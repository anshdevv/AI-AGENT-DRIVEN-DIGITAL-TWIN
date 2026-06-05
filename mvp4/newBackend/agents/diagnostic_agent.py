# agents/diagnostic_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic node — runs after the entire conversation (booking + triage done).
#
# DESIGN:
#   - Reads ctx["medgemma_raw_history"] to reconstruct clean clinical Q&A
#     (MedGemma's own raw questions + patient answers — never Qwen's rephrase)
#   - Generates a proper SOAP note (Subjective / Objective / Assessment / Plan)
#     via a System + Human two-message prompt so MedGemma doesn't hallucinate
#   - Validates the output contains all four SOAP headers before saving
#   - Falls back to a deterministic SOAP template if MedGemma goes off-rails
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from agents.mcp_tools import save_case_notes

try:
    PKT = ZoneInfo("Asia/Karachi")
except ZoneInfoNotFoundError:
    PKT = timezone(timedelta(hours=5))

BOOKING_CTX_DIR = Path("booking_context")

# ── MedGemma client ───────────────────────────────────────────────────────────

_med_llm: ChatOllama | None = None


def _get_med_llm() -> ChatOllama:
    global _med_llm
    if _med_llm is None:
        _med_llm = ChatOllama(model="medgemma:4b", temperature=0.0)
    return _med_llm


# ── Tag stripping ─────────────────────────────────────────────────────────────

_TAG_RE = re.compile(
    r"\[MEDGEMMA_SUMMARY:[^\]]*\]"
    r"|\[SYMPTOM_LOGGED:[^\]]+\]"
    r"|\[START_TRIAGE\]"
    r"|\[END_CALL\]"
    r"|\[TRIAGE_COMPLETE\]"
    r"|CLINICAL_SUMMARY:.*?(?=\n\n|\Z)",
    re.DOTALL,
)


def _strip_tags(text: str) -> str:
    return _TAG_RE.sub("", text).strip()


# ── Q&A extractor from medgemma_raw_history ───────────────────────────────────

def _extract_medgemma_qa(ctx: dict) -> list[tuple[str, str]]:
    """
    Parse ctx["medgemma_raw_history"] into clean (question, answer) pairs.

    History structure written by triage_agent:
        [0] {"role": "user",      "content": "<original complaint>"}   ← skip
        [1] {"role": "assistant", "content": "<MedGemma raw Q1>"}
        [2] {"role": "user",      "content": "<patient answer 1>"}
        [3] {"role": "assistant", "content": "<MedGemma raw Q2>"}
        [4] {"role": "user",      "content": "<patient answer 2>"}
        …

    We skip index 0 (that's the complaint, not a Q&A pair) and walk the rest
    pairing assistant entries with the user entry immediately following.
    """
    history = ctx.get("medgemma_raw_history", [])
    pairs: list[tuple[str, str]] = []

    i = 1   # skip index 0 (original complaint)
    while i < len(history):
        entry = history[i]
        if entry["role"] == "assistant":
            # Clean the question — strip tags and trailing summary blocks
            question = _strip_tags(entry["content"])
            question = re.split(r"\[TRIAGE_COMPLETE\]|CLINICAL_SUMMARY:", question)[0].strip()
            if i + 1 < len(history) and history[i + 1]["role"] == "user":
                answer = history[i + 1]["content"].strip()
                if question and answer:
                    pairs.append((question, answer))
                i += 2
            else:
                i += 1
        else:
            i += 1

    return pairs


def _format_qa_for_prompt(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return "No triage questions recorded."
    return "\n".join(f"Q{n}: {q}\nA{n}: {a}" for n, (q, a) in enumerate(pairs, 1))


def _format_qa_for_report(pairs: list[tuple[str, str]]) -> str:
    if not pairs:
        return "  No triage questions recorded."
    lines = []
    for n, (q, a) in enumerate(pairs, 1):
        lines.append(f"  Q{n}: {q}")
        lines.append(f"  A{n}: {a}")
    return "\n".join(lines)


# ── SOAP system prompt ────────────────────────────────────────────────────────
# Instructions ONLY — no patient data — so MedGemma stays on-task.

_SOAP_SYSTEM = """\
You are a clinical documentation assistant for a hospital outpatient department.
Your ONLY job is to produce a SOAP note and pre-consultation summary for the attending physician.

STRICT RULES:
1. Use ONLY the patient data provided in the human message. Do NOT invent anything.
2. Do NOT make a final diagnosis. Do NOT prescribe medication.
3. Write "Not recorded." for any field where data is absent.
4. Output ONLY the report — no preamble, no questions, no explanation.
5. Follow EXACTLY this format (keep every divider, label, and section header):

=========================================
     PRE-CONSULTATION SOAP NOTE
=========================================
DATE                : {date}
ATTENDING PHYSICIAN : {doctor}
SPECIALTY           : {specialization}

PATIENT
  Name  : {name}
  Phone : {phone}

─────────────────────────────────────────
S — SUBJECTIVE
─────────────────────────────────────────
Chief Complaint:
  <one sentence — what the patient reports in their own words>

History of Present Illness (HPI):
  <narrative paragraph: onset, location, duration, character,
   aggravating/relieving factors, associated symptoms — drawn
   ONLY from the Triage Q&A provided>

Triage Q&A (clinical pre-screening):
{qa}

Past Medical / Surgical History:
  {history}

─────────────────────────────────────────
O — OBJECTIVE
─────────────────────────────────────────
Vital Signs    : Not recorded (pre-consultation triage only)
Physical Exam  : Deferred — to be completed by attending physician

Symptom Severity  : {severity}
Symptom Match     : {dataset_match}

─────────────────────────────────────────
A — ASSESSMENT
─────────────────────────────────────────
Clinical Impression:
  <2-3 sentences: plausible differentials based ONLY on the
   Subjective section — do NOT give a definitive diagnosis>

Routing Rationale : {routing_reason}

─────────────────────────────────────────
P — PLAN
─────────────────────────────────────────
Recommended Specialist : {specialist}
Booking Status         : {booking_status}

Precautions / Red Flags to Discuss:
  {precautions}

Next Steps:
  1. Attending physician to perform full history and physical examination.
  2. Order investigations as clinically indicated.
  3. Review and update this note after consultation.
=========================================
"""

# ── Human / data prompt ───────────────────────────────────────────────────────
# Patient data ONLY — no instructions — so MedGemma has a clean HumanMessage.

_DATA_TEMPLATE = """\
Please complete the SOAP note using ONLY the following patient data:

ATTENDING PHYSICIAN : {doctor}
SPECIALTY           : {specialization}
PATIENT NAME        : {name}
PATIENT PHONE       : {phone}
DATE                : {date}
CHIEF COMPLAINT     : {complaint}
PAST HISTORY        : {history}
RECOMMENDED SPEC    : {specialist}
BOOKING STATUS      : {booking_status}
ROUTING REASON      : {routing_reason}
SEVERITY            : {severity}
SYMPTOM MATCH       : {dataset_match}
PRECAUTIONS         : {precautions}

TRIAGE Q&A (MedGemma clinical questions + patient answers — use for HPI):
{qa}

CLINICAL SUMMARY FROM TRIAGE:
{summary}
"""


# ── Fallback deterministic SOAP note ─────────────────────────────────────────

def _clean_dataset_match(raw: str) -> str:
    """
    Strip the raw 'Top candidate conditions' block from the symptom lookup output.
    Keep only the clinically useful summary lines (severity, recommendation, specialist).
    The raw conditions list has too many false positives at low scores (50% on a single
    symptom match) and makes the SOAP note look noisy and unprofessional.
    """
    if not raw:
        return "Not recorded"
    lines = raw.splitlines()
    clean = []
    skip  = False
    for line in lines:
        stripped = line.strip()
        # Start skipping at the conditions block
        if re.search(r"top candidate conditions|matched symptoms|precautions", stripped, re.I):
            skip = True
        if skip:
            continue
        # Drop the divider line and SYMPTOM LOOKUP header
        if re.match(r"──+", stripped) or "SYMPTOM LOOKUP CONTEXT" in stripped:
            continue
        if stripped:
            clean.append(stripped)
    return "\n  ".join(clean) if clean else "Not recorded"


def _build_fallback_report(
    date: str,
    doctor_name: str,
    specialization: str,
    patient_name: str,
    patient_phone: str,
    symptom: str,
    past_history: str,
    formatted_qa: str,
    clinical_summary: str,
    specialist: str,
    booking_status: str,
    routing_reason: str,
    severity: str,
    dataset_match: str,
    precautions: str,
) -> str:
    return (
        "=========================================\n"
        "     PRE-CONSULTATION SOAP NOTE\n"
        "=========================================\n"
        f"DATE                : {date}\n"
        f"ATTENDING PHYSICIAN : {doctor_name}\n"
        f"SPECIALTY           : {specialization}\n\n"
        "PATIENT\n"
        f"  Name  : {patient_name}\n"
        f"  Phone : {patient_phone}\n\n"
        "─────────────────────────────────────────\n"
        "S — SUBJECTIVE\n"
        "─────────────────────────────────────────\n"
        "Chief Complaint:\n"
        f"  {symptom}\n\n"
        "History of Present Illness (HPI):\n"
        f"  {clinical_summary}\n\n"
        "Triage Q&A (clinical pre-screening):\n"
        f"{formatted_qa}\n\n"
        "Past Medical / Surgical History:\n"
        f"  {past_history}\n\n"
        "─────────────────────────────────────────\n"
        "O — OBJECTIVE\n"
        "─────────────────────────────────────────\n"
        "Vital Signs    : Not recorded (pre-consultation triage only)\n"
        "Physical Exam  : Deferred — to be completed by attending physician\n\n"
        f"Symptom Severity  : {severity}\n"
        f"Symptom Match     : {dataset_match}\n\n"
        "─────────────────────────────────────────\n"
        "A — ASSESSMENT\n"
        "─────────────────────────────────────────\n"
        "Clinical Impression:\n"
        "  Pre-consultation triage completed. See Chief Complaint and Q&A above.\n"
        "  Formal assessment to be performed by the attending physician.\n\n"
        f"Routing Rationale : {routing_reason}\n\n"
        "─────────────────────────────────────────\n"
        "P — PLAN\n"
        "─────────────────────────────────────────\n"
        f"Recommended Specialist : {specialist}\n"
        f"Booking Status         : {booking_status}\n\n"
        "Precautions / Red Flags to Discuss:\n"
        f"  {precautions}\n\n"
        "Next Steps:\n"
        "  1. Attending physician to perform full history and physical examination.\n"
        "  2. Order investigations as clinically indicated.\n"
        "  3. Review and update this note after consultation.\n"
        "=========================================\n"
        "(Generated by deterministic fallback — MedGemma output invalid)"
    )


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_report(text: str) -> bool:
    """Return False if any SOAP section header is missing (hallucination signal)."""
    required = ["S — SUBJECTIVE", "O — OBJECTIVE", "A — ASSESSMENT", "P — PLAN"]
    return all(marker in text for marker in required)


# ── Debug printer ─────────────────────────────────────────────────────────────

def _print_diagnostic_context(sys_msg: str, human_msg: str) -> None:
    print("\n╔" + "═" * 60 + "╗")
    print("║  📋 DIAGNOSTIC MEDGEMMA INPUT (SOAP NOTE)                  ║")
    print("╠" + "═" * 60 + "╣")
    for label, content in [("[SYSTEM]", sys_msg), ("[HUMAN / PATIENT DATA]", human_msg)]:
        print(f"║  {label}")
        print("╟" + "─" * 60 + "╢")
        for line in content.splitlines():
            while len(line) > 58:
                print(f"║  {line[:58]}")
                line = line[58:]
            print(f"║  {line}")
        print("╟" + "─" * 60 + "╢")
    print("╚" + "═" * 60 + "╝\n")


# ── Main node ─────────────────────────────────────────────────────────────────

def diagnostic_node(state: dict) -> dict:
    print("\n" + "=" * 54)
    print("📋 [Diagnostic] Generating SOAP note with MedGemma")

    # ── Pull state ────────────────────────────────────────────────
    symptom   = state.get("extracted_symptom", "Not recorded")
    profile   = state.get("patient_profile") or {}
    ctx       = state.get("booking_context") or {}
    patient   = ctx.get("patient", {})
    doctor    = ctx.get("selected_doctor", {})
    appt      = ctx.get("appointment", {})

    doctor_name    = profile.get("booked_doctor") or doctor.get("name") or "Unknown"
    specialization = profile.get("doctor_specialization") or doctor.get("specialization") or "General Physician"
    patient_name   = patient.get("name") or "Unknown"
    patient_phone  = patient.get("phone") or "Unknown"
    past_history   = profile.get("past_history") or "Not provided"
    booking_id     = appt.get("booking_id")
    routing_reason = ctx.get("routing_reason") or "Not recorded"
    severity       = ctx.get("triage_severity") or "Unknown"
    specialist     = ctx.get("recommended_specialist") or specialization
    precautions    = profile.get("precautions") or "Discuss with attending physician."
    date_str       = datetime.now(PKT).strftime("%Y-%m-%d %H:%M PKT")
    dataset_match  = ctx.get("final_symptom_match") or "Not recorded"
    dataset_match  = _clean_dataset_match(dataset_match)   # strip noisy conditions block

    # Booking status line
    if appt.get("confirmed") and booking_id:
        booking_status = (
            f"CONFIRMED — Booking ID {booking_id}  "
            f"({appt.get('date', '?')} at {appt.get('time', '?')})"
        )
    elif appt.get("date"):
        booking_status = f"Pending confirmation — {appt.get('date')} at {appt.get('time', '?')}"
    else:
        booking_status = "Not yet booked"

    # ── Extract clean Q&A from MedGemma's isolated history ───────
    medgemma_qa_pairs = _extract_medgemma_qa(ctx)

    if medgemma_qa_pairs:
        print(f"   [Diagnostic] Using medgemma_raw_history → {len(medgemma_qa_pairs)} Q&A pairs")
    else:
        # Fallback: parse state triage_qa strings ("Q: ...\nA: ...")
        print("   [Diagnostic] medgemma_raw_history empty — falling back to triage_qa strings")
        for pair_str in list(state.get("triage_qa") or ctx.get("triage_qa") or []):
            lines = pair_str.strip().splitlines()
            q = next((l[2:].strip() for l in lines if l.startswith("Q:")), "")
            a = next((l[2:].strip() for l in lines if l.startswith("A:")), "")
            if q and a:
                medgemma_qa_pairs.append((q, a))
        print(f"   [Diagnostic] Fallback produced {len(medgemma_qa_pairs)} pairs")

    formatted_qa_prompt = _format_qa_for_prompt(medgemma_qa_pairs)
    formatted_qa_report = _format_qa_for_report(medgemma_qa_pairs)

    # ── Extract clinical summary from messages ────────────────────
    clinical_summary = "See triage Q&A above."
    for m in reversed(list(state.get("messages", []))):
        content = str(m.content)
        if "MEDGEMMA_SUMMARY:" in content:
            start = content.find("MEDGEMMA_SUMMARY:") + 17
            end   = content.find("]", start)
            if end > start:
                clinical_summary = content[start:end].strip()
            break

    # ── Build System + Human prompts ──────────────────────────────
    system_text = _SOAP_SYSTEM.format(
        date           = date_str,
        doctor         = doctor_name,
        specialization = specialization,
        name           = patient_name,
        phone          = patient_phone,
        history        = past_history,
        qa             = formatted_qa_report,
        severity       = severity,
        dataset_match  = dataset_match,
        routing_reason = routing_reason,
        specialist     = specialist,
        booking_status = booking_status,
        precautions    = precautions,
    )

    human_text = _DATA_TEMPLATE.format(
        date           = date_str,
        doctor         = doctor_name,
        specialization = specialization,
        name           = patient_name,
        phone          = patient_phone,
        complaint      = symptom,
        history        = past_history,
        qa             = formatted_qa_prompt,
        summary        = clinical_summary,
        specialist     = specialist,
        booking_status = booking_status,
        routing_reason = routing_reason,
        severity       = severity,
        dataset_match  = dataset_match,
        precautions    = precautions,
    )

    _print_diagnostic_context(system_text, human_text)

    # ── Call MedGemma ─────────────────────────────────────────────
    report = ""
    medgemma_succeeded = False
    try:
        med_llm  = _get_med_llm()
        response = med_llm.invoke([
            SystemMessage(content=system_text),
            HumanMessage(content=human_text),
        ])
        raw = str(response.content).strip()
        print(f"🧠 [Diagnostic] MedGemma output ({len(raw)} chars):\n{raw[:600]}")

        if _validate_report(raw):
            report = raw
            medgemma_succeeded = True
            print("✅ [Diagnostic] SOAP note validated — all four SOAP sections present")
        else:
            print("⚠️  [Diagnostic] Validation FAILED — expected SOAP headers missing, using fallback")
            print(f"   First 200 chars: {raw[:200]}")

    except Exception as e:
        print(f"❌ [Diagnostic] MedGemma failed: {e} — using deterministic fallback")

    # ── Fallback ──────────────────────────────────────────────────
    if not medgemma_succeeded:
        report = _build_fallback_report(
            date           = date_str,
            doctor_name    = doctor_name,
            specialization = specialization,
            patient_name   = patient_name,
            patient_phone  = patient_phone,
            symptom        = symptom,
            past_history   = past_history,
            formatted_qa   = formatted_qa_report,
            clinical_summary = clinical_summary,
            specialist     = specialist,
            booking_status = booking_status,
            routing_reason = routing_reason,
            severity       = severity,
            dataset_match  = dataset_match,
            precautions    = precautions,
        )

    # ── Save to Supabase ──────────────────────────────────────────
    if booking_id:
        try:
            result = save_case_notes.invoke({
                "appointment_id": booking_id,
                "notes": report,
            })
            print(f"💾 [Diagnostic] SOAP note saved to Supabase → {result}")
        except Exception as e:
            print(f"❌ [Diagnostic] Could not save to Supabase: {e}")
    else:
        print("⚠️  [Diagnostic] No booking_id found — SOAP note not saved to Supabase")

    # ── Update JSON sidecar ───────────────────────────────────────
    session_id = ctx.get("session_id")
    if session_id:
        try:
            safe_id   = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
            json_path = BOOKING_CTX_DIR / f"{safe_id}.json"
            existing  = json.loads(json_path.read_text(encoding="utf-8")) if json_path.exists() else {}
            existing["diagnostic_report"]       = report
            existing["diagnostic_generated_at"] = datetime.now(PKT).isoformat()
            existing["triage_qa"]               = list(state.get("triage_qa") or [])
            existing["medgemma_qa_pairs"]        = [
                {"q": q, "a": a} for q, a in medgemma_qa_pairs
            ]
            existing["medgemma_succeeded"]       = medgemma_succeeded
            json_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"💾 [Diagnostic] Session JSON updated → {json_path.name}")
        except Exception as e:
            print(f"❌ [Diagnostic] Could not update session JSON: {e}")

    print(f"\n📄 [SOAP Report]:\n{report}")

    # ── Save as styled HTML ───────────────────────────────────────
    # Parse the medgemma_qa_pairs for the HTML table
    _qa_for_html = medgemma_qa_pairs[:6]   # cap at 6 for display
    _conditions  = []
    if dataset_match:
        for _line in dataset_match.split("\n"):
            _m = re.search(r"(\w[\w ]+)\s+\(score\s+(\d+%?)\)", _line)
            if _m:
                _pre = re.search(r"Precautions\s*:\s*(.+)", _line)
                _conditions.append({
                    "name":       _m.group(1).strip(),
                    "score":      _m.group(2),
                    "symptoms":   "headache" if "headache" in report.lower() else symptom,
                    "precautions": _pre.group(1).strip() if _pre else "Discuss with physician.",
                })
    # Extract HPI from report text
    _hpi = ""
    _hpi_m = re.search(r"History of Present Illness.*?:\s*(.+?)(?:\n|$)", report, re.IGNORECASE)
    if _hpi_m:
        _hpi = _hpi_m.group(1).strip()
    if not _hpi:
        _hpi = f"Patient reports {symptom}. Duration and progression as noted in triage Q&A."
    _clinical_m = re.search(r"Clinical Impression.*?:\s*(.+?)(?:\n|$)", report, re.IGNORECASE)
    _clinical_impression = _clinical_m.group(1).strip() if _clinical_m else "Refer to SOAP note text above."

    _soap_html_path = _save_soap_html(
        session_id     = ctx.get("session_id", ""),
        booking_id     = str(booking_id or ""),
        patient_name   = patient_name,
        date_str       = date_str,
        doctor_name    = doctor_name,
        specialization = specialization,
        patient_phone  = patient_phone,
        symptom        = symptom,
        hpi            = _hpi,
        qa_pairs       = _qa_for_html,
        severity       = severity,
        dataset_match  = dataset_match,
        conditions     = _conditions,
        clinical_impression = _clinical_impression,
        routing_reason = routing_reason,
        specialist     = specialist,
        booking_date   = booking_status.replace("Booking ID:", "").strip() if booking_status else "",
        precautions    = precautions,
    )
    if _soap_html_path and session_id:
        try:
            safe_id   = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
            json_path = BOOKING_CTX_DIR / f"{safe_id}.json"
            if json_path.exists():
                _jdata = json.loads(json_path.read_text(encoding="utf-8"))
                _jdata["soap_html_path"] = _soap_html_path
                json_path.write_text(json.dumps(_jdata, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass

    return {"final_diagnostic_report": report}


# ─────────────────────────────────────────────────────────────────────────────
# HTML SOAP NOTE GENERATOR
# ─────────────────────────────────────────────────────────────────────────────

_SOAP_HTML_DIR = Path("soap_notes")

def _build_soap_html(
    date_str: str, doctor_name: str, specialization: str,
    patient_name: str, patient_phone: str, symptom: str,
    hpi: str, qa_pairs: list[tuple[str,str]],
    severity: str, dataset_match: str, conditions: list[dict],
    clinical_impression: str, routing_reason: str,
    specialist: str, booking_id: str, booking_date: str,
    precautions: str,
) -> str:
    """Generate a fully styled HTML SOAP note from structured data."""

    qa_rows = ""
    for i, (q, a) in enumerate(qa_pairs, 1):
        q_clean = q.replace("<", "&lt;").replace(">", "&gt;")
        a_clean = a.replace("<", "&lt;").replace(">", "&gt;")
        qa_rows += f"""
        <div class="qa-row">
          <div class="qa-cell"><span class="qnum">Q{i}:</span> {q_clean}</div>
          <div class="qa-cell"><span class="anum">A{i}:</span> {a_clean}</div>
        </div>"""

    cond_rows = ""
    for i, c in enumerate(conditions[:3], 1):
        cond_rows += f"""
        <div class="condition">
          <div class="num">{i}</div>
          <div>
            <div class="condition-title">{c.get('name','')} <span class="muted">({c.get('score','')})</span></div>
            <div>Matched symptoms: {c.get('symptoms','')}</div>
            <div><strong class="label-strong">Precautions:</strong> {c.get('precautions','')}</div>
          </div>
        </div>"""

    booking_badge = (
        f'<span class="booking-badge">✅ CONFIRMED</span> Booking ID {booking_id}<br>'
        f'<span class="muted">{booking_date}</span>'
        if booking_id else "Not yet booked"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1.0"/>
  <title>Pre-Consultation SOAP Note — {patient_name}</title>
  <style>
    :root{{--navy:#0b2f5b;--navy-2:#123f75;--teal:#087d80;--teal-soft:#e8f7f6;--blue-soft:#eef6ff;--gray-50:#f7fafc;--gray-100:#eef2f6;--gray-200:#d9e2ec;--gray-500:#66788a;--text:#182433;--muted:#596b7c;--white:#ffffff;--shadow:0 18px 45px rgba(11,47,91,0.10);--radius:22px}}
    *{{box-sizing:border-box}}
    body{{margin:0;font-family:Inter,ui-sans-serif,system-ui,-apple-system,sans-serif;background:linear-gradient(135deg,#f4f8fb 0%,#eef7f7 100%);color:var(--text);line-height:1.55}}
    .page{{width:min(1180px,calc(100% - 32px));margin:32px auto;background:rgba(255,255,255,0.94);border:1px solid rgba(217,226,236,0.85);border-radius:30px;box-shadow:var(--shadow);overflow:hidden}}
    .topbar{{background:linear-gradient(135deg,var(--navy) 0%,var(--teal) 100%);color:white;padding:28px 34px;display:flex;align-items:center;justify-content:space-between;gap:24px}}
    .brand{{display:flex;gap:18px;align-items:center}}
    .logo{{width:68px;height:68px;border-radius:20px;background:rgba(255,255,255,0.16);display:grid;place-items:center;font-size:34px;border:1px solid rgba(255,255,255,0.28)}}
    .title-block h1{{margin:0;font-size:clamp(28px,4vw,44px);line-height:1.02;letter-spacing:-0.04em;font-weight:850}}
    .title-block p{{margin:8px 0 0;color:rgba(255,255,255,0.82);font-size:15px}}
    .status-pill{{display:inline-flex;align-items:center;gap:8px;background:rgba(255,255,255,0.14);border:1px solid rgba(255,255,255,0.24);border-radius:999px;padding:10px 16px;font-weight:750;white-space:nowrap}}
    .content{{padding:30px 34px 36px}}
    .meta-grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:22px}}
    .meta-card{{background:var(--white);border:1px solid var(--gray-200);border-radius:18px;padding:18px;min-height:110px;box-shadow:0 8px 18px rgba(11,47,91,0.04)}}
    .meta-label{{display:flex;gap:8px;align-items:center;color:var(--teal);font-size:12px;font-weight:850;letter-spacing:0.08em;text-transform:uppercase;margin-bottom:8px}}
    .meta-value{{font-size:18px;font-weight:750;color:var(--navy)}}
    .meta-subvalue{{margin-top:3px;color:var(--muted);font-size:14px}}
    .section{{background:var(--white);border:1px solid var(--gray-200);border-radius:var(--radius);margin-top:18px;overflow:hidden;box-shadow:0 10px 26px rgba(11,47,91,0.05)}}
    .section-header{{display:flex;align-items:center;gap:14px;padding:18px 22px;border-bottom:1px solid var(--gray-200);background:linear-gradient(90deg,#ffffff 0%,#f7fbfd 100%)}}
    .letter{{width:48px;height:48px;border-radius:15px;display:grid;place-items:center;color:white;font-size:26px;font-weight:900;box-shadow:0 10px 20px rgba(8,125,128,0.18)}}
    .letter.teal{{background:var(--teal)}}.letter.navy{{background:var(--navy)}}
    .section-header h2{{margin:0;font-size:24px;color:var(--navy);letter-spacing:-0.02em}}
    .section-body{{padding:22px}}
    .two-col{{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}}
    .field-list{{display:grid;gap:16px}}
    .field{{display:grid;grid-template-columns:28px 1fr;gap:10px;align-items:start}}
    .dot{{width:9px;height:9px;background:var(--teal);border-radius:999px;margin-top:9px;justify-self:center}}
    .field strong,.label-strong{{color:var(--teal);font-weight:850}}
    .qa-table{{border:1px solid #c8dde2;border-radius:18px;overflow:hidden;background:#fbfefe}}
    .qa-title{{background:var(--teal-soft);color:var(--teal);font-weight:850;padding:12px 16px;border-bottom:1px solid #c8dde2}}
    .qa-row{{display:grid;grid-template-columns:1.2fr 0.8fr;border-bottom:1px solid #e2edf0}}
    .qa-row:last-child{{border-bottom:0}}
    .qa-cell{{padding:12px 16px;font-size:14px}}
    .qa-cell+.qa-cell{{border-left:1px solid #e2edf0;background:#ffffff}}
    .qnum,.anum{{color:var(--teal);font-weight:900;margin-right:6px}}
    .objective-grid{{display:grid;grid-template-columns:0.78fr 1.22fr;gap:20px;align-items:start}}
    .icon-list{{display:grid;gap:18px}}
    .icon-field{{display:grid;grid-template-columns:46px 1fr;gap:12px;align-items:center}}
    .mini-icon{{width:46px;height:46px;border-radius:16px;background:var(--blue-soft);display:grid;place-items:center;color:var(--navy);font-size:22px}}
    .inset-card{{border:1px solid #b9d0ef;border-radius:18px;overflow:hidden;background:#fbfdff}}
    .inset-head{{background:var(--navy-2);color:white;padding:13px 16px;font-weight:850;display:flex;align-items:center;gap:10px}}
    .match-summary{{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;padding:16px;background:#f8fbff;border-bottom:1px solid #d6e5f7}}
    .metric{{border-right:1px solid #d6e5f7;padding-right:10px}}
    .metric:last-child{{border-right:0}}
    .metric span{{display:block;color:var(--muted);font-size:12px;font-weight:700}}
    .metric b{{color:var(--teal);font-size:16px}}
    .conditions{{padding:16px}}
    .conditions h3{{margin:0 0 12px;color:var(--navy);font-size:16px}}
    .condition{{display:grid;grid-template-columns:30px 1fr;gap:10px;padding:11px 0;border-bottom:1px dashed #d8e5f2}}
    .condition:last-child{{border-bottom:0}}
    .num{{width:28px;height:28px;border-radius:999px;background:var(--navy);color:white;display:grid;place-items:center;font-weight:850;font-size:13px}}
    .condition-title{{font-weight:850;color:var(--navy)}}
    .muted{{color:var(--muted)}}
    .assessment-plan-grid{{display:grid;grid-template-columns:1fr 0.9fr;gap:20px;align-items:start}}
    .next-steps{{background:linear-gradient(135deg,#f3f8ff 0%,#edf7fb 100%);border:1px solid #cbdff3;border-radius:18px;padding:18px}}
    .next-steps h3{{margin:0 0 12px;color:var(--navy);font-size:18px}}
    .step{{display:grid;grid-template-columns:28px 1fr;gap:10px;margin:12px 0}}
    .step-number{{width:28px;height:28px;border-radius:999px;background:var(--navy);color:white;display:grid;place-items:center;font-size:13px;font-weight:900}}
    .booking-badge{{display:inline-flex;align-items:center;gap:8px;background:#ecfbf5;color:#08724f;border:1px solid #bcebd8;border-radius:999px;padding:6px 11px;font-weight:850;margin-left:4px}}
    .footer-strip{{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:18px}}
    .footer-card{{background:white;border:1px solid var(--gray-200);border-radius:18px;padding:18px;display:grid;grid-template-columns:46px 1fr;gap:12px;align-items:center}}
    .actions{{display:flex;justify-content:flex-end;gap:12px;margin-top:24px}}
    button{{border:0;border-radius:999px;padding:12px 18px;font-weight:850;cursor:pointer;color:white;background:linear-gradient(135deg,var(--navy) 0%,var(--teal) 100%);box-shadow:0 10px 22px rgba(8,125,128,0.18)}}
    @media(max-width:900px){{.two-col,.objective-grid,.assessment-plan-grid,.footer-strip{{grid-template-columns:1fr}}.meta-grid{{grid-template-columns:repeat(2,1fr)}}}}
    @media print{{body{{background:white}}.page{{width:100%;margin:0;border:0;box-shadow:none;border-radius:0}}.actions{{display:none}}.topbar{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}}}
  </style>
</head>
<body>
  <main class="page">
    <header class="topbar">
      <div class="brand">
        <div class="logo">📋</div>
        <div class="title-block">
          <h1>Pre-Consultation SOAP Note</h1>
          <p>Clinical pre-screening summary prepared before physician consultation</p>
        </div>
      </div>
      <div class="status-pill">✅ Booking Confirmed</div>
    </header>
    <section class="content">
      <div class="meta-grid">
        <article class="meta-card"><div class="meta-label">📅 Date</div><div class="meta-value">{date_str[:10]}</div><div class="meta-subvalue">{date_str[11:16]} PKT</div></article>
        <article class="meta-card"><div class="meta-label">🩺 Physician</div><div class="meta-value">{doctor_name}</div><div class="meta-subvalue">Attending physician</div></article>
        <article class="meta-card"><div class="meta-label">🏥 Specialty</div><div class="meta-value">{specialization}</div><div class="meta-subvalue">Recommended route</div></article>
        <article class="meta-card"><div class="meta-label">👤 Patient</div><div class="meta-value">{patient_name}</div><div class="meta-subvalue">Phone: {patient_phone}</div></article>
      </div>
      <section class="section">
        <div class="section-header"><div class="letter teal">S</div><h2>S — Subjective</h2></div>
        <div class="section-body two-col">
          <div class="field-list">
            <div class="field"><div class="dot"></div><div><strong>Chief Complaint:</strong> {symptom}</div></div>
            <div class="field"><div class="dot"></div><div><strong>History of Present Illness (HPI):</strong> {hpi}</div></div>
            <div class="field"><div class="dot"></div><div><strong>Past Medical / Surgical History:</strong> Not provided</div></div>
          </div>
          <aside class="qa-table">
            <div class="qa-title">Triage Q&amp;A (clinical pre-screening)</div>
            {qa_rows}
          </aside>
        </div>
      </section>
      <section class="section">
        <div class="section-header"><div class="letter navy">O</div><h2>O — Objective</h2></div>
        <div class="section-body objective-grid">
          <div class="icon-list">
            <div class="icon-field"><div class="mini-icon">♡</div><div><span class="label-strong">Vital Signs:</span> Not recorded (pre-consultation triage only)</div></div>
            <div class="icon-field"><div class="mini-icon">📝</div><div><span class="label-strong">Physical Exam:</span> Deferred — to be completed by attending physician</div></div>
            <div class="icon-field"><div class="mini-icon">📊</div><div><span class="label-strong">Symptom Severity:</span> {severity}</div></div>
          </div>
          <aside class="inset-card">
            <div class="inset-head">🎯 Symptom Match</div>
            <div class="match-summary">
              <div class="metric"><span>Severity estimate</span><b>{severity}</b></div>
              <div class="metric"><span>Recommendation</span><b>Physician review</b></div>
              <div class="metric"><span>Suggested specialist</span><b>{specialist}</b></div>
            </div>
            <div class="conditions"><h3>Top candidate conditions</h3>{cond_rows}</div>
          </aside>
        </div>
      </section>
      <section class="section">
        <div class="section-header"><div class="letter teal">A</div><h2>A — Assessment</h2></div>
        <div class="section-body">
          <div class="field-list">
            <div class="field"><div class="dot"></div><div><strong>Clinical Impression:</strong> {clinical_impression}</div></div>
            <div class="field"><div class="dot"></div><div><strong>Routing Rationale:</strong> {routing_reason}</div></div>
          </div>
        </div>
      </section>
      <section class="section">
        <div class="section-header"><div class="letter navy">P</div><h2>P — Plan</h2></div>
        <div class="section-body assessment-plan-grid">
          <div class="field-list">
            <div class="field"><div class="dot"></div><div><strong>Recommended Specialist:</strong> {specialist}</div></div>
            <div class="field"><div class="dot"></div><div><strong>Booking Status:</strong> {booking_badge}</div></div>
            <div class="field"><div class="dot"></div><div><strong>Precautions / Red Flags to Discuss:</strong> {precautions}</div></div>
          </div>
          <aside class="next-steps">
            <h3>Next Steps</h3>
            <div class="step"><div class="step-number">1</div><div>Attending physician to perform full history and physical examination.</div></div>
            <div class="step"><div class="step-number">2</div><div>Order investigations as clinically indicated.</div></div>
            <div class="step"><div class="step-number">3</div><div>Review and update this note after consultation.</div></div>
          </aside>
        </div>
      </section>
      <div class="footer-strip">
        <div class="footer-card"><div class="mini-icon">⚠️</div><div><strong class="label-strong">PRECAUTIONS:</strong> {precautions}</div></div>
        <div class="footer-card"><div class="mini-icon">📄</div><div><strong class="label-strong">CLINICAL SUMMARY FROM TRIAGE:</strong><br>See triage Q&amp;A above.</div></div>
      </div>
      <div class="actions"><button onclick="window.print()">Print / Save as PDF</button></div>
    </section>
  </main>
</body>
</html>"""


def _save_soap_html(
    session_id: str, booking_id: str, patient_name: str,
    **kwargs
) -> str | None:
    """Save the HTML SOAP note to soap_notes/ and return the file path."""
    try:
        _SOAP_HTML_DIR.mkdir(exist_ok=True)
        safe_name  = re.sub(r"[^a-zA-Z0-9]", "_", patient_name or "patient")
        safe_bid   = re.sub(r"[^a-zA-Z0-9]", "_", str(booking_id or session_id[:8]))
        filename   = f"SOAP_{safe_bid}_{safe_name}.html"
        path       = _SOAP_HTML_DIR / filename
        html       = _build_soap_html(
            patient_name=patient_name,
            booking_id=booking_id,
            **kwargs,
        )
        path.write_text(html, encoding="utf-8")
        print(f"🌐 [Diagnostic] HTML SOAP note saved → {path}")
        return str(path)
    except Exception as e:
        print(f"❌ [Diagnostic] HTML save failed: {e}")
        return None