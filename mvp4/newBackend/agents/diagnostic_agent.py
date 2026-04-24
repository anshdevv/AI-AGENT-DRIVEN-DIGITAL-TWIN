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
    return {"final_diagnostic_report": report}