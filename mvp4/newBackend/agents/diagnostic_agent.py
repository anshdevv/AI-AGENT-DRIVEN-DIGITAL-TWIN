# agents/diagnostic_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Diagnostic node — runs after the entire conversation (booking + triage done).
#
# Takes everything in state, asks MedGemma to produce a structured clinical
# summary for the doctor, saves it to Supabase, returns the report in state.
#
# The doctor sees this report when they open the appointment.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.messages import SystemMessage, AIMessage
from langchain_ollama import ChatOllama

from agents.mcp_tools import save_case_notes

try:
    PKT = ZoneInfo("Asia/Karachi")
except ZoneInfoNotFoundError:
    PKT = timezone(timedelta(hours=5))

BOOKING_CTX_DIR = Path("booking_context")

# ── MedGemma client (shared with triage_agent if already warm) ────────────────
_med_llm: ChatOllama | None = None

def _get_med_llm() -> ChatOllama:
    global _med_llm
    if _med_llm is None:
        _med_llm = ChatOllama(model="medgemma:4b", temperature=0.0)
    return _med_llm


# ── Diagnostic system prompt ──────────────────────────────────────────────────
_DIAG_SYSTEM = """You are a clinical documentation assistant.
Your job is to produce a clean, structured pre-consultation report for the attending physician.
You are NOT diagnosing. You are organising information already collected during triage.

Write the report in this exact format:

=========================================
      PRE-CONSULTATION CLINICAL REPORT
=========================================
ATTENDING PHYSICIAN: {doctor}
SPECIALTY: {specialization}

PATIENT:
  Name : {name}
  Phone: {phone}

CHIEF COMPLAINT:
  {complaint}

PAST MEDICAL HISTORY:
  {history}

TRIAGE Q&A:
{qa}

CLINICAL SUMMARY FROM TRIAGE:
  {summary}

SUGGESTED SPECIALIST:
  {specialist}

PRECAUTIONS TO DISCUSS:
  {precautions}
=========================================

Fill in each section from the information provided.
If a field is missing, write "Not recorded."
Do not add diagnoses, drug names, or information not in the input.
"""


def diagnostic_node(state: dict) -> dict:
    print("\n" + "="*54)
    print("📋 [Diagnostic] Generating clinical report with MedGemma")

    # ── Pull everything from state ────────────────────────────────
    symptom    = state.get("extracted_symptom", "Not recorded")
    profile    = state.get("patient_profile") or {}
    ctx        = state.get("booking_context") or {}
    triage_qa  = state.get("triage_qa") or []
    patient    = ctx.get("patient", {})
    doctor     = ctx.get("selected_doctor", {})
    appt       = ctx.get("appointment", {})

    doctor_name    = profile.get("booked_doctor") or doctor.get("name") or "Unknown"
    specialization = profile.get("doctor_specialization") or doctor.get("specialization") or "General Physician"
    patient_name   = patient.get("name") or "Unknown"
    patient_phone  = patient.get("phone") or "Unknown"
    past_history   = profile.get("past_history") or "Not provided"
    booking_id     = appt.get("booking_id")

    # Format Q&A
    if triage_qa:
        formatted_qa = "\n".join([f"  {pair}" for pair in triage_qa])
    else:
        formatted_qa = "  No triage questions recorded."

    # Extract clinical summary from triage messages if present
    clinical_summary = "See triage Q&A above."
    for m in reversed(list(state.get("messages", []))):
        content = str(m.content)
        if "MEDGEMMA_SUMMARY:" in content:
            start = content.find("MEDGEMMA_SUMMARY:") + 17
            end   = content.find("]", start)
            if end > start:
                clinical_summary = content[start:end].strip()
            break

    # Suggested specialist from symptom_lookup (if available in profile)
    specialist  = profile.get("doctor_specialization") or specialization
    precautions = profile.get("precautions") or "Discuss with attending physician."

    # ── Build user prompt for MedGemma ────────────────────────────
    user_prompt = _DIAG_SYSTEM.format(
        doctor        = doctor_name,
        specialization= specialization,
        name          = patient_name,
        phone         = patient_phone,
        complaint     = symptom,
        history       = past_history,
        qa            = formatted_qa,
        summary       = clinical_summary,
        specialist    = specialist,
        precautions   = precautions,
    )

    # ── Call MedGemma ─────────────────────────────────────────────
    report = ""
    try:
        med_llm  = _get_med_llm()
        response = med_llm.invoke([SystemMessage(content=user_prompt)])
        report   = str(response.content).strip()
        print(f"✅ [Diagnostic] Report generated ({len(report)} chars)")
    except Exception as e:
        print(f"❌ [Diagnostic] MedGemma failed: {e} — using fallback report")
        report = (
            "=========================================\n"
            "      PRE-CONSULTATION CLINICAL REPORT\n"
            "=========================================\n"
            f"ATTENDING PHYSICIAN: {doctor_name}\n"
            f"SPECIALTY: {specialization}\n\n"
            f"PATIENT: {patient_name} ({patient_phone})\n\n"
            f"CHIEF COMPLAINT: {symptom}\n\n"
            f"PAST HISTORY: {past_history}\n\n"
            "TRIAGE Q&A:\n"
            f"{formatted_qa}\n\n"
            f"CLINICAL SUMMARY: {clinical_summary}\n"
            "=========================================\n"
            "(Generated by fallback — MedGemma unavailable)"
        )

    # ── Save to Supabase if we have a booking ID ──────────────────
    if booking_id:
        try:
            result = save_case_notes.invoke({
                "appointment_id": booking_id,
                "notes": report,
            })
            print(f"💾 [Diagnostic] Final report saved to Supabase → {result}")
        except Exception as e:
            print(f"❌ [Diagnostic] Could not save to Supabase: {e}")
    else:
        print("⚠️  [Diagnostic] No booking_id — report not saved to Supabase")

    # ── Save full session record to JSON sidecar ──────────────────
    session_id = ctx.get("session_id")
    if session_id:
        try:
            safe_id   = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
            json_path = BOOKING_CTX_DIR / f"{safe_id}.json"
            # Load existing context and append the report
            if json_path.exists():
                existing = json.loads(json_path.read_text(encoding="utf-8"))
            else:
                existing = {}
            existing["diagnostic_report"]      = report
            existing["diagnostic_generated_at"] = datetime.now(PKT).isoformat()
            existing["triage_qa"]               = list(state.get("triage_qa") or [])
            json_path.write_text(
                json.dumps(existing, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"💾 [Diagnostic] Session JSON updated → {json_path.name}")
        except Exception as e:
            print(f"❌ [Diagnostic] Could not update session JSON: {e}")

    print(f"\n📄 [Report]:\n{report}")
    return {"final_diagnostic_report": report}