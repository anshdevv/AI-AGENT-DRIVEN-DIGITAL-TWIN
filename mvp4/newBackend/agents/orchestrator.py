# agents/orchestrator.py
# ─────────────────────────────────────────────────────────────────────────────
# Single orchestrator — Qwen 32B as supervisor.
#
# FLOW:
#   1. User describes symptom
#   2. Triage node (MedGemma → Qwen direct chain) asks up to 8 clinical questions
#   3. Supervisor (Qwen) handles booking via deterministic tool calls
#   4. create_booking is gated by a code-level YES check in tool_executor
#   5. Diagnostic node (MedGemma) writes the final doctor report
#
# KEY DESIGN DECISIONS:
#   - Option B triage: MedGemma + Qwen chained inside triage_node (no disguise)
#   - triage_router → END while triage is active; → supervisor_node when done
#   - One supervisor LLM, but create_booking can ONLY fire when code confirms YES
#   - Session JSON named by session_id (never "default")
#   - tool_executor blocks re-fetching patient data if already in context
#   - Triage runs FIRST; booking step starts at collect_patient after triage
#   - Triage Q&A saved to Supabase via save_case_notes
#   - Slot extraction is LLM-based (no regex) — handles all languages/formats
#     Default time when none is stated: 11:00 AM
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

from typing import Annotated, Any, TypedDict, Sequence
import operator
import time
import json
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.messages import BaseMessage, AIMessage, SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.llm_config import get_llm
from agents.mcp_tools import ALL_TOOLS
from agents.triage_agent import triage_node
from agents.diagnostic_agent import diagnostic_node

try:
    PKT = ZoneInfo("Asia/Karachi")
except ZoneInfoNotFoundError:
    PKT = timezone(timedelta(hours=5))


# ═══════════════════════════════════════════════════════════════════
# TESTING FLAGS
# ═══════════════════════════════════════════════════════════════════
# Set this to False when you want the assistant to follow the detected/user language again.
FORCE_ENGLISH_TEST = True


def _force_english_for_testing(ctx: dict) -> dict:
    """Testing-only language lock: keep supervisor/Qwen-facing replies in English."""
    if FORCE_ENGLISH_TEST:
        ctx["patient_language"] = "en"
    return ctx


# ═══════════════════════════════════════════════════════════════════
# BOOKING CONTEXT — per-session JSON sidecar
# ═══════════════════════════════════════════════════════════════════

BOOKING_CTX_DIR = Path("booking_context")
BOOKING_CTX_DIR.mkdir(exist_ok=True)


def _booking_ctx_path(session_id: str) -> Path:
    if not session_id or session_id == "default":
        raise ValueError(f"session_id must be a real value. Got: '{session_id}'")
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
    return BOOKING_CTX_DIR / f"{safe}.json"


def _empty_booking_context(session_id: str) -> dict:
    return {
        "session_id":               session_id,
        "created_at":               datetime.now(PKT).isoformat(),
        "last_updated":             None,
        "patient_language":         "en",
        "step":                     "collect_patient_history",
        "prime_complaint":          None,
        "initial_complaint_hint":   None,   # complaint hinted during history phase — no repeat needed
        "recommended_specialist":   None,
        "routing_decision":         None,
        "triage_completed":         False,
        "patient_history_checked":  False,
        "patient_history_available": False,
        "patient_history_data":     None,
        # ── Algorithmic history field progression ────────────────────────
        "history_field_turn_count": {},
        "skipped_history_fields":   [],
        "collected_this_session":   [],
        "recent_case_notes":        None,
        # ── Orchestrator-controlled triage dimensions ─────────────────────
        "triage_dimensions":        None,   # [(dim_key, question_text), ...] computed once on turn 0
        # ── Triage tracking ──────────────────────────────────────────────
        "triage_questions_asked":   0,
        "triage_qa":                [],
        "medgemma_raw_history":     [],
        "accumulated_symptoms":     [],
        "symptom_context_block":    None,
        "final_symptom_match":      None,
        # ── Human handoff / waitlist ──────────────────────────────────────
        "human_handoff_pending":    False,
        "human_handoff_confirmed":  False,
        "waitlist_position":        None,
        "patient": {
            "id":             None,
            "name":           None,
            "phone":          None,
            "age":            None,
            "gender":         None,
            "marital_status": None,
        },
        "selected_doctor": {
            "id":             None,
            "name":           None,
            "specialization": None,
        },
        "pending_slot": {
            "date": None,
            "time": None,
        },
        "appointment": {
            "date":       None,
            "time":       None,
            "confirmed":  False,
            "booking_id": None,
        },
    }



# ── Nurse-style question text for each history field ─────────────────────────
_QUESTION_FOR_FIELD: dict[str, str] = {
    "age":               "How old are you?",
    "gender":            "Are you male or female?",
    "marital_status":    "Are you married or single?",
    "chronic_conditions": (
        "Do you have any existing health conditions — such as diabetes, "
        "high blood pressure, heart disease, asthma, or thyroid problems?"
    ),
    "medications":       "Are you currently taking any medications?",
    "drug_allergies":    "Do you have any known allergies to medications?",
    "general_allergies": (
        "Do you have any allergies to food, dust, pollen, animal fur, "
        "or other environmental triggers?"
    ),
    "family_history":    "Does heart disease, diabetes, or cancer run in your family?",
    "smoking_status":    "Do you smoke or drink alcohol?",
    # Female, age 12–55 (no marital gate — periods are not marital-dependent)
    "menstrual_history": (
        "How is your menstrual cycle? Is it regular, and do you experience "
        "any pain or heavy bleeding during your periods?"
    ),
    "lmp_date":          "When was the first day of your last menstrual period?",
    # Female + married + age 12–55 only
    "pregnancy_status":  "Are you currently pregnant or could you be pregnant?",
    "obstetric_history": (
        "Have you had any previous pregnancies? "
        "Were they normal deliveries or C-sections?"
    ),
    # Age gates
    "fall_history":      "Have you had any recent falls or balance problems?",
    "vaccination_status":"Are the child's vaccinations up to date?",
}


# ── What counts as a complete answer for each field ──────────────────────────
# For single-answer fields this is obvious — but for multi-part fields Qwen
# must keep asking follow-up questions WITHIN the field before saving.
_FIELD_COMPLETE_CRITERIA: dict[str, str] = {
    "age":               "a number",
    "gender":            "male or female",
    "marital_status":    "married, single, widowed, or divorced",
    "chronic_conditions":"a list of conditions OR confirmation of 'none'",
    "medications":       "a list of medications OR confirmation of 'none'",
    "drug_allergies":    "a list OR confirmation of 'none'",
    "general_allergies": "a list OR confirmation of 'none'",
    "family_history":    "family conditions mentioned OR confirmation of 'none'",
    "smoking_status":    "smoking status AND alcohol use — both answered",
    # Multi-part — Qwen must collect ALL parts before saving
    "menstrual_history": (
        "BOTH: (1) regularity — regular or irregular, "
        "AND (2) pain or discomfort — present or absent"
    ),
    "lmp_date":          "approximate date or timeframe of last period",
    "pregnancy_status":  "yes or no (currently pregnant)",
    "obstetric_history": (
        "BOTH: (1) whether previous pregnancies occurred, "
        "AND (2) if yes — delivery type: normal or C-section"
    ),
    "fall_history":      "yes or no, and if yes: how recently",
    "vaccination_status":"up to date, not up to date, or unsure",
}


def _compute_required_history_fields(patient: dict, history_data: str | None) -> list[str]:
    """
    Deterministically decide which history fields still need to be collected.
    Called in Python — no LLM involved in this decision.

    Demographics checked:   age, gender, marital_status
    Base medical history:   chronic_conditions, medications, drug_allergies,
                            general_allergies, family_history, smoking_status
    Female 12-55:           menstrual_history, lmp_date
    Female + married 12-55: pregnancy_status, obstetric_history
    Age 60+:                fall_history
    Age < 12:               vaccination_status
    """
    age     = patient.get("age")
    gender  = (patient.get("gender")         or "").strip().lower()
    marital = (patient.get("marital_status") or "").strip().lower()
    hd      = (history_data or "").lower()

    def _has(keyword: str) -> bool:
        if not hd or keyword not in hd:
            return False
        idx = hd.find(keyword)
        snippet = hd[idx:idx + 60]
        return "none reported" not in snippet and "not recorded" not in snippet

    required: list[str] = []

    # ── Demographics (patients table) ─────────────────────────────────────────
    if age is None:
        required.append("age")
    if not gender:
        required.append("gender")
    if not marital:
        required.append("marital_status")

    # ── Base medical history (patient_history table) ──────────────────────────
    if not _has("chronic conditions"):
        required.append("chronic_conditions")
    if not _has("medications"):
        required.append("medications")
    if not _has("drug allergies"):
        required.append("drug_allergies")
    if not _has("general allergies"):
        required.append("general_allergies")
    if not _has("family history"):
        required.append("family_history")
    if not _has("smoking"):
        required.append("smoking_status")

    # ── Demographic gates ─────────────────────────────────────────────────────
    is_female  = gender in ("female", "f", "woman", "girl")
    is_married = marital == "married"

    # Menstrual history: all females 12-55 (no marital gate)
    if is_female and age is not None and 12 <= int(age) <= 55:
        if not _has("menstrual"):
            required.append("menstrual_history")
        if not _has("lmp") and not _has("last menstrual"):
            required.append("lmp_date")

    # Pregnancy & obstetric history: female + married + 12-55 only
    # Asking about pregnancy to an unmarried woman is culturally inappropriate in Pakistan.
    if is_female and is_married and age is not None and 12 <= int(age) <= 55:
        if not _has("pregnancy status") and not _has("pregnancy"):
            required.append("pregnancy_status")
        if not _has("obstetric") and not _has("c-section") and not _has("delivery"):
            required.append("obstetric_history")

    # Elderly fall / polypharmacy screen
    if age is not None and int(age) >= 60:
        if not _has("fall"):
            required.append("fall_history")

    # Paediatric vaccination
    if age is not None and int(age) < 12:
        if not _has("vaccination"):
            required.append("vaccination_status")

    return required



# ── Maximum turns allowed per history field before force-advancing ────────────
_MAX_TURNS_PER_HISTORY_FIELD = 3


def _enforce_history_field_limits(ctx: dict) -> None:
    """
    Hard algorithmic gate against infinite loops in history collection.

    Called once per supervisor turn while step == collect_patient_history.
    Increments the turn counter for the currently active field.
    If the counter reaches _MAX_TURNS_PER_HISTORY_FIELD, the field is
    force-removed from required_history_fields and added to skipped_history_fields
    so the pipeline always moves forward regardless of Qwen's behaviour.

    The soft signal to Qwen (in the directive) says 'ask follow-ups' — but this
    function is the hard enforcement that ensures the field limit is respected.
    """
    required = ctx.get("required_history_fields")
    if not required:
        return   # nothing to enforce

    current_field = required[0]
    counts = ctx.setdefault("history_field_turn_count", {})
    counts[current_field] = counts.get(current_field, 0) + 1

    print(
        f"🔢 [HistoryGate] field='{current_field}'  "
        f"turn {counts[current_field]}/{_MAX_TURNS_PER_HISTORY_FIELD}"
    )

    if counts[current_field] >= _MAX_TURNS_PER_HISTORY_FIELD:
        print(
            f"⏭️  [HistoryGate] '{current_field}' hit {_MAX_TURNS_PER_HISTORY_FIELD}-turn limit "
            f"— force-advancing to next field"
        )
        required.pop(0)
        ctx["required_history_fields"] = required
        counts[current_field] = 0   # reset so it doesn't immediately re-trigger if re-added
        skipped = ctx.setdefault("skipped_history_fields", [])
        if current_field not in skipped:
            skipped.append(current_field)
        print(f"   Skipped: {skipped}  |  Remaining: {required}")



def _compute_triage_dimensions(complaint: str, ctx: dict) -> list[tuple[str, str]]:
    """
    Compute the ordered list of (dimension_key, question_text) for this triage session.
    Called once on the first triage turn and cached in ctx['triage_dimensions'].

    Priority: red_flag → duration → character → severity → associated → complaint_specific
    At most 6 dimensions. Condition-specific branch replaces generic modifying-factors.
    """
    c     = complaint.lower()
    p     = ctx.get("patient", {})
    hist  = (ctx.get("patient_history_data") or "").lower()
    age   = p.get("age")
    g     = (p.get("gender") or "").lower()
    m     = (p.get("marital_status") or "").lower()
    female  = g in ("female", "f", "woman", "girl")
    married = m == "married"

    def _has(kw: str) -> bool:
        return kw in hist

    dims: list[tuple[str, str]] = []

    # ── 1. Red flag (complaint-specific) ─────────────────────────────────────
    if any(w in c for w in ("chest", "heart", "pressure", "tightness")):
        dims.append(("red_flag",
            "Are you also experiencing sweating, pain in your left arm or jaw, "
            "or difficulty breathing?"))
    elif any(w in c for w in ("head", "headache")):
        dims.append(("red_flag",
            "Is this the worst headache of your life, or do you have any sudden "
            "weakness, facial drooping, or slurred speech?"))
    elif any(w in c for w in ("breath", "breathing", "shortness")):
        dims.append(("red_flag",
            "Are you able to speak in full sentences, and are your lips "
            "and fingertips a normal colour?"))
    elif any(w in c for w in ("abdomen", "stomach", "belly", "abdominal")):
        dims.append(("red_flag",
            "Is the pain so severe you cannot touch your abdomen, or have you "
            "noticed any blood in your stool or vomit?"))
    else:
        dims.append(("red_flag",
            "Are you experiencing any severe chest pain, difficulty breathing, "
            "sudden confusion, or uncontrolled bleeding?"))

    # ── 2. Duration ──────────────────────────────────────────────────────────
    dims.append(("duration",
        "How long have you had this, and did it come on suddenly or gradually?"))

    # ── 3. Character ─────────────────────────────────────────────────────────
    dims.append(("character",
        "How would you describe it — sharp, dull, burning, throbbing, "
        "or more of a pressure feeling?"))

    # ── 4. Severity ──────────────────────────────────────────────────────────
    dims.append(("severity",
        "On a scale of 1 to 10, how severe is it right now?"))

    # ── 5. Associated symptoms ───────────────────────────────────────────────
    dims.append(("associated",
        "Are you experiencing anything else alongside this — "
        "fever, nausea, vomiting, dizziness, or other symptoms?"))

    # ── 6. Complaint + condition-specific (highest relevant one) ─────────────
    if _has("diabetes") and any(w in c for w in ("dizzy", "dizziness", "faint", "weak", "shak")):
        dims.append(("complaint_specific",
            "When did you last check your blood sugar, and did you take your "
            "diabetes medication today?"))
    elif (_has("hypertension") or _has("blood pressure")) and any(w in c for w in ("head", "headache", "dizzy", "vision")):
        dims.append(("complaint_specific",
            "Have you taken your blood pressure medication today, and have you "
            "noticed any changes in your vision?"))
    elif (_has("heart") or _has("cardiac")) and any(w in c for w in ("chest", "breath", "palpitat")):
        dims.append(("complaint_specific",
            "Does the discomfort spread to your arm, jaw, or back?"))
    elif _has("asthma") and any(w in c for w in ("breath", "wheeze", "cough")):
        dims.append(("complaint_specific",
            "Have you used your rescue inhaler today, and if so how many times?"))
    elif female and married and any(w in c for w in ("abdomen", "stomach", "pelvic", "pelvis")):
        dims.append(("complaint_specific",
            "Is there any possibility you could be pregnant?"))
    elif age and int(age) >= 60 and any(w in c for w in ("fall", "dizzy", "balance", "weak")):
        dims.append(("complaint_specific",
            "Have you had any recent falls, and are you steady on your feet?"))
    else:
        dims.append(("modifying_factors",
            "Does anything make it better or worse — rest, movement, "
            "eating, or a certain position?"))

    return dims


def _determine_routing(ctx: dict) -> str:
    """
    GP-first rule:
      First visit (or new complaint) → General Physician
      Second visit with same complaint specialization → recommended specialist
      Emergency flag → EMERGENCY

    Uses recent_case_notes (fetched automatically after patient lookup)
    and the triage-recommended specialist to decide.
    """
    if ctx.get("routing_decision"):
        return ctx["routing_decision"]   # already decided this session

    recommended = ctx.get("recommended_specialist") or ""
    notes       = (ctx.get("recent_case_notes") or "").lower()

    # Emergency always overrides
    if "emergency" in recommended.lower():
        return "EMERGENCY"

    # No prior notes → definitely first visit → GP
    if not notes or "no case notes" in notes or "no recent" in notes:
        return "General Physician"

    # Prior notes exist — check if same specialist was seen before
    # e.g. recommended = "Neurologist" and prior notes mention Neurologist → returning
    if recommended and recommended.lower() not in ("general physician", "gp", ""):
        # strip to the first word for a fuzzy match (e.g. "Neurologist" in notes)
        specialist_keyword = recommended.split("/")[0].strip().lower()
        if specialist_keyword and specialist_keyword in notes:
            return recommended   # returning patient for same complaint → specialist

    # Prior notes but different complaint → GP again
    return "General Physician"


def load_booking_context(session_id: str) -> dict:
    try:
        path = _booking_ctx_path(session_id)
    except ValueError as e:
        print(f"⚠️  [BookingCtx] {e} — creating in-memory context")
        return _empty_booking_context(session_id)

    if path.exists():
        try:
            data  = json.loads(path.read_text(encoding="utf-8"))
            empty = _empty_booking_context(session_id)
            for k, v in empty.items():
                data.setdefault(k, v)
            print(f"📂 [BookingCtx] Loaded {path.name}  step={data['step']}")
            return data
        except Exception as e:
            print(f"⚠️  [BookingCtx] Load failed ({e}) — starting fresh")
    return _empty_booking_context(session_id)


def save_booking_context(session_id: str, ctx: dict) -> None:
    ctx["last_updated"] = datetime.now(PKT).isoformat()
    try:
        path = _booking_ctx_path(session_id)
        path.write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"💾 [BookingCtx] step={ctx['step']} → saved {path.name}")
    except Exception as e:
        print(f"❌ [BookingCtx] Save failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# BOOKING STATE MACHINE
# ═══════════════════════════════════════════════════════════════════

def _advance_step(ctx: dict) -> None:
    p                  = ctx["patient"]
    d                  = ctx["selected_doctor"]
    s                  = ctx["pending_slot"]
    history_checked    = ctx.get("patient_history_checked", False)
    history_available  = ctx.get("patient_history_available", False)

    print(f"\n🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔀 [AdvanceStep] patient.id          = {p['id']}")
    print(f"🔀 [AdvanceStep] patient.name        = {p['name']}")
    print(f"🔀 [AdvanceStep] patient.phone       = {p['phone']}")
    print(f"🔀 [AdvanceStep] doctor.id           = {d['id']}")
    print(f"🔀 [AdvanceStep] doctor.name         = {d['name']}")
    print(f"🔀 [AdvanceStep] pending_slot        = {s['date']} at {s['time']}")
    print(f"🔀 [AdvanceStep] confirmed           = {ctx['appointment']['confirmed']}")
    print(f"🔀 [AdvanceStep] history_checked     = {history_checked}")
    print(f"🔀 [AdvanceStep] history_available   = {history_available}")
    print(f"🔀 [AdvanceStep] triage_completed    = {ctx.get('triage_completed')}")

    if ctx["appointment"]["confirmed"]:
        ctx["step"] = "completed"
        print(f"🔀 [AdvanceStep] → step = 'completed' ✅")
        print(f"🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return

    # Human handoff takes priority over everything else
    if ctx.get("human_handoff_confirmed"):
        ctx["step"] = "await_human"
        print(f"🔀 [AdvanceStep] → step = 'await_human' (patient in waitlist)")
        print(f"🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return

    if p["id"] and d["id"] and s["date"] and s["time"]:
        ctx["step"] = "await_confirmation"
        print(f"🔀 [AdvanceStep] → step = 'await_confirmation' ✅")
        print(f"🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return

    # ── Phase 0.5: collect patient history before triage ─────────────────────
    # Stay in collect_patient_history until required_history_fields is an empty list.
    # None means it hasn't been computed yet (still waiting for patient.id).
    required = ctx.get("required_history_fields")
    if not p["id"] or required is None or len(required) > 0:
        ctx["step"] = "collect_patient_history"
        reason = []
        if not p["id"]:                      reason.append("no patient.id")
        if required is None:                 reason.append("fields not yet computed")
        elif len(required) > 0:              reason.append(f"{len(required)} field(s) remaining: {required}")
        print(f"🔀 [AdvanceStep] → step = 'collect_patient_history' ({', '.join(reason)})")
        print(f"🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return

    # ── History confirmed — proceed to triage ────────────────────────────────
    if not ctx.get("triage_completed"):
        ctx["step"] = "collect_patient"
        print(f"🔀 [AdvanceStep] → step = 'collect_patient' (history done, triage pending)")
        print(f"🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return

    if not d["id"]:
        ctx["step"] = "collect_doctor"
        print(f"🔀 [AdvanceStep] → step = 'collect_doctor' ❌ missing doctor.id")
        print(f"🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return

    ctx["step"] = "collect_slot"
    missing = []
    if not s["date"]: missing.append("date")
    if not s["time"]: missing.append("time")
    print(f"🔀 [AdvanceStep] → step = 'collect_slot' ❌ missing: {', '.join(missing)}")
    print(f"🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


def _get_booking_directive(ctx: dict, today: str, tomorrow: str) -> str:
    step = ctx.get("step", "collect_patient")
    p    = ctx["patient"]
    d    = ctx["selected_doctor"]
    s    = ctx["pending_slot"]

    lines = ["━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"]
    lines.append("CURRENT BOOKING STATE:")
    lines.append(f"  step                = {step}")
    lines.append(f"  prime_complaint     = {ctx.get('prime_complaint') or '–'}")
    lines.append(f"  specialist_needed   = {ctx.get('recommended_specialist') or '–'}")
    lines.append(f"  triage_completed    = {ctx.get('triage_completed')}")
    lines.append(f"  triage_questions    = {ctx.get('triage_questions_asked', 0)}")
    lines.append(
        f"  patient             = {p['name']} (ID={p['id']}, phone={p['phone']})"
        if p["id"] else "  patient             = not yet looked up"
    )
    lines.append(
        f"  doctor              = Dr. {d['name']} (ID={d['id']}, {d['specialization']})"
        if d["id"] else "  doctor              = not yet selected"
    )
    lines.append(
        f"  pending_slot        = {s['date']} at {s['time']}"
        if (s["date"] or s["time"]) else "  pending_slot        = not yet chosen"
    )
    if ctx["appointment"]["confirmed"]:
        lines.append(f"  appointment         = CONFIRMED (ID={ctx['appointment']['booking_id']})")
    lines.append("")

    # ── Human handoff — takes priority over all step directives ──────────────
    if ctx.get("human_handoff_pending") and not ctx.get("human_handoff_confirmed"):
        complaint = ctx.get("prime_complaint") or ctx.get("initial_complaint_hint") or "their concern"
        lines.append("YOUR NEXT ACTION: Patient has requested a human agent.")
        lines.append("  Tell the patient:")
        lines.append("  'The current estimated wait time for a human agent is approximately")
        lines.append("  10–15 minutes. Would you like to be added to the queue?'")
        lines.append("")
        lines.append("  If patient says YES:")
        lines.append(f"    Call add_to_waitlist(session_id='{ctx.get('session_id', '')}',")
        lines.append(f"    patient_id={p.get('id')}, patient_name='{p.get('name')}',")
        lines.append(f"    phone='{p.get('phone')}', complaint='{complaint}')")
        lines.append("    Then output: [HUMAN_CONFIRMED]")
        lines.append("")
        lines.append("  If patient says NO:")
        lines.append("    Resume the normal booking flow from where you left off.")
        lines.append("  ⛔ DO NOT start triage or booking until patient answers this question.")

    # ── Phase 0.5: patient history ────────────────────────────────────────────
    elif step == "collect_patient_history":
        pid   = p.get("id")
        phone = p.get("phone")

        lines.append("YOUR NEXT ACTION: Collect patient profile and medical history.")
        lines.append("  ⛔ ONLY these tools are allowed right now:")
        lines.append("     lookup_customer_profile · register_customer_profile")
        lines.append("     update_patient_demographics · get_patient_history · save_patient_history")
        lines.append("  ⛔ DO NOT call any booking or specialist tools.")
        lines.append("  ⛔ DO NOT output [SYMPTOM_LOGGED:] or [START_TRIAGE] yet.")
        lines.append("")

        if not phone and not pid:
            lines.append("  ▶ SUB-STEP 1: Ask for phone number.")
            lines.append("    Greet the patient warmly and ask: 'Could I get your phone number?'")
            lines.append("    Even if they mentioned a symptom — ask for phone first.")

        elif phone and not pid:
            lines.append(f"  ✓ Phone collected: {phone}")
            lines.append("  ▶ SUB-STEP 2: Call lookup_customer_profile now.")

        elif pid and ctx.get("required_history_fields") is None:
            lines.append(f"  ✓ Patient identified: {p.get('name')} (ID={pid})")
            lines.append("  ▶ SUB-STEP 3: Call get_patient_history now to load existing history.")

        elif pid and ctx.get("required_history_fields") is not None:
            remaining = ctx.get("required_history_fields", [])
            collected = ctx.get("collected_this_session", [])

            if remaining:
                current_field = remaining[0]
                question      = _QUESTION_FOR_FIELD.get(
                    current_field, f"Please tell me about: {current_field}"
                )
                criteria = _FIELD_COMPLETE_CRITERIA.get(current_field, "a clear answer")

                lines.append(f"  ✓ Patient: {p.get('name')} (ID={pid})")
                lines.append(f"  ✓ Collected this session: {collected or 'none yet'}")
                lines.append(f"  ▶ Currently collecting: [{current_field}]")

                turns_used = ctx.get("history_field_turn_count", {}).get(current_field, 0)
                turns_left = _MAX_TURNS_PER_HISTORY_FIELD - turns_used
                lines.append(f"    Turn {turns_used}/{_MAX_TURNS_PER_HISTORY_FIELD} on this field — {turns_left} turn(s) remaining before auto-advance")
                lines.append(f"    Initial question: '{question}'")
                lines.append(f"    Complete when you have: {criteria}")
                lines.append("")
                lines.append("  IMPORTANT RULES for this field:")
                lines.append("  1. Ask the question if not yet asked.")
                lines.append("  2. If the patient's answer is PARTIAL — only answered part of")
                lines.append(f"     what's needed — ask ONE natural follow-up to get the rest.")
                lines.append("  3. Only call save_patient_history when the COMPLETE criteria")
                lines.append("     above is satisfied. Not before.")
                lines.append("  4. Do NOT jump to the next topic until this field is saved.")
                lines.append("  5. If the patient goes off-topic, gently redirect:")
                lines.append(f"     'I'll note that — just to finish up, {question}'")
                lines.append("")

                _DEMOGRAPHIC_FIELDS = {"age", "gender", "marital_status"}
                if current_field in _DEMOGRAPHIC_FIELDS:
                    lines.append(f"  When complete → call update_patient_demographics(patient_id={pid}, {current_field}=<answer>)")
                else:
                    lines.append(f"  When complete → call save_patient_history(patient_id={pid}, {current_field}=<full answer summary>)")

                lines.append(f"  Fields remaining after this: {remaining[1:] or 'none — all done'}")

            else:
                lines.append(f"  ✓ ALL FIELDS COLLECTED for {p.get('name')}.")
                lines.append("  ▶ Now ask: 'What brings you in today?'")
                lines.append("    When the patient mentions a symptom output:")
                lines.append("    [SYMPTOM_LOGGED: <symptom>]")
                lines.append("    [START_TRIAGE]")

    # ── GUARD: only trigger triage when step is collect_patient AND flag is unset.
    # If step has already advanced (collect_doctor, collect_slot, etc.) triage
    # clearly happened — don't re-trigger it even if the flag was somehow lost.
    elif not ctx.get("triage_completed") and step == "collect_patient":
        lines.append("YOUR NEXT ACTION: Medical Triage.")
        lines.append("  If the user mentions ANY medical symptom, you MUST output exactly:")
        lines.append("  [SYMPTOM_LOGGED: <symptom>]")
        lines.append("  [START_TRIAGE]")
        lines.append("  DO NOT ask for phone number, name, or try to book until triage is finished!")

    elif step == "collect_patient":
        if not ctx.get("triage_completed"):
            hint = ctx.get("initial_complaint_hint")
            lines.append("YOUR NEXT ACTION: Start medical triage.")
            lines.append(f"  Patient {p.get('name') or 'identified'} — history collection complete.")
            if hint:
                lines.append(f"  ✓ Patient already mentioned their complaint: '{hint}'")
                lines.append(f"  DO NOT ask 'what brings you in today?' — use the hint above.")
                lines.append(f"  You may say: 'I see you came in for {hint}. Let me start your assessment.'")
                lines.append(f"  Then immediately output:")
                lines.append(f"  [SYMPTOM_LOGGED: {hint}]")
                lines.append(f"  [START_TRIAGE]")
            else:
                lines.append("  Ask: 'What brings you in today?'")
                lines.append("  When the patient mentions a symptom output BOTH:")
                lines.append("  [SYMPTOM_LOGGED: <symptom>]")
                lines.append("  [START_TRIAGE]")
            lines.append("  ⛔ Triage MUST happen before any booking. DO NOT skip to doctor selection.")

        else:
            routing = _determine_routing(ctx)
            ctx["routing_decision"] = routing
            if routing == "EMERGENCY":
                lines.append("YOUR NEXT ACTION: ⚠️ EMERGENCY — Escalate immediately.")
                lines.append("  Tell the patient to call 115 or go to the nearest ER now.")
                lines.append("  Do NOT proceed with booking.")
            else:
                lines.append("YOUR NEXT ACTION: Triage complete. Start the booking flow.")
                lines.append(f"  Routing decision → {routing}")
                if routing == "General Physician":
                    lines.append("  Reason: first visit or new complaint type — GP protocol.")
                else:
                    lines.append("  Reason: returning patient with prior visit for same complaint.")
                lines.append(f"  1. Tell the patient: triage complete, we recommend seeing a {routing}.")
                lines.append(f"  2. Patient already identified: {p.get('name')} (ID={p.get('id')}).")
                lines.append(f"  3. Proceed to doctor selection — DO NOT ask for phone number again.")

    elif step == "await_human":
        position = ctx.get("waitlist_position", "?")
        lines.append("YOUR NEXT ACTION: Human handoff — patient is in the waitlist.")
        lines.append(f"  Patient position in queue: #{position}")
        lines.append("  Tell the patient: 'You are number {position} in the queue.")
        lines.append("  A human agent will be with you shortly. Thank you for your patience.'")
        lines.append("  If the patient says they want to cancel: call cancel_waitlist(session_id)")
        lines.append("  and resume the normal booking flow.")


    elif step == "collect_doctor":
        spec = ctx.get("recommended_specialist", "")
        if d["id"]:
            # Doctor already resolved — should not normally land here, but be safe
            lines.append("YOUR NEXT ACTION: Doctor is already selected — do NOT call any doctor tools.")
            lines.append(f"  Doctor: Dr. {d['name']} (ID={d['id']}, {d['specialization']})")
            lines.append("  Just ask the patient which date they prefer for their appointment.")
        else:
            lines.append("YOUR NEXT ACTION: Find and select a doctor.")
            lines.append(f"  1. Call get_doctors_by_specialization(specialization='{spec}') to list available doctors.")
            lines.append("  2. If only one doctor is returned, auto-select them and immediately ask for a preferred date.")
            lines.append("  3. If multiple doctors, present them and ask the patient to pick one.")
            lines.append("  ⛔ DO NOT call get_doctor_profile — the ID is extracted automatically from the list.")
            lines.append("  ⛔ DO NOT call get_doctors_by_specialization more than once.")

    elif step == "collect_slot":
        d_name     = d.get("name", "Unknown")
        d_id       = d.get("id")
        schedule   = ctx.get("doctor_schedule", [])

        lines.append("YOUR NEXT ACTION: Help the patient pick a date and time slot.")
        lines.append(f"  TODAY = {today} | TOMORROW = {tomorrow}")
        lines.append("")

        if schedule:
            lines.append(f"  ✅ Dr. {d_name}'s weekly schedule (already fetched — DO NOT call get_doctor_profile again):")
            for entry in schedule:
                lines.append(f"     {entry['day']}: {entry['start']} – {entry['end']}")
            lines.append("")
            lines.append("  WORKFLOW:")
            lines.append("  1. If patient has NOT said which day they want → present the schedule above and ask.")
            lines.append("  2. Once patient states a day (e.g. 'monday') or day+time (e.g. 'monday 9:30 am'):")
            lines.append(f"     Call: find_provider_availability(doctor_id={d_id}, date='<day>', time='<HH:MM or omit>')")
            lines.append("  3. Show the returned free slots. Ask the patient to confirm one.")
            lines.append("  4. Once patient confirms a specific slot → show full booking summary and ask YES/NO.")
        else:
            lines.append(f"  ⛔ DO NOT call get_doctor_profile or any doctor-lookup tool.")
            lines.append(f"  STEP 1: Ask the patient which day they would like to see Dr. {d_name}.")
            lines.append(f"  STEP 2: Once they give a day, call find_provider_availability(doctor_id={d_id}, date='<chosen_day>').")
            lines.append("  STEP 3: Show free slots, ask patient to pick one.")

        lines.append("")
        lines.append("  ⛔ NEVER call find_provider_availability without a date.")
        lines.append(f"  ⛔ ALWAYS pass doctor_id={d_id} — never leave doctor unspecified.")
        lines.append("  ⛔ DO NOT call get_doctor_schedule — use find_provider_availability.")
        lines.append("  ⛔ NEVER present appointment time options from your own knowledge.")
        lines.append("     You MUST call find_provider_availability first — only show slots it returns.")

    if ctx.get("triage_completed"):
        lines.append("")
        lines.append("⛔ TRIAGE IS DONE — ABSOLUTE PROHIBITION:")
        lines.append(f"  Complaint already logged: '{ctx.get('prime_complaint', '–')}'")
        lines.append("  NEVER emit [SYMPTOM_LOGGED:...] or [START_TRIAGE] again under any circumstances.")
        lines.append("  These tags are only valid before triage. Emitting them now will break the flow.")

    elif step == "await_confirmation":
        lines.append("YOUR NEXT ACTION: Show summary. Ask 'Shall I confirm? (yes/no)'. Call NO tools.")
        lines.append(f"  Patient : {p['name']} (ID={p['id']})")
        lines.append(f"  Doctor  : Dr. {d['name']} (ID={d['id']})")
        lines.append(f"  Date    : {s['date']} at {s['time']}")
        lines.append("  DO NOT call create_booking — code handles it after YES.")

    elif step == "completed":
        lines.append("YOUR NEXT ACTION: Booking is confirmed.")
        lines.append(f"  1. Warmly confirm: Dr. {d['name']}, {s['date']} at {s['time']}, Booking ID={ctx['appointment']['booking_id']}")
        lines.append("  2. Ask ONE time: 'Is there anything else I can help you with today?'")
        lines.append("  3. If the patient says no / goodbye / done / nothing else → output exactly: [END_CALL]")
        lines.append("     Do NOT say goodbye yourself — the system will send the goodbye message.")
        lines.append("  4. If they have a real follow-up question, answer it, then ask once more.")
        lines.append("  ⛔ DO NOT call any tools. ⛔ DO NOT output [END_CALL] unless patient declines.")

    lines.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# STATE
# ═══════════════════════════════════════════════════════════════════

class ConversationState(TypedDict, total=False):
    session_id:              str
    messages:                Annotated[Sequence[BaseMessage], operator.add]
    triage_active:           bool
    interaction_completed:   bool
    extracted_symptom:       str
    patient_profile:         dict[str, Any]
    final_diagnostic_report: str
    triage_qa:               Annotated[list[str], operator.add]
    booking_context:         dict


# ═══════════════════════════════════════════════════════════════════
# LLM + TOOL MAP
# ═══════════════════════════════════════════════════════════════════

_llm_with_tools: Any = None



# ── Per-phase tool whitelist ──────────────────────────────────────────────────
# Each step gets exactly the tools it needs. The LLM is bound with only those
# tools so it cannot generate calls for tools outside its current phase.
_PHASE_TOOL_NAMES: dict[str, list[str]] = {
    # Phase 0.5 — identity + history only
    "collect_patient_history": [
        "lookup_customer_profile",
        "register_customer_profile",
        "update_patient_demographics",
        "get_patient_history",
        "save_patient_history",
    ],
    # Phase 1-4 — chief complaint → triage → specialist recommendation
    "collect_patient": [
        "recommend_specialist_tool",
    ],
    # Booking phase — find and select a doctor
    "collect_doctor": [
        "recommend_specialist_tool",
        "get_doctors_by_specialization",
        "get_doctor_profile",
        "get_doctor_schedule",
    ],
    # Booking phase — find available slot
    "collect_slot": [
        "get_doctor_profile",
        "get_doctor_schedule",
        "find_provider_availability",
    ],
    # Booking phase — confirm
    "await_confirmation": [
        "create_booking",
        "find_provider_availability",
    ],
    # Human waitlist phase
    "await_human": [
        "get_waitlist_position",
        "cancel_waitlist",
    ],
    "completed": [],
}

_llm_by_step:   dict[str, Any]  = {}
_llm_with_tools: Any            = None   # kept for backward-compat; not used in main flow


def _get_llm_for_step(step: str):
    """Return an LLM bound with only the tools allowed for this step."""
    global _llm_by_step
    if step not in _llm_by_step:
        allowed_names = _PHASE_TOOL_NAMES.get(step)
        if allowed_names is None:
            # Unknown step — bind everything as a safe fallback
            tools = ALL_TOOLS
            label = f"{step}(all-fallback)"
        else:
            tools = [t for t in ALL_TOOLS if t.name in allowed_names]
            label = step
        print(f"🔧 [LLM] Binding tools for step='{label}': {[t.name for t in tools]}")
        base = get_llm(temperature=0.1)
        _llm_by_step[step] = base.bind_tools(tools) if tools else base
    return _llm_by_step[step]


def _get_llm_with_tools():
    """Legacy helper — calls _get_llm_for_step with unknown step (all tools)."""
    return _get_llm_for_step("__all__")


_HANDOFF_TOOL_NAMES = {"add_to_waitlist", "get_waitlist_position", "cancel_waitlist"}


def _get_llm_with_handoff_tools(base_step: str):
    """
    LLM binding that adds waitlist tools on top of the base step's tools.
    Used when human_handoff_pending=True so Qwen can call add_to_waitlist
    from any step without Guard 0 blocking it.
    Cached per base_step.
    """
    cache_key = f"__handoff__{base_step}"
    if cache_key not in _llm_by_step:
        base_names = set(_PHASE_TOOL_NAMES.get(base_step, []))
        all_names  = base_names | _HANDOFF_TOOL_NAMES
        tools      = [t for t in ALL_TOOLS if t.name in all_names]
        base       = get_llm(temperature=0.1)
        _llm_by_step[cache_key] = base.bind_tools(tools) if tools else base
        print(f"🔧 [LLM] Binding handoff tools for step='{base_step}': {[t.name for t in tools]}")
    return _llm_by_step[cache_key]


# Tool map used by tool_executor_node to look up and call the actual function
_TOOL_MAP: dict[str, Any] = {t.name: t for t in ALL_TOOLS}

_SLOT_EXTRACTOR_LLM: Any = None

def _get_slot_extractor_llm():
    """Lazy-init a raw (no tools) LLM used only for slot date/time extraction."""
    global _SLOT_EXTRACTOR_LLM
    if _SLOT_EXTRACTOR_LLM is None:
        _SLOT_EXTRACTOR_LLM = get_llm(temperature=0.0)
    return _SLOT_EXTRACTOR_LLM


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

MAX_HISTORY_MESSAGES = 20   # Raised from 12 — triage now runs up to 8 rounds
MAX_TOOLS_PER_TURN   = 2    # Hard cap — prevents Groq 400 from long tool chains

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|confirm|book\s*it|go\s*ahead|sure|ok|okay|"
    r"haan|ji|bilkul|theek\s*hai|kar\s*do|confirm\s*kar|"
    r"book\s*(this|the|my)?\s*appointment|"
    r"please\s*book|do\s*it|proceed|sounds\s*good|perfect|great|"
    r"appointment\s*(book\s*kar|confirm\s*kar|kar\s*do))\b",
    re.IGNORECASE,
)
_INVALID_VALUES = {"unknown", "none", "null", "n/a", "", "undefined", "?"}
_TOOL_REQUIRED_ARGS = {
    "create_booking":          {"patient_id": "patient ID", "doctor_id": "doctor ID", "date": "date", "time": "time"},
    "lookup_customer_profile": {"phone": "patient phone number"},
}

# Used by _extract_from_tool_result — still needed for tool result parsing
_SCHEDULE_RE = re.compile(
    r"\b(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r":\s*(\d{1,2}:\d{2})\s*[\u2013\-]\s*(\d{1,2}:\d{2})",
    re.IGNORECASE,
)
_DATE_RE    = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
_BOOKING_ID_RE = re.compile(r"(?i)Appointment\s+ID:\s*(\d+)")


def _is_invalid(val: Any) -> bool:
    return val is None or str(val).strip().lower() in _INVALID_VALUES


def _trim_messages(messages: list) -> list:
    if len(messages) > MAX_HISTORY_MESSAGES:
        trimmed = messages[-MAX_HISTORY_MESSAGES:]
        print(f"✂️  [History] Trimmed {len(messages)} → {len(trimmed)}")
        return trimmed
    return messages


def _build_safe_messages(raw: list) -> list:
    """
    Prepare message history for Qwen (supervisor).
    Removes:
      - AIMessages that have tool_calls (replaced with content-only version)
      - ToolMessages that have no preceding AIMessage with tool_calls (orphans)
      - Empty AIMessages (blank content, no tool_calls) — these are triage silencers
    This prevents Groq 400 'tool_use_failed' from malformed tool sequences in history.
    """
    # First pass: which tool_call IDs are legitimately present?
    valid_tool_call_ids: set[str] = set()
    for msg in raw:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                valid_tool_call_ids.add(tc.get("id", ""))

    safe = []
    for msg in raw:
        if msg.type == "ai":
            if getattr(msg, "tool_calls", None):
                # Flatten — keep content only, drop the tool_calls
                content = str(msg.content) if msg.content else ""
                safe.append(AIMessage(content=content))
            elif not str(msg.content).strip():
                # Empty AI message (e.g. triage silencer) — skip entirely
                continue
            else:
                safe.append(msg)
        elif msg.type == "tool":
            # Only include if there was a matching tool_call AIMessage
            tid = getattr(msg, "tool_call_id", None)
            if tid and tid in valid_tool_call_ids:
                safe.append(msg)
            else:
                print(f"✂️  [SafeMsg] Dropped orphaned ToolMessage tool_call_id={tid}")
        else:
            safe.append(msg)
    return safe


def _count_tools_since_last_human(messages: list) -> int:
    count = 0
    for msg in reversed(messages):
        if msg.type == "human":
            break
        if msg.type == "tool":
            count += 1
    return count


def _last_human_text(messages: list) -> str:
    for m in reversed(messages):
        if m.type == "human":
            return str(m.content).strip()
    return ""


def _invoke_with_retry(llm, msgs: list, retries: int = 2, delay: float = 3.0):
    last_exc = None
    for attempt in range(1, retries + 2):
        try:
            if attempt > 1:
                print(f"🔄 [LLM] Retry {attempt}…")
                time.sleep(delay)
            return llm.invoke(msgs)
        except Exception as e:
            err = str(e)
            if any(k in err for k in ["ReadError", "10054", "ConnectionError", "RemoteDisconnected", "forcibly closed"]):
                print(f"⚠️  [LLM] Connection error (attempt {attempt}): {err[:100]}")
                last_exc = e
            else:
                raise
    raise last_exc


def _validate_tool_calls(response) -> object | None:
    for tc in (getattr(response, "tool_calls", None) or []):
        name    = tc.get("name", "")
        args    = tc.get("args", {})
        missing = [desc for arg, desc in _TOOL_REQUIRED_ARGS.get(name, {}).items() if _is_invalid(args.get(arg))]
        if missing:
            missing_str = " and ".join(missing)
            print(f"🚫 [Validator] Blocked '{name}' — missing: {missing_str}")
            return AIMessage(content=f"I still need {missing_str} to proceed. Could you share that?")
    return None


def _print_messages(messages: list, label: str = "Messages") -> None:
    print(f"\n📋 [{label}] {len(messages)} msgs:")
    for i, msg in enumerate(messages):
        role    = msg.__class__.__name__.replace("Message", "")
        content = str(msg.content)[:120].replace("\n", " ")
        tcalls  = f" tools={[tc['name'] for tc in msg.tool_calls]}" if getattr(msg, "tool_calls", None) else ""
        print(f"   [{i}] {role}{tcalls}: {content}")


# ═══════════════════════════════════════════════════════════════════
# PER-TOOL BOOKING CONTEXT EXTRACTORS
# ═══════════════════════════════════════════════════════════════════

_PATIENT_ID_RE        = re.compile(r"['\"]id['\"]\s*:\s*(\d+)")
_PATIENT_NAME_RE      = re.compile(r"['\"]name['\"]\s*:\s*['\"]([A-Za-z][A-Za-z ]{1,40}?)['\"]")
_PATIENT_AGE_RE       = re.compile(r"['\"]age['\"]\s*:\s*(\d+(?:\.\d+)?)")
_PATIENT_GENDER_RE    = re.compile(r"['\"]gender['\"]\s*:\s*['\"]([^'\"]{1,20})['\"]")
_PATIENT_MARITAL_RE   = re.compile(r"['\"]marital_status['\"]\s*:\s*['\"]([^'\"]{1,20})['\"]")
_DOCTOR_ID_RE    = re.compile(r"\(ID:\s*(\d+)\)")
_DOCTOR_NAME_RE  = re.compile(r"Dr\.\s+([A-Za-z][A-Za-z .]{1,40}?)(?:\s*[\(|,\n]|$)")
_DOCTOR_SPEC_RE  = re.compile(r"\|\s*([A-Za-z][A-Za-z /]+?)\s*(?:Fee|$|\|)")
_SPECIALIST_RE   = re.compile(r"(?i)(?:Recommended specialist|specialist|see a?n?)\s*[:\-]\s*([A-Za-z][A-Za-z /]{3,40})")


def _extract_from_tool_result(tool_name: str, tool_args: dict, result_str: str, ctx: dict) -> bool:
    changed = False
    print(f"\n🔍 [ToolExtract] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔍 [ToolExtract] tool        = '{tool_name}'")
    print(f"🔍 [ToolExtract] args        = {tool_args}")
    print(f"🔍 [ToolExtract] result[:200]= {result_str[:200]}")
    print(f"🔍 [ToolExtract] BEFORE: patient.id={ctx['patient']['id']}  doctor.id={ctx['selected_doctor']['id']}  slot={ctx['pending_slot']}")

    if tool_name == "lookup_customer_profile":
        print(f"   result preview: {result_str[:300]}")
        if not ctx["patient"]["phone"]:
            phone = tool_args.get("phone") or tool_args.get("name")
            if phone:
                ctx["patient"]["phone"] = str(phone)
                changed = True
                print(f"   → patient.phone = '{phone}'")
        if not ctx["patient"]["id"]:
            m = _PATIENT_ID_RE.search(result_str)
            if m:
                ctx["patient"]["id"] = int(m.group(1))
                changed = True
                print(f"   → patient.id = {ctx['patient']['id']}")
            else:
                print(f"   ⚠️  COULD NOT extract patient.id — regex found nothing in: {result_str[:200]}")
        if not ctx["patient"]["name"]:
            m = _PATIENT_NAME_RE.search(result_str)
            if m:
                ctx["patient"]["name"] = m.group(1).strip()
                changed = True
                print(f"   → patient.name = '{ctx['patient']['name']}'")
        # Extract demographics if stored in patients table
        if ctx["patient"].get("age") is None:
            m = _PATIENT_AGE_RE.search(result_str)
            if m:
                ctx["patient"]["age"] = int(float(m.group(1)))
                changed = True
                print(f"   → patient.age = {ctx['patient']['age']}")
        if not ctx["patient"].get("gender"):
            m = _PATIENT_GENDER_RE.search(result_str)
            if m:
                ctx["patient"]["gender"] = m.group(1).strip().lower()
                changed = True
                print(f"   → patient.gender = '{ctx['patient']['gender']}'")
        if not ctx["patient"].get("marital_status"):
            m = _PATIENT_MARITAL_RE.search(result_str)
            if m:
                ctx["patient"]["marital_status"] = m.group(1).strip().lower()
                changed = True
                print(f"   → patient.marital_status = '{ctx['patient']['marital_status']}'")

    elif tool_name in ("get_doctor_profile", "get_doctor_schedule"):
        print(f"   result preview: {result_str[:300]}")
        if not ctx["selected_doctor"]["id"]:
            if tool_args.get("doctor_id"):
                ctx["selected_doctor"]["id"] = int(tool_args["doctor_id"])
                changed = True
            else:
                m = _DOCTOR_ID_RE.search(result_str)
                if m:
                    ctx["selected_doctor"]["id"] = int(m.group(1))
                    changed = True
            print(f"   → doctor.id = {ctx['selected_doctor']['id']}")
        if not ctx["selected_doctor"]["name"]:
            if tool_args.get("doctor_name"):
                ctx["selected_doctor"]["name"] = str(tool_args["doctor_name"])
                changed = True
            else:
                m = _DOCTOR_NAME_RE.search(result_str)
                if m:
                    ctx["selected_doctor"]["name"] = m.group(1).strip()
                    changed = True
            print(f"   → doctor.name = '{ctx['selected_doctor']['name']}'")
        if not ctx["selected_doctor"]["specialization"]:
            m = _DOCTOR_SPEC_RE.search(result_str)
            if m:
                ctx["selected_doctor"]["specialization"] = m.group(1).strip()
                changed = True
            print(f"   → doctor.spec = '{ctx['selected_doctor']['specialization']}'")
        # ── Cache the weekly schedule so we never re-fetch it ─────
        schedule_entries = [
            {"day": m.group(1).title(), "start": m.group(2), "end": m.group(3)}
            for m in _SCHEDULE_RE.finditer(result_str)
        ]
        if schedule_entries:
            ctx["doctor_schedule"] = schedule_entries
            changed = True
            print(f"   → doctor_schedule cached: {schedule_entries}")

    elif tool_name in ("get_doctors_by_specialization", "query_database_table"):
        print(f"   result preview: {result_str[:300]}")
        if not ctx["selected_doctor"]["specialization"]:
            spec = tool_args.get("specialization")
            if spec:
                ctx["selected_doctor"]["specialization"] = str(spec)
                changed = True
                print(f"   → doctor.spec (from args) = '{spec}'")

        _RAW_DOC_ID_RE   = re.compile(r"['\"]id['\"]\s*:\s*(\d+)")
        _RAW_DOC_NAME_RE = re.compile(r"['\"]name['\"]\s*:\s*['\"]([^'\"]+)['\"]")
        _FMT_DOC_RE      = re.compile(
            r"-\s*Dr\.?\s*([A-Za-z][A-Za-z .]{1,40}?)\s*\(ID:\s*(\d+)\)",
            re.IGNORECASE,
        )

        fmt_matches = _FMT_DOC_RE.findall(result_str)
        raw_ids     = _RAW_DOC_ID_RE.findall(result_str)
        raw_names   = _RAW_DOC_NAME_RE.findall(result_str)

        patient_id_str = str(ctx["patient"]["id"]) if ctx["patient"]["id"] else None
        patient_name   = ctx["patient"]["name"] or ""

        if fmt_matches and len(fmt_matches) == 1 and not ctx["selected_doctor"]["id"]:
            name, doc_id = fmt_matches[0]
            ctx["selected_doctor"]["id"]   = int(doc_id)
            ctx["selected_doctor"]["name"] = name.strip()
            changed = True
            print(f"   → (fmt) doctor.id={doc_id}  doctor.name='{name.strip()}'")
        elif raw_ids and not ctx["selected_doctor"]["id"]:
            doc_ids = [i for i in raw_ids if i != patient_id_str]
            if len(doc_ids) == 1:
                ctx["selected_doctor"]["id"] = int(doc_ids[0])
                changed = True
                print(f"   → (raw) doctor.id={doc_ids[0]}")
            elif doc_ids:
                print(f"   ℹ️  Multiple doctor IDs found {doc_ids} — user must pick one")

        if raw_names and not ctx["selected_doctor"]["name"]:
            doc_names = [n for n in raw_names if n.lower() != patient_name.lower()]
            if len(doc_names) == 1:
                ctx["selected_doctor"]["name"] = doc_names[0].strip()
                changed = True
                print(f"   → (raw) doctor.name='{doc_names[0].strip()}'")

        print(f"   After extract: doctor.id={ctx['selected_doctor']['id']}  doctor.name='{ctx['selected_doctor']['name']}'")

    elif tool_name == "get_patient_history":
        print(f"   result preview: {result_str[:300]}")
        ctx["patient_history_checked"] = True
        if "Patient history found" in result_str:
            ctx["patient_history_available"] = True
            ctx["patient_history_data"]      = result_str
            print(f"   → patient_history_available = True (existing record loaded)")
        else:
            ctx["patient_history_available"] = False
            print(f"   → patient_history_available = False (no history on file)")

        # Compute required fields now that we know both patient demographics and history state
        required = _compute_required_history_fields(ctx["patient"], ctx.get("patient_history_data"))
        ctx["required_history_fields"] = required
        changed = True
        print(f"   → required_history_fields computed: {required}")

        # Proactively fetch recent case notes to support GP/specialist routing later
        if ctx["patient"]["id"] and not ctx.get("recent_case_notes"):
            try:
                from agents.mcp_tools import get_recent_case_notes as _gcn
                notes = _gcn.invoke({"patient_id": ctx["patient"]["id"]})
                ctx["recent_case_notes"] = str(notes)
                print(f"   → recent_case_notes fetched ({len(ctx['recent_case_notes'])} chars)")
            except Exception as e:
                print(f"   ⚠️  Could not fetch recent case notes: {e}")

    elif tool_name == "update_patient_demographics":
        print(f"   result preview: {result_str[:200]}")
        if "updated successfully" in result_str.lower():
            # Mirror the saved values back into ctx["patient"] so subsequent
            # directive renders see the correct demographics immediately.
            for field in ("age", "gender", "marital_status"):
                val = tool_args.get(field)
                if val is not None:
                    ctx["patient"][field] = val
                    # Remove from required list
                    reqs = ctx.get("required_history_fields") or []
                    if field in reqs:
                        reqs.remove(field)
                    ctx["required_history_fields"] = reqs
                    coll = ctx.get("collected_this_session", [])
                    if field not in coll:
                        coll.append(field)
                    ctx["collected_this_session"] = coll

            # Recompute in case demographic change unlocked new gates
            # (e.g. gender+marital_status now set → pregnancy gate may open)
            ctx["required_history_fields"] = _compute_required_history_fields(
                ctx["patient"], ctx.get("patient_history_data")
            )
            # Reset counter for fields that were just saved
            counts = ctx.setdefault("history_field_turn_count", {})
            for field in ("age", "gender", "marital_status"):
                if tool_args.get(field) is not None:
                    counts[field] = 0
            changed = True
            print(f"   → demographics updated; required_history_fields recomputed: {ctx['required_history_fields']}")

    elif tool_name == "save_patient_history":
        print(f"   result preview: {result_str[:200]}")
        if "saved successfully" in result_str.lower():
            ctx["patient_history_available"] = True
            ctx["patient_history_checked"]   = True

            # Mark saved fields as collected
            _HISTORY_FIELDS = {
                "chronic_conditions", "medications", "drug_allergies",
                "general_allergies", "family_history", "smoking_status",
                "pregnancy_status", "lmp_date", "menstrual_history",
                "obstetric_history", "fall_history", "vaccination_status",
            }
            reqs = ctx.get("required_history_fields") or []
            coll = ctx.get("collected_this_session", [])
            for field in _HISTORY_FIELDS:
                if tool_args.get(field) is not None and field in reqs:
                    reqs.remove(field)
                    if field not in coll:
                        coll.append(field)
            ctx["required_history_fields"] = reqs
            ctx["collected_this_session"]   = coll
            # Reset turn counter for saved fields so re-collection gets a fresh count
            counts = ctx.setdefault("history_field_turn_count", {})
            for field in _HISTORY_FIELDS:
                if tool_args.get(field) is not None:
                    counts[field] = 0
            changed = True
            print(f"   → history saved; required_history_fields remaining: {reqs}")

            # Fetch recent case notes if not already done
            if ctx["patient"]["id"] and not ctx.get("recent_case_notes"):
                try:
                    from agents.mcp_tools import get_recent_case_notes as _gcn
                    notes = _gcn.invoke({"patient_id": ctx["patient"]["id"]})
                    ctx["recent_case_notes"] = str(notes)
                    print(f"   → recent_case_notes fetched ({len(ctx['recent_case_notes'])} chars)")
                except Exception as e:
                    print(f"   ⚠️  Could not fetch recent case notes: {e}")
        else:
            print(f"   ⚠️  save_patient_history did not confirm success: {result_str}")

    elif tool_name == "add_to_waitlist":
        print(f"   result preview: {result_str[:200]}")
        if "position:" in result_str.lower():
            m = re.search(r"position[:\s]+(\d+)", result_str, re.IGNORECASE)
            if m:
                ctx["waitlist_position"]   = int(m.group(1))
            ctx["human_handoff_confirmed"] = True
            ctx["human_handoff_pending"]   = False
            changed = True
            print(f"   → waitlist confirmed, position={ctx.get('waitlist_position')}")

    elif tool_name == "cancel_waitlist":
        if "removed" in result_str.lower() or "cancelled" in result_str.lower():
            ctx["human_handoff_confirmed"] = False
            ctx["human_handoff_pending"]   = False
            ctx["waitlist_position"]        = None
            _advance_step(ctx)
            changed = True
            print(f"   → waitlist cancelled, resuming normal flow")
        print(f"   result preview: {result_str[:200]}")
        if not ctx["recommended_specialist"]:
            m = _SPECIALIST_RE.search(result_str)
            if m:
                ctx["recommended_specialist"] = m.group(1).strip()
            elif len(result_str.strip()) < 80:
                ctx["recommended_specialist"] = result_str.strip()
            if ctx["recommended_specialist"]:
                changed = True
                print(f"   → recommended_specialist = '{ctx['recommended_specialist']}'")

    elif tool_name == "find_provider_availability":
        print(f"   result preview: {result_str[:300]}")
        # Grab the resolved date echoed back in the result
        date_m = _DATE_RE.search(result_str)
        if date_m and not ctx["pending_slot"]["date"]:
            ctx["pending_slot"]["date"] = date_m.group(1)
            changed = True
            print(f"   → pending_slot.date (from availability result) = '{date_m.group(1)}'")

    elif tool_name == "create_booking":
        print(f"   result preview: {result_str[:300]}")
        if "Appointment confirmed" in result_str and not ctx["appointment"]["confirmed"]:
            ctx["appointment"]["date"]      = str(tool_args.get("date", ""))
            ctx["appointment"]["time"]      = str(tool_args.get("time", ""))
            ctx["appointment"]["confirmed"] = True
            if not ctx["patient"]["id"] and tool_args.get("patient_id"):
                ctx["patient"]["id"] = int(tool_args["patient_id"])
            if not ctx["selected_doctor"]["id"] and tool_args.get("doctor_id"):
                ctx["selected_doctor"]["id"] = int(tool_args["doctor_id"])
            m = _BOOKING_ID_RE.search(result_str)
            if m:
                ctx["appointment"]["booking_id"] = int(m.group(1))
            changed = True
            print(f"   ✅ BOOKING CONFIRMED — date={ctx['appointment']['date']} "
                  f"time={ctx['appointment']['time']} booking_id={ctx['appointment']['booking_id']}")
        else:
            print(f"   ⚠️  create_booking did NOT return 'Appointment confirmed'")
            print(f"       Full result: {result_str}")

    if changed:
        _advance_step(ctx)
        print(f"🔍 [ToolExtract] AFTER:  patient.id={ctx['patient']['id']}  doctor.id={ctx['selected_doctor']['id']}  slot={ctx['pending_slot']}  step={ctx['step']}")
    else:
        print(f"🔍 [ToolExtract] AFTER:  no changes extracted from this tool result")
    print(f"🔍 [ToolExtract] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
    return changed


# ═══════════════════════════════════════════════════════════════════
# TOOL EXECUTOR NODE
# ═══════════════════════════════════════════════════════════════════

def tool_executor_node(state: ConversationState) -> dict:
    session_id = state.get("session_id") or "default"
    ctx        = state.get("booking_context") or load_booking_context(session_id)
    messages   = list(state.get("messages", []))

    print(f"\n🔧 [ToolExecutor] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔧 [ToolExecutor] session_id  = {session_id}")
    print(f"🔧 [ToolExecutor] step        = {ctx['step']}")
    print(f"🔧 [ToolExecutor] patient.id  = {ctx['patient']['id']}  name={ctx['patient']['name']}")
    print(f"🔧 [ToolExecutor] doctor.id   = {ctx['selected_doctor']['id']}  name={ctx['selected_doctor']['name']}")
    print(f"🔧 [ToolExecutor] slot        = {ctx['pending_slot']}")
    print(f"🔧 [ToolExecutor] confirmed   = {ctx['appointment']['confirmed']}")
    print(f"🔧 [ToolExecutor] num_msgs    = {len(messages)}")

    if not messages:
        print("   ⚠️  No messages")
        return {"messages": [], "booking_context": ctx}

    last_msg = messages[-1]
    if not getattr(last_msg, "tool_calls", None):
        print("   ⚠️  Last message has no tool_calls")
        return {"messages": [], "booking_context": ctx}

    tool_calls = last_msg.tool_calls
    print(f"   Calls to run: {[tc['name'] for tc in tool_calls]}")

    tool_messages: list[ToolMessage] = []
    ctx_changed = False

    # ── GUARD 0: enforce per-phase tool whitelist at execution time ───────────
    # Derives the allowed set from _PHASE_TOOL_NAMES so it stays in sync
    # with the per-step LLM binding automatically.
    _current_step  = ctx.get("step", "")
    _phase_allowed = set(_PHASE_TOOL_NAMES.get(_current_step, []))
    # If step is unknown or not in the map, allow everything (safe fallback)
    _phase_restricted = bool(_phase_allowed)

    for tc in tool_calls:
        tool_name    = tc.get("name", "")
        tool_args    = tc.get("args", {})
        tool_call_id = tc.get("id", "")

        print(f"\n   ▶ '{tool_name}' args={tool_args}")

        if _phase_restricted and tool_name not in _phase_allowed:
            # Always allow waitlist/handoff tools when human handoff is pending
            if tool_name in _HANDOFF_TOOL_NAMES and ctx.get("human_handoff_pending"):
                print(f"   ✅ [Guard0] '{tool_name}' allowed — human handoff pending")
            else:
                msg = (
                    f"⛔ Tool '{tool_name}' is not allowed in step '{_current_step}'. "
                    f"Allowed tools for this step: {sorted(_phase_allowed)}. "
                    f"Follow the directive and only call tools from that list."
                )
                print(f"   🚫 [Guard0] Blocked '{tool_name}' — not in phase whitelist {sorted(_phase_allowed)}")
                tool_messages.append(ToolMessage(content=msg, tool_call_id=tool_call_id, name=tool_name))
                continue

        print(f"\n   ▶ '{tool_name}' args={tool_args}")

        # ── GUARD 1: Never re-fetch patient ──────────────────────
        if tool_name == "lookup_customer_profile" and ctx["patient"]["id"]:
            msg = f"Patient already loaded: {ctx['patient']['name']} (ID={ctx['patient']['id']}). Skipping."
            print(f"   🚫 [Guard1] {msg}")
            tool_messages.append(ToolMessage(content=msg, tool_call_id=tool_call_id, name=tool_name))
            continue

        # ── GUARD 2: Never re-fetch doctor info if ID already set ─
        _DOCTOR_TOOLS = ("get_doctor_profile", "get_doctor_schedule", "get_doctors_by_specialization")
        if tool_name in _DOCTOR_TOOLS and ctx["selected_doctor"]["id"]:
            msg = (
                f"Doctor already loaded: Dr. {ctx['selected_doctor']['name']} "
                f"(ID={ctx['selected_doctor']['id']}, {ctx['selected_doctor']['specialization']}). "
                f"Do NOT call any doctor-lookup tools again. "
                f"Just ask the patient which date they prefer for their appointment."
            )
            print(f"   🚫 [Guard2] {msg}")
            tool_messages.append(ToolMessage(content=msg, tool_call_id=tool_call_id, name=tool_name))
            continue

        # ── GUARD 4: find_provider_availability must always have a doctor identifier.
        if tool_name == "find_provider_availability":
            args = dict(tool_args)
            has_doctor = args.get("doctor_id") or args.get("doctor_name") or args.get("specialization")
            if not has_doctor:
                known_id   = ctx["selected_doctor"].get("id")
                known_name = ctx["selected_doctor"].get("name")
                if known_id:
                    args["doctor_id"] = known_id
                    print(f"   💉 [Guard4] Auto-injected doctor_id={known_id} into find_provider_availability")
                elif known_name:
                    args["doctor_name"] = known_name
                    print(f"   💉 [Guard4] Auto-injected doctor_name='{known_name}' into find_provider_availability")
                else:
                    known_spec = ctx.get("recommended_specialist") or ctx["selected_doctor"].get("specialization")
                    if known_spec:
                        args["specialization"] = known_spec
                        print(f"   💉 [Guard4] Auto-injected specialization='{known_spec}' into find_provider_availability")
            if not args.get("date"):
                block_msg = (
                    "find_provider_availability blocked — no date provided. "
                    "Ask the patient which day they prefer before checking availability."
                )
                print(f"   🚫 [Guard4] BLOCKED — no date in find_provider_availability call")
                tool_messages.append(ToolMessage(content=block_msg, tool_call_id=tool_call_id, name=tool_name))
                continue
            tool_args = args

        # ── GUARD 3: create_booking requires confirmed YES ────────
        if tool_name == "create_booking":
            last_human = _last_human_text(messages)
            is_yes     = bool(_YES_RE.search(last_human))
            step       = ctx.get("step", "")

            # ── AUTO-HYDRATE slot from tool args ──────────────────────────────
            args_date = str(tool_args.get("date", "")).strip()
            args_time = str(tool_args.get("time", "")).strip()
            if args_date and not ctx["pending_slot"]["date"]:
                ctx["pending_slot"]["date"] = args_date
                print(f"   💉 [BookingGate] Auto-set slot.date={args_date} from tool args")
            if args_time and not ctx["pending_slot"]["time"]:
                ctx["pending_slot"]["time"] = args_time
                print(f"   💉 [BookingGate] Auto-set slot.time={args_time} from tool args")
            if ctx["pending_slot"]["date"] and ctx["pending_slot"]["time"]:
                _advance_step(ctx)
                save_booking_context(session_id, ctx)
                step = ctx.get("step", "")
                print(f"   💉 [BookingGate] Step re-evaluated → '{step}'")

            print(f"\n   🔐 [BookingGate] ═══════════════════════════════════════════")
            print(f"   🔐 step            = '{step}'  (must be 'await_confirmation')")
            print(f"   🔐 last_human      = '{last_human[:120]}'")
            print(f"   🔐 is_yes          = {is_yes}  (regex: {_YES_RE.pattern[:60]})")
            print(f"   🔐 patient.id      = {ctx['patient']['id']}")
            print(f"   🔐 patient.name    = {ctx['patient']['name']}")
            print(f"   🔐 doctor.id       = {ctx['selected_doctor']['id']}")
            print(f"   🔐 doctor.name     = {ctx['selected_doctor']['name']}")
            print(f"   🔐 slot.date       = {ctx['pending_slot']['date']}")
            print(f"   🔐 slot.time       = {ctx['pending_slot']['time']}")
            print(f"   🔐 confirmed       = {ctx['appointment']['confirmed']}")
            print(f"   🔐 tool_args       = {tool_args}")
            print(f"   🔐 --- Full message history ({len(messages)} msgs) ---")
            for i, m in enumerate(messages):
                role    = m.__class__.__name__.replace("Message", "")
                content = str(m.content)[:150].replace("\n", " ")
                print(f"   🔐   [{i}] {role}: {content}")
            print(f"   🔐 ═══════════════════════════════════════════════════════════")

            if step != "await_confirmation":
                p = ctx["patient"]
                d = ctx["selected_doctor"]
                s = ctx["pending_slot"]
                reasons = []
                if not p["id"]:     reasons.append(f"patient.id is None")
                if not d["id"]:     reasons.append(f"doctor.id is None")
                if not s["date"]:   reasons.append(f"pending_slot.date is None")
                if not s["time"]:   reasons.append(f"pending_slot.time is None")
                reasons_str = " | ".join(reasons) if reasons else "unknown reason"
                block_msg = (
                    f"Booking blocked — step is '{step}', must be 'await_confirmation'. "
                    f"Root cause: {reasons_str}. "
                    f"Show the appointment summary and ask the patient to confirm first."
                )
                print(f"   🚫 [BookingGate] BLOCKED — wrong step. Root cause: {reasons_str}")
                tool_messages.append(ToolMessage(content=block_msg, tool_call_id=tool_call_id, name=tool_name))
                continue

            if not is_yes:
                block_msg = (
                    f"Booking blocked — patient has not confirmed yet. "
                    f"Last message: '{last_human[:80]}'. "
                    f"Ask: 'Shall I confirm this appointment? (yes/no)'"
                )
                print(f"   🚫 [BookingGate] BLOCKED — no YES in last message")
                tool_messages.append(ToolMessage(content=block_msg, tool_call_id=tool_call_id, name=tool_name))
                continue

            print(f"   ✅ [BookingGate] ALLOWED — step is correct and patient said YES")

        # ── Run the tool ──────────────────────────────────────────
        fn = _TOOL_MAP.get(tool_name)
        if not fn:
            result_str = f"Error: unknown tool '{tool_name}'"
            print(f"   ❌ Unknown tool")
        else:
            try:
                result_str = fn.invoke(tool_args)
                print(f"   ✅ Result: {str(result_str)[:200]}")
            except Exception as exc:
                result_str = f"Tool error: {exc}"
                print(f"   ❌ Exception: {exc}")

        changed     = _extract_from_tool_result(tool_name, tool_args, str(result_str), ctx)
        ctx_changed = ctx_changed or changed

        # ── Try slot extraction ONLY after find_provider_availability ──
        # Calling it after get_doctors_by_specialization is wrong —
        # last_human is just the phone number at that point, not a slot.
        if tool_name == "find_provider_availability" and ctx["step"] == "collect_slot" and not ctx["pending_slot"]["time"]:
            _try_extract_pending_slot_inline(messages, ctx)
            if ctx["pending_slot"]["time"]:
                ctx_changed = True

        tool_messages.append(ToolMessage(
            content=str(result_str),
            tool_call_id=tool_call_id,
            name=tool_name,
        ))

    if ctx_changed:
        save_booking_context(session_id, ctx)

    print(f"\n🔧 [ToolExecutor] Done — step={ctx['step']}  ctx_changed={ctx_changed}")
    return {"messages": tool_messages, "booking_context": ctx}


# ═══════════════════════════════════════════════════════════════════
# PENDING SLOT EXTRACTOR  (LLM-based — no regex)
# ═══════════════════════════════════════════════════════════════════

def _try_extract_pending_slot_inline(messages: list, ctx: dict) -> None:
    """
    LLM-based slot extraction.

    Sends the last 8 messages to a small Qwen model with a strict JSON-only
    prompt. Handles all languages, formats, and spoken styles:
      - "11 am", "11:00", "11.00 AM", "3.30 PM"
      - Urdu / Roman Urdu time expressions
      - Patient said YES where the time was shown in an earlier AI message

    Default: if a date is known but no time can be found anywhere, falls back
    to 11:00 (the clinic's standard morning slot).
    """
    last_human = _last_human_text(messages)

    print(f"\n🔍 [SlotExtract] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔍 [SlotExtract] last_human   = '{last_human[:120]}'")
    print(f"🔍 [SlotExtract] current slot = date={ctx['pending_slot']['date']}  time={ctx['pending_slot']['time']}")

    # Build a short context window — last 8 messages is plenty
    recent        = messages[-8:]
    context_lines = []
    for m in recent:
        role    = "Patient" if m.type == "human" else "Assistant"
        content = str(m.content).strip().replace("\n", " ")[:200]
        context_lines.append(f"{role}: {content}")
    context_block = "\n".join(context_lines)

    known_date = ctx["pending_slot"].get("date") or "unknown"

    prompt = f"""You are a date/time extractor for a hospital appointment booking system in Pakistan.

CONVERSATION SO FAR:
{context_block}

ALREADY KNOWN:
- date: {known_date}

TASK:
Extract the appointment date and time the patient wants.

Date rules:
- Return a YYYY-MM-DD string.
- If the already-known date is not "unknown", use it unless the patient explicitly changes it.
- Return null only if genuinely unclear.
- Urdu day names (check the raw Urdu in the conversation, NOT the English translation which may be wrong):
    پیر = Monday | منگل = Tuesday | بدھ = Wednesday | جمعرات = Thursday
    جمعہ = Friday | ہفتہ = Saturday | اتوار = Sunday
  Roman Urdu: peer/pir=Monday, mangal=Tuesday, budh=Wednesday,
    jumerat/jumeraat=Thursday, jumma/juma=Friday, hafta=Saturday, itwar/etwar=Sunday

Time rules:
- Return HH:MM in 24-hour format.
- Convert: "11 am" → "11:00", "3.30 PM" → "15:30", "2:00 PM" → "14:00", "9 baj ke" → "09:00".
- If the patient said YES/confirm/book without stating a time, find the most recently mentioned time anywhere in the Assistant messages (the assistant showed available slots — use that time).
- Return null ONLY if no time appears anywhere in the conversation.

Respond with ONLY valid JSON. No explanation, no markdown fences, no extra text.

{{"date": "YYYY-MM-DD or null", "time": "HH:MM or null"}}"""

    print(f"🔍 [SlotExtract] Calling LLM for extraction…")
    try:
        response  = _get_slot_extractor_llm().invoke([HumanMessage(content=prompt)])
        raw       = str(response.content).strip()
        # Strip <think>...</think> blocks emitted by Qwen reasoning models
        raw       = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        # Strip accidental markdown fences just in case
        raw       = re.sub(r"^```[a-z]*\n?", "", raw)
        raw       = re.sub(r"\n?```$",        "", raw)
        raw       = raw.strip()
        print(f"🔍 [SlotExtract] LLM raw output: {raw[:200]}")
        extracted = json.loads(raw)
    except Exception as e:
        print(f"🔍 [SlotExtract] ❌ LLM extraction failed: {e} — slot NOT set")
        print(f"🔍 [SlotExtract] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return

    ext_date = extracted.get("date") or None
    ext_time = extracted.get("time") or None

    # Normalise string "null" / "none" the model might return as text
    if isinstance(ext_date, str) and ext_date.lower() in ("null", "none", "unknown", ""):
        ext_date = None
    if isinstance(ext_time, str) and ext_time.lower() in ("null", "none", "unknown", ""):
        ext_time = None

    print(f"🔍 [SlotExtract] Extracted → date={ext_date}  time={ext_time}")

    # ── Default time fallback ─────────────────────────────────────
    # If no time was found anywhere but we at least have a date,
    # default to 11:00 AM (clinic standard morning slot).
    if not ext_time:
        final_date = ctx["pending_slot"].get("date") or ext_date
        if final_date:
            ext_time = "11:00"
            print(f"🔍 [SlotExtract] ⚠️  No time found — defaulting to 11:00")
        else:
            print(f"🔍 [SlotExtract] ❌ RESULT: no time and no date — slot NOT set")
            print(f"🔍 [SlotExtract] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            return

    # Prefer the date already in ctx (it came from find_provider_availability),
    # fall back to whatever the LLM found in the conversation
    final_date = ctx["pending_slot"].get("date") or ext_date

    if not final_date:
        print(f"🔍 [SlotExtract] ❌ RESULT: have time='{ext_time}' but NO date — slot NOT set")
        print(f"🔍 [SlotExtract] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return

    ctx["pending_slot"]["time"] = ext_time
    ctx["pending_slot"]["date"] = final_date
    _advance_step(ctx)
    print(f"🔍 [SlotExtract] ✅ RESULT: slot SET → {final_date} at {ext_time}  step={ctx['step']}")
    print(f"🔍 [SlotExtract] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


def _try_extract_pending_slot(state: dict, ctx: dict, session_id: str) -> None:
    """Supervisor-side wrapper — saves to disk if the slot was extracted."""
    print(f"\n📅 [SlotExtract-Wrapper] step={ctx['step']}  slot.time={ctx['pending_slot']['time']}  slot.date={ctx['pending_slot']['date']}")
    if ctx["step"] != "collect_slot":
        print(f"📅 [SlotExtract-Wrapper] SKIPPED — step is '{ctx['step']}', not 'collect_slot'")
        return
    if ctx["pending_slot"]["time"]:
        print(f"📅 [SlotExtract-Wrapper] SKIPPED — slot.time already set: '{ctx['pending_slot']['time']}'")
        return
    messages = list(state.get("messages", []))
    _try_extract_pending_slot_inline(messages, ctx)
    if ctx["pending_slot"]["time"]:
        save_booking_context(session_id, ctx)
        print(f"📅 [SlotExtract-Wrapper] Saved slot to disk")


# ═══════════════════════════════════════════════════════════════════
# SUPERVISOR NODE
# ═══════════════════════════════════════════════════════════════════

def supervisor_node(state: ConversationState) -> dict:
    now          = datetime.now(PKT)
    today_str    = now.strftime("%A, %Y-%m-%d")
    tomorrow_str = (now + timedelta(days=1)).strftime("%A, %Y-%m-%d")

    session_id = state.get("session_id")
    if not session_id:
        session_id = "session_" + now.strftime("%Y%m%d_%H%M%S")
        print(f"⚠️  [Supervisor] session_id missing — generated: {session_id}")

    ctx             = state.get("booking_context") or load_booking_context(session_id)
    ctx             = _force_english_for_testing(ctx)
    raw_messages    = _trim_messages(list(state.get("messages", [])))
    safe_messages   = _build_safe_messages(raw_messages)
    tools_this_turn = _count_tools_since_last_human(list(state.get("messages", [])))

    print("\n" + "=" * 54)
    print(f"🧠 [Supervisor] step={ctx['step']}  tools={tools_this_turn}/{MAX_TOOLS_PER_TURN}")
    print(f"   triage_active={state.get('triage_active')}  triage_done={ctx.get('triage_completed')}")
    print(f"   triage_questions={ctx.get('triage_questions_asked', 0)}")
    print(f"   patient.id={ctx['patient']['id']}  doctor.id={ctx['selected_doctor']['id']}")
    print(f"   slot={ctx['pending_slot']}  confirmed={ctx['appointment']['confirmed']}")
    print("=" * 54)
    _print_messages(safe_messages, "Supervisor input")

    _try_extract_pending_slot(state, ctx, session_id)

    # ── Algorithmic field-limit enforcement ───────────────────────────────────
    # Must run BEFORE the directive is built so the directive reflects any
    # force-advance that happened this turn.
    if ctx.get("step") == "collect_patient_history":
        _enforce_history_field_limits(ctx)
        save_booking_context(session_id, ctx)

    booking_directive = _get_booking_directive(ctx, today_str, tomorrow_str)

    prompt_path   = Path(__file__).parent.parent / "prompts" / "supervisor_system.md"
    static_prompt = (
        prompt_path.read_text(encoding="utf-8")
        if prompt_path.exists()
        else "You are a bilingual (English/Urdu/Roman Urdu) hospital concierge AI. Be kind and empathetic."
    )

    sys_prompt = SystemMessage(content=f"""{static_prompt}

DATE CONTEXT (Pakistan Standard Time):
  Today    : {today_str}
  Tomorrow : {tomorrow_str}

PATIENT LANGUAGE: {ctx.get("patient_language", "en")}
{"IMPORTANT: The patient speaks Urdu. You MUST reply in simple everyday Urdu (nastaliq script). NOT Roman Urdu, NOT English. Natural spoken Urdu only." if ctx.get("patient_language") in ("ur", "urdu") else "Reply in plain conversational English only. Do not use Urdu, Roman Urdu, Hindi, or mixed-script text."}

{booking_directive}

ABSOLUTE PROHIBITIONS:
  Never call a tool with null/unknown values.
  Never call lookup_customer_profile if patient.id is set above.
  Never call get_doctor_schedule — use find_provider_availability instead.
  Never call create_booking — code handles it after YES.
  Never call find_provider_availability without a date — ask the patient first.
  Always pass doctor_id or doctor_name when calling find_provider_availability.
  One tool per turn, then reply to the user.
  Never present appointment times that did not come from find_provider_availability.
{"  TRIAGE DONE: NEVER emit [SYMPTOM_LOGGED:...] or [START_TRIAGE] — triage is complete." if ctx.get("triage_completed") else ""}

SPECIAL TAGS (output these exact strings when needed):
  [TRANSFER_TO_HUMAN] — output this (alone, no other text) if the patient says they want to speak to a human, a real person, a human agent, or uses phrases like "insaan se baat", "banda chahiye", "agent se milna", "انسان سے بات", "حقیقی نمائندے". The system will immediately hand off the call.
""")

    print("📡 [Supervisor] Calling LLM…")
    current_step = ctx.get("step", "collect_patient_history")
    try:
        # When human handoff is pending, include waitlist tools regardless of step
        if ctx.get("human_handoff_pending") or ctx.get("step") == "await_human":
            llm_to_use = _get_llm_with_handoff_tools(current_step)
        else:
            llm_to_use = _get_llm_for_step(current_step)
        response = _invoke_with_retry(llm_to_use, [sys_prompt] + safe_messages)
    except Exception as e:
        print(f"❌ [Supervisor] LLM error: {e}")
        return {
            "messages":              [AIMessage(content="I'm having trouble. Please repeat that.")],
            "booking_context":       ctx,
            "triage_active":         state.get("triage_active") or False,
            "interaction_completed": state.get("interaction_completed") or False,
            "extracted_symptom":     state.get("extracted_symptom"),
            "session_id":            session_id,
        }

    print(f"\n🤖 [LLM] response: {repr(str(response.content)[:300])}")
    if getattr(response, "tool_calls", None):
        print(f"🛠️  [LLM] tool_calls: {[tc['name'] for tc in response.tool_calls]}")
        blocked = _validate_tool_calls(response)
        if blocked:
            print("🚫 [Supervisor] Tool call blocked by validator")
            response = blocked

    response_text         = str(response.content)
    extracted_symptom     = state.get("extracted_symptom")
    triage_active         = state.get("triage_active") or False
    interaction_completed = state.get("interaction_completed") or False

    if "[SYMPTOM_LOGGED:" in response_text and ctx.get("step") != "collect_patient_history":
        start   = response_text.find("[SYMPTOM_LOGGED:") + 16
        end     = response_text.find("]", start)
        symptom = response_text[start:end].strip()
        extracted_symptom = symptom
        if not ctx.get("prime_complaint"):
            ctx["prime_complaint"] = symptom
            _advance_step(ctx)
            save_booking_context(session_id, ctx)
        print(f"📝 [Supervisor] symptom='{symptom}'")
    elif "[SYMPTOM_LOGGED:" in response_text and ctx.get("step") == "collect_patient_history":
        start   = response_text.find("[SYMPTOM_LOGGED:") + 16
        end     = response_text.find("]", start)
        symptom = response_text[start:end].strip()
        # Store for later so patient doesn't have to repeat the complaint
        if not ctx.get("initial_complaint_hint"):
            ctx["initial_complaint_hint"] = symptom
            save_booking_context(session_id, ctx)
            print(f"📝 [Supervisor] 💾 Complaint hint stored: '{symptom}' (will be used post-history)")
        print(f"📝 [Supervisor] ⛔ SYMPTOM_LOGGED suppressed during history phase: '{symptom}'")

    # ── Human handoff detection ───────────────────────────────────────────────
    if "[HUMAN_REQUESTED]" in response_text:
        ctx["human_handoff_pending"] = True
        save_booking_context(session_id, ctx)
        print(f"🧑 [Supervisor] Human handoff requested by patient")

    if "[HUMAN_CONFIRMED]" in response_text and ctx.get("human_handoff_pending"):
        ctx["human_handoff_confirmed"] = True
        ctx["human_handoff_pending"]   = False
        _advance_step(ctx)
        save_booking_context(session_id, ctx)
        print(f"🧑 [Supervisor] Human handoff confirmed — step = await_human")

    if "[START_TRIAGE]" in response_text and ctx.get("step") != "collect_patient_history":
        triage_active = True
        print("🚦 [Supervisor] triage_active = True — muting supervisor reply, triage_node will speak")
        response = AIMessage(content="")
    elif "[START_TRIAGE]" in response_text and ctx.get("step") == "collect_patient_history":
        print("🚦 [Supervisor] ⛔ START_TRIAGE suppressed — still in collect_patient_history phase")

    if "[END_CALL]" in response_text:
        booking_done = (
            ctx.get("appointment", {}).get("confirmed")
            or ctx.get("step") == "completed"
        )
        if not booking_done:
            print(
                f"⚠️  [Supervisor] [END_CALL] blocked — booking not done yet "
                f"(step={ctx.get('step')}, confirmed={ctx.get('appointment', {}).get('confirmed')})"
            )
            cleaned = response_text.replace("[END_CALL]", "").strip()
            cleaned = re.sub(r"\[SYMPTOM_LOGGED:[^\]]*\]", "", cleaned).strip()
            if not cleaned:
                lang = ctx.get("patient_language", "en")
                if lang in ("ur", "urdu"):
                    cleaned = "معذرت، ہم ابھی آپ کی appointment مکمل نہیں کر سکے۔ کیا آپ آگے بڑھنا چاہیں گے؟"
                else:
                    cleaned = "Sorry, we haven't finished booking your appointment yet. Shall we continue?"
            response = AIMessage(content=cleaned)
        else:
            interaction_completed = True
            ctx["triage_completed"] = True
            save_booking_context(session_id, ctx)
            print("🏁 [Supervisor] interaction_completed = True")

            lang = ctx.get("patient_language", "en")
            if lang in ("ur", "urdu"):
                goodbye = "آپ کا شکریہ کہ آپ نے ہم سے رابطہ کیا۔ اپنا خیال رکھیں، خدا حافظ۔ 🙏"
            else:
                goodbye = "Thank you for your time. Take care, and goodbye! 🙏"
            response = AIMessage(content=goodbye)
            print(f"👋 [Supervisor] Goodbye sent ({lang}): {goodbye}")

    if "[TRANSFER_TO_HUMAN]" in response_text:
        print("🔄 [Supervisor] [TRANSFER_TO_HUMAN] detected — stopping and handing off")
        interaction_completed = True
        save_booking_context(session_id, ctx)
        lang = ctx.get("patient_language", "en")
        if lang in ("ur", "urdu"):
            transfer_msg = "آپ کو ابھی ایک انسانی نمائندے سے منسلک کیا جا رہا ہے۔ براہ کرم انتظار کریں۔"
        else:
            transfer_msg = "Transferring you to a human agent now. Please hold."
        print(f"🔄 [Supervisor] Transfer message ({lang}): {transfer_msg}")
        response = AIMessage(content=transfer_msg)

    if not str(response.content).strip() and not triage_active:
        print("⚠️  [Supervisor] EMPTY response returned and triage not active — this is the silent-AI bug.")
        print(f"   response_text was: {repr(response_text[:300])}")
        print(f"   step={ctx.get('step')}  triage_active={triage_active}  interaction_completed={interaction_completed}")

    return {
        "messages":              [response],
        "extracted_symptom":     extracted_symptom,
        "triage_active":         triage_active,
        "interaction_completed": interaction_completed,
        "booking_context":       ctx,
        "session_id":            session_id,
    }


# ═══════════════════════════════════════════════════════════════════
# ROUTING
# ═══════════════════════════════════════════════════════════════════

def entry_router(state: ConversationState) -> str:
    if state.get("triage_active"):
        print("🔀 [Entry] → triage_node")
        return "triage_node"
    if state.get("interaction_completed"):
        print("🔀 [Entry] → diagnostic_node")
        return "diagnostic_node"
    print("🔀 [Entry] → supervisor_node")
    return "supervisor_node"


def supervisor_router(state: ConversationState) -> str:
    messages = list(state.get("messages", []))
    if not messages:
        return END

    last = messages[-1]

    if state.get("interaction_completed"):
        print("🔀 [Router] → diagnostic_node")
        return "diagnostic_node"

    if getattr(last, "tool_calls", None):
        tools_done = _count_tools_since_last_human(messages)
        if tools_done >= MAX_TOOLS_PER_TURN:
            print(f"🚫 [Router] Tool limit hit → END")
            return END
        print(f"🔀 [Router] → tool_executor ({tools_done + 1}/{MAX_TOOLS_PER_TURN})")
        return "tool_executor"

    if state.get("triage_active"):
        print("🔀 [Router] → triage_node")
        return "triage_node"

    print("🔀 [Router] → END")
    return END


def triage_router(state: ConversationState) -> str:
    """
    Option B routing — triage_node already produced the patient-facing AIMessage.

    - triage still active  →  END  (wait for patient's next reply)
    - triage complete      →  supervisor_node  (proceed to booking)
    """
    if state.get("triage_active"):
        print("🔀 [TriageRouter] question sent → END (awaiting patient reply)")
        return END
    else:
        print("🔀 [TriageRouter] triage complete → supervisor_node (booking)")
        return "supervisor_node"


# ═══════════════════════════════════════════════════════════════════
# GRAPH
# ═══════════════════════════════════════════════════════════════════

builder = StateGraph(ConversationState)
builder.add_node("supervisor_node", supervisor_node)
builder.add_node("tool_executor",   tool_executor_node)
builder.add_node("triage_node",     triage_node)
builder.add_node("diagnostic_node", diagnostic_node)

builder.add_conditional_edges(START,             entry_router)
builder.add_conditional_edges("supervisor_node", supervisor_router)
builder.add_edge("tool_executor",                "supervisor_node")
builder.add_conditional_edges("triage_node",     triage_router)
builder.add_edge("diagnostic_node",              END)

memory = MemorySaver()
orchestrator_graph = builder.compile(checkpointer=memory)

print("✅ [Orchestrator] Ready.")
print(f"   tools={len(ALL_TOOLS)}  max_per_turn={MAX_TOOLS_PER_TURN}")
print(f"   ctx_dir={BOOKING_CTX_DIR.resolve()}")
print(f"   triage: Option B (MedGemma→Qwen direct chain, max {8} questions)")
print("   create_booking: code-gated YES check in tool_executor")
print("   session JSON: named by session_id (never 'default')")
print("   slot extraction: LLM-based (default time: 11:00 AM)")