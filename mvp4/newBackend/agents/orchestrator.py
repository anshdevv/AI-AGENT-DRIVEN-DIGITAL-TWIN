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
from agents.chat_memory import store_message, search_relevant, backfill_session_patient_id
from agents.policy_rag  import search_policy

import os
# ── Feature flags (set in .env to disable expensive features) ─────────────
_USE_SYMPTOM_LOOKUP = os.getenv("USE_SYMPTOM_LOOKUP", "true").lower() != "false"
_USE_LLM_JUDGE      = os.getenv("USE_LLM_JUDGE",      "false").lower() == "true"
print(f"🔧 [Features] symptom_lookup={'ON' if _USE_SYMPTOM_LOOKUP else 'OFF'}  "
      f"llm_judge={'ON' if _USE_LLM_JUDGE else 'OFF'}")

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
    "smoking_status":    "Do you currently smoke or use any tobacco products?",
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
    "marital_status":    "married or single — ONLY these two options, no others",
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
        """
        Returns True if this field's label appears in the history string.
        Presence means the field exists in the DB row (even if the value is 'none').
        'None reported' IS a valid collected answer — don't re-ask it.
        """
        if not hd or keyword not in hd:
            return False
        return True   # label present → field was collected

    required: list[str] = []

    # ── Demographics (patients table) ─────────────────────────────────────────
    if age is None:
        required.append("age")
    if not gender:
        required.append("gender")
    # Only ask marital status for adults — never ask a minor
    if not marital and (age is None or int(age) >= 18):
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
    # Age gate: don't ask an under-14 about smoking
    if not _has("smoking") and (age is None or int(age) >= 14):
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



# ── Per-field turn limit: after this many turns on one field, force-advance ──
_MAX_TURNS_PER_HISTORY_FIELD = 3


def _enforce_history_field_limits(ctx: dict) -> None:
    """
    Per-field turn limit. Tracks how many turns have elapsed while
    required_history_fields[0] is the current field. After 3 turns
    without a save, force-saves "not provided" and removes the field.

    This prevents the model staying on one field forever without saving,
    while still allowing the auto-save to resolve it naturally first.
    The per-turn counter resets whenever auto-save successfully clears
    a field (see _try_auto_save_history_field).
    """
    required = ctx.get("required_history_fields")
    if not required:
        return

    current_field = required[0]
    counts = ctx.setdefault("history_field_turn_count", {})
    counts[current_field] = counts.get(current_field, 0) + 1
    turn = counts[current_field]

    print(
        f"🔢 [HistoryGate] field='{current_field}'  turn {turn}/{_MAX_TURNS_PER_HISTORY_FIELD}  "
        f"remaining_fields={required}"
    )

    if turn < _MAX_TURNS_PER_HISTORY_FIELD:
        return

    # ── Force-advance ─────────────────────────────────────────────────────────
    print(f"⏭️  [HistoryGate] '{current_field}' hit {_MAX_TURNS_PER_HISTORY_FIELD}-turn limit — force-advancing")
    patient_id = ctx.get("patient", {}).get("id")
    _DEMO = {"age", "gender", "marital_status"}
    if patient_id and current_field not in _DEMO:
        try:
            from agents.mcp_tools import save_patient_history as _sph
            _sph.invoke({"patient_id": patient_id, current_field: "not provided"})
            print(f"   → Force-saved '{current_field}'='not provided'")
        except Exception as e:
            print(f"   ⚠️  Force-save failed: {e}")

    reqs = list(ctx.get("required_history_fields") or [])
    skipped = ctx.setdefault("skipped_history_fields", [])
    if current_field in reqs:
        reqs.remove(current_field)
    if current_field not in skipped:
        skipped.append(current_field)
    ctx["required_history_fields"] = reqs
    counts[current_field] = 0
    print(f"   Skipped: {skipped}  |  Remaining: {reqs}")



def _compute_triage_dimensions(complaint: str, ctx: dict) -> list[tuple[str, str]]:
    """
    Compute the ordered list of (dimension_key, question_text) for this triage session.
    Called once on the first triage turn and cached in ctx['triage_dimensions'].

    Priority: red_flag → duration → character → severity → associated → complaint_specific
    At most 6 dimensions. Condition-specific branch replaces generic modifying-factors.

    SPECIAL CASE: routine / checkup / screening / follow-up visits are NOT symptomatic.
    OLDCARTS makes no sense for them — patient gets confused by "how long have you had
    this?" / "is it sharp or dull?" when there is no symptom. For these, ask one
    intent-clarifying question and complete triage.
    """
    c     = complaint.lower()
    p     = ctx.get("patient", {})
    hist  = (ctx.get("patient_history_data") or "").lower()
    age   = p.get("age")
    g     = (p.get("gender") or "").lower()
    m     = (p.get("marital_status") or "").lower()
    female  = g in ("female", "f", "woman", "girl")
    married = m == "married"

    # ── SPECIAL CASE: Routine / preventive / non-symptomatic visit ───────────
    # Detect these BEFORE building OLDCARTS dimensions. One brief question only.
    _ROUTINE_KW = (
        "routine", "checkup", "check-up", "check up", "screening",
        "follow-up", "follow up", "followup", "annual", "yearly",
        "preventive", "preventative", "physical exam", "wellness",
        "well visit", "general checkup", "general check",
    )
    if any(kw in c for kw in _ROUTINE_KW):
        return [(
            "routine_intent",
            "Just to confirm — is there any specific concern you'd like the doctor "
            "to look at during this visit, or is this purely a routine check?",
        )]

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
    Routing rules:
      - Emergency                                → EMERGENCY
      - Severe                                   → specialist + human flag
      - No prior notes                           → GP (first visit)
      - Prior visit > 6 months                   → new episode → GP
      - Prior visit, different complaint         → GP
      - Prior visit, same complaint, < 6 months  → prior specialist (or recommended)
    """
    from datetime import datetime, timezone

    if ctx.get("routing_decision"):
        return ctx["routing_decision"]

    recommended = (ctx.get("recommended_specialist") or "General Physician").strip()
    notes       = (ctx.get("recent_case_notes") or "").lower()
    severity    = (ctx.get("triage_severity")    or "Unknown").lower()
    complaint   = (ctx.get("prime_complaint")    or ctx.get("extracted_symptom") or "").lower()

    if "emergency" in recommended.lower() or severity in ("emergency", "critical"):
        return "EMERGENCY"

    if severity == "severe":
        if not ctx.get("human_handoff_pending"):
            ctx["human_handoff_pending"] = True
            print("   [Routing] Severe — flagging for human attention")
        spec = recommended if recommended.lower() not in ("general physician", "gp", "") else "Specialist"
        print(f"   [Routing] Severe → {spec} (+ human flag)")
        return spec

    if not notes or "no case notes" in notes or "no prior" in notes or "no recent" in notes:
        print("   [Routing] No prior notes → first visit → General Physician")
        return "General Physician"

    if complaint:
        complaint_words = set(re.sub(r"[^a-z\s]", "", complaint).split())
        complaint_words -= {"i", "a", "an", "the", "have", "had", "am", "is", "my", "with"}
        matches = sum(1 for w in complaint_words if len(w) > 3 and w in notes)

        if matches >= 1:
            date_matches = re.findall(r"(\d{4}-\d{2}-\d{2})", notes)
            days_ago = None
            if date_matches:
                try:
                    last_dt = datetime.strptime(max(date_matches), "%Y-%m-%d").replace(tzinfo=timezone.utc)
                    days_ago = (datetime.now(timezone.utc) - last_dt).days
                except Exception:
                    pass

            if days_ago is not None and days_ago > 180:
                print(f"   [Routing] Prior visit {days_ago}d ago (>180) → new episode → GP")
                return "General Physician"

            prior_spec_m = re.search(
                r"(?:recommended specialist|specialty)[\s]*[:=][\s]*([a-z][a-z\s/]+?)(?:[\n]|\.|booking|$)",
                notes, re.IGNORECASE,
            )
            prior_spec = prior_spec_m.group(1).strip().title() if prior_spec_m else ""
            if prior_spec.lower() in ("general physician", "gp", ""):
                prior_spec = ""

            ctx["is_returning_same_complaint"] = True
            chosen = prior_spec or recommended

            if chosen.lower() in ("general physician", "gp", ""):
                print(f"   [Routing] Returning ({days_ago}d ago) — prior was GP → General Physician")
                return "General Physician"

            print(f"   [Routing] Returning ({days_ago}d ago, {matches} kw) → {chosen}")
            return chosen

    print("   [Routing] Prior notes but different/new complaint → General Physician")
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

    # ── Phase 0.6: history collected but not yet confirmed by patient ────────
    # Stay in confirm_patient_history until patient explicitly approves the
    # parsed history record. The supervisor handles parsing + save on YES.
    if ctx.get("history_pending_confirmation"):
        ctx["step"] = "confirm_patient_history"
        print(f"🔀 [AdvanceStep] → step = 'confirm_patient_history' (awaiting patient YES)")
        print(f"🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return

    # ── History confirmed — proceed to triage ────────────────────────────────
    if not ctx.get("triage_completed"):
        ctx["step"] = "collect_patient"
        ctx["triage_completed"] = False
        # DO NOT set triage_active=True here. triage_active is only set by the
        # supervisor when the patient gives their complaint and the supervisor
        # outputs [START_TRIAGE]. Setting it here bypasses the supervisor entirely
        # and triage starts with symptom=None because no complaint was captured.
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
        lines.append("     update_patient_demographics · get_patient_history")
        lines.append("  ⛔ DO NOT call save_patient_history (history is batched + saved later).")
        lines.append("  ⛔ DO NOT call any booking or specialist tools.")
        lines.append("  ⛔ DO NOT output [SYMPTOM_LOGGED:] or [START_TRIAGE] yet.")
        lines.append("")

        if not phone and not pid:
            lines.append("  ▶ SUB-STEP 1: Ask for phone number.")
            lines.append("    Greet the patient warmly and ask: 'Could I get your phone number?'")
            lines.append("    Even if they mentioned a symptom — ask for phone first.")
            lines.append("    BUT: if patient mentioned a complaint in this message, include:")
            lines.append("    [SYMPTOM_LOGGED: <their complaint>]")
            lines.append("    at the very start of your reply — the system will store and hide it.")
            lines.append("    Example: '[SYMPTOM_LOGGED: headache] Could I get your phone number?'")

        elif phone and not pid:
            lines.append(f"  ✓ Phone collected: {phone}")
            if ctx.get("patient_lookup_failed"):
                lines.append("  ✓ Lookup ran — patient is NEW (not in system).")
                lines.append("  ▶ Ask for their name (first name is fine).")
                lines.append("  ▶ Then call: register_customer_profile(name=<name>, phone=<phone>)")
                lines.append("  ⛔ ONLY name and phone in this call. Nothing else.")
                lines.append("  ⛔ Age/gender come LATER via update_patient_demographics.")
            else:
                lines.append("  ▶ SUB-STEP 2: Call lookup_customer_profile now.")

        elif pid and ctx.get("required_history_fields") is None:
            lines.append(f"  ✓ Patient identified: {p.get('name')} (ID={pid})")
            lines.append("  ▶ SUB-STEP 3: Call get_patient_history now to load existing history.")

        elif pid and ctx.get("required_history_fields") is not None:
            remaining = ctx.get("required_history_fields", [])
            collected = ctx.get("collected_this_session", [])

            known_demo = []
            if p.get("age"):            known_demo.append(f"age={p['age']}")
            if p.get("gender"):         known_demo.append(f"gender={p['gender']}")
            if p.get("marital_status"): known_demo.append(f"marital_status={p['marital_status']}")

            if remaining:
                current_field = remaining[0]
                question  = _QUESTION_FOR_FIELD.get(current_field, f"Please tell me about: {current_field}")
                criteria  = _FIELD_COMPLETE_CRITERIA.get(current_field, "a clear answer")

                _DEMOGRAPHIC_FIELDS = {"age", "gender", "marital_status"}
                is_demographic     = current_field in _DEMOGRAPHIC_FIELDS

                lines.append(f"  ✓ Patient: {p.get('name')} (ID={pid})")
                if known_demo:
                    lines.append(f"  ✓ Already known — DO NOT re-ask: {', '.join(known_demo)}")
                lines.append(f"  ✓ Collected this session: {collected or 'none yet'}")
                lines.append(f"  ▶ Currently collecting: [{current_field}] "
                             f"({'demographic' if is_demographic else 'history'})")

                field_turns = ctx.get("history_field_turn_count", {}).get(current_field, 0)
                lines.append(f"    Field turn {field_turns}/{_MAX_TURNS_PER_HISTORY_FIELD} (auto-advances if unanswered)")
                lines.append(f"    Question: '{question}'")
                lines.append(f"    ⛔ Ask this EXACTLY as written.")
                lines.append(f"    Complete when you have: {criteria}")
                lines.append("")

                if is_demographic:
                    # Demographics still save per-field via update_patient_demographics
                    lines.append("  RULES (demographic field):")
                    lines.append("  1. Ask the question if not yet asked.")
                    lines.append("  2. If answer is partial, ask ONE follow-up to complete it.")
                    lines.append(f"  3. When complete → call update_patient_demographics(patient_id={pid}, {current_field}=<answer>)")
                    lines.append("  4. Then briefly acknowledge and move to the next field.")
                else:
                    # ── HISTORY FIELD: LLM-driven completion via [FIELD_COMPLETE] ─
                    next_field    = remaining[1] if len(remaining) > 1 else None
                    next_question = _QUESTION_FOR_FIELD.get(next_field, "") if next_field else ""
                    lines.append("  RULES (history field — DO NOT CALL ANY TOOL):")
                    lines.append("  1. Ask the question if not yet asked.")
                    lines.append("  2. If the answer is unclear or 'yes/yeah/sure' → ask ONE follow-up.")
                    lines.append("  3. ⛔ DO NOT call save_patient_history — it is NOT bound.")
                    lines.append("  4. When you have a CLEAR answer: briefly acknowledge,")
                    lines.append("     ask the NEXT field question in the SAME message,")
                    lines.append("     then end with [FIELD_COMPLETE]. One round-trip per two fields.")
                    if next_question:
                        lines.append(f"  5. NEXT question to ask: \"{next_question}\"")
                        lines.append("     Examples:")
                        lines.append(f"       Patient: 'No'          → 'Got it! {next_question} [FIELD_COMPLETE]'")
                        lines.append(f"       Patient: 'Penicillin'  → 'Noted. {next_question} [FIELD_COMPLETE]'")
                        lines.append(f"       Patient: 'Dust allergy' → 'Understood. {next_question} [FIELD_COMPLETE]'")
                    else:
                        lines.append("  5. This is the LAST field — just acknowledge clearly:")
                        lines.append("       Patient: 'No'    → 'Got it. [FIELD_COMPLETE]'")
                        lines.append("       Patient: 'Never' → 'Noted. [FIELD_COMPLETE]'")
                        lines.append("  ⛔ DO NOT ask 'What brings you in today?' or any complaint question.")
                        lines.append("  ⛔ DO NOT ask about symptoms. The system handles the next step automatically.")
                        lines.append("  ⛔ Your response must be 1-2 words + [FIELD_COMPLETE]. Nothing more.")
                    lines.append("  6. Unclear answers (follow-up first, NO marker yet):")
                    lines.append("       Patient: 'Yes'          → 'Could you tell me which one?'")
                    lines.append("       Patient: 'I take pills' → 'What kind of pills?'")
                    lines.append(f"  7. Off-topic: 'I\'ll note that — {question}'")
                    lines.append("  8. ⛔ Never reply with ONLY '[FIELD_COMPLETE]' — always")
                    lines.append("     include either the next question or an acknowledgement.")

                lines.append(f"  Fields remaining after this: {remaining[1:] or 'none — all done'}")

            else:
                # Shouldn't normally reach here — _advance_step moves past empty
                # required list — but keep a sane fallback.
                lines.append(f"  ✓ ALL FIELDS COLLECTED for {p.get('name')}.")
                lines.append("  ▶ Ask: 'What brings you in today?'")
                lines.append("  When the patient mentions a complaint, output ONLY these two lines:")
                lines.append("  [SYMPTOM_LOGGED: <their complaint>]")
                lines.append("  [START_TRIAGE]")

    # ── New step: show patient the parsed history, await YES/NO ──────────────
    elif step == "confirm_patient_history":
        parsed = ctx.get("parsed_history") or {}
        lines.append("YOUR NEXT ACTION: Show the patient their parsed medical-history record.")
        lines.append("  ⛔ ABSOLUTE RULES — read carefully:")
        lines.append("  ⛔ DO NOT call any tools or functions of any kind at this step.")
        lines.append("  ⛔ DO NOT output `<tool_call>`, `<invoke>`, `<parameter>`, `<|DSML|>`,")
        lines.append("     `tool_calls`, `save_medical_history`, `save_patient_history`,")
        lines.append("     JSON blocks, code fences, or ANY function-call-like syntax.")
        lines.append("  ⛔ The system has already saved everything. The database write happens")
        lines.append("     PROGRAMMATICALLY in Python when the patient replies YES — you do NOT")
        lines.append("     and CANNOT trigger it from here. Trying to call save_patient_history")
        lines.append("     would CORRUPT the record. Just speak to the patient as plain text.")
        lines.append("")
        lines.append("  Reply with ONLY this plain-text format (no other content, no markdown")
        lines.append("  fences, no tool calls, no JSON):")
        lines.append("")
        lines.append("    📋 Here's the medical history I've gathered:")

        _DEMO = {"age", "gender", "marital_status"}
        rendered_any = False
        # Render non-demographic fields first (the medical content)
        for fld, val in parsed.items():
            if fld in _DEMO:
                continue
            if val and str(val).strip() and str(val).strip().lower() != "none":
                lines.append(f"      • {fld.replace('_', ' ').title()}: {val}")
                rendered_any = True
            elif val and str(val).strip().lower() == "none":
                # Still show "none" explicitly — patient should know nothing
                # was missed, just confirmed-absent.
                lines.append(f"      • {fld.replace('_', ' ').title()}: None")
                rendered_any = True
        # Then demographics for context
        for fld in ("age", "gender", "marital_status"):
            val = parsed.get(fld)
            if val and str(val).strip():
                lines.append(f"      • {fld.replace('_', ' ').title()}: {val}")
                rendered_any = True

        if not rendered_any:
            # Defensive — should never happen after the safety-net parser runs,
            # but if it does, tell the LLM to just move forward without showing
            # an empty record (the patient would be confused).
            lines.append("      • (no specific history recorded)")
            lines.append("")
            lines.append("  ⚠️  No structured fields available — just say:")
            lines.append("    'Thanks! I have your details on file. Shall we continue?'")
            lines.append("    and wait for YES/NO.")
        else:
            lines.append("")
            lines.append("  Then ask: 'Is this correct? (yes / no — and tell me what to change)'")

        lines.append("")
        lines.append("  Behaviour on patient reply:")
        lines.append("  • YES → orchestrator saves to DB programmatically. Do NOT save yourself.")
        lines.append("  • NO + correction → orchestrator re-parses. Just acknowledge their correction.")
        lines.append("  • Anything else → repeat the record briefly and ask yes/no again.")

    # ── GUARD: only trigger triage when step is collect_patient AND flag is unset.
    elif not ctx.get("triage_completed") and step == "collect_patient":
        hint = ctx.get("initial_complaint_hint")
        lines.append("YOUR NEXT ACTION: Ask the patient what brings them in today.")
        # Tell the LLM what history was collected so it doesn't say "no history on file"
        _hist = ctx.get("patient_history_data") or ""
        if _hist and "history found" in _hist.lower():
            lines.append(f"  ✓ Medical history on file (already collected — do NOT say 'no history on file').")
        elif ctx.get("history_checked"):
            lines.append(f"  ✓ History check complete.")
        lines.append("  ▶ ALWAYS ask: 'What brings you in today?' or 'What's the reason for your visit?'")
        lines.append("  ⛔ Do NOT skip this question — triage accuracy depends on the patient's own words.")
        if hint:
            lines.append(f"  ℹ️  Patient may have mentioned '{hint}' earlier — you may reference it naturally,")
            lines.append(f"      e.g. 'I see you mentioned {hint} — is that what you're coming in for today?'")
            lines.append(f"      But still WAIT for their confirmation before starting triage.")
        lines.append("  ▶ When patient states their reason: output ONLY:")
        lines.append("  [SYMPTOM_LOGGED: <their complaint>]")
        lines.append("  [START_TRIAGE]")

    elif step == "collect_patient":
        if False:
            pass
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
            # Doctor already resolved — guide toward slot selection naturally
            lines.append("YOUR NEXT ACTION: Transition smoothly into booking.")
            lines.append(f"  The triage has recommended: {d['specialization']}.")
            lines.append(f"  Doctor found: Dr. {d['name']} ({d['specialization']}, ID={d['id']}).")
            lines.append("  Present this as a warm recommendation — not a done deal:")
            lines.append(f"  e.g. 'Based on your assessment, I'd suggest Dr. {d['name']}, a {d['specialization']}.")
            lines.append(f"       Shall we check available slots?'")
            lines.append("  Once patient agrees, ask which day they prefer (today or tomorrow).")
        else:
            lines.append("YOUR NEXT ACTION: Find a doctor and present as a recommendation.")
            lines.append(f"  1. Call get_doctors_by_specialization(specialization='{spec}').")
            lines.append("  2. Present the result warmly: 'Based on your assessment, I'd suggest...'")
            lines.append("  3. If only one doctor, recommend them and ask if patient wants to proceed.")
            lines.append("  4. If multiple doctors, briefly describe each and let patient choose.")
            lines.append("  ⛔ DO NOT call get_doctor_profile.")
            lines.append("  ⛔ DO NOT call get_doctors_by_specialization more than once.")

    elif step == "collect_slot":
        d_name     = d.get("name", "Unknown")
        d_id       = d.get("id")
        schedule   = ctx.get("doctor_schedule", [])
        pending_date = ctx["pending_slot"].get("date")
        slots_fetched = ctx.get("slots_fetched_for_date")

        lines.append("YOUR NEXT ACTION: Help the patient pick a date and time slot.")
        lines.append(f"  TODAY = {today} | TOMORROW = {tomorrow}")
        lines.append("")

        if pending_date and not slots_fetched:
            # Date is known but availability hasn't been checked — must call tool NOW
            lines.append(f"  ✅ Patient chose date: {pending_date}")
            lines.append(f"  ▶ CALL NOW: find_provider_availability(doctor_id={d_id}, date='{pending_date}')")
            lines.append("  ⛔ DO NOT ask anything — call the tool immediately.")
        elif slots_fetched and ctx.get("availability_result"):
            # Availability already pre-fetched — present naturally, let patient pick
            lines.append(f"  ✅ I've checked Dr. {d_name}'s schedule for {slots_fetched}.")
            lines.append(f"  Available times:")
            lines.append(f"  {ctx['availability_result'][:800]}")
            lines.append("")
            lines.append("  WORKFLOW:")
            lines.append("  1. Present the slots warmly: 'Here are the available times with Dr. [Name]:'")
            lines.append("  2. Ask patient to pick one: 'Which time works best for you?'")
            lines.append("  3. Once they pick → show full summary and ask 'Shall I confirm this booking? (Yes/No)'")
        elif schedule:
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
        lines.append("YOUR NEXT ACTION: Confirm the appointment.")
        lines.append(f"  Patient : {p['name']} (ID={p['id']})")
        lines.append(f"  Doctor  : Dr. {d['name']} (ID={d['id']})")
        lines.append(f"  Date    : {s['date']} at {s['time']}")
        lines.append("")
        lines.append("  WORKFLOW:")
        lines.append("  1. If the patient has NOT yet said yes/no → show the summary above and ask:")
        lines.append("     'Shall I confirm this appointment? (yes/no)'")
        lines.append("  2. If the patient says YES → call create_booking immediately:")
        complaint_for_book = (ctx.get("prime_complaint") or "").strip().replace("'", "")[:200]
        lines.append(
            f"     create_booking(patient_id={p['id']}, doctor_id={d['id']}, "
            f"date='{s['date']}', time='{s['time']}', "
            f"chief_complaint='{complaint_for_book}')"
        )
        lines.append("  3. If create_booking succeeds → warmly confirm and wait for next message.")
        lines.append("  4. If create_booking fails (slot taken) → tell the patient that slot is gone,")
        lines.append("     show the updated free slots from the tool result, and ask them to pick again.")
        lines.append("  ⛔ Only call create_booking — no other tools at this step.")

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
    # Phase 0.5 — identity + history collection (questions only, no save)
    # save_patient_history is INTENTIONALLY excluded — history is batched and
    # confirmed via the confirm_patient_history step below, then saved once.
    "collect_patient_history": [
        "lookup_customer_profile",
        "register_customer_profile",
        "update_patient_demographics",
        "get_patient_history",
    ],
    # Phase 0.6 — show batched history to patient, await YES/NO confirmation.
    # No tools bound — the orchestrator handles save programmatically on YES.
    "confirm_patient_history": [],
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
        "find_provider_availability",   # needed when patient names a date before step advances
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

MAX_HISTORY_MESSAGES = 30   # Raised: history phase + triage easily exceeds 20
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
    "create_booking":          {"patient_id": "patient ID", "doctor_id": "doctor ID", "date": "date", "time": "time", "chief_complaint": "the patient's chief complaint from triage"},
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
    Prepare message history for the supervisor LLM.
    Rules:
      - AI messages WITH tool_calls are kept as-is (stripping them breaks the
        tool_call_id pairing that DeepSeek and Groq both require).
      - ToolMessages with no matching preceding tool_call are dropped (orphans).
      - Duplicate ToolMessages for the same tool_call_id are deduplicated.
      - Empty AI messages (blank content, no tool_calls) are dropped.
    """
    # Collect all valid tool_call IDs from AI messages
    valid_tool_call_ids: set[str] = set()
    for msg in raw:
        if msg.type == "ai" and getattr(msg, "tool_calls", None):
            for tc in msg.tool_calls:
                valid_tool_call_ids.add(tc.get("id", ""))

    safe = []
    _seen_tool_ids: set[str] = set()
    for msg in raw:
        if msg.type == "ai":
            if not str(msg.content).strip() and not getattr(msg, "tool_calls", None):
                # Empty AI message with no tool_calls — drop (triage silencer etc.)
                continue
            safe.append(msg)   # keep tool_calls intact — required by DeepSeek + Groq
        elif msg.type == "tool":
            tid = getattr(msg, "tool_call_id", None)
            if tid and tid in valid_tool_call_ids:
                if tid in _seen_tool_ids:
                    print(f"✂️  [SafeMsg] Dropped duplicate ToolMessage tool_call_id={tid}")
                else:
                    _seen_tool_ids.add(tid)
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
    """
    Invoke the LLM with up to `retries` extra attempts.

    Retries on:
      - Connection / network errors (ReadError, 10054, etc.)
      - Effectively-empty responses: Qwen3 sometimes returns ONLY a
        <think>…</think> block with no visible content. After the caller
        strips that block the response appears blank, which triggers the
        downstream fallback every time. We detect this here — before
        returning — so the model gets another chance to give a real answer.
    """
    _THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
    last_exc  = None
    last_resp = None
    for attempt in range(1, retries + 2):
        try:
            if attempt > 1:
                print(f"🔄 [LLM] Retry {attempt}…")
                time.sleep(delay)
            resp      = llm.invoke(msgs)
            last_resp = resp
            # Check for effective emptiness after stripping think blocks
            effective = _THINK_RE.sub("", str(resp.content or "")).strip()
            has_tools = bool(getattr(resp, "tool_calls", None))
            if not effective and not has_tools:
                print(f"⚠️  [LLM] Effectively-empty response (attempt {attempt}/{retries + 1}) — retrying")
                continue
            return resp
        except Exception as e:
            err = str(e)
            if any(k in err for k in ["ReadError", "10054", "ConnectionError", "RemoteDisconnected", "forcibly closed"]):
                print(f"⚠️  [LLM] Connection error (attempt {attempt}): {err[:100]}")
                last_exc = e
            else:
                raise
    # All attempts exhausted
    if last_exc:
        raise last_exc
    # All attempts returned empty — return last response; downstream fallback will handle it
    return last_resp


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
                # Backfill patient_id on early messages stored before ID was known
                backfill_session_patient_id(ctx.get("session_id", ""), ctx["patient"]["id"])
            else:
                print(f"   ⚠️  COULD NOT extract patient.id — regex found nothing in: {result_str[:200]}")
                if "no patient profile found" in result_str.lower():
                    ctx["patient_lookup_failed"] = True
                    changed = True
                    print("   → patient_lookup_failed = True (new patient — registration required)")
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

    elif tool_name == "register_customer_profile":
        # ── Extract patient.id and name from registration success ─────────────
        # e.g. "Patient registered successfully. ID: 7, Name: ahsan"
        # Without this, patient.id stays None after registration, the directive
        # never computes required_history_fields, and the supervisor asks all
        # demographic questions at once instead of one at a time.
        print(f"   result preview: {result_str[:300]}")
        m = re.search(r"\bID[:\s]+(\d+)\b", result_str, re.IGNORECASE)
        if m and not ctx["patient"]["id"]:
            ctx["patient"]["id"] = int(m.group(1))
            changed = True
            print(f"   → patient.id = {ctx['patient']['id']} (from registration)")
            backfill_session_patient_id(ctx.get("session_id", ""), ctx["patient"]["id"])
        m = re.search(r"\bName[:\s]+([A-Za-z][A-Za-z ]{1,40}?)(?:\.|,|\n|$)", result_str, re.IGNORECASE)
        if m and not ctx["patient"]["name"]:
            ctx["patient"]["name"] = m.group(1).strip()
            changed = True
            print(f"   → patient.name = '{ctx['patient']['name']}' (from registration)")
            # Clear complaint hint if it's just the patient's name (not a real complaint).
            # This happens when patient gives their name during registration and the
            # auto-hint mistakenly stored it as the chief complaint.
            hint = ctx.get("initial_complaint_hint", "")
            if hint and hint.lower().strip() == ctx["patient"]["name"].lower().strip():
                ctx["initial_complaint_hint"] = None
                changed = True
                print(f"   ⚠️  [AutoHint] Cleared hint '{hint}' — matched patient name, not a complaint")
        if not ctx["patient"]["phone"]:
            phone = tool_args.get("phone")
            if phone:
                ctx["patient"]["phone"] = str(phone)
                changed = True
                print(f"   → patient.phone = '{phone}' (from registration args)")

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

        # ── Auto-fetch availability for the next working day once doctor is known ─
        # Programmatic: don't wait for the LLM to call find_provider_availability.
        # Tries tomorrow first; if the doctor has no slots, walks forward up to 7
        # days so we always cache a date that has real times to show the patient.
        doc_id_known = ctx["selected_doctor"]["id"]
        if doc_id_known and not ctx.get("slots_fetched_for_date"):
            try:
                from agents.mcp_tools import find_provider_availability as _fpa
                found_date = None
                found_result = None
                for days_ahead in range(1, 8):   # tomorrow … 7 days out
                    candidate = (datetime.now(PKT) + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                    avail_result = _fpa.invoke({"doctor_id": doc_id_known, "date": candidate})
                    avail_str = str(avail_result)
                    # Only accept a result that actually contains time slots
                    if "No available slots" not in avail_str and avail_str.strip():
                        found_date   = candidate
                        found_result = avail_str
                        break
                if found_date:
                    ctx["slots_fetched_for_date"] = found_date
                    ctx["availability_result"]    = found_result
                    changed = True
                    print(f"   🗓️  [AutoAvail] Pre-fetched availability for {found_date}: {found_result[:150]}")
                else:
                    print(f"   ⚠️  [AutoAvail] Doctor {doc_id_known} has no available slots in the next 7 days")
            except Exception as e:
                print(f"   ⚠️  [AutoAvail] Could not pre-fetch availability: {e}")

    elif tool_name == "get_patient_history":
        # ── Process the patient history result and compute required fields ────
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

            # Mark saved fields as collected — remove from both required AND skipped
            _HISTORY_FIELDS = {
                "chronic_conditions", "medications", "drug_allergies",
                "general_allergies", "family_history", "smoking_status",
                "pregnancy_status", "lmp_date", "menstrual_history",
                "obstetric_history", "fall_history", "vaccination_status",
            }
            reqs    = ctx.get("required_history_fields") or []
            coll    = ctx.get("collected_this_session", [])
            skipped = ctx.get("skipped_history_fields", [])
            for field in _HISTORY_FIELDS:
                if tool_args.get(field) is not None:
                    if field in reqs:    reqs.remove(field)
                    if field in skipped: skipped.remove(field)
                    if field not in coll: coll.append(field)
            ctx["required_history_fields"] = reqs
            ctx["collected_this_session"]   = coll
            ctx["skipped_history_fields"]   = skipped
            # Reset turn counter for saved fields
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
        fetched_date = None
        if date_m:
            fetched_date = date_m.group(1)
            if not ctx["pending_slot"]["date"]:
                ctx["pending_slot"]["date"] = fetched_date
                changed = True
                print(f"   → pending_slot.date (from availability result) = '{fetched_date}'")
        # Mark that we've fetched slots for this date — the slot extractor needs
        # this to know it's safe to default time to 11:00 if patient doesn't pick
        known_date = fetched_date or tool_args.get("date") or ctx["pending_slot"].get("date")
        if known_date:
            ctx["slots_fetched_for_date"] = known_date
            changed = True
            print(f"   → slots_fetched_for_date = '{known_date}' (slot extractor can now apply 11:00 default)")

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
            # Booking failed (slot taken or validation error).
            # Clear the pending time so _advance_step drops back to collect_slot
            # and the LLM re-asks the patient to pick from current free slots.
            print(f"   ⚠️  create_booking did NOT return 'Appointment confirmed'")
            print(f"       Full result: {result_str}")
            ctx["pending_slot"]["time"] = None
            # Re-fetch availability so the directive immediately shows fresh slots
            try:
                from agents.mcp_tools import find_provider_availability as _fpa
                pending_date = ctx["pending_slot"].get("date")
                doc_id       = ctx["selected_doctor"]["id"]
                if pending_date and doc_id:
                    refreshed = _fpa.invoke({"doctor_id": doc_id, "date": pending_date})
                    ctx["availability_result"]  = str(refreshed)
                    ctx["slots_fetched_for_date"] = pending_date
                    print(f"   🔄 [BookingRetry] Refreshed availability: {str(refreshed)[:120]}")
            except Exception as _e:
                print(f"   ⚠️  [BookingRetry] Could not refresh availability: {_e}")
            changed = True

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
    # ── Deduplicate tool calls ────────────────────────────────────────────────
    # The LLM sometimes emits the same tool name+args twice. Each creates a
    # separate ToolMessage. If two ToolMessages share the same name and the
    # second has no distinct matching tool_call_id, Groq rejects the next
    # request with "tool message must follow a message with tool_calls".
    _seen_signatures: set[str] = set()
    _deduped_calls: list = []
    for _tc in tool_calls:
        _sig = f"{_tc.get('name','')}::{sorted(_tc.get('args',{}).items())}"
        if _sig not in _seen_signatures:
            _seen_signatures.add(_sig)
            _deduped_calls.append(_tc)
        else:
            print(f"   ⚠️  [ToolExec] Duplicate call '{_tc.get('name')}' removed — would create orphaned ToolMessage")
    tool_calls = _deduped_calls
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

        # ── GUARD 3: create_booking only at await_confirmation step ────────
        if tool_name == "create_booking":
            last_human = _last_human_text(messages)
            step       = ctx.get("step", "")

            # ── SYNC pending_slot with what LLM put in tool args ─────────────
            # IMPORTANT: always update (not just when empty).
            # Without this, if patient picks 09:00 after a stale 11:00 default,
            # the slot stays at 11:00 and the supervisor shows the wrong summary.
            args_date = str(tool_args.get("date", "")).strip()
            args_time = str(tool_args.get("time", "")).strip()
            if args_date and args_date != ctx["pending_slot"].get("date"):
                ctx["pending_slot"]["date"] = args_date
                print(f"   💉 [BookingGate] Synced slot.date={args_date} from tool args")
            if args_time and args_time != ctx["pending_slot"].get("time"):
                ctx["pending_slot"]["time"] = args_time
                print(f"   💉 [BookingGate] Synced slot.time={args_time} from tool args (was {ctx['pending_slot'].get('time')})")
            if ctx["pending_slot"]["date"] and ctx["pending_slot"]["time"]:
                _advance_step(ctx)
                save_booking_context(session_id, ctx)
                step = ctx.get("step", "")
                print(f"   💉 [BookingGate] Step re-evaluated → '{step}'")

            print(f"\n   🔐 [BookingGate] ═══════════════════════════════════════════")
            print(f"   🔐 step            = '{step}'  (must be 'await_confirmation')")
            print(f"   🔐 last_human      = '{last_human[:120]}'")
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

            # Trust the LLM to only call create_booking after explicit patient confirmation.
            # The step == "await_confirmation" guard above is sufficient.
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

def _parse_first_available_slot(avail_str: str) -> str | None:
    """
    Extract the first available time from a find_provider_availability result string.
    Handles formats like 'Slot: 09:00', 'Morning (06:00-11:59): 09:00, 09:30'
    Returns HH:MM string or None.
    """
    if not avail_str:
        return None
    # Match any HH:MM pattern that comes after "Slot:", "Morning", "Afternoon" etc.
    m = re.search(r"(?:Slot:|(?:Morning|Afternoon|Evening)\s*[^:]*:)\s*(\d{1,2}:\d{2})", avail_str)
    if m:
        return m.group(1)
    # Fallback: grab first HH:MM anywhere in the string
    m = re.search(r"\b(\d{1,2}:\d{2})\b", avail_str)
    return m.group(1) if m else None


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

    # ── Fast path: vague / flexible responses → default to tomorrow 11:00 ───
    # Patient says "any slot", "any time", "whenever", "first available" etc.
    # Don't loop asking for a specific day — just book tomorrow morning.
    _FLEXIBLE_PATTERNS = re.compile(
        r"^(any\s*(slot|time|day|date|appointment|available)?|"
        r"whenever|anytime|any\s*day|first\s*available|"
        r"doesn'?t matter|don'?t care|up to you|your\s*choice|"
        r"koi\s*bhi|kab\s*bhi|jo\s*bhi|chalega|chale\s*ga|theek\s*hai|"
        r"ok|okay|sure|fine|yes|yeah|yep|yup|haan|han)\s*[\.\!]*$",
        re.IGNORECASE,
    )
    if last_human and _FLEXIBLE_PATTERNS.match(last_human.strip()):
        # Use pre-fetched availability date; parse first real slot from availability result
        default_date = ctx.get("slots_fetched_for_date") or (datetime.now(PKT) + timedelta(days=1)).strftime("%Y-%m-%d")
        avail_str    = ctx.get("availability_result", "")
        first_slot   = _parse_first_available_slot(avail_str)
        ctx["pending_slot"]["date"] = default_date
        ctx["pending_slot"]["time"] = first_slot or "09:00"
        print(f"🔍 [SlotExtract] ✅ Vague response — using first available: {default_date} at {ctx['pending_slot']['time']}")
        print("🔍 [SlotExtract] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return

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
    # Only default to 11:00 AFTER find_provider_availability has been called
    # for this date (ctx["slots_fetched_for_date"] is set).
    # If we default early the step jumps to await_confirmation before the
    # patient ever sees actual available slots from the doctor's schedule.
    if not ext_time:
        final_date = ctx["pending_slot"].get("date") or ext_date
        slots_fetched = ctx.get("slots_fetched_for_date")
        if final_date and slots_fetched and slots_fetched == final_date:
            # Slots were fetched for this date — safe to default to 11:00
            ext_time = "11:00"
            print(f"🔍 [SlotExtract] ⚠️  No time found but slots fetched — defaulting to 11:00")
        elif final_date:
            # Date known but availability not yet checked — store date only,
            # leave time null so step stays at collect_slot and the directive
            # tells the LLM to call find_provider_availability.
            ctx["pending_slot"]["date"] = final_date
            print(f"🔍 [SlotExtract] ℹ️  Date={final_date} stored — awaiting find_provider_availability before setting time")
            print("🔍 [SlotExtract] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            return
        else:
            print(f"🔍 [SlotExtract] ❌ RESULT: no time and no date — slot NOT set")
            print("🔍 [SlotExtract] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
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
    step = ctx["step"]
    print(f"\n📅 [SlotExtract-Wrapper] step={step}  slot.time={ctx['pending_slot']['time']}  slot.date={ctx['pending_slot']['date']}")
    if step not in ("collect_slot", "await_confirmation"):
        print(f"📅 [SlotExtract-Wrapper] SKIPPED — step is '{step}'")
        return
    # At await_confirmation: only re-extract if patient is giving a new slot
    # (i.e. not saying yes/no — those are handled by BookingGate).
    if step == "await_confirmation":
        last_human = _last_human_text(list(state.get("messages", [])))
        if _YES_RE.search(last_human or "") or re.search(r"\bno\b|\bnope\b|\bcancel\b", last_human or "", re.I):
            print(f"📅 [SlotExtract-Wrapper] SKIPPED — await_confirmation, patient said yes/no (not a slot)")
            return
        # Patient gave a new slot — extract and update
        messages = list(state.get("messages", []))
        old_time = ctx["pending_slot"]["time"]
        _try_extract_pending_slot_inline(messages, ctx)
        if ctx["pending_slot"]["time"] and ctx["pending_slot"]["time"] != old_time:
            save_booking_context(session_id, ctx)
            print(f"📅 [SlotExtract-Wrapper] ✅ Slot updated at await_confirmation: {old_time} → {ctx['pending_slot']['time']}")
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

def _try_auto_save_demographic(ctx: dict, msg: str, session_id: str) -> None:
    """
    For simple one-word demographic fields (age, gender, marital_status),
    extract the answer from the patient's message and save it directly.
    This prevents the LLM from skipping the tool call and jumping to the next field.
    Only fires when the current required field is one of these three.
    """
    required = ctx.get("required_history_fields")
    if not required:
        return
    current_field = required[0]
    if current_field not in ("age", "gender", "marital_status"):
        return

    patient_id = ctx["patient"].get("id")
    if not patient_id:
        return

    msg_lower = msg.strip().lower()
    payload: dict | None = None

    if current_field == "age":
        m = re.search(r"\b(\d{1,3})\b", msg_lower)
        if m and 0 < int(m.group(1)) < 130:
            payload = {"age": int(m.group(1))}

    elif current_field == "gender":
        if re.search(r"\bmale\b|\bman\b|\bboy\b|\bmard\b", msg_lower) and "female" not in msg_lower:
            payload = {"gender": "male"}
        elif re.search(r"\bfemale\b|\bwoman\b|\bgirl\b|\baurat\b|\bkhatoon\b", msg_lower):
            payload = {"gender": "female"}

    elif current_field == "marital_status":
        if re.search(r"\bsingle\b|\bunmarried\b|\bbachelor\b|\bnot married\b", msg_lower):
            payload = {"marital_status": "single"}
        elif re.search(r"\bmarried\b|\bshadi\b|\bnikah\b", msg_lower):
            payload = {"marital_status": "married"}

    if not payload:
        return

    try:
        from agents.mcp_tools import update_patient_demographics as _upd
        result = _upd.invoke({"patient_id": patient_id, **payload})
        if "updated successfully" in result.lower():
            for field, val in payload.items():
                ctx["patient"][field] = val
            ctx["required_history_fields"] = _compute_required_history_fields(
                ctx["patient"], ctx.get("patient_history_data")
            )
            coll = ctx.setdefault("collected_this_session", [])
            for f in payload:
                if f not in coll:
                    coll.append(f)
            counts = ctx.setdefault("history_field_turn_count", {})
            for f in payload:
                counts[f] = 0
            # If demographics happened to be the last missing pieces, flag for
            # confirmation. The supervisor will then run the LLM parser (which
            # picks up demographics from ctx["patient"] directly).
            if not ctx.get("required_history_fields"):
                ctx["history_pending_confirmation"] = True
                print(f"📋 [AutoSave] All history collected (demographics last) — "
                      f"flagging for patient confirmation")
            save_booking_context(session_id, ctx)
            print(f"🤖 [AutoSave] {payload} auto-saved for patient_id={patient_id}")
    except Exception as e:
        print(f"⚠️  [AutoSave] Failed: {e}")


def _history_fields_to_extract(ctx: dict) -> list[str]:
    """
    Determine which non-demographic history fields the parser should try to
    extract from the conversation.

    Robust to two LLM-quirk failure modes:
      1. LLM forgot to emit [FIELD_COMPLETE] markers → collected_this_session
         is empty / partial.
      2. LLM force-advanced past fields without saving → required_history_fields
         is empty.

    Resolution strategy (union, then minus demographics):
      • collected_this_session  (fields the LLM explicitly marked complete)
      • required_history_fields (fields still on the queue right now)
      • _compute_required_history_fields(...) re-derived for the patient
        (deterministic, based on age/gender/marital_status — gives the full
        set of fields that SHOULD have been collected this session)

    This way even if the LLM walked through every question without ever
    emitting [FIELD_COMPLETE], we still know which fields the conversation
    covered and can hand them to the batch parser.
    """
    _DEMO = {"age", "gender", "marital_status"}
    fields: set[str] = set()
    fields.update(ctx.get("collected_this_session", []) or [])
    fields.update(ctx.get("required_history_fields", []) or [])
    try:
        derived = _compute_required_history_fields(
            ctx.get("patient", {}) or {},
            ctx.get("patient_history_data"),
        )
        fields.update(derived or [])
    except Exception as e:
        print(f"⚠️  [HistoryFields] _compute_required_history_fields failed: {e}")
    fields.difference_update(_DEMO)
    # Preserve a sensible order (alphabetical is fine — parser doesn't care)
    return sorted(fields)


def _parse_history_with_llm(messages: list, fields_to_extract: list[str]) -> dict:
    """
    Batch-extract structured medical history from a conversation.

    Takes the full message list and the list of history field names to
    populate. Calls the supervisor LLM (DeepSeek in production) ONCE with
    a strict JSON-only prompt, then validates and sanitises the output.

    Returns a dict mapping field names to extracted values. Fields that
    were never discussed in the conversation are omitted. Negative answers
    ("no allergies") become the string "none".

    This replaces the per-turn save approach which couldn't handle:
      - "Yes" followed by follow-up "Levothyroxine 75mcg" → maps to medications
      - "I have asthma and BP and thyroid" → splits into chronic_conditions
      - "Penicillin... actually also sulfa" → combines into drug_allergies
      - Patient correcting themselves later in the conversation
    """
    if not messages or not fields_to_extract:
        return {}

    # Build a clean transcript — only NURSE questions and PATIENT answers.
    # Skip tool messages, empty messages, and AI messages that look like
    # internal markers.
    transcript_lines = []
    for m in messages:
        content = str(getattr(m, "content", "") or "").strip()
        if not content:
            continue
        if m.type == "ai":
            # Skip directive markers and pure tool-call responses
            if content.startswith("[") and content.endswith("]"):
                continue
            transcript_lines.append(f"NURSE: {content[:400]}")
        elif m.type == "human":
            transcript_lines.append(f"PATIENT: {content[:400]}")
        # tool messages skipped

    if not transcript_lines:
        return {}

    transcript = "\n".join(transcript_lines)
    fields_list = "\n".join(f"  - {f}" for f in fields_to_extract)

    system_prompt = (
        "You extract structured medical history from a nurse-patient conversation. "
        "Return ONLY a single JSON object — no preamble, no explanation, no markdown fences."
    )

    user_prompt = f"""Extract values for these fields from the conversation below:
{fields_list}

Rules:
1. If the patient clearly denied or said "no/none/nope" → use "none"
2. Combine multi-turn answers — if nurse asked "any medications?" → patient said "Yes" → nurse asked "which?" → patient said "Levothyroxine 75mcg" → the value is "Levothyroxine 75mcg daily"
3. Combine list-style answers — "dust and pollen" → "dust, pollen"
4. If a field was never discussed at all → use null (omit it)
5. Do NOT invent details that aren't in the conversation
6. Keep each value concise (under 200 characters)

Conversation:
{transcript}

Return JSON now:"""

    try:
        from agents.llm_config import get_llm
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = get_llm(temperature=0)
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        raw = str(response.content).strip()
        # Strip markdown code fences if the model added them anyway
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```\s*$", "", raw)
        # Find the first JSON object substring
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            print(f"⚠️  [HistoryParser] No JSON found in response: {raw[:200]}")
            return {}
        parsed = json.loads(m.group(0))

        # Sanitise: only include known fields, coerce to string, cap length
        out: dict[str, str] = {}
        for fld in fields_to_extract:
            v = parsed.get(fld)
            if v is None or v == "":
                continue
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v if x)
            v = str(v).strip()
            if v and v.lower() not in {"null", "none mentioned", "not mentioned", "n/a"}:
                out[fld] = v[:300]
        print(f"📋 [HistoryParser] Extracted {len(out)}/{len(fields_to_extract)} fields: "
              f"{list(out.keys())}")
        return out
    except json.JSONDecodeError as e:
        print(f"⚠️  [HistoryParser] JSON parse failed: {e}")
        return {}
    except Exception as e:
        print(f"⚠️  [HistoryParser] LLM call failed: {e}")
        return {}


def _try_auto_save_history_field(ctx: dict, msg: str, session_id: str) -> None:
    """
    Symptom-bleed safety check during the patient-history collection phase.

    Field advancement is now LLM-driven via the [FIELD_COMPLETE] marker
    (handled in supervisor_node's post-LLM processing). This function only
    handles the safety case where a patient describes a SYMPTOM during the
    history phase — that message should never be treated as a history-field
    answer (e.g. "I have a severe headache" must NOT be saved as medications).

    When detected, we:
      • Store the message as initial_complaint_hint for triage to pick up later
      • Skip any field-advancement decision — the LLM decides via FIELD_COMPLETE

    Per-turn turn counts (used by _enforce_history_field_limits as a stuck-
    state safety net) are still incremented by that function — not here.
    """
    required = ctx.get("required_history_fields")
    if not required:
        return
    current_field = required[0]
    _DEMO = {"age", "gender", "marital_status"}
    if current_field in _DEMO:
        return    # demographics handled by _try_auto_save_demographic
    _HISTORY_FIELDS = {
        "chronic_conditions", "medications", "drug_allergies", "general_allergies",
        "family_history", "smoking_status", "vaccination_status",
        "menstrual_history", "lmp_date", "pregnancy_status", "obstetric_history",
        "fall_history",
    }
    if current_field not in _HISTORY_FIELDS:
        return

    patient_id = ctx["patient"].get("id")
    if not patient_id:
        return

    msg_clean = msg.strip()
    if not msg_clean:
        return
    msg_lower = msg_clean.lower()

    # ── Symptom-bleed safety check ─────────────────────────────────────────────
    # If the patient describes a symptom (e.g. "I have a headache") during
    # history collection, that's a complaint — not an answer to whichever
    # history field is current. Capture it as initial_complaint_hint and skip
    # any field advancement. The LLM should not emit [FIELD_COMPLETE] for this
    # message either, since the answer to the actual question wasn't given.
    _SYMPTOM_INDICATORS = (
        "headache", "head ache", "migraine", "fever", "nausea", "vomit",
        "dizzy", "dizziness", "pain", "ache", "hurt", "hurting",
        "i have", "i've had", "i'm having", "i am having", "i feel",
        "rash", "itch", "swollen", "cough", "sore throat", "burning",
        "chest pain", "shortness of breath", "trouble breathing", "stuck",
        "discomfort",
    )
    if current_field != "chronic_conditions":
        if any(ind in msg_lower for ind in _SYMPTOM_INDICATORS):
            print(f"⚠️  [HistorySafety] Message looks like a complaint, not a "
                  f"{current_field} answer: '{msg_clean[:60]}'")
            if not ctx.get("initial_complaint_hint"):
                ctx["initial_complaint_hint"] = msg_clean
                save_booking_context(session_id, ctx)
                print(f"💡 [HistorySafety] Stored as initial_complaint_hint")
            return
    # That's it — no more regex-based field advancement. The supervisor's
    # post-LLM [FIELD_COMPLETE] handler advances the field head when the LLM
    # decides the answer is satisfactory.


def _build_fallback_response(ctx: dict) -> str:
    """
    Generate a sensible patient-facing response when the LLM returns empty.
    Based entirely on ctx state — no LLM call. Patient should never see silence.
    """
    step = ctx.get("step", "collect_patient_history")
    p    = ctx.get("patient", {})
    name = p.get("name", "")
    first_name = name.split()[0] if name else ""

    if step == "collect_patient_history":
        remaining = ctx.get("required_history_fields") or []
        if remaining:
            field    = remaining[0]
            question = _QUESTION_FOR_FIELD.get(
                field, f"Could you tell me about your {field.replace('_', ' ')}?"
            )
            prefix = f"Thanks{', ' + first_name if first_name else ''}! " if not ctx.get("history_field_turn_count", {}).get(field) else ""
            return f"{prefix}{question}"
        else:
            hint = ctx.get("initial_complaint_hint")
            if hint:
                return f"Thank you for the information. You mentioned '{hint}' — let me start your assessment."
            return "Thank you! What brings you in today?"

    elif step == "collect_patient":
        hint = ctx.get("initial_complaint_hint")
        if hint:
            return f"I see you came in for {hint}. Let me start your assessment."
        return "What brings you in today?"

    elif step == "collect_doctor":
        spec = ctx.get("selected_doctor", {}).get("specialization", "")
        if spec:
            return f"Let me find available {spec} doctors for you."
        return "Let me find a suitable doctor based on your assessment."

    elif step == "collect_slot":
        d_name = ctx.get("selected_doctor", {}).get("name", "the doctor")
        avail  = ctx.get("availability_result", "")
        if avail:
            return f"Here are the available slots with Dr. {d_name}:\n{avail}\n\nWhich time works for you?"
        return f"Which day would you like to see Dr. {d_name} — today or tomorrow?"

    elif step == "await_confirmation":
        s      = ctx.get("pending_slot", {})
        d_name = ctx.get("selected_doctor", {}).get("name", "the doctor")
        date   = s.get("date", "")
        time   = s.get("time", "")
        return f"To confirm: appointment with Dr. {d_name} on {date} at {time}. Shall I book this? (Yes / No)"

    elif step == "completed":
        return "Your appointment has been booked. Is there anything else I can help you with?"

    return "I'm here to help. Could you please continue?"


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
    _advance_step(ctx)
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

    # ── Chat memory: persist user message + retrieve relevant past context ────
    patient_id    = ctx["patient"].get("id")
    last_user_msg = _last_human_text(raw_messages)
    if last_user_msg:
        store_message(session_id, patient_id, "user", last_user_msg)
    relevant_history = search_relevant(patient_id, last_user_msg) if patient_id and last_user_msg else ""
    if relevant_history:
        print(f"🧠 [Memory] Retrieved relevant past context ({len(relevant_history)} chars)")

    # ── Policy RAG lookup ─────────────────────────────────────────────────────
    # When the patient asks about hospital policy (fees, cancellation, hours, etc.)
    # find the relevant policy chunk and inject it. Skipped for tool-call-only turns.
    policy_context = search_policy(last_user_msg) if last_user_msg else ""
    if policy_context:
        print(f"📋 [PolicyRAG] Injecting policy context ({len(policy_context)} chars)")

    # ── Programmatic complaint hint capture ───────────────────────────────────
    # Only store if the first message is an actual complaint, not a greeting.
    # "hi", "hello", "good morning", "hey", "assalam o alaikum" etc. should
    # never be stored as the chief complaint.
    _GREETING_RE = re.compile(
        r"^(hi+|hey+|hello+|helo|salam|assalam|walaikum|good\s*(morning|evening|afternoon|day)|"
        r"hola|namaste|howdy|greetings?|yo|sup|what'?s\s*up|"
        # Administrative / booking phrases — not medical complaints
        r"i\s*(want|need|would like|'d like)\s*(to\s*)?(book|make|schedule|get)\s*(an?\s*)?(appointment|booking|slot|visit)|"
        r"(book|make|schedule)\s*(an?\s*)?(appointment|booking)|"
        r"i\s*want\s*(to\s*)?(see\s*)?(a\s*)?(doctor|physician|specialist)|"
        r"appointment\s*(please|chahiye|chahiye ga)?"
        r")\s*[\.,!?]*$",
        re.IGNORECASE,
    )
    _has_medical_word = re.search(
        r"\b(pain|ache|fever|cough|cold|nausea|dizzy|vomit|bleed|rash|itch|swollen|"
        r"throat|head|chest|back|stomach|ear|eye|nose|breathing|tired|weak|numb|burn|"
        r"injury|wound|cut|fracture|broke|break|hurt|symptom|ill|sick|unwell|problem|"
        r"complaint)\b",
        last_user_msg or "", re.IGNORECASE,
    )
    if (not ctx["patient"].get("id")
            and not ctx.get("initial_complaint_hint")
            and last_user_msg
            and not re.match(r"^[\+\d][\d\s\-]{8,14}$", last_user_msg.strip())
            and not _GREETING_RE.match(last_user_msg.strip())
            and _has_medical_word):   # only store if message has a medical word
        ctx["initial_complaint_hint"] = last_user_msg.strip()
        save_booking_context(session_id, ctx)
        print(f"💡 [AutoHint] Stored initial complaint hint: '{last_user_msg[:80]}'")

    # ── Auto-detect phone number in last message ──────────────────────────────
    # When patient gives their phone number, ctx["patient"]["phone"] is still None
    # because lookup_customer_profile hasn't been called yet. Without this, the
    # directive keeps showing "ask for phone" and the supervisor asks for name instead.
    if (not ctx["patient"]["phone"]
            and not ctx["patient"]["id"]
            and last_user_msg
            and re.match(r"^[\+\d][\d\s\-]{8,14}$", last_user_msg.strip())):
        ctx["patient"]["phone"] = last_user_msg.strip()
        save_booking_context(session_id, ctx)
        print(f"📱 [AutoPhone] Detected phone in message: '{ctx['patient']['phone']}'")

    # ── Auto-save demographic fields algorithmically ───────────────────────────
    # LLM sometimes skips update_patient_demographics and moves to next question.
    # For age/gender/marital_status, extract and save directly from patient message.
    if (ctx.get("step") == "collect_patient_history"
            and ctx["patient"].get("id")
            and last_user_msg):
        _try_auto_save_demographic(ctx, last_user_msg, session_id)
        # Then auto-save the current history field too (chronic_conditions,
        # medications, allergies, etc) — same idea, same problem: LLM skips
        # save_patient_history and moves on, causing data loss on force-advance.
        _try_auto_save_history_field(ctx, last_user_msg, session_id)

    # ── If history collection just finished, run the batch parser ─────────────
    # _try_auto_save_history_field sets history_pending_confirmation when the
    # last field's head is advanced. We then make ONE LLM call to extract
    # structured values from the whole history conversation, store them in
    # ctx["parsed_history"], and let _advance_step move us to the
    # confirm_patient_history step.
    #
    # Failure handling: parser may return {} on JSON error / LLM hiccup.
    # Retry once on the next turn. On second failure, skip the confirmation
    # step entirely with what we have (demographics + initial complaint) so
    # the patient never gets stuck staring at a blank record.
    if (ctx.get("history_pending_confirmation")
            and ctx.get("step") == "collect_patient_history"
            and not ctx.get("parsed_history_built")):

        _DEMO = {"age", "gender", "marital_status"}
        # Use the deterministic field list — robust to LLM skipping FIELD_COMPLETE.
        # See _history_fields_to_extract docstring for the union strategy.
        history_fields = _history_fields_to_extract(ctx)

        parse_attempts = ctx.get("history_parse_attempts", 0) + 1
        ctx["history_parse_attempts"] = parse_attempts

        parsed_fields: dict = {}
        if history_fields:
            print(f"📋 [HistoryParse] Running batch parser on {len(history_fields)} fields "
                  f"(attempt {parse_attempts}): {history_fields}")
            parsed_fields = _parse_history_with_llm(
                list(state.get("messages", [])),
                history_fields,
            )

        if parsed_fields or not history_fields:
            # Success (or nothing to parse — pure demographics case)
            parsed = ctx.setdefault("parsed_history", {})
            for k, v in parsed_fields.items():
                parsed[k] = v
            for demo_key in ("age", "gender", "marital_status"):
                v = ctx.get("patient", {}).get(demo_key)
                if v is not None and v != "":
                    parsed[demo_key] = v
            ctx["parsed_history_built"] = True
            # CRITICAL: clear required_history_fields so _advance_step can
            # move past collect_patient_history. Without this, _advance_step
            # sees remaining fields and returns early, never reaching the
            # history_pending_confirmation check → stuck forever.
            ctx["required_history_fields"] = []
            print(f"📋 [HistoryParse] Cleared required_history_fields → advancing to confirm step")
            _advance_step(ctx)
            save_booking_context(session_id, ctx)
        elif parse_attempts >= 2:
            # Two parse attempts failed — bail out gracefully. Skip the
            # confirmation step entirely with whatever we have so the
            # patient isn't stuck. Demographics survive, history fields
            # are left as whatever the auto-save captured (none, in the
            # new architecture). Triage will still run.
            print(f"⚠️  [HistoryParse] Parser failed {parse_attempts}x — "
                  f"skipping confirmation, proceeding to triage with degraded record")
            parsed = ctx.setdefault("parsed_history", {})
            for demo_key in ("age", "gender", "marital_status"):
                v = ctx.get("patient", {}).get(demo_key)
                if v is not None and v != "":
                    parsed[demo_key] = v
            ctx["parsed_history_built"] = True
            ctx["history_confirmed"] = True            # skip confirm step
            ctx["history_pending_confirmation"] = False
            ctx["history_parse_degraded"] = True        # diagnostic flag
            _advance_step(ctx)
            save_booking_context(session_id, ctx)
        else:
            # First failure — keep the pending flag set so we retry next turn.
            # Don't mark parsed_history_built. Don't advance step.
            print(f"⚠️  [HistoryParse] Parser returned empty (attempt {parse_attempts}/2) — "
                  f"will retry next turn")
            save_booking_context(session_id, ctx)

    # ── Confirm-step CORRECTION handler: re-parse when patient pushes back ────
    # If the patient sees the confirmation summary and says "no" or otherwise
    # signals a correction ("actually my meds are X"), we re-run the parser.
    # The parser sees the new correction message in the conversation history
    # and produces an updated record. The directive then re-renders it for
    # another confirmation pass. Capped at 3 attempts to avoid infinite loops.
    if (ctx.get("step") == "confirm_patient_history"
            and last_user_msg
            and not _YES_RE.search(last_user_msg)):

        _CORRECTION_RE = re.compile(
            r"(\bno\b|\bnope\b|\bnah\b|\bwrong\b|\bincorrect\b|\bactually\b"
            r"|\bchange\b|\bupdate\b|\bfix\b|\bedit\b|\bcorrect\b"
            r"|that\'?s\s+not|that\s+is\s+not|let\s+me|i\s+meant)",
            re.IGNORECASE,
        )
        if _CORRECTION_RE.search(last_user_msg):
            attempts = ctx.get("history_reparse_attempts", 0)
            _DEMO = {"age", "gender", "marital_status"}
            # Same robustness: don't rely solely on collected_this_session.
            history_fields = _history_fields_to_extract(ctx)
            if history_fields and attempts < 3:
                print(f"📋 [HistoryReparse] Patient signalled correction "
                      f"(attempt {attempts + 1}/3) — re-running parser with full conversation")
                parsed_fields = _parse_history_with_llm(
                    list(state.get("messages", [])),
                    history_fields,
                )
                if parsed_fields:
                    # OVERWRITE — the latest correction should supersede prior values
                    parsed = ctx.setdefault("parsed_history", {})
                    for k, v in parsed_fields.items():
                        parsed[k] = v
                    # Re-mirror demographics (in case patient corrected age/gender too)
                    for demo_key in ("age", "gender", "marital_status"):
                        v = ctx.get("patient", {}).get(demo_key)
                        if v is not None and v != "":
                            parsed[demo_key] = v
                    ctx["history_reparse_attempts"] = attempts + 1
                    save_booking_context(session_id, ctx)
                    print(f"📋 [HistoryReparse] Updated record — directive will re-show "
                          f"with new values: {list(parsed.keys())}")
                else:
                    print(f"⚠️  [HistoryReparse] Re-parse returned empty — keeping previous record")
            elif attempts >= 3:
                print(f"⚠️  [HistoryReparse] Hit 3-attempt cap — letting LLM handle "
                      f"this turn conversationally without re-parse")

    # ── Safety net: if we arrived at confirm_patient_history with an empty
    # parsed_history record, run the parser NOW. This is the last line of
    # defense — happens when either:
    #   (a) the FieldFallback path advanced the step but the parser failed
    #   (b) some other path set history_pending_confirmation+parsed_history_built
    #       without populating parsed_history
    # Without this, the directive renders an empty bullet list and the LLM
    # hallucinates a save tool call to compensate.
    if (ctx.get("step") == "confirm_patient_history"
            and not ctx.get("history_confirmed")
            and not ctx.get("history_safety_parse_done")):

        parsed_existing = ctx.get("parsed_history") or {}
        _DEMO = {"age", "gender", "marital_status"}
        has_non_demo = any(
            k not in _DEMO and v is not None and str(v).strip()
            for k, v in parsed_existing.items()
        )
        if not has_non_demo:
            print("🛟 [HistorySafetyNet] confirm_patient_history reached with empty "
                  "parsed_history — running parser as last resort")
            history_fields = _history_fields_to_extract(ctx)
            parsed_fields: dict = {}
            if history_fields:
                try:
                    parsed_fields = _parse_history_with_llm(
                        list(state.get("messages", [])),
                        history_fields,
                    )
                except Exception as e:
                    print(f"⚠️  [HistorySafetyNet] parser raised: {e}")
                    parsed_fields = {}

            parsed = ctx.setdefault("parsed_history", {})
            for k, v in parsed_fields.items():
                parsed[k] = v
            # Mirror demographics
            for demo_key in ("age", "gender", "marital_status"):
                v = ctx.get("patient", {}).get(demo_key)
                if v is not None and v != "":
                    parsed[demo_key] = v
            ctx["history_safety_parse_done"] = True
            print(f"🛟 [HistorySafetyNet] parsed_history populated with "
                  f"{len(parsed)} keys: {list(parsed.keys())}")
            save_booking_context(session_id, ctx)

    # ── Confirm-step handler: programmatic YES advances the flow ──────────────
    # During confirm_patient_history we show the patient the parsed record and
    # ask them to confirm. On YES we (a) do the single batch save of all
    # parsed history fields to the database, (b) clear the flag, (c) re-compute
    # the step (which will move forward into triage), and (d) queue a brief
    # AI confirmation message so the patient isn't left hanging. We do this
    # BEFORE the LLM call so the model doesn't accidentally re-ask the
    # confirmation question or hallucinate the save.
    if (ctx.get("step") == "confirm_patient_history"
            and last_user_msg
            and _YES_RE.search(last_user_msg)):
        print("✅ [HistoryConfirm] Patient confirmed history record — running batch save")

        parsed     = ctx.get("parsed_history") or {}
        patient_id = ctx.get("patient", {}).get("id")
        _DEMO      = {"age", "gender", "marital_status"}

        # Save the history fields in ONE call (demographics are already
        # in the patients table from per-turn update_patient_demographics)
        history_payload = {
            k: v for k, v in parsed.items()
            if k not in _DEMO and v is not None and str(v).strip()
        }

        # ── LAST-CHANCE PARSE on YES with empty payload ───────────────────────
        # If the patient said YES but we somehow have nothing to save, run the
        # parser one more time. Better than silently saving an empty record.
        if patient_id and not history_payload and not ctx.get("history_yes_reparse_done"):
            print("🛟 [HistoryConfirm] Empty payload at YES — last-chance parser run")
            ctx["history_yes_reparse_done"] = True
            history_fields = _history_fields_to_extract(ctx)
            if history_fields:
                try:
                    parsed_fields = _parse_history_with_llm(
                        list(state.get("messages", [])),
                        history_fields,
                    )
                    if parsed_fields:
                        for k, v in parsed_fields.items():
                            parsed[k] = v
                        history_payload = {
                            k: v for k, v in parsed.items()
                            if k not in _DEMO and v is not None and str(v).strip()
                        }
                        print(f"🛟 [HistoryConfirm] Last-chance parse recovered "
                              f"{len(history_payload)} field(s)")
                except Exception as e:
                    print(f"⚠️  [HistoryConfirm] Last-chance parse failed: {e}")

        if patient_id and history_payload:
            try:
                from agents.mcp_tools import save_patient_history as _sph
                result = _sph.invoke({"patient_id": patient_id, **history_payload})
                if "saved successfully" in str(result).lower():
                    print(f"✅ [HistoryConfirm] Batch saved {len(history_payload)} fields: "
                          f"{list(history_payload.keys())}")
                else:
                    print(f"⚠️  [HistoryConfirm] Batch save returned unexpected result: {result}")
            except Exception as e:
                print(f"❌ [HistoryConfirm] Batch save failed: {e}")
        else:
            print(f"⚠️  [HistoryConfirm] Nothing to save (patient_id={patient_id}, "
                  f"fields={len(history_payload)})")

        ctx["history_pending_confirmation"] = False
        ctx["history_confirmed"] = True
        _advance_step(ctx)
        save_booking_context(session_id, ctx)
        # Short programmatic confirmation message — keeps UI flowing.
        first_name = (ctx.get("patient", {}).get("name") or "").split()[0]
        hint       = ctx.get("initial_complaint_hint") or ""
        prefix     = f"Thanks{', ' + first_name if first_name else ''}! " if first_name else "Thanks! "
        if hint:
            ack_msg = f"{prefix}Your medical history is confirmed. Now let me ask about your reason for visiting today — you mentioned '{hint}'. Could you tell me a bit more?"
        else:
            ack_msg = f"{prefix}Your medical history is confirmed. What brings you in today?"
        return {
            "messages":              [AIMessage(content=ack_msg)],
            "booking_context":       ctx,
            "triage_active":         False,
            "interaction_completed": False,
            "extracted_symptom":     state.get("extracted_symptom"),
            "session_id":            session_id,
        }

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

{f"── MEMORY: RELEVANT PAST CONTEXT (verified before this turn) ──{chr(10)}{relevant_history}" if relevant_history else ""}

{policy_context}

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
        if ctx.get("human_handoff_pending") or ctx.get("step") == "await_human":
            llm_to_use = _get_llm_with_handoff_tools(current_step)
        elif current_step == "collect_patient" and not ctx.get("triage_completed"):
            # Pre-triage: no tools — supervisor just asks "what brings you in today?"
            # Having recommend_specialist_tool bound causes the model to call it instead
            llm_to_use = get_llm(temperature=0.1)
            print("🔧 [LLM] Pre-triage collect_patient — no tools bound")
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

    # Strip <think>...</think> blocks that Qwen sometimes leaks into its response
    raw_content = str(response.content)
    cleaned_content = re.sub(r"<think>.*?</think>", "", raw_content, flags=re.DOTALL).strip()
    if cleaned_content != raw_content:
        print("🧹 [Supervisor] Stripped <think> block from response")
        response = AIMessage(content=cleaned_content, tool_calls=getattr(response, "tool_calls", None) or [])

    # ── Strip hallucinated DSML / inline tool-call markup ─────────────────────
    # Qwen 32B on Groq occasionally outputs DeepSeek-style tool calls AS PLAIN
    # TEXT (the `<｜DSML｜>` markup, sometimes mixed with `<tool_call>` tags
    # or fenced JSON). This leaks garbled syntax to the WhatsApp user and
    # the tool never actually runs because LangChain doesn't see it as a
    # real tool_call.
    #
    # Strategy: HARD STOP at the first sign of tool-call markup. Once the
    # model starts hallucinating tool syntax, everything that follows is
    # garbage — preamble like "Alright, let me save your..." is kept, the
    # markup and everything after it is deleted. If the cleaned result is
    # too short to be useful, substitute a clean step-appropriate message.
    raw_content_2 = str(response.content)
    _DSML_HARD_STOP = re.compile(
        r"[<\[]?[｜\|]+\s*DSML[\s\S]*$"
        r"|<\s*\|\s*[A-Za-z]{2,8}\s*\|[\s\S]*$"
        r"|<\s*/?\s*tool_calls?\b[\s\S]*$"
        r"|<\s*/?\s*invoke\b[\s\S]*$"
        r"|<\s*/?\s*parameter\b[\s\S]*$"
        r"|```\s*(?:tool_code|tool_calls?|function_call)\b[\s\S]*$",
        re.IGNORECASE,
    )
    stripped_dsml = _DSML_HARD_STOP.sub("", raw_content_2)
    was_stripped  = stripped_dsml != raw_content_2

    if was_stripped:
        # Only collapse whitespace when we actually cut something — otherwise
        # we'd squash the newline-formatted bullet lists in normal responses.
        cleaned_dsml = re.sub(r"[ \t]+", " ", stripped_dsml).strip()
        cleaned_dsml = re.sub(r"\n{3,}", "\n\n", cleaned_dsml)

        print("🧹 [Supervisor] Stripped hallucinated DSML/tool-call markup from response")
        # If stripping left only the "Alright, let me save..." preamble (or
        # nothing), substitute a clean step-appropriate message so the
        # patient never sees fragments or a "let me save" promise that won't
        # actually happen as the model expects.
        is_save_preamble = bool(
            re.match(
                r"^\s*(alright|ok(ay)?|sure|now|first|let me|i'?ll)\s*[,!.\-]*\s*"
                r"(let me\s+)?(save|store|record|put|update|persist)\b",
                cleaned_dsml, re.IGNORECASE,
            )
        )
        if len(cleaned_dsml) < 10 or is_save_preamble:
            cur_step = ctx.get("step", "")
            if cur_step == "confirm_patient_history":
                cleaned_dsml = (
                    "Let me show you what I've recorded — please confirm if everything looks right."
                )
            else:
                cleaned_dsml = "One moment please."
            print(f"🧹 [Supervisor] Substituted clean message for step={cur_step!r}")
        response = AIMessage(
            content   = cleaned_dsml,
            tool_calls= getattr(response, "tool_calls", None) or [],
        )

    response_text         = str(response.content)
    extracted_symptom     = state.get("extracted_symptom")
    triage_active         = state.get("triage_active") or False
    interaction_completed = state.get("interaction_completed") or False

    # ── FALLBACK: LLM skipped [FIELD_COMPLETE] and went straight to complaint ──
    # The LLM sometimes collects all history fields in one conversational pass
    # without emitting [FIELD_COMPLETE] tags. When it then asks "What brings
    # you in today?" the fields are answered in context but not recorded.
    # Detect this and force-trigger the batch parser so we don't skip confirmation.
    _COMPLAINT_Q_RE = re.compile(
        r"(what brings you in|what('?s| is) (the reason|your reason|your concern)|"
        r"what symptoms|what('?s| is) (wrong|bothering|the problem)|"
        r"how can (i|we) help you today|reason for (your )?visit)",
        re.IGNORECASE,
    )
    if (ctx.get("step") == "collect_patient_history"
            and ctx.get("required_history_fields")
            and not ctx.get("history_pending_confirmation")
            and _COMPLAINT_Q_RE.search(response_text)):
        print("⚠️  [FieldFallback] LLM asked complaint question while fields remain — "
              "running batch parse now (LLM skipped [FIELD_COMPLETE])")

        # ── CRITICAL: Run the parser inline BEFORE advancing the step ─────────
        # Without this, parsed_history stays empty and the next turn's
        # confirm_patient_history directive shows the patient an empty record.
        # The parser block at line ~2579 only fires when step is still
        # collect_patient_history — but we're about to advance past that.
        history_fields = _history_fields_to_extract(ctx)
        parsed_fields: dict = {}
        if history_fields:
            print(f"📋 [FieldFallback→HistoryParse] Extracting {len(history_fields)} fields "
                  f"from conversation: {history_fields}")
            try:
                parsed_fields = _parse_history_with_llm(
                    list(state.get("messages", [])),
                    history_fields,
                )
            except Exception as e:
                print(f"⚠️  [FieldFallback→HistoryParse] parser raised: {e}")
                parsed_fields = {}

        parsed = ctx.setdefault("parsed_history", {})
        for k, v in parsed_fields.items():
            parsed[k] = v
        # Mirror demographics into parsed_history for the confirmation render
        for demo_key in ("age", "gender", "marital_status"):
            v = ctx.get("patient", {}).get(demo_key)
            if v is not None and v != "":
                parsed[demo_key] = v
        ctx["parsed_history_built"] = True
        ctx["history_pending_confirmation"] = True
        ctx["required_history_fields"] = []   # clear so _advance_step proceeds
        ctx["history_parse_attempts"] = ctx.get("history_parse_attempts", 0) + 1
        print(f"📋 [FieldFallback→HistoryParse] parsed_history now has "
              f"{len(parsed)} keys: {list(parsed.keys())}")

        _advance_step(ctx)
        save_booking_context(session_id, ctx)
        # Replace the complaint question with a hold message — confirmation first
        response = AIMessage(
            content   = "Thank you! Let me just confirm what I've noted before we proceed.",
            tool_calls= [],
        )

    # ── [FIELD_COMPLETE] handler — LLM-driven field advancement ────────────────
    # During collect_patient_history (history-fields phase), the directive
    # instructs the LLM to emit [FIELD_COMPLETE] whenever it has a clear
    # answer for the current field. This replaces brittle regex on the
    # patient's message — the LLM has full context and can correctly handle
    # multi-turn answers, "no X but yes Y", corrections, etc.
    if "[FIELD_COMPLETE]" in response_text and ctx.get("step") == "collect_patient_history":
        required = ctx.get("required_history_fields") or []
        if required:
            current_field = required[0]
            _DEMO = {"age", "gender", "marital_status"}
            if current_field not in _DEMO:
                reqs = list(required)
                reqs.remove(current_field)
                ctx["required_history_fields"] = reqs
                coll = ctx.setdefault("collected_this_session", [])
                if current_field not in coll:
                    coll.append(current_field)
                counts = ctx.setdefault("history_field_turn_count", {})
                counts[current_field] = 0
                print(f"📝 [FieldComplete] LLM signalled '{current_field}' is complete — advanced")
                if not reqs:
                    ctx["history_pending_confirmation"] = True
                    print(f"📋 [FieldComplete] All history fields done — flagging for batch parse")
                save_booking_context(session_id, ctx)
            else:
                print(f"⚠️  [FieldComplete] LLM emitted marker for demographic field "
                      f"'{current_field}' — ignored (demographics use update_patient_demographics)")
        # Strip the marker from the response — patient should never see it
        cleaned = re.sub(r"\[FIELD_COMPLETE\]", "", response_text).strip()
        response = AIMessage(
            content   = cleaned,
            tool_calls= getattr(response, "tool_calls", None) or [],
        )
        response_text = cleaned

    _HISTORY_STEPS = {"collect_patient_history", "confirm_patient_history"}
    if "[SYMPTOM_LOGGED:" in response_text and ctx.get("step") not in _HISTORY_STEPS:
        start   = response_text.find("[SYMPTOM_LOGGED:") + 16
        end     = response_text.find("]", start)
        symptom = response_text[start:end].strip()
        # Strip common filler prefixes so MedGemma gets a clean complaint
        symptom = re.sub(
            r"^(yes[,\s]+|no[,\s]+|i (am|have|am having|was having|got|feel|have got)\s+a?\s*)",
            "", symptom, flags=re.IGNORECASE,
        ).strip()
        extracted_symptom = symptom
        if not ctx.get("prime_complaint"):
            ctx["prime_complaint"] = symptom
            _advance_step(ctx)
            save_booking_context(session_id, ctx)
        print(f"📝 [Supervisor] symptom='{symptom}'")
        # Programmatically trigger triage — never rely on LLM emitting [START_TRIAGE]
        if not ctx.get("triage_completed"):
            triage_active = True
            ctx["triage_active"] = True
            response = AIMessage(content="")   # triage_node will speak
            save_booking_context(session_id, ctx)
            print("🚦 [Supervisor] Triage auto-triggered programmatically after SYMPTOM_LOGGED")
    elif "[SYMPTOM_LOGGED:" in response_text and ctx.get("step") in _HISTORY_STEPS:
        start   = response_text.find("[SYMPTOM_LOGGED:") + 16
        end     = response_text.find("]", start)
        symptom = response_text[start:end].strip()
        # Store the complaint hint so patient doesn't have to repeat it later
        if not ctx.get("initial_complaint_hint"):
            ctx["initial_complaint_hint"] = symptom
            print(f"📝 [Supervisor] 💾 Complaint hint stored: '{symptom}' (used post-history)")
        print(f"📝 [Supervisor] ⛔ SYMPTOM_LOGGED suppressed during history phase: '{symptom}'")
        save_booking_context(session_id, ctx)

        # If all history fields are done, advance step so triage fires next turn
        required = ctx.get("required_history_fields")
        if required is not None and len(required) == 0:
            _advance_step(ctx)
            save_booking_context(session_id, ctx)
            print(f"📝 [Supervisor] ✅ Step advanced to '{ctx['step']}' — triage fires next turn")
            # Let the existing response through (stripped) — no redirect needed
            cleaned = re.sub(r"\[SYMPTOM_LOGGED:[^\]]*\]", "", response_text)
            cleaned = re.sub(r"\[START_TRIAGE\]", "", cleaned).strip()
            response = AIMessage(content=cleaned, tool_calls=[])
        else:
            # History still in progress — cancel all tool calls, redirect to current field
            current_field = (required or [""])[0] if required else ""
            redirect_q    = _QUESTION_FOR_FIELD.get(current_field, "")
            redirect_msg  = (
                f"I noted that — we'll get to your {symptom} shortly. "
                f"First, let me finish collecting your medical history. "
                f"{redirect_q}"
            ).strip()
            print(f"📝 [Supervisor] ↩️  Redirecting to history field '{current_field}'")
            response = AIMessage(content=redirect_msg, tool_calls=[])  # ← cancel tool calls

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

    if "[START_TRIAGE]" in response_text and ctx.get("triage_completed"):
        cleaned = re.sub(r"\[START_TRIAGE\]", "", response_text).strip()
        print("🚦 [Supervisor] ⛔ START_TRIAGE suppressed — triage already complete")
        response = AIMessage(content=cleaned)
    elif "[START_TRIAGE]" in response_text and ctx.get("step") != "collect_patient_history":
        triage_active = True
        print("🚦 [Supervisor] triage_active = True — muting supervisor reply, triage_node will speak")
        response = AIMessage(content="")
    elif "[START_TRIAGE]" in response_text and ctx.get("step") == "collect_patient_history":
        print("🚦 [Supervisor] ⛔ START_TRIAGE suppressed — still in collect_patient_history phase")
        # Defensive: the LLM often adds "describe your symptoms" text that confuses
        # the patient — they answer with the complaint, which then triggers the
        # history auto-save to save the complaint as the current field. Replace
        # the response with the proper field question so we stay on track.
        remaining = ctx.get("required_history_fields") or []
        if remaining:
            current_field = remaining[0]
            question = _QUESTION_FOR_FIELD.get(
                current_field,
                f"Could you tell me about your {current_field.replace('_', ' ')}?"
            )
            replacement = f"Thanks. Just a couple more questions to finish your history — {question}"
            response = AIMessage(content=replacement)
            print(f"🚦 [Supervisor] Replaced LLM response with proper field question: '{question[:60]}'")
        else:
            # No remaining fields but step hasn't advanced yet — accept START_TRIAGE.
            # Clear the tag and let the entry_router handle the next turn.
            cleaned = re.sub(r"\[START_TRIAGE\]", "", response_text).strip()
            cleaned = re.sub(r"\[SYMPTOM_LOGGED:[^\]]*\]", "", cleaned).strip()
            ctx["step"] = "collect_patient"
            save_booking_context(session_id, ctx)
            response = AIMessage(content=cleaned)
            print(f"🚦 [Supervisor] All fields done — accepted START_TRIAGE, step advanced to collect_patient")

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

    if (not str(response.content).strip()
            and not triage_active
            and not getattr(response, "tool_calls", None)):
        print("⚠️  [Supervisor] EMPTY response with no tool calls — generating contextual fallback")
        print(f"   step={ctx.get('step')}")
        response = AIMessage(content=_build_fallback_response(ctx))

    # ── Chat memory: persist assistant response ───────────────────────────────
    assistant_text = str(response.content).strip()
    if assistant_text and patient_id:
        store_message(session_id, patient_id, "assistant", assistant_text)

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
def entry_router(state):
    ctx = state.get("booking_context", {})

    # 1. SOAP note — fires once AFTER patient says goodbye (interaction_completed)
    # NOT immediately on booking confirmation — patient may still have questions.
    if state.get("interaction_completed") and not ctx.get("diagnostic_report"):
        print("🔀 [EntryRouter] interaction complete + no report → diagnostic_node")
        return "diagnostic_node"

    # 2. LLM Judge — fires after SOAP, only when USE_LLM_JUDGE=true
    if (_USE_LLM_JUDGE
            and state.get("interaction_completed")
            and ctx.get("diagnostic_report")
            and not ctx.get("judge_report")):
        print("🔀 [EntryRouter] SOAP done → judge_node")
        return "judge_node"

    if ctx.get("triage_active"):
        return "triage_node"

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

# Judge node — only active when USE_LLM_JUDGE=true, runs after diagnostic
if _USE_LLM_JUDGE:
    try:
        from agents.judge_agent import judge_node
        builder.add_node("judge_node", judge_node)
        builder.add_edge("judge_node", END)
        print("🔧 [Graph] judge_node added to graph")
    except ImportError:
        print("⚠️  [Graph] judge_agent not found — judge_node skipped")

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