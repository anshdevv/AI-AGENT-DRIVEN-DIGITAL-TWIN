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
        "step":                     "collect_patient",
        "prime_complaint":          None,
        "recommended_specialist":   None,
        "triage_completed":         False,
        "triage_questions_asked":   0,          # incremented each turn in triage_node
        "triage_qa":                [],         # Q&A pairs — saved here AND flushed to Supabase
        "medgemma_raw_history":     [],         # Parallel history: only MedGemma's own raw outputs
        "accumulated_symptoms":     [],         # patient answers collected across all triage turns
        "symptom_context_block":    None,       # turn-1 lookup result (cached, injected into MedGemma prompt)
        "final_symptom_match":      None,       # lookup result after full triage (used for specialist routing)
        "patient": {
            "id":    None,
            "name":  None,
            "phone": None,
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
    p = ctx["patient"]
    d = ctx["selected_doctor"]
    s = ctx["pending_slot"]

    print(f"\n🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"🔀 [AdvanceStep] patient.id      = {p['id']}         ← must be set")
    print(f"🔀 [AdvanceStep] patient.name    = {p['name']}")
    print(f"🔀 [AdvanceStep] doctor.id       = {d['id']}         ← must be set")
    print(f"🔀 [AdvanceStep] doctor.name     = {d['name']}")
    print(f"🔀 [AdvanceStep] pending_slot    = {s['date']} at {s['time']}  ← both must be set")
    print(f"🔀 [AdvanceStep] confirmed       = {ctx['appointment']['confirmed']}")
    print(f"🔀 [AdvanceStep] booking_id      = {ctx['appointment'].get('booking_id')}")

    if ctx["appointment"]["confirmed"]:
        ctx["step"] = "completed"
        print(f"🔀 [AdvanceStep] → step = 'completed' ✅ (appointment already confirmed)")
        print(f"🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return
    if p["id"] and d["id"] and s["date"] and s["time"]:
        ctx["step"] = "await_confirmation"
        print(f"🔀 [AdvanceStep] → step = 'await_confirmation' ✅ all data present")
        print(f"🔀 [AdvanceStep] ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        return
    if not p["id"]:
        ctx["step"] = "collect_patient"
        print(f"🔀 [AdvanceStep] → step = 'collect_patient' ❌ missing patient.id")
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
    print(f"🔀 [AdvanceStep] → step = 'collect_slot' ❌ missing slot fields: {', '.join(missing)}")
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

    # ── GUARD: only trigger triage when step is collect_patient AND flag is unset.
    # If step has already advanced (collect_doctor, collect_slot, etc.) triage
    # clearly happened — don't re-trigger it even if the flag was somehow lost.
    if not ctx.get("triage_completed") and step == "collect_patient":
        lines.append("YOUR NEXT ACTION: Medical Triage.")
        lines.append("  If the user mentions ANY medical symptom, you MUST output exactly:")
        lines.append("  [SYMPTOM_LOGGED: <symptom>]")
        lines.append("  [START_TRIAGE]")
        lines.append("  DO NOT ask for phone number, name, or try to book until triage is finished!")

    elif step == "collect_patient":
        specialist = ctx.get("recommended_specialist") or "a specialist"
        lines.append("YOUR NEXT ACTION: Triage is done. Start the booking flow.")
        lines.append(f"  1. First, tell the patient warmly: triage is complete and based on their")
        lines.append(f"     symptoms we recommend seeing a {specialist}.")
        lines.append(f"  2. Then ask for their phone number so you can look up their profile.")
        lines.append(f"  3. Once you have the phone number, call lookup_customer_profile.")
        lines.append("  DO NOT call lookup_customer_profile if patient.id is already set.")

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


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        print("🔧 [LLM] Binding tools (once at startup)...")
        _llm_with_tools = get_llm(temperature=0.1).bind_tools(ALL_TOOLS)
        print(f"✅ [LLM] Tools bound: {[t.name for t in ALL_TOOLS]}")
    return _llm_with_tools


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

_PATIENT_ID_RE   = re.compile(r"['\"]id['\"]\s*:\s*(\d+)")
_PATIENT_NAME_RE = re.compile(r"['\"]name['\"]\s*:\s*['\"]([A-Za-z][A-Za-z ]{1,40}?)['\"]")
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

    elif tool_name == "recommend_specialist_tool":
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

    for tc in tool_calls:
        tool_name    = tc.get("name", "")
        tool_args    = tc.get("args", {})
        tool_call_id = tc.get("id", "")

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
    try:
        response = _invoke_with_retry(_get_llm_with_tools(), [sys_prompt] + safe_messages)
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

    if "[SYMPTOM_LOGGED:" in response_text:
        start   = response_text.find("[SYMPTOM_LOGGED:") + 16
        end     = response_text.find("]", start)
        symptom = response_text[start:end].strip()
        extracted_symptom = symptom
        if not ctx.get("prime_complaint"):
            ctx["prime_complaint"] = symptom
            _advance_step(ctx)
            save_booking_context(session_id, ctx)
        print(f"📝 [Supervisor] symptom='{symptom}'")

    if "[START_TRIAGE]" in response_text:
        triage_active = True
        print("🚦 [Supervisor] triage_active = True — muting supervisor reply, triage_node will speak")
        response = AIMessage(content="")

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