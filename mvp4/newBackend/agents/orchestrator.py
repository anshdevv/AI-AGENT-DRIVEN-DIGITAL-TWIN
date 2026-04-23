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
        "step":                     "collect_patient",
        "prime_complaint":          None,
        "recommended_specialist":   None,
        "triage_completed":         False,
        "triage_questions_asked":   0,          # incremented each turn in triage_node
        "triage_qa":                [],         # Q&A pairs — saved here AND flushed to Supabase
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
    if ctx["appointment"]["confirmed"]:
        ctx["step"] = "completed"
        return
    p = ctx["patient"]
    d = ctx["selected_doctor"]
    s = ctx["pending_slot"]
    if p["id"] and d["id"] and s["date"] and s["time"]:
        ctx["step"] = "await_confirmation"
        return
    if not p["id"]:
        ctx["step"] = "collect_patient"
        return
    if not d["id"]:
        ctx["step"] = "collect_doctor"
        return
    ctx["step"] = "collect_slot"


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

    if not ctx.get("triage_completed"):
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
        lines.append("YOUR NEXT ACTION: Find and select a doctor.")
        lines.append(f"  Call get_doctors_by_specialization('{spec}') to list doctors.")
        lines.append("  Ask which doctor the patient prefers, then call get_doctor_profile to confirm ID.")

    elif step == "collect_slot":
        lines.append("YOUR NEXT ACTION: Find an available time slot.")
        lines.append(f"  Call find_provider_availability(doctor_id={d['id']}, date='today' or ask patient).")
        lines.append(f"  TODAY = {today} | TOMORROW = {tomorrow}")
        lines.append("  DO NOT call get_doctor_schedule — use find_provider_availability.")

    elif step == "await_confirmation":
        lines.append("YOUR NEXT ACTION: Show summary. Ask 'Shall I confirm? (yes/no)'. Call NO tools.")
        lines.append(f"  Patient : {p['name']} (ID={p['id']})")
        lines.append(f"  Doctor  : Dr. {d['name']} (ID={d['id']})")
        lines.append(f"  Date    : {s['date']} at {s['time']}")
        lines.append("  DO NOT call create_booking — code handles it after YES.")

    elif step == "completed":
        lines.append("YOUR NEXT ACTION: Booking is confirmed. Follow these steps in order:")
        lines.append(f"  1. Confirm the booking warmly: Dr. {d['name']}, {s['date']} at {s['time']}, Booking ID={ctx['appointment']['booking_id']}")
        lines.append("  2. Ask: 'Is there anything else I can help you with today?'")
        lines.append("  3. If the patient says no / goodbye / nothing else → output [END_CALL]")
        lines.append("  4. If they have more questions, answer them first, then ask again.")
        lines.append("  ⛔ DO NOT call any tools.")

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


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

MAX_HISTORY_MESSAGES = 20   # Raised from 12 — triage now runs up to 8 rounds
MAX_TOOLS_PER_TURN   = 2    # Hard cap — prevents Groq 400 from long tool chains

_YES_RE = re.compile(
    r"\b(yes|yeah|yep|yup|confirm|book\s*it|go\s*ahead|sure|ok|okay|"
    r"haan|ji|bilkul|theek\s*hai|kar\s*do|confirm\s*kar)\b",
    re.IGNORECASE,
)
_INVALID_VALUES = {"unknown", "none", "null", "n/a", "", "undefined", "?"}
_TOOL_REQUIRED_ARGS = {
    "create_booking":          {"patient_id": "patient ID", "doctor_id": "doctor ID", "date": "date", "time": "time"},
    "lookup_customer_profile": {"phone": "patient phone number"},
}
_TIME_RE = re.compile(r"\b(\d{1,2}:\d{2})\b")
_DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


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

_PATIENT_ID_RE   = re.compile(r"['\"]id['\"]\s*:\s*(\d+)")                          # matches 'id': 3
_PATIENT_NAME_RE = re.compile(r"['\"]name['\"]\s*:\s*['\"]([A-Za-z][A-Za-z ]{1,40}?)['\"]")  # matches 'name': 'ashal'
_DOCTOR_ID_RE    = re.compile(r"\(ID:\s*(\d+)\)")
_DOCTOR_NAME_RE  = re.compile(r"Dr\.\s+([A-Za-z][A-Za-z .]{1,40}?)(?:\s*[\(|,\n]|$)")
_DOCTOR_SPEC_RE  = re.compile(r"\|\s*([A-Za-z][A-Za-z /]+?)\s*(?:Fee|$|\|)")
_BOOKING_ID_RE   = re.compile(r"(?i)Appointment\s+ID:\s*(\d+)")
_SPECIALIST_RE   = re.compile(r"(?i)(?:Recommended specialist|specialist|see a?n?)\s*[:\-]\s*([A-Za-z][A-Za-z /]{3,40})")


def _extract_from_tool_result(tool_name: str, tool_args: dict, result_str: str, ctx: dict) -> bool:
    changed = False
    print(f"🔍 [ToolExtract] Processing '{tool_name}' result")

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

    elif tool_name == "get_doctors_by_specialization":
        print(f"   result preview: {result_str[:300]}")
        if not ctx["selected_doctor"]["specialization"]:
            spec = tool_args.get("specialization")
            if spec:
                ctx["selected_doctor"]["specialization"] = str(spec)
                changed = True
                print(f"   → doctor.spec (from args) = '{spec}'")

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
        # Slot captured later in _try_extract_pending_slot after user picks a time

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
        print(f"   → step advanced to '{ctx['step']}'")

    return changed


# ═══════════════════════════════════════════════════════════════════
# TOOL EXECUTOR NODE
# ═══════════════════════════════════════════════════════════════════

def tool_executor_node(state: ConversationState) -> dict:
    session_id = state.get("session_id") or "default"
    ctx        = state.get("booking_context") or load_booking_context(session_id)
    messages   = list(state.get("messages", []))

    print(f"\n🔧 [ToolExecutor] Entering — step={ctx['step']}  session={session_id}")

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

        # ── GUARD 2: Never re-fetch doctor profile if ID already set ─
        if tool_name in ("get_doctor_profile", "get_doctor_schedule") and ctx["selected_doctor"]["id"]:
            msg = (
                f"Doctor already loaded: Dr. {ctx['selected_doctor']['name']} "
                f"(ID={ctx['selected_doctor']['id']}, {ctx['selected_doctor']['specialization']}). Skipping."
            )
            print(f"   🚫 [Guard2] {msg}")
            tool_messages.append(ToolMessage(content=msg, tool_call_id=tool_call_id, name=tool_name))
            continue

        # ── GUARD 3: create_booking requires confirmed YES ────────
        if tool_name == "create_booking":
            last_human = _last_human_text(messages)
            is_yes     = bool(_YES_RE.search(last_human))
            step       = ctx.get("step", "")

            print(f"   🔐 [BookingGate] step='{step}'  last_human='{last_human[:80]}'  is_yes={is_yes}")

            if step != "await_confirmation":
                block_msg = (
                    f"Booking blocked — step is '{step}', must be 'await_confirmation'. "
                    f"Show the appointment summary and ask the patient to confirm first."
                )
                print(f"   🚫 [BookingGate] BLOCKED — wrong step")
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

            print(f"   ✅ [BookingGate] ALLOWED — patient said YES")

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

        # ── Try slot extraction after every tool result ───────────
        # Needed because the user may have already said "10:30 AM" in their
        # previous message — we can only set the slot once we have a date
        # from a tool result (find_provider_availability returns the date).
        if ctx["step"] == "collect_slot" and not ctx["pending_slot"]["time"]:
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
# PENDING SLOT EXTRACTOR
# ═══════════════════════════════════════════════════════════════════

def _try_extract_pending_slot_inline(messages: list, ctx: dict) -> None:
    """
    Core slot extraction — takes messages directly.
    Called from both tool_executor (after tool result arrives) and supervisor_node.
    Extracts time from last human message + date from most recent message with a date.
    """
    last_human = _last_human_text(messages)
    time_m     = _TIME_RE.search(last_human)
    if not time_m:
        # Also try parsing written time like "10 30" or "10:30 AM"
        written = re.search(r"\b(\d{1,2})\s+(\d{2})\b", last_human)
        if written:
            time_m = re.search(r"\b(\d{1,2}:\d{2})\b",
                               f"{written.group(1)}:{written.group(2)}")
    if not time_m:
        return

    candidate_date = None
    for m in reversed(messages):
        date_m = _DATE_RE.search(str(m.content))
        if date_m:
            candidate_date = date_m.group(1)
            break

    if candidate_date:
        ctx["pending_slot"]["time"] = time_m.group(1)
        ctx["pending_slot"]["date"] = candidate_date
        _advance_step(ctx)
        print(f"📅 [SlotExtract] slot={candidate_date} at {time_m.group(1)} → step={ctx['step']}")


def _try_extract_pending_slot(state: dict, ctx: dict, session_id: str) -> None:
    """Supervisor-side wrapper — saves to disk if slot was extracted."""
    if ctx["step"] != "collect_slot" or ctx["pending_slot"]["time"]:
        return
    messages = list(state.get("messages", []))
    _try_extract_pending_slot_inline(messages, ctx)
    if ctx["pending_slot"]["time"]:
        save_booking_context(session_id, ctx)


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
{"IMPORTANT: The patient speaks Urdu. You MUST reply in simple everyday Urdu (nastaliq script). NOT Roman Urdu, NOT English. Natural spoken Urdu only." if ctx.get("patient_language") in ("ur", "urdu") else "Reply in plain English."}

{booking_directive}

ABSOLUTE PROHIBITIONS:
  Never call a tool with null/unknown values.
  Never call lookup_customer_profile if patient.id is set above.
  Never call get_doctor_schedule — use find_provider_availability.
  Never call create_booking — code handles it after YES.
  One tool per turn, then reply to the user.
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
        # Suppress supervisor's patient-facing text so triage_node owns the conversation
        # Keep a silent AIMessage so LangGraph state is valid
        response = AIMessage(content="")

    if "[END_CALL]" in response_text:
        interaction_completed = True
        ctx["triage_completed"] = True
        save_booking_context(session_id, ctx)
        print("🏁 [Supervisor] interaction_completed = True")

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
        # Supervisor just emitted [START_TRIAGE] — hand off to triage_node
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