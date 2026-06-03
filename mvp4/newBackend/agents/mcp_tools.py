from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage

from config import supabase
from agents.llm_config import get_llm


try:
    PKT = ZoneInfo("Asia/Karachi")
except ZoneInfoNotFoundError:  # pragma: no cover - Windows installs may miss tzdata
    PKT = timezone(timedelta(hours=5))


WORD_RE = re.compile(r"[a-z0-9]+")
NUMBERED_ITEM_RE = re.compile(r"^\d+\.\s*(.+)$")
SPECIALIZATION_ALIASES: dict[str, list[list[str]]] = {
    "Gastroenterologist / General Physician": [
        ["stomach", "pain"],
        ["stomach", "ache"],
        ["stomach", "hurts"],
        ["abdominal", "pain"],
        ["abdominal", "cramps"],
        ["belly", "pain"],
        ["belly", "hurts"],
        ["nausea"],
        ["bloating"],
        ["vomiting"],
        ["diarrhea"],
    ],
    "Cardiologist": [
        ["chest", "pain"],
        ["chest", "hurts"],
        ["shortness", "breath"],
        ["heart", "palpitations"],
        ["irregular", "heartbeat"],
    ],
    "ENT or General Physician": [
        ["cough"],
        ["cold"],
        ["throat", "pain"],
        ["ear", "pain"],
        ["hearing", "loss"],
    ],
    "Dermatologist": [
        ["skin", "rash"],
        ["rash"],
        ["itching"],
        ["itch"],
    ],
    "Neurologist": [
        ["headache"],
        ["dizziness"],
        ["head", "pain"],
        ["head", "hurts"],
    ],
    "Orthopedic": [
        ["back", "pain"],
        ["back", "hurts"],
        ["joint", "pain"],
        ["joint", "hurts"],
    ],
    "Urologist": [
        ["urinary", "burning"],
        ["frequent", "urination"],
    ],
    "Endocrinologist": [["diabetes"]],
    "Gynecologist": [["pregnancy"]],
    "Psychiatrist / Psychologist": [
        ["mental", "stress"],
        ["anxiety"],
        ["depression"],
    ],
    "Ophthalmologist": [
        ["eye", "redness"],
        ["blurry", "vision"],
    ],
}
WEEKDAY_ALIASES = {
    "mon": "monday",
    "monday": "monday",
    "tue": "tuesday",
    "tues": "tuesday",
    "tuesday": "tuesday",
    "wed": "wednesday",
    "wednesday": "wednesday",
    "thu": "thursday",
    "thur": "thursday",
    "thurs": "thursday",
    "thursday": "thursday",
    "fri": "friday",
    "friday": "friday",
    "sat": "saturday",
    "saturday": "saturday",
    "sun": "sunday",
    "sunday": "sunday",
}
WEEKDAY_ORDER = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
KNOWN_SUPABASE_TABLES = [
    "doctors",
    "patients",
    "doctor_availability",
    "slots",
    "appointments",
    "appointment_events",
    "patient_history",
    "human_waitlist",            # human agent queue
]
SUPPORTED_SPECIALIZATIONS_FALLBACK = [
    "Cardiologist",
    "Dermatologist",
    "Neurologist",
    "Pediatrician",
    "Orthopedic",
    "Gynecologist",
]
APPROVED_TRIAGE_FLOW_BY_KEY: dict[str, str] = {}
GENERAL_TRIAGE_FLOW = "general_intake.md"


# =====================================================================
# DATACLASSES
# =====================================================================

@dataclass(slots=True)
class ToolCallResult:
    ok: bool
    data: dict[str, Any]
    error: str | None = None


@dataclass(slots=True)
class SymptomEvidence:
    specialization: str
    matched_phrases: list[str]
    score: float


# =====================================================================
# PURE STATIC HELPERS (extracted from class for reuse in tools)
# =====================================================================

def _parse_date(raw_value: str | None) -> datetime:
    now = datetime.now(PKT)
    print(now)
    if not raw_value:
        return now
    lowered = raw_value.strip().lower()
    if lowered == "today":
        return now
    if lowered == "tomorrow":
        print(now + timedelta(days=1))
        return now + timedelta(days=1)
    if lowered == "day after tomorrow":
        return now + timedelta(days=2)
    if lowered in WEEKDAY_ALIASES:
        target_day = WEEKDAY_ALIASES[lowered]
        delta = (WEEKDAY_ORDER.index(target_day) - now.weekday()) % 7
        return now + timedelta(days=delta)
    for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y"):
        try:
            parsed = datetime.strptime(raw_value, fmt)
            return parsed.replace(tzinfo=PKT)
        except ValueError:
            continue
    raise ValueError("Date must be today, tomorrow, a weekday, or look like 2026-04-06.")


def _parse_time(raw_value: str):
    cleaned = raw_value.strip().lower()
    for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"):
        try:
            return datetime.strptime(cleaned.upper(), fmt).time()
        except ValueError:
            continue
    raise ValueError("Time must look like 14:30 or 2:30 PM.")


def _normalize_text(text: str) -> str:
    return " ".join(WORD_RE.findall(text.lower()))


def _extract_words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _normalize_phrase_tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def _phrase_match_score(tokens: list[str], normalized_text: str, token_set: set[str]) -> float:
    if not tokens:
        return 0.0
    phrase = " ".join(tokens)
    if phrase and phrase in normalized_text:
        return 0.95 if len(tokens) > 1 else 0.8
    overlap = len(set(tokens) & token_set)
    if overlap == len(tokens):
        return 0.82 if len(tokens) > 1 else 0.7
    if len(tokens) > 1 and overlap >= len(tokens) - 1:
        return 0.45
    return 0.0


def _split_specialization_options(specialization: str) -> list[str]:
    options = re.split(r"/|\bor\b", specialization, flags=re.IGNORECASE)
    return [option.strip() for option in options if option.strip()]


def _specialization_matches(provider_specialization: str, requested_option: str) -> bool:
    provider_tokens = WORD_RE.findall(provider_specialization.lower())
    requested_tokens = WORD_RE.findall(requested_option.lower())
    if not provider_tokens or not requested_tokens:
        return False
    provider_text = " ".join(provider_tokens)
    if len(requested_tokens) == 1:
        return requested_tokens[0] in set(provider_tokens)
    return " ".join(requested_tokens) in provider_text


def _specialization_is_supported(specialization: str, supported_specializations: list[str]) -> bool:
    if not specialization:
        return True
    supported = {item.lower() for item in supported_specializations}
    for option in _split_specialization_options(specialization):
        if option.lower() in supported:
            return True
    return False


def _normalize_doctor_record(provider: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(provider)
    if "name" in provider and "Name" not in normalized:
        normalized["Name"] = provider["name"]
    if "Name" in provider and "name" not in normalized:
        normalized["name"] = provider["Name"]
    if "specialization" in provider and "Specialization" not in normalized:
        normalized["Specialization"] = provider["specialization"]
    if "Specialization" in provider and "specialization" not in normalized:
        normalized["specialization"] = provider["Specialization"]
    if "experience_years" in provider:
        normalized["Experience"] = f"{provider['experience_years']} years"
        normalized["experience_years"] = provider["experience_years"]
    elif "Experience" in provider and "experience_years" not in normalized:
        try:
            normalized["experience_years"] = int("".join(WORD_RE.findall(str(provider["Experience"]))))
        except Exception:
            normalized["experience_years"] = None
    if "consultation_fee" in provider and "consultation_fee" not in normalized:
        normalized["consultation_fee"] = provider["consultation_fee"]
    return normalized


def _pick_best_name_match(query: str, doctors: list[dict[str, Any]]) -> dict[str, Any]:
    normalized_query = set(WORD_RE.findall(query.lower()))
    best_doctor = doctors[0]
    best_score = -1
    for doctor in doctors:
        name_tokens = set(WORD_RE.findall(str(doctor.get("Name", "")).lower()))
        score = len(normalized_query & name_tokens)
        if score > best_score:
            best_score = score
            best_doctor = doctor
    return best_doctor


def _day_of_week_from_date(value: datetime) -> int:
    # In the schema, 0=Sunday
    return (value.weekday() + 1) % 7


def _day_of_week_name(value: int | None) -> str:
    if value is None:
        return ""
    try:
        names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        return names[int(value) % 7]
    except Exception:
        return str(value)


def _format_db_time(raw_value: str) -> str:
    if not raw_value:
        return ""
    try:
        return datetime.strptime(raw_value, "%H:%M:%S").strftime("%H:%M")
    except ValueError:
        return raw_value


def _slot_starts_at(slot: dict[str, Any], target_time: datetime.time) -> bool:
    slot_start = datetime.fromisoformat(str(slot["start_time"])).timetz().replace(tzinfo=None)
    return slot_start == target_time


def _find_nearest_slot(
    slots: list[dict[str, Any]],
    target_time: datetime.time,
    tolerance_minutes: int = 30,
) -> dict[str, Any] | None:
    """
    Return the available slot whose start time is closest to target_time,
    within tolerance_minutes. Prefers slots at-or-after target_time.
    Returns None if no slot is within tolerance.
    """
    target_delta = timedelta(hours=target_time.hour, minutes=target_time.minute)
    best_slot  = None
    best_diff  = timedelta(minutes=tolerance_minutes + 1)

    for slot in slots:
        slot_start = datetime.fromisoformat(str(slot["start_time"])).timetz().replace(tzinfo=None)
        slot_delta = timedelta(hours=slot_start.hour, minutes=slot_start.minute)
        diff = abs(slot_delta - target_delta)
        if diff < best_diff:
            best_diff = diff
            best_slot = slot

    return best_slot


def _format_slot_start(slot: dict[str, Any]) -> str:
    return datetime.fromisoformat(str(slot["start_time"])).strftime("%H:%M")


def _is_blocking_slot(slot: dict[str, Any], active_appointment_slot_ids: set[int]) -> bool:
    status = str(slot.get("status", "")).strip().lower()
    if slot.get("id") in active_appointment_slot_ids:
        return True
    return status in {"booked", "confirmed", "held"}


def _slot_overlaps(
    slot_start: datetime,
    slot_end: datetime,
    blocked_start: datetime,
    blocked_end: datetime,
) -> bool:
    return slot_start < blocked_end and blocked_start < slot_end


def _contains_term(normalized_text: str, token_set: set[str], term: str) -> bool:
    term_tokens = WORD_RE.findall(term.lower())
    if not term_tokens:
        return False
    if len(term_tokens) == 1:
        return term_tokens[0] in token_set
    return " ".join(term_tokens) in normalized_text


def _apply_query_filter(query: Any, column: str, op: str, value: Any) -> Any:
    if op == "eq":
        return query.eq(column, value)
    if op == "neq":
        return query.neq(column, value)
    if op == "ilike":
        return query.ilike(column, value)
    if op == "like":
        return query.like(column, value)
    if op == "gte":
        return query.gte(column, value)
    if op == "lte":
        return query.lte(column, value)
    if op == "gt":
        return query.gt(column, value)
    if op == "lt":
        return query.lt(column, value)
    if op == "in":
        values = value if isinstance(value, list) else [value]
        return query.in_(column, values)
    raise ValueError(f"Unsupported filter op: {op}")


def _get_schedule_rows(doctor_id: int, weekday: int) -> list[dict[str, Any]]:
    if not supabase:
        return []
    print(f"   [ScheduleRows] Querying doctor_availability: doctor_id={doctor_id}, day_of_week={weekday}")
    response = (
        supabase.table("doctor_availability")
        .select("*")
        .eq("doctor_id", doctor_id)
        .eq("day_of_week", weekday)
        .execute()
    )
    rows = sorted(
        list(response.data or []),
        key=lambda row: (
            str(row.get("start_time", "")),
            str(row.get("end_time", "")),
        ),
    )
    print(f"   [ScheduleRows] Found {len(rows)} row(s): {[{'start': r.get('start_time'), 'end': r.get('end_time'), 'duration': r.get('slot_duration_minutes')} for r in rows]}")
    return rows


def _build_bookable_slots(
    *,
    doctor_id: int,
    target_date: datetime,
    schedule_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not supabase:
        return []

    day_start = datetime.combine(target_date.date(), datetime.min.time()).replace(tzinfo=PKT)
    day_end   = day_start + timedelta(days=1)

    print(f"   [BuildSlots] doctor_id={doctor_id}  date={target_date.date()}  window={day_start.isoformat()} → {day_end.isoformat()}")

    slot_response = (
        supabase.table("slots")
        .select("*")
        .eq("doctor_id", doctor_id)
        .gte("start_time", day_start.isoformat())
        .lt("start_time", day_end.isoformat())
        .execute()
    )
    print(f"   [BuildSlots] Existing slots in DB for this day: {len(slot_response.data or [])}")
    for s in (slot_response.data or []):
        print(f"      slot id={s.get('id')} start={s.get('start_time')} status={s.get('status')}")

    appointment_response = (
        supabase.table("appointments")
        .select("slot_id, status")
        .eq("doctor_id", doctor_id)
        .execute()
    )
    active_appointment_slot_ids = {
        int(item["slot_id"])
        for item in (appointment_response.data or [])
        if item.get("slot_id") is not None
        and str(item.get("status", "")).strip().lower() not in {"cancelled", "canceled", "no_show"}
    }
    print(f"   [BuildSlots] Active appointment slot_ids (blocking): {active_appointment_slot_ids}")

    blocked_windows = []
    for slot in slot_response.data or []:
        if not _is_blocking_slot(slot, active_appointment_slot_ids):
            continue
        blocked_windows.append(
            (
                datetime.fromisoformat(str(slot["start_time"])),
                datetime.fromisoformat(str(slot["end_time"])),
            )
        )
    print(f"   [BuildSlots] Blocked windows: {[(str(s), str(e)) for s, e in blocked_windows]}")

    bookable_slots: list[dict[str, Any]] = []
    for schedule_row in schedule_rows:
        duration_minutes = int(schedule_row.get("slot_duration_minutes") or 15)
        start_time       = datetime.strptime(str(schedule_row["start_time"]), "%H:%M:%S").time()
        end_time         = datetime.strptime(str(schedule_row["end_time"]),   "%H:%M:%S").time()
        slot_start       = datetime.combine(target_date.date(), start_time).replace(tzinfo=PKT)
        schedule_end     = datetime.combine(target_date.date(), end_time).replace(tzinfo=PKT)

        print(f"   [BuildSlots] Schedule row: {start_time} → {end_time}  duration={duration_minutes}min")

        while slot_start + timedelta(minutes=duration_minutes) <= schedule_end:
            slot_end = slot_start + timedelta(minutes=duration_minutes)
            if any(
                _slot_overlaps(slot_start, slot_end, blocked_start, blocked_end)
                for blocked_start, blocked_end in blocked_windows
            ):
                slot_start = slot_end
                continue

            bookable_slots.append(
                {
                    "doctor_id":             doctor_id,
                    "start_time":            slot_start.isoformat(),
                    "end_time":              slot_end.isoformat(),
                    "slot_duration_minutes": duration_minutes,
                    "source_schedule_id":    schedule_row.get("id"),
                    "status":                "available",
                }
            )
            slot_start = slot_end

    print(f"   [BuildSlots] → {len(bookable_slots)} bookable slot(s) remaining after blocking")
    return bookable_slots


def _split_mapping_line(line: str) -> tuple[str, str] | None:
    for separator in ("→", "->"):
        if separator in line:
            symptom_text, specialization = [part.strip() for part in line.split(separator, 1)]
            return symptom_text, specialization
    return None





def _get_supported_specializations() -> list[str]:
    if not supabase:
        return list(SUPPORTED_SPECIALIZATIONS_FALLBACK)
    try:
        response = supabase.table("doctors").select("specialization").execute()
    except Exception:
        return list(SUPPORTED_SPECIALIZATIONS_FALLBACK)
    raw_values = [str(row.get("specialization", "")).strip() for row in (response.data or [])]
    discovered = {value for value in raw_values if value}
    ordered = [item for item in SUPPORTED_SPECIALIZATIONS_FALLBACK if item in discovered]
    extras = sorted(discovered - set(ordered))
    return ordered + extras if ordered or extras else list(SUPPORTED_SPECIALIZATIONS_FALLBACK)


def _lookup_providers_by_specialization(specialization: str) -> list[dict[str, Any]]:
    if not supabase:
        return []
    providers_by_id: dict[Any, dict[str, Any]] = {}
    for option in _split_specialization_options(specialization):
        response = (
            supabase.table("doctors")
            .select("*")
            .ilike("specialization", f"%{option}%")
            .execute()
        )
        for provider in response.data or []:
            provider_specialization = str(provider.get("specialization", ""))
            if not _specialization_matches(provider_specialization, option):
                continue
            providers_by_id[provider.get("id")] = provider
    return [_normalize_doctor_record(provider) for provider in providers_by_id.values()]


def _pick_question_flow_path(symptom: str, specialization: str, question_flow_dir: Path) -> Path:
    normalized = _normalize_text(f"{symptom} {specialization}")
    token_set = set(_extract_words(f"{symptom} {specialization}"))
    approved_key = None
    for key in APPROVED_TRIAGE_FLOW_BY_KEY:
        if _contains_term(normalized, token_set, key):
            approved_key = key
            break
    if approved_key:
        return question_flow_dir / APPROVED_TRIAGE_FLOW_BY_KEY[approved_key]
    return question_flow_dir / GENERAL_TRIAGE_FLOW


def _parse_question_flow(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {
            "questions": ["Can you briefly describe the main issue for the doctor?"],
            "red_flags": [],
        }
    questions: list[str] = []
    red_flags: list[str] = []
    section = "questions"
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        cleaned = raw_line.strip()
        if not cleaned:
            continue
        if cleaned.lower().startswith("red flags"):
            section = "red_flags"
            continue
        if cleaned.lower().startswith("rules"):
            section = "rules"
            continue
        question_match = NUMBERED_ITEM_RE.match(cleaned)
        if question_match:
            questions.append(question_match.group(1).strip())
            continue
        if section == "red_flags" and cleaned.startswith("-"):
            red_flags.append(cleaned.lstrip("- ").strip())
    return {
        "questions": questions or ["Can you briefly describe the main issue for the doctor?"],
        "red_flags": red_flags,
    }


# =====================================================================
# LANGCHAIN TOOLS — @tool decorated, all using the helpers above
# =====================================================================

# Make sure to import this at the top of mcp_tools.py if it's not there!
from langchain_community.chat_models import ChatOllama

@tool
def recommend_specialist_tool(symptom: str) -> str:
    """
    Use this tool whenever a patient mentions a symptom but hasn't chosen a doctor.
    Takes the patient's exact symptom or complaint (in any language/phrasing) and
    returns the recommended specialist department from the hospital's actual DB.
    """
    print(f"🛠️ [Tool] recommend_specialist_tool: symptom='{symptom}'")

    # Fetch real specializations from DB to constraint the LLM
    supported_specializations = _get_supported_specializations()

    try:
        # 🧠 USE LOCAL MEDGEMMA INSTEAD OF THE MAIN API
        llm = ChatOllama(model="medgemma:4b", temperature=0.0)
        
        sys_prompt = SystemMessage(content=f"""
        You are a medical routing assistant for a hospital.
        Map the patient's symptom to the correct department.

        AVAILABLE DEPARTMENTS: {', '.join(supported_specializations)}

        RULES:
        1. Reply with ONLY the exact department name from the list above.
        2. Do not explain your reasoning or add any other text.
        3. If completely unsure, output "General Physician".
        """)
        
        user_prompt = HumanMessage(content=f"Patient symptom: {symptom}")
        specialist = llm.invoke([sys_prompt, user_prompt]).content.strip()
        
        print(f"   ↳ MedGemma matched '{symptom}' to: {specialist}")
        return f"Recommended specialist: {specialist}"
        
    except Exception as e:
        print(f"   ↳ MedGemma error: {e}. Falling back to: General Physician")
        return "Recommended specialist: General Physician"

@tool
def search_knowledge(query: str) -> str:
    """
    Search FAQ and policy content for customer support answers.
    Use for questions about hospital policies, pricing, general info.
    """
    print(f"🛠️ [Tool] search_knowledge: query='{query}'")
    # If a KnowledgeBase instance is available in your context, wire it in.
    # For now returns a graceful message — swap with knowledge_base.search(query).
    return f"Knowledge search for '{query}': No knowledge base connected. Please answer from context."


@tool
def list_database_tables() -> str:
    """List the Supabase tables this assistant can inspect."""
    print("🛠️ [Tool] list_database_tables")
    return f"Available tables: {', '.join(KNOWN_SUPABASE_TABLES)} (read-only via query_database_table)"


@tool
def query_database_table(
    table_name: str,
    columns: str = "*",
    limit: int = 20,
    filters: list[dict[str, Any]] | None = None,
    order_by: str | None = None,
    ascending: bool = True,
) -> str:
    """
    Read rows from a Supabase table with optional filters.
    Use for ad-hoc data lookups. Supports filters with ops: eq, neq, ilike, gte, lte, gt, lt, in.
    Example filters: [{"column": "specialization", "op": "ilike", "value": "%cardio%"}]
    """
    print(f"🛠️ [Tool] query_database_table: table={table_name}")
    if not supabase:
        return "Database not connected."

    table_name = str(table_name).strip()
    if not table_name:
        return "Error: table_name is required."

    safe_limit = max(1, min(int(limit), 100))
    query = supabase.table(table_name).select(columns or "*")

    for filter_item in filters or []:
        column = str(filter_item.get("column") or "").strip()
        op = str(filter_item.get("op") or "eq").strip().lower()
        value = filter_item.get("value")
        if not column:
            continue
        try:
            query = _apply_query_filter(query, column, op, value)
        except ValueError as e:
            return f"Filter error: {e}"

    if order_by:
        query = query.order(str(order_by).strip(), desc=not ascending)

    try:
        response = query.limit(safe_limit).execute()
        rows = response.data or []
        if not rows:
            return f"No rows found in '{table_name}' matching those filters."
        return f"Found {len(rows)} row(s) in '{table_name}':\n{rows}"
    except Exception as e:
        return f"Database error: {e}"


@tool
def lookup_customer_profile(phone: str) -> str:
    """
    Look up a patient profile by phone number.
    Returns the patient record if found, or indicates no record exists.
    """
    print(f"🛠️ [Tool] lookup_customer_profile: phone='{phone}'")
    if not supabase:
        return "Database not connected."
    response = supabase.table("patients").select("*").eq("phone", phone).limit(1).execute()
    if response.data:
        return f"Patient found: {response.data[0]}"
    return f"No patient profile found for phone: {phone}"


@tool
def register_customer_profile(name: str, phone: str, gender: str | None = None, age: float | None = None) -> str:
    """
    Create a new patient profile with name and phone number.
    Optionally include gender and age.
    """
    print(f"🛠️ [Tool] register_customer_profile: name='{name}', phone='{phone}'")
    if not supabase:
        return "Database not connected."
    payload = {"name": name, "phone": phone}
    if gender:
        payload["gender"] = gender
    if age is not None:
        payload["age"] = age
    try:
        response = supabase.table("patients").insert(payload).execute()
        profile = response.data[0] if response.data else None
        if profile:
            return f"Patient registered successfully. ID: {profile.get('id')}, Name: {profile.get('name')}"
        return "Failed to register patient."
    except Exception as e:
        return f"Error registering patient: {e}"




@tool
def get_doctor_profile(doctor_name: str | None = None, doctor_id: int | None = None) -> str:
    """
    Fetch a doctor's full profile together with their weekly schedule.
    Provide either doctor_name or doctor_id.
    """
    print(f"🛠️ [Tool] get_doctor_profile: name='{doctor_name}', id={doctor_id}")
    if not supabase:
        return "Database not connected."

    doctor: dict[str, Any] | None = None
    if doctor_id is not None:
        response = supabase.table("doctors").select("*").eq("id", doctor_id).limit(1).execute()
        doctor = response.data[0] if response.data else None
    elif doctor_name:
        response = supabase.table("doctors").select("*").ilike("name", f"%{doctor_name}%").execute()
        if response.data:
            doctor = _pick_best_name_match(doctor_name, response.data)

    if not doctor:
        return f"No doctor found matching '{doctor_name or doctor_id}'."

    doctor = _normalize_doctor_record(doctor)
    slots_response = (
        supabase.table("doctor_availability")
        .select("*")
        .eq("doctor_id", doctor["id"])
        .execute()
    )
    schedule = [
        {
            "day": _day_of_week_name(slot.get("day_of_week")),
            "start": _format_db_time(slot.get("start_time", "")),
            "end": _format_db_time(slot.get("end_time", "")),
        }
        for slot in (slots_response.data or [])
    ]

    lines = [
        f"Dr. {doctor.get('name')} (ID: {doctor.get('id')}) | {doctor.get('specialization')}",
        f"Fee: {doctor.get('consultation_fee', 'N/A')} | Experience: {doctor.get('experience_years', 'N/A')} yrs",
        "Schedule:",
    ]
    for s in schedule:
        lines.append(f"  {s['day']}: {s['start']} – {s['end']}")
    return "\n".join(lines) if schedule else "\n".join(lines) + "\n  No schedule found."


@tool
def find_provider_availability(
    specialization: str | None = None,
    doctor_name: str | None = None,
    doctor_id: int | None = None,
    date: str | None = None,
    time: str | None = None,
) -> str:
    """
    Find available appointment slots for a doctor or specialization on a given date.
    Date accepts: today, tomorrow, day after tomorrow, weekday names (monday etc), or YYYY-MM-DD.
    Time (optional) filters to a specific start time, e.g. '14:30' or '2:30 PM'.
    Identify the doctor via doctor_id (preferred when already selected), doctor_name, or specialization.
    Checks actual booked/held slots to show only truly free times.
    """
    print(f"🛠️ [Tool] find_provider_availability: spec={specialization!r}, doctor={doctor_name!r}, doctor_id={doctor_id}, date={date!r}, time={time!r}")
    if not supabase:
        return "Database not connected."

    # Resolve providers — doctor_id takes priority (most precise), then name, then specialization
    if doctor_id is not None:
        print(f"   [Availability] Looking up by doctor_id={doctor_id}")
        response = supabase.table("doctors").select("*").eq("id", doctor_id).limit(1).execute()
        providers = [_normalize_doctor_record(p) for p in (response.data or [])]
        if not providers:
            return f"No doctor found with ID {doctor_id}."
    elif doctor_name:
        # Normalize: strip "Dr." prefix so "Dr. Afaf Irfan" matches DB value "Dr.Afaf Irfan"
        search_name = re.sub(r"^[Dd][Rr]\.?\s*", "", doctor_name).strip()
        print(f"   [Availability] Looking up by name: original='{doctor_name}' → search='{search_name}'")
        response = supabase.table("doctors").select("*").ilike("name", f"%{search_name}%").execute()
        providers = [_normalize_doctor_record(p) for p in (response.data or [])]
        print(f"   [Availability] Name search returned {len(providers)} provider(s): {[p.get('name') for p in providers]}")
    elif specialization:
        print(f"   [Availability] Looking up by specialization='{specialization}'")
        providers = _lookup_providers_by_specialization(specialization)
        print(f"   [Availability] Specialization search returned {len(providers)} provider(s)")
    else:
        return "Provide at least a doctor_id, doctor_name, or specialization."

    if not providers:
        return "No providers found matching that criteria."

    try:
        target_date = _parse_date(date)
    except ValueError as e:
        return f"Invalid date: {e}"
    target_weekday = _day_of_week_from_date(target_date)
    target_time    = _parse_time(time) if time else None
    available      = []

    print(f"   [Availability] Target date: {target_date.strftime('%A %Y-%m-%d')}  weekday_index={target_weekday}  time_filter={target_time}")

    for provider in providers:
        p_id   = provider["id"]
        p_name = provider.get("name", "Unknown")
        schedule_rows = _get_schedule_rows(p_id, target_weekday)
        print(f"   [Availability] Dr.{p_name} (ID={p_id}): schedule_rows={len(schedule_rows)}")
        if not schedule_rows:
            print(f"   [Availability] ⚠️  No schedule rows for weekday_index={target_weekday} — doctor may not work this day")
            continue

        all_slots = _build_bookable_slots(
            doctor_id=p_id,
            target_date=target_date,
            schedule_rows=schedule_rows,
        )
        print(f"   [Availability] Dr.{p_name}: {len(all_slots)} free slot(s) built")
        if all_slots:
            print(f"   [Availability] First few slots: {[_format_slot_start(s) for s in all_slots[:4]]}")

        if target_time is not None:
            # Try exact match first
            exact = [s for s in all_slots if _slot_starts_at(s, target_time)]
            if exact:
                print(f"   [Availability] Exact time match found for {target_time}")
                matching_slots = exact
                time_note = None
            else:
                # Fall back to nearest available slot within 30 minutes
                nearest = _find_nearest_slot(all_slots, target_time, tolerance_minutes=30)
                if nearest:
                    nearest_str = _format_slot_start(nearest)
                    print(f"   [Availability] No exact match for {target_time}; nearest slot={nearest_str}")
                    matching_slots = [nearest]
                    time_note = f"(closest to {time} — exact time not available)"
                else:
                    print(f"   [Availability] No slot within 30 min of {target_time}")
                    matching_slots = []
                    time_note = None
        else:
            matching_slots = all_slots
            time_note = None

        if matching_slots:
            display_slots = [_format_slot_start(s) for s in matching_slots[:6]]
            extra_count   = len(all_slots) - len(display_slots) if not target_time else 0
            available.append((provider, display_slots, len(all_slots), time_note, extra_count))

    if not available:
        day_name  = target_date.strftime("%A")
        date_str  = target_date.strftime("%Y-%m-%d")
        if target_time:
            # Show all slots even though requested time has no match
            # Re-run without time filter for a helpful "here's what IS available" message
            fallback = []
            for provider in providers:
                rows = _get_schedule_rows(provider["id"], target_weekday)
                if not rows:
                    continue
                slots = _build_bookable_slots(doctor_id=provider["id"], target_date=target_date, schedule_rows=rows)
                if slots:
                    fallback.append((provider, [_format_slot_start(s) for s in slots[:6]], len(slots)))
            if fallback:
                lines = [f"No slot at {time} on {day_name} ({date_str}). Available slots:"]
                for provider, display_slots, total in fallback:
                    extra = f" (+{total - len(display_slots)} more)" if total > len(display_slots) else ""
                    lines.append(f"  Dr. {provider.get('name')} ({provider.get('specialization')}): {', '.join(display_slots)}{extra}")
                return "\n".join(lines)
        return f"No available slots found on {day_name} ({date_str})."

    lines = [f"Available slots on {target_date.strftime('%A, %Y-%m-%d')}:"]
    for provider, display_slots, total, time_note, extra_count in available:
        note_str  = f" {time_note}" if time_note else ""
        extra_str = f" (+{extra_count} more)" if extra_count > 0 else ""
        lines.append(
            f"  Dr. {provider.get('name')} ({provider.get('specialization')}): "
            f"{', '.join(display_slots)}{extra_str}{note_str}"
        )
    return "\n".join(lines)


@tool
def get_doctor_schedule(doctor_id: int | None = None, doctor_name: str | None = None) -> str:
    """
    Get the weekly consultation schedule for a specific doctor.
    Provide either doctor_id or doctor_name.
    """
    print(f"🛠️ [Tool] get_doctor_schedule: id={doctor_id}, name='{doctor_name}'")
    if not supabase:
        return "Database not connected."

    doctor: dict[str, Any] | None = None
    if doctor_id is not None:
        response = supabase.table("doctors").select("*").eq("id", doctor_id).limit(1).execute()
        doctor = response.data[0] if response.data else None
    elif doctor_name:
        response = supabase.table("doctors").select("*").ilike("name", f"%{doctor_name}%").execute()
        if response.data:
            doctor = _pick_best_name_match(doctor_name, response.data)

    if not doctor:
        return f"No doctor found for '{doctor_name or doctor_id}'."

    doctor = _normalize_doctor_record(doctor)
    slots_response = (
        supabase.table("doctor_availability")
        .select("*")
        .eq("doctor_id", doctor["id"])
        .execute()
    )
    schedule = [
        {
            "day": _day_of_week_name(slot.get("day_of_week")),
            "start": _format_db_time(slot.get("start_time", "")),
            "end": _format_db_time(slot.get("end_time", "")),
        }
        for slot in (slots_response.data or [])
    ]
    if not schedule:
        return f"Dr. {doctor.get('name')} has no schedule entries."

    lines = [f"Schedule for Dr. {doctor.get('name')} ({doctor.get('specialization')}):"]
    for s in schedule:
        lines.append(f"  {s['day']}: {s['start']} – {s['end']}")
    return "\n".join(lines)



@tool
def create_booking(patient_id: int, doctor_id: int, date: str, time: str) -> str:
    """
    Create a confirmed appointment for an existing patient with an existing doctor.
    Requires: patient_id (int), doctor_id (int), date (e.g. '2026-04-20' or 'tomorrow'), time (e.g. '14:30').
    Validates the slot is actually free before booking. Creates both slot and appointment records.
    Also logs an appointment_events entry for audit trail.
    """
    print(f"🛠️ [Tool] create_booking: patient_id={patient_id}, doctor_id={doctor_id}, date={date}, time={time}")
    if not supabase:
        return "Database not connected."

    try:
        target_date = _parse_date(date)
        target_time = _parse_time(time)
    except ValueError as exc:
        return f"Invalid date/time: {exc}"

    weekday = _day_of_week_from_date(target_date)
    schedule_rows = _get_schedule_rows(doctor_id, weekday)
    if not schedule_rows:
        return f"No schedule for doctor {doctor_id} on {target_date.strftime('%A')}."

    bookable_slots = _build_bookable_slots(
        doctor_id=doctor_id,
        target_date=target_date,
        schedule_rows=schedule_rows,
    )
    chosen_slot = next(
        (slot for slot in bookable_slots if _slot_starts_at(slot, target_time)),
        None,
    )
    if not chosen_slot:
        return f"The {time} slot on {date} is no longer available. Please choose another time."

    # Create slot record
    try:
        created_slot = supabase.table("slots").insert(
            {
                "doctor_id": doctor_id,
                "start_time": chosen_slot["start_time"],
                "end_time": chosen_slot["end_time"],
                "status": "booked",
            }
        ).execute()
        slot_record = created_slot.data[0] if created_slot.data else None
    except Exception as e:
        return f"Failed to reserve slot: {e}"

    if not slot_record:
        return "Slot could not be reserved."

    # Create appointment record
    try:
        appointment_payload = {
            "slot_id": slot_record["id"],
            "doctor_id": doctor_id,
            "patient_id": patient_id,
            "status": "booked",
        }
        response = supabase.table("appointments").insert(appointment_payload).execute()
        appointment = response.data[0] if response.data else None
    except Exception as e:
        # Roll back slot
        supabase.table("slots").update({"status": "available"}).eq("id", slot_record["id"]).execute()
        return f"Failed to create appointment: {e}"

    if not appointment:
        supabase.table("slots").update({"status": "available"}).eq("id", slot_record["id"]).execute()
        return "Appointment could not be saved."

    # Audit trail
    try:
        supabase.table("appointment_events").insert(
            {"appointment_id": appointment["id"], "event_type": "created"}
        ).execute()
    except Exception:
        pass  # Non-fatal

    appt_date = target_date.strftime("%A, %Y-%m-%d")
    appt_time = target_time.strftime("%H:%M")
    return (
        f"✅ Appointment confirmed!\n"
        f"  Appointment ID: {appointment['id']}\n"
        f"  Date: {appt_date} at {appt_time}\n"
        f"  Doctor ID: {doctor_id} | Patient ID: {patient_id}"
    )


@tool
def get_recent_case_notes(patient_id: int) -> str:
    """
    Fetch the last 5 appointment notes for a patient.
    Useful for giving the doctor context on returning patients.
    """
    print(f"🛠️ [Tool] get_recent_case_notes: patient_id={patient_id}")
    if not supabase:
        return "Database not connected."
    response = (
        supabase.table("appointments")
        .select("id, notes, doctor_id, created_at")
        .eq("patient_id", patient_id)
        .neq("notes", None)
        .order("created_at", desc=True)
        .limit(5)
        .execute()
    )
    notes = response.data or []
    if not notes:
        return f"No case notes found for patient {patient_id}."
    lines = [f"Last {len(notes)} note(s) for patient {patient_id}:"]
    for n in notes:
        lines.append(f"  [{n.get('created_at', 'N/A')}] Appt#{n['id']} — {n.get('notes', '')}")
    return "\n".join(lines)


@tool
def save_case_notes(appointment_id: int, notes: str) -> str:
    """
    Save intake or triage notes against an existing appointment.
    Call this after the triage flow is complete to persist what the patient described.
    """
    print(f"🛠️ [Tool] save_case_notes: appointment_id={appointment_id}")
    if not supabase:
        return "Database not connected."
    try:
        response = (
            supabase.table("appointments")
            .update({"notes": notes})
            .eq("id", appointment_id)
            .execute()
        )
        if response.data:
            return f"Notes saved for appointment {appointment_id}."
        return f"No appointment found with ID {appointment_id}."
    except Exception as e:
        return f"Error saving notes: {e}"

@tool
def get_doctors_by_specialization(specialization: str) -> str:
    """
    Get a list of doctors based on their medical specialization (e.g., 'Cardiologist', 'Dermatologist').
    Use this when a user asks "Who are your cardiologists?" or "Do you have a pediatrician?".
    """
    print(f"🛠️ [Tool] get_doctors_by_specialization: '{specialization}'")
    if not supabase:
        return "Database not connected."

    # Fix the plural bug (e.g., "Cardiologists" -> "Cardiologist")
    clean_spec = specialization.strip()
    if clean_spec.lower().endswith('s'):
        clean_spec = clean_spec[:-1]

    try:
        response = (
            supabase.table("doctors")
            .select("id, name, specialization, experience_years, consultation_fee")
            .ilike("specialization", f"%{clean_spec}%")
            .execute()
        )
        
        doctors = response.data or []
        if not doctors:
            return f"No doctors found for specialization: {specialization}."

        lines = [f"Doctors specializing in {specialization}:"]
        for d in doctors:
            lines.append(f"  - Dr. {d.get('name')} (ID: {d.get('id')}) | {d.get('experience_years', 'N/A')} yrs exp | Fee: Rs. {d.get('consultation_fee', 'N/A')}")
        
        return "\n".join(lines)
        
    except Exception as e:
        return f"Database error while searching for doctors: {e}"


# =====================================================================
# PATIENT HISTORY TOOLS
# =====================================================================

@tool
def update_patient_demographics(
    patient_id: int,
    age: int | None = None,
    gender: str | None = None,
    marital_status: str | None = None,
) -> str:
    """
    Update age, gender, and/or marital_status on an existing patient record.
    Call this during the history intake phase when collecting demographic info.
    Only updates fields that are explicitly provided.
    """
    print(f"🛠️ [Tool] update_patient_demographics: patient_id={patient_id} age={age} gender={gender} marital={marital_status}")
    if not supabase:
        return "Database not connected."
    payload: dict = {}
    if age is not None:
        payload["age"] = age
    if gender is not None:
        payload["gender"] = gender.strip().lower()
    if marital_status is not None:
        payload["marital_status"] = marital_status.strip().lower()
    if not payload:
        return "No fields provided to update."
    try:
        supabase.table("patients").update(payload).eq("id", patient_id).execute()
        updated = ", ".join(f"{k}={v}" for k, v in payload.items())
        return f"Patient demographics updated successfully for patient_id {patient_id}: {updated}."
    except Exception as e:
        return f"Error updating patient demographics: {e}"

@tool
def get_patient_history(patient_id: int) -> str:

    """
    Check if a patient has medical history on file.
    Returns the full history record if found, or indicates none exists.
    Call this after lookup_customer_profile to decide whether to collect history.
    """
    print(f"🛠️ [Tool] get_patient_history: patient_id={patient_id}")
    if not supabase:
        return "Database not connected."
    try:
        response = (
            supabase.table("patient_history")
            .select("*")
            .eq("patient_id", patient_id)
            .limit(1)
            .execute()
        )
        if response.data:
            h = response.data[0]
            return (
                f"Patient history found:\n"
                f"  Chronic conditions : {h.get('chronic_conditions') or 'None reported'}\n"
                f"  Medications        : {h.get('medications') or 'None reported'}\n"
                f"  Drug allergies     : {h.get('drug_allergies') or 'None reported'}\n"
                f"  Family history     : {h.get('family_history') or 'None reported'}\n"
                f"  Smoking / alcohol  : {h.get('smoking_status') or 'Not recorded'}\n"
                f"  Last updated       : {h.get('last_updated') or 'Unknown'}"
            )
        return f"No medical history on file for patient_id {patient_id}."
    except Exception as e:
        return f"Could not retrieve patient history: {e}"


@tool
def save_patient_history(
    patient_id: int,
    chronic_conditions: str | None = None,
    medications: str | None = None,
    drug_allergies: str | None = None,
    general_allergies: str | None = None,
    family_history: str | None = None,
    smoking_status: str | None = None,
    menstrual_history: str | None = None,
    lmp_date: str | None = None,
    pregnancy_status: str | None = None,
    obstetric_history: str | None = None,
    fall_history: str | None = None,
    vaccination_status: str | None = None,
) -> str:
    """
    Save or update a patient's medical history.
    Upserts on patient_id — safe to call even if a record already exists.
    Call this after collecting nurse-style history questions.
    Provide only the fields you have collected — omit the rest.
    """
    print(f"🛠️ [Tool] save_patient_history: patient_id={patient_id}")
    if not supabase:
        return "Database not connected."

    payload: dict = {
        "patient_id":   patient_id,
        "last_updated": datetime.now(PKT).isoformat(),
    }
    for field, val in [
        ("chronic_conditions", chronic_conditions),
        ("medications",        medications),
        ("drug_allergies",     drug_allergies),
        ("general_allergies",  general_allergies),
        ("family_history",     family_history),
        ("smoking_status",     smoking_status),
        ("menstrual_history",  menstrual_history),
        ("lmp_date",           lmp_date),
        ("pregnancy_status",   pregnancy_status),
        ("obstetric_history",  obstetric_history),
        ("fall_history",       fall_history),
        ("vaccination_status", vaccination_status),
    ]:
        if val is not None:
            payload[field] = val

    try:
        response = (
            supabase.table("patient_history")
            .upsert(payload, on_conflict="patient_id")
            .execute()
        )
        if response.data:
            return f"Patient history saved successfully for patient_id {patient_id}."
        return "Failed to save patient history — no data returned."
    except Exception as e:
        return f"Error saving patient history: {e}"


# =====================================================================
# TOOL LIST — import this in your orchestrator
# =====================================================================



# =====================================================================
# HUMAN WAITLIST TOOLS
# =====================================================================

@tool
def add_to_waitlist(
    session_id: str,
    patient_id: int | None = None,
    patient_name: str | None = None,
    phone: str | None = None,
    complaint: str | None = None,
) -> str:
    """
    Add a patient to the human agent waitlist.
    Returns their position and estimated wait time.
    Call this after the patient confirms they want to speak with a human.
    """
    print(f"🛠️ [Tool] add_to_waitlist: session_id={session_id}")
    if not supabase:
        return "Database not connected."
    try:
        # Count how many are already waiting
        pending = (
            supabase.table("human_waitlist")
            .select("id", count="exact")
            .eq("status", "waiting")
            .execute()
        )
        position = (pending.count or 0) + 1
        wait_minutes = position * 5   # ~5 min per person

        supabase.table("human_waitlist").insert({
            "session_id":   session_id,
            "patient_id":   patient_id,
            "patient_name": patient_name,
            "phone":        phone,
            "complaint":    complaint,
            "status":       "waiting",
            "position":     position,
            "created_at":   datetime.now(PKT).isoformat(),
        }).execute()

        return (
            f"Added to waitlist. Position: {position}. "
            f"Estimated wait time: approximately {wait_minutes} minutes."
        )
    except Exception as e:
        return f"Error adding to waitlist: {e}"


@tool
def get_waitlist_position(session_id: str) -> str:
    """
    Get the current queue position and estimated wait for a session.
    """
    if not supabase:
        return "Database not connected."
    try:
        result = (
            supabase.table("human_waitlist")
            .select("position, status, created_at")
            .eq("session_id", session_id)
            .eq("status", "waiting")
            .limit(1)
            .execute()
        )
        if result.data:
            r = result.data[0]
            wait = r["position"] * 5
            return f"Position: {r['position']}. Estimated wait: ~{wait} minutes. Status: {r['status']}."
        return "Session not found in waitlist or already answered."
    except Exception as e:
        return f"Error: {e}"


@tool
def cancel_waitlist(session_id: str) -> str:
    """
    Remove a patient from the human waitlist (they changed their mind).
    """
    if not supabase:
        return "Database not connected."
    try:
        supabase.table("human_waitlist").update({"status": "cancelled"}).eq("session_id", session_id).execute()
        return f"Removed from waitlist. Continuing with automated booking."
    except Exception as e:
        return f"Error cancelling waitlist: {e}"


# =====================================================================
# TOOL LIST — import this in your orchestrator
# =====================================================================

ALL_TOOLS = [
    recommend_specialist_tool,
    get_doctors_by_specialization,
    search_knowledge,
    list_database_tables,
    query_database_table,
    lookup_customer_profile,
    register_customer_profile,
    update_patient_demographics,
    get_doctor_profile,
    find_provider_availability,
    get_doctor_schedule,
    create_booking,
    get_recent_case_notes,
    save_case_notes,
    get_patient_history,
    save_patient_history,
    add_to_waitlist,
    cancel_waitlist,
    get_waitlist_position,
]