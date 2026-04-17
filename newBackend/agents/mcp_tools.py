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
    if not raw_value:
        return now
    lowered = raw_value.strip().lower()
    if lowered == "today":
        return now
    if lowered == "tomorrow":
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
    response = (
        supabase.table("doctor_availability")
        .select("*")
        .eq("doctor_id", doctor_id)
        .eq("day_of_week", weekday)
        .execute()
    )
    return sorted(
        list(response.data or []),
        key=lambda row: (
            str(row.get("start_time", "")),
            str(row.get("end_time", "")),
        ),
    )


def _build_bookable_slots(
    *,
    doctor_id: int,
    target_date: datetime,
    schedule_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not supabase:
        return []

    day_start = datetime.combine(target_date.date(), datetime.min.time()).replace(tzinfo=PKT)
    day_end = day_start + timedelta(days=1)
    slot_response = (
        supabase.table("slots")
        .select("*")
        .eq("doctor_id", doctor_id)
        .gte("start_time", day_start.isoformat())
        .lt("start_time", day_end.isoformat())
        .execute()
    )
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

    bookable_slots: list[dict[str, Any]] = []
    for schedule_row in schedule_rows:
        duration_minutes = int(schedule_row.get("slot_duration_minutes") or 15)
        start_time = datetime.strptime(str(schedule_row["start_time"]), "%H:%M:%S").time()
        end_time = datetime.strptime(str(schedule_row["end_time"]), "%H:%M:%S").time()
        slot_start = datetime.combine(target_date.date(), start_time).replace(tzinfo=PKT)
        schedule_end = datetime.combine(target_date.date(), end_time).replace(tzinfo=PKT)

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
                    "doctor_id": doctor_id,
                    "start_time": slot_start.isoformat(),
                    "end_time": slot_end.isoformat(),
                    "slot_duration_minutes": duration_minutes,
                    "source_schedule_id": schedule_row.get("id"),
                    "status": "available",
                }
            )
            slot_start = slot_end

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
        # Strict temperature 0 for deterministic, safe routing
        llm = get_llm(temperature=0.0)
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
        
        print(f"   ↳ LLM matched '{symptom}' to: {specialist}")
        return f"Recommended specialist: {specialist}"
        
    except Exception as e:
        print(f"   ↳ LLM error: {e}. Falling back to: General Physician")
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
        f"Dr. {doctor.get('name')} | {doctor.get('specialization')}",
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
    date: str | None = None,
    time: str | None = None,
) -> str:
    """
    Find available appointment slots for a doctor or specialization on a given date.
    Date accepts: today, tomorrow, day after tomorrow, weekday names (monday etc), or YYYY-MM-DD.
    Time (optional) filters to a specific start time, e.g. '14:30' or '2:30 PM'.
    Checks actual booked/held slots to show only truly free times.
    """
    print(f"🛠️ [Tool] find_provider_availability: spec={specialization}, doctor={doctor_name}, date={date}, time={time}")
    if not supabase:
        return "Database not connected."

    # Resolve providers
    if doctor_name:
        response = supabase.table("doctors").select("*").ilike("name", f"%{doctor_name}%").execute()
        providers = [_normalize_doctor_record(p) for p in (response.data or [])]
    elif specialization:
        providers = _lookup_providers_by_specialization(specialization)
    else:
        return "Provide at least a specialization or doctor_name."

    if not providers:
        return "No providers found matching that criteria."

    target_date = _parse_date(date)
    target_weekday = _day_of_week_from_date(target_date)
    target_time = _parse_time(time) if time else None
    available = []

    for provider in providers:
        schedule_rows = _get_schedule_rows(provider["id"], target_weekday)
        if not schedule_rows:
            continue

        matching_slots = _build_bookable_slots(
            doctor_id=provider["id"],
            target_date=target_date,
            schedule_rows=schedule_rows,
        )
        if target_time is not None:
            matching_slots = [s for s in matching_slots if _slot_starts_at(s, target_time)]

        if matching_slots:
            display_slots = [_format_slot_start(s) for s in matching_slots[:6]]
            available.append((provider, display_slots, len(matching_slots)))

    if not available:
        day_name = target_date.strftime("%A")
        date_str = target_date.strftime("%Y-%m-%d")
        return f"No available slots found on {day_name} ({date_str})."

    lines = [f"Available slots on {target_date.strftime('%A, %Y-%m-%d')}:"]
    for provider, display_slots, total in available:
        slots_str = ", ".join(display_slots)
        extra = f" (+{total - len(display_slots)} more)" if total > len(display_slots) else ""
        lines.append(f"  Dr. {provider.get('name')} ({provider.get('specialization')}): {slots_str}{extra}")
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


# =====================================================================
# TOOL LIST — import this in your orchestrator
# =====================================================================

ALL_TOOLS = [
    recommend_specialist_tool,
    search_knowledge,
    list_database_tables,
    query_database_table,
    lookup_customer_profile,
    register_customer_profile,
    get_doctor_profile,
    find_provider_availability,
    get_doctor_schedule,
    create_booking,
    get_recent_case_notes,
    save_case_notes,
]


class ToolsWrapper:
    def __init__(self, tools: list):
        self.tools = tools

    def tool_definitions(self) -> list[dict[str, Any]]:
        definitions = []
        for tool in self.tools:
            schema = tool.get_input_schema()
            definitions.append({
                "name": tool.name,
                "description": tool.description,
                "inputSchema": schema,
            })
        return definitions

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolCallResult:
        for tool in self.tools:
            if tool.name == name:
                try:
                    result = tool.invoke(arguments)
                    return ToolCallResult(ok=True, data={"result": result}, error=None)
                except Exception as e:
                    return ToolCallResult(ok=False, data={}, error=str(e))
        return ToolCallResult(ok=False, data={}, error=f"Tool {name} not found")


tools = ToolsWrapper(ALL_TOOLS)