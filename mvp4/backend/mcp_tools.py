from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from .config import supabase
from .domain import DomainConfig
from .knowledge import KnowledgeBase


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


class CustomerServiceTools:
    def __init__(self, domain: DomainConfig, knowledge_base: KnowledgeBase) -> None:
        self.domain = domain
        self.knowledge_base = knowledge_base
        self.symptom_map = self._load_symptom_map()

    def tool_definitions(self) -> list[dict[str, Any]]:
        return [
            {
                "name": "search_knowledge",
                "description": "Search FAQ and policy content for customer support answers.",
                "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
            },
            {
                "name": "list_database_tables",
                "description": "List the Supabase tables this assistant can inspect for context.",
                "inputSchema": {"type": "object", "properties": {}},
            },
            {
                "name": "query_database_table",
                "description": "Read rows from a Supabase table with optional filters for context-aware assistance.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "table_name": {"type": "string"},
                        "columns": {"type": "string"},
                        "limit": {"type": "number"},
                        "filters": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "column": {"type": "string"},
                                    "op": {"type": "string"},
                                    "value": {},
                                },
                            },
                        },
                        "order_by": {"type": "string"},
                        "ascending": {"type": "boolean"},
                    },
                    "required": ["table_name"],
                },
            },
            {
                "name": "match_symptoms_to_specialization",
                "description": "Map a symptom description to the most relevant doctor specialization.",
                "inputSchema": {"type": "object", "properties": {"symptom": {"type": "string"}}},
            },
            {
                "name": "lookup_customer_profile",
                "description": "Look up a customer or patient profile by phone number.",
                "inputSchema": {"type": "object", "properties": {"phone": {"type": "string"}}},
            },
            {
                "name": "register_customer_profile",
                "description": "Create a customer or patient profile.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "phone": {"type": "string"},
                        "gender": {"type": "string"},
                        "age": {"type": "number"},
                    },
                },
            },
            {
                "name": "recommend_service_provider",
                "description": "Find doctors or other providers by specialization, symptom, or name.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "specialization": {"type": "string"},
                        "symptom": {"type": "string"},
                        "doctor_name": {"type": "string"},
                    },
                },
            },
            {
                "name": "get_doctor_profile",
                "description": "Fetch a doctor's profile together with the weekly schedule.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doctor_name": {"type": "string"},
                        "doctor_id": {"type": "number"},
                    },
                },
            },
            {
                "name": "find_provider_availability",
                "description": "Find which providers are available on a specific date and time.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "specialization": {"type": "string"},
                        "doctor_name": {"type": "string"},
                        "date": {"type": "string"},
                        "time": {"type": "string"},
                    },
                },
            },
            {
                "name": "get_doctor_schedule",
                "description": "Get the weekly consultation schedule for a specific doctor.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "doctor_id": {"type": "number"},
                        "doctor_name": {"type": "string"},
                    },
                },
            },
            {
                "name": "get_triage_flow",
                "description": "Get safe intake questions and red flags for a symptom or specialization.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "symptom": {"type": "string"},
                        "specialization": {"type": "string"},
                    },
                },
            },
            {
                "name": "create_booking",
                "description": "Create an appointment or booking for a customer.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "patient_id": {"type": "number"},
                        "doctor_id": {"type": "number"},
                        "date": {"type": "string"},
                        "time": {"type": "string"},
                    },
                },
            },
            {
                "name": "get_recent_case_notes",
                "description": "Fetch recent appointment notes for a customer.",
                "inputSchema": {"type": "object", "properties": {"patient_id": {"type": "number"}}},
            },
            {
                "name": "save_case_notes",
                "description": "Persist intake or triage notes for the current booking.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"appointment_id": {"type": "number"}, "notes": {"type": "string"}},
                },
            },
        ]

    def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> ToolCallResult:
        args = arguments or {}
        handlers = {
            "search_knowledge": self.search_knowledge,
            "list_database_tables": self.list_database_tables,
            "query_database_table": self.query_database_table,
            "match_symptoms_to_specialization": self.match_symptoms_to_specialization,
            "lookup_customer_profile": self.lookup_customer_profile,
            "register_customer_profile": self.register_customer_profile,
            "recommend_service_provider": self.recommend_service_provider,
            "get_doctor_profile": self.get_doctor_profile,
            "find_provider_availability": self.find_provider_availability,
            "get_doctor_schedule": self.get_doctor_schedule,
            "get_triage_flow": self.get_triage_flow,
            "create_booking": self.create_booking,
            "get_recent_case_notes": self.get_recent_case_notes,
            "save_case_notes": self.save_case_notes,
        }
        handler = handlers.get(name)
        if not handler:
            return ToolCallResult(ok=False, data={}, error=f"Unknown tool: {name}")

        try:
            return ToolCallResult(ok=True, data=handler(**args))
        except Exception as exc:
            return ToolCallResult(ok=False, data={}, error=str(exc))

    def list_database_tables(self) -> dict[str, Any]:
        return {
            "tables": list(KNOWN_SUPABASE_TABLES),
            "read_only": True,
            "supports_arbitrary_table_query": True,
        }

    def query_database_table(
        self,
        table_name: str,
        columns: str = "*",
        limit: int = 20,
        filters: list[dict[str, Any]] | None = None,
        order_by: str | None = None,
        ascending: bool = True,
    ) -> dict[str, Any]:
        if not supabase:
            raise RuntimeError("Supabase is not configured.")

        table_name = str(table_name).strip()
        if not table_name:
            raise ValueError("table_name is required.")

        safe_limit = max(1, min(int(limit), 100))
        query = supabase.table(table_name).select(columns or "*")

        for filter_item in filters or []:
            column = str(filter_item.get("column") or "").strip()
            op = str(filter_item.get("op") or "eq").strip().lower()
            value = filter_item.get("value")
            if not column:
                continue
            query = self._apply_query_filter(query, column, op, value)

        if order_by:
            query = query.order(str(order_by).strip(), desc=not ascending)

        response = query.limit(safe_limit).execute()
        rows = response.data or []
        columns_seen = sorted({key for row in rows if isinstance(row, dict) for key in row.keys()})
        return {
            "table_name": table_name,
            "rows": rows,
            "columns": columns_seen,
            "count": len(rows),
            "read_only": True,
        }

    def _load_symptom_map(self) -> list[tuple[list[list[str]], str]]:
        path = self.domain.symptom_map_path
        if not path.exists():
            return []

        mappings: list[tuple[list[list[str]], str]] = []
        text = path.read_text(encoding="utf-8", errors="replace")
        for line in text.splitlines():
            parsed = self._split_mapping_line(line)
            if not parsed:
                continue

            symptom_text, specialization = parsed
            phrase_groups = [
                self._normalize_phrase_tokens(item)
                for item in symptom_text.split(",")
                if item.strip()
            ]
            phrase_groups = [tokens for tokens in phrase_groups if tokens]
            if phrase_groups:
                mappings.append((phrase_groups, specialization))

        for specialization, aliases in SPECIALIZATION_ALIASES.items():
            mappings.append((aliases, specialization))
        return mappings

    def infer_specialization(self, text: str) -> str | None:
        result = self.match_symptoms_to_specialization(text)
        return result.get("specialization")

    def search_knowledge(self, query: str) -> dict[str, Any]:
        hits = self.knowledge_base.search(query)
        return {
            "matches": [
                {"source": hit.source, "content": hit.content, "score": hit.score}
                for hit in hits
            ]
        }

    def match_symptoms_to_specialization(self, symptom: str) -> dict[str, Any]:
        normalized_text = self._normalize_text(symptom)
        token_set = set(self._extract_words(symptom))
        if not normalized_text and not token_set:
            return {"specialization": None, "confidence": 0.0, "matches": []}

        grouped_matches: dict[str, SymptomEvidence] = {}
        for phrase_groups, specialization in self.symptom_map:
            matched_phrases: list[str] = []
            best_group_score = 0.0
            for tokens in phrase_groups:
                score = self._phrase_match_score(tokens, normalized_text, token_set)
                if score <= 0:
                    continue
                matched_phrases.append(" ".join(tokens))
                best_group_score = max(best_group_score, score)

            if not matched_phrases:
                continue

            evidence = grouped_matches.get(specialization)
            total_score = min(0.99, best_group_score + (0.07 * max(len(matched_phrases) - 1, 0)))
            if evidence:
                merged_phrases = sorted(set(evidence.matched_phrases + matched_phrases))
                grouped_matches[specialization] = SymptomEvidence(
                    specialization=specialization,
                    matched_phrases=merged_phrases,
                    score=max(evidence.score, total_score),
                )
            else:
                grouped_matches[specialization] = SymptomEvidence(
                    specialization=specialization,
                    matched_phrases=sorted(set(matched_phrases)),
                    score=total_score,
                )

        ranked = sorted(grouped_matches.values(), key=lambda item: item.score, reverse=True)
        best = ranked[0] if ranked else None
        return {
            "specialization": best.specialization if best else None,
            "confidence": round(best.score, 2) if best else 0.0,
            "matches": [
                {
                    "specialization": item.specialization,
                    "matched_phrases": item.matched_phrases,
                    "score": round(item.score, 2),
                }
                for item in ranked[:3]
            ],
        }

    def lookup_customer_profile(self, phone: str) -> dict[str, Any]:
        if not supabase:
            return {"profile": None}
        response = supabase.table("patients").select("*").eq("phone", phone).limit(1).execute()
        profile = response.data[0] if response.data else None
        return {"profile": profile}

    def register_customer_profile(
        self,
        name: str,
        phone: str,
        gender: str | None = None,
        age: float | None = None,
    ) -> dict[str, Any]:
        if not supabase:
            raise RuntimeError("Supabase is not configured.")
        payload = {"name": name, "phone": phone}
        response = supabase.table("patients").insert(payload).execute()
        profile = response.data[0] if response.data else None
        return {"profile": profile}

    def recommend_service_provider(
        self,
        specialization: str | None = None,
        symptom: str | None = None,
        doctor_name: str | None = None,
    ) -> dict[str, Any]:
        if not supabase:
            return {
                "providers": [],
                "specialization": specialization,
                "symptom_match": None,
                "supported_specializations": list(SUPPORTED_SPECIALIZATIONS_FALLBACK),
                "requested_specialization_supported": bool(not specialization),
            }

        symptom_match = self.match_symptoms_to_specialization(symptom or "") if symptom else None
        inferred_specialization = specialization or (symptom_match or {}).get("specialization") or ""
        supported_specializations = self.get_supported_specializations()
        if doctor_name:
            response = supabase.table("doctors").select("*").ilike("name", f"%{doctor_name}%").execute()
            providers = response.data or []
        elif inferred_specialization:
            providers = self._lookup_providers_by_specialization(inferred_specialization)
        else:
            providers = []

        providers = [self._normalize_doctor_record(provider) for provider in providers]
        return {
            "providers": providers,
            "specialization": inferred_specialization,
            "symptom_match": symptom_match,
            "supported_specializations": supported_specializations,
            "requested_specialization_supported": self._specialization_is_supported(
                inferred_specialization,
                supported_specializations,
            ),
        }

    def get_doctor_profile(
        self,
        doctor_name: str | None = None,
        doctor_id: int | None = None,
    ) -> dict[str, Any]:
        schedule_payload = self.get_doctor_schedule(doctor_id=doctor_id, doctor_name=doctor_name)
        return {
            "doctor": schedule_payload.get("doctor"),
            "schedule": schedule_payload.get("schedule", []),
        }

    def find_provider_availability(
        self,
        specialization: str | None = None,
        doctor_name: str | None = None,
        date: str | None = None,
        time: str | None = None,
    ) -> dict[str, Any]:
        provider_result = self.recommend_service_provider(
            specialization=specialization,
            doctor_name=doctor_name,
        )
        providers = provider_result["providers"]
        target_date = self._parse_date(date)
        target_weekday = self._day_of_week_from_date(target_date)
        target_time = self._parse_time(time) if time else None
        available: list[dict[str, Any]] = []

        for provider in providers:
            if not supabase:
                continue
            schedule_rows = self._get_schedule_rows(provider["id"], target_weekday)
            if not schedule_rows:
                continue

            matching_slots = self._build_bookable_slots(
                doctor_id=provider["id"],
                target_date=target_date,
                schedule_rows=schedule_rows,
            )
            if target_time is not None:
                matching_slots = [
                    slot
                    for slot in matching_slots
                    if self._slot_starts_at(slot, target_time)
                ]

            if matching_slots:
                display_slots = [self._format_slot_start(slot) for slot in matching_slots[:6]]
                available.append(
                    {
                        "doctor": provider,
                        "slot": matching_slots[0],
                        "slots": matching_slots,
                        "display_slot": display_slots[0],
                        "display_slots": display_slots,
                        "slot_count": len(matching_slots),
                    }
                )

        return {
            "available": available,
            "date": target_date.strftime("%Y-%m-%d"),
            "weekday": str(target_weekday),
            "specialization": provider_result["specialization"],
            "supported_specializations": provider_result.get("supported_specializations", []),
            "requested_specialization_supported": provider_result.get("requested_specialization_supported", True),
        }
    
    def get_doctor_schedule(
        self,
        doctor_id: int | None = None,
        doctor_name: str | None = None,
    ) -> dict[str, Any]:
        if not supabase:
            return {"doctor": None, "schedule": []}

        doctor: dict[str, Any] | None = None
        if doctor_id is not None:
            response = supabase.table("doctors").select("*").eq("id", doctor_id).limit(1).execute()
            doctor = response.data[0] if response.data else None
        elif doctor_name:
            response = supabase.table("doctors").select("*").ilike("name", f"%{doctor_name}%").execute()
            if response.data:
                doctor = self._pick_best_name_match(doctor_name, response.data)

        if not doctor:
            return {"doctor": None, "schedule": []}

        doctor = self._normalize_doctor_record(doctor)
        slots_response = (
            supabase.table("doctor_availability")
            .select("*")
            .eq("doctor_id", doctor["id"])
            .execute()
        )
        schedule = [
            {
                "days": self._day_of_week_name(slot.get("day_of_week")),
                "start_time": self._format_db_time(slot.get("start_time", "")),
                "end_time": self._format_db_time(slot.get("end_time", "")),
            }
            for slot in (slots_response.data or [])
        ]
        return {"doctor": doctor, "schedule": schedule}

    def get_triage_flow(
        self,
        symptom: str | None = None,
        specialization: str | None = None,
    ) -> dict[str, Any]:
        path = self._pick_question_flow_path(symptom or "", specialization or "")
        parsed = self._parse_question_flow(path)
        return {
            "flow_name": path.stem,
            "questions": parsed["questions"],
            "red_flags": parsed["red_flags"],
            "path": str(path),
        }

    def create_booking(self, patient_id: int, doctor_id: int, date: str, time: str) -> dict[str, Any]:
        if not supabase:
            raise RuntimeError("Supabase is not configured.")
        try:
            target_date = self._parse_date(date)
            target_time = self._parse_time(time)
        except ValueError as exc:
            return {
                "appointment": None,
                "slot": None,
                "error_code": "invalid_datetime",
                "message": str(exc),
            }

        weekday = self._day_of_week_from_date(target_date)
        schedule_rows = self._get_schedule_rows(doctor_id, weekday)
        if not schedule_rows:
            return {
                "appointment": None,
                "slot": None,
                "error_code": "schedule_unavailable",
                "message": "No schedule is loaded for that doctor on the requested day.",
            }

        bookable_slots = self._build_bookable_slots(
            doctor_id=doctor_id,
            target_date=target_date,
            schedule_rows=schedule_rows,
        )
        chosen_slot = next(
            (slot for slot in bookable_slots if self._slot_starts_at(slot, target_time)),
            None,
        )
        if not chosen_slot:
            return {
                "appointment": None,
                "slot": None,
                "error_code": "slot_unavailable",
                "message": "The requested time is no longer open.",
            }

        created_slot = supabase.table("slots").insert(
            {
                "doctor_id": doctor_id,
                "start_time": chosen_slot["start_time"],
                "end_time": chosen_slot["end_time"],
                "status": "booked",
            }
        ).execute()
        slot_record = created_slot.data[0] if created_slot.data else None
        if not slot_record:
            return {
                "appointment": None,
                "slot": None,
                "error_code": "slot_create_failed",
                "message": "The slot could not be reserved.",
            }

        appointment_payload = {
            "slot_id": slot_record["id"],
            "doctor_id": doctor_id,
            "patient_id": patient_id,
            "status": "booked",
        }
        response = supabase.table("appointments").insert(appointment_payload).execute()
        appointment = response.data[0] if response.data else None
        if not appointment:
            supabase.table("slots").update({"status": "available"}).eq("id", slot_record["id"]).execute()
            return {
                "appointment": None,
                "slot": slot_record,
                "error_code": "appointment_create_failed",
                "message": "The appointment row could not be created.",
            }

        try:
            supabase.table("appointment_events").insert(
                {"appointment_id": appointment["id"], "event_type": "created"}
            ).execute()
        except Exception:
            pass

        return {
            "appointment": appointment,
            "slot": slot_record,
            "error_code": None,
        }

    def get_recent_case_notes(self, patient_id: int) -> dict[str, Any]:
        if not supabase:
            return {"notes": []}
        response = (
            supabase.table("appointments")
            .select("id, notes, doctor_id, created_at")
            .eq("patient_id", patient_id)
            .neq("notes", None)
            .order("created_at", desc=True)
            .limit(5)
            .execute()
        )
        return {"notes": response.data or []}

    def save_case_notes(self, appointment_id: int, notes: str) -> dict[str, Any]:
        if not supabase:
            raise RuntimeError("Supabase is not configured.")
        response = (
            supabase.table("appointments")
            .update({"notes": notes})
            .eq("id", appointment_id)
            .execute()
        )
        return {
            "appointment": response.data[0] if response.data else None,
            "stored": bool(response.data),
            "summary": notes,
        }

    def _lookup_providers_by_specialization(self, specialization: str) -> list[dict[str, Any]]:
        if not supabase:
            return []

        providers_by_id: dict[Any, dict[str, Any]] = {}
        for option in self._split_specialization_options(specialization):
            response = (
                supabase.table("doctors")
                .select("*")
                .ilike("specialization", f"%{option}%")
                .execute()
            )
            for provider in response.data or []:
                provider_specialization = str(provider.get("specialization", ""))
                if not self._specialization_matches(provider_specialization, option):
                    continue
                providers_by_id[provider.get("id")] = provider
        return [self._normalize_doctor_record(provider) for provider in providers_by_id.values()]

    def _pick_question_flow_path(self, symptom: str, specialization: str) -> Path:
        normalized = self._normalize_text(f"{symptom} {specialization}")
        token_set = set(self._extract_words(f"{symptom} {specialization}"))
        approved_key = None
        for key in APPROVED_TRIAGE_FLOW_BY_KEY:
            if self._contains_term(normalized, token_set, key):
                approved_key = key
                break
        if approved_key:
            return self.domain.question_flow_dir / APPROVED_TRIAGE_FLOW_BY_KEY[approved_key]
        return self.domain.question_flow_dir / GENERAL_TRIAGE_FLOW

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _parse_time(raw_value: str):
        cleaned = raw_value.strip().lower()
        for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p", "%I%p"):
            try:
                return datetime.strptime(cleaned.upper(), fmt).time()
            except ValueError:
                continue
        raise ValueError("Time must look like 14:30 or 2:30 PM.")

    @staticmethod
    def _weekday_matches(day_expression: str, target: str) -> bool:
        values = [item.strip().lower() for item in re.split(r"[,\s]+", day_expression) if item.strip()]
        if target in values:
            return True
        if "-" not in day_expression:
            return False

        day_order = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
        start, end = [item.strip().lower() for item in day_expression.split("-", 1)]
        if start not in day_order or end not in day_order or target not in day_order:
            return False

        start_index = day_order.index(start)
        end_index = day_order.index(end)
        target_index = day_order.index(target)
        if start_index <= end_index:
            return start_index <= target_index <= end_index
        return target_index >= start_index or target_index <= end_index

    @staticmethod
    def _split_mapping_line(line: str) -> tuple[str, str] | None:
        for separator in ("â†’", "→", "Ã¢â€ â€™", "->"):
            if separator in line:
                symptom_text, specialization = [part.strip() for part in line.split(separator, 1)]
                return symptom_text, specialization
        return None

    @staticmethod
    def _normalize_text(text: str) -> str:
        return " ".join(WORD_RE.findall(text.lower()))

    @staticmethod
    def _extract_words(text: str) -> list[str]:
        return WORD_RE.findall(text.lower())

    @staticmethod
    def _normalize_phrase_tokens(text: str) -> list[str]:
        return WORD_RE.findall(text.lower())

    @staticmethod
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

    @staticmethod
    def _split_specialization_options(specialization: str) -> list[str]:
        options = re.split(r"/|\bor\b", specialization, flags=re.IGNORECASE)
        return [option.strip() for option in options if option.strip()]

    def get_supported_specializations(self) -> list[str]:
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

    @staticmethod
    def _format_db_time(raw_value: str) -> str:
        if not raw_value:
            return ""
        try:
            return datetime.strptime(raw_value, "%H:%M:%S").strftime("%H:%M")
        except ValueError:
            return raw_value

    @staticmethod
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

    @staticmethod
    def _specialization_is_supported(specialization: str, supported_specializations: list[str]) -> bool:
        if not specialization:
            return True
        supported = {item.lower() for item in supported_specializations}
        for option in CustomerServiceTools._split_specialization_options(specialization):
            if option.lower() in supported:
                return True
        return False

    @staticmethod
    def _slot_starts_at(slot: dict[str, Any], target_time: datetime.time) -> bool:
        slot_start = datetime.fromisoformat(str(slot["start_time"])).timetz().replace(tzinfo=None)
        return slot_start == target_time

    @staticmethod
    def _format_slot_start(slot: dict[str, Any]) -> str:
        return datetime.fromisoformat(str(slot["start_time"])).strftime("%H:%M")

    @staticmethod
    def _is_blocking_slot(slot: dict[str, Any], active_appointment_slot_ids: set[int]) -> bool:
        status = str(slot.get("status", "")).strip().lower()
        if slot.get("id") in active_appointment_slot_ids:
            return True
        return status in {"booked", "confirmed", "held"}

    @staticmethod
    def _slot_overlaps(
        slot_start: datetime,
        slot_end: datetime,
        blocked_start: datetime,
        blocked_end: datetime,
    ) -> bool:
        return slot_start < blocked_end and blocked_start < slot_end

    def _get_schedule_rows(self, doctor_id: int, weekday: int) -> list[dict[str, Any]]:
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
        self,
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
            if not self._is_blocking_slot(slot, active_appointment_slot_ids):
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
                    self._slot_overlaps(slot_start, slot_end, blocked_start, blocked_end)
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

    @staticmethod
    def _day_of_week_from_date(value: datetime) -> int:
        # In the schema, 0=Sunday
        return (value.weekday() + 1) % 7

    @staticmethod
    def _day_of_week_name(value: int | None) -> str:
        if value is None:
            return ""
        try:
            names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
            return names[int(value) % 7]
        except Exception:
            return str(value)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def _specialization_matches(provider_specialization: str, requested_option: str) -> bool:
        provider_tokens = WORD_RE.findall(provider_specialization.lower())
        requested_tokens = WORD_RE.findall(requested_option.lower())
        if not provider_tokens or not requested_tokens:
            return False

        provider_text = " ".join(provider_tokens)
        if len(requested_tokens) == 1:
            return requested_tokens[0] in set(provider_tokens)
        return " ".join(requested_tokens) in provider_text

    @staticmethod
    def _contains_term(normalized_text: str, token_set: set[str], term: str) -> bool:
        term_tokens = WORD_RE.findall(term.lower())
        if not term_tokens:
            return False
        if len(term_tokens) == 1:
            return term_tokens[0] in token_set
        return " ".join(term_tokens) in normalized_text
