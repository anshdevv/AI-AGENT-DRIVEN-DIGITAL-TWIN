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
        response = supabase.table("Patient").select("*").eq("phone", phone).limit(1).execute()
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
        payload = {"Name": name, "phone": phone}
        if gender:
            payload["Gender"] = gender
        if age is not None:
            payload["age"] = age
        response = supabase.table("Patient").insert(payload).execute()
        profile = response.data[0] if response.data else None
        return {"profile": profile}

    def recommend_service_provider(
        self,
        specialization: str | None = None,
        symptom: str | None = None,
        doctor_name: str | None = None,
    ) -> dict[str, Any]:
        if not supabase:
            return {"providers": [], "specialization": specialization, "symptom_match": None}

        symptom_match = self.match_symptoms_to_specialization(symptom or "") if symptom else None
        inferred_specialization = specialization or (symptom_match or {}).get("specialization") or ""
        if doctor_name:
            response = supabase.table("Doctors").select("*").ilike("Name", f"%{doctor_name}%").execute()
            providers = response.data or []
        elif inferred_specialization:
            providers = self._lookup_providers_by_specialization(inferred_specialization)
        else:
            providers = []

        return {
            "providers": providers,
            "specialization": inferred_specialization,
            "symptom_match": symptom_match,
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
        target_weekday = target_date.strftime("%a").lower()
        target_time = self._parse_time(time) if time else None
        available: list[dict[str, Any]] = []

        for provider in providers:
            if not supabase:
                continue
            slots_response = (
                supabase.table("doctor_availability")
                .select("*")
                .eq("doctor_id", provider["id"])
                .execute()
            )
            matching_slots: list[dict[str, Any]] = []
            for slot in slots_response.data or []:
                if not self._weekday_matches(slot.get("days", ""), target_weekday):
                    continue

                start_time = datetime.strptime(slot["start_time"], "%H:%M:%S").time()
                end_time = datetime.strptime(slot["end_time"], "%H:%M:%S").time()
                if target_time and not (start_time <= target_time <= end_time):
                    continue

                matching_slots.append(
                    {
                        "slot": slot,
                        "display_slot": f"{start_time.strftime('%H:%M')} - {end_time.strftime('%H:%M')}",
                    }
                )
            if matching_slots:
                available.append(
                    {
                        "doctor": provider,
                        "slot": matching_slots[0]["slot"],
                        "slots": [entry["slot"] for entry in matching_slots],
                        "display_slot": matching_slots[0]["display_slot"],
                        "display_slots": [entry["display_slot"] for entry in matching_slots],
                    }
                )

        return {
            "available": available,
            "date": target_date.strftime("%Y-%m-%d"),
            "weekday": target_weekday,
            "specialization": provider_result["specialization"],
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
            response = supabase.table("Doctors").select("*").eq("id", doctor_id).limit(1).execute()
            doctor = response.data[0] if response.data else None
        elif doctor_name:
            response = supabase.table("Doctors").select("*").ilike("Name", f"%{doctor_name}%").execute()
            if response.data:
                doctor = self._pick_best_name_match(doctor_name, response.data)

        if not doctor:
            return {"doctor": None, "schedule": []}

        slots_response = (
            supabase.table("doctor_availability")
            .select("*")
            .eq("doctor_id", doctor["id"])
            .execute()
        )
        schedule = [
            {
                "days": slot.get("days", ""),
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
        payload = {
            "patient_id": patient_id,
            "doctor_id": doctor_id,
            "appointment_date": date,
            "time": time,
        }
        response = supabase.table("appointments").insert(payload).execute()
        appointment = response.data[0] if response.data else None
        return {"appointment": appointment}

    def get_recent_case_notes(self, patient_id: int) -> dict[str, Any]:
        return {"notes": []}

    def save_case_notes(self, appointment_id: int, notes: str) -> dict[str, Any]:
        return {
            "appointment": None,
            "stored": False,
            "summary": notes,
            "reason": "The current appointments schema does not include a notes column.",
        }

    def _lookup_providers_by_specialization(self, specialization: str) -> list[dict[str, Any]]:
        if not supabase:
            return []

        providers_by_id: dict[Any, dict[str, Any]] = {}
        for option in self._split_specialization_options(specialization):
            response = (
                supabase.table("Doctors")
                .select("*")
                .ilike("Specialization", f"%{option}%")
                .execute()
            )
            for provider in response.data or []:
                providers_by_id[provider.get("id")] = provider
        return list(providers_by_id.values())

    def _pick_question_flow_path(self, symptom: str, specialization: str) -> Path:
        lowered = f"{symptom} {specialization}".lower()
        if "stomach" in lowered or "gastro" in lowered or "abdominal" in lowered:
            return self.domain.question_flow_dir / "stomach_pain.md"
        if "head" in lowered or "neuro" in lowered or "dizziness" in lowered:
            return self.domain.question_flow_dir / "headache.md"
        if "cough" in lowered or "ent" in lowered or "cold" in lowered:
            return self.domain.question_flow_dir / "cough.md"
        return self.domain.question_flow_dir / "fever.md"

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
        for fmt in ("%Y/%m/%d", "%Y-%m-%d", "%d/%m/%Y"):
            try:
                parsed = datetime.strptime(raw_value, fmt)
                return parsed.replace(tzinfo=PKT)
            except ValueError:
                continue
        return now

    @staticmethod
    def _parse_time(raw_value: str):
        cleaned = raw_value.strip().lower()
        for fmt in ("%H:%M", "%I:%M %p", "%I:%M%p", "%I %p"):
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

    @staticmethod
    def _format_db_time(raw_value: str) -> str:
        if not raw_value:
            return ""
        try:
            return datetime.strptime(raw_value, "%H:%M:%S").strftime("%H:%M")
        except ValueError:
            return raw_value

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
