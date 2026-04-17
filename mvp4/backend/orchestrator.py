from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

try:
    from langgraph.checkpoint.memory import InMemorySaver
except Exception:  # pragma: no cover - optional import guard
    InMemorySaver = None

from .domain import get_domain_config
from .graph import ConversationState, create_conversation_graph
from .knowledge import KnowledgeBase
from .llm import llm
from .mcp_tools import CustomerServiceTools, ToolCallResult
from .settings import settings


try:
    PKT = ZoneInfo("Asia/Karachi")
except ZoneInfoNotFoundError:  # pragma: no cover - Windows installs may miss tzdata
    PKT = timezone(timedelta(hours=5))


PHONE_RE = re.compile(r"(?:\+92|0)?\d{10,11}")
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}(?:\s?[ap]m)?|\d{1,2}\s?[ap]m)\b", re.IGNORECASE)
DATE_RE = re.compile(r"\b(?:\d{4}[/-]\d{2}[/-]\d{2}|\d{2}/\d{2}/\d{4})\b")
DOCTOR_RE = re.compile(r"\bdr\.?\s*([a-zA-Z]+(?:[\s.]+[a-zA-Z]+)*)", re.IGNORECASE)
NAMEISH_RE = re.compile(r"^[A-Za-z]+(?:\s+[A-Za-z]+)+$")
WORD_RE = re.compile(r"[a-z0-9]+")

GREETING_WORDS = {"hello", "hi", "hey", "salam", "assalam", "assalamualaikum"}
HANDOFF_WORDS = {"human", "representative", "agent", "manager", "complaint"}
BOOKING_WORDS = {"book", "booking", "appointment", "reserve"}
DOCTOR_INFO_WORDS = {
    "available",
    "availability",
    "schedule",
    "timing",
    "time",
    "when",
    "consulting",
    "consultation",
    "experience",
    "profile",
    "fee",
    "fees",
}
DOCTOR_PROFILE_PHRASES = {
    "who is",
    "tell me about",
    "about dr",
    "about doctor",
    "doctor info",
    "doctor information",
    "doctor profile",
    "profile of",
}
AVAILABILITY_QUERY_WORDS = {"available", "availability", "slot", "slots", "free", "open", "today", "tomorrow"}
DOCTOR_NAME_STOP_WORDS = (
    DOCTOR_INFO_WORDS
    | BOOKING_WORDS
    | AVAILABILITY_QUERY_WORDS
    | {"doctor", "doctors", "specialist", "specialists", "profile", "about", "please", "now"}
)
ALTERNATIVE_PROVIDER_WORDS = {"another", "other", "anyother", "else", "different"}
FAQ_WORDS = {
    "timing",
    "hours",
    "insurance",
    "lab",
    "test",
    "scan",
    "xray",
    "location",
    "address",
    "parking",
    "open",
    "close",
    "visit",
    "visiting",
}
MEDICATION_WORDS = {
    "medicine",
    "medication",
    "prescribe",
    "prescription",
    "tablet",
    "antibiotic",
    "painkiller",
    "dose",
    "dosage",
}
SYMPTOM_HINT_WORDS = {
    "pain",
    "fever",
    "cough",
    "rash",
    "headache",
    "dizziness",
    "vomiting",
    "nausea",
    "diarrhea",
    "bloating",
    "fatigue",
    "shortness",
    "breath",
    "itching",
}
URGENT_PHRASES = {
    "chest pain",
    "shortness of breath",
    "difficulty breathing",
    "trouble breathing",
    "coughing blood",
    "severe bleeding",
    "unconscious",
    "passed out",
    "loss of consciousness",
    "stroke",
    "heart attack",
    "suicidal",
}
SPECIALIZATION_HINTS = {
    "cardiologist": "Cardiologist",
    "cardiology": "Cardiologist",
    "ent": "ENT or General Physician",
    "ear nose throat": "ENT or General Physician",
    "dermatologist": "Dermatologist",
    "skin specialist": "Dermatologist",
    "neurologist": "Neurologist",
    "pediatrician": "Pediatrician",
    "paediatrician": "Pediatrician",
    "child specialist": "Pediatrician",
    "orthopedic": "Orthopedic",
    "orthopaedic": "Orthopedic",
    "orthopeadic": "Orthopedic",
    "orthopaedist": "Orthopedic",
    "urologist": "Urologist",
    "endocrinologist": "Endocrinologist",
    "gynecologist": "Gynecologist",
    "gynaecologist": "Gynecologist",
    "psychiatrist": "Psychiatrist / Psychologist",
    "psychologist": "Psychiatrist / Psychologist",
    "ophthalmologist": "Ophthalmologist",
    "eye specialist": "Ophthalmologist",
    "gastroenterologist": "Gastroenterologist / General Physician",
    "general physician": "General Physician",
    "physician": "General Physician",
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
ORDINAL_MAP = {
    "first": 0,
    "1st": 0,
    "second": 1,
    "2nd": 1,
    "third": 2,
    "3rd": 2,
    "fourth": 3,
    "4th": 3,
}
RESET_PHRASES = {"start over", "reset", "forget that", "new appointment"}


@dataclass(slots=True)
class SessionState:
    session_id: str
    history: list[dict[str, Any]] = field(default_factory=list)
    slots: dict[str, Any] = field(default_factory=dict)
    workflow: dict[str, Any] = field(default_factory=dict)
    last_intent: str = "clarify"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OrchestratorResult:
    session_id: str
    reply: str
    intent: str
    action: str
    state: SessionState
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntentDecision:
    intent: str
    confidence: float
    used_llm: bool = False


class HybridIntentClassifier:
    def __init__(self, tools: CustomerServiceTools) -> None:
        self.tools = tools

    def classify(self, state: ConversationState) -> IntentDecision:
        message = str(state.get("current_message", "")).strip()
        lowered = message.lower()
        entities = state.get("current_entities", {}) or {}
        booking_state = state.get("booking_state", {}) or {}
        triage_state = state.get("triage_state", {}) or {}

        if state.get("safety_flags"):
            return IntentDecision("safety", 1.0)
        if self._contains_any_token(lowered, HANDOFF_WORDS):
            return IntentDecision("handoff", 0.98)
        if self._contains_any_token(lowered, MEDICATION_WORDS):
            return IntentDecision("safety", 0.95)
        if self._is_greeting(lowered):
            return IntentDecision("greeting", 0.94)

        if triage_state.get("status") == "collecting":
            if not self._looks_like_topic_switch(lowered, entities):
                return IntentDecision("triage", 0.9)

        if booking_state.get("status") in {"collect_phone", "collect_name", "collect_details", "needs_provider"}:
            if self._looks_like_doctor_info(lowered, entities, state):
                return IntentDecision("doctor_info", 0.84)
            if self._looks_like_faq(lowered):
                return IntentDecision("faq", 0.78)
            return IntentDecision("book_appointment", 0.88)

        if self._looks_like_booking(lowered, entities):
            return IntentDecision("book_appointment", 0.96)
        if self._looks_like_doctor_info(lowered, entities, state):
            return IntentDecision("doctor_info", 0.9)
        if self._looks_like_recommendation(lowered, entities, state):
            return IntentDecision("recommend_doctor", 0.9)
        if self._looks_like_faq(lowered):
            return IntentDecision("faq", 0.82)

        if llm.enabled:
            decision = self._llm_backup(message, state)
            if decision:
                return decision
        return IntentDecision("clarify", 0.3)

    @staticmethod
    def _contains_any_token(lowered: str, options: set[str]) -> bool:
        normalized = " ".join(WORD_RE.findall(lowered))
        token_set = set(normalized.split())
        for option in options:
            option_tokens = WORD_RE.findall(option.lower())
            if not option_tokens:
                continue
            if len(option_tokens) == 1:
                if option_tokens[0] in token_set:
                    return True
                continue
            if " ".join(option_tokens) in normalized:
                return True
        return False

    @staticmethod
    def _is_greeting(lowered: str) -> bool:
        words = lowered.split()
        return bool(words) and len(words) <= 4 and any(word in GREETING_WORDS for word in words)

    def _looks_like_booking(self, lowered: str, entities: dict[str, Any]) -> bool:
        if self._contains_any_token(lowered, BOOKING_WORDS):
            return True
        if "schedule appointment" in lowered or "schedule a visit" in lowered:
            return True
        return bool(entities.get("date") or entities.get("time")) and bool(
            entities.get("doctor_name") or entities.get("specialization")
        )

    def _looks_like_doctor_info(self, lowered: str, entities: dict[str, Any], state: ConversationState) -> bool:
        if self._looks_like_alternative_provider_request(lowered, entities, state):
            return False

        if entities.get("specialization") and not entities.get("doctor_name"):
            if any(word in lowered for word in AVAILABILITY_QUERY_WORDS | {"which", "what", "doctors"}):
                return False

        asks_profile = any(phrase in lowered for phrase in DOCTOR_PROFILE_PHRASES)
        asks_schedule = any(word in lowered for word in DOCTOR_INFO_WORDS)
        has_doctor_context = bool(state.get("slots", {}).get("doctor_name") or state.get("doctor_profile"))
        explicit_doctor_reference = bool(
            entities.get("doctor_name")
            or re.search(r"\bdr(?:\.|\s)?[a-z]", lowered)
            or "this doctor" in lowered
            or "that doctor" in lowered
            or "same doctor" in lowered
        )
        return (explicit_doctor_reference or has_doctor_context) and (asks_profile or asks_schedule)

    def _looks_like_recommendation(self, lowered: str, entities: dict[str, Any], state: ConversationState) -> bool:
        if self._looks_like_alternative_provider_request(lowered, entities, state):
            return True
        has_provider_context = bool(
            state.get("slots", {}).get("specialization")
            or state.get("candidate_doctors")
            or state.get("slots", {}).get("doctor_name")
            or state.get("doctor_profile")
        )
        if has_provider_context and self._contains_any_token(lowered, AVAILABILITY_QUERY_WORDS):
            if "doctor" in lowered or "available" in lowered or "schedule" in lowered:
                return True
        if entities.get("symptom") or entities.get("specialization"):
            return True
        if "which doctor" in lowered or "what doctor" in lowered or "specialist" in lowered:
            return True
        if "recommend" in lowered and "doctor" in lowered:
            return True
        return any(word in lowered for word in SYMPTOM_HINT_WORDS)

    def _looks_like_faq(self, lowered: str) -> bool:
        return any(word in lowered for word in FAQ_WORDS)

    def _looks_like_topic_switch(self, lowered: str, entities: dict[str, Any]) -> bool:
        return (
            self._looks_like_booking(lowered, entities)
            or self._looks_like_doctor_info(lowered, entities, {})
            or self._looks_like_faq(lowered)
            or self._contains_any_token(lowered, HANDOFF_WORDS)
        )

    def _looks_like_alternative_provider_request(
        self,
        lowered: str,
        entities: dict[str, Any],
        state: ConversationState,
    ) -> bool:
        asks_alternative = any(word in lowered for word in ALTERNATIVE_PROVIDER_WORDS)
        asks_availability = any(word in lowered for word in AVAILABILITY_QUERY_WORDS) or bool(
            entities.get("date") or entities.get("time")
        )
        if "another doctor" in lowered or "other doctor" in lowered or "any other doctor" in lowered:
            asks_alternative = True
        has_context = bool(
            entities.get("specialization")
            or state.get("slots", {}).get("specialization")
            or state.get("candidate_doctors")
            or state.get("doctor_profile")
            or state.get("slots", {}).get("doctor_name")
        )
        return has_context and (asks_alternative or ("doctor" in lowered and asks_availability))

    @staticmethod
    def _llm_backup(message: str, state: ConversationState) -> IntentDecision | None:
        history = list(state.get("history", []) or [])
        recent_turns = history[-6:]
        history_text = "\n".join(
            f"- {item.get('role', 'unknown')}: {str(item.get('content', '')).strip()}"
            for item in recent_turns
            if str(item.get("content", "")).strip()
        ) or "- no prior turns"
        result = llm.complete_json(
            f"""
Classify the user's latest healthcare concierge message.

You MUST read the conversation context first, not just the latest message.
You MUST detect whether the user is continuing the current topic or changing topics.
If the user refers to "the doctor", "this doctor", "she", "he", "same doctor", or asks about availability/schedule,
resolve that against the active doctor or booking context in the state.
If the user changes topics, choose the new intent instead of staying on the old one.

Allowed intents:
- greeting
- faq
- doctor_info
- recommend_doctor
- book_appointment
- triage
- handoff
- safety
- clarify

Return strict JSON only like {{"intent":"faq","confidence":0.71}}.

Conversation state:
- booking_status: {state.get("booking_state", {}).get("status", "idle")}
- triage_status: {state.get("triage_state", {}).get("status", "idle")}
- slots: {state.get("slots", {})}
- extracted_entities: {state.get("current_entities", {})}
- candidate_doctors: {state.get("candidate_doctors", [])}
- active_doctor_profile: {state.get("doctor_profile")}

Recent conversation:
{history_text}

User message:
{message}
""",
            model=settings.classifier_model,
            fallback={},
        )
        intent = str(result.get("intent") or "").strip()
        if not intent:
            return None
        confidence = result.get("confidence")
        try:
            parsed_confidence = float(confidence)
        except (TypeError, ValueError):
            parsed_confidence = 0.55
        return IntentDecision(intent=intent, confidence=max(parsed_confidence, 0.55), used_llm=True)


class MedicalConversationDirector:
    def __init__(self, tools: CustomerServiceTools, domain, knowledge_base: KnowledgeBase) -> None:
        self.tools = tools
        self.domain = domain
        self.knowledge_base = knowledge_base
        self.intent_classifier = HybridIntentClassifier(tools)

    def contextualize_turn(self, state: ConversationState) -> dict[str, Any]:
        message = str(state.get("current_message", "")).strip()
        channel = str(state.get("channel", "chat"))
        slots = dict(state.get("slots", {}) or {})
        booking_state = dict(state.get("booking_state", {}) or {})
        triage_state = dict(state.get("triage_state", {}) or {})
        candidate_doctors = list(state.get("candidate_doctors", []) or [])

        if booking_state.get("status") == "completed":
            booking_state = {
                "status": "idle",
                "patient_profile": booking_state.get("patient_profile"),
                "appointment": booking_state.get("appointment"),
            }

        if any(phrase in message.lower() for phrase in RESET_PHRASES):
            booking_state = {
                "status": "idle",
                "patient_profile": booking_state.get("patient_profile"),
            }
            triage_state = {"status": "idle"}
            candidate_doctors = []

        current_entities = self._extract_entities(message, state)
        merged_slots = self._merge_slots(slots, current_entities)

        doctor_name = current_entities.get("doctor_name")
        current_profile = state.get("doctor_profile")
        explicit_specialization = current_entities.get("specialization")
        specialization_changed = bool(
            explicit_specialization
            and slots.get("specialization")
            and self._normalize_doctor_key(str(explicit_specialization)) != self._normalize_doctor_key(str(slots.get("specialization")))
        )
        doctor_changed = bool(
            doctor_name
            and slots.get("doctor_name")
            and self._normalize_doctor_key(str(doctor_name)) != self._normalize_doctor_key(str(slots.get("doctor_name")))
        )
        if specialization_changed or doctor_changed:
            candidate_doctors = []
            current_profile = None
            for key in ("symptom", "doctor_name", "doctor_id", "date", "time"):
                merged_slots.pop(key, None)
            if explicit_specialization:
                merged_slots["specialization"] = explicit_specialization
            if doctor_name:
                merged_slots["doctor_name"] = doctor_name
            if current_entities.get("doctor_id"):
                merged_slots["doctor_id"] = current_entities.get("doctor_id")
        if doctor_name and current_profile and doctor_name.lower() != str(current_profile.get("Name", "")).lower():
            current_profile = None

        return {
            "history": [
                {
                    "role": "user",
                    "content": message,
                    "channel": channel,
                    "timestamp": datetime.now(PKT).isoformat(),
                }
            ],
            "slots": merged_slots,
            "current_entities": current_entities,
            "reply": "",
            "action": "",
            "metadata": {},
            "next_agent": "finalize_turn",
            "booking_state": booking_state or {"status": "idle"},
            "triage_state": triage_state or {"status": "idle"},
            "candidate_doctors": candidate_doctors,
            "doctor_profile": current_profile,
            "safety_flags": self._detect_safety_flags(message),
        }

    def classify_intent_node(self, state: ConversationState) -> dict[str, Any]:
        decision = self.intent_classifier.classify(state)
        metadata = {
            "intent_confidence": round(decision.confidence, 2),
            "intent_used_llm": decision.used_llm,
        }
        return {
            "intent": decision.intent,
            "intent_confidence": decision.confidence,
            "metadata": metadata,
            "next_agent": "finalize_turn",
        }

    def route_intent(self, state: ConversationState) -> str:
        mapping = {
            "greeting": "greeting_agent",
            "faq": "faq_agent",
            "doctor_info": "doctor_info_agent",
            "recommend_doctor": "recommend_doctor_agent",
            "book_appointment": "booking_agent",
            "triage": "triage_agent",
            "handoff": "handoff_agent",
            "safety": "safety_agent",
            "clarify": "clarify_agent",
        }
        return mapping.get(str(state.get("intent", "clarify")), "clarify_agent")

    @staticmethod
    def route_after_agent(state: ConversationState) -> str:
        next_agent = str(state.get("next_agent", "finalize_turn"))
        if next_agent in {"recommend_doctor_agent", "booking_agent", "triage_agent"}:
            return next_agent
        return "finalize_turn"

    def greeting_agent(self, state: ConversationState) -> dict[str, Any]:
        return self._reply(
            state,
            reply="I can help with doctor matching, schedules, booking, and a short intake. Tell me what you need.",
            action="greeting",
        )

    def faq_agent(self, state: ConversationState) -> dict[str, Any]:
        query = str(state.get("current_message", "")).strip()
        lowered_query = query.lower()
        current_entities = dict(state.get("current_entities", {}) or {})
        slots = dict(state.get("slots", {}) or {})
        has_doctor_context = bool(slots.get("doctor_name") or state.get("doctor_profile"))
        has_provider_context = bool(
            has_doctor_context
            or slots.get("specialization")
            or state.get("candidate_doctors")
        )
        if self._message_requests_availability(lowered_query, current_entities) or "schedule" in lowered_query:
            if has_doctor_context:
                return self.doctor_info_agent(state)
            if has_provider_context:
                return self.recommend_doctor_agent(state)

        tool_result = self.tools.call_tool("search_knowledge", {"query": query})
        tool_events = [self._tool_event("search_knowledge", {"query": query}, tool_result)]

        if not tool_result.ok:
            return self._reply(
                state,
                reply=self.domain.fallback,
                action="knowledge_error",
                tool_events=tool_events,
            )

        matches = tool_result.data.get("matches", [])
        if not matches:
            return self._reply(
                state,
                reply=self.domain.fallback,
                action="knowledge_fallback",
                tool_events=tool_events,
            )

        top_matches = matches[:2]
        context_text = "\n\n".join(f"{item['source']}:\n{item['content']}" for item in top_matches)
        reply = None
        if llm.enabled:
            history = list(state.get("history", []) or [])
            recent_turns = history[-4:]
            history_text = "\n".join(
                f"- {item.get('role', 'unknown')}: {str(item.get('content', '')).strip()}"
                for item in recent_turns
                if str(item.get("content", "")).strip()
            ) or "- no prior turns"
            reply = llm.complete(
                f"""
You are a warm hospital concierge assistant.
You MUST read the recent conversation context before answering.
You MUST notice when the user is continuing the current topic versus changing topics.
Answer the user using only the knowledge below.
If the knowledge does not answer the user's question, do not invent facts.
If the user is asking about a doctor-specific schedule or availability, do not answer with generic hospital FAQ.

Recent conversation:
{history_text}

Structured context:
- slots: {slots}
- extracted_entities: {current_entities}
- active_doctor_profile: {state.get("doctor_profile")}

Knowledge:
{context_text}

User:
{query}
""",
                model=settings.action_model,
                temperature=0.15,
                max_tokens=220,
            )
        if not reply:
            reply = top_matches[0]["content"]

        return self._reply(
            state,
            reply=reply,
            action="knowledge_lookup",
            metadata={"sources": [item["source"] for item in top_matches]},
            tool_events=tool_events,
        )

    def safety_agent(self, state: ConversationState) -> dict[str, Any]:
        flags = state.get("safety_flags", []) or []
        slots = dict(state.get("slots", {}) or {})
        specialization = slots.get("specialization")
        if "medication_request" in flags:
            reply = (
                "I can help you choose the right doctor, book an appointment, and collect intake details, "
                "but I can't prescribe medicines or advise on dosage. A licensed doctor should handle that."
            )
            if specialization:
                reply += f" Based on your symptoms, {specialization} would be the right specialty to consult."
            return self._reply(state, reply=reply, action="medical_boundary")

        reply = (
            "Some of what you described sounds urgent, so I should not guide this through chat. "
            "Please contact local emergency services or go to the nearest emergency department right away."
        )
        return self._reply(state, reply=reply, action="urgent_handoff")

    def handoff_agent(self, state: ConversationState) -> dict[str, Any]:
        return self._reply(
            state,
            reply=self.domain.escalation_message,
            action="handoff",
        )

    def clarify_agent(self, state: ConversationState) -> dict[str, Any]:
        return self._reply(
            state,
            reply="I can suggest the right specialist, share a doctor's schedule, or book an appointment. Tell me which one you need.",
            action="clarify",
        )

    def doctor_info_agent(self, state: ConversationState) -> dict[str, Any]:
        slots = dict(state.get("slots", {}) or {})
        current_entities = dict(state.get("current_entities", {}) or {})
        current_message = str(state.get("current_message", ""))
        lowered_message = current_message.lower()
        doctor_name = slots.get("doctor_name")
        doctor_id = slots.get("doctor_id")
        if not doctor_name and not doctor_id:
            return self._reply(
                state,
                reply="Please tell me the doctor's name, and I can share the profile and schedule.",
                action="doctor_info_missing_name",
            )

        arguments = {"doctor_name": doctor_name, "doctor_id": doctor_id}
        tool_result = self.tools.call_tool("get_doctor_profile", arguments)
        tool_events = [self._tool_event("get_doctor_profile", arguments, tool_result)]
        if not tool_result.ok:
            return self._reply(
                state,
                reply="I couldn't fetch that doctor's information right now, but I can still help you try another doctor or continue with booking.",
                action="doctor_info_error",
                tool_events=tool_events,
            )

        doctor = tool_result.data.get("doctor")
        schedule = tool_result.data.get("schedule", [])
        if not doctor:
            return self._reply(
                state,
                reply="I couldn't find that doctor yet. If you want, tell me the specialty or symptoms and I'll suggest the closest match.",
                action="doctor_info_not_found",
                tool_events=tool_events,
            )

        requested_dates = self._extract_requested_dates(current_message, current_entities)
        requested_time = current_entities.get("time")
        wants_specific_availability = self._message_requests_specific_availability(
            lowered_message,
            current_entities,
            requested_dates,
        )
        if wants_specific_availability:
            availability_summaries: list[str] = []
            lookup_dates = requested_dates or [current_entities.get("date") or "today"]
            for requested_date in lookup_dates:
                availability_args = {
                    "specialization": doctor.get("Specialization"),
                    "doctor_name": doctor.get("Name"),
                    "date": requested_date,
                    "time": requested_time,
                }
                availability_result = self.tools.call_tool("find_provider_availability", availability_args)
                tool_events.append(self._tool_event("find_provider_availability", availability_args, availability_result))
                if not availability_result.ok:
                    return self._reply(
                        state,
                        reply=self._format_datetime_error_reply(availability_result.error),
                        action="availability_error",
                        tool_events=tool_events,
                    )

                requested_date_label = self._format_requested_date_label(
                    requested_date,
                    availability_result.data.get("date"),
                )
                available_entries = availability_result.data.get("available", [])
                selected_entry = self._choose_available_doctor(available_entries, str(doctor.get("Name", ""))) if available_entries else None
                matching_slots = []
                if selected_entry:
                    matching_slots = list(selected_entry.get("display_slots") or [])
                    if not matching_slots and selected_entry.get("display_slot"):
                        matching_slots = [str(selected_entry.get("display_slot"))]

                if matching_slots:
                    slot_text = ", ".join(matching_slots)
                    if requested_time:
                        availability_summaries.append(
                            f"{requested_date_label}: available around {requested_time} during {slot_text}."
                        )
                    else:
                        availability_summaries.append(f"{requested_date_label}: {slot_text}.")
                else:
                    if requested_time:
                        availability_summaries.append(
                            f"{requested_date_label}: no consulting hours around {requested_time}."
                        )
                    else:
                        availability_summaries.append(f"{requested_date_label}: no published consulting hours.")

            if availability_summaries:
                availability_reply = (
                    f"Availability for {self._display_provider_name(str(doctor.get('Name', 'Unknown')))}: "
                    + " ".join(availability_summaries)
                )
            else:
                availability_reply = "I couldn't check the date-specific availability right now."
        else:
            availability_reply = None

        reply_parts = [self._format_doctor_profile(doctor)]
        if availability_reply:
            reply_parts.append(availability_reply)
        if schedule and not wants_specific_availability:
            reply_parts.append(f"Full weekly schedule: {self._format_schedule(schedule)}.")
        elif not schedule:
            reply_parts.append("I don't have a published weekly schedule for this doctor yet.")
        reply_parts.append("If you'd like, I can book an appointment with this doctor next.")

        updated_slots = dict(slots)
        updated_slots["doctor_name"] = doctor.get("Name")
        updated_slots["doctor_id"] = doctor.get("id")
        if doctor.get("Specialization"):
            updated_slots["specialization"] = doctor.get("Specialization")

        return {
            "reply": " ".join(reply_parts),
            "action": "doctor_info",
            "metadata": {"doctor_id": doctor.get("id")},
            "tool_events": tool_events,
            "doctor_profile": doctor,
            "slots": updated_slots,
            "next_agent": "finalize_turn",
        }

    def recommend_doctor_agent(self, state: ConversationState) -> dict[str, Any]:
        slots = dict(state.get("slots", {}) or {})
        current_entities = dict(state.get("current_entities", {}) or {})
        booking_state = dict(state.get("booking_state", {}) or {})
        lowered_message = str(state.get("current_message", "")).lower()
        symptom = current_entities.get("symptom") or slots.get("symptom")
        specialization = current_entities.get("specialization") or slots.get("specialization")
        doctor_name = current_entities.get("doctor_name") or slots.get("doctor_name")
        symptom_match = current_entities.get("symptom_match")
        explicit_specialization_requested = bool(current_entities.get("specialization"))
        symptom_guided_request = bool(symptom and not explicit_specialization_requested)
        wants_alternative = self._message_requests_alternative_provider(lowered_message)
        wants_availability = self._message_requests_availability(lowered_message, current_entities)

        if not specialization and symptom_match:
            specialization = symptom_match.get("specialization")

        if not specialization and not doctor_name:
            return self._reply(
                state,
                reply=(
                    "Tell me the symptoms or the kind of specialist you're looking for, and I'll narrow down the right doctors for you."
                ),
                action="recommendation_clarify",
            )

        lookup_doctor_name = None if wants_alternative else doctor_name
        arguments = {
            "specialization": specialization,
            "symptom": symptom,
            "doctor_name": lookup_doctor_name,
        }
        tool_result = self.tools.call_tool("recommend_service_provider", arguments)
        tool_events = [self._tool_event("recommend_service_provider", arguments, tool_result)]

        if not tool_result.ok:
            reply = "I couldn't complete the doctor lookup right now."
            if specialization:
                if symptom_guided_request:
                    reply += f" Based on the symptoms, {specialization} is still the right department to check."
                else:
                    reply += f" I understood that you're looking for a {specialization}."
            reply += " If you'd like, we can keep going and I can collect booking details for a human follow-up."
            return self._reply(
                state,
                reply=reply,
                action="recommendation_error",
                tool_events=tool_events,
            )

        providers = tool_result.data.get("providers", [])
        resolved_specialization = tool_result.data.get("specialization") or specialization
        resolved_match = tool_result.data.get("symptom_match") or symptom_match or {}
        supported_specializations = tool_result.data.get("supported_specializations") or self.tools.get_supported_specializations()
        requested_specialization_supported = bool(
            tool_result.data.get("requested_specialization_supported", True)
        )
        updated_slots = dict(slots)
        if resolved_specialization:
            updated_slots["specialization"] = resolved_specialization
        if symptom:
            updated_slots["symptom"] = symptom

        providers = self._filter_provider_alternatives(
            providers,
            current_doctor_name=slots.get("doctor_name"),
            wants_alternative=wants_alternative,
        )

        if not providers:
            supported_text = self._supported_departments_text(supported_specializations)
            if resolved_specialization and not requested_specialization_supported:
                reply = f"I can't auto-book {resolved_specialization} yet. I currently support {supported_text}."
                if symptom_guided_request:
                    reply = f"That symptom route is not in the live roster yet. I currently support {supported_text}."
                return {
                    "reply": reply,
                    "action": "recommendation_unsupported_specialty",
                    "metadata": {
                        "specialization": resolved_specialization,
                        "supported_specializations": supported_specializations,
                    },
                    "tool_events": tool_events,
                    "slots": updated_slots,
                    "candidate_doctors": [],
                    "next_agent": "finalize_turn",
                }
            if resolved_specialization:
                if symptom_guided_request:
                    reply = (
                        f"Based on what you described, {resolved_specialization} looks like the right specialty, "
                        f"but I couldn't find any {resolved_specialization} doctors in the current database."
                    )
                else:
                    reply = (
                        f"I understood that you're looking for a {resolved_specialization}, "
                        f"but I couldn't find any {resolved_specialization} doctors in the current database."
                    )
                reply += " If you'd like, I can help with a related specialty or collect details for manual follow-up."
            else:
                reply = "I need a little more symptom detail before I can suggest the right specialist."
            return {
                "reply": reply,
                "action": "recommendation_specialty_only",
                "metadata": {"specialization": resolved_specialization},
                "tool_events": tool_events,
                "slots": updated_slots,
                "candidate_doctors": [],
                "next_agent": "finalize_turn",
            }

        if wants_availability:
            availability_args = {
                "specialization": resolved_specialization,
                "doctor_name": None if wants_alternative else doctor_name,
                "date": current_entities.get("date") or "today",
                "time": current_entities.get("time"),
            }
            availability_result = self.tools.call_tool("find_provider_availability", availability_args)
            tool_events.append(self._tool_event("find_provider_availability", availability_args, availability_result))
            if not availability_result.ok:
                return self._reply(
                    state,
                    reply=self._format_datetime_error_reply(availability_result.error),
                    action="availability_error",
                    tool_events=tool_events,
                )

            available_entries = availability_result.data.get("available", [])
            filtered_entries = self._filter_available_alternatives(
                available_entries,
                current_doctor_name=slots.get("doctor_name"),
                wants_alternative=wants_alternative,
            )
            requested_date = availability_result.data.get("date")
            if not filtered_entries:
                specialty_label = resolved_specialization or "doctors"
                if wants_alternative and slots.get("doctor_name"):
                    reply = (
                        f"I couldn't find another {specialty_label} option available on {requested_date}. "
                        "If you'd like, I can check tomorrow or a different time."
                    )
                else:
                    reply = (
                        f"I couldn't find an available {specialty_label} slot on {requested_date}. "
                        "If you'd like, I can check another date or time."
                    )
                return {
                    "reply": reply,
                    "action": "availability_empty",
                    "metadata": {"specialization": resolved_specialization, "date": requested_date},
                    "tool_events": tool_events,
                    "slots": updated_slots,
                    "candidate_doctors": [],
                    "next_agent": "finalize_turn",
                }

            visible_entries = filtered_entries[:4]
            candidate_doctors = [entry.get("doctor", {}) for entry in filtered_entries[:5]]
            provider_lines = [self._format_available_provider_line(entry) for entry in visible_entries]
            if wants_alternative and slots.get("doctor_name"):
                intro = f"Here are other {resolved_specialization or 'doctor'} options available on {requested_date}:"
            else:
                intro = f"Here are doctors available on {requested_date}:"
            return {
                "reply": "\n".join([intro, *provider_lines]),
                "action": "availability_lookup",
                "metadata": {"specialization": resolved_specialization, "date": requested_date},
                "tool_events": tool_events,
                "slots": updated_slots,
                "candidate_doctors": candidate_doctors,
                "next_agent": "finalize_turn",
            }

        visible_providers = providers[:4]
        provider_lines = [self._format_provider_line(item) for item in visible_providers]
        match_phrases = resolved_match.get("matches", []) if symptom_guided_request else []
        intro = "Here are doctors who fit your request:"
        if resolved_specialization:
            if symptom_guided_request:
                intro = f"Based on the symptoms, {resolved_specialization} is the right specialty. Here are suitable doctors:"
            else:
                intro = f"Here are {resolved_specialization} doctors I found:"
        if booking_state.get("status") == "needs_provider":
            intro += " Pick one and I'll continue the booking."
        else:
            intro += " Tell me who you'd like, and I can book the appointment."

        next_agent = "finalize_turn"
        if booking_state.get("status") == "needs_provider" and len(visible_providers) == 1:
            only_doctor = visible_providers[0]
            updated_slots["doctor_name"] = only_doctor.get("Name")
            updated_slots["doctor_id"] = only_doctor.get("id")
            next_agent = "booking_agent"

        return {
            "reply": "\n".join([intro, *provider_lines]),
            "action": "recommend_doctor",
            "metadata": {
                "specialization": resolved_specialization,
                "matched_terms": match_phrases,
            },
            "tool_events": tool_events,
            "candidate_doctors": providers[:5],
            "slots": updated_slots,
            "next_agent": next_agent,
        }

    def booking_agent(self, state: ConversationState) -> dict[str, Any]:
        slots = dict(state.get("slots", {}) or {})
        booking_state = dict(state.get("booking_state", {}) or {})
        current_message = str(state.get("current_message", "")).strip()
        current_status = str(booking_state.get("status") or "idle")
        tool_events: list[dict[str, Any]] = []

        if current_status == "completed":
            booking_state = {
                "status": "idle",
                "patient_profile": booking_state.get("patient_profile"),
            }

        if not slots.get("doctor_name") and not slots.get("specialization"):
            return self._reply(
                state,
                reply="Tell me the doctor's name or the symptoms first, and I'll line up the right appointment.",
                action="booking_missing_target",
            )

        if not slots.get("doctor_name"):
            booking_state["status"] = "needs_provider"
            return {
                "booking_state": booking_state,
                "reply": "",
                "action": "booking_delegate_recommendation",
                "next_agent": "recommend_doctor_agent",
            }

        phone = slots.get("phone")
        if not phone:
            booking_state["status"] = "collect_phone"
            return {
                "booking_state": booking_state,
                "reply": "Please share your mobile number to continue.",
                "action": "collect_phone",
                "next_agent": "finalize_turn",
            }

        patient_profile = booking_state.get("patient_profile")
        if not patient_profile:
            lookup = self.tools.call_tool("lookup_customer_profile", {"phone": phone})
            tool_events.append(self._tool_event("lookup_customer_profile", {"phone": phone}, lookup))
            if lookup.ok:
                patient_profile = lookup.data.get("profile")

            if not patient_profile:
                if current_status == "collect_name" and self._looks_like_full_name(current_message):
                    create_args = {"name": current_message, "phone": phone}
                    created = self.tools.call_tool("register_customer_profile", create_args)
                    tool_events.append(self._tool_event("register_customer_profile", create_args, created))
                    if not created.ok:
                        return self._reply(
                            state,
                            reply="I couldn't create the patient profile right now. Please try again in a moment.",
                            action="profile_error",
                            tool_events=tool_events,
                        )
                    patient_profile = created.data.get("profile")
                else:
                    booking_state["status"] = "collect_name"
                    return {
                        "booking_state": booking_state,
                        "reply": "I couldn't find a patient profile for that number. Please send your full name.",
                        "action": "collect_name",
                        "tool_events": tool_events,
                        "next_agent": "finalize_turn",
                    }

        booking_state["patient_profile"] = patient_profile

        if not slots.get("date"):
            booking_state["status"] = "collect_details"
            return {
                "booking_state": booking_state,
                "reply": "What date would you like? You can say today, tomorrow, or use YYYY-MM-DD.",
                "action": "collect_date",
                "tool_events": tool_events,
                "next_agent": "finalize_turn",
            }

        if not slots.get("time"):
            booking_state["status"] = "collect_details"
            date_label = self._humanize_date_reference(slots.get("date"))
            return {
                "booking_state": booking_state,
                "reply": f"{date_label} noted. What time works best? For example, 14:30 or 2:30 PM.",
                "action": "collect_time",
                "tool_events": tool_events,
                "next_agent": "finalize_turn",
            }

        availability_args = {
            "specialization": slots.get("specialization"),
            "doctor_name": slots.get("doctor_name"),
            "date": slots.get("date"),
            "time": slots.get("time"),
        }
        availability = self.tools.call_tool("find_provider_availability", availability_args)
        tool_events.append(self._tool_event("find_provider_availability", availability_args, availability))
        if not availability.ok:
            booking_state["status"] = "collect_details"
            return {
                "booking_state": booking_state,
                "reply": self._format_datetime_error_reply(availability.error),
                "action": "availability_error",
                "tool_events": tool_events,
                "next_agent": "finalize_turn",
            }

        available = availability.data.get("available", [])
        if not available:
            alternative_args = dict(availability_args)
            alternative_args["time"] = None
            alternatives = self.tools.call_tool("find_provider_availability", alternative_args)
            tool_events.append(self._tool_event("find_provider_availability", alternative_args, alternatives))
            alternative_text = self._format_alternative_slots_reply(alternatives.data.get("available", []))
            return self._reply(
                state,
                reply=(
                    f"I couldn't find an open slot for {slots.get('doctor_name')} on {availability.data.get('date')} at {slots.get('time')}. "
                    f"{alternative_text}"
                ),
                action="availability_empty",
                tool_events=tool_events,
            )

        chosen = self._choose_available_doctor(available, slots.get("doctor_name"))
        create_args = {
            "patient_id": patient_profile["id"],
            "doctor_id": chosen["doctor"]["id"],
            "date": availability.data["date"],
            "time": slots["time"],
        }
        created = self.tools.call_tool("create_booking", create_args)
        tool_events.append(self._tool_event("create_booking", create_args, created))
        if not created.ok:
            return self._reply(
                state,
                reply="I couldn't create the booking right now. Please try again in a moment.",
                action="booking_error",
                tool_events=tool_events,
            )

        appointment = created.data.get("appointment") or {}
        error_code = created.data.get("error_code")
        if error_code or not appointment:
            booking_state["status"] = "collect_details"
            return {
                "booking_state": booking_state,
                "reply": self._format_booking_error_reply(created.data),
                "action": "booking_error",
                "tool_events": tool_events,
                "next_agent": "finalize_turn",
            }
        booking_state.update(
            {
                "status": "completed",
                "appointment": appointment,
                "patient_profile": patient_profile,
            }
        )

        updated_slots = dict(slots)
        updated_slots["doctor_name"] = chosen["doctor"].get("Name")
        updated_slots["doctor_id"] = chosen["doctor"].get("id")
        if chosen["doctor"].get("Specialization"):
            updated_slots["specialization"] = chosen["doctor"].get("Specialization")

        triage_state = {
            "status": "pending_start",
            "appointment_id": appointment.get("id"),
            "doctor_name": chosen["doctor"].get("Name"),
            "specialization": chosen["doctor"].get("Specialization"),
            "answers": [],
            "intro": (
                f"Your appointment is confirmed with {self._display_provider_name(chosen['doctor'].get('Name', 'the doctor'))} "
                f"on {availability.data['date']} at {slots['time']}."
            ),
        }
        return {
            "booking_state": booking_state,
            "triage_state": triage_state,
            "slots": updated_slots,
            "metadata": {"appointment_id": appointment.get("id")},
            "tool_events": tool_events,
            "reply": "",
            "action": "booking_created",
            "next_agent": "triage_agent",
        }

    def triage_agent(self, state: ConversationState) -> dict[str, Any]:
        triage_state = dict(state.get("triage_state", {}) or {})
        slots = dict(state.get("slots", {}) or {})
        current_message = str(state.get("current_message", "")).strip()
        tool_events: list[dict[str, Any]] = []

        if triage_state.get("status") in {None, "idle", "pending_start"}:
            flow_args = {
                "symptom": slots.get("symptom"),
                "specialization": triage_state.get("specialization") or slots.get("specialization"),
            }
            flow = self.tools.call_tool("get_triage_flow", flow_args)
            tool_events.append(self._tool_event("get_triage_flow", flow_args, flow))
            if not flow.ok:
                return self._reply(
                    state,
                    reply="I couldn't load the intake questions right now, but your booking is still in place.",
                    action="triage_error",
                    tool_events=tool_events,
                )

            questions = flow.data.get("questions", [])
            first_question = questions[0] if questions else "Can you briefly describe the main issue for the doctor?"
            triage_state.update(
                {
                    "status": "collecting",
                    "flow_name": flow.data.get("flow_name"),
                    "questions": questions or [first_question],
                    "red_flags": flow.data.get("red_flags", []),
                    "current_index": 0,
                    "answers": [],
                }
            )
            intro = triage_state.pop("intro", None) or "To help the doctor prepare, I'd like to ask a few quick intake questions."
            return {
                "triage_state": triage_state,
                "reply": f"{intro} {first_question}",
                "action": "triage_question",
                "tool_events": tool_events,
                "next_agent": "finalize_turn",
            }

        if triage_state.get("status") != "collecting":
            return self._reply(
                state,
                reply="There isn't an active triage flow right now, but I can start one if you need it.",
                action="triage_idle",
            )

        questions = list(triage_state.get("questions", []) or [])
        current_index = int(triage_state.get("current_index", 0))
        answers = list(triage_state.get("answers", []) or [])
        if questions:
            previous_question = questions[current_index]
            answers.append({"question": previous_question, "answer": current_message})
            triage_state["answers"] = answers

        if self._matches_red_flag(current_message, triage_state.get("red_flags", [])):
            summary = self._build_triage_summary(answers)
            appointment_id = triage_state.get("appointment_id")
            if appointment_id:
                save_result = self.tools.call_tool("save_case_notes", {"appointment_id": appointment_id, "notes": summary})
                tool_events.append(self._tool_event("save_case_notes", {"appointment_id": appointment_id, "notes": summary}, save_result))
            triage_state.update({"status": "completed", "summary": summary, "red_flag_triggered": True})
            reply = (
                "Thank you. What you described could be urgent, so I can't assess it further here. "
                "Please contact emergency services or the nearest emergency department right away."
            )
            return {
                "triage_state": triage_state,
                "reply": reply,
                "action": "triage_red_flag",
                "metadata": {"triage_summary": summary},
                "tool_events": tool_events,
                "next_agent": "finalize_turn",
            }

        next_index = current_index + 1
        if next_index < len(questions):
            triage_state["current_index"] = next_index
            return {
                "triage_state": triage_state,
                "reply": questions[next_index],
                "action": "triage_question",
                "next_agent": "finalize_turn",
            }

        summary = self._build_triage_summary(answers)
        appointment_id = triage_state.get("appointment_id")
        if appointment_id:
            save_result = self.tools.call_tool("save_case_notes", {"appointment_id": appointment_id, "notes": summary})
            tool_events.append(self._tool_event("save_case_notes", {"appointment_id": appointment_id, "notes": summary}, save_result))
        triage_state.update({"status": "completed", "summary": summary})
        return {
            "triage_state": triage_state,
            "reply": "Thank you. I've prepared a short intake summary for the doctor, and your appointment is all set.",
            "action": "triage_complete",
            "metadata": {"triage_summary": summary},
            "tool_events": tool_events,
            "next_agent": "finalize_turn",
        }

    def finalize_turn(self, state: ConversationState) -> dict[str, Any]:
        reply = str(state.get("reply") or self.domain.fallback).strip()
        current_message = str(state.get("current_message", "")).strip()

        # --- NEW BILINGUAL AUTO-TRANSLATOR ---
        # If the user spoke Urdu, translate the hardcoded English reply into Urdu!
        if llm.enabled and current_message and reply:
            translated_reply = llm.complete(
                f"""
                You are a translation assistant for a hospital concierge.
                The system wants to say: "{reply}"
                The user just asked: "{current_message}"
                
                If the user is speaking Urdu or Roman Urdu, translate the system's message into native Urdu script. 
                If the user is speaking English, leave the system's message exactly as it is in English.
                Return ONLY the final text, no quotes or extra commentary.
                """,
                model=settings.action_model,
                temperature=0.1,
                max_tokens=250,
            )
            if translated_reply:
                reply = translated_reply
        # ---------------------------------------

        return {
            "history": [
                {
                    "role": "assistant",
                    "content": reply,
                    "channel": state.get("channel", "chat"),
                    "timestamp": datetime.now(PKT).isoformat(),
                }
            ],
            "last_intent": str(state.get("intent", "clarify")),
            "next_agent": "finalize_turn",
        }

    def _extract_entities(self, message: str, state: ConversationState) -> dict[str, Any]:
        entities: dict[str, Any] = {}
        lowered = message.lower()

        compact_message = re.sub(r"\s+", "", message)
        phone_match = PHONE_RE.search(compact_message)
        if phone_match:
            entities["phone"] = phone_match.group(0)

        email_match = EMAIL_RE.search(message)
        if email_match:
            entities["email"] = email_match.group(0)

        doctor_match = DOCTOR_RE.search(message)
        if doctor_match:
            cleaned_doctor_name = self._clean_doctor_name_candidate(doctor_match.group(1))
            if cleaned_doctor_name:
                entities["doctor_name"] = cleaned_doctor_name

        candidate = self._resolve_candidate_reference(message, state)
        if candidate:
            entities.setdefault("doctor_name", candidate.get("Name"))
            entities.setdefault("doctor_id", candidate.get("id"))
            if candidate.get("Specialization"):
                entities.setdefault("specialization", candidate.get("Specialization"))

        time_match = TIME_RE.search(message)
        if time_match:
            entities["time"] = time_match.group(1).strip()

        date_match = DATE_RE.search(message)
        if date_match:
            entities["date"] = date_match.group(0)
        else:
            relative_date = self._extract_relative_date(message)
            if relative_date:
                entities["date"] = relative_date

        explicit_specialization = self._extract_specialization(lowered)
        if explicit_specialization:
            entities["specialization"] = explicit_specialization

        is_symptom_message = self._looks_like_symptom_message(lowered)
        if is_symptom_message:
            symptom_match_result = self.tools.call_tool("match_symptoms_to_specialization", {"symptom": message})
            if symptom_match_result.ok and symptom_match_result.data.get("specialization"):
                entities["symptom_match"] = symptom_match_result.data
                entities.setdefault("specialization", symptom_match_result.data.get("specialization"))
                entities["symptom"] = message.strip()

        if not entities.get("symptom") and is_symptom_message:
            entities["symptom"] = message.strip()

        return entities

    @staticmethod
    def _merge_slots(existing: dict[str, Any], extracted: dict[str, Any]) -> dict[str, Any]:
        merged = dict(existing)
        for key in ("phone", "email", "doctor_name", "doctor_id", "specialization", "symptom", "date", "time"):
            value = extracted.get(key)
            if value:
                merged[key] = value
        return merged

    @staticmethod
    def _extract_relative_date(message: str) -> str | None:
        lowered = message.lower()
        if "day after tomorrow" in lowered:
            return "day after tomorrow"
        if "tomorrow" in lowered:
            return "tomorrow"
        if "today" in lowered:
            return "today"
        for phrase, canonical in sorted(WEEKDAY_ALIASES.items(), key=lambda item: -len(item[0])):
            if re.search(rf"\b{re.escape(phrase)}\b", lowered):
                return canonical
        return None

    def _extract_specialization(self, lowered: str) -> str | None:
        normalized = " ".join(WORD_RE.findall(lowered))
        token_set = set(normalized.split())

        rag_specializations = sorted(
            {specialization for _, specialization in self.tools.symptom_map if specialization},
            key=lambda item: (-len(WORD_RE.findall(item.lower())), -len(item)),
        )
        for specialization in rag_specializations:
            if self._message_contains_term(normalized, token_set, specialization):
                return specialization

        ordered_aliases = sorted(
            SPECIALIZATION_HINTS.items(),
            key=lambda item: (-len(WORD_RE.findall(item[0].lower())), -len(item[0])),
        )
        for phrase, specialization in ordered_aliases:
            if self._message_contains_term(normalized, token_set, phrase):
                return specialization
        for phrase, specialization in ordered_aliases:
            if self._message_fuzzy_contains_term(normalized.split(), phrase):
                return specialization
        return None

    @staticmethod
    def _message_contains_term(normalized: str, token_set: set[str], phrase: str) -> bool:
        phrase_tokens = WORD_RE.findall(phrase.lower())
        if not phrase_tokens:
            return False
        if len(phrase_tokens) == 1:
            return phrase_tokens[0] in token_set
        return " ".join(phrase_tokens) in normalized

    @staticmethod
    def _message_fuzzy_contains_term(message_tokens: list[str], phrase: str) -> bool:
        phrase_tokens = WORD_RE.findall(phrase.lower())
        if not phrase_tokens:
            return False
        if len(phrase_tokens) == 1:
            target = phrase_tokens[0]
            if len(target) < 6:
                return False
            return any(
                len(token) >= 6 and SequenceMatcher(None, token, target).ratio() >= 0.84
                for token in message_tokens
            )
        window = len(phrase_tokens)
        if len(message_tokens) < window:
            return False
        phrase_text = " ".join(phrase_tokens)
        for index in range(len(message_tokens) - window + 1):
            candidate = " ".join(message_tokens[index : index + window])
            if SequenceMatcher(None, candidate, phrase_text).ratio() >= 0.9:
                return True
        return False

    @staticmethod
    def _looks_like_symptom_message(lowered: str) -> bool:
        return (
            any(word in lowered for word in SYMPTOM_HINT_WORDS)
            or "i have" in lowered
            or "i am having" in lowered
            or "suffering" in lowered
        )

    @staticmethod
    def _detect_safety_flags(message: str) -> list[str]:
        lowered = message.lower()
        flags: list[str] = []
        if any(word in lowered for word in MEDICATION_WORDS):
            flags.append("medication_request")
        if any(phrase in lowered for phrase in URGENT_PHRASES):
            flags.append("urgent_symptom")
        return flags

    def _resolve_candidate_reference(self, message: str, state: ConversationState) -> dict[str, Any] | None:
        lowered = message.lower()
        message_tokens = set(WORD_RE.findall(lowered))
        candidates = list(state.get("candidate_doctors", []) or [])
        if not candidates:
            return None

        for token, index in ORDINAL_MAP.items():
            if re.search(rf"\b{re.escape(token)}\b", lowered) and index < len(candidates):
                return candidates[index]

        if any(phrase in lowered for phrase in ("that doctor", "that one", "this doctor", "this one")):
            doctor_name = state.get("slots", {}).get("doctor_name")
            if doctor_name:
                matched = self._find_doctor_by_name(candidates, str(doctor_name))
                if matched:
                    return matched
            if len(candidates) == 1:
                return candidates[0]

        for candidate in candidates:
            candidate_name = str(candidate.get("Name", "")).lower()
            tokens = [token for token in WORD_RE.findall(candidate_name) if token != "dr"]
            if tokens and set(tokens[-2:]).issubset(message_tokens):
                return candidate
        return None

    @staticmethod
    def _find_doctor_by_name(doctors: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
        lowered_name = MedicalConversationDirector._normalize_doctor_key(name)
        for doctor in doctors:
            if MedicalConversationDirector._normalize_doctor_key(str(doctor.get("Name", ""))) == lowered_name:
                return doctor
        return None

    @staticmethod
    def _looks_like_full_name(message: str) -> bool:
        return bool(NAMEISH_RE.match(message.strip()))

    @staticmethod
    def _clean_doctor_name_candidate(raw_name: str) -> str:
        tokens = [token for token in WORD_RE.findall(raw_name.lower()) if token]
        while tokens and tokens[-1] in DOCTOR_NAME_STOP_WORDS:
            tokens.pop()
        if not tokens:
            return ""
        return " ".join(token.capitalize() for token in tokens[:4])

    @staticmethod
    def _choose_available_doctor(available: list[dict[str, Any]], target_name: str | None) -> dict[str, Any]:
        if not target_name:
            return available[0]
        lowered_target = MedicalConversationDirector._normalize_doctor_key(target_name)
        for entry in available:
            doctor_name = MedicalConversationDirector._normalize_doctor_key(str(entry.get("doctor", {}).get("Name", "")))
            if lowered_target and (lowered_target in doctor_name or doctor_name in lowered_target):
                return entry
        return available[0]

    @staticmethod
    def _format_provider_line(provider: dict[str, Any]) -> str:
        name = str(provider.get("Name", "Unknown")).strip()
        specialization = str(provider.get("Specialization", "General")).strip()
        experience = provider.get("Experience")
        experience_text = MedicalConversationDirector._format_experience(experience)
        suffix = f", {experience_text}" if experience_text else ""
        return f"- {MedicalConversationDirector._display_provider_name(name)} ({specialization}{suffix})"

    @staticmethod
    def _format_available_provider_line(entry: dict[str, Any]) -> str:
        doctor = entry.get("doctor", {})
        name = MedicalConversationDirector._display_provider_name(str(doctor.get("Name", "Unknown")))
        specialization = str(doctor.get("Specialization", "General")).strip()
        experience_text = MedicalConversationDirector._format_experience(doctor.get("Experience"))
        display_slots = list(entry.get("display_slots") or [])
        if display_slots:
            visible_slots = [str(slot).strip() for slot in display_slots if str(slot).strip()][:4]
            more_count = max(int(entry.get("slot_count") or len(display_slots)) - len(visible_slots), 0)
            slot_text = ", ".join(visible_slots)
            if more_count:
                slot_text += f" (+{more_count} more)"
        else:
            slot_text = str(entry.get("display_slot") or "time available").strip()
        details: list[str] = [specialization]
        if experience_text:
            details.append(experience_text)
        details.append(f"Slots: {slot_text}")
        return f"- {name} ({', '.join(details)})"

    def _supported_departments_text(self, supported_specializations: list[str] | None = None) -> str:
        departments = supported_specializations or self.tools.get_supported_specializations()
        return ", ".join(departments)

    def _format_datetime_error_reply(self, error: str | None) -> str:
        lowered_error = str(error or "").lower()
        if "date must" in lowered_error:
            return "Please send the date as today, tomorrow, a weekday, or YYYY-MM-DD."
        if "time must" in lowered_error:
            return "Please send the time as HH:MM or with AM/PM."
        return "I couldn't check that date and time. Please try again."

    def _format_alternative_slots_reply(self, available_entries: list[dict[str, Any]]) -> str:
        if not available_entries:
            return "Send another time and I'll check again."
        chosen = available_entries[0]
        doctor_name = self._display_provider_name(str(chosen.get("doctor", {}).get("Name", "the doctor")))
        slots = list(chosen.get("display_slots") or [])
        visible = ", ".join(slots[:4]) if slots else str(chosen.get("display_slot") or "").strip()
        if not visible:
            return "Send another time and I'll check again."
        return f"Closest open times with {doctor_name} are {visible}. Send one of those if you'd like."

    def _format_booking_error_reply(self, booking_result: dict[str, Any]) -> str:
        error_code = str(booking_result.get("error_code") or "").strip().lower()
        if error_code == "invalid_datetime":
            return self._format_datetime_error_reply(booking_result.get("message"))
        if error_code == "schedule_unavailable":
            return "That doctor does not have a loaded schedule for the requested day yet."
        if error_code == "slot_unavailable":
            return "That time is no longer available. Send another time and I'll recheck."
        return "I couldn't complete the booking just now. Please try again."

    @staticmethod
    def _message_requests_availability(lowered: str, entities: dict[str, Any]) -> bool:
        return any(word in lowered for word in AVAILABILITY_QUERY_WORDS) or bool(
            entities.get("date") or entities.get("time")
        )

    @staticmethod
    def _message_requests_specific_availability(
        lowered: str,
        entities: dict[str, Any],
        requested_dates: list[str],
    ) -> bool:
        if entities.get("time") or requested_dates:
            return True
        return any(word in lowered for word in {"today", "tomorrow", "slot", "slots", "free", "open"})

    @staticmethod
    def _extract_requested_dates(message: str, entities: dict[str, Any]) -> list[str]:
        matches: list[tuple[int, str]] = []
        lowered = message.lower()
        masked = lowered
        for label in ("day after tomorrow", "today", "tomorrow"):
            pattern = re.compile(rf"\b{re.escape(label)}\b")
            for match in pattern.finditer(masked):
                matches.append((match.start(), label))
            masked = pattern.sub(lambda found: " " * len(found.group(0)), masked)

        for phrase, canonical in sorted(WEEKDAY_ALIASES.items(), key=lambda item: -len(item[0])):
            pattern = re.compile(rf"\b{re.escape(phrase)}\b")
            for match in pattern.finditer(masked):
                matches.append((match.start(), canonical))
            masked = pattern.sub(lambda found: " " * len(found.group(0)), masked)

        for match in DATE_RE.finditer(message):
            matches.append((match.start(), match.group(0)))

        if not matches and entities.get("date"):
            return [str(entities.get("date"))]

        ordered: list[str] = []
        seen: set[str] = set()
        for _, label in sorted(matches, key=lambda item: item[0]):
            if label not in seen:
                ordered.append(label)
                seen.add(label)
        return ordered

    @staticmethod
    def _format_requested_date_label(raw_date: str | None, resolved_date: str | None) -> str:
        cleaned = (raw_date or "").strip()
        if not cleaned:
            return resolved_date or "the requested date"

        lowered = cleaned.lower()
        if resolved_date and lowered in {"today", "tomorrow", "day after tomorrow", *WEEKDAY_ORDER}:
            return f"{MedicalConversationDirector._humanize_date_reference(lowered)} ({resolved_date})"
        return resolved_date or cleaned

    @staticmethod
    def _humanize_date_reference(date_value: str | None) -> str:
        cleaned = str(date_value or "").strip()
        if not cleaned:
            return "that date"
        lowered = cleaned.lower()
        if lowered in WEEKDAY_ALIASES:
            return WEEKDAY_ALIASES[lowered].capitalize()
        if lowered in {"today", "tomorrow", "day after tomorrow"}:
            return lowered.capitalize()
        return cleaned

    @staticmethod
    def _message_requests_alternative_provider(lowered: str) -> bool:
        if "another doctor" in lowered or "other doctor" in lowered or "any other doctor" in lowered:
            return True
        return any(word in lowered for word in ALTERNATIVE_PROVIDER_WORDS)

    @staticmethod
    def _filter_provider_alternatives(
        providers: list[dict[str, Any]],
        *,
        current_doctor_name: str | None,
        wants_alternative: bool,
    ) -> list[dict[str, Any]]:
        if not wants_alternative or not current_doctor_name:
            return providers
        current_key = MedicalConversationDirector._normalize_doctor_key(current_doctor_name)
        return [
            provider
            for provider in providers
            if MedicalConversationDirector._normalize_doctor_key(str(provider.get("Name", ""))) != current_key
        ]

    @staticmethod
    def _filter_available_alternatives(
        available_entries: list[dict[str, Any]],
        *,
        current_doctor_name: str | None,
        wants_alternative: bool,
    ) -> list[dict[str, Any]]:
        if not wants_alternative or not current_doctor_name:
            return available_entries
        current_key = MedicalConversationDirector._normalize_doctor_key(current_doctor_name)
        return [
            entry
            for entry in available_entries
            if MedicalConversationDirector._normalize_doctor_key(str(entry.get("doctor", {}).get("Name", ""))) != current_key
        ]

    @staticmethod
    def _normalize_doctor_key(name: str) -> str:
        tokens = [token for token in WORD_RE.findall(name.lower()) if token != "dr"]
        return " ".join(tokens)

    @staticmethod
    def _format_experience(experience: Any) -> str:
        if experience in (None, ""):
            return ""
        raw = str(experience).strip()
        lowered = raw.lower()
        if "experience" in lowered:
            return raw
        if "year" in lowered:
            return f"{raw} experience"
        if raw.isdigit():
            return f"{raw} years experience"
        return raw

    @staticmethod
    def _format_doctor_profile(doctor: dict[str, Any]) -> str:
        name = MedicalConversationDirector._display_provider_name(str(doctor.get("Name", "Unknown")))
        specialization = doctor.get("Specialization") or "General"
        experience = doctor.get("Experience")
        experience_text = MedicalConversationDirector._format_experience(experience)
        if experience_text:
            return f"{name} is a {specialization} specialist with {experience_text}."
        return f"{name} is a {specialization} specialist."

    @staticmethod
    def _format_schedule(schedule: list[dict[str, Any]]) -> str:
        formatted = []
        for slot in schedule:
            days = MedicalConversationDirector._format_schedule_days(str(slot.get("days", "")))
            start_time = slot.get("start_time", "")
            end_time = slot.get("end_time", "")
            if days and start_time and end_time:
                formatted.append(f"{days} {start_time}-{end_time}")
        return "; ".join(formatted) if formatted else "schedule unavailable"

    @staticmethod
    def _format_schedule_days(days: str) -> str:
        return re.sub(r"\b[a-z]{3,9}\b", lambda match: match.group(0).capitalize(), days.strip().lower())

    @staticmethod
    def _display_provider_name(name: str) -> str:
        cleaned = re.sub(r"^dr\.?\s*", "", name.strip(), flags=re.IGNORECASE)
        return f"Dr. {cleaned}"

    def _matches_red_flag(self, answer: str, red_flags: list[str]) -> bool:
        lowered = answer.lower()
        if any(phrase in lowered for phrase in URGENT_PHRASES):
            return True
        normalized_answer = " ".join(WORD_RE.findall(lowered))
        for red_flag in red_flags:
            normalized_flag = " ".join(WORD_RE.findall(red_flag.lower()))
            if normalized_flag and normalized_flag in normalized_answer:
                return True
        return False

    def _build_triage_summary(self, answers: list[dict[str, str]]) -> str:
        raw_summary = "\n".join(f"Q: {item['question']}\nA: {item['answer']}" for item in answers)
        if llm.enabled and raw_summary:
            reply = llm.complete(
                f"""
Summarize these patient intake notes for a doctor in 5 short lines.
Do not diagnose. Do not prescribe. Keep it factual.

{raw_summary}
""",
                model=settings.triage_model,
                temperature=0.1,
                max_tokens=180,
            )
            if reply:
                return reply
        return raw_summary

    @staticmethod
    def _tool_event(name: str, arguments: dict[str, Any], result: ToolCallResult) -> dict[str, Any]:
        event = {
            "name": name,
            "arguments": arguments,
            "ok": result.ok,
            "timestamp": datetime.now(PKT).isoformat(),
        }
        if result.ok:
            event["result_keys"] = sorted(result.data.keys())
        else:
            event["error"] = result.error
        return event

    @staticmethod
    def _reply(
        state: ConversationState,
        *,
        reply: str,
        action: str,
        metadata: dict[str, Any] | None = None,
        tool_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return {
            "reply": reply.strip(),
            "action": action,
            "metadata": metadata or {},
            "tool_events": tool_events or [],
            "next_agent": "finalize_turn",
        }


class CustomerServiceOrchestrator:
    def __init__(self) -> None:
        self.domain = get_domain_config(settings.app_domain)
        self.knowledge_base = KnowledgeBase(self.domain)
        self.tools = CustomerServiceTools(self.domain, self.knowledge_base)
        self.director = MedicalConversationDirector(self.tools, self.domain, self.knowledge_base)
        checkpointer = InMemorySaver() if InMemorySaver is not None else None
        self.graph = create_conversation_graph(self.director, checkpointer=checkpointer)

    def process(self, *, session_id: str, message: str, channel: str = "chat") -> OrchestratorResult:
        clean_message = (message or "").strip()
        if not clean_message:
            return OrchestratorResult(
                session_id=session_id,
                reply=self.domain.fallback,
                intent="clarify",
                action="empty_message",
                state=SessionState(session_id=session_id),
            )

        result_state = self.graph.invoke(
            {
                "session_id": session_id,
                "current_message": clean_message,
                "channel": channel,
            },
            config={"configurable": {"thread_id": session_id}},
        )
        session_state = self._to_session_state(session_id, result_state)
        return OrchestratorResult(
            session_id=session_id,
            reply=str(result_state.get("reply") or self.domain.fallback).strip(),
            intent=str(result_state.get("intent") or "clarify"),
            action=str(result_state.get("action") or "reply"),
            state=session_state,
            metadata=dict(result_state.get("metadata", {}) or {}),
        )

    @staticmethod
    def _to_session_state(session_id: str, state: ConversationState) -> SessionState:
        workflow = {
            "booking": dict(state.get("booking_state", {}) or {}),
            "triage": dict(state.get("triage_state", {}) or {}),
        }
        metadata = dict(state.get("metadata", {}) or {})
        if state.get("candidate_doctors"):
            metadata["candidate_doctors"] = state.get("candidate_doctors")
        if state.get("doctor_profile"):
            metadata["doctor_profile"] = state.get("doctor_profile")
        if state.get("tool_events"):
            metadata["tool_events"] = list(state.get("tool_events") or [])[-10:]
        return SessionState(
            session_id=session_id,
            history=list(state.get("history", []) or []),
            slots=dict(state.get("slots", {}) or {}),
            workflow=workflow,
            last_intent=str(state.get("intent") or "clarify"),
            metadata=metadata,
        )


orchestrator = CustomerServiceOrchestrator()
