from __future__ import annotations

import copy
import unittest
from dataclasses import dataclass
from datetime import datetime, timezone
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.domain import get_domain_config
from backend.graph import create_conversation_graph
from backend.knowledge import KnowledgeBase
from backend.mcp_tools import CustomerServiceTools, ToolCallResult
from backend.orchestrator import InMemorySaver, MedicalConversationDirector, OrchestratorResult, SessionState


@dataclass
class FakeResponse:
    data: list[dict]
    count: int | None = None


class FakeQuery:
    def __init__(self, database: "FakeSupabase", table_name: str) -> None:
        self.database = database
        self.table_name = table_name
        self.filters: list[tuple[str, str, object]] = []
        self.operation = "select"
        self.payload: object = None
        self.limit_value: int | None = None
        self.order_by_column: str | None = None
        self.order_desc = False
        self.count_mode: str | None = None

    def select(self, _columns: str = "*", count: str | None = None) -> "FakeQuery":
        self.operation = "select"
        self.count_mode = count
        return self

    def insert(self, payload: object) -> "FakeQuery":
        self.operation = "insert"
        self.payload = payload
        return self

    def update(self, payload: dict) -> "FakeQuery":
        self.operation = "update"
        self.payload = payload
        return self

    def eq(self, column: str, value: object) -> "FakeQuery":
        self.filters.append(("eq", column, value))
        return self

    def neq(self, column: str, value: object) -> "FakeQuery":
        self.filters.append(("neq", column, value))
        return self

    def ilike(self, column: str, value: object) -> "FakeQuery":
        self.filters.append(("ilike", column, value))
        return self

    def gte(self, column: str, value: object) -> "FakeQuery":
        self.filters.append(("gte", column, value))
        return self

    def lte(self, column: str, value: object) -> "FakeQuery":
        self.filters.append(("lte", column, value))
        return self

    def gt(self, column: str, value: object) -> "FakeQuery":
        self.filters.append(("gt", column, value))
        return self

    def lt(self, column: str, value: object) -> "FakeQuery":
        self.filters.append(("lt", column, value))
        return self

    def in_(self, column: str, values: list[object]) -> "FakeQuery":
        self.filters.append(("in", column, values))
        return self

    def limit(self, value: int) -> "FakeQuery":
        self.limit_value = value
        return self

    def order(self, column: str, desc: bool = False) -> "FakeQuery":
        self.order_by_column = column
        self.order_desc = desc
        return self

    def execute(self) -> FakeResponse:
        if self.operation == "select":
            rows = [copy.deepcopy(row) for row in self.database.tables[self.table_name] if self._matches(row)]
            total_count = len(rows)
            if self.order_by_column:
                rows.sort(key=lambda row: row.get(self.order_by_column), reverse=self.order_desc)
            if self.limit_value is not None:
                rows = rows[: self.limit_value]
            return FakeResponse(data=rows, count=total_count if self.count_mode else None)

        if self.operation == "insert":
            payloads = self.payload if isinstance(self.payload, list) else [self.payload]
            created: list[dict] = []
            for item in payloads:
                row = copy.deepcopy(item)
                if "id" not in row:
                    row["id"] = self.database.next_id(self.table_name)
                if "created_at" not in row:
                    row["created_at"] = datetime.now(timezone.utc).isoformat()
                self.database.tables[self.table_name].append(row)
                created.append(copy.deepcopy(row))
            return FakeResponse(data=created)

        if self.operation == "update":
            updated: list[dict] = []
            for row in self.database.tables[self.table_name]:
                if not self._matches(row):
                    continue
                row.update(copy.deepcopy(self.payload))
                updated.append(copy.deepcopy(row))
            return FakeResponse(data=updated)

        raise ValueError(f"Unsupported fake operation: {self.operation}")

    def _matches(self, row: dict) -> bool:
        for op, column, value in self.filters:
            row_value = row.get(column)
            if op == "eq" and row_value != value:
                return False
            if op == "neq" and row_value == value:
                return False
            if op == "ilike":
                needle = str(value).replace("%", "").lower()
                if needle not in str(row_value or "").lower():
                    return False
            if op == "gte" and row_value < value:
                return False
            if op == "lte" and row_value > value:
                return False
            if op == "gt" and row_value <= value:
                return False
            if op == "lt" and row_value >= value:
                return False
            if op == "in" and row_value not in value:
                return False
        return True


class FakeSupabase:
    def __init__(self, tables: dict[str, list[dict]]) -> None:
        self.tables = {name: copy.deepcopy(rows) for name, rows in tables.items()}

    def table(self, name: str) -> FakeQuery:
        self.tables.setdefault(name, [])
        return FakeQuery(self, name)

    def next_id(self, table_name: str) -> int:
        rows = self.tables.setdefault(table_name, [])
        if not rows:
            return 1
        return max(int(row.get("id", 0)) for row in rows) + 1


def build_fake_database() -> FakeSupabase:
    return FakeSupabase(
        {
            "doctors": [
                {"id": 1, "name": "Dr. Ahmed Khan", "specialization": "Cardiologist", "experience_years": 12, "consultation_fee": 3000.0},
                {"id": 2, "name": "Dr. Sara Ali", "specialization": "Dermatologist", "experience_years": 8, "consultation_fee": 2500.0},
                {"id": 3, "name": "Dr. Usman Raza", "specialization": "Neurologist", "experience_years": 15, "consultation_fee": 4000.0},
                {"id": 4, "name": "Dr. Hina Malik", "specialization": "Pediatrician", "experience_years": 10, "consultation_fee": 2000.0},
                {"id": 5, "name": "Dr. Bilal Sheikh", "specialization": "Orthopedic", "experience_years": 9, "consultation_fee": 2800.0},
                {"id": 6, "name": "Dr. Ayesha Noor", "specialization": "Gynecologist", "experience_years": 11, "consultation_fee": 3200.0},
            ],
            "doctor_availability": [
                {"id": 1, "doctor_id": 3, "day_of_week": 1, "start_time": "14:00:00", "end_time": "16:00:00", "slot_duration_minutes": 30},
                {"id": 2, "doctor_id": 1, "day_of_week": 1, "start_time": "09:00:00", "end_time": "11:00:00", "slot_duration_minutes": 30},
            ],
            "patients": [
                {"id": 1, "name": "Sara Patient", "phone": "03001234567", "email": "sara@example.com", "created_at": "2026-04-01T10:00:00+00:00"},
            ],
            "slots": [],
            "appointments": [],
            "appointment_events": [],
        }
    )


class TestOrchestrator:
    def __init__(self, tools: CustomerServiceTools) -> None:
        self.domain = get_domain_config("healthcare")
        self.knowledge_base = KnowledgeBase(self.domain)
        self.tools = tools
        self.director = MedicalConversationDirector(self.tools, self.domain, self.knowledge_base)
        checkpointer = InMemorySaver() if InMemorySaver is not None else None
        self.graph = create_conversation_graph(self.director, checkpointer=checkpointer)

    def process(self, *, session_id: str, message: str, channel: str = "chat") -> OrchestratorResult:
        result_state = self.graph.invoke(
            {"session_id": session_id, "current_message": message, "channel": channel},
            config={"configurable": {"thread_id": session_id}},
        )
        session_state = SessionState(
            session_id=session_id,
            history=list(result_state.get("history", []) or []),
            slots=dict(result_state.get("slots", {}) or {}),
            workflow={
                "booking": dict(result_state.get("booking_state", {}) or {}),
                "triage": dict(result_state.get("triage_state", {}) or {}),
            },
            last_intent=str(result_state.get("intent") or "clarify"),
            metadata=dict(result_state.get("metadata", {}) or {}),
        )
        return OrchestratorResult(
            session_id=session_id,
            reply=str(result_state.get("reply") or "").strip(),
            intent=str(result_state.get("intent") or "clarify"),
            action=str(result_state.get("action") or "reply"),
            state=session_state,
            metadata=dict(result_state.get("metadata", {}) or {}),
        )


class ConciergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_db = build_fake_database()
        self.domain = get_domain_config("healthcare")
        self.knowledge_base = KnowledgeBase(self.domain)
        self.supabase_patcher = patch("backend.mcp_tools.supabase", self.fake_db)
        self.llm_patcher = patch("backend.orchestrator.llm.enabled", False)
        self.supabase_patcher.start()
        self.llm_patcher.start()
        self.addCleanup(self.supabase_patcher.stop)
        self.addCleanup(self.llm_patcher.stop)
        self.tools = CustomerServiceTools(self.domain, self.knowledge_base)
        self.director = MedicalConversationDirector(self.tools, self.domain, self.knowledge_base)

    def test_find_provider_availability_returns_discrete_slots(self) -> None:
        self.fake_db.tables["slots"].append(
            {
                "id": 10,
                "doctor_id": 3,
                "start_time": "2026-04-06T14:30:00+05:00",
                "end_time": "2026-04-06T15:00:00+05:00",
                "status": "booked",
                "created_at": "2026-04-01T10:00:00+00:00",
            }
        )
        self.fake_db.tables["appointments"].append(
            {
                "id": 20,
                "slot_id": 10,
                "doctor_id": 3,
                "patient_id": 1,
                "status": "booked",
                "created_at": "2026-04-01T10:05:00+00:00",
            }
        )

        result = self.tools.find_provider_availability(doctor_name="Usman", date="2026-04-06")

        self.assertEqual(result["date"], "2026-04-06")
        self.assertEqual(result["available"][0]["display_slots"], ["14:00", "15:00", "15:30"])

    def test_create_booking_rejects_invalid_datetime_and_double_booking(self) -> None:
        invalid = self.tools.create_booking(patient_id=1, doctor_id=3, date="2026-99-99", time="14:00")
        self.assertEqual(invalid["error_code"], "invalid_datetime")

        created = self.tools.create_booking(patient_id=1, doctor_id=3, date="2026-04-06", time="14:00")
        self.assertIsNone(created["error_code"])
        self.assertIsNotNone(created["appointment"])

        duplicate = self.tools.create_booking(patient_id=1, doctor_id=3, date="2026-04-06", time="14:00")
        self.assertEqual(duplicate["error_code"], "slot_unavailable")

    def test_booking_agent_offers_alternatives_when_requested_time_is_unavailable(self) -> None:
        self.fake_db.tables["slots"].append(
            {
                "id": 30,
                "doctor_id": 3,
                "start_time": "2026-04-06T14:00:00+05:00",
                "end_time": "2026-04-06T14:30:00+05:00",
                "status": "booked",
                "created_at": "2026-04-01T09:00:00+00:00",
            }
        )
        self.fake_db.tables["appointments"].append(
            {
                "id": 31,
                "slot_id": 30,
                "doctor_id": 3,
                "patient_id": 1,
                "status": "booked",
                "created_at": "2026-04-01T09:05:00+00:00",
            }
        )

        state = {
            "slots": {
                "doctor_name": "Usman Raza",
                "doctor_id": 3,
                "specialization": "Neurologist",
                "phone": "03001234567",
                "date": "2026-04-06",
                "time": "14:00",
            },
            "booking_state": {},
            "current_message": "14:00",
        }

        result = self.director.booking_agent(state)

        self.assertEqual(result["action"], "availability_empty")
        self.assertIn("Closest open times", result["reply"])
        self.assertIn("14:30", result["reply"])

    def test_unsupported_specialty_returns_supported_departments(self) -> None:
        state = {
            "slots": {},
            "current_entities": {"specialization": "ENT or General Physician"},
            "current_message": "I need an ENT doctor",
            "booking_state": {},
        }

        result = self.director.recommend_doctor_agent(state)

        self.assertEqual(result["action"], "recommendation_unsupported_specialty")
        self.assertIn("I currently support", result["reply"])
        self.assertIn("Pediatrician", result["reply"])

    def test_booking_tool_failure_does_not_confirm_appointment(self) -> None:
        def fake_call_tool(name: str, arguments: dict | None = None) -> ToolCallResult:
            if name == "lookup_customer_profile":
                return ToolCallResult(ok=True, data={"profile": {"id": 1, "name": "Sara Patient", "phone": "03001234567"}})
            if name == "find_provider_availability":
                return ToolCallResult(
                    ok=True,
                    data={
                        "available": [
                            {
                                "doctor": {"id": 3, "Name": "Dr. Usman Raza", "Specialization": "Neurologist"},
                                "slot": {"start_time": "2026-04-06T14:00:00+05:00", "end_time": "2026-04-06T14:30:00+05:00"},
                                "display_slot": "14:00",
                                "display_slots": ["14:00"],
                            }
                        ],
                        "date": "2026-04-06",
                    },
                )
            if name == "create_booking":
                return ToolCallResult(ok=False, data={}, error="db down")
            raise AssertionError(f"Unexpected tool call: {name}")

        director = MedicalConversationDirector(self.tools, self.domain, self.knowledge_base)
        with patch.object(director.tools, "call_tool", side_effect=fake_call_tool):
            result = director.booking_agent(
                {
                    "slots": {
                        "doctor_name": "Usman Raza",
                        "doctor_id": 3,
                        "specialization": "Neurologist",
                        "phone": "03001234567",
                        "date": "2026-04-06",
                        "time": "14:00",
                    },
                    "booking_state": {},
                    "current_message": "14:00",
                }
            )

        self.assertEqual(result["action"], "booking_error")
        self.assertNotIn("confirmed", result["reply"].lower())

    def test_urgent_symptom_immediately_escalates(self) -> None:
        state = self.director.contextualize_turn(
            {"current_message": "I have chest pain and shortness of breath", "channel": "chat"}
        )
        state.update(self.director.classify_intent_node(state))
        result = self.director.safety_agent(state)

        self.assertEqual(result["action"], "urgent_handoff")
        self.assertIn("emergency", result["reply"].lower())

    def test_triage_completion_without_appointment_id_does_not_store_notes(self) -> None:
        with patch.object(self.tools, "call_tool", wraps=self.tools.call_tool) as wrapped_call_tool:
            result = self.director.triage_agent(
                {
                    "slots": {},
                    "current_message": "No other symptoms.",
                    "triage_state": {
                        "status": "collecting",
                        "questions": ["Anything else the doctor should know?"],
                        "current_index": 0,
                        "answers": [],
                        "red_flags": [],
                    },
                }
            )

        called_tools = [call.args[0] for call in wrapped_call_tool.call_args_list]
        self.assertEqual(result["action"], "triage_complete")
        self.assertNotIn("save_case_notes", called_tools)

    def test_chat_endpoint_smoke_flow_for_existing_patient(self) -> None:
        test_orchestrator = TestOrchestrator(self.tools)
        with patch("backend.main.orchestrator", test_orchestrator):
            from backend.main import app

            client = TestClient(app)
            session_id = "smoke-existing"

            first = client.post("/chat", json={"session_id": session_id, "user_input": "I have headache and dizziness"})
            second = client.post("/chat", json={"session_id": session_id, "user_input": "book the first one"})
            third = client.post("/chat", json={"session_id": session_id, "user_input": "03001234567"})
            fourth = client.post("/chat", json={"session_id": session_id, "user_input": "2026-04-06"})
            fifth = client.post("/chat", json={"session_id": session_id, "user_input": "2:00 PM"})

        self.assertEqual(first.status_code, 200)
        self.assertIn("Neurologist", first.json()["reply"])
        self.assertEqual(second.json()["action"], "collect_phone")
        self.assertEqual(third.json()["action"], "collect_date")
        self.assertEqual(fourth.json()["action"], "collect_time")
        self.assertEqual(fifth.json()["action"], "triage_question")
        self.assertIn("appointment is confirmed", fifth.json()["reply"].lower())

    def test_new_patient_happy_path_creates_profile_and_books(self) -> None:
        test_orchestrator = TestOrchestrator(self.tools)
        session_id = "new-patient"

        step1 = test_orchestrator.process(session_id=session_id, message="Book Dr. Usman Raza")
        step2 = test_orchestrator.process(session_id=session_id, message="03111234567")
        step3 = test_orchestrator.process(session_id=session_id, message="Amina Khan")
        step4 = test_orchestrator.process(session_id=session_id, message="2026-04-06")
        step5 = test_orchestrator.process(session_id=session_id, message="14:30")

        phones = [row["phone"] for row in self.fake_db.tables["patients"]]
        self.assertEqual(step1.action, "collect_phone")
        self.assertEqual(step2.action, "collect_name")
        self.assertEqual(step3.action, "collect_date")
        self.assertEqual(step4.action, "collect_time")
        self.assertEqual(step5.action, "triage_question")
        self.assertIn("03111234567", phones)

    def test_partial_doctor_name_question_keeps_doctor_context_clean(self) -> None:
        test_orchestrator = TestOrchestrator(self.tools)
        session_id = "doctor-info-clean"

        first = test_orchestrator.process(session_id=session_id, message="i want to see a cardiologist")
        second = test_orchestrator.process(session_id=session_id, message="when is dr ahmed available")

        self.assertEqual(first.action, "recommend_doctor")
        self.assertEqual(second.action, "doctor_info")
        self.assertIn("Dr. Ahmed Khan", second.reply)

    def test_specialty_availability_after_doctor_query_stays_in_recommendation_mode(self) -> None:
        test_orchestrator = TestOrchestrator(self.tools)
        session_id = "specialty-availability"

        test_orchestrator.process(session_id=session_id, message="i want to see a cardiologist")
        test_orchestrator.process(session_id=session_id, message="when is dr ahmed available")
        third = test_orchestrator.process(session_id=session_id, message="what cardiologist are available")

        self.assertEqual(third.action, "availability_lookup")
        self.assertIn("Here are doctors available", third.reply)


if __name__ == "__main__":
    unittest.main()
