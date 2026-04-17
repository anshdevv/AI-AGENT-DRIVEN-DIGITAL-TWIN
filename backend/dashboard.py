from __future__ import annotations

from typing import Any


TRIAGE_DIAGNOSIS_NOTE = "This is the triage agent's summary for the doctor, not a final diagnosis."


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return text


def _pick_text(*values: Any) -> str:
    for value in values:
        text = _clean_text(value)
        if text:
            return text
    return ""


def _normalize_answers(raw_answers: Any) -> list[dict[str, str]]:
    answers: list[dict[str, str]] = []
    for item in _as_list(raw_answers):
        row = _as_dict(item)
        question = _clean_text(row.get("question"))
        answer = _clean_text(row.get("answer"))
        if question or answer:
            answers.append({"question": question, "answer": answer})
    return answers


def _attention_status(triage_status: str, *, has_summary: bool, red_flag_triggered: bool) -> str:
    if red_flag_triggered:
        return "Urgent review recommended"
    if has_summary:
        return "Ready for doctor review"
    if triage_status in {"collecting", "pending_start"}:
        return "Intake in progress"
    return "Waiting for intake"


def build_session_dashboard(session_id: str, state: dict[str, Any] | None) -> dict[str, Any]:
    state = state if isinstance(state, dict) else {}

    booking_state = _as_dict(state.get("booking_state"))
    triage_state = _as_dict(state.get("triage_state"))
    slots = _as_dict(state.get("slots"))
    metadata = _as_dict(state.get("metadata"))
    doctor_profile = _as_dict(state.get("doctor_profile"))
    current_entities = _as_dict(state.get("current_entities"))

    patient_profile = _as_dict(booking_state.get("patient_profile"))
    appointment = _as_dict(booking_state.get("appointment"))
    answers = _normalize_answers(triage_state.get("answers"))
    summary = _pick_text(triage_state.get("summary"), metadata.get("triage_summary"))
    triage_status = _pick_text(triage_state.get("status"), "idle")
    complaint = _pick_text(slots.get("symptom"), current_entities.get("symptom"))
    doctor_name = _pick_text(
        doctor_profile.get("Name"),
        slots.get("doctor_name"),
        triage_state.get("doctor_name"),
    )
    doctor_specialization = _pick_text(
        doctor_profile.get("Specialization"),
        slots.get("specialization"),
        triage_state.get("specialization"),
    )
    red_flag_triggered = bool(triage_state.get("red_flag_triggered"))

    return {
        "session_id": session_id,
        "has_data": bool(patient_profile or appointment or complaint or answers or summary),
        "patient": {
            "id": patient_profile.get("id"),
            "name": _pick_text(patient_profile.get("name")),
            "phone": _pick_text(patient_profile.get("phone")),
            "gender": _pick_text(patient_profile.get("gender")),
            "age": patient_profile.get("age"),
        },
        "doctor": {
            "id": doctor_profile.get("id") or appointment.get("doctor_id") or slots.get("doctor_id"),
            "name": doctor_name,
            "specialization": doctor_specialization,
        },
        "appointment": {
            "id": appointment.get("id"),
            "status": _pick_text(appointment.get("status"), booking_state.get("status")),
            "date": _pick_text(slots.get("date")),
            "time": _pick_text(slots.get("time")),
            "slot_id": appointment.get("slot_id"),
        },
        "complaint": complaint,
        "triage": {
            "status": triage_status,
            "flow_name": _pick_text(triage_state.get("flow_name")),
            "summary": summary,
            "answers": answers,
            "red_flag_triggered": red_flag_triggered,
            "attention_status": _attention_status(
                triage_status,
                has_summary=bool(summary),
                red_flag_triggered=red_flag_triggered,
            ),
            "diagnosis_note": TRIAGE_DIAGNOSIS_NOTE,
        },
    }


def get_session_dashboard(orchestrator: Any, session_id: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if orchestrator is not None:
        try:
            state = orchestrator.get_session_state(session_id)
        except Exception:
            state = {}
    return build_session_dashboard(session_id, state)
