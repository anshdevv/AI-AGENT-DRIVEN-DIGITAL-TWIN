from __future__ import annotations

import base64
import json
import os
import re
import secrets
import threading
import uuid
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, HumanMessage

from config import settings, supabase
from agents.orchestrator import BOOKING_CTX_DIR, load_booking_context, save_booking_context, orchestrator_graph
from agents.voice_agent import voice_service
from agents.pipecat_pipeline import PipecatCallPipeline
from agents.judge_agent import SAFE_HANDOFF_REPLY, judge_agent_reply

try:
    PKT = ZoneInfo("Asia/Karachi")
except ZoneInfoNotFoundError:
    PKT = timezone(timedelta(hours=5))


app = FastAPI(title="Medical Concierge Agent (LangGraph Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# session_id -> detected language code
_session_lang: dict[str, str] = {}
_human_handoff: dict[str, bool] = {}
_human_inbox: dict[str, list[dict[str, str]]] = defaultdict(list)
_active_call_ws: dict[str, WebSocket] = {}
_handoff_sessions: dict[str, dict[str, Any]] = {}
_auth_sessions: dict[str, dict[str, Any]] = {}
_session_locks_guard = threading.Lock()
_session_locks: dict[str, threading.RLock] = {}

_HUMAN_PATTERNS = tuple(
    p.strip().lower()
    for p in os.getenv(
        "HUMAN_HANDOFF_PATTERNS",
        "talk to human,talk to a human,speak to human,speak to a human,chat with human,chat with a human,human agent,real person,live agent,representative,doctor please,actual doctor",
    ).split(",")
    if p.strip()
)
_HANDOFF_START_MSG = os.getenv(
    "HUMAN_HANDOFF_START_MSG",
    "Sure, I am routing you to a human care specialist now. Please continue here and they will reply shortly.",
)
_HANDOFF_WAIT_MSG = os.getenv(
    "HUMAN_HANDOFF_WAIT_MSG",
    "Your message has been shared with the human care team. Please wait for their reply.",
)


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_input: str = Field(..., min_length=1)
    channel: str = "chat"


class VoiceRequest(BaseModel):
    session_id: str | None = None
    transcript: str | None = None
    audio_base64: str | None = None
    mime_type: str = "audio/webm"


class HumanMessageRequest(BaseModel):
    session_id: str
    message: str = Field(..., min_length=1)
    sender: str = "human"


class CorrectionRequest(BaseModel):
    note: str = Field(..., min_length=1)
    category: str = "general"
    target_message_index: int | None = None


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class LoginResponse(BaseModel):
    token: str
    role: str
    username: str
    doctor_id: int | None = None
    doctor_name: str | None = None


def _wants_human(text: str) -> bool:
    t = (text or "").strip().lower()
    return any(p in t for p in _HUMAN_PATTERNS)


@contextmanager
def _session_turn_lock(session_id: str):
    """Serialize one patient's full turn while allowing other patients in parallel."""
    key = str(session_id or "").strip()
    with _session_locks_guard:
        lock = _session_locks.setdefault(key, threading.RLock())
    lock.acquire()
    try:
        yield
    finally:
        lock.release()


def _handoff_state(start: bool) -> dict:
    return {
        "messages": [AIMessage(content=_HANDOFF_START_MSG if start else _HANDOFF_WAIT_MSG)],
        "triage_active": False,
        "human_handoff": True,
    }


def _now_iso() -> str:
    return datetime.now(PKT).isoformat()


def _auth_error() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Authentication required.",
        headers={"WWW-Authenticate": "Bearer"},
    )


def _get_auth_user(authorization: str | None = Header(default=None)) -> dict[str, str]:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise _auth_error()
    token = authorization.split(" ", 1)[1].strip()
    user = _auth_sessions.get(token)
    if not user:
        raise _auth_error()
    return user


def _require_role(*roles: str):
    allowed = set(roles)

    def dependency(user: dict[str, str] = Depends(_get_auth_user)) -> dict[str, str]:
        if user.get("role") not in allowed:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions.")
        return user

    return dependency


def _public_ctx(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "session_id": ctx.get("session_id"),
        "created_at": ctx.get("created_at"),
        "last_updated": ctx.get("last_updated"),
        "step": ctx.get("step"),
        "prime_complaint": ctx.get("prime_complaint"),
        "recommended_specialist": ctx.get("recommended_specialist"),
        "patient": ctx.get("patient") or {},
        "selected_doctor": ctx.get("selected_doctor") or {},
        "appointment": ctx.get("appointment") or {},
        "triage_completed": ctx.get("triage_completed", False),
        "human_handoff": _human_handoff.get(str(ctx.get("session_id")), False),
    }


def _persist_transcript_entry(session_id: str, entry: dict[str, Any]) -> None:
    try:
        ctx = load_booking_context(session_id)
        transcript = list(ctx.get("transcript") or [])
        transcript.append(entry)
        ctx["transcript"] = transcript[-300:]
        save_booking_context(session_id, ctx)
    except Exception as exc:
        print(f"[Transcript] persist failed for {session_id}: {exc}")


def _record_transcript(
    session_id: str,
    *,
    sender: str,
    text: str,
    channel: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    clean = str(text or "").strip()
    if not clean:
        return
    entry = {
        "at": _now_iso(),
        "sender": sender,
        "text": clean,
        "channel": channel,
        "metadata": metadata or {},
    }
    _persist_transcript_entry(session_id, entry)
    if _human_handoff.get(session_id):
        _mark_handoff(session_id, status_value=_handoff_sessions.get(session_id, {}).get("status", "active"))


def _mark_handoff(session_id: str, *, status_value: str = "pending", reason: str | None = None) -> None:
    ctx = load_booking_context(session_id)
    existing = _handoff_sessions.get(session_id, {})
    session = {
        "session_id": session_id,
        "status": status_value,
        "reason": reason or existing.get("reason") or "Patient requested human support.",
        "created_at": existing.get("created_at") or _now_iso(),
        "updated_at": _now_iso(),
        "patient": ctx.get("patient") or {},
        "prime_complaint": ctx.get("prime_complaint"),
        "recommended_specialist": ctx.get("recommended_specialist"),
        "last_message": "",
    }
    transcript = ctx.get("transcript") or []
    if transcript:
        session["last_message"] = transcript[-1].get("text", "")
    _handoff_sessions[session_id] = session
    ctx["human_handoff"] = True
    ctx["handoff_status"] = status_value
    ctx["handoff_reason"] = session["reason"]
    save_booking_context(session_id, ctx)


def _set_reply(state: dict[str, Any], reply_text: str) -> None:
    messages = list(state.get("messages") or [])
    if messages:
        messages[-1] = AIMessage(content=reply_text)
    else:
        messages = [AIMessage(content=reply_text)]
    state["messages"] = messages


def _finalize_agent_reply(
    *,
    session_id: str,
    user_message: str,
    channel: str,
    state: dict[str, Any],
    skip_judge: bool = False,
) -> dict[str, Any]:
    messages = list(state.get("messages") or [])
    reply_text = str(messages[-1].content) if messages else "I couldn't process that."

    if not skip_judge and not state.get("human_handoff", False):
        verdict = judge_agent_reply(
            user_message=user_message,
            draft_reply=reply_text,
            channel=channel,
            data_context={
                "booking_context": state.get("booking_context") or {},
                "triage_active": state.get("triage_active", False),
            },
        )
        state["judge"] = {
            "approved": verdict.approved,
            "accuracy_risk": verdict.accuracy_risk,
            "medical_safety_risk": verdict.medical_safety_risk,
            "reason": verdict.reason,
        }
        if not verdict.approved:
            _human_handoff[session_id] = True
            _mark_handoff(session_id, status_value="pending", reason=f"Judge flagged agent reply: {verdict.reason}")
            reply_text = verdict.safe_reply or SAFE_HANDOFF_REPLY
            _set_reply(state, reply_text)
            state["human_handoff"] = True

    _record_transcript(session_id, sender="assistant", text=reply_text, channel=channel, metadata={"judge": state.get("judge")})
    return state


def _collect_by_id(rows: list[dict[str, Any]], key: str = "id") -> dict[Any, dict[str, Any]]:
    return {row.get(key): row for row in rows if row.get(key) is not None}


def _coerce_int(value: Any) -> int | None:
    try:
        if value is None or str(value).strip() == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _session_file_for_id(session_id: str) -> Any:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", str(session_id or ""))
    return BOOKING_CTX_DIR / f"{safe}.json"


def _read_context_file(path: Any) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[CRM] Could not read session file {path}: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    data.setdefault("session_id", path.stem)
    return data


def _load_all_booking_contexts() -> list[dict[str, Any]]:
    contexts: list[dict[str, Any]] = []
    try:
        files = sorted(BOOKING_CTX_DIR.glob("*.json"))
    except Exception as exc:
        print(f"[CRM] Could not list booking contexts: {exc}")
        return contexts
    for path in files:
        ctx = _read_context_file(path)
        if ctx:
            contexts.append(ctx)
    contexts.sort(key=lambda item: item.get("last_updated") or item.get("created_at") or "", reverse=True)
    return contexts


def _load_existing_context(session_id: str) -> dict[str, Any]:
    path = _session_file_for_id(session_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Session not found.")
    return load_booking_context(session_id)


def _normalize_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        match = re.search(r"-?\d+(?:\.\d+)?", str(value))
        if not match:
            return None
        number = float(match.group(0))
    if number <= 1:
        number *= 100
    return round(max(0, min(number, 100)), 1)


def _score_from_mapping(mapping: dict[str, Any], names: tuple[str, ...]) -> float | None:
    for name in names:
        if name in mapping:
            score = _normalize_score(mapping.get(name))
            if score is not None:
                return score
    for key in ("evaluation", "evaluations", "metrics", "quality_scores", "ragas", "scores"):
        nested = mapping.get(key)
        if isinstance(nested, dict):
            score = _score_from_mapping(nested, names)
            if score is not None:
                return score
    return None


def _explicit_quality_score(ctx: dict[str, Any], names: tuple[str, ...]) -> float | None:
    score = _score_from_mapping(ctx, names)
    if score is not None:
        return score
    for entry in ctx.get("transcript") or []:
        metadata = entry.get("metadata") if isinstance(entry, dict) else None
        if isinstance(metadata, dict):
            score = _score_from_mapping(metadata, names)
            if score is not None:
                return score
    return None


def _judge_proxy_scores(ctx: dict[str, Any]) -> dict[str, Any]:
    judged = []
    for entry in ctx.get("transcript") or []:
        metadata = entry.get("metadata") if isinstance(entry, dict) else None
        judge = metadata.get("judge") if isinstance(metadata, dict) else None
        if isinstance(judge, dict):
            judged.append(judge)

    if not judged:
        return {"faithfulness": None, "relevance": None, "source": "pending", "judged_turns": 0}

    faithful = [
        item
        for item in judged
        if item.get("approved") is True and not item.get("accuracy_risk") and not item.get("medical_safety_risk")
    ]
    relevant = [item for item in judged if item.get("approved") is True and not item.get("medical_safety_risk")]
    total = max(len(judged), 1)
    return {
        "faithfulness": round(len(faithful) / total * 100, 1),
        "relevance": round(len(relevant) / total * 100, 1),
        "source": "judge_proxy",
        "judged_turns": len(judged),
    }


def _quality_scores(ctx: dict[str, Any]) -> dict[str, Any]:
    faithfulness = _explicit_quality_score(ctx, ("faithfulness_score", "faithfulness", "faithful_score"))
    relevance = _explicit_quality_score(ctx, ("relevance_score", "relevance", "answer_relevance", "answer_relevancy"))
    source = "explicit" if faithfulness is not None or relevance is not None else "pending"
    proxy = _judge_proxy_scores(ctx)
    if faithfulness is None:
        faithfulness = proxy["faithfulness"]
    if relevance is None:
        relevance = proxy["relevance"]
    if source == "pending" and proxy["source"] != "pending":
        source = proxy["source"]
    return {
        "faithfulness": faithfulness,
        "relevance": relevance,
        "source": source,
        "judged_turns": proxy["judged_turns"],
    }


def _last_transcript_entry(ctx: dict[str, Any]) -> dict[str, Any]:
    transcript = [item for item in (ctx.get("transcript") or []) if isinstance(item, dict)]
    return transcript[-1] if transcript else {}


def _session_status(ctx: dict[str, Any]) -> str:
    if ctx.get("appointment", {}).get("confirmed"):
        return "booked"
    if ctx.get("human_handoff") or ctx.get("handoff_status"):
        return "handoff"
    if ctx.get("triage_completed"):
        return "triaged"
    return "active"


def _session_summary(ctx: dict[str, Any]) -> dict[str, Any]:
    transcript = [item for item in (ctx.get("transcript") or []) if isinstance(item, dict)]
    last_entry = _last_transcript_entry(ctx)
    corrections = list(ctx.get("llm_corrections") or [])
    return {
        "session_id": ctx.get("session_id"),
        "created_at": ctx.get("created_at"),
        "last_updated": ctx.get("last_updated"),
        "status": _session_status(ctx),
        "step": ctx.get("step"),
        "prime_complaint": ctx.get("prime_complaint"),
        "recommended_specialist": ctx.get("recommended_specialist"),
        "triage_completed": ctx.get("triage_completed", False),
        "triage_questions_asked": ctx.get("triage_questions_asked", 0),
        "patient": ctx.get("patient") or {},
        "selected_doctor": ctx.get("selected_doctor") or {},
        "appointment": ctx.get("appointment") or {},
        "transcript_count": len(transcript),
        "last_message": last_entry.get("text", ""),
        "last_sender": last_entry.get("sender", ""),
        "scores": _quality_scores(ctx),
        "corrections_count": len(corrections),
        "latest_correction": corrections[-1] if corrections else None,
        "has_report": bool(ctx.get("diagnostic_report")),
        "handoff_status": ctx.get("handoff_status"),
    }


def _session_detail(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "summary": _session_summary(ctx),
        "context": _public_ctx(ctx),
        "transcript": ctx.get("transcript") or [],
        "triage_qa": ctx.get("triage_qa") or [],
        "diagnostic_report": ctx.get("diagnostic_report") or "",
        "corrections": ctx.get("llm_corrections") or [],
        "preference_tuning_notes": ctx.get("preference_tuning_notes") or [],
        "final_symptom_match": ctx.get("final_symptom_match") or ctx.get("symptom_context_block") or "",
    }


def _find_doctor_for_login(username: str) -> dict[str, Any] | None:
    if not supabase:
        return None
    cleaned = username.strip()
    if not cleaned:
        return None

    try:
        numeric_id = _coerce_int(cleaned)
        if numeric_id is not None:
            response = supabase.table("doctors").select("*").eq("id", numeric_id).limit(1).execute()
            if response.data:
                return response.data[0]

        response = supabase.table("doctors").select("*").limit(1000).execute()
        candidates = response.data or []
    except Exception as exc:
        print(f"[Auth] Doctor lookup failed: {exc}")
        return None

    lowered = cleaned.lower()
    for doctor in candidates:
        values = [
            doctor.get("username"),
            doctor.get("email"),
            doctor.get("phone"),
            doctor.get("name"),
            doctor.get("Name"),
        ]
        if any(str(value or "").strip().lower() == lowered for value in values):
            return doctor

    for doctor in candidates:
        name = str(doctor.get("name") or doctor.get("Name") or "").strip().lower()
        if lowered and lowered in name:
            return doctor
    return None


def _public_doctor(doctor: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": doctor.get("id"),
        "name": doctor.get("name") or doctor.get("Name") or "Unknown",
        "specialization": doctor.get("specialization") or doctor.get("Specialization") or "",
    }


def process_with_langgraph(session_id: str, message: str, channel: str, target_lang: str = "en") -> dict:
    """Push a message through LangGraph using session_id as the persistent thread."""
    _record_transcript(session_id, sender="patient", text=message, channel=channel)
    human_requested = _wants_human(message)
    if human_requested:
        _human_handoff[session_id] = True
        if message.strip():
            _human_inbox[session_id].append({"sender": "patient", "message": message, "at": _now_iso()})
        _mark_handoff(session_id, status_value="pending", reason="Patient requested a human.")
        ctx = load_booking_context(session_id)
        return {
            "session_id": session_id,
            "messages": [AIMessage(content=_HANDOFF_START_MSG)],
            "triage_active": False,
            "human_handoff": True,
            "booking_context": ctx,
        }

    if _human_handoff.get(session_id):
        if message.strip():
            _human_inbox[session_id].append({"sender": "patient", "message": message, "at": _now_iso()})
        _mark_handoff(session_id, status_value=_handoff_sessions.get(session_id, {}).get("status", "active"))
        ctx = load_booking_context(session_id)
        return {
            "session_id": session_id,
            "messages": [AIMessage(content="")],
            "triage_active": False,
            "human_handoff": True,
            "awaiting_human": True,
            "booking_context": ctx,
        }

    ctx = load_booking_context(session_id)

    if target_lang and ctx.get("patient_language") != target_lang:
        ctx["patient_language"] = target_lang
        save_booking_context(session_id, ctx)

    config = {"configurable": {"thread_id": session_id}}
    input_state = {
        "session_id": session_id,
        "messages": [HumanMessage(content=message)],
        "booking_context": ctx,
    }

    print(f"\n[API] invoke -> thread='{session_id}' | lang='{target_lang}' | msg='{message[:80]}'")
    state = orchestrator_graph.invoke(input_state, config=config)
    state["human_handoff"] = _human_handoff.get(session_id, False)
    return state


@app.post("/auth/login", response_model=LoginResponse)
def login(request: LoginRequest) -> LoginResponse:
    username = request.username.strip()
    password = request.password
    role = None
    auth_payload: dict[str, Any] | None = None
    if username == settings.admin_username and secrets.compare_digest(password, settings.admin_password):
        role = "admin"
        auth_payload = {"username": username, "role": role}
    elif username == settings.csr_username and secrets.compare_digest(password, settings.csr_password):
        role = "csr"
        auth_payload = {"username": username, "role": role}
    elif secrets.compare_digest(password, settings.doctor_portal_password):
        doctor = _find_doctor_for_login(username)
        if doctor:
            role = "doctor"
            public_doctor = _public_doctor(doctor)
            auth_payload = {
                "username": public_doctor["name"],
                "role": role,
                "doctor_id": public_doctor["id"],
                "doctor_name": public_doctor["name"],
            }
    if not role:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid username or password.")

    token = secrets.token_urlsafe(32)
    _auth_sessions[token] = auth_payload or {"username": username, "role": role}
    return LoginResponse(token=token, **_auth_sessions[token])


@app.get("/auth/me")
def auth_me(user: dict[str, Any] = Depends(_get_auth_user)) -> dict:
    return {
        "username": user["username"],
        "role": user["role"],
        "doctor_id": user.get("doctor_id"),
        "doctor_name": user.get("doctor_name"),
    }


@app.get("/admin/appointments")
def admin_appointments(_: dict[str, str] = Depends(_require_role("admin"))) -> dict:
    if not supabase:
        raise HTTPException(status_code=503, detail="Database not connected.")
    try:
        appointments = supabase.table("appointments").select("*").limit(500).execute().data or []
        doctors = _collect_by_id(supabase.table("doctors").select("*").limit(500).execute().data or [])
        patients = _collect_by_id(supabase.table("patients").select("*").limit(500).execute().data or [])
        slots = _collect_by_id(supabase.table("slots").select("*").limit(1000).execute().data or [])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Database error: {exc}") from exc

    rows = []
    for appt in appointments:
        doctor = doctors.get(appt.get("doctor_id"), {})
        patient = patients.get(appt.get("patient_id"), {})
        slot = slots.get(appt.get("slot_id"), {})
        rows.append(
            {
                "id": appt.get("id"),
                "status": appt.get("status"),
                "created_at": appt.get("created_at"),
                "notes": appt.get("notes") or "",
                "doctor": {
                    "id": doctor.get("id") or appt.get("doctor_id"),
                    "name": doctor.get("name") or doctor.get("Name") or "Unknown",
                    "specialization": doctor.get("specialization") or doctor.get("Specialization") or "",
                },
                "patient": {
                    "id": patient.get("id") or appt.get("patient_id"),
                    "name": patient.get("name") or patient.get("Name") or "Unknown",
                    "phone": patient.get("phone") or "",
                },
                "slot": {
                    "id": slot.get("id") or appt.get("slot_id"),
                    "start_time": slot.get("start_time"),
                    "end_time": slot.get("end_time"),
                    "status": slot.get("status"),
                },
            }
        )
    rows.sort(key=lambda item: item.get("slot", {}).get("start_time") or item.get("created_at") or "", reverse=True)
    return {"appointments": rows, "count": len(rows)}


@app.get("/admin/crm")
def admin_crm(_: dict[str, Any] = Depends(_require_role("admin"))) -> dict:
    sessions = [_session_summary(ctx) for ctx in _load_all_booking_contexts()]
    active = [item for item in sessions if item.get("status") == "active"]
    handoffs = [item for item in sessions if item.get("status") == "handoff"]
    booked = [item for item in sessions if item.get("status") == "booked"]
    scored = [item for item in sessions if item.get("scores", {}).get("faithfulness") is not None]
    avg_faithfulness = (
        round(sum(item["scores"]["faithfulness"] for item in scored) / len(scored), 1)
        if scored else None
    )
    relevant = [item for item in sessions if item.get("scores", {}).get("relevance") is not None]
    avg_relevance = (
        round(sum(item["scores"]["relevance"] for item in relevant) / len(relevant), 1)
        if relevant else None
    )
    return {
        "sessions": sessions,
        "count": len(sessions),
        "stats": {
            "active": len(active),
            "handoffs": len(handoffs),
            "booked": len(booked),
            "corrections": sum(int(item.get("corrections_count") or 0) for item in sessions),
            "avg_faithfulness": avg_faithfulness,
            "avg_relevance": avg_relevance,
        },
    }


@app.get("/admin/chats/{session_id}")
def admin_chat_detail(session_id: str, _: dict[str, Any] = Depends(_require_role("admin"))) -> dict:
    ctx = _load_existing_context(session_id)
    return _session_detail(ctx)


@app.post("/admin/chats/{session_id}/corrections")
def admin_add_correction(
    session_id: str,
    request: CorrectionRequest,
    user: dict[str, Any] = Depends(_require_role("admin")),
) -> dict:
    ctx = _load_existing_context(session_id)
    corrections = list(ctx.get("llm_corrections") or [])
    entry = {
        "id": str(uuid.uuid4()),
        "at": _now_iso(),
        "author": user.get("username", "admin"),
        "category": request.category.strip() or "general",
        "target_message_index": request.target_message_index,
        "note": request.note.strip(),
    }
    corrections.append(entry)
    ctx["llm_corrections"] = corrections
    ctx["preference_tuning_notes"] = [
        {
            "category": item.get("category", "general"),
            "note": item.get("note", ""),
            "session_id": session_id,
            "created_at": item.get("at"),
        }
        for item in corrections
        if item.get("note")
    ]
    save_booking_context(session_id, ctx)
    return {"ok": True, "correction": entry, "session": _session_summary(ctx)}


@app.get("/admin/preference-notes")
def admin_preference_notes(_: dict[str, Any] = Depends(_require_role("admin"))) -> dict:
    notes: list[dict[str, Any]] = []
    for ctx in _load_all_booking_contexts():
        for item in ctx.get("preference_tuning_notes") or []:
            if isinstance(item, dict):
                notes.append(
                    {
                        **item,
                        "session_id": item.get("session_id") or ctx.get("session_id"),
                        "patient": ctx.get("patient") or {},
                        "prime_complaint": ctx.get("prime_complaint"),
                    }
                )
    notes.sort(key=lambda item: item.get("created_at") or "", reverse=True)
    return {"notes": notes, "count": len(notes)}


@app.get("/doctor/me")
def doctor_me(user: dict[str, Any] = Depends(_require_role("doctor"))) -> dict:
    return {
        "doctor": {
            "id": user.get("doctor_id"),
            "name": user.get("doctor_name") or user.get("username"),
        }
    }


@app.get("/doctor/patients")
def doctor_patients(user: dict[str, Any] = Depends(_require_role("doctor"))) -> dict:
    doctor_id = _coerce_int(user.get("doctor_id"))
    if doctor_id is None:
        raise HTTPException(status_code=403, detail="Doctor account is not linked to a doctor record.")

    contexts = _load_all_booking_contexts()
    def local_doctor_rows() -> dict:
        matched = [
            ctx for ctx in contexts
            if _coerce_int((ctx.get("selected_doctor") or {}).get("id")) == doctor_id
        ]
        return {
            "doctor": {"id": doctor_id, "name": user.get("doctor_name") or user.get("username")},
            "patients": [
                {
                    "appointment": ctx.get("appointment") or {},
                    "patient": ctx.get("patient") or {},
                    "slot": {},
                    "clinical_notes": ctx.get("diagnostic_report") or "\n\n".join(ctx.get("triage_qa") or []),
                    "sessions": [_session_summary(ctx)],
                }
                for ctx in matched
            ],
            "count": len(matched),
        }

    if not supabase:
        return local_doctor_rows()

    try:
        appointments = (
            supabase.table("appointments")
            .select("*")
            .eq("doctor_id", doctor_id)
            .limit(500)
            .execute()
            .data
            or []
        )
        patients = _collect_by_id(supabase.table("patients").select("*").limit(1000).execute().data or [])
        slots = _collect_by_id(supabase.table("slots").select("*").limit(1500).execute().data or [])
    except Exception as exc:
        print(f"[DoctorCRM] Database unavailable, using local context fallback: {exc}")
        return local_doctor_rows()

    rows = []
    for appt in appointments:
        patient = patients.get(appt.get("patient_id"), {})
        slot = slots.get(appt.get("slot_id"), {})
        appointment_id = appt.get("id")
        patient_id = appt.get("patient_id")
        related_contexts = [
            ctx for ctx in contexts
            if _coerce_int((ctx.get("selected_doctor") or {}).get("id")) == doctor_id
            and (
                _coerce_int((ctx.get("appointment") or {}).get("booking_id")) == _coerce_int(appointment_id)
                or _coerce_int((ctx.get("patient") or {}).get("id")) == _coerce_int(patient_id)
            )
        ]
        related_contexts.sort(key=lambda item: item.get("last_updated") or item.get("created_at") or "", reverse=True)
        fallback_notes = ""
        if related_contexts:
            latest = related_contexts[0]
            fallback_notes = latest.get("diagnostic_report") or "\n\n".join(latest.get("triage_qa") or [])
        rows.append(
            {
                "appointment": {
                    "id": appointment_id,
                    "status": appt.get("status"),
                    "created_at": appt.get("created_at"),
                    "notes": appt.get("notes") or "",
                },
                "patient": {
                    "id": patient.get("id") or patient_id,
                    "name": patient.get("name") or patient.get("Name") or "Unknown",
                    "phone": patient.get("phone") or "",
                    "age": patient.get("age"),
                    "gender": patient.get("gender"),
                },
                "slot": {
                    "id": slot.get("id") or appt.get("slot_id"),
                    "start_time": slot.get("start_time"),
                    "end_time": slot.get("end_time"),
                    "status": slot.get("status"),
                },
                "clinical_notes": appt.get("notes") or fallback_notes,
                "sessions": [_session_summary(ctx) for ctx in related_contexts],
            }
        )

    rows.sort(key=lambda item: item.get("slot", {}).get("start_time") or item.get("appointment", {}).get("created_at") or "", reverse=True)
    return {
        "doctor": {"id": doctor_id, "name": user.get("doctor_name") or user.get("username")},
        "patients": rows,
        "count": len(rows),
    }


@app.get("/csr/handoffs")
def csr_handoffs(_: dict[str, str] = Depends(_require_role("csr"))) -> dict:
    sessions = sorted(_handoff_sessions.values(), key=lambda item: item.get("updated_at", ""), reverse=True)
    return {"handoffs": sessions, "count": len(sessions)}


@app.get("/csr/handoffs/{session_id}")
def csr_handoff_detail(session_id: str, _: dict[str, str] = Depends(_require_role("csr"))) -> dict:
    if session_id not in _handoff_sessions and not _human_handoff.get(session_id):
        raise HTTPException(status_code=404, detail="Handoff session not found.")
    ctx = load_booking_context(session_id)
    return {
        "handoff": _handoff_sessions.get(session_id) or {"session_id": session_id, "status": "active"},
        "context": _public_ctx(ctx),
        "transcript": ctx.get("transcript") or [],
        "queued_messages": _human_inbox.get(session_id, []),
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    session_id = request.session_id or str(uuid.uuid4())
    with _session_turn_lock(session_id):
        state = process_with_langgraph(session_id, request.user_input, request.channel)
        state = _finalize_agent_reply(
            session_id=session_id,
            user_message=request.user_input,
            channel=request.channel,
            state=state,
        )
        messages = state.get("messages", [])
        reply_text = messages[-1].content if messages else "I couldn't process that."

        return {
            "session_id": session_id,
            "reply": reply_text,
            "triage_active": state.get("triage_active", False),
            "human_handoff": state.get("human_handoff", False),
        }


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "domain": settings.app_domain,
        "huggingface_llm": bool(settings.huggingface_api_key),
        "judge_model": settings.judge_model,
        "elevenlabs_stt": voice_service.supports_server_stt,
        "elevenlabs_tts": voice_service.supports_server_tts,
    }


@app.post("/voice/message")
async def voice_message(request: VoiceRequest) -> dict:
    session_id = request.session_id or str(uuid.uuid4())
    audio_bytes = base64.b64decode(request.audio_base64) if request.audio_base64 else None

    raw_transcript, detected_lang = await voice_service.transcribe_raw(
        audio_bytes=audio_bytes,
        mime_type=request.mime_type,
        transcript_hint=request.transcript,
    )

    if detected_lang not in ("en", "english"):
        _session_lang[session_id] = detected_lang

    if detected_lang not in ("en", "english") and audio_bytes and voice_service.supports_server_stt:
        transcript = await voice_service.translate_to_english(
            audio_bytes=audio_bytes,
            mime_type=request.mime_type,
        )
        if not transcript.strip():
            transcript = raw_transcript
    else:
        transcript = raw_transcript

    target_lang = _session_lang.get(session_id, "en")
    with _session_turn_lock(session_id):
        state = process_with_langgraph(session_id, transcript, "voice_message", target_lang)
        state = _finalize_agent_reply(
            session_id=session_id,
            user_message=transcript,
            channel="voice_message",
            state=state,
        )
        messages = state.get("messages", [])
        reply_text = messages[-1].content if messages else "I'm sorry, I couldn't process that."
    audio_reply = await voice_service.synthesize_base64_wav(reply_text) if str(reply_text).strip() else ""

    return {
        "session_id": session_id,
        "transcript": transcript,
        "detected_language": detected_lang,
        "reply": reply_text,
        "triage_active": state.get("triage_active", False),
        "human_handoff": state.get("human_handoff", False),
        "audio_base64": audio_reply,
    }


@app.post("/human/message")
async def human_message(
    request: HumanMessageRequest,
    _: dict[str, str] = Depends(_require_role("csr", "admin")),
) -> dict:
    session_id = request.session_id
    with _session_turn_lock(session_id):
        _human_handoff[session_id] = True
        _mark_handoff(session_id, status_value="active", reason="CSR joined the conversation.")
        _human_inbox[session_id].append({"sender": request.sender, "message": request.message, "at": _now_iso()})
        _record_transcript(session_id, sender=request.sender or "human", text=request.message, channel="human")
    ws = _active_call_ws.get(session_id)
    if ws:
        try:
            await ws.send_json({"type": "human_message", "text": request.message, "sender": request.sender})
        except Exception:
            pass
    return {"ok": True, "session_id": session_id, "queued_messages": len(_human_inbox.get(session_id, []))}


@app.get("/human/messages/{session_id}")
def human_messages(session_id: str) -> dict:
    messages = [
        item
        for item in _human_inbox.get(session_id, [])
        if str(item.get("sender", "")).lower() not in {"patient", "user"}
    ]
    return {"session_id": session_id, "messages": messages, "count": len(messages)}


@app.websocket("/ws/call/{session_id}")
async def call_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    print(f"[WS] Client connected: session_id='{session_id}'")
    _active_call_ws[session_id] = websocket

    def process_call_turn(call_session_id: str, message: str, channel: str, target_lang: str = "en") -> dict:
        with _session_turn_lock(call_session_id):
            state = process_with_langgraph(call_session_id, message, channel, target_lang)
            return _finalize_agent_reply(
                session_id=call_session_id,
                user_message=message,
                channel=channel,
                state=state,
            )

    pipeline = PipecatCallPipeline(
        websocket=websocket,
        session_id=session_id,
        voice_service=voice_service,
        process_with_langgraph=process_call_turn,
        session_lang_store=_session_lang,
    )
    await pipeline.start()

    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except WebSocketDisconnect:
                raise
            except Exception as e:
                print(f"[WS] receive_json error: {e}")
                break
            await pipeline.handle_client_message(payload)
    except WebSocketDisconnect:
        print(f"[WS] Client disconnected: session_id='{session_id}'")
    finally:
        await pipeline.shutdown()
        _session_lang.pop(session_id, None)
        _active_call_ws.pop(session_id, None)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
