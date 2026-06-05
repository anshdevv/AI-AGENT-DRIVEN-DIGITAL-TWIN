# main.py
# ─────────────────────────────────────────────────────────────────────────────
# RAG (dialect_middleware) and Judge LLM are commented out.
# Re-enable when pipeline phases are stable.
# ─────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import base64
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from config import settings
from agents.orchestrator import orchestrator_graph, load_booking_context, save_booking_context
from agents.voice_agent import voice_service

# ── RAG layer — DISABLED (comment back in when pipeline is stable) ────────────
# from rag.dialect_middleware import dialect_middleware   # noqa: F401
# from rag.judge_llm import JudgeLLM

# ── Judge singleton — DISABLED ────────────────────────────────────────────────
# ESCALATION_MESSAGE = (
#     "I'm having some trouble understanding. "
#     "Let me transfer you to a human agent who can help you better."
# )
# _judge: JudgeLLM | None = None
# def _get_judge() -> JudgeLLM | None:
#     global _judge
#     if _judge is not None:
#         return _judge
#     if not settings.groq_api_key:
#         print("⚠️  [Judge] GROQ_API_KEY missing — judge disabled.")
#         return None
#     try:
#         _judge = JudgeLLM(groq_api_key=settings.groq_api_key, model="qwen/qwen3-32b")
#         print("✅ [Judge] JudgeLLM initialised.")
#     except Exception as e:
#         print(f"⚠️  [Judge] Init failed: {e}")
#     return _judge


import re

app = FastAPI(title="Medical Concierge Agent (LangGraph Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Internal tag / think-block sanitiser ──────────────────────────────────────
_INTERNAL_TAGS = re.compile(
    r"<think>.*?</think>"               # Qwen reasoning blocks
    r"|(\[SYMPTOM_LOGGED:[^\]]*\])"     # pipeline control tags
    r"|(\[START_TRIAGE\])"
    r"|(\[RECOMMEND_SPECIALIST\])"
    r"|(\[END_CALL\])"
    r"|(\[TRANSFER_TO_HUMAN\])"
    r"|(\[HUMAN_CONFIRMED\])"
    r"|(\[HUMAN_REQUESTED\])",
    re.DOTALL,
)

def _sanitize_reply(text: str) -> str:
    """Remove internal pipeline tags and Qwen think blocks before sending to client."""
    cleaned = _INTERNAL_TAGS.sub("", text).strip()
    return cleaned if cleaned else ""

# ── Session language store ─────────────────────────────────────────────────────
_session_lang: dict[str, str] = {}


# ── Request / Response models ─────────────────────────────────────────────────
class ChatRequest(BaseModel):
    session_id: str | None = None
    user_input: str = Field(..., min_length=1)
    channel: str = "chat"


class VoiceRequest(BaseModel):
    session_id: str | None = None
    transcript: str | None = None
    audio_base64: str | None = None
    mime_type: str = "audio/webm"


# ── Core LangGraph helper ─────────────────────────────────────────────────────
def process_with_langgraph(
    session_id: str, message: str, channel: str, target_lang: str = "en"
) -> dict:
    ctx = load_booking_context(session_id)
    if target_lang and ctx.get("patient_language") != target_lang:
        ctx["patient_language"] = target_lang
        save_booking_context(session_id, ctx)

    config      = {"configurable": {"thread_id": session_id}}
    input_state = {
        "session_id":      session_id,
        "messages":        [HumanMessage(content=message)],
        "booking_context": ctx,
    }
    print(f"\n📨 [API] invoke → thread='{session_id}' | lang='{target_lang}' | msg='{message[:80]}'")
    return orchestrator_graph.invoke(input_state, config=config)


# ── /chat ─────────────────────────────────────────────────────────────────────
@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    session_id = request.session_id or str(uuid.uuid4())
    if not request.session_id:
        print(f"🆕 [API] New session: {session_id}")
    else:
        print(f"♻️  [API] Continuing session: {session_id}")

    state = process_with_langgraph(session_id, request.user_input, request.channel)

    messages = state.get("messages", [])
    reply_text = messages[-1].content if messages else "I couldn't process that."
    reply_text = _sanitize_reply(reply_text) or "One moment please."

    # ── Judge evaluation — DISABLED ───────────────────────────────────────────
    # judge = _get_judge()
    # if judge:
    #     judge_result = judge.evaluate(
    #         query=request.user_input, response=reply_text,
    #         session_id=session_id, tool_events=state.get("tool_events"),
    #     )
    #     if judge_result.get("hitl_trigger"):
    #         reply_text = ESCALATION_MESSAGE

    return {
        "session_id":    session_id,
        "reply":         reply_text,
        "triage_active": state.get("triage_active", False),
        "judge":         {},   # placeholder — re-enable judge to populate
    }


# ── /health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health() -> dict:
    return {
        "ok":              True,
        "domain":          settings.app_domain,
        "huggingface_llm": bool(settings.huggingface_api_key),
        "groq_stt":        voice_service.supports_server_stt,
        "edge_tts":        voice_service.supports_server_tts,
        "dialect_rag":     False,   # disabled — re-enable when wired back in
        "judge_enabled":   False,   # disabled — re-enable when wired back in
    }


# ── /voice/message ────────────────────────────────────────────────────────────
@app.post("/voice/message")
async def voice_message(request: VoiceRequest) -> dict:
    session_id  = request.session_id or str(uuid.uuid4())
    audio_bytes = base64.b64decode(request.audio_base64) if request.audio_base64 else None

    raw_transcript, detected_lang = await voice_service.transcribe_raw(
        audio_bytes=audio_bytes,
        mime_type=request.mime_type,
    )

    if detected_lang not in ("en", "english"):
        _session_lang[session_id] = detected_lang

    if detected_lang not in ("en", "english") and audio_bytes:
        transcript = await voice_service.translate_to_english(
            audio_bytes=audio_bytes,
            mime_type=request.mime_type,
        )
        if not transcript.strip():
            transcript = raw_transcript
    else:
        transcript = raw_transcript

    target_lang = _session_lang.get(session_id, "en")
    state       = process_with_langgraph(session_id, transcript, "voice_message", target_lang)
    messages    = state.get("messages", [])
    reply_text  = _sanitize_reply(messages[-1].content if messages else '') or "I'm sorry, I couldn't process that."

    # ── Judge evaluation — DISABLED ───────────────────────────────────────────
    # judge = _get_judge()
    # if judge:
    #     judge_result = judge.evaluate(...)
    #     if judge_result.get("hitl_trigger"):
    #         reply_text = ESCALATION_MESSAGE

    audio_reply = await voice_service.synthesize_base64_wav(reply_text)

    return {
        "session_id":        session_id,
        "transcript":        transcript,
        "detected_language": detected_lang,
        "reply":             reply_text,
        "triage_active":     state.get("triage_active", False),
        "audio_base64":      audio_reply,
        "judge":             {},
    }


# ── /ws/call/{session_id} ─────────────────────────────────────────────────────
@app.websocket("/ws/call/{session_id}")
async def call_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    print(f"🔌 [WS] Connected: {session_id}")
    await websocket.send_json({
        "type":       "call_ready",
        "session_id": session_id,
        "server_tts": voice_service.supports_server_tts,
        "server_stt": voice_service.supports_server_stt,
    })

    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except Exception as e:
                print(f"❌ [WS] receive_json failed: {e}")
                break

            message_type = payload.get("type")

            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if message_type == "user_audio":
                audio_b64 = payload.get("audio_base64") or ""
                mime_type = payload.get("mime_type", "audio/webm")
                if not audio_b64:
                    await websocket.send_json({"type": "error", "message": "audio_base64 is empty."})
                    continue

                audio_bytes = base64.b64decode(audio_b64)
                try:
                    raw_transcript, detected_lang = await voice_service.transcribe_raw(
                        audio_bytes=audio_bytes, mime_type=mime_type,
                    )
                except Exception as e:
                    print(f"❌ [WS] transcribe_raw failed: {e}")
                    await websocket.send_json({"type": "error", "message": "Speech recognition failed."})
                    continue

                if detected_lang not in ("en", "english"):
                    _session_lang[session_id] = detected_lang

                if not raw_transcript.strip():
                    continue

                if detected_lang not in ("en", "english"):
                    try:
                        english_transcript = await voice_service.translate_to_english(
                            audio_bytes=audio_bytes, mime_type=mime_type,
                        )
                        if not english_transcript.strip():
                            english_transcript = raw_transcript
                    except Exception as e:
                        english_transcript = raw_transcript
                else:
                    english_transcript = raw_transcript

                await websocket.send_json({
                    "type":              "user_transcript_echo",
                    "text":              english_transcript,
                    "detected_language": detected_lang,
                })
                transcript = english_transcript

            elif message_type == "user_transcript":
                transcript = (payload.get("text") or "").strip()
                if not transcript:
                    await websocket.send_json({"type": "error", "message": "Transcript is empty."})
                    continue
                detected_lang = "en"
            else:
                await websocket.send_json({"type": "error", "message": "Unsupported message type."})
                continue

            # ── Orchestrator (Judge disabled) ─────────────────────────────────
            try:
                target_lang = _session_lang.get(session_id, "en")
                state       = process_with_langgraph(session_id, transcript, "call", target_lang)
                messages    = state.get("messages", [])
                reply_text  = _sanitize_reply(messages[-1].content if messages else '') or "I'm sorry, I couldn't process that."

                # Judge disabled:
                # judge = _get_judge()
                # if judge:
                #     judge_result = judge.evaluate(...)
                #     if judge_result.get("hitl_trigger"):
                #         reply_text = ESCALATION_MESSAGE

                audio_reply = await voice_service.synthesize_base64_wav(reply_text)
                await websocket.send_json({
                    "type":              "assistant_response",
                    "text":              reply_text,
                    "detected_language": target_lang,
                    "triage_active":     state.get("triage_active", False),
                    "audio_base64":      audio_reply,
                    "judge":             {},
                })
            except Exception as e:
                print(f"❌ [WS] Orchestrator/send error: {e}")
                await websocket.send_json({
                    "type":    "error",
                    "message": "Something went wrong. Please try again.",
                })

    except WebSocketDisconnect:
        _session_lang.pop(session_id, None)
        print(f"🔌 [WS] Disconnected: {session_id}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)