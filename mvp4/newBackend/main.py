from __future__ import annotations

import base64
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from config import settings
from agents.orchestrator import load_booking_context, save_booking_context, orchestrator_graph
from agents.voice_agent import voice_service
from agents.pipecat_pipeline import PipecatCallPipeline


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


class ChatRequest(BaseModel):
    session_id: str | None = None
    user_input: str = Field(..., min_length=1)
    channel: str = "chat"


class VoiceRequest(BaseModel):
    session_id: str | None = None
    transcript: str | None = None
    audio_base64: str | None = None
    mime_type: str = "audio/webm"


def process_with_langgraph(session_id: str, message: str, channel: str, target_lang: str = "en") -> dict:
    """Push a message through LangGraph using session_id as the persistent thread."""
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
    return orchestrator_graph.invoke(input_state, config=config)


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    session_id = request.session_id or str(uuid.uuid4())
    state = process_with_langgraph(session_id, request.user_input, request.channel)
    messages = state.get("messages", [])
    reply_text = messages[-1].content if messages else "I couldn't process that."

    return {
        "session_id": session_id,
        "reply": reply_text,
        "triage_active": state.get("triage_active", False),
    }


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "domain": settings.app_domain,
        "huggingface_llm": bool(settings.huggingface_api_key),
        "groq_stt": voice_service.supports_server_stt,
        "edge_tts": voice_service.supports_server_tts,
    }


@app.post("/voice/message")
async def voice_message(request: VoiceRequest) -> dict:
    session_id = request.session_id or str(uuid.uuid4())
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
    state = process_with_langgraph(session_id, transcript, "voice_message", target_lang)
    messages = state.get("messages", [])
    reply_text = messages[-1].content if messages else "I'm sorry, I couldn't process that."
    audio_reply = await voice_service.synthesize_base64_wav(reply_text)

    return {
        "session_id": session_id,
        "transcript": transcript,
        "detected_language": detected_lang,
        "reply": reply_text,
        "triage_active": state.get("triage_active", False),
        "audio_base64": audio_reply,
    }


@app.websocket("/ws/call/{session_id}")
async def call_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    print(f"[WS] Client connected: session_id='{session_id}'")

    pipeline = PipecatCallPipeline(
        websocket=websocket,
        session_id=session_id,
        voice_service=voice_service,
        process_with_langgraph=process_with_langgraph,
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
