# main.py
from __future__ import annotations

import base64
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage

from config import settings
from agents.orchestrator import orchestrator_graph
from agents.voice_agent import voice_service

app = FastAPI(title="Medical Concierge Agent (LangGraph Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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


def process_with_langgraph(session_id: str, message: str, channel: str) -> dict:
    """Push a message through LangGraph using session_id as the persistent thread."""
    config = {"configurable": {"thread_id": session_id}}

    # ✅ FIX: Only pass the NEW message here.
    # LangGraph's MemorySaver already holds the full history for this thread_id.
    # If you also pass old messages here, they get appended AGAIN → message count doubles.
    input_state = {
        "messages": [HumanMessage(content=message)],
        # Do NOT pass session_id or other state fields here — MemorySaver owns them.
    }

    print(f"\n📨 [API] invoke → thread='{session_id}' | msg='{message[:80]}'")
    result_state = orchestrator_graph.invoke(input_state, config=config)
    return result_state


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    session_id = request.session_id or str(uuid.uuid4())

    if not request.session_id:
        print(f"🆕 [API] New session minted: {session_id}")
    else:
        print(f"♻️  [API] Continuing session: {session_id}")

    state = process_with_langgraph(session_id, request.user_input, request.channel)

    messages = state.get("messages", [])

    # ✅ Print the full message array so you can see what's in it
    print(f"\n📋 [API] Full message array ({len(messages)} messages):")
    for i, msg in enumerate(messages):
        role = msg.__class__.__name__.replace("Message", "")
        content_preview = str(msg.content)[:120].replace("\n", " ")
        print(f"   [{i}] {role}: {content_preview}")

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

    transcript = await voice_service.transcribe_audio(
        audio_bytes=audio_bytes,
        mime_type=request.mime_type,
        transcript_hint=request.transcript,
    )

    state = process_with_langgraph(session_id, transcript, "voice_message")
    messages = state.get("messages", [])
    reply_text = messages[-1].content if messages else "I'm sorry, I couldn't process that."
    audio_reply = await voice_service.synthesize_base64_wav(reply_text)

    return {
        "session_id": session_id,
        "transcript": transcript,
        "reply": reply_text,
        "triage_active": state.get("triage_active", False),
        "audio_base64": audio_reply,
    }


@app.websocket("/ws/call/{session_id}")
async def call_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    print(f"🔌 [WS] Client connected: session_id='{session_id}'")
    await websocket.send_json({
        "type": "call_ready",
        "session_id": session_id,
        "server_tts": voice_service.supports_server_tts,
        "server_stt": voice_service.supports_server_stt,
    })
    try:
        while True:
            payload = await websocket.receive_json()
            message_type = payload.get("type")

            if message_type == "ping":
                await websocket.send_json({"type": "pong"})
                continue

            if message_type != "user_transcript":
                await websocket.send_json({"type": "error", "message": "Unsupported message type."})
                continue

            transcript = (payload.get("text") or "").strip()
            if not transcript:
                await websocket.send_json({"type": "error", "message": "Transcript is empty."})
                continue

            state = process_with_langgraph(session_id, transcript, "call")
            messages = state.get("messages", [])
            reply_text = messages[-1].content if messages else "I'm sorry, I couldn't process that."
            audio_reply = await voice_service.synthesize_base64_wav(reply_text)

            await websocket.send_json({
                "type": "assistant_response",
                "text": reply_text,
                "triage_active": state.get("triage_active", False),
                "audio_base64": audio_reply,
            })
    except WebSocketDisconnect:
        print(f"🔌 [WS] Client disconnected: session_id='{session_id}'")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)