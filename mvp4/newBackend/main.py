# main.py
from __future__ import annotations

import base64
import uuid
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 1. Import from our new centralized config
from config import settings

# 2. Import from our new modular agents folder
from agents.orchestrator import orchestrator_graph
from agents.mcp_server import router as mcp_router
from agents.voice_agent import voice_service


app = FastAPI(title="Medical Concierge Agent (LangGraph Edition)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach the MCP endpoints
app.include_router(mcp_router)


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
    """Helper to push messages through LangGraph and extract the new state."""
    config = {"configurable": {"thread_id": session_id}}
    input_state = {
        "session_id": session_id,
        "user_input": message,
        "channel": channel
    }
    # LangGraph automatically handles caching/history via the thread_id
    result_state = orchestrator_graph.invoke(input_state, config=config)
    return result_state


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "domain": settings.app_domain,
        "huggingface_llm": bool(settings.huggingface_api_key),
        "groq_stt": voice_service.supports_server_stt,
        "edge_tts": voice_service.supports_server_tts,
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    session_id = request.session_id or str(uuid.uuid4())
    state = process_with_langgraph(session_id, request.user_input, request.channel)
    
    return {
        "session_id": session_id,
        "reply": state.get("response", "I'm sorry, I couldn't process that."),
        "intent": state.get("intent", "unknown"),
        "action": state.get("booking_step", "reply"), # Adjust based on your final state schema
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
    reply_text = state.get("response", "I'm sorry, I couldn't process that.")
    audio_reply = await voice_service.synthesize_base64_wav(reply_text)
    
    return {
        "session_id": session_id,
        "transcript": transcript,
        "reply": reply_text,
        "intent": state.get("intent", "unknown"),
        "action": state.get("booking_step", "reply"),
        "audio_base64": audio_reply,
    }


@app.websocket("/ws/call/{session_id}")
async def call_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    await websocket.send_json(
        {
            "type": "call_ready",
            "session_id": session_id,
            "server_tts": voice_service.supports_server_tts,
            "server_stt": voice_service.supports_server_stt,
        }
    )
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
            reply_text = state.get("response", "I'm sorry, I couldn't process that.")
            audio_reply = await voice_service.synthesize_base64_wav(reply_text)
            
            await websocket.send_json(
                {
                    "type": "assistant_response",
                    "text": reply_text,
                    "intent": state.get("intent", "unknown"),
                    "action": state.get("booking_step", "reply"),
                    "audio_base64": audio_reply,
                }
            )
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)