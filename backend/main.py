from __future__ import annotations

import base64
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

if __package__ in (None, ""):
    workspace_root = Path(__file__).resolve().parents[1]
    if str(workspace_root) not in sys.path:
        sys.path.insert(0, str(workspace_root))

    from backend.mcp_server import router as mcp_router
    from backend.orchestrator import orchestrator
    from backend.settings import settings
    from backend.voice_service import voice_service
else:
    from .mcp_server import router as mcp_router
    from .orchestrator import orchestrator
    from .settings import settings
    from .voice_service import voice_service


app = FastAPI(title="Medical Concierge Agent")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins or ["*"],
    allow_origin_regex=settings.cors_origin_regex or None,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
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


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "domain": settings.app_domain,
        "google_llm": settings.has_google,
        "groq_stt": voice_service.supports_server_stt,
        "edge_tts": voice_service.supports_server_tts,
    }


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    session_id = request.session_id or str(uuid.uuid4())
    result = orchestrator.process(session_id=session_id, message=request.user_input, channel=request.channel)
    return {
        "session_id": session_id,
        "reply": result.reply,
        "intent": result.intent,
        "action": result.action,
        "metadata": result.metadata,
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
    result = orchestrator.process(session_id=session_id, message=transcript, channel="voice_message")
    audio_reply = await voice_service.synthesize_base64_wav(result.reply)
    return {
        "session_id": session_id,
        "transcript": transcript,
        "reply": result.reply,
        "intent": result.intent,
        "action": result.action,
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

            result = orchestrator.process(session_id=session_id, message=transcript, channel="call")
            audio_reply = await voice_service.synthesize_base64_wav(result.reply)
            await websocket.send_json(
                {
                    "type": "assistant_response",
                    "text": result.reply,
                    "intent": result.intent,
                    "action": result.action,
                    "audio_base64": audio_reply,
                }
            )
    except WebSocketDisconnect:
        return


if __name__ == "__main__":
    import uvicorn

    import_target = "main:app" if __package__ in (None, "") else "backend.main:app"
    uvicorn.run(import_target, host="0.0.0.0", port=8000, reload=True)
