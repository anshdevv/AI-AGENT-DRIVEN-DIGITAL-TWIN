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

# ── Session language store ─────────────────────────────────────────────────────
# Maps session_id → ISO 639-1 language code detected by Whisper.
# Kept in-memory; survives the lifetime of the process (resets on restart).
# If you need persistence across restarts, store in Redis or a DB instead.
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


from langchain_core.messages import HumanMessage
# Make sure to import your context helpers from the orchestrator
from agents.orchestrator import load_booking_context, save_booking_context, orchestrator_graph

def process_with_langgraph(session_id: str, message: str, channel: str, target_lang: str = "en") -> dict:
    """Push a message through LangGraph using session_id as the persistent thread."""
    
    # 1. Load the persistent JSON context for this session
    ctx = load_booking_context(session_id)
    
    # 2. Update the language so Qwen knows exactly how to format the reply
    if target_lang and ctx.get("patient_language") != target_lang:
        ctx["patient_language"] = target_lang
        save_booking_context(session_id, ctx)

    # 3. Setup LangGraph config and state
    config = {"configurable": {"thread_id": session_id}}
    
    input_state = {
        "session_id": session_id,
        "messages": [HumanMessage(content=message)],
        "booking_context": ctx  # Inject the updated context into the graph state
    }

    print(f"\n📨 [API] invoke → thread='{session_id}' | lang='{target_lang}' | msg='{message[:80]}'")
    
    # 4. Run the graph
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

    # ── STT: transcribe in native script, detect language ─────────────────────
    raw_transcript, detected_lang = await voice_service.transcribe_raw(
        audio_bytes=audio_bytes,
        mime_type=request.mime_type,
    )

    # Persist detected language for this session
    if detected_lang not in ("en", "english"):
        _session_lang[session_id] = detected_lang
        print(f"🌐 [Session] Stored lang='{detected_lang}' for session '{session_id}'")

    # ── Translate to English for MedGemma ───────────────────────────────────────
    if detected_lang not in ("en", "english") and audio_bytes:
        transcript = await voice_service.translate_to_english(
            audio_bytes=audio_bytes,
            mime_type=request.mime_type,
        )
        if not transcript.strip():
            transcript = raw_transcript
    else:
        transcript = raw_transcript

    # ── Orchestrator: MedGemma receives clean English ──────────────────────────
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
    print(f"🔌 [WS] Client connected: session_id='{session_id}'")
    await websocket.send_json({
        "type": "call_ready",
        "session_id": session_id,
        "server_tts": voice_service.supports_server_tts,
        "server_stt": voice_service.supports_server_stt,
    })

    try:
        while True:
            try:
                payload = await websocket.receive_json()
            except Exception as e:
                print(f"❌ [WS] receive_json failed: {e} — closing")
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

                # ── 1. TRANSCRIBE (native script, auto-detects en/ur) ──
                try:
                    raw_transcript, detected_lang = await voice_service.transcribe_raw(
                        audio_bytes=audio_bytes,
                        mime_type=mime_type,
                    )
                except Exception as e:
                    print(f"❌ [WS] transcribe_raw failed: {e}")
                    await websocket.send_json({"type": "error", "message": "Speech recognition failed, please try again."})
                    continue

                if detected_lang not in ("en", "english"):
                    _session_lang[session_id] = detected_lang
                    print(f"🌐 [WS] Stored lang='{detected_lang}' for session '{session_id}'")

                print(f"🎤 [WS] Patient spoke ({detected_lang}): {raw_transcript}")

                if not raw_transcript.strip():
                    continue  # Silence, skip turn

                # ── 2. TRANSLATE via Whisper (only if non-English) ──
                if detected_lang not in ("en", "english"):
                    try:
                        english_transcript = await voice_service.translate_to_english(
                            audio_bytes=audio_bytes,
                            mime_type=mime_type,
                        )
                        # If translation came back empty, fall back to raw text
                        if not english_transcript.strip():
                            print("⚠️ [WS] Whisper translation empty — using raw transcript as fallback")
                            english_transcript = raw_transcript
                    except Exception as e:
                        print(f"❌ [WS] translate_to_english failed: {e} — using raw transcript")
                        english_transcript = raw_transcript
                else:
                    english_transcript = raw_transcript

                print(f"🔄 [WS] English for MedGemma: {english_transcript[:120]}")

                # ── 3. ECHO TO FRONTEND (show English so the chat makes sense) ──
                await websocket.send_json({
                    "type": "user_transcript_echo",
                    "text": english_transcript,
                    "detected_language": detected_lang,
                })

                # We set the transcript variable here so the rest of the flow uses the ENGLISH version
                transcript = english_transcript

            elif message_type == "user_transcript":
                # Plain-text fallback (testing / browser without mic)
                transcript = (payload.get("text") or "").strip()
                if not transcript:
                    await websocket.send_json({"type": "error", "message": "Transcript is empty."})
                    continue
                detected_lang = "en"  # text path always assumed English

            else:
                await websocket.send_json({"type": "error", "message": "Unsupported message type."})
                continue

            # ── Orchestrator: Qwen responds directly in patient's language ─────
            try:
                target_lang = _session_lang.get(session_id, "en")
                state = process_with_langgraph(session_id, transcript, "call", target_lang)
                messages = state.get("messages", [])
                reply_text = messages[-1].content if messages else "I'm sorry, I couldn't process that."

                # ── TTS: Qwen already responded in patient's language ─────────
                audio_reply = await voice_service.synthesize_base64_wav(reply_text)

                await websocket.send_json({
                    "type": "assistant_response",
                    "text": reply_text,
                    "detected_language": target_lang,
                    "triage_active": state.get("triage_active", False),
                    "audio_base64": audio_reply,
                })
            except Exception as e:
                print(f"❌ [WS] Orchestrator/send error: {e}")
                await websocket.send_json({"type": "error", "message": "Something went wrong processing your message. Please try again."})

    except WebSocketDisconnect:
        # Clean up session language on disconnect
        _session_lang.pop(session_id, None)
        print(f"🔌 [WS] Client disconnected: session_id='{session_id}'")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)