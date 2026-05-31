# Medical Concierge MVP

A LangGraph-powered medical concierge prototype for conversational triage, booking, and care routing.

## Scope

- Handles chat, voice note, and live call interactions.
- Supports symptom-to-specialist recommendation, doctor lookup, appointment booking, and pre-visit triage.
- Preserves session context per patient via `newBackend/booking_context/*.json`.
- Includes human handoff and CSR/doctor review flows.
- Does not prescribe medication or provide emergency medical diagnosis.

## Architecture

- Backend: `newBackend/main.py` (FastAPI)
- Orchestration: `newBackend/agents/orchestrator.py` (LangGraph)
- Agent logic: `newBackend/agents/`
- Voice pipeline: `newBackend/agents/voice_agent.py`
- WhatsApp bridge: `newBackend/whatsapp/` (optional, unofficial)

## Primary API Endpoints

- `POST /chat` — text conversation turn
- `POST /voice/message` — voice note upload, STT, optional translation, and reply
- `WS /ws/call/{session_id}` — live audio/WebSocket call pipeline
- `POST /human/message` — CSR/human agent sends a message into a patient session
- `GET /human/messages/{session_id}` — read queued human messages for a session
- `GET /health` — service health and integration status
- `GET /csr/handoffs` — CSR handoff queue
- `GET /csr/handoffs/{session_id}` — handoff session detail

## Core Features

- Persistent patient session state for booking, triage, and transcript history
- Agent-level orchestration with LangGraph for routing and multi-step medical workflows
- Voice note transcription and synthesis using `newBackend/agents/voice_agent.py`
- Human handoff detection and support for explicit CSR/doctor messaging
- Booking context stored as JSON in `newBackend/booking_context/`

## Deployment

### Backend

```bash
cd newBackend
python -m pip install -r requirements.txt
python main.py
```

Or from repository root:

```bash
python api.py
```

### Frontend

```bash
cd frontend
npm install
npm start
```

## Configuration

The backend loads `.env` from `newBackend/.env` and supports these environment variables:

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `HUGGINGFACE_API_KEY`
- `GROQ_API_KEY`
- `ELEVENLABS_API_KEY` / `ElevenLabs`
- `ELEVENLABS_STT_MODEL`
- `ELEVENLABS_TTS_MODEL`
- `ACTION_MODEL`
- `APP_DOMAIN`
- `CORS_ORIGINS`
- `ADMIN_USERNAME`
- `ADMIN_PASSWORD`
- `CSR_USERNAME`
- `CSR_PASSWORD`
- `DOCTOR_PORTAL_PASSWORD`

## Notes

- `newBackend/booking_context/` stores per-session JSON to keep patient data and transcripts between requests.
- The WhatsApp bridge in `newBackend/whatsapp/` is optional and uses an unofficial Web client; it is not production-safe.
- Voice STT/TTS is enabled only when supported credentials and packages are configured.
- The project is a prototype; the current focus is on conversational booking and triage workflows rather than full clinical decision support.
