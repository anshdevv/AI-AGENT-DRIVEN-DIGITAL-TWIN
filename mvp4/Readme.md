# Medical Concierge MVP

This project now uses a LangGraph-based medical concierge instead of the older manual customer-service router.

## What It Handles

- chat
- voice notes
- live call transcripts
- symptom-to-specialty routing
- doctor info and schedule lookups
- appointment booking
- safe pre-visit triage

The assistant does **not** prescribe medication or handle emergency advice beyond urgent escalation.

## Backend Flow

1. A hybrid intent classifier runs first.
   It uses heuristics and entity extraction for speed, with an LLM only as backup.
2. LangGraph routes the turn to the right specialist agent.
3. Agents share the same session context:
   recommendation, doctor info, booking, triage, FAQ, safety, and handoff.
4. MCP tools power the business actions and can also be called over `/mcp`.

## RAG + Tools

- FAQ RAG: `backend/rag/faq`
- symptom-to-specialization map: `backend/rag/mapping/symptoms_to_specialization.md`
- triage question flows: `backend/rag/question_flows`

Key MCP tools:

- `match_symptoms_to_specialization`
- `recommend_service_provider`
- `get_doctor_profile`
- `find_provider_availability`
- `create_booking`
- `get_triage_flow`

## API Surface

- `POST /chat`
- `POST /voice/message`
- `WS /ws/call/{session_id}`
- `GET /mcp`
- `POST /mcp`

All three user channels share the same backend session state.

## Voice

The app keeps the same voice-note and live-call surfaces, and the backend voice service still supports server-side STT/TTS when the optional keys and packages are available. The old `voice chit` folder remains as reference material for voice handling ideas.

## Environment

Detected keys used by this repo:

- `SUPABASE_URL`
- `SUPABASE_KEY`
- `GOOGLE_API_KEY`
- `GROQ_API_KEY`
- `ELEVENLABS` or `ElevenLabs`
- `CLASSIFIER_MODEL`
- `ACTION_MODEL`
- `TRIAGE_MODEL`
- `APP_DOMAIN`
- `CORS_ORIGINS`

## Run

Use the project virtual environment so `langgraph` and the other pinned packages are available.

```bash
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe api.py
```

For the frontend:

```bash
cd frontend
npm install
npm start
```
