# WhatsApp Bridge (Unofficial)

Relays patient WhatsApp messages to the Medical Concierge `/chat` pipeline using
[Baileys](https://github.com/WhiskeySockets/Baileys) (automates WhatsApp Web).

> ⚠️ **Unofficial / ToS risk.** This is not the WhatsApp Business API. It automates a
> logged-in WhatsApp Web session and violates WhatsApp's Terms of Service. The number
> **can be banned**. Use a secondary / throwaway number, not a personal one.

## How it works
```
Patient WhatsApp ⇄ this bridge (Node) ⇄ POST /chat (FastAPI) ⇄ LangGraph agents
```
- `session_id` = the patient's WhatsApp JID (e.g. `923001234567@s.whatsapp.net`), so each
  patient keeps persistent triage/booking context (`booking_context/*.json`) automatically.
- `channel` is sent as `"whatsapp"`.

## Setup
1. Start the backend first:
   ```bash
   cd ..        # newBackend/
   python main.py   # serves on http://localhost:8000
   ```
2. Install and run the bridge:
   ```bash
   npm install
   node index.js
   ```
3. A QR code prints in the terminal. On the bot's phone open
   **WhatsApp → Settings → Linked Devices → Link a Device** and scan it.
   Auth is saved to `./wa_auth/` so you only scan once.

## Configuration (env vars)
| Var | Default | Purpose |
|-----|---------|---------|
| `BACKEND_URLS` | (unset) | Comma-separated list of backends for scale-out (see below) |
| `BACKEND_URL` | `http://localhost:8000` | Single backend (used if `BACKEND_URLS` unset) |
| `WA_AUTH_DIR` | `./wa_auth` | Where the WhatsApp session is stored |
| `WA_REQUEST_TIMEOUT_MS` | `120000` | Backend request timeout (agents can be slow) |
| `WA_SEND_AUDIO_REPLY` | `0` | `1` = also send a synthesized voice-note reply |
| `LOG_LEVEL` | `warn` | Baileys/pino log level |

## Scaling to multiple backends (sticky routing)
Run several backend instances on different ports, then point the bridge at all of them:

```bash
# terminal 1..N (each its own port)
cd ..
python main.py                 # port 8000
# (set PORT/host per instance, e.g. uvicorn main:app --port 8001)

# bridge
set BACKEND_URLS=http://localhost:8000,http://localhost:8001,http://localhost:8002   # Windows
node index.js
```

The bridge hashes each patient's number to **one** backend and always sends that
patient there (`backendFor()` in `index.js`). This is **sticky routing**: it keeps
each backend's in-memory state (LangGraph `MemorySaver`, handoff/language dicts)
consistent for a given patient.

> ⚠️ Sticky routing relies on the same `session_id` always hashing to the same
> backend. If you add/remove backends from the list, existing patients may be
> re-routed and lose their in-memory thread (booking_context on disk is still shared).
> All backends share the same Ollama/MedGemma — to parallelize triage, raise
> `OLLAMA_NUM_PARALLEL` or run more Ollama instances.

## Notes
- Only 1:1 text chats are handled; groups, status broadcasts and own messages are ignored.
- To re-link a different number, delete `wa_auth/` and restart.
- Human handoff: the backend returns the handoff text in `reply`, which is forwarded as-is.
  (A CSR pushing replies *out* to WhatsApp would need a future `/whatsapp/outbound` endpoint.)
