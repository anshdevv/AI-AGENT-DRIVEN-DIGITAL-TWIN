// WhatsApp bridge (unofficial, Baileys) for the Medical Concierge.
// Relays inbound patient WhatsApp messages to the FastAPI /chat endpoint
// and sends the agent's reply back over WhatsApp.
//
// Session model: session_id = sender JID (e.g. 923001234567@s.whatsapp.net).
// The backend sanitizes this into a per-patient context file, so each
// patient keeps persistent triage/booking context automatically.
//
// ⚠️  Unofficial: this automates WhatsApp Web and violates WhatsApp ToS.
//     Use a secondary/throwaway number — the number can be banned.

import makeWASocket, {
  useMultiFileAuthState,
  DisconnectReason,
  fetchLatestBaileysVersion,
  downloadMediaMessage,
} from "@whiskeysockets/baileys";
import axios from "axios";
import pino from "pino";
import qrcode from "qrcode-terminal";

// One or more backend instances. Set BACKEND_URLS to a comma-separated list to
// scale out, e.g. "http://localhost:8000,http://localhost:8001,http://localhost:8002".
// Falls back to single BACKEND_URL (default localhost:8000).
const BACKENDS = (process.env.BACKEND_URLS || process.env.BACKEND_URL || "http://localhost:8000")
  .split(",")
  .map((u) => u.trim().replace(/\/$/, ""))
  .filter(Boolean);

const AUTH_DIR = process.env.WA_AUTH_DIR || "./wa_auth";
const REQUEST_TIMEOUT_MS = Number(process.env.WA_REQUEST_TIMEOUT_MS || 120000);
// Set to "1" to also send a synthesized audio reply back as a WhatsApp voice note.
const SEND_AUDIO_REPLY = process.env.WA_SEND_AUDIO_REPLY === "1";

const logger = pino({ level: process.env.LOG_LEVEL || "warn" });

// Sticky routing: a given session_id (patient) ALWAYS maps to the same backend,
// so that backend's in-memory state (MemorySaver, handoff/lang dicts) stays valid.
function hashString(s) {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = (h * 31 + s.charCodeAt(i)) | 0; // 32-bit rolling hash
  }
  return Math.abs(h);
}

function backendFor(sessionId) {
  return BACKENDS[hashString(sessionId) % BACKENDS.length];
}

// Unwrap ephemeral / view-once envelopes so we see the real content.
function innerMessage(m) {
  if (!m) return m;
  return (
    m.ephemeralMessage?.message ||
    m.viewOnceMessage?.message ||
    m.viewOnceMessageV2?.message ||
    m.documentWithCaptionMessage?.message ||
    m
  );
}

// Extract plain text from the various WhatsApp message shapes.
function extractText(msg) {
  const m = innerMessage(msg.message);
  if (!m) return "";
  return (
    m.conversation ||
    m.extendedTextMessage?.text ||
    m.imageMessage?.caption ||
    m.videoMessage?.caption ||
    m.buttonsResponseMessage?.selectedButtonId ||
    m.listResponseMessage?.singleSelectReply?.selectedRowId ||
    ""
  ).trim();
}

// Return the audio (voice note) message object if present.
function audioPart(msg) {
  const m = innerMessage(msg.message);
  return m?.audioMessage || null;
}

async function callBackend(sessionId, text) {
  const base = backendFor(sessionId);
  const { data } = await axios.post(
    `${base}/chat`,
    { session_id: sessionId, user_input: text, channel: "whatsapp" },
    { timeout: REQUEST_TIMEOUT_MS, headers: { "Content-Type": "application/json" } }
  );
  return data; // { session_id, reply, triage_active, human_handoff }
}

// Download a WhatsApp voice note, send it to /voice/message (STT + agent + TTS).
async function callVoiceBackend(sock, msg, sessionId) {
  const base = backendFor(sessionId);
  const buffer = await downloadMediaMessage(
    msg,
    "buffer",
    {},
    { logger, reuploadRequest: sock.updateMediaMessage }
  );
  const audio = audioPart(msg);
  const mime = (audio?.mimetype || "audio/ogg").split(";")[0].trim();
  const { data } = await axios.post(
    `${base}/voice/message`,
    {
      session_id: sessionId,
      audio_base64: buffer.toString("base64"),
      mime_type: mime,
    },
    { timeout: REQUEST_TIMEOUT_MS, headers: { "Content-Type": "application/json" } }
  );
  return data; // { session_id, transcript, detected_language, reply, audio_base64 }
}

// ── Per-session FIFO queues ──────────────────────────────────────────────
// Messages from the SAME patient run in order (so their booking_context file
// is never read/written concurrently); DIFFERENT patients run in parallel.
const sessionQueues = new Map();

function enqueue(sessionId, task) {
  const prev = sessionQueues.get(sessionId) || Promise.resolve();
  const next = prev.then(task).catch((e) => console.error("❌ task error:", e.message));
  // Clean up the map entry once this is the tail of the chain.
  sessionQueues.set(
    sessionId,
    next.finally(() => {
      if (sessionQueues.get(sessionId) === next) sessionQueues.delete(sessionId);
    })
  );
}

// Handle one inbound message end-to-end (download/route/reply).
async function processMessage(sock, msg, jid, sessionId, text, audio) {
  await sock.sendPresenceUpdate("composing", jid);

  console.log(`   ↳ routed to ${backendFor(sessionId)}`);

  let reply = "";
  let audioReplyB64 = "";
  try {
    if (audio) {
      console.log(`📥🎙️ [${sessionId}] voice note (${audio.seconds || "?"}s)`);
      const data = await callVoiceBackend(sock, msg, sessionId);
      console.log(`   📝 transcript: "${data?.transcript || ""}" lang=${data?.detected_language}`);
      reply = (data?.reply || "").trim();
      audioReplyB64 = data?.audio_base64 || "";
    } else {
      console.log(`📥 [${sessionId}] ${text}`);
      const data = await callBackend(sessionId, text);
      reply = (data?.reply || "").trim();
    }
  } catch (err) {
    console.error(`❌ Backend error for ${sessionId}:`, err.message);
    reply = "Sorry, I'm having trouble right now. Please try again in a moment.";
  }

  await sock.sendPresenceUpdate("paused", jid);
  if (reply) {
    await sock.sendMessage(jid, { text: reply });
    console.log(`📤 [${sessionId}] ${reply.slice(0, 80)}`);
  }
  if (SEND_AUDIO_REPLY && audioReplyB64) {
    try {
      await sock.sendMessage(jid, {
        audio: Buffer.from(audioReplyB64, "base64"),
        mimetype: "audio/mp4",
        ptt: true,
      });
      console.log(`📤🎙️ [${sessionId}] audio reply sent`);
    } catch (err) {
      console.error(`⚠️  audio reply failed:`, err.message);
    }
  }
}

async function start() {
  const { state, saveCreds } = await useMultiFileAuthState(AUTH_DIR);
  const { version } = await fetchLatestBaileysVersion();

  const sock = makeWASocket({
    version,
    auth: state,
    logger,
    printQRInTerminal: false, // we render the QR ourselves below
    markOnlineOnConnect: false,
  });

  sock.ev.on("creds.update", saveCreds);

  sock.ev.on("connection.update", (update) => {
    const { connection, lastDisconnect, qr } = update;
    if (qr) {
      console.log("\n📱 Scan this QR with the bot's WhatsApp (Linked Devices):\n");
      qrcode.generate(qr, { small: true });
    }
    if (connection === "open") {
      console.log(`✅ WhatsApp connected. Backends (${BACKENDS.length}): ${BACKENDS.join(", ")}`);
    }
    if (connection === "close") {
      const statusCode = lastDisconnect?.error?.output?.statusCode;
      const loggedOut = statusCode === DisconnectReason.loggedOut;
      console.log(`⚠️  Connection closed (code=${statusCode}). loggedOut=${loggedOut}`);
      if (!loggedOut) {
        console.log("🔄 Reconnecting...");
        start();
      } else {
        console.log("❌ Logged out. Delete the auth folder and re-scan the QR.");
      }
    }
  });

  sock.ev.on("messages.upsert", ({ messages, type }) => {
    console.log(`🔔 upsert type=${type} count=${messages.length}`);

    for (const msg of messages) {
      const jid = msg.key?.remoteJid || "";            // may be @lid or @s.whatsapp.net
      const altJid = msg.key?.remoteJidAlt || "";       // phone-number JID when jid is @lid
      const text = extractText(msg);
      const audio = audioPart(msg);
      console.log(`   • jid=${jid} alt=${altJid} fromMe=${msg.key?.fromMe} text="${text}" audio=${!!audio}`);

      // Ignore our own messages, groups, status broadcasts, and newsletters.
      if (msg.key?.fromMe) continue;
      const isUser = jid.endsWith("@s.whatsapp.net") || jid.endsWith("@lid");
      if (!isUser) continue;
      if (!text && !audio) continue;

      // Stable per-patient id: prefer the phone-number JID; fall back to the LID.
      const sessionId = altJid && altJid.endsWith("@s.whatsapp.net") ? altJid : jid;

      // Enqueue without awaiting: different patients run in parallel,
      // a single patient's messages run in arrival order.
      enqueue(sessionId, () => processMessage(sock, msg, jid, sessionId, text, audio));
    }
  });

  // WhatsApp calls cannot be answered by unofficial libs — auto-reject and guide
  // the patient to send a voice note instead.
  sock.ev.on("call", async (calls) => {
    for (const call of calls) {
      if (call.status !== "offer") continue;
      console.log(`📞 Incoming call from ${call.from} — rejecting, suggesting voice note`);
      try {
        await sock.rejectCall(call.id, call.from);
        await sock.sendMessage(call.from, {
          text: "I can't take live calls here. Please type your message or send a voice note and I'll help you right away.",
        });
      } catch (err) {
        console.error("⚠️  call handling failed:", err.message);
      }
    }
  });
}

start().catch((err) => {
  console.error("Fatal:", err);
  process.exit(1);
});
