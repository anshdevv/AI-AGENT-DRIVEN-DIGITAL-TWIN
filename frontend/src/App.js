import { startTransition, useEffect, useRef, useState } from "react";
import "./index.css";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";
const WS_BASE = process.env.REACT_APP_WS_BASE || "";
const GATE_START = Number(process.env.REACT_APP_CALL_GATE_START || 18);
const GATE_STOP = Number(process.env.REACT_APP_CALL_GATE_STOP || 12);
const VAD_MIN = Number(process.env.REACT_APP_CALL_VAD_MIN || 8);
const INTERRUPT = Number(process.env.REACT_APP_CALL_INTERRUPT || 13);
const VAD_TICK_MS = Number(process.env.REACT_APP_CALL_VAD_TICK_MS || 60);
const SILENCE_HOLD_MS = Number(process.env.REACT_APP_CALL_SILENCE_HOLD_MS || 420);
const MAX_CHUNK_MS = Number(process.env.REACT_APP_CALL_MAX_CHUNK_MS || 10000);

const newId = () => window.crypto?.randomUUID?.() || `s-${Date.now()}`;
const b64 = (blob) =>
  new Promise((r, j) => {
    const f = new FileReader();
    f.onloadend = () => r(String(f.result || "").split(",")[1] || "");
    f.onerror = j;
    f.readAsDataURL(blob);
  });
const trimSlash = (s) => String(s || "").replace(/\/+$/, "");
const wsOrigin = () => {
  const explicit = trimSlash(WS_BASE);
  if (explicit) return explicit;
  try {
    const raw = String(API_BASE || "").trim();
    const abs = raw.startsWith("http://") || raw.startsWith("https://") ? raw : `${window.location.origin}${raw.startsWith("/") ? "" : "/"}${raw}`;
    const u = new URL(abs);
    return `${u.protocol === "https:" ? "wss" : "ws"}://${u.host}`;
  } catch {
    return `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`;
  }
};
const wsCallUrl = (sessionId) => `${wsOrigin()}/ws/call/${encodeURIComponent(sessionId)}`;

export default function App() {
  const [sessionId] = useState(newId);
  const [messages, setMessages] = useState([
    { id: "w", sender: "bot", channel: "system", text: "Medical concierge is online." },
  ]);
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [callOn, setCallOn] = useState(false);
  const [callStatus, setCallStatus] = useState("Offline");
  const [recording, setRecording] = useState(false);
  const [handoff, setHandoff] = useState(false);

  const endRef = useRef(null);
  const wsRef = useRef(null);
  const pingRef = useRef(null);
  const callStreamRef = useRef(null);
  const callRecRef = useRef(null);
  const callCtxRef = useRef(null);
  const callActiveRef = useRef(false);
  const callConnectingRef = useRef(false);
  const audioRef = useRef(null);
  const speakingRef = useRef(false);
  const genRef = useRef(0);
  const micRecRef = useRef(null);
  const micChunksRef = useRef([]);

  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages, busy, callStatus]);
  useEffect(() => {
    return () => {
      callActiveRef.current = false;
      const rec = callRecRef.current;
      if (rec && rec.state !== "inactive") {
        rec.ondataavailable = null;
        rec.stop();
      }
      const mrec = micRecRef.current;
      if (mrec && mrec.state === "recording") mrec.stop();
      stopAudio();
      if (pingRef.current) clearInterval(pingRef.current);
      pingRef.current = null;
      const s = callStreamRef.current;
      if (s) s.getTracks().forEach((t) => t.stop());
      callStreamRef.current = null;
      if (callCtxRef.current) {
        try {
          callCtxRef.current.close();
        } catch (_) {}
      }
      wsRef.current?.close();
    };
  }, []);

  const add = (sender, textMsg, channel = "chat") =>
    startTransition(() =>
      setMessages((m) => [...m, { id: `${Date.now()}-${Math.random()}`, sender, text: textMsg, channel }])
    );

  const stopAudio = () => {
    const a = audioRef.current;
    if (!a) return;
    try {
      a.pause();
    } catch (_) {}
    a.onended = null;
    a.onerror = null;
    audioRef.current = null;
    speakingRef.current = false;
  };

  const playAudio = (audio_base64, generation_id) => {
    if (!audio_base64) return;
    stopAudio();
    genRef.current = Number(generation_id || 0);
    const a = new Audio(`data:audio/mp3;base64,${audio_base64}`);
    audioRef.current = a;
    speakingRef.current = true;
    const done = () => {
      if (audioRef.current === a) audioRef.current = null;
      speakingRef.current = false;
      if (genRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "assistant_playback_done", generation_id: genRef.current }));
      }
    };
    a.onended = done;
    a.onerror = done;
    a.play().catch(done);
  };

  const sendChat = async () => {
    const t = text.trim();
    if (!t || busy) return;
    add("user", t);
    setText("");
    setBusy(true);
    try {
      const r = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, user_input: t, channel: "chat" }),
      });
      const d = await r.json();
      setHandoff(Boolean(d.human_handoff));
      add("bot", d.reply, d.human_handoff ? "human" : "chat");
    } catch {
      add("bot", "Backend unavailable.", "error");
    } finally {
      setBusy(false);
    }
  };

  const startVoiceNote = async () => {
    if (!navigator.mediaDevices?.getUserMedia || recording) return;
    try {
      const s = await navigator.mediaDevices.getUserMedia({ audio: true });
      micChunksRef.current = [];
      const rec = new MediaRecorder(s);
      micRecRef.current = rec;
      setRecording(true);
      rec.ondataavailable = (e) => e.data?.size && micChunksRef.current.push(e.data);
      rec.onstop = async () => {
        setRecording(false);
        s.getTracks().forEach((t) => t.stop());
        const blob = new Blob(micChunksRef.current, { type: rec.mimeType || "audio/webm" });
        const audio_base64 = await b64(blob);
        setBusy(true);
        add("user", "Voice note", "voice");
        try {
          const r = await fetch(`${API_BASE}/voice/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, audio_base64, mime_type: blob.type }),
          });
          const d = await r.json();
          setHandoff(Boolean(d.human_handoff));
          if (d.transcript) add("user", d.transcript, "voice");
          add("bot", d.reply, d.human_handoff ? "human" : "voice");
          playAudio(d.audio_base64, 0);
        } catch {
          add("bot", "Voice processing failed.", "error");
        } finally {
          setBusy(false);
        }
      };
      rec.start();
    } catch {
      add("bot", "Mic permission denied.", "error");
    }
  };

  const stopVoiceNote = (silent = false) => {
    const rec = micRecRef.current;
    if (rec && rec.state === "recording") rec.stop();
    if (!silent) setRecording(false);
  };

  const startCallRecorder = (stream) => {
    const prev = callRecRef.current;
    if (prev && prev.state !== "inactive") {
      prev.ondataavailable = null;
      prev.stop();
    }
    if (callCtxRef.current) {
      try {
        callCtxRef.current.close();
      } catch (_) {}
    }

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    callCtxRef.current = ctx;
    const an = ctx.createAnalyser();
    an.fftSize = 256;
    ctx.createMediaStreamSource(stream).connect(an);
    const arr = new Uint8Array(an.frequencyBinCount);
    let rec = null;
    let chunks = [];
    let peak = 0;
    let speaking = false;
    let stopping = false;
    let silenceMs = 0;
    let startedAt = 0;
    let lastInt = 0;
    const type = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"].find((t) => MediaRecorder.isTypeSupported(t)) || "";

    const stopChunk = () => {
      if (!rec || !speaking || stopping) return;
      stopping = true;
      if (rec.state === "recording") rec.stop();
    };

    const startChunk = () => {
      if (speaking || !callActiveRef.current || wsRef.current?.readyState !== WebSocket.OPEN) return;
      chunks = [];
      peak = 0;
      silenceMs = 0;
      startedAt = Date.now();
      rec = new MediaRecorder(stream, type ? { mimeType: type } : {});
      callRecRef.current = rec;
      speaking = true;
      stopping = false;
      rec.ondataavailable = (e) => e.data?.size && chunks.push(e.data);
      rec.onstop = async () => {
        const local = chunks;
        const p = peak;
        chunks = [];
        peak = 0;
        silenceMs = 0;
        speaking = false;
        stopping = false;
        if (!callActiveRef.current || wsRef.current?.readyState !== WebSocket.OPEN || !local.length) return;
        const blob = new Blob(local, { type: rec?.mimeType || type || "audio/webm" });
        if (blob.size < 1000 || p < VAD_MIN) return;
        wsRef.current.send(JSON.stringify({ type: "user_audio", audio_base64: await b64(blob), mime_type: blob.type }));
      };
      rec.start();
    };

    const tick = setInterval(() => {
      if (!callActiveRef.current || !callStreamRef.current) {
        stopChunk();
        clearInterval(tick);
        return;
      }
      an.getByteFrequencyData(arr);
      const avg = arr.reduce((a, b) => a + b, 0) / arr.length;
      if (avg > peak) peak = avg;
      if (
        speakingRef.current &&
        avg >= INTERRUPT &&
        Date.now() - lastInt > 350 &&
        wsRef.current?.readyState === WebSocket.OPEN
      ) {
        lastInt = Date.now();
        wsRef.current.send(JSON.stringify({ type: "interrupt" }));
        stopAudio();
      }
      if (!speaking) {
        if (avg >= GATE_START) startChunk();
        return;
      }
      silenceMs = avg <= GATE_STOP ? silenceMs + VAD_TICK_MS : 0;
      const elapsed = Date.now() - startedAt;
      if (silenceMs >= SILENCE_HOLD_MS || elapsed >= MAX_CHUNK_MS) stopChunk();
    }, VAD_TICK_MS);
  };

  const startCall = async () => {
    if (callActiveRef.current || callConnectingRef.current) return;
    callConnectingRef.current = true;
    setCallStatus("Connecting...");
    const ws = new WebSocket(wsCallUrl(sessionId));
    wsRef.current = ws;
    ws.onopen = async () => {
      callConnectingRef.current = false;
      callActiveRef.current = true;
      setCallOn(true);
      setCallStatus("Connected");
      pingRef.current && clearInterval(pingRef.current);
      pingRef.current = setInterval(() => wsRef.current?.readyState === WebSocket.OPEN && wsRef.current.send('{"type":"ping"}'), 12000);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
        callStreamRef.current = stream;
        startCallRecorder(stream);
      } catch {
        setCallStatus("Mic unavailable");
      }
    };
    ws.onmessage = (e) => {
      const p = JSON.parse(e.data);
      if (p.type === "call_ready") setCallStatus("Live");
      if (p.type === "user_transcript_echo") add("user", p.text, "call");
      if (p.type === "assistant_response") {
        setHandoff(Boolean(p.human_handoff));
        add("bot", p.text, p.human_handoff ? "human" : "call");
      }
      if (p.type === "assistant_audio") playAudio(p.audio_base64, p.generation_id);
      if (p.type === "assistant_interrupted") stopAudio();
      if (p.type === "human_message") add("bot", p.text, "human");
      if (p.type === "error") add("bot", p.message || "Call error", "error");
    };
    ws.onclose = (e) => {
      callConnectingRef.current = false;
      setCallStatus(`Disconnected (${e.code || 1000})`);
      stopCall(false);
    };
    ws.onerror = () => {
      callConnectingRef.current = false;
      setCallStatus("Connection error");
    };
  };

  const stopCall = (closeSocket = true) => {
    callConnectingRef.current = false;
    callActiveRef.current = false;
    setCallOn(false);
    setCallStatus("Offline");
    stopAudio();
    if (pingRef.current) clearInterval(pingRef.current);
    pingRef.current = null;
    const rec = callRecRef.current;
    if (rec && rec.state !== "inactive") {
      rec.ondataavailable = null;
      rec.stop();
    }
    callRecRef.current = null;
    const s = callStreamRef.current;
    if (s) s.getTracks().forEach((t) => t.stop());
    callStreamRef.current = null;
    if (callCtxRef.current) {
      try {
        callCtxRef.current.close();
      } catch (_) {}
      callCtxRef.current = null;
    }
    if (closeSocket) wsRef.current?.close();
    wsRef.current = null;
  };

  const onSend = () => (recording ? stopVoiceNote() : sendChat());

  return (
    <div className="wa-root">
      <section className="wa-card">
        <header className="wa-top">
          <div className="wa-user">
            <span className="wa-avatar">MC</span>
            <div>
              <h1>Medical Concierge</h1>
              <small>{handoff ? "Human handoff active" : callStatus}</small>
            </div>
          </div>
          <button className={`wa-call ${callOn ? "on" : ""}`} onClick={callOn ? stopCall : startCall} aria-label="toggle-call">
            {callOn ? "📴" : "📞"}
          </button>
        </header>

        <main className="wa-chat">
          {messages.map((m) => (
            <article key={m.id} className={`wa-msg ${m.sender === "user" ? "me" : "them"} ${m.channel === "human" ? "human" : ""}`}>
              <p>{m.text}</p>
            </article>
          ))}
          {busy ? <div className="wa-typing">Typing...</div> : null}
          <div ref={endRef} />
        </main>

        <footer className="wa-compose">
          <button className={`wa-mic ${recording ? "on" : ""}`} onClick={recording ? () => stopVoiceNote() : startVoiceNote} aria-label="voice-note">
            {recording ? "■" : "🎤"}
          </button>
          <input
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && onSend()}
            placeholder={handoff ? "Message human care team..." : "Type a message"}
          />
          <button className="wa-send" onClick={onSend} disabled={busy}>
            ➤
          </button>
        </footer>
      </section>
    </div>
  );
}
