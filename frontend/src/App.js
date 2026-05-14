import { startTransition, useCallback, useEffect, useMemo, useRef, useState } from "react";
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
const AUTH_KEY = "medical_concierge_auth";

const newId = () => window.crypto?.randomUUID?.() || `s-${Date.now()}`;
const trimSlash = (s) => String(s || "").replace(/\/+$/, "");
const authHeaders = (auth) => (auth?.token ? { Authorization: `Bearer ${auth.token}` } : {});
const b64 = (blob) =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(String(reader.result || "").split(",")[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });

const wsOrigin = () => {
  const explicit = trimSlash(WS_BASE);
  if (explicit) return explicit;
  try {
    const raw = String(API_BASE || "").trim();
    const abs = raw.startsWith("http://") || raw.startsWith("https://") ? raw : `${window.location.origin}${raw.startsWith("/") ? "" : "/"}${raw}`;
    const url = new URL(abs);
    return `${url.protocol === "https:" ? "wss" : "ws"}://${url.host}`;
  } catch {
    return `${window.location.protocol === "https:" ? "wss" : "ws"}://${window.location.host}`;
  }
};

const wsCallUrl = (sessionId) => `${wsOrigin()}/ws/call/${encodeURIComponent(sessionId)}`;
const formatDate = (value) => {
  if (!value) return "Not scheduled";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
};

function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const submit = async (event) => {
    event.preventDefault();
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/auth/login`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ username, password }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Login failed.");
      onLogin(data);
    } catch (err) {
      setError(err.message || "Login failed.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <main className="login-shell">
      <form className="login-panel" onSubmit={submit}>
        <div>
          <p className="eyebrow">Medical Concierge</p>
          <h1>Staff sign in</h1>
          <p className="muted">Use the admin or CSR account configured on the backend.</p>
        </div>
        <label>
          Username
          <input value={username} onChange={(event) => setUsername(event.target.value)} autoComplete="username" />
        </label>
        <label>
          Password
          <input type="password" value={password} onChange={(event) => setPassword(event.target.value)} autoComplete="current-password" />
        </label>
        {error ? <p className="form-error">{error}</p> : null}
        <button className="primary-action" type="submit" disabled={busy}>
          {busy ? "Signing in..." : "Sign in"}
        </button>
      </form>
    </main>
  );
}

function AppHeader({ auth, onLogout }) {
  return (
    <header className="app-header">
      <div>
        <p className="eyebrow">Medical Concierge</p>
        <h1>{auth.role === "admin" ? "Admin Dashboard" : "CSR Workspace"}</h1>
      </div>
      <div className="session-chip">
        <span>{auth.username}</span>
        <strong>{auth.role}</strong>
        <button onClick={onLogout} aria-label="Sign out">Sign out</button>
      </div>
    </header>
  );
}

function AdminDashboard({ auth }) {
  const [appointments, setAppointments] = useState([]);
  const [busy, setBusy] = useState(true);
  const [error, setError] = useState("");
  const [selectedReport, setSelectedReport] = useState(null);

  const loadAppointments = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const response = await fetch(`${API_BASE}/admin/appointments`, { headers: authHeaders(auth) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to load appointments.");
      setAppointments(data.appointments || []);
    } catch (err) {
      setError(err.message || "Unable to load appointments.");
    } finally {
      setBusy(false);
    }
  }, [auth]);

  useEffect(() => {
    loadAppointments();
  }, [loadAppointments]);

  return (
    <section className="admin-dashboard">
      <div className="section-bar">
        <div>
          <h2>Appointments</h2>
          <p>{appointments.length} records</p>
        </div>
        <button className="secondary-action" onClick={loadAppointments}>Refresh</button>
      </div>
      {error ? <p className="form-error">{error}</p> : null}
      {busy ? <div className="empty-state">Loading appointments...</div> : null}
      {!busy && appointments.length === 0 ? <div className="empty-state">No appointments found.</div> : null}
      {!busy && appointments.length > 0 ? (
        <div className="appointment-table">
          <div className="table-row table-head">
            <span>Time</span>
            <span>Doctor</span>
            <span>Patient</span>
            <span>Status</span>
            <span>Report</span>
          </div>
          {appointments.map((appointment) => (
            <article className="table-row" key={appointment.id}>
              <span>{formatDate(appointment.slot?.start_time || appointment.created_at)}</span>
              <span>
                <strong>Dr. {appointment.doctor?.name}</strong>
                <small>{appointment.doctor?.specialization || "Specialization not recorded"}</small>
              </span>
              <span>
                <strong>{appointment.patient?.name}</strong>
                <small>{appointment.patient?.phone || "No phone"}</small>
              </span>
              <span><mark>{appointment.status || "unknown"}</mark></span>
              <span>
                <button className="secondary-action compact-button" onClick={() => setSelectedReport(appointment)}>
                  View report
                </button>
              </span>
            </article>
          ))}
        </div>
      ) : null}
      {selectedReport ? (
        <div className="modal-backdrop" role="presentation" onClick={() => setSelectedReport(null)}>
          <section className="report-modal" role="dialog" aria-modal="true" aria-label="Appointment report" onClick={(event) => event.stopPropagation()}>
            <header>
              <div>
                <p className="eyebrow">Appointment #{selectedReport.id}</p>
                <h2>{selectedReport.patient?.name || "Unknown patient"}</h2>
                <p className="muted">Dr. {selectedReport.doctor?.name || "Unknown"} - {formatDate(selectedReport.slot?.start_time || selectedReport.created_at)}</p>
              </div>
              <button className="secondary-action" onClick={() => setSelectedReport(null)}>Close</button>
            </header>
            <pre>{selectedReport.notes || "No report or notes recorded for this appointment."}</pre>
          </section>
        </div>
      ) : null}
    </section>
  );
}

function CsrHandoffConsole({ auth }) {
  const [handoffs, setHandoffs] = useState([]);
  const [selectedId, setSelectedId] = useState("");
  const [detail, setDetail] = useState(null);
  const [reply, setReply] = useState("");
  const [error, setError] = useState("");

  const selected = useMemo(
    () => handoffs.find((item) => item.session_id === selectedId) || handoffs[0],
    [handoffs, selectedId]
  );

  const loadHandoffs = useCallback(async () => {
    if (!auth?.token) return;
    try {
      const response = await fetch(`${API_BASE}/csr/handoffs`, { headers: authHeaders(auth) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to load handoffs.");
      setHandoffs(data.handoffs || []);
      setError("");
    } catch (err) {
      setError(err.message || "Unable to load handoffs.");
    }
  }, [auth]);

  const loadDetail = useCallback(async (sessionId) => {
    if (!auth?.token) return;
    if (!sessionId) {
      setDetail(null);
      return;
    }
    try {
      const response = await fetch(`${API_BASE}/csr/handoffs/${encodeURIComponent(sessionId)}`, { headers: authHeaders(auth) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to load handoff.");
      setDetail(data);
      setError("");
    } catch (err) {
      setError(err.message || "Unable to load handoff.");
    }
  }, [auth]);

  useEffect(() => {
    if (!auth?.token) return undefined;
    loadHandoffs();
    const timer = setInterval(loadHandoffs, 2500);
    return () => clearInterval(timer);
  }, [auth?.token, loadHandoffs]);

  useEffect(() => {
    if (selected?.session_id) {
      setSelectedId(selected.session_id);
      loadDetail(selected.session_id);
    }
  }, [selected?.session_id, loadDetail]);

  useEffect(() => {
    if (!auth?.token || !selectedId) return undefined;
    const timer = setInterval(() => loadDetail(selectedId), 1500);
    return () => clearInterval(timer);
  }, [auth?.token, loadDetail, selectedId]);

  const sendReply = async () => {
    if (!detail?.context?.session_id || !reply.trim()) return;
    try {
      const response = await fetch(`${API_BASE}/human/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json", ...authHeaders(auth) },
        body: JSON.stringify({ session_id: detail.context.session_id, message: reply.trim(), sender: "csr" }),
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "Unable to send reply.");
      setReply("");
      await loadDetail(detail.context.session_id);
      await loadHandoffs();
    } catch (err) {
      setError(err.message || "Unable to send reply.");
    }
  };

  return (
    <section className="handoff-console">
      <div className="handoff-list">
        <div className="section-bar compact">
          <div>
            <h2>Human handoffs</h2>
            <p>{handoffs.length} active</p>
          </div>
          <button className="icon-action" onClick={loadHandoffs} aria-label="Refresh handoffs">Refresh</button>
        </div>
        {error ? <p className="form-error">{error}</p> : null}
        {handoffs.length === 0 ? <div className="empty-state">No patients are waiting.</div> : null}
        {handoffs.map((handoff) => (
          <button
            key={handoff.session_id}
            className={`handoff-item ${handoff.session_id === selectedId ? "active" : ""}`}
            onClick={() => setSelectedId(handoff.session_id)}
          >
            <strong>{handoff.patient?.name || "Unknown patient"}</strong>
            <span>{handoff.prime_complaint || handoff.reason}</span>
            <small>{handoff.status}</small>
          </button>
        ))}
      </div>
      <div className="handoff-detail">
        {detail ? (
          <>
            <div className="detail-head">
              <div>
                <h2>{detail.context?.patient?.name || "Patient session"}</h2>
                <p>{detail.context?.patient?.phone || detail.context?.session_id}</p>
              </div>
              <mark>{detail.handoff?.status || "active"}</mark>
            </div>
            <div className="transcript">
              {(detail.transcript || []).map((item, index) => (
                <article className={`transcript-line ${item.sender}`} key={`${item.at}-${index}`}>
                  <small>{item.sender} - {item.channel}</small>
                  <p>{item.text}</p>
                </article>
              ))}
            </div>
            <div className="csr-reply">
              <textarea value={reply} onChange={(event) => setReply(event.target.value)} placeholder="Reply as CSR..." />
              <button className="primary-action" onClick={sendReply}>Send reply</button>
            </div>
          </>
        ) : (
          <div className="empty-state">Select a handoff to view the conversation.</div>
        )}
      </div>
    </section>
  );
}

function PatientChatPanel() {
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
  const seenHumanRef = useRef(new Set());

  const add = useCallback((sender, textMsg, channel = "chat") => {
    startTransition(() =>
      setMessages((items) => [...items, { id: `${Date.now()}-${Math.random()}`, sender, text: textMsg, channel }])
    );
  }, []);

  const stopAudio = useCallback(() => {
    const audio = audioRef.current;
    if (!audio) return;
    try {
      audio.pause();
    } catch (_) {}
    audio.onended = null;
    audio.onerror = null;
    audioRef.current = null;
    speakingRef.current = false;
  }, []);

  const playAudio = useCallback((audio_base64, generation_id) => {
    if (!audio_base64) return;
    stopAudio();
    genRef.current = Number(generation_id || 0);
    const audio = new Audio(`data:audio/mp3;base64,${audio_base64}`);
    audioRef.current = audio;
    speakingRef.current = true;
    const done = () => {
      if (audioRef.current === audio) audioRef.current = null;
      speakingRef.current = false;
      if (genRef.current && wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "assistant_playback_done", generation_id: genRef.current }));
      }
    };
    audio.onended = done;
    audio.onerror = done;
    audio.play().catch(done);
  }, [stopAudio]);

  const stopCall = useCallback((closeSocket = true) => {
    callConnectingRef.current = false;
    callActiveRef.current = false;
    setCallOn(false);
    setCallStatus("Offline");
    stopAudio();
    if (pingRef.current) clearInterval(pingRef.current);
    pingRef.current = null;
    const recorder = callRecRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.ondataavailable = null;
      recorder.stop();
    }
    callRecRef.current = null;
    const stream = callStreamRef.current;
    if (stream) stream.getTracks().forEach((track) => track.stop());
    callStreamRef.current = null;
    if (callCtxRef.current) {
      try {
        callCtxRef.current.close();
      } catch (_) {}
      callCtxRef.current = null;
    }
    if (closeSocket) wsRef.current?.close();
    wsRef.current = null;
  }, [stopAudio]);

  useEffect(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), [messages, busy, callStatus]);

  useEffect(() => {
    if (!handoff) return undefined;
    const timer = setInterval(async () => {
      try {
        const response = await fetch(`${API_BASE}/human/messages/${encodeURIComponent(sessionId)}`);
        const data = await response.json();
        for (const item of data.messages || []) {
          const key = `${item.at || ""}-${item.sender}-${item.message}`;
          if (seenHumanRef.current.has(key)) continue;
          seenHumanRef.current.add(key);
          add("bot", item.message, "human");
          setHandoff(true);
        }
      } catch (_) {}
    }, 1500);
    return () => clearInterval(timer);
  }, [add, handoff, sessionId]);

  useEffect(() => {
    return () => {
      callActiveRef.current = false;
      const recorder = callRecRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.ondataavailable = null;
        recorder.stop();
      }
      const micRecorder = micRecRef.current;
      if (micRecorder && micRecorder.state === "recording") micRecorder.stop();
      stopAudio();
      if (pingRef.current) clearInterval(pingRef.current);
      const stream = callStreamRef.current;
      if (stream) stream.getTracks().forEach((track) => track.stop());
      if (callCtxRef.current) {
        try {
          callCtxRef.current.close();
        } catch (_) {}
      }
      wsRef.current?.close();
    };
  }, [stopAudio]);

  const sendChat = async () => {
    const clean = text.trim();
    if (!clean || busy) return;
    add("user", clean);
    setText("");
    setBusy(true);
    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, user_input: clean, channel: "chat" }),
      });
      const data = await response.json();
      setHandoff(Boolean(data.human_handoff));
      if (data.reply) add("bot", data.reply, data.human_handoff ? "human" : "chat");
    } catch {
      add("bot", "Backend unavailable.", "error");
    } finally {
      setBusy(false);
    }
  };

  const startVoiceNote = async () => {
    if (!navigator.mediaDevices?.getUserMedia || recording) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      micChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      micRecRef.current = recorder;
      setRecording(true);
      recorder.ondataavailable = (event) => event.data?.size && micChunksRef.current.push(event.data);
      recorder.onstop = async () => {
        setRecording(false);
        stream.getTracks().forEach((track) => track.stop());
        const blob = new Blob(micChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        const audio_base64 = await b64(blob);
        setBusy(true);
        add("user", "Voice note", "voice");
        try {
          const response = await fetch(`${API_BASE}/voice/message`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: sessionId, audio_base64, mime_type: blob.type }),
          });
          const data = await response.json();
          setHandoff(Boolean(data.human_handoff));
          if (data.transcript) add("user", data.transcript, "voice");
          if (data.reply) add("bot", data.reply, data.human_handoff ? "human" : "voice");
          playAudio(data.audio_base64, 0);
        } catch {
          add("bot", "Voice processing failed.", "error");
        } finally {
          setBusy(false);
        }
      };
      recorder.start();
    } catch {
      add("bot", "Mic permission denied.", "error");
    }
  };

  const stopVoiceNote = () => {
    const recorder = micRecRef.current;
    if (recorder && recorder.state === "recording") recorder.stop();
    setRecording(false);
  };

  const startCallRecorder = (stream) => {
    const previous = callRecRef.current;
    if (previous && previous.state !== "inactive") {
      previous.ondataavailable = null;
      previous.stop();
    }
    if (callCtxRef.current) {
      try {
        callCtxRef.current.close();
      } catch (_) {}
    }

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    callCtxRef.current = ctx;
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    ctx.createMediaStreamSource(stream).connect(analyser);
    const levels = new Uint8Array(analyser.frequencyBinCount);
    let recorder = null;
    let chunks = [];
    let peak = 0;
    let speaking = false;
    let stopping = false;
    let silenceMs = 0;
    let startedAt = 0;
    let lastInterrupt = 0;
    const type = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"].find((item) => MediaRecorder.isTypeSupported(item)) || "";

    const stopChunk = () => {
      if (!recorder || !speaking || stopping) return;
      stopping = true;
      if (recorder.state === "recording") recorder.stop();
    };

    const startChunk = () => {
      if (speaking || !callActiveRef.current || wsRef.current?.readyState !== WebSocket.OPEN) return;
      chunks = [];
      peak = 0;
      silenceMs = 0;
      startedAt = Date.now();
      recorder = new MediaRecorder(stream, type ? { mimeType: type } : {});
      callRecRef.current = recorder;
      speaking = true;
      stopping = false;
      recorder.ondataavailable = (event) => event.data?.size && chunks.push(event.data);
      recorder.onstop = async () => {
        const local = chunks;
        const currentPeak = peak;
        chunks = [];
        peak = 0;
        silenceMs = 0;
        speaking = false;
        stopping = false;
        if (!callActiveRef.current || wsRef.current?.readyState !== WebSocket.OPEN || !local.length) return;
        const blob = new Blob(local, { type: recorder?.mimeType || type || "audio/webm" });
        if (blob.size < 1000 || currentPeak < VAD_MIN) return;
        wsRef.current.send(JSON.stringify({ type: "user_audio", audio_base64: await b64(blob), mime_type: blob.type }));
      };
      recorder.start();
    };

    const tick = setInterval(() => {
      if (!callActiveRef.current || !callStreamRef.current) {
        stopChunk();
        clearInterval(tick);
        return;
      }
      analyser.getByteFrequencyData(levels);
      const avg = levels.reduce((sum, value) => sum + value, 0) / levels.length;
      if (avg > peak) peak = avg;
      if (speakingRef.current && avg >= INTERRUPT && Date.now() - lastInterrupt > 350 && wsRef.current?.readyState === WebSocket.OPEN) {
        lastInterrupt = Date.now();
        wsRef.current.send(JSON.stringify({ type: "interrupt" }));
        stopAudio();
      }
      if (!speaking) {
        if (avg >= GATE_START) startChunk();
        return;
      }
      silenceMs = avg <= GATE_STOP ? silenceMs + VAD_TICK_MS : 0;
      if (silenceMs >= SILENCE_HOLD_MS || Date.now() - startedAt >= MAX_CHUNK_MS) stopChunk();
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
      if (pingRef.current) clearInterval(pingRef.current);
      pingRef.current = setInterval(() => wsRef.current?.readyState === WebSocket.OPEN && wsRef.current.send('{"type":"ping"}'), 12000);
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
        callStreamRef.current = stream;
        startCallRecorder(stream);
      } catch {
        setCallStatus("Mic unavailable");
      }
    };
    ws.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "call_ready") setCallStatus("Live");
      if (payload.type === "user_transcript_echo") add("user", payload.text, "call");
      if (payload.type === "assistant_response") {
        setHandoff(Boolean(payload.human_handoff));
        if (payload.text) add("bot", payload.text, payload.human_handoff ? "human" : "call");
      }
      if (payload.type === "assistant_audio") playAudio(payload.audio_base64, payload.generation_id);
      if (payload.type === "assistant_interrupted") stopAudio();
      if (payload.type === "human_message") add("bot", payload.text, "human");
      if (payload.type === "error") add("bot", payload.message || "Call error", "error");
    };
    ws.onclose = (event) => {
      callConnectingRef.current = false;
      setCallStatus(`Disconnected (${event.code || 1000})`);
      stopCall(false);
    };
    ws.onerror = () => {
      callConnectingRef.current = false;
      setCallStatus("Connection error");
    };
  };

  const onSend = () => (recording ? stopVoiceNote() : sendChat());

  return (
    <section className="patient-panel">
      <header className="chat-top">
        <div>
          <h2>Medical Concierge</h2>
          <p>{handoff ? "Human handoff active" : callStatus}</p>
        </div>
        <button className={`call-action ${callOn ? "on" : ""}`} onClick={callOn ? () => stopCall() : startCall} aria-label="Toggle call">
          {callOn ? "End call" : "Start call"}
        </button>
      </header>
      <main className="chat-stream">
        {messages.map((message) => (
          <article key={message.id} className={`chat-message ${message.sender === "user" ? "me" : "them"} ${message.channel === "human" ? "human" : ""}`}>
            <p>{message.text}</p>
          </article>
        ))}
        {busy ? <div className="typing">Typing...</div> : null}
        <div ref={endRef} />
      </main>
      <footer className="compose">
        <button className={`round-action ${recording ? "on" : ""}`} onClick={recording ? stopVoiceNote : startVoiceNote} aria-label="Voice note">
          {recording ? "Stop" : "Mic"}
        </button>
        <input
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={(event) => event.key === "Enter" && onSend()}
          placeholder={handoff ? "Message human care team..." : "Type a message"}
        />
        <button className="primary-action compact-button" onClick={onSend} disabled={busy}>Send</button>
      </footer>
    </section>
  );
}

function CsrWorkspace({ auth }) {
  return (
    <div className="csr-workspace">
      <CsrHandoffConsole auth={auth} />
    </div>
  );
}

export default function App() {
  const path = window.location.pathname.toLowerCase();
  const targetRole = path.startsWith("/admin") ? "admin" : path.startsWith("/csr") || path.startsWith("/staff") ? "csr" : "";
  const [auth, setAuth] = useState(() => {
    try {
      return JSON.parse(window.localStorage.getItem(AUTH_KEY) || "null");
    } catch {
      return null;
    }
  });

  const handleLogin = (nextAuth) => {
    window.localStorage.setItem(AUTH_KEY, JSON.stringify(nextAuth));
    setAuth(nextAuth);
  };

  const logout = () => {
    window.localStorage.removeItem(AUTH_KEY);
    setAuth(null);
  };

  useEffect(() => {
    const token = auth?.token;
    const role = auth?.role;
    if (!targetRole || !token || role !== targetRole) return undefined;
    let cancelled = false;
    fetch(`${API_BASE}/auth/me`, { headers: { Authorization: `Bearer ${token}` } })
      .then(async (response) => {
        if (cancelled) return;
        if (!response.ok) logout();
      })
      .catch(() => {
        if (!cancelled) logout();
      });
    return () => {
      cancelled = true;
    };
  }, [auth?.role, auth?.token, targetRole]);

  if (!targetRole) {
    return (
      <main className="public-chat-shell">
        <PatientChatPanel />
      </main>
    );
  }

  if (!auth || auth.role !== targetRole) return <LoginScreen onLogin={handleLogin} />;

  return (
    <main className="app-shell">
      <AppHeader auth={auth} onLogout={logout} />
      {targetRole === "admin" ? <AdminDashboard auth={auth} /> : <CsrWorkspace auth={auth} />}
    </main>
  );
}
