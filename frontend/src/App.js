import { startTransition, useDeferredValue, useEffect, useRef, useState } from "react";
import "./index.css";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

const PAGE_TABS = [
  { id: "messages", label: "Messages" },
  { id: "diagnostics", label: "Diagnostics" },
];

const DEFAULT_DIAGNOSIS_NOTE = "This is the triage agent's summary for the doctor, not a final diagnosis.";
const DEFAULT_VOICE_STATUS = "Tap the mic to record a voice note.";
const DEFAULT_CALL_STATUS = "Call is offline";

function createSessionId() {
  if (window.crypto?.randomUUID) {
    return window.crypto.randomUUID();
  }
  return `session-${Date.now()}`;
}

function createRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    return null;
  }

  const recognition = new Recognition();
  recognition.lang = "en-US";
  recognition.interimResults = false;
  recognition.continuous = true;
  return recognition;
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      const result = String(reader.result || "");
      resolve(result.split(",")[1] || "");
    };
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

function speakResponse(text, audioBase64) {
  window.speechSynthesis?.cancel();

  if (audioBase64) {
    const audio = new Audio(`data:audio/wav;base64,${audioBase64}`);
    audio.play().catch(() => {});
    return;
  }

  if (!window.speechSynthesis || !text) {
    return;
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  window.speechSynthesis.speak(utterance);
}

function formatLabel(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatValue(value, fallback = "Not yet shared") {
  if (value === 0) {
    return "0";
  }
  if (value === false) {
    return "No";
  }
  const text = String(value || "").trim();
  return text || fallback;
}

function createEmptyDiagnostics(sessionId) {
  return {
    session_id: sessionId,
    has_data: false,
    patient: {
      id: "",
      name: "",
      phone: "",
      gender: "",
      age: "",
    },
    doctor: {
      id: "",
      name: "",
      specialization: "",
    },
    appointment: {
      id: "",
      status: "",
      date: "",
      time: "",
      slot_id: "",
    },
    complaint: "",
    triage: {
      status: "idle",
      flow_name: "",
      summary: "",
      answers: [],
      red_flag_triggered: false,
      attention_status: "Waiting for intake",
      diagnosis_note: DEFAULT_DIAGNOSIS_NOTE,
    },
  };
}

function normalizeDiagnostics(payload, sessionId) {
  const fallback = createEmptyDiagnostics(sessionId);
  if (!payload || typeof payload !== "object") {
    return fallback;
  }

  return {
    session_id: payload.session_id || fallback.session_id,
    has_data: Boolean(payload.has_data),
    patient: {
      ...fallback.patient,
      ...(payload.patient && typeof payload.patient === "object" ? payload.patient : {}),
    },
    doctor: {
      ...fallback.doctor,
      ...(payload.doctor && typeof payload.doctor === "object" ? payload.doctor : {}),
    },
    appointment: {
      ...fallback.appointment,
      ...(payload.appointment && typeof payload.appointment === "object" ? payload.appointment : {}),
    },
    complaint: String(payload.complaint || ""),
    triage: {
      ...fallback.triage,
      ...(payload.triage && typeof payload.triage === "object" ? payload.triage : {}),
      answers: Array.isArray(payload?.triage?.answers) ? payload.triage.answers : [],
      diagnosis_note: payload?.triage?.diagnosis_note || DEFAULT_DIAGNOSIS_NOTE,
    },
  };
}

async function requestDiagnostics(sessionId) {
  const response = await fetch(`${API_BASE}/dashboard/${sessionId}`);
  if (!response.ok) {
    throw new Error("Unable to load diagnostics.");
  }
  const data = await response.json();
  return normalizeDiagnostics(data, sessionId);
}

function getTriageTone(triage) {
  if (triage.red_flag_triggered) {
    return "alert";
  }
  if (triage.summary) {
    return "ready";
  }
  return "neutral";
}

function PhoneIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none">
      <path
        d="M6.6 10.8a15.5 15.5 0 0 0 6.6 6.6l2.2-2.2a1 1 0 0 1 1-.24c1.08.36 2.24.54 3.4.54a1 1 0 0 1 1 1V20a1 1 0 0 1-1 1C10.73 21 3 13.27 3 3.8a1 1 0 0 1 1-1h3.52a1 1 0 0 1 1 1c0 1.18.18 2.32.54 3.4a1 1 0 0 1-.24 1l-2.22 2.6Z"
        fill="currentColor"
      />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none">
      <path
        d="M12 15a3.5 3.5 0 0 0 3.5-3.5V6.5a3.5 3.5 0 1 0-7 0v5A3.5 3.5 0 0 0 12 15Zm6-3.5a1 1 0 1 0-2 0 4 4 0 1 1-8 0 1 1 0 1 0-2 0 6 6 0 0 0 5 5.91V20H9a1 1 0 1 0 0 2h6a1 1 0 1 0 0-2h-2v-2.59A6 6 0 0 0 18 11.5Z"
        fill="currentColor"
      />
    </svg>
  );
}

function SendIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none">
      <path d="M3.6 11.3 19.8 4.2c.9-.4 1.8.5 1.4 1.4l-7.1 16.2c-.4.9-1.7.8-2-.2l-1.8-6-6-1.8c-1-.3-1.1-1.6-.2-2Z" fill="currentColor" />
    </svg>
  );
}

function StopIcon() {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none">
      <rect x="6.5" y="6.5" width="11" height="11" rx="2.5" fill="currentColor" />
    </svg>
  );
}

function MessageBubble({ message }) {
  const isUser = message.sender === "user";

  return (
    <article className={isUser ? "message-bubble user" : "message-bubble bot"}>
      <div className="bubble-meta">
        <span>{isUser ? "You" : "PulseCore"}</span>
        <span>{formatLabel(message.channel)}</span>
      </div>
      {message.audioUrl ? (
        <div className="voice-note-block">
          <audio className="message-audio" controls preload="metadata" src={message.audioUrl}>
            Your browser does not support audio playback.
          </audio>
        </div>
      ) : null}
      {message.text ? <p>{message.text}</p> : null}
    </article>
  );
}

function DetailItem({ label, value }) {
  return (
    <div className="detail-item">
      <span className="detail-label">{label}</span>
      <strong className="detail-value">{formatValue(value)}</strong>
    </div>
  );
}

function AnswerItem({ answer, index }) {
  return (
    <article className="answer-item">
      <span className="detail-label">Question {index + 1}</span>
      <p className="answer-question">{formatValue(answer.question)}</p>
      <p className="answer-response">{formatValue(answer.answer)}</p>
    </article>
  );
}

function App() {
  const [sessionId] = useState(createSessionId);
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      sender: "bot",
      text: "PulseCore is ready. You can chat, send a voice note, or start a live call for symptoms, doctor info, booking, and triage.",
      channel: "system",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activePage, setActivePage] = useState("messages");
  const [voiceStatus, setVoiceStatus] = useState(DEFAULT_VOICE_STATUS);
  const [callStatus, setCallStatus] = useState(DEFAULT_CALL_STATUS);
  const [callActive, setCallActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [diagnostics, setDiagnostics] = useState(() => ({
    loading: false,
    loaded: false,
    error: "",
    syncedAt: "",
    data: createEmptyDiagnostics(sessionId),
  }));

  const deferredMessages = useDeferredValue(messages);
  const scrollRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const voiceRecognitionRef = useRef(null);
  const callRecognitionRef = useRef(null);
  const callSocketRef = useRef(null);
  const callActiveRef = useRef(false);
  const voiceAudioUrlsRef = useRef([]);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [deferredMessages, loading]);

  useEffect(() => {
    if (activePage !== "diagnostics") {
      return undefined;
    }

    let ignore = false;
    setDiagnostics((current) => ({
      ...current,
      loading: true,
      error: "",
    }));

    requestDiagnostics(sessionId)
      .then((data) => {
        if (ignore) {
          return;
        }
        setDiagnostics({
          loading: false,
          loaded: true,
          error: "",
          syncedAt: new Date().toISOString(),
          data,
        });
      })
      .catch(() => {
        if (ignore) {
          return;
        }
        setDiagnostics((current) => ({
          ...current,
          loading: false,
          error: "Diagnostics could not be loaded right now.",
        }));
      });

    return () => {
      ignore = true;
    };
  }, [activePage, sessionId]);

  useEffect(() => {
    return () => {
      window.speechSynthesis?.cancel();
      callActiveRef.current = false;
      callSocketRef.current?.close();

      if (callRecognitionRef.current) {
        callRecognitionRef.current.onend = null;
        callRecognitionRef.current.stop();
      }

      if (voiceRecognitionRef.current?.engine) {
        voiceRecognitionRef.current.engine.stop();
      }

      const recorder = mediaRecorderRef.current;
      if (recorder) {
        recorder.onstop = null;
        recorder.stream?.getTracks?.().forEach((track) => track.stop());
      }

      voiceAudioUrlsRef.current.forEach((url) => URL.revokeObjectURL(url));
      voiceAudioUrlsRef.current = [];
    };
  }, []);

  function appendMessage(sender, text, channel, extra = {}) {
    const id = extra.id || `${Date.now()}-${Math.random()}`;
    startTransition(() => {
      setMessages((current) => [
        ...current,
        { id, sender, text, channel, ...extra },
      ]);
    });
    return id;
  }

  function updateMessage(messageId, updates) {
    startTransition(() => {
      setMessages((current) =>
        current.map((message) => {
          if (message.id !== messageId) {
            return message;
          }

          const nextUpdates = typeof updates === "function" ? updates(message) : updates;
          return { ...message, ...nextUpdates };
        })
      );
    });
  }

  async function refreshDiagnostics(silent = true) {
    if (!silent) {
      setDiagnostics((current) => ({
        ...current,
        loading: true,
        error: "",
      }));
    }

    try {
      const data = await requestDiagnostics(sessionId);
      setDiagnostics({
        loading: false,
        loaded: true,
        error: "",
        syncedAt: new Date().toISOString(),
        data,
      });
    } catch (error) {
      if (!silent) {
        setDiagnostics((current) => ({
          ...current,
          loading: false,
          error: "Diagnostics could not be loaded right now.",
        }));
      }
    }
  }

  async function sendChatMessage() {
    const text = input.trim();
    if (!text || loading) {
      return;
    }

    appendMessage("user", text, "chat");
    setInput("");
    setLoading(true);

    try {
      const response = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          user_input: text,
          channel: "chat",
        }),
      });

      if (!response.ok) {
        throw new Error("Chat request failed.");
      }

      const data = await response.json();
      appendMessage("bot", data.reply, data.action || "chat");
      await refreshDiagnostics(true);
    } catch (error) {
      appendMessage("bot", "I couldn't reach the backend. Check that the API is running on port 8000.", "error");
    } finally {
      setLoading(false);
    }
  }

  async function startVoiceRecording() {
    if (isRecording || loading || callActive) {
      return;
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setVoiceStatus("Microphone access is not available in this browser.");
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunksRef.current = [];

      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      setIsRecording(true);
      setVoiceStatus("Recording voice note...");

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        setIsRecording(false);

        const tracks = stream.getTracks();
        tracks.forEach((track) => track.stop());

        const transcriptHint = voiceRecognitionRef.current?.finalTranscript || "";
        voiceRecognitionRef.current = null;

        const audioBlob = new Blob(recordedChunksRef.current, {
          type: recorder.mimeType || "audio/webm",
        });
        const localAudioUrl = URL.createObjectURL(audioBlob);
        voiceAudioUrlsRef.current.push(localAudioUrl);
        const audioBase64 = await blobToBase64(audioBlob);
        await sendVoiceMessage(audioBase64, audioBlob.type, transcriptHint, localAudioUrl);
      };

      const recognition = createRecognition();
      if (recognition) {
        voiceRecognitionRef.current = { engine: recognition, finalTranscript: "" };
        recognition.onresult = (event) => {
          const latest = event.results[event.results.length - 1];
          if (latest?.isFinal && voiceRecognitionRef.current) {
            voiceRecognitionRef.current.finalTranscript = latest[0].transcript.trim();
          }
        };
        try {
          recognition.start();
        } catch (error) {
          voiceRecognitionRef.current = null;
        }
      } else {
        voiceRecognitionRef.current = null;
      }

      recorder.start();
    } catch (error) {
      setIsRecording(false);
      setVoiceStatus("Microphone access failed.");
    }
  }

  function stopVoiceRecording() {
    const recorder = mediaRecorderRef.current;
    if (!recorder || recorder.state === "inactive") {
      return;
    }

    setVoiceStatus("Sending voice note...");
    if (voiceRecognitionRef.current?.engine) {
      voiceRecognitionRef.current.engine.stop();
    }
    recorder.stop();
  }

  async function sendVoiceMessage(audioBase64, mimeType, transcriptHint, localAudioUrl) {
    setLoading(true);
    const voiceMessageId = appendMessage("user", transcriptHint || "Voice note", "voice", {
      audioUrl: localAudioUrl,
    });

    try {
      const response = await fetch(`${API_BASE}/voice/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          transcript: transcriptHint,
          audio_base64: audioBase64,
          mime_type: mimeType,
        }),
      });

      if (!response.ok) {
        throw new Error("Voice request failed.");
      }

      const data = await response.json();
      if (data.transcript && data.transcript !== transcriptHint) {
        updateMessage(voiceMessageId, { text: data.transcript });
      }
      appendMessage("bot", data.reply, data.action || "voice");
      speakResponse(data.reply, data.audio_base64);
      setVoiceStatus("Voice note processed.");
      await refreshDiagnostics(true);
    } catch (error) {
      appendMessage("bot", "Voice processing failed. If browser speech recognition is available, try live call mode.", "error");
      setVoiceStatus("Voice note failed.");
    } finally {
      setLoading(false);
    }
  }

  function startCall() {
    if (callActive || loading || isRecording) {
      return;
    }

    const socket = new WebSocket(`${API_BASE.replace("http", "ws")}/ws/call/${sessionId}`);
    callSocketRef.current = socket;

    socket.onopen = () => {
      callActiveRef.current = true;
      setCallActive(true);
      setCallStatus("Live call connected");
      appendMessage("bot", "Live call connected. Start speaking when you're ready.", "call");
      startCallRecognition();
    };

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);

      if (payload.type === "assistant_response") {
        appendMessage("bot", payload.text, payload.action || "call");
        speakResponse(payload.text, payload.audio_base64);
        refreshDiagnostics(true);
      }

      if (payload.type === "call_ready") {
        setCallStatus(
          payload.server_stt
            ? "Live call connected with server voice support"
            : "Live call connected using browser speech"
        );
      }
    };

    socket.onclose = () => {
      callActiveRef.current = false;
      callSocketRef.current = null;
      setCallActive(false);
      setCallStatus("Call ended");
      stopCallRecognition();
    };

    socket.onerror = () => {
      callActiveRef.current = false;
      callSocketRef.current = null;
      setCallActive(false);
      setCallStatus("Call connection failed");
      appendMessage("bot", "I couldn't connect the live call. You can keep using chat or voice notes.", "error");
      stopCallRecognition();
    };
  }

  function stopCall() {
    callActiveRef.current = false;
    callSocketRef.current?.close();
    setCallActive(false);
    setCallStatus("Call ended");
    stopCallRecognition();
    window.speechSynthesis?.cancel();
  }

  function startCallRecognition() {
    const recognition = createRecognition();
    if (!recognition) {
      setCallStatus("Call connected. Browser speech recognition is unavailable, so use chat or voice notes here.");
      return;
    }

    callRecognitionRef.current = recognition;
    recognition.onresult = (event) => {
      const latest = event.results[event.results.length - 1];
      if (!latest?.isFinal) {
        return;
      }

      const transcript = latest[0].transcript.trim();
      if (!transcript || callSocketRef.current?.readyState !== WebSocket.OPEN) {
        return;
      }

      appendMessage("user", transcript, "call");
      callSocketRef.current.send(JSON.stringify({ type: "user_transcript", text: transcript }));
    };

    recognition.onend = () => {
      if (callActiveRef.current) {
        try {
          recognition.start();
        } catch (error) {}
      }
    };

    try {
      recognition.start();
    } catch (error) {
      setCallStatus("Call connected. Speech recognition could not start in this browser.");
    }
  }

  function stopCallRecognition() {
    if (callRecognitionRef.current) {
      callRecognitionRef.current.onend = null;
      callRecognitionRef.current.stop();
      callRecognitionRef.current = null;
    }
  }

  const diagnosticsData = diagnostics.data;
  const triage = diagnosticsData.triage;
  const hasDiagnosticsContent = Boolean(
    diagnosticsData.has_data || diagnosticsData.complaint || triage.summary || triage.answers.length
  );
  const hasDraftText = input.trim().length > 0;
  const activityMessage = isRecording
    ? voiceStatus
    : callActive
      ? callStatus
      : voiceStatus !== DEFAULT_VOICE_STATUS
        ? voiceStatus
        : callStatus !== DEFAULT_CALL_STATUS
          ? callStatus
          : "Type a message or tap the mic to record a voice note.";
  const composerButtonLabel = isRecording
    ? "Stop recording and send"
    : hasDraftText
      ? "Send message"
      : "Record voice note";
  const composerButtonDisabled = isRecording ? false : loading || (callActive && !hasDraftText);

  function handleComposerAction() {
    if (isRecording) {
      stopVoiceRecording();
      return;
    }
    if (hasDraftText) {
      sendChatMessage();
      return;
    }
    startVoiceRecording();
  }

  function handleCallAction() {
    if (callActive) {
      stopCall();
      return;
    }
    startCall();
  }

  return (
    <div className="app-shell">
      <section className="mobile-frame">
        <header className="topbar">
          <div className="topbar-row">
            <div className="topbar-main">
              <div className="avatar">MC</div>
              <div className="topbar-copy">
                <p className="topbar-title">PulseCore</p>
                <p className="topbar-subtitle">Simple care chat for all ages, with text, voice notes, and live calls.</p>
              </div>
            </div>

            <div className="topbar-actions">
              <button
                className={callActive ? "icon-button active" : "icon-button"}
                onClick={handleCallAction}
                type="button"
                aria-label={callActive ? "End live call" : "Start live call"}
                title={callActive ? "End live call" : "Start live call"}
              >
                <PhoneIcon />
              </button>
            </div>
          </div>

          <div className="topbar-meta">
            <span className={`status-pill ${callActive ? "ready" : "neutral"}`}>
              {callActive ? "Live call on" : "Ready to help"}
            </span>
            <span className="status-pill neutral">Session {sessionId.slice(0, 8)}</span>
          </div>
        </header>

        <nav className="page-tabs" aria-label="App pages">
          {PAGE_TABS.map((page) => (
            <button
              key={page.id}
              className={activePage === page.id ? "tab-button active" : "tab-button"}
              onClick={() => setActivePage(page.id)}
              type="button"
            >
              {page.label}
            </button>
          ))}
        </nav>

        {activePage === "messages" ? (
          <div className="messages-screen">
            <div className={isRecording ? "activity-strip recording" : "activity-strip"}>
              <strong>{isRecording ? "Recording voice note" : callActive ? "Live call" : "Chat"}</strong>
              <span>{activityMessage}</span>
            </div>

            <main className="conversation-surface">
              <div className="conversation-list">
                {deferredMessages.map((message) => (
                  <MessageBubble key={message.id} message={message} />
                ))}
                {loading ? <div className="loading-chip">Assistant is preparing the next reply...</div> : null}
                <div ref={scrollRef} />
              </div>
            </main>

            <footer className="composer-shell">
              <div className="composer-row">
                <input
                  className="composer-input"
                  value={input}
                  onChange={(event) => setInput(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      sendChatMessage();
                    }
                  }}
                  placeholder={
                    isRecording
                      ? "Recording voice note..."
                      : "Describe symptoms, ask about a doctor, or start a booking..."
                  }
                  disabled={isRecording}
                />
                <button
                  className={isRecording ? "composer-action-button recording" : "composer-action-button"}
                  onClick={handleComposerAction}
                  disabled={composerButtonDisabled}
                  type="button"
                  aria-label={composerButtonLabel}
                  title={composerButtonLabel}
                >
                  {isRecording ? <StopIcon /> : hasDraftText ? <SendIcon /> : <MicIcon />}
                </button>
              </div>
            </footer>
          </div>
        ) : (
          <main className="diagnostics-screen">
            <section className="diagnostics-header">
              <div>
                <p className="section-label">Diagnostics page</p>
                <h2 className="diagnostics-title">Patient summary and triage notes</h2>
                <p className="section-copy">
                  This page pulls the active session details so the doctor can quickly review the patient snapshot.
                </p>
              </div>
              <button className="secondary-button" onClick={() => refreshDiagnostics(false)} type="button">
                Refresh
              </button>
            </section>

            {diagnostics.error ? <div className="inline-banner error">{diagnostics.error}</div> : null}

            <section className="summary-panel emphasis">
              <div className="summary-topline">
                <span className={`status-pill ${getTriageTone(triage)}`}>{formatValue(triage.attention_status)}</span>
                <span className="summary-session">Session {diagnosticsData.session_id.slice(0, 8)}</span>
              </div>
              <h3 className="summary-title">
                {diagnosticsData.patient.name ? diagnosticsData.patient.name : "No patient profile yet"}
              </h3>
              <p className="summary-copy">
                {diagnosticsData.complaint
                  ? `Chief concern: ${diagnosticsData.complaint}`
                  : "Once the patient shares symptoms and completes intake, the summary will appear here."}
              </p>
              <p className="summary-note">{triage.diagnosis_note}</p>
            </section>

            <section className="summary-panel">
              <p className="section-label">Patient details</p>
              <div className="detail-grid">
                <DetailItem label="Patient ID" value={diagnosticsData.patient.id} />
                <DetailItem label="Phone" value={diagnosticsData.patient.phone} />
                <DetailItem label="Gender" value={diagnosticsData.patient.gender} />
                <DetailItem label="Age" value={diagnosticsData.patient.age} />
              </div>
            </section>

            <section className="summary-panel">
              <p className="section-label">Visit details</p>
              <div className="detail-grid">
                <DetailItem label="Doctor" value={diagnosticsData.doctor.name} />
                <DetailItem label="Specialization" value={diagnosticsData.doctor.specialization} />
                <DetailItem label="Appointment date" value={diagnosticsData.appointment.date} />
                <DetailItem label="Appointment time" value={diagnosticsData.appointment.time} />
                <DetailItem label="Appointment status" value={diagnosticsData.appointment.status} />
                <DetailItem label="Triage status" value={triage.status} />
              </div>
            </section>

            <section className="summary-panel">
              <p className="section-label">Triage summary</p>
              {triage.summary ? (
                <p className="summary-text">{triage.summary}</p>
              ) : (
                <div className="empty-note">No triage summary is available yet.</div>
              )}
            </section>

            <section className="summary-panel">
              <p className="section-label">Recorded intake answers</p>
              {triage.answers.length ? (
                <div className="answer-list">
                  {triage.answers.map((answer, index) => (
                    <AnswerItem key={`${answer.question}-${index}`} answer={answer} index={index} />
                  ))}
                </div>
              ) : (
                <div className="empty-note">
                  Intake answers will show here after the patient completes the triage questions.
                </div>
              )}
            </section>

            {!hasDiagnosticsContent && !diagnostics.loading ? (
              <div className="inline-banner">No diagnostic data has been captured in this session yet.</div>
            ) : null}

            {diagnostics.loading ? <div className="loading-chip diagnostics-loading">Refreshing diagnostics...</div> : null}
            {diagnostics.syncedAt ? (
              <p className="sync-copy">Last refreshed in this browser session.</p>
            ) : null}
          </main>
        )}
      </section>
    </div>
  );
}

export default App;
