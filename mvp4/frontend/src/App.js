import { startTransition, useEffect, useRef, useState } from "react";
import "./index.css";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

function createSessionId() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return `session-${Date.now()}`;
}

function createRecognition() {
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) return null;
  const r = new Recognition();
  r.lang = "en-US";
  r.interimResults = false;
  r.continuous = true;
  return r;
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(String(reader.result || "").split(",")[1] || "");
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
}

// ---------------------------------------------------------------------------
// MicVisualiser — live volume bar drawn with AnalyserNode + canvas
// ---------------------------------------------------------------------------
function MicVisualiser({ stream }) {
  const canvasRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    if (!stream) return;

    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const analyser = ctx.createAnalyser();
    analyser.fftSize = 256;
    const source = ctx.createMediaStreamSource(stream);
    source.connect(analyser);

    const data = new Uint8Array(analyser.frequencyBinCount);
    const canvas = canvasRef.current;

    function draw() {
      rafRef.current = requestAnimationFrame(draw);
      analyser.getByteFrequencyData(data);
      const avg = data.reduce((a, b) => a + b, 0) / data.length;
      const c = canvas.getContext("2d");
      c.clearRect(0, 0, canvas.width, canvas.height);

      // Background track
      c.fillStyle = "rgba(83,59,45,0.10)";
      c.beginPath();
      c.roundRect(0, canvas.height / 2 - 4, canvas.width, 8, 4);
      c.fill();

      // Filled portion
      const pct = Math.min(avg / 128, 1);
      const grad = c.createLinearGradient(0, 0, canvas.width, 0);
      grad.addColorStop(0, "#bf5c3f");
      grad.addColorStop(1, "#d88e5a");
      c.fillStyle = grad;
      c.beginPath();
      c.roundRect(0, canvas.height / 2 - 4, canvas.width * pct, 8, 4);
      c.fill();

      // Tip dot
      if (pct > 0.02) {
        c.beginPath();
        c.arc(canvas.width * pct, canvas.height / 2, 6, 0, Math.PI * 2);
        c.fillStyle = "#8d3821";
        c.fill();
      }
    }
    draw();

    return () => {
      cancelAnimationFrame(rafRef.current);
      source.disconnect();
      ctx.close();
    };
  }, [stream]);

  return (
    <canvas
      ref={canvasRef}
      width={260}
      height={28}
      style={{ display: "block", width: "100%", height: 28, borderRadius: 8 }}
    />
  );
}

// ---------------------------------------------------------------------------
// speakResponse — plays TTS audio and fires onStart/onEnd so the caller can
// pause recognition while the bot is speaking (prevents TTS echo loop).
// ---------------------------------------------------------------------------
function speakResponse(text, audioBase64, { onStart, onEnd } = {}) {
  if (audioBase64) {
    const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);
    onStart?.();
    audio.onended = () => onEnd?.();
    audio.onerror = () => onEnd?.();
    audio.play().catch(() => onEnd?.());
    return;
  }

  if (!window.speechSynthesis || !text) return;
  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
  onStart?.();
  utterance.onend = () => onEnd?.();
  utterance.onerror = () => onEnd?.();
  window.speechSynthesis.speak(utterance);
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
function App() {
  const [sessionId] = useState(createSessionId);
  const [messages, setMessages] = useState([
    {
      id: "welcome",
      sender: "bot",
      text: "Medical concierge is ready. You can chat, send a voice note, or start a live call for symptoms, doctor info, booking, and triage.",
      channel: "system",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [activeMode, setActiveMode] = useState("chat");
  const [voiceStatus, setVoiceStatus] = useState("Idle");
  const [callStatus, setCallStatus] = useState("Call is offline");
  const [callActive, setCallActive] = useState(false);
  const [callLogs, setCallLogs] = useState([]);

  // Mic streams exposed to the visualiser
  const [liveStream, setLiveStream] = useState(null);  // voice-note mode
  const [callStream, setCallStream] = useState(null);  // call mode

  const scrollRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const voiceRecognitionRef = useRef(null);
  const callRecognitionRef = useRef(null);
  const callSocketRef = useRef(null);
  const callActiveRef = useRef(false);
  const vadTimerRef = useRef(null);
  const audioCtxRef = useRef(null);
  const callStreamRef = useRef(null);   // ref mirror of callStream — safe to read inside callbacks
  // True while bot audio / TTS is playing — recognition must be silenced
  const isSpeakingRef = useRef(false);
  const listenStartRef = useRef(null);  // timestamp when mic opened

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, voiceStatus, callStatus]);

  function appendMessage(sender, text, channel) {
    startTransition(() => {
      setMessages((cur) => [
        ...cur,
        { id: `${Date.now()}-${Math.random()}`, sender, text, channel },
      ]);
    });
  }

  function pushLog(msg) {
    const ts = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    setCallLogs((cur) => [...cur.slice(-49), `${ts}  ${msg}`]); // keep last 50
    console.log(`[Call] ${msg}`);
  }

  // ── Chat ──────────────────────────────────────────────────────────────────
  async function sendChatMessage() {
    const text = input.trim();
    if (!text || loading) return;
    appendMessage("user", text, "chat");
    setInput("");
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, user_input: text, channel: "chat" }),
      });
      const data = await res.json();
      appendMessage("bot", data.reply, data.action || "chat");
    } catch {
      appendMessage("bot", "I couldn't reach the backend. Check that the API is running on port 8000.", "error");
    } finally {
      setLoading(false);
    }
  }

  // ── Voice note ────────────────────────────────────────────────────────────
  async function startVoiceRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setVoiceStatus("Microphone access is not available in this browser.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setLiveStream(stream);
      recordedChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      setVoiceStatus("Recording — speak now");

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunksRef.current.push(e.data);
      };
      recorder.onstop = async () => {
        stream.getTracks().forEach((t) => t.stop());
        setLiveStream(null);
        const blob = new Blob(recordedChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        const b64 = await blobToBase64(blob);
        const hint = voiceRecognitionRef.current?.finalTranscript || "";
        await sendVoiceMessage(b64, blob.type, hint);
      };

      const recognition = createRecognition();
      if (recognition) {
        voiceRecognitionRef.current = { engine: recognition, finalTranscript: "" };
        recognition.onresult = (e) => {
          const latest = e.results[e.results.length - 1];
          if (latest?.isFinal)
            voiceRecognitionRef.current.finalTranscript = latest[0].transcript.trim();
        };
        recognition.start();
      } else {
        voiceRecognitionRef.current = null;
      }
      recorder.start();
    } catch {
      setVoiceStatus("Microphone access failed.");
    }
  }

  function stopVoiceRecording() {
    mediaRecorderRef.current?.stop();
    voiceRecognitionRef.current?.engine?.stop();
    setVoiceStatus("Sending voice note...");
  }

  async function sendVoiceMessage(audioBase64, mimeType, transcriptHint) {
    setLoading(true);
    appendMessage("user", transcriptHint || "Voice note sent", "voice");
    try {
      const res = await fetch(`${API_BASE}/voice/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: sessionId,
          transcript: transcriptHint,
          audio_base64: audioBase64,
          mime_type: mimeType,
        }),
      });
      const data = await res.json();
      if (data.transcript && data.transcript !== transcriptHint)
        appendMessage("user", data.transcript, "voice transcript");
      appendMessage("bot", data.reply, data.action || "voice");
      speakResponse(data.reply, data.audio_base64);
      setVoiceStatus("Voice note processed.");
    } catch {
      appendMessage("bot", "Voice processing failed. Try live call mode.", "error");
      setVoiceStatus("Voice note failed.");
    } finally {
      setLoading(false);
    }
  }

  // ── Call ──────────────────────────────────────────────────────────────────
  // When bot speaks we STOP the recorder entirely so no corrupt partial chunks
  // accumulate. When bot finishes we restart it fresh with a clean container.
  function pauseCallRecognition() {
    isSpeakingRef.current = true;
    if (listenStartRef.current) {
      const secs = ((Date.now() - listenStartRef.current) / 1000).toFixed(1);
      pushLog(`🔇 Mic closed — was listening for ${secs}s`);
      listenStartRef.current = null;
    } else {
      pushLog("🔇 Mic closed — bot is speaking");
    }
    const recorder = callRecognitionRef.current;
    if (recorder && recorder.state === "recording") {
      recorder.ondataavailable = null;  // discard any partial chunk — bot is speaking
      recorder.onstop = null;           // don't spawn next recorder while bot speaks
      recorder.stop();
    }
  }

  function resumeCallRecognition() {
    isSpeakingRef.current = false;
    const stream = callStreamRef.current;
    if (!stream || !callActiveRef.current) return;
    // Small delay so the mic settles before we start recording again
    setTimeout(() => {
      if (!callActiveRef.current || isSpeakingRef.current) return;
      pushLog("🎙️ Mic reopening after bot finished speaking...");
      startCallRecorder(stream);
    }, 300);
  }

  function startCall() {
    if (callActive) return;

    const socket = new WebSocket(`${API_BASE.replace("http", "ws")}/ws/call/${sessionId}`);
    callSocketRef.current = socket;

    socket.onopen = () => {
      callActiveRef.current = true;
      setCallActive(true);
      setCallStatus("Live call connected");
      setCallLogs([]);  // fresh log for new call
      appendMessage("bot", "Live call connected. Start speaking when you're ready.", "call");
      pushLog("🔌 WebSocket connected");

      // Play intro then open mic
      const introAudio = new Audio("/intro.mp3");
      introAudio.play()
        .then(() => { introAudio.onended = startCallRecognitionWithStream; })
        .catch(() => startCallRecognitionWithStream());
    };

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "user_transcript_echo") {
        // Whisper heard this — show it as a user bubble
        appendMessage("user", payload.text, "call");
      }
      if (payload.type === "assistant_response") {
        pushLog(`🤖 Bot responding — mic will pause`);
        appendMessage("bot", payload.text, payload.action || "call");
        speakResponse(payload.text, payload.audio_base64, {
          onStart: pauseCallRecognition,
          onEnd: resumeCallRecognition,
        });
      }
      if (payload.type === "call_ready") {
        setCallStatus(
          payload.server_stt
            ? "Live call — server STT active"
            : "Live call — browser speech"
        );
      }
    };

    socket.onclose = () => {
      callActiveRef.current = false;
      setCallActive(false);
      setCallStatus("Call ended");
      pushLog("🔌 WebSocket closed");
      stopCallRecognition();
    };

    socket.onerror = () => {
      callActiveRef.current = false;
      setCallStatus("Call connection failed");
      setCallActive(false);
      pushLog("❌ WebSocket error");
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

  // How often we slice and send an audio chunk (ms).
  // 4s is long enough for a full sentence, short enough to feel responsive.
  const CHUNK_INTERVAL_MS = 4000;

  async function startCallRecognitionWithStream() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCallStatus("Microphone unavailable — use chat or voice notes.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      setCallStream(stream);
      callStreamRef.current = stream;
      startCallRecorder(stream);
    } catch {
      setCallStatus("Microphone access failed.");
    }
  }

  function startCallRecorder(stream) {
    // Kill any existing recorder first — never run two at once
    const existing = callRecognitionRef.current;
    if (existing && existing.state !== "inactive") {
      existing.ondataavailable = null;
      existing.stop();
    }
    callRecognitionRef.current = null;

    // ── VAD setup: track peak volume during each recording window ──
    // We sample the analyser every 100ms and store the max seen.
    // If the whole window was silent we drop the chunk.
    const VAD_THRESHOLD = 60;  // 0-255 scale — real speech typically hits 60+
    let peakVolume = 0;
    let vadInterval = null;

    if (audioCtxRef.current) {
      try { audioCtxRef.current.close(); } catch (_) {}
    }
    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtxRef.current = audioCtx;
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    const freqData = new Uint8Array(analyser.frequencyBinCount);

    vadInterval = setInterval(() => {
      analyser.getByteFrequencyData(freqData);
      const avg = freqData.reduce((a, b) => a + b, 0) / freqData.length;
      if (avg > peakVolume) peakVolume = avg;
    }, 100);

    const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"].find(
      (t) => MediaRecorder.isTypeSupported(t)
    ) || "";

    // ── Stop/restart loop — each recorder instance gets its own fresh WebM
    // header, so every blob sent to Groq is a valid standalone file. ──────────
    let chunkIntervalId = null;

    function spawnRecorder() {
      if (!callActiveRef.current || isSpeakingRef.current) return;

      const rec = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      callRecognitionRef.current = rec;

      const windowStart = Date.now();
      peakVolume = 0;

      rec.ondataavailable = async (e) => {
        const listenedMs = Date.now() - windowStart;
        const peak = peakVolume;

        if (e.data.size < 3000) {
          pushLog(`⏭️ Dropped — too small (${e.data.size}B)`);
          return;
        }
        if (peak < VAD_THRESHOLD) {
          pushLog(`🔕 Dropped — silence (peak vol ${peak.toFixed(1)}, threshold ${VAD_THRESHOLD})`);
          return;
        }
        if (isSpeakingRef.current) {
          pushLog("⏭️ Dropped — bot started speaking");
          return;
        }
        if (callSocketRef.current?.readyState !== WebSocket.OPEN) return;

        pushLog(`📤 Sending ${(e.data.size / 1024).toFixed(1)}KB — peak vol ${peak.toFixed(1)} (${(listenedMs / 1000).toFixed(1)}s)`);

        const b64 = await blobToBase64(e.data);
        callSocketRef.current.send(JSON.stringify({
          type: "user_audio",
          audio_base64: b64,
          mime_type: e.data.type || mimeType || "audio/webm",
        }));
      };

      rec.onstart = () => {
        listenStartRef.current = Date.now();
        pushLog(`🎙️ Listening... (${CHUNK_INTERVAL_MS / 1000}s window)`);
      };

      rec.onstop = () => {
        // Spawn next recorder immediately after this one finishes
        if (callActiveRef.current && !isSpeakingRef.current) {
          spawnRecorder();
        }
      };

      rec.start();

      // Stop after the window — triggers ondataavailable then onstop
      setTimeout(() => {
        if (rec.state === "recording") rec.stop();
      }, CHUNK_INTERVAL_MS);
    }

    // Clean up VAD interval when the whole call stops (stopCallRecognition clears the recorder ref)
    const origVadCleanup = () => {
      if (vadInterval) { clearInterval(vadInterval); vadInterval = null; }
      if (chunkIntervalId) { clearInterval(chunkIntervalId); chunkIntervalId = null; }
    };

    // Patch stopCallRecognition cleanup onto the stream tracks
    stream.getTracks().forEach(t => {
      const origStop = t.stop.bind(t);
      t.stop = () => { origVadCleanup(); origStop(); };
    });

    spawnRecorder();
    setCallStatus("Live call — Whisper STT active");
  }

  function stopCallRecognition() {
    const recorder = callRecognitionRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.ondataavailable = null;
      recorder.stop();
    }
    callRecognitionRef.current = null;
    const stream = callStreamRef.current;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    callStreamRef.current = null;
    setCallStream(null);
  }


  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="app-shell">
      <div className="ambient ambient-one" />
      <div className="ambient ambient-two" />

      <section className="panel">
        <header className="hero">
          <div>
            <p className="eyebrow">AI Medical Concierge</p>
            <h1>One shared agent for chat, voice notes, and live calls.</h1>
            <p className="hero-copy">
              Symptom-aware routing, doctor discovery, appointment booking, and safe pre-visit triage in one flow.
            </p>
          </div>

          <div className="status-card">
            <span>Session</span>
            <strong>{sessionId.slice(0, 8)}</strong>
            <small>{callStatus}</small>
          </div>
        </header>

        <div className="mode-strip">
          {["chat", "voice", "call"].map((mode) => (
            <button
              key={mode}
              className={activeMode === mode ? "mode-pill active" : "mode-pill"}
              onClick={() => setActiveMode(mode)}
            >
              {mode}
            </button>
          ))}
        </div>

        <main className="conversation">
          {messages.map((message) => (
            <article
              key={message.id}
              className={message.sender === "user" ? "bubble user" : "bubble bot"}
            >
              <span className="channel-tag">{message.channel}</span>
              <p>{message.text}</p>
            </article>
          ))}
          {loading ? <div className="loading-bar">Thinking...</div> : null}
          <div ref={scrollRef} />
        </main>

        <footer className="composer">
          {activeMode === "chat" ? (
            <div className="composer-row">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") sendChatMessage(); }}
                placeholder="Describe symptoms, ask about a doctor, or start a booking..."
              />
              <button onClick={sendChatMessage}>Send</button>
            </div>
          ) : null}

          {activeMode === "voice" ? (
            <div className="voice-panel">
              <p>{voiceStatus}</p>
              {liveStream ? (
                <div className="mic-vis-wrap">
                  <span className="mic-dot" />
                  <MicVisualiser stream={liveStream} />
                </div>
              ) : null}
              <div className="voice-actions">
                <button onClick={startVoiceRecording} disabled={!!liveStream}>
                  Start voice note
                </button>
                <button className="secondary" onClick={stopVoiceRecording} disabled={!liveStream}>
                  Stop and send
                </button>
              </div>
            </div>
          ) : null}

          {activeMode === "call" ? (
            <div className="voice-panel">
              <p>{callStatus}</p>
              {callStream ? (
                <div className="mic-vis-wrap">
                  <span className="mic-dot" />
                  <MicVisualiser stream={callStream} />
                </div>
              ) : null}
              <div className="voice-actions">
                <button onClick={startCall} disabled={callActive}>
                  Start live call
                </button>
                <button className="secondary" onClick={stopCall} disabled={!callActive}>
                  End call
                </button>
              </div>
              {callLogs.length > 0 ? (
                <div style={{
                  marginTop: 10,
                  maxHeight: 160,
                  overflowY: "auto",
                  background: "rgba(0,0,0,0.55)",
                  borderRadius: 8,
                  padding: "6px 10px",
                  fontFamily: "monospace",
                  fontSize: 11,
                  color: "#b8ffc8",
                  lineHeight: 1.6,
                }}>
                  {callLogs.map((line, i) => <div key={i}>{line}</div>)}
                </div>
              ) : null}
            </div>
          ) : null}
        </footer>
      </section>
    </div>
  );
}

export default App;