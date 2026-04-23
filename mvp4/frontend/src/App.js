import { startTransition, useEffect, useRef, useState } from "react";
import "./index.css";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";
const CALL_CHUNK_MS = 4000;
const CALL_VAD_THRESHOLD = 1;
const CALL_GATE_START_THRESHOLD = 18;
const CALL_GATE_STOP_THRESHOLD = 15;
const INTERRUPT_THRESHOLD = 18;

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

      c.fillStyle = "rgba(83,59,45,0.10)";
      c.beginPath();
      c.roundRect(0, canvas.height / 2 - 4, canvas.width, 8, 4);
      c.fill();

      const pct = Math.min(avg / 128, 1);
      const grad = c.createLinearGradient(0, 0, canvas.width, 0);
      grad.addColorStop(0, "#bf5c3f");
      grad.addColorStop(1, "#d88e5a");
      c.fillStyle = grad;
      c.beginPath();
      c.roundRect(0, canvas.height / 2 - 4, canvas.width * pct, 8, 4);
      c.fill();

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
  const [liveStream, setLiveStream] = useState(null);
  const [callStream, setCallStream] = useState(null);

  const scrollRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const voiceRecognitionRef = useRef(null);

  const callSocketRef = useRef(null);
  const callRecorderRef = useRef(null);
  const callStreamRef = useRef(null);
  const callAudioCtxRef = useRef(null);
  const callActiveRef = useRef(false);
  const callAudioRef = useRef(null);
  const botSpeakingRef = useRef(false);
  const activeGenerationRef = useRef(0);
  const lastInterruptSignalRef = useRef(0);
  const pingTimerRef = useRef(null);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, voiceStatus, callStatus]);

  useEffect(() => {
    return () => {
      callActiveRef.current = false;
      try {
        callSocketRef.current?.close();
      } catch (_) {}
      if (pingTimerRef.current) {
        clearInterval(pingTimerRef.current);
        pingTimerRef.current = null;
      }
      const recorder = callRecorderRef.current;
      if (recorder && recorder.state !== "inactive") {
        recorder.ondataavailable = null;
        recorder.stop();
      }
      const stream = callStreamRef.current;
      if (stream) stream.getTracks().forEach((t) => t.stop());
      try {
        callAudioCtxRef.current?.close();
      } catch (_) {}
      try {
        callAudioRef.current?.pause();
      } catch (_) {}
    };
  }, []);

  function appendMessage(sender, text, channel) {
    startTransition(() => {
      setMessages((cur) => [
  
        ...cur,
        { id: `${Date.now()}-${Math.random()}`, sender, text, channel },
      ]);
    });
  }

  function pushLog(msg) {
    const ts = new Date().toLocaleTimeString([], {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    });
    setCallLogs((cur) => [...cur.slice(-49), `${ts}  ${msg}`]);
    console.log(`[Call] ${msg}`);
  }

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
      setVoiceStatus("Recording - speak now");

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
          if (latest?.isFinal) {
            voiceRecognitionRef.current.finalTranscript = latest[0].transcript.trim();
          }
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
      if (data.transcript && data.transcript !== transcriptHint) appendMessage("user", data.transcript, "voice transcript");
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

  function notifyPlaybackDone(generationId) {
    if (!generationId) return;
    const socket = callSocketRef.current;
    if (socket?.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify({ type: "assistant_playback_done", generation_id: generationId }));
    }
  }

  function stopAssistantPlayback(reason = "") {
    const current = callAudioRef.current;
    if (current) {
      try {
        current.pause();
      } catch (_) {}
      current.onended = null;
      current.onerror = null;
      callAudioRef.current = null;
    }
    const hadSpeech = botSpeakingRef.current;
    botSpeakingRef.current = false;
    if (hadSpeech) pushLog(`Assistant audio stopped${reason ? ` (${reason})` : ""}`);
  }

  function playAssistantAudio(payload) {
    const audioBase64 = payload.audio_base64;
    const generationId = Number(payload.generation_id || 0);
    activeGenerationRef.current = generationId;
    stopAssistantPlayback();

    if (!audioBase64) {
      botSpeakingRef.current = false;
      return;
    }

    const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);
    callAudioRef.current = audio;
    botSpeakingRef.current = true;

    audio.onended = () => {
      if (callAudioRef.current === audio) callAudioRef.current = null;
      botSpeakingRef.current = false;
      notifyPlaybackDone(generationId);
      pushLog("Assistant finished speaking");
    };
    audio.onerror = () => {
      if (callAudioRef.current === audio) callAudioRef.current = null;
      botSpeakingRef.current = false;
      notifyPlaybackDone(generationId);
      pushLog("Assistant audio playback error");
    };

    audio.play().catch(() => {
      botSpeakingRef.current = false;
      notifyPlaybackDone(generationId);
      pushLog("Assistant audio failed to start");
    });
  }

  function startCall() {
    if (callActiveRef.current) return;

    const socket = new WebSocket(`${API_BASE.replace("http", "ws")}/ws/call/${sessionId}`);
    callSocketRef.current = socket;

    socket.onopen = () => {
      callActiveRef.current = true;
      setCallActive(true);
      setCallStatus("Live call connected");
      setCallLogs([]);
      appendMessage("bot", "Live call connected. Start speaking when you're ready.", "call");
      pushLog("WebSocket connected");
      if (pingTimerRef.current) clearInterval(pingTimerRef.current);
      pingTimerRef.current = setInterval(() => {
        if (callSocketRef.current?.readyState === WebSocket.OPEN) {
          callSocketRef.current.send(JSON.stringify({ type: "ping" }));
        }
      }, 12000);

      const introAudio = new Audio("/intro.mp3");
      introAudio.play().then(() => {
        introAudio.onended = startCallRecognitionWithStream;
      }).catch(() => startCallRecognitionWithStream());
    };

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "call_ready") {
        const mode = payload.pipeline === "pipecat" ? "Pipecat pipeline active" : "Live call active";
        setCallStatus(mode);
        pushLog(mode);
      }
      if (payload.type === "user_transcript_echo") {
        appendMessage("user", payload.text, "call");
      }
      if (payload.type === "assistant_response") {
        appendMessage("bot", payload.text, payload.action || "call");
        pushLog("Assistant response received");
      }
      if (payload.type === "assistant_audio") {
        pushLog("Assistant audio received");
        playAssistantAudio(payload);
      }
      if (payload.type === "assistant_interrupted") {
        stopAssistantPlayback("barge-in");
        pushLog("Interruption acknowledged by server");
      }
      if (payload.type === "error") {
        pushLog(`Server error: ${payload.message || "unknown error"}`);
      }
    };

    socket.onclose = (event) => {
      callActiveRef.current = false;
      setCallActive(false);
      setCallStatus("Call ended");
      if (pingTimerRef.current) {
        clearInterval(pingTimerRef.current);
        pingTimerRef.current = null;
      }
      pushLog(`WebSocket closed (code=${event.code}, reason='${event.reason || "none"}')`);
      stopAssistantPlayback("call ended");
      stopCallRecognition();
    };

    socket.onerror = () => {
      callActiveRef.current = false;
      setCallActive(false);
      setCallStatus("Call connection failed");
      pushLog("WebSocket error");
      stopAssistantPlayback("socket error");
      stopCallRecognition();
    };
  }

  function stopCall() {
    callActiveRef.current = false;
    setCallActive(false);
    setCallStatus("Call ended");
    stopAssistantPlayback("manual end");
    stopCallRecognition();
    callSocketRef.current?.close();
    callSocketRef.current = null;
    if (pingTimerRef.current) {
      clearInterval(pingTimerRef.current);
      pingTimerRef.current = null;
    }
  }

  async function startCallRecognitionWithStream() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setCallStatus("Microphone unavailable - use chat or voice notes.");
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      setCallStream(stream);
      callStreamRef.current = stream;
      startCallRecorder(stream);
    } catch {
      setCallStatus("Microphone access failed.");
    }
  }

  function startCallRecorder(stream) {
    const existing = callRecorderRef.current;
    if (existing && existing.state !== "inactive") {
      existing.ondataavailable = null;
      existing.stop();
    }
    callRecorderRef.current = null;

    if (callAudioCtxRef.current) {
      try {
        callAudioCtxRef.current.close();
      } catch (_) {}
    }

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    callAudioCtxRef.current = audioCtx;
    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 256;
    audioCtx.createMediaStreamSource(stream).connect(analyser);
    const freqData = new Uint8Array(analyser.frequencyBinCount);

    let peakVolume = 0;
    let currentVolume = 0;
    let gateOpen = false;
    let highFrames = 0;
    let lowFrames = 0;
    const vadInterval = setInterval(() => {
      analyser.getByteFrequencyData(freqData);
      const avg = freqData.reduce((a, b) => a + b, 0) / freqData.length;
      currentVolume = avg;
      if (avg > peakVolume) peakVolume = avg;

      // Interruption should be immediate even while gated.
      const now = Date.now();
      if (
        botSpeakingRef.current &&
        avg >= INTERRUPT_THRESHOLD &&
        now - lastInterruptSignalRef.current > 350 &&
        callSocketRef.current?.readyState === WebSocket.OPEN
      ) {
        lastInterruptSignalRef.current = now;
        callSocketRef.current.send(JSON.stringify({ type: "interrupt" }));
        stopAssistantPlayback("user started speaking");
        pushLog(`Barge-in detected (vol ${avg.toFixed(1)})`);
      }

      // Voice gate with hysteresis:
      // open only after sustained high signal, close after sustained low signal.
      if (!gateOpen) {
        if (avg >= CALL_GATE_START_THRESHOLD) {
          highFrames += 1;
          if (highFrames >= 3) {
            gateOpen = true;
            lowFrames = 0;
            pushLog(`Voice gate OPEN (>= ${CALL_GATE_START_THRESHOLD})`);
          }
        } else {
          highFrames = 0;
        }
      } else {
        if (avg <= CALL_GATE_STOP_THRESHOLD) {
          lowFrames += 1;
          if (lowFrames >= 8) {
            gateOpen = false;
            highFrames = 0;
            pushLog(`Voice gate CLOSED (<= ${CALL_GATE_STOP_THRESHOLD})`);
          }
        } else {
          lowFrames = 0;
        }
      }
    }, 80);

    const mimeType = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg"].find(
      (t) => MediaRecorder.isTypeSupported(t)
    ) || "";

    pushLog(`Listening in ${CALL_CHUNK_MS / 1000}s chunks`);
    const runRecordingWindow = () => {
      if (!callActiveRef.current || !callStreamRef.current) return;
      if (!gateOpen) {
        setTimeout(runRecordingWindow, 140);
        return;
      }

      const chunks = [];
      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : {});
      callRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data?.size > 0) chunks.push(e.data);
      };

      recorder.onstop = async () => {
        const peak = peakVolume;
        peakVolume = 0;

        if (!callActiveRef.current) {
          clearInterval(vadInterval);
          return;
        }

        const blob = new Blob(chunks, { type: recorder.mimeType || mimeType || "audio/webm" });
        if (blob.size < 1000) {
          pushLog(`Dropped chunk: too small (${blob.size}B)`);
          runRecordingWindow();
          return;
        }
        if (peak < CALL_VAD_THRESHOLD) {
          pushLog(`Dropped chunk: low volume (${peak.toFixed(1)} < ${CALL_VAD_THRESHOLD})`);
          runRecordingWindow();
          return;
        }
        if (callSocketRef.current?.readyState !== WebSocket.OPEN) {
          runRecordingWindow();
          return;
        }

        if (!gateOpen && currentVolume < CALL_GATE_START_THRESHOLD) {
          runRecordingWindow();
          return;
        }

        const b64 = await blobToBase64(blob);
        callSocketRef.current.send(
          JSON.stringify({
            type: "user_audio",
            audio_base64: b64,
            mime_type: blob.type || mimeType || "audio/webm",
          })
        );
        pushLog(`Sent audio chunk ${(blob.size / 1024).toFixed(1)}KB (peak ${peak.toFixed(1)})`);
        runRecordingWindow();
      };

      recorder.start();
      setTimeout(() => {
        if (recorder.state === "recording") recorder.stop();
      }, CALL_CHUNK_MS);
    };

    runRecordingWindow();
    setCallStatus("Live call - listening");
  }

  function stopCallRecognition() {
    const recorder = callRecorderRef.current;
    if (recorder && recorder.state !== "inactive") {
      recorder.ondataavailable = null;
      recorder.stop();
    }
    callRecorderRef.current = null;

    const stream = callStreamRef.current;
    if (stream) stream.getTracks().forEach((t) => t.stop());
    callStreamRef.current = null;
    setCallStream(null);

    if (callAudioCtxRef.current) {
      try {
        callAudioCtxRef.current.close();
      } catch (_) {}
      callAudioCtxRef.current = null;
    }
  }

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
            <article key={message.id} className={message.sender === "user" ? "bubble user" : "bubble bot"}>
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
                onKeyDown={(e) => {
                  if (e.key === "Enter") sendChatMessage();
                }}
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
                <div
                  style={{
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
                  }}
                >
                  {callLogs.map((line, i) => (
                    <div key={i}>{line}</div>
                  ))}
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
