import { startTransition, useEffect, useRef, useState } from "react";
import "./index.css";

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

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
  if (audioBase64) {
    const audio = new Audio(`data:audio/mp3;base64,${audioBase64}`);
    audio.play().catch(() => {});
    return;
  }

  if (!window.speechSynthesis || !text) {
    return;
  }

  window.speechSynthesis.cancel();
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.rate = 1;
  utterance.pitch = 1;
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

  const scrollRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const recordedChunksRef = useRef([]);
  const voiceRecognitionRef = useRef(null);
  const callRecognitionRef = useRef(null);
  const callSocketRef = useRef(null);
  const callActiveRef = useRef(false);

  useEffect(() => {
    scrollRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading, voiceStatus, callStatus]);

  function appendMessage(sender, text, channel) {
    startTransition(() => {
      setMessages((current) => [
        ...current,
        { id: `${Date.now()}-${Math.random()}`, sender, text, channel },
      ]);
    });
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
      const data = await response.json();
      appendMessage("bot", data.reply, data.action || "chat");
    } catch (error) {
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
      recordedChunksRef.current = [];
      const recorder = new MediaRecorder(stream);
      mediaRecorderRef.current = recorder;
      setVoiceStatus("Recording voice note...");

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          recordedChunksRef.current.push(event.data);
        }
      };

      recorder.onstop = async () => {
        const tracks = stream.getTracks();
        tracks.forEach((track) => track.stop());

        const audioBlob = new Blob(recordedChunksRef.current, { type: recorder.mimeType || "audio/webm" });
        const audioBase64 = await blobToBase64(audioBlob);
        const transcriptHint = voiceRecognitionRef.current?.finalTranscript || "";
        await sendVoiceMessage(audioBase64, audioBlob.type, transcriptHint);
      };

      const recognition = createRecognition();
      if (recognition) {
        voiceRecognitionRef.current = { engine: recognition, finalTranscript: "" };
        recognition.onresult = (event) => {
          const latest = event.results[event.results.length - 1];
          if (latest?.isFinal) {
            voiceRecognitionRef.current.finalTranscript = latest[0].transcript.trim();
          }
        };
        recognition.start();
      } else {
        voiceRecognitionRef.current = null;
      }

      recorder.start();
    } catch (error) {
      setVoiceStatus("Microphone access failed.");
    }
  }

  function stopVoiceRecording() {
    mediaRecorderRef.current?.stop();
    if (voiceRecognitionRef.current?.engine) {
      voiceRecognitionRef.current.engine.stop();
    }
    setVoiceStatus("Sending voice note...");
  }

  async function sendVoiceMessage(audioBase64, mimeType, transcriptHint) {
    setLoading(true);
    appendMessage("user", transcriptHint || "Voice note sent", "voice");

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
      const data = await response.json();
      if (data.transcript && data.transcript !== transcriptHint) {
        appendMessage("user", data.transcript, "voice transcript");
      }
      appendMessage("bot", data.reply, data.action || "voice");
      speakResponse(data.reply, data.audio_base64);
      setVoiceStatus("Voice note processed.");
    } catch (error) {
      appendMessage("bot", "Voice processing failed. If browser speech recognition is available, try live call mode.", "error");
      setVoiceStatus("Voice note failed.");
    } finally {
      setLoading(false);
    }
  }

  function startCall() {
    if (callActive) {
      return;
    }

    const socket = new WebSocket(`${API_BASE.replace("http", "ws")}/ws/call/${sessionId}`);
    callSocketRef.current = socket;

    socket.onopen = () => {
      callActiveRef.current = true;
      setCallActive(true);
      setCallStatus("Live call connected");
      appendMessage("bot", "Live call connected. Start speaking when you're ready.", "call");
      const introAudio = new Audio("/intro.mp3");
      introAudio.play().then(() => {
          // PRO TIP: Wait for the intro to finish playing before turning on the microphone!
          introAudio.onended = () => {
              startCallRecognition();
          };
          }).catch((error) => {
          console.log("Browser blocked auto-play, starting mic anyway.", error);
          startCallRecognition(); // Fallback if audio fails
      });
    };

    socket.onmessage = (event) => {
      const payload = JSON.parse(event.data);
      if (payload.type === "assistant_response") {
        appendMessage("bot", payload.text, payload.action || "call");
        speakResponse(payload.text, payload.audio_base64);
      }
      if (payload.type === "call_ready") {
        setCallStatus(payload.server_stt ? "Live call connected with server voice support" : "Live call connected using browser speech");
      }
    };

    socket.onclose = () => {
      callActiveRef.current = false;
      setCallActive(false);
      setCallStatus("Call ended");
      stopCallRecognition();
    };

    socket.onerror = () => {
      callActiveRef.current = false;
      setCallStatus("Call connection failed");
      setCallActive(false);
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
        recognition.start();
      }
    };
    recognition.start();
  }

  function stopCallRecognition() {
    if (callRecognitionRef.current) {
      callRecognitionRef.current.onend = null;
      callRecognitionRef.current.stop();
      callRecognitionRef.current = null;
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
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    sendChatMessage();
                  }
                }}
                placeholder="Describe symptoms, ask about a doctor, or start a booking..."
              />
              <button onClick={sendChatMessage}>Send</button>
            </div>
          ) : null}

          {activeMode === "voice" ? (
            <div className="voice-panel">
              <p>{voiceStatus}</p>
              <div className="voice-actions">
                <button onClick={startVoiceRecording}>Start voice note</button>
                <button className="secondary" onClick={stopVoiceRecording}>
                  Stop and send
                </button>
              </div>
            </div>
          ) : null}

          {activeMode === "call" ? (
            <div className="voice-panel">
              <p>{callStatus}</p>
              <div className="voice-actions">
                <button onClick={startCall} disabled={callActive}>
                  Start live call
                </button>
                <button className="secondary" onClick={stopCall} disabled={!callActive}>
                  End call
                </button>
              </div>
            </div>
          ) : null}
        </footer>
      </section>
    </div>
  );
}

export default App;
