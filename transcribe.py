import asyncio
import json
from flask import Flask, render_template_string
from flask_sock import Sock
from deepgram import DeepgramClient

app = Flask(__name__)
sock = Sock(app)

DEEPGRAM_API_KEY = "5df4c91593d3ada2f6affd516f744a1782769396"

HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Live STT</title>
</head>
<body style="font-family: Arial; padding: 40px;">
    <h2>Live Speech → Text (Deepgram)</h2>

    <button onclick="start()">Start</button>
    <button onclick="stop()">Stop</button>

    <h3>Transcript</h3>
    <div id="output" style="border:1px solid #ccc; padding:10px; min-height:100px;"></div>

<script>
let socket;
let mediaRecorder;

async function start() {
    socket = new WebSocket("ws://" + location.host + "/listen");

    socket.onmessage = (msg) => {
        document.getElementById("output").innerText = msg.data;
    };

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

    mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

    mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0 && socket.readyState === 1) {
            event.data.arrayBuffer().then(buffer => {
                socket.send(buffer);
            });
        }
    };

    mediaRecorder.start(250); // send chunks every 250ms
}

function stop() {
    if (mediaRecorder) mediaRecorder.stop();
    if (socket) socket.close();
}
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)



@sock.route("/listen")
def listen(ws):
    dg = DeepgramClient(api_key=DEEPGRAM_API_KEY)

    # create websocket connection to Deepgram
    dg_socket = dg.listen.v("1")

    transcript = ""

    def on_message(result, **kwargs):
        nonlocal transcript
        try:
            sentence = result["channel"]["alternatives"][0]["transcript"]
            if sentence:
                transcript += " " + sentence
                ws.send(transcript)
        except:
            pass

    dg_socket.on("transcript", on_message)

    dg_socket.start({
        "model": "nova-2",
        "language": "en-US",
        "encoding": "opus",
        "sample_rate": 48000,
    })

    try:
        while True:
            data = ws.receive()
            if data is None:
                break
            dg_socket.send(data)
    finally:
        dg_socket.finish()
if __name__ == "__main__":
    app.run(debug=True)