import io
import pyaudio # <--- NEW IMPORT
from elevenlabs import ElevenLabs
from pydub import AudioSegment

from elevenlabs.client import AsyncElevenLabs

client = ElevenLabs(api_key="")

def tts_sentence_audio(text):
    stream = client.text_to_speech.convert(
        voice_id="oH8YmZXJYEZq5ScgoGn9",
        model_id="eleven_multilingual_v2",
        text=text,
        output_format="mp3_44100_128"
    )

    audio_buffer = io.BytesIO()
    for chunk in stream:
        if chunk:
            audio_buffer.write(chunk)

    audio_buffer.seek(0)
    audio = AudioSegment.from_mp3(audio_buffer)
    # --- PYAUDIO PLAYBACK ---
    p = pyaudio.PyAudio()
    
    # Open a playback stream
    stream_out = p.open(format=p.get_format_from_width(audio.sample_width),
                        channels=audio.channels,
                        rate=audio.frame_rate,
                        output=True)
# Play the audio in chunks
    chunk_size = 1024
    data = audio.raw_data
    
    while data:
        stream_out.write(data[:chunk_size])
        data = data[chunk_size:]

    # Stop and close the PyAudio stream
    stream_out.stop_stream()
    stream_out.close()
    p.terminate()