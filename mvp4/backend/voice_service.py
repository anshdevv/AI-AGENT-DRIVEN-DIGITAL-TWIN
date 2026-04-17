from __future__ import annotations

import base64
import io
from typing import Any

from .settings import settings

try:
    from groq import Groq
except Exception:  # pragma: no cover - optional runtime dependency
    Groq = None

try:
    from elevenlabs.client import AsyncElevenLabs
except Exception:  # pragma: no cover
    AsyncElevenLabs = None


class VoiceService:
    def __init__(self) -> None:
        self.groq_client = Groq(api_key=settings.groq_api_key) if Groq and settings.has_groq else None
        
        # Initialize ElevenLabs with the key from your .env
        self.eleven_client = AsyncElevenLabs(
            api_key=settings.elevenlabs_api_key
        ) if AsyncElevenLabs and settings.elevenlabs_api_key else None

    @property
    def supports_server_stt(self) -> bool:
        return self.groq_client is not None

    @property
    def supports_server_tts(self) -> bool:
        return self.eleven_client is not None

    async def transcribe_audio(
        self,
        *,
        audio_bytes: bytes | None,
        mime_type: str | None,
        transcript_hint: str | None,
    ) -> str:
        # We only use the browser's transcript if Groq is disconnected or audio failed
        if not audio_bytes or not self.supports_server_stt:
            return transcript_hint.strip() if (transcript_hint and transcript_hint.strip()) else ""

        file_name = self._guess_file_name(mime_type)
        
        # We send the audio to Whisper with a 'prompt' that sets the context.
        # This prevents Whisper from auto-translating Urdu into English.
        response = self.groq_client.audio.transcriptions.create(
            file=(file_name, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="text",
            prompt="The user might speak in English, pure Urdu script, or Roman Urdu. Please transcribe exactly what is said without translating it to English.",
        )
        
        return str(response).strip()
    async def synthesize_base64_wav(self, text: str) -> str | None:
        if not text or not self.supports_server_tts:
            return None

        # FIX: Removed the 'await' keyword from this line!
        generator = self.eleven_client.text_to_speech.convert(
            voice_id="nPczCjzI2devNBz1zQrb",
            model_id="eleven_v3",
            text=text,
            output_format="mp3_44100_128"
        )
        audio_buffer = io.BytesIO()
        # The 'async for' loop is where the actual awaiting happens
        async for chunk in generator:
            if chunk:
                audio_buffer.write(chunk)

        # Get the raw MP3 bytes and encode them to base64 to send to React
        mp3_bytes = audio_buffer.getvalue()
        return base64.b64encode(mp3_bytes).decode("utf-8")

    @staticmethod
    def _guess_file_name(mime_type: str | None) -> str:
        if mime_type == "audio/wav":
            return "message.wav"
        if mime_type == "audio/mp3":
            return "message.mp3"
        if mime_type == "audio/ogg":
            return "message.ogg"
        return "message.webm"


voice_service = VoiceService()