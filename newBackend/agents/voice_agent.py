from __future__ import annotations

import base64
import io
import sys
import wave
from pathlib import Path
from typing import Any

# Ensure the workspace root is importable so sibling package imports work
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

try:
    from backend.settings import settings as settings
except ModuleNotFoundError:  # pragma: no cover - fallback when running from a different CWD
    from config import settings

try:
    import edge_tts
    import numpy as np
except Exception:  # pragma: no cover - optional runtime dependency
    edge_tts = None
    np = None

try:
    from groq import Groq
except Exception:  # pragma: no cover - optional runtime dependency
    Groq = None


class VoiceService:
    def __init__(self) -> None:
        self.groq_client = Groq(api_key=settings.groq_api_key) if Groq and settings.has_groq else None

    @property
    def supports_server_stt(self) -> bool:
        return self.groq_client is not None

    @property
    def supports_server_tts(self) -> bool:
        return edge_tts is not None and np is not None

    async def transcribe_audio(
        self,
        *,
        audio_bytes: bytes | None,
        mime_type: str | None,
        transcript_hint: str | None,
    ) -> str:
        if transcript_hint and transcript_hint.strip():
            return transcript_hint.strip()
        if not audio_bytes or not self.supports_server_stt:
            return ""

        file_name = self._guess_file_name(mime_type)
        response = self.groq_client.audio.transcriptions.create(
            file=(file_name, audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="text",
        )
        return str(response).strip()

    async def synthesize_base64_wav(self, text: str) -> str | None:
        if not text or not self.supports_server_tts:
            return None
        pcm_audio, sample_rate = await self._synthesize_pcm(text)
        wav_bytes = self._pcm_to_wav_bytes(pcm_audio, sample_rate)
        return base64.b64encode(wav_bytes).decode("utf-8")

    async def _synthesize_pcm(self, text: str) -> tuple[Any, int]:
        communicate = edge_tts.Communicate(
            text,
            "en-US-AndrewNeural",
            codec="audio-24khz-16bit-mono-pcm",
        )
        pcm_buffer = io.BytesIO()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                pcm_buffer.write(chunk["data"])

        pcm_bytes = pcm_buffer.getvalue()
        audio = np.frombuffer(pcm_bytes, dtype=np.int16)
        return audio, 24_000

    @staticmethod
    def _pcm_to_wav_bytes(audio, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio.tobytes())
        return buffer.getvalue()

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
