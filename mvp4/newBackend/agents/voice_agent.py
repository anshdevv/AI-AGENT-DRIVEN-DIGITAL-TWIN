# agents/voice_agent.py
from __future__ import annotations

import base64
import io

from config import settings

try:
    from groq import AsyncGroq
except Exception:
    AsyncGroq = None

try:
    from elevenlabs.client import AsyncElevenLabs
except Exception:
    AsyncElevenLabs = None


class VoiceService:
    def __init__(self) -> None:
        self.groq_client = (
            AsyncGroq(api_key=settings.groq_api_key)
            if AsyncGroq and getattr(settings, "groq_api_key", None)
            else None
        )
        self.eleven_client = (
            AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
            if AsyncElevenLabs and getattr(settings, "elevenlabs_api_key", None)
            else None
        )

    @property
    def supports_server_stt(self) -> bool:
        return self.groq_client is not None

    @property
    def supports_server_tts(self) -> bool:
        return self.eleven_client is not None

    async def transcribe_audio(
        self,
        audio_bytes: bytes | None = None,
        mime_type: str | None = None,
        transcript_hint: str | None = None,
    ) -> str:
        if not audio_bytes or not self.supports_server_stt:
            return transcript_hint.strip() if (transcript_hint and transcript_hint.strip()) else ""

        file_name = self._guess_file_name(mime_type)

        try:
            transcription = await self.groq_client.audio.transcriptions.create(
                file=(file_name, audio_bytes),
                model="whisper-large-v3-turbo",
                response_format="text",
                prompt=(
                    "The user might speak in English, pure Urdu script, or Roman Urdu. "
                    "Transcribe exactly what is said without translating. "
                    "Medical words: Cardiologist, Neurologist, Orthopedic, Dermatologist, "
                    "Gastroenterologist, Panadol, Paracetamol, OPD, appointment, symptoms, "
                    "sar dard, pait dard, bukhar, ulti, thakan, chakkar."
                ),
            )
            return str(transcription).strip()
        except Exception as e:
            print(f"🎤 [STT Error]: {e}")
            return transcript_hint or "Could not transcribe audio."

    async def synthesize_base64_wav(self, text: str) -> str | None:
        """
        Convert text to speech via ElevenLabs.
        Returns base64-encoded MP3 string, or None if TTS is unavailable.
        Text is cleaned of any internal system tags before sending.
        """
        if not text or not self.supports_server_tts:
            return None

        clean_text = _strip_system_tags(text)
        if not clean_text.strip():
            return None

        try:
            generator = self.eleven_client.text_to_speech.convert(
                voice_id="nPczCjzI2devNBz1zQrb",
                model_id="eleven_turbo_v2_5",   # lower latency than eleven_v3 for voice calls
                text=clean_text,
                output_format="mp3_44100_128",
            )
            audio_buffer = io.BytesIO()
            async for chunk in generator:
                if chunk:
                    audio_buffer.write(chunk)

            mp3_bytes = audio_buffer.getvalue()
            if not mp3_bytes:
                print("⚠️  [TTS] ElevenLabs returned empty audio")
                return None

            encoded = base64.b64encode(mp3_bytes).decode("utf-8")
            print(f"🔊 [TTS] Generated {len(mp3_bytes):,} bytes of audio")
            return encoded

        except Exception as e:
            print(f"🔊 [TTS Error]: {e}")
            return None

    @staticmethod
    def _guess_file_name(mime_type: str | None) -> str:
        mapping = {
            "audio/wav":  "message.wav",
            "audio/mp3":  "message.mp3",
            "audio/ogg":  "message.ogg",
            "audio/webm": "message.webm",
        }
        return mapping.get(mime_type or "", "message.webm")


def _strip_system_tags(text: str) -> str:
    """
    Remove internal orchestrator/triage tags that should never be
    read aloud to the patient.
    e.g. [MEDGEMMA_SUMMARY: ...], [TRIAGE_COMPLETE], [START_TRIAGE], etc.
    """
    import re
    # Remove block tags with content: [TAG: content]
    text = re.sub(r"\[MEDGEMMA_SUMMARY:[^\]]*\]", "", text)
    text = re.sub(r"\[SYMPTOM_LOGGED:[^\]]*\]", "", text)
    # Remove standalone control tags
    for tag in ["[TRIAGE_COMPLETE]", "[START_TRIAGE]", "[END_CALL]"]:
        text = text.replace(tag, "")
    return text.strip()


voice_service = VoiceService()