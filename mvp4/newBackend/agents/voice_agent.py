# agents/voice_agent.py
import base64
from groq import AsyncGroq
from config import settings

class VoiceService:
    def __init__(self):
        self.supports_server_stt = bool(settings.groq_api_key)
        self.supports_server_tts = False # Set to True once you plug in ElevenLabs
        
        if self.supports_server_stt:
            self.groq_client = AsyncGroq(api_key=settings.groq_api_key)

    async def transcribe_audio(self, audio_bytes: bytes, mime_type: str, transcript_hint: str = None) -> str:
        if not self.supports_server_stt or not audio_bytes:
            return transcript_hint or "Audio not processed."
        
        # Groq expects a tuple of (filename, file_bytes)
        file_tuple = ("audio.webm", audio_bytes)
        
        # MEDICAL & SLANG PROMPT DICTIONARY
        # This forces Whisper to spell these words correctly if it hears something similar
        medical_context_prompt = (
            "Cardiologist, Neurologist, Orthopedic, Dermatologist, Gastroenterologist, "
            "Panadol, Paracetamol, OPD, appointment, symptoms. "
            "saar ma dard, pait dard, bukhar, ulti, thakan, chakar."
        )
        
        try:
            transcription = await self.groq_client.audio.transcriptions.create(
                file=file_tuple,
                model="whisper-large-v3",
                prompt=medical_context_prompt,
                response_format="text"
            )
            return transcription
        except Exception as e:
            print(f"🎤 [STT Error]: {e}")
            return "Could not transcribe audio."

    async def synthesize_base64_wav(self, text: str) -> str:
        # TODO: Plug in ElevenLabs or EdgeTTS here later.
        # For now, return empty so it doesn't crash.
        return ""

voice_service = VoiceService()