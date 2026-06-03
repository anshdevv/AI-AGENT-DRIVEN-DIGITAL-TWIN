# agents/voice_agent.py
from __future__ import annotations

import asyncio
import base64
import io
import re as _re

from config import settings

try:
    from groq import AsyncGroq
except Exception:
    AsyncGroq = None

try:
    from elevenlabs.client import AsyncElevenLabs
except Exception:
    AsyncElevenLabs = None

try:
    from huggingface_hub import AsyncInferenceClient as HFAsyncClient
except Exception:
    HFAsyncClient = None

# ── Language code → translation instruction for the LLM ───────────────────────
_LANG_NAMES: dict[str, str] = {
    "ur": "Urdu", "hi": "Hindi", "ar": "Arabic", "fa": "Persian/Farsi",
    "pa": "Punjabi", "bn": "Bengali", "ps": "Pashto", "sd": "Sindhi",
    "urdu": "Urdu", "hindi": "Hindi", "arabic": "Arabic", "punjabi": "Punjabi",
    "bengali": "Bengali", "pashto": "Pashto", "sindhi": "Sindhi",
}

_ENGLISH_CODES = {"en", "english"}
_HF_WHISPER_MODEL = "openai/whisper-large-v3"

# Testing-only language lock. Set False when you want language auto-detection again.
FORCE_ENGLISH_TEST = False


def _detect_lang_from_text(text: str) -> str:
    """Detect language by checking the Unicode script of the transcribed text."""
    if not text or not text.strip():
        return "en"
    if _re.search(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]", text):
        return "ur"
    if _re.search(r"[ऀ-ॿ]", text):
        return "hi"
    if _re.search(r"[਀-੿]", text):
        return "pa"
    if _re.search(r"[ঀ-৿]", text):
        return "bn"
    return "en"


class VoiceService:
    def __init__(self) -> None:
        # 🎛️ TOGGLE THIS TO SWITCH BETWEEN GROQ AND HUGGING FACE
        self.USE_GROQ_AS_PRIMARY = True  
        raw_key = settings.groq_api_key
        print(f"🔑 [VoiceService] GROQ_API_KEY loaded: {'✅ SET' if raw_key else '❌ EMPTY/MISSING'}")
        if raw_key:
            print(f"🔑 [VoiceService] Key preview: {raw_key[:8]}...{raw_key[-4:]} (len={len(raw_key)})")
        else:
            print("🔑 [VoiceService] ⚠️ Key is empty — Groq client will NOT be created!")

        self.groq_client = (
            AsyncGroq(api_key=settings.groq_api_key)
            if AsyncGroq and settings.groq_api_key
            else None
        )
        print(f"🔑 [VoiceService] Groq client created: {'✅ Yes' if self.groq_client else '❌ No'}")

        self.hf_client = (
            HFAsyncClient(
                provider="hf-inference",
                api_key=settings.huggingface_api_key,
            )
            if HFAsyncClient and getattr(settings, "huggingface_api_key", None)
            else None
        )

        self.eleven_client = (
            AsyncElevenLabs(api_key=settings.elevenlabs_api_key)
            if AsyncElevenLabs and getattr(settings, "elevenlabs_api_key", None)
            else None
        )

        print(f"🎤 [VoiceService] Initialized. Primary Engine set to: {'GROQ' if self.USE_GROQ_AS_PRIMARY else 'HUGGING FACE'}")

    @property
    def supports_server_stt(self) -> bool:
        return self.hf_client is not None or self.groq_client is not None

    @property
    def supports_server_tts(self) -> bool:
        return self.eleven_client is not None


    # ── 1. TRANSCRIBE RAW ──────────────────────────────────────────────────
    async def transcribe_raw(
        self,
        audio_bytes: bytes,
        mime_type: str | None = "audio/webm",
    ) -> tuple[str, str]:
        """
        Returns (text_for_pipeline, detected_language_code).

        Normal mode:
          - Groq/HF Whisper transcribes in the spoken language.
          - Non-English is treated as Urdu for the rest of the pipeline.

        FORCE_ENGLISH_TEST mode:
          - Groq uses Whisper's TRANSLATION endpoint, so Urdu/Hindi/Roman Urdu
            speech is converted to English text before it reaches Qwen/MedGemma.
          - This is the correct way to stop Urdu text from entering the app.
          - Overriding detected_lang alone is not enough because transcription
            has already happened by then.
        """
        if not audio_bytes or not self.supports_server_stt:
            return "", "en"

        # Reject suspiciously small audio — likely silence or a failed recording
        if len(audio_bytes) < 1000:
            print(f"🚫 [STT] Audio too small ({len(audio_bytes)} bytes) — likely silence, skipping")
            return "", "en"

        file_name = self._guess_file_name(mime_type)

        # ── ENGLISH-ONLY TEST MODE ──────────────────────────────────────────
        # Important: use translations.create, not transcriptions.create.
        # transcriptions.create can still output Urdu script; translations.create
        # returns English text.
        if FORCE_ENGLISH_TEST and self.groq_client:
            try:
                print("🎤 [STT] FORCE_ENGLISH_TEST=True — using Groq Whisper TRANSLATION → English...")
                result = await self.groq_client.audio.translations.create(
                    file=(file_name, bytes(audio_bytes)),
                    model="whisper-large-v3",
                    response_format="text",
                    prompt="The patient is describing medical symptoms, appointment details, names, dates, and times.",
                )
                english_text = (
                    result if isinstance(result, str)
                    else getattr(result, "text", str(result))
                ).strip()

                if not english_text:
                    return "", "en"

                _HALLUCINATION_SIGNALS = (
                    "transcribe exactly", "spoken in english", "spoken in urdu",
                    "the user may", "native script", "may speak in",
                    "describing his medical", "describing their medical",
                    "headache, fever, and pain", "medical symptoms",
                    "such as headache", "fever, and pain",
                    "born on march", "loved one in", "a loved one",
                    "the person born",
                )
                if any(sig in english_text.lower() for sig in _HALLUCINATION_SIGNALS):
                    print(f"🚫 [STT] Translation hallucination detected — dropping: {english_text[:80]}")
                    return "", "en"

                print(f"🔄 [STT] English transcript for pipeline: {english_text[:120]}")
                return english_text, "en"
            except Exception as e:
                print(f"❌ [Groq-Translate STT Error]: {e}")
                # Fall through to normal transcription only if translation fails.
                # This keeps the app usable, but may not be English-only.

        if self.USE_GROQ_AS_PRIMARY and self.groq_client:
            try:
                print("🎤 [STT] Using Groq (whisper-large-v3) — transcription...")
                result = await self.groq_client.audio.transcriptions.create(
                    file=(file_name, bytes(audio_bytes)),
                    model="whisper-large-v3",
                    response_format="verbose_json",
                    prompt="symptoms: pain, fever, cough, headache, nausea, vomiting, dizziness",
                )
                original_text = (getattr(result, "text", None) or "").strip()
                detected_lang = (getattr(result, "language", None) or "en").lower().strip()

                # ── no_speech_prob guard — most reliable silence detector ──
                segments = getattr(result, "segments", None) or []
                if segments:
                    avg_no_speech = sum(s.get("no_speech_prob", 0) for s in segments) / len(segments)
                    print(f"🔇 [STT] avg no_speech_prob={avg_no_speech:.2f}")
                    if avg_no_speech > 0.6:
                        print(f"🚫 [STT] High no_speech_prob ({avg_no_speech:.2f}) — dropping hallucination: {original_text[:80]}")
                        return "", "en"

                _HALLUCINATION_SIGNALS = (
                    "transcribe exactly", "spoken in english", "spoken in urdu",
                    "the user may", "native script", "may speak in",
                    "describing his medical", "describing their medical",
                    "headache, fever, and pain", "medical symptoms",
                    "such as headache", "fever, and pain",
                    "born on march", "loved one in", "a loved one",
                    "the person born",
                )

                import unicodedata
                def _has_cjk(text: str) -> bool:
                    return any(
                        unicodedata.category(c) in ("Lo",) and "\u4e00" <= c <= "\u9fff"
                        or "\u3040" <= c <= "\u30ff"
                        for c in text
                    )

                if _has_cjk(original_text):
                    print(f"🚫 [STT] CJK characters detected — multilingual hallucination, dropping: {original_text[:80]}")
                    return "", "en"
                if any(sig in original_text.lower() for sig in _HALLUCINATION_SIGNALS):
                    print(f"🚫 [STT] Hallucination detected — dropping: {original_text[:80]}")
                    return "", "en"

                # Normal mode: only English is English. Everything else → Urdu.
                if detected_lang not in ("en", "english"):
                    print(f"🔤 [STT] Non-English detected ('{detected_lang}') → treating as Urdu")
                    detected_lang = "ur"

                return original_text, detected_lang
            except Exception as e:
                print(f"❌ [Groq-STT Error]: {e}")
                return "", "en"

        elif self.hf_client:
            try:
                print(f"🎤 [STT] Using Hugging Face ({_HF_WHISPER_MODEL})...")
                result = await self.hf_client.automatic_speech_recognition(
                    audio_bytes, model=_HF_WHISPER_MODEL
                )
                original_text = result.get("text", "") if isinstance(result, dict) else getattr(result, "text", str(result))
                original_text = original_text.strip()

                # HF fallback cannot reliably force translation in this wrapper.
                # In English-test mode, do not allow Urdu-script text into the pipeline.
                if FORCE_ENGLISH_TEST:
                    detected = _detect_lang_from_text(original_text)
                    if detected != "en":
                        print(f"🚫 [STT] FORCE_ENGLISH_TEST=True but HF returned non-English text ({detected}); dropping transcript")
                        return "", "en"
                    return original_text, "en"

                detected_lang = _detect_lang_from_text(original_text)
                return original_text, detected_lang
            except Exception as e:
                print(f"❌ [HF-STT Error]: {e}")
                return "", "en"

        return "", "en"


    # ── 2. TRANSLATE TO ENGLISH via Whisper (for MedGemma) ─────────────────
    async def translate_to_english(self, audio_bytes: bytes, mime_type: str | None = "audio/webm") -> str:
        """
        Uses Whisper's built-in translation task to convert any language → English.
        No LLM involved — no <think> blocks, no extra round-trip.
        Falls back to empty string on failure.
        """
        if not audio_bytes or not self.groq_client:
            return ""

        file_name = self._guess_file_name(mime_type)
        print("🔄 [Translate] Whisper translate task → English for MedGemma...")
        try:
            result = await self.groq_client.audio.translations.create(
                file=(file_name, bytes(audio_bytes)),
                model="whisper-large-v3",   # turbo doesn't support translations endpoint
                response_format="text",
                prompt="The patient is describing their medical symptoms, such as headache, fever, and pain.",
            )
            # Groq returns a plain string for response_format="text"
            english_text = (result if isinstance(result, str) else getattr(result, "text", str(result))).strip()
            print(f"🔄 [Translate] English output: {english_text[:120]}")
            return english_text
        except Exception as e:
            print(f"❌ [Whisper-Translate Error]: {e}")
            return ""


    # ── TTS & HELPERS ──────────────────────────────────────────────────────────
    async def synthesize_base64_wav(self, text: str) -> str | None:
        if not text or not self.supports_server_tts:
            return None

        clean_text = _strip_system_tags(text)
        if not clean_text.strip():
            return None

        try:
            generator = self.eleven_client.text_to_speech.convert(
                voice_id="nPczCjzI2devNBz1zQrb",
                model_id="eleven_turbo_v2_5",
                text=clean_text,
                output_format="mp3_44100_128",
            )
            audio_buffer = io.BytesIO()
            async for chunk in generator:
                if chunk:
                    audio_buffer.write(chunk)

            mp3_bytes = audio_buffer.getvalue()
            if not mp3_bytes:
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
    import re
    text = re.sub(r"\[MEDGEMMA_SUMMARY:[^\]]*\]", "", text)
    text = re.sub(r"\[SYMPTOM_LOGGED:[^\]]*\]", "", text)
    for tag in ["[TRIAGE_COMPLETE]", "[START_TRIAGE]", "[END_CALL]"]:
        text = text.replace(tag, "")
    return text.strip()


voice_service = VoiceService()
