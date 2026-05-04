from __future__ import annotations

import asyncio
import base64
import json
import re as _re
import uuid
from urllib import error as url_error
from urllib import parse as url_parse
from urllib import request as url_request

from config import settings


def _detect_lang_from_text(text: str) -> str:
    if not text or not text.strip():
        return "en"
    if _re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", text):
        return "ur"
    if _re.search(r"[\u0900-\u097F]", text):
        return "hi"
    if _re.search(r"[\u0A00-\u0A7F]", text):
        return "pa"
    if _re.search(r"[\u0980-\u09FF]", text):
        return "bn"
    return "en"


class VoiceService:
    def __init__(self) -> None:
        self.elevenlabs_api_key = settings.elevenlabs_api_key
        print(
            f"[VoiceService] ElevenLabs voice stack: {'enabled' if self.elevenlabs_api_key else 'disabled'} "
            f"(stt_model={settings.elevenlabs_stt_model}, stt_hint={settings.elevenlabs_stt_language_hint or 'auto'}, "
            f"tts_model={settings.elevenlabs_tts_model}, voice={settings.elevenlabs_voice_id})"
        )

    @property
    def supports_server_stt(self) -> bool:
        return bool(self.elevenlabs_api_key)

    @property
    def supports_server_tts(self) -> bool:
        return bool(self.elevenlabs_api_key)

    async def transcribe_raw(
        self,
        audio_bytes: bytes | None,
        mime_type: str | None = "audio/webm",
        transcript_hint: str | None = None,
    ) -> tuple[str, str]:
        normalized_mime_type = self._normalize_mime_type(mime_type)
        if audio_bytes and self.elevenlabs_api_key:
            try:
                original_text, detected_lang = await asyncio.to_thread(
                    self._elevenlabs_transcribe,
                    bytes(audio_bytes),
                    normalized_mime_type,
                    settings.elevenlabs_stt_language_hint or None,
                )
                if original_text:
                    return original_text, detected_lang

                if settings.elevenlabs_stt_language_hint:
                    original_text, detected_lang = await asyncio.to_thread(
                        self._elevenlabs_transcribe,
                        bytes(audio_bytes),
                        normalized_mime_type,
                        None,
                    )
                    if original_text:
                        return original_text, detected_lang
            except Exception as e:
                print(f"[STT error] {e}")

        hint = (transcript_hint or "").strip()
        if not hint:
            return "", "en"
        return hint, _detect_lang_from_text(hint)

    async def translate_to_english(self, audio_bytes: bytes, mime_type: str | None = "audio/webm") -> str:
        del audio_bytes, mime_type
        return ""

    async def synthesize_base64_wav(self, text: str) -> str | None:
        if not text or not self.supports_server_tts:
            return None

        clean_text = _strip_system_tags(text)
        if not clean_text:
            return None

        detected_lang = _detect_lang_from_text(clean_text)
        primary_model = settings.elevenlabs_tts_model or "eleven_turbo_v2_5"
        fallback_model = settings.elevenlabs_tts_fallback_model or "eleven_turbo_v2_5"
        candidate_models = [primary_model]
        if detected_lang == "ur" and fallback_model not in candidate_models:
            candidate_models.insert(0, fallback_model)
        elif fallback_model not in candidate_models:
            candidate_models.append(fallback_model)

        last_error: Exception | None = None
        for model_id in candidate_models:
            try:
                mp3_bytes = await asyncio.to_thread(
                    self._elevenlabs_synthesize_mp3,
                    clean_text,
                    model_id,
                    detected_lang,
                )
                if mp3_bytes:
                    return base64.b64encode(mp3_bytes).decode("utf-8")
            except Exception as e:
                last_error = e
                print(f"[TTS synthesis failed][model={model_id}][lang={detected_lang}] {e}")

        if last_error is not None:
            print(f"[TTS synthesis failed] exhausted models for lang={detected_lang}")
        return None

    @staticmethod
    def _guess_file_name(mime_type: str | None) -> str:
        mapping = {
            "audio/wav": "message.wav",
            "audio/mp3": "message.mp3",
            "audio/ogg": "message.ogg",
            "audio/webm": "message.webm",
        }
        return mapping.get(mime_type or "", "message.webm")

    @staticmethod
    def _normalize_mime_type(mime_type: str | None) -> str:
        cleaned = (mime_type or "audio/webm").split(";", 1)[0].strip().lower()
        if cleaned in {"audio/webm", "audio/ogg", "audio/wav", "audio/mpeg", "audio/mp3"}:
            return cleaned
        return "audio/webm"

    def _elevenlabs_transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str,
        language_hint: str | None,
    ) -> tuple[str, str]:
        if not self.elevenlabs_api_key:
            raise RuntimeError("ElevenLabs API key is missing.")

        boundary = f"----CodexBoundary{uuid.uuid4().hex}"
        file_name = self._guess_file_name(mime_type)
        body = self._build_multipart_body(
            boundary=boundary,
            model_id=settings.elevenlabs_stt_model or "scribe_v2",
            file_name=file_name,
            mime_type=mime_type,
            audio_bytes=audio_bytes,
            language_code=language_hint,
        )

        req = url_request.Request(
            "https://api.elevenlabs.io/v1/speech-to-text",
            data=body,
            method="POST",
            headers={
                "xi-api-key": self.elevenlabs_api_key,
                "Content-Type": f"multipart/form-data; boundary={boundary}",
                "Accept": "application/json",
            },
        )
        try:
            with url_request.urlopen(req, timeout=30) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except url_error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ElevenLabs STT HTTP {e.code}: {body[:200]}") from e
        except Exception as e:
            raise RuntimeError(f"ElevenLabs STT request failed: {e}") from e

        transcript = str(payload.get("text") or "").strip()
        detected_lang = self._normalize_detected_language(
            str(payload.get("language_code") or "en").lower().strip() or "en",
            transcript,
            language_hint,
        )
        return transcript, detected_lang

    @staticmethod
    def _build_multipart_body(
        *,
        boundary: str,
        model_id: str,
        file_name: str,
        mime_type: str,
        audio_bytes: bytes,
        language_code: str | None,
    ) -> bytes:
        parts: list[bytes] = []
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="model_id"\r\n\r\n'
                f"{model_id}\r\n"
            ).encode("utf-8")
        )
        if language_code:
            parts.append(
                (
                    f"--{boundary}\r\n"
                    f'Content-Disposition: form-data; name="language_code"\r\n\r\n'
                    f"{language_code}\r\n"
                ).encode("utf-8")
            )
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="file"; filename="{file_name}"\r\n'
                f"Content-Type: {mime_type}\r\n\r\n"
            ).encode("utf-8")
        )
        parts.append(audio_bytes)
        parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
        return b"".join(parts)

    @staticmethod
    def _normalize_detected_language(
        detected_lang: str,
        transcript: str,
        language_hint: str | None,
    ) -> str:
        lang = (detected_lang or "en").strip().lower()
        hint = (language_hint or "").strip().lower()

        if lang in {"hi", "hindi"}:
            if hint == "ur" or _re.search(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]", transcript or ""):
                return "ur"
        return lang

    def _elevenlabs_synthesize_mp3(
        self,
        text: str,
        model_id: str,
        language_code: str | None,
    ) -> bytes:
        if not self.elevenlabs_api_key:
            raise RuntimeError("ElevenLabs API key is missing.")

        voice_id = settings.elevenlabs_voice_id
        if not voice_id:
            raise RuntimeError("ElevenLabs voice ID is missing.")

        query = url_parse.urlencode(
            {
                "output_format": settings.elevenlabs_output_format or "mp3_44100_128",
            }
        )
        endpoint = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?{query}"
        payload: dict[str, str] = {
            "text": text,
            "model_id": model_id,
        }
        if language_code and language_code != "en":
            payload["language_code"] = language_code
        body = json.dumps(payload).encode("utf-8")
        req = url_request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "xi-api-key": self.elevenlabs_api_key,
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
            },
        )
        try:
            with url_request.urlopen(req, timeout=30) as resp:
                return resp.read()
        except url_error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"ElevenLabs HTTP {e.code}: {body[:200]}") from e
        except Exception as e:
            raise RuntimeError(f"ElevenLabs request failed: {e}") from e


def _strip_system_tags(text: str) -> str:
    import re

    text = re.sub(r"\[MEDGEMMA_SUMMARY:[^\]]*\]", "", text)
    text = re.sub(r"\[SYMPTOM_LOGGED:[^\]]*\]", "", text)
    for tag in ["[TRIAGE_COMPLETE]", "[START_TRIAGE]", "[END_CALL]"]:
        text = text.replace(tag, "")
    return text.strip()


voice_service = VoiceService()
