from __future__ import annotations

import asyncio
import base64
from dataclasses import dataclass
from typing import Any, Callable

from fastapi import WebSocket


@dataclass(slots=True)
class PipelineTurn:
    transcript: str
    channel: str = "call"


class PipecatCallPipeline:
    """
    Persistent bidirectional call pipeline:
      user audio -> STT/translation -> LangGraph -> TTS -> client playback

    This is intentionally long-lived for the entire websocket session and supports
    interruption: if user speaks while assistant audio is still playing, backend
    emits `assistant_interrupted` so frontend can stop playback immediately.
    """

    def __init__(
        self,
        *,
        websocket: WebSocket,
        session_id: str,
        voice_service: Any,
        process_with_langgraph: Callable[[str, str, str, str], dict],
        session_lang_store: dict[str, str],
    ) -> None:
        self.websocket = websocket
        self.session_id = session_id
        self.voice_service = voice_service
        self.process_with_langgraph = process_with_langgraph
        self.session_lang_store = session_lang_store

        self._turn_queue: asyncio.Queue[PipelineTurn] = asyncio.Queue()
        self._worker_task: asyncio.Task | None = None
        self._closed = False

        self._assistant_speaking = False
        self._generation_id = 0
        self._lock = asyncio.Lock()

    async def start(self) -> None:
        self._worker_task = asyncio.create_task(self._turn_worker())
        await self._safe_send(
            {
                "type": "call_ready",
                "session_id": self.session_id,
                "server_tts": self.voice_service.supports_server_tts,
                "server_stt": self.voice_service.supports_server_stt,
                "pipeline": "pipecat",
            }
        )

    async def shutdown(self) -> None:
        self._closed = True
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    async def handle_client_message(self, payload: dict[str, Any]) -> None:
        message_type = payload.get("type")
        print(f"[Pipecat] <- {message_type}")

        if message_type == "ping":
            await self._safe_send({"type": "pong"})
            return

        if message_type == "interrupt":
            await self._interrupt_assistant(reason="user_interrupt_signal")
            return

        if message_type == "assistant_playback_done":
            async with self._lock:
                generation_id = int(payload.get("generation_id") or 0)
                if generation_id == self._generation_id:
                    self._assistant_speaking = False
            return

        if message_type == "user_audio":
            await self._handle_user_audio(payload)
            return

        if message_type == "user_transcript":
            text = (payload.get("text") or "").strip()
            if not text:
                await self._safe_send({"type": "error", "message": "Transcript is empty."})
                return
            await self._enqueue_turn(text)
            return

        await self._safe_send({"type": "error", "message": "Unsupported message type."})

    async def _handle_user_audio(self, payload: dict[str, Any]) -> None:
        audio_b64 = payload.get("audio_base64") or ""
        mime_type = payload.get("mime_type", "audio/webm")

        if not audio_b64:
            await self._safe_send({"type": "error", "message": "audio_base64 is empty."})
            return

        try:
            audio_bytes = base64.b64decode(audio_b64)
        except Exception:
            await self._safe_send({"type": "error", "message": "Invalid audio payload."})
            return
        print(f"[Pipecat] audio bytes={len(audio_bytes)} mime={mime_type}")

        try:
            raw_transcript, detected_lang = await self.voice_service.transcribe_raw(
                audio_bytes=audio_bytes,
                mime_type=mime_type,
            )
        except Exception as e:
            await self._safe_send({"type": "error", "message": f"Speech recognition failed: {e}"})
            return

        if detected_lang not in ("en", "english"):
            self.session_lang_store[self.session_id] = detected_lang

        if not raw_transcript.strip():
            print("[Pipecat] empty transcript from STT")
            return

        if detected_lang not in ("en", "english"):
            try:
                english_transcript = await self.voice_service.translate_to_english(
                    audio_bytes=audio_bytes,
                    mime_type=mime_type,
                )
                transcript = english_transcript.strip() or raw_transcript
            except Exception:
                transcript = raw_transcript
        else:
            transcript = raw_transcript

        await self._interrupt_assistant(reason="user_barge_in")
        await self._safe_send(
            {
                "type": "user_transcript_echo",
                "text": transcript,
                "detected_language": detected_lang,
            }
        )
        await self._enqueue_turn(transcript)

    async def _enqueue_turn(self, transcript: str) -> None:
        await self._turn_queue.put(PipelineTurn(transcript=transcript, channel="call"))
        print(f"[Pipecat] queued turn, queue_size={self._turn_queue.qsize()}")

    async def _turn_worker(self) -> None:
        while not self._closed:
            turn = await self._turn_queue.get()
            target_lang = self.session_lang_store.get(self.session_id, "en")

            try:
                async with self._lock:
                    self._generation_id += 1
                    generation_id = self._generation_id

                state = await asyncio.to_thread(
                    self.process_with_langgraph,
                    self.session_id,
                    turn.transcript,
                    turn.channel,
                    target_lang,
                )
                messages = state.get("messages", [])
                reply_text = messages[-1].content if messages else "I'm sorry, I couldn't process that."

                await self._safe_send(
                    {
                        "type": "assistant_response",
                        "text": reply_text,
                        "detected_language": target_lang,
                        "triage_active": state.get("triage_active", False),
                        "human_handoff": state.get("human_handoff", False),
                        "generation_id": generation_id,
                    }
                )
                print(f"[Pipecat] -> assistant_response(gen={generation_id})")

                # Generate TTS after text is already sent to reduce perceived latency.
                audio_reply = await self.voice_service.synthesize_base64_wav(reply_text)

                async with self._lock:
                    # If generation changed, this turn was interrupted/replaced.
                    if generation_id != self._generation_id:
                        print(f"[Pipecat] stale audio dropped for gen={generation_id}")
                        continue
                    self._assistant_speaking = bool(audio_reply)

                await self._safe_send(
                    {
                        "type": "assistant_audio",
                        "audio_base64": audio_reply,
                        "generation_id": generation_id,
                    }
                )
                print(f"[Pipecat] -> assistant_audio(gen={generation_id})")
            except Exception as e:
                print(f"[Pipecat] worker error: {e}")
                await self._safe_send(
                    {
                        "type": "error",
                        "message": f"Pipeline processing failed: {e}",
                    }
                )
            finally:
                self._turn_queue.task_done()

    async def _interrupt_assistant(self, *, reason: str) -> None:
        async with self._lock:
            interrupted_generation = self._generation_id
            was_speaking = self._assistant_speaking
            self._assistant_speaking = False
            # Invalidate any pending/stale TTS for current generation.
            self._generation_id += 1

        if was_speaking:
            await self._safe_send(
                {
                    "type": "assistant_interrupted",
                    "reason": reason,
                    "generation_id": interrupted_generation,
                }
            )

    async def _safe_send(self, payload: dict[str, Any]) -> None:
        if self._closed:
            return
        try:
            await self.websocket.send_json(payload)
        except Exception:
            self._closed = True
