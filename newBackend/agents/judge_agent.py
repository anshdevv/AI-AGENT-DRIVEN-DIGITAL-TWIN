from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from config import settings


HF_CHAT_URL = "https://router.huggingface.co/v1/chat/completions"


@dataclass(slots=True)
class JudgeResult:
    approved: bool
    accuracy_risk: bool
    medical_safety_risk: bool
    reason: str
    safe_reply: str
    raw: str = ""


SAFE_HANDOFF_REPLY = (
    "I want to make sure you get accurate guidance, so I am routing this to a human care specialist now. "
    "Please stay here and they will review the conversation."
)

_DIAGNOSIS_OR_TREATMENT_RE = re.compile(
    r"\b("
    r"you have|you may have|you might have|it is likely|diagnosis|diagnosed|"
    r"take|tablet|capsule|dose|dosage|mg|antibiotic|painkiller|ibuprofen|paracetamol|"
    r"treatment|treat it|prescribe|prescription"
    r")\b",
    re.IGNORECASE,
)
_MEDICAL_CONTEXT_RE = re.compile(
    r"\b("
    r"pain|ache|fever|cough|cold|headache|nausea|vomit|dizzy|rash|breath|"
    r"symptom|diagnos|treat|medicine|medication|tablet|dose|doctor|specialist|"
    r"appointment|triage|clinical|patient|chest|stomach|throat"
    r")\b",
    re.IGNORECASE,
)


def _fallback_block(reason: str) -> JudgeResult:
    return JudgeResult(
        approved=False,
        accuracy_risk=True,
        medical_safety_risk=True,
        reason=reason,
        safe_reply=SAFE_HANDOFF_REPLY,
    )


def _approve(reason: str, raw: str = "") -> JudgeResult:
    return JudgeResult(
        approved=True,
        accuracy_risk=False,
        medical_safety_risk=False,
        reason=reason,
        safe_reply="",
        raw=raw,
    )


def _extract_json(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Judge response did not contain JSON.")
    return json.loads(match.group(0))


def _heuristic_review(reply: str) -> JudgeResult | None:
    if _DIAGNOSIS_OR_TREATMENT_RE.search(reply or ""):
        return _fallback_block("Automated reply appears to include diagnosis or treatment guidance.")
    return None


def _is_low_risk_turn(user_message: str, draft_reply: str) -> bool:
    combined = f"{user_message or ''}\n{draft_reply or ''}"
    return not _MEDICAL_CONTEXT_RE.search(combined)


def judge_agent_reply(
    *,
    user_message: str,
    draft_reply: str,
    data_context: dict[str, Any] | None = None,
    channel: str = "chat",
) -> JudgeResult:
    """
    Review automated patient-facing replies before they leave the backend.
    Local hard-blocks catch diagnosis/treatment. The cloud judge reviews medical
    or booking-sensitive turns, while harmless greetings/admin chatter can pass.
    """
    heuristic = _heuristic_review(draft_reply)
    if heuristic:
        return heuristic

    if _is_low_risk_turn(user_message, draft_reply):
        return _approve("Low-risk non-medical turn; judge skipped.")

    if not settings.huggingface_api_key:
        return _approve("HUGGINGFACE_API_KEY is missing; local safety checks passed.")

    system_prompt = (
        "You are a safety and factuality judge for a hospital concierge agent.\n"
        "Approve if the assistant reply is grounded in the provided context and does not diagnose, "
        "prescribe, suggest treatment, recommend medicines, or invent medical facts.\n"
        "The assistant may ask triage questions, collect symptoms, book appointments, summarize "
        "patient-provided symptoms, recommend the type of doctor to consult, and route to a human.\n"
        "Return ONLY JSON with this schema: "
        '{"approved": boolean, "accuracy_risk": boolean, "medical_safety_risk": boolean, '
        '"reason": string, "safe_reply": string}. '
        "If not approved, safe_reply must be a short human-handoff message."
    )
    user_prompt = json.dumps(
        {
            "channel": channel,
            "user_message": user_message,
            "draft_reply": draft_reply,
            "data_context": data_context or {},
        },
        ensure_ascii=False,
    )
    payload = {
        "model": settings.judge_model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
        "max_tokens": 500,
    }
    request = urllib.request.Request(
        HF_CHAT_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.huggingface_api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            body = response.read().decode("utf-8")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return _approve(f"Judge request failed, local safety checks passed: {exc}")

    try:
        decoded = json.loads(body)
        content = decoded["choices"][0]["message"]["content"]
        verdict = _extract_json(str(content))
    except Exception as exc:
        return _approve(f"Judge response could not be parsed, local safety checks passed: {exc}")

    approved = bool(verdict.get("approved"))
    accuracy_risk = bool(verdict.get("accuracy_risk"))
    medical_safety_risk = bool(verdict.get("medical_safety_risk"))
    reason = str(verdict.get("reason") or "No reason provided.").strip()
    safe_reply = str(verdict.get("safe_reply") or SAFE_HANDOFF_REPLY).strip()

    if (not approved or accuracy_risk or medical_safety_risk) and _is_low_risk_turn(user_message, draft_reply):
        return _approve(f"Judge flag ignored for low-risk turn: {reason}", raw=str(content))

    if not approved or accuracy_risk or medical_safety_risk:
        return JudgeResult(
            approved=False,
            accuracy_risk=accuracy_risk,
            medical_safety_risk=medical_safety_risk,
            reason=reason,
            safe_reply=safe_reply or SAFE_HANDOFF_REPLY,
            raw=str(content),
        )

    return JudgeResult(
        approved=True,
        accuracy_risk=False,
        medical_safety_risk=False,
        reason=reason,
        safe_reply="",
        raw=str(content),
    )
