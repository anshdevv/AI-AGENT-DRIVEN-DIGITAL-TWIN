# agents/triage_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Option B: MedGemma → Qwen direct chain inside one node.
# No disguise messages. No cross-node routing tricks.
#
# Flow per turn:
#   1. MedGemma reads CLEAN clinical history → produces raw clinical question
#   2. Qwen rephrases it warmly for the patient → AIMessage returned
#   3. triage_router → END   (wait for patient reply)
#
# On completion ([TRIAGE_COMPLETE] or 8 questions reached):
#   1. Qwen generates handoff message
#   2. triage_active = False → triage_router → supervisor_node → booking
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import re
from pathlib import Path

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from agents.llm_config import get_llm
from agents.symptom_lookup import lookup, format_for_prompt
from agents.mcp_tools import save_case_notes

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "triage_system.md"

MAX_TRIAGE_QUESTIONS = 8


# ── Prompt loader ─────────────────────────────────────────────────────────────

def _load_triage_prompt() -> str:
    if _PROMPT_PATH.exists():
        return _PROMPT_PATH.read_text(encoding="utf-8")
    return (
        "You are a clinical pre-triage assistant.\n"
        "Ask ONE focused follow-up question per turn about the patient's complaint.\n"
        f"Max {MAX_TRIAGE_QUESTIONS} questions total. "
        "When done, output [TRIAGE_COMPLETE] and a CLINICAL_SUMMARY block."
    )


# ── MedGemma client ───────────────────────────────────────────────────────────

_med_llm: ChatOllama | None = None


def _get_med_llm() -> ChatOllama:
    global _med_llm
    if _med_llm is None:
        print("🔧 [Triage] Connecting to Ollama medgemma:4b ...")
        _med_llm = ChatOllama(model="medgemma:4b", temperature=0.0)
    return _med_llm


# ── Qwen client (rephrasing only — warm & patient-facing) ─────────────────────

_qwen_llm = None


def _get_qwen_llm():
    global _qwen_llm
    if _qwen_llm is None:
        print("🔧 [Triage] Initialising Qwen rephrasing client ...")
        _qwen_llm = get_llm(temperature=0.3)
    return _qwen_llm


# ── Helpers ───────────────────────────────────────────────────────────────────

_SUMMARY_RE = re.compile(r"CLINICAL_SUMMARY:(.*?)(?:\Z|\[)", re.DOTALL | re.IGNORECASE)
_SYSTEM_TAG_RE = re.compile(r"\[SYMPTOM_LOGGED:[^\]]+\]|\[START_TRIAGE\]|\[END_CALL\]")


def _extract_clinical_summary(text: str) -> str:
    m = _SUMMARY_RE.search(text)
    if m:
        return m.group(1).strip()
    parts = text.split("[TRIAGE_COMPLETE]", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _get_questions_asked(state: dict) -> int:
    """booking_context is the single source of truth for question count."""
    return state.get("booking_context", {}).get("triage_questions_asked", 0)


def _build_triage_messages(state: dict, sys_prompt_text: str) -> list:
    """
    Build a clean history for MedGemma.
    Only includes messages from AFTER [START_TRIAGE] — so MedGemma only sees
    the real clinical Q&A, not the pre-triage symptom description or booking noise.
    This prevents repeated questions caused by seeing the same symptom multiple times.
    """
    sys_msg  = SystemMessage(content=sys_prompt_text)
    all_msgs = list(state.get("messages", []))

    # Find the index of the message containing [START_TRIAGE]
    triage_start_idx = 0
    for i, m in enumerate(all_msgs):
        if m.type == "ai" and "[START_TRIAGE]" in str(m.content):
            triage_start_idx = i + 1   # everything AFTER the trigger message
            break

    triage_msgs = all_msgs[triage_start_idx:]

    # Strip orchestrator control tags from AI messages
    clean = []
    for m in triage_msgs:
        content = str(m.content)
        if m.type == "ai":
            stripped = _SYSTEM_TAG_RE.sub("", content).strip()
            if stripped:
                clean.append(AIMessage(content=stripped))
        else:
            clean.append(m)

    print(f"   [TriageMsgs] Using {len(clean)} msgs (post-START_TRIAGE, of {len(all_msgs)} total)")
    return [sys_msg] + clean


def _record_qa_pair(state: dict, existing_qa: list[str]) -> list[str]:
    """
    Find the most recent (AI question, Human answer) pair and append it.
    With Option B the history is clean so this reliably finds real pairs.
    """
    messages = list(state.get("messages", []))
    last_human_answer = ""
    last_ai_question = ""

    for m in reversed(messages):
        if not last_human_answer and m.type == "human":
            last_human_answer = str(m.content).strip()
        elif last_human_answer and m.type == "ai":
            # Strip any leftover routing tags before storing
            question = _SYSTEM_TAG_RE.sub("", str(m.content)).strip()
            if question:
                last_ai_question = question
            break

    if last_ai_question and last_human_answer:
        existing_qa.append(f"Q: {last_ai_question}\nA: {last_human_answer}")

    return existing_qa


def _strip_think(text: str) -> str:
    """Remove <think>...</think> reasoning blocks that Qwen emits."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def _rephrase_with_qwen(raw_text: str) -> str:
    try:
        qwen = _get_qwen_llm()
        response = qwen.invoke([
            SystemMessage(content=(
                "You are a kind, empathetic hospital assistant. "
                "Rephrase the following clinical question into warm, conversational language. "
                "Match the language of the conversation (English, Urdu, or Roman Urdu). "
                "Do NOT mention MedGemma, AI, or any system. "
                "Ask exactly ONE question. Be concise."
            )),
            HumanMessage(content=raw_text),
        ])
        cleaned = _strip_think(str(response.content))
        return cleaned
    except Exception as e:
        print(f"⚠️  [Triage] Qwen rephrase failed ({e}) — using MedGemma output directly")
        return raw_text
def _save_triage_to_supabase(ctx: dict, clinical_summary: str, qa_pairs: list[str]) -> None:
    booking_id = ctx.get("appointment", {}).get("booking_id")
    if not booking_id:
        return
    qa_text = "\n".join(qa_pairs) if qa_pairs else "No Q&A recorded."
    notes = (
        f"=== TRIAGE NOTES ===\n{qa_text}\n\n"
        f"=== CLINICAL SUMMARY ===\n{clinical_summary}\n"
    )
    try:
        save_case_notes.invoke({"appointment_id": booking_id, "notes": notes})
        print(f"💾 [Triage] Notes saved for booking_id={booking_id}")
    except Exception as e:
        print(f"❌ [Triage] Failed to save notes: {e}")


# ── Main node ─────────────────────────────────────────────────────────────────

def triage_node(state: dict) -> dict:
    print("\n" + "=" * 54)
    print("⚕️  [Triage] Entering triage_node")

    symptom         = state.get("extracted_symptom", "")
    ctx             = state.get("booking_context", {})
    profile         = state.get("patient_profile") or {}
    triage_qa       = list(state.get("triage_qa", []))
    questions_asked = _get_questions_asked(state)

    print(f"   symptom='{symptom}'  questions_asked={questions_asked}/{MAX_TRIAGE_QUESTIONS}")

    # ── Accumulate patient responses each turn ────────────────────
    messages = list(state.get("messages", []))
    last_human = next(
        (str(m.content) for m in reversed(messages) if m.type == "human"), ""
    )
    if last_human:
        accumulated = ctx.get("accumulated_symptoms", [])
        accumulated.append(last_human)
        ctx["accumulated_symptoms"] = accumulated
        print(f"   accumulated_symptoms count={len(accumulated)}")

    # ── Symptom lookup — TURN 1 ONLY (context/hypothesis, no specialist yet) ──
    # On subsequent turns MedGemma uses its conversation history.
    # Specialist is only determined at _complete_triage after full picture is known.
    symptom_context_block = ctx.get("symptom_context_block", "")
    if symptom and not symptom_context_block:
        tokens = [s.strip() for s in symptom.replace(",", " ").split() if s.strip()]
        try:
            match = lookup(tokens)
            symptom_context_block = format_for_prompt(match)
            ctx["symptom_context_block"] = symptom_context_block   # cache — only run once
            print(f"🔍 [Triage] Turn-1 lookup → candidates: {[m.disease for m in match.top_matches]}")
            print(f"   (specialist NOT set yet — waiting for full triage)")
        except Exception as e:
            print(f"⚠️  [Triage] symptom_lookup failed: {e}")
    else:
        print("⚠️  [Triage] No symptom in state — supervisor may have missed [SYMPTOM_LOGGED]")

    # ── Hard exit at question limit ────────────────────────────────
    if questions_asked >= MAX_TRIAGE_QUESTIONS:
        print(f"🔔 [Triage] Reached {MAX_TRIAGE_QUESTIONS}-question limit — completing")
        summary = (
            f"Complaint: {symptom}\n"
            f"Suggested specialist: {profile.get('doctor_specialization', 'General Physician')}"
        )
        return _complete_triage(state, summary, triage_qa, profile, ctx)

    # ── Record the most recent Q&A pair ───────────────────────────
    triage_qa = _record_qa_pair(state, triage_qa)

    # ── Build MedGemma system prompt ──────────────────────────────
    base_prompt    = _load_triage_prompt()
    past_history   = profile.get("past_history", "Not provided")
    doctor_name    = (
        profile.get("booked_doctor")
        or ctx.get("selected_doctor", {}).get("name", "Unknown")
    )
    specialization = (
        profile.get("doctor_specialization")
        or ctx.get("selected_doctor", {}).get("specialization", "General Physician")
    )
    questions_left = MAX_TRIAGE_QUESTIONS - questions_asked

    sys_prompt_text = (
        f"{base_prompt}\n\n"
        f"PATIENT CONTEXT:\n"
        f"  Complaint : {symptom or 'Not specified'}\n"
        f"  History   : {past_history}\n"
        f"  Doctor    : Dr. {doctor_name} ({specialization})\n\n"
        f"{symptom_context_block}\n"
        f"IMPORTANT: {questions_left} question(s) left. "
        f"Ask ONE focused clinical question, or output [TRIAGE_COMPLETE]."
    )

    # ── STEP 1: MedGemma — clinical reasoning ─────────────────────
    try:
        med_llm  = _get_med_llm()
        msgs     = _build_triage_messages(state, sys_prompt_text)
        response = med_llm.invoke(msgs)
        raw_text = str(response.content).strip()
        print(f"🧠 [MedGemma] → {raw_text[:150]}")
    except Exception as e:
        print(f"❌ [Triage] MedGemma failed: {e}")
        summary = f"Complaint: {symptom}. MedGemma unavailable."
        return _complete_triage(state, summary, triage_qa, profile, ctx)

    # ── Check for triage completion ────────────────────────────────
    if "[TRIAGE_COMPLETE]" in raw_text:
        return _complete_triage(
            state,
            _extract_clinical_summary(raw_text),
            triage_qa,
            profile,
            ctx,
        )

    # ── STEP 2: Qwen — patient-facing rephrasing ───────────────────
    # Direct call inside the same node. No HumanMessage disguise.
    # No supervisor routing required. History stays clean.
    polished = _rephrase_with_qwen(raw_text)
    print(f"💬 [Qwen→Patient] → {polished[:150]}")

    # Increment question counter
    ctx["triage_questions_asked"] = questions_asked + 1

    return {
        "messages":        [AIMessage(content=polished)],
        "triage_active":   True,
        "patient_profile": profile,
        "triage_qa":       triage_qa,
        "booking_context": ctx,
    }


# ── Completion ────────────────────────────────────────────────────────────────

def _complete_triage(
    state: dict,
    clinical_summary: str,
    qa_pairs: list[str],
    profile: dict,
    ctx: dict,
) -> dict:
    print("✅ [Triage] COMPLETE — running final specialist lookup")

    # ── FINAL LOOKUP: use ALL accumulated symptoms for accurate routing ──────
    # This is the definitive specialist decision — not the turn-1 hypothesis.
    initial     = ctx.get("prime_complaint", "")
    all_answers = ctx.get("accumulated_symptoms", [])
    full_text   = initial + " " + " ".join(all_answers)
    all_tokens  = [s.strip() for s in full_text.replace(",", " ").split() if s.strip()]

    specialist = ctx.get("recommended_specialist") or "General Physician"  # fallback

    if all_tokens:
        try:
            final_match = lookup(all_tokens)
            specialist  = final_match.suggested_specialist
            ctx["recommended_specialist"]  = specialist
            ctx["final_symptom_match"]     = format_for_prompt(final_match)
            profile["doctor_specialization"] = specialist
            print(f"🎯 [Triage] Final specialist after full triage: {specialist}")
            print(f"   Top matches: {[(m.disease, m.score) for m in final_match.top_matches]}")
        except Exception as e:
            print(f"⚠️  [Triage] Final lookup failed: {e} — keeping '{specialist}'")

    _save_triage_to_supabase(ctx, clinical_summary, qa_pairs)

    ctx["triage_completed"]       = True
    ctx["triage_questions_asked"] = ctx.get("triage_questions_asked", 0)
    ctx["triage_qa"]              = qa_pairs   # persist Q&A in JSON sidecar

    # Qwen generates the handoff — respects user's language, mentions correct specialist
    handoff = _rephrase_with_qwen(
        f"Triage is now complete. Inform the patient warmly that we have gathered "
        f"all the information we need and based on their symptoms we recommend "
        f"seeing a {specialist}. Ask if they would like to proceed with booking "
        f"an appointment with a {specialist}."
    )

    return {
        "triage_active":   False,
        "messages":        [AIMessage(content=handoff)],
        "patient_profile": profile,
        "triage_qa":       qa_pairs,
        "booking_context": ctx,
    }