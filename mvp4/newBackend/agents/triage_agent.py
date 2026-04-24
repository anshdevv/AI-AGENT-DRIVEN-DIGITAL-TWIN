# agents/triage_agent.py
# ─────────────────────────────────────────────────────────────────────────────
# Option B: MedGemma → Qwen direct chain inside one node.
#
# FIXES & NEW FEATURES (v2):
#   1. CONTEXT FIX: triage_start_idx stored in booking_context on first call
#      so MedGemma never re-sees pre-triage noise → no more repeated questions
#   2. PATIENT HISTORY: prior appointments + notes fetched from Supabase and
#      injected into MedGemma's system prompt on turn 1
#   3. SEVERITY-BASED DYNAMIC QUESTION LIMIT:
#      Severe   → max 4 questions (escalate fast)
#      Moderate → max 6 questions
#      Mild     → max 6 questions (still want full picture)
#      Unknown  → max 8 questions (fallback)
#   4. GP-FIRST REFERRAL LOGIC in _complete_triage:
#      - Severe                    → direct specialist (skip GP)
#      - Patient explicitly named a doctor type → honor request
#      - First visit for complaint → General Physician
#      - Returning patient         → specialist
#   5. FULL MEDGEMMA CONTEXT PRINT: every turn prints exactly what MedGemma
#      receives so you can debug easily
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import json
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from langchain_ollama import ChatOllama

from agents.llm_config import get_llm
from agents.symptom_lookup import lookup, format_for_prompt
from agents.mcp_tools import save_case_notes, get_recent_case_notes

try:
    PKT = ZoneInfo("Asia/Karachi")
except ZoneInfoNotFoundError:
    PKT = timezone(timedelta(hours=5))

BOOKING_CTX_DIR = Path("booking_context")
BOOKING_CTX_DIR.mkdir(exist_ok=True)

_PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "triage_system.md"


# ── Booking context disk-persistence (mirrors orchestrator's save_booking_context)
# We cannot import orchestrator here (circular), so we do the same write inline.
# This is critical: the API handler re-loads booking_context from disk on every
# turn, so any in-memory mutation (like medgemma_raw_history) will be LOST unless
# we immediately flush it to the JSON sidecar.

def _persist_booking_context(ctx: dict) -> None:
    """Write ctx to its session JSON file so the next API turn sees the updates."""
    session_id = ctx.get("session_id")
    if not session_id or session_id == "default":
        print("⚠️  [TriagePersist] No valid session_id — skipping disk save")
        return
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", session_id)
    path = BOOKING_CTX_DIR / f"{safe}.json"
    ctx["last_updated"] = datetime.now(PKT).isoformat()
    try:
        path.write_text(json.dumps(ctx, indent=2, ensure_ascii=False), encoding="utf-8")
        history_len = len(ctx.get("medgemma_raw_history", []))
        print(f"💾 [TriagePersist] Saved ctx → {path.name}  "
              f"medgemma_raw_history={history_len} entries  "
              f"contents={[e['role'] for e in ctx.get('medgemma_raw_history', [])]}")
    except Exception as e:
        print(f"❌ [TriagePersist] Failed to save ctx: {e}")


# Dynamic limits by severity — Severe gets escalated faster
MAX_QUESTIONS_BY_SEVERITY = {
    "Severe":   4,
    "Moderate": 6,
    "Mild":     6,
    "Unknown":  8,
}
MAX_TRIAGE_QUESTIONS = 8  # absolute fallback ceiling


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

_SUMMARY_RE     = re.compile(r"CLINICAL_SUMMARY:(.*?)(?:\Z|\[)", re.DOTALL | re.IGNORECASE)
# ALL internal tags that must never reach the patient's screen or TTS
_SYSTEM_TAG_RE  = re.compile(
    r"\[MEDGEMMA_SUMMARY:[^\]]*\]"
    r"|\[SYMPTOM_LOGGED:[^\]]+\]"
    r"|\[START_TRIAGE\]"
    r"|\[END_CALL\]"
    r"|\[TRIAGE_COMPLETE\]"
    r"|CLINICAL_SUMMARY:.*?(?=\n\n|\Z)",
    re.DOTALL,
)


def _extract_clinical_summary(text: str) -> str:
    m = _SUMMARY_RE.search(text)
    if m:
        return m.group(1).strip()
    parts = text.split("[TRIAGE_COMPLETE]", 1)
    return parts[1].strip() if len(parts) > 1 else ""


def _get_questions_asked(state: dict) -> int:
    return state.get("booking_context", {}).get("triage_questions_asked", 0)


def _get_max_questions(state: dict) -> int:
    """Dynamic limit based on severity from the first symptom lookup."""
    ctx = state.get("booking_context", {})
    severity = ctx.get("triage_severity", "Unknown")
    limit = MAX_QUESTIONS_BY_SEVERITY.get(severity, MAX_TRIAGE_QUESTIONS)
    print(f"   [DynamicLimit] severity='{severity}' → max_questions={limit}")
    return limit


def _print_medgemma_context(msgs: list) -> None:
    """Pretty-print the full context being sent to MedGemma — debug only."""
    print("\n" + "╔" + "═" * 60 + "╗")
    print("║  📋 MEDGEMMA FULL CONTEXT                                  ║")
    print("╠" + "═" * 60 + "╣")
    for i, msg in enumerate(msgs):
        role = msg.__class__.__name__.replace("Message", "").upper()
        content = str(msg.content)
        print(f"║  [{i}] {role}")
        print("╟" + "─" * 60 + "╢")
        # Print full content with line wrapping at 58 chars
        for line in content.splitlines():
            while len(line) > 58:
                print(f"║  {line[:58]}")
                line = line[58:]
            print(f"║  {line}")
        print("╟" + "─" * 60 + "╢")
    print("╚" + "═" * 60 + "╝\n")


def _build_triage_messages(state: dict, sys_prompt_text: str) -> list:
    """
    Build a clean, ISOLATED history exclusively for MedGemma.

    MedGemma's context is completely separate from the main LangGraph messages
    array so it NEVER sees Qwen's rephrased/translated output.

    Turn 1  → seeds ctx["medgemma_raw_history"] with the patient's original
              complaint (all pre-triage human text)
    Turn N+ → appends the latest patient answer only when last history entry was AI
    MedGemma's raw response is appended to medgemma_raw_history by triage_node
    AFTER invoke, so the final sequence per call is:
      user: <original complaint>
      assistant: <MedGemma raw Q1>    ← NOT Qwen's version
      user: <patient answer 1>
      assistant: <MedGemma raw Q2>
      …
    """
    sys_msg  = SystemMessage(content=sys_prompt_text)
    all_msgs = list(state.get("messages", []))
    ctx      = state.get("booking_context", {})

    # Safe init — backward compat with sessions that pre-date this field
    raw_history: list[dict] = ctx.setdefault("medgemma_raw_history", [])

    # ── Locate triage start index (stored once, stable across turns) ─────────
    triage_start_idx = ctx.get("triage_start_msg_index")
    if triage_start_idx is None:
        triage_start_idx = 0
        for i, m in enumerate(all_msgs):
            if m.type == "ai" and "[START_TRIAGE]" in str(m.content):
                triage_start_idx = i + 1
                break
        if triage_start_idx == 0:
            triage_start_idx = ctx.get("triage_start_msg_index_fallback", 0)
        ctx["triage_start_msg_index"] = triage_start_idx
        print(f"   [TriageMsgs] Stored triage_start_idx={triage_start_idx} (first call)")
    else:
        print(f"   [TriageMsgs] Using stored triage_start_idx={triage_start_idx}")

    print(f"   [MedGemmaHistory] At build time: {len(raw_history)} entries, "
          f"roles={[e['role'] for e in raw_history]}")

    # ── TURN 1: seed with the original complaint ──────────────────────────────
    if not raw_history:
        pre_triage_text = " ".join(
            str(m.content) for m in all_msgs[:triage_start_idx] if m.type == "human"
        ).strip()
        if not pre_triage_text:
            pre_triage_text = state.get("extracted_symptom", "I have a medical complaint.")
        raw_history.append({"role": "user", "content": pre_triage_text})
        print(f"   [MedGemmaHistory] Turn-1 seed → complaint: '{pre_triage_text[:80]}'")

    # ── TURN N+: append latest patient answer (only when MedGemma last spoke) ─
    elif raw_history[-1]["role"] == "assistant":
        last_human = next(
            (str(m.content) for m in reversed(all_msgs) if m.type == "human"), ""
        ).strip()
        if last_human:
            # Guard: don't duplicate if already appended
            last_user_content = next(
                (e["content"] for e in reversed(raw_history) if e["role"] == "user"), ""
            )
            if last_user_content.strip() != last_human:
                raw_history.append({"role": "user", "content": last_human})
                print(f"   [MedGemmaHistory] Turn-N answer appended: '{last_human[:60]}...'")
            else:
                print(f"   [MedGemmaHistory] Answer already in history — skip duplicate")
        else:
            print(f"   [MedGemmaHistory] No human message found to append")
    else:
        print(f"   [MedGemmaHistory] Last entry is 'user' — waiting for MedGemma to respond first")

    # ── Convert to LangChain messages ─────────────────────────────────────────
    clean = []
    for item in raw_history:
        if item["role"] == "user":
            clean.append(HumanMessage(content=item["content"]))
        else:
            stripped = _SYSTEM_TAG_RE.sub("", item["content"]).strip()
            clean.append(AIMessage(content=stripped or item["content"]))

    print(f"   [TriageMsgs] {len(clean)} msgs built from MedGemma's private history "
          f"(Qwen outputs fully excluded)")
    return [sys_msg] + clean


def _record_qa_pair(state: dict, existing_qa: list[str]) -> list[str]:
    messages = list(state.get("messages", []))
    last_human_answer = ""
    last_ai_question  = ""

    for m in reversed(messages):
        if not last_human_answer and m.type == "human":
            last_human_answer = str(m.content).strip()
        elif last_human_answer and m.type == "ai":
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


def _strip_all_system_tags(text: str) -> str:
    """Strip ALL internal orchestrator/triage tags before text reaches the patient."""
    cleaned = _SYSTEM_TAG_RE.sub("", text)
    cleaned = re.sub(r"\[MEDGEMMA_SUMMARY:[^\]]*\]", "", cleaned)
    cleaned = re.sub(r"CLINICAL_SUMMARY:.*?(?=\n\n|\Z)", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _rephrase_with_qwen(raw_text: str, patient_language: str = "en") -> str:
    # Always strip system tags from input before sending to Qwen
    clean_input = _strip_all_system_tags(raw_text)
    if not clean_input:
        return ""

    is_urdu = patient_language in ("ur", "urdu")
    language_instruction = (
        "Reply in simple everyday Urdu (nastaliq script). "
        "Use natural spoken words like: aap ko, kya, kab se, kitna, theek hai, batayein. "
        "NOT formal/literary Urdu. NOT Roman Urdu. Actual Urdu script."
        if is_urdu else
        "Reply in plain conversational English."
    )

    try:
        qwen = _get_qwen_llm()
        response = qwen.invoke([
            SystemMessage(content=(
                "You are a warm, friendly hospital receptionist.\n"
                "Rephrase the given clinical question into natural easy language for the patient.\n\n"
                f"LANGUAGE: {language_instruction}\n\n"
                "RULES:\n"
                "- ONE question only. One or two short sentences max.\n"
                "- Sound like a human talking, not a form.\n"
                "- Do NOT mention AI, MedGemma, triage, system, or any technical term.\n"
                "- Do NOT output any tags like [TRIAGE_COMPLETE] or [MEDGEMMA_SUMMARY].\n"
                "- Output ONLY the rephrased question, nothing else."
            )),
            HumanMessage(content=clean_input),
        ])
        result = _strip_think(str(response.content))
        # Final safety pass — remove any tags Qwen accidentally echoed
        return _strip_all_system_tags(result)
    except Exception as e:
        print(f"⚠️  [Triage] Qwen rephrase failed ({e}) — using stripped MedGemma output")
        return clean_input


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


# ── Patient history fetcher ───────────────────────────────────────────────────

def _fetch_patient_history(ctx: dict) -> str:
    """
    Fetch prior appointment notes for this patient from Supabase.
    Returns a formatted block for injection into MedGemma's system prompt.
    Returns empty string if no history found or patient not yet identified.
    """
    patient_id = ctx.get("patient", {}).get("id")
    if not patient_id:
        return ""

    try:
        result = get_recent_case_notes.invoke({"patient_id": patient_id})
        if "No case notes" in result or not result.strip():
            return "── PATIENT HISTORY ──\nNo prior visits on record.\n────────────────────"
        return f"── PATIENT HISTORY (last 5 visits) ──\n{result}\n────────────────────"
    except Exception as e:
        print(f"⚠️  [Triage] Could not fetch patient history: {e}")
        return ""


def _detect_explicit_doctor_request(state: dict) -> str | None:
    """
    Check if the patient explicitly asked for a specific type of doctor
    anywhere in the conversation before triage started.
    Returns the specialization string if found, None otherwise.

    Examples:
      "I want to see a cardiologist"  → "Cardiologist"
      "book me with a skin doctor"    → "Dermatologist"
      "I need a general physician"    → "General Physician"
    """
    EXPLICIT_REQUEST_PATTERNS = [
        (re.compile(r"\b(cardiolog\w+)\b", re.I),         "Cardiologist"),
        (re.compile(r"\b(neurolog\w+)\b", re.I),          "Neurologist"),
        (re.compile(r"\b(dermatolog\w+|skin\s+doctor)\b", re.I), "Dermatologist"),
        (re.compile(r"\b(orthoped\w+|bone\s+doctor)\b",   re.I), "Orthopedic"),
        (re.compile(r"\b(gastroenterolog\w+)\b",          re.I), "Gastroenterologist"),
        (re.compile(r"\b(psychiatr\w+|psycholog\w+)\b",   re.I), "Psychiatrist"),
        (re.compile(r"\b(gynecolog\w+|gynaecolog\w+)\b",  re.I), "Gynecologist"),
        (re.compile(r"\b(pulmonolog\w+|lung\s+doctor)\b", re.I), "Pulmonologist"),
        (re.compile(r"\b(urolog\w+)\b",                   re.I), "Urologist"),
        (re.compile(r"\b(endocrinolog\w+)\b",              re.I), "Endocrinologist"),
        (re.compile(r"\b(general\s+physician|gp|family\s+doctor)\b", re.I), "General Physician"),
        (re.compile(r"\b(pediatr\w+|child\s+doctor)\b",  re.I), "Pediatrician"),
    ]
    messages = list(state.get("messages", []))
    # Only look at messages before triage started
    ctx = state.get("booking_context", {})
    triage_start_idx = ctx.get("triage_start_msg_index", len(messages))

    pre_triage_text = " ".join(
        str(m.content) for m in messages[:triage_start_idx] if m.type == "human"
    )

    for pattern, specialization in EXPLICIT_REQUEST_PATTERNS:
        if pattern.search(pre_triage_text):
            print(f"🎯 [Triage] Explicit doctor request detected: '{specialization}'")
            return specialization

    return None


def _check_returning_patient_for_complaint(ctx: dict, symptom: str) -> bool:
    """
    Returns True if patient has prior appointment notes mentioning the
    same complaint area — indicating they've already seen a GP for this.
    Uses simple keyword overlap between complaint and prior notes.
    """
    patient_id = ctx.get("patient", {}).get("id")
    if not patient_id:
        return False

    try:
        result = get_recent_case_notes.invoke({"patient_id": patient_id})
        if "No case notes" in result or not result.strip():
            return False

        # Simple overlap check — if symptom keywords appear in past notes
        symptom_tokens = set(re.findall(r"[a-z]+", symptom.lower()))
        notes_tokens   = set(re.findall(r"[a-z]+", result.lower()))
        # Remove noise words
        stop = {"the", "a", "an", "of", "and", "or", "is", "was", "for", "to",
                "in", "at", "with", "no", "not", "this", "that", "has", "have"}
        symptom_tokens -= stop
        overlap = symptom_tokens & notes_tokens
        overlap_ratio = len(overlap) / max(len(symptom_tokens), 1)

        returning = overlap_ratio >= 0.3   # 30%+ keyword overlap = likely same complaint
        print(f"   [ReturningCheck] symptom_tokens={symptom_tokens}  "
              f"overlap={overlap}  ratio={overlap_ratio:.2f}  returning={returning}")
        return returning
    except Exception as e:
        print(f"⚠️  [Triage] Returning patient check failed: {e}")
        return False


# ── Main node ─────────────────────────────────────────────────────────────────

def triage_node(state: dict) -> dict:
    print("\n" + "=" * 54)
    print("⚕️  [Triage] Entering triage_node")

    symptom         = state.get("extracted_symptom", "")
    ctx             = state.get("booking_context", {})
    profile         = state.get("patient_profile") or {}
    triage_qa       = list(state.get("triage_qa", []))
    questions_asked = _get_questions_asked(state)

    print(f"   symptom='{symptom}'  questions_asked={questions_asked}")

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

    # ── Symptom lookup — TURN 1 ONLY ─────────────────────────────
    symptom_context_block = ctx.get("symptom_context_block", "")
    if symptom and not symptom_context_block:
        tokens = [s.strip() for s in symptom.replace(",", " ").split() if s.strip()]
        try:
            match = lookup(tokens)
            symptom_context_block = format_for_prompt(match)
            ctx["symptom_context_block"] = symptom_context_block
            # ── Store severity for dynamic question limit ─────────
            ctx["triage_severity"] = match.severity
            print(f"🔍 [Triage] Turn-1 lookup → "
                  f"candidates: {[m.disease for m in match.top_matches]}  "
                  f"severity={match.severity}")
        except Exception as e:
            print(f"⚠️  [Triage] symptom_lookup failed: {e}")
            ctx["triage_severity"] = "Unknown"
    else:
        if not symptom:
            print("⚠️  [Triage] No symptom in state")

    # ── Patient history — TURN 1 ONLY ────────────────────────────
    patient_history_block = ctx.get("patient_history_block", "")
    if not patient_history_block:
        patient_history_block = _fetch_patient_history(ctx)
        if patient_history_block:
            ctx["patient_history_block"] = patient_history_block
            print(f"📋 [Triage] Patient history fetched:\n{patient_history_block[:200]}...")
        else:
            ctx["patient_history_block"] = ""   # mark as attempted so we don't retry

    # ── Dynamic question limit ────────────────────────────────────
    max_questions = _get_max_questions(state)

    # ── Hard exit at question limit ───────────────────────────────
    if questions_asked >= max_questions:
        print(f"🔔 [Triage] Reached {max_questions}-question limit — completing")
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
    questions_left = max_questions - questions_asked
    severity_so_far = ctx.get("triage_severity", "Unknown")

    sys_prompt_text = (
        f"{base_prompt}\n\n"
        f"PATIENT CONTEXT:\n"
        f"  Complaint        : {symptom or 'Not specified'}\n"
        f"  Medical History  : {past_history}\n"
        f"  Doctor           : Dr. {doctor_name} ({specialization})\n"
        f"  Severity (so far): {severity_so_far}\n\n"
        f"{symptom_context_block}\n"
        f"{patient_history_block}\n"
        f"IMPORTANT: {questions_left} question(s) left (severity={severity_so_far} → max={max_questions}). "
        f"Ask ONE focused clinical question, or output [TRIAGE_COMPLETE] if you have enough info."
    )

    # ── STEP 1: MedGemma — clinical reasoning ─────────────────────
    try:
        med_llm  = _get_med_llm()
        msgs     = _build_triage_messages(state, sys_prompt_text)

        # ── FULL CONTEXT PRINT ────────────────────────────────────
        _print_medgemma_context(msgs)

        response = med_llm.invoke(msgs)
        raw_text = str(response.content).strip()
        print(f"🧠 [MedGemma raw output] → {raw_text[:300]}")

        # ── Append MedGemma's raw output to its ISOLATED history ──────────────
        # MUST happen here — before Qwen rephrases — so we store the clinical
        # English text, never Qwen's translated/rephrased version.
        # We also immediately persist to disk because the API handler reloads
        # booking_context from disk on every turn, so in-memory changes are lost.
        history = ctx.setdefault("medgemma_raw_history", [])
        history.append({"role": "assistant", "content": raw_text})
        print(f"📝 [MedGemmaHistory] Appended assistant entry "
              f"(total entries now: {len(history)}) "
              f"→ roles: {[e['role'] for e in history]}")
        _persist_booking_context(ctx)

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
    patient_language = ctx.get("patient_language", "en")
    polished = _rephrase_with_qwen(raw_text, patient_language)
    print(f"💬 [Qwen→Patient] → {polished[:200]}")

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
    print("✅ [Triage] COMPLETE — running final specialist lookup + GP-first routing")

    # ── FINAL LOOKUP: use ALL accumulated symptoms ────────────────
    initial     = ctx.get("prime_complaint", "")
    all_answers = ctx.get("accumulated_symptoms", [])
    full_text   = initial + " " + " ".join(all_answers)
    all_tokens  = [s.strip() for s in full_text.replace(",", " ").split() if s.strip()]

    dataset_specialist = ctx.get("recommended_specialist") or "General Physician"
    final_severity     = ctx.get("triage_severity", "Unknown")

    if all_tokens:
        try:
            final_match        = lookup(all_tokens)
            dataset_specialist = final_match.suggested_specialist
            final_severity     = final_match.severity
            ctx["recommended_specialist"] = dataset_specialist
            ctx["final_symptom_match"]    = format_for_prompt(final_match)
            ctx["triage_severity"]        = final_severity
            profile["doctor_specialization"] = dataset_specialist
            print(f"🎯 [Triage] Dataset specialist: {dataset_specialist}  severity: {final_severity}")
            print(f"   Top matches: {[(m.disease, m.score) for m in final_match.top_matches]}")
        except Exception as e:
            print(f"⚠️  [Triage] Final lookup failed: {e} — keeping '{dataset_specialist}'")

    # ── GP-FIRST ROUTING LOGIC ─────────────────────────────────────
    #
    # Priority order:
    #   1. Severe → skip GP, go straight to specialist
    #   2. Patient explicitly requested a doctor type → honor it
    #   3. Returning patient for same complaint → specialist (they've seen GP already)
    #   4. First visit → General Physician
    #
    symptom = state.get("extracted_symptom", initial)

    explicit_request  = _detect_explicit_doctor_request(state)
    is_returning      = _check_returning_patient_for_complaint(ctx, symptom)
    is_severe         = final_severity == "Severe"

    print(f"\n   ── ROUTING DECISION ──")
    print(f"   severity       = {final_severity}  is_severe={is_severe}")
    print(f"   explicit_req   = {explicit_request}")
    print(f"   is_returning   = {is_returning}")
    print(f"   dataset_spec   = {dataset_specialist}")

    if is_severe:
        final_doctor_type = dataset_specialist
        routing_reason = f"SEVERE symptoms → direct specialist ({dataset_specialist})"
    elif explicit_request:
        final_doctor_type = explicit_request
        routing_reason = f"Patient explicitly requested → {explicit_request}"
    elif is_returning:
        final_doctor_type = dataset_specialist
        routing_reason = f"Returning patient (same complaint) → specialist ({dataset_specialist})"
    else:
        final_doctor_type = "General Physician"
        routing_reason = "First visit / non-severe → General Physician first"

    print(f"   ➤  FINAL DECISION: {final_doctor_type}  [{routing_reason}]")
    print(f"   ───────────────────────────────")

    ctx["recommended_specialist"]    = final_doctor_type
    ctx["routing_reason"]            = routing_reason
    profile["doctor_specialization"] = final_doctor_type

    _save_triage_to_supabase(ctx, clinical_summary, qa_pairs)

    ctx["triage_completed"]       = True
    ctx["triage_questions_asked"] = ctx.get("triage_questions_asked", 0)
    ctx["triage_qa"]              = qa_pairs

    # ── CRITICAL: flush to disk NOW so the next API turn sees triage_completed=True.
    # Without this, _persist_booking_context() was called earlier (when appending
    # MedGemma's raw output) BEFORE triage_completed was set, so the JSON sidecar
    # always had triage_completed=False — causing the supervisor to re-trigger
    # [START_TRIAGE] on every subsequent message.
    _persist_booking_context(ctx)

    # Build handoff message
    if final_doctor_type == "General Physician":
        handoff_prompt = (
            f"Triage is now complete. Inform the patient warmly that we have gathered "
            f"all the information we need. Based on their symptoms, we recommend starting "
            f"with a General Physician who can assess them and refer to a specialist if needed. "
            f"Ask if they would like to proceed with booking an appointment with a General Physician."
        )
    else:
        handoff_prompt = (
            f"Triage is now complete. Inform the patient warmly that we have gathered "
            f"all the information we need. Based on their symptoms ({routing_reason.lower()}), "
            f"we recommend seeing a {final_doctor_type}. "
            f"Ask if they would like to proceed with booking an appointment with a {final_doctor_type}."
        )

    patient_language = ctx.get("patient_language", "en")
    handoff = _rephrase_with_qwen(handoff_prompt, patient_language)

    return {
        "triage_active":   False,
        "messages":        [AIMessage(content=handoff)],
        "patient_profile": profile,
        "triage_qa":       qa_pairs,
        "booking_context": ctx,
    }