#!/usr/bin/env python3
"""
Patient simulator for the test harness.
═══════════════════════════════════════

Two-tier design:
  1. KEYWORD ROUTER — answers fixed-fact questions (demographics, allergies,
     yes/no history items, booking confirmations) directly from the persona
     JSON. No LLM call. Fast and deterministic.
  2. LLM FALLBACK — for open-ended questions (symptom descriptions, anything
     not matched by the router), calls a small local model via Ollama with
     a SHORT fact-sheet prompt (no growing chat history, which confuses 4B
     models on long conversations).

Most history-collection turns hit tier 1. Most triage-drilldown turns hit
tier 1 too. The LLM only kicks in for the few genuinely open-ended turns
("describe the pain", "anything else you want to add", etc.).
"""
import re
import json
import requests
from pathlib import Path


# ── Ollama config — override via env if needed ─────────────────────────────
import os
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434/api/chat")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b-instruct")


# ── Helpers used by route handlers ─────────────────────────────────────────

def _yes_no_field(persona, key, yes_template, no_template):
    """Render a yes/no answer based on whether persona['history'][key] has content."""
    val = (persona.get("history", {}) or {}).get(key) or ""
    v = str(val).strip()
    if v and v.lower() not in ("none", "no", "never", "n/a", ""):
        return yes_template.format(val=v)
    return no_template


# ── Keyword router ─────────────────────────────────────────────────────────
# Each entry: (regex pattern, callable(persona) -> reply string)
# Order matters: earlier patterns win, so more specific patterns go first.

_ROUTES = [
    # ── Identity ────────────────────────────────────────────────────────────
    (r"\b(phone\s*number|share\s*your\s*phone|cell\s*number|contact\s*number)\b",
     lambda p: p.get("phone", "")),
    (r"\b(your\s*name|share\s*your\s*name|first\s*name|tell\s*me\s*your\s*name|"
     r"create\s*a\s*profile|register\s*you|what\s*should\s*i\s*call)\b",
     lambda p: p.get("name", "")),

    # ── Demographics ────────────────────────────────────────────────────────
    (r"\b(how\s*old|what'?s\s*your\s*age|your\s*age|age\s*\?)\b",
     lambda p: f"I am {p['demographics']['age']} years old"),
    (r"\b(male\s*or\s*female|your\s*gender|man\s*or\s*woman)\b",
     lambda p: p['demographics']['gender'].capitalize()),
    (r"\b(married|single|marital\s*status|are\s*you\s*married)\b",
     lambda p: p['demographics']['marital_status'].capitalize()),

    # ── History fields ──────────────────────────────────────────────────────
    (r"\b(chronic|existing\s*(health|medical)|any\s*(health\s*)?conditions?|"
     r"long-?term\s*condition|managing\s*any|conditions?\s*you\s*have|"
     r"diabetes.*high\s*blood\s*pressure|thyroid|asthma)\b",
     lambda p: _yes_no_field(p, "chronic_conditions",
                              "Yes, {val}", "No, none of those")),

    (r"\b(any\s*medications?|currently\s*taking|on\s*any\s*meds|any\s*pills|"
     r"prescribed\s*any|taking\s*any\s*medicine)\b",
     lambda p: _yes_no_field(p, "medications",
                              "Yes, I take {val}", "No, no medications")),

    (r"\b(allerg\w*\s*to\s*medications?|drug\s*allerg|penicillin|aspirin|"
     r"allergic\s*to\s*any\s*medicines?|react\w*\s*to\s*(any\s*)?medicines?|"
     r"reaction\s*to\s*medicines?|reaction\s*to\s*drugs?|"
     r"known\s*allerg\w*\s*to\s*med)",
     lambda p: _yes_no_field(p, "drug_allergies",
                              "Yes, {val}", "No drug allergies")),

    (r"\b(allergies?\s*to\s*food|environmental|pollen|animal\s*fur|"
     r"pet\s*allerg|other\s*allerg|seasonal\s*allerg|"
     r"general\s*allerg|allerg\w*\s*(to|like).*dust|"
     r"allerg\w*\s*(to|like).*food)",
     lambda p: _yes_no_field(p, "general_allergies",
                              "Yes, {val}", "No other allergies")),

    (r"\b(family\s*history|runs?\s*in\s*your\s*family|in\s*the\s*family|"
     r"family\s*member|parents?\s*have|heredit|genetic|relatives?\s*have|"
     r"anyone\s*in\s*your\s*(immediate\s*)?family)\b",
     lambda p: _yes_no_field(p, "family_history",
                              "Yes, {val}", "No family history of major illness")),

    (r"\b(smoke|smoking|tobacco|cigarette|cigar|vape|vaping|chewing\s*tobacco)\b",
     lambda p: _yes_no_field(p, "smoking_status",
                              "Yes, {val}", "No, I don't smoke")),

    # ── Female-specific ─────────────────────────────────────────────────────
    (r"\b(could\s*you\s*be\s*pregnant|are\s*you\s*pregnant|chance.*pregnant|"
     r"any\s*chance.*pregnant|currently\s*pregnant|pregnancy)\b",
     lambda p: _yes_no_field(p, "pregnancy_status",
                              "Yes, {val}", "No, not pregnant")),
    (r"\b(last\s*menstrual\s*period|last\s*period|lmp|first\s*day\s*of\s*your\s*last|"
     r"when\s*was\s*your\s*last\s*period)\b",
     lambda p: (p.get("history", {}) or {}).get("lmp_date") or "About two weeks ago"),
    (r"\b(menstrual|period|cycle|menstruation)\b",
     lambda p: (p.get("history", {}) or {}).get("menstrual_history") or "Regular, no issues"),
    (r"\b(obstetric|previous\s*pregnan|prior\s*pregnan|deliveries|delivery|"
     r"c-section|cesarean)\b",
     lambda p: (p.get("history", {}) or {}).get("obstetric_history") or "No previous pregnancies"),

    # ── Elderly-specific ────────────────────────────────────────────────────
    (r"\b(fall\w*\s*(recently|in\s*the\s*last|history)|fallen\s*recently|"
     r"trip\s*and\s*fall|balance\s*problem)\b",
     lambda p: _yes_no_field(p, "fall_history",
                              "Yes, {val}", "No, no falls")),
    (r"\b(vaccin\w*|immuniz\w*|shots\s*up\s*to\s*date)\b",
     lambda p: (p.get("history", {}) or {}).get("vaccination_status") or "Up to date"),

    # ── Complaint phase (kickoff) ───────────────────────────────────────────
    (r"\b(what\s*brings\s*you\s*in|main\s*reason|main\s*concern|"
     r"what'?s\s*the\s*problem|what'?s\s*bothering|"
     r"reason\s*for\s*your\s*visit|symptoms\s*or\s*concerns|"
     r"seems\s*to\s*be\s*the\s*problem|how\s*can\s*i\s*help\s*you\s*today)\b",
     lambda p: f"I have {p['complaint']['main']}"),

    # ── Triage drilldown ────────────────────────────────────────────────────
    (r"\b(how\s*long|since\s*when|how\s*many\s*days|for\s*how\s*long|"
     r"duration|started\s*when)\b",
     lambda p: f"About {p['complaint'].get('duration', 'a few days')}"),

    (r"\b(scale\s*(of|from)?\s*1\s*to\s*10|rate\s*(your|the)\s*pain|"
     r"out\s*of\s*10|how\s*severe|severity|how\s*bad)\b",
     lambda p: f"About {p['complaint'].get('severity', 5)} out of 10"),

    (r"\b(white\s*patches|pus|spots\s*on\s*your\s*throat|patches\s*in\s*the\s*back)\b",
     lambda p: "Yes" if p['complaint'].get('white_patches') else "No, none"),

    (r"\b(trouble\s*breathing|shortness\s*of\s*breath|hard\s*to\s*breathe|"
     r"opening\s*your\s*mouth|swollen.*throat|throat\s*feels\s*swollen)\b",
     lambda p: "Yes, a little" if p['complaint'].get('breathing_trouble') else "No, no trouble"),

    (r"\b(fever|runny\s*nose|cough|swollen\s*glands|associated\s*symptoms?|"
     r"any\s*other\s*symptoms|symptoms\s*like)\b",
     lambda p: ("Yes, " + ", ".join(p['complaint'].get('associated', []))
                if p['complaint'].get('associated')
                else "No, nothing else")),

    (r"\b(anyone\s*sick|near\s*anyone\s*sick|exposed?\s*to|"
     r"contact\s*with\s*anyone|been\s*around\s*anyone|air-?conditioned|"
     r"travel|crowd)\b",
     lambda p: p['complaint'].get('context', 'No, no exposure I know of')),

    (r"\b(only\s*when\s*you\s*swallow|when\s*you\s*swallow|hurt\s*to\s*swallow|"
     r"pain\s*all\s*the\s*time|just\s*when)\b",
     lambda p: ("Mostly when I swallow"
                if p['complaint'].get('swallowing_pain')
                else "All the time")),

    (r"\b(where\s*is\s*the\s*pain|location\s*of\s*(the\s*)?pain|"
     r"where.*hurt|side\s*of\s*your\s*head|both\s*sides|one\s*side|"
     r"front|back|temple)\b",
     lambda p: p['complaint'].get('location', "It's all over")),

    (r"\b(throb|throbbing|sharp|dull|pressure|stabbing|ache|aching|"
     r"type\s*of\s*pain|describe\s*the\s*pain|what\s*does\s*it\s*feel\s*like)\b",
     lambda p: p['complaint'].get('quality', "It feels like pressure")),

    (r"\b(sudden\s*weakness|facial\s*drooping|trouble\s*speaking|"
     r"worst.*headache|red\s*flag)\b",
     lambda p: "No, nothing like that"),

    # ── Booking phase ───────────────────────────────────────────────────────
    (r"\b(would\s*you\s*like\s*to\s*book|book\s*an\s*appointment|"
     r"shall\s*i\s*book|book\s*you\s*in|book\s*it|with\s*this\s*doctor)\b",
     lambda p: "Yes please book me"),

    (r"\b(which\s*day|what\s*day|preferred\s*day|when\s*would\s*you\s*like|"
     r"works\s*for\s*you)\b",
     lambda p: p.get('preferred_day', 'Whatever day is next available')),

    (r"\b(which\s*time|what\s*time|preferred\s*time|morning|afternoon|evening|"
     r"available\s*slots?|which\s*slot)\b",
     lambda p: p.get('preferred_slot', 'Earliest available is fine')),

    (r"\b(shall\s*i\s*go\s*ahead|confirm\s*this|all\s*set|"
     r"go\s*ahead\s*and\s*book|confirm\s*the\s*appointment|book\s*the\s*appointment)\b",
     lambda p: "Yes confirm please"),

    (r"\b(anything\s*else|further\s*assistance|help\s*you\s*with\s*anything|"
     r"any\s*other\s*questions?)\b",
     lambda p: "No, that's all, thank you"),
]


def _route_to_persona(persona: dict, bot_msg: str) -> str | None:
    """Try to answer from persona without calling the LLM. Returns None if no match."""
    if not bot_msg:
        return None
    msg = bot_msg.lower()
    for pattern, responder in _ROUTES:
        if re.search(pattern, msg, re.IGNORECASE):
            try:
                reply = responder(persona)
                if reply:
                    return reply
            except Exception:
                continue
    return None


# ── LLM fallback ───────────────────────────────────────────────────────────

def _llm_fallback(persona: dict, bot_msg: str) -> str:
    """Call the local model for open-ended replies. The fact sheet is summarised
    fresh each turn (no growing chat history) to keep the small model on rails."""
    d = persona.get("demographics", {}) or {}
    h = persona.get("history", {}) or {}
    c = persona.get("complaint", {}) or {}

    facts = [
        f"You are a {d.get('age','adult')}-year-old "
        f"{d.get('gender','person')}, {d.get('marital_status','single')}, "
        f"named {persona.get('name', 'the patient')}.",
    ]
    if c.get("main"):     facts.append(f"You came to the clinic because of: {c['main']}.")
    if c.get("duration"): facts.append(f"This has been going on for {c['duration']}.")
    if c.get("severity"): facts.append(f"Pain/discomfort severity is {c['severity']}/10.")
    if c.get("associated"):
        facts.append(f"Other symptoms you also have: {', '.join(c['associated'])}.")
    if c.get("context"):
        facts.append(f"Possibly relevant context: {c['context']}.")
    extras = [f"{k}: {v}" for k, v in h.items()
              if v and str(v).lower() not in ("none","no","never","")]
    if extras:
        facts.append("Health background: " + "; ".join(extras) + ".")

    system = (
        " ".join(facts) + "\n\n"
        "Stay in character as the patient. Reply in ONE short sentence — like "
        "a real patient texting on WhatsApp. Answer what the nurse just asked. "
        "If asked something not in your profile, give a brief 'no' or 'I'm not "
        "sure'. Do NOT role-play the nurse, do NOT explain your reasoning, do "
        "NOT volunteer extra information. Keep it natural, 1-2 sentences max."
    )

    try:
        r = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user",   "content": f"Nurse just said: \"{bot_msg}\"\nYour reply:"},
            ],
            "stream": False,
            "options": {"temperature": 0.6, "num_predict": 80, "num_ctx": 2048},
        }, timeout=60)
        r.raise_for_status()
        reply = r.json()["message"]["content"].strip()
        # Strip leading "Patient:" / quotes / "Me:"
        reply = re.sub(r"^(patient|me|i)\s*[:\-]\s*", "", reply, flags=re.IGNORECASE)
        reply = reply.strip('"').strip("'").strip()
        # Strip any role-play prefix like "*nods*"
        reply = re.sub(r"^\*[^*]+\*\s*", "", reply).strip()
        return reply or "Could you repeat that?"
    except Exception as e:
        print(f"  ⚠️  LLM fallback failed: {e}")
        return "Could you repeat that?"


# ── Public API ─────────────────────────────────────────────────────────────

def simulate_patient_reply(persona: dict, bot_msg: str) -> tuple[str, str]:
    """
    Generate a patient reply to the bot's message.
    Returns (reply_text, source) where source is 'persona' or 'llm'.
    """
    routed = _route_to_persona(persona, bot_msg)
    if routed is not None:
        return routed, "persona"
    return _llm_fallback(persona, bot_msg), "llm"


# ── Conversation termination ───────────────────────────────────────────────

_DONE_PATTERNS = [
    r"appointment\s*confirmed",
    r"\bgoodbye\b",
    r"take\s*care",
    r"have\s*a\s*great\s*day",
    r"booking\s*id\s*[:#]?\s*\d+",
    r"thank\s*you\s*for\s*(your\s*time|reaching\s*out|using)",
]

def is_done(bot_msg: str) -> bool:
    if not bot_msg:
        return False
    msg = bot_msg.lower()
    return any(re.search(p, msg) for p in _DONE_PATTERNS)


# ── Persona loading ────────────────────────────────────────────────────────

def load_persona(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_personas(directory: Path) -> list[dict]:
    paths = sorted(directory.glob("*.json"))
    return [load_persona(p) for p in paths]


# ── Self-test ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Self-test (no LLM needed) — checking the keyword router:\n")
    sample = {
        "name": "Ali Hasan",
        "phone": "03001234001",
        "demographics": {"age": 26, "gender": "male", "marital_status": "single"},
        "history": {
            "chronic_conditions": "none",
            "medications": "none",
            "drug_allergies": "none",
            "general_allergies": "dust",
            "family_history": "none",
            "smoking_status": "no",
        },
        "complaint": {
            "main": "a sore throat",
            "duration": "3 days",
            "severity": 6,
            "associated": ["mild fever", "runny nose"],
            "context": "my younger brother had flu last week",
            "swallowing_pain": True,
            "white_patches": False,
            "breathing_trouble": False,
        },
    }
    tests = [
        "Could I get your phone number?",
        "What's your name?",
        "How old are you, Ali?",
        "Are you male or female?",
        "Are you married or single?",
        "Do you have any chronic conditions like diabetes or asthma?",
        "Are you currently taking any medications?",
        "Do you have any allergies to medications?",
        "Do you have any other allergies — like to pollen, dust, or food?",
        "Does heart disease run in your family?",
        "Do you smoke?",
        "What brings you in today?",
        "How long has the sore throat been bothering you?",
        "On a scale of 1 to 10?",
        "Any white patches in your throat?",
        "Any other symptoms like fever or cough?",
        "Have you been near anyone sick?",
        "Would you like to book an appointment?",
        "What day works for you?",
        "What time works?",
        "Shall I confirm this booking?",
        "Anything else I can help you with?",
    ]
    persona_count = llm_count = 0
    for q in tests:
        reply = _route_to_persona(sample, q)
        if reply is None:
            src = "llm"
            reply = "<would call LLM>"
            llm_count += 1
        else:
            src = "persona"
            persona_count += 1
        print(f"  [{src:7}] Q: {q}\n           A: {reply}\n")

    print(f"Coverage: {persona_count}/{len(tests)} answered from persona "
          f"({100*persona_count//len(tests)}%), "
          f"{llm_count} would need the LLM.")
