# agents/symptom_lookup.py
# ─────────────────────────────────────────────────────────────────────────────
# Loads the 4 medical CSVs once at startup.
# Exposes one function: lookup(symptoms: list[str]) → SymptomMatch
#
# No ML, no embeddings, no decision tree.
# Pure set-intersection scoring — fast, transparent, explainable.
#
# SAFETY FIX: Urgent symptom clusters (e.g. chest + sweating) always override
# the dataset score so a cardiac patient is never routed to General Physician
# just because malaria happened to tie on score.
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ── Path config — adjust if your CSVs live elsewhere ─────────────────────────
_DATA_DIR = Path(__file__).parent.parent / "data"

SYNAPSE_CSV       = _DATA_DIR / "Patient_symptoms.csv"
DISEASE_SYM_CSV   = _DATA_DIR / "DiseaseAndSymptoms.csv"
DISEASE_VEC_CSV   = _DATA_DIR / "Disease_and_symptoms_dataset.csv"
PRECAUTION_CSV    = _DATA_DIR / "Disease_precaution.csv"


# ── Specialist mapping (disease → recommended specialist) ────────────────────
DISEASE_TO_SPECIALIST: dict[str, str] = {
    "panic disorder":            "Psychiatrist",
    "anxiety":                   "Psychiatrist",
    "depression":                "Psychiatrist",
    "migraine":                  "Neurologist",
    "epilepsy":                  "Neurologist",
    "hypertension":              "Cardiologist",
    "heart attack":              "Cardiologist",
    "coronary artery disease":   "Cardiologist",
    "diabetes":                  "Endocrinologist",
    "hypothyroidism":            "Endocrinologist",
    "hyperthyroidism":           "Endocrinologist",
    "fungal infection":          "Dermatologist",
    "psoriasis":                 "Dermatologist",
    "acne":                      "Dermatologist",
    "allergy":                   "General Physician",
    "drug reaction":             "General Physician",
    "common cold":               "General Physician",
    "pneumonia":                 "Pulmonologist",
    "tuberculosis":              "Pulmonologist",
    "bronchial asthma":          "Pulmonologist",
    "malaria":                   "General Physician",
    "typhoid":                   "General Physician",
    "dengue":                    "General Physician",
    "gastroenteritis":           "Gastroenterologist",
    "peptic ulcer disease":      "Gastroenterologist",
    "gerd":                      "Gastroenterologist",
    "jaundice":                  "Gastroenterologist",
    "hepatitis":                 "Gastroenterologist",
    "urinary tract infection":   "Urologist",
    "chronic kidney disease":    "Nephrologist",
    "arthritis":                 "Orthopedic",
    "osteoarthritis":            "Orthopedic",
    "cervical spondylosis":      "Orthopedic",
    "dimorphic hemorrhoids":     "General Surgeon",
    "varicose veins":            "General Surgeon",
    "paralysis":                 "Neurologist",
    "aids":                      "Infectious Disease",
    "chicken pox":               "General Physician",
}


# ── Urgent override clusters ──────────────────────────────────────────────────
# When these token subsets are present in the patient's symptoms, the specialist
# is forced REGARDLESS of what the dataset scoring returns.
# This prevents dangerous mis-routing due to score ties.
#
# Rules:
#   - Use the smallest unambiguous set of tokens that clinically implies urgency.
#   - Order by priority (first match wins).
#
_URGENT_OVERRIDES: list[tuple[frozenset[str], str]] = [
    # Cardiac
    (frozenset({"chest", "sweating"}),          "Cardiologist"),
    (frozenset({"chest", "pain", "arm"}),        "Cardiologist"),
    (frozenset({"chest", "tightness"}),          "Cardiologist"),
    (frozenset({"chest", "pressure"}),           "Cardiologist"),
    (frozenset({"chest", "breathless"}),         "Cardiologist"),
    (frozenset({"palpitations"}),                "Cardiologist"),
    # Neurological
    (frozenset({"seizure"}),                     "Neurologist"),
    (frozenset({"paralysis"}),                   "Neurologist"),
    (frozenset({"sudden", "weakness"}),          "Neurologist"),
    (frozenset({"sudden", "numbness"}),          "Neurologist"),
    # Respiratory
    (frozenset({"breathless", "cough", "fever"}),"Pulmonologist"),
    # Abdominal
    (frozenset({"abdomen", "severe"}),           "Gastroenterologist"),
]


# ─────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DiseaseMatch:
    disease:          str
    score:            float          # 0.0 – 1.0
    matched_symptoms: list[str]
    precautions:      list[str]
    specialist:       str


@dataclass
class SymptomMatch:
    top_matches:          list[DiseaseMatch]   # up to 3
    severity:             str                  # "Mild" | "Moderate" | "Severe" | "Unknown"
    recommendation:       str                  # from SYNAPSE dataset
    suggested_specialist: str                  # top match specialist (or "General Physician")


# ─────────────────────────────────────────────────────────────────────────────
# MODULE-LEVEL DATA STORE (loaded once)
# ─────────────────────────────────────────────────────────────────────────────

# disease → set of symptom tokens
_disease_symptom_map: dict[str, set[str]] = {}

# disease → list of precaution strings
_precaution_map: dict[str, list[str]] = {}

# SYNAPSE rows: list of {"symptoms": set[str], "severity": str, "recommendation": str}
_synapse_rows: list[dict] = []

_loaded = False


def _tok(text: str) -> str:
    """Normalise a symptom string to a comparable token."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip()).strip()


def _load_disease_symptom_csv() -> None:
    """DiseaseAndSymptoms.csv  →  disease → set of symptom tokens."""
    if not DISEASE_SYM_CSV.exists():
        print(f"⚠️  [SymptomLookup] Missing {DISEASE_SYM_CSV.name} — skipping")
        return
    with open(DISEASE_SYM_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = _tok(row.get("Disease", ""))
            if not disease:
                continue
            symptoms: set[str] = set()
            for col, val in row.items():
                if col == "Disease":
                    continue
                v = _tok(val)
                if v:
                    symptoms.add(v)
            if disease in _disease_symptom_map:
                _disease_symptom_map[disease].update(symptoms)
            else:
                _disease_symptom_map[disease] = symptoms
    print(f"✅ [SymptomLookup] DiseaseAndSymptoms loaded — {len(_disease_symptom_map)} diseases")


def _load_disease_vector_csv() -> None:
    """Disease_and_symptoms_dataset.csv (binary vectors) → merge into map."""
    if not DISEASE_VEC_CSV.exists():
        print(f"⚠️  [SymptomLookup] Missing {DISEASE_VEC_CSV.name} — skipping")
        return
    with open(DISEASE_VEC_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        symptom_cols = [h for h in headers if h != "diseases"]
        for row in reader:
            disease = _tok(row.get("diseases", ""))
            if not disease:
                continue
            symptoms = {_tok(col) for col in symptom_cols if row.get(col, "0").strip() == "1"}
            if disease in _disease_symptom_map:
                _disease_symptom_map[disease].update(symptoms)
            else:
                _disease_symptom_map[disease] = symptoms
    print(f"✅ [SymptomLookup] Disease vector CSV merged — {len(_disease_symptom_map)} total diseases")


def _load_precaution_csv() -> None:
    """Disease_precaution.csv → disease → [precaution1..4]."""
    if not PRECAUTION_CSV.exists():
        print(f"⚠️  [SymptomLookup] Missing {PRECAUTION_CSV.name} — skipping")
        return
    with open(PRECAUTION_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            disease = _tok(row.get("Disease", ""))
            if not disease:
                continue
            precautions = [
                row.get(f"Precaution_{i}", "").strip()
                for i in range(1, 5)
                if row.get(f"Precaution_{i}", "").strip()
            ]
            _precaution_map[disease] = precautions
    print(f"✅ [SymptomLookup] Precautions loaded — {len(_precaution_map)} diseases")


def _load_synapse_csv() -> None:
    """SYNAPSE CSV → severity + recommendation per symptom cluster."""
    if not SYNAPSE_CSV.exists():
        print(f"⚠️  [SymptomLookup] Missing {SYNAPSE_CSV.name} — skipping")
        return
    with open(SYNAPSE_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw = row.get("Symptoms", "")
            symptoms = {_tok(s) for s in raw.split(",") if s.strip()}
            _synapse_rows.append({
                "symptoms":       symptoms,
                "severity":       row.get("Severity", "Unknown").strip(),
                "recommendation": row.get("Final Recommendation", "").strip(),
            })
    print(f"✅ [SymptomLookup] SYNAPSE loaded — {len(_synapse_rows)} rows")


def _ensure_loaded() -> None:
    global _loaded
    if _loaded:
        return
    print("🔄 [SymptomLookup] Loading medical datasets...")
    _load_disease_symptom_csv()
    _load_disease_vector_csv()
    _load_precaution_csv()
    _load_synapse_csv()
    _loaded = True
    print(f"✅ [SymptomLookup] Ready — {len(_disease_symptom_map)} diseases indexed")


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def lookup(patient_symptoms: list[str], top_n: int = 3) -> SymptomMatch:
    """
    Given a list of symptom strings from the patient, return the top_n
    matching diseases with scores, precautions, and specialist recommendation.

    Safety guarantee: urgent symptom clusters (cardiac, neuro, etc.) always
    produce the correct specialist regardless of dataset score ties.

    Usage:
        result = lookup(["chest tightness", "sweating", "nausea"])
        # result.suggested_specialist  →  "Cardiologist"  (not "General Physician")
    """
    _ensure_loaded()

    # ── Stop-word filter ─────────────────────────────────────────────────────
    # These words appear in normal speech but have no diagnostic value.
    # Keeping them inflates patient_tokens, lowers scores for real symptoms,
    # and causes random SYNAPSE rows to match via filler words.
    _STOP = frozenset({
        # Pronouns / auxiliary verbs / articles
        "i", "am", "have", "having", "had", "has", "a", "an", "the", "be",
        "been", "is", "are", "was", "were", "do", "did", "does",
        # Personal pronouns
        "my", "me", "im", "ive", "its",
        # Conjunctions / prepositions
        "some", "also", "and", "or", "but", "with", "for", "on", "at",
        "in", "it", "this", "that", "of", "to", "by", "from", "about",
        # Degree adverbs
        "very", "really", "quite", "little", "bit", "just", "so",
        # Action words that add no clinical meaning
        "feel", "feeling", "getting", "got", "keep", "experiencing",
        "suffering", "noticed", "experiencing",
        # Common short answers from triage Q&A (these pollute _complete_triage token set)
        "no", "yes", "not", "none", "nothing", "nope", "yeah",
        "okay", "ok", "sure", "please", "thank", "thanks",
    })

    # Normalise patient symptoms, stripping stop words from individual tokens
    patient_tokens: set[str] = set()
    for s in patient_symptoms:
        full_tok = _tok(s)
        if full_tok and full_tok not in _STOP:
            patient_tokens.add(full_tok)
        for word in full_tok.split():
            if word and word not in _STOP:
                patient_tokens.add(word)

    if not patient_tokens:
        # All tokens were stop words — fall back to raw (shouldn't normally happen)
        for s in patient_symptoms:
            patient_tokens.add(_tok(s))

    # ── SAFETY: Check urgent overrides BEFORE dataset scoring ────────────────
    forced_specialist: str | None = None
    for trigger_tokens, specialist in _URGENT_OVERRIDES:
        if trigger_tokens.issubset(patient_tokens):
            forced_specialist = specialist
            print(f"🚨 [SymptomLookup] Urgent override → {specialist} (trigger={trigger_tokens})")
            break

    # ── Score each disease ────────────────────────────────────────────────────
    # Score = matched / disease_symptom_count (how much of this disease's
    # profile does the patient cover?). Previously we divided by patient_tokens
    # which was polluted by stop words. Dividing by the disease's own symptom
    # count is more stable and directly measures "does this complaint fit the
    # disease profile?"
    scored: list[tuple[float, str, list[str]]] = []
    for disease, disease_symptoms in _disease_symptom_map.items():
        if not disease_symptoms:
            continue
        matched = patient_tokens & disease_symptoms
        if not matched:
            continue
        score = len(matched) / len(disease_symptoms)
        scored.append((score, disease, sorted(matched)))

    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:top_n]

    matches: list[DiseaseMatch] = []
    for score, disease, matched in top:
        specialist = DISEASE_TO_SPECIALIST.get(disease, "General Physician")
        precautions = _precaution_map.get(disease, [])
        matches.append(DiseaseMatch(
            disease=disease,
            score=round(score, 3),
            matched_symptoms=matched,
            precautions=precautions,
            specialist=specialist,
        ))

    # ── Severity from SYNAPSE ─────────────────────────────────────────────────
    # Use ROW COVERAGE RATIO: how much of a SYNAPSE row's expected symptom set
    # does this patient cover?
    #
    # Old bug: raw overlap count → "fever" matched a Severe row containing
    # {fever, chest_pain, breathlessness, confusion} with overlap=1 and won
    # because nothing scored higher. Patient got Severe for a simple fever.
    #
    # Fix: a Severe row with 5 symptoms where patient only has 1 → coverage 0.2.
    # A Moderate row with 2 symptoms where patient has 1 → coverage 0.5 → wins.
    # Minimum threshold of 0.35: if no row is ≥35% covered, return "Unknown"
    # so the question limit stays generous rather than prematurely capping.
    severity = "Unknown"
    recommendation = "Doctor Consultation"
    best_coverage = 0.0
    _MIN_COVERAGE = 0.35   # patient must match at least 35% of a SYNAPSE row

    for row in _synapse_rows:
        row_syms = row["symptoms"]
        if not row_syms:
            continue
        overlap = len(patient_tokens & row_syms)
        if not overlap:
            continue
        coverage = overlap / len(row_syms)
        if coverage > best_coverage:
            best_coverage = coverage
            severity = row["severity"]
            recommendation = row["recommendation"]

    if best_coverage < _MIN_COVERAGE:
        severity = "Unknown"   # not enough symptom coverage to assign a severity
        recommendation = "Doctor Consultation"

    # Urgent override takes priority; otherwise use dataset top match
    suggested_specialist = (
        forced_specialist
        or (matches[0].specialist if matches else "General Physician")
    )

    print(
        f"🔍 [SymptomLookup] patient_tokens={sorted(patient_tokens)} "
        f"→ top={[(m.disease, m.score) for m in matches]} "
        f"severity={severity} (best_coverage={best_coverage:.2f})  specialist={suggested_specialist}"
    )

    return SymptomMatch(
        top_matches=matches,
        severity=severity,
        recommendation=recommendation,
        suggested_specialist=suggested_specialist,
    )


def format_for_prompt(match: SymptomMatch) -> str:
    """
    Render a SymptomMatch into a compact block suitable for injecting
    into a MedGemma system prompt.
    """
    lines = ["── SYMPTOM LOOKUP CONTEXT (from medical datasets) ──"]
    lines.append(f"Severity estimate   : {match.severity}")
    lines.append(f"Recommendation      : {match.recommendation}")
    lines.append(f"Suggested specialist: {match.suggested_specialist}")
    lines.append("")
    lines.append("Top candidate conditions:")
    for i, m in enumerate(match.top_matches, 1):
        lines.append(f"  {i}. {m.disease.title()}  (score {m.score:.0%})")
        lines.append(f"     Matched symptoms : {', '.join(m.matched_symptoms)}")
        if m.precautions:
            lines.append(f"     Precautions      : {'; '.join(m.precautions)}")
    lines.append("────────────────────────────────────────────────────")
    return "\n".join(lines)