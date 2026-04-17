from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
RAG_DIR = BASE_DIR / "rag"


@dataclass(frozen=True, slots=True)
class DomainConfig:
    name: str
    greeting: str
    fallback: str
    escalation_message: str
    faq_dir: Path
    symptom_map_path: Path
    question_flow_dir: Path


HEALTHCARE_DOMAIN = DomainConfig(
    name="healthcare",
    greeting="I can help with doctors, schedules, appointments, and a short intake.",
    fallback="I can help with doctors, schedules, appointments, and a short intake. Tell me what you need.",
    escalation_message="If you'd prefer, I can help hand this over to a human representative. If this is an emergency, please contact your local emergency services right away.",
    faq_dir=RAG_DIR / "faq",
    symptom_map_path=RAG_DIR / "mapping" / "symptoms_to_specialization.md",
    question_flow_dir=RAG_DIR / "question_flows",
)


def get_domain_config(domain_name: str) -> DomainConfig:
    if domain_name == "healthcare":
        return HEALTHCARE_DOMAIN
    return HEALTHCARE_DOMAIN
