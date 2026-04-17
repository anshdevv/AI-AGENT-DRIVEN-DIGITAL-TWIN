# agents/triage_agent.py
import spacy
from typing import Any, Dict
from langchain_core.messages import SystemMessage, AIMessage

from agents.llm_config import get_llm
from rag.engine import MedicalRAG

# Load NER model
try:
    nlp = spacy.load("en_core_sci_sm")
except OSError:
    import spacy.cli
    spacy.cli.download("en_core_sci_sm")
    nlp = spacy.load("en_core_sci_sm")

# Initialize RAG Engine globally so it only loads the CSV once
rag_db = MedicalRAG()

def extract_medical_entities(text: str) -> list[str]:
    """Uses SciSpaCy to extract medical terms from raw text."""
    doc = nlp(text)
    return list(set(ent.text.lower() for ent in doc.ents))

def execute_triage_step(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Executes a single turn of the triage conversation.
    Takes the LangGraph ConversationState and returns state updates.
    """
    print("⚕️ [Medical Triage] Analyzing input and fetching DB context...")
    
    profile = state.get("patient_profile", {})
    if not profile:
        profile = {
            "past_history": "No known history provided yet.",
            "booked_doctor": "Unknown",
            "doctor_specialization": "General Physician"
        } 
        
    symptom = state.get("extracted_symptom", "Unknown complaint")
    messages = state.get("messages", [])
    
    new_qa = []
    latest_user_text = ""
    
    # 1. Update QA History & Grab latest text for NER
    if len(messages) >= 2 and messages[-1].type == "human" and messages[-2].type == "ai":
        last_q = str(messages[-2].content)
        last_a = str(messages[-1].content)
        latest_user_text = last_a
        
        if "[START_TRIAGE]" not in last_q:
            qa_pair = f"Q: {last_q}\nA: {last_a}"
            new_qa.append(qa_pair)
            print(f"📝 [Saved Triage Q&A]: {qa_pair}")

    # 2. NER Extraction
    extracted_entities = extract_medical_entities(latest_user_text) if latest_user_text else []
    if extracted_entities:
        print(f"🔍 [NER Detected]: {extracted_entities}")

    # 3. RAG Lookup
    search_terms = [symptom] + extracted_entities
    rag_context = rag_db.retrieve(search_terms)

    # 4. LLM Evaluation
# In agents/triage_agent.py (inside execute_triage_step)
    
    # 4. LLM Evaluation
    llm = get_llm(
        temperature=0.1, 
        custom_model="hjogidasani/medical-triage-llama-3.1-8b"
    )
    
    sys_prompt = SystemMessage(content=f"""
    You are a clinical triage assistant.
    
    APPOINTMENT CONTEXT:
    - Specialty: {profile.get('doctor_specialization')}
    
    PATIENT DATA:
    - Chief Complaint: {symptom}
    - Detected Entities: {', '.join(extracted_entities) if extracted_entities else 'None'}
    - Past Medical History: {profile.get('past_history')}
    
    MEDICAL KNOWLEDGE (RAG CONTEXT):
    {rag_context}
    
    RULES:
    1. Based on the RAG context and symptoms, ask ONE logical follow-up question to determine severity or duration.
    2. Do NOT ask more than 3 questions total across the conversation.
    3. Once you have enough information to form a summary, do NOT say goodbye. Output EXACTLY: [TRIAGE_COMPLETE]
    """)
    
    response = llm.invoke([sys_prompt] + messages[-2:])
    response_text = str(response.content)
    
    # 5. Stop Logic check
    if "[TRIAGE_COMPLETE]" in response_text:
        print("✅ [Triage Done] Handing back to Supervisor...")
        transition_msg = AIMessage(content="I have noted all your symptoms for the doctor. Is there anything else I can help you with today?")
        return {
            "triage_active": False,
            "messages": [transition_msg],
            "patient_profile": profile,
            "triage_qa": new_qa
        }
        
    return {
        "messages": [response],
        "patient_profile": profile,
        "triage_qa": new_qa
    }