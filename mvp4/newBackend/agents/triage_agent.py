# agents/triage_agent.py
from langchain_core.messages import SystemMessage, AIMessage
from langchain_ollama import ChatOllama

def triage_node(state: dict):
    print("⚕️ [Medical Triage] Fetching DB context & asking clinical questions...")
    
    profile = state.get("patient_profile")
    if not profile:
        profile = {
            "past_history": "Patient has a history of mild asthma. No known allergies.",
            "booked_doctor": "Dr. Ahmed Khan",
            "doctor_specialization": "Cardiologist"
        } 
    
    symptom = state.get("extracted_symptom", "Unknown complaint")
    messages = state.get("messages", [])
    
    new_qa = []
    if len(messages) >= 2 and messages[-1].type == "human" and messages[-2].type == "ai":
        last_q = messages[-2].content
        last_a = messages[-1].content
        if "[START_TRIAGE]" not in last_q:
            qa_pair = f"Q: {last_q}\nA: {last_a}"
            new_qa.append(qa_pair)
            print(f"📝 [Saved Triage Q&A]: {qa_pair}")

    # FIXED: Using Local Ollama (MedGemma) instead of the cloud LLM
    try:
        med_llm = ChatOllama(model="medgemma:4b", temperature=0.2)
    except Exception as e:
        print("⚠️ [Ollama Error] Make sure Ollama is running!")
        # Fallback to prevent crash
        return {
            "triage_active": False,
            "messages": [AIMessage(content="I am having trouble connecting to my clinical database. Is there anything else I can help you with today?")]
        }

    sys_prompt = SystemMessage(content=f"""
    You are a clinical triage assistant. 
    
    APPOINTMENT CONTEXT:
    - Specialty: {profile.get('doctor_specialization', 'physician')} this is the doctor booked for today 
    
    PATIENT DATA FROM DATABASE:
    - Chief Complaint: {symptom}
    - Past Medical History: {profile.get('past_history')}
    
    Based on the chief complaint and history, ask ONE logical follow-up question to narrow down symptoms.
    Don't ask too many questions so the person gets annoyed. Max 2-3 questions.
    
    Once you have enough info, do NOT say goodbye. Instead, output EXACTLY: [TRIAGE_COMPLETE]
    """)
    
    try:
        response = med_llm.invoke([sys_prompt] + messages[-2:]) 
    except Exception as e:
        print(f"⚠️ [Ollama Invocation Error]: {e}")
        response = AIMessage(content="[TRIAGE_COMPLETE]")
    
    if "[TRIAGE_COMPLETE]" in str(response.content):
        print("✅ [Triage Done] Handing back to Supervisor...")
        transition_msg = AIMessage(content="I have noted all your symptoms for the doctor. Is there anything else I can help you with today, like booking another appointment?")
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