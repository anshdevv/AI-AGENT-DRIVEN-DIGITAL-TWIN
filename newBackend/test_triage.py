from langchain_core.messages import HumanMessage, AIMessage
from agents.triage_agent import execute_triage_step

state = {
    "session_id": "test-1",
    "messages": [
        AIMessage(content="How can I help you today?"),
        HumanMessage(content="I have sharp chest pain and shortness of breath."),
    ],
    "triage_active": True,
    "interaction_completed": False,
    "extracted_symptom": "chest pain",
    "patient_profile": {
        "past_history": "No known conditions",
        "booked_doctor": "Dr. Ahmed Khan",
        "doctor_specialization": "Cardiologist",
    },
    "final_diagnostic_report": "",
    "triage_qa": [],
}

result = execute_triage_step(state)
print("=== RESULT ===")
print(result)
print("=== RESPONSE TEXT ===")
print(result["messages"][0].content if result["messages"] else None)
