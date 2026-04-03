# from langchain_google_genai import ChatGoogleGenerativeAI
from ..config import OPENROUTER_API_KEY, supabase
import os
import datetime
from langchain_openai import ChatOpenAI


# llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", api_key=GOOGLE_API_KEY)

llm =ChatOpenAI(
    model_name="tngtech/deepseek-r1t2-chimera:free",  # pick any free model
    temperature=0,  # deterministic for classification
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1"
)

class MedicalTriage:
    def __call__(self, state):
        user_input = state.get("user_input", "")
        
        # 1. DYNAMIC SYMPTOM: Fallback to 'general' if unknown, not 'cough'
        symptom = state.get("triage_symptom", "general")
        
        # 2. STATE RETRIEVAL: These come from bk_apt.py (Booking Node)
        patient_data = state.get("patient_data", {})
        patient_name = patient_data.get("Name", "Unknown")
        patient_id = state.get("patient_id") # If None, we can't check DB history
        
        # Current Q&A for *this* session
        current_session_history = state.get("medical_info", [])

        # --- 3. FETCH PAST DATABASE HISTORY ---
        db_history_text = "No previous medical records found."
        if patient_id:
            try:
                # Query Supabase: Get notes from previous appointments
                # filtering out empty notes
                res = (supabase.table("appointments")
                       .select("appointment_date, notes, Doctor:Doctors(Name, Specialization)")
                       .eq("patient_id", patient_id)
                       .neq("notes", "null") # filtered where notes exist
                       .order("appointment_date", desc=True)
                       .limit(5) # context window limit
                       .execute())
                
                if res.data:
                    lines = []
                    for appt in res.data:
                        date = appt.get("appointment_date", "Unknown Date")
                        doc = appt.get("Doctor", {}).get("Name", "Unknown Doc")
                        notes = appt.get("notes", "No notes")
                        if notes:
                            lines.append(f"- [{date}] Saw Dr. {doc}: {notes}")
                    
                    if lines:
                        db_history_text = "\n".join(lines)
            except Exception as e:
                print(f"Error fetching DB history: {e}")

        # --- 4. LOAD RAG PROTOCOL ---
        # Tries specific symptom file, falls back to general.md
        path = f"backend/rag/question_flows/{symptom}.md"
        if not os.path.exists(path):
            path = "backend/rag/triage/general.md"
            # If even general.md is missing, stop.
            if not os.path.exists(path):
                state["triage_active"] = False
                state["response"] = "Appointment booked. See you then!"
                return state

        with open(path, "r") as f:
            protocol = f.read()

        # --- 5. UPDATE CURRENT SESSION HISTORY ---
        # Add User's answer to the list (if it's not the start)
        if user_input and current_session_history:
            current_session_history.append(f"Patient: {user_input}")
        specific_complaint = state.get("patient_complaint", "Not stated")
        symptom_category = state.get("triage_symptom", "general")
        # --- 6. INTELLIGENT PROMPT ---
        prompt = f"""
You are a smart triage nurse at a hospital.

### PATIENT CONTEXT
Name: {patient_name}
**Presenting Complaint:** "{specific_complaint}"
Symptom Category: {symptom}

### PAST MEDICAL HISTORY (From Database)
"{db_history_text}"

### TRIAGE PROTOCOL (Questions to ask)
{protocol}

### CURRENT CONVERSATION
{current_session_history}

### INSTRUCTIONS

1. **Analyze Past History:** If the patient's past history answers a protocol question (e.g., "Do you smoke?" might be in past notes), DO NOT ask it again. Assume the answer is the same unless it's time-sensitive (like "do you have fever now").
2.**Acknowledge the Complaint:** The patient has specifically stated: "{specific_complaint}". 
   - If the Protocol asks "Do you have X?", and the complaint confirms X, SKIP that question.
   - Example: If complaint is "stomach pain", do NOT ask "Do you have stomach pain?". Ask "How long have you had it?".
3. **Follow Protocol:** Ask the NEXT unanswered question from the Protocol.
4. **Be Adaptive:** If the user says something alarming (Red Flag), stop and output "DONE".
5. **Finish:** When all necessary info is gathered, output "DONE".

Output ONLY the next question or "DONE".
"""
        response = llm.invoke(prompt).content.strip()

        # --- 7. HANDLE RESPONSE ---
        if "DONE" in response:
            state["triage_active"] = False
            state["response"] = "Thank you. I have updated your file with your new symptoms. The doctor will see you shortly."
            
            # SAVE the new notes to the current appointment in DB
            save_notes_to_db(state.get("appointment_id"), current_session_history)
            
        else:
            state["response"] = response
            # Save the Nurse's question to history so we have context next turn
            current_session_history.append(f"Nurse: {response}")
            state["medical_info"] = current_session_history

        return state

def save_notes_to_db(appt_id, history_list):
    """Updates the 'notes' column in the appointments table"""
    if not appt_id or not history_list:
        return
    
    # Combine list into a readable block
    # e.g. "Nurse: Fever? \n Patient: Yes"
    notes_blob = "\n".join(history_list)
    
    try:
        supabase.table("appointments").update({"notes": notes_blob}).eq("id", appt_id).execute()
        print(f"Saved notes for Appt {appt_id}")
    except Exception as e:
        print(f"DB Save Error: {e}")