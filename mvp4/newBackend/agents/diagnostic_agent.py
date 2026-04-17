# agents/diagnostic_agent.py

def diagnostic_node(state: dict):
    print("📋 [Diagnostic Agent] Skipping LLM, compiling dummy report...")
    
    symptom = state.get("extracted_symptom", "Unknown")
    profile = state.get("patient_profile", {})
    triage_qa_list = state.get("triage_qa", [])
    
    formatted_qa = "\n".join([f"  {qa}" for qa in triage_qa_list]) if triage_qa_list else "  No additional questions asked."
    
    dummy_report = f"""
=========================================
      CLINICAL DIAGNOSTIC REPORT
=========================================
ATTENDING PHYSICIAN: {profile.get('booked_doctor', 'Unknown')}
SPECIALTY: {profile.get('doctor_specialization', 'General')}

PATIENT BACKGROUND:
- Past History: {profile.get('past_history', 'None on file')}
- Chief Complaint: {symptom}

TRIAGE Q&A:
{formatted_qa}

STATUS: MVP Dummy Data Generated.
=========================================
"""
    
    print(f"✅ [Report Generated]:\n{dummy_report}")
    return {"final_diagnostic_report": dummy_report}