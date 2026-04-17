# agents/orchestrator.py
from typing import Annotated, Any, TypedDict, Sequence
import operator

# LangChain & LangGraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode

# Local imports
from agents.llm_config import get_llm
# TODO: We will uncomment this in the next step when we build mcp_tools.py
# from agents.mcp_tools import recommend_specialist_tool, check_availability_tool, book_appointment_tool

# ------------------------------------------------------------------------
# 1. THE STATE
# ------------------------------------------------------------------------
class ConversationState(TypedDict):
    session_id: str
    messages: Annotated[Sequence[BaseMessage], operator.add] 
    
    # Phase Tracking
    triage_active: bool
    interaction_completed: bool
    
    # Clean Data Passing
    extracted_symptom: str
    patient_profile: dict[str, Any]
    final_diagnostic_report: str
    triage_qa: Annotated[list[str], operator.add]

# ------------------------------------------------------------------------
# 2. THE SUPERVISOR NODE
# ------------------------------------------------------------------------
def supervisor_node(state: ConversationState):
    print("🧠 [Supervisor] Analyzing...")
    messages = state.get("messages", [])
    
    # tools = [recommend_specialist_tool, check_availability_tool, book_appointment_tool]
    tools = [] 
    
    llm = get_llm(temperature=0.1)
    llm_with_tools = llm.bind_tools(tools) if tools else llm
    
    sys_prompt = SystemMessage(content="""
    You are a bilingual (English, Urdu, Roman Urdu) hospital concierge.
    Your tone must always be incredibly kind, patient, and empathetic. 

    YOUR JOBS:
    1. Help patients book appointments. Guide them patiently step-by-step.
    2. RECOMMENDATIONS: If a patient mentions a problem but hasn't named a doctor, ALWAYS use the `recommend_specialist_tool`.
    3. IMPORTANT: If the user mentions ANY medical symptom, do NOT attempt to diagnose. Acknowledge their discomfort, continue booking, and output exactly: 
       [SYMPTOM_LOGGED: <translate_their_symptom_to_english_here>]
    4. Once the booking is completely finished AND a symptom was logged, output: [START_TRIAGE]
    5. POST-TRIAGE: If the user has finished triage and says they don't need anything else (e.g., "no thanks", "goodbye", "Allah hafiz"), you MUST output exactly: [END_CALL]
    """)
    
    response = llm_with_tools.invoke([sys_prompt] + messages)
    response_text = str(response.content)
    
    if "[SYMPTOM_LOGGED:" in response_text:
        symptom_start = response_text.find("[SYMPTOM_LOGGED:") + 16
        symptom_end = response_text.find("]", symptom_start)
        state["extracted_symptom"] = response_text[symptom_start:symptom_end].strip()
        print(f"📝 [Extracted Symptom Saved]: {state['extracted_symptom']}")
    
    if "[START_TRIAGE]" in response_text:
        state["triage_active"] = True

    if "[END_CALL]" in response_text:
        state["interaction_completed"] = True
        
    return {
        "messages": [response], 
        "extracted_symptom": state.get("extracted_symptom"), 
        "triage_active": state.get("triage_active"),
        "interaction_completed": state.get("interaction_completed")
    }

# ------------------------------------------------------------------------
# 3. THE TRIAGE NODE (MedGemma)
# ------------------------------------------------------------------------
def triage_node(state: ConversationState):
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

    llm = get_llm(temperature=0.2)
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
    
    response = llm.invoke([sys_prompt] + messages[-2:]) 
    
    if "[TRIAGE_COMPLETE]" in str(response.content):
        print("✅ [Triage Done] Handing back to Supervisor...")
        # Unlock Triage, and have the AI seamlessly ask the transition question
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

# ------------------------------------------------------------------------
# 4. THE DIAGNOSTIC NODE
# ------------------------------------------------------------------------
def diagnostic_node(state: ConversationState):
    print("📋 [Diagnostic Agent] Compiling final physician report...")
    
    symptom = state.get("extracted_symptom", "Unknown")
    profile = state.get("patient_profile", {})
    
    triage_qa_list = state.get("triage_qa", [])
    formatted_qa = "\n\n".join(triage_qa_list) if triage_qa_list else "No additional triage questions were asked."
    
    llm = get_llm(temperature=0.0)
    sys_prompt = SystemMessage(content=f"""
    You are an expert medical summarizer. The triage call has just ended.
    
    APPOINTMENT CONTEXT (ATTENDING PHYSICIAN):
    - Doctor: {profile.get('booked_doctor', 'Unknown Doctor')}
    - Specialty: {profile.get('doctor_specialization', 'General')}
    
    PATIENT BACKGROUND:
    - Past History: {profile.get('past_history')}
    - Initial Complaint: {symptom}
    
    SPECIFIC Q&A GATHERED DURING TRIAGE:
    {formatted_qa}
    
    Review the gathered information and draft a concise, professional clinical diagnostic report.
    FORMAT:
    - Attending Physician & Specialty
    - Chief Complaint
    - Gathered Symptoms (Summarized from Q&A)
    - Red Flags / Urgency Level
    """)
    
    response = llm.invoke([sys_prompt])
    report = response.content
    print(f"✅ [Report Generated]:\n{report}")
    
    return {"final_diagnostic_report": report}

# ------------------------------------------------------------------------
# 5. EDGE ROUTING LOGIC
# ------------------------------------------------------------------------
def entry_router(state: ConversationState):
    """Decides who handles the user's incoming message."""
    if state.get("triage_active"):
        return "triage_node"
    return "supervisor_node"

def supervisor_router(state: ConversationState):
    """Routes the flow after the Supervisor speaks."""
    messages = state.get("messages", [])
    if not messages:
        return END
    last_message = messages[-1]
    
    # If Supervisor flipped the END switch, go generate the report
    if state.get("interaction_completed"):
        return "diagnostic_node"
        
    # If Supervisor called a tool, run the tool
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
        
    # If Supervisor flipped the triage switch, go to MedGemma
    if state.get("triage_active"):
        return "triage_node"
        
    return END

def triage_router(state: ConversationState):
    """Routes the flow after MedGemma (Triage) speaks."""
    # If MedGemma turned triage OFF, route back to Supervisor for the "Anything else?" loop
    if not state.get("triage_active"):
        return "supervisor_node"
        
    # Otherwise, wait for the user to answer MedGemma's question
    return END

# ------------------------------------------------------------------------
# 6. BUILD THE GRAPH
# ------------------------------------------------------------------------
builder = StateGraph(ConversationState)

builder.add_node("supervisor_node", supervisor_node)
builder.add_node("tools", ToolNode([])) 
builder.add_node("triage_node", triage_node)
builder.add_node("diagnostic_node", diagnostic_node) 

# The user's message enters via the Entry Router (skips Supervisor if in Triage)
builder.add_conditional_edges(START, entry_router)

builder.add_conditional_edges("supervisor_node", supervisor_router)
builder.add_edge("tools", "supervisor_node") 
builder.add_conditional_edges("triage_node", triage_router)
builder.add_edge("diagnostic_node", END)

memory = MemorySaver()
orchestrator_graph = builder.compile(checkpointer=memory)