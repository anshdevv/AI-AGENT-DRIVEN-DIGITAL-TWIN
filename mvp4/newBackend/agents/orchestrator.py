# agents/orchestrator.py
from typing import Annotated, Any, TypedDict, Sequence
import operator
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from datetime import datetime, timedelta, timezone
from agents.llm_config import get_llm
from agents.mcp_tools import ALL_TOOLS
from agents.triage_agent import triage_node
from agents.diagnostic_agent import diagnostic_node
try:
    PKT = ZoneInfo("Asia/Karachi")
except ZoneInfoNotFoundError:  # pragma: no cover - Windows installs may miss tzdata
    PKT = timezone(timedelta(hours=5))

# ------------------------------------------------------------------------
# 1. THE STATE
# ------------------------------------------------------------------------
class ConversationState(TypedDict, total=False):
    session_id: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    triage_active: bool
    interaction_completed: bool
    extracted_symptom: str
    patient_profile: dict[str, Any]
    final_diagnostic_report: str
    triage_qa: Annotated[list[str], operator.add]

# ------------------------------------------------------------------------
# 2. LLM CACHE — initialized ONCE at startup, never inside a node
# ------------------------------------------------------------------------
_llm_with_tools = None

def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        print("🔧 [LLM] Initializing LLM + binding tools (startup only)...")
        _llm_with_tools = get_llm(temperature=0.1).bind_tools(ALL_TOOLS)
        print(f"✅ [LLM] Ready. Tools bound: {[t.name for t in ALL_TOOLS]}")
    return _llm_with_tools

# ------------------------------------------------------------------------
# 3. HELPERS
# ------------------------------------------------------------------------
MAX_HISTORY_MESSAGES = 10

def _trim_messages(messages: list) -> list:
    if len(messages) > MAX_HISTORY_MESSAGES:
        trimmed = messages[-MAX_HISTORY_MESSAGES:]
        print(f"✂️  [History] Trimmed {len(messages)} → {len(trimmed)} messages")
        return trimmed
    return messages

def _invoke_with_retry(llm, prompt_messages: list, retries: int = 2, delay: float = 3.0):
    last_exc = None
    for attempt in range(1, retries + 2):
        try:
            if attempt > 1:
                print(f"🔄 [LLM] Retry attempt {attempt}...")
                time.sleep(delay)
            return llm.invoke(prompt_messages)
        except Exception as e:
            err = str(e)
            if any(kw in err for kw in ["ReadError", "10054", "ConnectionError", "RemoteDisconnected", "forcibly closed"]):
                print(f"⚠️  [LLM] Connection dropped (attempt {attempt}): {err[:120]}")
                last_exc = e
            else:
                raise
    raise last_exc

def _print_messages(messages: list, label: str = "Messages"):
    """Print full message array for debugging."""
    print(f"\n📋 [{label}] {len(messages)} message(s):")
    for i, msg in enumerate(messages):
        role = msg.__class__.__name__.replace("Message", "")
        content = str(msg.content)[:150].replace("\n", " ")
        tool_info = ""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_info = f" | tools={[tc['name'] for tc in msg.tool_calls]}"
        print(f"   [{i}] {role}{tool_info}: {content}")

# ------------------------------------------------------------------------
# 4. THE SUPERVISOR NODE
# ------------------------------------------------------------------------
def supervisor_node(state: ConversationState):
    # ✅ Compute real dates at call time so LLM always gets today's actual date
    now          = datetime.now(PKT)
    today_str    = now.strftime("%A, %Y-%m-%d")
    tomorrow_str = (now + timedelta(days=1)).strftime("%A, %Y-%m-%d")

    messages = _trim_messages(list(state.get("messages", [])))

    print("\n" + "="*50)
    print("🧠 [Supervisor] Entered")
    print(f"   today                : {today_str}")
    print(f"   triage_active        : {state.get('triage_active')}")
    print(f"   interaction_completed: {state.get('interaction_completed')}")
    print(f"   extracted_symptom    : {state.get('extracted_symptom')}")
    print(f"   message count        : {len(messages)}")
    print("="*50)
    _print_messages(messages, label="Supervisor input")

    llm_with_tools = _get_llm_with_tools()

    sys_prompt = SystemMessage(content=f"""
You are a bilingual (English, Urdu, Roman Urdu) hospital concierge AI.
Your tone must always be incredibly kind, patient, and empathetic.

DATE CONTEXT (Pakistan Standard Time):
- Today    : {today_str}
- Tomorrow : {tomorrow_str}
- NEVER guess or invent dates. Only use dates from this context or from tool results.
- When the user says "tomorrow", use {tomorrow_str} exactly. Do NOT ask them to confirm it.
- Never display a specific calendar date unless it came from a tool result.

YOUR JOBS:
1. Help patients book appointments. Guide them patiently step-by-step.
2. RECOMMENDATIONS: If a patient mentions a symptom but hasn't named a doctor,
   use `recommend_specialist_tool` to suggest the right specialist.
3. SYMPTOM LOGGING: If the user mentions ANY medical symptom, acknowledge their
   discomfort, continue helping with booking, and output exactly:
   [SYMPTOM_LOGGED: <english_translation_of_symptom>]
4. START TRIAGE: Once the booking is FULLY complete AND a symptom was logged,
   output exactly: [START_TRIAGE]
5. END CALL: After triage, if the user says goodbye or no more help needed
   (e.g., "no thanks", "Allah hafiz", "bye"), output exactly: [END_CALL]

TOOL RULES — VERY IMPORTANT:
- Do NOT call `find_provider_availability` or `get_doctor_schedule` unless the
  user has explicitly given you BOTH a doctor name AND a date. Ask first.
- Do NOT call `create_booking` unless you have patient_id, doctor_id, date,
  AND time all confirmed by the user. Collect them one at a time if needed.
- Do NOT call `lookup_customer_profile` unless the user has given their name
  or phone number.
- When in doubt, ask the user for missing info instead of calling a tool.
""")

    print("📡 [Supervisor] Invoking LLM...")
    try:
        response = _invoke_with_retry(llm_with_tools, [sys_prompt] + messages)
    except Exception as e:
        print(f"❌ [Supervisor] LLM failed after retries: {e}")
        fallback = AIMessage(content="I'm sorry, I'm having trouble connecting right now. Could you please repeat that?")
        return {
            "messages": [fallback],
            "triage_active": state.get("triage_active") or False,
            "interaction_completed": state.get("interaction_completed") or False,
            "extracted_symptom": state.get("extracted_symptom"),
        }

    response_text = str(response.content)
    print(f"💬 [Supervisor] Response: '{response_text[:300]}{'...' if len(response_text) > 300 else ''}'")

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🛠️  [Supervisor] Tool calls: {[tc['name'] for tc in response.tool_calls]}")

    extracted_symptom     = state.get("extracted_symptom")
    triage_active         = state.get("triage_active") or False
    interaction_completed = state.get("interaction_completed") or False

    if "[SYMPTOM_LOGGED:" in response_text:
        start = response_text.find("[SYMPTOM_LOGGED:") + 16
        end   = response_text.find("]", start)
        extracted_symptom = response_text[start:end].strip()
        print(f"📝 [Supervisor] Symptom logged: '{extracted_symptom}'")

    if "[START_TRIAGE]" in response_text:
        triage_active = True
        print("🚦 [Supervisor] → triage_active = True")

    if "[END_CALL]" in response_text:
        interaction_completed = True
        print("🏁 [Supervisor] → interaction_completed = True")

    print(f"📤 [Supervisor] Out: triage_active={triage_active}, interaction_completed={interaction_completed}")

    return {
        "messages": [response],
        "extracted_symptom": extracted_symptom,
        "triage_active": triage_active,
        "interaction_completed": interaction_completed,
    }

# ------------------------------------------------------------------------
# 5. ROUTING LOGIC
# ------------------------------------------------------------------------
def entry_router(state: ConversationState):
    triage_active = state.get("triage_active") or False
    dest = "triage_node" if triage_active else "supervisor_node"
    print(f"🔀 [Entry Router] triage_active={triage_active} → '{dest}'")
    return dest

def supervisor_router(state: ConversationState):
    messages = state.get("messages", [])
    if not messages:
        print("🔀 [Supervisor Router] No messages → END")
        return END

    last_message = messages[-1]

    if state.get("interaction_completed"):
        print("🔀 [Supervisor Router] interaction_completed → diagnostic_node")
        return "diagnostic_node"

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"🔀 [Supervisor Router] Tool calls → tools")
        return "tools"

    if state.get("triage_active"):
        print("🔀 [Supervisor Router] triage_active → triage_node")
        return "triage_node"

    print("🔀 [Supervisor Router] Normal reply → END")
    return END

def triage_router(state: ConversationState):
    if not (state.get("triage_active") or False):
        print("🔀 [Triage Router] Triage done → supervisor_node")
        return "supervisor_node"
    print("🔀 [Triage Router] Awaiting user → END")
    return END

# ------------------------------------------------------------------------
# 6. BUILD THE GRAPH
# ------------------------------------------------------------------------
builder = StateGraph(ConversationState)

builder.add_node("supervisor_node", supervisor_node)
builder.add_node("tools", ToolNode(ALL_TOOLS))
builder.add_node("triage_node", triage_node)
builder.add_node("diagnostic_node", diagnostic_node)

builder.add_conditional_edges(START, entry_router)
builder.add_conditional_edges("supervisor_node", supervisor_router)
builder.add_edge("tools", "supervisor_node")
builder.add_conditional_edges("triage_node", triage_router)
builder.add_edge("diagnostic_node", END)

memory = MemorySaver()
orchestrator_graph = builder.compile(checkpointer=memory)

print("✅ [Orchestrator] Graph compiled.")
print(f"   Tools registered: {len(ALL_TOOLS)}")