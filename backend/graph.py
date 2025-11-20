from langgraph.graph import StateGraph, START, END
from typing import TypedDict

from .Nodes.intent import IntentClassifier
from .Nodes.rec_doc import RecommendDoctor
from .Nodes.bk_apt import BookAppointment
from .Nodes.general import GeneralQuery


# --- 1. Define the state schema ---
class ChatState(TypedDict, total=False):
    user_input: str
    intent: str
    response: str
    doctor_name:str
    specialization:str
    date:str
    time:str
    context:list


# --- 2. Create the graph ---
def create_graph():
    graph = StateGraph(ChatState)   # ✅ Pass schema here

    graph.add_node("classify_intent", IntentClassifier())
    graph.add_node("recommend_doctor", RecommendDoctor())
    graph.add_node("book_appointment", BookAppointment())
    graph.add_node("general_query", GeneralQuery())

    graph.add_edge(START, "classify_intent")

# conditional routing
    graph.add_conditional_edges(
        "classify_intent",
        lambda state: state.get("intent"),
        {
            "recommend_doctor": "recommend_doctor",
            "book_appointment": "book_appointment",
            "general_query": "general_query",
        },
    )

    # terminal edges
    graph.add_edge("recommend_doctor", END)
    graph.add_edge("book_appointment", END)
    graph.add_edge("general_query", END)


    return graph
