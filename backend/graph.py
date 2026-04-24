from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, START, StateGraph


class ConversationState(TypedDict, total=False):
    session_id: str
    channel: str
    current_message: str
    history: Annotated[list[dict[str, Any]], operator.add]
    tool_events: Annotated[list[dict[str, Any]], operator.add]
    slots: dict[str, Any]
    current_entities: dict[str, Any]
    intent: str
    intent_confidence: float
    next_agent: str
    reply: str
    action: str
    metadata: dict[str, Any]
    candidate_doctors: list[dict[str, Any]]
    doctor_profile: dict[str, Any] | None
    booking_state: dict[str, Any]
    triage_state: dict[str, Any]
    safety_flags: list[str]
    last_intent: str


def create_conversation_graph(agent, checkpointer: Any | None = None) -> Any:
    graph = StateGraph(ConversationState)

    graph.add_node("contextualize_turn", agent.contextualize_turn)
    graph.add_node("classify_intent", agent.classify_intent_node)
    graph.add_node("greeting_agent", agent.greeting_agent)
    graph.add_node("faq_agent", agent.faq_agent)
    graph.add_node("safety_agent", agent.safety_agent)
    graph.add_node("handoff_agent", agent.handoff_agent)
    graph.add_node("clarify_agent", agent.clarify_agent)
    graph.add_node("doctor_info_agent", agent.doctor_info_agent)
    graph.add_node("recommend_doctor_agent", agent.recommend_doctor_agent)
    graph.add_node("booking_agent", agent.booking_agent)
    graph.add_node("triage_agent", agent.triage_agent)
    graph.add_node("finalize_turn", agent.finalize_turn)

    graph.add_edge(START, "contextualize_turn")
    graph.add_edge("contextualize_turn", "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        agent.route_intent,
        {
            "greeting_agent": "greeting_agent",
            "faq_agent": "faq_agent",
            "safety_agent": "safety_agent",
            "handoff_agent": "handoff_agent",
            "clarify_agent": "clarify_agent",
            "doctor_info_agent": "doctor_info_agent",
            "recommend_doctor_agent": "recommend_doctor_agent",
            "booking_agent": "booking_agent",
            "triage_agent": "triage_agent",
        },
    )

    for node_name in (
        "greeting_agent",
        "faq_agent",
        "safety_agent",
        "handoff_agent",
        "clarify_agent",
        "doctor_info_agent",
        "recommend_doctor_agent",
        "booking_agent",
        "triage_agent",
    ):
        graph.add_conditional_edges(
            node_name,
            agent.route_after_agent,
            {
                "recommend_doctor_agent": "recommend_doctor_agent",
                "booking_agent": "booking_agent",
                "triage_agent": "triage_agent",
                "finalize_turn": "finalize_turn",
            },
        )

    graph.add_edge("finalize_turn", END)
    return graph.compile(checkpointer=checkpointer)
