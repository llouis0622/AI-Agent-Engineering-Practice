from typing import TypedDict, Annotated, Optional
from langgraph.graph import StateGraph, START, END
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

llm = init_chat_model(model="gpt-5-mini")


class AgentState(TypedDict):
    user_message: str
    user_id: str
    issue_type: Optional[str]
    step_result: Optional[str]
    response: Optional[str]


def categorize_issue(state: AgentState) -> AgentState:
    prompt = (
        f"이 지원 요청을 'billing' 또는 'technical'로 분류하세요.\n\n"
        f"메시지: {state['user_message']}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    kind = response.content.strip().lower()
    if "billing" in kind:
        kind = "billing"
    elif "technical" in kind:
        kind = "technical"
    else:
        kind = "technical"  # 기본값

    return {"issue_type": kind}


def handle_invoice(state: AgentState) -> AgentState:
    return {"step_result": f"Invoice details for {state['user_id']}"}


def handle_refund(state: AgentState) -> AgentState:
    return {"step_result": "Refund process initiated"}


def handle_login(state: AgentState) -> AgentState:
    return {"step_result": "Password reset link sent"}


def handle_performance(state: AgentState) -> AgentState:
    return {"step_result": "Performance metrics analyzed"}


def summarize_response(state: AgentState) -> AgentState:
    details = state.get("step_result", "")
    prompt = f"다음 내용을 바탕으로 간결한 고객 응답을 작성하세요: {details}"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"response": response.content.strip()}


graph_builder = StateGraph(AgentState)

graph_builder.add_node("categorize_issue", categorize_issue)
graph_builder.add_node("handle_invoice", handle_invoice)
graph_builder.add_node("handle_refund", handle_refund)
graph_builder.add_node("handle_login", handle_login)
graph_builder.add_node("handle_performance", handle_performance)
graph_builder.add_node("summarize_response", summarize_response)

graph_builder.add_edge(START, "categorize_issue")


def top_router(state: AgentState):
    return "billing" if state["issue_type"] == "billing" else "technical"


graph_builder.add_conditional_edges(
    "categorize_issue",
    top_router,
    {"billing": "handle_invoice", "technical": "handle_login"}
)


def billing_router(state: AgentState):
    msg = state["user_message"].lower()
    return "refund" if "refund" in msg else "invoice_end"


graph_builder.add_conditional_edges(
    "handle_invoice",
    billing_router,
    {"refund": "handle_refund", "invoice_end": "summarize_response"}
)


def tech_router(state: AgentState):
    msg = state["user_message"].lower()
    return "performance" if "performance" in msg else "login_end"


graph_builder.add_conditional_edges(
    "handle_login",
    tech_router,
    {"performance": "handle_performance", "login_end": "summarize_response"}
)

graph_builder.add_edge("handle_refund", "summarize_response")
graph_builder.add_edge("handle_performance", "summarize_response")

graph_builder.add_edge("summarize_response", END)

graph = graph_builder.compile()

if __name__ == "__main__":
    initial_state = {
        "user_message": "안녕하세요, 인보이스와 (가능하다면) 환불 관련 도움을 받고 싶습니다.",
        "user_id": "U1234"
    }

    result = graph.invoke(initial_state)
    print(result["response"])
