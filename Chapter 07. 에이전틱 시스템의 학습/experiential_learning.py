from typing import Annotated
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langchain_core.messages import HumanMessage
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError(
        "OPENAI_API_KEY가 설정되지 않았습니다."
        "환경변수 또는 .env 파일에서 설정해주세요."
    )

llm = init_chat_model(model="gpt-5-mini", temperature=0)


def call_model(state: MessagesState):
    response = llm.invoke(state["messages"])
    return {"messages": response}


class InsightAgent:
    def __init__(self):
        self.insights = []
        self.promoted_insights = []
        self.demoted_insights = []
        self.reflections = []

    def generate_insight(self, observation):
        messages = [HumanMessage(content=f"다음 관찰을 바탕으로 인사이트를 생성하세요: '{observation}'")]

        builder = StateGraph(MessagesState)
        builder.add_node("generate_insight", call_model)
        builder.add_edge(START, "generate_insight")
        graph = builder.compile()

        result = graph.invoke({"messages": messages})

        generated_insight = result["messages"][-1].content
        self.insights.append(generated_insight)
        print(f"생성된 인사이트: {generated_insight}")
        return generated_insight

    def promote_insight(self, insight):
        if insight in self.insights:
            self.insights.remove(insight)
            self.promoted_insights.append(insight)
            print(f"승격된 인사이트: {insight}")
        else:
            print(f"'{insight}'인사이트를 찾을 수 없습니다.")

    def demote_insight(self, insight):
        if insight in self.promoted_insights:
            self.promoted_insights.remove(insight)
            self.demoted_insights.append(insight)
            print(f"강등된 인사이트: {insight}")
        else:
            print(f"'{insight}'인사이트를 찾을 수 없습니다.")

    def edit_insight(self, old_insight, new_insight):
        if old_insight in self.insights:
            index = self.insights.index(old_insight)
            self.insights[index] = new_insight
        elif old_insight in self.promoted_insights:
            index = self.promoted_insights.index(old_insight)
            self.promoted_insights[index] = new_insight
        elif old_insight in self.demoted_insights:
            index = self.demoted_insights.index(old_insight)
            self.demoted_insights[index] = new_insight
        else:
            print(f"'{old_insight}'인사이트를 찾을 수 없습니다.")
            return
        print(f"수정된 인사이트: '{old_insight}' -> '{new_insight}'")

    def show_insights(self):
        print("\n현재 인사이트:")
        print(f"인사이트: {self.insights}")
        print(f"승격된 인사이트: {self.promoted_insights}")
        print(f"강등된 인사이트: {self.demoted_insights}")

    def reflect(self, reflexion_prompt):
        builder = StateGraph(MessagesState)
        builder.add_node("reflection", call_model)
        builder.add_edge(START, "reflection")
        graph = builder.compile()

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=reflexion_prompt
                    )
                ]
            }
        )
        reflection = result["messages"][-1].content
        self.reflections.append(reflection)
        print(f"성찰: {reflection}")


agent = InsightAgent()

reports = [
    ("웹사이트 트래픽이 15% 증가했지만, 바운스율이 40%에서 55%로 급격히 증가했습니다.",
     False),
    ("이메일 열람률이 25%로 향상되었지만, 20% 목표를 초과했습니다.", True),
    ("장바구니 포기율이 60%에서 68%로 증가했지만, 50% 목표를 놓쳤습니다.",
     False),
    ("평균 주문 가치가 8% 증가했지만, 5% 증가 목표를 놓쳤습니다.", True),
    ("신규 구독자 수가 5% 감소했지만, 10% 성장 목표를 놓쳤습니다.",
     False),
]
for text, hit_target in reports:
    insight = agent.generate_insight(text)
    if hit_target:
        agent.promote_insight(insight)
    else:
        agent.demote_insight(insight)
if agent.promoted_insights:
    original = agent.promoted_insights[0]
    agent.edit_insight(original, f'개선된 인사이트: {original} 방문자 경험 개선을 위한 랜딩 페이지 UX 변경 조사')
agent.show_insights()
reflection_prompt = (
        "승격된 인사이트를 바탕으로, 다음 분기에 실행할 수 있는 하나의 고영향 실험을 제안하세요:" + f"\n{agent.promoted_insights}"
)
agent.reflect(reflection_prompt)
