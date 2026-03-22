from langchain_core.runnables import RunnableLambda
from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

llm_model = init_chat_model(model="gpt-5-mini", temperature=0)
llm = RunnableLambda(llm_model.invoke)

prompt = RunnableLambda(lambda text:
                        PromptTemplate.from_template(text).format_prompt().to_messages())

chain = prompt | llm

if __name__ == "__main__":
    result = chain.invoke("프랑스의 수도는 어디인가요?")
    print(result.content)
