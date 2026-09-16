from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-5.6-terra", temperature=0, reasoning_effort="none")
prompt = hub.pull("rlm/rag-prompt")

generation_chain = prompt | llm | StrOutputParser()