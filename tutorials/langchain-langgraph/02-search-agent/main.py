from dotenv import load_dotenv
load_dotenv()
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

tavily = TavilyClient()


@tool
def search(query: str) -> str:
    """
    Tool that search over internet
    Args:
       query : the query to search for
    Returns:
       The search result
    """
    return tavily.search(query)


llm = ChatOpenAI(model= "gpt-5")
tools = [search]
agent = create_agent(model=llm,tools=tools)

def main():
    result = agent.invoke({"messages": HumanMessage(content="what is the weather in ankara")})
    print(result)

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
