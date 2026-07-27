import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from tavily import TavilyClient
from langchain_tavily import TavilySearch
from typing import List
from pydantic import BaseModel, Field


load_dotenv()

openai_llm = ChatOpenAI(temperature=0, model="gpt-5")
anthropic_llm = ChatAnthropic(temperature=0, model="claude-sonnet-4-6")
# tavily_client = TavilyClient()


# @tool
# def search(query: str):
#     """
#     Tool to search over internet. Use this tool if searching over internet is required.
#     :param query:
#     :return: str
#     """
#     return tavily_client.search(query=query)

class Source(BaseModel):
    """Schema for a source used by the agent"""
    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for the agent response"""
    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used to generate the answer")


def main():
    print("Type your query > ")
    tools = [TavilySearch()]
    agent = create_agent(model=anthropic_llm, tools=tools, response_format=AgentResponse)
    query = input()

    response = agent.invoke({"messages": [HumanMessage(content=query)]})
    print(response)


if __name__ == "__main__":
    main()
