from langchain_core.tools import tool
from langsmith import traceable
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain.chat_models import init_chat_model
from typing import Union
load_dotenv()


@tool
def get_product_price(product: str) -> Union[int, str]:
    """Look up the product price
    :param product: str
    :return price: int

    """
    product_price_map = {"headphone": 100, "watch": 200, "pen": 50}

    try:
        return product_price_map[product]
    except KeyError:
        return "No product available"


@tool
def get_discounted_price(original_price: int, tier: str) -> Union[int, str]:
    """Look up the discounted price for a tier
    :param original_price: int
    :param tier: str
    :return discounted_price: int
    """

    discount_percentage_map = {"gold": 25, "silver": 20, "bronze": 10}
    try:
        discounted_price = original_price * (1 - discount_percentage_map[tier] / 100)
        return discounted_price
    except KeyError:
        return "requested tier unavailable."


tools = [get_product_price, get_discounted_price]
# llm = init_chat_model("anthropic:claude-sonnet-4-6", temperature=0)
llm = init_chat_model("openai:gpt-5", temperature=0)

llm_with_tools = llm.bind_tools(tools)
# just dictionary with tool function name as key and function itself as value.
tools_map = {t.name: t for t in tools}


@traceable(name="Agent-Under-Hood")
def run_agent(query_: str):
    system_prompt = "You are an assistant agent in an e-commerce website. You response customers' query about product" \
                    "price and discounted price according to their tier. You must follow the following strict rules " \
                    "at all times: \n\n (1) Never ever response price on your own.\n (2) Always call tools provided " \
                    "to get original price as well as discounted price.\n (3) If you cannot get prices from tools, " \
                    "just say so.  Never ever invent it.\n (4) Strictly refuse to answer any unrelated query."
    max_iteration = 10
    messages = [SystemMessage(system_prompt), HumanMessage(query_)]

    for i in range(max_iteration):
        response = llm_with_tools.invoke(messages)
        messages.append(response)  # no need AIMessage here. response is already AIMessage
        if response.tool_calls:
            for tool_call in response.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                tool_response = tools_map[function_name].invoke(function_args)
                messages.append(ToolMessage(content=str(tool_response), tool_call_id=tool_call_id))
            i += 1
        else:  # this is a final answer
            print(response.content)
            break


if __name__ == "__main__":
    query = "What is the discounted price of a pen for the platinum tier member."
    # query = "What is the diameter of the earth?"
    run_agent(query)
