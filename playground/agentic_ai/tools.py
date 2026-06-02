from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain.messages import AIMessage

#---- Search tool for ai to search on web ----
search_tool = DuckDuckGoSearchRun()

#---- Costum tool for ai for calculation purpose ----
@tool(
    "calculator",
    description="Performs arithmetic calculations. Use this for any math problems.",
)
def calc(expression: str) -> str:
    """Evaluate mathematical expressions."""
    return str(eval(expression))

#---- it will show all the tool call that our ai did to get the answer ----
# note : it will take the agent's respone['messages'] as input
def get_tool_call_data(chat_data:dict) -> None:
    try:
        for msg in chat_data["data"]:
            if isinstance(msg, AIMessage) and msg.tool_calls:
                for tool_call in msg.tool_calls:
                    print(f"Tool Name: {tool_call['name']}")
                    print(f"Arguments: {tool_call['args']}")
                    print(f"ID:        {tool_call['id']}\n")
    except Exception as e:
        print(e)
