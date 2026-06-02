from langchain.agents import create_agent
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver
from tools import search_tool, get_tool_call_data, calc

#---- storing ai conversation ----
chat_data = {"data": []}

llm = ChatOllama(model="qwen3.5:0.8b")
agent = create_agent(model=llm, tools=[search_tool, calc], checkpointer=MemorySaver())

#---- function to ask queries ---- 
def ask_ai_agent(query: str):
    try:
        response = agent.invoke(
            {"messages": [{"role": "user", "content": query}]},
            {"configurable": {"thread_id": "srsr2"}},
        )
        print(response["messages"][-1].content)
        chat_data["data"] = response["messages"]
    except Exception as e:
        print(e)

#---- Starting loop ----
print("choose \n (1) to ask questions\n (2) to get the toll calling data\n (3) to exit")
print("-" * 100)

while True:
    option = input("choise : ")
    if option == "1":
        query = input("what can i help you for your research : ")
        ask_ai_agent(query)
    elif option == "2":
        get_tool_call_data(chat_data)
    elif option == "3":
        print("thank you :)")
        break
    else:
        print("invalid choise")
    print("-" * 100)
