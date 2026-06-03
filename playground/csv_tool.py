from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_ollama import ChatOllama
import pandas as pd

# 1. Load the uploaded CSV
df = pd.read_csv("..\docs\Book1.csv")

# 2. Initialize your LLM
llm = ChatOllama(model="qwen3.5:0.8b")

# 3. Create the agent
agent_executor = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    agent_type="tool-calling", # Efficient for tool calling
    allow_dangerous_code=True # Required to run python code locally
)

# 4. Run a query
response = agent_executor.invoke({"input": "how may zero values in 24th row"})
print(response["output"])