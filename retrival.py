from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = OllamaEmbeddings(model="bge-m3:latest")
chat_model = ChatOllama(model="granite3.1-moe:1b")

def retrival_pipeline(query):
    db = Chroma(
        embedding_function=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )
    # query = "Who were the original creators of the Tom and Jerry series in 1940?"
    retriver = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3},
    )
    relevent_docs = retriver.invoke(query)
    combined_input = f"""based on the following documents, please answer this question : {query}
    documents : {chr(10).join([f"- {doc.page_content}" for doc in relevent_docs])}
    please provide a clear, helpful answer using only the information from these documents. if you can't find the answer in documnets, say i don't have enough information to answer that question based on the provided documents.
    """
    message = [
        SystemMessage(content="you are a helpful assistent."),
        HumanMessage(content=combined_input),
    ]
    result = chat_model.invoke(message)
    # print(f"AI answer : {result.content}")
    return result.content
    # for chunk in chat_model.stream(message):
    #     print(chunk.content, end="", flush=True)