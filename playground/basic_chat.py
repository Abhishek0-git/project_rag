from dotenv import load_dotenv
import os
import time
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

persist_directory = "db/chroma_db"
embedding_model = OllamaEmbeddings(model="bge-m3:latest")
chat_model = ChatOllama(model="granite3.1-moe:1b")
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"},
)

def ingestion_pipeline():
    loader = DirectoryLoader(
        path="docs",
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)

    batch_size = 50
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Storing chunks {i} to {i + len(batch)}...")
        vector_store.add_documents(documents=batch)


def retrival_pipeline():
    query = "Who were the original creators of the Tom and Jerry series in 1940?"
    retriver = vector_store.as_retriever(
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
    # result = chat_model.invoke(message)
    # print(f"AI answer : {result.content}")
    for chunk in chat_model.stream(message):
        print(chunk.content, end="", flush=True)


def main():
    # ingestion_pipeline()
    retrival_pipeline()


if __name__ == "__main__":
    main()
