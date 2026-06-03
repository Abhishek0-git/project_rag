import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.document_loaders import TextLoader, PyMuPDFLoader

persist_directory = "db/chroma_db"
embedding_model = OllamaEmbeddings(model="bge-m3:latest")
chat_model = ChatOllama(model="granite3.1-moe:1b")
vector_store = Chroma(
    embedding_function=embedding_model,
    persist_directory=persist_directory,
    collection_metadata={"hnsw:space": "cosine"},
)


def load_documents(file_path="docs"):
    print(f"loading document {file_path}....")
    loader = None
    clean_path = file_path.strip()
    if clean_path.lower().endswith(".pdf"):
        loader = PyMuPDFLoader(clean_path, mode="single")
    elif clean_path.lower().endswith(".txt"):
        loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    for i, doc in enumerate(documents):
        print(f"\nDocumlnet No. {i+1}")
        print(f"Source : {doc.metadata['source']}")
        print(f"Content Length : {len(doc.page_content)} characters")
        print(f"Content Preview : {doc.page_content[:100]}.....")
    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=40):
    print("\nSplitting documents into chunks.....\n")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("creating enbedding and storing into ChromaDb.")
    batch_size = 50
    print("--- Creating vector store ---")
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Storing chunks {i} to {i + len(batch)}...")
        vector_store.add_documents(documents=batch)
    print(f"--- Vector store created and saved to {persist_directory} ---")
    return vector_store


def ask_query(query):
    retriver = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.3},
    )
    relevent_docs = retriver.invoke(query)
    context = chr(10).join([f"- {doc.page_content}" for doc in relevent_docs])
    combined_input = f"""You are an expert assistant designed to answer questions accurately using ONLY the provided context.

Context:
---------------------
{context}
---------------------

Given the context above, please answer the following question. 

Strict Rules for Your Response:
1. Rely ONLY on the clear facts directly mentioned in the context. Do not assume, extrapolate, or bring in outside knowledge.
2. If the context does not contain the answer to the question, state exactly: "I cannot find the answer in the provided documents." Do not try to make up an answer.
3. Keep your response concise, factual, and directly relevant to the question.

Question: {query}
    """
    message = [
        SystemMessage(content="you are a helpful assistent."),
        HumanMessage(content=combined_input),
    ]
    # result = chat_model.invoke(message)
    # return result.content
    for chunk in chat_model.stream(message):
        print(chunk.content, end="", flush=True)
