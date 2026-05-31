from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

load_dotenv()

def load_documents(document, doc_path="docs"):
    print(f"loading document {document}....")

    if not os.path.exists(doc_path):
        raise FileNotFoundError(f"{doc_path} not exist....")

    loader = DirectoryLoader(
        path=doc_path,
        glob=f"{document}",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},  # used to encode the file data
    )
    documents = loader.load()

    # if len(documents) == 0:
    #     raise FileNotFoundError(f"{doc_path} is Empty....")

    for i, doc in enumerate(documents):
        print(f"\nDocumlnet No. {i+1}")
        print(f"Source : {doc.metadata['source']}")
        print(f"Content Length : {len(doc.page_content)} characters")
        print(f"Content Preview : {doc.page_content[:100]}.....")

    return documents


def split_documents(documents, chunk_size=800, chunk_overlap=0):
    print("\nSplitting documents into chunks.....\n")

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    # for i, chunk in enumerate(chunks[300:305]):
    #     print(f"\nChunk No. {i+1}")
    #     print(f"Source : {chunk.metadata['source']}")
    #     print(f"Content Length : {len(chunk.page_content)} characters")
    #     print(f"Content Preview : {chunk.page_content[:100]}.....")
    #     print("_" * 50)

    # print(f".... and {len(chunks) - 5} more.")

    return chunks


def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print("creating enbedding and storing into ChromaDb.")

    batch_size = 50

    embedding_model = OllamaEmbeddings(
        model="bge-m3:latest"
    )

    print("--- Creating vector store ---")
    vector_store = Chroma.from_documents(
        documents=chunks[:batch_size],
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"},
    )

    # 2. Add the remaining chunks in controlled batches
    for i in range(batch_size, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        print(f"Storing chunks {i} to {i + len(batch)}...")
        vector_store.add_documents(documents=batch)
        # time.sleep(2)

    print(f"--- Vector store created and saved to {persist_directory} ---")

    return True


# def main():
#     print("main function")

#     # Loading the file
#     documents = load_documents()

#     # Chunking the file
#     chunks = split_documents(documents)

#     # Embedding and storing in ChromaDb
#     create_vector_store(chunks)


# if __name__ == "__main__":
#     main()
