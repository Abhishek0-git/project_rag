from langchain_community.retrievers import BM25Retriever
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.documents import Document

doc_list = [
    Document(page_content="I like apples", metadata={"id": 1}),
    Document(page_content="I like oranges", metadata={"id": 2}),
    Document(page_content="apples and oranges are fruits", metadata={"id": 3}),
    Document(page_content="I like computers by Apple", metadata={"id": 4}),
    Document(page_content="I love fruit juice", metadata={"id": 5}),
]


def test_retrivers(
    model,doc_list=doc_list, query="apple", k=3, weights=[0.5, 0.5], metadata=False
):
    try:
        sparse_retriever = BM25Retriever.from_documents(doc_list)
        sparse_retriever.k = k

        embedding_model = OllamaEmbeddings(model=model)
        vector_db = Chroma.from_documents(doc_list, embedding_model)
        dense_retriever = vector_db.as_retriever()
        dense_retriever.search_kwargs = {"k": k}

        hybrid_retriever = EnsembleRetriever(
            retrievers=[sparse_retriever, dense_retriever], weights=weights
        )
    except Exception as exception:
        print(exception)
        return

    print(f"\n=== Query: '{query}' ===")

    print("\n" + ("    Sparse (BM25) Results    ").center(100, "-"))
    for doc in sparse_retriever.invoke(query):
        print(f"- {doc.page_content}")
        if metadata:
            print(f"-- metadata - {doc.metadata}")

    print("\n" + ("    Dense (Semantic) Results    ").center(100, "-"))
    for doc in dense_retriever.invoke(query):
        print(f"- {doc.page_content}")
        if metadata:
            print(f"-- metadata - {doc.metadata}")

    print("\n" + ("    Hybrid Results    ").center(100, "-"))
    for doc in hybrid_retriever.invoke(query):
        print(f"- {doc.page_content}")
        if metadata:
            print(f"-- metadata - {doc.metadata}")

    print("-" * 100)


if __name__ == "__main__":
    test_retrivers()
