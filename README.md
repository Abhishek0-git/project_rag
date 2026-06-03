# Project RAG 

Hey there! This is a local RAG (Retrieval-Augmented Generation) pipeline I've been hacking away at using Python and LangChain. 

This isn't just a standalone project—it's actually step one of a much bigger plan. The goal is to build a full-scale, autonomous AI Agent, and this repository contains its core "knowledge tool" for saving data and pulling up info when asked.

Everything runs completely locally on your machine, so your files stay private.

---

## What it does right now

*   **Chat with your files:** You can feed it any local `.txt` or `.pdf` file and start asking questions about it immediately.
*   **Zero guessing/hallucinations:** I've locked the LLM down with strict prompt rules. It will *only* answer using facts from the documents you gave it. If the answer isn't in there, it'll just flat out tell you it can't find it instead of making things up.
*   **Local Storage:** It splits up your files and saves them as vectors in a local **ChromaDB** instance using local embeddings.

---

## 📸 A quick look at it running

### 1. Feeding it a document
Here's me dropping a LangChain cheat sheet PDF into the terminal. It instantly processes it, chunks the text, saves it to ChromaDB, and answers a specific question perfectly based on that file.

![Terminal Ingestion]<img width="1920" height="1080" alt="Screenshot 2026-06-02 225420" src="https://github.com/user-attachments/assets/f4b67526-b6d2-485b-97a7-3ec28e79649e" />

### 2. Live streaming & code formatting
It handles follow-up technical questions with ease and spits out perfectly formatted code blocks directly in the terminal before you exit the loop.

![Terminal Stream]<img width="1920" height="1080" alt="Screenshot 2026-06-02 225450" src="https://github.com/user-attachments/assets/d501e982-0824-4492-8ab3-9958140bdb51" />

---

## What's coming next (The Roadmap)

Since this is just the foundation for a bigger AI agent, I have a ton of stuff planned:
*   **More file types:** Making it smart enough to read `.csv`, `.docx`, and images (`.png`, `.jpeg`).
*   **Document Generation:** Giving the agent the power to actually draft and export its own `.pdf` files from scratch.
*   **Turning it into a server:** I’ll be adding API endpoints soon so this can run as a backend service and talk to other apps.
*   **Agentic Workflows:** Integrating custom machine learning models so the agent can reason, make decisions, call its own tools, and handle complex jobs completely on its own.

---

## 🤝 Let's team up! (Looking for a Frontend Dev)

Now that the backend logic is solid and I'm moving towards an API-server structure, **I'm looking for a Frontend Developer to collaborate with!** 

If you want to team up and build a clean, modern web interface/dashboard for this AI agent, please reach out, open an issue, or drop a DM! 

---

## Tech Stack
*   Python
*   LangChain / LangChain Community
*   ChromaDB (Local Vector DB)
*   Ollama (For local models & embeddings)

---

## How to run it locally

1. Clone the repo:
```bash
   git clone [https://github.com/Abhishek0-git/project_rag.git](https://github.com/Abhishek0-git/project_rag.git)
   cd project_rag
```
Set up your virtual environment and install your packages (langchain, langchain-chroma, langchain-ollama, pymupdf).

Make sure your local Ollama server is running with your embedding and chat models pulled.

Fire up the CLI:
```base
  python app.py
```

---

## 📄 Code Documentation: `advanced_retrieval.py`

This module contains the core heavy-lifting logic for the document ingestion pipeline and the multi-query retrieval chain. It handles everything from file parsing and semantic splitting to vector database management and streaming execution.

Below is a detailed breakdown of the functions implemented in this file:

---

## 📄 Code Documentation: `advanced_retrieval.py`

This module handles all the backend heavy-lifting for document loading, splitting, database indexing, and advanced multi-query searching.

---

### ⚙️ Global Configurations & Customization

These attributes are configured as module-level global variables inside `advanced_retrieval.py`. Because they are exposed, you can easily overwrite or re-initialize them directly from your `app.py` file to swap models or database paths dynamically.

| Attribute | Type | Default Value | Description |
| :--- | :--- | :--- | :--- |
| `persist_directory` | `str` | `"db/chroma_db"` | The local folder where ChromaDB saves its vector indices. |
| `embedding_model` | `OllamaEmbeddings` | `model="bge-m3:latest"` | The model used to convert text chunks into vector embeddings. |
| `chat_model` | `ChatOllama` | `model="granite3.1-moe:1b"` | The LLM engine handling query expansion and contextual response generation. |
| `vector_store` | `Chroma` | `Chroma(...)` | The active vector database configuration using `cosine` distance similarity matching. |

*Want to change models or paths on the fly?* Just update them on your imported instance inside `app.py` before kicking off your pipelines:
```python
import advanced_retrieval

advanced_retrieval.persist_directory = "db/custom_db_path"
# Re-assign or update the models/vector_store as needed!
```

---

### 1. `load_documents(file_path)`
Responsible for targeting and loading localized documents into standardized LangChain document structures.

*   **Parameters:** 
    *   `file_path` *(str)*: The relative or absolute path to the target document.
*   **Behavior:** 
    *   Sanitizes the input string and dynamically selects the correct parsing class based on the file extension.
    *   Utilizes `PyMuPDFLoader` for `.pdf` files to extract text data accurately on a single-page strategy.
    *   Utilizes `TextLoader` with `utf-8` encoding for `.txt` files.
    *   Prints localized console telemetry showcasing document index numbers, original metadata sources, and text previews for transparency.
*   **Returns:** 
    *   `List[Document]`: A list of loaded LangChain document objects containing raw content and source metadata.

---

### 2. `split_documents(documents, chunk_size=800, chunk_overlap=40)`
Chunks the raw extracted text into smaller, overlapping segments to stay within LLM token constraints and preserve structural meaning.

*   **Parameters:**
    *   `documents` *(List[Document])*: The raw document objects retrieved from the loader function.
    *   `chunk_size` *(int, default=800)*: The maximum character length of an individual text segment.
    *   `chunk_overlap` *(int, default=40)*: The character overlap buffer maintained between consecutive chunks to avoid cutting off contextual phrases.
*   **Behavior:**
    *   Initializes a `RecursiveCharacterTextSplitter` configured with the parameters above.
    *   Recursively looks for natural text boundaries (like paragraphs, sentences, and words) to slice the content cleanly.
*   **Returns:**
    *   `List[Document]`: A list of smaller, chunked document pieces optimized for vectorization.

---

### 3. `create_vector_store(chunks, persist_directory="db/chroma_db")`
Generates high-dimensional semantic embeddings from the text chunks and indexes them safely in a persistent local vector database.

*   **Parameters:**
    *   `chunks` *(List[Document])*: The optimized text segments generated by the splitting pipeline.
    *   `persist_directory` *(str, default="db/chroma_db")*: The local directory where the Chroma DB vector files are physically serialized.
*   **Behavior:**
    *   Processes the data array using a **batch-processing loop** (handling 50 chunks at a time) to avoid memory spikes or overloading the embedding service.
    *   Uses local `OllamaEmbeddings` powered by the `bge-m3` model to convert strings into mathematical vectors.
    *   Stores the embeddings within a persistent **Chroma** instance structured with `cosine` similarity space configurations.
*   **Returns:**
    *   `Chroma`: An instantiated vector store object ready for immediate querying.

---

### 4. `ask_query(query)`
Executes the advanced multi-query generation process, retrieves matching documents from the vector base, and coordinates the response generation.

*   **Parameters:**
    *   `query` *(str)*: The natural language question submitted by the user.
*   **Behavior:**
    *   **Multi-Perspective Expansion:** Pass a specialized prompt into `MultiQueryRetriever` using the `granite3.1-moe:1b` model. This automatically forces the model to rewrite the initial user query from 5 distinct perspectives, significantly improving similarity search matches.
    *   **Context Compression:** Fetches the documents matching these variations and formats them inside a zero-hallucination instruction prompt template.
    *   **LCEL Chain Execution:** Orchestrates data flow cleanly using LangChain Expression Language (`RunnablePassthrough` -> `Prompt` -> `Model` -> `OutputParser`).
    *   **Real-time Streaming:** Consumes the generation stream directly, flushing chunk updates instantly to the terminal output to minimize perceived latency.
*   **Returns:**
    *   `None` *(Outputs the answers directly to stdout in a stream format)*.
