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
