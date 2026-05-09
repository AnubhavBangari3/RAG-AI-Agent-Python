# RAG AI Agent in Python

A local RAG AI Agent built with FastAPI, Inngest, Qdrant, LlamaIndex, Sentence Transformers, and Ollama.

This project can ingest PDF files, split them into chunks, generate free local embeddings, store them in Qdrant, and answer questions using a local Ollama LLM.

## Tech Stack

- Python
- FastAPI
- Inngest
- Qdrant Vector Database
- LlamaIndex PDF Reader
- Sentence Transformers for free local embeddings
- Ollama for free local LLM answers
- uv for Python dependency management

## Project Structure

```text
RAG AI Agent in Python/
│
├── main.py
├── data_loader.py
├── custom_types.py
├── pyproject.toml
├── README.md
├── uv.lock
│
└── qdrant_storage/
    ├── vector_db.py
    ├── aliases/
    ├── collections/
    └── raft_state.json
```

## Features

- PDF ingestion using Inngest background functions
- PDF chunking using LlamaIndex SentenceSplitter
- Free local embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Vector storage and similarity search using Qdrant
- Free local answer generation using Ollama `llama3`
- FastAPI health check endpoint
- Event-based workflow with Inngest

## Prerequisites

Install these before starting:

- Python 3.13+
- uv
- Docker Desktop
- Node.js and npm
- Ollama
- Git

## 1. Clone the Repository

```bash
git clone https://github.com/AnubhavBangari3/RAG-AI-Agent-Python.git
cd RAG-AI-Agent-Python
```

## 2. Initialize Project with uv

If the project is already initialized, skip this step.

```bash
uv init .
```

## 3. Install Python Dependencies

```bash
uv add fastapi inngest llama-index-core llama-index-readers-file python-dotenv qdrant-client uvicorn streamlit sentence-transformers torch requests pydantic
```

## 4. Install and Start Ollama

Download Ollama from:

```text
https://ollama.com/download
```

Pull the local model:

```bash
ollama pull llama3
```

Start Ollama model:

```bash
ollama run llama3
```

Keep this terminal running.

## 5. Start Qdrant Vector Database

Make sure Docker Desktop is running first.

For PowerShell:

```powershell
docker run -d --name qdrant -p 6333:6333 -v "${pwd}/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

For CMD:

```bat
docker run -d --name qdrant -p 6333:6333 -v "%cd%/qdrant_storage:/qdrant/storage" qdrant/qdrant
```

Check container:

```bash
docker ps
```

Qdrant should be available at:

```text
http://localhost:6333
```

## 6. Start FastAPI Server

This is the command you were missing.

```bash
uv run uvicorn main:app --reload
```

FastAPI will run at:

```text
http://127.0.0.1:8000
```

Health check:

```text
http://127.0.0.1:8000/
```

Expected response:

```json
{
  "status": "running",
  "app": "RAG AI Agent"
}
```

## 7. Start Inngest Dev Server

Open a new terminal and run:

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

Inngest UI will open at:

```text
http://localhost:8288
```

## Required Running Terminals

You need these terminals running together:

### Terminal 1: Qdrant

```bash
docker ps
```

If Qdrant is not running:

```bash
docker start qdrant
```

### Terminal 2: FastAPI

```bash
uv run uvicorn main:app --reload
```

### Terminal 3: Inngest

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

### Terminal 4: Ollama

```bash
ollama run llama3
```

## Testing the Project

### Step 1: Add a PDF

Place your PDF inside the project root folder.

Example:

```text
RAG AI Agent in Python/test.pdf
```

### Step 2: Trigger PDF Ingestion

Open Inngest UI:

```text
http://localhost:8288
```

Trigger event:

```text
rag/ingest_pdf
```

Payload:

```json
{
  "data": {
    "pdf_path": "test.pdf",
    "source_id": "test_pdf"
  }
}
```

Expected result:

```json
{
  "ingested": 10
}
```

The number may change depending on PDF size.

### Step 3: Query the PDF

Trigger event:

```text
rag/query_pdf_ai
```

Payload:

```json
{
  "data": {
    "question": "What is this PDF about?",
    "top_k": 5
  }
}
```

Expected result:

```json
{
  "answer": "Generated answer from the PDF context",
  "sources": ["test_pdf"],
  "num_contexts": 5
}
```

## Important Notes

### Do not commit cache files

Add this to `.gitignore`:

```gitignore
__pycache__/
*.pyc
.venv/
qdrant_storage/collections/
qdrant_storage/aliases/
qdrant_storage/raft_state.json
```

### If Qdrant container already exists

Use:

```bash
docker start qdrant
```

Instead of running `docker run` again.

### If Docker gives daemon error

Open Docker Desktop first and wait until the engine starts.

Then run:

```bash
docker ps
```

### If Inngest functions are not visible

Make sure FastAPI is running first:

```bash
uv run uvicorn main:app --reload
```

Then start Inngest:

```bash
npx inngest-cli@latest dev -u http://127.0.0.1:8000/api/inngest --no-discovery
```

### If Ollama is not responding

Run:

```bash
ollama run llama3
```

Then retry the query event.

## Git Commands

```bash
git add .
git commit -m "Update README with setup and testing steps"
git push origin main
```

## Summary

This project runs a fully local RAG pipeline:

```text
PDF
↓
Chunks
↓
Local Embeddings
↓
Qdrant Vector DB
↓
Question Search
↓
Ollama LLM
↓
Final Answer
```
