# Import logging utilities
import logging

# Generate unique IDs for vector records
import uuid

# FastAPI framework
from fastapi import FastAPI

# Load environment variables from .env
from dotenv import load_dotenv

# Inngest SDK for background workflows
import inngest
import inngest.fast_api

# Custom Pydantic response/request models
from custom_types import (
    RAGChunkAndSrc,
    RAGUpsertResult
)

# PDF loader and embedding utilities
from data_loader import (
    load_and_chunk_pdf,
    embed_texts
)

# Qdrant vector database handler
from qdrant_storage.vector_db import QdrantStorage


# Load environment variables
load_dotenv()


# Create Inngest client
inngest_client = inngest.Inngest(

    # Application ID shown in Inngest dashboard
    app_id="rag_app",

    # Use Uvicorn logger
    logger=logging.getLogger("uvicorn"),

    # Local development mode
    is_production=False,

    # Serialize Pydantic models automatically
    serializer=inngest.PydanticSerializer(),
)


# -------------------------------------------------------------------
# PDF INGESTION FUNCTION
# -------------------------------------------------------------------

@inngest_client.create_function(

    # Function name shown in dashboard
    fn_id="RAG: Ingest PDF",

    # Event trigger name
    trigger=inngest.TriggerEvent(
        event="rag/ingest_pdf"
    ),
)
async def rag_ingest_pdf(
    ctx: inngest.Context
):

    # ---------------------------------------------------------------
    # Step 1: Load PDF and split into chunks
    # ---------------------------------------------------------------
    def _load(
        ctx: inngest.Context
    ) -> RAGChunkAndSrc:

        # PDF path received from event
        pdf_path = ctx.event.data["pdf_path"]

        # Source ID for tracking document
        source_id = ctx.event.data.get(
            "source_id",
            pdf_path
        )

        # Load and chunk PDF text
        chunks = load_and_chunk_pdf(
            pdf_path
        )

        # Return chunk data
        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id
        )

    # ---------------------------------------------------------------
    # Step 2: Create embeddings and store in Qdrant
    # ---------------------------------------------------------------
    def _upsert(
        chunks_and_src: RAGChunkAndSrc
    ) -> RAGUpsertResult:

        # Extract chunks and source
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        # Generate embeddings for chunks
        vectors = embed_texts(
            chunks
        )

        # Generate unique vector IDs
        ids = [
            str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{source_id}:{i}"
                )
            )
            for i in range(len(chunks))
        ]

        # Metadata payload for each chunk
        payloads = [
            {
                "source": source_id,
                "text": chunks[i]
            }
            for i in range(len(chunks))
        ]

        # Store vectors in Qdrant
        QdrantStorage().upsert(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )

        # Return ingestion result
        return RAGUpsertResult(
            ingested=len(chunks)
        )

    # Execute PDF loading step
    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc,
    )

    # Execute embedding + storage step
    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    # Return final result
    return ingested.model_dump()


# -------------------------------------------------------------------
# IMPORTS FOR RAG QUERY FUNCTION
# -------------------------------------------------------------------

# HTTP requests for Ollama API calls
import requests

# Additional response models
from custom_types import (
    RAGSearchResult,
    RAGQueryResult
)


# -------------------------------------------------------------------
# PDF QUESTION ANSWERING FUNCTION
# -------------------------------------------------------------------

@inngest_client.create_function(

    # Function name
    fn_id="RAG: Query PDF",

    # Trigger event
    trigger=inngest.TriggerEvent(
        event="rag/query_pdf_ai"
    )
)
async def rag_query_pdf_ai(
    ctx: inngest.Context
):

    # ---------------------------------------------------------------
    # Step 1: Search vector database
    # ---------------------------------------------------------------
    def _search(
        question: str,
        top_k: int = 5
    ) -> RAGSearchResult:

        # Convert question into embedding vector
        query_vec = embed_texts(
            [question]
        )[0]

        # Initialize Qdrant storage
        store = QdrantStorage()

        # Search similar chunks
        found = store.search(
            query_vec,
            top_k
        )

        # Return retrieved contexts
        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"]
        )

    # Extract question from event
    question = ctx.event.data["question"]

    # Number of chunks to retrieve
    top_k = int(
        ctx.event.data.get(
            "top_k",
            5
        )
    )

    # Run vector search step
    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult
    )

    # ---------------------------------------------------------------
    # Step 2: Build context prompt
    # ---------------------------------------------------------------

    # Join retrieved chunks into single context block
    context_block = "\n\n".join(
        f"- {c}"
        for c in found.contexts
    )

    # Prompt for local LLM
    prompt = f"""
Use the following context to answer the question.

Context:
{context_block}

Question:
{question}

Answer concisely using only the context above.
"""

    # ---------------------------------------------------------------
    # Step 3: Call local Ollama LLM
    # ---------------------------------------------------------------

    response = requests.post(

        # Ollama local API
        "http://localhost:11434/api/generate",

        # Request body
        json={

            # Local model name
            "model": "llama3",

            # Prompt sent to model
            "prompt": prompt,

            # Disable streaming
            "stream": False
        }
    )

    # Parse JSON response
    result = response.json()

    # Extract generated answer
    answer = result["response"]

    # Return final response
    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts)
    ).model_dump()


# -------------------------------------------------------------------
# FASTAPI APPLICATION
# -------------------------------------------------------------------

# Create FastAPI app
app = FastAPI()


# Health check endpoint
@app.get("/")
def health_check():

    return {
        "status": "running",
        "app": "RAG AI Agent"
    }


# Register Inngest functions with FastAPI
inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[
        rag_ingest_pdf,
        rag_query_pdf_ai
    ],
)