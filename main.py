import logging
import uuid

from fastapi import FastAPI
from dotenv import load_dotenv

import inngest
import inngest.fast_api

from custom_types import RAGChunkAndSrc, RAGUpsertResult
from data_loader import load_and_chunk_pdf, embed_texts
from qdrant_storage.vector_db import QdrantStorage


load_dotenv()


inngest_client = inngest.Inngest(
    app_id="rag_app",
    logger=logging.getLogger("uvicorn"),
    is_production=False,
    serializer=inngest.PydanticSerializer(),
)


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf"),
)
async def rag_ingest_pdf(ctx: inngest.Context):

    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        pdf_path = ctx.event.data["pdf_path"]
        source_id = ctx.event.data.get("source_id", pdf_path)

        chunks = load_and_chunk_pdf(pdf_path)

        return RAGChunkAndSrc(
            chunks=chunks,
            source_id=source_id
        )

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id

        vectors = embed_texts(chunks)

        ids = [
            str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}"))
            for i in range(len(chunks))
        ]

        payloads = [
            {
                "source": source_id,
                "text": chunks[i]
            }
            for i in range(len(chunks))
        ]

        QdrantStorage().upsert(
            ids=ids,
            vectors=vectors,
            payloads=payloads
        )

        return RAGUpsertResult(
            ingested=len(chunks)
        )

    chunks_and_src = await ctx.step.run(
        "load-and-chunk",
        lambda: _load(ctx),
        output_type=RAGChunkAndSrc,
    )

    ingested = await ctx.step.run(
        "embed-and-upsert",
        lambda: _upsert(chunks_and_src),
        output_type=RAGUpsertResult,
    )

    return ingested.model_dump()



import requests

from custom_types import (
    RAGSearchResult,
    RAGQueryResult
)


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(
        event="rag/query_pdf_ai"
    )
)
async def rag_query_pdf_ai(
    ctx: inngest.Context
):

    # Search vector database
    def _search(
        question: str,
        top_k: int = 5
    ) -> RAGSearchResult:

        query_vec = embed_texts(
            [question]
        )[0]

        store = QdrantStorage()

        found = store.search(
            query_vec,
            top_k
        )

        return RAGSearchResult(
            contexts=found["contexts"],
            sources=found["sources"]
        )

    question = ctx.event.data["question"]

    top_k = int(
        ctx.event.data.get(
            "top_k",
            5
        )
    )

    found = await ctx.step.run(
        "embed-and-search",
        lambda: _search(question, top_k),
        output_type=RAGSearchResult
    )

    # Create context block
    context_block = "\n\n".join(
        f"- {c}"
        for c in found.contexts
    )

    # Final prompt
    prompt = f"""
Use the following context to answer the question.

Context:
{context_block}

Question:
{question}

Answer concisely using only the context above.
"""

    # Call FREE local Ollama model
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )

    result = response.json()

    answer = result["response"]

    return RAGQueryResult(
        answer=answer,
        sources=found.sources,
        num_contexts=len(found.contexts)
    ).model_dump()


app = FastAPI()


@app.get("/")
def health_check():
    return {
        "status": "running",
        "app": "RAG AI Agent"
    }


inngest.fast_api.serve(
    app,
    inngest_client,
    functions=[rag_ingest_pdf,rag_query_pdf_ai],
)