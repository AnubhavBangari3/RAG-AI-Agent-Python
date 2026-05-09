import pydantic


# Store chunks and source information
class RAGChunkAndSrc(pydantic.BaseModel):

    # List of text chunks
    chunks: list[str]

    # PDF/document source ID
    source_id: str = None


# Result after vector DB ingestion
class RAGUpsertResult(pydantic.BaseModel):

    # Number of vectors inserted
    ingested: int


# Search response from vector DB
class RAGSearchResult(pydantic.BaseModel):

    # Retrieved matching contexts
    contexts: list[str]

    # Source documents
    sources: list[str]


# Final RAG answer response
class RAGQueryResult(pydantic.BaseModel):

    # LLM generated answer
    answer: str

    # Source documents used
    sources: list[str]

    # Number of retrieved contexts
    num_contexts: int