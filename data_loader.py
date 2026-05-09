# PDF reader from LlamaIndex
from llama_index.readers.file import PDFReader

# Text chunking utility
from llama_index.core.node_parser import SentenceSplitter

# Free local embedding model
from sentence_transformers import SentenceTransformer

# Type hint support
from typing import List


# -------------------------------------------------------------------
# LOCAL EMBEDDING MODEL
# -------------------------------------------------------------------

# Load free HuggingFace embedding model locally
embedding_model = SentenceTransformer(

    # Small and fast embedding model
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Embedding dimension produced by MiniLM model
EMBED_DIM = 384


# -------------------------------------------------------------------
# TEXT SPLITTER CONFIGURATION
# -------------------------------------------------------------------

# Split long documents into overlapping chunks
splitter = SentenceSplitter(

    # Maximum chunk size
    chunk_size=1000,

    # Overlap between chunks for better context continuity
    chunk_overlap=200
)


# -------------------------------------------------------------------
# LOAD PDF AND SPLIT INTO CHUNKS
# -------------------------------------------------------------------

def load_and_chunk_pdf(
    path: str
):

    # Read PDF file
    docs = PDFReader().load_data(
        file=path
    )

    # Extract valid text from PDF pages
    texts = [
        d.text
        for d in docs
        if getattr(d, "text", None)
    ]

    # Store final chunks
    chunks = []

    # Split each page into smaller chunks
    for t in texts:

        chunks.extend(
            splitter.split_text(t)
        )

    # Return all chunks
    return chunks


# -------------------------------------------------------------------
# GENERATE TEXT EMBEDDINGS
# -------------------------------------------------------------------

def embed_texts(
    texts: List[str]
) -> List[List[float]]:

    # Convert text into embeddings
    embeddings = embedding_model.encode(

        # Input texts
        texts,

        # Show embedding progress
        show_progress_bar=True
    )

    # Convert numpy array into Python list
    return embeddings.tolist()