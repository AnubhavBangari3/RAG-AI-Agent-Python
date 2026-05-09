from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from sentence_transformers import SentenceTransformer
from typing import List


# Free local embedding model
embedding_model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Embedding dimension for MiniLM model
EMBED_DIM = 384


# Split large text into chunks
splitter = SentenceSplitter(
    chunk_size=1000,
    chunk_overlap=200
)


# Load PDF and split into chunks
def load_and_chunk_pdf(path: str):

    # Read PDF
    docs = PDFReader().load_data(file=path)

    # Extract text from pages
    texts = [
        d.text
        for d in docs
        if getattr(d, "text", None)
    ]

    chunks = []

    # Split text into smaller chunks
    for t in texts:
        chunks.extend(
            splitter.split_text(t)
        )

    return chunks


# Generate embeddings locally for free
def embed_texts(texts: List[str]) -> List[List[float]]:

    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=True
    )

    return embeddings.tolist()