# Import Qdrant client to connect with Qdrant vector database
from qdrant_client import QdrantClient

# Import required Qdrant models
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)


# Wrapper class for all Qdrant vector database operations
class QdrantStorage:

    # Initialize Qdrant connection and collection
    def __init__(
        self,
        url="http://localhost:6333",
        collection="docs",
        dim=384
    ):

        # Create Qdrant client
        self.client = QdrantClient(
            url=url,
            timeout=30
        )

        # Store collection name
        self.collection = collection

        # Check if collection already exists
        if not self.client.collection_exists(
            collection_name=self.collection
        ):

            # Create collection with vector dimension and similarity metric
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )

    # Insert or update vectors into Qdrant
    def upsert(
        self,
        ids,
        vectors,
        payloads
    ):

        # Prepare Qdrant points
        points = []

        # Convert ids, vectors, and payloads into PointStruct objects
        for i in range(len(ids)):

            points.append(
                PointStruct(
                    id=ids[i],
                    vector=vectors[i],
                    payload=payloads[i]
                )
            )

        # Store points in Qdrant collection
        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    # Search similar vectors from Qdrant
    def search(
        self,
        query_vector,
        top_k=5
    ):

        # Query Qdrant using the question embedding
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        # Store matched text chunks
        contexts = []

        # Store unique source names
        sources = set()

        # Extract payload data from search results
        for r in results.points:

            # Get metadata payload
            payload = r.payload or {}

            # Get matched text chunk
            text = payload.get(
                "text",
                ""
            )

            # Get source document name/path
            source = payload.get(
                "source",
                ""
            )

            # Add valid matched text to response
            if text:
                contexts.append(text)
                sources.add(source)

        # Return retrieved contexts and sources
        return {
            "contexts": contexts,
            "sources": list(sources)
        }