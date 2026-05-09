from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct
)


class QdrantStorage:

    def __init__(
        self,
        url="http://localhost:6333",
        collection="docs",
        dim=384
    ):

        self.client = QdrantClient(
            url=url,
            timeout=30
        )

        self.collection = collection

        # Create collection if not exists
        if not self.client.collection_exists(
            collection_name=self.collection
        ):

            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(
                    size=dim,
                    distance=Distance.COSINE
                )
            )

    # Insert vectors
    def upsert(
        self,
        ids,
        vectors,
        payloads
    ):

        points = []

        for i in range(len(ids)):

            points.append(
                PointStruct(
                    id=ids[i],
                    vector=vectors[i],
                    payload=payloads[i]
                )
            )

        self.client.upsert(
            collection_name=self.collection,
            points=points
        )

    # Search vectors
    def search(
        self,
        query_vector,
        top_k=5
    ):

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k,
            with_payload=True
        )

        contexts = []
        sources = set()

        for r in results.points:

            payload = r.payload or {}

            text = payload.get(
                "text",
                ""
            )

            source = payload.get(
                "source",
                ""
            )

            if text:
                contexts.append(text)
                sources.add(source)

        return {
            "contexts": contexts,
            "sources": list(sources)
        }