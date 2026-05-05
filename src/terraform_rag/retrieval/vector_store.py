import chromadb
from pathlib import Path
from typing import List

CHROMA_PATH = Path(__file__).parents[3] / "data" / "chroma"
COLLECTION_NAME = "terraform_docs"

def _get_collection():
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

def upsert_chunks(
    ids: List[str],
    chunks: List[str],
    embeddings: List[List[float]],
    metadatas: List[dict],
) -> None:
    _get_collection().upsert(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

def query(embedding: List[float], n_results: int = 5, where: dict | None = None) -> dict:
    kwargs = dict(
        query_embeddings=[embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    if where:
        kwargs["where"] = where
    return _get_collection().query(**kwargs)