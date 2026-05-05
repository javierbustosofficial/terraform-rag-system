from dotenv import load_dotenv
load_dotenv()

from typing import List
import openai
from sentence_transformers import CrossEncoder
from terraform_rag.ingestion.embedder import embed_chunks
from terraform_rag.retrieval.vector_store import query as vs_query

# Tunable constants
SEMANTIC_WEIGHT = 0.7
KEYWORD_WEIGHT = 0.3
CANDIDATE_POOL = 30       # fetched from ChromaDB
HYBRID_KEEP = 15          # passed to cross-encoder after hybrid re-score
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_cross_encoder = None  # lazy-loaded on first use

def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    return _cross_encoder


def _rewrite_query(query_text: str) -> str:
    client = openai.OpenAI()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a search query optimiser for Terraform documentation. "
                    "Rewrite the user's question into a clear, specific query that will "
                    "retrieve the most relevant documentation. Return only the rewritten "
                    "query, nothing else."
                ),
            },
            {"role": "user", "content": query_text},
        ],
        temperature=0,
    )
    return resp.choices[0].message.content.strip()


def _keyword_score(text: str, terms: List[str]) -> float:
    text_lower = text.lower()
    return sum(1 for t in terms if t in text_lower) / len(terms) if terms else 0.0


def retrieve(
    query_text: str,
    n_results: int = 5,
    section: str | None = None,
) -> List[dict]:
    # Stage 1: rewrite query
    rewritten = _rewrite_query(query_text)

    # Stage 2: embed rewritten query
    embedding = embed_chunks([rewritten])[0]

    # Stage 3: vector search
    where = {"section": section} if section else None
    raw = vs_query(embedding, n_results=CANDIDATE_POOL, where=where)

    # Stage 4: hybrid keyword re-score, keep top HYBRID_KEEP
    terms = rewritten.lower().split()
    candidates = []
    for doc, meta, dist in zip(
        raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        sem = 1 - dist
        kw = _keyword_score(doc, terms)
        blended = SEMANTIC_WEIGHT * sem + KEYWORD_WEIGHT * kw
        candidates.append({"text": doc, "metadata": meta, "score": blended})

    candidates.sort(key=lambda x: x["score"], reverse=True)
    candidates = candidates[:HYBRID_KEEP]

    # Stage 5: cross-encoder re-rank
    ce = _get_cross_encoder()
    pairs = [[query_text, c["text"]] for c in candidates]  # use original query for CE
    ce_scores = ce.predict(pairs)

    for candidate, ce_score in zip(candidates, ce_scores):
        candidate["score"] = float(ce_score)

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:n_results]