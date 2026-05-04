from typing import List
import openai

EMBED_MODEL = "text-embedding-3-small"
BATCH_SIZE = 500

def embed_chunks(chunks: List[str]) -> List[List[float]]:
    client = openai.OpenAI()  # reads OPENAI_API_KEY from env
    all_embeddings = []
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        response = client.embeddings.create(model=EMBED_MODEL, input=batch)
        batch_embeddings = [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        all_embeddings.extend(batch_embeddings)
    return all_embeddings