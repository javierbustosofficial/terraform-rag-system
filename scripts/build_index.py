from dotenv import load_dotenv
load_dotenv()

from terraform_rag.ingestion.loader import load_all
from terraform_rag.ingestion.chunker import markdown_document_chunking
from terraform_rag.ingestion.embedder import embed_chunks
from terraform_rag.retrieval.vector_store import upsert_chunks

def build_index():
    docs = load_all()
    all_ids, all_texts, all_metas = [], [], []

    for doc in docs:
        chunks = markdown_document_chunking(doc.content)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc.metadata['source_path']}::chunk_{i}"
            meta = {**doc.metadata, "chunk_index": i}
            all_ids.append(chunk_id)
            all_texts.append(chunk)
            all_metas.append(meta)

    print(f"Embedding {len(all_texts)} chunks...")
    embeddings = embed_chunks(all_texts)

    upsert_chunks(all_ids, all_texts, embeddings, all_metas)
    print(f"Stored {len(all_ids)} chunks in ChromaDB.")

if __name__ == "__main__":
    build_index()