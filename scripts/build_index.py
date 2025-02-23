import os
import json
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR = "../data/processed_chunks"
VECTOR_STORE_PATH = "../data/vector_store/faiss.index"
TEXT_STORE_PATH = "../data/vector_store/chunk_texts.json"

def build_bm25_index(text_chunks):
    """Builds a BM25 index from chunked text"""
    tokenized_chunks = [chunk.split() for chunk in text_chunks]
    return BM25Okapi(tokenized_chunks)

def build_faiss_index(text_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    """Builds a FAISS index from chunked text embeddings"""
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_chunks, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return index, embeddings

def process_documents():
    """Loads text chunks, builds BM25 & FAISS indexes, and saves them"""
    all_chunks = []
    
    # Load chunked text files
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):
            with open(os.path.join(DATA_DIR, filename), "r", encoding="utf-8") as f:
                all_chunks.extend(json.load(f))

    # Build indexes
    print("Building BM25 index...")
    bm25 = build_bm25_index(all_chunks)
    
    print("Building FAISS index...")
    faiss_index, _ = build_faiss_index(all_chunks)

    # Save FAISS index
    faiss.write_index(faiss_index, VECTOR_STORE_PATH)
    
    # Save text chunks for retrieval
    with open(TEXT_STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=4)
    
    print(f"Indexes built & saved! ({len(all_chunks)} documents indexed)")

if __name__ == "__main__":
    process_documents()
