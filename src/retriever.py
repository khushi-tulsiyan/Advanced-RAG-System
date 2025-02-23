import os
import faiss
import numpy as np
import json
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Tuple

class HybridRetriever:
    def __init__(self, vector_model: str = "msmarco-distilbert-base-v4"):
        """Initializes the hybrid retriever with FAISS (dense) and BM25 (sparse)."""
        self.vector_model = SentenceTransformer(vector_model)
        self.bm25 = None
        self.index = None
        self.documents = []
    
    def build_bm25(self, processed_chunks_dir: str):
        """Builds a BM25 index from processed text chunks."""
        corpus = []
        self.documents = []
        
        for filename in os.listdir(processed_chunks_dir):
            if filename.endswith(".json"):
                with open(os.path.join(processed_chunks_dir, filename), "r", encoding="utf-8") as f:
                    chunks = json.load(f)
                    corpus.extend(chunks)
                    self.documents.extend(chunks)
        
        tokenized_corpus = [doc.split() for doc in corpus]
        self.bm25 = BM25Okapi(tokenized_corpus)
        print(f"BM25 index built with {len(corpus)} chunks.")
    
    def build_faiss(self, vector_store_dir: str):
        """Builds a FAISS index from processed text chunks."""
        embeddings = self.vector_model.encode(self.documents, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)
        faiss.write_index(self.index, os.path.join(vector_store_dir, "faiss.index"))
        print(f"FAISS index built and stored in {vector_store_dir}.")
    
    def hybrid_search(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Performs hybrid retrieval using both BM25 and FAISS."""
        # BM25 retrieval
        bm25_scores = self.bm25.get_scores(query.split())
        bm25_top_k = np.argsort(bm25_scores)[-top_k:][::-1]
        
        # Dense retrieval via FAISS
        query_embedding = self.vector_model.encode([query], convert_to_numpy=True)
        _, faiss_top_k = self.index.search(query_embedding, top_k)
        
        # Merge results
        hybrid_results = {}
        for idx in bm25_top_k:
            hybrid_results[self.documents[idx]] = bm25_scores[idx]
        for idx in faiss_top_k[0]:
            hybrid_results[self.documents[idx]] = hybrid_results.get(self.documents[idx], 0) + 1
        
        return sorted(hybrid_results.items(), key=lambda x: x[1], reverse=True)[:top_k]

# Example usage
if __name__ == "__main__":
    retriever = HybridRetriever()
    retriever.build_bm25("data/processed_chunks")
    retriever.build_faiss("data/vector_store")
    query = "How do transformers work in NLP?"
    results = retriever.hybrid_search(query)
    for doc, score in results:
        print(f"Score: {score:.4f} | Doc: {doc}")
