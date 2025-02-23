import os
import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from transformers import pipeline

class RAGPipeline:
    def __init__(self, chunk_folder="../data/processed_chunks", vector_store="../data/vector_store/faiss.index"):
        """Initialize retrieval & reranking pipeline"""
        self.chunk_folder = chunk_folder
        self.vector_store = vector_store
        
        # Load FAISS index
        self.embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.index, self.chunk_texts = self.load_faiss_index()
        
        # Load BM25 index
        self.bm25, self.chunk_texts_bm25 = self.build_bm25_index()
        
        # Load reranker model
        self.reranker = pipeline("text-classification", model="cross-encoder/ms-marco-MiniLM-L-6-v2")

    def load_faiss_index(self):
        """Loads FAISS vector store and corresponding texts"""
        index = faiss.read_index(self.vector_store)
        with open(self.chunk_folder + "/chunk_texts.json", "r", encoding="utf-8") as f:
            chunk_texts = json.load(f)
        return index, chunk_texts

    def build_bm25_index(self):
        """Builds BM25 index from chunked documents"""
        texts = []
        for filename in os.listdir(self.chunk_folder):
            if filename.endswith(".json"):
                with open(os.path.join(self.chunk_folder, filename), "r", encoding="utf-8") as f:
                    texts.extend(json.load(f))
        
        tokenized_texts = [text.split() for text in texts]
        return BM25Okapi(tokenized_texts), texts

    def hybrid_search(self, query, top_k=10):
        """Performs hybrid retrieval using FAISS + BM25"""
        # BM25 search
        bm25_scores = self.bm25.get_scores(query.split())
        top_bm25_indices = np.argsort(bm25_scores)[-top_k:]

        # FAISS search
        query_embedding = self.embedding_model.encode([query])
        _, top_faiss_indices = self.index.search(query_embedding, top_k)
        top_faiss_indices = top_faiss_indices[0]

        # Merge results
        combined_indices = list(set(top_bm25_indices) | set(top_faiss_indices))
        retrieved_docs = [self.chunk_texts[idx] for idx in combined_indices]
        return retrieved_docs

    def rerank_results(self, query, retrieved_docs):
        """Reranks retrieved documents using cross-encoder"""
        pairs = [[query, doc] for doc in retrieved_docs]
        scores = self.reranker(pairs)
        ranked_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1]['score'], reverse=True)
        return [doc for doc, _ in ranked_docs]

    def retrieve_answer(self, query, top_k=5):
        """Main function: Retrieves & reranks the best document"""
        retrieved_docs = self.hybrid_search(query, top_k)
        ranked_docs = self.rerank_results(query, retrieved_docs)
        return ranked_docs[:top_k]

# Example usage
if __name__ == "__main__":
    rag = RAGPipeline()
    query = "What is deep learning?"
    results = rag.retrieve_answer(query)
    for idx, res in enumerate(results):
        print(f"{idx+1}. {res}\n")
