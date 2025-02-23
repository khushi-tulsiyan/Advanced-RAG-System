from transformers import pipeline
from typing import List, Tuple

class CrossEncoderReranker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """Initializes the cross-encoder reranker."""
        self.reranker = pipeline("text-classification", model=model_name)
    
    def rerank(self, query: str, retrieved_docs: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
        """Reranks retrieved documents using a cross-encoder."""
        inputs = [f"{query} [SEP] {doc[0]}" for doc in retrieved_docs]
        scores = self.reranker(inputs)
        
        reranked_results = sorted(
            zip([doc[0] for doc in retrieved_docs], [score['score'] for score in scores]),
            key=lambda x: x[1],
            reverse=True
        )
        return reranked_results

# Example usage
if __name__ == "__main__":
    reranker = CrossEncoderReranker()
    query = "How do transformers work in NLP?"
    retrieved_docs = [
        ("Transformers revolutionized NLP with self-attention mechanisms.", 1.2),
        ("FAISS is a library for similarity search of dense vectors.", 0.8),
        ("BM25 is used to estimate document relevance in search engines.", 0.9)
    ]
    reranked_results = reranker.rerank(query, retrieved_docs)
    for doc, score in reranked_results:
        print(f"Score: {score:.4f} | Doc: {doc}")
