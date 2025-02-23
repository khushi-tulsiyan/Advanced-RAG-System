import requests
import time

API_URL = "http://127.0.0.1:8000/query"

def test_api():
    """Tests API response time for multiple queries"""
    queries = ["What is deep learning?", "How does RAG work?", "Explain transformers in AI"]
    
    for query in queries:
        start_time = time.time()
        response = requests.post(API_URL, json={"query": query, "top_k": 3})
        elapsed_time = time.time() - start_time
        
        print(f"Query: {query}")
        print(f"Response Time: {elapsed_time:.3f}s")
        print(f"Results: {response.json()['results']}\n")

if __name__ == "__main__":
    test_api()
