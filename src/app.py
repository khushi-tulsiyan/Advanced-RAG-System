from fastapi import FastAPI, Query
from pydantic import BaseModel
from pipeline import RAGPipeline

# Initialize API & RAG System
app = FastAPI(title="Advanced RAG System", version="1.0")
rag_pipeline = RAGPipeline()

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

@app.get("/")
def home():
    return {"message": "Welcome to the Advanced RAG System!"}

@app.post("/query")
async def retrieve_answer(request: QueryRequest):
    """Retrieves top-k relevant documents for a given query"""
    results = rag_pipeline.retrieve_answer(request.query, request.top_k)
    return {"query": request.query, "results": results}

# Run API: `uvicorn src.app:app --reload`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
