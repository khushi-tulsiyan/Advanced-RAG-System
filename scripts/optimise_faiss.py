import faiss
import numpy as np

VECTOR_STORE_PATH = "../data/vector_store/faiss.index"
OPTIMIZED_VECTOR_STORE_PATH = "../data/vector_store/faiss_optimized.index"

def optimize_faiss():
    """Applies FAISS quantization for memory-efficient retrieval"""
    index = faiss.read_index(VECTOR_STORE_PATH)
    d = index.d  # Vector dimensions

    print(f"Original FAISS index size: {index.ntotal} vectors")

    # Apply Product Quantization (IVFPQ)
    quantizer = faiss.IndexFlatL2(d)
    index_ivfpq = faiss.IndexIVFPQ(quantizer, d, 100, 8, 8)
    
    # Train & add vectors
    index_ivfpq.train(np.random.rand(5000, d).astype(np.float32))  # Sample training data
    index_ivfpq.add(index.reconstruct_n(0, index.ntotal))

    faiss.write_index(index_ivfpq, OPTIMIZED_VECTOR_STORE_PATH)
    print(f"Optimized FAISS index saved at {OPTIMIZED_VECTOR_STORE_PATH}")

if __name__ == "__main__":
    optimize_faiss()
