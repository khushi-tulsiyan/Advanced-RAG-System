import os
import json
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TextChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        """Initializes the text chunker with specified chunk size and overlap."""
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )

    def chunk_text(self, text: str) -> List[str]:
        """Splits text into overlapping chunks for better retrieval performance."""
        return self.splitter.split_text(text)

    def process_documents(self, input_dir: str, output_dir: str):
        """Processes all text documents in a directory and saves chunked versions."""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for filename in os.listdir(input_dir):
            if filename.endswith(".txt"):
                with open(os.path.join(input_dir, filename), "r", encoding="utf-8") as f:
                    text = f.read()
                
                chunks = self.chunk_text(text)
                output_file = os.path.join(output_dir, filename.replace(".txt", "_chunks.json"))
                
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(chunks, f, indent=4)
                
                print(f"Processed {filename}: {len(chunks)} chunks created.")

# Example usage
if __name__ == "__main__":
    chunker = TextChunker()
    chunker.process_documents("data/raw_docs", "data/processed_chunks")
