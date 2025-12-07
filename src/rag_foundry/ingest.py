import os
from typing import List, Generator
from pathlib import Path
from rag_foundry.interfaces import Document

class DocumentLoader:
    def load_directory(self, path: str) -> List[Document]:
        documents = []
        path_obj = Path(path)
        if not path_obj.exists():
            raise FileNotFoundError(f"Directory not found: {path}")

        for root, _, files in os.walk(path):
            for file in files:
                if file.endswith((".md", ".txt")):
                    full_path = Path(root) / file
                    with open(full_path, "r", encoding="utf-8") as f:
                        text = f.read()
                        documents.append(
                            Document(
                                content=text,
                                metadata={"source": str(full_path), "filename": file}
                            )
                        )
        return documents

class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        chunks = []
        for doc in documents:
            text = doc.content
            # Simple character-based splitting for now
            # TODO: Improve with tokenizer-aware or recursive splitting
            start = 0
            while start < len(text):
                end = min(start + self.chunk_size, len(text))
                chunk_text = text[start:end]
                chunks.append(
                    Document(
                        content=chunk_text,
                        metadata={**doc.metadata, "start_char": start, "end_char": end}
                    )
                )
                start += self.chunk_size - self.chunk_overlap
        return chunks
