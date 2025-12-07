import faiss
import pickle
import numpy as np
import os
from pathlib import Path
from typing import List
from rag_foundry.interfaces import Document, BaseEmbedder
import logging

logger = logging.getLogger(__name__)

class FaissIndexer:
    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

    def build_index(self, documents: List[Document], output_path: str):
        if not documents:
            logger.warning("No documents to index.")
            return

        texts = [doc.content for doc in documents]
        logger.info(f"Embedding {len(texts)} documents...")
        embeddings = self.embedder.embed_texts(texts)
        dimension = len(embeddings[0])
        
        # Initialize FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(np.array(embeddings).astype('float32'))
        
        # Save index and metadata
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(index, output_path)
        
        # Save metadata (store actual text and metadata mapped by index ID)
        metadata_path = output_path + ".meta"
        with open(metadata_path, "wb") as f:
            pickle.dump(documents, f)
            
        logger.info(f"Index saved to {output_path}")

    @staticmethod
    def load(index_path: str):
        # Implementation moved to FAISSRetriever to keep Indexer focused on building
        pass
