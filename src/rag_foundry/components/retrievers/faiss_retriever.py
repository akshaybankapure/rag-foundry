import faiss
import pickle
import numpy as np
from typing import List, Optional
from pathlib import Path
from rag_foundry.interfaces import BaseRetriever, RetrievalResult, Document, BaseEmbedder
from rag_foundry.registry import register_retriever
from rag_foundry.components.embeddings import SentenceTransformerEmbedder
import logging

logger = logging.getLogger(__name__)

@register_retriever("faiss")
class FaissRetriever(BaseRetriever):
    def __init__(
        self,
        index_path: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        top_k: int = 5,
        **kwargs
    ):
        self.index_path = index_path
        self.top_k = top_k
        self.embedder: BaseEmbedder = SentenceTransformerEmbedder(embedding_model)
        
        if not Path(index_path).exists():
            raise FileNotFoundError(f"Index file not found: {index_path}")
            
        self.index = faiss.read_index(index_path)
        
        meta_path = index_path + ".meta"
        if not Path(meta_path).exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")
            
        with open(meta_path, "rb") as f:
            self.documents: List[Document] = pickle.load(f)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        k = top_k or self.top_k
        query_vector = self.embedder.embed_query(query)
        query_np = np.array([query_vector]).astype('float32')
        
        distances, indices = self.index.search(query_np, k)
        
        results = []
        for i, idx in enumerate(indices[0]):
            if idx == -1:  # No more results
                continue
            doc = self.documents[idx]
            # Convert L2 distance to a similarity-like score (simple inversion for now or raw distance)
            # L2 distance: 0 is identical. 
            score = float(distances[0][i])
            results.append(RetrievalResult(document=doc, score=score))
            
        return results
