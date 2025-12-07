from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from dataclasses import dataclass

@dataclass
class Document:
    """Represents a chunk of text with metadata."""
    content: str
    metadata: Dict[str, Any]

@dataclass
class RetrievalResult:
    """Result from a retriever."""
    document: Document
    score: float

class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """Retrieve documents relevant to the query."""
        pass

class BaseGenerator(ABC):
    """Abstract base class for LLM generators."""
    
    @abstractmethod
    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        """Generate a response given prompts."""
        pass

class BaseEmbedder(ABC):
    """Abstract base class for embedding models."""
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        pass
    
    @abstractmethod
    def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        pass
