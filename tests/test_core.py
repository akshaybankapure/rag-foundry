import pytest
from rag_foundry.ingest import TextSplitter
from rag_foundry.interfaces import Document

def test_text_splitter():
    text = "Hello world. This is a test document."
    doc = Document(content=text, metadata={})
    
    # Split with very small chunk size to force splitting
    splitter = TextSplitter(chunk_size=10, chunk_overlap=2)
    chunks = splitter.split_documents([doc])
    
    assert len(chunks) > 1
    assert chunks[0].content == "Hello worl"
    
def test_plugin_registry():
    from rag_foundry.registry import register_retriever, get_retriever_class, RETRIEVER_REGISTRY
    
    @register_retriever("test_retriever")
    class TestRetriever:
        pass
        
    assert "test_retriever" in RETRIEVER_REGISTRY
    assert get_retriever_class("test_retriever") == TestRetriever
