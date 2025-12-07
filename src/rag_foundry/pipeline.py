import time
from typing import Dict, Any, List
from jinja2 import Template
import logging

from rag_foundry.config import PipelineConfig
from rag_foundry.interfaces import RetrievalResult
from rag_foundry.registry import get_retriever_class, get_generator_class

logger = logging.getLogger(__name__)

class RAGPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        
        # Initialize Retriever
        retriever_cls = get_retriever_class(config.retriever.type)
        self.retriever = retriever_cls(
            index_path=config.retriever.index_path,
            embedding_model=config.retriever.embedding.model_name,
            top_k=config.retriever.top_k,
            **config.retriever.options
        )
        
        # Initialize Generator
        generator_cls = get_generator_class(config.generator.type)
        self.generator = generator_cls(
            model=config.generator.model,
            max_tokens=config.generator.max_tokens,
            temperature=config.generator.temperature,
            **config.generator.options
        )
        
        self.prompt_template = Template(config.prompt_template)

    def run(self, question: str) -> Dict[str, Any]:
        """
        Run the full RAG pipeline for a single question.
        Returns a dictionary with answer, retrieved contexts, and timing info.
        """
        start_time = time.time()
        
        # 1. Retrieve
        t0 = time.time()
        retrieval_results: List[RetrievalResult] = self.retriever.retrieve(question)
        retrieval_latency = time.time() - t0
        
        # 2. Format Context
        context_text = "\n\n".join([r.document.content for r in retrieval_results])
        
        # 3. Generate
        t0 = time.time()
        user_prompt = self.prompt_template.render(context=context_text, question=question)
        system_prompt = "You are a helpful assistant." # Can be configurable too
        
        answer = self.generator.generate(system_prompt=system_prompt, user_prompt=user_prompt)
        generation_latency = time.time() - t0
        
        total_latency = time.time() - start_time
        
        return {
            "pipeline_id": self.config.pipeline_id,
            "question": question,
            "answer": answer,
            "retrieved_docs": [
                {"content": r.document.content, "metadata": r.document.metadata, "score": r.score}
                for r in retrieval_results
            ],
            "metrics": {
                "retrieval_latency_ms": round(retrieval_latency * 1000, 2),
                "generation_latency_ms": round(generation_latency * 1000, 2),
                "total_latency_ms": round(total_latency * 1000, 2)
            }
        }
