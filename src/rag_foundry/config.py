from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

class EmbeddingConfig(BaseModel):
    provider: str = "sentence-transformers"
    model_name: str = "all-MiniLM-L6-v2"

class RetrieverConfig(BaseModel):
    type: str
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    index_path: str = "storage/default_index"
    top_k: int = 5
    options: Dict[str, Any] = Field(default_factory=dict)

class GeneratorConfig(BaseModel):
    type: str
    model: str
    max_tokens: int = 512
    temperature: float = 0.0
    options: Dict[str, Any] = Field(default_factory=dict)

class PipelineConfig(BaseModel):
    pipeline_id: str
    retriever: RetrieverConfig
    generator: GeneratorConfig
    prompt_template: str

class Settings(BaseSettings):
    """Global application settings."""
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    rag_foundry_env: str = "dev"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
