from typing import Dict, Type, Any, Callable
import logging

logger = logging.getLogger(__name__)

# Registry dictionaries
RETRIEVER_REGISTRY: Dict[str, Type] = {}
GENERATOR_REGISTRY: Dict[str, Type] = {}
METRIC_REGISTRY: Dict[str, Callable] = {}

def register_retriever(name: str):
    """Decorator to register a retriever class."""
    def decorator(cls):
        RETRIEVER_REGISTRY[name] = cls
        logger.debug(f"Registered retriever: {name} -> {cls.__name__}")
        return cls
    return decorator

def register_generator(name: str):
    """Decorator to register a generator class."""
    def decorator(cls):
        GENERATOR_REGISTRY[name] = cls
        logger.debug(f"Registered generator: {name} -> {cls.__name__}")
        return cls
    return decorator

def register_metric(name: str):
    """Decorator to register a metric function."""
    def decorator(func):
        METRIC_REGISTRY[name] = func
        logger.debug(f"Registered metric: {name} -> {func.__name__}")
        return func
    return decorator

def get_retriever_class(name: str) -> Type:
    if name not in RETRIEVER_REGISTRY:
        raise ValueError(f"Retriever '{name}' not found. Available: {list(RETRIEVER_REGISTRY.keys())}")
    return RETRIEVER_REGISTRY[name]

def get_generator_class(name: str) -> Type:
    if name not in GENERATOR_REGISTRY:
        raise ValueError(f"Generator '{name}' not found. Available: {list(GENERATOR_REGISTRY.keys())}")
    return GENERATOR_REGISTRY[name]
