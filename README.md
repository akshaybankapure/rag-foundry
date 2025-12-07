# RAG Foundry 🏭

> **A modular, extensbile, CI-friendly RAG evaluation and experimentation framework.**

RAG Foundry is an open-source tool designed to help engineering teams build, measure, and improve Retrieval Augmented Generation (RAG) pipelines. It provides a declarative way to define pipelines, a rigorous evaluation system using "LLM-as-a-Judge", and a web dashboard to inspect results.

## 🚀 Key Features

*   **Declarative Pipelines**: Define your entire RAG flow (retriever, prompt, model) in simple YAML.
*   **Modular Architecture**: Plugin-based design for Retrievers, Generators, and Embeddings. Easily swap FAISS for Chroma, or OpenAI for Anthropic.
*   **LLM-as-a-Judge**: Built-in evaluator to score answers on **Faithfulness** and **Correctness**.
*   **Quantitative Metrics**: Track Hit Rate, MRR, Latency, and Cost.
*   **CI/CD Integration**: Run regression tests on every PR to prevent performance degradation.
*   **Web Dashboard**: Visual UI to inspect answers and retrieved contexts side-by-side.

## 📦 Installation

Requires Python 3.11+.

```bash
pip install -e .
```

## ⚡ Quick Start

### 1. Configure Environment
Create a `.env` file:
```bash
OPENAI_API_KEY=sk-...
```

### 2. Ingest Data
Ingest a directory of documents into a FAISS index:
```bash
rag-foundry ingest corpora/sample/ --output-index storage/index.faiss
```

### 3. Run an Experiment
Run a comparison experiment using a pipeline config:
```bash
rag-foundry run configs/experiments/sample_experiment.yaml
```

### 4. View Results
Start the dashboard to view the evaluation report:
```bash
rag-foundry serve
```
Visit `http://localhost:8000` to see your results.

## 🧠 Architecture

RAG Foundry is built on a plugin system:
- **Registry**: `src/rag_foundry/registry.py` handles component registration.
- **Pipeline Runner**: `src/rag_foundry/pipeline.py` orchestrates the flow.
- **Evaluation**: `src/rag_foundry/evaluation/runner.py` executes experiments.

## 🛠 Extensibility

Adding a new component is easy! Just use the decorators:

```python
from rag_foundry.interfaces import BaseRetriever
from rag_foundry.registry import register_retriever

@register_retriever("my_custom_retriever")
class MyRetriever(BaseRetriever):
    def retrieve(self, query: str, top_k: int = 5):
        # Your logic here
        pass
```

Then use `type: "my_custom_retriever"` in your pipeline YAML.

## 🤝 Contributing
See [CONTRIBUTING.md](CONTRIBUTING.md) for details on how to submit PRs and run tests.

## 📄 License
MIT
