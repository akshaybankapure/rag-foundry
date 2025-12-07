import typer
import logging
import uvicorn
from pathlib import Path
from rag_foundry.ingest import DocumentLoader, TextSplitter
from rag_foundry.components.embeddings import SentenceTransformerEmbedder
from rag_foundry.indexing import FaissIndexer
from rag_foundry.evaluation.runner import ExperimentRunner

app = typer.Typer(help="RAG Foundry CLI")

@app.command()
def ingest(
    input_dir: str = typer.Argument(..., help="Directory containing corpus files"),
    output_index: str = typer.Option("storage/index.faiss", help="Output path for FAISS index"),
    chunk_size: int = typer.Option(500, help="Chunk size in characters"),
    embedding_model: str = typer.Option("all-MiniLM-L6-v2", help="Embedding model name")
):
    """Ingest documents and build a vector index."""
    typer.echo(f"Loading documents from {input_dir}...")
    loader = DocumentLoader()
    docs = loader.load_directory(input_dir)
    typer.echo(f"Loaded {len(docs)} documents.")
    
    splitter = TextSplitter(chunk_size=chunk_size)
    chunks = splitter.split_documents(docs)
    typer.echo(f"Created {len(chunks)} chunks.")
    
    embedder = SentenceTransformerEmbedder(model_name=embedding_model)
    indexer = FaissIndexer(embedder)
    indexer.build_index(chunks, output_index)
    typer.echo("Ingestion complete.")

@app.command()
def run(
    config: str = typer.Argument(..., help="Path to experiment config YAML")
):
    """Run an evaluation experiment."""
    runner = ExperimentRunner(config)
    runner.run()
    typer.echo("Experiment complete.")

@app.command()
def serve(
    host: str = "0.0.0.0",
    port: int = 8000
):
    """Start the Web UI server."""
    typer.echo(f"Starting server at http://{host}:{port}")
    uvicorn.run("rag_foundry.server:app", host=host, port=port, reload=True)

if __name__ == "__main__":
    app()
