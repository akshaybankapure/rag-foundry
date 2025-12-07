from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
import json
import os

app = FastAPI(title="RAG Foundry UI")

# Serve static files for the frontend
# We assume the UI index.html is in src/rag_foundry/ui/
UI_DIR = Path(__file__).parent / "ui"
REPORTS_DIR = Path("reports")

app.mount("/static", StaticFiles(directory=UI_DIR), name="static")

@app.get("/")
async def read_index():
    return FileResponse(UI_DIR / "index.html")

@app.get("/api/experiments")
def list_experiments():
    if not REPORTS_DIR.exists():
        return []
    
    experiments = []
    for d in REPORTS_DIR.iterdir():
        if d.is_dir():
            summary_path = d / "summary.md"
            if summary_path.exists():
                experiments.append({
                    "id": d.name,
                    "timestamp": d.name.split("_")[0], # simplified
                    "path": str(d)
                })
    return sorted(experiments, key=lambda x: x["id"], reverse=True)

@app.get("/api/experiments/{experiment_id}")
def get_experiment_details(experiment_id: str):
    path = REPORTS_DIR / experiment_id / "results.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    
    with open(path, "r") as f:
        return json.load(f)
