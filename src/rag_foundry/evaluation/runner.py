import json
import logging
import yaml
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from rag_foundry.config import PipelineConfig
from rag_foundry.pipeline import RAGPipeline
from rag_foundry.evaluation.judge import LLMJudge

logger = logging.getLogger(__name__)

class ExperimentRunner:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.exp_config = yaml.safe_load(f)
        
        self.dataset_path = self.exp_config["dataset"]
        self.pipelines_ids = self.exp_config["pipelines"]
        self.reports_dir = Path("reports") / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        self.judge = LLMJudge()

    def load_dataset(self) -> List[Dict]:
        dataset = []
        path = Path(self.dataset_path)
        if path.suffix == ".jsonl":
            with open(path, "r") as f:
                for line in f:
                    dataset.append(json.loads(line))
        elif path.suffix == ".json":
            with open(path, "r") as f:
                dataset = json.load(f)
        return dataset

    def load_pipelines(self) -> List[RAGPipeline]:
        pipelines = []
        for pid in self.pipelines_ids:
            # Assume pipeline configs are stored in a standard location or passed in config
            # For simplicity, let's assume `configs/pipelines/{pid}.yaml`
            p_path = Path("configs/pipelines") / f"{pid}.yaml"
            if not p_path.exists():
                raise FileNotFoundError(f"Pipeline config not found: {p_path}")
                
            with open(p_path, "r") as f:
                p_data = yaml.safe_load(f)
                # Inject ID if missing
                if "pipeline_id" not in p_data:
                    p_data["pipeline_id"] = pid
                
                config = PipelineConfig(**p_data)
                pipelines.append(RAGPipeline(config))
        return pipelines

    def run(self):
        dataset = self.load_dataset()
        pipelines = self.load_pipelines()
        
        results = {} # pipeline_id -> list of result objects
        
        for pipeline in pipelines:
            logger.info(f"Running pipeline: {pipeline.config.pipeline_id}")
            p_results = []
            
            for item in dataset:
                q = item["question"]
                gt = item.get("ground_truth", "")
                
                # Run Pipeline
                output = pipeline.run(q)
                
                # Run Judge
                context_str = "\n".join([d["content"] for d in output["retrieved_docs"]])
                scores = self.judge.evaluate(q, output["answer"], context_str, gt)
                
                result_entry = {
                    **output,
                    "ground_truth": gt,
                    "judge_scores": scores
                }
                p_results.append(result_entry)
            
            results[pipeline.config.pipeline_id] = p_results

        self.save_reports(results)

    def save_reports(self, results: Dict[str, List[Any]]):
        # 1. Full JSON
        with open(self.reports_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
            
        # 2. Markdown Summary
        summary_lines = ["# Experiment Results\n"]
        for pid, res_list in results.items():
            avg_faith = sum(r["judge_scores"]["faithfulness"] for r in res_list) / len(res_list)
            avg_corr = sum(r["judge_scores"]["correctness"] for r in res_list) / len(res_list)
            avg_lat = sum(r["metrics"]["total_latency_ms"] for r in res_list) / len(res_list)
            
            summary_lines.append(f"## Pipeline: {pid}")
            summary_lines.append(f"- **Faithfulness**: {avg_faith:.2f}/5")
            summary_lines.append(f"- **Correctness**: {avg_corr:.2f}/5")
            summary_lines.append(f"- **Avg Latency**: {avg_lat:.2f}ms")
            summary_lines.append("")
            
        with open(self.reports_dir / "summary.md", "w") as f:
            f.write("\n".join(summary_lines))
        
        logger.info(f"Reports saved to {self.reports_dir}")
