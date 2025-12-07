import logging
import json
from rag_foundry.components.generators.openai_generator import OpenAIGenerator

logger = logging.getLogger(__name__)

JUDGE_PROMPT_TEMPLATE = """
You are an impartial judge evaluating the quality of a RAG system's answer.

Question: {question}
Ground Truth: {ground_truth}
Retrieved Context: {context}
Model Answer: {answer}

Please evaluate the answer on two criteria:
1. Faithfulness (1-5): Is the answer derived *only* from the Retrieved Context? (5 = fully supported, 1 = hallucinated/contradictory)
2. Correctness (1-5): Is the answer correct relative to the Ground Truth? (5 = perfectly correct, 1 = completely wrong)

Provide a short explanation.

Output valid JSON strictly in this format:
{{
  "faithfulness": <int>,
  "correctness": <int>,
  "reasoning": "<string>"
}}
"""

class LLMJudge:
    def __init__(self, model: str = "gpt-4o-mini"):
        # We reuse the Generator wrapper for consistency
        self.generator = OpenAIGenerator(model=model, temperature=0.0)

    def evaluate(self, question: str, answer: str, context: str, ground_truth: str) -> dict:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            question=question,
            ground_truth=ground_truth,
            context=context,
            answer=answer
        )
        
        try:
            response_text = self.generator.generate(
                system_prompt="You are a helpful judge. Output JSON only.",
                user_prompt=prompt
            )
            # Naive JSON extraction (strip markdown code blocks if present)
            clean_text = response_text.strip()
            if clean_text.startswith("```json"):
                clean_text = clean_text[7:]
            if clean_text.endswith("```"):
                clean_text = clean_text[:-3]
            
            return json.loads(clean_text)
        except Exception as e:
            logger.error(f"Judge evaluation failed: {e}")
            return {"faithfulness": 0, "correctness": 0, "reasoning": "Judge failed"}
