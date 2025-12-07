import os
import logging
from typing import Optional
from openai import OpenAI
from rag_foundry.interfaces import BaseGenerator
from rag_foundry.registry import register_generator
from rag_foundry.config import settings

logger = logging.getLogger(__name__)

@register_generator("openai_chat")
class OpenAIGenerator(BaseGenerator):
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0, max_tokens: int = 512, **kwargs):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = kwargs.get("api_key") or settings.openai_api_key
        
        if not self.api_key:
            # Check env var as fallback if not in settings yet
            self.api_key = os.getenv("OPENAI_API_KEY") 
        
        if not self.api_key:
            logger.warning("OPENAI_API_KEY not set. OpenAIGenerator will list fail if called.")

        self.client = OpenAI(api_key=self.api_key)

    def generate(self, system_prompt: str, user_prompt: str, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI generation failed: {e}")
            raise
