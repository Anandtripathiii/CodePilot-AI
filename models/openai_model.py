"""OpenAI wrapper (LLM #2)."""

import os
from typing import Any

from models.base import BaseModel, Reply

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class OpenAIModel(BaseModel):
    name = "openai"
    label = "OpenAI"

    def __init__(self) -> None:
        self.api_key: str = os.getenv("OPENAI_API_KEY", "").strip()
        self.model_name: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._client: Any = None

        if OpenAI is not None and self.api_key:
            self._client = OpenAI(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str) -> Reply:
        if not self.available:
            return self.fail("OpenAI is not configured. Add OPENAI_API_KEY to .env.")

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1400,
            )
        except Exception as exc:
            return self.fail(f"OpenAI request failed: {exc}")

        text = response.choices[0].message.content or ""
        if not text.strip():
            return self.fail("OpenAI returned an empty response. Try rewording.")
        return self.ok(text)
