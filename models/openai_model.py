"""OpenAI wrapper (LLM #2)."""

import os
from typing import Any

from models.base import BaseModel, Reply

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIModel(BaseModel):
    name = "openai"
    label = "OpenAI"

    def __init__(self, api_key: str = "", model_name: str = "") -> None:
        self.api_key: str = (api_key or os.getenv("OPENAI_API_KEY", "")).strip()
        self.model_name: str = (
            model_name or os.getenv("OPENAI_MODEL", "") or DEFAULT_MODEL
        ).strip()
        self._client: Any = None

        if OpenAI is not None and self.api_key:
            self._client = OpenAI(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str) -> Reply:
        if OpenAI is None:
            return self.fail("The openai package is not installed.")
        if not self.available:
            return self.fail(
                "No OpenAI key. Add one with the 'API keys' button — it is "
                "stored in your browser only."
            )

        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1400,
            )
        except Exception as exc:
            message = str(exc)
            if "401" in message or "invalid_api_key" in message:
                return self.fail("OpenAI rejected that key. Check it and try again.")
            return self.fail(f"OpenAI request failed: {exc}")

        text = response.choices[0].message.content or ""
        if not text.strip():
            return self.fail("OpenAI returned an empty response. Try rewording.")
        return self.ok(text)
