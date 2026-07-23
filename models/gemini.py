"""Google Gemini wrapper (LLM #1).

Uses the current `google-genai` SDK. The older `google-generativeai`
package is deprecated and its models return 404, so this is not optional.

    pip install google-genai
"""

import os
from typing import Any

from models.base import BaseModel, Reply

try:
    from google import genai
except ImportError:  # library not installed yet
    genai = None

# Gemini model IDs change often and retired ones return a 404.
# Override with GEMINI_MODEL in .env when this one is eventually replaced.
DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiModel(BaseModel):
    name = "gemini"
    label = "Gemini"

    def __init__(self) -> None:
        self.api_key: str = os.getenv("GEMINI_API_KEY", "").strip()
        self.model_name: str = os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self._client: Any = None

        if genai is not None and self.api_key:
            self._client = genai.Client(api_key=self.api_key)

    @property
    def available(self) -> bool:
        return self._client is not None

    def generate(self, prompt: str) -> Reply:
        if genai is None:
            return self.fail(
                "The google-genai package is not installed. "
                "Run: pip install google-genai"
            )
        if not self.available:
            return self.fail("Gemini is not configured. Add GEMINI_API_KEY to .env.")

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        except Exception as exc:
            message = str(exc)
            # A 404 nearly always means the model ID has been retired.
            if "404" in message or "not found" in message.lower():
                return self.fail(
                    f"The model '{self.model_name}' was not found — Google has "
                    "probably retired it. Set GEMINI_MODEL in your .env to a "
                    "current one and restart the server. Current list: "
                    "https://ai.google.dev/gemini-api/docs/models"
                )
            return self.fail(f"Gemini request failed: {exc}")

        text = getattr(response, "text", "") or ""
        if not text.strip():
            return self.fail("Gemini returned an empty response. Try rewording.")
        return self.ok(text)
