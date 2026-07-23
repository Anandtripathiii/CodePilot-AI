"""Google Gemini wrapper (LLM #1).

Uses the current `google-genai` SDK. The older `google-generativeai`
package is deprecated and its models return 404, so this is not optional.
"""

import os
from typing import Any

from models.base import BaseModel, Reply

try:
    from google import genai
except ImportError:  # library not installed yet
    genai = None

# Gemini model IDs change often and retired ones return a 404.
DEFAULT_MODEL = "gemini-3.6-flash"


class GeminiModel(BaseModel):
    name = "gemini"
    label = "Gemini"

    def __init__(self, api_key: str = "", model_name: str = "") -> None:
        # An explicit key wins; otherwise fall back to the server's own.
        self.api_key: str = (api_key or os.getenv("GEMINI_API_KEY", "")).strip()
        self.model_name: str = (
            model_name or os.getenv("GEMINI_MODEL", "") or DEFAULT_MODEL
        ).strip()
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
            return self.fail(
                "No Gemini key. Add one with the 'API keys' button — it is "
                "stored in your browser only."
            )

        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )
        except Exception as exc:
            message = str(exc)
            if "404" in message or "not found" in message.lower():
                return self.fail(
                    f"The model '{self.model_name}' was not found — Google has "
                    "probably retired it. Pick a current one from "
                    "https://ai.google.dev/gemini-api/docs/models"
                )
            if "API key" in message or "401" in message or "403" in message:
                return self.fail("Gemini rejected that key. Check it and try again.")
            return self.fail(f"Gemini request failed: {exc}")

        text = getattr(response, "text", "") or ""
        if not text.strip():
            return self.fail("Gemini returned an empty response. Try rewording.")
        return self.ok(text)
