"""The contract every LLM wrapper implements.

Each model returns the same shape so app.py never needs to know which
provider it is talking to.
"""

from typing import TypedDict


class Reply(TypedDict):
    """One model response. `error` is None on success, a message on failure."""

    text: str
    error: str | None


class BaseModel:
    name: str = "base"
    label: str = "Base"

    @property
    def available(self) -> bool:
        """True when an API key is present and the client loaded."""
        return False

    def generate(self, prompt: str) -> Reply:
        raise NotImplementedError

    # Small constructors so subclasses stay short.
    @staticmethod
    def ok(text: str) -> Reply:
        return Reply(text=text.strip(), error=None)

    @staticmethod
    def fail(message: str) -> Reply:
        return Reply(text="", error=message)
