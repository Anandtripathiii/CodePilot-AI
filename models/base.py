"""The contract every LLM wrapper implements.

A wrapper can be built two ways:

    GeminiModel()                     -> uses the server's own key, if set
    GeminiModel(key, "model-name")    -> uses a key supplied by a visitor

The second form is what makes "bring your own key" work: nothing is stored
on the server, the key lives in the visitor's browser and is used for the
length of one request.
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
        """True when a key is present and the client loaded."""
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
