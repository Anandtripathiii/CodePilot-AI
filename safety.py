"""Safety layer.

Two jobs:
  1. check_prompt   — refuse harmful requests, strip prompt-injection attempts
  2. check_response — a last look at model output before it reaches the browser

This is a coding assistant, so the bar is: help with any legitimate
programming question, refuse things whose only purpose is to attack
someone else's system or a person.
"""

import re
from dataclasses import dataclass
from typing import Final

# Requests we do not help with, whatever the wording.
BLOCKED_PATTERNS: Final[list[str]] = [
    r"\b(write|build|make|create|generate)\b.{0,40}\b(ransomware|malware|keylogger|rootkit|botnet|trojan|computer virus|worm)\b",
    r"\bddos\b.{0,30}\b(script|tool|attack|someone|website)\b",
    r"\b(steal|harvest|exfiltrate|dump)\b.{0,30}\b(credentials|passwords|credit card|cookies|session tokens)\b",
    r"\bcrack\b.{0,25}\b(password|licence|license|activation|serial)\b",
    r"\b(hack|break)\s+into\b.{0,40}\b(account|server|wifi|network|database|phone)\b",
    r"\bphishing\b.{0,30}\b(page|site|email|kit|template)\b",
    r"\bbypass\b.{0,30}\b(authentication|paywall|drm|2fa|two.factor)\b",
]

# Attempts to overwrite the system prompt. These are removed, not blocked —
# the rest of the message is usually a normal question.
INJECTION_PATTERNS: Final[list[str]] = [
    r"ignore (all|any|the) (previous|prior|above) (instructions|prompts?|rules)",
    r"disregard (your|all|the) (rules|instructions|guidelines|system prompt)",
    r"you are now (an?|in) [^.\n]{0,60}",
    r"(reveal|print|show|repeat) (your|the) (system prompt|instructions|initial prompt)",
    r"\bdeveloper mode\b",
    r"\bDAN mode\b",
    r"pretend (you have|there are) no (rules|restrictions|filters)",
]

# Anything that looks like a real secret is masked before display.
SECRET_PATTERNS: Final[list[tuple[str, str]]] = [
    (r"sk-[A-Za-z0-9]{20,}", "sk-***"),
    (r"AIza[0-9A-Za-z\-_]{30,}", "AIza***"),      # older Google keys
    (r"AQ\.[A-Za-z0-9\-_]{20,}", "AQ.***"),       # current Google AI Studio keys
    (r"ghp_[A-Za-z0-9]{30,}", "ghp_***"),
    (r"AKIA[0-9A-Z]{16}", "AKIA***"),
]

MAX_LENGTH: Final[int] = 12_000

BLOCK_MESSAGE: Final[str] = (
    "That request looks like it is aimed at attacking a system rather than "
    "building one, so CodePilot will not answer it. Ask about the same topic "
    "defensively — how an attack works, or how to protect against it — and it "
    "will help."
)

EMPTY_MESSAGE: Final[str] = (
    "That message was only an instruction override, so there is nothing to answer."
)

# Compiled once at import. Patterns never change at runtime.
_BLOCKED = [re.compile(p, re.IGNORECASE) for p in BLOCKED_PATTERNS]
_INJECTIONS = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
_SECRETS = [(re.compile(p), mask) for p, mask in SECRET_PATTERNS]


@dataclass(frozen=True, slots=True)
class Verdict:
    """The result of screening one prompt."""

    allowed: bool
    cleaned_prompt: str
    reason: str = ""


def check_prompt(text: str) -> Verdict:
    """Screen a user prompt before it reaches a model."""
    for pattern in _BLOCKED:
        if pattern.search(text):
            return Verdict(False, "", BLOCK_MESSAGE)

    cleaned = text
    for pattern in _INJECTIONS:
        cleaned = pattern.sub("[removed]", cleaned)

    cleaned = cleaned[:MAX_LENGTH].strip()

    if not cleaned or cleaned == "[removed]":
        return Verdict(False, "", EMPTY_MESSAGE)

    return Verdict(True, cleaned)


def check_response(text: str) -> str:
    """Mask anything that looks like a live API key before display."""
    for pattern, mask in _SECRETS:
        text = pattern.sub(mask, text)
    return text
