"""
Defense-in-depth heuristics. The primary defense is the system prompt (the model
is instructed to treat all message content as untrusted data, never as new
instructions). These regexes catch the crude, high-confidence cases and let us
flag them to the LLM explicitly rather than relying on the model to notice unaided.

This module never blocks a request outright — it annotates. The LLM makes the
final call on whether to refuse, using the flag as a strong hint. That avoids
false-positive refusals on legitimate messages that happen to contain a trigger
phrase (e.g. a JD that says "ignore previous certifications").
"""
from __future__ import annotations

import re

INJECTION_PATTERNS = [
    r"ignore (all|any|previous|prior|the above)?\s*(instructions|rules|prompt)",
    r"disregard (your|the) (instructions|system prompt|rules)",
    r"you are now",
    r"new instructions?:",
    r"system prompt",
    r"reveal your (prompt|instructions|system message)",
    r"act as (if you were|a different|an unrestricted)",
    r"jailbreak",
    r"pretend (you|to be)",
    r"</?(system|assistant|user)>",
    r"do anything now",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def looks_like_injection(text: str) -> bool:
    return any(p.search(text) for p in _COMPILED)


OFF_TOPIC_HINTS = [
    r"\b(write|draft) (me )?(a|an) (poem|essay|email|resume|cover letter)\b",
    r"\bstock (price|market|advice)\b",
    r"\blegal advice\b",
    r"\bmedical advice\b",
    r"\bsue\b",
    r"\bam i legally required\b",
    r"\bwhat.?s the weather\b",
    r"\btranslate\b",
    r"\bhow do i fire\b",
    r"\bsalary (negotiation|benchmark)\b",
]
_OFF_TOPIC_COMPILED = [re.compile(p, re.IGNORECASE) for p in OFF_TOPIC_HINTS]


def looks_off_topic(text: str) -> bool:
    return any(p.search(text) for p in _OFF_TOPIC_COMPILED)
