"""
LLM integration wrapper supporting Anthropic and Ollama.

Anthropic uses forced tool-use for guaranteed structured output.

Ollama has no equivalent of Anthropic's forced tool-use, so we lean on two
things instead, in order of preference:

  1. Ollama's `format` field set to a JSON Schema (constrained/"structured
     output" decoding, supported since Ollama 0.5 — this uses a grammar under
     the hood, so it's a genuine guarantee, not a prompting trick). If the
     running Ollama version is too old to understand a schema in `format`,
     the server returns a 400 and we transparently fall back to...
  2. `format: "json"` (the older, coarser "just emit valid JSON" mode), paired
     with a defensive extractor that strips code fences and does balanced-
     brace scanning (not a greedy regex) to pull the JSON object out even if
     the model still adds stray prose around it.

Both paths return a dict shaped like RESPOND_TOOL's input_schema:
{action, reply, selected_entity_ids, end_of_conversation}. The prompt text in
app/agent.py MUST describe this same shape — if you change one, change the
other.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import anthropic
import requests

from app.config import (
    ANTHROPIC_API_KEY,
    ANTHROPIC_MODEL,
    LLM_MAX_TOKENS,
    LLM_PROVIDER,
    LLM_TIMEOUT_SECONDS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
)

logger = logging.getLogger("shl_agent.llm")

RESPOND_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["clarify", "recommend", "refine", "compare", "refuse"],
        },
        "reply": {"type": "string"},
        "selected_entity_ids": {
            "type": "array",
            "items": {"type": "string"},
            "maxItems": 10,
        },
        "end_of_conversation": {"type": "boolean"},
    },
    "required": ["action", "reply", "selected_entity_ids", "end_of_conversation"],
}

RESPOND_TOOL = {
    "name": "respond",
    "description": (
        "Return the agent's next turn: the reply text, the classified action, "
        "which catalog candidates (by entity_id) to recommend right now, and "
        "whether the conversation is complete."
    ),
    "input_schema": RESPOND_SCHEMA,
}


class LLMError(RuntimeError):
    pass


def _get_provider() -> str:
    return (os.getenv("LLM_PROVIDER", LLM_PROVIDER or "anthropic")).lower()


# --------------------------------------------------------------------------
# JSON extraction helpers (used only for the Ollama format="json" fallback --
# the schema-constrained path and Anthropic's tool-use should both already
# return clean, directly-parseable JSON)
# --------------------------------------------------------------------------
def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        # drop the opening fence line (``` or ```json) and a trailing ``` if present
        lines = text.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _find_balanced_json_object(text: str) -> Optional[str]:
    """Scan for the first top-level {...} block using brace counting, so we
    don't depend on a regex that breaks the moment there's nested braces or
    trailing prose after the JSON."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _extract_json(raw_text: str) -> Dict[str, Any]:
    text = _strip_code_fences(raw_text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    candidate = _find_balanced_json_object(text)
    if candidate is not None:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass

    snippet = raw_text[:300].replace("\n", " ")
    raise LLMError(f"Response was not valid/extractable JSON. Raw (truncated): {snippet!r}")


def _validate_respond_shape(parsed: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(parsed, dict):
        raise LLMError(f"Parsed JSON was not an object: {type(parsed)}")
    # Fill safe defaults for anything missing rather than hard-failing — a
    # model that gets 3/4 fields right shouldn't blow the whole turn.
    parsed.setdefault("action", "clarify")
    parsed.setdefault("reply", "")
    parsed.setdefault("selected_entity_ids", [])
    parsed.setdefault("end_of_conversation", False)
    if not isinstance(parsed["selected_entity_ids"], list):
        parsed["selected_entity_ids"] = []
    parsed["selected_entity_ids"] = [str(x) for x in parsed["selected_entity_ids"]]
    return parsed


# --------------------------------------------------------------------------
# Anthropic
# --------------------------------------------------------------------------
def _call_anthropic(system_prompt: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    api_key = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    if not api_key:
        raise LLMError("ANTHROPIC_API_KEY is not set. Export it before starting the server.")

    client = anthropic.Anthropic(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)

    try:
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", LLM_MAX_TOKENS)),
            system=system_prompt,
            messages=conversation,
            tools=[RESPOND_TOOL],
            tool_choice={"type": "tool", "name": "respond"},
        )
    except anthropic.APITimeoutError as e:
        raise LLMError(f"LLM call timed out: {e}") from e
    except anthropic.APIStatusError as e:
        raise LLMError(f"LLM API error: {e}") from e
    except Exception as e:  # noqa: BLE001
        raise LLMError(f"LLM call failed: {e}") from e

    for block in resp.content:
        if block.type == "tool_use" and block.name == "respond":
            return _validate_respond_shape(dict(block.input))

    raise LLMError("Model did not return a tool_use 'respond' block.")


# --------------------------------------------------------------------------
# Ollama
# --------------------------------------------------------------------------
def _post_ollama(url: str, payload: Dict[str, Any], timeout: float) -> requests.Response:
    return requests.post(url, json=payload, timeout=timeout)


def _call_ollama(system_prompt: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/chat"
    model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL or "llama3.1:8b")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", LLM_TIMEOUT_SECONDS))
    max_tokens = int(os.getenv("LLM_MAX_TOKENS", LLM_MAX_TOKENS))

    messages = [{"role": "system", "content": system_prompt}] + list(conversation)

    base_payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": max_tokens},
    }

    # --- Attempt 1: schema-constrained structured output (Ollama >= 0.5) ---
    schema_payload = {**base_payload, "format": RESPOND_SCHEMA}
    try:
        resp = _post_ollama(url, schema_payload, timeout)
    except requests.RequestException as e:
        raise LLMError(f"Ollama not reachable at {base_url}: {e}") from e

    if resp.status_code == 400:
        logger.warning(
            "Ollama rejected schema-constrained format (likely an older Ollama "
            "version) — falling back to format='json'. Response: %s",
            resp.text[:300],
        )
        # --- Attempt 2: coarse json mode + defensive extraction ---
        json_mode_payload = {**base_payload, "format": "json"}
        try:
            resp = _post_ollama(url, json_mode_payload, timeout)
        except requests.RequestException as e:
            raise LLMError(f"Ollama not reachable at {base_url}: {e}") from e

    try:
        resp.raise_for_status()
    except requests.HTTPError as e:
        raise LLMError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}") from e

    try:
        data = resp.json()
    except ValueError as e:
        raise LLMError(f"Ollama returned a non-JSON HTTP body: {resp.text[:300]}") from e

    content = (data.get("message") or {}).get("content", "")
    if not content:
        raise LLMError(f"Ollama returned an empty message. Full response: {json.dumps(data)[:300]}")

    parsed = _extract_json(content)
    return _validate_respond_shape(parsed)


# --------------------------------------------------------------------------
# Public entry point (structured — used by app.agent for the recommender itself)
# --------------------------------------------------------------------------
def call_agent(system_prompt: str, conversation: List[Dict[str, str]]) -> Dict[str, Any]:
    provider = _get_provider()
    logger.info("Using LLM provider: %s", provider)

    if provider == "anthropic":
        return _call_anthropic(system_prompt, conversation)
    if provider == "ollama":
        return _call_ollama(system_prompt, conversation)
    raise LLMError(f"Unsupported LLM_PROVIDER: {provider}")


# --------------------------------------------------------------------------
# Freeform text entry point (used by eval/run_llm_simulated_eval.py to play the
# simulated user — no forced schema needed there, just a natural reply)
# --------------------------------------------------------------------------
def _call_anthropic_freeform(
    system_prompt: str, conversation: List[Dict[str, str]], max_tokens: int
) -> str:
    api_key = os.getenv("ANTHROPIC_API_KEY", ANTHROPIC_API_KEY)
    if not api_key:
        raise LLMError(
            "ANTHROPIC_API_KEY is not set (needed for the Anthropic-backed simulated "
            "user, independent of whichever provider the agent itself uses)."
        )
    client = anthropic.Anthropic(api_key=api_key, timeout=LLM_TIMEOUT_SECONDS)
    try:
        resp = client.messages.create(
            model=os.getenv("ANTHROPIC_MODEL", ANTHROPIC_MODEL),
            max_tokens=max_tokens,
            system=system_prompt,
            messages=conversation,
        )
    except Exception as e:  # noqa: BLE001
        raise LLMError(f"Anthropic freeform call failed: {e}") from e
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _call_ollama_freeform(
    system_prompt: str, conversation: List[Dict[str, str]], max_tokens: int
) -> str:
    base_url = os.getenv("OLLAMA_BASE_URL", OLLAMA_BASE_URL or "http://localhost:11434").rstrip("/")
    url = f"{base_url}/api/chat"
    model = os.getenv("OLLAMA_MODEL", OLLAMA_MODEL or "llama3.1:8b")
    timeout = float(os.getenv("LLM_TIMEOUT_SECONDS", LLM_TIMEOUT_SECONDS))

    messages = [{"role": "system", "content": system_prompt}] + list(conversation)
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": max_tokens},
    }
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise LLMError(f"Ollama not reachable at {base_url}: {e}") from e

    try:
        data = resp.json()
    except ValueError as e:
        raise LLMError(f"Ollama returned a non-JSON HTTP body: {resp.text[:300]}") from e

    content = (data.get("message") or {}).get("content", "")
    if not content:
        raise LLMError(f"Ollama returned an empty message. Full response: {json.dumps(data)[:300]}")
    return content.strip()


def call_freeform(
    system_prompt: str,
    conversation: List[Dict[str, str]],
    max_tokens: int = 300,
    provider: Optional[str] = None,
) -> str:
    """Plain-text completion, no forced schema. `provider` overrides LLM_PROVIDER
    for just this call — used by the eval harness so you can e.g. run the agent
    on Ollama but still simulate the user with Claude (an 8B local model is a
    noticeably weaker persona role-player than it is a JSON-schema follower, so
    decoupling the two is worth it if you have an Anthropic key available even
    just for eval)."""
    provider = (provider or _get_provider()).lower()
    if provider == "anthropic":
        return _call_anthropic_freeform(system_prompt, conversation, max_tokens)
    if provider == "ollama":
        return _call_ollama_freeform(system_prompt, conversation, max_tokens)
    raise LLMError(f"Unsupported provider: {provider}")