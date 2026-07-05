"""
Parses the markdown-rendered conversation traces (the "## Conversation" / "### Turn N"
/ "**User**" / "**Agent**" format used in the assignment's dev examples) into a
simple JSON structure:

[
  {
    "trace_id": "trace_0",
    "turns": [
      {"user": "...", "reference_agent_reply": "...", "reference_end_of_conversation": false},
      ...
    ]
  },
  ...
]

Usage:
    python -m eval.load_md_traces path/to/traces.md > eval/traces/parsed.json

This is for SCRIPTED replay only (comparing shape/behavior against a human-written
reference) — it does NOT give you persona/fact-driven simulated-user evaluation,
since these markdown transcripts don't encode the underlying facts. For the real
grading methodology (LLM-simulated user + Recall@10), use
eval/run_llm_simulated_eval.py with structured persona/fact trace files instead
(see eval/traces/example_trace.json for the schema).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Dict, List

CONVO_SPLIT = re.compile(r"^## Conversation\s*$", re.MULTILINE)
TURN_SPLIT = re.compile(r"^### Turn \d+\s*$", re.MULTILINE)
USER_RE = re.compile(r"\*\*User\*\*\s*\n+>\s?(.+?)(?=\n\n\*\*Agent\*\*)", re.DOTALL)
AGENT_RE = re.compile(
    r"\*\*Agent\*\*\s*\n+(.+?)(?=\n_`end_of_conversation`)", re.DOTALL
)
EOC_RE = re.compile(r"`end_of_conversation`:\s*\*\*(true|false)\*\*")


def parse_markdown(text: str) -> List[Dict]:
    traces = []
    conversations = CONVO_SPLIT.split(text)[1:]  # drop preamble before first split
    for ci, convo in enumerate(conversations):
        turns_raw = TURN_SPLIT.split(convo)[1:]  # drop text before first "### Turn"
        turns = []
        for turn_text in turns_raw:
            user_match = USER_RE.search(turn_text)
            agent_match = AGENT_RE.search(turn_text)
            eoc_match = EOC_RE.search(turn_text)
            if not user_match or not agent_match:
                continue
            user_text = user_match.group(1).strip()
            # collapse markdown quote markers on multi-line quotes
            user_text = re.sub(r"^>\s?", "", user_text, flags=re.MULTILINE).strip()
            agent_text = agent_match.group(1).strip()
            eoc = eoc_match.group(1) == "true" if eoc_match else False
            turns.append(
                {
                    "user": user_text,
                    "reference_agent_reply": agent_text,
                    "reference_end_of_conversation": eoc,
                }
            )
        if turns:
            traces.append({"trace_id": f"md_trace_{ci}", "turns": turns})
    return traces


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m eval.load_md_traces <path_to_markdown_file>", file=sys.stderr)
        sys.exit(1)
    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    traces = parse_markdown(text)
    print(json.dumps(traces, indent=2))


if __name__ == "__main__":
    main()
