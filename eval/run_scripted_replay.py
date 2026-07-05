"""
Scripted replay: feeds the literal user turns from each trace to the agent
in-process (imports app.agent.handle_chat directly — no HTTP server needed) and
checks the HARD EVAL constraints from the assignment:

  - schema compliance (Pydantic validates this for us)
  - every recommendation URL exists in the loaded catalog
  - recommendation count is 0, or between 1 and 10 inclusive
  - turn count doesn't exceed 8

It also prints the agent's actual reply next to the human-written reference reply
so you can eyeball behavioral drift (this is NOT a scoring signal — replies won't
match verbatim, that's expected and fine).

Usage:
    export ANTHROPIC_API_KEY=...
    python -m eval.load_md_traces eval/traces/my_traces.md > eval/traces/parsed.json
    python -m eval.run_scripted_replay eval/traces/parsed.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

# Allow running as `python -m eval.run_scripted_replay` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import handle_chat  # noqa: E402
from app.catalog import get_catalog  # noqa: E402
from app.schemas import Message  # noqa: E402


def check_hard_evals(catalog_urls: set, turns_seen: int, recommendations: List[Dict]) -> List[str]:
    problems = []
    if turns_seen > 8:
        problems.append(f"turn cap exceeded ({turns_seen} > 8)")
    n = len(recommendations)
    if n not in (0,) and not (1 <= n <= 10):
        problems.append(f"recommendation count {n} outside [0] or [1,10]")
    for rec in recommendations:
        if rec["url"] not in catalog_urls:
            problems.append(f"URL not in catalog: {rec['url']}")
    return problems


def run_trace(trace: Dict, catalog_urls: set) -> Dict:
    history: List[Message] = []
    turn_logs = []
    all_problems = []
    for i, turn in enumerate(trace["turns"], start=1):
        history.append(Message(role="user", content=turn["user"]))
        response = handle_chat(history)
        history.append(Message(role="assistant", content=response.reply))

        recs = [r.model_dump() for r in response.recommendations]
        problems = check_hard_evals(catalog_urls, len(history), recs)
        all_problems.extend(problems)

        turn_logs.append(
            {
                "turn": i,
                "user": turn["user"],
                "agent_reply": response.reply,
                "reference_reply": turn.get("reference_agent_reply", ""),
                "n_recommendations": len(recs),
                "end_of_conversation": response.end_of_conversation,
                "reference_end_of_conversation": turn.get("reference_end_of_conversation"),
                "problems": problems,
            }
        )
    return {
        "trace_id": trace["trace_id"],
        "turns": turn_logs,
        "hard_eval_pass": len(all_problems) == 0,
        "problems": all_problems,
    }


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m eval.run_scripted_replay <parsed_traces.json>", file=sys.stderr)
        sys.exit(1)

    traces = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    catalog = get_catalog()
    catalog_urls = {i.url for i in catalog.items}

    results = []
    for trace in traces:
        print(f"\n=== {trace['trace_id']} ===")
        result = run_trace(trace, catalog_urls)
        results.append(result)
        for t in result["turns"]:
            status = "OK " if not t["problems"] else "FAIL"
            print(f"  [{status}] turn {t['turn']} | recs={t['n_recommendations']} | eoc={t['end_of_conversation']}")
            print(f"      user:      {t['user'][:100]}")
            print(f"      agent:     {t['agent_reply'][:140]}")
            print(f"      reference: {t['reference_reply'][:140]}")
            for p in t["problems"]:
                print(f"      !! {p}")

    n_pass = sum(1 for r in results if r["hard_eval_pass"])
    print(f"\n=== Summary: {n_pass}/{len(results)} traces passed hard-eval checks ===")


if __name__ == "__main__":
    main()
