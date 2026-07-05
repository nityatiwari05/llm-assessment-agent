
"""
Mirrors the assignment's actual grading methodology: an LLM plays the persona/facts
given in each trace file, has a real multi-turn conversation with the agent (capped
at 8 turns), and we score:

  - hard evals: schema compliance, catalog-only URLs, recommendation count, turn cap
  - Recall@10 against `expected_shortlist_names` (name match, case-insensitive)
  - mean Recall@10 across all traces

Runs the agent IN-PROCESS by default (imports app.agent.handle_chat) for speed;
pass --http-endpoint to hit a deployed server instead, exercising the real API.
The simulated user and the agent under test are independently configurable
providers -- by default both follow LLM_PROVIDER from your environment/.env, but
you can override the simulated user specifically with --user-provider (useful if
your agent runs on local Ollama but you have an Anthropic key available and want
a stronger persona role-player driving the conversation; small local models tend
to be noticeably weaker at "stay in character as a hiring manager" than they are
at "follow a JSON schema").

Usage:
    python -m eval.run_llm_simulated_eval "eval/traces/*.json"
    python -m eval.run_llm_simulated_eval "eval/traces/*.json" --user-provider anthropic
    python -m eval.run_llm_simulated_eval "eval/traces/*.json" --http-endpoint https://your-app.onrender.com/chat
"""

from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import httpx
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.agent import handle_chat  # noqa: E402
from app.catalog import get_catalog  # noqa: E402
from app.config import ANTHROPIC_API_KEY, LLM_PROVIDER, OLLAMA_BASE_URL  # noqa: E402
from app.llm_client import LLMError, call_freeform  # noqa: E402
from app.schemas import Message  # noqa: E402

MAX_TURNS = 8
SIMULATED_USER_SYSTEM = """You are role-playing as a user talking to an assessment-recommendation \
chatbot, per the persona and facts below. Stay in character. Answer questions truthfully and only \
from the facts given; if asked about something not covered, say you don't have a strong preference. \
Once the assistant gives you a shortlist that reasonably covers your stated needs, accept it in your \
next message (e.g. "That looks good, thanks.") and do not ask for further changes. Keep messages short \
and natural, like a busy hiring manager typing quickly. Output ONLY the message text, nothing else -- \
no quotation marks around it, no "As the user, I would say" framing.

PERSONA:
{persona}

FACTS:
{facts}
"""

def simulate_user_turn(persona: str, facts: Dict, history: List[Dict], user_provider: Optional[str]) -> str:
    system = SIMULATED_USER_SYSTEM.format(persona=persona, facts=json.dumps(facts, indent=2))
    flipped = []
    for m in history:
        role = "user" if m["role"] == "assistant" else "assistant"
        flipped.append({"role": role, "content": m["content"]})
    if not flipped:
        flipped = [{"role": "user", "content": "(Start the conversation as this persona.)"}]
    return call_freeform(system, flipped, max_tokens=200, provider=user_provider)

def call_agent_inprocess(history: List[Dict]) -> Dict:
    msgs = [Message(role=m["role"], content=m["content"]) for m in history]
    response = handle_chat(msgs)
    return {
        "reply": response.reply,
        "recommendations": [r.model_dump() for r in response.recommendations],
        "end_of_conversation": response.end_of_conversation,
    }

def call_agent_http(endpoint: str, history: List[Dict]) -> Dict:
    payload = {"messages": [{"role": m["role"], "content": m["content"]} for m in history]}
    r = httpx.post(endpoint, json=payload, timeout=30.0)
    r.raise_for_status()
    return r.json()

def recall_at_10(recommended_names: List[str], expected_names: List[str]) -> float:
    if not expected_names:
        return 1.0
    rec_lower = {n.lower() for n in recommended_names[:10]}
    hit = sum(1 for e in expected_names if e.lower() in rec_lower)
    return hit / len(expected_names)

def run_trace(trace: Dict, catalog_urls: set, http_endpoint: Optional[str], user_provider: Optional[str]) -> Dict:
    history: List[Dict] = []
    problems: List[str] = []
    last_recs: List[Dict] = []
    for turn in range(MAX_TURNS):
        try:
            user_msg = simulate_user_turn(trace["persona"], trace["facts"], history, user_provider)
        except LLMError as e:
            problems.append(f"turn {turn+1}: simulated user call failed: {e}")
            break
        history.append({"role": "user", "content": user_msg})
        try:
            if http_endpoint:
                agent_out = call_agent_http(http_endpoint, history)
            else:
                agent_out = call_agent_inprocess(history)
        except Exception as e:  # noqa: BLE001
            problems.append(f"turn {turn+1}: agent call failed: {e}")
            break
        history.append({"role": "assistant", "content": agent_out["reply"]})
        recs = agent_out.get("recommendations") or []
        n = len(recs)
        if n != 0 and not (1 <= n <= 10):
            problems.append(f"turn {turn+1}: recommendation count {n} outside [0] or [1,10]")
        for rec in recs:
            if rec["url"] not in catalog_urls:
                problems.append(f"turn {turn+1}: URL not in catalog: {rec['url']}")
        if recs:
            last_recs = recs
        if agent_out.get("end_of_conversation"):
            break
    else:
        problems.append(f"conversation did not end within {MAX_TURNS} turns")

    recommended_names = [r["name"] for r in last_recs]
    recall = recall_at_10(recommended_names, trace.get("expected_shortlist_names", []))

    return {
        "trace_id": trace["trace_id"],
        "n_turns": len(history) // 2,
        "final_recommendations": recommended_names,
        "expected": trace.get("expected_shortlist_names", []),
        "recall_at_10": recall,
        "hard_eval_pass": len(problems) == 0,
        "problems": problems,
        "transcript": history,
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_globs", nargs="+")
    parser.add_argument("--http-endpoint", default=None, help="POST /chat URL of a deployed server")
    parser.add_argument(
        "--user-provider",
        default=None,
        choices=["anthropic", "ollama"],
        help="Override which provider plays the simulated user (defaults to LLM_PROVIDER, "
        "i.e. whatever the agent itself is configured to use).",
    )

    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    effective_user_provider = (args.user_provider or LLM_PROVIDER).lower()
    if effective_user_provider == "anthropic" and not ANTHROPIC_API_KEY:
        print(
            "ANTHROPIC_API_KEY is not set, but the simulated user is configured to use "
            "Anthropic (LLM_PROVIDER=anthropic, or --user-provider anthropic). Either "
            "set ANTHROPIC_API_KEY, or pass --user-provider ollama to simulate the user "
            "locally too (lower persona-quality, but zero external dependency).",
            file=sys.stderr
        )
        sys.exit(1)
    if effective_user_provider == "ollama":
        print(f"Simulated user will run against Ollama at {OLLAMA_BASE_URL}.", file=sys.stderr)

    trace_paths: List[str] = []
    for g in args.trace_globs:
        trace_paths.extend(sorted(glob.glob(g)))
    trace_paths = [p for p in trace_paths if Path(p).name != "example_trace.json"]
    if not trace_paths:
        print(
            "No trace files matched (excluding example_trace.json by filename). "
            "Add real traces to eval/traces/ -- if you copied the template, make "
            "sure you renamed the file, not just trace_id inside it.",
            file=sys.stderr,
        )
        sys.exit(1)

    catalog = get_catalog()
    catalog_urls = {i.url for i in catalog.items}

    results = []

    for p in trace_paths:
        trace = json.loads(Path(p).read_text(encoding="utf-8"))
        print(f"\n=== Running {trace['trace_id']} ({p}) ===")
        result = run_trace(trace, catalog_urls, args.http_endpoint, args.user_provider)
        results.append(result)
        print(
            f"  recall@10: {result['recall_at_10']:.2f} | "
            f"hard_eval_pass: {result['hard_eval_pass']} | turns: {result['n_turns']}"
        )
        print(f"  expected:  {result['expected']}")
        print(f"  got:       {result['final_recommendations']}")
        if result["problems"]:
            for p_ in result["problems"]:
                print(f"  !! {p_}")

        if args.verbose:
            for m in result["transcript"]:
                print(f"    [{m['role']}] {m['content'][:160]}")


    mean_recall = sum(r["recall_at_10"] for r in results) / len(results)
    n_hard_pass = sum(1 for r in results if r["hard_eval_pass"])
    print("\n=== SUMMARY ===")
    print(f"traces run:        {len(results)}")
    print(f"hard-eval pass:    {n_hard_pass}/{len(results)}")
    print(f"mean recall@10:    {mean_recall:.3f}")

    Path("eval/last_run_results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nFull transcripts written to eval/last_run_results.json")

if __name__ == "__main__":
    main()

