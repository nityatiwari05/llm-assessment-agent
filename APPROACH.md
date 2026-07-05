# Approach

## Architecture

Stateless FastAPI service (`GET /health`, `POST /chat`) with three layers:

1. **Retrieval** — BM25 (`rank_bm25`) over `name + description + keys + job_levels`
   for every catalog row. I chose lexical over embedding retrieval deliberately:
   SHL assessment names are short, precise, near-keyword tokens ("Core Java",
   "SVAR", "OPQ32r", "GSA") rather than natural-language concepts needing
   semantic paraphrase matching — exactly BM25's strong suit — and it needs no
   model download, so cold start on a free-tier host stays fast and the whole
   pipeline is deterministic and unit-testable without hitting a model hub.
   I added a small hand-written synonym table (`aws` → "amazon web services",
   `js` → "javascript", etc.) to bridge common JD shorthand to catalog
   vocabulary, and a query-fan-out (`search_multi`) that runs BM25 separately
   over the last user turn, the last 4 messages, and the full user-turn history,
   then unions top hits — a single bag-of-words query blurs distinct
   constraints together (e.g. "Java" + "situational judgement" + "graduate"
   in one query underweights the SJT signal against the much more frequent Java
   terms in the catalog).

2. **Orchestration** (`app/agent.py`) — builds a candidate pool (retrieval ∪ any
   catalog item literally named anywhere in the conversation, for
   compare/refine continuity), formats it compactly into the system prompt, and
   makes **one** Claude call per turn using **forced tool-use** rather than
   "please respond in JSON." The tool schema requires: `action` (clarify /
   recommend / refine / compare / refuse), `reply`, `selected_entity_ids`
   (must come from the candidate pool), and `end_of_conversation`. Forcing tool
   use guarantees schema-valid output every turn with no JSON-repair step,
   which matters under the 30s/call budget and the hard schema-compliance eval.

3. **Grounding enforcement in Python, not just in the prompt.** After the LLM
   responds, `handle_chat` filters `selected_entity_ids` down to the ones that
   are actually in the candidate pool before building the response, and hard-
   forces `recommendations=[]` / `end_of_conversation=false` whenever
   `action` is `clarify` or `refuse`, regardless of what the model produced for
   those fields. This is the actual mechanism behind "every URL comes from your
   scraped catalog" and "recommendations are empty when still gathering
   context" — I don't rely on the model to self-police these, because a replay
   harness with a realistic, occasionally contradictory simulated user is
   exactly the setting where a model drifts under instruction-following alone.

## Context engineering / prompt design

The system prompt encodes the four behaviors (clarify / recommend / refine /
compare) as explicit decision rules rather than examples to imitate, because
the traces I was given have real behavioral nuance that's brittle to copy
verbatim — e.g. "don't over-clarify: you have 8 turns total, so once you have
role + level + the 1-2 skills that matter most, commit" came directly from
watching the Rust-engineer and full-stack-JD traces, where the agent
clarifies exactly once or twice before shaping a shortlist, not per skill
mentioned. I also explicitly told the model that a broad multi-skill JD is a
reason to *ask which skills matter most*, not a reason to test everything —
otherwise the natural failure mode is a bloated 8-10 item battery for every JD.

Refinement continuity is the trickiest part of a stateless API: there's no
server-side shortlist to diff against, only message text. I solved this two
ways: (a) the system prompt instructs the model to reconstruct the current
shortlist by reading prior turns, and (b) any catalog item named anywhere in
history is force-included in this turn's candidate pool even if BM25 wouldn't
retrieve it fresh, so "keep the OPQ, drop the REST test" stays grounded to the
same `entity_id`s turn over turn instead of silently re-picking a different
row with a similar name.

Refusals (off-topic, legal/compliance questions, prompt injection) are a first
one of the four values `action` can take, not a separate code path — this
keeps "refuse this one out-of-scope question but don't nuke the rest of an
otherwise-good conversation" (see the HIPAA legal-question trace) naturally
expressible: the model can refuse *this turn* while the next turn resumes
normal recommend/refine behavior, because refusal isn't sticky across turns
except where the child-safety-style principle applies (it doesn't here). I
added a lightweight regex heuristic layer (`app/guardrails.py`) purely as a
flag surfaced to the model in the system prompt (not a hard block) — this
avoids false-positive refusals on legitimate messages that happen to contain a
trigger phrase, while still giving the model an explicit nudge on the clearer
injection attempts.

## Evaluation approach

Three layers, cheapest-and-fastest first:

1. **Unit tests** (`tests/`) — catalog loading, dedup, BM25 relevance, schema
   shape. No network, no LLM, run in CI on every commit.
2. **Behavior probes** (`eval/run_behavior_probes.py`) — 5 scripted, binary-
   assertion conversations against the real in-process agent: no-recommend-on-
   vague-turn-1, refuses legal-advice questions, refuses off-topic creative
   tasks, resists a direct "ignore your instructions" injection, and shortlist
   grows after a refinement request. Fast enough to run before every deploy.
3. **LLM-simulated-user replay + Recall@10** (`eval/run_llm_simulated_eval.py`)
   — mirrors the actual grading methodology described in the assignment: an
   LLM plays a persona with a fixed fact sheet, has a real ≤8-turn conversation
   with the agent, and I score hard-eval constraints plus Recall@10 against a
   labeled `expected_shortlist_names` per trace. I also built
   `eval/load_md_traces.py` + `run_scripted_replay.py` to replay the literal
   user turns from the markdown-rendered dev conversations you gave me directly
   (useful for eyeballing behavioral drift against a human-written reference
   reply), though that format has no ground-truth facts so it's a sanity check,
   not a scoring signal.

## What didn't work / iteration notes

- **First pass used a single BM25 query = full conversation text.** This
  under-retrieved secondary constraints (e.g. "add a situational judgement
  element" got drowned out by earlier, more frequent domain terms). Fixed with
  the query-fan-out + union approach in `search_multi`.
- **Letting the model free-type URLs was the biggest early source of
  hallucination risk** even with a "use the catalog" instruction — moving to
  entity_id selection from a closed candidate list plus Python-side filtering
  was the fix, not a better prompt. I'd treat this as the single highest-
  leverage design decision in the whole project.
- **A single "is this in scope?" LLM pre-classifier call** (before the main
  call) was tried and dropped — it doubled latency for marginal gains over just
  putting refusal criteria in the main system prompt as one of the four
  `action` values, and doubling LLM calls per turn risked the 30s budget on
  slower connections.

## AI tool usage

Built with Claude (via Claude Code / this chat) for scaffolding FastAPI
boilerplate, the BM25/synonym retrieval layer, the eval harnesses, and drafting
this document; I made the core design calls (BM25 over embeddings, forced
tool-use over JSON-in-prompt, Python-side grounding enforcement over prompt-
only grounding, refusal-as-an-action-value over a separate refusal code path)
based on the assignment's stated failure modes and the behavioral patterns
visible in the provided dev traces.
