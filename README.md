# SHL Assessment Recommender

Conversational agent that recommends SHL assessments via a stateless `POST /chat`
API, backed by BM25 retrieval over the SHL product catalog and an LLM
(Claude, via the Anthropic API) for dialogue management, grounded recommendation
selection, and comparisons.

```
shl-agent/
  app/
    main.py         FastAPI app: GET /health, POST /chat
    agent.py         orchestration: retrieval -> system prompt -> LLM -> grounding
    catalog.py        loads + normalizes data/catalog.json
    retrieval.py       BM25 lexical search over the catalog
    llm_client.py       Anthropic API wrapper (forced tool-use for structured output)
    guardrails.py        prompt-injection / off-topic heuristics
    schemas.py            Pydantic request/response models (matches the spec exactly)
    config.py               env-based configuration
  data/
    catalog.json     <-- YOU ADD THIS (see below) — not committed, it's your input data
  eval/
    load_md_traces.py            parses markdown-style dev traces -> JSON
    run_scripted_replay.py       replays real user turns, checks hard-eval constraints
    run_llm_simulated_eval.py    LLM-simulated user + Recall@10 (matches grading methodology)
    run_behavior_probes.py       5 scripted binary-assertion probes
    traces/example_trace.json    documents the persona/fact trace schema
  scripts/
    validate_catalog.py    sanity-checks data/catalog.json before you deploy
    scrape_catalog.py       optional skeleton to re-scrape shl.com if data goes stale
  tests/                      unit tests, no network/LLM calls
  Dockerfile
  requirements.txt
  .env.example
```

## 1. Getting the catalog data

This repo does not ship the catalog — you already have it (either from the
assignment's provided scrape, or the raw JSON you pasted into our conversation).

Save it as `data/catalog.json`. Expected shape — a JSON array of records like:

```json
{
  "entity_id": "4084",
  "name": "Java 8 (New)",
  "link": "https://www.shl.com/products/product-catalog/view/java-8-new/",
  "job_levels": ["Mid-Professional", "Professional Individual Contributor"],
  "languages": ["English (USA)"],
  "duration": "18 minutes",
  "status": "ok",
  "remote": "yes",
  "adaptive": "no",
  "description": "Multi-choice test that measures the knowledge of Java class design...",
  "keys": ["Knowledge & Skills"]
}
```

If your file uses different key names, either rename them to match, or edit the
handful of `rec.get(...)` lines in `app/catalog.py::load_catalog` — it's a small,
single function.

**Catalog scoping (Individual Test Solutions only):** `app/catalog.py` has an
`EXCLUDE_NAME_PATTERNS` list, empty by default. If your scrape mixes in true
pre-packaged Job Solutions you want excluded, add name substrings there. I left
it empty because several of the dev traces you gave me legitimately recommend
catalog rows named like "Entry Level Customer Serv-Retail & Contact Center" and
"Manufac. & Indust. - Safety & Dependability 8.0" as individual line items, so a
blanket "Solution" filter would have wrongly dropped correct answers.

Then sanity-check it:

```bash
python scripts/validate_catalog.py
```

## 2. Install

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Configure

```bash
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY
export $(grep -v '^#' .env | xargs)   # or use python-dotenv / your shell's env loading
```

## 4. Run locally

```bash
uvicorn app.main:app --reload --port 8000
```

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## 5. Test

```bash
pytest                              # unit tests, no network/LLM calls, fast
python -m eval.run_behavior_probes  # 5 scripted probes, needs ANTHROPIC_API_KEY
```

## 6. Evaluate against traces

**The assignment's real persona/fact trace files** (from the zip, with labeled
expected shortlists) — this is the harness that mirrors the actual grading
methodology (LLM-simulated user + Recall@10 + hard evals). Convert each trace to
the schema documented in `eval/traces/example_trace.json`
(`trace_id`, `persona`, `facts`, `expected_shortlist_names`), drop them in
`eval/traces/`, then:

```bash
python -m eval.run_llm_simulated_eval "eval/traces/*.json"
```

This prints per-trace Recall@10 and hard-eval pass/fail, plus a mean Recall@10
summary, and writes full transcripts to `eval/last_run_results.json`. Add
`--http-endpoint https://your-deployed-app/chat` to run it against a live
deployment instead of in-process.

## 7. Deploy

Any container host works (Render, Fly, Railway, HF Spaces). Example for Render:

1. Push this repo to GitHub.
2. New Web Service -> connect repo -> Docker runtime (uses the provided `Dockerfile`).
3. Set env var `ANTHROPIC_API_KEY` in the Render dashboard.
4. Make sure `data/catalog.json` is committed to the repo (or mounted some other
   way) — the Dockerfile copies `./data` into the image.
5. Deploy. Cold start: the evaluator allows up to 2 minutes for the first
   `/health` call — this app's startup just loads a JSON file and builds a BM25
   index in memory, so cold start is dominated by container spin-up, not our code.

For local Docker:

```bash
docker build -t shl-agent .
docker run -p 8000:8000 -e ANTHROPIC_API_KEY=sk-... shl-agent
```

## Design notes

- **Retrieval is BM25, not embeddings.** Assessment names are short, precise
  skill/product tokens ("Core Java", "OPQ32r", "SVAR") — a keyword-heavy domain
  where lexical search is strong, fast (no model download), and fully
  deterministic. `app/retrieval.py` documents how to swap in embeddings later
  behind the same interface if you want to.
- **Grounding is enforced in Python, not just prompted.** The LLM picks
  `entity_id`s from a candidate pool built by retrieval; `app/agent.py` then
  discards any id the model returns that isn't in that pool before building the
  response. This is what guarantees "every URL comes from the scraped catalog"
  regardless of what the model does.
- **Structured output uses forced tool-use**, not "please output JSON" in the
  prompt — this avoids JSON-repair logic under a hard 30s timeout.
- **State is reconstructed from the full message history every call** (the API
  is stateless) — the system prompt instructs the model to read prior turns to
  infer the current shortlist before applying refinements, and a fuzzy
  name-match force-includes any catalog item already named in the conversation
  into the candidate pool, so "keep X" / "drop Y" stays consistent even when
  BM25 alone wouldn't have re-surfaced X this turn.
- **Refusals are a first-class action**, not just prompt text — off-topic /
  legal / injection attempts get `action="refuse"`, which is enforced in Python
  to always carry empty recommendations and `end_of_conversation=false`.
