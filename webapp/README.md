# Shopping Copilot — Local Web App & Evaluation Dashboard

A product-level, environment-adaptive frontend for the conversational shopping agent.
The core retrieval/rerank/decision pipeline is untouched: the web app only lets a
non-technical user pick which engine combination to run (LLM provider/model, rerank,
retrieval backend, output strategy, enhancement toggles) — or let **Auto (LUT)** pick
the best config for the current environment.

## Highlights

- **Environment-adaptive**: a runtime panel exposes every engine option the project
  supports. `Auto (LUT)` shows the recommended config for the detected environment
  (`device × dense × llm × network`) with its expected Technical Score.
- **Offline by default, zero cost**: the default config runs fully local (rule intent,
  no rerank, hold-back output). No API key is required; no token is consumed.
- **Online on explicit choice**: DeepSeek / OpenAI intent recognition and semantic
  rerank (qwen3-rerank MaaS, chat-LLM) are available, but only enabled when the user
  selects them and confirms the paid-API dialog.
- **Key security**: API keys are sent to the local backend, held in-memory in the
  runtime only, never written to disk, never logged, and never returned by the API.
  The key field is intentionally cleared after applying a config.
- **Token/cost visibility**: every assistant turn shows the token usage and an
  approximate USD cost when an online LLM call happened.

## Quick start

```bash
# 1. Python web dependencies
pip install -r requirements-web.txt

# 2. (Optional) build the Evaluation Dashboard once
cd dashboard
npm install
npm run build
cd ..

# 3. Start the app (default: offline, Auto retrieval, hold-back)
python -m webapp --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 — the agent build re-indexes the 50k catalog on the first
config switch and can take ~30s; the status pill shows `Loading → Ready`.

## Runtime configuration panel

The sidebar exposes the engine selectors rendered from `GET /api/runtime`:

| Control | Options |
|---|---|
| LLM provider | Off (rule-based) / DeepSeek / OpenAI |
| Model | provider-specific (deepseek-chat, gpt-4o-mini, ...) |
| API key | optional; kept in server memory only |
| Semantic rerank | Off / Auto (qwen3→chat→rule) / qwen3-rerank (MaaS) / Chat LLM |
| Retrieval backend | Auto (BLaIR dense if available) / BM25 / Dense / Hybrid |
| Output strategy | Hold-back (default) / Full Top-10 / Hold-back + confidence gate |
| Enhancements | LLM intent, constraint fingerprint, category expand, paraphrase |

Selecting any online option (LLM provider, non-off rerank, or LLM intent) shows a
confirm dialog: “This enables online AI features … may incur cost”. Applying a config
rebuilds the engine when needed and starts a new chat (`sessions_reset=true`).

The **Auto (LUT) banner** displays the recommendation for the current environment and
its expected Technical Score; the default launch already uses that recommendation.

## Real-time usage & cost recording (Live usage)

Every chat turn is recorded by the web runtime and exposed through `GET /api/metrics`
(no API keys, ever). When an online LLM is enabled, each turn captures:

- provider / model, retrieval / rerank backend, output strategy;
- `prompt_tokens` / `completion_tokens` (sum of intent LLM + semantic rerank usage);
- estimated USD cost (approximate per-1M-token pricing table in `webapp/metrics.py`);
- latency and timestamp.

The **Evaluation Dashboard ? Live usage** view (served at `/dashboard`) polls
`/api/metrics` every 5 seconds and shows KPIs (total turns, online turns, total
tokens, estimated cost), a per-provider breakdown, and a recent-turns table.

- Recording is in-memory for the process lifetime by default. To append every event
  to a JSONL file (survives restarts), set `WEBAPP_METRICS_LOG=/path/to/usage.jsonl`
  before starting the app.
- If the dashboard is opened as a static GitHub Pages build (no local web app), the
  Live usage panel degrades gracefully with a "Live metrics unavailable" message.

## Evaluation Dashboard

If `dashboard/dist/` exists (built with `npm run build`), the dashboard is mounted at
**/dashboard** and linked from the sidebar. It visualizes the public-200 evaluation
(HR/MRR/MTTC/TS), the generalization-gap disclosure (rule-only vs LLM on paraphrased
queries), and per-strategy A/B comparisons.

## HTTP API (subset used by the panel)

- `GET /api/runtime` — environment fingerprint, LUT recommendation, active config
  summary, and the option catalog for the selectors (never includes API keys).
- `POST /api/runtime/config` — switch engine config; body fields are whitelisted,
  unknown keys are ignored; returns the new runtime info with `sessions_reset: true`.
- `POST /api/sessions` / `POST /api/sessions/{id}/messages` — chat; `agent_response`
  carries `recommendations`, `ask_attribute`, `message`, and `usage` (tokens).
- `GET /api/products/{parent_asin}` — product detail for the drawer.

## Tests

```bash
python -m pytest tests/test_webapp_api.py tests/test_webapp_service.py \
  tests/test_webapp_catalog.py tests/test_webapp_static.py
```

46 tests cover the HTTP API, session service, catalog presenter, static assets, and
security guarantees (no key leakage in runtime responses, key stripping in config
payloads, 503 while loading/failed).
