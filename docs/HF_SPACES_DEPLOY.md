# Deploy to Hugging Face Spaces (rules-only offline demo)

Public demo of the Shopping Copilot web app. **Default = fully offline / rules-only /
zero cost**. No API keys are shipped; users can bring their own keys in the UI
(kept in server memory only, never persisted).

## What gets deployed

- Product UI (`webapp/`) + Evaluation Dashboard (`dashboard/dist`, built in-image)
- Rules-only Agent (stdlib + SQLite FTS5; no torch/transformers/models)
- Optional user-supplied LLM (DeepSeek / OpenAI / qwen3-rerank) via the runtime panel

## 1. Create the Space

1. Go to https://huggingface.co/new-space
2. Space name: e.g. `shopping-copilot-demo`
3. **SDK: Docker** (custom) · Hardware: CPU basic (free)
4. Create

## 2. Push this repo to the Space (with the catalog)

The frozen catalog is git-ignored, so it must be force-added to the Space repo:

```bash
# add the Space as a remote (from this repo)
git remote add hf https://huggingface.co/spaces/<your-name>/<space-name>
git fetch hf

# force-include the 50k catalog (57.7 MB) + keep assets/analysis
git add -f data/catalog.jsonl
git add data/assets data/analysis data/seeds
git commit -m "deploy: rules-only offline demo (HF Spaces)"

# push to the Space main branch
git push hf main
```

> Note: if you create the Space "from a GitHub repo" instead, the Docker build
> will clone the GitHub repo which does NOT contain `data/catalog.jsonl` (git-ignored),
> so the Agent cannot build. Use the push-above method.

## 3. Configuration (no keys required)

| Setting | Value |
|---|---|
| LLM providers | leave unset → offline by default |
| `LLM_PROVIDER` | `none` (already the default in the image) |
| Port | Spaces injects `PORT=7860`; the image uses `${PORT:-7860}` |

Users open the app, and if they want online features they enter their **own**
DeepSeek/OpenAI key in the runtime panel (in-memory only, never written/logged).

## 4. Verify after first build

- `https://huggingface.co/spaces/<your-name>/<space-name>` → product UI
- `/dashboard` → Evaluation Dashboard
- First config build indexes 50k catalog (~30-60 s): status pill `Loading → Ready`
- Health: `/api/health`

## Do NOT push

- `data/offline_blair_embeds*.npy` (195 MB) — rules-only; dense will show "unavailable" and auto-fall back to BM25
- any model weights / caches — keep the image small and offline
- `.env` / API keys — never

## Local validation

```bash
docker build -t shopping-copilot-hf .   # or use the venv check:
python -m venv .venv && .venv/bin/pip install -r requirements-web.txt numpy
python -c "import webapp.app; from agent.main_agent import Agent; print('ok')"
```
