# Evaluation Copilot Dashboard

Public, source-backed visualization of the frozen Version A evaluation.

## Local development

```text
npm ci
npm run check
npm run dev
```

## Production build

```text
npm run build
```

The build is written to `dist/`. GitHub Actions deploys this directory to GitHub Pages.

## Live usage view (served by the local web app)

`Live usage` fetches `GET /api/metrics` from the same origin. When the dashboard is
served by the web app (`python -m webapp`), it shows real-time per-turn token usage
and estimated cost for online LLM turns, refreshing every 5 seconds. On a static
GitHub Pages deployment the panel shows a graceful "Live metrics unavailable" notice.

## Public data boundary

The bundled `src/data/dashboardData.json` contains only reviewed evaluation metrics, 200 compact Session result records and version provenance. It does not contain:

- API keys, tokens, cookies or environment variables;
- local absolute paths;
- Catalog rows or compressed Catalog files;
- raw per-turn conversation text;
- temporary evaluation snapshots;
- private evaluation data.

Official public-200, self-built generalization and self-built robustness results are visibly separated and are never combined into one score.

Frozen Agent: `4dab398a82b399076b7d201009ea9ab3bdc7909a`.
