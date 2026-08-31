# syntax=docker/dockerfile:1
# ---- Stage 1: build the Evaluation Dashboard (static React app) ----
FROM node:22-alpine AS dashboard-build
WORKDIR /app
COPY dashboard/package.json dashboard/package-lock.json ./
RUN npm ci
COPY dashboard/ .
RUN npm run build

# ---- Stage 2: rules-only offline runtime (stdlib + sqlite + tiny web deps) ----
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    LLM_PROVIDER=none \
    LLM_INTENT_ENABLE=0 \
    RETRIEVAL_BACKEND=auto

WORKDIR /app

# Web + rules-only runtime deps (no torch/transformers/models -> small image)
COPY requirements-web.txt .
RUN pip install -r requirements-web.txt numpy

# Runtime code (rules-only offline Agent + webapp)
COPY agent/ agent/
COPY config/ config/
COPY webapp/ webapp/
COPY llm/ llm/
COPY utils/ utils/
COPY starter/ starter/

# Data (catalog is required; assets/analysis optional-but-used; NO 195MB blair npy, NO models)
COPY data/catalog.jsonl data/catalog.jsonl
COPY data/assets/ data/assets/
COPY data/analysis/ data/analysis/
COPY data/seeds/ data/seeds/

# Built dashboard from stage 1
COPY --from=dashboard-build /app/dist dashboard/dist/

EXPOSE 7860
CMD ["sh", "-c", "python -m webapp --host 0.0.0.0 --port ${PORT:-7860}"]
