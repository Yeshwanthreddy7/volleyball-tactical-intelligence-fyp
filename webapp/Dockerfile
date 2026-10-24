# Volleyball Tactical Analysis - full stack image (React frontend + FastAPI + GPU engine).
# Build from the repository root (finalyearproject/):
#   docker build -f webapp/Dockerfile -t vta-web .
# Run with an NVIDIA GPU (nvidia-container-toolkit on the host):
#   docker compose -f webapp/docker-compose.yml up -d

# ---------- 1. frontend
FROM node:20-alpine AS web
WORKDIR /web
COPY webapp/frontend/package.json webapp/frontend/package-lock.json* ./
RUN npm ci --include=dev || npm install --include=dev
COPY webapp/frontend/ ./
RUN npm run build

# ---------- 2. backend + research code
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 LOKY_MAX_CPU_COUNT=4 \
    VTA_REPO_ROOT=/app VTA_STORAGE_DIR=/data VTA_DEMOS_DIR=/app/webapp/storage/demos \
    VTA_FRONTEND_DIST=/app/webapp/frontend/dist
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY webapp/backend/requirements.txt webapp/backend/requirements.txt
RUN pip install -r webapp/backend/requirements.txt

# research code and models the engine runs (see webapp/.dockerignore for the full list)
COPY . /app
COPY --from=web /web/dist /app/webapp/frontend/dist

WORKDIR /app/webapp/backend
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
