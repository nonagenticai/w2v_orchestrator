# Multi-stage Dockerfile for w2v_orchestrator
#
# Builds from the public base image. The shared library (`w2v-common`) resolves
# from the public git source declared in pyproject.toml, so no private index or
# host mapping is needed.
# Build with: docker buildx build -f Dockerfile .

FROM python:3.14-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /uvx /bin/

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    curl \
    ca-certificates \
    git \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for layer caching
COPY pyproject.toml uv.lock README.md ./
COPY alembic.ini ./

# Resolve and install deps; w2v-common comes from its public git tag
RUN uv sync --frozen --no-dev --no-install-project

# Now copy source and install the project itself
COPY src/ ./src/

RUN uv sync --frozen --no-dev

# ============================================================================
# Stage 2: Runtime
# ============================================================================
FROM python:3.14-slim AS production

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 orchestrator && \
    mkdir -p /app /app/logs /app/data && \
    chown -R orchestrator:orchestrator /app

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder --chown=orchestrator:orchestrator /app/.venv /app/.venv

# Copy application code and resources
COPY --chown=orchestrator:orchestrator src/ ./src/
COPY --chown=orchestrator:orchestrator templates/ ./templates/
COPY --chown=orchestrator:orchestrator scaffold/ ./scaffold/
# One-off maintenance scripts (e.g. backfill_project_status) — runnable in-pod.
COPY --chown=orchestrator:orchestrator scripts/ ./scripts/
# Bundled royalty-free sound-effects library — read in-pod by
# scripts/seed_sound_effects.py to seed the sound-effects object-store bucket.
COPY --chown=orchestrator:orchestrator assets/ ./assets/
COPY --chown=orchestrator:orchestrator alembic.ini ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app

USER 1000:1000

# Port 6300 (W2V Orchestrator)
EXPOSE 6300
# Dedicated plaintext Prometheus metrics port (API_METRICS_PORT, default 6301).
EXPOSE 6301

# A deployment's own livenessProbe / readinessProbe are the source of truth for
# health; a Dockerfile HEALTHCHECK would only drift from them.

# Migrations are expected to run as a separate job before the container starts.
# Keeping `alembic upgrade head` in CMD as well would crashloop on a bad
# migration instead of failing fast while existing replicas keep serving.
CMD ["python", "-m", "src.asgi"]
