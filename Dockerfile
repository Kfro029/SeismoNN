FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml uv.lock README.md ./
COPY seismonn ./seismonn
COPY scripts ./scripts
COPY configs ./configs

RUN uv sync --frozen --no-dev

EXPOSE 8000

ENV SEISMONN_CHECKPOINT=/app/checkpoints/best.pt
ENV SEISMONN_DEVICE=cpu

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD uv run python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

CMD ["uv", "run", "uvicorn", "seismonn.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
