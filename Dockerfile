FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Runtime libraries used by scientific Python packages and image/data readers.
RUN apt-get update \
    && apt-get install --no-install-recommends -y ca-certificates libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

# The project config pins PyTorch packages to its CUDA 13.0 wheel index.
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:${PATH}" \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

VOLUME ["/app/data", "/app/output"]

ENTRYPOINT ["python", "-m", "hmc.main"]
CMD ["--help"]
