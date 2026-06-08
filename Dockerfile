# syntax=docker/dockerfile:1
#
# Multi-stage image for the fine-tuned contract-extractor serving layer.
#
#   Stage 1 (builder)  — CUDA *devel* base; builds a self-contained venv with
#                        the app + GPU serving stack (unsloth/bitsandbytes).
#   Stage 2 (runtime)  — CUDA *runtime* base; copies just the venv + app code,
#                        so the final image stays as small as a CUDA image gets.
#
# unsloth/bitsandbytes are intentionally NOT in pyproject (they are GPU-only and
# fail to install on macOS/CI), so they are installed here, in the image only.
#
# Build:  docker build -t contract-extractor .
# Run  :  docker run --gpus all -p 8000:8000 --env-file .env contract-extractor
#         (the model is GPU-only; on a machine without NVIDIA GPUs the build
#          still succeeds but the container cannot load the model.)

# ---------------------------------------------------------------------------
# Stage 1 — builder
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Python 3.11 (pyproject requires >=3.11; ubuntu22.04 ships 3.10) via deadsnakes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev \
        build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Self-contained virtualenv we can lift wholesale into the runtime stage.
ENV VIRTUAL_ENV=/opt/venv
RUN python3.11 -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install third-party deps first (better layer caching), then the package.
COPY pyproject.toml ./
COPY extractor ./extractor
COPY training ./training
COPY evaluation ./evaluation
COPY README.md ./README.md
RUN pip install . \
    # GPU serving stack — kept out of pyproject (GPU-only, breaks CPU installs).
    && pip install "unsloth"

# ---------------------------------------------------------------------------
# Stage 2 — runtime
# ---------------------------------------------------------------------------
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04 AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # Default to the published Hub adapter so the image works for anyone.
    EXTRACTOR_ADAPTER_PATH=solankiom/llama-3.1-8b-contract-extractor

# Runtime needs the Python 3.11 interpreter the venv was built against, plus
# curl for the container healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates curl \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 \
    && rm -rf /var/lib/apt/lists/*

# Bring over the prebuilt venv and the application source.
COPY --from=builder /opt/venv /opt/venv
WORKDIR /app
COPY extractor ./extractor
COPY training ./training
COPY evaluation ./evaluation

# Run as a non-root user.
RUN useradd --create-home --uid 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Llama 3.1 takes ~45s to load; give startup a generous grace period before the
# healthcheck can mark the container unhealthy.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "extractor.api:app", "--host", "0.0.0.0", "--port", "8000"]
