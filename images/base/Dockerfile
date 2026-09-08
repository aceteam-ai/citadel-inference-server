# citadel-inference-server :base
#
# The runtime core + torch. Every per-family image (images/<family>/) layers
# its model dependencies on top of THIS image so families never share (or
# fight over) transformers/diffusers pins.
#
# Base OS: nvidia/cuda:*-base-* -- the SMALL flavor. It carries only the
# driver-facing libs and sets NVIDIA_VISIBLE_DEVICES/NVIDIA_DRIVER_CAPABILITIES
# so the nvidia container runtime exposes the GPU; the CUDA runtime libraries
# themselves come bundled with the torch cu128 wheels (nvidia-*-cu12 pip
# packages), so the fatter *-runtime-* image would duplicate ~2 GB for
# nothing. CUDA major matches the torch wheel index (cu128).
#
# Build context is the REPO ROOT (see .github/workflows/ci.yml):
#   docker build -f images/base/Dockerfile -t citadel-inference-server:base .
ARG CUDA_IMAGE=nvidia/cuda:12.8.1-base-ubuntu24.04
FROM ${CUDA_IMAGE}

# torch + torchaudio pinned together: omnivoice requires torchaudio>=2.4, and
# installing it LATER from PyPI could drag a mismatched CPU torch along.
ARG TORCH_VERSION=2.8.0
ARG TORCH_INDEX=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# python3 = 3.12 on ubuntu 24.04. libsndfile1 backs soundfile (wav encode).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 python3-venv ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# uv for fast, reproducible installs (workspace convention: uv, never pip).
COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /usr/local/bin/uv

# A venv (ubuntu 24.04's system python is externally-managed, PEP 668).
ENV VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH
RUN uv venv --python /usr/bin/python3 "$VIRTUAL_ENV"

RUN uv pip install \
        "torch==${TORCH_VERSION}" "torchaudio==${TORCH_VERSION}" \
        --index-url "${TORCH_INDEX}"

WORKDIR /app
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY cis ./cis
RUN uv pip install --no-deps . && uv pip install \
        "fastapi>=0.115" "uvicorn[standard]>=0.30" "pydantic>=2.0" \
        "numpy>=1.26" "soundfile>=0.12" "huggingface_hub>=0.26"

# Runtime contract defaults. HF_HOME is bind-mounted from the node's
# ~/citadel-cache/huggingface; the hub layout lands in HF_HOME/hub (never set
# HF_HUB_CACHE=HF_HOME -- see cis/hub.py).
ENV HF_HOME=/root/.cache/huggingface \
    PORT=8000 \
    CIS_DEVICE=auto \
    CIS_DTYPE=auto \
    CIS_SLOTS=2 \
    CIS_MAX_INPUT_CHARS=5000 \
    CIS_PRELOAD=true

EXPOSE 8000

# curl-free healthcheck: readiness = /health body's model_loaded (the same
# gate citadel-cli's synthesizeHealthReady uses). start-period covers a
# first-start weights pull.
HEALTHCHECK --interval=30s --timeout=10s --retries=5 --start-period=300s \
    CMD python -c "import urllib.request,json,os,sys; d=json.load(urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','8000')+'/health')); sys.exit(0 if d.get('model_loaded') else 1)"

CMD ["python", "-m", "cis"]
