# citadel-inference-server :tts
#
# The TTS family layer: :base + `pip install omnivoice`. Published as
#   ghcr.io/aceteam-ai/citadel-inference-server:tts        (floating)
#   ghcr.io/aceteam-ai/citadel-inference-server:tts-<sha>  (immutable)
# and referenced by citadel-cli's embedded omnivoice compose (P1).
#
# torch/torchaudio are already present from :base (cu128 wheels), so uv
# resolves omnivoice's torch>=2.4 / torchaudio>=2.4 against them and pulls no
# second torch from PyPI.
#
#   docker build -f images/tts/Dockerfile \
#     --build-arg BASE_IMAGE=citadel-inference-server:base -t citadel-inference-server:tts .
ARG BASE_IMAGE=ghcr.io/aceteam-ai/citadel-inference-server:base
FROM ${BASE_IMAGE}

ARG OMNIVOICE_VERSION=0.2.1
RUN uv pip install "omnivoice==${OMNIVOICE_VERSION}"

# The pinned cross-phase contract (design doc §6.2 / the P0 brief).
ENV CIS_TASK=tts \
    CIS_ADAPTER=omnivoice \
    CIS_MODEL=k2-fsa/OmniVoice

# Weights are NOT baked in: the checkpoint is CC-BY-NC and ~1 GB, and every
# node already mounts a shared HF cache. First start self-provisions into
# HF_HOME (after a disk preflight), or finds a MODEL_CACHE_PULL pre-fetch.
