# citadel-inference-server (CIS)

Generic, config-driven inference server for the **AceTeam Sovereign Compute Fabric**.

> **Status: P0** — base runtime + `tts` task + OmniVoice adapter + images + CI.
> Design of record:
> [`aceteam-ai/citadel-cli#1007`](https://github.com/aceteam-ai/citadel-cli/pull/1007)
> — `docs/design-generic-inference-server.md`.

## Why this exists

Onboarding a new inference model as a Citadel "service" has meant one of two
things that both scale badly:

- a bespoke Python wrapper checked into **citadel-cli** (bonsai, diffusers,
  whisper, hermes, meeting, nvr, ...), which turns a pure-Go orchestration CLI
  into a junk drawer of model code; or
- a one-off image repo per model (gliner2).

CIS replaces per-model wrappers with **one shared runtime + thin per-model-family
adapters**. citadel-cli then carries only a compose file + a handful of
registration points — **never model code**.

## Architecture

- **Base runtime** (`cis/`): env-driven config (`cis/config.py`), HuggingFace
  cache placement + self-provisioning with a disk preflight (`cis/hub.py`,
  `cis/preflight.py`), lazy **single-flight** model load (`cis/loader.py`),
  `/health` + `/info` (`cis/app.py`), a request semaphore sized by `CIS_SLOTS`,
  receipt headers.
- **Adapters** (`cis/adapters/`): a **task** decides which routes mount; a
  **family** decides how the model is called.
  - `tts.py` — the `tts` task: `POST /v1/audio/speech`, kokoro-byte-identical.
  - `omnivoice.py` — the OmniVoice family (`k2-fsa/OmniVoice`).
  - `stub.py` — a dependency-free family used by the hermetic tests and the
    CI image smoke test.
  - `__init__.py` — the `(task, family)` registry; `CIS_ADAPTER` picks the
    family, and it must implement `CIS_TASK`.
- **Images** (`images/`): `:base` (CUDA base + torch + the runtime) and `:tts`
  (`:base` + `pip install omnivoice`). Container listens on **:8000**.

"Config-only onboarding" holds **within an already-implemented family**; a new
family costs exactly one adapter file here — and never anything in citadel-cli.

## Running

```bash
docker run --rm --gpus all -p 127.0.0.1:8214:8000 \
  -v ~/citadel-cache/huggingface:/root/.cache/huggingface \
  ghcr.io/aceteam-ai/citadel-inference-server:tts
```

First start self-provisions `k2-fsa/OmniVoice` into the mounted HF cache
(after a disk preflight); a citadel-cli `MODEL_CACHE_PULL` pre-fetch of the
same repo is found and nothing is downloaded. `/health` answers `200
{"status":"loading"}` from the moment the socket is bound and flips to `"up"`
with `model_loaded: true` once weights are on the device.

```bash
curl -s localhost:8214/health
curl -s localhost:8214/info | jq .voices
curl -s localhost:8214/v1/audio/speech -H 'content-type: application/json' \
  -d '{"input":"Hello from the fabric.","voice":"female-british"}' -o out.wav -D -
```

## Config contract (`CIS_*`)

Compose-native environment variables; `Config.from_env` (`cis/config.py`) is
the authority for defaults and parsing.

| Var | Default | Meaning |
|---|---|---|
| `CIS_TASK` | required (`tts` in the image) | `tts`. Decides which routes mount. |
| `CIS_ADAPTER` | required (`omnivoice` in the image) | adapter family: `omnivoice`, `stub`. |
| `CIS_MODEL` | required (`k2-fsa/OmniVoice` in the image) | HF repo id or local path. |
| `CIS_MODEL_REVISION` | unset (`main`) | HF revision pin. |
| `CIS_DEVICE` | `auto` | `auto` \| `cpu` \| `cuda` \| `cuda:N`. `auto` = CUDA when visible. An explicit `cuda` with no visible GPU fails the load loudly. |
| `CIS_DTYPE` | `auto` | `float16` \| `bfloat16` \| `float32`; `auto` = fp16 on CUDA, fp32 on CPU. |
| `CIS_SLOTS` | `2` | concurrency semaphore around synthesis. |
| `CIS_MAX_INPUT_CHARS` | `5000` | 413 above this. |
| `CIS_PRELOAD` | `true` | load at startup (in the background, socket already bound) vs lazily on first request. |
| `CIS_DISK_PREFLIGHT` | `true` | refuse a self-provisioning download that cannot fit (fail-closed on a confirmed shortfall, fail-open when the estimate itself fails). |
| `CIS_EXTRA` | `{}` | JSON object of family knobs. OmniVoice: `num_step` (32), `model_license` (override). Malformed JSON refuses startup. |
| `PORT` | `8000` | container listen port. |
| `HF_HOME` | `/root/.cache/huggingface` | HF cache root (bind-mount it). The hub layout lands in `HF_HOME/hub` — never point `HF_HUB_CACHE` at `HF_HOME`. |
| `HF_TOKEN` | unset | HF auth for gated repos. |

## The `tts` wire contract

### `POST /v1/audio/speech`

```json
{"input": "text", "voice": "auto", "response_format": "wav", "instructions": "female, british accent"}
```

| Field | Default | Notes |
|---|---|---|
| `input` | required | > `CIS_MAX_INPUT_CHARS` → 413 |
| `voice` | `"auto"` | **enumerable**: `"auto"` + the presets in `/info`.`voices`. Unknown → 400. Never free text. |
| `response_format` | `"wav"` | `wav` only in v1 (PCM-16 RIFF). Anything else → 400. |
| `instructions` | absent | free-form "voice design", handed to the adapter verbatim; **overrides** the `voice` preset. OmniVoice validates it against its closed attribute vocabulary and an invalid value → 400 with upstream's message (which lists the valid items). |
| `language` | absent | optional language hint → OmniVoice `language=`. |
| `speed` | absent | optional `0.5..2.0` → OmniVoice `speed=`. |
| `model` | absent | accepted and ignored (OpenAI-client compatibility, as kokoro). |

Response: `200` with the raw audio bytes, `Content-Type: audio/wav`, and:

```
X-TTS-Model-Version: omnivoice-0.2.1+k2-fsa/OmniVoice
X-TTS-Cache-Key: <sha256 over model_version, voice, instructions, language, format, speed, text>
X-TTS-Chars: <len(input)>
X-TTS-Duration-Seconds: <samples / sample_rate, 3dp>
X-TTS-Cache-Hit: 0
```

Errors: `{"detail": "..."}` with 400/413/422/500/503 (FastAPI's shape — what
kokoro emits; citadel-cli surfaces the body verbatim).

### `GET /health` — always HTTP 200

```json
{"status": "up" | "loading", "model_loaded": true, "model_version": "...", "slots": 2, "device": "cuda"}
```

Readiness = `model_loaded: true` (the gate citadel-cli's
`synthesizeHealthReady` uses). `last_error` is added after a failed lazy load.

### `GET /info`

```json
{"model": "k2-fsa/OmniVoice", "adapter": "omnivoice", "task": "tts",
 "model_license": "CC-BY-NC", "voices": ["auto", "female", "male", "..."],
 "model_loaded": true, "model_version": "...", "revision": null,
 "default_voice": "auto", "default_format": "wav", "formats": ["wav"],
 "capacity": {"slots": 2, "max_input_chars": 5000},
 "adapter_info": {"device": "cuda", "dtype": "float16", "num_step": 32, "sample_rate": 24000, "voice_cloning": false},
 "cis_version": "0.1.0", "load_seconds": 41.2}
```

The first five keys are the pinned cross-phase contract; the rest are additive.

## OmniVoice notes

- Loaded once at startup: `OmniVoice.from_pretrained(<local snapshot>, device_map=<device>, dtype=<torch dtype>)`;
  `model.generate(text=, instruct=, language=, speed=, num_step=)` → `list[np.ndarray]` @ `model.sampling_rate` (24 kHz).
- **Voice cloning (`ref_audio`/`ref_text`) is deferred to v2.** The adapter
  never passes a cloning kwarg and never asks `from_pretrained` for the ASR
  model, so no Whisper download can occur (`tests/test_omnivoice_adapter.py`
  asserts key absence).
- `voice` presets are composed only of the attribute items upstream's
  `_resolve_instruct` accepts (gender / age / pitch / whisper / accent).
- The checkpoint's license is reported as the model card states it —
  `CC-BY-NC`, no version given upstream. The code (`pip install omnivoice`)
  is Apache-2.0.

## Development

```bash
uv sync --extra dev
uv run ruff check .
HF_HUB_OFFLINE=1 uv run pytest -q         # hermetic: stub adapter + fake OmniVoice; no GPU, no download

docker build -f images/base/Dockerfile -t cis:base .
docker build -f images/tts/Dockerfile --build-arg BASE_IMAGE=cis:base -t cis:tts .
scripts/smoke.sh cis:tts                  # boots the image with CIS_ADAPTER=stub, checks the contract
```

CI (`.github/workflows/ci.yml`) runs the suite and builds + smoke-tests both
images on every PR without pushing; `publish.yml` pushes
`ghcr.io/aceteam-ai/citadel-inference-server:{base,tts}` (+ `-<sha>` tags) on
every push to `main`.

## License

Elastic License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE) (the
AceTeam org standard, matching citadel-cli).
