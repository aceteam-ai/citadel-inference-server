# citadel-inference-server (CIS)

Generic, config-driven inference server for the **AceTeam Sovereign Compute Fabric**.

> **Status: WIP / P0 (scaffolding).** Design of record:
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

- **Base runtime** (`cis/`): the ~80% every existing sidecar already shares —
  env-driven config, HuggingFace cache management, lazy single-flight model
  load, `/health` (`{status, model_loaded}`), `/info`, a request semaphore,
  receipt headers, disk preflight, self-provisioning.
- **Adapters** (`cis/adapters/<family>.py`): one small file per model **family**
  (TTS, image, OCR, ASR), mapping that family's native call onto the right
  OpenAI-compatible endpoint (`/v1/audio/speech`, `/v1/images/...`, etc.).
  LLM chat is explicitly **out of scope** — vLLM / llama.cpp / ollama already
  *are* that server.
- **Images**: one `:base` image + one thin layer per family for dependency
  isolation.

"Config-only onboarding" holds **within an already-implemented family**; a new
family costs exactly one adapter file here — and never anything in citadel-cli.

First consumer: **OmniVoice** (TTS family), served *kokoro-byte-identically* so
citadel-cli's existing `SYNTHESIZE_SPEECH` path routes to it with only an added
`backend` field.

## Config contract (`CIS_*`)

Compose-native environment variables; self-provisioning by default:

`CIS_TASK`, `CIS_ADAPTER`, `CIS_MODEL`, `CIS_DEVICE`, `CIS_DTYPE`, `CIS_SLOTS`,
`CIS_MAX_INPUT_CHARS`, `CIS_PRELOAD`, `CIS_EXTRA` (JSON), plus per-family knobs.

## Planned layout

```
cis/                  base runtime (config, cache, load, health, receipts)
cis/adapters/         per-family adapters (tts, image, ocr, asr)
images/               per-family Dockerfiles (base + thin layers)
tests/
```

## License

TBD in P0 — will align with the AceTeam org standard for citadel-cli.
