#!/usr/bin/env bash
# Hermetic smoke test of a BUILT image: boots it with the dependency-free
# `stub` adapter (no GPU, no network, no model pull) and exercises the full
# HTTP contract citadel-cli's SynthesizeSpeechHandler speaks.
#
#   scripts/smoke.sh <image> [host-port]
#
# Exit non-zero on any contract violation. Used by CI on every PR (against the
# locally-built :tts image) and before publishing.
set -euo pipefail

IMAGE="${1:?usage: smoke.sh <image> [host-port]}"
HOST_PORT="${2:-18000}"
NAME="cis-smoke-$$"
BASE="http://127.0.0.1:${HOST_PORT}"

cleanup() {
    echo "--- container logs (${NAME}) ---" >&2
    docker logs "${NAME}" >&2 2>&1 || true
    docker rm -f "${NAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

echo "smoke: starting ${IMAGE} as ${NAME} on :${HOST_PORT} (adapter=stub)"
docker run -d --name "${NAME}" -p "${HOST_PORT}:8000" \
    -e CIS_ADAPTER=stub -e CIS_MODEL=stub/smoke -e CIS_DEVICE=cpu \
    -e CIS_EXTRA='{"load_delay_s": 1}' \
    "${IMAGE}" >/dev/null

# 1. /health is reachable and ALWAYS 200 (loading -> up), never a 503.
deadline=$((SECONDS + 90))
loaded=""
while [ $SECONDS -lt $deadline ]; do
    if body=$(curl -sf "${BASE}/health" 2>/dev/null); then
        echo "health: ${body}"
        if echo "${body}" | python3 -c 'import sys,json; d=json.load(sys.stdin); sys.exit(0 if d.get("model_loaded") is True and d.get("status")=="up" else 1)'; then
            loaded=1; break
        fi
    fi
    sleep 1
done
[ -n "${loaded}" ] || { echo "FAIL: /health never reported model_loaded:true" >&2; exit 1; }

# 2. /info pinned shape.
info=$(curl -sf "${BASE}/info")
echo "info: ${info}"
echo "${info}" | python3 -c '
import sys, json
d = json.load(sys.stdin)
for k in ("model", "adapter", "task", "model_license", "voices"):
    assert k in d, f"/info missing {k}"
assert d["task"] == "tts" and d["adapter"] == "stub" and d["model"] == "stub/smoke"
assert isinstance(d["voices"], list) and d["voices"][0] == "auto"
assert d["formats"] == ["wav"]
print("info ok")
'

# 3. /v1/audio/speech: wav bytes + the five receipt headers.
hdr=$(mktemp); out=$(mktemp)
code=$(curl -s -o "${out}" -D "${hdr}" -w '%{http_code}' \
    -H 'Content-Type: application/json' \
    -d '{"input":"smoke test","voice":"auto","response_format":"wav"}' \
    "${BASE}/v1/audio/speech")
echo "speech: HTTP ${code}, $(wc -c < "${out}") bytes"
[ "${code}" = "200" ] || { echo "FAIL: expected 200"; cat "${out}"; exit 1; }
head -c 4 "${out}" | grep -q RIFF || { echo "FAIL: body is not RIFF/WAVE"; exit 1; }
grep -qi '^content-type: audio/wav' "${hdr}" || { echo "FAIL: content-type"; cat "${hdr}"; exit 1; }
for h in X-TTS-Model-Version X-TTS-Cache-Key X-TTS-Chars X-TTS-Duration-Seconds X-TTS-Cache-Hit; do
    grep -qi "^${h}: " "${hdr}" || { echo "FAIL: missing header ${h}"; cat "${hdr}"; exit 1; }
done
grep -qi '^X-TTS-Chars: 10' "${hdr}" || { echo "FAIL: X-TTS-Chars"; cat "${hdr}"; exit 1; }

# 4. Rejections are 4xx with a {"detail"} body.
code=$(curl -s -o "${out}" -w '%{http_code}' -H 'Content-Type: application/json' \
    -d '{"input":"x","response_format":"opus"}' "${BASE}/v1/audio/speech")
[ "${code}" = "400" ] || { echo "FAIL: opus should 400, got ${code}"; exit 1; }
grep -q '"detail"' "${out}" || { echo "FAIL: error body shape"; cat "${out}"; exit 1; }
code=$(curl -s -o "${out}" -w '%{http_code}' -H 'Content-Type: application/json' \
    -d '{"input":"x","voice":"not-a-voice"}' "${BASE}/v1/audio/speech")
[ "${code}" = "400" ] || { echo "FAIL: unknown voice should 400, got ${code}"; exit 1; }

rm -f "${hdr}" "${out}"
echo "smoke: PASS"
