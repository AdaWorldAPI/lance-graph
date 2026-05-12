#!/usr/bin/env bash
#
# codec_sweep.sh — curl-driven lab iteration for codec sweeps.
#
# Reads a YAML config under configs/codec/, converts to JSON via yq, POSTs
# to the running shader-lab's /v1/shader/sweep endpoint, pretty-prints the
# WireSweepResponse.
#
# Prereqs: yq (mikefarah/yq ≥ v4), curl, jq, a running shader-lab binary
# on SHADER_LAB_URL (default http://localhost:3001).
#
# Usage:
#   scripts/codec_sweep.sh configs/codec/00_pr220_baseline.yaml
#   SHADER_LAB_URL=http://10.0.0.5:3001 scripts/codec_sweep.sh configs/codec/10_wider_codebook.yaml
#
# Output: JSON WireSweepResponse with per-grid-point stub results (until
# D2.2 lands real decode-and-compare). Every result row has stub:true.

set -euo pipefail

SHADER_LAB_URL="${SHADER_LAB_URL:-http://localhost:3001}"
ENDPOINT="${SHADER_LAB_URL}/v1/shader/sweep"

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <yaml-config>"
    echo "  example: $0 configs/codec/00_pr220_baseline.yaml"
    exit 2
fi

CONFIG="$1"

if [[ ! -f "$CONFIG" ]]; then
    echo "config not found: $CONFIG" >&2
    exit 2
fi

if ! command -v yq >/dev/null 2>&1; then
    echo "yq not installed — install mikefarah/yq (https://github.com/mikefarah/yq)" >&2
    exit 2
fi

# Convert YAML → JSON; POST to endpoint; pretty-print response.
json_body=$(yq -o=json '.' "$CONFIG")

echo "=== POST $ENDPOINT ==="
echo "--- request ---"
echo "$json_body" | jq '.'
echo
echo "--- response ---"

response=$(curl -sS -X POST "$ENDPOINT" \
    -H "Content-Type: application/json" \
    -d "$json_body")

echo "$response" | jq '.'

echo
echo "=== Stub honesty check ==="
# Per EPIPHANIES.md 2026-04-20 "D0.2 stub flag is anti-#219 defense at
# the type level" — the check MUST fail the script (not just log) when
# the flag is absent or false. Until D2.2 lands real decode-and-compare,
# Phase 0/2 runs return stub:true. A non-stub response here means
# either the wrong endpoint was hit, the response was malformed, or
# (worst case) the server silently shipped non-stub code and this
# script is now pretending synthetic numbers are real.

stub_flag=$(echo "$response" | jq -r '.results[0].stub // "missing"')
echo "results[0].stub = $stub_flag"

case "$stub_flag" in
    true)
        echo "OK — Phase 0 stub honored. (D2.2 will flip this to false when real decode lands;"
        echo "     at that point, flip this check too.)"
        ;;
    false)
        echo "FAIL — results[0].stub is false but D2.2 has not landed." >&2
        echo "        This script refuses to treat non-stub output as real during Phase 0." >&2
        echo "        Either the server is running non-scaffold code (update this check)," >&2
        echo "        or the request hit the wrong endpoint / unexpected handler." >&2
        exit 3
        ;;
    *)
        echo "FAIL — results[0].stub missing or unparseable (got: $stub_flag)." >&2
        echo "        Response may be malformed or an error payload." >&2
        echo "        Inspect the --- response --- section above." >&2
        exit 3
        ;;
esac
