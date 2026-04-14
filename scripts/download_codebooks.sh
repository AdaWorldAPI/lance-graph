#!/bin/bash
# download_codebooks.sh — Download HHTL-D codebooks from GitHub release
#
# Selective download: only the palette groups you need.
# Full download: single archive with everything.
#
# Usage:
#   # Full download (11 MB)
#   ./scripts/download_codebooks.sh --all
#
#   # Single-file safetensors only
#   ./scripts/download_codebooks.sh --safetensors
#
#   # Manifest only (inspect what's available)
#   ./scripts/download_codebooks.sh --manifest
#
#   # Specific version
#   ./scripts/download_codebooks.sh --all --tag v0.1.0
#
#   # Custom output dir
#   ./scripts/download_codebooks.sh --all --dir /opt/codebooks

set -euo pipefail

REPO="AdaWorldAPI/lance-graph"
TAG="latest"
OUT_DIR="./codebooks"
MODE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)          MODE="all"; shift ;;
        --safetensors)  MODE="safetensors"; shift ;;
        --manifest)     MODE="manifest"; shift ;;
        --tag)          TAG="$2"; shift 2 ;;
        --dir)          OUT_DIR="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: $0 [--all|--safetensors|--manifest] [--tag TAG] [--dir DIR]"
            exit 0 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ -z "$MODE" ]; then
    echo "Usage: $0 [--all|--safetensors|--manifest] [--tag TAG] [--dir DIR]"
    exit 1
fi

mkdir -p "$OUT_DIR"

if [ "$TAG" = "latest" ]; then
    BASE_URL="https://github.com/${REPO}/releases/latest/download"
else
    BASE_URL="https://github.com/${REPO}/releases/download/${TAG}"
fi

echo "═══ HHTL-D Codebook Download ═══"
echo "  Source: ${REPO} @ ${TAG}"
echo "  Output: ${OUT_DIR}"
echo ""

case $MODE in
    all)
        echo "Downloading full archive..."
        curl -L --progress-bar \
            "${BASE_URL}/qwen3-tts-1.7b-hhtld.tar.gz" | \
            tar xz -C "$OUT_DIR"
        echo ""
        echo "Contents:"
        find "$OUT_DIR" -type f | sort | while read f; do
            echo "  $(du -h "$f" | cut -f1)  $(basename "$f")"
        done
        ;;

    safetensors)
        echo "Downloading model_hhtld.safetensors..."
        curl -L --progress-bar \
            "${BASE_URL}/model_hhtld.safetensors" \
            -o "${OUT_DIR}/model_hhtld.safetensors"
        echo "  $(du -h "${OUT_DIR}/model_hhtld.safetensors" | cut -f1)  model_hhtld.safetensors"
        ;;

    manifest)
        echo "Downloading manifest.json..."
        curl -sL "${BASE_URL}/manifest.json" -o "${OUT_DIR}/manifest.json"
        echo ""
        python3 -c "
import json
m = json.load(open('${OUT_DIR}/manifest.json'))
print(f'Model:       {m[\"original_model\"]}')
print(f'Compression: {m[\"compression_ratio\"]:.0f}:1')
print(f'Palette k:   {m[\"palette_k\"]}')
print(f'Groups:      {len(m[\"groups\"])}')
print(f'Passthrough: {len(m[\"passthrough\"])} tensors')
print()
for name, g in sorted(m['groups'].items()):
    n_files = len(g['files'])
    total = g['bytes']
    print(f'  {name}: {n_files} files, {total:,} B')
" 2>/dev/null || cat "${OUT_DIR}/manifest.json"
        ;;
esac

echo ""
echo "═══ DONE ═══"
