#!/bin/bash
# bake_hhtld_codebooks.sh — Bake HHTL-D codebooks from Qwen3-TTS-1.7B
#
# Downloads the safetensors from HuggingFace, runs the encoder,
# splits output into per-group codebook files, generates manifest.json,
# and optionally creates a GitHub release.
#
# Usage:
#   ./scripts/bake_hhtld_codebooks.sh [--release v0.1.0]
#
# Requirements:
#   - Rust 1.94+ with cargo
#   - ~4 GB free disk for model download
#   - ndarray crate at ../ndarray (or set NDARRAY_PATH)
#
# Output in ./codebooks/:
#   manifest.json
#   palettes/talker_gate.palette.bgz     (per-group palette + distance + route)
#   palettes/talker_up.palette.bgz
#   ...
#   entries/talker_gate.entries.bin       (flat HhtlDEntry × N rows)
#   entries/talker_up.entries.bin
#   ...
#   passthrough/norms.bin                (all norm weights concatenated)
#   passthrough/biases.bin
#   model_hhtld.safetensors              (single-file alternative)

set -euo pipefail

MODEL_ID="Qwen/Qwen3-TTS-12Hz-1.7B-Base"
MODEL_DIR="./models/Qwen3-TTS-12Hz-1.7B-Base"
CODEBOOK_DIR="./codebooks"
RELEASE_TAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --release) RELEASE_TAG="$2"; shift 2 ;;
        --model-dir) MODEL_DIR="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

echo "═══ HHTL-D CODEBOOK BAKER ═══"
echo ""
echo "  Model:   $MODEL_ID"
echo "  Dir:     $MODEL_DIR"
echo "  Output:  $CODEBOOK_DIR"
echo ""

# ─── Step 1: Download model if needed ──────────────────────────────
if [ ! -f "${MODEL_DIR}/model.safetensors" ]; then
    echo "[1] Downloading model from HuggingFace..."
    mkdir -p "$MODEL_DIR"

    # Download safetensors
    curl -L --progress-bar \
        "https://huggingface.co/${MODEL_ID}/resolve/main/model.safetensors" \
        -o "${MODEL_DIR}/model.safetensors"

    # Download config
    curl -sL "https://huggingface.co/${MODEL_ID}/resolve/main/config.json" \
        -o "${MODEL_DIR}/config.json"

    echo "  Downloaded: $(du -h "${MODEL_DIR}/model.safetensors" | cut -f1)"
else
    echo "[1] Model already at ${MODEL_DIR}/model.safetensors"
    echo "  Size: $(du -h "${MODEL_DIR}/model.safetensors" | cut -f1)"
fi

# ─── Step 2: Run HHTL-D encoder ───────────────────────────────────
echo ""
echo "[2] Running HHTL-D encoder..."
cargo run --release --example tts_17b_hhtld_encode \
    --manifest-path crates/thinking-engine/Cargo.toml \
    -- "${MODEL_DIR}/model.safetensors"

HHTLD_FILE="${MODEL_DIR}/model_hhtld.safetensors"
if [ ! -f "$HHTLD_FILE" ]; then
    echo "ERROR: Encoder did not produce ${HHTLD_FILE}"
    exit 1
fi
echo "  Encoded: $(du -h "$HHTLD_FILE" | cut -f1)"

# ─── Step 3: Split into per-group files ────────────────────────────
echo ""
echo "[3] Splitting into per-group codebook files..."
mkdir -p "${CODEBOOK_DIR}/palettes" "${CODEBOOK_DIR}/entries" "${CODEBOOK_DIR}/passthrough"

# Use Python to split the safetensors into individual files
python3 << PYEOF
import struct, json, os

st_path = "${HHTLD_FILE}"
out_dir = "${CODEBOOK_DIR}"

with open(st_path, 'rb') as f:
    header_size = struct.unpack('<Q', f.read(8))[0]
    header = json.loads(f.read(header_size).decode())
    data_offset = 8 + header_size

    meta = header.pop('__metadata__', {})
    manifest = {
        'encoding': meta.get('encoding', 'bgz-hhtl-d'),
        'version': int(meta.get('version', '1')),
        'original_model': meta.get('original_model', ''),
        'palette_k': int(meta.get('palette_k', '256')),
        'compression_ratio': float(meta.get('compression_ratio', '0')),
        'total_output_bytes': int(meta.get('total_output_bytes', '0')),
        'groups': {},
        'passthrough': [],
    }

    for name, info in sorted(header.items()):
        offsets = info['data_offsets']
        begin, end = offsets[0], offsets[1]
        size = end - begin

        f.seek(data_offset + begin)
        data = f.read(size)

        if name.startswith('passthrough.'):
            real_name = name[len('passthrough.'):]
            pt_path = os.path.join(out_dir, 'passthrough', real_name.replace('.', '_') + '.bin')
            with open(pt_path, 'wb') as out:
                out.write(data)
            manifest['passthrough'].append({
                'name': real_name,
                'file': os.path.basename(pt_path),
                'dtype': info['dtype'],
                'shape': info['shape'],
                'bytes': size,
            })
        else:
            # Parse: role_name.tensor_type
            parts = name.rsplit('.', 1)
            if len(parts) == 2:
                role_name, tensor_type = parts
            else:
                role_name, tensor_type = name, 'data'

            if role_name not in manifest['groups']:
                manifest['groups'][role_name] = {'files': {}, 'bytes': 0}

            if tensor_type in ('palette', 'distance_table', 'route_table', 'hip_families', 'gamma_meta'):
                fpath = os.path.join(out_dir, 'palettes', f'{role_name}.{tensor_type}.bin')
            elif tensor_type == 'hhtld_entries':
                fpath = os.path.join(out_dir, 'entries', f'{role_name}.entries.bin')
            elif tensor_type == 'original_shape':
                fpath = os.path.join(out_dir, 'entries', f'{role_name}.shape.bin')
            else:
                fpath = os.path.join(out_dir, 'entries', f'{role_name}.{tensor_type}.bin')

            with open(fpath, 'wb') as out:
                out.write(data)

            manifest['groups'][role_name]['files'][tensor_type] = {
                'file': os.path.relpath(fpath, out_dir),
                'bytes': size,
                'shape': info['shape'],
            }
            manifest['groups'][role_name]['bytes'] += size

    # Copy the single-file version too
    import shutil
    shutil.copy2(st_path, os.path.join(out_dir, 'model_hhtld.safetensors'))

    # Write manifest
    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w') as out:
        json.dump(manifest, out, indent=2)

    total_palette = sum(
        sum(f['bytes'] for k, f in g['files'].items() if k != 'hhtld_entries')
        for g in manifest['groups'].values()
    )
    total_entries = sum(
        g['files'].get('hhtld_entries', {}).get('bytes', 0)
        for g in manifest['groups'].values()
    )
    total_pt = sum(p['bytes'] for p in manifest['passthrough'])
    n_groups = len(manifest['groups'])

    print(f"  Groups:      {n_groups}")
    print(f"  Palettes:    {total_palette:,} B ({total_palette/1024:.1f} KB)")
    print(f"  Entries:     {total_entries:,} B ({total_entries/1024:.1f} KB)")
    print(f"  Passthrough: {total_pt:,} B ({total_pt/1024:.1f} KB)")
    print(f"  Manifest:    {manifest_path}")
PYEOF

# ─── Step 4: Create tar archive for release ────────────────────────
echo ""
echo "[4] Creating release archive..."
ARCHIVE="${CODEBOOK_DIR}/qwen3-tts-1.7b-hhtld.tar.gz"
tar -czf "$ARCHIVE" -C "$CODEBOOK_DIR" \
    manifest.json palettes/ entries/ passthrough/ model_hhtld.safetensors
echo "  Archive: $(du -h "$ARCHIVE" | cut -f1)"

# ─── Step 5: GitHub release (optional) ─────────────────────────────
if [ -n "$RELEASE_TAG" ]; then
    echo ""
    echo "[5] Creating GitHub release ${RELEASE_TAG}..."

    GH_TOKEN="${GH_TOKEN:-}"
    REPO="AdaWorldAPI/lance-graph"

    if [ -z "$GH_TOKEN" ]; then
        echo "  ERROR: Set GH_TOKEN env var for release upload"
        exit 1
    fi

    # Create release
    RELEASE_ID=$(curl -s -H "Authorization: token $GH_TOKEN" \
        -H "Content-Type: application/json" \
        -d "{
            \"tag_name\": \"${RELEASE_TAG}\",
            \"name\": \"BGZ-HHTL-D Codebooks: Qwen3-TTS-1.7B (${RELEASE_TAG})\",
            \"body\": \"HHTL-D encoded codebooks for Qwen3-TTS-12Hz-1.7B-Base.\n\nCompression: 3.86 GB → 11.2 MB (343:1).\nPalette groups: 26 shared palettes.\n\nDownload the archive or pull individual files.\n\n\`\`\`sh\ndocker pull ghcr.io/adaworldapi/lance-graph-tts:${RELEASE_TAG}\n\`\`\`\",
            \"draft\": false,
            \"prerelease\": false
        }" \
        "https://api.github.com/repos/$REPO/releases" | python3 -c "import sys,json; print(json.load(sys.stdin)['id'])")

    echo "  Release ID: $RELEASE_ID"

    # Upload archive
    echo "  Uploading archive..."
    curl -s -H "Authorization: token $GH_TOKEN" \
        -H "Content-Type: application/gzip" \
        --data-binary @"$ARCHIVE" \
        "https://uploads.github.com/repos/$REPO/releases/$RELEASE_ID/assets?name=qwen3-tts-1.7b-hhtld.tar.gz" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Uploaded: {d[\"name\"]} ({d[\"size\"]/1e6:.1f} MB)')"

    # Upload single-file safetensors
    echo "  Uploading safetensors..."
    curl -s -H "Authorization: token $GH_TOKEN" \
        -H "Content-Type: application/octet-stream" \
        --data-binary @"${CODEBOOK_DIR}/model_hhtld.safetensors" \
        "https://uploads.github.com/repos/$REPO/releases/$RELEASE_ID/assets?name=model_hhtld.safetensors" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Uploaded: {d[\"name\"]} ({d[\"size\"]/1e6:.1f} MB)')"

    # Upload manifest separately (for selective download)
    echo "  Uploading manifest..."
    curl -s -H "Authorization: token $GH_TOKEN" \
        -H "Content-Type: application/json" \
        --data-binary @"${CODEBOOK_DIR}/manifest.json" \
        "https://uploads.github.com/repos/$REPO/releases/$RELEASE_ID/assets?name=manifest.json" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'  Uploaded: {d[\"name\"]}')"

    echo "  Release: https://github.com/$REPO/releases/tag/${RELEASE_TAG}"
fi

echo ""
echo "═══ DONE ═══"
echo ""
echo "Files in ${CODEBOOK_DIR}/:"
find "$CODEBOOK_DIR" -type f | sort | while read f; do
    echo "  $(du -h "$f" | cut -f1)  $f"
done
