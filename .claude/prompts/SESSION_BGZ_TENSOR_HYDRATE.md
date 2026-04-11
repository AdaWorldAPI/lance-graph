# SESSION: bgz-tensor Hydration

## STATUS: REFERENCE STUB

File was previously a 0-byte placeholder created in `5cd2713` and cleaned up
in `9eee5a6`. Restored with minimal content grounding the hydration workflow
so the file has a reason to exist in `prompts/`. Expand as needed when a
real session lands on hydration work.

## CONTEXT

**Repo**: `AdaWorldAPI/lance-graph`
**Crate**: `crates/bgz-tensor/` (standalone, 0 deps)
**Binary**: `crates/bgz-tensor/src/hydrate.rs` (224 LOC, feature-gated)
**Data source**: `data/manifest.json` — SHA256 for all 41 bgz7 shards
**Release**: GitHub Release `v0.1.0-bgz-data` — 41 bgz7 assets, 685 MB total

Hydration is the process of getting precomputed bgz7 tensor indexes onto disk
so bgz-tensor can serve attention-as-table-lookup without running a GGUF
forward pass. The hydrate binary is the CLI surface for that workflow.

## FEATURE-FLAG-GATED MODELS

From `hydrate.rs:49-54`:

```
qwen35-9b       80 MB  — quick thinking, shallow routing
qwen35-27b-v1  174 MB  — Opus 4.5 behavior (deep reasoning)
qwen35-27b-v2  174 MB  — Opus 4.6 precision (code/format)
qwen35-full    430 MB  — all variants
```

Zero download by default. Enable a feature flag to opt in.

## COMMANDS

```bash
# Show all models and hydration status
cargo run -p bgz-tensor --features hydrate --bin hydrate -- --list

# Download all feature-enabled models
cargo run -p bgz-tensor --features hydrate --bin hydrate -- --download

# Download a specific model
cargo run -p bgz-tensor --features hydrate --bin hydrate -- --download qwen35-9b

# Stream from HuggingFace, build bgz7 locally (skips GitHub release)
cargo run -p bgz-tensor --features hydrate --bin hydrate -- --reindex qwen35-9b

# Check SHA256 of existing shards
cargo run -p bgz-tensor --features hydrate --bin hydrate -- --verify qwen35-9b
```

## KEY API SURFACE

`bgz_tensor::manifest` module (used by `hydrate.rs`):

- `load_manifest()` — read `data/manifest.json`
- `is_hydrated(name, shard_count)` — presence + count check
- `is_enabled(name)` — feature-flag check
- `enabled_models()` — list of currently enabled names
- `bgz7_path(name, shard)` — on-disk path for a given shard
- `verify_sha256(path, expected)` — integrity check

## SHARD LAYOUT

Each enabled model is split into N shards; hydration downloads or streams
them in parallel. The 41 total shards across all models map to HuggingFace
safetensor slices, with a manifest.json mapping (model, shard_idx) →
(bgz7_filename, sha256, size_bytes).

## WHAT A HYDRATION SESSION WOULD COVER

Possible scopes for a future real session on this prompt file:

1. **Resumable downloads**: handle partial shards gracefully on network failure.
2. **Parallel shard fetch**: currently sequential; AVX/tokio-friendly concurrency.
3. **Lazy mmap hydration**: don't download until a shard is actually queried.
4. **CI hydration**: GitHub Action that hydrates enabled models into release
   cache, so downstream builds are fast.
5. **Verification replay**: post-hydration SHA256 scan before trusting the cache.
6. **Railway hydration budget**: 685 MB is larger than runtime RAM budget;
   hydrate at build-time (32 GB phase), runtime mmap-only.

## RELATED

- `crates/bgz-tensor/data/manifest.json` — shard manifest
- `crates/bgz-tensor/src/hhtl_cache.rs` — consumer (HHTL k=64/k=256 cache)
- `.claude/phases/integration_phases.md § Phase 5.11` — Build-Time vs Runtime
  separation (hydration belongs to Build-Time)

## OUT OF SCOPE

- Building new bgz7 indexes from scratch — that is a separate session on
  the encoding side (`bgz-tensor/src/encode.rs` if/when it exists).
- Changes to the bgz7 format itself — hydration is pure transport.
- Railway deployment orchestration — that is `.claude/phases/` material.
