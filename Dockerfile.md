# lance-graph Docker CPU Detection & SIMD Dispatch

## Three-Tier Build Strategy

| Target | Dockerfile | RUSTFLAGS | Use case |
|---|---|---|---|
| **Portable (AVX2)** | `Dockerfile` | `-C target-cpu=x86-64-v3` | GitHub CI, general servers |
| **AVX-512 pinned** | `Dockerfile.avx512` | `-C target-cpu=x86-64-v4` | Production (Skylake-X+) |
| **HHTL-D TTS** | `Dockerfile.hhtld` | (inherits) | TTS inference container |
| **Local dev** | `.cargo/config.toml` | `-C target-cpu=x86-64-v4` | Developer machines |

## How lance-graph Uses SIMD

lance-graph delegates all SIMD work to **ndarray** (mandatory dependency).
ndarray's `src/simd.rs` polyfill provides the dispatch:

```
Consumer code (lance-graph):
  ndarray::hpc::bitwise::hamming_distance_raw(a, b)
  ndarray::simd::F32x16::mul_add(b, c)
  ndarray::hpc::renderer::integrate_simd(pos, vel, dt, damp)

Polyfill (ndarray simd.rs):
  ┌─────────────────────────┐
  │ compile-time target_cpu │
  ├─────────┬───────────────┤
  │ v4      │ v3 / lower    │
  ├─────────┼───────────────┤
  │ __m512  │ 2× __m256 or  │
  │ native  │ scalar loop   │
  └─────────┴───────────────┘
  +
  ┌──────────────────────────────┐
  │ runtime LazyLock<Tier>       │
  │ is_x86_feature_detected!()  │
  │ → per-function AVX-512 even │
  │   when compiled at v3       │
  └──────────────────────────────┘
```

### What lance-graph calls from ndarray SIMD

| lance-graph location | ndarray function | What it does |
|---|---|---|
| `driver.rs` (shader hot loop) | `bitwise::hamming_distance_raw` | Content-plane Hamming pre-pass (16K-bit fingerprints) |
| `vector_ops.rs` (DataFusion UDF) | `bitwise::hamming_distance_raw` | SQL `hamming_distance()` function |
| `fingerprint.rs` (graph) | `bitwise::hamming_distance_raw` | Graph fingerprint similarity |
| `blasgraph/types.rs` | Own AVX-512/AVX2 Hamming | Hand-rolled (predates ndarray integration) |

### `.cargo/config.toml` vs CI RUSTFLAGS

**Important:** `RUSTFLAGS` env var **replaces** (not appends to) the `rustflags`
array in `.cargo/config.toml`. This is a Cargo design decision.

lance-graph's `.cargo/config.toml` sets `target-cpu=x86-64-v4` for local dev.
CI workflows set `RUSTFLAGS="-C debuginfo=1 -C target-cpu=x86-64-v3"` which
**overrides** config.toml entirely. The CI binary targets AVX2.

This is intentional:
- Local dev: maximum SIMD (AVX-512, everything inlined)
- CI: portable (AVX2, runtime detection for anything higher)
- Production Docker: choose `Dockerfile` (AVX2) or `Dockerfile.avx512`

## AMX Detection

Intel AMX (Sapphire Rapids+) is detected at runtime by ndarray:
`ndarray::hpc::amx_matmul::amx_available()` checks CPUID + OS XSAVE support.
AMX kernels are always compiled in and gated at call sites. No Dockerfile
or RUSTFLAGS change needed — it works with any `target-cpu`.

## NEON (ARM / aarch64 / Raspberry Pi)

ndarray detects NEON automatically on aarch64 (it's mandatory). The `dotprod`
extension (Pi 5 / A76+) is runtime-detected for 4× int8 throughput.
lance-graph inherits this via ndarray; no ARM-specific configuration needed.

## Choosing the Right Dockerfile

```
GitHub CI / PR checks     → Dockerfile (AVX2, -C target-cpu=x86-64-v3)
Railway / production      → Dockerfile.avx512 (-C target-cpu=x86-64-v4)
TTS inference             → Dockerfile.hhtld (downloads codebooks + runs decoder)
Raspberry Pi / ARM        → Dockerfile (NEON auto-detected at runtime)
Maximum compatibility     → docker build --build-arg RUSTFLAGS="-C target-cpu=x86-64"
```

## Verifying CPU Features

```bash
# Inside the container:
cat /proc/cpuinfo | grep -oP 'avx512\w+' | sort -u

# From Rust (ndarray):
use ndarray::hpc::simd_caps::simd_caps;
println!("{:?}", simd_caps());  // CpuCaps { avx512: true, avx2: true, fma: true, ... }
```

## Build Examples

```bash
# Default (AVX2) — safe everywhere
docker build -t lance-graph-test .

# AVX-512 pinned — production servers
docker build -f Dockerfile.avx512 -t lance-graph-avx512 .

# TTS inference
docker build -f Dockerfile.hhtld \
  --build-arg RELEASE_TAG=v0.1.0 \
  -t lance-graph-tts:v0.1.0 .
```
