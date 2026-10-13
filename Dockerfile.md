# lance-graph Docker CPU Detection & SIMD Dispatch

## Build Targets

Production always runs on x86-64-v4 (AVX-512), so every image and the wheel
build for v4.

| Target | File | Flags | Use case |
|---|---|---|---|
| **Deploy (canonical)** | `Dockerfile.avx512` | `-C target-cpu=x86-64-v4` | Railway / production deploys |
| **Same-tier fallback** | `Dockerfile` | `-C target-cpu=x86-64-v4` | Compile/test image; a deploy that forgets the file name still gets v4 |
| **HHTL-D TTS** | `Dockerfile.hhtld` | `.cargo/config-v4.toml` | TTS inference container |
| **Local dev** | `.cargo/config.toml` | `-C target-cpu=native` | Developer machines (host CPU) |
| **CI tests** | `build.yml`, `rust-test.yml`, `style.yml` | `-C target-cpu=x86-64-v3` | Not production; never cite a timing from it |

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

lance-graph's `.cargo/config.toml` sets `target-cpu=native` for local dev.
The test workflows set `RUSTFLAGS="-C debuginfo=0 -C target-cpu=x86-64-v3"`,
which **overrides** config.toml entirely, so those CI binaries target AVX2.

- Local dev: the host CPU (`native`).
- CI tests: AVX2. Production does not run this arm, so a timing from it is
  not evidence.
- Production: x86-64-v4, in every Dockerfile and the wheel. On a host without
  AVX-512, ndarray's startup `cpu_guard` names the missing features and exits
  before `main` instead of dying with SIGILL.

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
Railway / production      → Dockerfile.avx512 (canonical); Dockerfile is the same-tier fallback
TTS inference             → Dockerfile.hhtld (config-v4.toml; downloads codebooks)
Raspberry Pi / ARM        → Dockerfile.hhtld (config-v4.toml applies to x86_64 only)
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
# Production tier (x86-64-v4)
docker build -t lance-graph-test .

# TTS inference
docker build -f Dockerfile.hhtld \
  --build-arg RELEASE_TAG=v0.1.0 \
  -t lance-graph-tts:v0.1.0 .
```
