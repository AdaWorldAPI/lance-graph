# 2026-10-13 — Production builds target x86-64-v4 (D-PROD-V4)

**Status:** DONE (build targets); the CI test tier and probe guards are OPEN.

**DECISION:** everything shipped from this tree builds for x86-64-v4: the
Dockerfiles, the thinking-engine images and the wheel. There is no v3
production variant.

**BASIS:** production always runs on v4. Measured exposure: Railway hosts
about 95% v4, Claude Code sessions about 70% v4. A v3 build ships the arm
production never runs, and a timing measured on it is not evidence for
production.

**SCOPE:**

| site | before | after |
|---|---|---|
| `Dockerfile` | `RUSTFLAGS -C target-cpu=x86-64-v3` | `native` (fallback; the build host's CPU) |
| `Dockerfile.avx512` (canonical deploy) | `x86-64-v4` | `sapphirerapids` (Railway hosts are Sapphire Rapids, Emerald Rapids or newer, per the maintainer) |
| `Dockerfile.hhtld`, `crates/cognitive-stack/Dockerfile`, `crates/symbiont/Dockerfile` | `cp config-v3.toml config.toml` | `cp config-v4.toml config.toml` (x86_64 only; arm64 untouched) |
| `crates/thinking-engine/Dockerfile`, `Dockerfile.railway` | `CARGO_TARGET_X86_64_…_RUSTFLAGS x86-64-v3` | `x86-64-v4` |
| `python/Makefile` (`make publish`) | `x86-64-v3` on x86_64 hosts | `x86-64-v4` |
| new `.cargo/config-v4.toml` | — | `-Ctarget-cpu=x86-64-v4`, mirrors ndarray's file without `-Dwarnings` |

**Measured:** `cargo --config .cargo/config-v4.toml` JOINS with
`.cargo/config.toml`'s `native` (rustc receives `-Ctarget-cpu=native
-Ctarget-cpu=x86-64-v4`). The later flag wins: the cfg equals v4 alone
(23 target features), not native (32 on this Cascade Lake host). The
Dockerfiles copy the file over `config.toml`, so they never join.

**Debug 0 in CI (reported by the maintainer, not re-measured here):** a
build without debug info drops from 16-19 GB to 3.5 GB of disk and from
22-42 to 4-9 minutes. With `--target`, RUSTFLAGS no longer reaches build
scripts and proc-macros (measured: a build script compiled with
`-C debuginfo=2`), so the v4 job also sets `CARGO_PROFILE_DEV_DEBUG=0` and
`CARGO_PROFILE_TEST_DEBUG=0`.

**Safety:** on a host without AVX-512, ndarray's startup `cpu_guard` names
the missing features and exits before `main`; the binary does not die with
SIGILL mid-run. That holds only for binaries that link ndarray.

**OPEN:**
- `build.yml` gains `linux-build-v4`: it compiles and links lance-graph's
  tests at x86-64-v4 with `--target`, so build scripts stay at the host
  baseline, and runs nothing (most GitHub runners are v3). The v4 arm is now
  compiled in CI but still never executed there.
- `build.yml`, `rust-test.yml` and `style.yml` still RUN tests on a v3 pin
  instead of per-runner `native` (the #1526 resolver).
- Probes and examples do not yet refuse to report timings off v4.
- `Dockerfile.avx512` builds `sapphirerapids`; its build stage runs the
  tests, so that arm executes on the Railway build host and nowhere else.
  CI compiles only the v4 arm; the Sapphire Rapids-only code (VNNI, BF16,
  FP16) is not compiled in lance-graph CI.
- Plain `Dockerfile` builds `native` (maintainer's decision). It tunes to the
  build host; if the run host lacks a compiled-in feature, ndarray's startup
  `cpu_guard` stops the binary before `main`, and ndarray#360 warns at build
  time about enabled features the guard has no row for.
- Neither Dockerfile provides the ndarray sibling (`/ndarray`) that the
  default features and `lance-graph-planner` require, so as written they
  cannot resolve dependencies; `crates/symbiont/Dockerfile` clones it.
- Not verified with `docker build` here.

**REVISIT WHEN:** a production host without AVX-512 appears.
