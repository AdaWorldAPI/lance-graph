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
| `Dockerfile` | `RUSTFLAGS -C target-cpu=x86-64-v3` | `x86-64-v4` |
| `Dockerfile.hhtld`, `crates/cognitive-stack/Dockerfile`, `crates/symbiont/Dockerfile` | `cp config-v3.toml config.toml` | `cp config-v4.toml config.toml` (x86_64 only; arm64 untouched) |
| `crates/thinking-engine/Dockerfile`, `Dockerfile.railway` | `CARGO_TARGET_X86_64_…_RUSTFLAGS x86-64-v3` | `x86-64-v4` |
| `python/Makefile` (`make publish`) | `x86-64-v3` on x86_64 hosts | `x86-64-v4` |
| new `.cargo/config-v4.toml` | — | `-Ctarget-cpu=x86-64-v4`, mirrors ndarray's file without `-Dwarnings` |

**Measured:** `cargo --config .cargo/config-v4.toml` JOINS with
`.cargo/config.toml`'s `native` (rustc receives `-Ctarget-cpu=native
-Ctarget-cpu=x86-64-v4`). The later flag wins: the cfg equals v4 alone
(23 target features), not native (32 on this Cascade Lake host). The
Dockerfiles copy the file over `config.toml`, so they never join.

**Safety:** on a host without AVX-512, ndarray's startup `cpu_guard` names
the missing features and exits before `main`; the binary does not die with
SIGILL mid-run. That holds only for binaries that link ndarray.

**OPEN:**
- `build.yml`, `rust-test.yml` and `style.yml` still test the v3 arm only;
  production's AVX-512 code paths have no coverage in lance-graph's main CI.
- Probes and examples do not yet refuse to report timings off v4.
- `Dockerfile.avx512` now builds the same tier as `Dockerfile`; it is kept
  until nothing names it.
- Not verified with `docker build` here.

**REVISIT WHEN:** a production host without AVX-512 appears.
