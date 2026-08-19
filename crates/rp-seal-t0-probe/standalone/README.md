# E2 re-verification — standalone economics benches (Tier 0)

E2's cost-model numbers were produced with these standalone programs
(compiled with bare `rustc -O`, no cargo — the researcher brief's letter).
This directory commits the sources so the numbers stay re-runnable; the
re-verification below was an INDEPENDENT fresh compile + run
(min of 3 warm passes; the first cold pass is first-touch-fault dominated
and must not be quoted). Wall-clock is legitimate here — this IS the
economics probe (T0.3: "wall-clock latency is an economics metric").

Run: `./run.sh` (compiles to a temp dir, runs all four).

## Re-verification results (2026-08-19, this container, min-of-3 warm)

**CONFIRMED — the decision-bearing claims:**
- **FNV invariance + cost** (`bench.rs`): `content_hash`-shaped serial
  FNV-1a over the 32 MiB payload = **41.5 / 41.4 / 41.2 ms at
  b = 1 / 4 / 64** — invariant under coalescing exactly as E2 measured
  (E2: 47.6/47.5/47.4; ≈0.79 vs 0.70 GB/s — same machine, different
  load). The 8-lane ILP variant is ~2× (ceiling probe only, not a valid
  digest). The claim that the seal's dominant CPU term does not amortize
  and structurally forbids an incremental variant STANDS.
- **b+1 byte amplification**: arithmetic ((1+C+D)·512) + E2's
  source-anchored finding that Lance does not squeeze the zeros back out
  at the pinned column — not machine-dependent; unchanged.
- **Durable boundary** (`bench4.rs`): small append+fsync 0.22–0.32 ms
  (E2: 0.13–0.16 — same order), 32 MiB append+fsync 285 ms (E2: 202),
  per-file tax ~0.36–0.45 ms/file → 4096 files/cycle ≈ 1.49 s (E2:
  0.91 s). The one-fsync-per-cycle amortization boundary and the
  per-petal-fragment non-viability both STAND (orders, not decimals).

**NOISY — flagged, not decision-bearing:**
- `arrow_payload_col` at b=64 varies **17.6–206 ms** across runs
  (allocator/page-reclaim noise at ~32 MiB); best-case matches E2's
  13.6 ms. Quote the bytes formula, never this timing.

**CONTESTED — E2's own self-disproof does not reproduce:**
- `bench3.rs` scatter (random vs sequential 512-B row writes): this run
  measures **2.66× at 32 MiB / 2.91× at 128 MiB / 3.83× at 512 MiB**
  where E2 recorded 0.97× / 1.41× / 2.30×. The size-gradient
  reproduces; the "random placement is FREE at 32 MiB" absolute does
  not (2.66× is not free). Shared-vCPU noise cuts both ways; E2's own
  prescription — the bench3 sweep on ≥3 machines with distinct L3 — is
  the resolution path. No decision rests on it: placement is optional
  storage economics per the §0 STORNO either way.

## The cache-miss metric: STRUCK for container probes (measured)

`/sys/bus/event_source/devices/` in this environment exposes NO `cpu`
PMU (only software/tracepoint/breakpoint/msr/power) — hardware
cache-miss counting is impossible here, not merely unwired. Per the
charter's "do not optimize a metric that was not measured": the metric
is struck for container-based probes and recorded as a bare-metal
follow-on. perf-event wiring lands only with hardware that exposes it.
