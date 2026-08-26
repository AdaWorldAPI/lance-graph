## 2026-08-19 — E-E2-REVERIFIED-SCATTER-CONTESTED-PMU-ABSENT-1

**Status:** FINDING (independent re-verification of E2, Tier 0).

E2's benches were standalone `rustc -O` programs (no cargo — the brief's
letter held); sources now committed under
`crates/rp-seal-t0-probe/standalone/` with `run.sh`. Independent fresh
compile + min-of-3 warm runs:

**CONFIRMED:** FNV seal invariance (41.5/41.4/41.2 ms at b=1/4/64 —
does not amortize, forbids incremental; decision stands); b+1 byte
amplification (arithmetic + source-anchored, unchanged); the durable
boundary orders (small fsync ~0.2-0.3 ms, 32 MiB ~285 ms, per-petal
4096-file cycle ~1.5 s — non-viable, stands).

**NOISY:** the arrow-column timing at b=64 spans 17.6-206 ms across
runs — quote the bytes formula, never this timing.

**CONTESTED:** E2's self-disproof "random placement is FREE at 32 MiB
(0.97x)" does NOT reproduce — this run: 2.66x/2.91x/3.83x at
32/128/512 MiB (gradient reproduces, absolute does not). Resolution =
E2's own multi-machine bench3 sweep. Nothing rests on it: placement is
optional economics per §0 either way.

**Cache-miss metric STRUCK for container probes, on measurement:** this
VM exposes NO cpu PMU (`/sys/bus/event_source/devices/` = software/
tracepoint/breakpoint/msr/power only) — hardware counters are
impossible here, not merely unwired. Bare-metal follow-on recorded.
Discharges M13's "wire perf-event or strike the metric".

**Tier 0 state:** T0.1-T0.3 + X-C2-1 + E2-reverify + PMU decision all
landed. E3's locality-debt metric rides the optional-economics track
(non-gating per §0) as that track's opener if placement work is ever
wanted.

