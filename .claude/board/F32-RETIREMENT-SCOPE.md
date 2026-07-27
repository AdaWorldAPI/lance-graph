# f32 retirement scope — IDENTIFICATION ONLY (2026-07-27)

> **This document scopes; it does not migrate.** Canon (settled 2026-07-27):
> truth / contradiction = **palette256 codes**; comparison = **`[a,b]` reads**;
> floats = **legacy**. Per the operator's implementation gate, migrating any
> value waits until its **resident tenant and write path exist** — neither does
> today (primer §14/§15). This file exists so that when the gate opens, the
> scope is already known and nobody re-derives it.
>
> Source: §12 substrate trace lane D + direct enumeration. Every entry
> `file:line`. **No code changed by this document.**

## Tier 1 — CORE CARRIER (retires first; blocked on a resident tenant)

| site | field | note |
|---|---|---|
| `nars/truth.rs:12` | `TruthValue.frequency: f32` | the NARS frequency; **declared** row home `MetaWord::nars_f`, unwired, 8B/4B width mismatch open |
| `nars/truth.rs:14` | `TruthValue.confidence: f32` | the NARS confidence; declared home `MetaWord::nars_c`, same status |
| `nars/belief.rs` | `Belief.contradiction: f32` | preserved dialectic depth (`max \|f₁−f₂\|`); **NO resident tenant exists** |

`TruthValue` is `Copy` and two `f32`s wide; it is the single carrier through
every tactic, the semiring, and the arena's revision path. **Retiring these three
fields IS the value migration** — everything in tier 2 is downstream arithmetic
that follows automatically once the carrier is palette-coded.

## Tier 2 — DOWNSTREAM DERIVED SCALARS (follow the carrier)

Computed *from* tier-1 values; no independent storage decision.

- `nars/insight.rs` — `coherence` (`:158`), `wonder` (`:169`),
  `confidence_entropy` (`:180`), `ratio` (`:202`), `GraphSignals.{coherence,
  wonder}` (`:45,48`), `revision_velocity` params (`:95,121,135`),
  `InsightMush.{insight, mush}` (`:221,226`), `detect(.., yield_theta)` (`:248`).
- `nars/basin_resonance.rs` — `{resonance, staunen, wisdom, evidence}`
  (`:75-81`), `{obs_weight, der_weight, stakes}` (`:114-118`), `stakes()` (`:123`).
- `nars/epiphany.rs` — `rate` (`:34`, `:64`).
- `nars/regulate.rs` — `{yield_theta, elevate_threshold}` (`:27,29`) — **threshold
  parameters**, not stored state; these become palette-index thresholds.

## Tier 3 — TRANSIENT WIDENING (delete, don't port)

- `physical::accumulate::TruthPropagatingSemiring` — an `f32 → f64 → f32`
  round-trip **per `adjacent_truth_propagate` call**. Pure precision theatre over
  values that will be u8 codes; retires with the carrier, ports to nothing.

## Explicitly OUT of scope

- **`QualiaColumn` 18×f32** — a ratified SoA column, not belief state.
- **The uncertainty lane** — the tiny per-edge float `Σ` sandwich (`Σ' = M·Σ·Mᵀ`,
  Pillar 6/7) is a **co-certified sibling** of the integer SELECT lane, not a
  competitor (ndarray `EPIPHANIES.md` 2026-05-26, "Two lanes"). It is certified
  PSD *metadata*, never bulk arithmetic. **Do not retire it.**
- **θ / Fisher-z aperture** — `similarity_z = atanh` is a scalar aperture, not a
  per-value float; the VALIDATED entry places it in the design deliberately.
- **Lab/calibration crates** — table-build float is the amortized reconstruction
  the doctrine permits.

## Gate (unchanged)

Retirement begins only when tier 1 has somewhere to land:

```text
resident belief tenant
  → owner-authorized mutation
  → synchronous Kanban transition
  → ahead-firing descriptor cast
  → new Lance standing-wave position
  → production temporal read
```

First missing link today is still *before* the write:
`MailboxSoA → SoaEnvelope ownership/writer seam → live Lance write path`.

## Order-of-fold caveat DISSOLVES on migration

The census flagged f32 fold-order sensitivity ((a+b)+c ≠ a+(b+c)) as needing a
per-site exact / tolerance / deterministic-order decision. Under palette256 the
algebra is **integer table lookup — exactly associative and commutative** — so
the caveat is *eliminated, not mitigated*. It applies only to the legacy code
above, i.e. only until it retires.
