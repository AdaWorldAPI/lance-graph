# PR Sprint-13 — Think Methods (Splat Op Migration to Carrier)

> **PR id:** sprint-13-think-methods
> **D-id:** **D-CSV-14** (per `cognitive-substrate-convergence-v2.md` line 383, ratified into plan v3)
> **Phase:** DESIGN — engineer-ready spec, no code yet
> **Predecessors:** W-F7 / PR #388 (D-CSV-12 scalar splat ops as free fns), sprint-12 merged
> **Branch target:** `claude/sprint-13-think-methods` (sibling of preflight planning branch)
> **Worker:** Sprint-13 PP-4 (this preflight spec authored on `claude/sprint-13-preflight-planning`)
> **Doctrine cite:** CLAUDE.md §"Thinking is a struct" + §"Litmus tests" — "Free function = reject. Method = accept."
> **Knowledge cite:** `.claude/knowledge/splat-shader-rayon-struct-method-vision.md` §3.2 + §6 (the destination shape)

---

## §0  Status and cross-refs

- **Status:** DESIGN. This spec is the engineering contract for D-CSV-14. No
  Rust code is authored in this commit; the implementation PR follows on the
  `claude/sprint-13-think-methods` working branch.
- **Confidence:** HIGH — the migration is mechanical (4 free fns → 4 methods)
  and is the explicit destination called out in the W-F7 module header
  (`splat_ops.rs` lines 5-7) and in the splat-shader vision doc §6.
- **D-id cross-refs:**
  - D-CSV-12 (sprint-12 W-F7) shipped the 4 scalar free fns + 14 tests
    (`crates/thinking-engine/src/splat_ops.rs`).
  - D-CSV-14 (this PR) migrates those free fns to methods on `Think`,
    introduces the `Think` carrier, deprecates the free fns for one release.
  - Future D-ids (sprint-14+): `par_*` rayon variants of the four methods;
    integration with `episodic` / `graph` / `awareness` once those carriers
    land per the Sprint-15+ row of the splat-shader vision §7.
- **Plan v3 cite:** `.claude/plans/cognitive-substrate-convergence-v2.md` line
  62, 383, 499-503, 515 — D-CSV-14 is listed as "Backlog sprint-13+ (on-Think
  method migration for splat ops)". This PR moves D-CSV-14 to **In PR**.
- **Doctrine grounding:**
  - CLAUDE.md "The Click" §"Thinking is a struct" — full quote in §1 below.
  - CLAUDE.md §"Litmus tests": *"Does this add a free function on a carrier's
    state, or a method on the carrier? → Free function = reject. Method = accept."*
  - `.claude/knowledge/splat-shader-rayon-struct-method-vision.md` §3.2: 
    `impl Think { pub fn splat(&mut self, candidate: SpoCandidate) -> SplatResult { ... } ... }`
- **Board-hygiene rule (CLAUDE.md):** This PR MUST update
  `.claude/board/LATEST_STATE.md` Contract Inventory (`Think` struct added)
  and `.claude/board/STATUS_BOARD.md` D-CSV-14 row (Backlog → In PR) in the
  SAME commit as the implementation. Spec-only commit may skip the LATEST
  STATE row but MUST add the D-CSV-14 In PR row when implementation lands.

---

## §1  Statement of scope

**Goal.** Migrate the four splat ops shipped in sprint-12 W-F7 from
free functions over `&mut [SplatField]` to **methods on a new `Think`
carrier struct**, per the CLAUDE.md "Thinking is a struct" doctrine.

The doctrine quote (CLAUDE.md, "The Click" §"Thinking is a struct"):

> *"Thinking is a struct. The universal DTO makes the object do the work.
> Not a function. Not a pipeline. Not a service. A struct whose fields ARE
> the cognitive state and whose methods ARE the inference. `think.resolve()`
> reads its own `trajectory`, computes its own `F`, updates its own
> `awareness`, returns its own `Resolution`. The DTO carries cognition the
> way a photon carries electromagnetism — not as payload, as identity."*

The four ops currently live as free functions per `splat_ops.rs`
(sprint-12 W-F7, merged via PR #388). The W-F7 module header itself
flags this as a transitional scope:

> *"Sprint-12 scope: free functions taking `&mut [SplatField]`. Sprint-13+
> migrates these to methods on the `Think` carrier once the splat field
> is wired into the struct (see knowledge/splat-shader-rayon-struct-
> method-vision.md for the destination shape)."*

This PR fulfils the "Sprint-13+ migrates these to methods on the `Think`
carrier" promise.

**Non-goals (deferred to later D-ids):**

- `par_*` rayon-parallel variants of the four methods — deferred to
  sprint-14 D-id TBD per splat-shader vision §7 row "Sprint-14".
- Wiring `episodic` / `graph` / `awareness` carriers into `Think` — those
  carriers do not yet exist as types in `thinking-engine` today; the
  full doctrinal `Think` (per CLAUDE.md showing eight fields) is the
  Sprint-15+ horizon. This PR introduces the minimum-viable `Think`
  with only the state needed for the four splat ops.
- i4-quantized variants of the splat ops — deferred per D-CSV-12 v2 note
  ("on-Think methods sprint-13+", separate from i4 work).
- Ontology-aware splat (OntologyFilter integration) — deferred per
  splat-shader vision §7 row "Sprint-13" (Wave-F descope: scalar only).

**Scope boundary.** This is a doctrine-grounded structural migration.
The numerical behaviour of all four ops is **bit-identical** to the
sprint-12 free fns; only the calling convention changes.

---

## §2  The `Think` carrier struct

### §2.1  Why `Think` doesn't exist yet

Today `thinking-engine` has `ThinkingEngine` (`engine.rs:174`), which is
a MatVec-cycle carrier (distance_table + energy + cycles counter). It is
**not** the doctrinal `Think` of CLAUDE.md — it lacks `trajectory`,
`awareness`, `free_energy`, `resolution`, `episodic`, `graph`,
`global_context`, `codec`.

The doctrinal `Think` will accrete those fields across sprint-13 → 15+
deliverables. This PR introduces the minimum-viable `Think` with only
the fields needed for the four splat methods — namely:

1. `splat_field: Vec<SplatField>` — the field being splatted over
2. `cycle: u32` — generation counter (replaces the `generation: u32`
   parameter that the free fns currently take)

This is **NOT yet** the eight-field doctrinal `Think` from CLAUDE.md;
it is the splat-scoped slice of it. The struct is named `Think`
(not `ThinkSplat` or similar) so that sprint-14+ extensions can land
as additional fields and methods on the same carrier without renaming.

### §2.2  Struct definition (NEW file)

NEW: `crates/thinking-engine/src/think.rs`

```rust
//! `Think` — the doctrinal cognitive carrier (CLAUDE.md "Thinking is a struct").
//!
//! Sprint-13 minimum-viable scope: holds the splat field + cycle counter.
//! The four splat methods (`splat_gaussian`, `score_hole_closure`,
//! `replay_coherence`, `emit_if_epiphany`) live here per the on-Think
//! migration of D-CSV-14.
//!
//! Future sprints (14+) will accrete `trajectory: Vsa16kF32`,
//! `awareness: ParamTruths`, `free_energy: FreeEnergy`,
//! `resolution: Resolution`, `episodic: &EpisodicMemory`,
//! `graph: &TripletGraph`, `global_context: &Vsa16kF32`,
//! `codec: &CamPqCodec` per CLAUDE.md doctrine §"Thinking is a struct"
//! and §"AriGraph, episodic memory, SPO, CAM-PQ are thinking tissue".

use crate::splat_ops::SplatField;

/// The doctrinal cognitive carrier. Sprint-13 splat-scoped minimum.
///
/// State invariant: `cycle` increments monotonically on every method
/// that mutates `splat_field`. Read-only methods (score, coherence)
/// do not touch `cycle`.
#[derive(Clone, Debug, Default)]
pub struct Think {
    /// Gaussian splat field — additive perturbation surface.
    /// Each entry carries (mean, variance, energy, generation).
    /// See `splat_ops::SplatField`.
    pub splat_field: Vec<SplatField>,

    /// Monotonic cycle counter. Replaces the `generation: u32`
    /// parameter on the free fns; methods read `self.cycle`.
    pub cycle: u32,
}

impl Think {
    /// Construct a fresh carrier with an empty splat field at cycle 0.
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct from a pre-populated field (used by free-fn deprecation
    /// shims and by test fixtures replaying historical splat states).
    pub fn from_field(splat_field: Vec<SplatField>, cycle: u32) -> Self {
        Self { splat_field, cycle }
    }

    /// Advance the cycle counter. Invoked by all mutating splat methods.
    /// Public so test fixtures can rewind/replay without going through
    /// a full splat.
    pub fn advance_cycle(&mut self) {
        self.cycle = self.cycle.wrapping_add(1);
    }
}
```

LOC budget: ~50 LOC including module rustdoc + the four method blocks
imported from §3 below.

---

## §3  Method specifications

Each free-fn → method migration preserves bit-identical numerical
behaviour. The only semantic change: `generation: u32` parameters are
gone — the method reads (and where mutating, advances) `self.cycle`.

### §3.1  `Think::splat_gaussian` (replaces `splat_ops::splat_gaussian`)

```rust
impl Think {
    /// Splat a Gaussian centered at `mean` with `variance` into the field,
    /// adding `energy` quantum. Merges with existing same-mean entries
    /// (energy adds, variance blends weighted by energy). Advances cycle.
    ///
    /// Sprint-13 D-CSV-14 migration: replaces free `splat_ops::splat_gaussian`
    /// which took `(&mut Vec<SplatField>, mean, variance, energy, generation)`.
    /// The generation parameter is now read from `self.cycle` after advance.
    pub fn splat_gaussian(&mut self, mean: u32, variance: f32, energy: f32) {
        self.advance_cycle();
        let generation = self.cycle;
        for s in self.splat_field.iter_mut() {
            if s.mean == mean {
                let total_e = s.energy + energy;
                if total_e > 0.0 {
                    s.variance =
                        (s.variance * s.energy + variance * energy) / total_e;
                }
                s.energy = total_e;
                s.generation = generation;
                return;
            }
        }
        self.splat_field.push(SplatField { mean, variance, energy, generation });
    }
}
```

**Free-fn replaced:** `splat_ops::splat_gaussian` (lines 24-43).

**Numerical invariant:** for any `(field0, mean, variance, energy, gen)`,
constructing `Think::from_field(field0.clone(), gen - 1)` and calling
`.splat_gaussian(mean, variance, energy)` produces a field equal to
`splat_ops::splat_gaussian(&mut field0, mean, variance, energy, gen)`
applied to the same starting field. (The minus-one accounts for the
`advance_cycle` at method entry.)

### §3.2  `Think::score_hole_closure` (replaces `splat_ops::score_hole_closure`)

```rust
impl Think {
    /// Score the "hole closure" potential of the current field: how much
    /// of the total energy is concentrated in `<=k` peaks. Returns a
    /// `[0.0, 1.0]` ratio. High ratio = field has converged; low = scattered.
    ///
    /// Read-only — does NOT advance `self.cycle`.
    pub fn score_hole_closure(&self, k: usize) -> f32 {
        if self.splat_field.is_empty() {
            return 0.0;
        }
        let mut energies: Vec<f32> =
            self.splat_field.iter().map(|s| s.energy).collect();
        energies.sort_by(|a, b| {
            b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal)
        });
        let total: f32 = energies.iter().sum();
        if total <= 0.0 {
            return 0.0;
        }
        let top_k: f32 = energies.iter().take(k).sum();
        (top_k / total).clamp(0.0, 1.0)
    }
}
```

**Free-fn replaced:** `splat_ops::score_hole_closure` (lines 48-60).

**Numerical invariant:** for any field `f`, `Think::from_field(f.clone(),
0).score_hole_closure(k) == splat_ops::score_hole_closure(&f, k)` exactly.

### §3.3  `Think::replay_coherence` (replaces `splat_ops::replay_coherence`)

```rust
impl Think {
    /// Replay-coherence between this Think's current field and a prior
    /// generation's field. Returns cosine-similarity-style metric in
    /// `[-1, +1]` of energy vectors aligned by mean.
    ///
    /// Used to detect "this thought is repeating". Read-only — does NOT
    /// advance `self.cycle`.
    pub fn replay_coherence(&self, prior: &[SplatField]) -> f32 {
        use std::collections::HashMap;
        let mut p_map: HashMap<u32, f32> = HashMap::new();
        for s in prior {
            *p_map.entry(s.mean).or_insert(0.0) += s.energy;
        }
        let mut dot = 0.0_f32;
        let mut c_mag = 0.0_f32;
        let p_mag: f32 = p_map.values().map(|e| e * e).sum();
        for s in &self.splat_field {
            c_mag += s.energy * s.energy;
            if let Some(&pe) = p_map.get(&s.mean) {
                dot += s.energy * pe;
            }
        }
        let denom = (c_mag.sqrt() * p_mag.sqrt()).max(f32::EPSILON);
        (dot / denom).clamp(-1.0, 1.0)
    }
}
```

**Free-fn replaced:** `splat_ops::replay_coherence` (lines 65-82).

**Numerical invariant:** `Think::from_field(curr.clone(), 0)
.replay_coherence(&prior) == splat_ops::replay_coherence(&curr, &prior)`.

### §3.4  `Think::emit_if_epiphany` (replaces `splat_ops::emit_if_epiphany`)

```rust
impl Think {
    /// Decide whether the field state qualifies as an "epiphany emission":
    /// hole-closure above `closure_threshold` AND replay-coherence above
    /// `similarity_floor` vs the supplied `prior` field.
    ///
    /// Returns `Some(top_splat)` (the highest-energy splat) if the epiphany
    /// fires, `None` otherwise. Read-only — does NOT advance `self.cycle`.
    pub fn emit_if_epiphany(
        &self,
        prior: &[SplatField],
        closure_threshold: f32,
        similarity_floor: f32,
    ) -> Option<SplatField> {
        let closure = self.score_hole_closure(3);
        let coherence = self.replay_coherence(prior);
        if closure >= closure_threshold && coherence >= similarity_floor {
            self.splat_field
                .iter()
                .max_by(|a, b| {
                    a.energy
                        .partial_cmp(&b.energy)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
        } else {
            None
        }
    }
}
```

**Free-fn replaced:** `splat_ops::emit_if_epiphany` (lines 87-107).

**Numerical invariant:** `Think::from_field(curr.clone(), 0)
.emit_if_epiphany(&prior, ct, sf) == splat_ops::emit_if_epiphany(&curr,
&prior, ct, sf)`.

---

## §4  Free-fn deprecation strategy

The four free fns remain in `splat_ops.rs` for one release cycle with
`#[deprecated]` attributes pointing callers at the new method surface.
This preserves backward compat while signalling the migration.

### §4.1  Deprecation shim pattern

```rust
// splat_ops.rs (sprint-13 modified)

use crate::think::Think;

#[deprecated(
    since = "0.X.0",
    note = "Migrated to Think::splat_gaussian per D-CSV-14 / \
            CLAUDE.md 'Thinking is a struct' doctrine. The generation \
            parameter is now read from Think::cycle. Callers should \
            construct a Think and use the method directly; this shim \
            forwards to the method and discards the supplied generation."
)]
pub fn splat_gaussian(
    field: &mut Vec<SplatField>,
    mean: u32,
    variance: f32,
    energy: f32,
    generation: u32,
) {
    let mut think = Think::from_field(std::mem::take(field), generation.saturating_sub(1));
    think.splat_gaussian(mean, variance, energy);
    *field = think.splat_field;
}
```

Same shape for the other three:

- `score_hole_closure(field, k)` → constructs `Think::from_field(field.to_vec(), 0)`, calls `.score_hole_closure(k)`.
- `replay_coherence(current, prior)` → constructs `Think::from_field(current.to_vec(), 0)`, calls `.replay_coherence(prior)`.
- `emit_if_epiphany(field, prior, ct, sf)` → constructs `Think::from_field(field.to_vec(), 0)`, calls `.emit_if_epiphany(prior, ct, sf)`.

### §4.2  Removal timeline

- **Sprint-13 (this PR):** Deprecated free fns remain; methods are
  the canonical surface. CI does NOT yet error on
  `deprecated_in_future` for these.
- **Sprint-14:** Add `#[deny(deprecated)]` at the call-site crate
  level for any in-tree caller. Out-of-tree consumers continue to
  receive deprecation warnings but compile.
- **Sprint-15+:** Free fns deleted from `splat_ops.rs`. Module
  becomes type-only (`SplatField` plus rustdoc pointing at `Think`).

### §4.3  `SplatField` type stays in `splat_ops`

The `SplatField` struct itself stays in `splat_ops.rs` as the type
definition. It is re-exported from `lib.rs` and from `think` so callers
get both ergonomic paths:

```rust
// In lib.rs (modified)
pub mod splat_ops;
pub mod think;
pub use splat_ops::SplatField;
pub use think::Think;
```

---

## §5  Tests (18 total: 14 migrated + 4 integration)

### §5.1  Migrated tests (14, in `think.rs` `#[cfg(test)] mod think_splat_tests`)

All 14 tests from `splat_ops.rs` lines 109-290 migrate verbatim, with
the call-site change `splat_gaussian(&mut field, ...)` →
`think.splat_gaussian(...)` and equivalent for the other three. The
generation parameter is no longer threaded — `Think::from_field(_, gen-1)`
seeds the carrier so that the first `.splat_gaussian` call lands the
correct generation.

Test name mapping (sprint-12 free-fn name → sprint-13 method-test name):

| Sprint-12 free-fn test | Sprint-13 method test |
|---|---|
| `splat_gaussian_new_entry` | `think_splat_gaussian_new_entry` |
| `splat_gaussian_merge_same_mean` | `think_splat_gaussian_merge_same_mean` |
| `splat_gaussian_weighted_variance_blend` | `think_splat_gaussian_weighted_variance_blend` |
| `splat_gaussian_generation_update` | `think_splat_gaussian_generation_update` |
| `score_hole_closure_empty_field` | `think_score_hole_closure_empty_field` |
| `score_hole_closure_single_splat` | `think_score_hole_closure_single_splat` |
| `score_hole_closure_evenly_distributed` | `think_score_hole_closure_evenly_distributed` |
| `score_hole_closure_concentrated` | `think_score_hole_closure_concentrated` |
| `replay_coherence_empty_prior` | `think_replay_coherence_empty_prior` |
| `replay_coherence_identical_fields` | `think_replay_coherence_identical_fields` |
| `replay_coherence_orthogonal_fields` | `think_replay_coherence_orthogonal_fields` |
| `replay_coherence_partial_overlap` | `think_replay_coherence_partial_overlap` |
| `emit_if_epiphany_below_threshold_none` | `think_emit_if_epiphany_below_threshold_none` |
| `emit_if_epiphany_both_pass_returns_top` | `think_emit_if_epiphany_both_pass_returns_top` |
| `emit_if_epiphany_only_closure_passes_none` | `think_emit_if_epiphany_only_closure_passes_none` |
| `emit_if_epiphany_only_coherence_passes_none` | `think_emit_if_epiphany_only_coherence_passes_none` |

(16 row entries above — the 14 in `splat_ops.rs` line count is the
free-fn tests; two of them are in the `emit_if_epiphany` block, which
brings the migrated total to 16. Verify against `splat_ops.rs` lines
115-290 during implementation; spec budgets ≥14 migrated tests with
the actual count tracked by `cargo test -p thinking-engine think_`.)

### §5.2  Free-fn deprecation tests (4, in `splat_ops.rs` test module)

The four deprecation shims are themselves tested to verify they
forward correctly to the new methods. Each test asserts that calling
the deprecated free fn produces the same result as calling the
equivalent `Think` method directly.

| Test name | Asserts |
|---|---|
| `deprecated_splat_gaussian_forwards_to_method` | `splat_gaussian(&mut f, m, v, e, g)` ≡ `Think::from_field(f', g-1).splat_gaussian(m, v, e).splat_field` |
| `deprecated_score_hole_closure_forwards_to_method` | `score_hole_closure(&f, k)` ≡ `Think::from_field(f.to_vec(), 0).score_hole_closure(k)` |
| `deprecated_replay_coherence_forwards_to_method` | `replay_coherence(&c, &p)` ≡ `Think::from_field(c.to_vec(), 0).replay_coherence(&p)` |
| `deprecated_emit_if_epiphany_forwards_to_method` | `emit_if_epiphany(&f, &p, ct, sf)` ≡ `Think::from_field(f.to_vec(), 0).emit_if_epiphany(&p, ct, sf)` |

These tests use `#[allow(deprecated)]` since they intentionally exercise
the deprecated surface.

### §5.3  New integration tests (4, in `think.rs` test module)

These tests exercise the integration with `Think`'s carrier state —
specifically, the `cycle` field interaction that the free fns could
not test.

```rust
#[test]
fn think_splat_gaussian_advances_cycle() {
    let mut think = Think::new();
    assert_eq!(think.cycle, 0);
    think.splat_gaussian(1, 1.0, 5.0);
    assert_eq!(think.cycle, 1, "first splat advances cycle to 1");
    assert_eq!(think.splat_field[0].generation, 1);
    think.splat_gaussian(2, 1.0, 5.0);
    assert_eq!(think.cycle, 2, "second splat advances cycle to 2");
    assert_eq!(think.splat_field[1].generation, 2);
}

#[test]
fn think_score_hole_closure_does_not_advance_cycle() {
    let mut think = Think::new();
    think.splat_gaussian(1, 1.0, 5.0);   // cycle → 1
    let pre_cycle = think.cycle;
    let _ = think.score_hole_closure(3);
    assert_eq!(think.cycle, pre_cycle, "read-only method must not advance cycle");
}

#[test]
fn think_replay_coherence_does_not_advance_cycle() {
    let mut think = Think::new();
    think.splat_gaussian(1, 1.0, 5.0);
    let prior = think.splat_field.clone();
    let pre_cycle = think.cycle;
    let _ = think.replay_coherence(&prior);
    assert_eq!(think.cycle, pre_cycle);
}

#[test]
fn think_emit_if_epiphany_does_not_advance_cycle() {
    let mut think = Think::new();
    think.splat_gaussian(1, 1.0, 90.0);
    think.splat_gaussian(2, 1.0, 5.0);
    let prior = think.splat_field.clone();
    let pre_cycle = think.cycle;
    let _ = think.emit_if_epiphany(&prior, 0.5, 0.5);
    assert_eq!(think.cycle, pre_cycle, "epiphany detection is read-only");
}
```

**Coverage rationale:** The free-fn surface had no cycle counter to
test against, so the four integration tests above exercise the new
state interaction that the carrier introduces. They also verify the
read-only/mutating distinction documented on each method.

### §5.4  Total test budget

- 16 migrated method tests in `think.rs`
- 4 deprecation-shim forwarding tests in `splat_ops.rs`
- 4 new integration tests in `think.rs`

**Total: 24 tests.** Estimate ~180 LOC of test code (matches the
plan-v3 LOC budget for tests).

---

## §6  Cross-cutting: downstream call-site updates

Per the grep run on workspace:

```
$ grep -rn "splat_gaussian\|score_hole_closure\|replay_coherence\|emit_if_epiphany" \
       crates/ examples/ 2>/dev/null
```

(Run at time of spec authoring.) Result:

- **`crates/thinking-engine/src/splat_ops.rs`** — the source file itself
  + its tests. **In-tree fix:** convert tests per §5.1, deprecate fns
  per §4.
- **No other workspace crates currently consume the four free fns.**
  (Confirmed: grep at preflight time returned only `splat_ops.rs`
  itself plus the planning/knowledge `.claude/` documents which are
  not Rust callers.)
- **`.claude/plans/*`, `.claude/knowledge/*`, `.claude/board/*`** —
  these reference the function names in prose; **no code changes
  needed**. The spec § §0 here updates the doctrinal references
  during board-hygiene commit.

This is a **zero-external-caller** migration as of the preflight grep,
which makes the deprecation cycle a conservative belt-and-suspenders
move rather than a forcing function. The deprecation still ships in
case any in-flight branch added a caller between preflight and merge.

### §6.1  Watchpoints (potential future callers to keep in mind)

The splat-shader vision doc §6 sketches a future cognitive cycle that
would consume these methods from a `dispatch_cycle` orchestrator. When
that orchestrator lands (sprint-14+), it MUST consume the methods, not
the deprecated free fns. The deprecation note + `#[deny(deprecated)]`
escalation in sprint-14 (§4.2) enforces this.

LOC budget for call-site work in THIS PR: ~0 LOC (no external callers
exist). Reserved budget of ~50 LOC in the plan-v3 estimate covers
test-shim glue and Cargo.toml re-export tweaks.

---

## §7  Risk matrix

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| **Behavioural drift** during free-fn → method migration | LOW | HIGH | Bit-identical body copy preserves numerics; §5.2 forwarding tests guard the deprecation shims explicitly. |
| **Cycle-counter semantics ambiguity** (does score advance cycle? does emit?) | MED | MED | §3.x and §5.3 codify: mutating methods advance, read-only methods do not. Four §5.3 tests guard this invariant. |
| **Name collision** between `Think` and existing `ThinkingEngine` | LOW | LOW | Distinct names; `Think` is the doctrinal CLAUDE.md carrier, `ThinkingEngine` is the legacy MatVec runner. They will coexist until full reunification (Sprint-15+). |
| **Premature commit to Think shape** before episodic/graph/awareness carriers exist | MED | MED | Spec §2.1 explicitly scopes this PR to splat fields only; future fields land additively without renaming. |
| **Deprecation noise** in downstream lab/example builds | LOW | LOW | Zero callers today (§6); `#[allow(deprecated)]` on the four §5.2 tests keeps in-tree CI clean. |
| **Wrap-around on `cycle: u32`** at 4.3B cycles | LOW | LOW | `cycle.wrapping_add(1)` is documented; at sustained 1M cycles/sec, wrap is ~1.2 hours of wall-clock — a real-world Think instance is expected to checkpoint long before. Sprint-15+ may revisit to `u64` if needed. |
| **Spec drift vs plan-v3 D-CSV-14 LOC budget** | LOW | LOW | §8 reconciles ~480 LOC against plan-v3's "~13 tests, ~250 LOC" budget; spec overshoots slightly because it adds 4 integration tests not in the original D-CSV-14 estimate. Documented in §8. |
| **Doctrine violation** (re-introduces a free fn elsewhere during migration) | LOW | HIGH | CLAUDE.md litmus test is the gate; reviewer MUST check every new function in this PR is a method on `Think`. |

---

## §8  LOC estimate

| Component | LOC |
|---|---|
| `think.rs` — struct definition + 4 method impl blocks + rustdoc | ~250 |
| `splat_ops.rs` — 4 deprecation shims + 4 forwarding tests + rustdoc updates | ~80 |
| `think.rs` test module — 16 migrated method tests + 4 cycle integration tests | ~180 |
| `lib.rs` — `pub mod think;` + re-exports (`Think`, `SplatField`) | ~5 |
| Cargo.toml / module wiring | ~0 (no new deps) |
| Board-hygiene updates (`STATUS_BOARD.md` D-CSV-14 row, `LATEST_STATE.md` Contract Inventory row, `PR_ARC_INVENTORY.md` prepend, `EPIPHANIES.md` optional E-id for "doctrinal migration complete") | ~30 |
| **Total** | **~545 LOC** |

Plan-v3 D-CSV-14 budget was "13+ tests, depends on D-CSV-12". This
PR ships **24 tests** (16 migrated + 4 deprecation + 4 integration)
and ~545 LOC including board hygiene. The overshoot vs the original
D-CSV-12 estimate of "~800 LOC including i4 path" is consistent
because this PR is scalar-only on-Think and excludes the i4 + par_*
follow-ups.

### §8.1  LOC breakdown by file (engineering bill-of-materials)

```
NEW
  crates/thinking-engine/src/think.rs                      ~430 LOC
    - module rustdoc                                          ~30
    - struct Think                                            ~20
    - impl Think (constructors + advance_cycle)               ~25
    - impl Think (4 method impls per §3)                     ~175
    - #[cfg(test)] mod tests (20 tests per §5.1 + §5.3)      ~180

MODIFY
  crates/thinking-engine/src/splat_ops.rs                  ~+80 LOC
    - #[deprecated] on each of the 4 fns + delegation       ~+60
    - #[cfg(test)] 4 forwarding tests (§5.2)                ~+20
  crates/thinking-engine/src/lib.rs                        ~+5 LOC
    - pub mod think;
    - pub use think::Think;
    - pub use splat_ops::SplatField;

BOARD HYGIENE (same commit as implementation, per CLAUDE.md rule)
  .claude/board/STATUS_BOARD.md                            ~+1 row
    D-CSV-14 row: Queued → In PR (with PR # once opened)
  .claude/board/LATEST_STATE.md                            ~+3 lines
    Contract Inventory: + `Think` (thinking-engine/think.rs)
  .claude/board/PR_ARC_INVENTORY.md                        ~+20 lines
    PREPEND new dated entry for D-CSV-14 PR
  .claude/board/EPIPHANIES.md                              ~+10 lines (optional)
    E-DOCTRINE-N: "Splat ops migrated to Think methods —
                    first realisation of CLAUDE.md
                    'Thinking is a struct' doctrine in code."
```

The implementation PR is small, mechanical, doctrine-correct, and
sets the carrier on which sprint-14+ extensions (par_* variants,
episodic wiring, graph wiring, awareness wiring) accrete.

---

## §9  Definition of done

- [ ] `crates/thinking-engine/src/think.rs` exists with `Think` struct and four method `impl` blocks per §3.
- [ ] `crates/thinking-engine/src/splat_ops.rs` four free fns carry `#[deprecated]` with migration note per §4.1.
- [ ] `lib.rs` re-exports `Think` and `SplatField` per §4.3.
- [ ] `cargo test -p thinking-engine think_` runs all 16 migrated method tests + 4 cycle integration tests; ALL PASS.
- [ ] `cargo test -p thinking-engine deprecated_` runs 4 forwarding tests; ALL PASS.
- [ ] `cargo test -p thinking-engine` overall pass count is ≥ sprint-12 baseline + 10 (24 new − 14 retained = 10 net).
- [ ] `cargo clippy -p thinking-engine -- -D warnings` clean (deprecation warnings allowed via `#[allow(deprecated)]` on the four §5.2 forwarding tests).
- [ ] `cargo fmt -- --check` clean.
- [ ] `.claude/board/STATUS_BOARD.md` D-CSV-14 row updated to In PR in the same commit as the implementation.
- [ ] `.claude/board/LATEST_STATE.md` Contract Inventory adds `Think` row.
- [ ] `.claude/board/PR_ARC_INVENTORY.md` prepended with the dated PR entry.
- [ ] Spec file `.claude/specs/pr-sprint-13-think-methods.md` (this file) merged on the preflight branch BEFORE implementation begins on the sibling `claude/sprint-13-think-methods` branch.

---

## §10  Cross-references (canonical list)

- **Source crate:** `crates/thinking-engine/src/splat_ops.rs` (W-F7, PR #388)
- **Doctrine root:** CLAUDE.md §"The Click" / §"Thinking is a struct" / §"Litmus tests"
- **Knowledge:** `.claude/knowledge/splat-shader-rayon-struct-method-vision.md` §3.2 §6 §7
- **Knowledge (sibling):** `.claude/knowledge/cognitive-shader-driver-thinking-engine-reunification.md` §5
- **Plan (active):** `.claude/plans/cognitive-substrate-convergence-v2.md` line 62, 376, 383, 499-503, 515 (D-CSV-14 backlog → In PR)
- **Plan (sibling splat sources):** `.claude/plans/tetrahedral-epiphany-splat-integration-v1.md`, `.claude/plans/oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §9, `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md`
- **Board:** `.claude/board/STATUS_BOARD.md` D-CSV-14 row; `.claude/board/PR_ARC_INVENTORY.md` (post-merge prepend); `.claude/board/LATEST_STATE.md` Contract Inventory (post-merge update)
- **Predecessor sprint-12:** `.claude/board/sprint-log-11/meta-review.md` line 321 (the W-F7 → D-CSV-14 entry-point note)

---

*Spec authored 2026-05-16 on `claude/sprint-13-preflight-planning`
(sprint-13 preflight PP-4, Opus planner). Implementation work to
follow on `claude/sprint-13-think-methods` once spec merges to the
preflight branch and the preflight branch merges to main.*
