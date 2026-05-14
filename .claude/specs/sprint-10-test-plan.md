# Sprint-10 Unified Test Plan

> **Author:** W11 (Sonnet, test-plan-unification worker)
> **Date:** 2026-05-14
> **Sprint:** sprint-log-10 (CCA2A pattern, 12 workers + 1 Opus meta)
> **Output target:** `.claude/specs/sprint-10-test-plan.md`
> **Status:** DRAFT — awaiting W1-W9 per-PR spec completion for exact test counts
> **Parent plan:** `.claude/plans/causaledge64-mailbox-rename-soa-v1.md`
> **Plans cited:** §3 (CausalEdge64 layout), §7 (PR sequencing + test plans), §8 (ndarray prerequisites), §12 (iron-rule compliance)
> **CI sources:** `scripts/miri-tests.sh` (lance-graph), `/home/user/ndarray/scripts/miri-tests.sh`, `.github/workflows/rust-test.yml`, `.github/workflows/style.yml`

---

## §1 Statement of Scope

This spec aggregates per-PR test plans from the W1-W9 worker specs (sprint-10 fleet) into a coherent test infrastructure plan that **gates the sprint-10 → sprint-11 transition**. It does not author new test logic — it specifies what must exist, where it must live, and what CI gates enforce it.

The sprint-10 CCA2A fleet produces 8 implementation specs (7 lance-graph PRs + 1 ndarray PR). This document defines:

1. **Unified test category taxonomy** across all 8 PRs.
2. **Aggregated unit-test count targets** per PR, with test file locations.
3. **Miri coverage growth plan** — how the Miri-clean test count grows from ~459 (current lance-graph) + ~301 (current ndarray polyfill) to ~1500 across both repos after sprint-11 implementation.
4. **Miri runner extension instructions** for both `scripts/miri-tests.sh` (lance-graph) and `ndarray/scripts/miri-tests.sh`.
5. **Clippy gate policy** — what must be clean before any sprint-11 PR can merge.
6. **Integration test plan** — cross-crate scenarios that no single PR's test suite covers.
7. **Perf benchmark suite** — criterion-based benchmarks gating sprint-11 acceptance.
8. **CI workflow additions** — new job definitions for `.github/workflows/rust-test.yml` and a new `miri-extended.yml`.
9. **Deferred items** — tests explicitly parked for sprint-12+.

**Why this spec exists:** The parent plan (`causaledge64-mailbox-rename-soa-v1.md`) sequences 8 PRs that jointly wire a new substrate layer. Individual PR test plans live in W1-W9 specs. But cross-PR integration scenarios (par-tile feeding lance-graph-supervisor feeding AriGraph) require a test author who can see all 8 PRs simultaneously. This spec is that author's mandate.

**Sprint-10 → sprint-11 gate:** A sprint-11 implementation wave is blocked until this spec is ratified (meta-review) AND every W1-W9 spec has been reviewed for test-plan consistency against this document. Implementation workers read this spec before writing a single test.

---

## §2 Test Categories (Five)

### 2.1 Unit Tests (per-crate, per-type/method)

Owned by each worker's spec. Each unit test targets a single type or method in isolation. Tests live in the crate's `src/` inline (`#[cfg(test)]` module) or in `crates/<crate>/tests/*.rs` integration-style files that still target a single crate.

**Aggregation target (from §3 table):** ~183 new unit tests across the 8 PRs.

**Coverage requirements per unit test:**
- Every new public `fn` or method must have at least one happy-path test.
- Methods with error paths must have at least one error-path test.
- `unsafe` blocks (if any) must have a companion Miri test (see §2.4).

### 2.2 Property Tests (proptest / quickcheck)

Property tests verify algebraic and behavioral invariants across random inputs. They are heavier than unit tests but lighter than integration tests.

**Mandated property test targets for sprint-11 (by PR):**

| Target | Property | Library | Location |
|---|---|---|---|
| `CausalEdge64` PAL8 encode/decode | Round-trip: `decode(encode(edge)) == edge` for all valid bit layouts | proptest | `crates/causal-edge/tests/pal8_round_trip.rs` |
| `CausalEdge64` G/W/truth accessor | Bit accessor orthogonality: setting G-slot does not disturb W-slot or truth band | proptest | same file |
| `AttentionMask` LRU eviction | LRU invariant: evicted slot index is always the least-recently-used; post-eviction lookup returns None | proptest | `crates/par-tile/tests/attention_mask_props.rs` |
| `MailboxSoA` lifecycle | Push-dispatch-drop: row count monotonic during push, decrements on drop; plasticity counters monotonic during emission | proptest | `crates/par-tile/tests/mailbox_soa_props.rs` |
| `SpoWitnessChain<N>` ordering | Append-only: witness sequence prefix-stable under concurrent emission (single-writer discipline) | proptest | `crates/lance-graph/tests/spo_witness_chain_props.rs` |
| `NarsTables` LUT correctness | NARS deduction identity: `revise(w, w) == w` for all valid truth values | quickcheck | `crates/causal-edge/tests/nars_tables_invariant.rs` |

**Total property tests target:** ~6 suites, each covering 5-20 property variants = ~60-120 individual property assertions.

### 2.3 Integration Tests (cross-crate)

Integration tests verify that two or more crates compose correctly. They link multiple crates together in a single test binary and live in `crates/<crate>/tests/` or a workspace-level `tests/e2e/` directory.

Full cross-PR integration test plan: §7. **Target: ~30 integration test scenarios across 7 cross-crate pairings.**

### 2.4 Miri Tests (UB Detection)

Miri tests are the existing unit and integration tests run under `cargo +nightly miri test`. They detect undefined behavior that Rust's type system permits but the memory model forbids: use-after-free, out-of-bounds access, data races, invalid pointer arithmetic.

**Current state (pre-sprint-11):**
- lance-graph `scripts/miri-tests.sh`: covers `lance-graph-contract`, `lance-graph-rbac`, `neural-debug`, `lance-graph-ontology` (no-default-features). ~459 Miri-clean tests across these 4 crates.
- ndarray `scripts/miri-tests.sh`: covers `ndarray` + `ndarray-rand` with `--features approx,serde,nightly-simd`, excluding `hpc::*` (except `hpc::byte_scan`), `simd::tests::*`, and `hpc::framebuffer::pyramid_tests::*`. ~301 Miri-clean tests in the passing subset.

**Post-sprint-11 target: ~1500 Miri-clean tests across both repos.** See §4 for the detailed growth path.

### 2.5 Perf Benchmarks (Criterion-based)

Performance benchmarks measure latency and throughput for hot-path operations. They run on `merge_group` and push-to-main only (not on every PR). Full plan: §8.

---

## §3 Aggregated Unit-Test Count Per PR

> **Note:** W1-W9 per-PR specs are not yet authored as of this W11 draft (W12 was the only preceding worker). The test counts below are derived from the parent plan §7 per-PR scopes, LOC estimates, and the standard coverage expectation of ~1 test per 10-20 LOC of new public API surface. These will be reconciled against actual W1-W9 spec counts by the meta-reviewer (Opus).

| PR | Spec author | Est. unit tests | New test file(s) | Parent plan §7 LOC estimate |
|---|---|---|---|---|
| `PR-NDARRAY-MIRI-COMPLETE` | W8 | ~60 | extend `simd_nightly/tests.rs`; new `u_word_gaps_test.rs` | ~200 LOC (method gaps + dispatch reroute) |
| `PR-CE64-MB-1` par-tile crate apex | W1 | ~30 | `crates/par-tile/tests/mailbox_tests.rs`; `crates/par-tile/tests/attention_mask_tests.rs` | ~1500 LOC new crate |
| `PR-CE64-MB-2` CausalEdge64 v2 layout | W2 | ~10 unit | `crates/causal-edge/tests/v2_layout_test.rs` | ~400 LOC |
| `PR-CE64-MB-2` PAL8 + NarsTables regression | W3 | ~5 regression | `crates/causal-edge/tests/pal8_round_trip.rs`; `crates/causal-edge/tests/nars_tables_invariant.rs` | (companion to W2) |
| `PR-CE64-MB-3` BindSpace E/F/G/H | W4 | ~15 | extend `crates/cognitive-shader-driver/tests/bindspace_tests.rs` | ~800 LOC |
| `PR-CE64-MB-4` AriGraph SPO-G | W5 | ~10 | `crates/lance-graph/tests/arigraph_spo_g.rs` | ~600 LOC |
| `PR-CE64-MB-5` MailboxSoA + AttentionMask | W6 | ~15 | extend `crates/par-tile/tests/integration.rs`; new `crates/par-tile/tests/mailbox_soa_tests.rs` | ~1200 LOC |
| `PR-CE64-MB-6` SigmaTierRouter | W7 | ~30 | `crates/lance-graph-supervisor/tests/sigma_router.rs` | ~1500 LOC |
| `PR-CE64-MB-7` bevy cull plugin | W9 | ~8 | `crates/bevy-cull-plugin/tests/integration.rs` | ~500 LOC |
| **Total** | | **~183 unit + ~60 property assertions** | | ~6700 LOC new across 8 PRs |

### 3.1 Per-PR Test Scope Detail

**PR-NDARRAY-MIRI-COMPLETE (W8 — cite `.claude/specs/pr-ndarray-miri-complete.md` when available)**

Source: parent plan §8 + ndarray `scripts/miri-tests.sh` comments.

Tests cover:
- `U16x32`, `U32x16`, `U64x8` method gaps: `simd_eq`, `simd_ne`, `simd_ge`, `simd_gt`, `simd_le`, `simd_lt`, `simd_clamp`, `select`, `zero()`, `to_bitmask`, `from_u8x64_lo`, `from_u8x64_hi`, `pack_saturate_u8`, `shl`, `shr` — ~14 methods × 3 types = 42 method-level unit tests.
- Symmetric `I16x32`, `I32x16`, `I64x8` gaps — ~6 methods × 3 types = 18 method-level unit tests.
- `cfg(miri)` dispatch reroute smoke test: `simd::F32x16::splat(1.0)` must not abort under `cargo +nightly miri test`.
- Total: ~60 new tests in ndarray's `simd_nightly/` test suite.

**Critical unlock:** W8's `cfg(miri)` dispatch reroute in `src/simd.rs` removes the AVX target-feature wall. This opens `simd::tests::*` (~130 tests) and many `hpc::*` paths to Miri checking. The `hpc::framebuffer::pyramid_tests::*` exclusion remains (19+ min runtime).

**PR-CE64-MB-1 par-tile crate apex (W1 — cite `.claude/specs/pr-ce64-mb-1-par-tile-crate.md` when available)**

Source: parent plan §6 par-tile crate description.

Tests cover:
- `Mailbox<T>` trait: `InMemoryMailbox` send/recv round-trip, capacity, close behavior.
- `AttentionMask`: `bind_g`, `lookup_g`, `resolve_g`, LRU eviction under slot exhaustion, rebinding.
- `MailboxSoA<N>`: row push, dispatch_cycle no-op on empty, drop_row reduces count.
- `BindSpaceView`: zero-copy borrow invariant (borrow does not clone BindSpace data).

**PR-CE64-MB-2 + PR-CE64-MB-2-test (W2 + W3 — cite `.claude/specs/pr-ce64-mb-2-causaledge64-v2.md` and `.claude/specs/pr-ce64-mb-2-pal8-nars-regression.md` when available)**

Source: parent plan §3 CausalEdge64 v2 layout; §7 PR-CE64-MB-2 scope.

W2 tests (layout accessors):
- `CausalEdge64::g_slot()` returns bits 51-55 correctly.
- `CausalEdge64::w_slot()` returns bits 56-61 correctly.
- `CausalEdge64::truth_band()` returns bits 62-63 correctly.
- Setting G-slot does not perturb W-slot or truth-band (bit orthogonality).
- v2 feature-flag: v1 consumers compile without the new accessors (backward compat gate).

W3 regression tests:
- PAL8 round-trip: `pal8_decode(pal8_encode(edge)) == edge` for 100 random valid edges.
- NarsTables LUT invariant: `deduction(w, w) == w` for all 256×256 truth pairs.
- EdgeColumn binary layout: write v2 CausalEdge64 array, read back as v1 — PAL8 bytes stable.

**PR-CE64-MB-3 BindSpace E/F/G/H (W4 — cite `.claude/specs/pr-ce64-mb-3-bindspace-efgh.md` when available)**

Source: parent plan §6 cognitive-shader-driver; `bindspace-columns-v1.md` Phase 2.

Tests cover:
- Column E (OntologyDelta): set/get round-trip.
- Column F (AwarenessColumn): 256-B/row layout, gated write, read returns correct slice.
- Column G (ModelBindingColumn): style-slot index stored and retrieved correctly.
- Column H (TypeColumn EntityTypeId u16): `push_typed(entity_type_id)` increments row count; `entity_type_id_at(row)` correct.
- `CollapseGate::MergeMode::Superposition`: both deltas preserved when XOR-equal (does not collapse to zero).
- `BindSpaceView` row-range borrow: exposes correct row subset.
- FIX-5 closure: `trust_below_floor` wiring test at `driver.rs:311` path.

**PR-CE64-MB-4 AriGraph SPO-G (W5 — cite `.claude/specs/pr-ce64-mb-4-arigraph-spo-g.md` when available)**

Source: parent plan §6 AriGraph; `ogit-g-context-bundle-v1.md` D-OGIT-G-1.

Tests cover:
- SPO-G quad mode: `insert_quad(S, P, O, G)` stores correctly; `query_by_g(g)` returns all matching quads.
- G filter: `query_spo_g(S, P, O, G)` returns only matching quads.
- Ghost-edge persistence: Pearl rung 3 edge inserts as ghost; survives serialization round-trip.
- `SpoWitnessChain<N>` packing: packed witness identity + replay_ref round-trip.
- NARS decay: `ghost.decay()` reduces confidence per NARS truth-revise formula.
- Reactivation: evidence for ghost edge promotes it to active.
- Chain truncation: `SpoWitnessChain<4>` truncates at 4 entries, oldest dropped.

**PR-CE64-MB-5 MailboxSoA + AttentionMask actor wiring (W6 — cite `.claude/specs/pr-ce64-mb-5-mailbox-soa-attentionmask.md` when available)**

Source: parent plan §5 MailboxSoA; D-CE64-MB-7/8/9.

Tests cover:
- `AttentionMask::bind_g` / `lookup_g` / `resolve_g` round-trip.
- LRU eviction: slot 0 evicted when all 32 G-slots full and new domain bound.
- `AttentionMask::broadcast`: all bound-domain compartments notified on domain eviction.
- `MailboxSoA` spawn-dispatch-prune lifecycle (property test D-CE64-MB-9).
- XOR-cancel: complementary mailboxes XOR to zero in EdgeColumn (via `CollapseGate::Xor`).
- Intent gate: `compartment.intent = None` prevents dispatch to Zone 3.
- Plasticity counter: `emit(edge)` increments plasticity counter for `(role, G)` pair; counter monotonic.
- `BindSpaceView` zero-copy: borrow flag asserts BindSpace internal buffer not copied.
- Write token: `BindSpaceView` write-token prevents concurrent writes (compile-time via borrow checker).

**PR-CE64-MB-6 SigmaTierRouter (W7 — cite `.claude/specs/pr-ce64-mb-6-sigma-tier-router.md` when available)**

Source: parent plan §6 lance-graph-supervisor; `linguistic-epiphanies-2026-04-19.md` E21 Σ10 Rubicon.

Tests cover:
- Banding policy: Σ1-Σ5 maps to TokioMailbox; Σ6-Σ8 maps to InMemoryMailbox; Σ9-Σ10 routes to L4 escalation (one test per tier band = 10 tests).
- JIT pipeline E2E: `spawn(style_slot, G_slot)` → AttentionMask resolves architectural style → `KernelHandle` consumed or JIT-compiled via `lance-graph-planner`.
- Elevation cascade: Σ7 → Σ8 → Σ9 escalation chain fires correctly on EPIPHANY-tier witness emission.
- Plasticity NARS-revise: `SigmaTierRouter` adjusts spawn priors based on plasticity counters.
- Three-trigger pruning: budget-exhaustion, XOR-cancel, and outcome-sufficient each individually trigger `drop_row`.
- Σ9-Σ10 escalation: routes EPIPHANY-tier output to `CallcenterSupervisor` without breaking existing supervisor.
- Per-thread shadow: each `SigmaTierRouter` instance is thread-local; no global mutable state.
- `InMemoryMailbox` p99 latency: micro-benchmark asserts p99 < 500 ns (see §8).

**PR-CE64-MB-7 bevy cull plugin (W9 — cite `.claude/specs/pr-ce64-mb-7-bevy-cull-plugin.md` when available)**

Source: parent plan §7 PR-CE64-MB-7; bevy session recommendations.

Tests cover:
- Correctness vs stock cull: 1000 entities; `NdarrayCullPlugin` and stock `check_visibility` agree on visible/hidden for all entities.
- Edge case: entity exactly on frustum boundary (both inside and outside, floating-point edge).
- Empty scene: 0 entities; plugin does not panic.
- Large scene: 10K entities; plugin completes within 1 frame budget (60 Hz = 16.7 ms).
- Miri compatibility: plugin code under `cfg(miri)` compiles and does not invoke unsafe intrinsics directly.

---

## §4 Miri Test Coverage Growth

### 4.1 Current State (Pre-Sprint-11)

**lance-graph** (per `scripts/miri-tests.sh` — shipped on branch `claude/resolve-pr-369-conflicts-ozMXd`, commit `6590b9e`):

Miri-clean crates:
- `lance-graph-contract` — ~15 inline tests
- `lance-graph-rbac` — ~30 tests
- `neural-debug` — ~10 tests
- `lance-graph-ontology` (no-default-features) — ~404 tests
- **Total: ~459 Miri-clean tests**

Excluded from Miri (FFI-blocked):
- `lance-graph` core (lance + arrow + datafusion FFI)
- `lance-graph-planner` (JIT/Cranelift FFI)
- `lance-graph-catalog` (lance FFI)
- `lance-graph-python` (PyO3 FFI)

**ndarray** (per `ndarray/scripts/miri-tests.sh`):

Miri-clean (currently):
- ndarray core + ndarray-rand with `--features approx,serde,nightly-simd`, minus `hpc::*` (except byte_scan), `simd::tests::*`, `hpc::framebuffer::pyramid_tests::*`.
- **~301 Miri-clean tests** in this subset.

Excluded from Miri (AVX target-feature wall):
- `simd::tests::*` — ~130 tests using `crate::simd::F32x16` etc. via AVX intrinsics.
- `hpc::*` (except `hpc::byte_scan`) — ~400+ tests across all hpc modules.
- `hpc::framebuffer::pyramid_tests::*` — 3 tests (19+ min each).

**Combined pre-sprint-11 total: ~760 Miri-clean tests.**

### 4.2 Post-Sprint-11 Target

**Mechanism A — W8 ndarray dispatch reroute (+~529 tests in ndarray)**

W8 adds `cfg(miri)` switch in `src/simd.rs` re-exporting from `simd_nightly` under Miri. This removes the AVX target-feature wall for:
- `simd::tests::*` — ~130 tests become Miri-clean.
- `hpc::activations::*` — ~30 tests.
- `hpc::fingerprint::*` — ~60 tests.
- `hpc::blas_level1::*` — ~80 tests.
- `hpc::nars::*` — ~40 tests.
- `hpc::styles::*` — ~49 tests.
- `hpc::causal_diff::*` — ~30 tests.
- Other `hpc::*` paths — ~110 tests.
- `hpc::framebuffer::pyramid_tests::*` remains excluded (19+ min runtime, not a UB signal).

**Mechanism B — New sprint-11 crates (pure Rust) (+~116 tests in lance-graph)**

- `par-tile` crate (W1, W6): pure Rust, no BLAS/FFI. ~45 unit tests Miri-runnable.
- `causal-edge` additions (W2, W3): PAL8 codec and accessor tests are pure Rust. ~15 Miri-clean.
- `cognitive-shader-driver` new columns (W4): BindSpace SoA Columns E/F/G/H = pure Rust SoA layout. ~15 Miri-clean.
- `lance-graph-supervisor` additions (W7): SigmaTierRouter with InMemoryMailbox uses crossbeam (Miri-friendly). Under Miri, Tokio-backed ractor shape is excluded via `#[cfg(not(miri))]`. ~30 Miri-clean.
- AriGraph SPO-G unit tests (W5): quad shape operations are pure Rust (Lance I/O FFI-blocked). ~10 Miri-clean.
- `bevy-cull-plugin` Miri-compat test (W9): 1 Miri-clean compilation test.

**Mechanism C — Expanded lance-graph Miri scope (+~145 tests in lance-graph)**

- `lance-graph-contract` gains 2-bit truth-band collapse + CollapseGate::Superposition unit tests (~5 tests).
- `lance-graph-supervisor` sigma-router tests added to Miri sweep (crossbeam-based paths). ~30 tests.
- Property tests (§2.2): ~60 assertions across 6 proptest suites (all pure Rust, Miri-runnable with `PROPTEST_CASES=100`).
- Integration tests for pure-Rust cross-crate paths (par-tile × causal-edge). ~10 Miri-clean.

### 4.3 Miri Coverage Tracking Table

| Repo | Pre-sprint-11 | Mechanism A | Mechanism B | Mechanism C | Post-sprint-11 |
|---|---|---|---|---|---|
| ndarray | ~301 | +529 | 0 | 0 | ~830 |
| lance-graph | ~459 | 0 | +116 | +145 | ~720 |
| **Total** | **~760** | **+529** | **+116** | **+145** | **~1550** |

**Conservative target: ~1500. Optimistic target: ~1550.**

---

## §5 Miri Test Runner Extensions

### 5.1 lance-graph `scripts/miri-tests.sh` Extensions

**After PR-CE64-MB-1 lands (par-tile crate):**

Add to `MIRI_SAFE_CRATES`:
```sh
MIRI_SAFE_CRATES="
    -p lance-graph-contract
    -p lance-graph-rbac
    -p neural-debug
    -p par-tile
"
```

Rationale: `par-tile` is pure Rust, no external deps beyond `crossbeam-channel` (Miri-friendly per upstream crossbeam Miri CI). `InMemoryMailbox` uses `VecDeque` + crossbeam; no inline asm, no BLAS.

**After PR-CE64-MB-6 lands (SigmaTierRouter):**

Add to `MIRI_SAFE_CRATES`:
```sh
    -p lance-graph-supervisor
```

Rationale: `SigmaTierRouter` unit tests target InMemoryMailbox routing (pure crossbeam). The Tokio-backed ractor shape is excluded via `#[cfg(not(miri))]` in the implementation; only the InMemoryMailbox path exercises under Miri.

**After W3 regressions land:**

Add property test invocation:
```sh
# Property tests for CausalEdge64 PAL8 round-trip + NarsTables invariant.
# proptest is Miri-friendly; run with reduced case count to control runtime.
export PROPTEST_CASES=100
cargo +nightly miri test -p causal-edge --features proptest -- pal8 nars_tables
```

**Full updated script structure (post-sprint-11):**
```sh
MIRI_SAFE_CRATES="
    -p lance-graph-contract
    -p lance-graph-rbac
    -p neural-debug
    -p par-tile
    -p lance-graph-supervisor
"

MIRI_SAFE_NO_DEFAULT="
    -p lance-graph-ontology --no-default-features
"

cargo +nightly miri test $MIRI_SAFE_CRATES
cargo +nightly miri test $MIRI_SAFE_NO_DEFAULT

export PROPTEST_CASES=100
cargo +nightly miri test -p causal-edge --features proptest
```

### 5.2 ndarray `scripts/miri-tests.sh` Extensions

**After PR-NDARRAY-MIRI-COMPLETE lands (W8):**

Drop the `!test(/^simd::tests::/)` exclusion clause. Updated filter (replacing the current 3-clause filter):

```sh
cargo +nightly miri nextest run -v \
    --no-fail-fast \
    -p ndarray -p ndarray-rand \
    --features approx,serde,nightly-simd \
    -E '!(
            test(/^hpc::/) - test(/^hpc::byte_scan/)
                          - test(/^hpc::activations::/)
                          - test(/^hpc::fingerprint::/)
                          - test(/^hpc::blas_level1::/)
                          - test(/^hpc::nars::/)
                          - test(/^hpc::styles::/)
                          - test(/^hpc::causal_diff::/)
        ) and !test(/^hpc::framebuffer::pyramid_tests::/)
       '
```

**Rationale for remaining exclusions:**
- `hpc::framebuffer::pyramid_tests::*`: 19+ minutes each under Miri. Performance constraint, not a UB signal. Retain exclusion with existing comment.
- `hpc::*` modules calling BLAS FFI (gemm, gemv, etc.): Miri cannot cross FFI boundaries.

**Comment update for audit trail:** The existing `# Filter rationale (3-clause AND)` comment must be updated to 2-clause AND after dropping the `simd::tests::*` exclusion, with note: "clause 2 (simd::tests::*) removed by PR-NDARRAY-MIRI-COMPLETE after cfg(miri) dispatch reroute in src/simd.rs."

---

## §6 Clippy Gate

Per `CLAUDE.md` "Clippy-first verification discipline" and the existing `.github/workflows/style.yml` Tier A/B structure.

### 6.1 Policy

**Sprint-10 spec production (current sprint):** Specs are markdown only; clippy does not apply.

**Sprint-11 implementation (next sprint):** Every implementation PR runs `cargo clippy` as the **first** CI gate before any test or build. A PR that fails clippy cannot merge.

**Implementation workers MUST:**
1. Run `cargo clippy --workspace --tests --no-deps -- -D warnings` locally before opening any PR.
2. Address all clippy violations before requesting review.
3. Never use `#[allow(clippy::...)]` without a comment explaining why the lint is incorrect for this specific case.

**CLAUDE.md rule (locked):** Never use `clippy --fix` for unused-import warnings — they signal missing wiring, not dead code.

### 6.2 Per-crate Clippy Tier Assignment (Post-Sprint-11)

| Crate | Current tier | Post-sprint-11 tier | Notes |
|---|---|---|---|
| `lance-graph-contract` | Tier A (mandatory, gating) | Tier A (maintained) | Zero-dep; clean baseline |
| `par-tile` (NEW) | — | **Tier A** | New crate, zero legacy debt; Tier A on first PR |
| `lance-graph-supervisor` new modules | Tier B (advisory) | **Tier A for new files** | `sigma_tier_router.rs` and sigma_router module = new code |
| `cognitive-shader-driver` new columns | Tier B (advisory) | **Tier A for new column modules** | `column_e.rs`, `column_f.rs`, `column_g.rs`, `column_h_ext.rs` = new files |
| `causal-edge` new accessors | Tier B (advisory) | **Tier A for new accessor methods** | v2 accessor methods + bit-layout module |
| `lance-graph` core | Tier B (continue-on-error) | Tier B (maintained) | ~91 pre-existing violations (TD-CLIPPY-LG-1); not sprint-11's debt |
| `lance-graph-planner` | Tier B | Tier B | No sprint-11 changes to planner core |

### 6.3 `style.yml` Extension

After PR-CE64-MB-1 lands, add to `style.yml` Clippy section:
```yaml
- name: Clippy par-tile (Tier A, mandatory)
  run: cargo clippy --manifest-path crates/par-tile/Cargo.toml --lib --tests -- -D warnings
```

After PR-CE64-MB-3/5/6 land, each new-file module in existing crates gets a targeted Tier A check using `--lib` scoped to the new module's manifest path.

### 6.4 Format Gate

All sprint-11 PRs must pass `cargo fmt --check`. New crates must be added to workspace `Cargo.toml` members list so `cargo fmt` covers them automatically.

---

## §7 Integration Test Plan (Cross-PR, ~30 Tests)

Integration tests are staged: some run once PR-CE64-MB-1 lands; others require the full chain.

### 7.1 par-tile × causal-edge (after PR-CE64-MB-1 + PR-CE64-MB-2)

**Test file:** `crates/causal-edge/tests/par_tile_integration.rs`

| Test | Assertion |
|---|---|
| `inmemory_mailbox_causal_edge_round_trip` | CausalEdge64 emit into InMemoryMailbox, recv, decode: all fields preserved |
| `causal_edge_v2_in_mailbox_capacity` | InMemoryMailbox with capacity=100 fills to 100 CausalEdge64 entries without drop |
| `causal_edge_g_slot_preserved_in_mailbox` | G-slot (bits 51-55) unchanged after send/recv |
| `causal_edge_truth_band_preserved` | Truth band (bits 62-63) unchanged after mailbox transit |

### 7.2 par-tile × cognitive-shader-driver (after PR-CE64-MB-1 + PR-CE64-MB-3)

**Test file:** `crates/cognitive-shader-driver/tests/par_tile_integration.rs`

| Test | Assertion |
|---|---|
| `bindspaceview_borrows_from_mailbox_soa` | MailboxSoA row borrow produces valid BindSpaceView with correct row range |
| `collapsegate_superposition_via_mailbox_soa` | Two complementary compartments emit XOR-equal edges; CollapseGate::Superposition preserves both |
| `entity_type_column_h_visible_via_bindspaceview` | EntityTypeId written via Column H visible through BindSpaceView row borrow |
| `awareness_column_f_gated_write_via_compartment` | AwarenessColumn write through CollapseGate modifies only emitting compartment's row range |

### 7.3 par-tile × lance-graph-supervisor (after PR-CE64-MB-1 + PR-CE64-MB-5 + PR-CE64-MB-6)

**Test file:** `crates/lance-graph-supervisor/tests/par_tile_integration.rs`

| Test | Assertion |
|---|---|
| `attentionmask_actor_spawns_as_child_of_callcenter_supervisor` | `AttentionMaskActor` spawns under `CallcenterSupervisor`; parent `pid()` is supervisor |
| `sigma_tier_router_sibling_of_attentionmask_actor` | `SigmaTierRouter` and `AttentionMaskActor` are siblings in the actor tree |
| `sigma_6_compartment_uses_inmemory_mailbox` | Σ6 spawn uses `InMemoryMailbox` backing, not TokioMailbox |
| `sigma_9_escalation_reaches_callcenter_supervisor` | Σ9 EPIPHANY witness emission arrives at `CallcenterSupervisor` |

### 7.4 lance-graph-supervisor × lance-graph-planner (after PR-CE64-MB-6)

**Test file:** `crates/lance-graph-supervisor/tests/planner_integration.rs`

| Test | Assertion |
|---|---|
| `sigma_9_escalation_reaches_planner_strategy_registry` | Σ9 escalation path invokes `lance-graph-planner` strategy registry lookup |
| `sigma_10_epiphany_triggers_jit_compile` | Σ10 EPIPHANY compartment spawn triggers JIT compile via `KernelHandle` if not cached |
| `kernel_handle_cached_on_second_sigma_10_spawn` | Second Σ10 spawn with same style-slot returns cached `KernelHandle` without recompile |

### 7.5 lance-graph-supervisor × AriGraph (after PR-CE64-MB-4 + PR-CE64-MB-6)

**Test file:** `crates/lance-graph-supervisor/tests/arigraph_integration.rs`

| Test | Assertion |
|---|---|
| `sigma_router_ghost_edge_lands_in_arigraph_spo_g` | SigmaTierRouter ghost-edge emission persists as SPO-G quad with correct G-slot |
| `arigraph_ghost_reactivation_spawns_compartment` | Evidence for ghost edge triggers SigmaTierRouter to spawn fresh Σ7 compartment |
| `nars_decay_on_ghost_reduces_confidence` | NARS truth-revise on ghost quad reduces confidence monotonically |
| `epiphany_witness_emits_spo_g_quad_with_correct_domain` | Σ9 EPIPHANY produces SPO-G quad where G = active OGIT domain pointer |

### 7.6 bevy-cull-plugin × par-tile (after PR-CE64-MB-7)

**Test file:** `crates/bevy-cull-plugin/tests/par_tile_integration.rs`

| Test | Assertion |
|---|---|
| `cull_system_spawns_compartments_per_frame` | 1K-entity bevy scene; cull system spawns at least 1 compartment per visible entity per frame |
| `viewvisibility_matches_stock_bevy_cull` | Per-entity `ViewVisibility` from plugin matches stock `check_visibility` for all 1K entities |
| `cull_system_drops_compartments_after_frame` | MailboxSoA row count returns to 0 after frame (no compartment leaks) |
| `cull_system_handles_empty_scene` | 0 entities; cull system runs without panic |

### 7.7 Full Pipeline End-to-End (after all 7 lance-graph PRs)

**Test file:** `tests/e2e/pipeline_e2e.rs` (workspace-level)

Exercises the full data flow from parent plan §1:
```
YAML agent card → ThinkingStyle (8-bit slot via AttentionMask)
  → KernelHandle JIT cached → compartment spawn (MailboxSoA, Σ7)
  → CausalEdge64 emit → CollapseGate Bundle → EdgeColumn write
  → AriGraph SPO-G commit → SigmaTierRouter Σ9 on high-surprise
```

| Test | Assertion |
|---|---|
| `full_pipeline_yaml_to_arigraph` | YAML agent card input → AriGraph SPO-G quad persisted with correct (S, P, O, G) |
| `full_pipeline_thinking_style_resolves_kernel_handle` | ThinkingStyle slot lookup produces KernelHandle; second call returns same handle |
| `full_pipeline_ghost_emitted_for_low_evidence` | Low-evidence CausalEdge64 creates ghost edge (Pearl rung 3) in AriGraph |
| `full_pipeline_sigma9_triggers_on_high_surprise` | High-surprise compartment (truth band `11`) escalates to Σ9 |

---

## §8 Perf Benchmark Suite

Benchmarks use `criterion`. They live in `crates/<crate>/benches/*.rs`. CI runs on `merge_group` and `push` to main only. Absolute latency targets are calibrated for ubuntu-24.04 github-hosted CI runner (2-core, 7 GB RAM).

### 8.1 InMemoryMailbox Latency

**File:** `crates/par-tile/benches/inmemory_mailbox.rs` — Driver: W7 spec

| Benchmark | Metric | Target |
|---|---|---|
| `inmemory_mailbox_send_recv_p50` | p50 single-threaded round-trip | < 200 ns |
| `inmemory_mailbox_send_recv_p99` | p99 single-threaded round-trip | **< 500 ns** (W7 cycle-speed budget) |
| `inmemory_mailbox_send_recv_p999` | p999 single-threaded | < 5 µs |
| `inmemory_mailbox_throughput_10k` | 10K messages/sec sustained | > 5M msg/s |

### 8.2 AttentionMask Bind/Lookup Latency

**File:** `crates/par-tile/benches/attention_mask.rs` — Driver: W6 spec

| Benchmark | Metric | Target |
|---|---|---|
| `attention_mask_bind_g_hot` | Bind G-slot (cache-hot) | < 50 ns |
| `attention_mask_lookup_g_hot` | Lookup G-slot (cache-hot) | **< 100 ns** |
| `attention_mask_lookup_g_cold` | Lookup triggering LRU eviction | **< 1 µs** |
| `attention_mask_resolve_g_cold_start` | Full cold-start (bind + lookup + evict) | < 2 µs |

### 8.3 MailboxSoA dispatch_cycle Throughput

**File:** `crates/par-tile/benches/mailbox_soa.rs` — Driver: W6 spec

| Benchmark | Metric | Target |
|---|---|---|
| `mailbox_soa_dispatch_100_compartments` | `dispatch_cycle()` latency for N=100 active rows | < 20 µs |
| `mailbox_soa_dispatch_1000_compartments` | N=1000 | < 200 µs (linear) |
| `mailbox_soa_dispatch_10000_compartments` | N=10000 | < 2 ms (linear) |
| `mailbox_soa_push_row_throughput` | Row push rate | > 1M rows/s |

### 8.4 AriGraph SPO-G Insert Throughput

**File:** `crates/lance-graph/benches/arigraph_spo_g.rs` — Driver: W5 spec

| Benchmark | Metric | Target |
|---|---|---|
| `arigraph_spog_insert_small` | 100-quad table insert | > 500K quads/s |
| `arigraph_spog_insert_medium` | 10K-quad table | > 100K quads/s |
| `arigraph_spog_query_by_g` | Query-by-G on 10K-quad table | < 10 µs per query |

### 8.5 CausalEdge64 PAL8 Encode/Decode Throughput

**File:** `crates/causal-edge/benches/pal8_throughput.rs` — Driver: W2/W3 specs

| Benchmark | Metric | Target |
|---|---|---|
| `pal8_encode_throughput` | Encode 10K values | > 50M enc/s (matches v1 baseline) |
| `pal8_decode_throughput` | Decode 10K PAL8-encoded values | > 50M dec/s |
| `pal8_v2_no_regression` | v2 accessors add no overhead vs v1 | delta < 5% |

### 8.6 Bevy Cull Plugin Per-Frame Cost

**File:** `crates/bevy-cull-plugin/benches/cull_bench.rs` — Driver: W9 spec

| Benchmark | Metric | Target |
|---|---|---|
| `cull_1k_entities` | Per-frame cull time, 1K entities | < 2 ms |
| `cull_10k_entities` | 10K entities | < 10 ms |
| `cull_100k_entities` | 100K entities (stress test) | < 50 ms |
| `cull_vs_stock_bevy_1k` | Plugin time / stock `check_visibility` time | ratio < 1.5× |

---

## §9 CI Workflow File Additions

The existing CI structure uses `rust-test.yml` (test gate), `style.yml` (clippy + fmt). No `ci.yaml` exists — do NOT create one. Sprint-11 additions go into `rust-test.yml` as new jobs, plus one new workflow for the extended Miri sweep.

### 9.1 New Jobs in `rust-test.yml`

Each job runs only when the relevant crate's paths are touched (path filter). Full job definitions for implementors:

```yaml
par-tile-tests:
  runs-on: ubuntu-24.04
  timeout-minutes: 15
  steps:
    # [standard checkout + ndarray sibling + toolchain]
    - name: Run par-tile tests (stable)
      run: cargo test --manifest-path crates/par-tile/Cargo.toml
    - name: Run par-tile property tests (nightly)
      run: cargo +nightly test --manifest-path crates/par-tile/Cargo.toml --features proptest

causaledge64-v2-regression:
  runs-on: ubuntu-24.04
  timeout-minutes: 10
  steps:
    - name: PAL8 round-trip + NarsTables invariant + EdgeColumn binary
      run: cargo test --manifest-path crates/causal-edge/Cargo.toml -- pal8 nars_tables edge_column

bindspace-efgh-tests:
  runs-on: ubuntu-24.04
  timeout-minutes: 15
  steps:
    - name: BindSpace E/F/G/H + CollapseGate::Superposition tests
      run: cargo test --manifest-path crates/cognitive-shader-driver/Cargo.toml

arigraph-spo-g-tests:
  runs-on: ubuntu-24.04
  timeout-minutes: 20
  steps:
    - name: Install protobuf-compiler
      run: sudo apt update && sudo apt install -y protobuf-compiler
    - name: AriGraph SPO-G integration tests
      run: cargo test --manifest-path crates/lance-graph/Cargo.toml -- arigraph_spo_g

sigma-router-tests:
  runs-on: ubuntu-24.04
  timeout-minutes: 20
  steps:
    - name: SigmaTierRouter + CallcenterSupervisor integration
      run: cargo test --manifest-path crates/lance-graph-supervisor/Cargo.toml

bevy-cull-plugin-tests:
  runs-on: ubuntu-24.04
  timeout-minutes: 15
  steps:
    - name: Install headless display
      run: sudo apt update && sudo apt install -y xvfb libvulkan-dev
    - name: bevy cull plugin tests (headless)
      run: xvfb-run cargo test --manifest-path crates/bevy-cull-plugin/Cargo.toml

miri-extended:
  runs-on: ubuntu-24.04
  timeout-minutes: 60
  # Run on merge_group and push; NOT on every PR (long-running)
  if: github.event_name != 'pull_request'
  steps:
    - name: Install miri + nextest
      run: |
        rustup +nightly component add miri
        cargo install cargo-nextest
    - name: Run lance-graph Miri sweep
      run: ./scripts/miri-tests.sh
    - name: Run ndarray Miri sweep
      run: |
        cd /home/user/ndarray
        ./scripts/miri-tests.sh
```

**Promotion gate:** Once `miri-extended` stays clean for 30 consecutive days on main, promote from informational to required PR check. 30-day soak validates stability before Miri failures become blocking.

### 9.2 `style.yml` Extension

After each new crate lands, add a new Tier A clippy step:
```yaml
# After PR-CE64-MB-1:
- name: Clippy par-tile (Tier A, mandatory)
  run: cargo clippy --manifest-path crates/par-tile/Cargo.toml --lib --tests -- -D warnings

# After PR-CE64-MB-3 (new column modules only, not legacy bindspace.rs):
- name: Clippy cognitive-shader-driver new columns (Tier A for new files)
  run: |
    cargo clippy --manifest-path crates/cognitive-shader-driver/Cargo.toml \
      -- -D warnings 2>&1 | grep -E "^error" | grep -v "bindspace.rs" | grep -E "column_[efgh]" || true
```

---

## §10 Sprint-11 → Sprint-12+ Deferred Test Items

| Deferred item | Reason | When |
|---|---|---|
| Full vendored-rayon benchmark vs `std::thread::scope` | OQ-5 (rayon vendor decision) deferred; benchmarks cannot run until vendor decision made | Sprint-12 (after OQ-5 ratified) |
| INT4-32D cold-start K-NN benchmark | OQ-4 wiring deferred to `pr-j-1-int4-32d-atoms.md` implementation sprint | Sprint-12 |
| Σ10 epiphany cross-session persistence tests | Requires Lance persistence infra; out of sprint-11 scope | Sprint-13+ |
| Additional bevy plugin tests (Splat / Cognitive / Audio) | Bevy session §Phase 4 items; W9's scope = NdarrayCullPlugin proof only | Sprint-12+ |
| `hpc::framebuffer::pyramid_tests::*` Miri coverage | 19+ min runtime; deferred until fixtures cfg(miri)-shortened | Sprint-12+ |
| BLAS-backed `hpc::blas_level3::*` Miri coverage | BLAS is FFI; Miri cannot cross FFI boundaries | Indeterminate |
| Full E2E benchmark with real lance-graph-planner JIT | JIT benchmarks require Cranelift AOT warmup; first-run cost skews criterion | Sprint-12 |

---

## §11 Risk Matrix

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| W8 dispatch reroute leaks nightly feature into stable compilation | Medium | High | W8 must use `#[cfg(all(feature = "nightly-simd", miri))]` not bare `#[cfg(miri)]`; CI stable build must pass without `nightly-simd` |
| `crossbeam-channel` Miri compatibility breaks on upstream update | Low | Medium | Pin crossbeam to known-Miri-clean version in Cargo.toml |
| `par-tile` pulls Miri-incompatible transitive dep | Low | High | No external deps rule for par-tile except `ractor` + crossbeam; validate with `cargo +nightly miri test` before opening PR |
| Miri `hpc::*` tests take too long after W8 reroute | Medium | Medium | Stage exclusion removal: start with small modules (activations, fingerprint), expand after profiling |
| Property tests (proptest) take too long under Miri | Medium | Low | Run with `PROPTEST_CASES=100` under Miri; add to MIRIFLAGS |
| PAL8 binary compat breaks on CausalEdge64 v2 (reserved bits were 0 in v1) | Medium | High | W3 regression test suite gates PR-CE64-MB-2 merge; v1 readers must get zero from bits 51-63 in v2 data (iron rule from parent plan §3) |
| bevy headless tests flaky on CI (xvfb race) | Medium | Low | Use `bevy/headless` feature + `--no-default-features` to skip window/render; test only the cull logic |

---

## §12 Iron-Rule Compliance Cross-Check

Per parent plan §12, every test in this plan must be consistent with the workspace's iron rules.

| Iron rule | Test plan compliance |
|---|---|
| **I-SUBSTRATE-MARKOV** (VSA bundling = Chapman-Kolmogorov semigroup) | No test bundles `CausalEdge64` via VSA. CausalEdge64 carries palette INDICES (identities). Tests verify accessor orthogonality and PAL8 round-trip — scalar operations, not VSA algebra. |
| **I-NOISE-FLOOR-JIRAK** (classical Berry-Esseen wrong under weak dependence) | Plasticity counter monotonicity test does not claim a statistical significance threshold. Any threshold-dependent test cites Jirak-derived bound explicitly in its `// INVARIANT:` comment. |
| **I-VSA-IDENTITIES** (VSA on IDENTITY fingerprints, never on content) | No test calls `vsa_bundle(edge1, edge2)` or similar. CausalEdge64 round-trip tests verify the 8-byte scalar. AttentionMask rename tests verify slot assignment (identity renaming). |
| **I1** (BindSpace read-only; writes only through CollapseGate) | Integration tests for BindSpaceView verify that direct write attempts fail at compile time (borrow checker enforces). Tests verify CollapseGate-gated writes succeed; raw alias attempts cannot compile. |
| **Method-on-carrier discipline** | No test calls a free function on a carrier's state. Tests invoke `edge.g_slot()`, `mask.lookup_g(slot)`, `soa.dispatch_cycle()` — all methods on carriers. |
| **Zone serialization rule** | No integration test deserializes a Wire DTO inside Zone 1 or 2. The pipeline E2E test (§7.7) verifies CausalEdge64 stays as `u64` inside Zones 1/2; becomes Wire DTO only at Zone 3 egress mock. |

---

## §13 Open Questions for Meta-Review

**OQ-T1 — W1-W9 test count reconciliation.** This spec uses parent-plan-derived estimates because W1-W9 specs do not exist at W11 draft time. Meta-reviewer must reconcile actual W1-W9 counts against §3 table. If any PR has significantly fewer tests than estimated, flag it.

**OQ-T2 — proptest Miri run time.** Running proptest with default 256 cases under Miri is untested. This spec recommends `PROPTEST_CASES=100` as mitigation. W8's PR description should include a proptest Miri time measurement to validate this assumption.

**OQ-T3 — bevy headless rendering approach.** This spec assumes `xvfb-run` + `libvulkan-dev` for headless rendering. An alternative is `bevy/headless` feature with `WinitPlugin` disabled. Meta-reviewer should confirm which approach W9's spec mandates after W9 drafts.

**OQ-T4 — causal-edge Miri scope.** `crates/causal-edge/` is not currently in `MIRI_SAFE_CRATES`. Adding it requires verifying its dependency closure has no FFI. The parent plan states causal-edge is pure Rust + NarsTables LUT (no BLAS, no lance FFI). W2's spec should confirm or deny this; if denied, §5.1 property test invocation must be removed.

**OQ-T5 — Miri and `ractor` in lance-graph-supervisor.** The supervisor Miri inclusion assumes `SigmaTierRouter`'s Miri path excludes Tokio-backed ractor shapes via `#[cfg(not(miri))]`. If W7's spec does not include these guards, the Miri sweep for `lance-graph-supervisor` will abort on Tokio FFI. Meta-reviewer must verify W7's spec addresses this explicitly.

---

## Appendix A: Existing CI Structure (Sprint-10 Baseline)

```
.github/workflows/
  rust-test.yml    — cargo test on lance-graph core + contract; llvm-cov coverage
  style.yml        — clippy Tier A (contract, mandatory) + Tier B (lance-graph, advisory) + rustfmt
  build.yml        — cargo build check
  jc-proof.yml     — JC pillar proofs
  release.yml      — release automation
  rust-publish.yml — crate publishing
```

No `ci.yaml` file exists. Sprint-11 implementors must add jobs to `rust-test.yml` per §9.1 above. No Miri CI job exists yet; `miri-extended` (§9.1) is the first.

Current Tier A clippy gate (mandatory, blocking):
```sh
cargo clippy --manifest-path crates/lance-graph-contract/Cargo.toml --lib --tests -- -D warnings
```

Current Tier B clippy gate (advisory, `continue-on-error: true`):
```sh
cargo clippy --manifest-path crates/lance-graph/Cargo.toml --lib --tests -- -D warnings
```

---

## Appendix B: Source Cross-References

| Source | Section used | Key information extracted |
|---|---|---|
| `causaledge64-mailbox-rename-soa-v1.md` §3 | §3.1 (v2 layout) | CausalEdge64 bit-field layout table; G bits 51-55, W bits 56-61, truth bits 62-63 |
| `causaledge64-mailbox-rename-soa-v1.md` §5 | §3.1 (MailboxSoA) | D-CE64-MB-7/8/9 deliverables; property test requirements |
| `causaledge64-mailbox-rename-soa-v1.md` §6 | §3.1 (per-crate) | Crate change inventory; LOC estimates per PR |
| `causaledge64-mailbox-rename-soa-v1.md` §7 | §3 table | PR sequencing; parallel-landability |
| `causaledge64-mailbox-rename-soa-v1.md` §8 | §4.1 | ndarray prerequisites; PR-NDARRAY-MIRI-COMPLETE scope |
| `causaledge64-mailbox-rename-soa-v1.md` §12 | §12 | Iron-rule compliance table |
| `scripts/miri-tests.sh` | §4.1, §5.1 | Current lance-graph Miri scope; MIRI_SAFE_CRATES list |
| `ndarray/scripts/miri-tests.sh` | §4.1, §5.2 | Current ndarray Miri scope; 3-clause filter rationale |
| `.github/workflows/rust-test.yml` | Appendix A, §9 | Existing CI job structure; no ci.yaml |
| `.github/workflows/style.yml` | §6, §9.2 | Clippy Tier A/B structure; format gate |
| `MANIFEST.md` sprint-log-10 | §3 | Fleet worker → PR mapping |
| `LATEST_STATE.md` | §1 (context) | Sprint-7 landed (#366); existing supervisor shape |
