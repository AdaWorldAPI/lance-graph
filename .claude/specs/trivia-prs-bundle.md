# Trivia PRs bundle — three quick wins

**Sprint:** Sprint-3
**Author:** W12 (Sprint-3 worker agent)
**Branch target:** `claude/tier-1-implementation-specs`
**Closes:** TD-CAM-DIST-REGISTRATION-9, TD-ADJ-THINK-EXPOSE-10, TD-DEEPNSM-NSM-COLLAPSE-11
**Total effort:** <1 day for one engineer to ship all three.
**Risk:** None / Low / Low (in PR order). Parallel-shippable with the
sprint-3 main critical path (PR-B-1 → PR-A-1 → ...).

These three PRs cash in the W6 ledger reframes: substrate that already
exists (cam_distance UDF, p64-bridge planes, deepnsm root crate) just
needs a one-line registration / a public method / a shim re-export.

---

## PR-CAM-DIST: register cam_distance UDF globally (TD-CAM-DIST-REGISTRATION-9)

**Effort: trivial (1 line + 1 test).**

### Goal

Move `cam_distance` UDF registration from opt-in to default in
`DataFusionPlanner`. Per the W6 ledger reframe, the substrate is
shipped (cam_pq/udf.rs:241/257/326); only the registration in
`DataFusionPlanner::new` is missing.

### Files

- `crates/lance-graph/src/datafusion_planner/mod.rs` —
  `DataFusionPlanner::new`. Add line:

  ```rust
  state = lance_graph::cam_pq::udf::register_cam_distance(state);
  ```

- `crates/lance-graph/tests/cam_distance_default.rs` — new 1-test file:

  ```rust
  #[test]
  fn default_planner_resolves_cam_distance_udf() {
      let planner = DataFusionPlanner::new(default_config(), default_runtime());
      let result = planner.execute("SELECT cam_distance(0, 1)").await;
      assert!(result.is_ok());
  }
  ```

### Acceptance

- [ ] `DataFusionPlanner::new` always registers `cam_distance`.
- [ ] 1 test green.
- [ ] CAM-DIST-1 entropy 3 → 2 (post-PR ledger update).

### Risk

None — pure additive registration; no consumer can break because no
consumer was using the opt-in registration path in production code.

---

## PR-ADJ-THINK-EXPOSE: tau_write() public API (TD-ADJ-THINK-EXPOSE-10)

**Effort: light (~30 LOC + 1 test).**

### Goal

Per the W6 ledger reframe, the `[u64; 64] × 8` planes inside
`p64-bridge::CognitiveShader` ALREADY ARE the ThinkingAdjacency
adjacency store. We just need to expose a public write API at the
τ-prefix `0x0D` slots that the I5 doctrine specifies.

### Files

- `crates/p64-bridge/src/lib.rs::cognitive_shader` — add
  `CognitiveShader::tau_write(&mut self, style_a: u8, style_b: u8, layer: u8)`
  and the matching read:

  ```rust
  impl<'a> CognitiveShader<'a> {
      pub fn tau_write(&mut self, style_a: u8, style_b: u8, layer: u8) {
          let block_row = (style_a as usize) / 4;
          let block_col = (style_b as usize) / 4;
          if block_row < 64 && block_col < 64 && layer < 8 {
              // The τ-prefix 0x0D corresponds to layer 5 (ABSTRACTS) per W6 reframe.
              self.planes[layer as usize][block_row] |= 1u64 << block_col;
          }
      }

      pub fn tau_read(&self, style_a: u8, style_b: u8, layer: u8) -> bool {
          let block_row = (style_a as usize) / 4;
          let block_col = (style_b as usize) / 4;
          if block_row < 64 && block_col < 64 && layer < 8 {
              (self.planes[layer as usize][block_row] & (1u64 << block_col)) != 0
          } else {
              false
          }
      }
  }
  ```

- `crates/p64-bridge/tests/tau_round_trip.rs` — new test:

  ```rust
  #[test]
  fn tau_write_then_read() {
      let mut planes = [[0u64; 64]; 8];
      let palette = build_test_palette(16);
      let semiring = PaletteSemiring::build(&palette);
      let mut shader = CognitiveShader::new(planes, &semiring);

      shader.tau_write(8, 12, 5);
      assert!(shader.tau_read(8, 12, 5));
      assert!(!shader.tau_read(8, 12, 6));  // different layer
  }
  ```

### Acceptance

- [ ] `tau_write` + `tau_read` public methods on `CognitiveShader`.
- [ ] 1 round-trip test green.
- [ ] ADJ-THINK-1 entropy 4 → 2 (post-PR ledger update).

### Risk

Low — additive public API on an existing type. The bit-layout (block_row =
style/4, layer ∈ [0,8)) is identical to what the substrate already uses
internally for the planes; we are exposing the same arithmetic, not
inventing a new index scheme.

### Open question for the engineer

The spec encodes `// τ-prefix 0x0D corresponds to layer 5 (ABSTRACTS)` as
a comment, not an enforced invariant. If callers should be forced through
a `TauPrefix::Abstracts` enum that maps to layer 5, raise that as a
follow-up — it is not required for the entropy reduction and would balloon
the PR beyond "trivia."

---

## PR-DEEPNSM-NSM-COLLAPSE: delete `lance-graph/src/nsm/` shim (TD-DEEPNSM-NSM-COLLAPSE-11)

**Effort: small (~30 LOC + 5 file deletes; existing tests cover).**

### Goal

Per LADYBUG-EQUIV-1
(`.claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md`): `nsm/` is
migration debt from external repo → embedded module → root crate. The
user clarified ages ago that promotion to root crate was the goal; the
embedded module is orphan residue.

### Files

DELETE:

- `crates/lance-graph/src/nsm/encoder.rs`    (~15 KB)
- `crates/lance-graph/src/nsm/parser.rs`     (~22 KB)
- `crates/lance-graph/src/nsm/similarity.rs` (~6.5 KB)
- `crates/lance-graph/src/nsm/tokenizer.rs`  (~18 KB)
- `crates/lance-graph/src/nsm/nsm_word.rs`   (~16.6 KB)

REPLACE: `crates/lance-graph/src/nsm/mod.rs` (491 B → ~30 LOC re-export
shim):

```rust
//! NSM module — thin shim that re-exports from the deepnsm root crate.
//! All types previously defined here moved to crates/deepnsm/ in PR ____
//! (planned: this PR).
//! See LADYBUG-EQUIV-1 in
//! .claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md
//! for migration history.

pub use deepnsm::encoder;
pub use deepnsm::parser;
pub use deepnsm::similarity;
pub use deepnsm::vocabulary as tokenizer;
pub use deepnsm::{codebook, pos, nsm_primes};  // covers nsm_word's content
```

### Pre-deletion verification (run via grep)

- `cargo check --workspace` succeeds (no consumers reference deleted-only
  types).
- If a test or downstream module DOES use `nsm_word::SomeType`, the
  re-export `pub use deepnsm::codebook::SomeType` covers it. The shim
  preserves every public path the previous embedded module exported.

### Acceptance

- [ ] 5 `.rs` files deleted.
- [ ] `mod.rs` replaced with shim.
- [ ] `cargo check --workspace` clean.
- [ ] All existing nsm-touching tests still pass via the shim.
- [ ] DEEPNSM-NSM-1 ledger row state-change appended (Migration debt →
      Closed; entropy 5 → 1).

### Risk

Low if the grep verification is clean; medium if a hidden import surfaces
(some downstream module quietly using a private helper from one of the
deleted files). Mitigation: the engineer runs `cargo check --workspace`
BEFORE pushing the deletion commit, then iterates on the shim re-export
list until clean.

---

## Bundle effort summary

| PR | Effort | Risk | Lines touched |
|---|---|---|---|
| PR-CAM-DIST | trivial | None — pure additive | +1 src, +1 test file |
| PR-ADJ-THINK-EXPOSE | light | Low — additive public API | ~30 LOC + 1 test |
| PR-DEEPNSM-NSM-COLLAPSE | small | Low if grep clean; medium if hidden import | -78 KB src, +30 LOC shim |

**Total:** ~1 working day for all three. **Net entropy delta: −5**
(CAM-DIST-1 3→2, ADJ-THINK-1 4→2, DEEPNSM-NSM-1 5→1).

These can ship in parallel with sprint-3 main critical path (PR-B-1 →
PR-A-1 → ...). None of them touch `SpoQuad`, `OntologyRegistry`, or
`OrchestrationBridge`, so no merge contention with W2/W3/W4.

### Recommended ship order

1. **PR-CAM-DIST first** — trivial diff, fastest review, immediate
   entropy win, zero coordination cost.
2. **PR-ADJ-THINK-EXPOSE second** — additive only, but the engineer
   should sanity-check that no other code path is already calling a
   private `tau_*` helper (grep `tau_write\|tau_read`).
3. **PR-DEEPNSM-NSM-COLLAPSE last** — has the highest blast radius
   (workspace-wide build check) and the most physical churn (file
   deletions). Land it after the two trivial ones to keep the bisect
   surface clean.

---

## Cross-references

- `.claude/board/TECH_DEBT.md` — TD-CAM-DIST-REGISTRATION-9,
  TD-ADJ-THINK-EXPOSE-10, TD-DEEPNSM-NSM-COLLAPSE-11.
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER_RESOLVED.md` —
  LADYBUG-EQUIV-1 (nsm migration history).
- `.claude/board/ARCHITECTURE_ENTROPY_LEDGER.md` — W6 reframes for
  CAM-DIST-1 and ADJ-THINK-1 (substrate-already-shipped findings).
- `.claude/specs/sprint-3-execution-plan.md` — W1 master plan.
- `.claude/specs/pr-a-1-spo-g-u32-slot.md` — sister Tier-1 spec
  (independent; no dependency).

---

## Open questions for the engineer

1. **PR-CAM-DIST signature.** The spec shows
   `register_cam_distance(state)` as taking and returning the
   DataFusion `SessionState`. Confirm against `cam_pq/udf.rs:241` that
   this matches the actual function signature; if it has been refactored
   to take `&mut SessionState`, adjust the one-liner accordingly.
2. **PR-ADJ-THINK-EXPOSE layer-5 invariant.** Should `tau_write` reject
   `layer != 5` for τ-prefix `0x0D`, or stay general (any layer 0..8)?
   Recommendation: stay general; the τ-prefix-to-layer mapping is a
   caller concern, not a substrate concern.
3. **PR-DEEPNSM-NSM-COLLAPSE re-export completeness.** Run
   `rg -l 'use crate::nsm::|use lance_graph::nsm::'` across the
   workspace before drafting the shim. Any unique import path not
   covered by the five `pub use deepnsm::…` lines means an additional
   re-export is needed.
