# W3+W4a — atomic implementation plan (code-grounded, v1)

> **Status:** v2 — 5-consolidation + 3-brutal-critic pass APPLIED (see § "v2 critique
> integration" below). v1 brutal verdict was **HOLD** (3 P0s); v2 integrates every
> P0/P1/P2. Architecture confirmed sound by all 8 angles (dto-soa ACCEPT, convergence
> single-body confirmed, firewall source-CLEAN, preflight zero-P0-drift); the fixes are
> the write-shim design, the prefilter window, OQ-D resolution, and parity-surface completion.
> **Date:** 2026-06-17. **Branch:** `claude/bindspace-mailbox-soa-w3-w4a`.
> **Parent plan:** `bindspace-singleton-to-mailbox-soa-v1.md` (the W2→W7 wiring plan on main;
> the v2 wiring snapshot lives on branch `claude/bindspace-mailbox-soa-wiring-plan`).
> **Unblocked:** W1 (#517), W1b (#518), W1c (#519) are ALL merged to `main`.
> The wiring plan's "BLOCKED:#518" gates are cleared. W2 (differential harness)
> and W3+W4a (atomic read/write re-point) are now buildable.
> **Operator constraints (binding):** two paths step by step; test the new before
> deleting the old; CausalEdge64 dedup precise; delete BindSpace LAST.

## v2 critique integration — what the 5+3 pass changed (HOLD → build-ready)

**Panel:** 5 consolidation (integration-lead, truth-architect, dto-soa-savant,
convergence-architect, firewall-warden) + 3 brutal (baton-handoff-auditor,
brutally-honest-tester, preflight-drift-auditor). Verdicts: architecture SOUND
(dto-soa ACCEPT enum-over-trait; convergence single-body confirmed + `edge()`
collapse real; preflight zero-P0-drift, line numbers exact; firewall source-CLEAN).
Blocking gaps fixed below.

- **C1 — write surface (was P0; baton P0-2, brutal-tester P0-1).** The read shim is
  `&self`; the WRITE path is separate — `driver.run()` writes NO bindspace columns
  (verified: zero `self.bindspace.*.set/write_`). Writes are engine_bridge **free fns**
  (`dispatch_busdto:234`, `persist_cycle:706`, **`ingest_codebook_indices:53`** — the
  4th writer v1 missed) called by **bins** (`serve.rs:156/695`, `grpc.rs:140`) and tests.
  The bin call-site conversion (`Arc::get_mut` → owned `&mut MailboxSoA`) is **W4b**, not
  this PR. **W3+W4a delivers the write SHIM** (a `BackingStoreWrite<'a>` `&mut` enum mirror
  with `set_content/set_qualia/set_edge/set_meta/set_entity_type/set_temporal/set_expert/
  set_sigma`; `set_edge` wraps `u64→CausalEdge64` on the singleton arm — `bindspace`
  `edges.set(u64)`:132 vs mailbox `set_edge(CausalEdge64)`:414) **exercised only by the
  W2 differential harness**, feature **default-OFF**. Production stays singleton-read +
  singleton-write until W4b/W7 → H-DW-1 satisfied (under the feature BOTH surfaces flip
  in the harness; production never half-flips).
- **C2 — prefilter window (was P0; convergence, baton P1-2, brutal-tester P0-3).** Mailbox
  arm MUST iterate `win.start.min(populated)..win.end.min(populated)`, matching
  `BindSpace::meta_prefilter`'s `start..end` (bindspace.rs:358-362). `0..populated` is a
  sentinel-lie: identical Vec shape, divergent rows on any `row_start>0` (wire.rs/grpc.rs
  pass `req.row_start`). **W2 differential MUST include a non-full-window case**
  (e.g. `ColumnWindow::new(1, len-1)`) or the gate passes green on the bug.
- **C3 — OQ-D MailboxId selection (was P0; integration-lead, baton P0-3, brutal-tester P1-2).**
  `ShaderDispatch` carries no `MailboxId` (cognitive_shader.rs:178-209). **Resolution
  (Option A, NO contract change):** under the feature the driver selects a single designated
  mailbox via a driver-crate `const DEFAULT_MAILBOX: MailboxId = 0` (`MailboxId` is a `u32`
  alias — collapse_gate.rs:121 — so it's a free const, not an associated const) +
  `debug_assert!(self.mailboxes.len() == 1)`. Multi-mailbox routing is **W5**. Reject
  Option B (adding `mailbox_id` to `ShaderDispatch` = zero-dep-contract change, W5 scope).
- **C4 — parity surface completion (was incomplete; truth-architect).** The W2 differential
  asserts the **whole `ShaderCrystal` field-by-field** (every f32 via `to_bits()`), NOT a
  hand-picked subset. v1 missed: `hit_count`, **`cycles_used`**, **`style_ord`** (the two
  prefilter-order canaries), `emitted_edges`+`emitted_edge_count`, `gate`, `persisted_row`,
  all of `MetaSummary` (3×f32+bool), `alpha_composite`, `ShaderHit._pad`. `to_bits()` is
  REQUIRED (same `run()` body + identical read bytes = bit-identical arithmetic; ULP
  tolerance would mask exactly the read-path bug this differential exists to catch).
- **C5 — unbind_busdto downgrade (was P1; baton P0-1, truth C1/C2, brutal-tester P1-1).**
  Feature-gate the lossy narrowing with a doc migration pointer (I-LEGACY-API-FEATURE-GATED).
  Keep the singleton arm's bit-exact dense-top_k assertion live via
  `#[cfg(not(feature="mailbox-thoughtspace"))]`; add a SEPARATE explicitly-lossy mailbox-arm
  assertion that non-headline `top_k[1..].0` recover as `0` (pin the loss, don't tolerate it).
  Never relax the existing test in place. **OQ-B = NOW-OK to land** (loss is
  `#[cfg(with-engine)]` lab-path, not read by live `run()`, extends an already-lossy
  register recovery per I-VSA-IDENTITIES) — no new column needed.
- **C6 — firewall lint must be concrete (was BLOCK; firewall-warden).** Replace the prose
  with a real `Grep`-CI rule failing on `(causal_diff|thinking_engine::layered)\s*::\s*CausalEdge64`
  + the `... as` aliased forms, scoped to `crates/cognitive-shader-driver/`, AND explicitly
  allowing the `repr(transparent)` cast sites at mailbox_soa.rs:606/620 (else false-positive).
  Or a `compile_fail`/trybuild guard. Source is CLEAN today (no twin imports) — the gate must
  have teeth.
- **C7 — resolved-favorably / notes.** `expert_at`:472 / `set_expert`:478 EXIST → `cycle_count`
  is LOSSLESS on the mailbox arm (truth escalation closed). Feature-gate asymmetry (preflight P2):
  `dispatch_busdto`/`unbind_busdto` are `#[cfg(with-engine)]`, `persist_cycle`/`ingest_codebook_indices`
  are ungated (default build) → the writer re-point reconciles `with-engine ∧ mailbox-thoughtspace`.
  `set_edge` is at :414 (v1 said :411 — doc-comment line). Board hygiene: the build PR updates
  `STATUS_BOARD.md` + `INTEGRATION_PLANS.md` in the same commit.

## Ground truth (verified against `main` HEAD d2f9b7d9, NOT guessed)

### Dispatch read surface — `driver.rs::run()` (the 6 reads to re-point)
| line | read | shim method |
|---|---|---|
| 172 | `self.bindspace.meta_prefilter(req.rows, &req.meta_prefilter)` | `prefilter` |
| 178, 455 | `self.bindspace.qualia.row(row).to_f32_17d()` | `qualia_row` |
| 204, 206, 290 | `self.bindspace.fingerprints.content_row(row)` | `content_row` |
| 244 | `CausalEdge64(self.bindspace.edges.get(row))` | `edge` |
| 356 | `self.bindspace.entity_type[r]` | `entity_type` |
| 360 | `self.bindspace.ontology()` | **NOT shim** — `Arc<OntologyRegistry>` on driver (W4b) |
| 661 | `row_count` = `self.bindspace.len` | `len` |
| 665 | `byte_footprint` = `self.bindspace.byte_footprint()` | (trait, both impl) |
| 556 | `self.awareness.write()` | unchanged (leaf RwLock) |
| 288 | `cycle_fp = [0u64; WORDS_PER_FP]` transient fold | unchanged (stack, never a column) |

### Mailbox accessor surface — `mailbox_soa.rs` (verified signatures)
- `content_row(row) -> &[u64]` (500) — byte-identical stride to `bs.fingerprints.content_row`
- `qualia_at(row) -> QualiaI4_16D` (420)
- `edge(row) -> CausalEdge64` (405) — **stores `causal_edge::CausalEdge64` NATIVELY** (`self.edges[row]`)
- `entity_type_at(row) -> u16` (444)
- `meta_at(row) -> MetaWord` (432) — no `meta_raw`; `MetaFilter::accepts` takes `MetaWord`
- `populated() -> usize` (317) — the prefilter bound (W1c)
- write side: `set_content` (506), `set_qualia` (426), `set_edge` (411), `apply_edges` (253)
- **no `meta_prefilter`** → shim mailbox arm iterates **`win.start.min(populated)..win.end.min(populated)`** ascending, `MetaFilter::accepts` (v2 P0-B: NOT `0..populated` — `BindSpace::meta_prefilter` honors the dispatch `ColumnWindow` start; wire/grpc pass non-zero `row_start`)
- `edges_raw()` (:592) and `meta_raw()` (:609) DO exist as `MailboxSoaView` trait methods (`unsafe repr(transparent)` casts) — the shim does NOT use them (typed `edge()`/`meta_at()` path is used instead; see CausalEdge64 resolution below). The firewall lint must allow the `repr(transparent)` cast sites.

### Writer surface — `engine_bridge.rs` (3 fns take BindSpace directly)
- `dispatch_busdto(bs: &mut BindSpace, row, bus, style_ord) -> usize` (234) → `bs.write_cycle_fingerprint` (243)
- `unbind_busdto(bs: &BindSpace, row) -> BusDto` (315) → reads `bs.fingerprints.cycle_row(row)` (334)
- `persist_cycle(bs: &mut BindSpace, row, bus, style_ord)` (706) → `bs.write_cycle_fingerprint` (707)

## RESOLVED: CausalEdge64 firewall (plan's `edges_raw()` worry is moot)

The wiring plan (baton-handoff P1) said "use `mb.edges_raw()[row]` (raw), not
`mb.edge(row).0` (typed bounce)." **Ground truth overturns the premise:**
`MailboxSoA` stores `CausalEdge64` natively (`edges: [CausalEdge64; N]`,
`edge(row)` returns `self.edges[row]` — already typed). `mailbox_soa` and
`driver` both reference the SAME re-exported `causal_edge::CausalEdge64`
(`causal_edge::CausalEdge64` == `causal_edge::edge::CausalEdge64`).

**Resolution — shim `edge` returns the typed `causal_edge::CausalEdge64`:**
- Singleton arm: `CausalEdge64(bs.edges.get(row))` (raw u64 wrapped, as today)
- Mailbox arm: `mb.edge(row)` (already typed — ZERO raw bounce)

This is STRICTLY stronger than the plan: no raw u64 ever surfaces on the
mailbox arm, no `edges_raw()` accessor added, firewall is one typed surface.
The CI firewall lint (bar `ndarray::hpc::causal_diff::CausalEdge64` AND
`thinking_engine::layered::CausalEdge64`) still ships.

## The shim — `backing.rs` (new, driver-crate-local, monomorphized)

```rust
// crates/cognitive-shader-driver/src/backing.rs
pub(crate) enum BackingStore<'a> {
    Singleton(&'a BindSpace),       // path A — live default
    #[cfg(feature = "mailbox-thoughtspace")]
    Mailbox(&'a MailboxSoA<1024>),  // path B — feature-gated
}

impl<'a> BackingStore<'a> {
    fn prefilter(&self, win: (u32,u32), f: &MetaFilter) -> Vec<u32>;
    fn content_row(&self, row: usize) -> &[u64];
    fn qualia_row(&self, row: usize) -> QualiaI4_16D;
    fn edge(&self, row: usize) -> CausalEdge64;   // typed both arms
    fn entity_type(&self, row: usize) -> u16;
    fn len(&self) -> usize;
}
```
- `run()` keeps ONE body, written against `BackingStore`. The feature flag
  selects which variant is *constructed* (in `dispatch`/`dispatch_with_sink`),
  NEVER `#[cfg]`-branches inside `run`.
- prefilter mailbox arm clamps to `populated()`, iterates `0..populated` ascending
  so `passed_rows` order matches `BindSpace::meta_prefilter`'s `for row in start..end`.
- `ontology` is NOT a shim method (stays `Arc<OntologyRegistry>`; re-homed at W4b).

## Feature gate (I-LEGACY-API-FEATURE-GATED-safe)
- ONE feature `mailbox-thoughtspace`, default **OFF** until W7.
- Gate the OWNER (which variant is built), not the fn body.
- No per-W sub-features.

## Writer re-point (W4a half — engine_bridge + persist path)
The writers and readers flip **together** in this one PR (H-DW-1: never W3
writers-only then W4 readers-later → silent same-row divergence).
- `dispatch_busdto` / `persist_cycle`: under feature, stop the
  `write_cycle_fingerprint` column write; keep `edges`/`meta`/`qualia` writes
  routed to the mailbox via `set_*`/`apply_edges`. Compute the Binary16K
  accumulator transiently, discard (cycle plane never a mailbox column).
- `unbind_busdto`: REWRITE (not re-point) with a DOWNGRADED contract — only the
  headline index (in `qualia[9]`) survives; non-headline `top_k` indices recovered
  from cycle-plane set-bits are NOT recoverable from mailbox columns (D-DIST-5
  exception). Differential gate + `busdto_bridge_test.rs` tolerance update land
  in THIS PR.

## In-PR gates (all mandatory, per wiring plan §6 W3+W4a)
1. **W2 differential harness** (one driver, two backing arms; same awareness/planes/
   semiring): assert `ShaderResonance` bit-identical via `f32::to_bits()` (top_k
   row/distance/predicates/cycle_index + resonance/entropy/std_dev bits) +
   `cycle_fingerprint` + `MaterializeProvenance`; mailbox built with
   `set_populated(len)` + prefilter clamped to `populated()`. Gate with
   `entity_type==0` rows (neutralizes ontology); second assertion `entity_type!=0`
   after W4b ontology re-home.
2. **BusDto differential**: `unbind_busdto(mailbox)` vs `unbind_busdto(bindspace)` —
   headline + energies + cycle_count + converged parity; non-headline-index loss
   documented in updated tolerance.
3. **temporal field-isolation matrix**: each migrated column write leaves all others
   byte-unchanged; emit-path `driver.rs:402-413 pack(temporal=cycle_index)` either
   lands in mailbox `temporal` column OR paired with `temporal_dropped_under_v2` test.
4. **CausalEdge64 firewall CI lint**: deny/grep that `cognitive-shader-driver`
   imports neither `ndarray::hpc::causal_diff::CausalEdge64` nor
   `thinking_engine::layered::CausalEdge64`; + `edge()` round-trip to typed.
5. **cycle-drop compile+bench**: `MailboxSoA` exposes no `cycle*` symbol; footprint
   ≈ 6 KB/row (content/topic/angle), not ≈ 71.6 KB/row.
6. **`--features mailbox-thoughtspace` CI matrix row** (build + W2 differential),
   live from this PR.

## Explicitly OUT of this PR (deferred to later W-steps)
- W4b: `Arc::get_mut` → owned `&mut MailboxSoA` (4 bin sites) + ontology re-home.
- W5: bins build the mailbox set.
- W6: death→SPO-G+Lance tombstone (HARD-BLOCKED surrealdb #41).
- W7: delete BindSpace + cycle plane + drop feature.

## Open questions for the 5+3 pass
- OQ-A: is W2 (the differential harness) a SEPARATE prior PR, or folded into the
  W3+W4a atomic PR as the gate? (Plan v2 §6 says "run on the W3+W4a atomic PR".)
- OQ-B: `unbind_busdto` rewrite with downgraded contract — acceptable now, or does
  the non-headline-index loss need a new mailbox column FIRST (escalate to operator)?
- OQ-C: should the shim be `enum BackingStore` or a generic `R: DriverRead` trait?
  (Plan: equivalent, monomorphized either way; enum is simpler, no new pub trait.)
- OQ-D: does `dispatch`/`dispatch_with_sink` construct the variant cleanly without
  a `#[cfg]` inside `run`, given `mailboxes: HashMap` is keyed by `MailboxId` — which
  mailbox does a singleton-shaped `ShaderDispatch` select? (req carries no mailbox id today.)
```
