# AUDIT — what needs fixing (2026-07-27, post-§12-trace consolidation)

> Every item is session-verified with evidence; freshness-swept today.
> Owner legend: **CODE-READY** (small, no design needed) · **GATED** (needs the
> operator's design ruling or the P0 path built first) · **USER-OWNED** (outside
> repo / needs operator access or action).

## P0 — the critical path (everything else is downstream of these)

| # | item | evidence | fix shape | owner |
|---|---|---|---|---|
| 1 | **The substrate write path is unbuilt.** `MailboxSoA` never implements `SoaEnvelope`; no Lance-version-producing code exists; `BatchWriter::cast()` has zero production callers. First missing hop: `MailboxSoA → SoaEnvelope seam → live Lance write`. | trace A (`exec-runs/trace-A-write-path.md`); primer §14 | build the seam — the ONE thing that unblocks the standing-wave path | **GATED** (design) |
| 2 | **No production temporal read.** `deinterlace` test-only end-to-end; sole `DeinterlaceRow` impl is a test struct; **no HLC producer exists anywhere in `crates/`**. | trace B | wire one production implementor + an HLC/tick source | **GATED** |
| 3 | **Belief state has no resident home.** Truth: declared only (`MetaWord::{nars_f,nars_c}`), no accessor; rung: shaped enum, zero row wiring; contradiction / premises / evidence: **no tenant at all**. | trace C | tenant decisions + jc-pillar cert per le-contract §3b | **GATED** |
| 4 | **`MetaWord` 8 B/4 B width mismatch** — already flagged open in le-contract; blocks the truth tenant specifically. | trace C | resolve the width before any truth wiring | **GATED** |

P0 sequencing: 4 → 3 → 1 → 2 is the dependency order for the gate path
(`resident tenant → owner mutation → Kanban → cast → Lance position → temporal read`).
Per §15: **there must never be a transfer** — complete this path, then retire heap authority.

## P1 — live defects & misleading surfaces (small, mostly actionable now)

| # | item | evidence | fix shape | owner |
|---|---|---|---|---|
| 5 | **`impl Distance for [u8;6]` measures as pure noise** (ρ −0.0030, recall@10 0.0125 vs exact) — byte-L1 over centroid *indices* is not a metric — and has **zero consumers**, so removal/rename breaks nothing. | probe `probe_palette256_ndarray`; ISSUES §G | delete the impl or rename to `byte_l1_NOT_a_distance`-class name + doc pointer | **CODE-READY** (one small PR) |
| 6 | **`cam.rs` ADC unrouted through `CamCodecContract`** — the standing May debt `cam-pq-production-wiring` (ndarray `cam_pq` shipped, unrouted). The f32 per-query-table shape (6 144 B/query) is the materialization the doctrine forbids. | ndarray EPIPHANIES 2026-05-26; census | wire ndarray's implementation through the contract per the May debt | **GATED** (touches contract surface) |
| 7 | **Arena-internal re-indexing** — `by_sc` rebuilt every closure pass (`belief.rs:285`); 3 independent `deg`/`by_pred`/`by_subj` rebuilds (`tactics.rs:161-168, 181-186, 337-344`). **⊘ DE-ESCALATED by primer §16** (serialization boundary is the forbidden thing): these are tier-2 ephemeral thinking-scratch — waste, not violations. They may never PERSIST as an index (tier-3 with no legitimation), but per-call rebuilds are legal. | trace D; §16 | optional perf work, any time; persistence remains forbidden | **CODE-READY** (optional) |
| 8 | **`rcr_floor_and_budget` pins arena-admission order** via a literal 5-element vector — determinism alone cannot satisfy it. Not a defect today; a migration landmine. | trace D verified by hand | at migration: preserve the order or rewrite the assertion — explicit decision | **GATED** (flagged, no action) |
| 9 | **`cargo fmt` broken on `lance-graph-cognitive`** — `container_bs/mod.rs:35` declares `#[cfg(test)] pub mod tests;`, `tests.rs` does not exist; rustfmt aborts on the whole crate. **Verified still broken today.** | freshness sweep | one line: delete the declaration, or add the file — which one is intended is the only question | **USER-OWNED** (one-line, needs intent call) |
| 10 | **Pre-existing `-D warnings` failures**: `lance-graph-ontology` deprecated `oxrdf::Subject` (verified still firing today); `lance-graph-cognitive` `triangle.rs` derivable-`Default` (unverified today — crate not addressable via workspace clippy). | freshness sweep | mechanical: `Subject → NamedOrBlankNode`; `#[derive(Default)]` | **CODE-READY** |
| 11 | **Doc-comments claiming unwired behaviour** — 4 sites fixed today (`batch_writer`, `reasoning_loop`, `witness_fabric`, `kanban::Commit`); the TD entry stays open because only the traced files were checked. | `TD-DOC-COMMENTS-CLAIM-UNWIRED-BEHAVIOUR` | grep-sweep the rest of the prose for behaviour claims vs call graphs | **CODE-READY** (grind) |

## P2 — hygiene & debt

| # | item | evidence | fix shape | owner |
|---|---|---|---|---|
| 12 | **Dead lab deps**: `bgz17` unconditional dep of planner, `bgz17`/`bgz-tensor` default-on features of core — **zero imports anywhere in the spine**. | `TD-BGZ-LAB-DEPS-DECLARED-NEVER-IMPORTED` | remove the dep lines (or wire them — decision) | **GATED** (Cargo policy) |
| 13 | **Stale "6×cosine²" doc labels** in deepnsm-v2 (`lib.rs:83`, `space.rs:158-163`) — code is SquaredL2 via contract `PairPalette`. | `TD-DEEPNSM-V2-COSINE-LABELS-STALE` | two-line doc fix | **CODE-READY** |
| 14 | **Twin-arena drift ledger never collected.** The one KNOWN drift (empty-stamp guard) is **FIXED — verified today, both arenas carry it** (`belief.rs:193` both). Residual drift unknown; the A0 census agent that was to diff them produced no output. | freshness sweep; lost A0 run | one Sonnet diff pass over the two `belief.rs` before any migration touches either | **CODE-READY** (grind) |
| 15 | **Stop hook defect** (`~/.claude/stop-hook-git-check.sh`): fires on GitHub merge commits and recommends `--reset-author`, which would force-push a rewrite of published `main`. Fix is one flag: `git log --no-merges`. | earlier session; edit blocked by auto-mode | one-flag edit outside the repo | **USER-OWNED** |
| 16 | **Stranded OGAR privacy remediation** — commits `a5c94d3`, `727f659` (113 files) unpushable (403 on every path); one private-codebook term still in public OGIT `origin/main` (`crates/ogar-emitter/src/projection_adapter.rs`). Task #7 remains in_progress. | earlier session | operator access/push; term removal on the public repo | **USER-OWNED** |
| 17 | **`euler_gamma_unfold` identity lives outside the data** — angle re-derived from a caller-held `member_index`; same defect shape as arena premises (identity not replay-stable). Lab crate; noted, not urgent. | `euler_fold.rs:257-261` | store or derive the index from addressed state — when the crate is next touched | **GATED** (lab) |
| 17b | **`NodeRowPacket::new(&rows, 0)` — promoted to the sharpest open question** (primer §16): it sits AT the Lance flush seam. If it is a re-encoding built to be reconstructed from → serialization boundary, forbidden; if it is the in-place LE view the sink drains → legal. Needs its read, not a guess. | trace A; §16 | one focused read of `canonical_node.rs:1479` region + the sink path | **CODE-READY** (read-only) |
| 18 | **Superseded probe retained** — `probe_adc_cosine_head_to_head.rs` (hand-rolled arms; fidelity numbers void) alongside its successor `probe_palette256_ndarray.rs`. Headers already carry the retraction. | this session | delete the old probe or keep as annotated history — cosmetic | **CODE-READY** |

## Resolved this session (for the record, so they aren't re-audited)
- Twin empty-stamp guard drift — **fixed** (verified both sides today).
- `AdjacencyBatch` owned re-packing — **replaced with borrowed view** (`ffca104`), 9/9 tests, clippy clean.
- 4 unwired-behaviour doc-comments — **downgraded** to DECLARED/TESTED-ONLY.
- f32 retirement **scope** — identified (`F32-RETIREMENT-SCOPE.md`); migration stays gated on P0 #3.
- The §13 CODE-PROVEN mislabelling — superseded by §15's three-column table.

## The honest shape of the audit
Nothing in P1/P2 moves the needle while P0 stands. The four P0 items are one
path: give beliefs a home (4→3), give the home a write seam (1), give the seam
a production read (2). Everything else is either housekeeping or blocked on that
path by the no-transfer ruling.
