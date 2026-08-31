# The spatial-mask / R2IL hypothesis vs. `lance-graph` — full audit report

**Date:** 2026-08-24. **Method:** eleven Sonnet Explore agents (raw JSONL transcripts in
`raw-transcripts/`, ~2.9 MB) plus direct source reads by the orchestrating session, run
against the working trees of `lance-graph`, `ndarray`, `OGAR`, `ruff`, and `MedCare-rs`
(read-only; MedCare-rs is private and touched only for cross-repo triangulation already
recorded in its own `CLAUDE.md`). Every claim below is graded **ESTABLISHED** (real, cited
code/tests), **PLAUSIBLE BUT UNPROVEN** (right shape, no closed contract), or **FALSE /
OVERCLAIMED** (the code contradicts it) — the same discipline this repo's own `EPIPHANIES.md`
uses, applied from outside.

**The hypothesis under test** (operator's own framing, preserved verbatim for reference):
*"Identity remains semantic. Relations are baked into stable address projections. R2IL is
the semantic ISA boundary. The hot substrate executes those projections as masks, so
ontology, hierarchy, membership, attention, spatial adjacency, token pairing, inheritance,
permissions, and selected p-code operations can share the same ALU primitives without
becoming the same semantics."* Not "everything is a bitmask" — specifically: *"everything
that enters the hot substrate must be reducible to stable-universe + plane + address
projection + mask operation."*

**Reading this document.** §1 is the raw-transcript index. §2 is the ten-challenge scorecard
(the original ask). §3 is the running-total ESTABLISHED / PLAUSIBLE / FALSE ledger, fully
merged across every audit wave, superseding earlier partial versions given mid-session. §4
is the six-way "rung" collision catalogue, a cross-cutting finding this hypothesis walked
straight into. §5 is the self-falsification episode catalogue — evidence about how much to
trust this codebase's *other* unverified claims. §6 answers the seven deliverable asks
(invariant / falsifier / hot-path budget / etc.) in final form. §7 is a citation index.

---

## §1 — Raw transcript index (`raw-transcripts/`)

| File | Agent scope | Result |
|---|---|---|
| `01-r2il-surface-and-probes.jsonl` | R2IL type/op definitions, all 11 `probe_r2il_*`/`probe_bpe_r2il_*`/`probe_stamp_*` examples, the two `.claude/plans/r2il-*`/`probe-revision-*` docs | Completed, full report (§2 Challenge 1, §3) |
| `01b-r2il-surface-interrupted-first-attempt.jsonl` | Same brief as above, first launch | **Interrupted before completion** (superseded by 01; kept for provenance) |
| `02-hhtl-clam-chaoda-morton-address-geometry.jsonl` | HHTL real address shape, CLAM/CHAODA in both `lance-graph` and `ndarray`, Morton ordering, Levenshtein+EWA "sandwich" claim, closure-as-bitmask vs. prefix-walk | Completed, full report (§2 Challenge 6b/9, §3) |
| `03-blasgraph-causaledge-96bit-rabitq-splat.jsonl` | BLASGraph semirings + Boolean-GEMM closure, `CausalEdge64` layout, the 12-byte/96-bit `EdgeBlock` shape, RaBitQ, Gaussian-splat/surfel naming, `bgz-tensor` attention, hex/axial adjacency | Completed, full report (§2 Challenges 2/5/10/11/12/13, §3) |
| `04-soa-rungs-alphaoverlay-revision-rubicon.jsonl` | `NodeRow`/`ValueTenant` SoA layout, `AlphaOverlay` sparsity, `persona-vs-rung-ladder.md`, `temporal.rs`/`counterfactual.rs` revision model, `CollapseGate`/`MergeMode::Superposition`, Rubicon/Libet/Heckhausen search | Completed, full report (§2 Challenges 3/4/7, §3, §4) |
| `05-bpe-token-trie-INTERRUPTED-no-final-report.jsonl` | BPE/token carrier, `deepnsm-v2` trie, tesseract-rs token-lane numbers, `(tile,cell,port,token,delta)` event shape | **Interrupted mid-grind (30 tool calls, no synthesis)** — findings in this doc for Challenge 6 come from the orchestrating session's own direct reads instead (`FamilyTrie`, `fsm.rs`, `vocab.rs`), not from this transcript |
| `06-stockfish-nnue-references.jsonl` | Every "stockfish" hit across `lance-graph`: real code vs. analogy vs. planned-and-GPL-fenced | Completed, full report (§3, "real non-dependency boundary" finding) |
| `07-ndarray-3dgs-splat3d-substrate.jsonl` | `ndarray/src/hpc/splat3d/` — `Gaussian3D`/`GaussianBatch`, ten `3DGS-*.md` plan docs, `benches/RESULTS.md` real numbers, `tile.rs` screen-space binning, `depth_cascade.rs` HHTL cascade, cross-repo wiring to `lance-graph`'s `jc::ewa_sandwich_3d` | Completed, full report (§2 Challenge 5 correction, §6 hot-path budget) |
| `08-p64-bridge-cognitive-shader-driver.jsonl` | `p64-bridge`'s `[[u64;64];8]` predicate-plane topology, `cognitive-shader-driver`'s `update_planes`/`cascade` compute cycle, timing-benchmark search including the "125 ns / 233 ns" figures | Completed, full report (§2 Challenge 5/8 correction, §6) |
| `09-pr-1000-1018-belief-abi-arc.jsonl` | Every board mention of PRs #1000–#1018 across `EPIPHANIES.md`/`LATEST_STATE.md`/`PR_ARC_INVENTORY.md` | Completed, full report (§3, §5) |
| `10-pr-1019-1040-frontier-check.jsonl` | Same board sweep for #1019–#1040 | Completed — confirms the documented frontier is #1019, everything above is undocumented |

Plus direct (non-agent) source reads by the orchestrating session, cited inline throughout:
`ruff_r2il::facet::VarnodeFacet` + its `r2il`/`r2sleigh` path-dep chain; `OGAR/crates/ogar-loco`
(full `lib.rs` header + `interpret_probe.rs` + `recipe_dispatch_bridge_probe.rs`);
`lance-graph-contract::recipe_dispatch` (full file, 361 lines); `lance-graph-supervisor::cycle_driver`
(PR #879, header + honesty ledger + tests); the `#911→Phase-A` persistence supersession in
`EPIPHANIES.md`/`LATEST_STATE.md`; `lance-graph-planner::elevation::cycle`/`kanban_actor.rs`
(the real Rubicon/Libet consumers, correcting an earlier under-scoped finding).

---

## §2 — The ten challenges, scored

### Challenge 1 — relation vs. compiled projection; is R2IL a compiler boundary?

**Verdict: doctrine yes, measured code no — and the two claims are separable.**

The design principle is real (`ARC-B-OWNERSHIP-AND-ADDRESSING-REASSESSMENT.md` §7: *"the
domain oracle keeping semantic authority and the ABI exposing address + mask + view"*), and
independently, elsewhere in the codebase, the *general* claim ("a source relation dies from
the hot path while remaining valid provenance") is genuinely established for `is_a`
specifically — see Challenge 9. But **R2IL's own instruction vocabulary is not a bitmask
ALU.** The one probe that actually executes R2IL-shaped ops (`probe_r2il_frontier_phase2.rs`)
frames it as typed register arithmetic (`Copy/IntAdd/IntMult/IntLeft` over `Vn{space,offset,
size}`), and the real corpus opcode census (`probe_r2il_real_episodes.rs`, 143 real x86-64
functions, 17,557 rows) is nine RISC-like opcodes: `branch, call, call_ind, cbranch, copy,
int_add, load, return, store`. Nothing resembling AND/OR/XOR/NOT/SHIFT/POPCNT/gather-scatter/
Boolean-GEMM appears in R2IL's own vocabulary anywhere it was actually run.

**R2IL's real address identity, closer to a `MaskRef` than anything native to `lance-graph`:**
`ruff_r2il::facet::VarnodeFacet` — 16 bytes, `classid(space-class) | offset_lo | offset_hi |
size`, prefix-routable, wrapping `r2il::Varnode` from a **path dependency on
`r2sleigh/crates/r2il`** (confirmed directly: `ruff/crates/ruff_r2il/Cargo.toml:19`). This
is the real Ghidra→SLEIGH→r2sleigh→R2IL lineage, exactly as the operator corrected
mid-session — `r2sleigh` is radare2's own SLEIGH-based lifter, and `r2il` is its
Rust-side Varnode representation, not Ghidra's raw p-code.

**Full R2IL execution status (from the belief-ABI arc, §3):** PR #1013 caught a real
register-clobber bug a happy-path RL learner would have kept (trust score e=0.812, above
the 0.75 admission bar, refused only by the falsification check verifying the actual
contract). PR #1014 reproduces on the real corpus: of 380 occurrences of the top opcode
trigram `(int_add,copy,store)`, exactly 1 is dataflow-chained — **99.7% over-admission** by
naive opcode-window matching, confirming def-use chains (not linear windows) are the real
macro carrier.

### Challenge 2 — the 6×2×8-bit bonding ABI: same physical shape, different semantic plane

**Verdict: ESTABLISHED, and cross-repo, deliberately mirrored — the strongest single
confirmed result in this whole audit.**

`lance-graph-contract/src/facet.rs:359-431` — `CascadeShape::{G6D2, G4D3, G3D4}`, all
`G·D=12` (96-bit), 8-bit-addressed: Rails `6×(u8:u8)`, other frameworks `4×(u8:u8:u8)`,
canonical GUID `3×(u8:u8:u8:u8)`. `OGAR/crates/ogar-loco/src/lib.rs:1-100` —
`LaneShape::{Pairs,Triples,Quads}`, the **identical** `6×2/4×3/3×4` carving, doc-stated as
*"mirroring the LE contract's CascadeShape."* Same 12-byte register, genuinely different
planes: one is a spatial/rail address geometry (`lance-graph-contract`), the other is a
program `(function:value)` call ABI (`ogar-loco`).

What's NOT established: the specific **hexagonal/6-directional-facet reading** the
hypothesis proposed. Every sanctioned reading of the 12-byte register (`CoarseOnly`/
`CoarseResidue`/`Pq32x4` for `EdgeBlock`; the G1/G2/G3 legacy carvings; the `6×(8:8)`/
`4×(8:8:8)`/`3×(8:8:8:8)` rail geometry) is content-blind and class-dispatched — none reads
it as 6 facets of 16 bits with spatial/coordinate meaning. That reading would be net-new.

### Challenge 3 — ten alpha-stacked rungs as overlays

**Verdict: PLAUSIBLE, right shape, wrong mechanism — see §4 for the full six-way collision.**

`AlphaOverlay` (`MedCare-rs/crates/medcare-nodesoa/src/alpha.rs`) is genuinely sparse
(thin-provisioned across an address space, proven by tests: `claiming_materialises_only_the_
visited_rows`, `a_revisit_keeps_the_first_stamp_and_does_not_grow_the_table`) and rung is a
real metadata field (`AlphaStamp{cycle,seq,rung,visits}`), durable via Lance. But the
overlay semantics are **first-write-wins, not additive** — a revisit at a *different* rung
does not create a second reading; the first rung is frozen. `rung[n] = rung[n-1] +
semantic_delta[n]` (additive stacking) is not what's implemented; single-value overwrite is.

`persona-vs-rung-ladder.md`'s content ladder (0-1 observation / 2 = 144 verb atoms / 3 = 34
NARS tactics / 4 = StyleFamily macros) matches the hypothesis's 10-rung *count* but assigns
content to only rungs 0-4, and its own open item O1 states the shipped `RungLevel` type
*"does NOT implement"* this content ladder — it's a ruling, not wired code.

### Challenge 4 — Rubicon + Libet as commitment geometry

**Verdict: ESTABLISHED as real, mechanical, load-bearing production architecture — the
second-strongest confirmed result in this audit, after Challenge 2.**

The canonical governing storage rule, stated twice verbatim in the board (`LATEST_STATE.md`,
Phase-A entry): *"Thinking is cheaper than persisting its intermediate control flow: a
thought runs its whole Rubicon ladder transiently and only an artifact-backed delta becomes
durable."* Mechanically real: cast payload non-empty = artifact = persisted; empty =
intent-only = ephemeral, discarded on regeneration, never reaching the writer
(`restage_held`'s empty payload IS the ephemerality mechanism).

**The real Rubicon consumers, corrected mid-session per the operator's own pointer:**
`lance-graph-supervisor/src/kanban_actor.rs:45,54` — a live (non-deprecated) *"message-free,
read-only census of Rubicon phases,"* with the phase DAG **never hardcoded**, derived from
`KanbanColumn::next_phases()`. `lance-graph-planner/src/elevation/cycle.rs` — a real M12
milestone unifying the per-cycle **−550,000 µs Libet anchor** (stamped at the exact
`Planning → CognitiveWork` crossing, `KanbanMove::libet_offset_us`) with per-strategy
patience budgets; explicitly **advisory, never gating** (*"an exhausted budget must not
deadlock a cycle"*).

**PR #879 (`cycle_driver.rs`, 2,618 lines, ratified as the complete production
phase-progression path)** gives exactly the two-phase geometry the challenge asked for:
pre-commit failure (before `Vn+1` exists) = deterministic regeneration, provably discardable;
post-commit interruption (`Vn+1` exists) = committed-history recovery only, *"the two
mechanisms share no state."* PR #911's attempt to make an optimistic fence "effective after
the fact" via a compensating delete was found categorically wrong and **replaced, not
patched** (`Dataset::delete` creates another version — *"there is no undo in an MVCC
manifest chain"*) — a real self-falsification, not a clean success story.

**Not found, checked directly:** a literal file named `revision.rs` (does not exist
anywhere in the repo); a −220ms intermediate Libet sub-band (only −550ms and the 0-crossing
appear anywhere).

### Challenge 5 — spatial BLASGraph + sparse active frontier

> **⊘ ADDENDUM 2026-08-31 — this structural verdict was later confirmed
> FUNCTIONALLY, by independent means.** This audit found no hex adjacency by
> grep (structure). Two 2026-08-26 experiments then measured what a hex
> topology would BUY, and it bought nothing:
> `E-Q6-HEX-FAILS-CONTENT-ADDRESSING-IS-CAPACITY-DESTROYING-UNDER-A-SKEWED-DISTRIBUTION-1`
> and `E-Q7-FREQUENCY-SIZING-RESCUES-THE-LEARNING-GATE-BUT-NOT-THE-INTERFERENCE-CLAIM-AND-THE-2-BYTE-RAILS-ARE-COMPLEMENTARY-NOT-COMPETING-1`
> (the second removes the first's capacity confound, strengthening it). Two
> independent lines — absent in the code, and useless when built — that had
> not been connected. Neither touches the `6×(u8:u8)` RAIL reading, which is
> canon and a different object; see the §7.2 addendum in
> `.claude/plans/r2il-machine-semantic-contract-v1.md`.

**Verdict: split finding — hexagonal/axial adjacency is FALSE/not-found everywhere audited;
a real, working rectangular active-frontier structure exists, but in `ndarray`, unconnected
to `lance-graph`'s cognitive substrate.**

No hexagonal/axial 6-neighbor adjacency scheme exists anywhere in `lance-graph`, `ndarray`,
or `OGAR` (exhaustive grep, zero hits for `axial`/`hex.*neighbor`/`hexagon` across all three).

But `ndarray/src/hpc/splat3d/tile.rs` (605 lines) implements real 16×16 screen-space tile
binning: `TileInstance` (16-byte, `#[repr(C,align(16))]`), sorted `(tile_id<<32|depth_bits)`
for O(1) depth-ordered per-tile access — a genuine fixed-neighbor spatial-adjacency structure
with sorted active-element lists, just rectangular/per-frame, not hexagonal/persistent. And
`ndarray/src/hpc/splat3d/depth_cascade.rs` (464 lines) implements exactly the "only active
cells pay compute" principle: `HhtlAction{Reject, KeepCoarse, Refine, ProjectExact,
RenderExact}` with real early-reject cascading, matching the corresponding
`3DGS-HHTL-CPU-cascade-plan.md`'s own sketch.

**The bridge that would connect this to `lance-graph`'s cognitive substrate does not exist:**
`3DGS-4x4-cognitive-shader-SoA-plan.md` is 100% proposal — `grep` for `Mat4x4`/`Sym4`/
`Block4`/`Splat4Carrier` returns zero hits anywhere in `ndarray/src`.

A second, structurally different real 4096-bit mask structure exists in `lance-graph` itself
— see Challenge 8.

### Challenge 6 — BPE becomes structural learning (EXPLORE→LEARNED→FROZEN)

**Verdict: PLAUSIBLE, upgraded mid-session from an earlier NOT-FOUND — real measured code
exists, but scoped CAN-FIT-NOT-YET-BUY, and outside `lance-graph` for its most mature form.**

Real crystallization exists: PR #1012 packs real BPE merges into **28 resident `[u8;12]`
`Copy` particles** under the fixed `6×(8:8)` geometry from Challenge 2 — actual code, not a
fixture. A real gate result names the triangle: *"frozen learned explore superposition of
what is more efficient and reusing that"* (S6, #1012). But the explicit grade is
`E-TOKEN-BPE-CAN-FIT-NOT-YET-BUY-1` — the geometry accepts the encoding; nothing yet depends
on it being there.

`(tile, cell, port, token, delta)` as a named event shape: **NOT FOUND** anywhere.

The BPE-merge-tree-as-HHTL-ancestry hypothesis was independently tried and refuted TWICE, in
different sessions, with the same finding: *"Encodability ≠ hierarchy"* — a binary merge DAG
is not lawful radix-prefix containment (three same-depth token pairs are prefixes of each
other in the DAG but not in the address geometry).

The most mature real token-seam work (`PROBE-TOKEN-SEAM-1`, 37 gates, 13 disable-runs, a
genuine post-hoc vacuity audit that found 5 real holes in its own falsifiers) lives in a
**different repo** (`AdaWorldAPI/paperless-rs`, because it needs a Tantivy dependency this
workspace doesn't carry) and is explicitly still **OPEN**, not merged.

Real ancestry crystallization for a *different* domain: `deepnsm-v2/src/ancestry.rs`
(`FamilyTrie`) independently re-derives the same "trie is crystallized memory, closure is
prefix-containment" law found for HHTL in Challenge 9 — first-parent-wins, cycle-residue
excluded, never silently dropped.

### Challenge 7 — superposition without quantum cosplay

**Verdict: ESTABLISHED, in the sharpest and most precise form found anywhere in this
audit — but for the NARS recipe-dispatch layer specifically, not the address/mask layer.**

`lance-graph-contract/src/recipe_dispatch.rs:34-46` (361 lines, 5 real tests) states, as
pre-existing production doctrine, almost exactly the challenge's own framing: *"Before
dispatch the awareness is in superposition: the witness register holds many bundled loci;
many recipes could fire. Dispatching a recipe... is a measurement that collapses it — but
deterministically: identity is recoverable by key, no measurement randomness (Schrödinger's
cat in a glass box)."* The gating mechanism (`nan_disqualifier`) is framed precisely as "a
conjugate variable not measurable in this basis" — an input the current tenants cannot
ground, so the recipe is skipped rather than read off noise. And every dispatch records its
own *triggering cause* (`RecipeStep::trigger`) so the orchestrator's own causal influence on
what gets found is logged, not hidden (the **Versuchsleitereffekt**, explicitly named).

`CollapseGate`'s `MergeMode::Superposition` variant is real and named (*"keep ALL deltas
without resolution"*) but **not demonstrated as an implementation** — every occurrence found
is either the definition or a test asserting what it does *not* trigger. The actual
`CollapseGate` enum (`Flow/Block/Hold`) lives in a different repo (`ndarray`), out of scope
for this audit pass. The one traced write path (`AlphaOverlay::claim`) is single-value
overwrite, not multi-reading accumulation.

### Challenge 8 — universes must remain distinct (MaskRef/MaskOp)

**Verdict: the LAW is ESTABLISHED and independently ratified; the STRUCT is not found
anywhere, by design, distributed instead.**

PR #1012's ratified law: *"CONTENT NEVER TRAVELS IN CLASSID. CLASSID SELECTS THE READING.
HHTL = WHERE. mask = WHAT. edges = HOW."* This is precisely the universe-separation the
hypothesis's `MaskRef{universe,plane,address_range}` proposes — just distributed across
classid/HHTL/mask/edges rather than named as one type. The nearest real (heterogeneous, by
design) analogs: `FieldMask(u64)`, `WideFieldMask`, `RowFocusMask{entries:Vec<
AttentionFocusFacet>}`, `NamedView{class,mask,template}` — a probe's own "grounded starting
facts" states these are *"heterogeneous semantic selector families; no universal CommonMask
contract has been established."*

A real, load-bearing 4096-bit predicate-plane structure exists — `cognitive-shader-driver::
driver.rs:75-79`, `RwLock<Box<[[u64;64];8]>>`, 8 named predicates (`CAUSES, ENABLES,
SUPPORTS, CONTRADICTS, REFINES, ABSTRACTS, GROUNDS, BECOMES`), row/col addressing a
256-entry SPO palette (`s_idx()/4`, `o_idx()/4`) — a real `{universe=predicate,
plane=block-row/col, address_range=4096-bit}` identity in spirit, though no such struct is
named. Its only mutator (`update_planes`) is a full external batch-replace, untested; the
runtime `cascade()` path is read-only, never writes back. This is NOT the write-back
masked-dynamics cycle the hypothesis proposes — it's a static/externally-swapped SPO
adjacency topology plus a stateless per-dispatch query.

### Challenge 9 — address geometry (stable numbering ≠ DFS contiguity ≠ Morton locality ≠ semantic locality ≠ closure compression)

**Verdict: the codebase independently ran and ratified essentially this exact caution,
under its own numbering, before this audit ever asked the question.**

The operator-ratified law from the belief-ABI arc (PR #1009, reached by withdrawing an
earlier overclaim): *"Hierarchy is the address space, not the ontology... HHTL need not
claim that reality is a tree."* PR #1006 found the overclaim it corrects — a bare
arena-position integer (`arena[37]`) had been treated as carrying semantic hierarchy:
*"an implementation accident wearing an HHTL costume."*

The real closure mechanism, ESTABLISHED and independently re-derived in three places:
`NiblePath::is_ancestor_of` (`hhtl.rs:172-183`) is `O(1)` prefix-containment over a
16-nibble/64-bit single-inheritance path (explicitly NOT a general DAG closure — multi-parent
routes to an orthogonal `FieldMask` bit instead, `hhtl.rs:28-32`); `FamilyTrie`
(`deepnsm-v2/ancestry.rs`) independently re-derives the identical law for a different domain;
`MedCare-rs`'s `obo_store.rs` independently converges on the same two-stage
prefix-exclude-then-bounded-walk pattern in a third, unrelated private repo.

**A real measured limit on "geometry = semantic locality," found and stated honestly rather
than assumed:** PR #1009's own falsifier G3 — *"one source observed through three sibling
basins pools naively to c=0.9444 — bit-identical to three genuinely independent sources...
`globality = geometry` is TRUE only with provenance."*

**No `DOWN[x]` materialized ancestor-closure bitmask (u64 or wider) exists anywhere** — the
real strategy is always prefix-arithmetic pruning falling back to a capped graph walk
(`class_view.rs`'s `MAX_HOPS=16`), never a precomputed OR-of-parents mask.

**No measured claim exists anywhere** that "semantically related closures have measurably
lower fragmentation under Morton ordering" — real Morton ordering (`onebrc-probe`, ~10%
addressing tax) is measured, but for hash-table slot routing, an unrelated use.

### Challenge 10 — up/down transpose cost

**Verdict: PARTIALLY ESTABLISHED, from an unexpected angle.**

No dedicated benchmark comparing materialized-transpose vs. bitsliced vs. block-transpose
vs. gather vs. SIMD-scan vs. dual-projection was found for the HHTL ancestry structures
specifically. But `cognitive-shader-driver::driver.rs` snapshots its `[[u64;64];8]` planes
under a read lock explicitly to guarantee a *consistent* view across one query cycle — i.e.
the codebase already treats "read the same state throughout one cascade" as a real cost
worth paying for, though this is about consistency, not about up/down transpose cost
specifically. This challenge is the least-answered of the ten; a real benchmark here would
be genuinely new information, not a rediscovery.

---

## §3 — Running-total ledger (fully merged, supersedes any earlier partial version stated mid-session)

### ESTABLISHED

1. The 12-byte `6×2/4×3/3×4` register carving, deliberately mirrored across `lance-graph-contract` and `OGAR`'s `ogar-loco` (Challenge 2).
2. `is_a`/HHTL closure = prefix-containment, independently re-derived in three repos, with an operator-ratified law explicitly rejecting the ontology-is-a-tree overclaim (Challenge 9).
3. The Rubicon/Libet commitment-phase model, mechanically real across `cycle_driver.rs` (#879), the Phase-A persistence rule, `elevation/cycle.rs`'s M12 budget unification, and `kanban_actor.rs`'s live phase census (Challenge 4).
4. `recipe_dispatch.rs`'s classical-superposition/deterministic-collapse doctrine, independently matching the hypothesis's own "not quantum cosplay" framing almost verbatim (Challenge 7).
5. The universe-separation law (`CONTENT NEVER TRAVELS IN CLASSID... HHTL=WHERE, mask=WHAT, edges=HOW`), operator-ratified in PR #1012 (Challenge 8).
6. RaBitQ (`bgz17::rabitq_compat.rs`) — real binary quantization, genuine XOR+popcount Hamming distance, Walsh-Hadamard rotation, 6 tests.
7. `bgz-tensor::AttentionSemiring` — exact match for "distance table (u16) + compose table (u8)," real O(1) multi-hop `compose_chain`.
8. `BLASGraph::HdrSemiring` — 7 real semiring variants including `Boolean`, real `GrBMatrix::mxm`.
9. `CausalEdge64`'s bit layout, const-asserted, with a real reserved 59-63 region (`TRUTH_SHIFT`/`SPARE_SHIFT`).
10. `NodeRow = 16|16|480`, const-asserted at compile time — the one truly stable substrate everything else sits on.
11. `ogar-loco` genuinely executes real programs, end-to-end proven via `PROBE-LOCO-INTERPRETER-1` → `PROBE-RECIPE-EXECUTION-1` → `PROBE-RECIPE-DISPATCH-BRIDGE-1`, the last closing with a byte-for-byte equality falsifier across all 34 recipe ids routed through the real call ABI vs. called directly.
12. Real, deliberately enforced non-dependency boundary: `stockfish-rs` is GPL-3.0-fenced (*"NEVER becomes a dependency"*), all exchange file-based, and the codebase's own `SYNERGY-MAP-S00-S07.md` self-audits its own claims about it as unverifiable.
13. Real screen-space active-frontier compute (`ndarray/tile.rs` + `depth_cascade.rs`), though unconnected to `lance-graph`.

### PLAUSIBLE BUT UNPROVEN

1. Rung-stacking as additive overlay (Challenge 3) — the real mechanism (`AlphaStamp`) is sparse and durable but first-write-wins, not additive.
2. `CollapseGate::Superposition` as a genuine multi-reading store (Challenge 7) — named, untested, and the real enum lives in a different, unaudited repo.
3. `MaskRef`/`MaskOp` as named types (Challenge 8) — the law is ratified, no struct embodies it.
4. BPE crystallization into the fixed register (Challenge 6) — real code exists (28 resident particles), explicitly graded CAN-FIT-NOT-YET-BUY.
5. Non-destructive revision as one three-phase construct (Challenge 4) — assembled from two real but separate subsystems (Lance time-travel reads; a partially-stubbed counterfactual mailbox), not one type.

### FALSE / OVERCLAIMED

1. Hexagonal/axial 6-neighbor spatial adjacency — zero hits anywhere audited (Challenge 5).
2. "6 directional 16-bit facets" reading of the 96-bit register — contradicted by every sanctioned reading found (Challenge 2).
3. Boolean-semiring GEMM realizing transitive closure via repeated squaring — the primitives exist, the doubling loop does not (Challenge 1-adjacent).
4. "One XOR + popcount gives structural + epistemic distance" for `CausalEdge64` — doc-comment aspiration only; the real distance function uses precomputed per-plane matrices.
5. "Width + height + CLAM + CHAODA + Levenshtein + EWA sandwich" as one wired HHTL pipeline — every ingredient real, combination explicitly tried and backed away from in the one place it was attempted (Challenge 9-adjacent).
6. Literal "surfel" address — zero hits anywhere; "splat" exists but is metaphorical CAM-plane compositing, not a spatial/geometric address.
7. The "125 ns"/"233 ns" hot-path figures — confirmed absent, by exhaustive string search, from `lance-graph`, the sibling `ndarray::p64` crate, and `ndarray`'s splat3d bench suite. Not merely unfound — actively searched for and not present.

---

## §4 — The "rung" collision: six distinct meanings found, three confused, three deliberate

This hypothesis's Challenge 3 walked directly into a real, pre-existing terminology collision
in this codebase, worth cataloguing in full since it recurs across nearly every thread:

1. **Shipped `RungLevel`** — Pearl causal-depth reading (0-2 observe / 3-5 intervene / 6-9 counterfactual) + homeostatic elevation policy.
2. **`persona-vs-rung-ladder.md`'s content ladder** — 0-1 observation / 2 = 144 verb atoms / 3 = 34 NARS tactics / 4 = StyleFamily macros. Explicitly stated NOT implemented by (1) — open item O1.
3. **`learning::cognitive_frameworks::Rung`** — Noise=0..Transcendent=9, interior ordinals stated to diverge from (1), tracked as unresolved debt `TD-THIRD-RUNG-LADDER-LEARNING`.
4. **`AlphaStamp.rung`** (`MedCare-rs::alpha.rs`) — per-address metadata, first-write-wins, durable via Lance.
5. **`recipe_dispatch::rung(id)`** — per-recipe derived value (`Tier` base + `RecipeInference::rung_delta`), 1..9, a pure function with no per-instance state.
6. **`recipe_loci::loci_rung`** — organ-depth, 0..6, from a sibling module in the same PR cluster (#780/#784).

Meanings 1-3 are the confused triad — genuinely unresolved, flagged as debt inside the
codebase itself. Meanings 5 and 6 are the opposite case, and instructive for how to read
rung-multiplicity generally: the codebase ran a real statistical battery cross-checking them
(Pearson +0.337, Spearman +0.343, ICC≈+0.34, **Cronbach's α = 0.504**, order-disorder 34.4%)
and concluded, in its own words, *"neither tautology (α≳0.8) nor orthogonal (α≲0.2) →
DISTINCT FACETS... measure each, never assume symmetry."* Lesson for this hypothesis:
multiple rung readings over one substrate are not automatically a defect — they're a defect
only when nobody has checked whether they're actually saying different things.

---

## §5 — Self-falsification episode catalogue

Cited throughout §2-3, collected here because their density is itself evidence about how
much to trust this codebase's *other*, unverified claims elsewhere:

- **PR #911 → Phase-A**: an optimistic-fence compensating-delete design found categorically
  wrong (*"Dataset::delete creates another version. It is not rollback."*) and replaced, not
  patched.
- **`E-A-A-PERMANENT-FAULT-REPORTED-AS-RETRYABLE-IS-AN-INFINITE-LOOP-1`**: a malformed
  511-byte payload misclassified as retryable, which under #879's own "safe to regenerate"
  contract would have looped forever; fixed by adding a permanent-by-construction error
  variant.
- **`E-A-DOC-COMMENT-IS-NOT-AN-ENFORCEMENT-1`** / **`E-A-RECONCILED-HEAD-IS-NOT-A-
  PUBLICATION-1`**: two paired findings that a prose warning is evidence the next layer will
  ignore it — the fix in both cases was making the distinction unrepresentable in a type, not
  documenting it harder.
- **PR #1013**: a register-clobber macro reaches trust score e=0.812 (above the 0.75 bar)
  under a deliberately sloppy happy-path oracle; refused only by the falsification check.
  *"Happy-path RL would have learned the clobber."*
- **PR #1014**: pre-registration "real top-1 trigram > shuffled" explicitly FAILED (380 vs
  387) — the refutation is recorded as a finding, not hidden.
- **PR #1012**: retracts its own headline recommendation (`Copula → relation concept →
  classid reference`) in the same PR, after discovering the answer was already shipped
  elsewhere — *"the intermediate hypothesis was therefore a proposal for shipped code —
  precisely the rediscovery tax `CLAUDE.md` names."*
- **PR #1017 (`PROBE-TOKEN-SEAM-1`)**: an independent post-hoc vacuity audit of its own
  *already-passing* 37-gate probe found five real holes (a byte-count check that would have
  missed a CRLF bug collapsing 300 spans into 1; a tautological threshold; two shape-only
  checks; one unexercised path) — and separately found two of its own disable-runs had been
  silently wrong (bad binary path, six consecutive false "no failure" reports).
- **PR #1006 → #1009**: an arena-position integer treated as carrying HHTL semantics
  (*"an implementation accident wearing an HHTL costume"*), caught and generalized into the
  ratified hierarchy-is-address-space-not-ontology law.
- **`interpret_probe.rs`**: reports its own kill condition (do the 34 recipes execute) as
  explicitly NOT TESTED rather than silently declaring victory on the parts it could test.

---

## §6 — The seven deliverable asks, final answers

**1. ESTABLISHED / 2. PLAUSIBLE / 3. FALSE** — see §3 (merged, current).

**4. THE ONE INVARIANT.** Unchanged from the first pass, and nothing in the extended audit
weakens it: *every substrate primitive that survives to the hot path is a fixed-width
register whose byte-groups are re-carved (not re-typed) per reading, keyed by an upstream
classid decision made once.* `NodeRow`'s 512-byte stride; the 12-byte register's three
carvings, mirrored deliberately across two repos; `NiblePath`'s 64 bits read two ways
(prefix-containment for closure, literal address for routing); `CausalEdge64`'s
const-asserted layout read through multiple named lenses without re-encoding. This is
narrower than the full hypothesis and survives everything found against it.

**5. THE ONE FALSIFIER.** Unchanged in design, now better-motivated: build a real `DOWN[x]`
bitmask over a synthetic 64-node DAG with one deliberate multi-parent exception, compare
against `NiblePath::is_ancestor_of` on every single-inheritance node (must agree exactly),
confirm the multi-parent case routes to `FieldMask` rather than corrupting either
representation, time at N=64 and N=4096 against the Morton-cascade model's own zero-drop
prediction. `depth_cascade.rs`'s real `HhtlAction` reject-early cascade is now a working
precedent for exactly this shape of cheap-reject-before-expensive-compute structure.

**6. HOT-PATH BUDGET.** The "125 ns / 233 ns" figures are now conclusively dead — actively
searched for and confirmed absent across `lance-graph`, `ndarray::p64`, and `ndarray`'s
splat3d bench suite. Real numbers to use instead, `ndarray/benches/RESULTS.md`, measured on
Sapphire Rapids AVX-512: `Spd3` sandwich covariance push-forward 209.96ns (scalar) → 90.41ns
(AVX-512, 16-wide batch), a 1.83× speedup against a ≥10× target, honestly reported as missed
(*"AoS↔SoA transpose overhead... dominates"*). `Spd3::eig` 125.66-130.82ns. No
rasterization-pass, fps, or full-pipeline number exists anywhere audited — every downstream
stage is explicitly flagged as not-yet-benched in the same document. If a hot-path number is
needed for the mask/register primitives specifically (not 3DGS), none exists yet; state that
plainly rather than reusing a number that cannot be traced.

**7. NO ARCHITECTURE FICTION.** Confirmed absent by exhaustive search, listed once, flatly:
`MaskRef`/`MaskOp` as named types; hexagonal/axial 6-neighbor tile adjacency anywhere;
`(tile,cell,port,token,delta)` as a named event shape; the EXPLORE/LEARNED/FROZEN triangle as
one named construct (though its two ends are separately real); `DOWN[x]` as an actual
materialized bitmask; rung-stacking as additive overlay; `CollapseGate` holding literal
multiple pending readings per address; a literal `revision.rs` file; a −220ms Libet sub-band;
the bridge from `ndarray`'s real 3DGS active-frontier machinery into `lance-graph`'s
cognitive substrate (`3DGS-4x4-cognitive-shader-SoA-plan.md` is 100% proposal, zero code).

---

## §7 — Citation index (file:line, by challenge)

- **Challenge 1 (R2IL):** `ruff/crates/ruff_r2il/Cargo.toml:19`; `ruff_r2il/src/facet.rs:1-60`;
  `lance-graph-planner/examples/probe_r2il_frontier_phase2.rs`; `probe_r2il_real_episodes.rs:253-267`;
  `lance-graph-contract/src/ogar_codebook.rs:113-117,149`.
- **Challenge 2 (12-byte register):** `lance-graph-contract/src/facet.rs:359-431`;
  `OGAR/crates/ogar-loco/src/lib.rs:1-100`; `lance-graph-contract/src/canonical_node.rs:645-704`.
- **Challenge 3 (rungs):** `MedCare-rs/crates/medcare-nodesoa/src/alpha.rs:106-297`;
  `.claude/v3/knowledge/persona-vs-rung-ladder.md:24-76`.
- **Challenge 4 (Rubicon/Libet):** `lance-graph-supervisor/src/cycle_driver.rs:1-160`;
  `lance-graph-supervisor/src/kanban_actor.rs:45,54`; `lance-graph-planner/src/elevation/cycle.rs:1-40`;
  `lance-graph-contract/src/kanban.rs` (`LIBET_COMMIT_WINDOW_US`); `.claude/board/EPIPHANIES.md`
  (Phase-A supersession entry, `E-A-PUBLISHED-MANIFEST-IS-HISTORY-RECONCILIATION-NOT-ROLLBACK-1`).
- **Challenge 5 (spatial):** `ndarray/src/hpc/splat3d/tile.rs`; `ndarray/src/hpc/splat3d/depth_cascade.rs`;
  `ndarray/.claude/plans/3DGS-4x4-cognitive-shader-SoA-plan.md`; `ndarray/.claude/plans/
  3DGS-HHTL-CPU-cascade-plan.md`.
- **Challenge 6 (BPE):** `deepnsm-v2/src/ancestry.rs` (`FamilyTrie`); `.claude/board/EPIPHANIES.md`
  #1012 entries (`E-TOKEN-BPE-CAN-FIT-NOT-YET-BUY-1`); `AdaWorldAPI/paperless-rs` (`PROBE-TOKEN-SEAM-1`,
  external repo, referenced only).
- **Challenge 7 (superposition):** `lance-graph-contract/src/recipe_dispatch.rs:1-361` (full file);
  `lance-graph-contract/src/collapse_gate.rs:18-64`.
- **Challenge 8 (universes):** `.claude/board/EPIPHANIES.md` #1012 law entry; `cognitive-shader-driver/
  src/driver.rs:75-190,266-267`; `p64-bridge/src/lib.rs:121-128,383-433`.
- **Challenge 9 (address geometry):** `lance-graph-contract/src/hhtl.rs:28-32,172-183`;
  `.claude/board/EPIPHANIES.md` PR #1006/#1009 entries; `MedCare-rs/crates/medcare-cohorts/src/
  obo_store.rs:1113-1198`.
- **Ogar-loco execution proof:** `OGAR/crates/ogar-loco/examples/interpret_probe.rs:1-60,780-798`;
  `lance-graph/crates/lance-graph-ogar/examples/recipe_dispatch_bridge_probe.rs:1-60`.
- **PR arc:** `.claude/board/EPIPHANIES.md` and `.claude/board/PR_ARC_INVENTORY.md`, PRs #879,
  #911, #912, #1000-#1014, #1017, #1019 (all others in #908-1060 checked and confirmed absent).
