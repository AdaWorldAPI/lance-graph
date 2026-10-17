# Thinking tables: wiring to prior plans (PRs #1447–#1596)

STATUS: WORKING-MODEL. Each row is WIRED (a plan, entry or crate carries it),
PARTIAL (parts are carried, the gap is named) or NEW (nothing prior carries
it). Every cited line was read in its source.

**Result.** The 64 tables themselves are the most thoroughly carried part:
D-HL decided, placed and measured them (budget, slot placement, identity table,
written-by word, both coverage orientations, givens layer), and D-HOLO H14c/H14d
measured the cross-table reads. What no prior work carries: FIFO order as the
stratum order (N1), a fixpoint run inside one thought table (N2, N3), naming the
written-by queries as the metacognition surface (N4), and the table's lifecycle
as one unit (N5).

## Why this document exists

The model below was stated in session on 2026-10-17, with the observation
that everything in it already exists in the work merged as PRs #1447–#1596,
except the 64 layers themselves. This document checks that claim element by
element. It does not introduce a mechanism; it points each element at the
plan, entry or crate that already carries it, and isolates what is new.

Method: Haiku substrate-mapper runs read every plan, knowledge note,
board entry and crate `lib.rs` changed by those PRs, in full, and copied the
lines that define, decide or measure something. Each wiring claim below is
then verified against the source itself, never against the map alone.

Separate input, same session: a ruff_cpp_spo harvest of PG-Ring (leapfrog
triejoin over compact ring indexes); its idea catalogue is
`.claude/research/2026-10-17-pgring-idea-catalogue.md` (#1600). It feeds D-ASM-11
(leapfrog as prior art to learn from, arm A seek over sealed lanes vs arm B
trie zipper), clean-room, and is cited here only where it bears on row 6/7.

## The model, element by element

| # | Element | Claim to verify | Candidate prior work | Status |
|---|---|---|---|---|
| 1 | 64 thoughts are 64 tables over one identity | each thought is a whole table over the shared NodeGuid rows; sparse write, merge on read | D-HL (ce64-hypothesis-layers-v1), D-HL-5 thought lanes, D-HOLO H14c | WIRED |
| 2 | Writes are isolated per thought | a thought writes only its own dirty bytes; no thought writes another | D-HL, D-WAL (cycle writer), sparse-write rulings | WIRED |
| 3 | Reads cross thoughts by presence-bit folds | `T_a[i] : T_b[i]` at the same coordinate, never a gather or foreign key | D-HOLO H14d `Thought(t)[T_a[i] : T_b[i]]`, presence planes P_l | WIRED |
| 4 | Provenance is the written-by word | per row, which thoughts wrote it, as presence bits | D-HL-3 evidential base, H14c/H14d presence planes | WIRED |
| 5 | Independence for revision is decided by source bits | per-delta source bitset, not the thought number | D-HL-3, revision layer invariants (G1–G7), isa::truth | WIRED (probe not run) |
| 6 | Set theory is the AST | ∩ AND, ∪ OR, − AND NOT, σ aperture, π table/field choice, ⋈ layout, μ fixpoint | relalg-v1 (D-RA-1..4), metta-datalog-assimilation (D-ASM), D-TELEPORT fold_ranges | PARTIAL |
| 7 | Datalog fixpoint inside a thought | semi-naive: Δ = new AND NOT old, done when popcount(Δ) = 0 | D-RA-3/4 (rules as algebra values, μ via the d′ driver), D-MTP-1 | PARTIAL |
| 8 | FIFO scheduling is stratification | a thought frees at fixpoint; FIFO order is the stratum order negation needs | D-ASM schedule, NAL schedule-depth audit, D-NALP | PARTIAL |
| 9 | Uses: time series, lenses, parallel NAL rules | one thought per year, per lens, or per NAL rule | D-WAL-6/7 temporal, nal-polyfill-v1, NAL audit inventory | PARTIAL |
| 10 | Foreign-key-shaped queries are native metacognition | who wrote it, who agrees (AND + popcount), who disagrees, as plane algebra | NAL-9 inventory, epistemic-frontier capstone (#1597), Simpson coresearch (#1596) | PARTIAL |
| 11 | The 64-layer table itself | 64 tables as the unit of thought, with its lifecycle | none expected | NEW (as a unit) |

## Wiring (each cited line read in source)

Paths are relative to the repository root. `#1597` is the epistemic-frontier
capstone, `.claude/knowledge/epistemic-frontier-capstone.md` (merged as
#1597); its lines are cited as `(#1597)`.

**Vocabulary (DECISION 2026-10-17).** The unit is a **thought**, numbered
`0..64` (thought 0 = givens, 1..63 = hypotheses; the W field). "Lane" is
reserved for SoA lanes: each thought owns up to 32 lane-tables (lane 0
identity, 1 edge, 2..31 value), so up to 2,048 lane-tables
(`.claude/plans/ce64-hypothesis-layers-v1.md:386-391`). The model's
"64 lanes = 64 tables" reads as 64 thought tables. BASIS: "lanes is already
used for SoA lanes". SCOPE: docs, code and PR text from here on.

### 1. 64 thought tables over one identity — WIRED

- `ce64-hypothesis-layers-v1.md:16-19`: "up to 64 parallel hypothesis layers
  over an immutable address space. Each layer is a sparse delta … keyed by
  `NodeGuid` / row id, merged on read."
- `:290-295` DECISION 2026-10-14: up to 64 tables; each writes only its delta;
  a thought is placed globally (a table of its own) or locally (one 16-byte slot
  of a shared table). `:301-302`: the key slot never changes; the 65th table is
  refused.
- `:470-476` identity table: 64k rows × 32 lanes, addressed by NodeGuid rank;
  `:484-488` chosen arm M: one 64-bit written-by word per row, overlays per
  (row, thought). `:576-577`: dense tiles use the Morton code as the row.
- `.claude/plans/d-holo-h14c-thought-first-v1.md:145-147`: one table per
  (thought, lane), indexed by the address, plus a presence plane per table.
- `.claude/board/entries/2026-10-11-photo-probe-64-populations.md:40-43`
  (D-PHOTO-0): with 64 resident populations over 1,048,576 addresses,
  bit-sliced planes win on every regime but one; they touch only the planes an
  expression names.

### 2. Writes isolated per table — WIRED

- `ce64-hypothesis-layers-v1.md:118-120` (D-HL-5): one writer per lane;
  `run_wave` and `run_wave_parallel` byte-identical.
- `:301-302`: a slot-placed thought may write and read only its own slot.
- `.claude/knowledge/split-tunnel-sparse-append-resident-fold.md` rule 1
  (sparse records, never a full row image) and rule 2 (one batch = one Lance
  version, one writer per mailbox; a racing writer is detected, not refused).
- Givens: only hydration writes layer 0; a conflicting hypothesis write is
  refused and recorded as a contradiction (`ce64-hypothesis-layers-v1.md:692-701`).

### 3. Reads cross tables at one coordinate, by presence — WIRED

- `d-holo-h14c-thought-first-v1.md:29-41`: `Thought(t)[Table(a)[i] : Table(b)[i]]`;
  "presence says whether a value is there, never where it is."
- `:107-122`: presence planes of 65,536 bits per (thought, lane); population,
  shared occupancy `popcount(Pa & Pb)`, Hamming `popcount(Pa ^ Pb)`, joint
  availability `A & Pa & Pb`, all as mask-risc programs.
- `:409-413` measured: 0.9–1.2 ns per 64-address word; session OR over 64
  thoughts 19–32 ns per word.
- `.claude/board/entries/2026-10-16-synergy-map-since-1579.md:282-303` (H4):
  a triangle is `R[a][b] ∧ S[b][c] ∧ T[a][c]`, each plane read at the
  coordinate itself; the work is the symbolic run intersection that chooses
  which coordinates to visit.
- `.claude/plans/d-holo-h14d-logic-mask-read-v1.md`: the four stages (logic,
  address mask R over presence planes and aperture, Lance 13 masked read,
  terminal); 34 % of 2,000 programs decided symbolically.

### 4. Provenance is the written-by word — WIRED

- `ce64-hypothesis-layers-v1.md:596-609`: R = one 64-bit written-by word per
  row, C = per-thought masks; queries Q-ID, Q-TH, Q-PAIR, Q-PERSP.
- `:643-660` measured: Q-PERSP 1.1 ns on the row words; aperture queries at most
  0.40 ns per row; both orientations kept, each query routed to its faster one.
- `d-holo-h14c-thought-first-v1.md:93-95`: "a read reports which thought
  supplied the value, so their evidence is not conflated."
- `#1597` capstone `:148`: W and the written-by word name the lane; they point
  nowhere. Its D-HL-1 note records W (CE64 53..58) as the thought layer
  (0 givens, 1..63 hypotheses).

### 5. Independence decided by source bits, not by the table — WIRED (probe not run)

- `ce64-hypothesis-layers-v1.md:73-90` (D-HL-3): each delta carries
  `(layer, seq, evidential base)`, a source bitset; the fixture separates layer
  from provenance (one observation in two layers counts once; two independents
  in one layer count twice); the layer-keyed rule is the disable. Not yet run.
- `.claude/plans/nal-polyfill-v1.md:366-375`: `Beliefs::seal_evidence` classes
  each piece (independent / duplicate / overlapping / no stamp); revision
  refuses overlap and missing provenance.
- `.claude/board/entries/2026-10-14-revision-layer-invariants.md:15-16`: G1 an
  echo earns zero weight; G2 a synthesis needs a new independent root.
- `.claude/board/entries/2026-10-12-nars-self-evidence-literature.md:15`: the
  same test from the literature side (Wang, Li & Hammer 2018).

### 6. Set theory as the AST — PARTIAL

- `.claude/plans/d-cmr-concept-chain-normalization-v1.md:80-118`: the set IR
  (`Runs`, `Res`, `∩ ∪ ∖`, complement over a universe, measures), describing
  only, never executing; `:110-113` home in the contract.
- `.claude/plans/relalg-v1.md:14-28`: D-RA-1 built `Rel` + `normalize` over one
  table; `:68-72`: no joins, no recursion.
- `.claude/board/entries/2026-10-16-synergy-map-since-1011.md:101-107` (D1):
  one set algebra in four spellings (relalg `Rel`, contract `Runs`, D-HOLO
  interval sets, quack `fold_ranges`); the hinge is D-RA-2.
- σ (aperture): `nal-polyfill-v1.md:322-328`, `SealedFacetLane::bound_aperture`.
  π (field choice): `:174-186`, the binder's `WideFieldMask`.
- ⋈: no operator. The join is the layout (H14c/H14d; the 2026-10-16 rulings
  that put VIOLATION banners on D-CMR-5 `:216`, relalg-v1 `:70` and the D-ASM
  entries).
- μ: not in relalg; see row 7.

### 7. Fixpoint inside a table — PARTIAL

- `.claude/board/entries/2026-10-03-coresearch-chained-hop.md:44-46`: the step
  `f ← Aᵀf .∗ ¬v` is Datalog semi-naive, DBSP and BFS alike; operand reuse
  carries one mask. That is the model's Δ = new AND NOT old.
- `:63`: the two-population semi-naive form (Δ and I) was SKIPPED as an extra
  undemanded population that bought nothing span-bound.
- `:51-53`: law A1 allows only demanded population-sized writes, so loop state
  needs an amendment; `:144`, `:171-175`: recursion lives in an external d′
  driver; full loop state inside mask-risc is held.
- `relalg-v1.md:63` (D-RA-4): bounded recursion as a frontier fixpoint in that
  driver. Its preimage step is gather-bannered (`:70`).
- `.claude/board/entries/2026-10-11-residue-fixed-point.md:10-19`
  (D-RESIDUE-FP-0, MEASURED): the reasoning loop runs to a fixed point; extra
  passes over the same evidence move no bit; only new evidence lowers the
  residue. That is the termination test the model's "done when popcount(Δ) = 0"
  needs, pinned for hydration.

### 8. FIFO scheduling as stratification — PARTIAL

- Stratification exists as rungs: `.claude/plans/self-reasoning-substrate-v1.md:100-109`
  (Layer 3, rungs = Tarski stratification) and `:254-266` (D-SRS-2, enforcement
  as a test).
- FIFO exists only as a scheduler baseline: `.claude/board/entries/2026-10-07-ce64-cycle-0-cross-cycle-executor.md:7`
  (a FIFO worklist that reads no entropy), `:47-48` (FIFO yields about twice
  the productive folds per second at 1M; informative work is front-loaded);
  `.claude/plans/sag-reflexive-cognition-from-1509-v1.md:76` (D-SAG-5-5
  scheduler factorial); `.claude/plans/sag-wisdom-frontier-hard-problems-v1.md:94`.
- A wave scheduler exists: `.claude/knowledge/nal-recipe-audit/schedule_depth.md:20-23`.
  A recipe is ready iff its `requires` are known; one wave runs; `known |=
  writes`; repeat until none is ready. Each wave is a stratum.
- Its limit is the model's point: `:72-80` the scheduler tracks one grounded
  bit per field, so a read after another recipe's write is invisible; `:89-98`
  a second wave needs produced objects with an identity and an input channel
  for them. A thought table's written-by bits are such an identity.
- `:133-135`: `AlphaTunnel` merges its 10 rung lanes in (rung, seq) order, and
  `:79-80` each rung lane gets its own seed copy.
- Nothing ties FIFO order to negation strata or to a table freeing at
  fixpoint. Search space for that negative: `grep -i fifo` over `.claude/plans`,
  `.claude/knowledge`, `.claude/board/entries` and `crates` (`*.md`, `*.rs`),
  19 files; every plan, knowledge and entry hit was read at its line. The 9
  crate hits (`onebrc-probe`, `holograph`, `lance-graph-ontology`,
  `lance-graph-archetype`, `lance-graph-planner` example, contract `savants.rs`,
  `deepnsm`) were not read.

### 9. Uses — PARTIAL

- One table per NAL strategy: `#1597` capstone `:221-225` ("A thought =
  (strategy, front end) compiled once into one program, running in one lane").
- Time: `d-holo-h14c-thought-first-v1.md:281-283` (`VersionClock`, D-WAL-6);
  `.claude/board/entries/2026-10-16-synergy-map-since-1011.md:79` (WAL arc; the
  version clock is not yet wired to a reader).
- Snapshot condition: `.claude/board/entries/2026-10-12-nars-self-evidence-literature.md:18`,
  a pinned Lance version k must replay identically while k+1 appends.
- Lenses: no source in the mapped range names a lens table.
- Caution from `.claude/board/entries/2026-10-17-coresearch-simpson-epistemic.md:28-33`:
  NAL revision of stratum beliefs equals the union recount, so revising across
  per-year or per-lens tables is the amalgamation Simpson's paradox warns about.

### 10. Foreign-key-shaped questions as metacognition — PARTIAL

- Who wrote it: Q-PERSP / Q-TH (row 4). Who agrees: `popcount(Pa & Pb)`
  (H14c `:112-118`). Who disagrees with the givens: the contradiction records
  (`ce64-hypothesis-layers-v1.md:696-699`).
- `d-holo-h14c-thought-first-v1.md:124-128`: presence agreement, value equality,
  semantic agreement, evidential independence and causal confidence are five
  different things. A popcount answers the first two only.
- NAL-9 steps 1–4 are covered by the saccade, hypothesis layer, givens and
  written-by words; steps 5–6 are what a NAL-9 thought must show (`#1597` `:227-233`).
- Simpson map `:72-75` (SEAM-STRATUM-PROVENANCE): stratum of origin is a fourth
  axis, separate from evidential independence.

### 11. The 64-table unit with its lifecycle — NEW as a unit

The parts exist (budget, placement, written-by word, givens, sessions,
validation ladder), but no plan treats "open a thought table, run a fixpoint in
it, commit or free it" as one lifecycle. The commit step is D-HL-8, unbuilt
(`ce64-hypothesis-layers-v1.md:36`).

**DECISION 2026-10-17 — the lifecycle is git branching.** A thought table is
both compute and storage. Thoughts are branches over main (thought 0, the
givens); at most 64 exist at once. A thought computes and stores in its own
table; when it is done it is flattened (squash-committed) into main, which
frees its slot so a new thought branch can open. BASIS: "A thought table is
compute and storage. It just needs to flatten/flush/squash the commit to the
main to open new thought branches. Same as if you use GitHub and max 64
branches." The squash is the D-HL-8 commit. Consequences for the open
points: O2 is closed (both), N2 is answered (a thought holds its own loop
state; its writes stay in its branch until the squash), and O4 narrows to
the squash itself (derived facts live in the branch; only the squash writes
main).

Points this ruling leaves to design, not yet decided:

- **The squash writes the next version, not the givens table.** Only
  hydration writes thought 0, and a given is never overwritten; the givens
  table is read-only per (classid, sealed version)
  (`ce64-hypothesis-layers-v1.md:702-725`). So the squash lands as the next
  sealed version of main, whose givens are rebuilt at seal; it must pass
  the revision gate (G1–G7), and a conflict with a given stays the existing
  contradiction record, never an overwrite.
- **Provenance through the squash.** Flattening a branch must keep each
  fact's source bitset (D-HL-3); otherwise revision in main cannot tell an
  echo from an independent confirmation (G1).
- **Branch count.** Main is thought 0 (the givens), so 63 hypothesis
  branches (1..63) can be open at once beside it (`:702-705`).
- **Simpson.** Squashing several per-stratum branches into one main is the
  amalgamation the Simpson map warns about (O5); strata need to survive the
  squash as a declared axis.

The nearest lifecycle prior is the goal state machine of
`.claude/plans/rubicon-volition-v1.md:24-50` (open, candidate, committed,
acting, evaluating, released), and the revision invariants R1–R4. It is for
goals, not thoughts, and its boundary ruling (`:106`) keeps volition out of
CE64 and mask-risc; a thought-table lifecycle can borrow its shape (explicit,
replayable transitions) but not its records.

## What is new

| # | item | evidence of absence | where it would land |
|---|---|---|---|
| N1 | FIFO order as the stratum order: a table frees at its fixpoint, and the next stratum may read it negated. The wave scheduler exists; what it lacks is produced objects with identity (schedule_depth `:89-98`) | row 8 search space | `self-reasoning-substrate-v1` D-SRS-2 or a new D-HL item; needs a test that a negated read of an unfinished table is refused |
| N2 | A thought table as the bracketed loop state the chained-hop round 2 asked for (one owner, branch lifetime, squash to main only through the ladder; answered by the row-11 DECISION) | chained-hop `:51-53` holds loop state pending an amendment; D-HL `:474` sessions are temporary | a decision on chained-hop option (b)/b-lite, recorded against D-HL-5 |
| N3 | Fixpoint (μ) inside one thought table | relalg `:68-72` (no recursion); D-RA-4 `:63` driver only | D-RA-4, once its step is rewritten without the gather-bannered preimage |
| N4 | The written-by / presence queries named as the metacognition surface | rows 4 and 10: the queries exist and are measured; no front end exposes them | NAL-9 strategy table (`#1597` §6) |
| N5 | The lifecycle as one unit | row 11 | D-HL-8 |

## Open points

| # | point | sources |
|---|---|---|
| O1 | CLOSED 2026-10-17: the word is **thought** (`0..64`); "lane" stays the SoA lane. | D-HL `:386-391`; vocabulary paragraph above |
| O2 | CLOSED 2026-10-17: both — a thought table is compute and storage, squash-committed to main like a git branch (row 11). Was: session or storage: D-HL says the 64 thoughts are session addresses, not storage; the capstone says a thought's materialization is its table's sparse storage layer. If kept, a cached population needs its own type over the same word layout, never `AlphaMask`. | D-HL `:474`; `#1597` `:148`; photo-probe `:57-63` |
| O3 | Precedence or fold: a merged read uses "higher thought index wins a shared lane", an overwrite rule; set questions use presence folds. Which reads use which is not written down. | D-HL `:412`, `:491`; H14c `:93-95` |
| O4 | NARROWED 2026-10-17: derived facts live in the thought's branch; only the squash writes main (row 11). Was: derived facts written into a table versus split-tunnel rule 7 (derived reads write nothing) and law A1 (only demanded population writes). | split-tunnel rule 7; chained-hop `:51` |
| O5 | Tables as strata: folding per-year or per-lens tables for revision reproduces Simpson's amalgamation; the stratum guard is only a probe. | Simpson map rows 1, 4 |
| O6 | The relalg absorption defect and the home of the set IR wait on D-RA-2. | synergy map D1; ledger N17 |
| O7 | The model's reading "W = the classid of the table" is not in any plan and is not recorded here as canon; `#1597` records W as the thought layer, and its classid commit was reverted. | `#1597` D-HL-1 note |
| O8 | Coverage: PRs #1447–#1496, #1504–#1591 mapped by Haiku (two items per map checked in source); #1499, #1500, #1593 and #1596 read directly; #1497, #1498, #1501–#1503, #1592, #1594 and #1595 not mapped. | maps in the session scratchpad |
