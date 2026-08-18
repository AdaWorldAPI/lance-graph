# LOTUS-FRONTIER-AUDIT — Phase 0 archaeology (research arc, 2026-08-18)

> Deliverable 1 of the LOTUS SEAL / FRACTAL COMMIT FRONTIER research charter
> (operator, 2026-08-18). Companion: `F-ORD-REAL-FALSIFIER.md` (deliverable 2,
> same directory). This document changes NO code; the paired PR lands only the
> pre-registered falsifier tests it describes.
>
> **Grading legend (charter):**
> - **VERIFIED** — read at file:line in this working tree this session.
> - **INFERENCE** — mechanism argued from VERIFIED parts; not yet measured.
> - **HYPOTHESIS** — the research claim itself; needs a falsifier to run.
> - **BLOCKER** — cannot be verified in this session; names what unblocks it.

## §0 The question under investigation

The charter investigates deterministic inverse-Morton 4×4 placement (a 16⁴
tree: 65,536 leaves → 4,096 → 256 → 16 → root), local 16-way sealing (a petal
seals when all 16 children are RESOLVED — present or explicitly empty),
comma-style anti-resonance modulation, temporal/frontier visibility, and
pre-publication immutable chunk writes — as a way out of the
whole-batch-barrier trap without breaking cycle atomicity.

The operator sharpened the underlying question mid-arc (2026-08-18):

> *"Can temporal.rs amortize enough so that the cycles become permeable?"*
> — with the cost anchor *"a whole 64k batch costs 233ms write and seal"*
> (operator-measured, out-of-tree; the provenance discipline of the D-LGJ-W8
> spec applies: OPERATOR-MEASURED, not re-derived here), the observation
> *"the Seal was potentially expensive too"*, the pipeline hypothesis
> *"if SoA are allowed to pull through with 4-16 cycle processing it can
> self-amortize for L1 cache cycles"*, and the mechanism bet *"the lotus or
> Pythagorean comma and Morton self-alignment sorting would try to make
> batches cheap enough."*

§6 answers this against the evidence. Short form: **the cycle does not become
permeable; it becomes thin** — and the seal-cost evidence (§1.4) shows the
operator's mechanism bet attacks the same root as the F-ORD defect.

## §1 The write path as it exists (VERIFIED)

### §1.1 Arrival mints the coordinate

`BatchWriter::cast` (`crates/lance-graph-planner/src/batch_writer.rs:132-138`)
mints `CastId(next_id++)` in ARRIVAL order. `CastId`'s derived `Ord` is
insertion order by design (`:84-90`). The module doc's description of the read
side is DECLARED-UNWIRED by its own header (`:12-20`; `cast()` has zero
production call sites — `TD-DOC-COMMENTS-CLAIM-UNWIRED-BEHAVIOUR`).

### §1.2 The coordinate becomes the order key

`collect_casts` (`crates/lance-graph-supervisor/src/cycle_driver.rs:357-393`)
derives `stream_position = position_base + cast.0` (`:385`). The RESTART leg is
already answered: `position_base` is a durable cursor
(`:340-347`; falsifier `restart_stable_stream_positions_survive_writer_reconstruction`,
`:1460`) — stream positions stay monotonic across writer rebuilds. What is NOT
answered is the within-cycle leg: two runs of the same semantic work with
different worker completion orders mint different `(owner → stream_position)`
assignments.

### §1.3 The publication identity eats the arrival coordinate — THE DEFECT

`DetachedCycleBatch::freeze` (`crates/lance-graph-planner/src/persist_sink.rs:377-390`)
stable-sorts by `stream_position` (`:378` via `order_cycle_stably`, `:133-135`)
— so far so good: completion order never becomes STORAGE order. But
`content_hash` (`:401-429`) then folds **the `stream_position` VALUES
themselves** into the batch hash (`:414`,
`eat(&s.stream_position.to_le_bytes())`). Since those values are minted from
arrival (§1.1-1.2), **`batch_hash` — the durable idempotency key and the
publication identity — depends on producer completion order even after the
sort.**

The struct's own doc claims the opposite (`:359-362`: *"Identical completed
sets yield identical hashes regardless of worker completion order"*). That
sentence is FALSE today. `F-ORD-REAL-FALSIFIER.md` carries the mechanism and
the pre-registered test (landed alongside this audit: a GREEN two-sided defect
pin + an `#[ignore]`d RED falsifier asserting the desired property).

What survives: the **row-keyed image leg holds** — `image` is
`row → last payload in stream order` (`:379-382`), `row = row_of(owner)` is
identity-derived, so with per-owner rows and per-owner payloads the coalesced
image is arrival-independent (the defect pin asserts this GREEN leg
explicitly).

### §1.4 The seal is O(batch bytes) with a full copy and a byte-at-a-time hash (VERIFIED)

The operator's *"the Seal was potentially expensive too"* is verifiable at
file:line — three full passes over the batch payload:

1. **Sort** — `order_cycle_stably` is a general `sort_by_key` (`:133-135`);
   its own doc already concedes the lotus-shaped alternative: *"when the key
   is already a dense slot index into the cycle image, a stable scatter into
   predetermined positions is cheaper than a general O(n log n) sort on the
   64k path"* (`:126-129`).
2. **Clone** — `freeze` clones every payload into the image
   (`:381`, `image.insert(s.row, s.payload.clone())`); for 64K × 512 B that is
   a 32 MB copy. A second full clone is held as the retry cache
   (`cycle_driver.rs:420-425`: *"The one clone held until commit success is
   the price of offering that cache"*).
3. **Hash** — `content_hash` is byte-at-a-time FNV-1a over every payload byte
   (`:405-427`) — a serial, SIMD-hostile pass over the full 32 MB.

INFERENCE: at 64K × 512 B these three passes make the seal itself
O(tens of ms) before Lance sees a byte, consistent with the operator's 233 ms
write+seal figure being dominated by more than the Lance append. Not measured
in-tree; `BENCHMARK-PLAN.md` (Phase 5 deliverable) owes the split
(seal-only vs append-only vs total).

### §1.5 The gates that already exist (VERIFIED)

- Artifact gate: zero non-empty payloads → `CommitOutcome::NoChange`, sink
  never invoked, no version minted (`persist_sink.rs:618-630`).
- ≤1 move per owner per cycle, enforced pre-seal; extra moves HELD, re-staged
  intent-only (empty payload — deliberately unable to trip the artifact gate)
  (`cycle_driver.rs:337-410`).
- Failure semantics: on WAL failure nothing publishes, nothing mutates;
  prescribed recovery is **deterministic regeneration from the unchanged Vn**
  (`cycle_driver.rs:419-425`). This doctrine is load-bearing for §6's
  pipeline answer.

## §2 Persistence capability (VERIFIED + the session BLOCKER)

### §2.1 What the store actually calls

`LanceCycleWriter` (`crates/lance-graph/src/graph/cycle_sink.rs`) uses ONLY the
high-level Lance API: the sole `lance` import is
`use lance::dataset::{Dataset, WriteMode, WriteParams}` (`:89`), and the SOLE
mutation site is `raw_append` (`:706-728`) — `Dataset::write(…, Create)` on
first-ever commit, `ds.append(reader, None)` thereafter. One `RecordBatch` per
cycle carries all three row kinds (FRAME=0 / LANDING=1 / IMAGE=2;
`FixedSizeBinary(512)` payloads; `:99-144`) in ONE atomic commit (`:825-885`).
No `Fragment`, `FragmentCreateBuilder`, `Transaction`, `Operation::*`, blob, or
cleanup/GC API appears anywhere in the file (exhaustive grep). Supporting
machinery, all VERIFIED: `bootstrap()` mints a real version when it creates
(`:647-690`); `guard_schema` rejects drift loudly (`:358-397`); `WriterClaim`
is in-process-only exclusivity (`:221-274`); the `(cycle, batch_hash)`
Reconciled fast path (`:774-824`); `max_cycle` startup watermark (`:399-431`);
`checkpoint_bound` caps recovery below the latecomer fence
(`cycle_driver.rs:912-938`); `recover_and_apply` replays strictly above the
per-owner `stream_position` watermark in stored order, never re-sorting
(`persist_sink.rs:640-…`).

### §2.2 BLOCKER — the lance crate source is absent from this sandbox

`~/.cargo/registry/src/index.crates.io-…/` contains zero `lance*` directories;
no `.crate` archives cached; `vendor/` holds only `ractor`; `Cargo.lock` pins
the whole family `=9.0.0`. **Charter task 2(a)-(d) — fragment-level
write-without-commit (`write_fragments` / `Fragment::create`), a two-phase
prepared-fragments commit shape, orphan/cleanup APIs, and blob storage — is
UNVERIFIABLE from source this session.** Everything §2.1 states is about what
`cycle_sink.rs` CALLS, which is real but does not bound what lance 9.0.0
OFFERS. Phase 6 (`PREPARED-ARTIFACT-PUBLICATION.md`) is **gated** on a
`cargo fetch` (network + disk budget permitting) or an operator-sanctioned
alternative source consult. Do not write Phase 6 from memory of lance APIs.

> **⊘ BLOCKER LIFTED 2026-08-18 (same day, operator pin ruling).** The
> operator's dependency ruling ("pinned via [patch] of the upstream
> repository git") sanctioned the upstream source consult: the EXACT v9.0.0
> tag is on disk (`/tmp/sources/lance-9`, matching the Cargo.lock checksum)
> plus current upstream (`/tmp/sources/lance-main`). Capability findings are
> deliberately NOT recorded here — they enter through the RP-SEAL research
> program's Domain-A independent passes (`.claude/plans/
> erasure-seals-compaction-research-v1.md` §11 independence rule) and are
> cross-checked at consolidation. The paragraph above stands as the honest
> record of what THIS session's Phase 0 could and could not verify at
> composition time.

### §2.3 The existing "merkle" is not a tree (VERIFIED)

`MerkleRoot::from_fingerprint` (`crates/lance-graph/src/graph/spo/merkle.rs:18-38`)
is a flat XOR-fold rolling hash over one fingerprint's words — no parent/child
relation, no 16-slot fan-out anywhere in the crate (grep: zero matches for any
`H(parent)=H(children)` construct). `verify_lineage` is structural-only by its
own admission (*"Known gap: does not re-hash"*, `:143-157`). **Consequence:
the lotus petal hash — `H(parent) = H(canonical child slots 0..15)` — is NEW
design, not a reuse.** Nothing existing carries it; nothing existing conflicts
with it.

## §3 Placement + comma prior art (census; VERIFIED unless noted)

| Candidate | Where | Verdict for lotus |
|---|---|---|
| `FacetTier::morton()` | `lance-graph-contract` `facet.rs` (canon V3: "256 = 4⁴ hierarchical ancestry") | **Strongest leaf-identity candidate** — the 4-ary centroid-ancestry reading is exactly the 16⁴ tree's per-tier structure |
| `hhtl::NiblePath`, `FAN_OUT=16` | lance-graph core | Production 16-ary router; the doctrine already says *"HHTL is the deterministic PLACE (16-ary); helix is the RESIDUE"* |
| `CurveRuler` stride-4-over-17 | helix | **The shipped comma** — triple-falsified positive (DFT anti-aliasing ev7b; `comma_quorum` N_eff 11.00/12 vs 1.00 without; `comma_awareness`). D-QUANTGATE names it the mandatory quantized-layer phase generator |
| `basin_placement_learning` 75.8% | probe | **Overclaim flag (INFERENCE):** the measured 75.8% is a BINARY split, not 4-ary — `E-BASIN-IS-A-NODE` cites it wider than the measurement supports; re-grade before reuse |
| `symbiont::domino::morton4` | symbiont | ⊘ STRUCK 2026-08-18 (operator no-go: symbiont DEPRECATED per the #879/#911/#912/#913 arc) — was: real 4×4 bijective Morton, SIMD-lane-scoped, not a placement authority. Not a candidate; kept as census history only |
| thinking-engine `domino.rs` | thinking-engine | FALSE LEAD — name collision, unrelated mechanism |
| Feistel / XorShift mixers | — | ABSENT from the workspace; a comma design needing one imports new code |

Cross-cutting census finding: Morton interleave is reinvented ≥10× across the
workspace with no canonical home in `ndarray/src` — consolidation onto
`FacetTier::morton()` (or an ndarray primitive it consumes) is a named seam
for the wave, not this PR.

## §4 Frontier visibility — the adversarial findings (VERIFIED history, HYPOTHESIS design)

1. **The space the frontier reopens was deliberately closed.**
   `DurableWitness` / `DurableCoordinate` were RETIRED 2026-08-02 for epistemic
   race safety — the ruling's shape: *an open cycle's output is never visible
   as a `Vn` input*. Any design that lets a reader consume sealed-but-
   unpublished petals is re-entering that space and must carry the retirement
   reasoning, not route around it.
2. **F-VISIBILITY currently holds by STRUCTURAL ABSENCE** — pre-publication
   state never touches Lance at all (§2.1: one append per cycle, nothing
   else). Any prepared-petal write changes the mechanism that makes this
   falsifier pass from "impossible" to "guarded" — a strictly weaker footing
   that needs its own falsifier.
3. **Doc drift, correction owed:** `temporal.rs:396` + `:410` still cite the
   retired `persist_sink::DurableWitness` / `DurableCoordinate` as "the
   production implementor" — stale since the retirement. Fold the correction
   into the Phase 2 PR (or its own doc-fix PR); flagged here so it is not
   rediscovered.
4. **The supervisor and the core store have no Cargo edge** —
   `lance-graph-supervisor` (collect/seal/recover) and
   `lance-graph::cycle_sink` (the Lance writer) are composed only in prose;
   no crate depends on the other. The lotus seam would be the first real
   composition; that is a design event, not a refactor.
5. **The riskiest assumption in the whole charter (HYPOTHESIS):** a "sealed"
   state short of the atomic commit either (a) becomes durable pre-publication
   — threatening cycle-atomicity and reopening finding 1 — or (b) stays
   in-memory — contradicting the restart-reuse motivation. The charter's
   git-object pattern (content-addressed petal objects durable EARLY, one
   root/manifest flip publishing LATE) is the only shape found that threads
   this needle, and it is exactly the shape §2.2's BLOCKER prevents verifying
   against lance 9.0.0 this session.

## §5 Charter maxims, restated against the evidence

- **ARRIVAL MAY FILL A SLOT, NEVER CREATE IT** — today arrival CREATES the
  coordinate (§1.1-1.3). This is the F-ORD-REAL defect in maxim form.
- **UNRESOLVED IS NOT EMPTY** — no present/empty distinction exists anywhere
  in the write path today; `SweepSlot` has no tri-state. New design.
- **SEALED MAY BE CONSUMED, OPEN MAY NOT** — today NOTHING pre-publication may
  be consumed (§4.2); the maxim names a weakening that must be earned.
- **PREPARED IS NOT PUBLISHED** — no prepared tier exists (§2.1); gated on
  §2.2.
- **CANONICALIZATION IS CONSTRUCTIVE, NOT A REPAIR SORT** — the repair sort is
  live at `persist_sink.rs:378`, and its own doc (`:126-129`) already names
  the constructive alternative. See §6.3.
- **IF COMMA DOESN'T SURVIVE MEASUREMENT, DELETE IT** — §3's comma evidence is
  from OTHER carriers (quorum/awareness/DFT); zero placement-tier comma
  measurements exist. The comma enters this design as HYPOTHESIS only.

## §6 The permeability question, answered (INFERENCE on VERIFIED parts)

**"Can temporal.rs amortize enough so that the cycles become permeable?"**

Decompose "permeable" by who wants through the wall:

### §6.1 Reader-plane permeability is the retired race — don't reopen it; make the wall THIN instead

A query reader consuming open-cycle state as history is precisely the
`DurableWitness` retirement (§4.1). The need it serves — fresher-than-published
reads — is met without it by raising **publication frequency**: smaller, more
frequent cycles. Two costs currently forbid that, and each has an owner:

- **Write-side cost** (seal + append per cycle): attacked by §6.3.
- **Read-side cost** (a reader now spans MANY small versions): this is exactly
  what `temporal.rs` amortizes — `QueryReference::at(v, rung)` + deinterlace
  is a zero-copy projection over a version RANGE; N small versions read as one
  sorted stream. **HYPOTHESIS to measure (F-AMORT): range-read cost stays
  ~flat as version count per unit of work rises 10×.** If F-AMORT holds, the
  wall's reader-visible thickness is bounded by commit latency, not batch
  size — permeability's *effect* without its race.

### §6.2 Compute-plane permeability is legal TODAY, and the crash story already exists

The operator's *"SoA allowed to pull through with 4-16 cycle processing"* is a
bounded pipeline: compute cycles N+1..N+k proceed while seals/appends for
N-k..N trail behind. Two regimes, sharply different:

- **Regime A (trailing publication, published-horizon reads):** every cycle
  still reads a PUBLISHED `Vn`; the horizon lags the compute frontier by ≤k.
  No epistemic rule is touched — the Vn rule constrains what may be READ AS
  HISTORY, not how far compute may run ahead of durability. Crash semantics
  come free: the prescribed recovery is already *deterministic regeneration
  from the unchanged Vn* (§1.5) — lose the in-flight window, regenerate ≤k
  cycles. The bounded window bounds the regeneration blast radius.
- **Regime B (frontier chaining):** cycle N+1 reads cycle N's sealed-but-
  unpublished in-memory image. This is dataflow forwarding, not epistemic
  history — but the committed frame's `base_version` must then be the version
  N *will* publish, a speculative basis that collapses the whole window on
  N's failure. Regime B is HYPOTHESIS; it needs the F-NO-SPECULATION /
  F-NO-BARRIER falsifiers and council review. Regime A needs neither.

### §6.3 The L1 claim, made mechanical — and the operator's mechanism bet lands on the defect's root

64K × 512 B = 32 MB: the whole batch is not L1-resident (nor L2). What the
4-16-cycle pull-through actually buys is **loop interchange**: the per-cycle
barrier forces cycle-major iteration (stream all 32 MB through DRAM every
cycle); a k-deep pipeline permits row-tile-major iteration — process one
petal-sized tile (16 rows × 512 B = 8 KB, comfortably L1) for k consecutive
cycles while it is hot, then move on. DRAM traffic for the compute phase drops
~k×; the seal's three O(bytes) passes (§1.4) overlap compute instead of
serializing behind it. **That is the honest form of "self-amortize for L1
cache cycles" — the barrier is not just a latency wall, it forces a
cache-hostile iteration order.** (INFERENCE; the benchmark plan owes
cycle-major vs tile-major at k ∈ {1,4,16}.)

And the mechanism bet — *"lotus/Pythagorean comma and Morton self-alignment
sorting would try to make batches cheap enough"* — attacks §1.4's three passes
at the root they share with the F-ORD defect:

1. **Sort → gone.** Identity-derived Morton slots mean the canonical form
   exists AT CAST TIME; sealing scatters into predetermined positions — the
   constructive canonicalization `order_cycle_stably`'s own doc already
   concedes is cheaper (`:126-129`).
2. **Clone → gone.** Content-addressed petal objects read the SoA backing
   store zero-copy at flush — the descriptor doctrine `batch_writer.rs`
   already states (*"deltas stay in the SoA backing store; the sink reads
   them at flush time"*).
3. **Hash → incremental and arrival-independent by construction.** A petal
   tree `H(parent)=H(child slots 0..15)` re-hashes only dirty petals,
   parallelizes 16-way, and — because slot positions are identity-derived —
   **is the F-ORD fix**: publication identity stops eating arrival
   coordinates because arrival no longer mints coordinates.

**One design, two problems:** the seal-cost problem and the
publication-identity defect are the same disease (canonicalization as repair
over arrival-minted coordinates) and the lotus placement is one cure for both.
The comma's role is confined and testable: Morton clustering gives intra-tile
locality (the L1 tiling above) but can hot-spot petals under strided identity
sequences; a deterministic bijective comma permutation at the petal-INDEX tier
spreads stride resonance while leaving intra-petal Morton locality intact.
That is a measurable trade (locality vs resonance), pre-registered as
F-COMMA-*; per §5's maxim, if it doesn't survive measurement it is deleted.

### §6.4 The rung ladder makes the frontier rung-QUALIFIED — sudoku filling without poisoning hindsight

The operator's closing refinement: *"temporal.rs rung ladder dependency would
allow for sudoku-like filling of known unknowns without poisoning hindsight."*

The load-bearing surface already exists, VERIFIED: `temporal.rs` grades **what
a reader at a given rung is allowed to know** (`EpistemicMode`, `:74-99`) —
`for_rung(0..=4) = Strict` (*"reason strictly in the present"*),
`for_rung(5..=8) = Aware` (*"admit hindsight"*), `for_rung(9..) = Retro`
(*"may spoiler-read"*); `QueryReference::at(ref_version, rung)` (`:167-173`)
carries it per read. The epistemic dimension the frontier needs is not new
machinery — it is this ladder pointed at the OTHER direction (ahead of the
publication horizon instead of behind it).

That reconciles §6.2's Regime B with §4.1's retirement, as HYPOTHESIS:

- The lotus tree's UNRESOLVED slots are **known unknowns** — the slot exists
  (identity-derived, §6.3), only its value is pending. Filling them is
  constraint propagation in arbitrary arrival order (the sudoku shape:
  arrival fills a slot, never creates it), petals sealing as their 16
  children resolve.
- **Visibility is rung-qualified, not binary.** A frontier fill is consumable
  only by readers whose rung explicitly opted into frontier awareness; the
  hindsight rungs (`Retro`, and everything reading published history as `Vn`)
  see ONLY sealed-and-published state. The 2026-08-02 retirement reasoning
  (*"an open cycle's output is never visible as a `Vn` input"*) is thereby
  PRESERVED at the hindsight rungs and relaxed only where a reader typed its
  tolerance — the race is not reopened, it is stratified.
- The falsifier this pre-registers (fold into F-VISIBILITY/F-STALE):
  **hindsight-invariance** — a hindsight-rung read must be bit-identical
  whether or not any frontier activity existed at read time. Frontier-rung
  reads may vary with the frontier; hindsight reads never. That is "without
  poisoning hindsight," made mechanical and two-sided.

### §6.5 The workload split — texts are LINEAR; GridLake tiles must not claim them

Operator (2026-08-18): *"Texts are organized linearly. Gridlake tiles would
make it contradictory."* This is a scoping law for the whole design, and it
exposes a SECOND verified doc-vs-code tension:

`SweepSlot::stream_position`'s own contract (`persist_sink.rs:168-183`) says
it is *"the caller's EXISTING canonical (textual/stream) order key — the
write-side deinterlace input. NOT a new coordinate… the witness-fabric order
key, already monotonic."* But `collect_casts` MINTS it from arrival
(`position_base + cast.0`, §1.2). The persistence contract was written for a
caller-supplied SEMANTIC key; the supervisor supplies an arrival mint. Two
workload families therefore need two different repairs, and conflating them
is the contradiction the operator names:

- **Linear/textual streams** (the temporal.rs sorted stream, episodic
  Markov-on-stream per `E-MARKOV-TEMPORAL-STREAM-1`, witness chains): stream
  order IS semantic. The fix is to CARRY the true textual/witness position
  through the cast instead of minting one — and then hashing it is CORRECT,
  because the coordinate is semantic identity, not arrival residue. Morton
  tiling must never claim this order key; deinterlace depends on it.
- **Grid/tile workloads** (owner-keyed SoA rows, spatial/semantic tiles):
  no meaningful linear order exists; the coordinate should be
  identity-derived (the lotus slot, §6.3) or absent from identity entirely.

The binding precedent already exists in canon: **domains bind the axes** (the
tier reading is class-resolved — "OSM: literal x/y; semantic: PQ subspace
pairs"). The order-key reading joins that list as a per-class/per-tenant
resolution — linear-stream classes keep a supplied semantic key; tile classes
get derived placement. One substrate, two sanctioned readings, resolved where
every other reading is resolved: at the class, never hardcoded in the sink.

### §6.6 RAM: chunking when the thought happens vs forcing the batch to freeze

Operator (2026-08-18): *"Forcing the whole batch to open for displaying a map
also affects RAM usage. The question is: is chunking when the thought happens
more efficient than forcing a batch to freeze?"*

**The freeze model's RAM shape, VERIFIED — the batch is resident up to THREE
times at seal:**

1. Staging holds OWNED payload bytes — `BatchWriter.pending_payloads:
   Vec<(CastId, P)>` with the production-shaped instantiation being
   `BatchWriter<Vec<u8>>` (`collect_casts`'s signature,
   `cycle_driver.rs:357-358`). This is itself a doc-vs-code drift: the
   module's own descriptor doctrine (`batch_writer.rs:30-39`, operator ruling
   Addendum-6: *"P is a DESCRIPTOR — never owned delta bytes; deltas stay in
   the SoA backing store; the sink reads them at flush time"*) is unrealized
   on the only wired path.
2. `freeze` clones every payload into the image (§1.4.2).
3. The retry cache holds a third copy until commit success (§1.4.2).

At 64K × 512 B that is ~32 MB × up to 3 transient, per cycle — plus the
read side: a map viewport that needs a handful of tiles today hydrates
whole-batch-shaped state, because nothing smaller than the cycle image exists
to hydrate.

**Chunk-at-thought (the lotus shape) converts the spike into a stream:** a
thought's 512 B lands in its identity-derived petal buffer when produced;
a petal seals at 16 resolved children (8 KB — one L1-resident tile, §6.3);
RAM high-water becomes O(open petals), not O(batch) × 3; and a map reader
hydrates exactly the petals its viewport touches — partial hydration is what
the content-addressed tree is FOR (the git-object read path).

**The honest counterweights (INFERENCE — this is a benchmark question, not a
settled one):**

- **Coalescing is a real freeze benefit** — *"64 same-row breaths durably cost
  ONE image row"* (`cycle_sink.rs:107`). Chunk-at-thought either writes every
  breath (write amplification × breaths/row) or keeps a per-petal coalescing
  window — the window is small (16 rows), but it must exist.
- **Small-write amplification** — 4,096 sealed petals × 8 KB as individual
  durable objects is the object-store anti-pattern; git itself answers this
  with packfiles. The petal tier likely stays in-memory (or in one streaming
  WAL) until root publication regardless — in which case the WRITE-side I/O
  comparison may be a wash and the real wins are RAM high-water, seal
  incrementality (§6.3), and read-side partial hydration.
- **Publication atomicity is untouched either way** — one cycle, one version
  (§4/§2.1); chunking changes what exists BELOW the commit, never how many
  commits there are.

Pre-registered as the F-MEM falsifier plus a benchmark row (Phase 5):
RAM high-water and seal latency, freeze-model vs petal-model, at 64K × 512 B
with breaths/row ∈ {1, 4, 64}, plus a viewport-read row (hydrate 1% of tiles:
bytes touched, freeze vs petal). If the petal model does not win RAM
high-water by a large factor, it is not carrying its complexity.

### §6.7 Verdict shape for the phases ahead

- The RED F-ORD-REAL falsifier lands NOW (this PR) — it is true regardless of
  which fix wins, and it is the gate every fix must turn green.
- Regime A pipelining + F-AMORT measurement need no epistemic weakening and no
  lance capability beyond what §2.1 verified — they are the low-risk half.
- Prepared petals (the git-object pattern) and Regime B are gated on §2.2's
  BLOCKER and on council review of §4's retirement reasoning.
- No metaphor becomes a type name without council approval (charter).

## §7 Index

| Deliverable | Status |
|---|---|
| 1. LOTUS-FRONTIER-AUDIT.md | this document |
| 2. F-ORD-REAL-FALSIFIER.md | landed alongside, with the pre-registered tests |
| 3-9 (design/experiment/benchmark/verdict docs) | NOT STARTED; 6 gated on §2.2, comma work gated on F-ORD fix design |

Falsifiers referenced: F-ORD-REAL (landed), F-AMORT, F-NO-BARRIER,
F-NO-SPECULATION, F-VISIBILITY, F-COMMA-*, F-LOCAL-SEAL, F-RESTART, F-DUP,
F-STALE, F-PUBLISH, F-64K, F-MEM — pre-registration doc is deliverable 8.
