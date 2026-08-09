# persistence-cycle-wal-bootstrap-v1 — the primitive cycle/WAL seam and its temporal/revision upgrade path

> **⊘ PARTIALLY SUPERSEDED (operator ruling 2026-08-09) — read
> `.claude/plans/persistence-artifact-backed-commit-v1.md` FIRST.** Guarantee 5
> below ("one successful cycle seal produces exactly one `DatasetVersion`") is
> now conditional: **no artifact-backed semantic change → no write → no new
> version**. An empty / intent-only cycle produces `CommitOutcome::NoChange`
> with ZERO store operations. Guarantees 1–4 and 6 survive unchanged, and
> guarantee 6 is strengthened (the writer additionally holds no owner borrow
> across the commit await). The §2 sparse-delta rule is now IMPLEMENTED — the
> concrete sink is `lance_graph::graph::cycle_sink::LanceCycleWriter`, whose
> coalesced image rows are the durable end-form. Append-only: nothing below is
> deleted; it is read through the newer contract.
>
> **Status:** ACTIVE (bootstrap SHIPPED in PR #878; upgrade phases PLANNED; the
> §2 sparse-delta storage rule is RATIFIED architecture, IMPLEMENTED in
> `LanceCycleWriter` as of Phase A 2026-08-09).
> **Date:** 2026-08-02.
> **Scope:** documentation-only architectural ruling. Records the *role* of the
> #878 persistence seam and the intended larger two-dimensional temporal
> architecture it bootstraps toward. Changes **no** Rust code, tests, public
> APIs, or `temporal.rs`.
> **Owns (narrowly):** "what #878 is and is NOT", the horizontal/vertical
> temporal split as it touches persistence, and the later
> `temporal.rs`-shadow + `revision.rs` correction path.
> **Does NOT own (cross-refs, never re-specifies):**
> - Horizontal temporal-stream detail → `temporal-markov-and-style-classes-v1.md`
>   (the ratified 2026-07-10 arc; `E-MARKOV-TEMPORAL-STREAM-1`).
> - D-MBX-A6 deliverable tracking → `.claude/board/STATUS_BOARD.md`
>   (D-MBX-A6-P1…P3e rows).
> - Per-row `write_row` cycle-gate → `mailbox-cycle-aware-write-contract-v1.md`
>   (a *different* deliverable — the SoA setter gate, not the WAL seam).
> - The reshape ruling itself → `.claude/board/EPIPHANIES.md`
>   `E-THE-DURABLE-UNIT-IS-THE-CYCLE-NOT-THE-CAST-ONE-WAL-WRITE-PER-SWEEP-1`
>   and `E-THE-PAIRED-MOVE-MUST-BE-DURABLE-CO-LOCATED-NOT-IN-MEMORY-ONLY-1`.
>
> This is the bootstrap-and-upgrade companion, not a competing architecture.

---

## 1. Current #878 role — a PRIMITIVE cycle/WAL bootstrap (deliberate)

#878 deliberately establishes a **primitive** persistence seam so that thinking
can become *wired and runnable* without first solving the complete cognitive
topology. It is scaffolding that carries load, chosen so the execution and
durability plumbing exists and is exercised **before** the final temporal or
cognitive representation is settled.

Its accepted responsibilities, end to end:

```
concurrent thought results
  → primitive slot collection and ordering
  → freeze one complete cycle
  → one amortized WAL write
  → one Lance DatasetVersion
```

The **hard guarantees** #878 makes (and only these):

1. All work in a cycle reads exactly **one sealed predecessor version `Vn`**.
2. **Open-cycle results are not visible as `Vn` input** (an unsealed cycle is
   excluded from the sealed read horizon).
3. The cycle is **frozen before durable I/O** (order + coalesce happen on a
   detached snapshot, then the append runs).
4. One successful cycle seal produces **exactly one WAL write**.
5. One successful cycle seal produces **exactly one `DatasetVersion`**.
6. The WAL **never receives a live mutable SoA borrow** (the persistence path
   holds no owner across I/O).

### The scalar slot/order model is explicitly PROVISIONAL

The current scalar slot + single-key ordering model is **sufficient** to
establish the execution and durability plumbing, and **nothing more**. It is
**not** claimed to be the final representation of temporal or cognitive order.
This document does **not** redesign or fix that limitation — recording the
boundary is the whole point. The sparse-delta storage ruling is §2; the
two-dimensional upgrade path is §3–§4; the accepted debts are §5.

---

## 2. A complete logical cycle is physically SPARSE (RATIFIED architecture, UNIMPLEMENTED in a concrete sink)

> **Status of this section:** the sparse-delta rule is **RATIFIED as
> architecture** and **UNIMPLEMENTED in a concrete Lance sink**. The #878
> bootstrap remains SHIPPED; this section governs the *future* concrete sink,
> not the merged contract-probe.

**"One complete cycle image" must NEVER be read as serializing every row merely
because every participant belonged to the cycle.** The load-bearing distinction:

```
complete logical cycle   ≠   full physical dataset rewrite
```

A cycle is **logically complete** when:

- all required participants reached the cycle boundary,
- all produced updates were collected,
- updates were temporally ordered / coalesced,
- all required lifecycle transitions were included,
- the resulting change set was frozen,
- the change set was committed atomically.

The **physical payload stays sparse**:

```
64k participants
  → N dirty rows (N may be ≪ 64k)
  → one frozen sparse delta batch
  → one WAL transaction
  → one DatasetVersion
```

Unchanged rows are **inherited from the sealed predecessor version** and MUST
NOT be serialized merely because they participated in the cycle.

### The storage invariant (RATIFIED)

> **«A cycle is globally complete but physically sparse. `commit_cycle`
> persists only the coalesced dirty-row set and the required durable transition
> metadata. Unchanged rows remain inherited from the sealed base version and do
> not become new row payloads merely because they were members of the cycle.»**

**"One WAL write per cycle" means one atomic durability boundary for the sparse
change set.** It does **not** mean one full 64k-row (~32 MiB) snapshot per
cycle.

### Participation is separate from mutation

Cycle participation / completion evidence is represented **compactly and
separately** from row payloads:

```
cycle completion evidence (small footer):
    cycle identity
    sealed base version
    expected / completed participation evidence or digest
    dirty-row count
    transition count
    batch digest

physical delta:
    only rows whose FINAL state changed
    only the required durable lifecycle transitions
```

**A participant that produced no state mutation MUST NOT require a 512-byte row
payload merely to prove participation.** (Cohort internals — participant-count
encoding, bitmap layout, ownership — are out of scope here; they belong to the
cohort architecture session.)

### Payload-duplication warning (honest bootstrap limitation)

The #878 contract-probe shape currently duplicates bytes in memory:

- `SweepSlot` owns payload bytes,
- `DetachedCycleBatch` retains the per-landing records,
- `freeze` also clones the final row payloads into the coalesced `image`.

This is acceptable for the in-memory contract-probe fake. **The concrete Lance
sink must NOT persist duplicate copies of both the per-landing payload bytes AND
the final coalesced row image.** The future concrete shape keeps the three
concerns distinct:

```
landing metadata / durable transitions
  + one detached coalesced dirty-row image
  + a small cycle footer / completion evidence
```

*(Recorded as a concrete-sink upgrade requirement — the Rust structs are NOT
redesigned in this task.)*

### Capacity + backpressure (concrete-sink ruling)

```
normal cycle:  a sparse dirty-row delta
worst case:    every row genuinely dirty → ≈ one full row slab
```

A genuinely dense cycle is **valid**, but it is an **explicit capacity event**,
not the default storage shape. The future concrete sink MUST define (numeric
thresholds are NOT chosen in this documentation task):

- maximum frozen cycles in flight,
- maximum bytes in flight,
- backpressure when WAL / storage falls behind,
- checkpoint / compaction policy,
- version-retention policy,
- disk-space monitoring + a refusal threshold.

### Required future falsifiers (concrete-sink, probe-first)

1. **Sparse-cycle** — 64k logical participants, 17 dirty rows → one WAL
   transaction → **exactly 17** coalesced row payloads written → one
   `DatasetVersion` → unchanged rows inherited from `Vn`.
2. **No-op-cycle policy** — zero dirty rows AND zero durable transitions → the
   sink follows ONE explicitly documented policy: *either* no new
   `DatasetVersion`, *or* a metadata-only cycle version. The policy is chosen
   before the concrete sink ships; it MUST never write a full empty row slab.
3. **Coalescing** — many updates to the same row in one cycle → **one** final
   row payload physically written; intermediate payload copies are NOT persisted
   as duplicate row state; the required transition history remains available
   separately.
4. **Dense-cycle capacity** — all rows genuinely dirty → one bounded dense
   batch; backpressure prevents an unbounded queue of dense frozen cycles; no
   silent disk exhaustion.
5. **Retention** — many `DatasetVersion`s accumulate → a documented
   retention / checkpoint policy bounds disk growth; versions inside the
   configured hindsight horizon remain readable.

**Scope fence for this section:** it concerns *physical persistence density
only*. It does not pull cohort topology (§6), horizontal partial ordering
(§3.1), or revision semantics (§4) into the concrete-sink design.

---

## 3. The intended larger architecture — two orthogonal dimensions

The final architecture is two **orthogonal** dimensions. #878 supplies the
vertical axis primitively and leaves the horizontal axis to a later phase.

### 2.1 Horizontal dimension — coherence within a frame (`temporal.rs`)

`temporal.rs` preserves and reconstructs **horizontal causal and contextual
coherence** — the relationships *among* the results that make up one frame.

For **deterministic streamed material** (e.g. the Bible), the initial
horizontal position is strongly determined by the source structure:

```
book → chapter → verse → token/span
```

Preserving this horizontal structure is required because it is what the local
awareness reads:

```
signed i4 neighbourhood pointers
  → relative-pronoun and anaphora relations
  → local grammatical and causal relationships
  → coherent standing-wave awareness
```

For **medical knowledge and higher-order thought**, horizontal relationships
may **not** be monotonically ordered — they may form **concurrent or partially
ordered neighbourhoods**. The final temporal architecture must therefore be able
to resolve **more than a scalar total order**. (This is precisely why the #878
scalar key is labelled provisional in §1.)

> The horizontal-stream mechanism itself — the version-range read
> (`QueryReference::at(v, rung)` + deinterlace), the ±5-generalizing window, the
> `LocalCausalRow`/`local_trajectories` reconstruction — is owned by
> `temporal-markov-and-style-classes-v1.md` and is **not** re-specified here.
> This document only records *that* the horizontal dimension is `temporal.rs`'s
> responsibility and *why* a scalar key is insufficient for it.

### 2.2 Vertical dimension — successive durable frames (`DatasetVersion`)

Lance `DatasetVersion` is the **vertical** axis. Each successful cycle seal
advances the durable frame:

```
Vn → Vn+1 → Vn+2
```

The version table therefore provides **both**:

- **vertical cognitive/rung progression** (each sealed frame is one discrete
  advance), and
- **cheap historical time-series lookup** (a version is selected without
  replaying any landing).

Consumers such as **Stockfish-style hindsight tests** can select versions from
the Lance version table because the frames have already been sealed as discrete
durable states — the lookup is over already-coherent frames.

> The version table performs **vertical** frame succession and lookup **only**.
> It does **not** perform horizontal causal ordering — that is the horizontal
> dimension's job (§3.1). Conflating the two is the error this section forecloses.

---

## 4. Planned temporal error-correction phase (later; not #878)

The primitive #878 slot model may initially produce frames whose **horizontal
coherence is incomplete** — the scalar order captures *that* results happened,
not the full partial-order structure among them.

A later phase may run `temporal.rs` as a **metacognitive / shadow correction
layer** over the produced frame. This layer **must not** require ractor actors
to wait synchronously for neighbours during thought execution — the emit path
stays wait-free.

Intended direction:

```
actors emit without neighbour waits
  → primitive cycle is collected
  → temporal coherence is evaluated (shadow pass)
  → inconsistencies become correction / revision input
  → a LATER cycle carries the corrected interpretation
```

`revision.rs` is the **planned complementary mechanism** for reinterpretation,
circular reasoning, and correction from the reflective side. (Distinct from the
NARS *belief-revision* of policy atoms in `triangle-tenants-gestalt-separation-v1.md`
— that revises `LearnedStyle`; this `revision.rs` feeds justified *reinterpretation*
into subsequent cognition.)

Conceptually, the three responsibilities stay separate:

```
temporal.rs   detects or reconstructs horizontal coherence
revision.rs   feeds justified correction into subsequent cognition
Lance versions preserve the successive durable vertical frames
```

**Iron constraint on the correction path:** a sealed historical version **must
not be silently rewritten** merely because a later metacognitive pass revised
its interpretation. Correction flows *forward* — into a later cycle / later
version — never *backward* over a sealed frame. (This is the vertical-axis
immutability that keeps the hindsight-lookup in §3.2 honest.)

---

## 5. Known limitations accepted for #878

Recorded explicitly as **upgrade points**, not as claims that #878 already
solves the final temporal model:

- The scalar slot model is **provisional**.
- A scalar key **may not capture** future partial-order cognition (§3.1).
- Cross-owner and equal-position **conflict semantics are not finalized**.
- Cycle **retry** and **production WAL idempotence** still require
  concrete-sink hardening.
- The **fake WAL sink proves the contract shape, not real crash durability**
  (`compile+test green ≠ storage proven`, the Ladybug lesson).
- The contract-probe shape **duplicates payload bytes in memory** (`SweepSlot` +
  `DetachedCycleBatch` landings + the coalesced `image`); the concrete sink must
  not persist both copies. Full statement + the sparse-delta storage invariant
  and the five concrete-sink falsifiers are in §2.
- The **complete horizontal temporal projection remains future work** (§3.1,
  cross-ref `temporal-markov-and-style-classes-v1.md`).

---

## 6. Scope exclusions (this document and the #878 bootstrap)

- Do **not** introduce or document detailed **cohort internals**.
- Do **not** mention a **fixed number of cohort slots**.
- Do **not** design **actor-neighbour waiting or firing dependencies** (the
  emit path is wait-free by §4).
- Do **not** invent new **semantic, temporal, rung, witness, branch, or
  ancestry** types.
- Do **not** revive `ThoughtWitness`, `basis`, `awareness_seq`, or **per-cast
  WAL persistence** (all retired by the reshape ruling; the durable unit is the
  cycle, not the cast).
- The **cohort execution model** is out of scope — it belongs to the separate
  cohort architecture work, not here.

---

## 7. Status snapshot

| Aspect | State |
|---|---|
| Cycle/WAL bootstrap seam (`persist_sink`) | **SHIPPED** in PR #878 (bootstrap; contract-probed, not storage-proven) |
| Vertical axis (`DatasetVersion` succession + lookup) | Established primitively by the seam |
| Horizontal axis (`temporal.rs` coherence over a frame) | **PLANNED** — owned by `temporal-markov-and-style-classes-v1.md` |
| Shadow temporal-coherence correction pass | **PLANNED** (§4) |
| `revision.rs` forward-correction mechanism | **PLANNED** (§4) |
| Concrete Lance sink (real crash durability) | **DEFERRED** — gated on crash falsifiers (§5) |
| Sparse-delta storage rule (complete cycle ≠ full rewrite) | **RATIFIED architecture, UNIMPLEMENTED in a concrete sink** (§2) |

The bootstrap exists so the rest can be built on a running, durable seam. The
scalar order is a load-bearing placeholder, and this document is the record that
it is a placeholder — nothing here claims the primitive ordering is the final
temporal architecture.
