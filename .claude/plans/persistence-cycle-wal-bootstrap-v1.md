# persistence-cycle-wal-bootstrap-v1 — the primitive cycle/WAL seam and its temporal/revision upgrade path

> **Status:** ACTIVE (bootstrap SHIPPED in PR #878; upgrade phases PLANNED).
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
boundary is the whole point. The upgrade path is §2–§3; the accepted debts are
§4.

---

## 2. The intended larger architecture — two orthogonal dimensions

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
> dimension's job (§2.1). Conflating the two is the error this section forecloses.

---

## 3. Planned temporal error-correction phase (later; not #878)

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
immutability that keeps the hindsight-lookup in §2.2 honest.)

---

## 4. Known limitations accepted for #878

Recorded explicitly as **upgrade points**, not as claims that #878 already
solves the final temporal model:

- The scalar slot model is **provisional**.
- A scalar key **may not capture** future partial-order cognition (§2.1).
- Cross-owner and equal-position **conflict semantics are not finalized**.
- Cycle **retry** and **production WAL idempotence** still require
  concrete-sink hardening.
- The **fake WAL sink proves the contract shape, not real crash durability**
  (`compile+test green ≠ storage proven`, the Ladybug lesson).
- The **complete horizontal temporal projection remains future work** (§2.1,
  cross-ref `temporal-markov-and-style-classes-v1.md`).

---

## 5. Scope exclusions (this document and the #878 bootstrap)

- Do **not** introduce or document detailed **cohort internals**.
- Do **not** mention a **fixed number of cohort slots**.
- Do **not** design **actor-neighbour waiting or firing dependencies** (the
  emit path is wait-free by §3).
- Do **not** invent new **semantic, temporal, rung, witness, branch, or
  ancestry** types.
- Do **not** revive `ThoughtWitness`, `basis`, `awareness_seq`, or **per-cast
  WAL persistence** (all retired by the reshape ruling; the durable unit is the
  cycle, not the cast).
- The **cohort execution model** is out of scope — it belongs to the separate
  cohort architecture work, not here.

---

## 6. Status snapshot

| Aspect | State |
|---|---|
| Cycle/WAL bootstrap seam (`persist_sink`) | **SHIPPED** in PR #878 (bootstrap; contract-probed, not storage-proven) |
| Vertical axis (`DatasetVersion` succession + lookup) | Established primitively by the seam |
| Horizontal axis (`temporal.rs` coherence over a frame) | **PLANNED** — owned by `temporal-markov-and-style-classes-v1.md` |
| Shadow temporal-coherence correction pass | **PLANNED** (§3) |
| `revision.rs` forward-correction mechanism | **PLANNED** (§3) |
| Concrete Lance sink (real crash durability) | **DEFERRED** — gated on crash falsifiers (§4) |

The bootstrap exists so the rest can be built on a running, durable seam. The
scalar order is a load-bearing placeholder, and this document is the record that
it is a placeholder — nothing here claims the primitive ordering is the final
temporal architecture.
