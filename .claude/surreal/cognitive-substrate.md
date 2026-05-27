# Cognitive Substrate Architecture — SurrealDB-on-Lance · ractor · NARS · zero-copy SoA

> Synthesized 2026-05-26 from design dialogue + **code verification**. This doc is
> deliberately honest: every claim is tagged **[CODE]** (verified in a repo),
> **[DESIGN]** (specified, partially built), or **[VISION]** (not in code yet).
> It replaces narrative with status so it survives a review.

## 0. Thesis

One Rust binary replaces **BEAM + ClickHouse** for a workload that sits in the
*gap* between them: compute-heavy, multi-model, transactional, zero-copy, embedded.
The win is collapsing compute + coordination + storage **and the serialization tax
between them** into one address space. **[DESIGN]**

Honest scope: BEAM (telecom-grade distributed I/O concurrency) and ClickHouse
(petabyte OLAP) remain better at *their* native jobs. The differentiator here is
**depth/autonomy per item, not throughput** — at 20–200 tickets/min a simple
queue+worker meets the throughput SLA too.

## 1. Zones

```
ZONE 0-2 (intern)   preemptiv (OS-Threads/rayon), zero-copy LE/SoA, NARS-Kern,
                    L1-resident attention focus. No async, no lock, no serialization
                    on the hot path. 20-200ns/cycle.
   │  lock-free ring (the ONE seam, non-blocking move)
ZONE 3  (membrane)  durability commit = SurrealDB-on-RocksDB; external I/O = OSINT;
                    structured integration = sqlx/API. Serialization lives ONLY here.
```

- **ractor is async-only** (tokio *or* async-std, both cooperative; no OS-thread/sync
  mode — verified, `slawlor/ractor`). So ractor lives **at the membrane** (cold path);
  the internal preemptive cores are **OS-threads/rayon, not ractor**. **[CODE: ractor async-only]**
- Preemption comes from the OS scheduler on real threads — a cooperative async runtime
  never preempts. A long L1-resident NARS burst on a cooperative runtime is lose-lose
  (no yield → starves peers; yield → evicts the L1 focus tile). Only OS-thread
  preemption escapes both. **[DESIGN]**

## 2. The one marker — Rubikon FLOW

```
collapse-gate FLOW = Rubikon-crossing = zone membrane = SurrealDB commit = WA batch boundary
```

One marker, five lenses. At **FLOW**, four things atomically: **externalize + persist
(flush/sink-in) + prune (forget) + log (replay unit)**. HOLD = stay internal
(reversible). BLOCK = discard (WA = 0). **[DESIGN]**

The numbers are *operational* (SLA/latency budget), **not** Libet's 550/200 ms —
those are simple-finger-movement constants, not a volition law, and a 200 ms window
cannot host a human veto (perceive+decide > 200 ms; Libet's veto is intracerebral).
Keep the *phase structure*; reject the neuro-numerology. **[DESIGN]**

## 3. Control model — attention + goalstate, NOT request

NARS runs free (anytime/AIKR); the tick **harvests** (Rubikon solved? → FLOW). The five
implicit request guarantees are now carried by: *what* = attention/budget · *when* =
goalstate check · *who* = commit→live-query · *SLA* = tick cadence · *backpressure* =
budget **fed by outbound saturation**. The tick is a sampling cadence, **not** a compute
deadline — thinking is unbounded by design. **[DESIGN/VISION — see §Verification]**

## 4. Data contracts

- **LE/SoA zero-copy contract** (data): pointer-free, position-independent; the same
  bytes as RAM / WAL / storage. **One format *mechanism*, many *schemas*** — do not
  force one rigid contract for data *and* orchestration (data = wide stable primitive
  columns; orchestration = few rich evolving records). **[DESIGN]**
- **Immutable, append-only ontology** (OGIT/OWL/DOLCE/CAM/labels): O(1) lookup via a
  version-sealed (minimal-)perfect-hash into an immutable arena. Append-only ⟹ stable
  pointers ⟹ cheap delta-versioning. Append **downward only** (DOLCE/OGIT-upper fixed).
  This makes the commit-time DOLCE constraint check **O(1)** (cheap solved-predicate).
  *O(1) classification ≠ O(1) computation* — Odoo business *logic* that computes over
  runtime data is not a lookup. **[DESIGN]**

## 5. Memory-hierarchy isomorphism

```
register (operands) ⊂ L1 (attention focus tile) ⊂ L2 (warm concepts) ⊂ RAM/Lance (full bag)
attention = cache tile · forgetting = eviction · attention-shift = tile reload (gather)
```
"Finish thinking in L1" = **cache-blocking for cognition**. The bag is NOT cache-bounded
(it's the durable store); the **focus** is. 20-200ns is a *cache-resident* number — one
RAM miss (~100ns) breaks it. **[DESIGN — precondition NOT yet in code, see §Verification]**

## 6. Execution & ownership (the borrow guarantee)

- **Mailbox = compile-time ownership bracket** at start/end. Message-passing = move
  semantics = exclusive ownership, enforced by the borrow checker at *compile time* →
  free at runtime; ractor's ~200ns is paid only at the two endpoints, never per cycle.
  `Send + 'static` **forces** owned transfer — a borrowed `&[u8]` view can't cross the
  mailbox. The owned span between brackets is lock-/race-free *by construction*. **[DESIGN]**
- Cross-runtime (ractor edge ↔ OS-thread core) is a **move over the lock-free ring**,
  never a shared `&mut`.

## 7. Durability & recovery

- **Epoch checkpoint**: durability at cycle boundaries, not within. One process = one
  failure domain ⟹ the checkpoint is **load-bearing** (only crash-recovery boundary).
- **commit = flush + prune-if-unreferenced** (committing *is* forgetting → cache stays
  bounded by throughput). Prune **after** durable (Lance-first); reference-aware eviction.
- **SoA-unit sink-in = event-sourced replay log.** Replay recovers committed **state**,
  **not** the reasoning **trajectory** (NARS under AIKR is resource/timing-dependent →
  not path-deterministic). Needs **log compaction** (snapshot fold) or replay/storage
  grow unbounded. **[DESIGN]**

## 8. Scale path

`kv-rocksdb` embedded (one binary, the elegant case) → `kv-tikv` distributed (Raft+RocksDB)
for HA. Same model/API; **code-level feature swap, ops-level data migration** (TiKV uses
its own RocksDB/Titan fork + cluster). Trades the one-binary elegance for HA — two
deployment *modes* of one codebase, not simultaneous. Use as the **review-shield against
"JanusGraph+Cassandra have HA"**: wrong tool for in-process cognitive compute, *and* we
have strongly-consistent (CP/Raft) HA when needed — arguably better-suited than Cassandra's
eventual consistency for auditable command-states. **[CODE: kv-rocksdb + kv-tikv backends exist]**

---

# Verification (2026-05-26) — Code / Design / Vision

## Verified present **[CODE]**

- **SurrealDB `kv-lance` backend**, default `WritePath::LsmWithWal`: WAL(fsync) +
  memtable (`dashmap`, generation-ordered) + background flusher → Lance `MergeInsertBuilder`.
  `core/src/kvs/lance/{wal,memtable,flusher,mod}.rs`.
- **`kv-rocksdb` + `kv-tikv` backends** present (the HA scale path).
- **WAL serializes via CBOR** (`ciborium`, `wal.rs:133`); Lance commit copies into Arrow
  (`schema.rs build_write_batch .collect()`). → the path is **NOT zero-copy as built**;
  the honest term is **zero-*serialization*** (achievable with raw-`#[repr(C)]`-LE + checksum
  replacing CBOR) + **one unavoidable engine-internal copy** (memtable/Arrow = the write itself).
- **NARS *data* model**: `NarsTruth` (freq+conf, pack/unpack) + `NarsBudget` packed in the
  holograph fingerprint schema (`holograph/src/width_16k/schema.rs`).
- **VSA superposition-bundle** (majority vote) in holograph: `HdrVector::bundle` /
  `bundle_weighted`, `representation.rs`, `bitpack.rs`, `sentence_crystal.rs`.
- **ractor is async-only** (tokio/async-std, cooperative; no preemptive/OS-thread mode).
- **splat3d render-depth certification** (ndarray PR #206, merged): DTO + scalar ref +
  SIMD batch + HHTL depth cascade + mesh anchor.

## The two load-bearing gaps

### Gap #1 — bounded NARS attention-bag: **NOT FOUND [VISION]**
`NarsTruth`/`NarsBudget` exist as *data*, but **no bounded NARS concept-bag with
forgetting** was found in `lance-graph`. The free-running reasoning *engine* (inference
loop, revision, bag-with-capacity) is not verifiably present. **Consequence:** the
cache-residency precondition the entire 20-200ns latency story rests on is **unimplemented**.
This is the #1 risk — the latency guarantee has no code backing yet.

### Gap #2 — commit merge is a **last-writer-wins relic [CODE, needs fix]**
The flusher's `MergeInsertBuilder.when_matched(WhenMatched::UpdateAll)` is LWW, and
`flusher.rs:251` states it was *"extracted from the prior commit_gate::single_lance_commit"*
— i.e. a **relic** from the (now-idle) `CommitGate`, not a deliberate semantics.
- `lance_graph_contract`'s KV-level `MergeMode::Bundle` is *defined* as "later writer wins",
  so the comment "LWW is consistent with Bundle" is self-consistent **at the KV level**.
- BUT the **cognitively-correct** merge is **superposition** (holograph majority-vote),
  which retains both contributions; LWW *discards* the earlier one.
- LWW is only safe under **hard key-partitioning** (same key never collides in a batch).
  The backend documents "conflicts **rare**" via BindSpace key-prefix sharding — and
  *rare ≠ never*: on the rare collision **LWW silently drops a contribution.**
- **Fix:** wire the holograph superposition-bundle into the merge point (memtable conflict
  or read-bundle-write in the flusher — Lance `MergeInsert` cannot superpose natively),
  **or** prove hard (not merely rare) key-partitioning.

## Other items still Design/Vision

- **Phase 4 (outcome/Bewerten)**: needed for *learned*, not just *produced* — no
  outcome→NARS-desire feedback path verified. Without it the credit-assignment loop
  (which thinking-style produced a good result) doesn't close. **[VISION]**
- **Preemptive internal execution** (OS-thread/rayon NARS cores + lock-free ring to the
  ractor edge): the *design*; not verified wired. **[DESIGN/VISION]**
- **zero-serialization commit** (raw-LE instead of CBOR/Arrow): **[VISION]** — current is CBOR.

## Load-bearing constraints (must hold or it breaks)

1. Attention focus hard-bounded to L1 (budget caps focus width) — **Gap #1, unbuilt**.
2. Forgetting = resource governor **and** latency guarantee (cache residency).
3. Backpressure: outbound saturation must feed back into the attention budget.
4. The cognitive merge must superpose where collisions are possible — **Gap #2, relic LWW**.
5. Cheap solved-predicate: NARS truth/desire + O(1) ontology lookup, no recompute/lock.
6. Tick snapshot lock-free (epoch/RCU); prune-after-durable; log compaction.
7. Hot cycle alloc-free, dispatch-free, inlined.

## Honest bilanz

**Strong & real:** the unification (one process, no inter-system serialization tax), the
marker-collapse, the cache-isomorphism, the non-request liveness, the borrow-checker-as-
compile-time-concurrency-proof — and a genuine HA path (TiKV). **Costs / not-yet:** the
bounded NARS bag (Gap #1) is the unbuilt keystone of the latency story; the commit merge
is still the LWW relic (Gap #2); zero-copy is really zero-serialization-once-changed; and
"learned" needs Phase 4. **Differentiator is depth/autonomy, not throughput.**
