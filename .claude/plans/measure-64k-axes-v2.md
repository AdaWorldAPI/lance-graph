# measure-64k-axes v2 — rolling epoch closure (operator-specified, 2026-08-05)

> **Supersedes v1's EXECUTION MODEL; keeps v1's arms as baseline +
> instrumentation.** v1's global-barrier shape survives as **Stage A0** (the
> comparison baseline) and its CSV/metrics harness is the shared
> instrumentation. The operator's correction: the 64k boundary stays the
> ACCOUNTING and VERSION boundary — it stops being the turnstile where every
> worker, cache line, encryption frame and disk write queues at once. A 64k
> barrier makes every car arrive at one tollbooth and calls the queue
> "amortisation"; the rolling model gives each neighbourhood an on-ramp,
> aligns lanes during the 200 ms closure window, and publishes the city-wide
> state only after every car is through, vetoed, parked, or explicitly
> delayed.

## The model, one line

**64k logical owners → rolling chunk closure → one published epoch.**

```
owner decision
  → provisional write-order registration
  → rolling Morton-ordered chunk append
  → 64k epoch manifest / DatasetVersion publication
```

Three terms, never conflated:

| level | meaning |
|---|---|
| Owner registration | one owner produced, vetoed, held or deferred its result |
| Chunk append | a physically aligned, encrypted segment is WRITTEN but not globally visible |
| Epoch seal | all 65,536 owners accounted; ONE manifest publishes Vn+1 |

**A chunk append is NOT a DatasetVersion. The epoch manifest is.** The
existing one-cycle/one-version logical contract is unchanged; what evolves is
the physical write mechanism: *one logical WAL transaction and one published
version, composed of multiple physical aligned appends.*

## D1 — identity vs write order (DECIDED: keep the 64k owners; chunk baton)

`MailboxId` keeps exactly ONE job: logical identity. Storage locality and
completion order move to a separate key:

```rust
struct WriteOrderKey { morton_chunk: u32, lane: u16, cycle_position: u64 }
```

Order: version outside → Morton chunk inside one version → owner lane inside
one chunk. Recovery identity stays `MailboxId` + monotonic stream position.

**Baton decision: a CHUNK baton, never an owner baton** (65,536 hand-offs =
a railway signal per wheel). The baton lives in the convergence layer;
workers never wait on it while thinking — they publish into assigned result
slots. Encryption may FINISH out of order; the baton makes APPENDS
monotonic: finished-early chunks park in their slot until the baton reaches
them.

## D2 — the Morton cascade (disk-aligned levels; two independent knobs)

```
L0  aligned disk page      4 KiB / 8 KiB / 16 KiB
L1  crypto + WAL segment   1 / 2 / 4 / 8 MiB
L2  logical 64k epoch      32 MiB canonical payload
L3  temporal series        16 epochs in DatasetVersion order
```

Page alignment (L0) and WAL segment size (L1) are INDEPENDENT knobs — a
1 MiB segment is 256×4 KiB = 128×8 KiB = 64×16 KiB pages. **4 KiB is never
an encryption chunk except as the deliberately pathological end** (tags,
nonces, syscall setup dominate). Within an epoch: chunks in Morton order,
owner lanes sorted inside each chunk. Across epochs: version order.

**temporal.rs gains a verified ordered-chunk fast path** —
`local_trajectories_from_ordered_chunks`: validate headers (version
monotonic, chunk sequence monotonic, owner lanes monotonic, stream positions
monotonic) then APPEND — no regroup, no sort. The generic fallback for
arbitrary interleaved history is retained; **generic and fast paths must
produce identical trajectory digests.** Chunks therefore fall out of the
grid cascade already sorted, continuously written, aligned with disk cache
and the 4k/8k/16k device units.

## D3 — the Libet closure window (200 ms as rolling veto budget, never a global sleep)

Per owner: thought finishes → provisional registration → veto deadline →
result REPLACEABLE until deadline → deadline expires → slot immutable.
Per chunk: all owners resolved OR the chunk's 200 ms local deadline reached
→ unresolved owners become Held/Deferred → chunk freezes → encrypt → append.
Earlier chunks close while later chunks still compute — the system waits
only for each chunk's bounded local window, never for all 64k thoughts.

```rust
enum ClosureState { Open, Registered, Vetoed, Held, Deferred, Frozen, Appended }
```

The semantic boundary: **before Frozen the free-will veto may replace or
cancel; after Frozen no mutation — correction is a NEW event in the next
epoch.** Registration is not commitment; encrypted bytes never shapeshift
under their tag. This amortises the free-will veto with write-order
registration instead of a barrier.

## D4 — what "64k complete" means (accounting, not completion)

NOT "all 65,536 thought bodies finished." IT IS: every owner identity has
exactly ONE accounting outcome — committed / vetoed / held / deferred /
absorbed. Only COMMITTED owners enter the sparse transition set (the #879
rule: a version is never permission to advance the whole fleet).

```rust
struct EpochManifest {
    owner_population: u32,  // 65_536
    committed: u32, vetoed: u32, held: u32, deferred: u32, absorbed: u32,
    chunk_count: u32, chunk_hash_root: [u8; 32],
}
// INVARIANT: committed+vetoed+held+deferred+absorbed == 65_536
```

## ⊘ D5 AMENDED (operator sanity-check, 2026-08-05 same day) — crypto is NOT a seal concern; removed from the seal benchmark entirely

**Verified from source before recording:** zero cryptographic operations
exist anywhere in the seal path (`batch_writer.rs`, `persist_sink.rs`,
`cycle_driver.rs` — the one grep hit for "nonce" is the `FnOnce` trait
name). The seal is deterministic ordering + cycle closure + batching +
version publication + one-WAL-append amortisation. **It was never a
cryptographic operation, and the earlier AEAD-in-the-seal framing
conflated two orthogonal layers.**

The corrected split — three independent curves, measured in this order:

```
A  pure seal:            thought → collect → seal → serialize     (no crypto)
B  seal + persistence:   seal → WAL → fsync                       (no crypto)
C  encryption:           evaluated LATER as a SEPARATE LAYER — and only
                         where encryption actually belongs (likely the
                         replication/transport boundary, NOT the seal path)
```

Without this split it is impossible to attribute a bottleneck among
sorting / cache locality / serialization / WAL / encryption / fsync.

Consequences:
- Every Stage-A and Stage-B measurement in this plan is **crypto-free**.
- The former "Stage B encryption arms" are DEFERRED to a future
  layer-placement decision ("where does encryption live?" precedes "what
  does it cost?"). The per-chunk AEAD design below is RETAINED as the
  recorded design for whenever that layer is evaluated — the nonce/AAD
  derivation rule and the crash contract remain correct for that future
  layer.
- The **AEADs-fork dependency decision is no longer blocking anything.**
- The Libet/rolling-closure optimization (D3) is purely a
  synchronization-stall reduction — **it has nothing to do with
  encryption** and the scheduling model is settled crypto-free.
- The crash contract (manifest-less chunks invisible) is an ORDERING
  property and stays in the crypto-free benchmark.

## D5-DEFERRED — encryption as a separate layer (design retained for later)

The expensive shape (collect 32 MiB → serialize → encrypt serially → append
→ sync) is one latency iceberg. Instead: resolved chunk → freeze → encrypt
INDEPENDENTLY on a bounded pool → append continuously in baton order →
final manifest sync. Per-chunk AEAD context with nonce/AAD derived from
**epoch id + dataset base version + chunk sequence + retry generation +
payload length — NEVER from chunk_id alone** (the same chunk number recurs
every epoch). The seal that got unbearably slow at 64k+ was the serial
whole-epoch shape; the rolling shape overlaps CPU encryption with disk
writing while the baton keeps appends deterministic.

**Crash contract:** physical chunks may land before the epoch publishes;
without the final manifest/footer they are an abandoned incomplete epoch —
invisible, reclaimed or ignored by recovery.

**Dependency decision (OPERATOR):** the real AEAD for Stage B comes from
the AdaWorldAPI fork per the P0 forks-only rule (the `AEADs` fork is in
repo scope; wiring = path/git dep decision). Stage B is SEQUENCED BEHIND
this decision; Stages A and C need no new dependency.

## D6 — where grind happens (measured separately; each its own curve)

- **CPU/memory:** 65,536 HashMap lookups; per-owner Vec allocation; clone
  during freeze; O(n log n) stable sort; BTreeMap coalescing; random access
  across 65,536 objects; TLB pressure; page faults; allocator
  fragmentation; NUMA migration.
- **Synchronisation:** shared BatchWriter mutex; global ready queue; one
  atomic per tiny op; false sharing in ready flags; baton spinning; load
  imbalance inside a chunk; one slow owner holding a chunk open.
- **Encryption:** serial AEAD over the epoch; tiny records; nonce/tag
  construction; copies into contiguous buffers; no hardware acceleration;
  one encryption thread feeding many writers.
- **Storage:** too many write syscalls; sync per chunk; unaligned trailing
  writes; page-cache eviction; dirty-page throttling; journal pressure;
  device queue saturation; write amplification.
- **Temporal:** rebuilding a BTreeMap every read; regrouping all 1,048,576
  rows; sorting already-ordered trajectories; cloning rows; reading every
  historical chunk when one owner range suffices.

## D7 — the 16-cycle interpretation (amortise or fall apart)

2 warm-ups + 16 measured epochs; classify the curve:

| shape | signature |
|---|---|
| Warm-up | cycles 1–3 improve, 4–16 flatten |
| Healthy amortisation | initial improvement, stable throughput, bounded baton lag, stable RSS/dirty pages |
| Cache turnover | early plateau; 8–16 modestly slower; LLC misses/faults rise but backlog bounded |
| Collapse | chunk-ready backlog grows per cycle; baton lag grows; epoch time exceeds 200 ms increasingly; encryption/dirty queue never drains |

Per-cycle queue metrics: ready chunks · encrypted-but-waiting chunks ·
appended chunks · max baton lag · deadline-held owners · bytes pending
encryption · dirty bytes · epoch wall time. **The collapse signal is
`backlog_end(n) − backlog_end(n−1)` staying positive after warm-up** —
accumulating debt, not amortising work. Sixteen cycles detect the onset;
they are not an endurance proof.

## Stages (no Cartesian explosion)

- **Stage A — layout, no encryption:** A0 global 64k barrier natural order
  (= v1's shape, the baseline; IN BUILD) · A1 rolling chunks natural order ·
  A2 rolling chunks Morton order — × page alignment {4,8,16 KiB} × WAL
  segment {1,2,4,8 MiB} × 16 cycles. Locates the storage/cache knee.
- **Stage B — seal + persistence (crypto-free, per the D5 amendment):**
  WAL + fsync on the best two Stage-A layouts — isolates storage cost from
  seal cost. (The former encryption arms are DEFERRED to the separate
  encryption-layer evaluation; the AEADs-fork dep decision no longer
  gates anything.)
- **Stage C — temporal recovery after 16 epochs:** generic
  `local_trajectories` · ordered Morton-chunk fast path · single-owner
  range lookup · full-fleet reconstruction · layer-2 epistemic deinterlace.
  Identical digests generic-vs-fast REQUIRED.

## Concurrency governance

**D-KIA-A2 is FROZEN unchanged** — the later publication gate. Operator
override recorded: **EXP-KIA-A2-ROLLING-CLOSURE** (non-claiming,
exploratory; explores the real concurrent topology and chooses the
benchmark shape; cannot pass or fail A2). Shape: owner thought bodies with
per-thread result buffers → write-order registration slots → chunk-local
closure → parallel encryption → ordered baton append → epoch manifest.
**No shared BatchWriter in the thought loop**; the convergence thread
consumes immutable `PreparedIntent { owner, closure, morton_key,
intended_move, payload_ref }` values. Asserts: all 65,536 accounted ·
owner binding preserved · generic and Morton temporal digests identical ·
one published DatasetVersion · incomplete chunks invisible · parallel and
sequential semantic result digests match.

## Sequencing

1. v1 lane lands → central release run → **A0 baseline numbers** (also
   validates the shared instrumentation).
2. Next lane: rolling closure + Morton (A1/A2) + queue metrics + manifest +
   temporal fast path (Stage C machinery) on the v1 harness.
3. Stage B (seal + persistence, crypto-free) on the settled layout.
4. EXP-KIA-A2-ROLLING-CLOSURE last, on the settled layout.
5. Encryption: a SEPARATE later arc, starting with the layer-placement
   decision (replication/transport vs storage), using the retained
   D5-DEFERRED design.

---

## ⊘ v4 cross-note (2026-08-05, append-only)

**v4 (`measure-64k-axes-v4.md`, the hot version window) composes with this
model one level UP — it does not supersede it.** This plan rolls chunks
*within* one cycle toward one epoch manifest; v4 batches sealed *epochs*
across the durable flush (publication clock decoupled from persistence
clock). Two clarifications a future reader needs:

- **The "one-cycle/one-version logical contract is unchanged" pin (§ above)
  SURVIVES under v4's recommended fork** (barrier flush: each seal still
  performs its own unsynced Lance commit → one real `DatasetVersion` per
  cycle; only the fdatasync is batched, 1 per K cycles). It would break
  under the rejected version-multiplexing fork — which is exactly half the
  reason that fork was rejected (v4 §3).
- **Two different 200 ms windows, never conflated:** D3's 200 ms is the
  *intra-cycle chunk-closure* deadline (rolling veto budget); v4's 200 ms
  is the *cross-cycle flush* deadline (Nagle-shaped barrier trigger). The
  Libet veto lives HERE, pre-seal (`ClosureState::Vetoed`) — v4 §H-3 pins
  that a published cycle is irrevocable and its flush queue is never a veto
  surface.
