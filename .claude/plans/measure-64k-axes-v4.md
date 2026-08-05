# measure-64k-axes v4 — the hot version window (operator-directed, 2026-08-05)

> **Reads with:** v2 (rolling epoch closure — COMPOSES, does not supersede),
> v3 (M-arm/O-arm results — this design survives both), and
> `.claude/knowledge/seal-vs-temporal-ordering-information.md` (whose
> properties 2 and 4 carry a dated caveat added by this plan). Nothing here is
> built or measured; the design was **panel-hardened before banking** (one
> canon-conflict sweep + one adversarial refuter, 2026-08-05) and the panel
> INVERTED the initial fork choice — recorded honestly in §4.

## 0. The model, one line

**Decouple the cognition clock from the persistence clock: publish every
sealed cycle to RAM immediately; make durability a batched background
barrier.**

```
think → seal cycle → PUBLISH (RAM, visible now)     ← cognition clock
think → seal cycle → PUBLISH
think → seal cycle → PUBLISH
        ...
background: FLUSH [n..n+K] as one vertical batch    ← persistence clock
            + ONE sync barrier ⇒ durable_head = n+K
```

The batch is **vertical — batching time, not owners**: the unit of the flush
is a run of consecutive sealed cycle images, which for Arrow/Lance is the
natural shape (appending immutable batches). Two watermarks replace one:

- `published_head` — the newest sealed cycle cognition may read (RAM).
- `durable_head` — the newest cycle that survives a crash (disk, advanced
  only at sync barriers).

The crash window is `(durable_head, published_head]`.

**Authorities do not move.** `temporal.rs` stays the chronology authority;
Lance stays the durable authority; the seal stays the ordering authority
(v3's O-arm result + `E-SEAL-AND-TEMPORAL-ARE-DIFFERENT-OBJECTS-1`). The hot
window is a *residency* statement, not a new authority: "versions
`durable_head+1 ..= published_head` are resident in RAM."

**Naming (canon-checked).** This is **the MailboxSoA fleet's hot version
window over sealed cycles** — NOT "VSA speaks Lance." Per
`E-MARKOV-TEMPORAL-STREAM-1` the VSA carrier is demoted to its
I-VSA-IDENTITIES niche and the in-RAM substrate is the `MailboxSoA` owner
fleet behind `MailboxFleet` (`cognitive-shader-driver/src/mailbox_soa.rs:58`,
`lance-graph-supervisor/src/cycle_driver.rs:179`). Any doc that says "VSA
speaks Lance versioning" is resurrecting the deprecated carrier framing;
the sweep confirmed "SoA horizon" has no canon citation either — use the
fleet name.

## 1. Why this beats "another cache layer"

A cache has **invalidation**; the hot window has only **eviction**. Because
sealed cycles are immutable and append-only, the window is not a copy of the
truth kept coherent with the truth — it *is* the head of the log, retained in
RAM past its durability point. No coherence protocol, no staleness, no
invalidation storm. `temporal.at(v)` becomes location-transparent: RAM if
`v > durable_head` and resident, Lance otherwise — the caller never learns
which.

And it aims at exactly what A0 measured: the expensive, **unstable** part of
the pipeline is `filesystem → page cache → writeback → allocator`
interaction — the sync — not sealing and not ownership. Batching K syncs into
one barrier amortizes precisely the phase A0 could not stabilize, while the
per-cycle seal (11–20 ms, stable across every arm) keeps running untouched.

## 2. What survives untouched (the load-bearing list)

- **The seal, per cycle, unchanged.** `freeze` still computes all four
  properties per cycle: cross-owner total order, arrival tie-breaking, the
  per-row fold, the cohort + `base_version` read horizon. The panel verdict:
  properties 1 and 3 are computed inside `freeze` and are not touched;
  property 4's cohort re-anchors on the SEAL event (see §5 caveat); property
  2 gains a durability caveat (see §5).
- **E-64K-1TO1's "ONE deterministic seal boundary per cycle"** — verdict
  COMPATIBLE: the operator order pins the *seal* boundary, not the physical
  append. Only the durability event moves.
- **v3's M-arm and O-arm results** — orthogonal. This design changes the
  durability *cadence*, not the per-cycle ordering (O-arm) and not the
  intra-cycle layout (M-arm). It survives both negative results.
- **v2's rolling epoch closure** — COMPOSES, one level down: v2 rolls chunks
  *within* one cycle toward one epoch manifest; v4 batches sealed *epochs*
  across the durable flush. v2's 200 ms is the intra-cycle chunk-closure
  deadline; v4's 200 ms is the cross-cycle flush deadline. **Two different
  200 ms windows — never conflate them.**

## 3. The fork the panel decided — barrier flush over version multiplexing

Lance mints exactly ONE dataset version per commit, so there were two ways to
flush K cycles:

**(i) Barrier flush — RECOMMENDED.** Keep 1 cycle = 1 Lance version. Each
seal performs its (unsynced) commit immediately — the cycle gets its REAL
Lance version number at publish time — and the background flush is **one sync
barrier** covering versions `durable_head+1 ..= v`. K cheap page-cache
manifest writes, ONE fdatasync. `durable_head` advances only when the barrier
returns.

**(ii) Version multiplexing — REJECTED for now.** K cycles share one Lance
version; `CycleId` becomes the fine clock. The adversarial panel killed the
"the type split pre-anticipated this" framing with citations:

- `temporal.rs` has **no cycle-within-version coordinate** —
  `QueryReference` carries `server_id/ref_version/hlc_tick/mode/rung`, no
  `CycleId`; `classify()` compares whole Lance versions. Under (ii) the
  epistemic horizon coarsens to K and **the no-hindsight guarantee degrades
  to up-to-(K−1) cycles of intra-version hindsight for a Strict reader** — a
  silent semantic break of the module's flagship property
  (`no_hindsight_streamed_known_game`, built on 1 event = 1 version).
- The two in-type candidates for a fine clock both fail by contract:
  `hlc_tick` is a cross-server causal tick (repurposing it IS the "third
  invented numbering" wearing a borrowed name, and it collides with the
  deinterlace sort key); `cast_seq` is owner-local ("Cross-owner values are
  never compared").
- The 1:1 binding is contractual at ≥6 sites (`persist_sink.rs:10-12`,
  `:94-96`, `:309-313`, `:323-327` + the `versions()` ladder falsifier
  `:864-872`; `cycle_driver.rs:103-105`, `:133-136`) plus
  cycle-count-as-version arithmetic in the measure harness. (ii) is a
  retrofit across all of them; (i) touches none.

So: **the type split (`CycleId` ≠ `DatasetVersion`) is a foothold, not an
anticipation** — and (i) does not even need the foothold. (ii) stays on the
shelf as the fallback if barrier flush measures insufficient, with its
hindsight-coarsening cost named as the price of admission.

## 4. The hardened invariants (each one bought by a landed attack)

**H-1 — checkpoint fencing.** `recover_and_apply` requires a THIRD durable
artifact besides RAM and the WAL: the per-owner `(phase, watermark)`
checkpoint (`persist_sink.rs:364-366`, `:379-382`). The naive "cognition and
record die together" claim was REFUTED as stated: a checkpoint cut from
published-but-unflushed state and made durable early leaves, after a crash, a
durable phase ahead of the surviving log (StalePhase corruption) AND a stale
watermark that silently skips legitimately durable later landings. The
repair, now an invariant: **checkpoint state may never be made durable ahead
of `durable_head` — it rides the same barrier or is fenced to it.** With H-1
(and the verified fact that nothing else escapes the RAM+WAL pair before
durability — the refuter's vector (a) found no side-effect leak in current
source), the die-together property holds: the crash loses the window and the
cognition that read it *together*, so restart-at-`durable_head` has no
divergence.

**H-2 — torn-tail cleanup.** Under barrier flush, a crash can leave unsynced
manifests above `durable_head` in torn or reordered states (without per-commit
sync, the filesystem may persist manifest j+1 while j's data pages are still
volatile — the refuter's vector (b)). So `durable_head` is NOT "the newest
manifest found"; it is **the newest fully-intact version at or below the last
barrier**, and recovery MUST determine it (barrier record or integrity probe)
and **remove/tombstone everything above it before the next writer starts** —
otherwise the durable prefix is not contiguous and both the base fence and the
watermark skip-logic silently corrupt.

**H-3 — the window is not a veto window.** A published cycle is irrevocable
from cognition's side the moment a successor chains on it. The flush queue is
append-only; nothing is ever dropped from it. The Libet veto lives PRE-SEAL
(v2 `ClosureState::Vetoed`, write-order registration) — never between publish
and barrier. Anyone "optimizing" by unqueueing a vetoed version corrupts the
chain.

**H-4 — zero-copy conditions (two sentences that keep the design legal).**
The sweep found the ruling that forbids "detached canonical state /
snapshots" verbatim (`E-AN-UNFILLED-SEMANTIC-SLOT-…-1`) and that
`DetachedCycleBatch` is literally documented as a snapshot (with the known
`freeze`-clones debt already on the board). The hot window is legal ONLY
under both of: (1) **the window retains the SINGLE freeze-output allocation
per cycle** — append-only, eviction-only, never re-minted, never a second
copy per version; (2) **the batched Lance append writes FROM those same
retained bytes** — the window IS the in-place backing store of the durable
write, i.e. genuinely the primary allocation, not a sidecar beside one.

**H-5 — reader visibility is rung-decided; there is NO pump
(operator-corrected 2026-08-05).** Two reads exist: *published* (cognition;
may read above `durable_head`) and *durable-only* (audit/compliance rung; a
durable-only read of an unflushed range FORCES the barrier for that range —
the transparent fall-through).

An earlier revision of this invariant said "the kanban pump rebases onto the
publish ack" — **that resurrected a deprecated mechanic and is retracted.**
There is no architectural pump, no acknowledgement-driven progression, and no
scheduler advancing cognition; the ack/pump framing belongs to the historical
compatibility surface only (`E-PROGRESSION-IS-EXISTENCE-NOT-COMMAND-1`,
completing the 2026-07-10 correction chain around
`E-KANBANSTEP-IS-THE-TRIGGER-1`). The authoritative execution path is:

```
think → seal → publish Lance version → next cycle reads the published version
```

The version becoming queryable IS the progression — nothing signals it,
acknowledges it, or schedules it. Durability trails publication
independently. Consequently the hot window is **not a message queue awaiting
acknowledgement**; it is a **resident horizon of immutable Lance versions**:
readers observe versions, writers publish versions, persistence catches up
on its own clock. The decoupling this design delivers needs no rewired
trigger — cycle n+1 reads published cycle n the moment it exists, which is
already the whole mechanism.

Ack/SLA/retry/notification vocabulary may legitimately survive ONLY in
external consumer surfaces (ticket-processing-style workflows) — an
application concern, never substrate mechanics.

## 5. Contract re-wordings this design owes (the sweep's site list)

Not code yet — but the day the build lane opens, these exact sites change
meaning and must change words, or the docs lie:

- `persist_sink.rs:288` "one **durable** append per cycle" → one *publishing*
  append per cycle; durability moves to the barrier op (a NEW trait method —
  no K-cycle/barrier operation exists in `WalSink` today, the refuter
  confirmed the design currently has "no contract home").
- `persist_sink.rs:304-308` "This is the ONLY durable op" → the only
  *publishing* op; the barrier is the durable op.
- `persist_sink.rs:39-43` + `:107`: `base_version` may name a
  published-but-not-yet-durable predecessor — legal under (i) because the
  unsynced commit already minted the real version; the doc must say
  "published predecessor," and `scan_sealed`'s "COMMITTED landings only" must
  name which head it reads against (published vs durable).
- The `wal_writes() == 1` falsifier (`persist_sink.rs:673`) SURVIVES under
  (i) — one append per cycle stays true; what becomes 1-per-K is the *sync*.
  A new falsifier is owed: `sync_barriers == 1` per K cycles, plus the H-2
  torn-tail recovery test.
- `.claude/knowledge/seal-vs-temporal-ordering-information.md` — dated
  caveats added by this plan: property 2's "once it is in the record, it *is*
  the durable fact" holds only at/below `durable_head` (above it, arrival
  order exists only in RAM and a crash loses it unreproducibly — exactly
  PROBE-SEAL-TIE-DENSITY's point); property 4's parenthetical "(one WAL
  append → one DatasetVersion)" re-anchors cohort membership on the SEAL
  event, which under (i) still mints one version per cycle.
- v2's pin "the existing one-cycle/one-version logical contract is
  unchanged" — **SURVIVES under (i)** (that is half the reason (i) won);
  under (ii) it would break, which is recorded on the shelf entry above.

## 6. Sizing and flush policy

- **Byte-budgeted, never count-budgeted.** A full 64k-owner cycle image is
  the 32 MiB canonical frame → 48 hot epochs = 1.5 GiB, 4 = 128 MiB. But a
  bursty consumer (one external event yielding several versions within tens
  of ms) produces KB-scale delta cycles. The window budget is bytes (with a
  count cap as the secondary bound); with `Arc`-rooted structural sharing
  across versions, N hot versions ≠ N full copies.
- **Adaptive flush (the Nagle shape):** barrier when
  `dirty_bytes ≥ budget/2` **or** `dirty_cycles ≥ 16` **or** `200 ms`
  elapsed since the oldest unflushed publish **or** memory pressure **or**
  shutdown **or** a durable-only reader forces a range (H-5). Quiet periods
  flush almost immediately; bursts coalesce into a handful of large
  sequential writes.

## 7. EXP-HOT-WINDOW — pre-registered measurements (none run)

Same discipline as v1–v3: one axis at a time, A0's spread guard inherited,
both directions named in advance.

- **P1 publish latency:** per-cycle publish (seal + unsynced commit) vs A0's
  seal+commit+sync. Expected: publish ≪ A0's commit path because the sync
  leaves the loop. If publish latency does NOT drop, the design's premise
  (the sync is the expensive part *in the loop*) is falsified for this
  workload and the layer is complexity without payoff — a KILL.
- **P2 barrier amortization curve:** K ∈ {1, 4, 16, 48} cycles per barrier;
  report durable throughput AND the barrier's own duration (it should scale
  sub-linearly in K if writeback overlaps; if barrier(K) ≈ K·barrier(1) the
  amortization is fictional — a KILL). WAL-knee instability warning applies
  verbatim: if the barrier phase shows A0's 6× swings, report the spread and
  claim nothing.
- **P3 crash-window cost:** the distribution of `published_head −
  durable_head` (cycles AND bytes) under the §6 policy. This is the price
  tag the operator accepts explicitly, not a number to bury.
- **P4 torn-tail recovery (H-2 falsifier):** kill -9 between commit and
  barrier; restart must (a) find the correct `durable_head`, (b) remove the
  torn tail, (c) replay to a state byte-identical with never-having-published
  the lost window. Both halves: a can-fire case (torn manifest present →
  detected) and a can-stay-silent case (clean barrier → nothing removed).
- **P5 checkpoint fence (H-1 falsifier):** attempt to persist an owner
  checkpoint above `durable_head`; the fence must refuse. Then the positive
  control: checkpoint riding the barrier lands.

## 8. Relationship to the open questions this session banked

- **PROBE-SEAL-TIE-DENSITY** becomes MORE urgent under this design: where
  ties exist, the seal's order derives partly from non-durable arrival — and
  the hot window widens the span in which that order exists ONLY in RAM.
- **ISS-MARM-T1-4X-A0-GAP** must be resolved before P1/P2 are compared
  against A0 numbers (same commensurability rule as v3).
- **TD-LANCE9-LANCEDB036-REMEASURE**: lance 9 / lancedb 0.36 are expected to
  change commit overhead — P1/P2 should be measured before and after that
  upgrade, since the fork's economics ((i)'s K manifest writes) move with it.
