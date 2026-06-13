# SurrealDB kv-lance hardening — VIEW-path durability for the polyglot SurrealQL surface (v1)

> **Status:** PROPOSAL (2026-06-13). Branch: `claude/kv-lance-hardening-plan-v1`.
>
> **Scope clarification up front:** this is hardening for the **VIEW** path, not
> the writer. The 2026-05-28 ruling (handover §2, `E-RUBICON-RACTOR`) and the
> 2026-06-09 `polyglot-container-query-membrane-v1` plan both pin:
> **LanceDB leads as the in-binary production durability store; SurrealDB is a
> view/dialect**. Production cognitive-state commits go via lance-graph direct,
> not through the SurrealQL hop. This plan hardens the **kv-lance backend** so
> the view path is production-grade for external/SurrealQL surfaces — and
> deliberately refuses to chase lance/lancedb capabilities past the
> `lance =7.0.0` / `lancedb =0.30.0` surface we already validate against.

## 1. Why this plan exists

Two findings from the inventory motivate it:

1. **kv-lance is already 18/19 implemented in-tree.** The polyglot plan §2.2
   records: `get/keys/keysr/scan/scanr` + writes + savepoints; MVCC via Lance
   versions; `Timeline` time-travel; background optimizer; SDK
   `Surreal::new::<Lance>(path)`; ~6 k lines of internal tests. So the work is
   not "implement kv-lance" — the `surreal_container` BLOCKED(C/D) marker is
   **stale**. The work is **hardening for production use as a view over the
   leading Lance dataset**, plus closing the named correctness gaps.
2. **`cognitive-substrate.md` (2026-05-26 verification) catalogues two named
   gaps in the path:**
   - **Gap #2 — LWW merge relic** (`flusher.rs:251`):
     `MergeInsertBuilder.when_matched(WhenMatched::UpdateAll)` is a relic from
     the now-idle `CommitGate`. On rare collisions it silently drops the earlier
     contribution. Acceptable for hard-partitioned keys; unsafe otherwise.
   - **Zero-copy is really zero-serialization** as built: WAL serialises via
     CBOR (ciborium, `wal.rs:133`), Lance commit copies into Arrow
     (`schema.rs build_write_batch.collect()`). Honest term + a path to actual
     zero-serialization with raw `#[repr(C)]`-LE + checksum.

## 2. Where kv-lance fits — the four-tier write/commit map

This plan supersedes my own earlier two-tier framing of the conversation
(Lance-only vs SurrealDB-on-RocksDB as peers). The corrected mapping is
**four tiers**, scoped by use case:

| Tier | Use case | Commit path | Status |
|---|---|---|---|
| **A** — In-binary production cognitive-state | hot path, same process | **lance-graph direct** (SoaEnvelope → Lance writers, this is what #487's ORM doctrine actually targets) | shipping per `witness_tombstone.rs` scaffold + lance-graph direct paths |
| **B-view** — External / multi-tenant / SurrealQL surface | view + dialect over leading Lance dataset | **SurrealDB on `kv-lance`** (hardened per **this plan**) | partial — 18/19 + 6 k tests internal; hardening below |
| **B-HA** — Distributed scale-out / Raft | HA cluster | **SurrealDB on `kv-tikv`** (Raft + RocksDB) | available; separate plan when HA enters scope |
| **B-fallback** — Operational-guarantees fallback | if kv-lance hardening hits an irreducible blocker | **SurrealDB on `kv-rocksdb`** (production-tested SurrealDB-native) | always available |

**Key invariant:** Tier A is the writer. Tier B is the view. They share the
underlying Lance dataset format (same `SoaEnvelope` geometry, same column
layout); the SurrealDB layer reads through the same Lance versions Tier A
writes. The "different carrier on purpose for clean separation" intent
articulated in `bindspace-singleton-to-mailbox-soa-v1.md` §7 is preserved by
the *tier scoping*, not by re-routing the production writer through SurrealDB.

## 3. Deliverables

### D-KVL-1 — Lance/lancedb version pin discipline [process]

Pin and assert `lance =7.0.0`, `lancedb =0.30.0` across the workspace (already
in `surreal_container/Cargo.toml`) **and** through the surrealdb fork's
kv-lance feature. Document the rationale: in-binary same-process consumer,
no external pressure to chase upstream, fewer breaking-change surfaces. Refuse
features released past these versions in the hardening work below; revisit
deliberately on a yearly cadence.

**Acceptance:** Cargo.toml assertions in `lance-graph`, `surreal_container`,
and the surrealdb fork's `core/Cargo.toml` (kv-lance feature). CI check that
committed version strings match the pin.

### D-KVL-2 — Gap #2: replace the LWW merge relic [semantics, P0]

`MergeInsertBuilder.when_matched(WhenMatched::UpdateAll)` (`flusher.rs:251`)
silently drops a contribution on key collision. For the view path, fix one of:

- **(a) Hard key-partitioning proof.** Encode the BindSpace prefix-sharding
  invariant in a property test; prove that two writes within a batch cannot
  collide. Adopt `MergeInsert` as a correct primitive *given* the partitioning.
- **(b) MergeMode-tagged callers.** Callers tag writes with explicit
  `MergeMode` (`Bundle` → read-bundle-write at the flusher; `Xor` → MergeInsert
  with partitioning assertion). Recommended: this preserves the holograph-
  superposition merge for any future write path that ingests external updates.

**Acceptance:** collision test (two concurrent writes, same key, distinct
contributions) — neither contribution silently dropped under the chosen merge
strategy. Documented partitioning proof if (a) is chosen.

### D-KVL-3 — WAL format: zero-serialization, not zero-CBOR [perf, correctness]

Replace ciborium-CBOR serialisation in `wal.rs:133` with raw `#[repr(C)]`-LE +
CRC32C. Decode path becomes memcpy from mmap (no parse); encode path stays
write-then-fsync. Add an explicit WAL-version byte so v0 (CBOR) and v1
(zero-serialisation) decoders are distinguishable and v0 files are explicitly
rejected by v1 readers without aliased-bit corruption (per `I-LEGACY-API-FEATURE-GATED`).

**Acceptance:** round-trip property test; benchmark improvement vs CBOR
baseline (rough target: 2-3× WAL throughput).

### D-KVL-4 — Crash recovery suite (consumer-side) [correctness]

The fork's ~6 k internal tests are SurrealDB-internal. Add lance-graph-consumer
crash-injection coverage:

- Kill during WAL fsync (partial trailing record at WAL tail)
- Kill during memtable flush (memtable rows not yet in Lance, WAL has source)
- Kill during Lance commit (memtable flushed, Lance version not committed)
- Kill during Lance version garbage-collect (timeline branch outlives the kill)

**Acceptance:** crash-injection suite that kills at each phase and verifies
all-or-nothing recovery against expected post-restart state. Gated on D-KVL-3
(WAL format) so the recovery tests target the v1 format from day one.

### D-KVL-5 — Concurrency / soak [correctness]

Multi-writer + multi-reader 1-hour minimum soak under realistic write mix.
Probes:

- LWW silent drops (should be eliminated by D-KVL-2)
- WAL/memtable race during compaction
- Memtable rotation deadlocks
- Timeline branch-point isolation

**Acceptance:** clean 1-hour soak; failure budget < 0.01 % under the chosen
merge strategy from D-KVL-2.

### D-KVL-6 — Schema evolution surface (lance 7.0.0 limits) [contract]

Document the supported subset under the pin: add-column (Lance versioned add);
drop-column = NOT supported under append-only versioning; rename = via
versioned schema swap; type-change = via column-evolution if Lance 7.0.0
exposes it, otherwise NOT supported. Tests at the supported boundary;
explicit rejection (not silent corruption) at the unsupported boundary.

**Acceptance:** doc + boundary tests.

### D-KVL-7 — surreal_container unblock (polyglot D-PG-6) [scope]

The polyglot plan's D-PG-6 ("optional: `surreal_container` unblock → Rubicon
kanban VIEW over leading LanceDB; ruling-compliant, off critical path")
becomes this plan's last deliverable. Replace the `BLOCKED(C/D)` placeholders
in `crates/surreal_container/{Cargo.toml,src/lib.rs}` with the now-known fork
coordinates + `kv-lance` feature flag, wire a minimal kanban-view smoke test.

**Acceptance:** `surreal_container` compiles against the pinned versions;
smoke test opens a SurrealStore over an existing Lance dataset and reads back
a row written by the Tier-A direct-Lance path. *This is the integration test
that proves the Tier-A writer ↔ Tier-B view contract holds.*

### D-KVL-8 — Doctrine record: tier-scoped ORM, kv-lance is the view [boards, P0]

Per `Mandatory Board-Hygiene Rule`, prepend:

- **`EPIPHANIES.md`** new entry `E-KVLANCE-VIEW-NOT-WRITER` (proposed name):

  > Outer boundary is ORM. Schema = OGAR. Storage choice is tier-scoped:
  > Tier A in-binary cognitive-state commit = lance-graph direct;
  > Tier B-view external/SurrealQL surface = SurrealDB on kv-lance
  > (hardened per `surrealdb-kv-lance-hardening-v1`, pinned to lance =7.0.0 /
  > lancedb =0.30.0); Tier B-HA = SurrealDB on kv-tikv; Tier B-fallback =
  > SurrealDB on kv-rocksdb. The #487 ORM doctrine
  > (`E-OUTER-BOUNDARY-IS-ORM-1`) is scoped to Tier A; this entry extends it
  > over the full tier map. Hand-rolled SoA→Lance is fine **as the Tier-A
  > writer** because it is in-process and we control the version surface; it
  > is **not** a substitute for a real database at any external surface, where
  > a production-tested SurrealDB-native backend (kv-rocksdb / kv-tikv /
  > hardened kv-lance) carries the durability guarantees.

- **`TECH_DEBT.md`** new debt entry pointing Gap #2 (`flusher.rs:251` LWW
  relic) → D-KVL-2 as the paid-by deliverable.
- **`INTEGRATION_PLANS.md`** prepend entry pointing here (this commit).
- **`STATUS_BOARD.md`** D-KVL-{1..8} rows.

**Acceptance:** four board files updated in the same commit as this plan (per
Board-Hygiene Rule). The EPIPHANIES wording lands only after user review of
the exact phrasing (the rest of the entries are mechanical).

## 4. Out of scope

- **kv-rocksdb hardening** — SurrealDB upstream already tests it. We adopt it
  via the existing SurrealDB-native integration; no work owned here.
- **kv-tikv / HA path** — separate plan when HA enters scope. D-KVL-3 (WAL
  format) may need re-evaluation under Raft.
- **lance-graph Tier-A writer improvements** — separate work; this plan is
  the kv-lance side.
- **Chasing lance/lancedb upstream past 7.0.0 / 0.30.0** — deliberate non-goal
  per D-KVL-1.

## 5. Sequencing

1. **D-KVL-1** (version pin discipline) — process gate, fast.
2. **D-KVL-8** (doctrine + board hygiene) — lands first to scope-shield the
   rest. Pending user review of the EPIPHANIES wording.
3. **D-KVL-2** (Gap #2 merge semantics) — biggest correctness issue.
4. **D-KVL-3** (WAL zero-serialisation) — biggest perf win; gates D-KVL-4.
5. **D-KVL-4** (crash recovery suite) — depends on D-KVL-3 WAL format.
6. **D-KVL-5** (soak), **D-KVL-6** (schema evolution surface) — parallel
   after D-KVL-2/3.
7. **D-KVL-7** (surreal_container unblock; polyglot D-PG-6) — last; off
   critical path per polyglot.

## 6. Risks

- **R1.** D-KVL-2 option (b) — MergeMode-tagged callers — requires a
  callable-side change that the fork's existing 6 k tests may not catch.
  *Mitigation:* D-KVL-5 soak under realistic write mix.
- **R2.** D-KVL-3 raw-LE breaks existing WAL files on disk.
  *Mitigation:* WAL version byte gates v0 rejection; migration tool authored
  separately, not in this plan.
- **R3.** Lance 7.0.0 schema-evolution surface (D-KVL-6) may be more limited
  than expected. *Mitigation:* discovery phase before D-KVL-6 implementation;
  doc the actual surface, then test at the boundary.
- **R4.** A future external Tier-B-view consumer might want write access (not
  only read). *Mitigation:* D-KVL-2 option (b) keeps the superposition merge
  available; the partitioning proof option (a) does NOT, so prefer (b) unless
  partitioning is provably hard.

## 7. References

- `.claude/handovers/2026-05-28-1200-pr-418-419-surreal-mailbox-baton-plan-map.md` §2
  (`E-RUBICON-RACTOR` ruling: LanceDB leads, SurrealDB is a view)
- `.claude/plans/polyglot-container-query-membrane-v1.md` (2026-06-09;
  inventory of kv-lance 18/19 status + D-PG-6 surreal_container unblock)
- `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md` §7 (two-egress-mode
  rule: external REST/sea-orm SQL vs internal SurrealDB → LanceDB | RocksDB)
- `.claude/surreal/cognitive-substrate.md` (2026-05-26 verification: Gap #2 +
  zero-copy honesty correction)
- `.claude/surreal/RECONCILIATION_with_canonical_plan.md` (the parallel
  derivation that folds into the canonical SoA plan)
- AdaWorldAPI/surrealdb fork: `core/src/kvs/lance/{wal,memtable,flusher,mod}.rs`,
  `core/src/kvs/api.rs` (`Transactable` contract)
- `crates/surreal_container/{Cargo.toml,src/lib.rs}` (BLOCKED markers; stale
  per polyglot §2.3, refreshed by D-KVL-7)
- PR #477 (three-tier model, merged 2026-06-07) and PR #487 (tombstone
  commit, open) — establish `E-OUTER-BOUNDARY-IS-ORM-1` that D-KVL-8 extends.
