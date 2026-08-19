# Seal finalization map — source-anchored (2026-08-19)

> Operator-mandated pre-implementation audit (STOP ruling): the data
> ownership/finalization path, the exact FNV site, the bytes FNV uniquely
> binds, the candidate hot-path closure seam, and the OLD-vs-TARGET
> double-touch accounting. **No implementation rides this document.**
> Companion ruling record: `CASCADE-ACCUMULATED-SEAL-SPEC.md` (the 12
> requirements, verbatim).

## 1. The ownership/finalization path, tier by tier

| event | where (source) | when |
|---|---|---|
| Payload bytes PRODUCED (hot) | caller materializes the 512-B witness payload and hands it to `BatchWriter::cast(on_behalf, moves, payload)` — `batch_writer.rs:132-138`; `BatchWriter<P>` is payload-generic, instantiated `BatchWriter<Vec<u8>>` by the driver | during the cycle, per cast |
| Intent visible (ahead-firing) | same `cast()` — intent recorded immediately, `CastId` monotonic = cast order | immediately |
| Landing coordinates minted | `collect_casts` (`cycle_driver.rs:358-393`): `SweepSlot{cycle, stream_position = position_base + CastId, owner, row = row_of(owner), paired_move (≤1/owner), payload}` — payloads MOVED, not copied | cycle close begins |
| **A row becomes FINAL** | `DetachedCycleBatch::freeze` (`persist_sink.rs:377-390`): `order_cycle_stably` by `stream_position`, then the per-row fold `image.insert(s.row, s.payload.clone())` — last stream position wins. Until freeze, any later cast can supersede a row; there is NO earlier row-finality event in source | at freeze |
| **The FNV pass (being deleted)** | `DetachedCycleBatch::content_hash` (`persist_sink.rs:402-428`), called from `freeze` (`:384`) | at freeze, after the fold |
| Retry-cache copy | `seal_cycle` (`cycle_driver.rs:~455`): `let frozen = casts.clone()` — one full extra payload copy held until commit success | at seal |
| Persist-layout copy | `LanceCycleWriter` (`lance-graph/src/graph/cycle_sink.rs`): Arrow `RecordBatch` build — three row kinds (frame / cast / image), `payload FixedSizeBinary(512)` **nullable** | at commit |
| ONE durable append | `Dataset::append` → ONE returned `DatasetVersion` (`CommitOutcome::Committed.version`) | at commit |

**PRESENT vs certified EMPTY, per tier:**
- Resident tier: `MailboxSoA<N>` (`cognitive-shader-driver/src/mailbox_soa.rs`,
  default `N = 1024` rows/mailbox) — `len` vs capacity; rows past `len` are
  explicitly "phantom rows" (`:203`).
- Cycle-image tier: a row is PRESENT iff it is a key of `freeze`'s
  `image: BTreeMap<u64, Vec<u8>>`; EMPTY = absent from the map. There is no
  certified-empty *record* — absence is implicit.
- Persist tier: the Arrow **null bitmap** — only image rows carry a non-null
  `payload`; frame/cast rows are payload-null (`cycle_sink.rs` schema,
  "nullable: only image rows carry it").

**⚠ Honest boundary — the cascade does not exist in the write path yet.**
No `4096→1024→256→64→16→4→1` reduction structure, no 16-row/8-KiB petal
unit, and no 64 K×512 B monolithic grid exists anywhere in the writer
tiers audited (`batch_writer.rs`, `cycle_driver.rs`, `persist_sink.rs`,
`cycle_sink.rs`, `mailbox_soa.rs`). What exists: per-mailbox SoA
(N = 1024), the 512-B witness-row ABI (`EPISODIC_WITNESS_BYTES`), and the
sparse per-cycle image keyed by `row: u64`. The cascade of the TARGET
diagram is therefore the structure the seal *introduces over the cycle's
witness set*, not one it hooks into — the leaf-granularity question
(row = 512 B vs petal = 16 rows = 8 KiB) is genuinely open exactly as the
ruling says, and must be settled by measurement at the seam, not assumed.

## 2. Exactly what FNV binds today (`persist_sink.rs:402-428`)

Once per batch: `frame.cycle` (8 B) ‖ `frame.base_version` (8 B — the
A_last read horizon, deliberately part of the idempotency identity per the
doc comment at `:394-401`). Then per canonical landing (ALL landings,
including same-row intermediates later coalesced away):
`stream_position` (8) ‖ `owner` ‖ `row` (8) ‖ move-tag
(`[1, from, to, exec]` or `[0]`) ‖ if move: `mailbox`,
`witness_chain_position` ‖ `payload.len()` (8) ‖ `payload` (≤512).

**What an image-only cascade root would NOT bind (the FNV-unique bytes):**
1. the frame identity `(cycle, base_version)` — pre-publication
   temporal/generation identity;
2. the full landing SEQUENCE — every landing's canonical coordinates
   (`stream_position, owner, row`), including superseded same-row
   intermediates (which ARE persisted as cast rows and replayed by
   `recover_and_apply`);
3. the kanban control plane — `paired_move` (from/to/exec), `mailbox`,
   `witness_chain_position`;
4. superseded intermediate payload bytes (persist-tier cast rows are
   payload-null, so these bytes are hash-bound today but NOT persisted —
   note: FNV currently binds MORE than the durable artifact carries).

## 3. ⊘ The candidate hot-path closure seam — SEAM A/B WAS THE WRONG QUESTION

> ⊘ Corrected by the register-grid ruling (spec doc, ARCHITECTURAL
> CORRECTION): the seam is NEITHER cast time NOR the freeze fold — both
> would hash bytes at a layer that must stay content-blind (cast) or
> byte-free (freeze freezes REGISTERS/pointers, not bytes). The seam is
> the ONE unavoidable flush-time dereference (`NodeRowPacket::as_le_bytes`
> → Lance serializer), where the same hot read feeds the leaf digest.
> §§3–5 below are retained as the pre-correction reasoning record.

Two real seams exist in source; both accumulate, neither adds a pass:

- **Seam A — cast time** (`BatchWriter::cast`, bytes hottest): digest each
  landing's metadata + payload as it is cast. Cast order IS canonical
  order within a cycle (`stream_position = position_base + CastId`,
  `CastId` monotonic — `cycle_driver.rs:340-347`), so a sequential
  landing-chain accumulator needs no sort and no physical-order
  dependency. Same-row supersede = petal re-digest + path re-bubble.
- **Seam B — freeze's existing per-row fold** (`persist_sink.rs:381-383`):
  the fold already walks every final payload to clone it into the image;
  fusing the petal digest INTO that walk touches bytes at a moment freeze
  already touches them and deletes the separate FNV pass outright. Lower
  disruption, slightly colder bytes than Seam A.

Either way the batch identity becomes a fixed-size header digest computed
from already-accumulated values at freeze:
`H(cycle ‖ base_version ‖ landing_chain_digest ‖ image_root)` — no
payload byte is read by the header step.

## 4. Idempotency audit: can `(cycle, batch_hash)` consume the root?

**YES — with the header binding above; FNV is deleted entirely, one
identity, no second hash.** The reconciliation contract
(`persist_sink.rs` `CommitOutcome::Reconciled`, `cycle_sink.rs`
HashConflict) compares an opaque `u64`-shaped identity for `(cycle, X)`;
nothing in it inspects how X was computed. The landing-chain digest
carries §2 items 2–4 accumulatively (≈40–70 B metadata per landing,
digested when staged — ~4 MiB total at C = 65,536, on the hot path, never
a post-pass); the header carries item 1. **No second full pass is needed
for idempotency — falsifier: payload bytes digested == payload bytes
produced, exactly once, and freeze touches zero payload bytes for
hashing.** The existing semantics re-pin: identical completed sets under
any completion order → identical identity; a re-derived frame →
`Fenced`/`HashConflict` fail-closed, unchanged.

**Receipt design point (ruling §"associate, don't pre-bind"):** the
returned `DatasetVersion` is NEVER a hash input (it doesn't exist
pre-write). The durable receipt (`FrameMeta` today: cycle, base_version,
batch_hash) gains the content root explicitly, so root (content identity)
and DatasetVersion (publication coordinate) associate durably without
either deriving from the other.

## 5. Payload bytes touched, OLD vs TARGET (C casts, D dirty rows, 512 B)

| pass | OLD | TARGET |
|---|---|---|
| produce (hot) | 1× | 1× (digest fused here or at the fold) |
| freeze fold clone | 1× (D rows) | 1× (D rows; petal digest fused if Seam B) |
| **FNV rescan** | **1× (ALL C payloads)** | **0 — deleted** |
| `seal_cycle` retry-cache clone | 1× (all C) | unchanged (separate concern, noted) |
| Arrow persist copy | 1× | 1× (the durable image itself — irreducible) |
| **total passes over payload** | **≈5** | **≈3, of which hashing = 0 extra** |

Leaf granularity (row 512 B vs petal 8 KiB), digest primitive
(CRC32C/BLAKE3/xxh3), and Seam A vs B are decided ONLY by measurement
inside this architecture (`ruling req 11`); the pre-STOP probe scaffold
(`rp-seal-t0-probe/src/cascade_seal.rs`, held uncommitted) becomes that
measurement harness once this map is ratified.


---

## 6. Register-grid archaeology (the mandated critical archaeology, 2026-08-19)

**Verdict: the target register/phase substrate EXISTS in source as an
operator-ruled contract and is NOT yet connected to persistence. Wire to
it; create nothing.**

| question | answer (source) |
|---|---|
| the pointer/registers | `NodeRowPacket<'a>{ rows: &[NodeRow], cycle }` — `lance-graph-contract/src/canonical_node.rs:1511`: a zero-copy `SoaEnvelope` over the SoA backing slab (16\|16\|480, 512-B stride, align 64); deliberately **NOT Clone/Copy** (operator ruling 2026-07-29: "copies are forbidden, borrows are only for the same mailbox") — the borrow cannot escape its mailbox. The inverse zero-copy read (LE bytes → `&[NodeRow]`) exists directly below it. |
| the canonical locus | `NodeGuid` — the 16-B key whose HEEL/HIP/TWIG nibble-interleave IS the Morton register lattice over ADDRESSES (OGAR canon: "x/y nibble-interleave = Morton in centroid space"). Addresses, never payload — the maxim is already the key's own design. |
| phase → order | constructive, twice over: `stream_position = position_base + CastId` (monotonic, `cycle_driver.rs:340-347` — no sort exists or is needed), and the flush contract (batch_writer.rs Addendum-6): the sink reads the LIVE store at flush, so stacked intents coalesce to last-state-wins with the move log carrying ordered history. Order is constructed by position, never repaired by sorting bytes. |
| register resolved | today: a row is final only implicitly (last same-row cast before collect). The register grid's resolved/present state is the piece the wiring ADDS — as register state, not as byte movement. |
| PRESENT vs certified EMPTY | `MailboxSoA<N>` `len` vs capacity ("phantom rows", mailbox_soa.rs:203) + the `NodeGuid` zero-fallback ladder (`is_bootstrap_address` guards); persist tier: the Arrow null bitmap. A certified-empty REGISTER record exists nowhere yet — added at the register tier by the wiring. |
| pointer frozen for one publication | target: freeze freezes REGISTERS (descriptor set + resolved mask), not bytes. Today's `DetachedCycleBatch::freeze` clones bytes — confirmed implementation debt, alongside `BatchWriter<Vec<u8>>` (the module's own doc: *"P is a DESCRIPTOR — (mailbox, dirty row-range, cycle) — never owned delta bytes"*) and `SweepSlot.payload`'s own doc (persist_sink.rs:191: *"a descriptor (a `NodeRowPacket` slice in production; bytes here)"*). |
| `as_le_bytes` on the real path | **nowhere live.** Callers: `crates/symbiont/src/bridge.rs:129` (operator-deprecated crate) + contract tests. The zero-copy flush contract is declared (Addendum-6) with `cast()` at zero production call sites (batch_writer.rs §12 trace, TD-DOC-COMMENTS-CLAIM-UNWIRED-BEHAVIOUR). |

**Corrected pass accounting (supersedes §5's TARGET column):** payload is
touched exactly TWICE ever — production (1) and the flush dereference (2),
where the same read feeds the Lance serializer and the leaf digest.
Hashing adds ZERO passes. Freeze touches zero payload bytes. The
Arrow/Lance serialization at the membrane is inside touch (2), not a
third pass.

**Corrected identity (supersedes §4's landing_chain formulation):**
ContentRoot binds final referenced durable content to canonical loci —
superseded payload bytes are NOT hashed (FNV's over-binding is dropped
deliberately, not preserved). ControlRoot carries the tiny persisted
trajectory/control metadata (the moves/sequence that ARE durable as cast
rows). BatchIdentity = H(cycle ‖ base_version ‖ ControlRoot ‖
ContentRoot); the returned DatasetVersion associates with the root in the
receipt, never as a hash input. The §4 verdict (FNV deleted entirely, one
identity, no second pass) STANDS — only the composition of the identity
is corrected.
