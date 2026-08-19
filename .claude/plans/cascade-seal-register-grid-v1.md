# Cascade-accumulated seal on the register grid — SPEC v1 (5+3 council object)

> **Status: SPEC v1 — council in progress (Phase 1).** This document is the
> Phase-0 spec for the 5+3 council convened 2026-08-19 on the operator's
> `/5plus3` invocation. It is the executable spec for task #25 (cascade seal
> measurement + implementation), hardened here BEFORE the operator ratifies
> PR #968. The council ratifies THE SPEC; code implementation stays gated on
> the #968 ready-flip (the STOP order's discharge condition).
>
> Panel swap declared per the card: lens 3 (code truth) runs on the
> `runtime-archaeologist` card directly (it exists locally).

---

## 1. FROZEN DECISIONS (the council may flag VIOLATES with evidence; never re-open on taste)

1. **The 12 seal requirements** — operator verbatim,
   `docs/lotus/CASCADE-ACCUMULATED-SEAL-SPEC.md` §Requirements (2026-08-19):
   no post-finalization payload rescan · no storage reread · no encryption ·
   no physical-order dependency · digest binds locus+resolved-state+content ·
   petal digest while hot · higher levels reduce child digests only · root
   before Lance publication · one DatasetVersion publishes image+root ·
   DatasetVersion = publication identity / root = content identity ·
   primitive benchmarked ONLY in-architecture · FNV deleted, not optimized.
2. **Register-grid correction** — operator verbatim, same doc §ARCHITECTURAL
   CORRECTION: Lotus holds registers (locus, pointer/descriptor, resolved
   state, phase, tiny digest state), never copied rows/petals/images; petal =
   16 register positions; phase + canonical register position CONSTRUCT
   ordering (no new sort/materialization on the write path); the digest seam
   is the ONE flush-time dereference (`ptr → SoA bytes → Lance serializer AND
   leaf digest in the same read`); identity split ContentRoot / ControlRoot /
   DatasetVersion with batch identity `H(cycle ‖ base_version ‖ ControlRoot ‖
   ContentRoot)`; superseded payload bytes are NOT hashed; cast stays
   content-blind (descriptor purity); MAXIM: "ZERO COPY UNTIL THE MEMBRANE."
3. **§0 STORNO** — `docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md` §0: physical
   iteration order never defines semantic replay order; canonical coordinates
   `(cycle, stream_position)`; compaction = optional storage economics only,
   never semantic repair, never a cognition prerequisite.
4. **T0.3 amendment** — ΔV = W_write − A_last; store coordinates, derive
   distance; wall-clock is economics only; the returned DatasetVersion is
   NEVER a pre-write hash input (which is why `base_version` — the sealed
   read horizon, known pre-write — is the version bound in the identity).
5. **Idempotency verdict** — `docs/lotus/SEAL-FINALIZATION-MAP.md` §4: the
   `(cycle, batch_hash)` reconciliation CAN consume the accumulated root; no
   second hash, no second pass. Operator: "Do not accept 'we need a second
   full pass for idempotency' without a falsifier."
6. **Payload touched exactly TWICE ever** — map §6: production write + the
   one flush dereference. The flush dereference pays for persistence AND
   integrity.
7. **I-LEGACY-API-FEATURE-GATED** (CLAUDE.md iron rule): the batch_hash
   algorithm change carries a version gate on the serialization path; stored
   FNV-era FrameMeta stays readable; no same-name silent semantic change.
8. **Zero-copy law** — `NodeRowPacket` is deliberately not `Clone`/`Copy`
   (operator 2026-07-29: "copies are forbidden, borrows are only for the
   same mailbox"); every SoA envelope is zero-copy creation→Lance tombstone
   (three-tier model, `docs/architecture/soa-three-tier-model.md`).
9. **Archaeology verdict is settled** — map §6: the register/phase substrate
   EXISTS as a declared contract (`NodeRowPacket`, Addendum-6 descriptor)
   and is UNWIRED to persistence. The work is to WIRE it; nothing new is
   created (no payload Morton tree inside seal code).
10. **Falsifiability rule** (CLAUDE.md P0): every guard disable-run
    red-then-green; anti-vacuity + can-stay-silent halves; thresholds get
    inertness tests.
11. **Leaf granularity is measured, not assumed** (operator STOP order: "Do
    not prematurely freeze 512 B as the leaf granularity") — the req-11
    bench measures 512-B-row-leaf vs 16-register-petal-leaf variants before
    the pin.
12. **X-C2-3 ECC is separate** — "Do not let ECC dictate the hot-path
    checksum geometry." Repair/coding arms never shape this design.
13. **Lance family upstream-authoritative** (`E-LANCE-IS-UPSTREAM-
    AUTHORITATIVE-1`): publication remains one durable DatasetVersion via
    the existing lance =9.0.0 pins; no fork, no storage-engine change.
14. **No agents run cargo** (operator, RP-SEAL arc). Savants/reviewers are
    read-only; the orchestrator runs all gates centrally at implementation
    time.

## 2. INPUT INVENTORY (verified 2026-08-19 against working tree @ 7267ec9)

`crates/lance-graph-planner/src/persist_sink.rs`
- `SweepSlot` :171-183 — `payload` today = owned bytes (doc :191 says
  "a descriptor (a `NodeRowPacket` slice in production; bytes here)");
  `stream_position` :183 = the canonical order key.
- `LandedSlot` :205; `FrameMeta` :214-220 (`cycle`, `base_version`,
  `batch_hash: u64`); `CommitOutcome` :227-252 (`NoChange` / `Committed` /
  `Reconciled`, hash-carrying); `CommitError` :258+ (`Fenced` /
  `HashConflict` :289 / `Ambiguous` :316-320 / `Io`).
- `DetachedCycleBatch::freeze` :377-390 — `order_cycle_stably` :378, per-row
  fold to image (row finality), `content_hash` call :383.
- `content_hash` :401-428 — **the FNV serial pass being deleted** (eats
  cycle, base_version, then per-landing stream_position/owner/row/move-tag/
  mailbox/witness_chain_position/payload-len/payload).
- `commit_cycle` :551 (WAL append + fence); `applied_through` watermark
  :642-653; `recover_and_apply` sorts by `(cycle, stream_position)` (T0.1,
  restart-only).

`crates/lance-graph-planner/src/batch_writer.rs`
- Addendum-6 zero-copy ruling :30-33 ("P is a DESCRIPTOR … sink reads them
  through `NodeRowPacket::as_le_bytes` at flush time").
- `BatchWriter<P>` :95; `cast()` :132 — zero production call sites (map §6);
  drain-releases-descriptor doc :177.

`crates/lance-graph-contract/src/canonical_node.rs`
- `NodeRowPacket<'a> { rows: &'a [NodeRow], cycle: u32 }` :1511-1514;
  `SoaEnvelope` impl :1540; `as_le_bytes()` :1553 — the 512-B-strided slab
  view; deliberately not Clone/Copy. Only live callers today: deprecated
  symbiont bridge + tests (map §6).

`crates/lance-graph-supervisor/src/cycle_driver.rs`
- `collect_casts` :357-393 (`stream_position = position_base + cast.0`
  :385); `seal_cycle` :434-437 (retry cache; exactly one WAL write).

`crates/lance-graph/src/graph/cycle_sink.rs`
- `LanceCycleWriter` Arrow schema :114-140; three row kinds :103-123
  (per-cast control rows + coalesced image rows); `payload
  FixedSizeBinary(512)` nullable :139-140; writer makes "one copy of each
  payload" :72 (the Arrow builder append — the membrane copy).

`crates/rp-seal-t0-probe/`
- `src/cascade_seal.rs` — HELD scaffold: `PetalDigest` trait :38-47;
  Blake3/Crc32c/Xxh3 impls :49-140; `CascadeSeal` accumulator with online
  bubble-up :144-240 (fanout parameterized; F-SEAL-NORESCAN /
  F-SEAL-ROOT-LATENCY accounting fields already present).
- `src/lib.rs` — X-C2-1 harness (S1U/S6 schemes, injections I1–I9,
  `run_one`, kill floor, null control); `tests/controls.rs` anti-vacuity
  gate.

Governing docs (frozen): `docs/lotus/CASCADE-ACCUMULATED-SEAL-SPEC.md`,
`docs/lotus/SEAL-FINALIZATION-MAP.md` §1–§6,
`docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md` §0.

## 3. THE PROPOSED RESOLUTION (committed design, in order)

### W1 — Descriptor purity (wire the declared contract)

- `SweepSlot` becomes generic in its payload the same way `BatchWriter<P>`
  already is: production instantiates the Addendum-6 descriptor shape
  `(mailbox, dirty row-range, cycle)` resolving to a `NodeRowPacket` borrow
  of the mailbox's SoA slab; tests keep owned bytes via a trivial impl.
  The bound is one trait, `PayloadSource`, with exactly one method
  (`resolve(&self) -> &[u8]` semantics at the flush seam) — implemented for
  the descriptor (via `NodeRowPacket::as_le_bytes`) and for `Vec<u8>`
  (identity). `freeze` folds REGISTERS (descriptors), not bytes: the
  per-row-finality fold selects the winning descriptor per row without
  dereferencing payloads.
- Cast stays content-blind: no hash, no byte read at `cast()` or in
  `collect_casts` (Frozen 2, 6).
- No new sort: the write path keeps cast-order = canonical order
  (`stream_position = position_base + CastId`); the T0.1 restart sort stays
  restart-only (Frozen 3).

### W2 — Digest rides the ONE flush dereference

- At the flush seam — the single point where payload bytes are read for the
  WAL append / Lance serializer — the SAME read feeds
  `CascadeSeal::resolve_petal`. Leaf digest binds
  `D(canonical locus (cycle, stream_position, owner, row) ‖ resolved/present
  flag ‖ base_version ‖ bytes)`. Unresolved/absent registers are stamped
  UNRESOLVED at finalize (absence is content identity; finalize touches zero
  payload bytes).
- The tree is a digest-register cascade only (fanout 4, the Morton
  2bit×2bit reading; scaffold already parameterizes fanout): higher levels
  reduce child digests, never payload. A petal = 16 register positions +
  resolved mask + digest state — never an 8-KiB buffer.
- **ContentRoot** = the cascade root over final resolved image registers
  (the per-row fold winners — the only payload bytes that persist).
  Superseded intermediate casts' bytes are NOT hashed (Frozen 2): under row
  finality they never reach the flush seam.
- **ControlRoot** = a digest over the durable control stream (per-cast
  coordinates: stream_position, owner, row, move-tag, mailbox,
  witness_chain_position — the non-payload fields the FNV used to eat, which
  persist as cycle_sink control rows). Accumulated online as casts are
  collected; no payload bytes involved.
- **Batch identity**: `batch_hash = fold64(H(cycle ‖ base_version ‖
  ControlRoot ‖ ContentRoot))` — `FrameMeta.batch_hash` stays `u64`
  (schema-stable width); the full roots travel in the commit receipt beside
  the returned DatasetVersion. Root exists BEFORE publication; one
  DatasetVersion publishes image + root (Frozen 1 reqs 8-10). The
  DatasetVersion is never a hash input (Frozen 4).

### W3 — FNV deletion + seal-version gate (I-LEGACY)

- `content_hash` (FNV) is DELETED from the write path (req 12). `FrameMeta`
  gains `seal_version: u8` — encoded frames v0 (absent field ⇒ FNV-era) stay
  readable; new frames mint v1.
- Reconciliation across versions: hash comparison is defined ONLY within the
  same seal_version. A recovery/re-submission meeting a stored v0 frame for
  a cycle it would re-freeze under v1 reconciles **by the fence chain**
  (cycle already durable ⇒ `Reconciled` keyed by cycle + stored identity),
  never `HashConflict` — a cross-version hash mismatch is not evidence of
  content divergence. FNV is never recomputed post-deletion; v0 frames are
  trusted by their stored value + fence-chain position. A falsifier pins
  this (G7 below) and a disable-run proves the gate can fire.

### W4 — req-11 in-architecture benchmark (the primitive decision)

- Wire `cascade_seal.rs` into the probe crate (deps: blake3, crc32c,
  xxhash-rust — probe-crate-only, never the planner) +
  `examples/cascade_seal_bench.rs`: petal-digest-while-hot · reduction cost
  · root-after-last-petal latency · single-petal incremental re-seal ·
  randomized arrival order · contrast row = the deleted whole-cycle FNV.
  Leaf-granularity variants measured: 512-B row-leaf vs 16-register
  petal-leaf (Frozen 11). Isolated hash throughput is inadmissible; the
  primitive is chosen from this bench alone. Wall-clock numbers are
  economics, labeled as such (Frozen 4).

### W5 — implementation sequencing (post-ratification, task #25)

Order: W4 bench (probe-only, zero substrate risk) → primitive pinned → W1
descriptor purity → W2 digest seam → W3 FNV deletion + gate — each wave its
own commit with its falsifiers, board hygiene same-commit, one central gate
run per wave (orchestrator only; Frozen 14).

## 4. NON-GOALS

- **X-C2-3 coding/ECC family** — separate arc; must not shape checksum
  geometry (Frozen 12).
- **Compaction** — economics track only; nothing here depends on it
  (Frozen 3).
- **Strict-sentinel removal / retention-tombstone policy** — Tier-1
  follow-ons, own PRs.
- **Code implementation inside THIS council pass** — the council ratifies
  the spec (extends #968); code starts on the operator's ready-flip.
- **Encryption** — never (req 3).
- **E2 scatter multi-machine sweep** — independent economics probe.

## 5. PRE-REGISTERED GATES (decided now; run centrally at implementation)

- G1 **F-SEAL-ORDER**: any arrival permutation → byte-identical roots.
- G2 **F-SEAL-NORESCAN**: `payload_bytes_digested` == payload bytes written,
  exactly once; finalize touches zero payload bytes.
- G3 **F-SEAL-PRESENCE**: unresolved petal ⇒ different root; erasing a
  resolved digest cannot be root-invisible.
- G4 **F-SEAL-ROOT-LATENCY**: after the last petal only that petal's path
  (log₄ n reduces) runs.
- G5 **F-TWICE**: instrumented dereference count proves payload bytes are
  touched exactly twice ever (production + flush).
- G6 **X-C2-1 on the root**: I3/I4/I5/I6 all move the root; kill floor
  (zero FA on I1–I3) holds; null control clean.
- G7 **Seal-version gate**: v0 frame decodes; same-cycle cross-version
  re-submission → `Reconciled`, never `HashConflict`; disable-run
  red-then-green.
- G8 Existing suites stay green (planner 357+, probe controls), zero
  behavior change before W1 lands; `cargo clippy -- -D warnings` + fmt
  clean per wave; `CARGO_PROFILE_DEV_DEBUG=0` discipline.
- G9 Board hygiene in the same commit per wave (STATUS_BOARD #25 row,
  EPIPHANIES if a finding emerges, AGENT_LOG council entry).

## 6. PER-SAVANT QUESTION SETS

Output contract (all savants): ≤10 findings, each = `(question #, verdict ∈
{CONFIRMS, VIOLATES, GAP, PRIOR-ART-AT, RISK}, file:line evidence, ≤2
sentences)`. No redesigns — a redesign urge files one RISK and stops.
Read-only; NO cargo; do not write any board file.

**S1 prior-art (prior-art-savant):**
1. Does any E-id / knowledge doc / plan already name a cycle-image digest,
   Merkle seal, or content/control root split overlapping W2 (e.g. ndarray
   `merkle_tree`, SPO merkle, medcare audit merkle chain)? PRIOR-ART-AT.
2. Is a version byte / seal-version gate on FrameMeta or WAL frames already
   implemented or specified anywhere?
3. Does a descriptor type matching Addendum-6's `(mailbox, row-range,
   cycle)` already exist by another name (avoid minting a duplicate)?
4. Would the planned board entry duplicate `E-SEAL-IS-ACCUMULATED-ON-THE-
   HOT-PATH-NOT-A-PASS-1` or `E-LOTUS-IS-A-REGISTER-GRID-NOT-A-BYTE-GRID-1`?

**S2 iron rules (iron-rule-savant):**
1. W3's gate vs I-LEGACY-API-FEATURE-GATED: v0 readable, no same-name
   silent change, version gate on the serialization path — YIELDS/VIOLATES?
2. W1/W2 vs the zero-copy law: any copy into Lotus/seal state? Does the
   `PayloadSource` bound preserve borrow-only-same-mailbox?
3. Any NEW hot-path serialization introduced (ADR-022/Firewall analog)?
4. Does folding the wide digest to u64 for `FrameMeta.batch_hash` weaken
   the fence identity relative to FNV64 under the stated non-adversarial
   threat model?
5. Does the cross-version `Reconciled`-by-fence-chain rule hit any AP
   anti-pattern (silent acceptance masking real divergence)?

**S3 code truth (runtime-archaeologist):**
1. Verify every file:line claim in §2 — CODED vs CLAIMED vs ABSENT.
2. Is `SweepSlot.payload` concretely `Vec<u8>` today? List every
   constructor/call site the `PayloadSource` generic would touch.
3. Where EXACTLY (file:line) does the current pipeline read payload bytes
   for the WAL append — is it genuinely once, and is that the seam W2 must
   join?
4. `cycle_sink.rs:72` says the writer makes "one copy of each payload" (the
   Arrow builder append). Does that membrane copy conflict with G2/G5's
   "touched exactly twice" accounting, or is it the same single flush read
   feeding the builder?
5. Where is `FrameMeta` encoded/decoded (the exact seam the `seal_version`
   byte gates)? Is there any second serialization site that could drift?

**S4 cascade impact (cascade-impact-savant):**
1. Full mandatory-pre-merge vs follow-up change list for W1–W4 (files,
   tests, docs, board rows).
2. Which existing tests assert `content_hash` / `batch_hash` values or FNV
   behavior and break on the algorithm change?
3. Who consumes `FrameMeta` / `CommitOutcome::{Committed, Reconciled}` hash
   fields outside persist_sink.rs?
4. What must the probe crate gain for G6 (root-level X-C2-1 arm) and does
   anything in `controls.rs` conflict?
5. Which docs must change in the same commits (CASCADE spec status lines,
   MAP, LATEST_STATE, STATUS_BOARD #25, PR_ARC on merge)?

**S5 different views (creative-explorer-savant):**
1. Strongest alternative reading of "petal = 16 registers" (per-mailbox vs
   per-cycle-image digest state) and its second-order consequence — no
   redesign, one RISK max.
2. Second-order consequence of the ContentRoot/ControlRoot split for
   temporal replay / QueryReference (does content identity enable
   cross-version dedup or replay verification later?).
3. Strongest failure reading of W3's cross-version `Reconciled` rule — is
   reconcile-by-fence-chain too permissive anywhere real?
4. Hidden coupling between the T0.1 restart sort and "phase constructs
   ordering": does RECOVERY need digest-tree awareness, or is the root
   recomputed on replay identical by construction?

## 7. CHANGE LEDGER (v1 → v2 → v3)

- v1: this document (Phase 0).
- v2: (Phase 2 — after the 5.)
- v3: (Phase 4/5 — after the 3 + fixes; ratified.)
