# Cascade-accumulated seal on the register grid — DRAFT v2 (5+3 council object)

> **Status: DRAFT v2 — Phase 2 consolidation complete; Phase 3 (the 3
> reviewers) sees THIS document only.** v1 was the Phase-0 spec; the 5
> savants (S1 prior-art/Opus, S2 iron-rules, S3 code-truth, S4
> cascade-impact, S5 different-views — 45 findings, 0 frozen-decision
> re-opens) produced the change ledger in §8. Raw savant output is banked
> in the session scratchpad and is not part of this document.
>
> The council ratifies THE SPEC; code implementation stays gated on the
> operator's #968 ready-flip (the STOP order's discharge condition).

---

## 1. FROZEN DECISIONS (unchanged from v1; the council verifies, never re-litigates)

1. **The 12 seal requirements** — operator verbatim,
   `docs/lotus/CASCADE-ACCUMULATED-SEAL-SPEC.md` §Requirements (2026-08-19):
   no post-finalization payload rescan · no storage reread · no encryption ·
   no physical-order dependency · digest binds locus+resolved-state+content ·
   petal digest while hot · higher levels reduce child digests only · root
   before Lance publication · one DatasetVersion publishes image+root ·
   DatasetVersion = publication identity / root = content identity ·
   primitive benchmarked ONLY in-architecture · FNV deleted, not optimized.
   *(Req 2's own verification clause: "no storage reread of anything the
   read did not already load" — load-bearing for §3 W3 below.)*
2. **Register-grid correction** — operator verbatim, same doc
   §ARCHITECTURAL CORRECTION: Lotus holds registers, never copied
   rows/petals/images; petal = 16 register positions; phase + canonical
   register position CONSTRUCT ordering; the digest seam is the ONE
   flush-time dereference; identity split ContentRoot / ControlRoot /
   DatasetVersion with `H(cycle ‖ base_version ‖ ControlRoot ‖
   ContentRoot)`; superseded payload bytes NOT hashed; cast content-blind;
   "ZERO COPY UNTIL THE MEMBRANE."
3. **§0 STORNO** — `docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md` §0:
   canonical replay coordinates; compaction = optional economics only.
4. **T0.3 amendment** — ΔV coordinates; wall-clock = economics; the
   returned DatasetVersion is NEVER a pre-write hash input. Shipped
   structural precedent: `AuditMerkleRoot::chain` whose canonical input
   excludes its own output (`crates/lance-graph-callcenter/src/
   unified_audit.rs:86-101,161-174`) — cited, not re-derived (S1-2).
5. **Idempotency verdict** — `docs/lotus/SEAL-FINALIZATION-MAP.md` §4: the
   `(cycle, batch_hash)` reconciliation CAN consume the accumulated root;
   no second hash, no second full pass.
6. **Payload touched exactly TWICE ever** — map §6 — **as the TARGET
   state.** Measured current state is FOUR touches (production +
   freeze-clone + FNV-read + flush-append; S3-5/S3-7): W1+W2 exist to
   collapse 4 → 2. G5 verifies the target, and the OLD count is the
   contrast row.
7. **I-LEGACY-API-FEATURE-GATED** — version gate on the serialization
   path; stored FNV-era frames readable; no same-name silent change.
8. **Zero-copy law** — `NodeRowPacket` not Clone/Copy; envelopes zero-copy
   creation→tombstone. The descriptor is a small copyable ADDRESS, never a
   packet handle (S2-4).
9. **Archaeology verdict settled** — wire the declared contract; create
   nothing new. Sharpened by S1-9: the descriptor TYPE already exists —
   see §3 W1.
10. **Falsifiability rule** — disable-runs red-then-green; anti-vacuity +
    can-stay-silent; threshold inertness.
11. **Leaf granularity measured, not assumed** — the W4 bench measures the
    granularity AND indexing-mode knobs before the pin.
12. **X-C2-3 ECC separate** — never shapes checksum geometry.
13. **Lance family upstream-authoritative** — publication via existing
    lance =9.0.0 pins.
14. **No agents run cargo**; orchestrator runs all gates centrally.

## 2. INPUT INVENTORY (v2 — corrected per S3, extended per S1)

`crates/lance-graph-planner/src/persist_sink.rs`
- `SweepSlot` :171-194 — **no generic parameter today**; `pub payload:
  Vec<u8>` :193; descriptor doc :191-193; `stream_position` :183 with the
  per-owner cross-cycle monotonicity contract :176-182 (**the write-order/
  restart-order agreement rests on this contract** — S5-7; cross-referenced
  here deliberately).
- `LandedSlot` :205; `FrameMeta` :214-220 (no version field; in THIS file
  it is only a trait-returned timeline row — the real serialization seam is
  in cycle_sink, below); `CommitOutcome` :227-252; `CommitError` :258+
  (`HashConflict` **:267**, Display :300 — v1's ":289" was misattributed;
  :289 is `Ambiguous`'s own `batch_hash` field; `Ambiguous` :287, Display
  :314-320).
- `DetachedCycleBatch::freeze` :377-390 — **today clones every payload into
  the owned image** (`image.insert(s.row, s.payload.clone())` :381) and
  `content_hash` :401-429 separately iterates every payload byte (FNV,
  invoked :383) — the two pre-flush touches W1/W2 delete.
- `commit_cycle` :551; watermark :642-653; `recover_and_apply` :677-729
  (restart-only sort; **never touches any digest** — the root is a
  write-once artifact, S5-8).
- Reconcile falsifiers in-tree: `randomized_completion_order_yields_the_
  same_batch_hash` :1943 (post-mint order only — weaker evidence once the
  in-tree G1 pair lands; re-verify, no rewrite; S4-3).

`crates/lance-graph-planner/src/batch_writer.rs`
- Addendum-6 ruling :30-33; `BatchWriter<P>` :95; `cast()` :132 (zero
  production call sites); drain doc :177. **The module doc's "(mailbox,
  dirty row-range, cycle)" phrasing is contradicted by every shipped
  descriptor implementation** — see §3 W1 (S1-9).

Shipped descriptor prior art (S1-9): `RowSpanDescriptor{row_lo, row_hi,
cycle}` at `examples/blw_rows.rs:521-538`, `examples/blw_tenant.rs:370`,
`examples/blw_fusion.rs:370`, `tests/probe_ignition.rs:298`,
`tests/d_ign_b_lenses.rs:403`; `DirtyRange{first_row, rows, cycle}` at
`cognitive-shader-driver/src/mailbox_soa.rs:1779-1786`. **All deliberately
mailbox-less** — "Ownership rides the cast pairing
(`BatchWriter::on_behalf_of`), never the DTO — the write-on-behalf iron
rule."

`crates/lance-graph-contract/src/canonical_node.rs`
- `NodeRowPacket` :1511-1514; `SoaEnvelope` impl :1540; `as_le_bytes`
  :1553; not Clone/Copy (:1492-1510 region).

`crates/lance-graph-contract/src/soa_envelope.rs`
- **The version-gate house shape** (S1-6): `ENVELOPE_LAYOUT_VERSION: u8 =
  2`, trait const `LAYOUT_VERSION`, mismatch error variant, reader-must-
  refuse rule (:41-54, :122, :224-226). `seal_version` mirrors THIS.

`crates/lance-graph-supervisor/src/cycle_driver.rs`
- `collect_casts` :357-393 (`stream_position = position_base + cast.0`
  :385 — **the ONE production SweepSlot mint**, from a drained
  `BatchWriter<Vec<u8>>`); `seal_cycle` :434-437.
- `CommitCycleOutcome` → `SealRecovery` mapping :132-150, :244, :463-471 —
  `HashConflict → Escalate` is unconditionally fail-closed today (S4-5).
- In-tree G1 falsifier pair (S4-2): defect-pin
  `f_ord_real_defect_pin_arrival_order_leaks_into_batch_hash` :2554
  (green today, self-documents "delete this pin when the fix lands") +
  `#[ignore]`d `f_ord_real_publication_identity_is_arrival_order_
  independent` :2604 — **deleted / un-ignored in the SAME commit as
  W2/W3.**

`crates/lance-graph/src/graph/cycle_sink.rs`
- **The real serialization seam** (S3-8): schema :114-140 (no version
  column today); `build_batch` :458-556 (`push_common` :474 writes
  cycle/base_version/batch_hash; Arrow append of image payload :532-534 —
  the flush touch); decode sites `find_frame` :560-596 (**projects
  `["batch_hash"]` only** :577) and `timeline` :1061-1101 (materializes
  `FrameMeta` :1093-1097). Reconcile comparison `stored_hash ==
  batch.batch_hash` :792-793, else `HashConflict` ~:804-809 — **zero
  version awareness** (S2-2).
- Second production SweepSlot mint: `scan_sealed` rebuild :1045
  (`payload: Vec::new()`).
- `batch_hash` persists as Arrow `UInt64` (:129) — the durable consumer
  the gate must reach (S4-4).
- `FrameMeta` construction sites for the field addition (S4-1, 9 total):
  cycle_sink.rs:1093, persist_sink.rs:937, cycle_driver.rs:1224 + 6
  test/example mocks (probe_ignition.rs:445, probe_ignition_64k.rs:229,
  d_ign_b_lenses.rs:537, measure_wal_curve.rs:1525, blw_tenant.rs:523,
  blw_fusion.rs:497). All mocks, no second real serialization site (S3-8).

`crates/rp-seal-t0-probe/`
- `cascade_seal.rs` — held scaffold (zero external callers, S3-6);
  `PetalDigest` :38-47; impls :49-140 (CRC32C width caveat :81-84);
  `CascadeSeal` :144-240.
- `lib.rs` `Scheme` :225-230 is per-locus — **structurally unable to carry
  the root-level arm** (S4-6); G6 gets a standalone harness.

Knowledge docs (v2 addition, S1-4): `.claude/knowledge/
seal-vs-temporal-ordering-information.md` — READ-BY names
`persist_sink::{freeze, order_cycle_stably, DetachedCycleBatch}`; carries
the tie-density finding governing G1's scope (S1-5) and the
pre-registered, still-unrun **PROBE-SEAL-TIE-DENSITY**.

Prior-art contrast set (S1-1/S1-3, cross-referenced, never adopted):
ndarray `hpc/merkle_tree.rs:22-31,62-78` + `hpc/seal.rs:21-52` (the
falsified S1U locus-unbound shape); `graph/spo/merkle.rs:4-12`
(`verify_lineage` checks structure without re-hashing — the cautionary
shape any W2 verifier must NOT repeat).

Governing docs (frozen): `docs/lotus/CASCADE-ACCUMULATED-SEAL-SPEC.md`,
`docs/lotus/SEAL-FINALIZATION-MAP.md`,
`docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md` §0.

## 3. THE PROPOSED RESOLUTION (v2 — committed design, in order)

### W1 — Descriptor purity (wire the declared contract)

- **Reuse the shipped mailbox-less descriptor shape** (v1's "(mailbox,
  row-range, cycle)" is WITHDRAWN — it would have re-added a field
  deliberately removed at six sites; S1-9). The production payload type is
  the `RowSpanDescriptor{row_lo, row_hi, cycle}` shape promoted from
  example/test code into the planner; ownership rides the cast pairing
  (`BatchWriter::on_behalf_of` → `SweepSlot.owner`), never the DTO. The
  descriptor resolves against the owner's SoA slab (identified by
  `SweepSlot.owner`) through `NodeRowPacket::as_le_bytes` at the flush
  seam only.
- `SweepSlot` gains a **new** type parameter with default —
  `SweepSlot<P = Vec<u8>>` (it has none today, S3-3) — so the ~28
  test/example construction sites compile unchanged; exactly 2 production
  sites move: `collect_casts` (cycle_driver.rs:383, mints the descriptor)
  and `scan_sealed` (cycle_sink.rs:1045, read-back rebuild). The bound is
  one trait, `PayloadSource` (new; nothing by this name exists — S3-3),
  resolving to `&[u8]` at the flush seam; `Vec<u8>` gets the identity
  impl.
- **`freeze` stops dereferencing payloads entirely** (S2-3/S3-5): the
  image becomes descriptor-keyed registers (`BTreeMap<row, P>` — the
  per-row-finality fold selects the winning DESCRIPTOR per row); the
  `payload.clone()` at :381 and the FNV byte-walk at :383/:401-429 are
  both deleted. After W1+W2 the only payload dereference anywhere in the
  pipeline is the flush read.
- Cast stays content-blind; no new sort on the write path; the T0.1
  restart sort stays restart-only. The write-order/restart-order agreement
  rests on the `stream_position` monotonicity contract
  (persist_sink.rs:176-182) — now cross-referenced from this spec (S5-7).

### W2 — Digest rides the ONE flush dereference

- At the flush seam (`build_batch`'s payload append, cycle_sink.rs:532-534
  — the same read that feeds the Arrow builder) the bytes ALSO feed
  `resolve_petal`. Leaf digest binds
  `D(cycle ‖ stream_position ‖ owner ‖ row ‖ resolved-flag ‖ base_version
  ‖ bytes)`.
- **Tree shape committed** (S5-1/S5-2): ONE per-cycle tree over the
  cycle's final image registers, **leaf index = rank of the global row id
  in ascending row order** — a pure function of durable coordinates, so
  cross-mailbox tree-position assignment is stable under ANY owner
  interleaving by construction (G1 gains the owner-interleave permutation
  arm). Presence/absence identity is carried by the locus-bearing leaf
  digests + the leaf count (erasing a row shifts ranks and count → root
  moves; F-SEAL-PRESENCE tests exactly this). The alternative fixed-4096-
  register-grid indexing (unresolved slots stamped UNRESOLVED at finalize)
  is measured in W4 under the same falsifiers — the indexing-mode pick is
  a W4 output beside the primitive pick (Frozen 11).
- **ContentRoot** = the cascade root over final image registers only
  (superseded intermediate bytes never reach the flush seam under row
  finality — not hashed, Frozen 2). **ControlRoot** = an online digest
  over the durable control stream (per-cast stream_position / owner / row
  / move-tag / mailbox / witness_chain_position — the fields FNV ate,
  which persist as cycle_sink control rows); accumulated at collect-time
  over coordinates only, zero payload bytes.
- **Batch identity**: `batch_hash = fold64(H(cycle ‖ base_version ‖
  ControlRoot ‖ ContentRoot))`. **Definitions committed** (S2-6):
  `H` = BLAKE3-256 over the fixed ~50-byte composition — a once-per-cycle
  operation over tiny fixed input, NOT the per-petal hot path req 11's
  bench governs (the bench still reports its cost for completeness);
  `fold64` = the first 8 bytes of `H`'s output read little-endian.
  **Identity floor** (S2-7): the identity path requires ≥64 bits of
  content entropy end-to-end, so the winning petal primitive's ROOT must
  be ≥8 bytes wide; `CRC32C/4` competes on petal-cost numbers but cannot
  be the identity root unless the bench also measures a widened-root
  pairing — the floor is a pre-registered constraint on W4's decision,
  not a post-hoc veto.
- FrameMeta.batch_hash stays `u64` (schema-stable width); full roots
  travel in the commit receipt beside the returned DatasetVersion. Root
  exists BEFORE publication; DatasetVersion never a hash input (Frozen 4,
  with the AuditMerkleRoot precedent cited).

### W3 — FNV deletion + seal-version gate (I-LEGACY, on the REAL seam)

- `content_hash` (FNV) deleted from the write path (req 12). The version
  gate lands on the ACTUAL serialization seam (S1-8/S2-1/S3-9): a new
  nullable Arrow column `seal_version` in `cycle_store_schema`
  (cycle_sink.rs:114-140) — absent/null ⇒ v0 (FNV-era), new frames write
  v1 — mirroring the shipped `ENVELOPE_LAYOUT_VERSION` shape
  (soa_envelope.rs:41-54,224-226; reader-must-refuse for versions it does
  not know). `FrameMeta` gains `seal_version: u8` (9 construction sites
  updated, S4-1); `find_frame`'s projection widens from `["batch_hash"]`
  to include the version column (S2-1); `timeline` materializes it.
- **The reconcile comparison is rewritten version-aware** (S2-2 — leaving
  cycle_sink.rs:792-793 as-is would send legitimate cross-version
  reconciles to `HashConflict`, the exact I-LEGACY failure shape). Rule,
  tightened per S5-5/S5-6/S2-8 (v1's reconcile-by-fence-chain-alone is
  WITHDRAWN as a silent-accept hole):
  - Same seal_version → hash comparison exactly as today.
  - **Cross-version → recompute-and-compare, restart-path-only**: the
    recovery scan already loads the cycle's durable sealed rows
    (`scan_sealed`); the v1 identity is recomputed over those
    already-loaded rows and compared against the re-submitted batch's v1
    hash. Match → `Reconciled` (same content under a new algorithm);
    mismatch → `HashConflict` (genuine divergence **fails closed**, and
    the existing `HashConflict → Escalate` recovery mapping
    (cycle_driver.rs:463-471) keeps its fail-closed meaning unchanged —
    S4-5). This satisfies req 2's own verification clause ("no storage
    reread of anything the read did not already load") because recovery
    already loads exactly these rows, and it avoids the spo-merkle
    cautionary shape (coordinates checked without re-reducing digests,
    S1-3). FNV is never recomputed; v0 stored values are never
    re-derived.
- Same commit: delete the defect-pin test (cycle_driver.rs:2554) and
  un-ignore its arrival-order-independence sibling (:2604) — the in-tree
  G1 falsifier (S4-2); LATEST_STATE prepend correcting the now-stale
  F-ORD narration (S4-8); CASCADE spec/MAP status lines flip (S4-9).

### W4 — req-11 in-architecture benchmark (the measured decisions)

- Wire `cascade_seal.rs` (deps blake3/crc32c/xxhash-rust, probe-crate
  only) + `examples/cascade_seal_bench.rs`: petal-digest-while-hot ·
  reduction · root-after-last-petal latency · single-petal incremental
  re-seal · randomized arrival order · contrast row = the deleted
  whole-cycle FNV · leaf granularity 512-B-row vs 16-register-petal ·
  **indexing mode rank-based vs fixed-grid** (W2) · **identity-floor
  compliance of each candidate** (W2). All wall-clock numbers are
  economics, labeled as such.
- **G6 harness is standalone** (S4-6): inject → `resolve_petal`×n →
  `finalize` → compare roots. The probe `Scheme` trait is per-locus and is
  NOT retrofitted (S4-10 accepted: no unifying-trait refactor).

### W5 — implementation sequencing (post-ratification, task #25)

W4 bench first (probe-only, zero substrate risk) → primitive + indexing
pinned → W1 descriptor purity → W2 digest seam → W3 FNV deletion + gate —
each wave its own commit with its falsifiers + board hygiene, one central
gate run per wave.

## 4. NON-GOALS (v2 additions marked)

- X-C2-3 ECC/coding family — separate; never shapes checksum geometry.
- Compaction — economics only.
- Strict-sentinel removal / retention-tombstone — Tier-1 follow-ons.
- Code implementation inside this council pass — gated on #968 ready-flip.
- Encryption — never (req 3).
- E2 scatter multi-machine sweep — independent.
- **(v2, S5-3/S5-4)** Replay-time root verification (recomputing
  ContentRoot over a replayed row set) and any QueryReference-level
  content verification: architecturally available, wired nowhere in
  W1–W4, and ContentRoot is strictly PER-CYCLE — equating it with a
  cross-cycle QueryReference horizon is a named future trap, not a
  feature of this arc. (The one exception is W3's cross-version
  reconcile, which is restart-path recompute over one cycle's
  already-loaded rows.)
- **(v2, S4-10)** No unifying trait over per-locus `Scheme` and the
  tree accumulator.

## 5. PRE-REGISTERED GATES (v2 — scoped and extended)

- G1 **F-SEAL-ORDER**: any arrival permutation — including cross-owner
  interleave permutation (S5-2 arm) — → byte-identical roots, **given
  distinct `stream_position`s**. The tie-density caveat is real and
  pre-registered (S1-5): where the stable sort breaks ties, arrival is
  the durable order by design (`seal-vs-temporal-ordering-information.md`)
  — **PROBE-SEAL-TIE-DENSITY** (unrun) decides whether the distinctness
  precondition is a law or a workload assumption; it joins the W4 wave.
  In-tree pairing: defect-pin :2554 deleted + :2604 un-ignored in the
  same commit (S4-2).
- G2 **F-SEAL-NORESCAN**: `payload_bytes_digested` == payload bytes
  written, exactly once; finalize touches zero payload bytes.
- G3 **F-SEAL-PRESENCE**: register erasure/absence moves the root under
  BOTH indexing modes (rank-based: rank+count shift; fixed-grid:
  UNRESOLVED stamp).
- G4 **F-SEAL-ROOT-LATENCY**: after the last petal only that petal's path
  runs.
- G5 **F-TWICE**: instrumented dereference count == 2 (production +
  flush) — verifying the TARGET; the OLD path's measured 4 is the
  contrast row (Frozen 6 as amended).
- G6 **X-C2-1 on the root**: standalone root-injection harness (S4-6);
  I3/I4/I5/I6 all move the root; kill floor; null control.
- G7 **Seal-version gate** (extended per S5-6): (a) v0 frame decodes;
  (b) same-content cross-version re-submission → `Reconciled`;
  (c) **divergent-content cross-version re-submission → `HashConflict`**
  (the fail-closed arm the recompute-and-compare rule exists for);
  (d) disable-run red-then-green on the version-awareness of the
  reconcile comparison.
- G8 Existing suites green (incl. persist_sink.rs:1943 re-verified,
  S4-3); clippy -D warnings + fmt per wave; debug=0 discipline.
- G9 Board hygiene same-commit per wave. **The ratification E-entry
  carries ONLY novel content** — seal_version + the cross-version
  recompute-and-compare rule + W4's measured numbers; petal-16 /
  identity-split / twice / FNV-deleted are already on the board verbatim
  and are cited, not restated (S1-10).

## 6. PER-SAVANT QUESTION SETS — discharged (Phase 1 complete; retained in v1's git history)

## 7. REVIEWER CONTRACT (Phase 3 — the 3 see THIS document only)

Per `.claude/agents/5plus3-council.md`: overclaim-auditor /
dilution-collapse-sentinel / firewall-warden; per spec section, verdict
`PASS / FIX(P1|P2) / BLOCK(P0)` + evidence; "looks good" without naming
why each section earns PASS is malformed.

## 8. CHANGE LEDGER (v1 → v2)

45 findings: 4 VIOLATES · 14 GAP · 6 PRIOR-ART-AT · 6 RISK · 15 CONFIRMS.
Zero frozen-decision re-opens. Dispositions:

| # | Finding | Disposition |
|---|---|---|
| L1 | S1-9 VIOLATES: mailbox-carrying descriptor contradicts 6 shipped mailbox-less sites | W1 rewritten: reuse `RowSpanDescriptor` shape, ownership via cast pairing. v1 design withdrawn |
| L2 | S1-10 VIOLATES: board-entry duplication | G9 rewritten: novel-content-only E-entry |
| L3 | S3-2 VIOLATES: HashConflict cited :289, is :267 | §2 corrected |
| L4 | S3-5/S3-7 + S2-3 VIOLATES/GAP: payload touched 4× today; "twice" was stated as current | Frozen 6 reworded (twice = TARGET); W1 explicitly deletes the freeze clone + FNV walk; G5 gains the 4-touch contrast row |
| L5 | S5-1/S5-2 GAP/RISK: tree shape ambiguous; cross-mailbox position stability ungated | W2 commits rank-of-row-id indexing (position from durable coordinates); fixed-grid measured in W4; G1 gains the owner-interleave arm |
| L6 | S5-5/S5-6 + S2-8 RISK/GAP: cross-version reconcile silently accepts divergence | W3 rule REPLACED: recompute-and-compare over recovery-loaded rows; divergence → HashConflict fail-closed; G7 gains arm (c). Req-2-compliant via its own verification clause |
| L7 | S2-2 GAP: bare hash comparison at cycle_sink.rs:792-793 would misfire cross-version | W3 names that exact comparison as the rewrite site |
| L8 | S1-8 + S2-1 + S3-9 GAP: FrameMeta has no encode site in persist_sink; gate must land on Arrow schema | W3 retargeted to cycle_store_schema + find_frame projection + timeline; FrameMeta field + 9 sites (S4-1) |
| L9 | S1-6/S1-7 PRIOR-ART-AT: ENVELOPE_LAYOUT_VERSION et al. | W3 mirrors that shape explicitly |
| L10 | S2-6 GAP: fold64 undefined | Defined: first 8 LE bytes of BLAKE3-256 over the fixed composition; identity-H is not the req-11 hot path (cost reported anyway) |
| L11 | S2-7 RISK: CRC32C 32-bit root < 64-bit identity floor | Identity floor pre-registered as a W4 decision constraint |
| L12 | S4-6 GAP + S4-10 RISK: G6 can't reuse Scheme; refactor temptation | Standalone harness committed; no-unifying-trait non-goal |
| L13 | S4-2 CONFIRMS: in-tree G1 falsifier pair | Wired into G1 + W3 same-commit obligations |
| L14 | S4-4/S4-5 CONFIRMS: Arrow UInt64 consumer; HashConflict→Escalate mapping | §2 inventory; W3 preserves Escalate's fail-closed meaning (works with L6) |
| L15 | S1-4/S1-5 GAP/RISK: missing knowledge doc; G1 tie-density scope | Doc added to §2; G1 scoped "given distinct stream_positions"; PROBE-SEAL-TIE-DENSITY joins W4 |
| L16 | S3-3 GAP: SweepSlot not generic; PayloadSource doesn't exist | W1 reworded: NEW default type param `SweepSlot<P = Vec<u8>>`; 2 production sites named |
| L17 | S4-8/S4-9 CONFIRMS: board/doc staleness on FNV deletion | W3 same-commit obligations |
| L18 | S5-3/S5-4/S5-8 GAP/RISK: replay verification unwired; QueryReference conflation; recovery digest-blindness | Explicit non-goal with the per-cycle-scope trap named |
| L19 | S5-7 GAP: monotonicity contract stated once | Cross-referenced in §2 and W1 |
| L20 | S1-1/S1-2/S1-3 PRIOR-ART-AT: ndarray merkle (falsified shape), AuditMerkleRoot (Frozen-4 precedent), spo merkle (cautionary) | Cited in §1.4, §2, and W3's verifier design |

Conflicts resolved: none adversarial — S5-5 and S2-8 independently flagged
the same hole (both credited, L6); S1-8 and S3-9/S2-1 converged on the
Arrow seam (L8). Losing positions recorded, not deleted: v1's
mailbox-carrying descriptor (L1) and reconcile-by-fence-chain-alone (L6)
are retained above as WITHDRAWN with their reasons.
