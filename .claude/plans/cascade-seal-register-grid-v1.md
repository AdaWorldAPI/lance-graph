# Cascade-accumulated seal on the register grid — RATIFIED v3 (5+3 council)

> **Status: RATIFIED v3 — council complete (Phases 0–5).** v1 = Phase-0 spec;
> v2 = Phase-2 consolidation of the 5 savants (S1 prior-art, S2 iron-rules,
> S3 code-truth, S4 cascade-impact, S5 different-views — 45 findings); v3 =
> Phase-4 fixes from the 3 reviewers (R1 overclaim-auditor, R2
> dilution-collapse-sentinel, R3 firewall-warden — 3×BLOCK(P0) + 4×FIX(P1) +
> 4×FIX(P2), all resolved, none argued away; ledger §8 L21–L28). Verdict
> counts and deltas are in the AGENT_LOG entry for this run.
>
> **This ratifies the SPEC. Code implementation stays gated on the
> operator's #968 ready-flip** (the STOP order's discharge condition). The
> operator may strike or amend any committed decision below at ratification;
> §8 marks the two points where the council exercised authoring discretion
> (L22, L24).

---

## 1. FROZEN DECISIONS (verbatim-checked against sources; items 6 and 11 carry DISCLOSED clarifications — no substance re-litigated)

1. **The 12 seal requirements** — operator verbatim,
   `docs/lotus/CASCADE-ACCUMULATED-SEAL-SPEC.md` §Requirements (2026-08-19):
   no post-finalization **full-cycle** payload rescan · no storage reread ·
   no encryption · no physical-order dependency · digest binds
   locus+resolved-state+content · petal digest while hot · higher levels
   reduce child digests only · root before Lance publication · one
   DatasetVersion publishes image+root · DatasetVersion = publication
   identity / root = content identity · primitive benchmarked ONLY
   in-architecture · FNV deleted, not optimized. *(v3: req 1's "full-cycle"
   qualifier restored verbatim — it is load-bearing for §3 W3, which now
   complies with it trivially rather than testing its edge.)*
2. **Register-grid correction** — operator verbatim, same doc
   §ARCHITECTURAL CORRECTION: Lotus holds registers, never copied
   rows/petals/images; petal = 16 register positions; phase + canonical
   register position CONSTRUCT ordering; the digest seam is the ONE
   flush-time dereference; identity split ContentRoot / ControlRoot /
   DatasetVersion with `H(cycle ‖ base_version ‖ ControlRoot ‖
   ContentRoot)`; superseded payload bytes NOT hashed; cast content-blind;
   "ZERO COPY UNTIL THE MEMBRANE." The stated closure hierarchy
   `4096 → 1024 → 256 → 64 → 16 → 4 → 1` traces to real SoA capacity
   (map §6: `MailboxSoA<N>` default N=1024 × 4 mailboxes) and is the
   PRIMARY indexing design in §3 W2.
3. **§0 STORNO** — `docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md` §0:
   canonical replay coordinates; compaction = optional economics only.
4. **T0.3 amendment** — ΔV coordinates; wall-clock = economics; the
   returned DatasetVersion is NEVER a pre-write hash input. Shipped
   precedent: `AuditMerkleRoot::chain` (canonical input excludes its own
   output; `crates/lance-graph-callcenter/src/unified_audit.rs:86-101,
   161-174`).
5. **Idempotency verdict** — `docs/lotus/SEAL-FINALIZATION-MAP.md` §4: the
   `(cycle, batch_hash)` reconciliation CAN consume the accumulated root;
   no second hash, no second full pass. *(v3: unqualifiedly true again —
   the v2 cross-version recompute that quietly excepted this is WITHDRAWN,
   §3 W3 / ledger L23.)*
6. **Payload touched exactly TWICE ever** — map §6 — **as the TARGET
   state** (clarification licensed by map §6 itself superseding §5's
   TARGET column; R2 concurs it is disambiguation, not dilution). Measured
   CURRENT state is FOUR touches (production + freeze-clone + FNV-read +
   flush-append; S3): W1+W2 collapse 4 → 2; G5 verifies the target with
   the 4-touch old path as contrast row.
7. **I-LEGACY-API-FEATURE-GATED** — version gate on the serialization
   path; stored FNV-era frames readable; no same-name silent change.
8. **Zero-copy law** — `NodeRowPacket` not Clone/Copy; envelopes zero-copy
   creation→tombstone. The descriptor is a small copyable ADDRESS, never a
   packet handle.
9. **Archaeology verdict settled** — wire the declared contract; create
   nothing new. The descriptor TYPE already exists mailbox-less at six
   sites (§2).
10. **Falsifiability rule** — disable-runs red-then-green; anti-vacuity +
    can-stay-silent; threshold inertness.
11. **Leaf GRANULARITY measured, not assumed** — scope per the map §1
    verbatim: row-512B vs petal-8KiB is the open, measured question.
    *(v3 clarification per R2: indexing MODE is a SEPARATE, explicitly
    NON-frozen open item — see §3 W2. Frozen 11 covers granularity only.)*
12. **X-C2-3 ECC separate** — never shapes checksum geometry.
13. **Lance family upstream-authoritative** — publication via existing
    lance =9.0.0 pins.
14. **No agents run cargo**; orchestrator runs all gates centrally.

## 2. INPUT INVENTORY (v3 — R1 spot-checked 10+ citations, all exact)

`crates/lance-graph-planner/src/persist_sink.rs`
- `SweepSlot` :171-194 — no generic parameter today; `pub payload:
  Vec<u8>` :193; descriptor doc :191-193; `stream_position` :183 with the
  per-owner cross-cycle monotonicity contract :176-182 (the write-order/
  restart-order agreement rests on this contract — cross-referenced
  deliberately).
- `LandedSlot` :205; `FrameMeta` :214-220 (no version field; in this file
  only a trait-returned timeline row); `CommitOutcome` :227-252;
  `CommitError` :258+ (`HashConflict` :267, Display :300; `Ambiguous`
  :287, its `batch_hash` field :289, Display :314-320 — **`Ambiguous` is
  the pre-existing fail-closed outcome §3 W3 routes cross-version cases
  to**; its own doc: "may or may not be durable", surfaced rather than
  guessed).
- `DetachedCycleBatch::freeze` :377-390 — today clones every payload into
  the owned image (:381) and `content_hash` :401-429 walks every payload
  byte (invoked :383) — the two pre-flush touches W1/W2 delete.
- `commit_cycle` :551; watermark :642-653; `recover_and_apply` :677-729 —
  restart-only sort; **touches no digest and, load-bearing for W3: the
  recovery path loads NO payload bytes** (see scan_sealed below).
- In-tree reconcile falsifier: `randomized_completion_order_yields_the_
  same_batch_hash` :1943 (post-mint order only; re-verify under the new
  hash, no rewrite).

`crates/lance-graph-planner/src/batch_writer.rs`
- Addendum-6 ruling :30-33; `BatchWriter<P>` :95; `cast()` :132 (zero
  production call sites); drain doc :177. The module doc's "(mailbox,
  dirty row-range, cycle)" phrasing is contradicted by every shipped
  descriptor implementation — §3 W1.

Shipped descriptor prior art: `RowSpanDescriptor{row_lo, row_hi, cycle}`
at `examples/blw_rows.rs:521-538`, `examples/blw_tenant.rs:370`,
`examples/blw_fusion.rs:370`, `tests/probe_ignition.rs:298`,
`tests/d_ign_b_lenses.rs:403`; `DirtyRange{first_row, rows, cycle}` at
`cognitive-shader-driver/src/mailbox_soa.rs:1779-1786`. All deliberately
mailbox-less — "Ownership rides the cast pairing
(`BatchWriter::on_behalf_of`), never the DTO — the write-on-behalf iron
rule."

`crates/lance-graph-contract/src/canonical_node.rs`
- `NodeRowPacket` :1511-1514; `SoaEnvelope` impl :1540; `as_le_bytes`
  :1553; not Clone/Copy (:1492-1510 region).

`crates/lance-graph-contract/src/soa_envelope.rs`
- The version-gate house shape: `ENVELOPE_LAYOUT_VERSION: u8 = 2` (:54),
  trait const `LAYOUT_VERSION`, mismatch error variant, reader-must-refuse
  rule (:45, :224-226). `seal_version` mirrors this.

`crates/lance-graph-supervisor/src/cycle_driver.rs`
- `collect_casts` :357-393 (`stream_position = position_base + cast.0`
  :385 — the ONE production SweepSlot mint, from a drained
  `BatchWriter<Vec<u8>>`); `seal_cycle` :434-437.
- `CommitCycleOutcome` → `SealRecovery` :132-150, :244, :463-471 —
  `HashConflict → Escalate` fail-closed today; W3 preserves that meaning
  and routes cross-version cases through the equally fail-closed
  `Ambiguous → Escalate`.
- In-tree G1 falsifier pair: defect-pin :2554 (self-documents "delete this
  pin when the fix lands") + `#[ignore]`d arrival-order-independence
  sibling :2604 — deleted / un-ignored in the SAME commit as W2/W3.

`crates/lance-graph/src/graph/cycle_sink.rs`
- The real serialization seam: schema :114-140 (no version column today);
  `build_batch` :458-556 (`push_common` :474; payload Arrow append
  :532-534 — the flush touch); decodes `find_frame` :560-596 (**projects
  `["batch_hash"]` ONLY** :577 — no row content reaches the reconcile
  comparison) and `timeline` :1061-1101 (FrameMeta :1093-1097). Reconcile
  comparison `stored_hash == batch.batch_hash` :792-793, else
  `HashConflict` ~:804-809 — zero version awareness today.
- **`scan_sealed` rebuild :1045 mints `payload: Vec::new()`** — the
  recovery/restart path loads NO payload bytes. (v3, per R2: this is the
  evidence that any cross-version content recompute would require a NEW
  full-cycle payload read — which req 1 verbatim forbids — and is why W3
  fails closed instead.)
- `batch_hash` persists as Arrow `UInt64` (:129) — the durable consumer
  the gate must reach.
- `FrameMeta` construction sites for the field addition (9): the 3 real
  (cycle_sink.rs:1093, persist_sink.rs:937, cycle_driver.rs:1224) + 6
  test/example mocks (probe_ignition.rs:445, probe_ignition_64k.rs:229,
  d_ign_b_lenses.rs:537, measure_wal_curve.rs:1525, blw_tenant.rs:523,
  blw_fusion.rs:497). No second real serialization site.

`crates/rp-seal-t0-probe/`
- `cascade_seal.rs` — held scaffold (zero external callers); `PetalDigest`
  :38-47; impls :49-140 (CRC32C width caveat :81-84); `CascadeSeal`
  :144-240.
- `lib.rs` `Scheme` :225-230 per-locus — G6 gets a standalone harness.

Knowledge doc: `.claude/knowledge/seal-vs-temporal-ordering-information.md`
— READ-BY names `persist_sink::{freeze, order_cycle_stably,
DetachedCycleBatch}`; carries the tie-density finding governing G1's scope
and the pre-registered, still-unrun **PROBE-SEAL-TIE-DENSITY**.

Prior-art contrast set (cross-referenced, never adopted): ndarray
`hpc/merkle_tree.rs:22-31,62-78` + `hpc/seal.rs:21-52` (the falsified S1U
locus-unbound shape); `graph/spo/merkle.rs:4-12` (`verify_lineage` checks
structure without re-hashing — the cautionary shape no verifier here
repeats).

Governing docs (frozen): `docs/lotus/CASCADE-ACCUMULATED-SEAL-SPEC.md`,
`docs/lotus/SEAL-FINALIZATION-MAP.md`,
`docs/lotus/RP-SEAL-CONSOLIDATION-PASS1.md` §0.

## 3. THE PROPOSED RESOLUTION (v3 — committed design, in order)

### W1 — Descriptor purity (wire the declared contract)

- Reuse the shipped mailbox-less descriptor shape (`RowSpanDescriptor
  {row_lo, row_hi, cycle}` promoted from example/test code into the
  planner); ownership rides the cast pairing (`BatchWriter::on_behalf_of`
  → `SweepSlot.owner`), never the DTO. The descriptor resolves against
  the owner's SoA slab through `NodeRowPacket::as_le_bytes` at the flush
  seam only.
- `SweepSlot` gains a NEW default type parameter — `SweepSlot<P =
  Vec<u8>>` — so the ~28 test/example construction sites compile
  unchanged; exactly 2 production sites move (`collect_casts`
  cycle_driver.rs:383; `scan_sealed` cycle_sink.rs:1045). One new trait,
  `PayloadSource`, resolving to `&[u8]` at the flush seam; `Vec<u8>` gets
  the identity impl.
- `freeze` stops dereferencing payloads entirely: the image becomes
  descriptor-keyed registers (per-row-finality fold selects the winning
  DESCRIPTOR per row); the clone at :381 and the FNV walk at :383/:401-429
  are deleted. After W1+W2 the only payload dereference in the pipeline is
  the flush read.
- Cast stays content-blind; no new sort on the write path; the T0.1
  restart sort stays restart-only; write/restart order agreement rests on
  the `stream_position` monotonicity contract (persist_sink.rs:176-182).

### W2 — Digest rides the ONE flush dereference

- At the flush seam (`build_batch`'s payload append, cycle_sink.rs:532-534
  — the same read that feeds the Arrow builder) the bytes ALSO feed
  `resolve_petal`. Leaf digest binds
  `D(cycle ‖ stream_position ‖ owner ‖ row ‖ resolved-flag ‖ base_version
  ‖ bytes)`.
- **Indexing — PRIMARY is the operator's fixed register grid** (v3, per
  R2; v2's rank-based commitment is demoted to challenger, ledger L24):
  ONE per-cycle tree over the fixed 4096-position register grid (the real
  SoA capacity, N=1024 × 4 mailboxes); **leaf slot = the global row id
  itself** — no rank arithmetic; absence = the UNRESOLVED stamp at
  finalize (a known slot, exactly the operator's stated design and the
  scaffold's existing shape). The rank-of-row-id variant is the
  explicitly NON-frozen challenger, measured in W4 for one stated reason:
  on sparse cycles (k dirty rows « 4096) fixed-grid finalize stamps
  4096−k absences (4096−k petal digests + full reduce), while rank-based
  digests only k leaves — the bench measures that trade under identical
  falsifiers (G3 covers absence identity in BOTH modes). The pick is a W4
  output beside the primitive pick.
- **Two claims, stated separately** (v3, per R1 — v2 conflated them):
  (a) *Tree-POSITION stability*: the leaf slot is a pure function of
  durable coordinates (the row id), so position assignment is independent
  of arrival and owner interleaving by construction — under BOTH indexing
  modes. (b) *Root-VALUE identity under arrival permutation* additionally
  requires the per-row fold WINNER to be order-independent, which holds
  **given distinct `stream_position`s** — exactly G1's scoping; where the
  stable sort breaks ties, arrival is the durable order by design, and
  the unrun **PROBE-SEAL-TIE-DENSITY** (W4) decides law vs workload
  assumption. W2 makes no claim stronger than G1 tests.
- **ContentRoot** = the cascade root over final image registers only
  (superseded intermediate bytes never reach the flush seam under row
  finality — not hashed). **ControlRoot** = an online digest over the
  durable control stream (per-cast stream_position / owner / row /
  move-tag / mailbox / witness_chain_position — the fields FNV ate, which
  persist as cycle_sink control rows); accumulated at collect-time over
  coordinates only, zero payload bytes.
- **Batch identity**: `batch_hash = fold64(H(cycle ‖ base_version ‖
  ControlRoot ‖ ContentRoot))`; `H` = BLAKE3-256 over the fixed ~50-byte
  composition (once per cycle, tiny fixed input — not the per-petal hot
  path req 11's bench governs; cost reported anyway); `fold64` = first 8
  bytes LE. **Identity floor**: ≥64 bits of content entropy end-to-end —
  the winning petal primitive's ROOT must be ≥8 bytes; `CRC32C/4`
  competes on petal cost but cannot be the identity root unless the bench
  also measures a widened-root pairing (pre-registered constraint on
  W4's decision).
- FrameMeta.batch_hash stays `u64`; full roots travel in the commit
  receipt beside the returned DatasetVersion. Root exists BEFORE
  publication; DatasetVersion never a hash input.

### W3 — FNV deletion + seal-version gate (I-LEGACY, on the REAL seam) — v3 rule

- `content_hash` (FNV) deleted from the write path (req 12). The version
  gate lands on the ACTUAL serialization seam: a new nullable Arrow
  column `seal_version` in `cycle_store_schema` (cycle_sink.rs:114-140) —
  absent/null ⇒ v0 (FNV-era), new frames write v1 — mirroring the shipped
  `ENVELOPE_LAYOUT_VERSION` shape (reader-must-refuse for unknown
  versions). `FrameMeta` gains `seal_version: u8` (9 sites); `find_frame`
  widens its projection from `["batch_hash"]` to include the version
  column; `timeline` materializes it.
- **Reconcile rule (v3 — both prior rules WITHDRAWN, ledger L23):**
  - Same seal_version → hash comparison exactly as today (`Reconciled` on
    match, `HashConflict` on mismatch). The steady-state fast path is
    unchanged — Frozen 5 holds unqualified.
  - **Cross-version → `CommitError::Ambiguous`** with a named
    version-migration cause, which the existing recovery mapping
    escalates (`Ambiguous`'s own doc: "may or may not be durable" —
    surfaced, never guessed). Rationale, stated honestly: same-content
    and divergent-content resubmissions are **indistinguishable across
    hash algorithms without reading payload bytes**; no path in the
    system loads those bytes at reconcile time (`find_frame` projects the
    hash column only; `scan_sealed` mints `payload: Vec::new()`), and
    adding such a read would be a post-finalization full-cycle payload
    rescan — forbidden by req 1 verbatim. Failing closed is therefore
    the only rule that satisfies the frozen requirements. The window is
    narrow by construction: a cross-version comparison requires a
    lost-response retry that STRADDLES the one-time seal-version flip.
    If the operator later wants auto-reconcile across that window, that
    is a deliberate operator relaxation — recorded as an option, not
    designed in.
  - Consequence: NO recompute, NO payload reads on any reconcile path,
    recovery stays digest-blind unconditionally, and reqs 1/2 are
    satisfied trivially rather than argued around.
- Same commit: delete the defect-pin (cycle_driver.rs:2554) + un-ignore
  the arrival-order-independence sibling (:2604); LATEST_STATE prepend
  correcting the stale F-ORD narration; CASCADE spec / MAP status lines
  flip.

### W4 — req-11 in-architecture benchmark (the measured decisions)

- Wire `cascade_seal.rs` (deps blake3/crc32c/xxhash-rust, probe-crate
  only) + `examples/cascade_seal_bench.rs`: petal-digest-while-hot ·
  reduction · root-after-last-petal latency · single-petal incremental
  re-seal · randomized arrival order · contrast row = the deleted
  whole-cycle FNV · leaf granularity 512-B-row vs 16-register-petal
  (Frozen 11) · indexing mode fixed-grid (primary) vs rank-based
  (challenger; sparse-cycle absence-stamp trade) · identity-floor
  compliance per candidate · PROBE-SEAL-TIE-DENSITY. Wall-clock =
  economics, labeled.
- G6's harness is standalone (inject → `resolve_petal`×n → `finalize` →
  compare roots); the probe `Scheme` trait is NOT retrofitted.

### W5 — implementation sequencing (post-#968-ready-flip, task #25)

Nothing in W1–W4 starts before the operator flips #968 ready (v3, per
R3). Then: W4 bench first (probe-only, zero substrate risk) → primitive +
granularity + indexing pinned → W1 descriptor purity → W2 digest seam →
W3 FNV deletion + gate — each wave its own commit with its falsifiers +
board hygiene, one central gate run per wave.

## 4. NON-GOALS

- X-C2-3 ECC/coding family — separate; never shapes checksum geometry.
- Compaction — economics only.
- Strict-sentinel removal / retention-tombstone — Tier-1 follow-ons.
- Code implementation before the #968 ready-flip.
- Encryption — never (req 3).
- E2 scatter multi-machine sweep — independent.
- Replay-time root verification (recomputing ContentRoot over a replayed
  row set) and QueryReference-level content verification: architecturally
  available, wired nowhere; ContentRoot is strictly PER-CYCLE — equating
  it with a cross-cycle QueryReference horizon is a named future trap.
  *(v3: with W3's recompute withdrawn there is NO exception — recovery
  and reconcile are digest-blind and payload-blind throughout this arc.)*
- No unifying trait over per-locus `Scheme` and the tree accumulator.

## 5. PRE-REGISTERED GATES

- G1 **F-SEAL-ORDER**: any arrival permutation — including cross-owner
  interleave — → byte-identical roots, **given distinct
  `stream_position`s** (tie-density caveat pre-registered;
  PROBE-SEAL-TIE-DENSITY joins W4 and decides law vs workload
  assumption). In-tree pairing: defect-pin :2554 deleted + :2604
  un-ignored same commit. W2's §3 claims are exactly this gate's claims —
  no stronger (v3, per R1).
- G2 **F-SEAL-NORESCAN**: `payload_bytes_digested` == payload bytes
  written, exactly once; finalize touches zero payload bytes.
- G3 **F-SEAL-PRESENCE**: register erasure/absence moves the root under
  BOTH indexing modes (fixed-grid: UNRESOLVED stamp; rank-based:
  rank+count shift).
- G4 **F-SEAL-ROOT-LATENCY**: after the last petal only that petal's path
  runs.
- G5 **F-TWICE**: instrumented dereference count == 2 (production +
  flush) — the TARGET; the old path's measured 4 is the contrast row.
- G6 **X-C2-1 on the root**: standalone root-injection harness;
  I3/I4/I5/I6 all move the root; kill floor; null control.
- G7 **Seal-version gate** (v3 arms): (a) v0 frame decodes; (b) SAME-
  version reconcile behavior byte-identical to today on both match and
  mismatch; (c) CROSS-version fence-mismatch → `Ambiguous`/Escalate —
  never silently `Reconciled`, never spurious `HashConflict`; (d)
  disable-run red-then-green on the comparison's version-awareness;
  (e) **zero payload dereferences on any reconcile path** — instrumented
  with G5's counter (the falsifier R2 asked for: proves W3 adds no
  storage read).
- G8 Existing suites green (incl. persist_sink.rs:1943 re-verified);
  clippy -D warnings + fmt per wave; debug=0 discipline.
- G9 Board hygiene same-commit per wave. The ratification E-entry carries
  ONLY novel content (the fail-closed cross-version rule + seal_version
  seam + W4's measured numbers); petal-16 / identity-split / twice /
  FNV-deleted are already on the board verbatim and are cited, not
  restated.

## 6. PER-SAVANT QUESTION SETS — discharged (Phase 1; in v1's git history)

## 7. PHASE-3 VERDICTS — discharged (Phase 4; per-reviewer verdicts in the AGENT_LOG entry; raw reports banked in session scratchpad)

## 8. CHANGE LEDGER (v1 → v2 → v3)

**Corrected headline (v3, per R1's §8 BLOCK):** zero operator rulings
re-litigated on substance; TWO frozen items carry disclosed
clarifications — Frozen 6 (twice = TARGET; licensed by map §6 superseding
§5's column) and Frozen 11 (scope restated to granularity-only; indexing
mode opened as an explicitly non-frozen measured item). Both are
amplifications of measurement, neither reverses a ruling; both are
flagged in §1 in place.

v1 → v2 (the 5 savants, 45 findings): L1–L20 as recorded in v2 (git
history of this file, commit a898825) — headline items: mailbox-carrying
descriptor WITHDRAWN for the shipped mailbox-less shape (L1);
touch-count reworded target-vs-measured (L4); tree-shape ambiguity
resolved (L5, superseded by L24); cross-version reconcile-by-fence-chain
WITHDRAWN for recompute-and-compare (L6, superseded by L23); seal gate
retargeted to the Arrow seam (L8); fold64 defined + identity floor (L10,
L11); G6 standalone harness (L12); G1 scoped + probe (L15).

v2 → v3 (the 3 reviewers):

| # | Reviewer finding | Disposition |
|---|---|---|
| L21 | R3 BLOCK(P0): model-tier token in the committed header | Stripped; siblings scrubbed; the AGENT_LOG entry names tiers by policy ROLE (per the council card), not by name-in-artifact |
| L22 | R3 FIX(P1): W5 "post-ratification" ambiguous vs the #968 gate | W5 renamed "post-#968-ready-flip"; header restates the gate. Council discretion note: the spec ratification (this document) and the operator's #968 ratification remain two distinct events, deliberately |
| L23 | R1 BLOCK(P0) + R2 BLOCK(P0), independent halves of one kill: W3's recompute-and-compare req-2 argument cited the wrong path (find_frame projects hash-only — R1) AND no path loads payload bytes at all (scan_sealed mints `payload: Vec::new()` — R2), while §1's dropped "full-cycle" hid the req-1 conflict (R2) | **Second withdrawal of a W3 rule.** v3 rule = fail closed: cross-version → `Ambiguous`/Escalate, no recompute, no payload reads anywhere on reconcile. "Full-cycle" restored in §1 req 1. Frozen 5 true unqualified again (resolves R1's §1 tension). G7 gains arms (b/c/e) |
| L24 | R2 BLOCK(P0): rank-based indexing committed as THE design demoted the operator's fixed-4096 grid under a mislabeled Frozen-11 citation | Fixed-grid restored as PRIMARY (leaf slot = row id, UNRESOLVED stamps); rank-based = explicitly non-frozen challenger with its one stated motivation (sparse-cycle absence-stamp cost); Frozen 11 restated granularity-only. Council discretion note: measuring the challenger is authoring discretion, disclosed here for the operator to strike |
| L25 | R1 BLOCK(P0): §8 "Zero frozen-decision re-opens" contradicted by the ledger's own rows | Headline replaced with the disclosed-clarifications form above |
| L26 | R1 FIX(P1) ×2: §1 header "unchanged from v1" false; W2 "by construction" dropped G1's precondition | Header reworded; W2 split into position-stability (by construction) vs root-value identity (G1-scoped, probe-gated) — the true leg kept, the overclaim removed (R2-collapse-safe) |
| L27 | R2 FIX(P2) ×3: §2 scan_sealed evidence uncross-referenced; §4 non-goal contradicted by W3's recompute; no gate falsifying "zero added reads" | §2 flags the evidence in place; §4 contradiction dissolved by L23 (no exception remains); G7(e) added |
| L28 | R2 §8 review: L1/L6 withdrawals confirmed as corrections, not collapses | Recorded; no action |

Conflicts: none adversarial — R1 and R2 converged on L23 from opposite
ends (both credited); the stricter verdict was taken everywhere it
applied. Losing positions retained above as WITHDRAWN with reasons, per
the anti-collapse rule.
