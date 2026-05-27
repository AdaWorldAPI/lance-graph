# Savant: SequenceGapAnomalyDetector  (id 6 · family 0x62 · lane L11)

**Tuple:** kind=PostingAnomaly · inference=Abduction · semiring=NarsTruth · style=Analytical
**Feeds Reasoner impl:** `PostingAnomalyReasoner`   (per the impl-per-ReasoningKind decision)

> dispatch: `ReasoningKind::PostingAnomaly` -> "abduce the most likely cause from the evidence trail"
> (`examples/savant_dispatch.rs:31`). Abduction -> `QueryStrategy::DnTreeFull`.

## What it decides (AXIS-B core)
Given a journal sequence that is **not contiguous** (a hole between `sequence_number` values within
one `(journal_id, sequence_prefix)` chain -- e.g. INV/2024/01042 -> INV/2024/01044, with 01043
absent), abduce **the most likely cause of the hole**: a *deleted posted entry* (a GoBD
Festschreibung breach) vs a benign cause (a draft discarded before ever taking a number, a
cross-prefix rename per L11 R22, an in-flight savepoint rollback per L11 R23, or an import
boundary). Output is a per-gap hypothesis with NARS `(frequency, confidence)`, not a verdict --
woa-rs surfaces it as an audit flag, never a write.

## Deterministic guard (AXIS-A -- stays in woa-rs)
Creation-time contiguity check `_is_end_of_seq_chain` (L11 R25, `sequence_mixin.py:L487-511`):
`max-min == len-1` AND highest is last in DB; plus the gap-free `_locked_increment` UNIQUE mechanism
(L11 R23, `sequence_mixin.py:L355-424`). These detect/prevent gaps *at write time*. The savant is
invoked only for the residual case: an **already-existing** hole the guard cannot attribute (R25
explicitly notes "detecting existing anomalous gaps ... is heuristic").

## Slot 1 -- Evidence (Arrow EvidenceRef)
Primary table `account_move` projected to the sequence-anomaly schema (one row per move in the
suspect chain window), `EvidenceRef { table: "account_move.sequence_chain", schema_fingerprint, rows }`:

| column | dtype | signal |
|---|---|---|
| `move_id` | `Int64` | identity of the move (or NULL sentinel row marking the inferred missing slot) |
| `journal_id` | `Int64` | chain partition key (part 1) |
| `sequence_prefix` | `Utf8` | chain partition key (part 2) -- hashes are **per `(journal_id, sequence_prefix)`** (L1 gotcha #8) |
| `sequence_number` | `Int64` | the contiguity axis; the gap = missing integer(s) in this column |
| `name` | `Utf8` | full Belegnummer; lets the reasoner detect rename contamination (INV vs FACT, L11 R22) |
| `state` | `Utf8` (`draft\|posted\|cancel`) | a `cancel` neighbour weakens "deleted"; `posted` neighbours on both sides strengthen it |
| `posted_before` | `Boolean` | a slot whose neighbours have `posted_before=true` raises P(missing one was also posted) (L1 R K3-12) |
| `inalterable_hash` | `Utf8`/nullable | hashed neighbours => chain was festgeschrieben => a hole = hash-chain break => strong "deleted posted entry" signal (L11 R10) |
| `date` | `Date32` | period boundary; a gap at a fiscal-year/month edge weakens "deleted" (sequence reset, L11 R20) |
| `create_date` | `Timestamp` | a present-but-later create_date adjacent to the hole hints at a savepoint-retry artefact (R23), not a deletion |

Discriminating window: rows where `sequence_number IN [gap_lo - k, gap_hi + k]` for the same
`(journal_id, sequence_prefix)`; `k` from `Budget.max_evidence_rows`.

## Slot 2 -- Odoo field -> signal map                 (cite L-doc file:lines)
- `sequence_number`, `sequence_prefix` <- `account.move._compute_split_sequence` stored fields -- `L11-COA-JOURNALS-LOCKDATES.md:130-132` (R26; `sequence_mixin.py:L47-48, L183-191`).
- contiguity definition (`max-min == len-1`, highest-is-last) <- `L11-COA-JOURNALS-LOCKDATES.md:125-128` (R25; `sequence_mixin.py:L487-511`).
- `name` / rename-reuse gotcha (alphabetical-max, INV>FACT) <- `L11-COA-JOURNALS-LOCKDATES.md:113-115` (R22; `sequence_mixin.py:L269-310`).
- `inalterable_hash` chain per `(journal_id, sequence_prefix)`, gap => `_get_chains_to_hash` UserError <- `L1-K3-POST.md:662-669, 791` (R K3-10; `account_move.py:L4683-4725`) and `L11-COA-JOURNALS-LOCKDATES.md:65-67` (R10).
- `state`/`posted_before` (name freeze; storno keeps both rows so deletion != cancellation) <- `L1-K3-POST.md:734-737` (R K3-12) and `L1-K3-POST.md:409-416` (R K3-6: storno = two rows, never deletes original).
- period-reset families (gap at year/month edge is benign) <- `L11-COA-JOURNALS-LOCKDATES.md:105-107` (R20; `sequence_mixin.py:L193-225`).
- savepoint-retry gap-prevention (present row, later create_date) <- `L11-COA-JOURNALS-LOCKDATES.md:117-119` (R23; `sequence_mixin.py:L355-424`).

## Slot 3 -- Property-level alignment
Decision stays **within family 0x62 SMBAccounting**. `account.move` -> `fibo:Transaction` and the
chain-integrity concept (hash, sequence) are ledger-internal; no cross-seam traversal is required to
*detect* the anomaly. PROPOSED only (no property-level axiom exists today -- `odoo_alignment.rs`
holds only class-level `owl:equivalentClass` rows): if the breach is later reported across the audit
seam, `odoo:inalterable_hash -> zugferd:integrityHash` / `fibofnd:hasIntegrityProof` would be the
property to traverse. For the AXIS-B decision itself: **N/A -- stays within 0x62**.

## Slot 4 -- AXIS-B decision in evidence terms
Let E = the sequence-chain window rows (slot 1) for one `(journal_id, sequence_prefix)` containing a
hole at `sequence_number = g`.

-> Conclusion C = `DeletedPostedEntry(journal_id, sequence_prefix, g)` (the GoBD-breach hypothesis),
emitted with NARS `(frequency, confidence)` where:
- **frequency** rises with: both neighbours `posted` (not `cancel`), both `posted_before=true`, both
  carry `inalterable_hash` (the hole breaks a festgeschrieben chain), `g` not on a period-reset
  boundary, a single isolated hole (not a run).
- **frequency** falls toward the benign hypothesis with: a `cancel` neighbour, `g` exactly at a
  fiscal-year/month reset edge, evidence of a prefix rename in `name`, or a later-`create_date`
  neighbour (savepoint artefact).
- **confidence** scales with window completeness (`rows` vs the ideal `gap_hi-gap_lo+1+2k`) and is
  capped by the phi-1 humility ceiling.

Discriminating features (ranked): `inalterable_hash` on neighbours >> `posted`+`posted_before` of
neighbours > distance of `g` from a period-reset boundary > `name`-prefix homogeneity >
`create_date` monotonicity. Abduction picks the **single most economical cause** consistent with E.

## Parity / GoBD notes
A confirmed deleted posted entry is a **GoBD para 146 / para 239 HGB** Unveraenderbarkeit breach
(Festschreibung). The savant output is suggestion-only (Iron Rule 7): it raises an audit flag for a
human; it must never auto-create, renumber, or backfill a move (renumbering would itself be a GoBD
violation). The hash-chain break (R K3-10) is the legally load-bearing corroborator -- a hole inside
a hashed range is materially stronger than a hole in an unhashed draft region.
