# Agent W5 Sprint Log — WitnessCorpus Pivot (Re-dispatch)

> **Worker:** W5 (re-dispatch; prior partial +21/-9 in commit f730528)
> **Branch:** claude/sprint-10-specs-patch-csv-prep
> **Date:** 2026-05-16
> **Target file:** .claude/specs/pr-ce64-mb-4-arigraph-spo-g.md

## Status: COMPLETE

All ~280 LOC delta applied in a single Python heredoc batch. No commits/pushes per instructions.

## Changes applied (8 patches)

| Patch | Location | Change |
|---|---|---|
| 1 | §3.3 | Replaced  (150 LOC) with full WitnessCorpus design (~220 LOC): ,  (CAM-PQ-indexed, unbounded),  (64-slot array), insert/query/cam_pq_search API, per-tenant lookup flow, Time-as-helper note |
| 2 | §3.4 (NEW) | Added W-slot semantics: Tier (3b)/Plasticity (2b)/State (1b) sub-fields; dispatch metadata ≠ epistemic confidence table |
| 3 | §5.1-5.3 | Replaced G-slot tenant routing with SoA partition + palette family-prefix + WitnessCorpus flow; retired SpoWitnessChain<32> truncation; added W5-INV-CHAIN-ORDER application at decoration time |
| 4 | §6 table | Updated witness.rs row: SpoWitnessChain<N> → WitnessEntry + WitnessCorpus + WitnessCorpusStore; LOC +150 → +280 |
| 5 | §7 T3 | Replaced  with  (covers all 3 invariants); added T8 (cam_pq_lookup_performance) + T9 (multi_root_store) |
| 6 | §8 risk | Replaced SpoWitnessChain<N> sizing risk with WitnessCorpus unbounded growth mitigation |
| 7 | §10 invariants | Added W5-INV-CHAIN-ORDER, W5-INV-WITNESS-UNBOUNDED, W5-INV-CAM-PQ-INDEX as iron rules with cross-refs |
| 8 | §11 exports | Updated to list WitnessCorpus types; added Retired list; added D-CSV-6/D-CSV-7 cross-refs |
| 9 | Footer | Updated LOC estimate 600 → 900; listed all 3 invariants + plan/knowledge cross-refs |

## Invariants established

- **W5-INV-CHAIN-ORDER** (iron): timestamp_ns ASC; source_url.hash() tie-break; Arc::make_mut CoW
- **W5-INV-WITNESS-UNBOUNDED** (iron): no <N> cap; salience-decay eviction only
- **W5-INV-CAM-PQ-INDEX** (iron): cam_pq_index is canonical; Vec scan forbidden in production

## Cross-refs verified

- cognitive-substrate-convergence-v1.md §5 L-3, L-6, L-9, L-16, L-17; §6 W-slot; §11 D-CSV-6/D-CSV-7; §12 W5 patch row
- spo-schema-and-mailbox-sidecar.md §2.4 (SPO-W cardinality → CAM-PQ)
- CLAUDE.md I-VSA-IDENTITIES (witness payloads use identity fingerprints)
