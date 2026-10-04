# 2026-10-04 — the first probes from the German grammar map: DET/PRON, case, right corner (D-LXC-23)

**Status:** MEASURED (debug-0, release). Runs:
- `ud_pos_eval` on UD r2.15 de/en/fr GSD/EWT;
- `ud_case_eval` on GSD train/test and HDT train-a-1/test, all fetched, not committed;
- `tekamolo_corners` on GSD test, Luther 1545 and Elberfelder 1905.

Each kill bar was fixed before its run. The source is `.claude/knowledge/german-grammar-rule-inventory.md`. Workers wrote `tekamolo_corners.rs` and `ud_case_eval.rs`; the orchestrator gated and ran them.

## Rank 1 — der/die/das, article vs pronoun (`ud_pos_eval`, new pairs (Det, Rel) and (Det, Noun))

German GSD has no `PronType=Rel`; its relative der is PRON, folded to Noun. So the German pair is (Det, Noun). The voters are "followed by a noun/adjective → det", "after a comma → pronoun", "followed by a determiner / verb-only word → pronoun" and the position tables.

| | frequency | joint quorum | pronoun precision | pronoun recall |
|---|---|---|---|---|
| German GSD (1,719 tokens) | 92.3 % | **96.0 %** | 90.0 % (72/80) | 54.1 % (72/133) |
| UD English | 95.1 % | 96.7 % | 83.0 % | 73.7 % |
| French GSD | 97.0 % | 98.4 % | 89.2 % | 63.5 % |

**PASS** on all three bars: precision ≥ 0.80, recall ≥ 0.50, fires ≤ 2× gold. German recall only just clears its bar. `ud_pos_eval` now prints second-reading precision and recall for every pair.

## Ranks 5–8 — case (`ud_case_eval`, new)

Each rule predicts Case= from form, position and train-mined tables. Gold is read only for scoring.

| rule | GSD (7,780 scored) | HDT (103,276 scored) | verdict |
|---|---|---|---|
| R1 Acc prepositions | 96.9 % on 322 vs 32.3 % | 90.0 % on 3,345 vs 40.8 % | PASS (HDT exactly at the 0.90 bar) |
| R1 Dat prepositions | 95.9 % on 895 | 92.5 % on 10,642 vs 57.7 % | PASS |
| R1 Gen prepositions | 58.2 % on 55 | 68.4 % vs 70.9 % baseline | KILL (Dat drift) |
| R3 Wechsel, the article decides | 98.7 % on 315 | **99.7 % on 6,458** | — |
| R3 Wechsel, verb prior vs prep majority (same tokens) | 74.8 % vs 68.0 % | **63.0 % vs 63.9 %** | **KILL** (the GSD pass did not hold on HDT) |
| R4 relative pronoun after a comma | 83.1 % on 118 vs 64.4 % | 80.1 % on 2,545 vs 63.8 % | PASS |
| R5 combined, first rule wins | 79.3 % vs 61.3 %, coverage 50.8 % | 80.0 % vs 66.2 %, coverage 58.0 % | — |

**Findings.**
- **Wo/Wohin is decided by the article's case.** Wherever the article form leaves one of Dat/Acc, it is right 99.7 % of the time. The verb prior adds nothing measurable on the cleaner gold.
- **R4 initially fired on nothing.** The spec mined on `PronType=Rel`, which GSD lacks. It now mines train under the same surface condition the test uses.
- **The largest residual confusion on HDT is Nom predicted as Gen** (2,665). It comes from R2 voting the train majority for the ambiguous `der`. Next: the one-nominative elimination (rank 7).

## Rank 15 — right corner vs left corner (`tekamolo_corners`, new)

Pre-registered prediction: right-corner reading overturns left-corner commitments more often on biblical text.

| corpus | overturn rate (95 % CI) | matched arm, post hoc: only the commit point moved | right-only lanes |
|---|---|---|---|
| UD German-GSD test | 12.3 % (10.3–14.3) | 12.2 % | 31 |
| Luther 1545 | 7.1 % (6.9–7.3) | 7.0 % | 10,861 |
| Elberfelder 1905 | 7.9 % (7.6–8.1) | 8.0 % | 5,171 |

**Verdict: KILL.** The prediction is falsified. The matched arm, post hoc, rules out the confound between commit point and ambiguity admission. On the Bible, the right reader's extra lanes come from admitting the ambiguous *da* (10,296 ambiguous wins on Luther), not from where it commits.

**OPEN.** Why does edited German overturn more? Candidate: GSD clauses are longer and carry more competing cues per clause. This is not measured, and the reason is left unexplained.
