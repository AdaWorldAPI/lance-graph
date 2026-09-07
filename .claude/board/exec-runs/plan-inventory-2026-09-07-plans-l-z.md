# Inventory: `.claude/plans/` — second half (L–Z, entries 104–211 of the sorted directory listing)

Repo: `/home/user/lance-graph`, checked out on `main` at `aeebfb23`. Range
confirmed: `ls .claude/plans | sort` has 211 entries; entry 104 is
`lance-convergence-staged-migration-v1.md`; this inventory covers entries
104–211 inclusive (108 files, through `wikidata-lazy-spine-hydration-v1.md`).

Method note (read carefully before trusting any single cell): for every
file the metadata region (first ~80–120 lines) was read with `Read`, the
verbatim `Status:`/`**Status:**` line was extracted (not a substring match —
see the Candidate-epiphanies section for a documented case where naive
substring matching would have misfired), and every `D-`-prefixed identifier
was extracted with a regex, THEN filtered for a documented false-positive
class (see below) before being reported as a `d_id`. The 25 flagged
"special attention" files plus every file with no status line were read in
full or substantially (metadata + body sections + tail). D-id board
cross-checks used `STATUS_BOARD.md`'s own rows, anchored on the row's
leading `| D-id |` cell.

**D-id extraction false positives found and corrected (read this before
citing any `d_ids` cell as ground truth):** the regex `D-[A-Z][A-Z0-9]*(-[A-Z0-9]+)+`
matches not only real deliverable IDs but also the tail of longer
epiphany-style sentence-IDs that begin `E-...` or `TD-...` — e.g. the
literal text `E-READ-NOT-GREP` contains the substring `D-NOT-GREP`, which
the naive regex reports as if it were a real `D-NOT-GREP` deliverable.
**81 such false positives were found and removed across 43 files** (full
list in Candidate epiphanies). The `d_ids` column below is the
POST-FILTER list. Some remaining entries in that column are still not
literal STATUS_BOARD deliverable rows but this repo's own house style of
citing a full-sentence finding as a `D-<SENTENCE>-N` id (e.g.
`D-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-CONTROL-1`) — these are legitimate
citations in-file, just not STATUS_BOARD rows, so they correctly show as
"not found" in the board cross-check rather than as errors.

## Table

| file | title | status_token | version | d_ids | prs | last_commit | verdict |
|---|---|---|---|---|---|---|---|
| `lance-convergence-staged-migration-v1.md` | lance convergence — the STAGED migration plan (v1) | MEASURED / ready-to-execute for stages 0–1; stage 3 is | v1 | D-BSW-0, D-BSW-2, D-LNC-0, D-LNC-1, D-LNC-2, D-LNC-3, D-LNC-4, D-LNC-5, D-LNC-6, D-LNC-7, D-MTS-6 | #1182, #1187, #1189, #1190, #8206*, #8589* | 2026-09-05 | OPEN — active 2026-09-05 arc; D-LNC-0..3 Shipped but D-LNC-4/5/6 + D-LNC-5a/D-MW-P2 still Queued/In PR |
| `lance-graph-business-logic-poc-via-woa-rs-v1.md` | Lance-Graph Business-Logic POC via woa-rs — v1 | Active (Draft) | v1 | D-LGMC-1, D-LGMC-15, D-LGMC-2, D-LGMC-22, D-LGMC-4, D-LGMC-5, D-LGMC-6, D-LGSMB-1, … (+39 more) | #150, #151, #152, #363, #372, #383, #390, #407 | 2026-05-25 | OPEN — Active (Draft) roadmap; references other plans' D-ids, ships nothing itself |
| `lance-graph-in-medcare-rs-v1.md` | lance-graph in MedCare-rs — v1 | Draft | v1 | D-LGMC-1, D-LGMC-10, D-LGMC-11, D-LGMC-12, D-LGMC-13, D-LGMC-14, D-LGMC-15, D-LGMC-16, … (+35 more) | #355 | 2026-05-25 | OPEN — Draft; D-LGMC-* deliverables not confirmed shipped |
| `lance-graph-in-smb-office-rs-v1.md` | lance-graph in smb-office-rs — v1 | Draft | v1 | D-LGMC-4, D-LGSMB-1, D-LGSMB-2, D-LGSMB-3, D-LGSMB-4, D-LGSMB-5, D-LGSMB-6, D-LGSMB-7, D-LGSMB-8, D-LGSMB-9, D-SDR-2, D-UB-2, D-UB-5, D-WLG-3 | #18, #273 | 2026-05-25 | OPEN — Draft; Phase A explicitly "Not started" |
| `lance-graph-in-woa-rs-v1.md` | lance-graph in woa-rs — v1 | Draft | v1 | D-LGMC-1, D-LGMC-2, D-LGMC-21, D-LGMC-4, D-LGSMB-1, D-UB-11, D-UB-4, D-UB-8, D-WLG-1, D-WLG-10..17, D-WLG-2..9 | #275 | 2026-05-25 | OPEN — Draft; 6-phase plan, no shipped confirmation found |
| `lance-graph-ontology-v5.md` | Plan: lance-graph-ontology v5 — post-merge follow-ons | Drafted (2026-05-07). Picks up where v4 left off | v5 | D-ONTO-V5-1 … D-ONTO-V5-15 (15 items) | - | 2026-05-07 | OPEN — 15-deliverable plan; only D-ONTO-V5-9 confirmed shipped (PR #355, per cross-repo citations), rest unconfirmed |
| `lance-graph-rdf-fma-snomed-v1.md` | lance-graph-rdf — FMA / SNOMED CT / RadLex import + named-graph context | plan, not implementation. v1. | v1 | - | #331 | 2026-05-04 | OPEN — explicitly "plan, not implementation" |
| `lance9-datafusion54-upgrade-probe-v1.md` | lance9-datafusion54-upgrade-probe-v1 — what breaks on lance 9 / lancedb 0.33 / DataFusion 54 / Rust 1.97.1 | MEASURED 2026-08-05. Assessment, not a landed migration | v1 | D-THE-NEW-ONE-1 (house-style citation, not a board row) | #14, #19, #93, #179, #243, #244, #245, #351, #891, #895 | 2026-08-05 | CLOSED — the assessment itself is complete per its own status; the actual migration it fed into landed later via `lance-convergence-staged-migration-v1.md` |
| `lf-integration-mapping-v1.md` | LF Integration Mapping — v1 | Active (2026-04-25) | v1 | - | #262, #263, #264 | 2026-04-25 | OPEN — Active, no closure signal found |
| `lite-unified-surrealql-lance-v1.md` | lite-unified-surrealql-lance-v1 — one store + one query surface, behind a feature gate | CONJECTURE / design. "Test via feature gate; do NOT commit" | v1 | - | - | 2026-06-18 | OPEN — CONJECTURE/design, explicit no-commit guardrail |
| `literature-probe-ladder-v1.md` | Literature as falsifier — the probe ladder — v1 | PROPOSED (doc-only). No code, no contract change, no board mutation | v1 | D-LIT-1, D-LIT-2, D-LIT-3, D-LIT-4 (D-LIT-5 in filename only, not a real id) | - | 2026-07-22 | OPEN — PROPOSED; D-LIT-1..4 all Queued on STATUS_BOARD |
| `mailbox-belief-update-and-substrate-test-v1.md` | mailbox-belief-update-and-substrate-test-v1 — "what did I learn" + two-axis substrate test | CONJECTURE / design. 5+3 council COMPLETE. Rides AFTER cycle-aware write | v1 | D-MBX-A3 | - | 2026-06-18 | OPEN — design ratified (council COMPLETE) but code (D-MBX-A3) is Queued on the board |
| `mailbox-cycle-aware-write-contract-v1.md` | mailbox-cycle-aware-write-contract-v1 — every SoA write carries/checks its cycle | CONJECTURE / design. 5+3-gated before code. | v1 | D-MBX-9 | #535 | 2026-06-18 | OPEN — file's own §"Open questions" says ALL 5+3 OQs are resolved, but D-MBX-9 is still Queued on STATUS_BOARD (gates on D-MBX-7/8 + surrealdb + D-PERSONA-5) |
| `mask-algebra-revision-read-v1.md` | mask-algebra-revision-read-v1 — PLAN (DRAFT) | DRAFT, awaiting operator ruling on §5 | v1 | D-ATOM-4, D-MAR-1, D-MAR-2 | #651, #1099 | 2026-08-31 | OPEN — D-MAR-1 Shipped, D-MAR-2 explicitly blocked pending the operator ruling |
| `measure-64k-axes-v1.md` | measure-64k-axes v1 — the corrected five-axis benchmark (operator-specified, 2026-08-05) | NONE | v1 (higher siblings v2/v3/v4 exist) | D-BLW-4, D-KIA-A2 | #445, #879 | 2026-08-05 | SUPERSEDED — v2/v3/v4 siblings exist |
| `measure-64k-axes-v2.md` | measure-64k-axes v2 — rolling epoch closure (operator-specified, 2026-08-05) | NONE | v2 (higher siblings v3/v4 exist) | D-KIA-A2 | #879 | 2026-08-05 | SUPERSEDED — v3/v4 siblings exist |
| `measure-64k-axes-v3.md` | measure-64k-axes v3 — the three arms Stage A0 earned (operator-directed, 2026-08-05) | NONE (read in full — see special note) | v3 (higher sibling v4 exists) | D-KIA-A2 | - | 2026-08-05 | SUPERSEDED — v4 sibling exists; its own M-arm/O-arm results are explicitly folded forward into v4 + a knowledge doc |
| `measure-64k-axes-v4.md` | measure-64k-axes v4 — the hot version window (operator-directed, 2026-08-05) | NONE | v4 (latest) | D-LANCE9-LANCEDB036-REMEASURE, D-SEMANTIC-SLOT (house-style citations) | - | 2026-08-05 | OPEN — latest in the v1-v4 chain, no status line found, no higher sibling |
| `medcare-consumer-pull-thinking-proof-v1.md` | medcare-consumer-pull-thinking-proof-v1 — one real medical thought through the already-live OGAR consumer path | ACTIVE (consolidation + proof target). Date 2026-08-02 | v1 | - | #879 | 2026-08-09 | OPEN — ACTIVE |
| `mul-calibration-not-verdict-v1.md` | Plan: MUL calibrates, it does not adjudicate | PROPOSAL (unbuilt) — 2026-08-26. PLAN/BOARD ONLY, no code | v1 | D-GATE-1, D-GATE-2, D-GATE-3, D-GATE-5, D-MCAL-0, D-MCAL-1..6, D-TSC-1 | #1045, #1052, #1054 | 2026-08-26 | OPEN — D-GATE-1..5 / D-MCAL-0 Queued |
| `mul-consumer-build-gate-v1.md` | Build gate: the D-MCAL arc against its real consumers | GATE RUN — 2026-08-27. Discharges D-MCAL-6 and the second half of… | v1 | D-MCAL-1..6 | #1045, #1065..1069 | 2026-08-27 | OPEN — "GATE RUN" reads as complete, but the file's own §4 states F-MUL-6's second half is "OPEN — not discharged" (MedCare-rs never built against this head) — this is exactly the naive-keyword trap the task warns about |
| `mul-consumer-census-v1.md` | Measurement: per-symbol MUL consumer census | MEASUREMENT COMPLETE — 2026-08-27. Measurement only, no code change | v1 | D-MCAL-1, D-MCAL-6 | #1045, #1052 | 2026-08-27 | CLOSED — measurement-only deliverable explicitly finished; D-MCAL-1/6 read Shipped on the board |
| `mul-ewa-trust-propagation-v1.md` | mul-ewa-trust-propagation-v1 — trust is point-wise today; the sandwich is a CANDIDATE propagation operator | PROPOSED — PLAN/BOARD ONLY. Measure-before-carve | v1 | D-MEP-0, D-MEP-1 | #1065, #1068, #1070, #1072, #1074 | 2026-08-29 | OPEN — D-MEP-0/1 Queued |
| `mul-gate-outcome-vs-ground-v1.md` | Plan: separate the gate OUTCOME from producer-owned GROUND | SUPERSEDED (thesis) 2026-08-26 | v1 | D-GATE-1..6, D-KW-2 | #1045, #1052 | 2026-08-26 | SUPERSEDED — explicit marker in status line |
| `multi-server-cognition-expansion-v1.md` | Multi-Server Cognition — Legacy-Stack Displacement + Expansion Readiness (v1) | PROPOSAL — rationale + expansion-readiness. NOT committed deliverables | v1 | D-CSV-6, D-SDR-26 | - | 2026-05-27 | OPEN |
| `normalized-entity-holy-grail-v1.md` | normalized-entity-holy-grail-v1 — typed unified normalization + Op chain over OGIT/OWL/DOLCE/Odoo | PROPOSAL. The trunk that unifies prior workspace work | v1 | D-NEH-1, D-ODOO-BP-1, D-ODOO-EXT-1 | #411, #420, #426, #427 | 2026-05-28 | OPEN |
| `north-star-integration-v1.md` | North-Star Integration — current state → the two-ViewAngle destination (v1) | RATIFIED (council resolved + gates ratified 2026-06-01) | v1 | D-VIEW-1 | #364, #366, #446, #448, #450 | 2026-06-01 | OPEN — the council DECISION is ratified, but the plan itself says it "enumerates open WIRING DECISIONS" before the actual A3→C7 run; D-VIEW-1 (Shipped) is only one item of the ladder |
| `ocr-canonical-soa-integration-v1.md` | OCR → Canonical SoA Integration v1 | PLANTED 2026-06-15 — design only | v1 | D-OCR-2, D-OCR-50, D-OCR-51, D-OCR-52, D-OCR-53 | #496, #498, #500 | 2026-06-16 | OPEN — D-OCR-50 partially shipped (#498); 51/52/53 and 3 named open decisions (§8) remain |
| `ocr-probes-v1.md` | OCR Transcode — Gating Probes v1 | PLANTED 2026-06-16 — from the 5-specialist framing of #497 | v1 | D-OCR-53 | #459, #496, #497, #498 | 2026-06-16 | OPEN |
| `octopus-causal-cot-audit-v1.md` | Octopus — measurement-first related-work + contract audit | MEASUREMENT REPORT. "No code. No new type. No rename." 2026-08-26 | v1 | D-ACR-7, D-AIF-1, D-AIF-2, D-ECG-1, D-ECG-4, D-ECG-6, D-OCT-1, D-OCT-11 | #1057, #1058 | 2026-08-26 | CLOSED — self-contained finished report by its own explicit claim |
| `odoo-business-logic-blueprint-v1.md` | odoo-business-logic-blueprint-v1 — typed Odoo entity DTOs | PROPOSAL. PREREQUISITE for `odoo-savant-reasoners-v2` Group F | v1 | D-ODOO-BP-1, D-ODOO-SAV-1, D-ODOO-SAV-2 | #407, #411..414, #416, #418..420 | 2026-05-28 | OPEN — the Group F it prerequisites is itself Queued |
| `odoo-classes-bitmask-render-v1.md` | odoo-classes-bitmask-render-v1 — the bounded-weekend fix classes.md prescribes | PLAN (pre-council). Authored 2026-05-30 | v1 | D-ARM-7, D-CHESS-BRINGUP-1, D-CLS-1..9, D-CLS-X | #436, #438, #439 | 2026-05-31 | OPEN — D-ARM-7 cited as "HARD PREREQ" and is still Queued |
| `odoo-savant-reasoners-v1.md` | odoo-savant-reasoners-v1 — lance-graph side of the Odoo richness harvest | PROPOSAL. Picks up the explicit cross-repo handover boundary | v1 (higher sibling v2 exists) | D-ODOO-SAV-1..4 | #412, #413 | 2026-05-27 | SUPERSEDED — v2 sibling exists; v2's own text says "v1 SHIPPED in PR #420" before v2 supersedes it as the active plan |
| `odoo-savant-reasoners-v2.md` | odoo-savant-reasoners-v2 — reshape: `Reasoner` trait → typed composition | PROPOSAL. v1 SHIPPED in PR #420 | v2 | D-ODOO-SAV-1, D-ODOO-SAV-4, D-ODOO-SAV-5 | #411, #414, #416, #418..420 | 2026-05-28 | OPEN — D-ODOO-SAV-5a..5e are all Queued on the board |
| `odoo-savant-roster-v1.md` | odoo-savant-roster-v1 | PROPOSAL (the lance-graph side of the woa-rs Odoo savant delegation) | v1 | D-ODOO-1..5 | #412, #413 | 2026-05-27 | OPEN |
| `odoo-source-extraction-v1.md` | odoo-source-extraction-v1 — TIER-1 Odoo source extraction | SHIPPED (Stage 1 complete 2026-05-28; EXT-1..6 landed; 48/53 entities backed; 5 exemptions documented) | v1 | D-ODOO-BP-1, D-ODOO-EXT, D-ODOO-EXT-1..6 | - | 2026-05-28 | CLOSED — explicit SHIPPED with documented, accounted-for exemptions (not silent incompleteness) |
| `ogar-ar-shape-endgame-v1.md` | ogar-ar-shape-endgame-v1 — increments + gates for the OGAR ontology compiler | PLAN (pre-council); becomes PLAN-RATIFIED after 5+3 verdict | v1 | - (D-ids are Inc-numbers, no board rows minted) | - | 2026-06-19 | OPEN — §11.3 shows all Incs PLAN-RATIFIED except Inc4 (conditional) and Inc5 (F5-real DEFERRED); no PR numbers cite the Incs landing |
| `ogar-sink-in-and-consumer-bridge-removal-v1.md` | Plan v1 — OGAR sink-in + consumer-bridge removal | PROPOSED (doc-only plan). Most of the mechanical migration is already shipped | v1 | D-OVC-1, D-SINK-1..5 | #29, #95..98, #105, #106, #109..111, … (+19 more) | 2026-06-30 | OPEN — underlying bridge-deprecation mechanism is shipped, but this plan's own D-SINK-2/3/5 (finish pull-migration, delete aliases, codebook mint) are not confirmed landed |
| `ogar-vocab-contract-codebook-migration-v1.md` | Migration — OGAR `ogar-vocab` codebook ⇄ `lance-graph-contract` classid (v1) | SHIPPING (2026-06-20). Operator signed off §5; D-OVC-1/2/4 landed | v1 | D-GV2-2, D-OVC-1..5 | #557, #560 | 2026-06-20 | CLOSED — §5 decisions all RESOLVED, D-OVC-1/2/4/5 landed per file body |
| `ogit-cascade-supabase-callcenter-v1.md` | OGIT-Cascade · Supabase Realtime · Callcenter Membrane — v1 | plan, not implementation | v1 | D-CASCADE-V1, D-CASCADE-V1-1..15 | #223, #352 | 2026-05-07 | OPEN |
| `ogit-g-context-bundle-v1.md` | OGIT-G Context Bundle — Tier-1 sub-plan (v1) | NONE (no explicit Status line; APPEND-ONLY governance sub-plan doc) | v1 | D-CASCADE-V1-7, D-OGIT-G-1, D-OGIT-G-2, D-OGIT-G-3, D-ONTO-V5-9 | #29, #98, #355 | 2026-05-12 | AMBIGUOUS — no status line; 3 deliverables described with no shipped/closed marker in the file |
| `open-ideas-fetch-v1.md` | Open-ideas fetch — three stale cards re-derived from the tree (v1) | MEASURED / ready-to-execute — PLANNING ONLY in this PR | v1 | D-OIF-0, D-OIF-1, D-OIF-1-DEC, D-OIF-2, D-OIF-2-DEC, D-OIF-3..7 | #76, #119, #121, #288, #301, #1171, #1185, #1188 | 2026-09-05 | OPEN — mixed board statuses: D-OIF-1 Superseded, D-OIF-1-DEC Withdrawn, most others Queued |
| `oracle-funnel-probe-v1.md` | oracle-funnel-probe-v1 — PROBE-ORACLE-FUNNEL: generate→validate→segment, staged | Stage 0 PRE-REGISTERED 2026-08-05 | v1 | D-BLW-5, D-RLG-1 | #241, #244, #245 | 2026-08-05 | OPEN — D-BLW-5 board status is "PAUSED by operator" (loop half) / payload half partially relaunched; D-RLG-1 "GATED (operator word + API)" |
| `oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` | Oxigraph + AriGraph + Cognitive Shader SoA Merge v1 | "architecture context / implementation prompt … not a claim that the implementation already exists" | v1 | - | - | 2026-05-05 | OPEN — pure architecture prompt, no D-ids, no PRs |
| `palantir-parity-cascade-v2.md` | Palantir Parity · Cascade · DTO Ladder — v2 | plan, not implementation | v2 (no higher sibling) | D-CASCADE-V1-2, D-CASCADE-V1-7, D-PARITY-V2-1..15 | #223, #272, #352 | 2026-05-07 | OPEN |
| `persistence-artifact-backed-commit-v1.md` | persistence-artifact-backed-commit-v1 — the canonical persistence contract | RATIFIED (operator ruling 2026-08-09). Phase A **implemented** on | v1 | - | #911 | 2026-08-09 | OPEN — only Phase A confirmed implemented; status wording implies later phases pending |
| `persistence-cycle-wal-bootstrap-v1.md` | persistence-cycle-wal-bootstrap-v1 — the primitive cycle/WAL seam | ACTIVE (bootstrap SHIPPED in PR #878; upgrade phases PLANNED) | v1 | D-MBX-A6, D-MBX-A6-P1 | #878, #912 | 2026-08-09 | OPEN — bootstrap shipped (D-MBX-A6-P1 Shipped) but upgrade phase D-MBX-A6 is Queued |
| `persistent-nars-kg-v1.md` | Persistent NARS Knowledge Graph — Integration Plan v1 | ACTIVE (2026-07-19) | v1 | D-GRAPH-1, D-INFER-DEDUCTIONS-RELATION-BLIND, D-TRUTH-1 (house-style citations, none found on board — see Mismatches) | #10, #11 | 2026-07-19 | OPEN |
| `pillar11-signature-certification-unification-v1.md` | pillar11-signature-certification-unification-v1 | W0-W4 SHIPPED, W5 DEFERRED (trigger measured, not fired) | v1 | D-SK-A | #289, #291, #350, #1111 | 2026-09-03 | CLOSED — main waves shipped; W5 is an explicit, non-blocking deferral (trigger threshold not yet crossed by real data); follow-up closed 2026-09-03 |
| `polyglot-container-query-membrane-v1.md` | Polyglot Container Query Membrane — SurrealQL AST + DataFusion UDF + Cypher (v1) | RESEARCH MAP + INTEGRATION PLAN. Grounded 2026-06-09 | v1 | D-IDENTITY-1, D-IDENTITY-2, D-MBX-6, D-PG-1..7 | #418, #482, #490 | 2026-06-13 | OPEN — D-PG-1/2/3 + D-MBX-6 Queued on board; a LATER plan (`soa-migration-diff-resolution-2026-06-13.md`) self-flags this one as "superseded in spirit" (author's own words in PR #484), but this file itself carries no such marker |
| `post-438-integration-options-v1.md` | post-438-integration-options-v1 — what to do next (council recalibration input) | OPTIONS LIST (pre-council). Composed 2026-05-30 | v1 | D-ARM-1, D-ARM-13, D-ARM-2, D-ARM-3 | #435..439 | 2026-05-30 | OPEN — D-ARM-1/2/3 Queued, only D-ARM-13 Shipped |
| `post-teardown-buildup-survey-v1.md` | Post-teardown buildup survey v1 — the six families as ingredients | SURVEY, read-only, plan-only (no code, no tenant, no ClassView…) | v1 | D-DCR-1, D-DCR-2, D-POP-1, D-POP-2 | #10, #298, #1134 | 2026-09-03 | OPEN — D-DCR-1/D-POP-1/D-POP-2 Shipped but D-DCR-2 In PR; the survey format is deliberately open-ended |
| `probe-excel-compute-dag-v1.md` | probe-excel-compute-dag-v1 — land `ClassView::compute_dag` on the clean 2-axis grid | CONJECTURE / probe scope. The named first proof for the one Core gap | v1 | - | - | 2026-06-18 | OPEN — not yet run |
| `probe-r2il-live-regfile-v1.md` | PROBE-R2IL-LIVE-REGFILE — plan v1 | GREEN (2026-08-26) — 18/18, seven disable runs red-then-green | v1 | - | - | 2026-08-26 | CLOSED — explicit GREEN, 18/18, disable-run discipline followed |
| `probe-revision-attention-view-1.md` | PROBE-REVISION-ATTENTION-VIEW-1 | NONE (no explicit Status line) | - | D-EDIT-ROUNDTRIP-1 (house-style citation) | #1000 | 2026-08-23 | AMBIGUOUS — thesis/probe doc with no recorded pass/fail outcome in the file |
| `q2-foundry-integration-v1.md` | Q2 Foundry-Equivalent Integration Plan — v1 | Proposed (2026-04-24) | v1 | - | #253..260 | 2026-04-24 | OPEN |
| `r2il-bpe-typed-genetic-recombination-v1.md` | R2IL × BPE as typed genetic recombination over the autopoiesis PALETTES (v1) | PROPOSAL, §7's three falsifiers now RUN | v1 | D-PERSONA-5 | #998 | 2026-08-24 | OPEN — the 3 falsifiers in §7 ran green, but the core recombination mechanism (§3–§5: the 4 operators, contract-checking pass, v3 admission loop) is explicitly still "unbuilt/unprobed" |
| `r2il-machine-semantic-contract-v1.md` | R2IL as the machine-semantic contract — documentation, plan, status, integration | PLAN (v1, 2026-08-25). One measured finding underneath | v1 | - | #285..287, #879, #998, #1012, #1013, #1023 | 2026-08-31 | OPEN — a live, long-running pre-registered probe queue through Q8b; Q2/Q3/Q4/Q5 still unresolved, Q3 is an external OGAR-mint gate |
| `reliability-checklist-arc-v1.md` | reliability-checklist-arc-v1 — integration plan as a LIST OF POSSIBILITIES | PROPOSAL / possibility menu (2026-05-30). NOT a committed sequence | v1 | D-MBX-11, D-MBX-A6, D-MBX-A6-P3 | #433, #439, #445 | 2026-06-14 | OPEN |
| `rosetta-codebook-convergence-v1.md` | rosetta-codebook-convergence v1 — the Bible Rosetta SoA + multi-codebook qualia agreement | PROPOSED (doc-only; D-RCC-1 is the gate, runnable today) | v1 | D-RCC-1..8, D-SCI-1 | #849, #850 | 2026-07-26 | OPEN — D-RCC-2/4/5/6 Queued |
| `rubicon-loco-rung-cognitive-fabric-v1.md` | rubicon-loco-rung-cognitive-fabric-v1 — one clockwork fabric, or a measured NO | PROPOSED — PLAN/BOARD ONLY. SOURCE-FIRST | v1 | D-ACR-1, D-ACR-2, D-ACR-3, D-ACR-7, D-ACR-8, D-R2IL-1, D-R2IL-5, D-RLR-1, D-RLR-2, D-RLR-4, D-RLR-5, D-RLR-6 | #70, #561, #565, #590, #879, #1074..1076, #1112, #1152, #1155, #1157 | 2026-09-03 | OPEN — the plan's own Wave-1 ladder (D-RLR-0..6) is entirely Queued or HELD on STATUS_BOARD; §I's "smallest first Wave" has not run |
| `rung-ladder-grounding-v1.md` | rung-ladder-grounding-v1 — ground agichat's RungShift ladder + CollapseGate as LE-contract | PROPOSAL | v1 | D-RUNG-1..4 | - | 2026-05-26 | OPEN — no board rows found for these ids |
| `rung-mul-grounding-v1.md` | rung-mul-grounding-v1 — the MUL fine-tuned into the ladder as an experience curve | PROPOSAL (follow-on to `rung-ladder-grounding-v1`) | v1 | D-RUNG-MUL-1..5 | - | 2026-05-26 | OPEN |
| `rung-persona-orchestration-v1.md` | rung-persona-orchestration-v1 — time-bound persona orchestration | PROPOSAL (sibling to `rung-mul-grounding-v1`) | v1 | D-PERSONA-1..6, D-RUNG-MUL, D-RUNG-MUL-1 | - | 2026-05-26 | OPEN — D-PERSONA-1 Shipped, 2/3/4/6 Queued, 5 blocked (mixed, mostly open) |
| `scientific-kg-substrate-v1.md` | Scientific-KG substrate — crawl → OCR → terms → reason → MUL (v1, scoping) | PROPOSED — scoping doc, no code | v1 | D-SCI-1..4, D-SCI-INSIGHT, D-SRS-1, D-SRS-3 | - | 2026-07-23 | OPEN — D-SCI-2 Queued, D-SCI-3 Blocked; D-SCI-INSIGHT/D-SRS-1/3 Shipped (mixed) |
| `self-reasoning-substrate-v1.md` | Self-Reasoning Substrate — the graph reasoning about itself (v1) | PROPOSED — doc-only. No code, no contract change, no build surface | v1 | D-SRS-1..4 | - | 2026-07-23 | CLOSED — see Mismatches: the status line is stale, all four D-SRS-* deliverables read Shipped on STATUS_BOARD with real file+test citations |
| `singleton-to-snapshot-nudge-v1.md` | Plan: Singleton → Snapshot Nudge | PROPOSAL | v1 | D-MBX-3, D-MBX-5, D-SNGL-1..7, D-SOA-SNAP-1, D-SOA-SNAP-2, D-SOA-SNAP-5 | - | 2026-06-07 | OPEN — D-SNGL-1/2/4 + D-MBX-3/5 Queued, D-SNGL-3 In progress |
| `soa-32-tenant-awareness-redundancy-v1.md` | Plan — 32-tenant 512-byte SoA (M20) | DRAFT (envelope-auditor gate pending) | v1 | D-MTS-6, D-TRI-1, D-TRI-2 | #717, #722, #725, #727, #729 | 2026-07-21 | OPEN |
| `soa-centroid-attention-field-synthesis-v1.md` | SoA Centroid Attention Field — Unified Synthesis v1 | PLANTED 2026-06-15. Gated on `cycle-coherent-soa-snapshot-v1` | v1 | D-OCR-53 | #495 | 2026-06-16 | OPEN — explicitly gated on a plastic-field COW mechanism that is itself still-authoritative/unimplemented; §6 lists 2 open decisions |
| `soa-migration-diff-resolution-2026-06-13.md` | SoA migration diff resolution — 2026-06-13 | NONE (meta-resolution/audit doc; per-plan status table instead of a top status) | - | D-CSV-1, D-IDENTITY-2, D-MBX-A1, D-MBX-A2, D-SNGL-3 | #383, #384, #386, #434, #470, #477..480, #482, #484, #486..490 | 2026-06-13 | CLOSED — a one-time, self-contained audit snapshot as of its own date; its own deliverable is the resolution itself |
| `soa-value-tenant-migration-v1-harvest.md` | SoA Value-Tenant Migration — Phase-1 Harvest Result | HARVEST (2026-06-25, cont.⁴³). Explicitly "NOT the migration" | - | D-GV2-1 | #500, #610 | 2026-06-25 | CLOSED — the Phase-1 harvest deliverable this doc commissioned is delivered, even though the overall migration continues in v2 |
| `soa-value-tenant-migration-v1.md` | SoA Value-Tenant Migration — Plan v1 (harvest brief + 5+3 sign-off) | BRIEF (2026-06-24). This is NOT the migration | v1 (higher sibling v2 exists) | - | #223, #477, #496, #500, #509, #511, #513, #605, #607 | 2026-06-25 | SUPERSEDED — v2 explicitly "Supersedes the v1 BRIEF's implicit single-pass framing" |
| `soa-value-tenant-migration-v2.md` | SoA Value-Tenant Migration v2 — Operator-Locked Phase Sequencing | OPERATOR-LOCKED SEQUENCING (2026-06-25). Supersedes v1's framing | v2 | D-ENVELOPE-PARSER (house-style citation) | #128, #613 | 2026-06-25 | OPEN — current version; Phase 2 gated on the two 5+3 panels |
| `splat-native-ultrasound-v1.md` | splat-native-ultrasound-v1 — integration plan (debt + future) | PROPOSAL / integration plan. Design-spec only; no code in this plan | v1 | D-MBX-10, D-MBX-A2, D-OSM-2, D-SPLAT-1..14 | #17, #25, #30, #31, #39, #162, #189, #426, #432..434, #467, #470, #7474†, #9663† | 2026-06-05 | OPEN — D-SPLAT-1/10/11 + D-MBX-10/D-OSM-2 all Queued |
| `sprint-5-through-9-roadmap-v1.md` | Sprint 5–9 Roadmap — D-SDR Follow-up to FMA Convergence to Compliance Certification | v1 plan, 2026-05-13 evening | v1 | D-SDR-1, D-SDR-2, D-SDR-3, D-SDR-39, D-SDR-6 | #363 | 2026-05-13 | OPEN — no explicit closure marker found; the plan is old (2026-05-13) relative to the 2026-09-07 session date and likely stale, but that is inferred, not confirmed |
| `sql-spo-ontology-bridge-v1.md` | SQL ↔ SPO Ontology Bridge — v1 Implementation Plan | Active | v1 | - | #308 | 2026-04-30 | OPEN |
| `streaming-arm-nars-discovery-v1.md` | streaming-arm-nars-discovery-v1 — Streaming association-rule discovery → NARS revision | PROPOSAL / integration plan. Spec only; no code in this plan | v1 | D-ARM-1..12, D-MBX-10, D-MBX-7, D-MBX-A1..3, D-ODOO-BP-1, D-ODOO-EXT-2 | #418, #433, #434 | 2026-05-29 | OPEN — D-ARM-1/2/3/10/11/12 all Queued |
| `substrate-comfort-zones-v1.md` | substrate-comfort-zones-v1 — where does each substrate formula feel at home? | RUN, §7 — the pre-registered hypothesis is REFUTED | v1 | D-CZ-0, D-CZ-1, D-CZ-2 (+ D-STRATUM-1, house-style citation) | #926..928, #930, #932, #936, #941, #944, #945, #947 | 2026-08-13 | CLOSED — a completed, definitive measurement (refutation is a result, not an unfinished state) |
| `supabase-subscriber-v1.md` | Supabase-shape Subscriber Flow Wire-up — v1 | In progress (2026-04-24) | v1 | - | - | 2026-04-24 | AMBIGUOUS — very old, generic "In progress" status with no later evidence checked this pass; likely stale but not verifiable without a deeper read |
| `super-domain-rbac-tenancy-v1.md` | Super-Domain RBAC + Multi-Tenancy — v1 | Active | v1 | D-ONTO-V5-5, D-SDR-1, D-SDR-10..15, … (40 total, many not board-tracked) | #275 | 2026-05-13 | OPEN — large multi-phase plan cited widely by other plans; many D-SDR-* rows remain Queued elsewhere on the board |
| `tarski-markov-hhtl-seam-v1.md` | TARSKI-MARKOV-HHTL — open questions register (NOT a plan) | HELD — OPEN QUESTIONS ONLY. Proposes no mechanism, licenses no work | v1 | D-WITNESS-1 (house-style citation) | #1007 | 2026-08-23 | AMBIGUOUS — explicitly "not a plan"; contains one withdrawn proposal plus questions that remain genuinely open |
| `temporal-markov-and-style-classes-v1.md` | temporal-markov-and-style-classes-v1 — the ratified 2026-07-10 cognition arc | ACTIVE (operator-ratified 2026-07-10) | v1 | D-CCF-4, D-EPIPHANY-SIG-1, D-MTS-1..6, D-ORG-1, D-ORG-2, D-SF-FILTER, D-TSC-1..4, D-TTV-1 | #11 | 2026-07-17 | OPEN — see special note; the gating probes D-MTS-1/2/3 are all still Queued on STATUS_BOARD, so the code migration this plan gates has not started |
| `tesseract-rs-ast-dll-codegen-v1.md` | tesseract-rs — AST-DLL C++→Rust Codegen Harness v1 | PLANTED 2026-06-15 v2 — layout IS in scope | v1 | D-OCR-30, D-OCR-31, D-OCR-40, D-OCR-41, D-OCR-42 | #498 | 2026-06-16 | OPEN |
| `tesseract-rs-layout-transcode-v1.md` | tesseract-rs — Layout (textord/ccstruct) 1:1 Transcode v1 | PLANTED 2026-06-15. FAITHFUL 1:1, raw-pointer where C++ is intrusive | v1 | D-OCR-30, D-OCR-31 | - | 2026-06-15 | OPEN |
| `tesseract-rs-recodebeam-transcode-v1.md` | tesseract-rs — recodebeam Decoder 1:1 Transcode v1 | PLANTED 2026-06-15. Decoder transcoded 1:1 over HOSTED posteriors | v1 | D-OCR-16, D-OCR-21, D-OCR-40 | - | 2026-06-15 | OPEN |
| `tesseract-rs-traineddata-gguf-v1.md` | tesseract-rs — traineddata → GGUF → embedanything Host v1 | PLANTED 2026-06-15. The LSTM is HOSTED, not transcoded | v1 | D-OCR-10, D-OCR-15, D-OCR-16, D-OCR-21, D-OCR-40 | - | 2026-06-15 | OPEN |
| `tesseract-rs-transcode-master-v1.md` | Tesseract → tesseract-rs — 1:1 Transcode Master Plan v2 | PLANTED 2026-06-15 v2 — design locked | v1 (filename; body is "v2" of the design) | D-OCR-10, D-OCR-15, D-OCR-16, D-OCR-21, D-OCR-30, D-OCR-31, D-OCR-40, D-OCR-42, D-OCR-50, D-OCR-52, D-OCR-53, D-OCR-NN | #498 | 2026-06-16 | OPEN — design locked but implementation not complete per file (a separate `tesseract-rs` repo carries the real transcode progress, per its own CLAUDE.md, not tracked on this board) |
| `tetrahedral-epiphany-splat-integration-v1.md` | Tetrahedral Epiphany Splat Integration Plan v1 | "integration plan / architecture proposal. No runtime implementation is claimed here" | v1 | - | - | 2026-05-02 | OPEN |
| `thinking-engine-harvest-closure-v1.md` | thinking-engine harvest & closure v1 — harvest every gem, retire the residue, close the chapter | PROPOSAL, plan-only (W0 = this census; no code moves in this PR) | v1 | D-ATOM-1, D-CSV-9, D-HOUSE-4, D-MBX-A6, D-PERSONA-1..6, D-PERT-1, D-REUNIFY-2..6, D-TEH-0..5, D-TRI-1, D-TRI-6, D-TSC-1, D-TSC-3, D-TTV-1, D-V3-W4 | #372, #387, #630, #717, #1051, #1137, #1142..1144, #1151 | 2026-09-03 | OPEN — W0 (this census) is done; the actual per-file fate-assignment / retirement work has not run |
| `thought-cycle-soa-awareness-integration-v1.md` | ThoughtCycleSoA awareness integration plan v1 | "Status: integration plan. No implementation claimed here" | v1 | - | #305, #322, #323 | 2026-05-01 | OPEN |
| `token-value-tenant-v1.md` | token-value-tenant-v1 — a byte-exact span address inside the 40,767-triple stream | PROPOSED — PLAN/BOARD ONLY. Measure-before-carve | v1 | D-TVT-0, D-CONSUMERS-1 (house-style citation) | #10 | 2026-08-28 | OPEN |
| `transcode-extend-core-probe-v1.md` | Transcode EXTEND-CORE + in-env probe slice — v1 (UNDER COUNCIL REVIEW) | PROPOSAL — under 5-consolidate + 3-brutal council review (2026-06-17) | v1 | D-CPP-CODEGEN-1 | #17 | 2026-06-17 | OPEN |
| `triangle-tenants-gestalt-separation-v1.md` | Triangle tenants × separated awareness surfaces × chess quarantine — v1 | DESIGN (operator-directed 2026-07-17; no bytes land with this) | v1 | D-SF-OPPONENT-1, D-SF-OPPONENT-3, D-TRI-1..6, D-TSC-1 | - | 2026-07-17 | OPEN — D-TRI-2/3/4/5 Queued, D-TRI-1 gated, D-TRI-6 In PR |
| `unified-bridge-consumer-migration-v1.md` | Unified-Bridge Consumer Migration — v1 | Draft | v1 | D-ONTO-V5-9, D-SDR-1, D-SDR-2, D-SDR-35, D-UB-1..14 | #355, #407, #408 | 2026-05-25 | OPEN |
| `unified-integration-v1.md` | Plan — Unified Integration: PersonaHub × ONNX × Archetype × MM-CoT × RoleDB | Active — brainstorm phase complete; deliverables defined; no code shipped yet | v1 | - | - | 2026-04-24 | OPEN — explicitly "no code shipped yet" |
| `unified-ogit-architecture-v1.md` | Unified OGIT Architecture — v1 (Master Synthesis) | NONE (master synthesis doc, no single top Status line) | v1 | - | - | 2026-05-12 | OPEN — no top status; §7 "Honest Self-Assessment" lists substantial not-yet-started items itself (no v2 written) |
| `unified-soa-convergence-v1-addendum-2026-05-29-review.md` | unified-soa-convergence-v1 — review addendum (post-merge, 2026-05-29) | REVIEW NOTES — review of the merged plan after PR #434 landed | - | D-MBX-11, D-MBX-9, D-MBX-A1, D-MBX-A2 | #434 | 2026-06-12 | CLOSED — its own §10 in-place edits were applied in the same commit; a self-contained, delivered addendum |
| `unified-soa-convergence-v1.md` | unified-soa-convergence-v1 — THE single little-endian SoA, end-to-end | PROPOSAL / integration plan. Design-spec only; no code in this plan | v1 | D-CE64-MB-1, D-CSV-10, D-CSV-7, D-MBX-2..12, D-MBX-A1..6, D-ODOO-2, D-ODOO-SAV-4, D-PERSONA-5 | #388, #414, #416..418, #433, #434, #445, #477, #490 | 2026-06-14 | OPEN — its own ERRATA note says the §4.2 stack-pin table is stale and points readers to `soa-migration-diff-resolution-2026-06-13.md` for the fuller diff; §1 rulings remain authoritative but substantial content is stale without a top-level SUPERSEDED marker |
| `unified-soa-rubikon-integration-v1.md` | Unified SoA — the Rubikon-model integration (v1) | "Status legend: SHIPPED / PARTIAL / PROPOSED" per-item (no single top status) | v1 | D-MBX-11, D-MBX-12, D-MBX-7, D-MBX-8, D-MBX-9, D-MBX-A6, D-MBX-A6-P3 | #437, #439, #477, #557 | 2026-06-20 | OPEN — a per-item legend, not a closed plan; D-MBX-7/8/9/12/A6 remain Queued |
| `v3-convergence-wiring-v1.md` | V3 Convergence Wiring — Plan v1 (wire, don't invent) | ACTIVE (2026-07-01) | v1 | - | - | 2026-07-02 | OPEN — D1a/D1b/D2/D6 shipped in-session per the file's own D-id table, but D3 Queued, D4 Planned, D7 a "deferred frontier" |
| `v3-substrate-integration-v1.md` | v3-substrate-integration-v1 — pointer stub | POINTER STUB — plan body lives at `.claude/v3/INTEGRATION-PLAN.md` | v1 | - | - | 2026-07-02 | OPEN — redirects to the still-very-active `.claude/v3/` program (see `CLAUDE.md` §V3 SUBSTRATE) |
| `w3-template-mask-v1.md` | W3 — the grammar template as StepMask: no finetuning, three tables — v1 | PROPOSED (doc-only). Gated on D-W3M-1 — no further W3 work | v1 | D-W3M-1, D-W3M-2, D-W3M-3 | - | 2026-07-22 | OPEN — all three Queued |
| `weather-soa-bake-v1.md` | weather-soa-bake-v1 — Zarr → NodeRow, the missing bake | PLAN (2026-08-13). Doc-only. No code shipped, no board file written BY THIS PLAN (own §11 lands its board rows in the same commit) | v1 | D-CZ-0, D-CZ-1, D-CZ-8, D-KIA-A2, D-WXA-1..5, D-WXB-1, D-WXB-4, D-WXC-1, D-WXS-0..12 | #460, #504, #901, #907, #920, #950 | 2026-08-13 | OPEN — its own D-WXS block is mostly Queued, D-WXS-0 explicitly Blocked (operator/OGAR classid mint) |
| `weather-substrate-evaluation-v1.md` | Evaluation Plan — Weather Normalized Substrate: KNOWN vs TO TEST (v1) | ACTIVE (audited 2026-08-11 — §8) | v1 | - | #498, #915, #917, #920..922 | 2026-08-11 | OPEN |
| `weather-substrate-poc-v1.md` | Weather Substrate POC — Plan v1 (encoder bake-off first, forecast second) | PROPOSED (doc-only). No code in this PR | v1 (higher sibling v2 exists) | D-BLW-5, D-HWV-1, D-WX-0..6 | #460, #879, #907 | 2026-08-10 | SUPERSEDED — v2 explicitly "Supersedes `weather-substrate-poc-v1.md`" |
| `weather-substrate-poc-v2.md` | Weather Substrate POC — v2 (jc-gated: representation → hardware → prediction) | PLAN (2026-08-10). Supersedes v1 (PR #914) | v2 (current) | D-WX-0, D-WXA-1..5, D-WXB-1..4, D-WXC-1..5 | #460, #879, #900..902, #907, #913, #914 | 2026-08-11 | OPEN — current version of the weather POC plan |
| `weather-w-probes-v1.md` | weather-w-probes-v1 — the W series as self-contained Sonnet worker briefs | ACTIVE for exploratory probes (W5, W2s-a, W6, W2s-b, W7) | v1 | - | #921, #926, #936, #938, #940 | 2026-08-12 | OPEN |
| `wikidata-lazy-spine-hydration-v1.md` | IMPLEMENTATION PLAN: Wikidata lazy-spine hydration v1 | QUEUED (all D-ids) | v1 | D-ARM-14, D-ARM-7, D-LWS-1..9 | #437, #441, #442, #445 | 2026-06-14 | OPEN — explicitly "nothing shipped yet" |

`*` = `lance#8206`/`lance#8589` are references to issue/PR numbers in the
**upstream `lance` crate's own GitHub repo**, not this repo's PRs — captured
by the `#\d+` regex without repo context; flagged so the number isn't
misread as a `lance-graph` PR.
`†` = `#7474`/`#9663` in `splat-native-ultrasound-v1.md` read as
out-of-range for this repo's PR numbering (which tops out in the low
1200s elsewhere in this pass) — not individually verified against GitHub
this session; likely also an external/upstream citation or a numeric
artifact, flagged rather than asserted.

## Special-attention notes

**`measure-64k-axes-v3.md`** (read in full). This is the third of a four-part
same-day experiment series (v1→v4, all dated 2026-08-05). v3 defines two
arms: **M-arm** (insert a Morton reorder before the seal/WAL write) and
**O-arm** (source the write-side row order from replay instead of the seal).
Both ran. M-arm's own framing requires the verdict to be the *sum*
(`Δtotal = reorder_cost − downstream_savings`), not the reorder cost alone —
the file doesn't give the final number, deferring it to v4. O-arm's result is
a **semantic failure, not just a slow one**: presorting cannot replace the
seal (digest divergence measured), so the standing position becomes "keep
`temporal.rs` as the read-side ordering authority and the seal as the
write-side one, and treat the gap between them as an explicit open research
question." What's left open, concretely: the M-arm net verdict (deferred to
v4), and the specific question "what does the seal compute that
`temporal.rs` does not encode" (four items named, three probes
pre-registered, tracked in a separate knowledge doc, not run in this file).

**`ocr-canonical-soa-integration-v1.md`.** Design-only sub-plan mapping OCR
tokens onto the canonical `NodeRow`. D-OCR-50 (token→NodeRow mapping) is
partially shipped via `ocr.rs`'s `LayoutBlock::to_node_row` (#498), but three
concrete gaps remain even within D-OCR-50 itself: token-grain nodes don't
exist yet (only block-grain), the HHTL layout-trie is unpopulated (0 depth),
and the OGAR OCR class was never minted (still riding the `0x0000_0000`
bootstrap classid). D-OCR-51 (value-tenant preset) and D-OCR-52 (DeepNSM +
CAM/PQ repair wiring) are unstarted. D-OCR-53 (bit-reproducible golden-file
harness) is explicitly gated on 50+51 landing first. Three open decisions in
§8 (dedicated `ValueTenant::OcrEvidence` vs riding existing tenants;
node-per-token vs node-per-line; character-confusion layer's crate home) are
unresolved.

**`ogar-ar-shape-endgame-v1.md`.** A 5-increment (Inc1-5) plan promoting five
CONJECTURE rows in a separate doctrine doc to FINDING via probes. Per its own
§11.3 "Final PLAN-RATIFIED status" table: Inc1/2/3a/3b are PLAN-RATIFIED,
Inc4 is PLAN-RATIFIED-conditional (operator must pick weaken-the-gate vs
defer-until-Odoo-arms), and Inc5 is split — the "smoke" half is ratified but
the load-bearing "F5-real" half (one executor pair — NativeLance or
SurrealAst — must do a REAL write, not a stub, with property-fuzzed binding
values) is explicitly DEFERRED. The doctrine's own §10 "litmus" claim stays
CONJECTURE until F5-real runs. No PR numbers appear in the file citing any
Inc as landed, so as read this plan is still pre-execution.

**`ogar-sink-in-and-consumer-bridge-removal-v1.md`.** Grounds a real, mostly
already-shipped state: all six per-consumer bridges (medcare/odoo/redmine/
smb/openproject/woa) are already `#[deprecated]` one-line aliases over a
single generic `UnifiedBridge<P: PortSpec>` harness — that half is DONE. What
remains open is layer (b): finishing each consumer's migration off the
deprecated alias onto a direct `Port::class_id()` pull, and only then
deleting the aliases (D-SINK-3), which is explicitly gated on that
consumer-side migration (D-SINK-2) plus porting the `bridge_scope_lock.rs`
cross-namespace-leak test onto the surviving path first. The classid realign
(merged-OSINT/FMA hi/lo ordering) is separately operator-gated and this plan
declines to pre-empt it.

**`oxigraph-arigraph-cognitive-shader-soa-merge-v1.md`.** Purely an
architecture-context/implementation-prompt document — the file's own second
line says "this is not a claim that the implementation already exists." No
D-ids, no PRs, no board presence. It sketches a 5-PR-shaped roadmap (schema →
witness lanes → search → prefetch → splat stubs) but nothing in the plan
itself, nor anywhere on the board, shows any of those PRs landing. Fully
open, essentially unstarted relative to this specific merge shape.

**`pillar11-signature-certification-unification-v1.md`.** Unusual in this
set for actually closing out: W0-W4 all measured and shipped, with three of
the plan's OWN pre-registered gates falsified by their own sweeps mid-run and
replaced (documented honestly as "the outcome the probe-first discipline
exists to produce," not hidden). W5 (a memory/time trigger threshold for a
"period-forced" regime) was measured but has **not fired** — the longest
in-tree signature path (4609) is well short of the trigger thresholds
(~11585 / ~33388) — so W5 stays a documented, non-blocking deferral rather
than an open task. A same-day follow-up (Cholesky PSD extension) that had sat
"Queued" for three days after actually shipping was explicitly closed in this
file on 2026-09-03, with a note that a stale Queued row "is a licence for
duplicate work."

**`probe-r2il-live-regfile-v1.md`.** A genuinely GREEN, closed probe (18/18,
seven disable-runs red-then-green) — but its own §9 makes an important
caveat explicit: **the CI workflow that would run this probe has literally
never executed** (`total_count: 0` on the sibling `r2sleigh` repo's GitHub
Actions) because it targets an unregistered self-hosted runner. Every green
number in the file is a *local* run only. The probe explicitly frames this as
the same "a gate that never runs is not a gate" lesson recorded once already
in `tesseract-rs`. Fixing the CI registration is named as the operator's
call, not done in this plan.

**`r2il-bpe-typed-genetic-recombination-v1.md`.** §7's three narrow
falsifiers (do splice points exist / does recombination round-trip / does a
recombined candidate produce a distinguishable counterfactual verdict) all
ran and all answered YES with real measured numbers (107/1056 ordered pairs
admit a splice point; 10 distinguishable recombined sequences plus 5
correctly-silent identity substitutions; a strong recombined splice pair
wins `minority_wins()` against an identical weak majority while an ordinary
baseline correctly does not). But the file is explicit that this proves only
the three narrow questions, not the mechanism: the four recombination
operators (§3), the contract-checking pass (§4), and the v3
admission/revision loop (§5) remain entirely "unbuilt/unprobed." The
counterfactual leg specifically only exercises the primitive that's
actually shipped (`deposit_counterfactual` + `minority_wins()`); the v3
`CounterfactualMailbox`/`revise_if_minority_wins` path is still a `todo!()`
stub, blocked on `D-PERSONA-5`.

**`r2il-machine-semantic-contract-v1.md`.** The largest and most active file
in this half (1427 lines), structured as documentation (§3) + an answer (§4)
+ a long, still-growing, pre-registered probe queue (§6-§11, through "Q8b").
Open items per its own §8.2 unified queue: **Q2** (a cheap 6502-lift
measurement, settles a conditional) is unresolved; **Q3** (the OGAR mint
decision for a custom memory space) is an EXTERNAL gate blocking "W2"; **Q4**
(op-row carving) "may proceed" but hasn't; **Q5** (the ontology-morphogenesis
probe) has its data half owned by a different repo. The most recent
completed work (Q8, run 2026-08-31, then Q8b) actively falsified the plan's
own hex-topology hypothesis: a degree-1 ablation matched the full
hex-topology arm to four decimal places at 5.5× less memory, meaning "the
six-neighbourness does nothing" — the instrument was reverted and the result
is board-only. Q8b then found the plan's own completion-task metric is not
even expressible on the workspace's real def-use carrier (BPE compresses
92-99.7% of chains to a single symbol, leaving no cued position). So the
plan's own probe apparatus, as specified, needs re-design before the next
question can be asked on it.

**`rubicon-loco-rung-cognitive-fabric-v1.md`.** A "one clockwork fabric, or
a measured NO" design doc explicitly marked SOURCE-FIRST (every §A-§J claim
carries a file:line). §I names "the ONE smallest first Wave" (`D-RLR-1`): a
non-rung-4 horizon invoking one existing Frozen atom via `ogar_loco`, writing
one alpha row, emitting a typed receipt, and replaying it through
`temporal.rs`. Per STATUS_BOARD, **every D-RLR-0..6 row is Queued or HELD** —
Wave 1 has not run. §J's 11 named falsifiers (F-RLR-1..11) are all still
live gates, not yet exercised. A prior version of this plan's own §F (rung
storage model) had to be corrected mid-session after being written without
reading the mandatory prior plan (`alpha-channel-rung-overlay-v1.md`) —
recorded as a self-caught process failure, not silently fixed.

**`soa-32-tenant-awareness-redundancy-v1.md`.** DRAFT status, "envelope-
auditor gate pending" — i.e. blocked on a review gate before landing. Its
own §6 is explicit about scope: it does not pre-commit the sibling-lane
count (jc-derived later), does not touch `CausalEdge64`/`EpisodicEdges64`
bit-fields, and does not claim the 64-bit awareness conjecture is wrong —
only that it builds the apparatus to test it. D-TRI-1/D-TRI-2 (the cross-
session convergence points) are Queued/gated on the board.

**`soa-migration-diff-resolution-2026-06-13.md`.** Not itself a plan — an
audit reconciling nine SoA/BindSpace/identity plans against shipped reality
post-#490. Marked CLOSED here because the audit itself is complete and
dated, but it names several OTHER plans as still genuinely open at the time
of writing: `cognitive-write-roundtrip-substrate-v1.md` ("still-authoritative
… blocked on `SoaEnvelope` impls"), `cycle-coherent-soa-snapshot-v1.md`
("still-authoritative … no implementor yet"), and
`singleton-to-snapshot-nudge-v1.md`'s `D-SNGL-3` ("still queued" as of the
audit — confirmed still "In progress" on the current board, three months
later). It also documents a full retirement list (§3.1): `CollapseGateEmission`,
`MailboxSoA::emit()`, the "Baton" carrier type, and the UUIDv8 `NodeGuid`
wrapper were all planned-then-deleted, useful context for anyone who finds
those names in an OLDER plan and assumes they're still current.

**`soa-value-tenant-migration-v1.md` / `-v1-harvest.md` / `-v2.md`.** A
three-document chain: v1 is a BRIEF commissioning a harvest; the harvest doc
is the filled inventory (its own deliverable, delivered); v2 is the
operator-locked sequencing that supersedes v1's "implicit single-pass
framing" with a forced two-phase order (identity/key-side V3 migration MUST
precede value-tenant/value-side reshaping, because the envelope parser
resolves `tail_variant` upstream of `value_schema`). v2's own §5 flags that
Phase 1's substance is largely an OGAR/MedCare-rs cross-repo read the harvest
session missed on a casing bug (`/home/user/ogar` vs the real
`/home/user/OGAR`) — so Phase 1's real content is that corrective sweep, not
yet confirmed done in this file.

**`thinking-engine-harvest-closure-v1.md`.** PROPOSAL, plan-only — its own
scope is explicitly "W0 = this census; no code moves in this PR." The
census itself (§0) is thorough: only two crates depend on
`crates/thinking-engine` in production, and the crate is workspace-excluded
and off CI with ~40 pre-existing clippy lints. Four small process/hygiene
fixes are recorded as already applied within this same census pass (dtype
routing correction, two `pub`→private visibility demotions, a test-count
correction from a stale board citation). But the actual "harvest every gem /
retire the residue" work this plan is named for — assigning a fate to each
of the 51 files — has not started; that's the next session's job.

**`thought-cycle-soa-awareness-integration-v1.md`.** A short, purely
conceptual integration plan ("No implementation claimed here"). Proposes
splitting the old `Vsa16kF32` carrier into `AwarenessPlane16K` /
`GrammarMarkovLens64` / `ReasoningWitness64` / `ThoughtCycleSoA`, all keyed
on shared `thought_id`/`cycle_id`. No D-ids of its own; entirely design
prose. What's left open is everything — this is a naming/shape proposal, not
a build plan with a status ladder.

**`unified-soa-convergence-v1.md`.** The 2026-05-29 foundational SoA plan
("five layered rulings"). Its own top-of-file ERRATA (added post-#490) is
the load-bearing fact: §1's rulings are still cited as authoritative
(anchor for `E-SOA-IS-THE-ONLY`), but §4.2's stack-pin table is stale (lance
bumped past what's written here at least twice since), and the fuller
picture lives in `soa-migration-diff-resolution-2026-06-13.md`. This plan is
therefore best read as "doctrine current, mechanics stale" rather than
either fully open or fully superseded — no single top-level marker settles
it either way, which is itself worth flagging for anyone citing this file's
stack-pin table as current.

**`unified-soa-rubikon-integration-v1.md`.** Uses an explicit per-item status
legend (✅ SHIPPED / ◐ PARTIAL / ☐ PROPOSED) rather than one top status. §6
"Honest status (no overclaim)" cross-references itself against
`unified-soa-convergence-v1.md`'s D-MBX ladder — most of those (D-MBX-7/8/9/
12/A6) remain Queued on the current board. The plan's real headline claim
(SurrealQL as "a co-equal lens on the one SoA, gated only by OQ-11.6") is
gated on uncommenting a fork dependency and filling in one stub function
(`read_via_kv_lance`) — a small, specific, still-open item.

**`v3-convergence-wiring-v1.md`.** ACTIVE, "wire, don't invent" — the
organizing finding is that most V3 substrate gaps are unwired seams, not
missing machinery. Of its 7 named deliverables, D1a/D1b/D2/D6 are marked
shipped in the same session that wrote the plan; D3 (q2 render probe) is
push-gated on a separate repo; D4 (registry consolidation) is merely
planned; D5 is recorded as an open issue (`ISS-Q2-CASCADE3-NIBBLE-ANCESTRY`)
rather than resolved; D7 (rig/rs-graph-llm orchestration loop) is explicitly
named "the only fully unwired angle" and deferred until the operator opens
it — that's the one item genuinely wide open.

**`weather-soa-bake-v1.md`.** Opens by correcting the framing it was
commissioned under: the predecessor plan (`weather-substrate-poc-v2.md`)
named a `D-WXA-5` gate and D-ids that were **never minted onto
STATUS_BOARD.md at all** (`grep -c "WXA"` = 0, independently re-verified by
the orchestrator) — a concrete, named instance of the
`ISS-PLAN-TRACKING-IS-UNENFORCED` board-hygiene gap this repo already
tracks. It further argues that gate's own threshold (Spearman ρ ≥ 0.98) is
"at serious risk of being VACUOUS" because a sibling plan already measured
that ρ *saturates* near 1.0 on smooth pressure fields regardless of
representation quality. This plan's own D-WXS block is Queued end-to-end
except D-WXS-0 (classid mint), which is explicitly Blocked on
operator/OGAR — so nothing in the actual bake has landed yet.

## Counts

| Verdict | Count |
|---|---|
| OPEN | 85 |
| CLOSED | 12 |
| SUPERSEDED | 7 |
| AMBIGUOUS | 4 |
| **Total** | **108** |

CLOSED: `lance9-datafusion54-upgrade-probe-v1.md`, `mul-consumer-census-v1.md`,
`octopus-causal-cot-audit-v1.md`, `odoo-source-extraction-v1.md`,
`ogar-vocab-contract-codebook-migration-v1.md`,
`pillar11-signature-certification-unification-v1.md`,
`probe-r2il-live-regfile-v1.md`, `self-reasoning-substrate-v1.md`,
`soa-migration-diff-resolution-2026-06-13.md`,
`soa-value-tenant-migration-v1-harvest.md`, `substrate-comfort-zones-v1.md`,
`unified-soa-convergence-v1-addendum-2026-05-29-review.md`.

SUPERSEDED: `measure-64k-axes-v1.md`, `measure-64k-axes-v2.md`,
`measure-64k-axes-v3.md`, `mul-gate-outcome-vs-ground-v1.md`,
`odoo-savant-reasoners-v1.md`, `soa-value-tenant-migration-v1.md`,
`weather-substrate-poc-v1.md`.

AMBIGUOUS: `ogit-g-context-bundle-v1.md`, `probe-revision-attention-view-1.md`,
`supabase-subscriber-v1.md`, `tarski-markov-hhtl-seam-v1.md`.

## Mismatches

Two genuine plan-vs-board disagreements found (a much smaller set than the
raw automated pass produced — see Candidate epiphanies for why most
apparent "mismatches" were discarded as false positives from naive keyword
matching):

1. **`self-reasoning-substrate-v1.md`** — the file's own status line reads
   "PROPOSED — doc-only. No code, no contract change, no build surface"
   (2026-07-23). STATUS_BOARD.md shows all four of its own named
   deliverables as Shipped, each with a real source file and test count:
   `D-SRS-1` (Shipped, `src/reason.rs` + 7 tests), `D-SRS-2` (Shipped,
   `src/{shape,ancestry}.rs` + 63 tests), `D-SRS-3` (Shipped — falsifier
   fired, conjecture not confirmed), `D-SRS-4` (Shipped — CONFIRMED
   positive, `src/introspect.rs` + 77 tests). The plan's own header text is
   stale relative to what actually shipped under its own D-ids.

2. **`persistent-nars-kg-v1.md`** and **`self-reasoning-substrate-v1.md`**
   both cite `D-GRAPH-1`, `D-TRUTH-1`, and `D-INFER-DEDUCTIONS-RELATION-BLIND`
   as deliverables — none of the three appears anywhere in
   `STATUS_BOARD.md` (confirmed by direct grep, not just the automated
   D-id map). Either these were never minted as board rows, or they were
   later renamed/absorbed into a different D-id family without a pointer
   left behind in either plan.

No other file in this range showed a plan claiming a specific D-id is
shipped/complete while STATUS_BOARD.md shows that same D-id Queued/Blocked/
Withdrawn — every apparent case surfaced by the first automated pass
resolved, on inspection, to either (a) the plan's own wording already
matching the board (e.g. `persistence-cycle-wal-bootstrap-v1.md` says
"bootstrap SHIPPED … upgrade phases PLANNED," and the board shows exactly
that split), or (b) a keyword false-positive in the crude classifier (see
Candidate epiphanies #1).

## Candidate epiphanies

Facts observed this pass, with counts and filenames, no speculation:

1. **Naive substring/keyword matching on "SHIPPED" misfires repeatedly,
   confirming the exact risk this repo's own `CLAUDE.md` documents for
   `ARCHIVE?` routing.** A crude classifier that flags a plan CLOSED
   whenever its status text contains "SHIPPED" produced false positives on
   at least: `weather-soa-bake-v1.md` (status literally says "**No** code
   shipped"), `unified-integration-v1.md` ("**no** code shipped yet"),
   `unified-soa-rubikon-integration-v1.md` (the word appears only inside a
   ✅/◐/☐ status-legend definition, not as an overall claim), and
   `mul-consumer-build-gate-v1.md` (status says "GATE RUN," which reads as
   complete, but the file's own §4 explicitly labels one falsifier "OPEN —
   not discharged"). Every verdict in the table above was therefore set
   from a full read of the status line in context, not a keyword scan.

2. **The `D-[A-Z]+(-[A-Z0-9]+)+` extraction regex produces real false
   positives by matching the tail of a longer `E-`/`TD-`-prefixed
   sentence-id.** Confirmed 81 false-positive "D-ids" across 43 files this
   pass (e.g. `E-READ-NOT-GREP` yields a spurious `D-NOT-GREP` in
   `north-star-integration-v1.md`; `E-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-
   CONTROL-1` yields a spurious `D-A-CONTROL-THAT-CANNOT-LOSE-IS-NO-
   CONTROL-1` in `mul-consumer-build-gate-v1.md`). All were detected by
   checking whether the character immediately preceding the match is
   alphabetic, and filtered out of the `d_ids` column above. Files most
   affected: `octopus-causal-cot-audit-v1.md` (4), `r2il-machine-semantic-
   contract-v1.md` (8), `thinking-engine-harvest-closure-v1.md` (5).

3. **This repo's own house style also uses a second, legitimate pattern
   that LOOKS like the same false-positive: `D-<FULL-SENTENCE>-N` as a
   real in-plan citation for a finding**, distinct from a STATUS_BOARD
   deliverable row (e.g. `D-IS-BOUND-TO-ITS-STATISTIC-AND-ITS-SAMPLE-1` in
   `pillar11-signature-certification-unification-v1.md`, which does
   correspond to a real, later-cited board finding). These pass the
   word-boundary filter (nothing alphabetic precedes them) but still show
   as "not found" in a literal STATUS_BOARD row lookup, because they're
   findings citations, not deliverable ids. A future automated pass should
   not treat "not found on STATUS_BOARD" as itself evidence of a tracking
   gap without checking whether the id is grammatically a deliverable or a
   sentence-citation.

4. **Some `#NNNN` citations reference an upstream repo's issue/PR tracker,
   not this repo's own.** `lance-convergence-staged-migration-v1.md` cites
   `lance#8206` and `lance#8589` (the upstream `lance` crate's GitHub
   issues) using bare `#NNNN` prose elsewhere in the same paragraph — a
   naive `#\d+` PR-citation regex captures these indistinguishably from
   this repo's own PR numbers. Flagged in the table rather than silently
   conflated.

5. **Six explicit `-v1`→`-v2`(→`-v3`→`-v4`) supersession chains exist in
   this half**, and in every case the higher-numbered file states the
   supersession explicitly in its own status line (never silent): the
   `measure-64k-axes` v1→v2→v3→v4 chain (3 files marked SUPERSEDED, 1
   current), `odoo-savant-reasoners` v1→v2, `soa-value-tenant-migration`
   v1→v2, and `weather-substrate-poc` v1→v2. No case of a stale
   lower-version file *without* a superseding sibling's explicit marker
   was found in this half (contrast with `polyglot-container-query-
   membrane-v1.md` below, which has no version sibling but is
   self-flagged as superseded by a THIRD file).

6. **A "superseded in spirit" self-flag lives in the superseding author's
   PR body, not in the superseded plan file itself, and not reachable by
   filename/version pattern.** `polyglot-container-query-membrane-v1.md`
   carries no `-v2` sibling and no `SUPERSEDED` marker of its own, but a
   LATER, unrelated-filename plan (`soa-migration-diff-resolution-
   2026-06-13.md`, itself in this half) quotes the author of PR #484
   self-flagging it as "superseded in discussion by the self-describing-
   key convergence." This repo's supersession tracking is therefore not
   fully captured by version-suffix conventions alone — some supersessions
   only exist as prose in a third document.

7. **15 files in this half (of 108) have no verbatim `Status:`/`**Status:**`
   line at all**, requiring a fallback read of the opening paragraph or
   (for 2 files) a full read: `ogit-g-context-bundle-v1.md`,
   `probe-revision-attention-view-1.md`, `soa-migration-diff-resolution-
   2026-06-13.md`, `tarski-markov-hhtl-seam-v1.md` (has a bare `Status:`
   without bold markers, on its own line), `token-value-tenant-v1.md`
   (same bare-Status pattern), `unified-ogit-architecture-v1.md`,
   `unified-soa-rubikon-integration-v1.md` (has a "Status legend" instead
   of a single status), `v3-substrate-integration-v1.md`, `wikidata-lazy-
   spine-hydration-v1.md` (bare `Status:` pattern), plus the four
   `measure-64k-axes-v{1,2,3,4}.md` files (all four have zero Status line
   of any form), `mul-ewa-trust-propagation-v1.md`,
   `r2il-bpe-typed-genetic-recombination-v1.md`, and
   `soa-value-tenant-migration-v1-harvest.md` (all three of the last group
   use a bare `Status:` — no `**` bold — that a `\*\*Status:?\*\*` regex
   alone would also miss).

8. **`weather-soa-bake-v1.md` documents, as its own §0.1, a real historical
   instance of exactly the D-id/board-tracking gap this task's inventory
   method is designed to catch**: it names a predecessor plan
   (`weather-substrate-poc-v2.md`) whose entire `D-WXA-*`/`D-WXB-*`/
   `D-WXC-*` deliverable ladder was never mirrored onto `STATUS_BOARD.md`,
   and states this as the plausible mechanism by which "a whole arc ran
   past [a gate] in Python without anyone tripping over an unmet gate."
   This is a first-party confirmation, from inside the corpus, that
   plan-D-ids not appearing on the board is a known, recurring failure
   mode here — not just an artifact of this inventory's own extraction
   method.

## Unread/uncertain

- **`sprint-5-through-9-roadmap-v1.md`** — only the metadata region was
  read (status line + head). It is dated 2026-05-13, four months before the
  current session date (2026-09-07), and cites only D-SDR-* ids that are
  extensively referenced (and partially shipped) across many other, later
  plans in this corpus, but this specific file's own closure state was not
  independently verified against a full read. Classified OPEN rather than
  CLOSED/SUPERSEDED purely because no explicit marker was found, not
  because staleness was ruled out.
- **`supabase-subscriber-v1.md`** — very short metadata read only (status
  line: "In progress (2026-04-24)", the oldest file in this half along with
  a few April 2026 plans). No later citation of this plan was found
  elsewhere in the corpus during this pass, which is itself mildly
  suggestive of abandonment, but that absence-of-citation check was not
  exhaustive (no full-corpus grep run specifically for this filename).
  Marked AMBIGUOUS rather than a guessed CLOSED or SUPERSEDED.
- **`tesseract-rs-*` five files** (`-ast-dll-codegen-v1.md`,
  `-layout-transcode-v1.md`, `-recodebeam-transcode-v1.md`,
  `-traineddata-gguf-v1.md`, `-transcode-master-v1.md`) — all read at
  metadata-region depth only (title/status/D-ids), not in full, because the
  actual transcode progress they describe is tracked in a SEPARATE sibling
  repo (`tesseract-rs`, which has its own extensive `CLAUDE.md` recording
  many shipped leaves — Leaf 1 through the full image-to-text pipeline —
  entirely outside this board's D-id namespace). All five are marked OPEN
  here on the lance-graph side purely because no D-OCR-* row on THIS
  repo's STATUS_BOARD.md confirms closure; the real, much more advanced
  status lives in the other repo and was not cross-checked this pass
  (out of scope for a `.claude/plans/` inventory of this repo).
- **`unified-ogit-architecture-v1.md`** — the 607-line master-synthesis
  file's §0-§4 and §7 (self-assessment) were read; §5 ("Proof of Vision")
  and §6 ("Cross-References") were not read in full. The OPEN verdict rests
  on §7's own admission of open items, which should hold regardless, but a
  full read of §5/§6 was not performed.