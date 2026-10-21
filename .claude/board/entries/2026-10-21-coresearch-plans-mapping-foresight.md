# 2026-10-21 — Co-research: mapping and foresight for the ClassView × Quack plans (#1667)

**Status:** EXPLORATION MAP. Nothing in it is adopted; the operator decides. Inputs: five scouts (code, internal prior art, literature, systems, concepts), a 20-row crosswalk, and three co-architects (bridge, firewall and fit, falsifier). Plans: #1667 (`d-qsp-…`, `d-rft-…`, `d-lfa-…`, not on main). Companion entries from the same run: `2026-10-21-pg-classview-polyfill-map.md` and `2026-10-21-eval-egglog-datafrog-ascent.md`.

## Question

Which steps in the three plans lack a semantic contract, a named oracle, a named failure mode, or an inside surface by file and line? Which failures will the plans predictably hit, judging by how established systems and the literature solved the same problems?

## The main finding

Three outside sources found the same defect independently: **one carrier or one equality used for two strata.** The brief seeded the search terms, so "independently" is weaker than it sounds.

| Reused for two strata | Where it breaks |
|---|---|
| K3 `=` used as the grouping equality | `GROUP BY`, `DISTINCT` and set operations treat NULLs as one group (`IS NOT DISTINCT FROM`) |
| A mask fold's zero used as the aggregate | An empty or all-NULL `SUM` is NULL in PostgreSQL, not 0 |
| A count carrier under recursion | A cycle keeps itself alive after its last external support is deleted |

Inside, four conventions carry four strata and share nothing:
- Quack `sql_where`: K3, WHERE only;
- `JoinExec::bag_count`: bag count;
- `Pop`: distinct, 256 slots;
- `set_rules::Population`: a flag that the caller declares rather than computes.

## Reconciled D-RFT IDs (proposed)

The plan owns the IDs. Within the plan, the prose wins over the table. The #1668 assessment supplies fixtures and inside surfaces, but never moves an ID. No renumbering is needed.

Order: 0 (landed) → 1 → {2, 3} → 4 → 5.

| ID | Contract | Change from today |
|---|---|---|
| D-RFT-0 | `normalize_kleene` keeps the TRUE rows. It applies at the WHERE terminal only, and NOT over its output is refused. | Adds the WHERE-terminal rule. |
| D-RFT-1 | One composition over a shared `JoinDomain` declares its stratum (Distinct or Bag(ℕ)) and its key equality. A NULL key matches nothing. No value-directed foreign gather. | All three documents already agree. |
| D-RFT-2 | The law-licensed rule catalogue. The first new rule is the substitution chain under `Arith`. egglog is an optional offline proof oracle only. | The plan table's "egglog path" becomes a sub-clause. |
| D-RFT-3 | Recursion is Distinct only. The closure equals datafrog's, with one `Iteration` per stratum. The 257th slot is refused. A recursive bag is refused. | Adds the bag refusal and the stratification refusal. |
| D-RFT-4 | Every route gives the same answer, in the Distinct stratum only. Slot generation and binding witness are linked: a rebind invalidates every holder. | The plan table's "think / counterfactual / proposal" row moves to D-LFA-1/2. |
| D-RFT-5 | Replay equals from-scratch. It covers (a) frontier invalidation, with the carrier named per stratum, and (b) thought-git replay from a D-TGB-4 receipt. | Makes the receipt prerequisite explicit. |

## Exploration map

### ADOPT-NOW (documentation only; no code)

| # | Idea | Plan step |
|---|---|---|
| A1 | Adopt the reconciled D-RFT ID table above. This gates any EXO-50 claim of D-RFT-2/4/5. | all D-RFT |
| A2 | Each step declares **stratum / carrier / equality** in the plan text and doc comments. This is a declaration, never a shared type. Promote it to a type only when a failing test shows two crates must exchange it at run time. | all D-QSP, D-RFT |
| A3 | Name a second equality, NotDistinct, used by `GROUP BY`, `DISTINCT`, set operations and recursive `UNION`. It is distinct from K3 `=`. | D-QSP-1/2/4 |
| A4 | `normalize_kleene` is WHERE-terminal. NOT over its result is refused. A (T,F) pair, if ever wanted, is a separate step, because adding it reopens D-RFT-0. | D-RFT-0, D-QSP-0 |
| A5 | `NOT IN`, `NOT EXISTS` and `EXCEPT` are three distinct parity rows. Rewriting one into another is known to be unsound on nullable columns. | D-QSP-2, D-RFT-1 |
| A6 | Recursive strata are Distinct only, unless a convergence proof exists. A recursion limit is a typed refusal, never a truncated answer. | D-RFT-3, D-QSP-4 |
| A7 | DuckDB is never a second correctness oracle. It is compared only on the shared stratum, after a declared dialect-normalization list. A PostgreSQL error is a typed refusal. | D-QSP-3/5 |
| A8 | Four "commit" events stay distinct: Rubicon intention, `KanbanColumn::Commit`, MUL authorization and the seal. Verbs permitted per column form a relation, which does not exist yet. `WaitForInput` is an external pause, not deliberation (this corrects #1668's assessment). | D-LFA-1/2 |

### PROBE (named, pre-registered, unrun)

All probes compare counts or scalar folds.

| Probe | Tests | Kill condition |
|---|---|---|
| P-NOTIN-TRIPLE | R={1,NULL}, S={NULL}: `NOT IN` 0, `NOT EXISTS` 2, `EXCEPT` 1 | PostgreSQL disagrees with 0/2/1 |
| P-NULLGROUP | NULL keys form one group under NotDistinct | PostgreSQL splits them |
| P-EMPTYSUM | Empty or all-NULL `SUM` is NULL. Ungrouped empty input gives 1 row; grouped empty input gives 0 rows. Real-zero `SUM` stays 0. | A real zero comes out NULL (over-correction) |
| P-EXCEPTALL | Multiplicity is max(m−n, 0). On a Boolean route the answer is a semantic refusal. | `EXCEPT ALL` is admitted without a count carrier |
| P-KLEENE-NOT | `NOT(normalize_kleene(x<5 AND NOT x<5))` on {3, NULL} keeps 2 rows; PostgreSQL keeps 1 | The composition is not refused |
| P-MUTUAL-SUPPORT | Two facts support each other. Delete the external seed and both clear. The twin with a second seed keeps both. | `Pop::repair` gives 0. That would contradict the predicted defect; the prediction is a conjecture. **Run first.** |
| P-BAGREC-CYCLE | A recursive bag over a cycle is refused; the Distinct arm equals the closure | A count is returned |
| P-STRAT-NEG | `A←¬B, B←¬A` is refused | It is accepted. Check that PostgreSQL rejects it before relying on it. |
| P-STALE-RESUME | A source moved, or interrupts were reordered, and resume refuses | Resume answers |
| P-ABA-MODDEL | Modify on one side and delete on the other is a named refusal | It resolves silently |
| P-REFUSAL-KIND | A RouteRefusal is retried on another route. A SemanticRefusal is never retried. | A semantic refusal is retried |
| P-TLP-COUNT | The counts for P, NOT P and P IS NULL sum to the table's row count | The sum differs |

Bridge findings worth a probe:
- `JoinExec::bag_count` sums `u64` without an overflow check (`mork-pathmap-probe/src/join.rs:653-659`). The contract should be exact below a stated row bound and refused above it.
- Grouping NULL keys can reuse the `(Col, Mask)` validity operand that `sql_where` already takes. No new carrier type is needed.

### PARK

- A ℤ-diff bag frontier (differential-dataflow shaped). Nothing implements it; drop it from the strata list until a consumer needs it.
- Semiring provenance polynomials (ℕ[X], Why(X)). They are an analysis lens only and must never be stored. They also collide with the 32-source cap.
- A (T,F)-pair `normalize_kleene` for CHECK constraints (D = {T, U}).
- sqllogictest as the case format. It is usable only with `rowsort` and a multiplicity check, never `valuesort`.

### SKIP

- A differential-dataflow engine.
- Dolt / lakeFS three-way merge vocabulary. Thought-git has one writer.
- LangGraph as the stale-resume oracle. It does not refuse a stale resume.
- NARS revision as a semiring or lattice merge. It is not idempotent.
- One merged provenance type: lineage, replay receipt and EpistemicState5 stay separate.

## Anchors the council found at risk

- **Quack `Filter::Semijoin` is a value-directed foreign gather** (`lance-graph-quack/src/lib.rs:152, 366-391`, lowering to `MaskOp::Gather`). ADMISSION B lists it as pending call-site classification. No `NOT IN`, `EXCEPT` or anti-join lowering may build on it until that classification lands.
- **Oracle scope.** `CLAUDE.md` already rules that answers are compared by folds, "for tests and oracles too". So every oracle inside the workspace is bound. Some existing oracles walk decoded rows:
  - `k3_pg_parity`'s Kleene reference evaluator;
  - the K3 unit tests in `set_algebra` / `set_rules`;
  - `join_probe.rs`;
  - `walk_mask.rs:361-372`.
  
  Only external processes (a PostgreSQL server, Soufflé) remain genuinely open.
- **D-LFA work before a pause.** `graph-flow` re-runs a node from its start on resume. Any Task that does work before a pause point conflicts with A6/A10 unless that work is pure compute.

## Open (operator)

1. Does "oracles compare by folds" admit a reference engine that holds tuples internally, if the comparison itself is fold against fold? This decides datafrog's role and the reference evaluators listed above.
2. Adopt the reconciled D-RFT table (A1) before anyone claims D-RFT-2/4/5 on EXO-50?
3. Who classifies `Semijoin` under ADMISSION B?

## Not searched

- The `Pop` tile body was not read in full; whether a tile carries counts is CLAIMED.
- The z8run-lance stale predicate is not on disk.
- From the systems scout: the original sqllogictest page, the lakeFS documentation, and the Roaring bitmaps papers.
- Primary sources not obtained: the Gupta–Mumick–Subrahmanian (1993) original and the Geerts–Poggi m-semiring papers.
- No probe was run.
