# 2026-10-20 — Thought git: replay is the API, the write is the Rubicon decision

**Status:** DECISION (operator, 2026-10-20) for the principle; WORKING-MODEL for the kanban mapping and the anomaly predicate. **Basis:** the operator's words (quoted in `thought-git-basics-v1.md` § "Correction: replay is the API, the write is the Rubicon decision") and the implementation read the same day (`alpha.rs`, `alpha_tunnel.rs`, `givens_layer_probe`, `thought_layers_probe`, the split-tunnel doc).

- One fold ≈ 1.7 ns, so one write buys 10⁶–10⁸ folds (operator: 233–350 ms per write; measured here: 1.7 ms floor, 5.8–10.5 ms `LanceCycleWriter` commit, 336 ms Lance scan).
- 99.9 % of thought git operations (branch, write, diff, status, squash, discard, replay) are autonomous in-memory heuristics with no Lance call.
- A thought without anomaly is squashed onto the stack and work continues on a new branch. Anomaly signals already in code: contradiction records, regressions (`visits > 1`), cold/hot surprise (`expected.xor(attended)`), divergence between thoughts (`differs`).
- Landing into Lance is the conscious Heckhausen Rubicon decision, taken when the result is satisfactory. A kanban card tracks the write intention, not the branch.
- Presence decides what is written: an absent lane is unknown (the NaN case), never zero; agreement with a given is not stored; a contradiction is a record, never a stored value.
- Rebase is replay of the receipt on the new base; v1 needs no recursion driver.
- "Satisfactory" is measured by automated hypothesis testing: each strategy is a branch on the same Sudoku fixture, compared and synthesized with `jc` statistics (an R-style API, interface OPEN). Engines in parallel: act-r, metta, nal, relalg, mask-risc exist; cog (Hyperon) and ona do not yet. First the strategies on Sudoku, then the 115 rows of `nal_inventory.tsv` as MeTTa programs.
- D-TGB-0 and D-TGB-1 re-specified; the old "slot taken at the Rubicon crossing" is struck.

- MeTTa strategies run PeTTa-style: compiled to relational clauses that execute on `datafrog` instead of SWI-Prolog; `ruff_metta_spo` extracts and classifies each rule under PeTTa's natives; `hyperon-experimental` is optional, for the interpreter-only rest and as a cross-check.
- Reasoning state (relalg, Datalog, datafrog), git metadata (PR entries) and replay receipts live in one separate table keyed by (data dataset, data version, thought id), so the data table and the metadata table do not depend on each other's schema. OPEN: the two writes are two Lance versions and not atomic together; data first, then metadata, and a missing metadata row is detected at replay.

OPEN: the anomaly predicate and thresholds; what "satisfactory" means for a write; card granularity.

**Addendum (operator, 2026-10-20):** "Du kannst alternativ gerne direkt die crates ansprechen das ist für das Internet Timing erstmal besser". DECISION: first runs may call the engine crates (act-r, datafrog, relalg, mask-risc, metta, nal) directly; the verb trait stays the target and they move behind it once it exists. Polyfill-never-legacy is unaffected. Plan: `thought-git-basics-v1.md`, section "Addressing the crates directly is allowed first".

**Addendum (operator, 2026-10-20):** "Kanban sollte später planungs Vokabular mit einer schönen schnellen effektiven API erhalten. Was lasse ich gerne offen. Es muß aber auch quack shaped folding polyfill methods verstehen". DECISION: D-TGB-9, later; vocabulary OPEN; fixed that planning questions are Quack-shaped folds (filter → mask, `Agg` fold, lowered to mask-risc), `PhaseCensus` as precedent.

**Addendum (operator, 2026-10-20):** "Internes Timing" (correcting "Internet Timing"): direct crate calls are for runtime cost, not schedule. "Ein schreibpfad braucht rust, keine API … Der kommt dagegen schon <> git/kanban": landing is concrete Rust, triggered by a kanban card, leaving its receipt in the metadata table. "datafrog > quack / folding RISC simd": the measured 8–15× Semijoin lowering rests on gather, forbidden since 2026-10-16; the allowed route is H14d's four stages (join is the layout). Recursion stays on `datafrog` until D-MTP-3. New D-TGB-10 (probe).

**Addendum (operator, 2026-10-20: "Mork hat die MIT Entscheidung bereits auf dem Board bestätigt"):** D-MPS-7 is decided, MIT. The board record the operator refers to was not found by this session in `.claude/` on main or in Linear P-EXO-3; the operator's words are the cited source. MeTTa over `lance-graph-mork` and MORK/MeTTa/relalg Datalog are no longer licence-blocked.
