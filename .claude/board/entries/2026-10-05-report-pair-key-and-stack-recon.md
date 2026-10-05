# Report Pair experiment superseded; destination-resolving fold gap and stack recon (2026-10-05)

**Status:** ⊘ SUPERSEDED IN PART (2026-10-05, same PR, before merge) — the Pair execution path described under "The cut" was REVERTED; this PR now ships only the corrected gap doc in `exec.rs` and this record. See "Correction" below. Recon findings and decisions 2–5 stand. ~~MEASURED — the code change and its tests are in this PR.~~ The recon findings are VERIFIED-IN-CODE against z8run `3a8a758`, lance-graph `97a3610d`, OGAR `e5de84e`, rs-graph-llm `e824977` and rig `d165343`. It ratifies the five decisions below and nothing more.

## Correction (2026-10-05) — why the cut below was superseded

`GroupAddr::Pair` addresses `hi · stride + lo`: the **dense product** of two dimension domains. Lowering every two-ordinal report to it made the Cartesian product the execution geometry because the primitive existed, not because any consumer demanded a dense cube. The intended fold model is the PowerShell-Hashtable / Excel-Pivot one: a row resolves its accumulator **destination** directly, and accumulator state scales with the **demanded / resolved destination universe**, not with an accidental Cartesian product. For IAM-sized dimensions (64k × 64k) a Pair sink is 4·10⁹ slots whatever the data demands.

- **What stays true:** Pair was useful *evidence* that Report re-implemented what Quack can fold, and the old doc's "substrate gap" claim was stale.
- **What was wrong:** Pair is not the general Pivot/Join execution model. A destination-resolving keyed fold is.
- **What the source shows (VERIFIED-IN-CODE):**
  - no fold path resolves a row to a destination outside a dense K-slot universe. ndarray/mask-risc/Quack have ONE keyed-reduction walker with three addresses — `Lane`, `Via` (`table[fk[i]]`), `Pair` — all writing a dense K-slot sink (ndarray `simd_masking_ops.rs` `masked_group_*`, mask-risc `ir.rs` `GroupKey`, Quack `GroupAddr`, lgj `plan_lower.rs` Local/Via).
  - The only compact structure is Report's `Layout::Sparse` RESULT index, reached by rescanning once per observed partition member.
- **Reverted:** `fold_major`, `fold_addr`, the Pair SUM normalization, the product budget, the Pair strides and decode, the Pair-pinning tests. Report is back to main's planner.
- **Kept:** `GroupAddr::Pair` in Quack/mask-risc/ndarray (correct for an explicitly dense mixed-radix problem; untouched here).
- **Shipped:** `exec.rs`'s gap doc now names the real gap: a destination-resolving keyed fold, stated semantically (bind/resolve → destination → fold), substrate-first.

## The cut (⊘ SUPERSEDED — reverted before merge; kept as the record of what was tried)

`lance-graph-report` planned two-dimensional reports as **one population pass per partition member**. Its module doc called the composite-key group fold "a named substrate gap". That gap had already closed: Quack has `GroupAddr::Pair { hi, lo, stride }`.

The physical planner now picks a **fold major**: the widest remaining ordinal whose product with the fold key's domain fits `domain_buffer_budget`. It folds both coordinates in ONE pass, through the pair key (`exec.rs` `fold_addr`).

| case | before | after |
|---|---|---|
| 20×30 plan | 20 passes | 1 pass |
| `population_scans` | 80 | 4 |

- Bucket coordinates, mask-set coordinates and a third ordinal stay partitions. Quack has no value-derived or multi-plane group key.
- Quack has no pair-keyed SUM terminal, so a pair-keyed SUM lowers to the NULL-preserving `GroupReduce { SumI32 }`. Its raw sink is normalized with Quack's own `normalize_group_sink` (empty → 0, the SUM identity). Count, Min and Max seed exactly the report's identities (0, `i64::MAX`, `i64::MIN`). Honest limit: the NULL-preserving sum is defined up to `GROUP_SUM_SYM_MAX_ROWS` = 2^32 − 1 rows, one fewer than `GroupSumI32`, so a pair-keyed SUM over a plane of exactly 2^32 rows is refused, not wrapped. Values (including `i32::MIN`) are unaffected.

**Tests:**
- New `pair_key_folds_two_ordinals_in_one_pass_and_equals_per_member_passes`: pair ≡ per-member on every cell, total and hidden-axis merge, over a mostly-empty 600-cell space.
  - Budget threshold: 600 pairs, 599 does not (can-fire / can-stay-silent).
  - **Disable run:** with the normalization removed, it goes red (`agnostic.rs:564`), then green when restored.
- `t9` now pins the pair programs, plus a per-member twin under a tight budget.
- The pass-budget test moved to three ordinals so it still fires at 20.

**Historical observation from review (not active work):** the Pair choice was greedy, not pass-minimal — with domains 100 × 60 × 60 and a 3600 budget it ran 3600 partition passes where folding 60 × 60 and partitioning over 100 would need 100. This is NOT an open optimization for Report: optimizing which dense product to fold is the superseded model. `GroupAddr::Pair` remains valid only where a consumer explicitly demands a dense mixed-radix destination universe; it is not the future implementation of sparse Pivot/Join/grouping.

**Other runs:**
- z8run-lance (`pivot_flow`, handle size under 128 bytes, `z8run_plan_is_the_native_plan`, zero-refold pivot): 10/10.
- report-ogar: 4/4.
- Report suite: 33/33.
- clippy `-D warnings` clean.

## Recon findings that set the scope

- **`pivot_flow()` already runs on Quack:** `lance-execute` → `ReportPlan::execute` → `quack::lower` → mask-risc. ReportPlan is a Quack frontend that duplicates some Quack semantics. Remaining duplicates: IN as an OR of equalities (Quack has `Filter::in_u32`), MEAN/NULL finalize (Quack has `lower_avg` + `GroupAgg::SumI32` + normalize), and its own Selection/Scalar membrane.
- **Quack has no execute and no result handle.** It stops at `Query` → `lower` → `Program`.
- **z8run-lance is not wired into z8run.** It is a workspace exclude, and `register_lance_nodes` is never called.
- **z8run-core runs a JSON row engine:** `database` (up to 1,000 rows into `FlowMessage`) → `aggregator` (group-by over JSON); `batch`/`loop` stream a population as messages.
- **graph-flow infers Kanban lifecycle from workflow status** (`storage_kanban.rs:158-173`: `Completed → Commit`, `Error → Prune`, `Commit → CognitiveWork`). This bypasses `can_transition_to`. graph-flow-kanban also no longer matches the contract (`KanbanMove.libet_offset_us`, `GateDecision {reason}`).
- **Session status is never persisted**, and the engine never builds `ExecutionStatus::Error`.
- **No carrier holds (coordinate space × version):** mask-risc `Planes`, `AbiBatch` and Quack `Query` identify populations by length. The z8run Envelope and the report generation use a free `u32`.
- **No dependency edge** between z8run, graph-flow and ogar-loco in any direction.

## DECISION (operator, 2026-10-05) — the provenance of these five rows is that choice; the evidence is the recon above

1. ~~**First PR** = the two-ordinal partition → `GroupAddr::Pair`, one pass.~~ ⊘ SUPERSEDED (operator, same day): Pair is not the fold model; this PR is reduced to revert + corrected record. See "Correction".
2. **The numeric Quack `Query` is the canonical boundary.** `bind::Draft` and `ReportPlan` are SIBLING frontends; so are a SQL frontend, an IAM frontend, a z8run island compiler and a Rig tool frontend. `Draft` stays small. A maximal pure z8run island lowers to one bound Query/program, not to "one Draft".
3. **Retire the graph-flow → KanbanColumn inference.** Keep generic session replay and resume. graph-flow emits facts and results (task completed, waiting for input, result ref, action receipt); the canonical Kanban + Revision owns lifecycle. graph-flow is not wired to the cycle-seal driver either.
4. **Enforce coordinate space × version at the population execution and result boundaries** (execute(Program, World), ResultRef, workspace). Do not put the dataset version into the `Query` IR to satisfy the invariant. Binder provenance (version-dependent CAM ordinals) is a separate question.
5. **No `Quack::execute` and no `Quack::ResultHandle` yet.** First determine who owns the execution membrane between a numeric Program and a versioned World. It is probably a backend adapter above mask-risc (#1330: "Quack owns query semantics; storage supplies capabilities").

**REVISIT WHEN:** a second frontend needs a capability that only `Draft` growth could give; or the execution membrane's owner is settled.

## OPEN

- **The missing primitive:** a destination-resolving keyed fold, stated semantically: BIND / RESOLVE (semantic coordinates, functional references) → a destination → ACCUMULATE directly into it. **Law:** a join/pivot used only to determine an aggregate destination must compile to destination resolution + fold — not to an intermediate relation, and not automatically to a dense product; accumulator state scales with the demanded / resolved destination universe. The normal fast path binds destinations beforehand (resident ordinal, CAM / codebook binding, `Via` or composed functional reference, or another pre-bound destination representation); only a destination universe that genuinely cannot be bound beforehand may use a query-local compact resolver. No physical representation is chosen here. Substrate-first; contract first — see the fold-contract recon.
- **`CoordSpec` has no functional-reference coordinate:** Report cannot group by `user → department` although `GroupAddr::Via` / `Filter::EqU32Via` already exist. A separable small step.
- Owner of the execution membrane (Program + World → ResultRef). It is needed by both z8run and graph-flow.
- Whether the merge-law re-roll of totals is a Quack fold or presentation.
- Whether `with_roles` may hide dimensions, since it merges over them: a GROUP BY done in the view.
- `ReportPlan::pivot`/`axis` docs say "pure metadata", but adding a coordinate changes the physical key (a data pivot).
- OGAR `handle_submit` runs `CapabilityExecutor` with no gate; the gate exists only in rs-graph-llm's `dispatch_via`.
- The `graph-flow-action-ogar` production path mints V1 (`NodeGuid::new`), reads the classid canon-low, and hardwires `guard_field_value = None`.
