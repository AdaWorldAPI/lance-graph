# ON × DO + watched WHERE: factored dispatch crosswalk and falsifier plan

**Status (2026-10-27): DOCS-ONLY RESEARCH/PLAN. NOT IMPLEMENTED, MEASURED, ADMITTED OR MERGE-AUTHORIZED.** This is an additive crosswalk for the existing owners, not a new contract, registry, engine, watcher service, source of truth, runtime or D-* ownership lane. GitHub owns code/plans; [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id) is a coordination inbox, not approval.

## Research question

Can the logical ON × WHEN × DO incidence in [odoo-rs #44](https://github.com/AdaWorldAPI/odoo-rs/pull/44) be **factored physically** into an ON × DO 256² presence bitmap (8 KiB) + method-identity/guard bindings + **borrowed, witnessed, reusable Quack WHERE populations** (one 65,536-bit / 8 KiB mask per applicable source-object population), with an optional source-attested changed-dependency/watch repair, while preserving full registry, three-valued guard, multiplicity, source-authorization and counterfactual semantics?

The 256² bitmap and the 65,536-row WHERE population are equal-sized **different coordinate domains**. Do not equate registry cells, source-object ordinals, DT64 predicate slots, DR focus positions, CE64 thought W bits, or a Palette256 code. The 256 capacity is a proposed blocked/padded physical view of #44's measured 194 ON × 219 DO, **not** 256 already harvested events or a new 256-predicate DT64 catalogue.

For event e, row R[e] is a candidate 256-bit DO mask, **not** final per-method dispatch. A separate immutable, versioned registration sidecar must preserve each MethodId, inherited/overridden status, typed guard and source-authorized binding. A representative statement of intended semantics is:

```
candidate_do = ON_DO[event]                 # presence only, possibly many MethodIds per bit
eligible(m, object, snapshot) =
    registered(m, registry_version)
    AND triggered(m, event)
    AND WHEN(m, object, snapshot) IS TRUE
    AND source_scope_allows(m, object, snapshot)
# unknown / denied / incomplete are not FALSE
```

For compatible source/coordinate witnesses the population computation may be an existing Quack/RelAlg mask algebra expression such as `P_event AND P_guard AND P_scope`. WHERE TRUE-only projection is appropriate at the terminal; the separate TRUE/FALSE/UNKNOWN evidence must survive wherever negation, explanation or counterfactual branching needs it. Presence of DO bit d **never** proves that every method occupying d is applicable.

## 1. Existing owners and evidence (read first; do not reimplement)

| Existing authority | What exists / precisely what does not | Reference |
|---|---|---|
| Odoo ON×WHEN×DO registry | #44 **MERGED**, test-only 194 ON × 5 provisional WHEN × 219 DO; 628 registrations, version/retract/coexistence, DO sidecar, independent checks. The 8 ambiguity cells arise at unattested ON: the corpus lacks onchange/constrains decorator arguments and real guard predicates. This is **not** a production watcher. | [odoo-rs #44](https://github.com/AdaWorldAPI/odoo-rs/pull/44), `docs/DT64-ON-WHEN-DO-REGISTRY.md` |
| Odoo Thought-Git follow-on | #45 **OPEN**, test-only unknown-guard branches and test-local NARS revision; equal weights are artificial and `NarsTruth` confidence clamp/order-dependence is explicitly reported. Not permission to interpret probabilities as guard facts. | [odoo-rs #45](https://github.com/AdaWorldAPI/odoo-rs/pull/45) |
| Canonical DT64 binding | Source-attested meaning, predicate/schema version, actual source owner/snapshot, ACL/completeness, codebook, focus coordinates, **borrowed** lane; compiled recipe distinct from detached result. Q0 eight-candidate read-only proof **proposed**, not run. No DT64 predicate evaluator or separate data store. | [D-DT64-Q0 mainline](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/d-dt64-q0-borrowed-quack-binding-seam-v1.md), [#1799](https://github.com/AdaWorldAPI/lance-graph/pull/1799) |
| Quack / DR256 WHERE | Existing `Filter::sql_where`, `lower_fused`, mask-risc `Keep / Count / Any`, borrowed `Filter::Plane`, gated predicates. #1792 P2 **MERGED LAB**; old detached predicate-dependent Keep silently retained **1,032** instead of **31** fresh rows. No authority for escaped Keep reuse. | [#1792](https://github.com/AdaWorldAPI/lance-graph/pull/1792), [capstone](https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/dr256-where-capstone-v1.md) |
| Global mask reuse economics | #1790/#1791 lab: 256² global mask = 8 KiB; count reuse advantageous from N=2 on measured compatible fixtures, while frequent rebuild and chaotic/full density can fail. These are workload-specific, **not** watcher microbenchmarks or production admission. | [#1790](https://github.com/AdaWorldAPI/lance-graph/pull/1790), [#1791](https://github.com/AdaWorldAPI/lance-graph/pull/1791) |
| Existing versioned population/repair | IAM `dir-rmap-probe::pop::Pop` carries version witness, per-user bitmap, `and`, `and_not`, `or`, borrowed plane and source-derived `repair`; losing one role-support path must not clear another. An existence example, **not** a transferable watch contract for arbitrary predicates. | [#1666](https://github.com/AdaWorldAPI/lance-graph/pull/1666), `crates/dir-rmap-probe/src/pop.rs` |
| K3 SQL/guards | Quack represents nullable operands as value + validity planes; `WHERE` keeps TRUE and excludes FALSE/UNKNOWN but cannot discard that distinction before `NOT` or counterfactual interpretation. `normalize_kleene` is explicitly WHERE-terminal. | [#1334](https://github.com/AdaWorldAPI/lance-graph/pull/1334), [#1677](https://github.com/AdaWorldAPI/lance-graph/pull/1677), [#1687](https://github.com/AdaWorldAPI/lance-graph/pull/1687) |
| Counterfactual analytical proposals | D-CFA #1813–#1818 are **OPEN DRAFT DOCS ONLY**, staged; #1814 typed incidence and #1815 protected Thought-Git replay own hypothesis-analysis semantics. Do not create a second counterfactual solver, re-interpret sibling worlds as compatible contributions, or claim 63 slots schedule parallelism. | [#1813](https://github.com/AdaWorldAPI/lance-graph/pull/1813), [#1814](https://github.com/AdaWorldAPI/lance-graph/pull/1814), [#1815](https://github.com/AdaWorldAPI/lance-graph/pull/1815) |
| Explicit exclusions | Source and test claims of the operator-rejected, closed-unmerged #1812 must **not** be harvested, adapted, cited as evidence, or used in any follow-on. Construct this proposal solely from independent mainline owners and cited probes. | EXO-50 epistemic quarantine |

## 2. Boundary: watch as a possible *source-owned invalidation*, not an invented service

1. **Cold/first demand:** owner exposes a source-pinned snapshot and authorization scope; existing Quack `WHERE` produces a typed eligible population or a refusal. Charge full scan/sweep/index/tile preparation costs and actual I/O.
2. **Compatible warm demand:** reuse a `Keep` selection *only while the same source-owned immutable borrow/snapshot and complete predicate/scope witnesses remain valid*. A compiled recipe can be reused on a new binding even if its previously computed mask cannot.
3. **Affected-set repair (candidate):** only when the owning source publishes a **complete** dependency change set for predicates, inherited definitions, referenced foreign rows, policy/ACL and codebook changes. Recompute every affected object's guard and update its population membership using the existing algebra/population machinery. A watch signal is an invalidation hint, **not** an authorization or version witness. Cross-row joins and aggregates can expand the affected set beyond the single changed row. Missing mutation coverage => full re-evaluation or **REFUSE**.
4. **Registration/coexistence:** if MethodId A and B share ON/DO but differ in WHEN, either may be eligible for a particular object. Never collapse their MethodIds, substitute last-wins, or call a different registry version a sibling writer. Guard-specific population should be anchored to method/registration identity or an exactly equivalent typed guard-expression group.
5. **No covert external execution:** `DO` records capability identity; actually invoking Odoo/SAP/Exchange effects is owned by source authorization and transactions. No generated business action in this probe.
6. **No prohibited mask escape:** a mask output by `ScatterOrU32` is demanded terminal-only under current mask-risc rules; #1791's witnessed-scattered-population carrier is **not** authorized. `Keep` plane reuse stays inside live, valid bind scope unless the explicit authority gap is solved.

## 3. Proposed experiment ladder (each step requires its own owner approval before code)

**A. Source census / falsifier preregistration (docs only):** enumerate actual ON event harvest, method IDs and guard source for Odoo; record every unattested onchange/constrains/WHERE operand as **UNKNOWN**, not inferred. Inventory in the real current Quack path which input planes are borrowed and where mutation closure is absent. Pin a source-independent scalar reference/evaluator; do not create a second production predicate interpreter.

**B. Factorization parity (test-only):** independently compute expected per-MethodId, per-object eligibility from source-scoped fixtures. Compare dense logical ON×WHEN×DO evaluation with factored ON×DO + `WHERE` (same registry/snapshot; compare exact identities, guard verdicts and whole masks, never popcount only). Include two different methods at identical ON/DO with disjoint guards; subclass override versus coexistence; one method with multiple capabilities; absent ON and missing guards. A 256-wide block/pad must not change #44 outcomes.

**C. Watch economics / repaired mask parity (test-only):** compare **cold full sweep**, **fresh Quack WHERE**, **same-borrow kept population**, **source-derived affected-set repair** and **forced rebuild/refusal** under 65,536-row sparse, clustered, full, chaotic, cross-record and multiversion distributions. Charges: watch preparation/subscription, dependency fanout, updates, invalidation/check, binds, data reads, mask fold, source access, memory footprint and I/O. Include N=1/2/4/8/64 compatible/incompatible uses; report crossover `N*`, not assumed 0-cost guards. Realizes existing P2 findings, not a new Quack executor.

**D. K3 / authority / counterfactual boundary (test-only):** exact TRUE/FALSE/UNKNOWN with validity, NOT and SQL WHERE-terminal projection; ACL/revocation, incomplete data, same stamp from different owners, codebook+coordinate changes. Unknown guard hypotheses require separately labelled worlds; **never** feed sibling branch writes to an ordinary union landing as though compatible. D-CFA / Thought-Git owners decide hypothesis admission, not this factorization plan.

**E. Admission decision:** source-attested correct and faster? **ADOPT** only when incremental mutation closure and valid masks are demonstrated. Otherwise **KEEP_DIRECT** (recompute inside borrow), **DEFER** (missing witness, real harvested guard), or **REJECT** (false positives, leakage, unpredictable cost). No production implementation implied by the decision table.

### Mandatory red/green disables

1. Remove MethodId multiplicity on one ON/DO bit -> at least one eligible coexisting writer disappears (**RED**).
2. Drop WHEN applicability / substitute provisional `unattested` as true -> exact result mismatch (**RED**).
3. Flip NULL/UNKNOWN to FALSE or negate WHERE-only truth as full K3 -> SQL/negation oracle mismatch (**RED**).
4. Retain detached stale `Keep` after property mutation (#1792 1,032 versus 31) -> REFUSE/recompute invariant fails (**RED**).
5. Miss one cross-row dependency or direct write with unchanged superficial stamp -> repair result differs from fresh source sweep (**RED**).
6. Change source owner, ACL, completeness, scope, codebook or coordinate seal -> stale cache must never be consumed (**RED**).
7. Remove one of two independent supporting paths -> the supported membership must remain (**RED** when blindly cleared).
8. Mix counterfactual sibling worlds or infer permission from branch probability -> world isolation / authorization test fails (**RED**).
9. Change only thought completion order / method-file order -> deterministic outputs remain identical (**stay green**); explicit source-defined precedence may be changed only by a pinned registry version.
10. Disable the mechanism thought to save time (watch/repair) and re-run charged benchmark; a speedup claim with no measurable delta must be **retracted**.

Actual tests and disables are **not run** in this documentation PR.

## 4. Linear EXO-50 map and ownership firewall

- [EXO-50](https://linear.app/exo-red/issue/EXO-50/agent-message-bus-gpt-xgrok-ccsession-id): new append-only handoff comment only, link this PR after creation. Existing issue status **Done** remains untouched. No claimed delivery/ACK for other-account Claude sessions.
- [EXO-46](https://linear.app/exo-red/issue/EXO-46): global↔local coordinate witness / production binding membrane **OPEN**. This plan cannot settle it.
- [EXO-126](https://linear.app/exo-red/issue/EXO-126): Quack SQL/K3 and relalg × Thought-Git coordination; all algebra stays with existing owners.
- [EXO-127](https://linear.app/exo-red/issue/EXO-127): admitted D-DRMC compiler algebra, not a newly defined WHEN compiler.
- [EXO-140](https://linear.app/exo-red/issue/EXO-140): ADMISSION B Semijoin/EqU32Via/Gather call-site scope not to be silently broadened.
- [EXO-98](https://linear.app/exo-red/issue/EXO-98) and [EXO-148](https://linear.app/exo-red/issue/EXO-148): Thought-Git reusable API / durable identity remain separately owned; D-CFA stages remain research.
- [EXO-109](https://linear.app/exo-red/issue/EXO-109), [EXO-132](https://linear.app/exo-red/issue/EXO-132), [EXO-133](https://linear.app/exo-red/issue/EXO-133), [EXO-135](https://linear.app/exo-red/issue/EXO-135): oracle independence, carrier closure, synthetic versus source-grounded claims, evidence→law admission.
- [EXO-152](https://linear.app/exo-red/issue/EXO-152): native numerical methods and ONNX/TFLite leaf certificates separately owned, not relevant to cheaper Boolean WHEN operations.
- NFR-OP-1: escaped-mask provenance/authority **OPEN**; NFR-OP-2: physical DR³ third axis **OPEN**. Neither is adjudicated by proposing 256² ON×DO plus separate WHERE masks.

**Review requested, not implementation requested:** DT64/Odoo owner checks harvested guard provenance and multiplicity; Quack/DR256 owner checks borrowed WHERE, occupancy and mask lifetime; source owner checks mutation closure; CFA/Thought-Git owners check counterfactual isolation. Reply with exact contradicting source/measurement and smallest test-only falsifier. No new storage layer, scheduler, general watcher runtime, planner, new algebra/ISA, hidden materialization, source mutation, automatic PR merge or board-status promotion.
