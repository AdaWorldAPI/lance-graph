# Co-research: mandatory codegraph, lazy reasoning behind DT64

**Date 2026-10-31 · Status OPEN (exploration map; nothing adopted by the council) · Plan refs: D-DYNBAG §18-§20**

**Question.** Can "every admitted executable method has a mandatory, versioned, addressable codegraph (proposed: R2IL); execution runs a bound native plan without hot-path graph traversal; reasoning is pushed down lazily through DT64" be made true without duplication? What is the minimal ownership and binding seam?

**Premise gate (premise-auditor): PREMISE-SPLIT.** The clause "R2IL owns the method representation" is PREMISE-WRONG as filed. R2IL (`ogar-r2il`) is one domain vocabulary: an arity table with no method identity and no revisions (`lib.rs:3-60`). "Codegraph" names two artifacts:
- (a) the static, versioned body (frame);
- (b) the index of executed ops (motion, per run).

The scouts answered four re-asked questions: Q1 who owns the body, Q2 is the binding a projection of it, Q3 what a receipt carries, Q4 where the cold analysis runs.

## Inside facts (VERIFIED-IN-CODE)

- OGAR `ActionDef` holds the body as `body_source: Option<String>`, with no revision and no hash (`ogar-vocab/src/lib.rs:396-457`).
- `ogar-loco::Program` is an executable body form that "does not mint keys" (`program.rs:38-49`). Executable `PROC_DEF` is OPEN.
- DT64 `MethodBinding` is hand-authored. Nothing derives it from an OGAR body (closed search over `lance-graph/crates` and `OGAR/crates`). `MethodRef` cannot name a body.
- Opaque ops are refused everywhere: R2IL `CallOther` has arity `None`; Tract returns `Refusal::Unregistered`. No opaque-node type exists.
- Per-op provenance exists only in r2ssa (`OpArena` `OpOrigin`/`Pass`).
- Content-derived revisions exist only in tract-harvest-probe (`FrameRevisions.source`), a lab crate.

## Outside evidence (read by section)

- **Soufflé provenance** (arXiv 1907.05045): each result carries a rule id and a proof height; proof trees are built on demand; overhead 1.27× time and 1.45× memory.
- **MLIR** (arXiv 2002.11054): one IR with many dialects; unknown ops are handled conservatively; effects are declared through traits and interfaces; each op carries a Location as provenance.
- **Incremental CodeQL** (arXiv 2308.09660): results are populated on demand, and incremental reuse works only with stable IDs across commits.
- **PROV-O**: `used`, `wasGeneratedBy`, `hadPlan`.
- **PostgreSQL plan cache**: a plan is re-planned when a recorded dependency changes.
- **ONNX Runtime**: a saved optimized model is valid only for its target, so a receipt must name the target.

## Exploration map

| # | idea | bridge | firewall | verdict | probe |
|---|---|---|---|---|---|
| 1 | Canonical method body lives in the OGAR IR (`ActionDef` → `ogar-loco::Program`), written in several vocabularies; R2IL is one of them | OPPORTUNITY | PASS | **ADOPT-NOW as the ownership rule** (doctrine already in OGAR-AS-IR; no code) | P1 |
| 2 | "R2IL owns the method representation" | DROP | TRAP | **SKIP** (contradicts the ruling that R2IL is executed and never pre-converted; R2IL has no identity) | — |
| 3 | Body revision = digest of canonical body bytes, identity only, never a correctness gate; held on the OGAR Core node | OPPORTUNITY | PASS (conditional) | **PROBE** | P2 |
| 4 | DT64 binding footprint derived from the body by a cold lowering outside DT64; DT64 validates (undeclared write refuses, inferred read warns) | WORTH-EXPLORING | CONFLICT (seam: OGAR codegen / lance-graph-ogar adapter) | **PROBE** | P3 |
| 5 | `MethodRef` + `body_rev`, frozen under D5; settle EXO-50 together | OPPORTUNITY | PASS | **PROBE** after P2/P3 (DT64 type change: owner decision) | P4 |
| 6 | Opaque native ops as declared extern words with arity + footprint in the OGAR body IR (never in the R2IL table) | WORTH-EXPLORING | CONFLICT (seam: OGAR body IR only, executed by the native adapter) | **PARK** pending an OGAR ruling | P5 |
| 7 | Receipt per run: `MethodRef`, `body_rev`, `plan_rev`, target, frame/route, output | OPPORTUNITY | PASS (per run, never per row) | **PROBE** | P7 |
| 8 | Re-plan or refuse when a recorded revision changes | WORTH-EXPLORING | PASS | **PARK** (frame/route rechecks exist; body revision depends on 3 and 5) | — |
| 9 | Lazy provenance | WORTH-EXPLORING | CONFLICT if per-row | **ADOPT-NOW as a rule:** the hot path emits only the per-run receipt; proofs are rebuilt cold by replay | — |
| 10 | Cold relational code projections (Defines/Uses/Calls/…) keyed by (MethodRef, body_rev), computed on demand, never stored as authority | WORTH-EXPLORING | CONFLICT if persisted | **PARK** until a body exists | P8 |
| 11 | Persist the e-graph in the IR | DROP | TRAP | **SKIP** | — |
| 12 | "Mandatory codegraph" | OPPORTUNITY as a mandatory body reference at admission | SPLIT | **PROBE** as "mandatory body reference"; **SKIP** as "mandatory R2IL graph" | P9 |
| 13 | #1892: DT64 effects → R2IL lowering | DROP | TRAP | **SKIP**: #1892 is superseded, do not merge. Reusable parts: the `Receipt` shape (re-key to body/plan revision), the `FoldDialect` run path | — |
| 14 | One def-use container for Tract / r2ssa / OnnxGraph | WORTH-EXPLORING | CONFLICT (no new shared type) | **PARK**; if pursued, a ClassView reading over the existing body container | P10 |

## Minimal seam (proposal, for the operator)

- **Body.** OGAR owns it. The Core node's `ActionDef` references one `ogar-loco::Program`, written in core words plus vocabulary words (R2IL, relalg, onnx, declared externs). `body_rev` is a digest of the canonical body bytes: an identity, never a gate. `body_source` stays as provenance text.
- **DT64.** Carries a reference, `MethodRef` + `body_rev`. Under D5 a new body means a derived classid plus a re-route. DT64 still executes nothing and gains no dependency on the interpreter.
- **Binding.** ON/WHEN stay authored in DT64. The effects/reads footprint is checked against a footprint derived by a cold lowering outside DT64 (OGAR codegen or the lance-graph-ogar adapter). Full derivation comes later, after P3 measures it.
- **Execution.** Prepare lowers the body once into a native plan with `plan_rev`; the plan is a cache, re-derivable from (`body_rev`, target). Run executes only the plan. Tract's `Plan` is the precedent.
- **Receipt.** One per run: `MethodRef`, `body_rev`, `plan_rev`, target, frame/route, output. Replay refuses on any mismatch, as D-TGB-4 does.
- **Reasoning.** XLOG / Egglog / DialEgg / Datalog run as cold passes keyed by (`MethodRef`, `body_rev`) and read the OGAR body. DT64 hands out the key, never the evaluation. Nothing is persisted as a second store.
- **Opaque ops.** Declared extern words with an arity and a footprint in the body IR. An undeclared op is still refused.

## Probes (pre-registered; lab only; P2 and P3 first, they gate P4, P7 and P9)

- **P1** Join an `ActionDef` and its `Program` under (classid, method, `body_rev`) without copying fields. Kill if the Program needs a field that lives only outside OGAR, or if the join needs a new store.
- **P2** Is `body_rev` stable across a no-op re-harvest, and does it change on a one-token semantic edit? Kill on either noise or insensitivity.
- **P3** Derive the binding footprint from `ActionDef` (reads/writes/calls/KausalSpec) and compare it with the hand-authored xlog-relcomp-probe bindings. Kill if any field needs invented information, or if registration refuses it with `UndeclaredRead`.
- **P4** Lab copy of DT64 with `body_rev` on `MethodRef`. Kill if registration needs the copied effects.
- **P5** A deliberately unregistered Jina op as an opaque node. Kill if the analysis silently treats it as pure.
- **P6** No hot-path traversal: lookups per run after bind must be zero, or at least independent of graph size (tiny vs Jina; sag fixture at N vs 16N). Report in ns/µs.
- **P7** Does the receipt tell runs apart when the body or the target changes? Today's `plan_id` would collide when the body changes but the binding does not.
- **P8** r2ssa rows keyed by (MethodRef, `body_rev`, OpOrigin). Kill if OpOrigin is unstable across identical harvests.
- **P9** Census: admitted methods with no OGAR body to point to. If there are more than 0, "mandatory" is a new authoring obligation.
- **P10** One def-use container across Tract / r2ssa / OnnxGraph. Kill if it needs `Any` or a blob.

## Not searched

- Translation validation / proof-carrying code (no primary source read).
- egglog and DialEgg papers beyond secondhand summaries.
- semiring provenance (not read).
- ProvONE (fetch failed).
- r2sleigh `r2engine` scheduling cost.
- OGAR `KausalSpec` variant bodies in depth.
- consumer repos (medcare-rs, odoo-rs), to count `ActionDef`s with a `body_source`.

## Closeout

PR #1890 | STATUS: open | OUTCOME: exploration map; #1892 marked superseded | OPEN: operator choice of probes; OGAR ruling on opaque extern words; executable `PROC_DEF`; EXO-50

## Operator correction (2026-10-31): `ogar-loco::Program` is not the method body

DECISION (operator). The October 30/31 codegraph architecture stands: **Kythe semantic facts + R2IL executable semantics + DT64/DR binding.** `ogar-loco::Program` is not the canonical method body, and the ogar-loco program model is rejected as that architecture. Struck, not deleted, in this entry: every statement that an `ActionDef` points to an `ogar-loco::Program`, that `body_rev` is a digest of such a Program, or that executable `PROC_DEF` in ogar-loco is a prerequisite.

Consequences:
- No `ogar-loco::Program` body, no new codegraph store, no second interpreter or method registry, no mandatory eager graph materialisation.
- P9's "no `ActionDef` links a Program" (category 1 = 0 of 50) is a fact about the data, not an admission gate. It blocks nothing.
- R2IL execution is kept without the ogar-loco program model: `FoldDialect::call` is inherent and the `ogar_loco::Dialect` impl is an optional adapter (#1895). The Kythe reading from witnessed facts exists (#1894).

OPEN: where the method's body revision is anchored once the Program is not the body.
