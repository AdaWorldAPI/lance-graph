# 2026-10-22 — /coresearch: the D-DRMC stack (#1700–#1704)

**Status:** EXPLORATION MAP. Nothing adopted; the operator chooses. Scope: the
five stacked PRs of D-DRMC (#1700 charter, #1701 typed DR64 binding, #1702
finite-population evidence, #1703 law/recipe identity, #1704 ACT-R handoff + CI).

## Question (after the premise gate)

The premise audit split the original question ("is this staging right, and what
does outside knowledge change?") into four, because the stack bundled concepts
with different signatures:

- **A. Frame identity:** which revisions identify the reference, who seals them,
  and how that seal stays apart from focus selection and authorization scope.
- **B. Finite-population evidence:** what makes a claim over one 64×64 focus
  checkable; how validity, K3 and completeness are carried.
- **C. Promotion:** under what typed step, if any, evidence may become a law
  premise. Evidence never becomes a proof.
- **D. Rebinding:** what a recipe keys on so it is binding-independent while each
  execution stays binding-guarded.

Verified by the main thread before the scouts ran: two different
`BindingWitness` types exist (dr-math `lib.rs:73`, 5 fields; egglog-r2il-probe
`catalogue.rs:39`, 3 fields), and the only shipped recipe falls back on any
binding change (`catalogue.rs:257`), which contradicts the charter's "the same
recipe rebinds to a fresh snapshot".

## Measured (main thread, release, not by the PRs, which never ran cargo)

| head | tests | clippy `-D warnings` | `cargo fmt --check` | CI |
|---|---|---|---|---|
| #1702 `34c43161` | 12/12 pass | clean | 17 diffs | no workflow |
| #1704 `63d27f35` | 20/20 pass | clean | 32 diffs | new `isolated-probe` job red at Format (#1703 and #1704) |

## Exploration map

Grades and evidence for every row are in the crosswalk (scratchpad, banked).
B/F/X = bridge / firewall / falsifier co-architect verdicts.

| # | idea | relation | B | F | verdict |
|---|---|---|---|---|---|
| X4 | `universal_on_focus` checks `Unknown` before `Counterexamples` (`population.rs:27-38`), so a refuted focus reports `Unknown(n)` and hides the counterexample. Certain-answer semantics (incomplete FCA, Libkin) put refuted first. The one test with both FALSE and UNKNOWN never calls it. | NEW (defect) | OPP | PASS | **ADOPT-NOW** in #1702: order Vacuous → Counterexamples → Unknown; test gamma `0b111`, R `0b001`, validity `0b011` ⇒ `== Counterexamples(1)`, plus both twins |
| X17 | gamma and R are bound to the same bits in `contract_handoff.rs:69-70` and `exact_complete_focus_passes_without_causal_claim`; recipe tests bind all-zero rows. "gamma ⊆ R" is then gamma ⊆ gamma and cannot fail. | NEW (test defect) | OPP | TRAP | **ADOPT-NOW**: distinct non-empty fixtures with anti-vacuity asserts; disable = read gamma in place of R |
| X20/X13 | Workflow added by #1704 is red at Format; installs `stable` while the repo pins 1.99; 32 rustfmt diffs. | PARTIAL | OPP | CONFLICT | **ADOPT-NOW**: `cargo fmt`, mirror `egglog-r2il-probe.yml`'s toolchain step |
| X15 | Stage 3 re-spells Stage B's law/proof/target identity. `ProofDisposition::CheckedByExternalAuthority` is caller-set; `offer_recipe` checks only a nonzero digest and checker revision, so a forged digest passes where `ReplayCapsule::recheck` refuses. Breaks the Stage B anchor (proofs checked by egglog's own checker) and "authorization is provenance". | PARTIAL / duplicate | OPP (partial replace) | **TRAP** | **ADOPT-NOW (remove)**: delete the caller-settable disposition; keep `LawShape` as the Boolean-DR64 domain key; take proof and target from Stage B (`ReplayCapsule::recheck`, `TargetAbi`). Probe X15 (forged-digest count `> 0` today) |
| X1 | The 5-field witness merges reference identity (source/map/coordinates), focus (motion) and authorization scope (policy), against the 2026-10-03 frame-vs-motion ruling. `dir-rmap-probe` already separates `Witness` (prep.rs:61-100) and `Focus` (focus.rs:70-79). PROV-DM: entity bundle / activity / agent. | PARTIAL | OPP | PASS | **ADOPT-NOW** in the probe: three objects, three checks, no `From` between frame and focus |
| X2 | A seal is store-issued identity (`Witness::of(store, version)`, `Weak<Snapshot>` by pointer), not caller equality. | ALREADY-HAVE elsewhere | OPP | PASS | **ADOPT-NOW** with X1: reuse dir-rmap's `Witness`; probe: two stores at version 0 must not alias |
| X3 | Validity is a carrier of R (Arrow validity bitmap, Alive2 value/poison, Quack `BoundField.validity`, D-QSP-0 NULL contract). #1702 types it as a third `BoundDr64` with its own method; a separate UNKNOWN carrier was already rejected here (EXO-50 /027, /028). | ALREADY-HAVE (probe diverges) | OPP | PASS | **ADOPT-NOW**: bind validity with R, Quack shape; probe: a validity bound to a foreign method must refuse |
| X6 | `validity: None` silently means all-known (`population.rs:81`), a closed world chosen by the caller's `Option`. Completeness is scoped, attested metadata on the data source (Razniewski & Nutt; Corman/Nutt/Savković). | NEW | OPP | CONFLICT | **PROBE**: `None` must refuse or report `Unattested`; any completeness record is local to the probe and store-issued with the frame, never in the contract |
| X8 | Key the recipe on its semantics; guard the binding on each run (fold contract "a prepared plan is not a valid execution world", PostgreSQL replanning, CompCert per-run validation, Enumo per-domain re-validation). Stage B's `Entry` holds one binding and `insert` overwrites. | PARTIAL | OPP | CONFLICT (out of fold only) | **PROBE**, owner egglog-r2il-probe: two certified bindings for one key; today the first falls back. Lookup and rebinding stay out of fold |
| X16 | `LawShape` keys on axis kinds only, so one law is reused for class 1 and class 22 without re-validation. Laws over sort variables need a parametric proof; Enumo kept 90–100 %, not all, across domains. | PARTIAL | OPP | CONFLICT | **PROBE**: class 22 with 2 counterexamples must not receive the law; shape lookup only nominates |
| X18 | `contract_handoff` feeds `false_pairs == 0` into ACT-R eligibility, unwraps `universal_on_focus` (inherits X4), and ties `Snapshot.witness_version` to the binding only by a coincidental literal (9). | PARTIAL | WORTH | CONFLICT | **PROBE** after X4: a census may veto, never qualify; snapshot derived from the binding |
| X7 | Evidence nominates, a separate validator admits (Ruler `is_valid`, Enumo `partition`, D-CMR §3.5, `causal_audit`). No conversion PopulationReport → premise exists, correctly, but only by absence. | PARTIAL | OPP (negative) | CONFLICT | **PROBE**: compile-fail test that no such conversion exists; a population-derived premise may only ever be a named, undischarged hypothesis that `promote` refuses |
| X5 | R bit × validity bit gives four codes, one duplicate. | NEW | WORTH | CONFLICT | **PARK**: canonical-form check `R ∧ ¬valid = 0` is cheap; Belnap "Both" only if a store merge ever writes it |
| X9 | One focus is an FCA formal context but not a support count; support needs counts under an explicit equivalence key over a window (2026-10-14 coverage DECISION). | PARTIAL | WORTH | CONFLICT | **PARK**: document `PopulationReport` as a per-focus census; support stays with population-law-crosscheck |
| X10 | Typed relation algebra: sort checks exist; composition/converse typing does not. | PARTIAL | WORTH | CONFLICT | **PARK**: composition belongs to dir-rmap compose / relalg, not this probe |
| X14 | A changed method *resolution* is its own invalidation input (PostgreSQL search_path hole); `MethodSignature` is caller-built. | NEW | WORTH | CONFLICT | **PARK** until a ClassView registry exposes a revision; then ask its owner |
| X21 | G0–G9 gate matrix, "27 atoms", heritage example, amortization design (END_TO_END_GATES.md). | NEW (prose) | park | PASS | **PARK**: label each gate CLAIMED until it cites a test |
| X11 | Bag is a different semiring; `BagNeedsMultiplicity` already refuses. | ALREADY-HAVE | DROP | PASS | **SKIP** (duplicate) |
| X12 | Proof re-check independent of its producer: Stage B `recheck` + i128 oracle. | ALREADY-HAVE | DROP | PASS | **SKIP** (duplicate; it is what X15 should consume) |
| X19 | `Capability` gate: a math offer never authorizes an effect. | ALREADY-HAVE | DROP | PASS | **SKIP**; tighten `is_err()` to `== EffectRequiresSeparateAuthorization` |

Tension resolved: authorization scope is not frame identity; it is checked as its
own guard at borrow time. The D-RMAP §7.4 cache key may still be the composite
(frame, scope) without merging the two types.

## Not searched

Differential Dataflow, DuckDB prepared statements, Isabelle/Coq kernels, Green et
al. PODS 2007 (read secondhand only), Necula proof-carrying code, Alive2's
internals beyond its value encoding, Libkin ICDT 2015 in full (abstract only).
The `sql_where` internals in Quack, the `set_rules` driver body and the
egglog-r2il-probe tests were not read. "No conversion PopulationReport → premise"
was checked in `population.rs` and `artifact.rs` in full, not across the whole
tree.

OPEN: who owns the frame seal for real sources (G1); whether `LawShape` laws can
ever be proved sort-parametric; whether Stage B's catalogue should hold several
bindings per key.
