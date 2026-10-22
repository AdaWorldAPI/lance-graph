# Mathematical foundry integration review, 2026-10-22

**Status: REVIEW GATES, not certification or production admission.** This document connects #1689–#1695 and records the falsifiers required to integrate the Stage D/E research branch into `main`. It is a review artifact, not an implementation claim. The branch also adds a dedicated Cargo CI workflow, because `crates/egglog-r2il-probe` is a workspace-excluded lab.

## Merge topology

- #1689 and #1692: source harvest, not executable admission.
- #1690: merged into main; MORK query_multi SET oracle and independent answer/cost split, plus egglog exact E1 Stage B. The MORK oracle is not a BAG execution engine; callback counts and deduplicated head tuples have different types.
- #1691: merged into main; separate three-valued Bangladesh site-service scalar oracle with independent grid, backup and telecom inputs. No grid risk becomes a measured tower-down observation.
- #1693: merged into main; E1 cold `ReplayCapsule`.
- #1694: merged into `lab/math-foundry-1-proof-replay-20261022`, not main.
- #1695: merged into `lab/math-foundry-2-certified-catalogue-20261022`, not main.
- This integration review is based on the latter branch after merge commit `67d060134e570c5cc99caa782af8fd6fcbeec5c3`; the catalogue and heritage remain lab-only.

## Gate P0: mathematical proof is not checked runtime substitutability

`artifact::ReplayCapsule::certify` proves `Arith::Exact` symbolic identities; `lower::run_steps` executes `INT_MULT` with *checked i64* semantics. For general inputs, a correct exact-integer final value is insufficient to establish that baseline and optimized programs have the same intermediate overflow/refusal behavior.

**Falsifier E-OVERFLOW-CANCEL:** For an eager checked baseline `a = i64::MAX * c; z = 0 * a`, input `c=2` refuses with overflow; a direct exact reduction of `z` to zero must NOT be accepted as equivalent on this input without an explicit valid-range/effect certificate. Test both `Ok(value)` and `Err(Overflow)` arms against a separately implemented, specified baseline. Do not assume egglog currently selects this reduction: verify extraction and keep a failure witness regardless of its tie-break.

**Admission:** either a cheaply checkable input-range witness covering every original intermediate step and the compiled steps, or an effect/error-preserving checked extraction. A certificate must include arithmetic domain and baseline evaluation semantics. A four-point i128 sample cannot discharge a quantified proof obligation. Remain in lab until falsified or resolved.

## Gate P0: contradictory or ambiguous premises

`Problem.premises` are fed to egglog as unrestricted `union` assertions while `oracle::solve` ignores a premise once its `lhs` is known. Duplicate contradictory equations, cycles and conflicting paths may therefore have no consistent assignment for arbitrary runtime inputs, yet a purely equational proof need not identify that domain restriction.

**Falsifier E-PREMISE:** `a=2*c; a=3*c` with `c=1` must not become a universally applicable compiled function. Require a coherent constraint/domain witness, or conservatively refuse ambiguous/cyclic/nonfunctional systems. Test duplicate `lhs`, cycles, disconnected premises and order permutation independently of egglog's extraction.

## Gate P1: cache is currently one entry per SemanticKey

`catalogue::Catalogue.entries` is `BTreeMap<SemanticKey, Entry>`; ABI and binding witness are stored inside `Entry`, not the map key. A second promotion for the same expression but a different target/binding **replaces** the earlier entry. The guard causes a conservative fallback, but the implementation does not yet demonstrate simultaneous multi-binding reuse.

**Falsifier E-MULTIBIND:** Promote the same expression for (target A, source generation 1) and (target B, generation 2), then request both. Either both remain separately retrievable, or the spec must label overwrite+fallback as a deliberate single-slot cache. For 256 × 4096 contexts, prefer an immutable `RecipeKey = (canonical term, typed laws, arithmetic domain, checker/toolchain version, target ABI)` and an independent **validated** `BindingKey = (dataset/version, scope, coordinate generation, source roots)`. A witness is supplied by the caller today; no source attestation is performed.

## Gate P1: independent proof verification and durable identity

`ReplayCapsule::recheck` regenerates egglog proof text with egglog and compares rendering bytes. That is conservative regeneration, **not** an independent proof checker, a portable proof format or an integrity-authenticated durable certificate.

**Falsifier E-PROOF:** mutate one premise, law/domain, proof edge, cost/target, lowered step, source root, checker identity or ABI; every material mutation must be rejected by an independently implemented verifier (or exact signed/hash-bound certificate format). A changed pretty-printer must not silently redefine proof validity. Pin the egglog, OGAR, R2IL and law-catalogue identities; branch=`main` git dependencies are not version pins.

Prefer a canonical typed proof DAG and explicit checker version; keep rendered text as a human-readable annex. Capsule digest, certification verdict and Thought-Git execution/commit receipts are three different objects.

## Gate P1: economic and execution certification

`CostEvidence` is provided by the caller and `pays_for_itself` checks a strict arithmetic inequality. This is an in-memory lab policy, not an independent performance measurement or statistical admission.

**Falsifier E-COST:** measure cold harvest/rewrite/proof-check/lowering plus lookup, binding, execution and fallback at fixed inputs; include a no-sharing competitor, an existing R2IL/relalg baseline, miss/stale/overflow arms, confidence ranges and source/version scope. Benchmark 256 templates × 4096 frames with at least two bindings for one template; measure total time and memory, not only the hot hit. Never relabel a synthetic test `CostEvidence` as empirical data.

## Gate P1: excluded-crate checks

The root workspace's green Build/Rust tests do not exercise this excluded crate. `.github/workflows/egglog-r2il-probe.yml` is the new targeted workflow and must finish green on this integration PR (fmt, clippy, release tests). The local Cargo toolchain and sibling ndarray checkout were unavailable to this review; all checks remain unverified until CI reports.

## Gate P2: historical mathematics, provenance and typed effects

`heritage::HeritageEntry` keeps source leads and mathematical capsules separate. Keep Leibniz and Euler **uncertified**. A modern formal reconstruction is not a checked historical attribution. Treat the 27 atoms as candidate semantic categories, not 27 machine opcodes or a universal grammar. Its current observed coverage is only `Const/Var/Multiply`.

The next study must version original source edition/folio, normalized statement, assumptions, theorem/checker, extraction/proof, target/ABI, actual input binding, output/overflow, and cold versus hot costs. D-TGB-4 should store **references** to these artifacts, not megabytes of proof text in each thought, and must keep proof checking, replay evidence and actual `LanceCycleWriter` commit acknowledgement distinct.

For #1691, reuse the typed **three-state physical oracle** only after independently versioned site/telecom/backup observations. Preserve `Unknown` and contradiction, per-tower GUID/window, and pure-versus-effect ordering. Do not map the synthetic legacy `grid_status` to independent observed grid power. Do not admit external `Gather` or physical side effects via algebraic equivalence.

## Admission decision

**Do not auto-merge.** This review enables a real integration diff and a missing CI gate; it does **not** certify the semantic counterexamples above as fixed. Maintain `egglog -> proof/rewrite` strictly cold and `R2IL -> FoldDialect -> mask-risc execute_into` as the permitted execution chain. Promote only after P0 invariants are either proven by witnesses/tests or explicitly restricted to the synthetic E1 admitted input domain.
