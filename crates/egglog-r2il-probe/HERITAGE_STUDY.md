# Stage E: mathematical heritage / 27-atom falsifier (lab only)

**Dependency chain (2026-10-22):** [#1690](https://github.com/AdaWorldAPI/lance-graph/pull/1690) (E1 compiler) and [#1693](https://github.com/AdaWorldAPI/lance-graph/pull/1693) (Stage C capsule) merged into `main`; [#1694](https://github.com/AdaWorldAPI/lance-graph/pull/1694) (Stage D catalogue) merged into the Stage C feature branch, and [#1695](https://github.com/AdaWorldAPI/lance-graph/pull/1695) (this Stage E) into the Stage D feature branch, not into `main`. Draft integration [#1698](https://github.com/AdaWorldAPI/lance-graph/pull/1698) carries the combined lab stack and dedicated excluded-workspace tests; it is not production-admitted.

## What this implements

The Rust `heritage` module distinguishes:
1. **Source lead**: an author/work/edition candidate. NOT a reviewed facsimile, historical influence proof or mathematical certification.
2. **Normalized mathematical statement**: a modern reconstruction with its assumptions, not necessarily the author's original proof.
3. **Checked computational artifact**: currently only the synthetic E1 exact-integer substitution, compiled by egglog and replayed through R2IL scalar `FoldDialect` with independent i128 oracle. E1 uses empty planes and scalar `INT_MULT`; it does NOT yet exercise a mask-RISC `execute_into` population fold. An exact symbolic proof is not a proof of checked-i64 intermediate-overflow equivalence for arbitrary inputs (see `REVIEW_GATES.md`).

The Leibniz 1684 product-rule and Euler-identity entries are **uncertified source candidates**. Neither is executable, because the current E1 language has only `Num/Var/Def/Mul` and does not express differentiation, complex exponentials or the necessary side conditions. A test refuses both, by design.

## 27-atom hypothesis, not a discovery

Candidate vocabulary: six term constructors, eight algebra operators, seven relational operators, six derivation constructs. These are *research-level semantic categories*, not R2IL opcodes or claims that 27 operations suffice for all mathematical domains. No atom is considered "covered" unless an independently certified compiled procedure exercises it. Today the synthetic E1 program witnesses just `Const, Var, Multiply`.

Measure on a future licensed corpus:
- Historical coverage: reviewed editions and unambiguous citations
- Formalization coverage: statements expressed with domain/assumption witnesses
- Proof coverage: mechanically checked lemmas, including independent proof-checker identity
- Lowering coverage: lawful mappings to extant R2IL opcodes/folds
- Minimum description length: vocabulary bytes + compiled corpus bytes; nonsharing control
- Replay cost, proof bytes, binding costs, cache hit rates and provenance retention

**A shorter encoding is not evidence of semantic preservation.** The optimizer must refuse a proposed common template that changes SQL NULL/UNKNOWN, Set/Bag, numeric overflow, effects, source identity or historical attribution.

## Thought-Git learning × audit integration (future gate)

Do not duplicate Thought-Git's store or insert proof events into the nanosecond hot loop. Its existing D-TGB-4 receipts and landed commit witnesses are distinct from egglog proof DAGs. The future link is a reference-only citation from a versioned derivation or historical study row to:
- certified law identity and assumptions
- compiled procedure identity and checker version
- source edition/provenance and any uncertainty
- runtime binding witness (separately invalidatable)
- promotion/demotion decision, counterexamples and cost samples

A replay recipe should be sufficient to recompute the outcome if the source snapshot/operands, codebook and checker are available. A hash alone is not a replay. An unsigned witness fold is not a cryptographic proof. Rubicon authorization and real LanceCycleWriter integration are out of scope.

## Next falsifiable stages (not shipped here)

1. Formalize the Leibniz product rule against an independent differentiation theorem checker. Preserve exact domain conditions, such as differentiability of both functions.
2. Add an explicit mapping from the formal derivative term graph into the low-code hypothesis, measuring needed *new* atoms instead of forcing a 27-atom success.
3. Compare Leibniz- and Newton-era equivalent formulas only after historical source reviews; equivalence does **not** establish intellectual influence.
4. E2: 256 templates × 4,096 focus contexts = 1,048,576 applications. Compare cold certified catalogue vs existing relalg/CSE baseline, share/no-share controls and **total** execution cost. No demonstrated throughput claim yet.
5. Add read-only Thought-Git proof/reference records and replay tests after EXO-98 owner coordination.

## Verification

`cd crates/egglog-r2il-probe && cargo test` (requires sibling OGAR and ndarray checkout).
No runnable historical theorem is claimed. The three source/sample fixtures remain entirely outside production.

**Integration check (2026-10-22):** #1698's new isolated Cargo job was triggered and stopped at `cargo fmt --check`; Clippy and tests were consequently skipped on that first run. Formatting and a complete targeted green CI run are open, independent of the semantic gates. The main-workspace checks are not substitutes.
