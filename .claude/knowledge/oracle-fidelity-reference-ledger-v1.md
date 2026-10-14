# D-ORACLE-0 source and status ledger (2026-10-14)

**Scope:** provenance map for a research proposal, not an external-oracle run. PR metadata and source snippets were inspected via connected GitHub. Every version must be frozen to immutable commit SHA for actual experiments.

## Original independent oracles

| Oracle | Source | Freeze/read before use | Status as of this research PR |
|---|---|---|---|
| OpenNARS Java | https://github.com/opennars/opennars | main source, inference implementations, stamps, truth functions, native test harness and versioning | Reference URL identified; NOT installed, run or evaluated |
| ONA C | https://github.com/opennars/OpenNARS-for-Applications | compiler flags, inference, event cycle, task inputs, choice/revision and native fixture syntax | Reference URL identified; NOT installed, run or evaluated |
| OpenPsi | https://github.com/opencog/openpsi | actual runtime/dependencies, needs/action traces, architecture of OpenCog integration | Reference URL identified; NOT installed, run or mapped to Rubikon |
| Hyperon Rust | https://github.com/trueagi-io/hyperon-experimental | Rust Cargo components, MeTTa test runner, bindings/rewrite trace, pin dependency SHA | Repository identified; NOT compiled, integrated or benchmarked |
| Pearl | Judea Pearl causal hierarchy, explicit SCM gold models (not merely SPO bitmasks) | structural causal model, confounding examples, intervention and counterfactual identification assumptions | External mathematical oracle specification must be written |
| Gadamer | Philosophical hermeneutics; no canonical numeric algorithm | formalize *our* policy with explicit name, admitted/denied evidence and context records | Engineering hypothesis; NOT claim of original executable Gadamer oracle |

Do **not** require native Hyperon, NARS and OpenPsi to return the same kind of truth object. The independent oracle may be a property or formal causal model where no directly runnable reference exists. Native-language and cross-model comparisons are different tables.

## Workspace evidence inspected on main

| Anchor | Grounded statement | Status / next falsifier |
|---|---|---|
| https://github.com/AdaWorldAPI/lance-graph/pull/1314 | multi-reading FSM with certain/alternative; was scoped to lexical grammar wave; current source may have more | Historical baseline, NOT comprehensive current-crate inventory |
| https://github.com/AdaWorldAPI/lance-graph/pull/1321 | position-vs-frequency homograph tests and language probing | Metrics are benchmarks on particular corpora, not ground truth for all grammar |
| https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/deepnsm-v2/src/lib.rs | currently exports belief, reason, tekamolo, hydrate, lexical, fsm etc. | Current built module presence verified by source read, integration/production wiring NOT verified |
| https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/deepnsm-v2/src/fsm.rs | parse_readings retains readings; answered_questions returns Pearl CausalMask for syntactic SPO coverage | Distinct grammar mask from causal proof semantics; F5 |
| https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/deepnsm-v2/src/tekamolo.rs | read_clause holds LaneHypothesis to right corner, then commits/abstains based on corpus evidence | Record each candidate rather than only committed ClauseRead output; F7 |
| https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/deepnsm-v2/src/reason.rs | derivation pointers, rung and fixed-point gates | Check identity/premise round-trip through engine |
| https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/deepnsm-v2/src/hydrate.rs | AddressedTriple + non-global-deduplicating scanpath preserving revisits | Preserve trajectory distinct from set coverage |
| https://github.com/AdaWorldAPI/lance-graph/pull/1319 | X17 was measured: derived empty stamp allowed pooling source already in ancestry | Re-run red-first on frozen main; F1/F2 |
| https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/deepnsm-v2/src/belief.rs | observe/revise_at has disjoint stamp guard, but close_transitive inserts derived entries with Stamp::default(); source id % 64 folding | Must not conflate derived/no direct receipt with provably disjoint; F1/F3 |
| https://github.com/AdaWorldAPI/lance-graph/pull/1407 | CE64 strict ISA and field contracts, revision truth-only, intervention and counterfactual refuse | Unsupported must stay refusal; test boundary/saturation |
| https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/causal-edge/src/isa.rs | truth::evidence_weight saturates; code comment records overflow/NaN-to-zero known defect | F4; math oracle and expected behavior first |
| https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/causal-edge/src/pearl.rs | SO/PO/SPO bits are described as Pearl ladder projections | A projection is not the required causal identification proof; F5 |
| https://github.com/AdaWorldAPI/lance-graph/pull/1410 | signed register law types, RelativeOffset/AxisPosition/Support, refused overflow | Separate law-authorized readings; no confidence/anaphora reinterpretation |
| https://github.com/AdaWorldAPI/lance-graph/pull/1408 | Nars16x8, MoorePalettePairs, MooreNars16 tenant contract | No automatic production consumer; check before assuming |
| https://github.com/AdaWorldAPI/lance-graph/pull/1227 | three incompatible 144-predicate design lineages | Do not choose a new authoritative codebook without full mapping |
| https://github.com/AdaWorldAPI/lance-graph/pull/1294 | P=0 where predicate palette absent; previously wrong SUPPORTS/CONTRADICTS inversion | Check mapping and writer before claims about actual CE64 predicate semantics |
| https://github.com/AdaWorldAPI/lance-graph/blob/main/.claude/plans/dialectic-engine-v1.md | prior S1-S12 design decisions: reachability vs NARS truth and stamp decisions | Reconcile original-oracle parity against prior decisions, never overwrite silently |
| https://github.com/AdaWorldAPI/lance-graph/pull/981 | contextual AttentionFocusFacet reading; axes are semantic neutral; prefix restrictions not address bitwise-OR | Same bytes can have different readings without changing ALU |

**Known unknowns to keep explicit:** PR #495/#497/#850 historical claims need direct local reading; correct sources for 144 palettes; OpenPsi runtime compatibility; native Hyperon trace API; oracle seed/config schema; full 144 input/expected fixture contents; whether X17 has been fixed on another worktree or unmerged branch; correctness of blanket Sim transitivity. Do not mark any unknown as DONE merely because the feature appears in a PR description.

## Naming and accountability

- Original = runnable upstream reference at recorded commit.
- Independent gold = explicit small mathematical/formal reference, constructed separately from implementation under test.
- LG candidate = current or experimental AdaWorldAPI implementation.
- Model parity = exact native behavior under comparable inputs and declared tolerances.
- Abstraction equivalence = preservation of model-specific observables under documented mapping.
- Performance = measured physical cost **after** parity, including hidden preparation/maintenance.
- Non-comparable = outcome preserved as such; no implicit score of zero or success.
