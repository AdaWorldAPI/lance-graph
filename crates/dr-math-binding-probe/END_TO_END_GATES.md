# D-DRMC-4 — integration, harvesting and downstream acceptance

**Status: STAGED LAB TEST + FUTURE DEPENDENCY MATRIX, not production completion.**

## What this stacked PR actually exercises

- An actual `ClassView` fixture declares `memberOf -> effectiveRoles` via the existing `compute_dag` method and its Kahn-order contract.
- Stage 1 binds typed `Identity×Property` DR64 planes as borrowed words against caller-supplied source/map/focus/scope numbers.
- Stage 2 compares constrained finite-focus observations to reference mask counting (and its scalar differential tests).
- Stage 3 gates a *synthetic* mathematical proof reference against a distinct compiled artifact identity. **The checked proof flag is not real checker attestation.**
- The existing `lance-graph-act-r` crate accepts eligible outputs to arbitrate a deterministic decision proposal and refuses a changed snapshot. No ActionDef or write, no Cognitive Shader Driver dependency.

This tests boundary shapes and refusals only. It does **not** run an egglog theorem compiler, execute R2IL, attest a Lance snapshot or execute a real class method. A positive CI job is *not* the science result.

## Remaining staged, independently owned gates

| Gate | Reused owner / implementation | Proof or falsifier |
| --- | --- | --- |
| G0 Class method -> operation | `class_view::compute_dag`; D-QSP/D-RFT #1667 and EXO-121 | Method-specific law, declared dependencies, non-participating fields, no semantic label inside hot register |
| G1 Real source / map seal | OGAR, D-RMAP-1A separate RelationshipMaps/MapPalettes, Quack binding | Immutable dataset+branch+version; map/index row-address seal; scope/ACL; coordinate rollover; >64 focus frontier; no invented foreign gather |
| G2 Discovery and constraint scope | DR bounded register + independent population observer | Empty/unknown/one false/out-of-focus counterexample, bag vs distinct; held-out reality generalization explicitly separate |
| G3 Oracle intake | MeTTa declaration frontend; MORK/PathMap unification; datafrog recursion; scalar/PG K3 | Count Set and BAG separately; output identity vs support; per-loop cycle; no self-support/negative recursion silently accepted |
| G4 Candidate rewrite discovery | Ruler/Enumo-style rule proposal, `set_rules`, egglog | Type/sort and arithmetic law guards, explicit premise consistency, finite fuel, independent oracle differential |
| G5 Effect-safe extraction | eggcc-inspired stateful extraction / GKAT `gkat-probe` | Prefetch and source/permission gates not reordered; K3 guard valid; side-effect proposal not committed; causal assumptions declared |
| G6 Certified compilation | `egglog-r2il-probe` #1690–#1698 (owner) | External portable proof checker, range/overflow/domain witness, checked target ABI, reproducible law/compiler version |
| G7 Physical execution | R2IL/Quack lowering → mask-risc/ndarray SIMD; direct prepared-DR native folds | Same answer vs independent oracle; no second hot evaluator; count bind/index/readahead/prefetch/scratch/output costs |
| G8 Deterministic orchestration | ACT-R, Rubicon, Kanban, cognitive-shader-driver; Thought-Git separate | Stale eligibility refusal; proposal vs action permission vs commit receipts; replay traces and separate writing authority |
| G9 Heritage and cross-domain reuse | Thought-Git audit references; proposed 27-atom hypothesis | Historical source quotation/attribution separately verified; one law + proof + compiled recipe, two different classes and snapshots; counterexample discovers bad abstraction |

### Math heritage example (not implemented)

A *product rule* ascribed to Leibniz is a source lead. Only after faithful source transcription, formal differentiation semantics, independent proof, domain constraints and target lowering may it be called a compiled theorem. Merely storing a label `Leibniz` or egglog rendering a term is no certificate. The 27 mathematical low-code atoms are a falsifiable empirical compression *hypothesis*; measure coverage over published proofs, not a universal basis claim.

### Performance / amortization

Design a controlled 256-rule × 4096-frame experiment with a baseline that must include non-sharing, cold compile/certification, binding/index building, source snapshot validation, state updates, warm/cold caches, direct DR register SIMD vs generic Quack/R2IL, generated code size, and false-positive/escape frontier. An expected fast fold by itself does not amortize a slow binder or compiler. No timings asserted here.

### Ownership and flow

- Do not modify overlapping #1698 mathematics-correction branch, #1667 method/SQL parity plans, EXO-98 Thought-Git storage, or `cognitive-shader-driver`.
- Preserve the fast resident-register SIMD ruling for #1685: consuming already-prepared bits is **not** a forbidden bypass of R2IL; R2IL remains the general polyfill, not compulsory redispatch of prepared operands.
- A law proof is not evidence of a causal connection in nature. Theorems, observational finite certificates, generalization evidence, executable artifact and committed effects are independently versioned objects.
- No merges or production API promotions without a separately owned change, targeted CI, independent truth oracle and reviewer agreement.

## Run

`cargo test --manifest-path crates/dr-math-binding-probe/Cargo.toml`

`cargo clippy --manifest-path crates/dr-math-binding-probe/Cargo.toml --all-targets -- -D warnings`

`cargo fmt --manifest-path crates/dr-math-binding-probe/Cargo.toml --all -- --check`

This crate has its own workspace and is intentionally not covered by every root-workspace test. The workflow in this PR provides a separate CI gate.
