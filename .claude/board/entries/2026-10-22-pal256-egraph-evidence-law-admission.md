# Cold mathematical candidate laws for typed evidence families

Scope **test-only / not egglog rule admission yet**. The Math Foundry E1 probe (#1690/#1693/#1696/#1698), R2IL and mask-risc remain sole owners. MeTTa/MORK/OpenNARS are source languages or independent differential oracles, never replacement hot execution engines.

Candidate laws:
1. `tanh(atanh(x)) = x` on the open real unit interval; finite-precision LUT cancellation needs a separate quantization/error proof.
2. `tanh(a+b)=(tanh(a)+tanh(b))/(1+tanh(a)tanh(b))` on finite real operands.
3. NARS positive/negative independent evidence counts merge by additive counts.
4. Bayesian log-likelihood contributions add only with correctly identified prior, likelihood and conditional independence.

Refuse:
- mapping arbitrary embedding cosine to Pearson sample r or sample-size evidence;
- treating `sum(atanh(cos))` as NARS revision;
- `c=tanh(|z|)` as sample-aware confidence;
- global e-graph `3σ -> 0` equivalence;
- float re-association, saturation, LUT roundtrips, or non-idempotent memory effects without typed guards.

Next cold stage: attach admitted typed predicates to existing `egglog-r2il-probe` *without* adding a second e-graph, emit R2IL via `FoldDialect`, independent oracle/replay capsule, compare to baseline under altered binding and ABI witness. Keep proof/law, compiled recipe, dataset/source evidence and execution receipts separate.

Run: `cargo test --manifest-path crates/egglog-r2il-probe/Cargo.toml --test palette256_typed_evidence_laws`.
