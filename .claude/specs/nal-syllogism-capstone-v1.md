# NAL Syllogism Capstone — figure resolution on CausalEdge64 (v1)

**Status:** KERNEL SHIPPED (`causal-edge::syllogism`, branch claude/jolly-cori-clnf9, f499546 + review fixes). Capstone INTEGRATION pending — this spec is the roadmap.
**Council:** 5-agent dev council (R1–R5, Opus, full-file reads per `E-READ-NOT-GREP`). R1/R2 SOUND (NAL-correct; firewall-clean; truth-fns byte-identical to `ndarray::hpc::nars` AND `forward()`). R3/R4 SOUND + doc fixes applied. R5 FIX-NEEDED **at the integration layer only** — the kernel is "a correct, well-documented O(1) kernel with zero callers." The capstone claim is real in shape, unwired in fact.

## What shipped (the kernel)

`CausalEdge64::figure(other) -> Option<Figure>` + `syllogize(other) -> Option<Syllogism>`. `Figure{Chain,ChainRev,SharedSubject,SharedObject}` = which SPO palette term two edges share, by integer equality — the hardwired analogue of the Pearl 2³ mask. The conclusion edge carries outer terms + canonical NARS truth (Deduction/Induction/Abduction) + signed v2 mantissa (+1/+2/−1) + AND-ed Pearl mask. 14 tests v2 / 13 v1; clippy/fmt/rustdoc clean.

**Scope boundary (R1, intentional):** the omitted Comparison/Exemplification/Analogy/Resemblance need a `<->` (Similarity) copula — `CausalEdge64` is *always* directed inheritance and has no copula field, so a `<->` conclusion is unrepresentable in this carrier. Correct to omit, not a gap.

## The capstone claim (user) and where it stands (R5)

> "NAL syllogism notation is the missing capstone for glueing all 3 reasoning methods with the 10-rung ladder and the JITson cranelift templates vs elixir. NAL notation and elixir complete each other."

- **3-path glue** — *designed, not wired.* All three paths produce/consume `CausalEdge64`; `syllogize(self, other)` is the shared binary op. But Path2 (EW64) stores `EdgeRef{family,local}` (NOT edges) and Path3 (WitnessTable) is a column-primitive not yet wired to edges. The resolvers don't exist yet.
- **10-rung ladder** — *aspirational.* No code maps `InferenceType`→`RungLevel`/`Operation`-lane. The ladder a syllogism *does* ride today is `CausalMask` (Pearl Level 1/2/3, already AND-ed in the conclusion), not `RungLevel`. `atoms::I4x32::pack/unpack` is still `todo!()` (blocked on 32-vs-33 + per-group sign/scale) — no lane to write.
- **cranelift ↔ elixir** — *thesis, not code.* `notation()` returns 4 hard-coded strings consumed by nobody but tests. No emitter turns a figure into a `JitTemplate` (`jit.rs`) or an Elixir clause. The SHARED declarative source both should compile from does not exist yet — **this is the actual missing capstone piece**, not the `Figure` enum.

## Roadmap (prioritized, firewall-clean, offline-first)

- **[P1 — THE capstone] `FIGURE_RULES` declarative table + dual emitters (R5 #3).** Replace the 4 hard-coded `notation()` strings with one `const FIGURE_RULES: [FigureRule; 4]` carrying `{figure, premise_shape, conclusion_shape, rule: InferenceType, mantissa: i8}`. Add two *pure offline* emitters that BOTH read it:
  - `fn jit_template(FigureRule) -> JitTemplate` — emit the JSON `jit.rs` accepts (a branch-free "match shared term → apply truth-fn" kernel; sibling of `cam_pq_cascade_template` in `jitson_kernel.rs`).
  - `fn elixir_clause(FigureRule) -> String` — emit the function head whose **doubled binding** (`def figure(m, p, _, s)`, M bound twice) IS the middle-term unification `figure()` does by integer equality. *This is the literal "Elixir and NAL complete each other": one table, two backends.* No Elixir runtime needed to emit.
- **[P0 — unblocks Path2] EW64→`CausalEdge64` resolver seam.** In a driver (NOT zero-dep `episodic_edges.rs`): `resolve(EdgeRef, &basin, &class) -> CausalEdge64` honoring the EdgeRef grammar — `family==0` ⇒ row's own basin; `family∈1..=15` ⇒ `class.cross_family_palette[family]` (OGIT class); `local` is **1-based** (`basin[local-1]`). Then `syllogize_hot(EpisodicEdges64,…)` folding **slot-0-anchored** (`hot[0].syllogize(hot[k])`, k=1..n) — NOT a blind left-fold (syllogize is non-symmetric; a left-fold `None`-cascades). ≤3 conclusions.
- **[P1 — correctness upgrade] Re-express `nars_engine::combinatorial_entailment` on `syllogize`+`Figure::Chain` (R4).** Today it assumes the chain positionally and **never checks `o1==s2`** — delegating absorbs that safety check. A real bug-fix, not just DRY.
- **[P1 — DRY] Factor `forward()`'s 4 truth arms onto the `syllogism.rs` private truth-fns (R2/R4/R5 #4).** Deduction-truth is now byte-identical in 5 places; collapse the two intra-`causal-edge` copies so a future tweak changes one site.
- **[P0 — unblocks Path3] WitnessTable→premise resolution.** `w_slot → WitnessEntry.spo_fact_ref → CausalEdge64`, so Path3 supporters also reach `syllogize`.
- **[P2 — gated] `InferenceType→RungLevel`/`Operation`-lane map (R5 #5).** Deduction→Shallow, Induction/Abduction→Structural/Counterfactual, mirroring the Pearl levels. **Gated** on `atoms::I4x32::pack/unpack` being un-`todo!()` + the 32-vs-33 + per-group sign/scale BLOCKED decisions.
- **[P2] Driver integration tests** (EW64 fold + witness-backed premises) — current tests only exercise the pure kernel.

## Firewall

Integer palette equality PROPOSES the figure (deterministic, offline, no float on the structural path); the NARS truth-function ADDRESSES. The codegen (P1) is pure offline string work — no LLM, no float on the hot path. Cross-ref: `E-NARS-FIGURE-CAPSTONE`, the PROPOSE/ADDRESS doctrine, `E-READ-NOT-GREP`.
