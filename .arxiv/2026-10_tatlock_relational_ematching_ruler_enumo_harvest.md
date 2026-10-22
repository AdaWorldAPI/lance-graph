# Tatlock / PLSE: relational e-matching, Ruler, Enumo

**Survey date:** 2026-10-22. **Status:** PRIMARY-SOURCE RESEARCH NOTES, not an LG implementation or approval to add an optimizer. This is a paraphrased paper harvest; numerical results belong to the cited authors' benchmarks.

## Sources and evidence scope

- Zhang, Wang, Willsey, Tatlock, [*Relational E-matching*](https://ztatlock.net/pubs/2022-popl-rematch/2022-popl-rematch.pdf), POPL 2022, DOI [10.1145/3498696](https://doi.org/10.1145/3498696). **Author PDF inspected** (§§1, 3–5). [Project page](https://ztatlock.net/pubs/2022-popl-rematch/).
- Nandi et al., [*Rewrite Rule Inference Using Equality Saturation*](https://ztatlock.net/pubs/2021-oopsla-ruler/2021-oopsla-ruler.pdf), OOPSLA 2021, DOI [10.1145/3485496](https://doi.org/10.1145/3485496). **Author PDF inspected**; [project page](https://ztatlock.net/pubs/2021-oopsla-ruler/).
- Pal et al., [*Equality Saturation Theory Exploration à la Carte*](https://ztatlock.net/pubs/2023-oopsla-enumo/2023-oopsla-enumo.pdf), OOPSLA 2023, DOI [10.1145/3622834](https://doi.org/10.1145/3622834). **Author PDF inspected** (§§2–3); [project page](https://ztatlock.net/pubs/2023-oopsla-enumo/).

## 1. Relational e-matching: use joins to find rewrite opportunities

**Paper-derived:** E-graphs represent e-nodes grouped into congruent e-classes. Traditional e-matching follows patterns by recursive backtracking. The paper instead translates each function symbol's nodes into a relation of child-class IDs and parent-class ID; matching one term pattern becomes a **conjunctive query** over these relations. Equality constraints (repeated pattern variables) become explicit join conditions rather than checks deferred until late in recursion. Generic join can then exploit variable ordering, functional dependencies and batching. Paper notes nonnested patterns should use a simple scan fast path, since building join indices can be wasteful. The worst-case-optimality theorem holds for e-matching under the paper's fixed-pattern/data-complexity assumptions; it does not imply that arbitrary LG joins or recursive paths are worst-case optimal.

**LG transfer hypothesis:** This is the closest correspondence to the existing global-to-local bind idea: a cold compiler can gather candidate term bindings into a **typed indexed focus domain** before applying many rewrite rules. Instead of re-walking a symbolic AST for every rule, compile patterns to relational joins with pinned e-class keys. But the e-class relation is the compiler's own relation over terms; a D-KEY16, same integer ordinal, or coincident source row coordinate is **not** automatically an equivalent e-class ID or entity identity.

**Small test REM1:** Run 256 overlapping typed rewrite patterns against a bounded e-graph using (a) simple recursive matcher and (b) generic-join-inspired matcher. Compare *complete match substitution multisets* via independent brute force. Measure cold index creation, variable-order search, matches/s, peak memory, proof identities and deterministic result order. Supply repeated-variable patterns, disconnected multipatterns, heavily shared symbols and degenerate one-node scans. A pattern-dependent win does not excuse unbounded index build cost.

## 2. Ruler: generate candidate laws instead of hand-authoring every rewrite

**Paper-derived:** Ruler reduces the exploration needed to infer equational rewrite rules by using equality saturation to compact many equivalent enumerated expressions, then filters candidate laws. The reported comparison found 5.8x smaller rulesets at 25x faster inference versus the specific CVC4-based baseline while retaining similar proving power on their benchmarks. These are research results for the tested languages, not an invariant over every domain.

**LG transfer hypothesis:** Treat Ruler as a **rule proposal factory**, not a source of automatically admitted laws. Existing contract laws own semantic authority. Candidates should come with a declared domain and rule provenance:
- Bool2 AND/OR on co-addressed masks;
- SQL K3 with validity (do not identify invalid NULL with valid FALSE or a separately modeled logical UNKNOWN);
- exact i32/i64 with specified overflow and terminal semantics;
- Bag versus Set/Distinct plus provenance;
- approximate f64 operators with explicitly defined accuracy/rounding policy.
A rule verified over finitely enumerated examples is still not necessarily valid over all operands or mixed effects.

**Small test RUL1:** Generate a candidate identity with Ruler-style enumeration over a tiny typed grammar; reject `UnionAll(R,R) = R` on a duplicate-row counterexample; accept `And(x,All) = x` only on a witnessed co-addressed Bool2 plane. Revalidate on a separate exhaustive microdomain, then independent evaluator; do **not** use synthesis's original samples as the sole oracle.

## 3. Enumo: programmable search policy beats one monolithic law generator

**Paper-derived:** Enumo is a DSL for programmable **equational theory exploration**. Users compose term enumeration, candidate generation and rule filtering/selection rather than accepting one opaque algorithm. Its fast-forwarding technique explores rules without requiring a target-language interpreter in that stage. This does *not* eliminate the need for a semantic oracle before LG accepts a law.

**LG transfer hypothesis:** Define small reproducible research strategies, not a production DSL:

```text
language=Bool2; grammar={And,Or,Not,All,Empty}
exploration=bounded(depth=4, terms=...)
filter=typed_oracle + independent_counterexamples
export=candidate_rule_id + premises + guard + proof_ref
```

Explore **separate** grammars for nullable SQL and bag/multiplicity. Keep a stable fingerprint of the original parser/law version, and invalidate cached rules when those versions change. This is particularly useful when 256 rules share only a small number of primitive operators.

**Small test ENUM1:** Two strategies explore the same bounded grammar: unguided enumeration and guided/enriched workload. Compare unique sound admissible laws per second, redundant laws, false candidates caught by independent oracle, e-graph growth and stale-law refusals. An optimization that learns more invalid rules is strictly worse.

## Contract boundary and suggested sequence

```text
      Ruler / Enumo (offline proposal)
                |
      typed law + oracle + guards
                |
  egglog e-classes + matching (relational matching candidate)
                |
  eggcc-inspired effect-safe extraction
                |
 contract -> relalg -> R2IL FoldDialect -> Quack -> mask-risc
```

This is a **research design**, not implemented end-to-end. Do not replace the established `lance-graph-relalg` ownership or R2IL SIMD polyfill. D-RA-2=A owns canonical set-law admission in `lance-graph-contract`; ADMISSION=B continues to prohibit foreign-value-directed cross-table positional reads. A faster matcher changes nothing about coordinate authority or write permission.

**Falsification gate:** A useful new matcher or inferred rewrite must preserve exact typed outputs, show a net compile+bind+execute+verify improvement, refuse stale focus/invalid law domains, and produce replayable provenance. Otherwise keep existing normalization and matchers.

**Linked notes:** [egglog](2304.04332_egglog_equality_saturation_datalog_excerpt.md), [eggcc](2026-10_eggcc_effect_safe_extraction_statewalk_r2il_map.md), [3LA](2026-10_3la_formal_interface_flexible_matching_application_validation.md).
