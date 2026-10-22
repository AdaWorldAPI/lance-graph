# 3LA 2024: formal semantics should choose and validate polyfill offloads

**Source:** Bo-Yuan Huang et al., *Application-level Validation of Accelerator Designs Using a Formal Software/Hardware Interface*, ACM TODAES 29(2), Article 35 (February 2024), [DOI:10.1145/3639051](https://doi.org/10.1145/3639051), 25 pages. **Material read:** user-supplied PDF `3639051.pdf`, 2026-10-22. This is **not presented as an arXiv preprint**: the `.arxiv` directory is the project's research-note shelf, not a claim about publication venue.

**Copyright/harvest mode:** bibliographic citation, original paraphrase and independent application mapping only. No full-text reproduction, figure copying, or redistributable PDF. **Status:** source-grounded research, hypothetical LG transfer, no deployed 3LA/ILA integration.

## Transcript of the ideas, paraphrased and ordered as the paper argues them

**T1. The missing loop is end-to-end application validation (§§1–2).** An early accelerator often has neither bespoke compiler integration nor a simulator fast enough for a full workload. Yet operator-level correctness is insufficient when quantization and shape or format conversions accumulate error. The authors target unmodified applications and early prototype designs.

**T2. One formal interface drives three tasks (§§2.1, 3.1).** The instruction-level abstraction (ILA) specifies observable accelerator state, instruction decode conditions and state updates, like an ISA for special-purpose devices. The same semantics support selecting accelerator instructions, emitting the corresponding MMIO commands, and generating an instruction-level simulator. ILA can separately be checked against RTL; the paper does *not* claim the compiler itself is automatically fully verified against arbitrary hardware.

**T3. Compiler/source granularity and device granularity rarely match (§§3.2–3.3).** A coarse hardware operator can correspond to many compiler-IR operations. The developer supplies one IR-to-accelerator mapping per accelerator operation, many-to-many where necessary, expressed over sizes/shapes rather than brittle literal syntax. Mappings can be validated using reference IR evaluation versus ILA simulation.

**T4. Flexible matching is more powerful than exact pattern matching (§3.3).** Generic IR algebraic rewrites and device-specific mapping rules coexist in equality saturation. E-graphs expose semantically equivalent shapes even if the source importer chooses a different tree. An extraction cost function chooses among candidates; their functional-validation-oriented experiments initially maximize accelerator calls, rather than establish a globally optimal runtime cost.

**T5. Constructive evidence beats a syntactic match (§§3.3, 5.3).** One LSTM RNN is represented by 566 Relay operators yet maps to one FlexASR operation. The authors report 66 FlexASR matches for a Transformer and 38 for ResMLP where exact matching found none. Glenside's general rewrites also enable convolution offloads onto accelerators without a direct convolution-specific mapping. These are compiler-match results, not proof that all rewrites are valid on all numeric formats.

**T6. Co-simulation finds a real design-level failure (§§3.5, 5.5; Table 4, p.18).** HLSCNN used narrow fixed-point values; per-operator testing did not reveal how severely weight quantization damaged full-model accuracy. ResNet-20/CIFAR-10: 91.55% reference, 29.75% initial HLSCNN, 92.10% after numerics tuning. MobileNet-V2/CIFAR-10: 92.40% reference, 10.35% initial, 91.50% tuned. MobileNet-V2/ImageNet: 72.30% reference, 0.10% initial, 68.85% tuned. The fix expanded weights to 16-bit fixed point and adjusted fixed-point binary positions; recovery is substantial but ImageNet is *not* identical to its reference.

**T7. The simulator is useful but not free (§§5.5–5.6).** Their automatically generated functional simulators allow early full-application exploration. For FlexASR the ILA simulator was reported 30× faster on average than RTL simulation; modeled offloads still have significant wall-clock simulation costs. Synthetic workloads on a FlexASR FPGA matched its simulator bit for bit. The paper's acceleration comparisons use operator cycle counts, not blanket live-system wall-clock throughput or mask-RISC numbers.

## LG application map: the interface is the missing formal seam

The conceptual import is not TVM, Relay, ILAng, custom MMIO, or a new executor. It is **specify observable semantics once, then derive valid offload selection, reference behavior and regression oracles**.

```text
MeTTa / MORK / Datalog / SQL / JC source
             |
      typed set/bag/K3/nullable contract
             |
  contract set algebra + relalg plan
             |
   egglog guided law search / CSE
             |
 eggcc-inspired effect-safe extraction
             |
    R2IL: typed FoldDialect procedure   <-- candidate formal SW/exec interface
             |
     Quack physical selection
             |
    mask-risc execute_into
             |
  independent scalar result / lineage oracle
             |
 authorization / source-version check / actual store acknowledgement
```

This is an **architectural research diagram**, not a statement that formal ILA models exist for R2IL or that the route above is completely implemented. Respect actual owners: `lance-graph-contract` owns canonical single-table set semantics under D-RA-2=A; `relalg` handles relational planning/lowering; R2IL supplies the SIMD polyfill; Quack/mask-risc provide the actual hot execution route. Existing `.claude/plans/d-relalg-unified-v1.md` is explicitly partly proposal.

### Proposed small machine-checkable contract surface

For each admitted physical procedure, pin:

```text
op_id, ABI_version,
operand_kinds, output_kind,
coordinate_domain + source_generation,
truth_domain (Bool2 | SQL_K3_with_validity | nonBoolean),
multiplicity (Set | Bag | Distinct),
numeric_domain + overflow/rounding,
allowed_pure_rewrites + their guards,
reads/writes/effect_order + authorization,
reference_scalar_semantics, refusal_cases
```

Do **not** invent a NULL *value* primitive where validity already carries SQL NULL. Three meanings remain distinct: (a) **valid FALSE**, (b) SQL **NULL/invalid**, (c) logical **UNKNOWN** in the chosen three-valued interface. The exact external representation is a contract to pin against current merged code and tests; this note does not reassign its bit layout. `COUNT`/Bag is not Boolean existence or Distinct. Equivalence proof does not prove physical binding, source freshness, or storage commit.

### How 3LA changes the optimization economics

- **Compile offloads into what already exists.** Candidate rewrite patterns target R2IL operations and existing Quack/mask-risc terminals rather than adding a second instruction VM. Map high-level `SUM/GROUP_SUM/KEEP`, predicate compositions and separately admitted join patterns to their actual lowerings.
- **Flexible matching can expose a pattern accidentally hidden by an importer.** A rewritten `And`, filtered aggregation, or co-addressed fold may surface a reusable operation without matching its original AST shape. This is the analogue of the paper's im2col/linear-layer discovery, *not* proof of a specific LG speedup.
- **Cost extraction is a separate decision.** Distinguish compile/saturation/extraction time, binding/preparation, source reads, SIMD folds, output materialization, dirty update, replay/checker and commit. “More offloads” can be strictly worse than fewer offloads when materialization or bind dominates.
- **Simulate full workflows, not merely operator identities.** An operator may be locally correct but change downstream SQL NULL behavior, multiplicity, overflow, epistemic state or result provenance. Compare the final typed observable outputs with an independent baseline, not just equal bit counts.

## Minimum independent falsifiers

| ID | Test | Invariant |
|---|---|---|
| LA1 shape-flex | Three equivalent pure predicates expressed via different nested AST shapes | Flexible matching yields one common legal fold, or explicitly reports a missing law; exact/scalar results match. |
| LA2 NULL/FALSE | SQL nullable operand cases: valid false, NULL invalid, and a separately defined logical UNKNOWN | No NULL collapses into valid FALSE; WHERE and projections agree with a PostgreSQL differential oracle under pinned validity rules. |
| LA3 Bag/Distinct | Two paths reach the same target, duplicate rows in input, `COUNT(*)` and `EXISTS` | Both semantics are separately correct; a Set-only rewrite cannot erase multiplicity. |
| LA4 source identity | Two sources use the same numeric local row ordinal for different identities; then generation changes | Different identities never become co-addressed by coincidence; stale bindings reject, even for an algebraically legal expression. |
| LA5 numeric boundary | i32 exact sufficient statistics versus f64 reorder/saturation cases | Integer legal factorizations preserve exact outputs; float reassociation needs an explicit bound/tolerance and cannot silently claim bitwise equality. |
| LA6 whole pipeline | Long mixed predicate/aggregate workflow and independent scalar interpreter | Test final result, NULL/Bag semantics, lineage/dirty distinctions and refusal behavior, not only intermediate fold equivalence. |
| LA7 economics | Existing R2IL/Quack baseline versus bounded e-graph extraction on reusable and nonsharing cases | Require an end-to-end win after binding and compilation amortization; otherwise keep the original route. |

## Ruling

**Research harvest only.** Prioritize a *contract-driven reference oracle + offload-validation harness* for one already-shipped R2IL subset, not a new compiler stack. If a formal machine interface proves useful later, its first artifact should be typed observable semantics and rejection tests, not synthetic MMIO or claims of complete hardware equivalence. Hardware ILA/RTL verification in 3LA is an inspiration, not a capability this LG note demonstrates.

**Adjacent records:** [egglog primary harvest](2304.04332_egglog_equality_saturation_datalog_excerpt.md); [eggcc effect-safe extraction](2026-10_eggcc_effect_safe_extraction_statewalk_r2il_map.md); [typed RelAlg ownership](../.claude/plans/d-relalg-unified-v1.md); [KAT law boundaries](2026-10_zucker_kleene_algebra_kat_derivatives_relalg_excerpt.md).
