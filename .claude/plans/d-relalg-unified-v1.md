# D-RELALG-UNIFIED: JC/R × Datalog × factorization × global-to-local execution

**Status:** PROPOSAL, documentation only (2026-10-20). No code in this plan's PR. Stage U1 is authorized; U2 waits on D-CMR-1. Every "exists" claim below was read at `origin/main` `a17a1a95` (file:line given).

**Three negatives are NOT YET FULLY VERIFIED.** They rest on grep plus directory listings only, not on a full read of the search space, and are marked (grep) where they appear:
1. JC has no expression IR (`crates/jc/src`, patterns `enum .*Expr`, `StatisticalRequest`, `enum Ast`, `enum Node`).
2. No code exists for the D-CMR set IR (`crates/lance-graph-contract/src`, patterns `struct Runs`, `enum SetExpr`).
3. No D-MSUB exists anywhere in the repo (`MSUB` over every `.md` and `.rs`).

Before a stage relies on one of them, read the named directory in full.

## Decisions (2026-10-20)

The same rulings, with their full qualifications (the accounting split `attest_cost` / `projection_build_cost` / `query_read_cost`, and D-MPS-8's conditional block), are recorded in `.claude/board/entries/2026-10-20-six-decisions.md` §1–§2. That entry is the authority where the two differ in wording.

DECISION (operator, 2026-10-20), verbatim: "ADMISSION = B: Counted, version-witnessed same-table seeks sind in Preparation zulässig. Cross-table positional gathers bleiben verboten. Frozen codebook lookups sind unter der definierten Attestation zulässig. D-MPS-8 ist nicht automatisch freigegeben."
- SCOPE: preparation-time reads. Arm B (#1633, #1639) is the admitted shape.
- BASIS: ledger (d) ADMISSION; `ISS-ADMISSION-PREP-BINDING`.
- REVISIT WHEN: a stage needs a positional read across tables.
- D-MPS-8, D-RPR-1, D-RST-4/6 and `quarantined-gather-hop` each still need their own argument against this rule; none is cleared by it.

DECISION (operator, 2026-10-20), verbatim: "D-RA-2 = A: Die kanonische Single-Table-Set-Algebra und ihre Rewrite-Regeln gehören in `lance-graph-contract`. `relalg` behält Lowering und relationale Planung. Keine allgemeine relationale AST für JC oder Quack einführen."
- SCOPE: the single-table set algebra (D-CMR-1) and its rewrite rules (D-CMR-6) move to the contract. relalg keeps lowering and relational planning.
- BASIS: `relalg-v1.md:61`; D-CMR §3.2.
- REVISIT WHEN: a rule needs information only relalg has.

Further rules from the same decision:
- U1 and U2 are accepted on measured row traversals, materializations and total cost, with the result set and the version semantics unchanged. Operator counts alone do not suffice.
- The 256-rule and 256-hop scenarios are separate benchmarks: sharing across rules is no evidence for amortization across hops.
- `ReadPolicy::DEFAULT` (#1639) stays a lab heuristic, and the exact API is unchanged. Open follow-ups: clustered-demand benchmarks and a direct test of the post-read filter.
- No merge on green CI alone: benchmarks, disable runs and board consistency are checked first.

## 0. Verdict in one paragraph

The diagram's centre, a shared RelAlg expression graph that both JC and Datalog lower into, does not exist. `lance-graph-relalg` is a single-table Boolean selection algebra with a terminal measure: seven `Rel` variants (`All, Empty, Plane(u16), Atom, And, Or, Not`, `ir.rs:85-100`), no joins, projection, grouping, variables, recursion, semiring or sharing (`lib.rs:28-29`: "recursion (`μ`) and the hop (preimage) are not built"). Quack has no cost model and does not consume relalg. JC has no expression IR (grep; not yet fully verified), and its R plan forbids building one that is "a general relational AST" (`jc-r-statistical-api-v1.md:86-89`). So the realistic target is **not** one new AST. Each layer gets the smallest extension that lets the layer above share work:
- a shared-plane binding (`let`) in relalg;
- a power-sums measure in the relalg `Query`;
- a co-addressed join over the D-MPS-8 `JoinDomain`;
- measured physical policy pins in Quack.

Both gates are now decided (§ Decisions):
- **ADMISSION = B** admits counted, version-witnessed same-table seeks and frozen codebook lookups under attestation. Cross-table positional gathers stay forbidden.
- **D-RA-2 = A** puts the set algebra and its rules in the contract.

The largest win needs neither gate. JC's statistics already lower to the shared sufficient-statistics terminals (`GroupPowerSumsI32` / `GroupCrossPowerSumsI32`). The Σ explosion of a correlation matrix collapses there today, if the requests are batched.

## 1. Ownership: the diagram against the repository

| Box in the diagram | Real owner today | State | What it needs |
|---|---|---|---|
| JC / R API | `crates/jc` (slice functions); `jc-r-statistical-api-v1.md` | EXISTS (stats), PLANNED (R API, D-JCR-0..8 all Queued) | D-JCR-1 `StatisticalRequest` is the lowering unit; it must stay typed, not an AST |
| Statistical expressions | jc functions over `PowerSums` / `CrossPowerSums` (`stats.rs:1282-1398`) | EXISTS | a batch form: many requests → one power-sum pass |
| Shared RelAlg AST | `lance-graph-relalg` `Rel` (selection only) | PARTIAL | `let`-bound planes (§5), power-sums measure (§5); joins via §8; home: `lance-graph-contract` (D-RA-2 = A; D-CMR set IR, `d-cmr-…:80-118`, PROPOSAL, no code (grep; not yet fully verified)) |
| Datalog frontends | crepe-plan-probe (`crepe_plan!`, flat positive non-recursive subset, `lib.rs:1-29`); datafrog and ascent as oracles; Soufflé/RAM PLANNED (`souffle-ram-64-thought-stack-v1.md:3,70`) | LAB | rule bodies lower to `Rel` per table plus `JoinDomain` joins; no second AST or VM (`souffle…:70`) |
| Algebraic optimizer | relalg `normalize.rs` (NNF, flatten, intervals, absorption, fixpoint); crepe-plan-probe `optimize.rs` (binding propagation, duplicate atom and rule elimination) | PARTIAL | cross-rule CSE (§7), hypergraph routing (`join_body.rs::is_alpha_acyclic` exists, no router) |
| Logical plan | `relalg::Query` → `lower.rs` → R2IL `FoldDialect` | EXISTS | — |
| Quack physical planner | `lance-graph-quack`: lowerings and `Filter`, **no cost model** (`lib.rs:27-42,674-688`), does not consume relalg | PARTIAL | measured policy pins first (§9), not a cost model |
| Global-to-local binding | `mork-pathmap-probe::rmap` (`Focus`, `Tile`, `Csr`, `rebind`), `join.rs::JoinDomain`; D-RMAP-1A BIND PROPOSED | PROBE | ADMISSION ruling; then a contract type |
| Direct / global execution | Quack → mask-risc | EXISTS | — |
| R2IL / mask-risc | `lance-graph-r2il` (`MAX_SLOTS = 8`, `call_owned`), `mask-risc` (`MaskOp`, ternlog fuser, power-sum terminals) | EXISTS | — |
| ndarray / SIMD | `ndarray::simd` via mask-risc `exec.rs:25` | EXISTS | numerical kernels (eigen, solvers) stay ndarray, outside the algebra |

**New contracts genuinely required:** the shared-plane binding, the power-sums measure, the co-addressed join operator, the BIND type (after ADMISSION), and a recursion driver (after ADMISSION). Everything else extends what exists.

## 2. Gap analysis

**Exists:**
- the Boolean selection algebra and its rewrites;
- the R2IL lowering with constant-slot width (`call_owned`, #1634);
- mask-risc with ternlog fusion and the 256-function dispatch;
- the power-sum and cross-power-sum terminals;
- JC statistics over those sums;
- binding propagation and duplicate elimination for flat Datalog;
- the α-acyclicity test;
- `seek_from` and `gallop_partition_point` on attested ordered lanes;
- Arm B addressed reads with a measured read policy (#1639);
- rmap `JoinDomain`, `Focus`, `Tile` and `TileExec::reach` (PROBE);
- the dir-sim compiled closure (#1635, MEASURED).

**Planned, unbuilt:**
- the JC R API (D-JCR);
- the contract set IR (D-CMR);
- relalg as algebra values (D-RA-3);
- Soufflé RAM (D-SRAM);
- BIND and its generations (D-RMAP-1A / -1A-V);
- recursive Datalog → relalg μ (D-MTP-3, not authorized).

**Blocked or ruled out:**
- recursion driver (`ISS-RECURSION-DRIVER`: "no lawful (co-addressed, non-gather) recursion driver exists");
- leapfrog D-ASM-11 (⛔ GATHER);
- D-SRAM-4 (its target is a banned gather);
- D-RA-4 and D-CMR-5 (preimage, gather);
- the quarantined `graph_hop.rs`.

**Absent, no plan:**
- a cost model or cardinality estimation (the DataFusion-era `lance-graph-planner` has stubs only, `histogram_cost.rs:29-38`);
- any shared IR spanning statistics and Datalog;
- prefetch or asynchronous staging.

## 3. JC/R integration

Statistical semantics stay in JC; relalg owns which rows and which grouping; mask-risc owns the pass. The lowering unit is the sufficient statistic, not the formula.

| JC request | Shared terminal | Then, in JC (scalar) |
|---|---|---|
| mean, variance (per group) | `GroupPowerSumsI32` (n, Σx, Σx²) | divide and subtract |
| covariance, Pearson, simple regression, R² | `GroupCrossPowerSumsI32` (n, Σx, Σy, Σx², Σy², Σxy) | `pearson_from_cross_power_sums` (`stats.rs:1342`) and siblings |
| one-way ANOVA, η² | `GroupPowerSumsI32` | `anova_from_power_sums` (`:1282`) |
| Spearman | ranks first (a lane transform), then the cross-power terminal | as Pearson on ranks |
| ICC, Cronbach α | per-item power sums plus cross sums over item pairs | existing formulas |
| covariance or correlation matrix of p variables | one cross-power pass per pair today; p(p+1)/2 pairs | see §6 |
| PCA, eigen, multiple regression | the cross-product matrix from the passes above | ndarray kernel; never a relational join |

**Integration rule.** A batch of `StatisticalRequest`s (D-JCR-1) is reduced to the set of distinct (selection, grouping key, lane pair) triples. Each triple is computed once. Every request reads its statistic from that set. This is CSE at the sufficient-statistic level, and it needs no new IR: the selection is a `relalg::Rel`, the key and lanes are already terminal parameters.

**Numerics.** mask-risc sums i32 lanes into exact integer accumulators. Sharing and reordering the passes is therefore exact, and rounding happens once, in JC's final f64 arithmetic. For f64 lanes, which mask-risc does not support (`jc-r…:46-51`), reordering is not exact: each transformation must state its error bound (pairwise or compensated summation) or refuse.

## 4. Datalog integration

A positive non-recursive rule body is a conjunction of atoms over tables. Today the probe lowers:
- the per-table selections to `Rel` (`native.rs`, through relalg);
- the join to a star over the D-MPS-8 shared `y` domain (`And` of presence planes, `join.rs:1-34`).

Arm B adds the addressed read. Crepe, ascent, datafrog and a Soufflé front end all lower the same way. A Soufflé front end contributes its schedule ideas only: index selection, magic sets, adornment. It must not bring its own VM (`souffle…:70`).

Recursion is the hard stop. The only co-addressed propagation step in the tree is `TileExec::reach` (`rmap.rs:858-870`). It runs over 256-slot tiles whose slots come from a BIND, so it is gated on ADMISSION. `walk_mask` (#1634, #1637) shows that bit planes win, but its inner loop reads delta rows at positions taken from adjacency values. That is a gather and does not qualify.

## 5. Smallest logical-IR extensions

1. **`let`-bound planes**, in the contract set IR (D-RA-2 = A); relalg lowers them. A program holds `lets` and `queries`, and `Plane(i)` is an immutable reference to `lets[i]`, valid only at the coordinate version the program was built for. Each `let` is computed once into a scratch plane, and every query reads it.
   - **Why not hash-consing inside `Rel`:** `call_owned` requires that no consumed `Val` is shared (`r2il lib.rs:570-587`). A DAG inside one fold would force plain `call` and give back the width limit #1634 removed.
   - Materializing a shared subexpression as a plane keeps every fold a tree. It costs one plane of `n/8` bytes per `let`.
2. **Power-sums measure.** Add `Measure::PowerSums { key, x }` and `Measure::CrossPowerSums { key, x, y }`, lowering to the existing terminals through `fold_cross_power_sums` (`r2il lib.rs:411`). This is the bridge from JC into relalg.
3. **Co-addressed join.** A conjunction of per-table `Rel`s over one `JoinDomain`. This is what the probe already does, named as an operator, so multi-table bodies stop being probe code.
4. **Not proposed:** variables as a general binder, semiring annotations on `Rel`, and a recursion node. Each needs ADMISSION or D-RA-2 first. Semirings would also need the bag/set split that D-CMR §3.2 leaves out ("sets, not bags").

## 6. Worked factorization: Σ_i Σ_j Σ_k A_ij B_jk

As relations `A(i, j, a)` and `B(j, k, b)`:

- **Naive:** enumerate (i, j, k), so |A ⋈_j B| products.
- **Factorized:** S = Σ_j (Σ_i a_ij)(Σ_k b_jk).
  1. `GroupSum(A, key = j, x = a)`: one pass over A.
  2. `GroupSum(B, key = j, x = b)`: one pass over B.
  3. One dot product over the j domain.

The two group sums land at the same j coordinate. That holds if j is the `JoinDomain` slot, which preparation assigns. Step 3 then reads co-addressed vectors and is not a gather.

**Cost:** from |A ⋈ B| to |A| + |B| + |dom j|.

**Exactness:** exact on integer lanes (distributivity in ℤ). On floats, the factorized form changes the rounding order. That is bounded but not identical, and it must be declared.

**Covariance matrix of p variables over n rows.** One `GroupCrossPowerSums` per pair costs p(p+1)/2 passes. The factorized form computes Σx_a once per variable (shared across pairs) and Σx_a x_b per pair: XᵀX, n·p² multiply-adds. That belongs in ndarray (GEMM), not in a relational join. The relational part ends at the selection that produces X's rows.

**Third moments M_ijk.**

| Request | Cost | What it is |
|---|---|---|
| the full tensor | p³ outputs | a real lower bound |
| one element | one pass over three lanes | — |
| contraction with weights w (Σ_jk M_ijk w_j w_k) | 2 passes per i | the projection u = Xw is computed first |
| a skewness test | needs only power sums | — |

The optimizer's job is to tell which of these the consumer asked for. It cannot make the full tensor cheaper.

## 7. Worked 256-rule example

| Rule | Body |
|---|---|
| 1 | `A(x,z) <- R(x,y), S(y,z)` |
| 2 | `B(x,z) <- R(x,y), S(y,z), P(z)` |
| 3 | `C(x,z) <- R(x,y), S(y,z), Q(x)` |
| … | … |

**Phase 1 (logical CSE).** Canonicalize each body by sorting atoms and renaming variables. This is what `optimize.rs` duplicate-atom and duplicate-rule elimination already does within one rule. Then find maximal shared sub-bodies across rules.

**Phase 2 (materialize).** `R ⋈_y S` becomes one `let`, a pair of presence planes over the y domain plus the joint mask. Rules 2 and 3 become `let₀ ∧ P(z)` and `let₀ ∧ Q(x)`, each one plane `And`.

**Count.** 256 rules sharing a two-atom core cost one core join plus 256 single-plane ANDs. Without sharing it would be 256 core joins.

**Kinds of reuse, kept apart:**

| Kind | Where it happens | Cost |
|---|---|---|
| Logical CSE | at compile time | free |
| Materialized reuse | the `let` plane | n/8 bytes |
| Semi-naive deltas | recursive rules only (§8) | — |
| Dense bitmap execution | per-op physical choice | — |
| Register reuse | the BIND geometry shared across rules | gated |

## 8. Recursion over up to 256 hops

**Shared-delta loop.** The relation state is one plane per derived relation over a bound domain. Each iteration:
1. Compute Δ.
2. Evaluate only the rule fragments whose body mentions a relation with nonzero Δ.
3. `new &= !old; old |= new`.
4. Stop when every Δ is empty.

This is the `walk_mask` loop. To be lawful it needs the step "rule fragment applied to Δ" to be co-addressed. With binary relations bound into 256-slot tiles, a hop is a Boolean tile product (D-RMAP-2 `tile_gemm.rs`, PARTIAL), and positions come from the tile layout, not from values.

**The gate.** That is exactly the BIND in preparation that ADMISSION decides. Without the ruling, no recursion stage can be built, only measured as a lab arm with the gather labelled.

**Benchmark requirement.** At least one input must need deep propagation. A chain or a ladder of length ≥ 200 forces the loop to run that many iterations (asserted, not assumed), plus a cyclic graph to check termination.

## 9. Physical planning in Quack

Quack has no cost model, and its own docs say a selectivity-only model cannot separate the measured regimes (`lib.rs:674-688`). The first step is therefore what #1639 just did for Arm B: per-decision **policy pins measured on fixtures**, each with a can-fire and a can-stay-silent test. Not a general cost model. Candidate decisions:

| Decision | Signal | Measured today |
|---|---|---|
| exact ranges / coalesced / full read | coverage of the table | #1639: full at ≥ 10% |
| dense plane / sorted intersection | demand density on the domain | not measured |
| materialize a `let` | number of readers × its pass cost against n/8 bytes | not measured |
| statistics: per-pair pass / ndarray GEMM | p and n | not measured |
| bind into a focus / stay global | reuse count, L1 fit (#1635: F wins only for reused foci with question sets ≤ L1) | #1635 |

A cost model becomes justified only after several pins exist and contradict a single selectivity rule.

## 10. Register integration (gated)

The register is BIND, `(thought, palette, focus, scope, version) → bound relations` (D-RMAP-1A `:82`). It is not a graph store. It owns slot assignment and a mapping witness.

**Reuse across rules and hops.** One BIND per (relation set, version) is shared by every rule and every hop. Each rule reads the same tile coordinates, so the geometry is paid once.

**Status.** #1635 measured that the compiled closure (B+), not the 256-slot focus (F), carries most of the gain. F wins only with reuse, a small question set and many groups.

Geometry stays experimental:
- 256 slots is one tile, 8 KiB for 256 × 256 bits.
- 256³ is a paging scheme. It is not a resident address space.
- Rules, hops, nodes and pages stay separate dimensions.

**Under ADMISSION = B.** A same-table, version-witnessed seek is admitted (Arm B, #1639). A frozen codebook lookup is admitted under attestation. A cross-table positional gather is not. A tile hop must therefore show, before U6, that its slot assignment is a frozen, attested codebook lookup and that the hop reads no position taken from another table's values. D-MPS-8 is not cleared by the decision and keeps its own review.

## 11. Lazy context: Wikidata, SAP

**Prefetch unit.** A predicted relationship page is a set of key ranges on an attested lane, which is exactly Arm B's ranges. Latency hiding means issuing the next stage's range reads (asynchronous Lance scans) while the current stage folds, so the read overlaps compute. A local pointer table does none of that. The measurable effect is the overlap: critical-path time against the sum of read time and fold time.

**Completeness.** Absent and unknown must stay distinct. Negation (`!R(x)`) is admitted only over a scope whose witness attests completeness for that relation and version. In an incomplete hydrated context, `!R` is refused, not answered false. This matters for Wikidata qualifiers and partial SAP authorization graphs alike.

**Fixtures.**
- Wikidata: a public dump subset, with QIDs and PIDs preserved.
- SAP: a synthetic org, position, role and authorization fixture, with no customer data. dir-sim's IAM fixture (#1635) is a ready template.

## 12. Smallest benchmark that separates the three mechanisms

One workload, run under three toggles. Each toggle changes one mechanism only.

| Mechanism | Toggle | What must change | What must not |
|---|---|---|---|
| Algebraic reduction | CSE / factorization on vs off | logical op count, passes, intermediate cardinality | the answer (bit-exact on integer lanes) |
| Physical improvement | same plan; plane vs sorted vs ndarray choice | time | op count |
| Latency hiding | same plan and representation; prefetch on vs off, cold store | critical-path time | bytes read |

**Workload.**
- One base table with 7 integer variables, n = 50k and 500k.
- 256 non-recursive rules drawn with a controlled shared-core rate (0%, 50%, 90%), plus 16 recursive rules over a chain of length 256 and one cycle (lab arm, gather labelled until ADMISSION).
- A JC batch: a 7×7 correlation matrix, 3 group-wise ANOVAs and one third-moment contraction.

**Arms.**
- A / B: crepe and ascent as baselines and oracles.
- C: `walk_mask`, the bit-plane executor.
- D: `let` CSE.
- E: D plus factorization.
- F: E plus a shared BIND (lab).
- G: F plus prefetch (lab).
- H: the JC batch lowered through the same `let`s and power-sum terminals. H must show a shared pass that the separate JC and Datalog paths each recomputed.

**Oracles.** R reference values for statistics; ndarray GEMM for XᵀX; crepe and ascent for the rule results; a chain fixture that asserts the iteration count.

## 13. Risks and falsifiers

| Risk | Falsifier |
|---|---|
| CSE of a `let` plane costs more than it saves (one reader, low selectivity) | D slower than A at 0% shared core |
| `call_owned` forbids sharing inside a fold; a DAG would restore the slot limit | depth/width test: any `let` design must keep `a_wide_disjunction…` (≤ 3 slots) green |
| Factorization changes float results | f64 arm with and without reordering, error bounded and reported, otherwise refused |
| A focus loses when the question set exceeds L1 (#1635: 1.5–2.1× slower) | F against E across question-set sizes |
| Prefetch predicts wrong pages | bytes read with prefetch must stay within ε of without; wasted reads counted |
| A recursion benchmark that converges in 2 iterations | assert the iteration count ≥ 200 on the chain |
| Negation over an incomplete hydrated context | test: `!R` under an incomplete scope is refused |
| Arm B's read crossover moves with size (#1639) | re-measure the pins at every new fixture size |

## 14. PR sequence (lab only, dependency-ordered)

| Stage | Content | Gate to start | Acceptance |
|---|---|---|---|
| U0 | decisions: D-RA-2 = A, ADMISSION = B | — | DONE 2026-10-20 (§ Decisions) |
| U1 | JC statistics batch over shared power-sum terminals (D-JCR-1 shape, probe crate) | none | H-lite: k requests → distinct-triple passes, bit-exact against per-request; R references |
| D-CMR-1 | dependency-free single-table set IR in `lance-graph-contract`, with algebraic law tests | — | laws hold on generated cases; disable runs red |
| U2 | `let`-bound planes on the contract IR: immutable `Plane(i)` references, versioned coordinates, each shared subexpression evaluated once; no DAG inside a fold | D-CMR-1 | rows traversed and planes materialized measured against recompute; width test stays ≤ 3 slots; answers and versions unchanged |
| D-CMR-6 | canonical rewrite rules into the contract | D-CMR-1 | existing relalg tests pass unchanged |
| U3 | cross-rule CSE in crepe-plan-probe on top of U2 | U2 | 256-rule fixture: op count before and after, answers equal to crepe and ascent |
| U4 | co-addressed join operator over `JoinDomain` (named, multi-table) | U2 | parity with the probe's star path; no gather (no positions taken from values) |
| U5 | Quack physical pins (dense vs sorted, materialize vs recompute) | U3, U4 | can-fire and can-stay-silent per pin; measured tables |
| U6 | recursion driver over BIND tiles | its slot assignment argued as an attested frozen-codebook lookup under ADMISSION = B | least fixpoint equals datafrog; ≥ 200 iterations asserted; cycles terminate; refusal past the bound; separate from the 256-rule benchmark |
| U7 | prefetch and staging of Arm B ranges | U6 or a non-recursive multi-stage plan | critical-path overlap measured; bytes read unchanged |
| U8 | Wikidata and SAP-shaped fixtures; completeness refusals | U6, U7 | refusal tests; witness carried end to end |

U1 needs nothing new and comes first. **U1 MEASURED (2026-10-20, `crates/jc-batch-probe`, `entries/2026-10-20-u1-jc-batch.md`):** 252 requests reduce to 2 selection and 42 cross passes; at 1M rows the batch is 6.4× faster with 11.5× fewer rows traversed, 8.5× fewer values read and 126× fewer mask bytes, and its answers equal the per-request answers bit for bit. The R reference is replaced by JC's slice functions because R is not installed. U2 builds on D-CMR-1. Both are accepted only on measured row traversals, materializations and total cost (§ Decisions), not on operator counts.
