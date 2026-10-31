# xlog-relcomp-probe

## 1. What this is

A test-only probe. It asks one question: can a restricted XLOG rule body be
lowered onto the existing 64x64 Boolean relation register in lance-graph, and
answered by mask propagation?

- It has no production API. Nothing in lance-graph depends on it.
- It is standalone. It has its own `[workspace]` and is not a lance-graph
  member.
- XLOG (BrainyBlaze/xlog) is pinned at
  `bff6da663308190148c5ce58a715d86525b62e67` (MIT OR Apache-2.0).

Run the tests:

```
cargo test --release --manifest-path crates/xlog-relcomp-probe/Cargo.toml
```

Run the cost probe:

```
cargo run --release --example cost --manifest-path crates/xlog-relcomp-probe/Cargo.toml
```

Public items in `src/lib.rs`: `admit`, `admit_program`, `facts`,
`AddressMap`, `register`, `propagate`, `ChainQuery`, `LowerRefusal`,
`DomainRefusal`, `EvalRefusal`, `Propagation`.

## 2. Gate A: which XLOG crates can be used

| Crate | Status |
|---|---|
| xlog-core, xlog-ir, xlog-stats, xlog-logic | Directly usable on CPU. Checked: they compile without CUDA (29 s debug check). |
| xlog-neural | CPU-buildable (pyo3 is optional). Not used. |
| xlog-runtime, xlog-prob, xlog-solve, xlog-cuda, xlog-gpu | CUDA-coupled (`cudarc` dependency). |

Consequence of the CUDA coupling: XLOG's executor, its PIR/exact
probabilistic path and its d-DNNF are not executable here.

xlog-core pins arrow 60. The lance stack pins arrow 58. Linking xlog-core
into the lance-graph graph would need two arrow majors. This is why the probe
is its own workspace.

Already implemented in lance-graph (not borrowed from XLOG):

- The 64x64 Boolean relation register, bind guards, chain typing and the
  compose kernel (`dr-math-binding-probe`, `relcomp`).
- GYO alpha-acyclicity (`lance-graph-relalg`, `join_body`).

Borrowed from XLOG: parsing, lowering to RIR, and the magic-set decision.
Not borrowed: the XLOG runtime. No XLOG AST is copied into lance-graph.

## 3. Gate B: one rule, two masks

Fixture:

```
R = {(1,2), (1,3), (9,7)}     S = {(2,4), (3,5), (7,4)}
q(Y) :- r(1, Y), s(Y, 4).
```

XLOG compiles it to
`Project([Column(1)], Join(Inner, keys [1]/[0], Filter(Scan r, col0==1), Filter(Scan s, col1==4)))`.

Results:

- Forward mask: {2, 3}. Backward mask: {2, 7}. Answer: {2}.
- The answer equals scalar enumeration and a tuple walk of XLOG's own RIR.
- Backward uses the existing `relcomp::compose` against a one-column selector
  (`C[z] = 1`). There is no transposed edge store and no new opcode.
- A 300-case deterministic differential against scalar enumeration passes
  (`differential_against_scalar_enumeration`).

Magic sets. XLOG's rewrite targets bound recursive queries only.
`#pragma magic_sets = auto` declines with "no bound recursive query eligible
for magic_sets". `= on` returns an error
(`magic_sets_decline_a_nonrecursive_query`).

Two plan variants appeared by running XLOG:

- (a) Unbound ends. The optimizer emits `ChainJoin` with the plain `Join` as
  `fallback`. `admit` accepts it only when both describe the same query.
  Otherwise it returns `ChainJoinDisagrees`.
- (b) Large unselective S. The optimizer swaps join sides (keys `[0]/[1]`,
  answer column 0). `admit` accepts both orders
  (`join_order_chosen_by_xlog_statistics_does_not_change_the_answer`).

Lesson: admitting the output of an optimizing frontend must be order- and
promotion-insensitive.

## 4. Refusals

Each refusal has a test. Nothing is approximated.

| Refusal | Case | Test |
|---|---|---|
| `NeedsInverse` | `s(4, Y)` | `refuses_reversed_orientation_without_an_inverse` |
| `NonPositiveBody` | negation (`not s(Y, 4)`) | `refuses_negation_unknown_is_not_false` |
| `NonPositiveBody` | comparison (`Z != 5`) | `refuses_comparisons_and_non_integer_facts` |
| `Probabilistic` | `0.5::r(1, 2)`; the RIR is deterministic, the program is not | `refuses_probabilistic_programs_even_when_the_rir_is_deterministic` |
| `FactTerm` | `r(9, -7)` | `refuses_comparisons_and_non_integer_facts` |
| `DerivedRelation` | a scanned predicate also defined by a rule (`r(X, Y) :- t(X, Y)`); its facts are not its whole relation | `refuses_a_scanned_relation_that_a_rule_derives` |
| `Chain(DomainMismatch)` | cross-domain alias; raised by the existing chain typing | `refuses_cross_domain_alias` |
| `AxisMismatch` | answer read in an axis the left operand does not produce | `refuses_cross_domain_alias` |
| `Bind(SourceChanged)` | stale frame | `refuses_a_stale_frame` |
| `DomainRefusal::Overflow` | 65 ids on one axis | `domain_boundary_63_64_65_and_tail_bits` |
| `ChainJoinDisagrees` | `ChainJoin` and its `fallback` differ | `refuses_a_chain_join_that_disagrees_with_its_fallback` |

## 5. Disable runs

Each guard was removed, the named test went red, and the guard was restored.
Anchors were asserted so a patch that did not apply could not pass as a
result.

| Guard removed | Result |
|---|---|
| AND of forward and backward | 3 tests red |
| Tail mask (`valid`) | boundary test red |
| `ChainJoin` fallback check | its test red |
| Probabilistic check | its test red |
| Inverse check | its test red |
| Negation check | 2 tests red |
| Overflow check | boundary test red |
| Commuted-join arm | join-order test red |
| Derived-relation check | its test red |

## 6. Gate C: the 64-slot boundary

Test: `domain_boundary_63_64_65_and_tail_bits`.

- 63 ids: slot 63 is not assigned. 64 ids: all bits valid. 65 ids: refused.
- Tail bits never leak. A stray bit past the domain in a register does not
  reach a mask.
- Non-contiguous ids (7, 2^31, `u32::MAX`) map to dense slots and back.
- Equal ids on two axes are two unrelated slots. `AddressMap` is per `Axis`.

## 7. Gate G: shared evidence

Tests are in `tests/gate_g.rs`. Two proofs share one fact:
`P1 = F and G`, `P2 = F and H`, with pF = 0.8, pG = 0.6, pH = 0.5.

- Exact, by enumerating the 2^3 worlds: P = 0.64.
- Derivatives with respect to (pF, pG, pH): (0.8, 0.4, 0.32).
- Noisy-OR over proofs gives 0.688. It overcounts because F is shared.
- Max over proofs undercounts.
- The Boolean support mask answers "is there a proof". It is not a
  probability. The register path therefore refuses probabilistic programs.

Scallop as an independent oracle (AdaWorldAPI/scallop `668bfb6`, `scli`,
built oracle-only with `RUSTC_BOOTSTRAP=1` on stable because scallop-core
uses `min_specialization`; it is not a dependency of anything). Program:

```
rel obs = {0.8::("F"), 0.6::("G"), 0.5::("H")}
rel p1() = obs("F") and obs("G")
rel p2() = obs("F") and obs("H")
rel q() = p1()
rel q() = p2()
query q
```

| `scli -p` | P(q) | Reading |
|---|---|---|
| `topkproofs -k 10` | 0.6400000000000001 | exact: proofs share the fact id of F |
| `addmultprob` | 0.88 | sums proof weights; overcounts |
| `minmaxprob` | 0.6 | min within a proof, max across; undercounts |
| `bool` | true | support only |

`scli` has no differentiable provenance, so the derivatives (0.8, 0.4, 0.32)
are checked by world enumeration only, not by Scallop. XLOG's own exact
probabilistic path (`xlog-prob`) is CUDA-coupled and was not run.

## 8. Gate I: cost

Host: `Intel(R) Xeon(R) Processor @ 2.80GHz`, avx512f, avx2. Release build.
Two runs, load about 0.5 to 0.9. One run, verbatim:

```
fixture         cold_us  c_alloc   c_bytes |  hot_ns  h_alc  words  rORs |  scal_ns  prep_ns  full_ns | ans
selective         372.1      896    113782 |   266.6   1.00     65     3 |     11.4     0.89     32.4 | 1
dense           15686.2    58298  14453982 |  2695.6   1.00     65  4096 |  87939.4     0.90   2319.9 | 64
skewed            677.1     1790    328614 |   317.2   1.00     65    64 |   2428.9     0.89     59.5 | 64
disconnected      343.0     1334    216634 |   292.1   1.00     65    32 |    399.2     0.91     43.4 | 0
adversarial      7831.6    29617   8984854 |  2504.7   1.00     65  4033 |  82872.6     0.91     82.6 | 1
```

Columns:

- `cold`: XLOG parse and compile, admission, fact extraction, address maps,
  register fill. Cold costs are reported in µs. They are not gates.
- `hot`: bind both registers, chain typing, propagate. In ns.
- `h_alc`: heap allocations per hot call. The 1 is the `Vec` that
  `Chain::new` takes.
- `words`: 64-bit words read. `rORs`: row-ORs done by `compose`.
- `scal`: scalar tuple enumeration.
- `prep`: prepared inverse column (two word reads and one AND).
- `full`: full R-compose-S. It answers "x reaches z", not which Y. Scale only.

Findings:

1. On the selective fixture, scalar enumeration (11–15 ns over two runs)
   beats the mask path (267–406 ns). The hot path is dominated by binding, typing and one allocation,
   not by bit work.
2. Backward by compose-with-selector costs one row OR per edge of S (4096 on
   dense, about 2.5 µs). A prepared inverse column costs about 0.9 ns, but it
   is a per-z prepared structure. That is an index.
3. Masks beat scalar by about 30x on dense (2.7 µs against 88 µs).
4. Cold cost is dominated by XLOG parsing fact literals: 15,686–21,422 µs
   on 8192 facts (two runs). Facts should come from the substrate, not from XLOG source text.
5. Bidirectional filtering did not beat a prepared index. Do not assume it
   does.

## 9. Not done

- Gate D (shader handoff). `ShaderDispatch` (lance-graph-contract
  `cognitive_shader.rs`) selects rows only by a contiguous `ColumnWindow`, a
  meta prefilter and an 8-bit predicate-plane `layer_mask`. No field accepts
  an arbitrary row-selection mask, so a relational answer cannot enter the
  shader read-only today. Reported as a contract gap. `layer_mask` was not
  repurposed.
- Gate E (64^3). Logical only. Not measured.
- Gate F (ONNX annotations). Not touched.
- Recursion and semi-naive evaluation.
- Cyclic bodies. Local consistency is not an answer.
- Bag semantics.
- K3/UNKNOWN as a third value.

## 10. Verdict

XLOG should be a **source-language frontend lowered into the native algebra,
consumed read-only as a pinned compiler and oracle dependency** for the
CPU-portable crates. It should not be forked: nothing needed changing
upstream. It should not be used as a runtime.

The smallest interface that changes the decision:

- A frozen, versioned RIR subset: Scan, Filter(col == const), Join(Inner),
  ChainJoin-with-fallback, Project(col).
- Facts supplied by the substrate, not parsed from text.

Before any wider adoption, one of these is needed: an arrow-major alignment,
or a facts-free `xlog-core` feature. Arrow 60 against 58 is the one hard
blocker to linking XLOG in-tree.

## 11. D-LITHO-1: Gate B as explicit steps, and bag multiplicity

`tests/litho.rs`. The D-LITHO plan is a mental model for this work; its
vocabulary is not used as types, names or steps here.

**Gate B as explicit steps.** Each is an existing call, and the test asserts
the steps equal `propagate`.

| Step | Gate B | Call |
|---|---|---|
| registers | R and S, filled once, then borrowed | `register`, `bind_dr64` |
| typing | one frame, focus and scope; R's columns are S's rows; a stale frame is refused | `Chain::new`, `bind_dr64` |
| bound constants | `col0 == 1` on R as row slot x; `col1 == 4` on S as selector `C[z] = 1` | `AddressMap::slot`, a 64 x 1 relation |
| forward | `R[x]` = {2, 3} | one row read |
| backward | preimage of {z} under S = {2, 7} | `compose`, then column 0 (3 row ORs) |
| answer | forward And backward = {2}, read as ids | `&`, `AddressMap::ids` |

Boolean throughout. The registers and address maps are prepared once per
frame revision; the slot and selector are query constants, prepared per query.

**Bag multiplicity (the failing fixture).** `q(Y) :- r(X, Y), s(Y, Z)` with
R = {(1,2), (5,2), (1,3), (9,7)}, S = {(2,4), (2,6), (3,4), (7,8)}. Y = 2 has
2 R witnesses × 2 S witnesses = 4 derivations; Y = 3 and Y = 7 have 1.
Expected {2: 4, 3: 1, 7: 1}, checked by:

1. scalar enumeration of (x, y, z) witnesses;
2. a bag walk of XLOG's own RIR, both for `q` and for
   `qc(Y, count(X)) :- r(X, Y), s(Y, Z)` (`GroupBy` over a non-deduplicating
   `Project`, which is XLOG's documented `count` semantics: it counts distinct
   full body bindings). A set walk gives {2: 2, ...}, the distinct-X count, so
   Gate B's set-valued oracle cannot be reused here.

Demand shape: 3 X, 3 Y, 3 Z ids; 4 of 9 cells set in each relation; one
64-slot Y word; largest count 4, i.e. three bits per Y.

Today: the register path answers support only ({2, 3, 7}); `admit` refuses
the count plan (`UnsupportedNode("GroupBy")`); the binder refuses a Bag
operand (`Refusal::BagNeedsMultiplicity`). The ignored
`the_register_path_carries_bag_counts` demands the counts and fails with
`{2: 1, ...}` vs `{2: 4, ...}`.

Disable runs: switching oracle 2 to set semantics turns
`both_oracles_agree_on_the_per_y_counts` red; dropping the backward step
turns `gate_b_as_explicit_steps_over_existing_calls` red.

**What carrying counts would need.** Gate B needs nothing new. Counts need
two things nothing defines yet:
- a per-Y count width, and what happens when a count exceeds it (the bound
  here is |X|·|Z| per Y, at most 4096, i.e. 13 bits);
- a per-Y product: for a two-atom chain the count is deg_R(y) · deg_S(y)
  (XLOG's own factorized count multiplies trie-range lengths the same way).

**The same operation through the production bind store**
(`tests/dto_binding.rs`, `lance-graph-dt64`). DR routes first: the answer
positions (the Y ids 2, 3, 7) are routed to the XLOG lens in the bind store's
route log, under a sealed `dt64::dr::FrameStore`. Then DT64 binds meaning:

- identities: `NodeGuid` references (V3 tail, lab lens classid `0x7F01_0001`,
  a stand-in, not a minted concept);
- property mappings: R and S as fields 0 and 1 of the lens, labelled by its
  `ClassView` through `Resolved::field_view`;
- method: a `MethodBinding` of `Population::Bag`, admitted by `MethodRef`.
  Its three steps name polyfill words by position: mask-risc's keyed count
  (`Terminal::GroupReduce`) for R and for S, then
  `ndarray::simd::checked_binary_i64(Mul)`. The polyfill is one lab lens
  (`0x7F0A_0001`); no word has a classid;
- expected values: {2: 4, 3: 1, 7: 1}, keyed by identity.

The binding refuses a stale frame (`MapChanged`) and a retracted position
(`Unrouted`; the old route version still resolves). Execution has no
production home: the admitted population is carried into the lab register,
whose `bind_dr64` refuses it (`BagNeedsMultiplicity`). That is the failing
half. `tests/litho.rs` keeps the lab frame store, because `relcomp::Chain`
accepts only that one.

**Epi5** (`tests/epi5_reading.rs`). An XLOG result written as a
`CausalEdge64` carries `Direct x Open` for a base fact and
`IndirectKnown x Open` for a composed edge whose Y is retained. A join
certifies nothing, and a bag count is never a certification or a truth value.
`raw5 == 0` reads as `Direct x Open`, so an unstamped composed edge would
claim a direct relation. Without a declaration for the XLOG lens, projection
refuses (`UndeclaredClass`).
