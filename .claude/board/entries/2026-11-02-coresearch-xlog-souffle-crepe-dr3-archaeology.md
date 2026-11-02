# /coresearch exploration map — XLOG / Soufflé / Crepe / DR³ integration archaeology (2026-11-02)

**Status:** EXPLORATION MAP. Ratifies nothing, implements nothing, adopts nothing on its own authority. Brief, crosswalk and raw scout notes were banked in the session scratchpad and are not part of the record; every claim below carries its file:line in the text or in the crosswalk row it cites. Council: 5 scouts (code cartographer, internal prior art, XLOG source, systems, literature+concepts; two outside scouts swapped to source archaeology, declared), 3 co-architects (bridge, firewall, falsifier), premise gate before and after. Sources: upstream BrainyBlaze/xlog `bff6da6`, AdaWorldAPI/xlog `84da1a8`, AdaWorldAPI/crepe `d9f2a11` (branch `ccr-d3ce1263-ied9yh`), lance-graph `main` `4fadef69d`, papers (graded in the crosswalk), operator source arXiv 2602.16707 (Tamagoyaki, SECTION-READ). 19 load-bearing citations re-read by a second reader: 19/19 as stated.

## 1. The answer in one paragraph
Neither premise holds as stated. Nothing was "lost in a transcode": lance-graph never took XLOG's compiler as a production dependency, by ruling (arrow 60 vs 58; no `RirNode` mirror; d-xlog-native:10-11). The ONLY path from any XLOG/Crepe compile artifact to an executor is a set of probe gates in `xlog-relcomp-probe` (own workspace) that admit a handful of shapes and refuse the rest. But the work is not a correct transcode either: XLOG's richest compile results are discarded upstream before any consumer could read them (MagicSetReport dropped, compile.rs:1496/1511; edge polarity dropped at stratify→lower; join estimates baked into tree shape, `RirMeta` schema-only; `RirNode::Fixpoint` dormant; semi-naive delta construction runtime-only and CUDA-only), and lance-graph re-implements semi-naive recursion at least four times, carries three Set/Bag types, two local-map mechanisms, two overflow contracts, and several tuple re-entry sites — two of them P0 traps.

## 2. Answers to the eight questions (evidence in the crosswalk)
1. **Upstream analyses vs reachability.** XLOG runs: desugar → meta → list → magic sets (opt-in) → negation safety → stratify → lower (exact bushy DP ≤10 atoms) → helper split → predicate pushdown → selectivity pass (triangle/4-cycle) → multiway/WCOJ promotion. What crosses into the IR: SCCs, strata, `is_recursive`, rule bodies (with order baked in), `MultiwayPlan` (decision + cost record). What stops: magic report (dropped), edge polarity (dropped), estimates (discarded), delta construction (runtime). What lance-graph reaches: only `ExecutionPlan.sccs/rules_by_scc` and body shapes, through probe gates. Production reaches nothing.
2. **Demand/magic/binding vs crepe_plan! vs Soufflé.** Same b/f vocabulary, three different jobs: XLOG and Soufflé rewrite the program across rules (both opt-in, both skip negation/aggregation); crepe! uses binding analysis only to choose one hash index per bound set; `crepe_plan!` exposes that analysis as data but drifts from crepe! twice (D1 `let`, D2 unsafe-rule labels). Soufflé shares ordered indexes (Dilworth chain cover); crepe builds one per bound set. Genuinely different: rewrite vs index choice.
3. **Recursion invariants across transitions.** XLOG: classical stratification, set semi-naive in runtime, hard iteration cap. The relcomp gate admits exactly one linear TC SCC and re-runs its own unbounded loop. lance-graph-xlog runs resident closures (n-round bound) as host loops over `ndarray::simd`; mask-risc has no recursion. Survives: set semantics, recursion flag, strata. Does not survive: polarity, termination contract (bounded / unbounded / capped-error differ), any semiring other than B. Negation crosses nowhere (excluded by ruling).
4. **Join order / selectivity / WCOJ / Free Join / aggregate fusion / factorized recursive deltas.** Decided at XLOG lowering/promotion, then dropped by design at admission (order-insensitive gates; Law 1). Resident DR³ expresses set support and per-key counts as folds; gather-based WCOJ is banned; a mask-AND generic join is the only WCOJ form that fits; Free Join needs bag leaves + an upstream binary plan (does not fit); factorized recursive deltas have no prior art and conflict (ring inverses vs ω-continuity) — an open question.
5. **XLOG program as DT64 bindings without a competing identity.** Today yes only vacuously: the canonical `XLOG_LENS` F100 is declared and never executed (r2il `prepare` refuses non-F000), the working path uses LAB classids, and competing identities exist elsewhere (MethodSignature, SHA-256 lowering digest, FNV plan_id). What cannot cross: per-word meaning (MISSING-1), polarity, semiring, termination contract.
6. **Equality.** Four non-interchangeable contracts: Soufflé eqrel (equivalence without congruence, extended delta), egg/egglog (congruence closure + lattice merge, inflationary, termination open), DialEgg (MLIR round trip; e-graph discarded each call), Tamagoyaki (e-graph as a persistent IR construct, straight-line pure code). XLOG has none (equality = filters + join keys). Inside: egglog offline proof only, by ruling.
7. **Reusing analysis without another executor.** Only by carrying named compile results as DATA into the existing admission gates; any import of XLOG's runtime (CUDA), egglog at runtime, or crepe's generated evaluator is a TRAP. Redundant today: the semi-naive copies, three Population types, two local maps. Must stay independent: mask-risc (reference algebra), DT64 (identity authority), the compile layer (outside).
8. **Code ingestion identity.** Codegraph keeps relational meaning but surrenders source identity to its own interner (`HashMap<Box<str>,u32>`), not to ClassView/DT64; Pos = interning order; `MapTable::new(1,1)` fixes revisions. Names re-enter in cold paths (fine) and in egglog-r2il at run time (conflict); per-refresh BTreeMap refolds in `wide::refresh` are on codegraph's production path (TRAP).

## 3. Verdicts per idea
**ADOPT-NOW** (firewall PASS, small, offline):
- C6: correct the stale gate citation — D-RA-2 was decided 2026-10-20; D-SRAM-2 is blocked by D-ASM-1..4 (board edit only).

**PROBE** (each pre-registered by the falsifier designer; homes are tests, never production; full designs in `raw/arch3-falsifiers.md`):
- P-A1 magic vs negation (Chen case reachable? strata are recomputed after magic, compile.rs:715-718).
- P-A2-D1 / P-A2-D2 crepe_plan! drifts (crepe-plan-probe tests).
- P-A3 polarity lost at stratify→lower.
- P-A4a Fixpoint dormant; P-A4b the 4+ semi-naive copies agree, compared by folds.
- P-A6 admission is order-insensitive.
- P-A10 eqrel vs congruence (f(a)=b, a=c ⊢ f(c)=b).
- P-B-D1 scattered mask consumed as an intermediate.
- P-B-D2 semiring: ℕ divergence on a 2-cycle; count-lane meanings.
- P-B-D3 overflow contracts.
- P-B-D4 per-round allocations (counting allocator).
- P-B-D5 termination bounds.
- P-C3 xlog source resolution (`cargo metadata`).
- P-I1 F100 never executes; P-I4 codegraph identity.

**PARK** (worth it, not now, reason):
- A3/A1 artifacts as admission data (polarity record; magic report) — after P-A3/P-A1 show the loss matters.
- A7 mask-AND generic join — after the gather/Yannakakis rulings are revisited.
- A8 aggregate kind + Law-7 legality as admission field — after MISSING-3.
- B4 admission = executability — needs the census rule 2 work.
- I2 collapse duplicate identities / Population types — follows decision D1.
- I3 egglog-r2il `PushVar(String)` → slots at lowering.
- A9 factorized recursive deltas — research question.

**SKIP**:
- XLOG runtime / CUDA executor, egglog as runtime evaluator, crepe's generated evaluator, arrow 60 in the production graph (TRAPs).
- Gather-based WCOJ, Free Join (gather ban; bag leaves + upstream plan).
- Unifying the four equality models (different contracts).
- Negation over derived programs (excluded by ruling, A5).
- Importing XLOG's join DP as a production planner (planning moves to loco/r2il).

## 4. Decisions only the operator can make (re-asked after premise gate 2: SPLIT on D1, D2, D6; WRONG-as-framed on D7)
- **D1a Where is the semiring declared?** (i) on the compiled program / binding signature — the compile layer, where the evaluation algebra belongs; (ii) on the population (contract `Population`); (iii) nowhere: refuse every non-Boolean evaluation. Separate concerns, decided elsewhere: count-lane meaning (multiplicity / maintenance support / refcount) is per-word meaning → ClassView / MISSING-1; overflow behaviour is an executor contract (B-D3).
- **D1b How do the three Set/Bag types collapse?** (identity work, census rule 1): into the contract `Population`, or keep the probe types as listed LAB rows.
- **D2a Which termination contracts exist?** Today three: bounded-by-n (Cube), unbounded (relcomp loop), capped-error (XLOG runtime). Pick the one contract a fixpoint must state (P-B-D5 measures them first).
- **D2b Where does iteration live?** (i) a compile-layer construct — a fixpoint as an ogar-loco / r2il program, consistent with "planning moves to loco/r2il" (the property belongs elsewhere); (ii) a mask-risc fixpoint program kind (ISA change, isa-anti-lasagne gate); (iii) one host-loop driver in lance-graph-xlog (firewall CONFLICT with D-SOC-FENCE, needs a tracked exception). Keeping the copies with conformance probe P-A4b is the status quo, not a home.
- **D3 Scattered mask in `support`** (B-D1, firewall CONFLICT). (i) a mask-risc terminal whose demanded result is the intersection of two scatters; (ii) amend ir.rs:329-337 — flagged: reshapes a reference-algebra law to fit one consumer; (iii) lower via `EqU32Via`/`Gather` over a resident plane — check against the gather ban first; (iv) re-addressing: decide which population axis `support` ranges over, so the two scatters stop being consumed as intermediates.
- **D4 relcomp's xlog source** (C3, firewall CONFLICT). Repoint to AdaWorldAPI/xlog at a pinned rev (the fork is tree-equal for the crates used) — or declare an upstream-pin exception. Gated on P-C3 (not run).
- **D5 `wide::refresh` refold** (TRAP on codegraph's production path, cube.rs:822/929) — a scheduling item, not a design fork: (i) BIN plan F7 (Quack counting, a fold over sorted lanes); (ii) refresh stops refolding at all (incremental maintenance); (iii) status quo with the trap tracked on the board.
- **D6 `crepe_plan!`.** First its ROLE: (i) a candidate compile artifact for D-SRAM-2; (ii) lab-only; (iii) retire it. Only for (i): pin a rev (the floating branch conflicts with pin policy) and fix drifts D1/D2 inside `crepe_plan!` (it diverges from crepe!, crepe itself is not wrong).
- **D7 (corrected question) Should XLOG's IR stop dropping compile records?** MagicSetReport and SCC edge polarity are compile-time properties of the program; the decision is whether the fork's IR (e.g. `RirMeta` / `ExecutionPlan`) keeps them and the compile layer consumes them — with probe gates reading them only to REFUSE. Premature until P-A1 and P-A3 show the loss matters (§3 PARK). Exit: the property belongs to the compile layer, and the executor never sees it.

## 5. Not searched (so "nothing found" is not "nothing exists")
- XLOG GPU/CUDA crates beyond classification; xlog-prob, xlog-induce, xlog-neural internals; pyxlog.
- Full reads of upstream compile.rs, lower.rs, optimizer.rs (range reads only).
- Primary sources not fetchable: Bancilhon et al. 1986, Gupta–Mumick–Subrahmanian 1993, Van Gelder et al., Downey–Sethi–Tarjan, Nelson–Oppen, Olteanu–Závodný, Amsterdamer et al. (secondhand or memory, and no claim rests on memory).
- Tamagoyaki §5 and benchmark tables; Soufflé SCC scheduling; egglog negation/stratification.
- ogar-loco / ogar-r2il internals; OGAR bodies (the identity row stopped at DT64/ClassView/F###).
- Licences not verified first-hand: egglog, Soufflé (UPL-1.0 per docs), DuckDB, DialEgg, Tamagoyaki, Free Join.
- No probe was run; every PROBE is a design.

## Persistent Relational Grammar: XLOG × Egglog × DialEgg × Tamagoyaki (extension, 2026-11-02)

**Status:** exploration only. Added after the map above, at the operator's request, as an extension of the same /coresearch run. The original scouts' briefs were left as they were.
- **Supplementary scouts:** three, all bounded. S6 read Tamagoyaki (arXiv 2602.16707) and the egglog fork source. S7 searched internal prior art on equality. S8 read the DR³, OGAR, R2IL and DT64 code.
- **Co-architects:** three, working from the extended crosswalk only.
- **Citations:** 10 load-bearing citations in the extended crosswalk were re-read by a second reader; all 10 were found. The co-architects' own additions were re-read by the orchestrator.
- **Not decided:** the architecture question below is still open. /5plus3 has not been run.

### 1. Crosswalk
Verdicts: ALREADY-HAVE / PARTIAL / NEW / CONFLICTS-ANCHOR.

| mechanism | where the code is | how it connects to the workspace | verdict |
|---|---|---|---|
| XLOG lowering and optimization | `compile.rs` drops the magic-set report (`:1496/:1511`); edge polarity is lost between stratify and lower; `RirMeta` holds only the schema | Stops at the relcomp probe gate. AST→RIR keeps relations, SCCs, strata and body shapes, which is enough for set Datalog. It keeps no op terms, no effects and no SSA, so it cannot hold e-classes. | PARTIAL |
| Soufflé / Crepe demand and rule analysis | crepe! uses binding analysis only to pick indexes; `crepe_plan!` drifts from it in two places (D1, D2) | Same adornment vocabulary as XLOG, used for a different job. Its one extra (Soufflé index-set choice) has no consumer inside the workspace. | PARTIAL (duplicate) |
| XLOG recursion and factorized deltas | semi-naive and `ChainJoin` exist only in the runtime (CUDA) | Reimplemented 4+ times inside the workspace, with no shared termination contract. No prior art exists for factorized recursive deltas. | PARTIAL + DUPLICATED |
| XLOG WCOJ / Free Join | `MultiwayPlan`; egglog's own matcher is a Free Join variant (`egglog-bridge/src/rule.rs:1-5`) | Gather-based WCOJ is rejected; Free Join works over bags | CONFLICTS-ANCHOR / NEW (mask-AND generic join) |
| Egglog equality and congruence | union-find; unions cannot be undone (`stage_remove` panics, `core-relations/src/uf/mod.rs:266-268`); congruence comes from the `UnionId` merge on key collision (`egglog-bridge/src/lib.rs:810-931`); rebuild is incremental only within one run; `push`/`pop` (a whole-graph clone) is the only undo | Reached only through program text, offline. The workspace has no union, canonicalize, rebuild, e-matching or extraction of its own (closed search). | PARTIAL (offline proof) |
| DialEgg | MLIR↔egglog; the whole e-graph is discarded on every call; 4939 ms cold on one case | Nothing is wired. The note's own gate (:113) applies: first show the smallest cross-dialect case that hand-written laws cannot handle. | NEW (not wired) |
| Tamagoyaki | The e-graph is the IR (`equivalence.class/graph/yield`). Selection is separate from replacement. It persists for one compilation flow only, covers pure straight-line code, and does not describe invalidation. | MLIR is absent. A class op in R2IL/OGAR would be a new IR construct. | NEW (principle). It does not show persistence across runs. |
| DT64 ClassView / F### admission | `admit` checks class / method / sig_rev / guard (lib.rs:832-858); D5 makes behaviour immutable | No home for an e-class, an SSA value or an op occurrence. Nothing relates two MethodBindings. | PARTIAL (identity authority only) |
| OGAR / R2IL execution bodies | R2IL `run` reaches `FoldDialect` through `call_step` and never builds an OGAR `FunctionBody` (binding.rs:353-397, 426/428); `FnSpec` carries no purity, effect or width information | "OGAR owns executable method bodies" is not true in code. The R2IL seam never refuses Bag. | PARTIAL |
| DR³ versioned sparse relation storage | dt64 `dr.rs` is addressing only ("Nothing here holds a value", :18-20); relations live in xlog `Cube`/`DeltaLog` (binary, a multiset with RETIRE, merge-on-read) | In memory only. The `DeltaLog` counter is separate from `RouteVersion`. Positions never renumber. Nothing records why two classes are equal. | PARTIAL (in-process); durable is NEW |
| `contract::set_rules` (overlooked) | `Semantics` (:72-80) and the closed `Refusal` set (:203-224); Bag refusals are already pinned (:1405-1443) | The in-house semantic-refusal vocabulary. Normalisation is destructive: one term, strictly-smaller rewrites only (`next.size() < m.size()`, :379), `apply` is private (:431). | ALREADY-HAVE (guards) / PARTIAL (no saturation) |
| position identity across dialects (overlooked) | polyfill aliases bind one position per word, each with a stated domain; ogar-r2il refuses `IntAdd` as core `ADD` | The existing answer to C12: equality is declared per position, never inferred from names | ALREADY-HAVE |

**Frozen boundaries, checked against the code:**
- **Violated in labs only:** "no strings or reconstruction" and "no name dispatch on the hot path". egglog-r2il re-parses extracted text and uses `PushVar(String)` at `lower.rs:28`; shared_relation regenerates egglog text.
- **Partial:** surrendering identity at typed ingestion, because Charon output goes to the codegraph interner rather than DT64. Also partial: DR³ as owner (dt64 addresses; Cube holds the rows; nothing is durable) and mask-risc as the resident executor (it is host loops over ndarray::simd).
- **Not true:** "OGAR owns bodies".
- **Hold:** DT64 identity; "oracles hold no authority"; equivalence, entailment, provenance and truth kept separate (e-class ids never leave egglog).

### 2. Connections the earlier map missed
- **`set_rules` already gives the refusal vocabulary a persistent e-class would need,** as rows E11 and E12 show. It is not yet a gate for arbitrary unions, for two reasons:
  - it judges only catalogue rules over `Measure` under a stated `Semantics`;
  - a rewrite that keeps the term size (commutativity, for example) is silently "not a step" rather than a refusal, so "unsafe" and "not smaller" cannot be told apart.
- **Equality as a justified record.** An equality can be stored as (rule id, `Semantics`, support addresses) with no refusals, instead of as a bare union. That record answers row E10's "nothing records why".
  - The pattern exists, for detection only: `tests/shared_relation.rs:303-316` finds a stale proof's retracted support by address without re-running egglog.
  - Repair still means re-running egglog.
- **egglog's matcher is itself a Free Join.** A second Free Join is therefore already in the dependency graph, offline.
- **Tamagoyaki's separation of selection from replacement maps onto DT64 directly.** Candidate forms are separate MethodBindings; selection is which one DT64 admits.
- **A key was already proposed (2026-10-31, entries/2026-10-31-coresearch-mandatory-codegraph.md:57),** but it has no anchor: "cold passes keyed by (MethodRef, body_rev)". The same entry leaves open where body_rev is anchored (:87-96).

### 3. Compiler lifetime vs durable versioned DR³: the open decision
How long each source keeps its equivalences:
- **Tamagoyaki:** for one compilation flow.
- **egglog:** for one process; the only undo is `push`/`pop` of the whole graph.
- **DR³:** versioned, but only in memory.

**(a) The e-graph is a versioned DR³ relation.**
- **Smallest shape:** one binary child relation per position, `canon(node, rep)` read through closure, and versions from `DeltaLog`.
- **It must solve four new things:**
  1. Merging without renumbering. That is sound for append-only inserts only.
  2. Retracting an equality. No source solves this; RETIRE floors the count and never undoes a union.
  3. One "valid at v" key that joins `DeltaLog` and `RouteVersion`.
  4. Durability.
- **Firewall verdict: TRAP.**
  - It would be a second authoritative program form beside DT64/R2IL. Stored extraction choices would carry authority over behaviour.
  - Every retraction rewrites the whole population under a sparse change.
  - A stored class invites being read as entailment.
- **Bridge verdict: PARK.**

**(b) DR³ holds canonical facts only; e-classes live only for one compile, keyed by (MethodRef, body_rev).**
- **Smallest shape:** DR³ stores admitted outcomes (chosen binding + (rule, `Semantics`) + support addresses). egglog and XLOG run cold, and their state dies with the call.
- **It must solve two things:**
  1. The body_rev anchor. This is open, and without it every invalidation is unscoped.
  2. The cold cost of re-saturation. That cost is reported in µs and is not a gate; DialEgg's 4939 ms is the reference.
- **What it gets for free:** retraction is just a frame advance plus a recompute of that MethodRef. DR³ already works that way.
- **Firewall verdict: PASS.**
- **Bridge verdict: WORTH-EXPLORING.**

**Premise gate (premise-auditor, after the co-architects).** The gate split D8; the evidence below is re-read in the code.

1. **(a) is a category error, not only a TRAP.** An e-class and a DR³ relation share no field of their signatures:

   | field | e-class | DR³ relation |
   |---|---|---|
   | lifetime | one saturation | versioned frames |
   | coordinates | term space | positions that never renumber |
   | writer | egglog rebuild | routing / ingest |
   | retraction | irreversible union | frame advance |

   Also, a durable DR³ does not exist (E10).
2. **(b) as written contradicts §5.** It stores "the chosen binding" in DR³, but selection belongs to DT64. Under the one name "canonical fact" it also bundles three objects:
   - the selection, which DT64 owns;
   - the justification record, a compiler witness;
   - the domain facts, which are what DR³ actually addresses.
3. **(a) and (b) do not exhaust the choices.** At least three more exist:
   - **(c)** Store nothing about equivalence. DR³ is untouched, and DT64 admits the chosen form as a separate MethodBinding or derived classid. This matches §2's Tamagoyaki mapping and the 2026-10-31 proposal. It is the only option that passes every signature test.
   - **(d)** E-classes stay compile-local, and the justification record is kept as a discardable compiler witness outside DR³.
   - **(e)** A discardable compile memo keyed by MethodRef. This is the D16 keep-vs-resaturate question.

**The trade, restated.** Every surviving option keeps e-classes compile-local. What is open is only:
- whether any equivalence state outlives one compile;
- where a justification record would live, if one is kept.

Whether keeping saves anything over cold re-saturation (D16) is UNKNOWN. The earlier framing said (b) moves the hard problem to a body_rev anchor; the gate rejected that (see D9 below). Whether keeping saves anything over re-saturating cold (operator D16) is UNKNOWN: no source measures keep against resaturate.

### 4. Smallest viable integration probe (does not presuppose (a) or (b))
`crates/xlog-relcomp-probe/tests/congruence.rs` (test only).

**What already exists.** The crate is its own workspace and already links:
- egglog `5deaa56`;
- BrainyBlaze xlog `bff6da6`;
- lance-graph-contract;
- lance-graph-dt64.

It reuses the `Shared` / `AddressMap` / `FrameStore` / `proof_edges` pattern from `tests/shared_relation.rs`.

**Fixture.** One sealed frame with two binary registers, `eq_base = {a~c}` and `fx = {f(a)=b, f(c)=d}`.

**Two sides, run on the same frame:**
- **egglog:** native congruence. `union (F a) b`, `union (F c) d`, `union a c`, then check whether `b = d`.
- **XLOG:** symmetric-transitive `eq` plus `eq(B,D) :- fx(A,B), fx(C,D), eq(A,C)`.

**Then:** retract `a~c` and advance the frame. Both sides must drop `b=d`, and the old egglog proof must name `a~c` as its stale support, found by address.

**The single yes/no:** can the existing XLOG relational path reproduce egglog's congruence as canonical facts on the same DR³ coordinates, before one retraction and after it?
- **YES:** equality facts can be derived without importing union-find.
- **NO:** congruence is egglog-only.
- **Either way,** the answer informs both (a) and (b) without choosing between them.

**Expected outcome, recorded before running.** `admit_closure` admits exactly one linear TC SCC, but symmetric-transitive `eq` plus the congruence rule is non-linear. So NO may come from the admission gate rather than from the semantics. If it does, that is a third answer ("the relational path cannot even state it"), and the probe must report which of the three it got.

**Twin:** `f(a)=b, g(c)=d, a~c` must not yield `b=d`.

**Reported, not gated:** egglog runs, register→text conversions, intermediate rows, and cold µs.

**Other pre-registered probes.** Full designs are in the scratchpad (`raw/arch6-ext-falsifiers.md`).

*Runnable today:*
- **P-E10:** retract one supporting fact under a test-filled `canon`.
  - Home: `lance-graph-xlog/tests/canon_retract.rs`.
  - Tests whether (a) is impossible, not a choice between (a) and (b), after the premise gate.
- **P3:** a refusal matrix over the shipped guards. It cites the existing Bag pins and adds `IntAdd`/`ADD`.
  - Home: `lance-graph-contract/tests/set_rules_refusal_matrix.rs`.
- **P-E9:** a set-only Complement rewrite on a Bag binding.
  - Fixture: multiplicities [2,1,1,3], A={0,1}. The correct answer is 4; the rewrite gives 1.
  - Home: `lance-graph-r2il/tests/integration/bag_rewrite.rs`, registered in `main.rs`.
  - Runnable if an admitted binding can express N−SUM.
- **P-E8:** two equivalent MethodBindings, `x*2` and `x+x`. Nothing in DT64 may relate them.
  - Home: `lance-graph-r2il/tests/integration/equivalent_bindings.rs`.
- **P4 baseline:** an edge added inside one of two disconnected chains must leave the other chain's rows and allocations untouched.
  - Home: `lance-graph-xlog/tests/publication.rs`.
- **P5:** XLOG against Crepe on `t(X,Z):-e(X,Y),e(Y,Z)`. No crate links both, so it is two half-tests over one committed fixture, compared by folds and never by rows.

*Partly runnable today:*
- **P6** (no-second-engine counts):
  - Today: a counting `#[global_allocator]`, which has precedent at `lance-graph-dt64/tests/dr.rs:381`.
  - Not yet: symbol-lookup, e-graph-construction and re-parse counters, which need a `#[cfg(test)]` hook.
  - Arms: the admitted run arm is reported in ns per call and must count 0. The cold arm is reported in µs and is not a gate.

*Not a probe yet* (no construct exists):
- **P1:** continuity across a second lowering. Only a re-saturation/re-parse baseline exists.
- **P3** as "route an e-class merge through the guards".
- **P4** at e-class granularity.

**Expected to fire.** Four KILLs are expected to fire: P2's version half, P-E10, P-E9 and P6's `PushVar`. They would confirm gaps already listed above; they are not new findings.

### 5. Semantic refusals and ownership risks
**Refusals any design must carry** (C11). Each is a hard refusal, and the vocabulary exists in `set_rules`/`laws.rs`:
- **Effects:** `Impure`, and refuse by default when purity is unknown, because `FnSpec` cannot state it.
- **Control-flow regions:** refuse merges across a region or branch; no source handles them.
- **Overflow:** `Checked`, plus no reassociation under Checked/Ieee.
- **Signedness:** never union the signed and unsigned forms of one op.
- **Floating point:** `Ieee`.
- **SQL 3VL:** `NotTwoValued`, and aliases are equal only over a non-empty selection.
- **Multiplicity:** `Bag`. This must become a refusal at the R2IL seam itself. Relational closure is never presented as congruence.
- **Cross-dialect:** no equality inferred from names.

**Ownership risks:**
- **E-class id:** no legitimate owner. Minted in DR³ it becomes an identity outside DT64; kept in egglog it is a string id. It should exist only inside one compile.
- **A persisted equivalence:** whoever stores it owns behaviour-equivalence claims without DT64 admission, and nobody owns withdrawing them.
- **Selection/extraction:** must be owned by DT64 admission. A choice stored in DR³, or made inside R2IL `run`, gives an oracle authority.
- **Rewrites today:** they land on `MethodBinding.effects`, so they edit DT64-held bindings that nobody checks for purity. "OGAR owns bodies" is not real in code.
- **"Valid at v":** no owner joins the `DeltaLog` counter to `RouteVersion`.
- **The justification record from §2:** it is a compiler witness and must not drift into a second store or be read as evidence provenance.

### 6. Verdicts
**ADOPT-NOW:** none. Every candidate either needs an operator decision or is a probe.

**PROBE** (test-only homes, designs above):
- the congruence probe (§4);
- P-E10, which separates (a) from (b);
- P3 refusal matrix;
- P-E9 Bag-at-seam;
- P-E8 equivalent bindings;
- P4 baseline;
- P5;
- P6 counters (allocator arm).

**PARK:**
- Any persistence of equivalence state beyond one compile (D8a/D8b). Reason: (a) fails the signature tests; (d) and (e) wait for the congruence probe, P-E10 and a keep-vs-resaturate measurement.
- The XLOG recursion duplicates. Reason: orthogonal to equality; covered by base D2.
- P1. Reason: no construct exists.

**SKIP:**
- DialEgg / MLIR ingestion, until its own :113 gate passes.
- An e-class as an R2IL/OGAR op (the literal Tamagoyaki transplant), because it would be a second IR.
- Crepe/Soufflé demand analysis for this arc, as a duplicate vocabulary.
- Gather-WCOJ and Free Join.
- egglog at runtime.
- Persisting union-find state outside `compile_inner`.

**Decisions only the operator can make** (re-asked after the premise gate):
- **D8a — does any equivalence state outlive one compile?**
  - Option (a), "the e-graph is a versioned DR³ relation", fails every signature test against the code as it stands. It stays on the menu only so the operator can overrule that reading.
  - Candidates:
    - (c) nothing persists;
    - (d) a discardable compiler witness;
    - (e) a discardable memo keyed by MethodRef.
  - The congruence probe and P-E10 inform this decision; neither picks an option.
- **D8b — if a justification record (rule, `Semantics`, support addresses) is kept, where does it live?**
  - Candidates:
    - nowhere;
    - a compiler-witness store, kept distinct from evidence provenance;
    - derived inside DT64 admission.
  - Not in DR³: DR³ addresses domain relations, and DT64 owns selection.
- **D9 — which key scopes invalidation?**
  - It was asked as "where is body_rev anchored". As coded that question dissolves. D5 makes a binding immutable under its classid (`lance-graph-dt64/src/lib.rs:32-40`), and the body is `MethodBinding.effects` (:197-213). So `MethodRef` already fixes the body.
  - Real candidate key: MethodRef × rule-catalogue/`Semantics` version × support version.
  - Sub-question: who owns the join of `DeltaLog` with `RouteVersion`?
  - Precondition for the operator: does "the Program is not the body" (2026-10-31 :87-96) intend bodies that change outside D5? If not, body_rev is not a coordinate.
- **D10 — the durable "persist e-graph" SKIP of 2026-10-31.**
  - It stands without a decision; the signature mismatch in D8a confirms it independently.
  - Separately, PARK: should one e-graph survive successive lowering passes inside one compile? That is P1, and no construct for it exists.
- **D11 — two questions, not one blanket "refuse Bag at the seam".** `Population::Bag` is a legitimate binding property.
  - **D11a:** must R2IL `run` honour `binding.population`, either refusing or executing Bag correctly, instead of treating every binding as a set?
  - **D11b:** must every path that changes `MethodBinding.effects` pass `set_rules`' Bag guards (:465/581/597, pinned :1405-1443)?
  - P-E9 must report which of the two failed.

### 7. Not searched
- Tamagoyaki §5 benchmarks and its artifact.
- DialEgg source.
- egglog proofs/provenance mode.
- ogar-loco internals.
- Any MLIR toolchain.
- Licences.
- No probe was run.

### Appendix: DEFERRED /5plus3 addendum (operator). NOT RELEASED.
Held verbatim in the session scratchpad (`addendum-operator.md`, section "DEFERRED /5plus3 ADDENDUM"). It runs only after the operator has picked D8–D11, in this order: finish /coresearch → review the crosswalk → choose an architecture → /5plus3 → only then implement. Its kill conditions are recorded there; none is evaluated here.

## Closeout
STATUS: measured (archaeology) | OUTCOME: exploration map plus the Persistent Relational Grammar extension. Nothing adopted. Base decisions D1–D7 and extension decisions D8a/D8b/D9/D10/D11a/D11b re-asked after the premise gates. 17 base probes and 9 extension probes pre-registered. | OPEN: every decision; no probe run; the deferred /5plus3 addendum is not released
