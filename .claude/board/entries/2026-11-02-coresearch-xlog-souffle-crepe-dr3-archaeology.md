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

## Closeout
STATUS: measured (archaeology) | OUTCOME: exploration map; nothing adopted; 7 operator decisions (D1–D7, re-asked after premise gate 2) and 17 pre-registered probes | OPEN: every decision; no probe run
