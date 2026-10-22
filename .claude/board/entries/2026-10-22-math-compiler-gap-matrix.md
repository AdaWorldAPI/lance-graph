# 2026-10-22 — Proof-carrying math compiler: Stage A gap matrix (egglog → R2IL)

**Status:** OPEN. Inventory only; no code. Read at `origin/main` `3b723df1`, egglog fork `main` `9c8396d`. Falsifiers E1–E5 are the ones registered in `.arxiv/2304.04332_egglog_equality_saturation_datalog_excerpt.md:23-28` (P6/P12 in `.arxiv/2026-10_qr256_gaussian_egglog_rotation_falsifier_paths.md`). D-RFT-2 (`d-rft-relalg-thought-git-frontier-v1.md`) is the owning plan; no new plan.

| Piece | State | Where | Note |
|---|---|---|---|
| egglog equality saturation + proofs | EXISTS (fork) | AdaWorldAPI/egglog `prove`, `prove-variants` (#1, #2), `ProofStore`/`ProofId` | no lance-graph crate depends on egglog today (no Cargo.toml names it) |
| independent proof check callable from outside egglog | MISSING | `ProofStore::check_proof` is `pub(crate)` | demonstrated gap for E4; the only fork change this slice needs |
| bounded saturation | EXISTS | egglog `run` with iteration limit; no node/time limit exposed as a typed outcome | E5 needs a typed "incomplete" result, not an error |
| typed single-table measure IR + guarded rules | EXISTS | contract `set_rules`: `Measure {Const, Count, Sum, CountRows, Opaque, Add, Sub}`, `Semantics {universe, arith, truth, sums, population}`, 8 rules with per-rule `Refusal` | the guard vocabulary (Exact/Wrapping/Checked/Ieee, TwoValued/ThreeValued, Distinct/Bag, Plain/Sym) already covers E3's OR/XOR-free half |
| multiplication / scale-by-constant in the measure IR | MISSING | `Measure` has no `Mul`/`Scale` | E1 (`z = 8c`) cannot be written in the contract IR today |
| Boolean OR vs GF(2) XOR as separate laws | PARTIAL | R2IL has `INT_OR` and `INT_XOR` as distinct ops; `set_algebra` is Boolean; no XOR set law | E3's OR/XOR arm needs an egglog sort split, not a contract change |
| relalg lowering to R2IL / mask-risc | EXISTS | `relalg::lower::run`, `run_with_program`, `run_as_written`; `program::run_program` (`let` planes, U2) | lowers `Rel` + `Measure {Count, Sum, Min, Max, Keep}` |
| scalar arithmetic in R2IL | EXISTS | `INT_ADD`, `INT_SUB`, `INT_MULT`, `INT_S_DIV`: scalar × scalar, checked `i64` | |
| lane arithmetic in R2IL | MISSING (by design) | `FoldError::LaneVersusLane` | a lane-scale `8 * c[i]` is not expressible; `SUM(8·c) = 8·SUM(c)` is (scalar after the fold) |
| reusable parameterized procedures | EXISTS | `r2il::procs` (`filter_lt`, `count`, `sum`, `group_sum`, `avg_parts`, …) | the compile target for B |
| group counts keyed by a slot | EXISTS | mask-risc `Terminal::GroupReduce { key: GroupKey, fold }` (COUNT/MIN/…), `GroupSumI32`, `GroupPowerSumsI32` | Example B's `GROUP_COUNT(R, y)` |
| dot product over a shared domain (`Σ_y r[y]·s[y]`) | MISSING | no Terminal with a dot/product doc line | Example B's last step is a scalar loop today (`join.rs` `bag_count`) |
| co-addressed domain for B | EXISTS (lab) | `mork-pathmap-probe::join::JoinDomain`, `JoinTable` (D-MPS-8, not admitted) | ADMISSION B review still owed |
| shared `let` planes (Example C) | EXISTS | contract `set_program` + `relalg::program` (U2 measured), `crepe-plan-probe::cse` (U3) | no egglog needed for C's baseline |
| Boolean composition (Example D) | EXISTS (lab) | `rmap::Tile`, `tile_gemm`, DR256 (#1666), `dir-rmap-probe` (#1685) | |
| Thought-Git replay receipt | EXISTS (lab) | `thought-git-probe::receipt` (generations, ABA guard) | no substitution/proof link; grep `substitut`/`egglog` in the crate: 0 hits |
| ClassView method DAG | EXISTS | contract `ClassView::compute_dag` (`class_view.rs:1403`), `ComputeEdge` | consumer for compiled-pattern identity, later |
| logical traceroute | PARTIAL | `set_rules::Trace { steps, refusals, budget_exhausted }` | rule-level only; no proof link, no binding requirements |
| three cache identities (law / pattern / register) | MISSING as types | witnesses exist separately (`RelationView.generation`, receipts) | |

**Smallest seam for Stage B (E1):** one lab crate that (1) encodes an exact-integer term language in egglog, (2) proves `z = 8c` from the premises with `prove`/`prove-variants`, (3) checks the proof with a newly public egglog checker, (4) extracts by a declared cost, (5) lowers to existing R2IL scalar ops over a `SUM` fold, (6) runs it on mask-risc, (7) compares with an exact oracle, (8) prints the traceroute. Open choice: whether the term language lives in the probe or `Measure` gains a guarded `Scale` in the contract (contract owner).
