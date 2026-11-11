# Review: AdaWorldAPI/ndarray #422

- **Head:** `f279fe75f4a5c4aff75f473a38508cf9e9d20aea` (base `97654984da601c4705bc2350aa586d39d7c57fd2`)
- **Evidence head, AdaWorldAPI/lance-graph:** `c7151a1ecb03d4523ece505e42de1e77ed23111e`
- **Scope:** PR #422 commits; consumer: lance-graph foundry-parity-lab (lib), extracted at c7151a1ecb03d4523ece505e42de1e77ed23111e; foundry-parity-lab and mask-risc unchanged at 317c4ffc04cb557d1142fd1331a0ccaedcd286de
- **Reviewer:** {"tool": "crai_report.py + build_q8_evidence.py"}
- **Run:** `PARTIAL` (mask-risc not in the Charon store (GeI32 read as source only); no test or benchmark executed)
- **Verdict:** INCOMPLETE REVIEW: run PARTIAL; not a defect verdict (3 open finding(s))
- **Semantics:** crai-rs CR-H0-REVIEW-SEMANTICS (c43ab35). Severity, evidence and confidence are separate axes.

## Q1 · admission · open

**Location:** `examples/u64x64/route_c.rs` · `u64x64 route C / A power-of-two quotient arms`

| severity | evidence class | confidence |
|---|---|---|
| not assessed | COMPILER_DERIVED | UNKNOWN |

**Claim**

> In the consumer's shipped code, the per-row power-of-two quotient specialisation in #422 has no call site to reach. #422's Rust changes are confined to the 'u64x64' example target, which no crate can depend on. The G1 and G6 paths in the Charon store reach no per-row stock quotient: their mask predicates compare a stock lane against a scalar threshold (G6: 'd * q', built at method registration), and mask-risc evaluates 'GeI32' as a plain compare. Not covered: mask-risc predicates other than 'GeI32', and G1's unshipped per-store min/argmin step (EXO-227 P1b).

**Evidence**

| grade | qualifier | statement | witness | revision |
|---|---|---|---|---|
| SOURCE_STRUCTURE |  | #422 changes 6 files: 3 Rust, 3 in example target(s) ['u64x64']; library Rust files changed: 0; other Rust files changed: 0 | `ndarray:97654984da60..f279fe75f4a5` | `f279fe75f4a5` |
| COMPILER_DERIVED |  | G1 capacity (arm_b): 'arm_b::{ItemQuery<'s>}::max_quantities' reaches 35 functions (3 only via dispatch or a closure body); divisions reached: ["arm_b::{ItemQuery<'s>}::cook_ops"]; other crates reached: ['lance_graph_mask_risc'] | `lance-graph:store` | `c7151a1ecb03` |
| COMPILER_DERIVED |  | G1 capacity (arm_b): 'arm_b::{ItemQuery<'s>}::network_capacity' reaches 27 functions (2 only via dispatch or a closure body); divisions reached: ["arm_b::{ItemQuery<'s>}::cook_ops"]; other crates reached: ['lance_graph_mask_risc'] | `lance-graph:store` | `c7151a1ecb03` |
| COMPILER_DERIVED |  | G6 CanCook (method): 'method::execute_can_cook' reaches 57 functions (7 only via dispatch or a closure body); divisions reached: none; other crates reached: ['lance_graph_contract', 'lance_graph_dt64', 'lance_graph_mask_risc', 'lance_graph_relalg'] | `lance-graph:store` | `c7151a1ecb03` |
| COMPILER_DERIVED |  | G6 CanCook (method): 'method::{PreparedMethod<'s>}::prepare' reaches 49 functions (7 only via dispatch or a closure body); divisions reached: none; other crates reached: ['lance_graph_contract', 'lance_graph_dt64', 'lance_graph_mask_risc', 'lance_graph_relalg'] | `lance-graph:store` | `c7151a1ecb03` |
| COMPILER_DERIVED |  | G6 CanCook (method): 'method::{PreparedMethod<'s>}::run' reaches 12 functions (0 only via dispatch or a closure body); divisions reached: none; other crates reached: ['lance_graph_mask_risc'] | `lance-graph:store` | `c7151a1ecb03` |
| COMPILER_DERIVED |  | integer division on an executable path, in 'arm_b::{ItemQuery<'s>}::cook_ops': '(q + r.per_staff - 1) / r.per_staff' | `lance-graph:crates/foundry-parity-lab/src/arm_b.rs:245` | `c7151a1ecb03` |
| SOURCE_STRUCTURE |  | mask-predicate threshold in 'arm_b::{ItemQuery<'s>}::cook_ops': 't: d * q' (names no lane) | `lance-graph:crates/foundry-parity-lab/src/arm_b.rs:235` | `c7151a1ecb03` |
| SOURCE_STRUCTURE |  | mask-predicate threshold in 'arm_b::{ItemQuery<'s>}::cook_ops': 't: (q + r.per_staff - 1) / r.per_staff' (names no lane) | `lance-graph:crates/foundry-parity-lab/src/arm_b.rs:245` | `c7151a1ecb03` |
| COMPILER_DERIVED |  | integer division at G6 method registration ('method::can_cook_guard', reached from ['method::methods'], not from the G6 execution entry points): '(q + r.per_staff - 1) / r.per_staff' | `lance-graph:crates/foundry-parity-lab/src/method.rs:58` | `c7151a1ecb03` |
| SOURCE_STRUCTURE |  | mask-predicate threshold in 'method::can_cook_guard': 'd * q' (names no lane) | `lance-graph:crates/foundry-parity-lab/src/method.rs:55` | `c7151a1ecb03` |
| SOURCE_STRUCTURE |  | mask-predicate threshold in 'method::can_cook_guard': 'staff' (names no lane) | `lance-graph:crates/foundry-parity-lab/src/method.rs:59` | `c7151a1ecb03` |
| COMPILER_DERIVED | POSSIBLE | reference arm 'oracle::{Oracle}::network_capacity' reaches 'oracle::{Oracle}::can_cook' through a closure body linked to its parent by Charon's name only: 'fx::stock(s, k, self.restocked) / d' | `lance-graph:crates/foundry-parity-lab/src/oracle.rs:62` | `c7151a1ecb03` |
| COMPILER_DERIVED | POSSIBLE | reference arm 'arm_a::{Objects}::network_capacity' reaches 'arm_a::{Objects}::can_cook' through a closure body linked to its parent by Charon's name only: 'v / d' | `lance-graph:crates/foundry-parity-lab/src/arm_a.rs:75` | `c7151a1ecb03` |
| SOURCE_STRUCTURE |  | mask-risc 'Pred::GeI32 { lane: u16, t: i32 }' is 'lane[i] >= t' with 't' a scalar 'i32' | `lance-graph:crates/lance-graph-mask-risc/src/ir.rs:133` | `c7151a1ecb03` |
| SOURCE_STRUCTURE |  | the executor passes 't' unchanged to 'ndarray::simd::ge_i32_to_mask' | `lance-graph:crates/lance-graph-mask-risc/src/exec.rs:588` | `c7151a1ecb03` |
| SOURCE_STRUCTURE |  | 'ge_i32_to_mask' is the complement of 'lt_i32_to_mask', a compare of each lane value against the splatted threshold; no division | `ndarray:src/simd_masking_ops.rs:3449` | `f279fe75f4a5` |

**Reachability**

- confirmed: `G1 capacity (arm_b): arm_b::{ItemQuery<'s>}::max_quantities`, `G1 capacity (arm_b): arm_b::{ItemQuery<'s>}::network_capacity`, `G6 CanCook (method): method::execute_can_cook`, `G6 CanCook (method): method::{PreparedMethod<'s>}::prepare`, `G6 CanCook (method): method::{PreparedMethod<'s>}::run`
- possible: `arm_a::{Objects}::network_capacity → arm_a::{Objects}::can_cook`, `oracle::{Oracle}::network_capacity → oracle::{Oracle}::can_cook`
- unknown: 96 closure bodies in the crate (linked to their parent by Charon's naming only (POSSIBLE), not by a call row); G1 per-store min/argmin (limiting resource) (not shipped: open in EXO-227 (P1b); this review covers shipped code only); lance-graph-mask-risc beyond `GeI32` (not in the Charon store; only the `GeI32` evaluation path was read, as source); production recipe divisor distribution (no production source; the only population is the synthetic fixture)

**Coverage**

- consumer_residuals: 0
- consumer_store: foundry_parity_lab library target
- ndarray_library_rust_changed: 0
- ndarray_other_rust_outside_examples: 0

**Falsifier**

- Run Charon on lance-graph-mask-risc and re-run the reach from the G1/G6 entry points into it: any `DivisionByZero` assertion on a path that takes a stock lane as an operand refutes the claim. Smallest next verification: the predicate kinds other than `GeI32` that the consumer builds.
- status: `NOT_RUN`

**No suggestion:** an admission finding, not a defect: no code change follows from the evidence

## Q2 · research · open

**Location:** `crates/foundry-parity-lab/src/arm_b.rs` · `arm_b::{ItemQuery}::cook_ops`

| severity | evidence class | confidence |
|---|---|---|
| not assessed | COMPILER_DERIVED | UNKNOWN |

**Claim**

> The executable path does evaluate integer division: '(q + per_staff - 1) / per_staff'. It is the threshold operand of a mask predicate and names no lane, so it is computed once when the predicate is built, not per row. G6 has the same ceiling division at method registration. In shipped code, a per-row quotient specialisation has nothing to apply to.

**Evidence**

| grade | qualifier | statement | witness | revision |
|---|---|---|---|---|
| COMPILER_DERIVED |  | division assertion (zero and overflow) at this site | `lance-graph:crates/foundry-parity-lab/src/arm_b.rs:245` | `c7151a1ecb03` |
| COMPILER_DERIVED |  | division assertion at G6 method registration | `lance-graph:crates/foundry-parity-lab/src/method.rs:58` | `c7151a1ecb03` |
| SOURCE_STRUCTURE |  | 't: (q + r.per_staff - 1) / r.per_staff' (names no lane) | `lance-graph:crates/foundry-parity-lab/src/arm_b.rs:245` | `c7151a1ecb03` |

**Reachability**

- confirmed: `arm_b::{ItemQuery<'s>}::cook_ops`, `method::can_cook_guard`
- possible: none observed
- unknown: none recorded

**Falsifier**

- Read the Pred::GeI32 threshold sites: the division's operands are query parameters, not lanes.
- status: `NOT_RUN`

**Suggestion** (inert proposal; applying it needs the owner's approval and revalidation against `f279fe75f4a5`)

> Narrow 'No quotient is evaluated on the executable path' to 'no per-row stock quotient is evaluated': a scalar ceiling division builds the staff threshold.

## Q3 · research · open

**Location:** `examples/u64x64/main.rs`

| severity | evidence class | confidence |
|---|---|---|
| not assessed | ASSERTED | UNKNOWN |

**Claim**

> The PR's benchmark results (power-of-two shift no-regret; best route only on AVX-512 hot) were not re-run here.

**Evidence**

| grade | qualifier | statement | witness | revision |
|---|---|---|---|---|
| ASSERTED |  | benchmark results and disable runs, author-reported (plan §17) | `ndarray:.claude/plans/u64x64-llvm-probe-v1.md:949` | `f279fe75f4a5` |

**Falsifier**

- Per tier T in v2/v3/v4, build with `env -u RUSTFLAGS cargo --config .cargo/config-$T.toml rustc --release --example u64x64` (plan §4) and run `u64x64 bench <scenario>` for the §16 scenarios; compare with §17. The F7/F8 disable arms (test `d > 0` instead of `is_power_of_two`) are parity checks: they must turn `u64x64 check` red, and say nothing about speed.
- status: `NOT_RUN`

**No suggestion:** a performance claim outside this review's evidence; no defect follows
