# Review: AdaWorldAPI/ndarray #422

- **Head:** `f279fe75f4a5c4aff75f473a38508cf9e9d20aea` (base `97654984da601c4705bc2350aa586d39d7c57fd2`)
- **Evidence head, AdaWorldAPI/lance-graph:** `317c4ffc04cb557d1142fd1331a0ccaedcd286de`
- **Scope:** PR #422 commits; consumer: lance-graph foundry-parity-lab (lib)
- **Reviewer:** {"tool": "crai_report.py + build_q8_evidence.py"}
- **Run:** `PARTIAL` (mask-risc internals not extracted; no test or benchmark executed)
- **Verdict:** NOT CLEAN: run PARTIAL
- **Semantics:** crai-rs CR-H0-REVIEW-SEMANTICS (c43ab35). Severity, evidence and confidence are separate axes.

## Q1 · admission · open

**Location:** `examples/u64x64/route_c.rs` · `u64x64 route C / A power-of-two quotient arms`

| severity | evidence class | confidence |
|---|---|---|
| not assessed | COMPILER_DERIVED | UNKNOWN |

**Claim**

> The per-row power-of-two quotient specialisation in #422 does not reach the named Foundry consumer. #422's Rust changes are confined to the 'u64x64' example target, which no crate can depend on. The consumer's executable G1 and G6 paths evaluate no per-row stock quotient: they compare stock against a precomputed 'd * q'.

**Evidence**

| grade | qualifier | statement | witness | revision |
|---|---|---|---|---|
| SOURCE_STRUCTURE |  | #422 changes 6 files: 3 Rust, all in example target(s) ['u64x64']; library Rust files changed: 0 | `ndarray:97654984da60..f279fe75f4a5` | `f279fe75f4a5` |
| COMPILER_DERIVED |  | G1 capacity (arm_b): `arm_b::{ItemQuery<'s>}::max_quantities` reaches 35 functions (3 only via dispatch or a closure body); divisions reached: ["arm_b::{ItemQuery<'s>}::cook_ops"]; other crates reached: ['lance_graph_mask_risc'] | `lance-graph:store` | `317c4ffc04cb` |
| COMPILER_DERIVED |  | G1 capacity (arm_b): `arm_b::{ItemQuery<'s>}::network_capacity` reaches 27 functions (2 only via dispatch or a closure body); divisions reached: ["arm_b::{ItemQuery<'s>}::cook_ops"]; other crates reached: ['lance_graph_mask_risc'] | `lance-graph:store` | `317c4ffc04cb` |
| COMPILER_DERIVED |  | G6 CanCook (method): `method::execute_can_cook` reaches 57 functions (7 only via dispatch or a closure body); divisions reached: none; other crates reached: ['lance_graph_contract', 'lance_graph_dt64', 'lance_graph_mask_risc', 'lance_graph_relalg'] | `lance-graph:store` | `317c4ffc04cb` |
| COMPILER_DERIVED |  | G6 CanCook (method): `method::{PreparedMethod<'s>}::prepare` reaches 49 functions (7 only via dispatch or a closure body); divisions reached: none; other crates reached: ['lance_graph_contract', 'lance_graph_dt64', 'lance_graph_mask_risc', 'lance_graph_relalg'] | `lance-graph:store` | `317c4ffc04cb` |
| COMPILER_DERIVED |  | G6 CanCook (method): `method::{PreparedMethod<'s>}::run` reaches 12 functions (0 only via dispatch or a closure body); divisions reached: none; other crates reached: ['lance_graph_mask_risc'] | `lance-graph:store` | `317c4ffc04cb` |
| COMPILER_DERIVED |  | integer division on an executable path, in `arm_b::{ItemQuery<'s>}::cook_ops`: `(q + r.per_staff - 1) / r.per_staff` | `lance-graph:crates/foundry-parity-lab/src/arm_b.rs:245` | `317c4ffc04cb` |
| SOURCE_STRUCTURE |  | mask-predicate threshold in `arm_b::{ItemQuery<'s>}::cook_ops`: `t: d * q` (names no lane) | `lance-graph:crates/foundry-parity-lab/src/arm_b.rs:235` | `317c4ffc04cb` |
| SOURCE_STRUCTURE |  | mask-predicate threshold in `arm_b::{ItemQuery<'s>}::cook_ops`: `t: (q + r.per_staff - 1) / r.per_staff` (names no lane) | `lance-graph:crates/foundry-parity-lab/src/arm_b.rs:245` | `317c4ffc04cb` |
| COMPILER_DERIVED | POSSIBLE | reference arm `oracle::{Oracle}::network_capacity` may reach `oracle::{Oracle}::can_cook`: `fx::stock(s, k, self.restocked) / d` | `lance-graph:crates/foundry-parity-lab/src/oracle.rs:62` | `317c4ffc04cb` |
| COMPILER_DERIVED | POSSIBLE | reference arm `arm_a::{Objects}::network_capacity` may reach `arm_a::{Objects}::can_cook`: `v / d` | `lance-graph:crates/foundry-parity-lab/src/arm_a.rs:75` | `317c4ffc04cb` |

**Reachability**

- confirmed: `G1 capacity (arm_b): arm_b::{ItemQuery<'s>}::max_quantities`, `G1 capacity (arm_b): arm_b::{ItemQuery<'s>}::network_capacity`, `G6 CanCook (method): method::execute_can_cook`, `G6 CanCook (method): method::{PreparedMethod<'s>}::prepare`, `G6 CanCook (method): method::{PreparedMethod<'s>}::run`
- possible: `arm_a::{Objects}::network_capacity → arm_a::{Objects}::can_cook`, `oracle::{Oracle}::network_capacity → oracle::{Oracle}::can_cook`
- unknown: 56 closure bodies (linked to their parent by Charon's naming only (POSSIBLE), not by a call row); lance-graph-mask-risc execution internals (not in the store; the G1/G6 paths call into it); production recipe divisor distribution (no production source; the only population is the synthetic fixture)

**Coverage**

- consumer_residuals: 0
- consumer_store: foundry_parity_lab library target
- ndarray_library_rust_changed: 0
- ndarray_other_rust_outside_examples: 0

**Falsifier**

- Extract lance-graph-mask-risc and re-run the reach: a per-row division reachable from `execute_into` with a stock-lane operand would refute the claim. Independently, make `Pred::GeI32` evaluation panic on any division and run foundry-parity-lab's tests.
- status: `NOT_RUN`

**No suggestion:** an admission finding, not a defect: no code change follows from the evidence

## Q2 · research · open

**Location:** `crates/foundry-parity-lab/src/arm_b.rs` · `arm_b::{ItemQuery}::cook_ops`

| severity | evidence class | confidence |
|---|---|---|
| not assessed | COMPILER_DERIVED | UNKNOWN |

**Claim**

> The executable path does evaluate integer division: '(q + per_staff - 1) / per_staff'. It is the threshold operand of a mask predicate and names no lane, so it is computed once when the predicate is built, not per row. A per-row quotient specialisation has nothing to apply to.

**Evidence**

| grade | qualifier | statement | witness | revision |
|---|---|---|---|---|
| COMPILER_DERIVED |  | division assertion (zero and overflow) at this site | `lance-graph:crates/foundry-parity-lab/src/arm_b.rs:245` | `317c4ffc04cb` |
| SOURCE_STRUCTURE |  | `t: (q + r.per_staff - 1) / r.per_staff` (names no lane) | `lance-graph:crates/foundry-parity-lab/src/arm_b.rs:245` | `317c4ffc04cb` |

**Reachability**

- confirmed: `arm_b::{ItemQuery<'s>}::cook_ops`
- possible: none observed
- unknown: none recorded

**Falsifier**

- Read the Pred::GeI32 threshold sites: the division's operands are query parameters, not lanes.
- status: `NOT_RUN`

**Suggestion** (inert proposal; applying it needs the owner's approval and revalidation against `f279fe75f4a5`)

> If the spec says 'no quotient is evaluated on the executable path', narrow it to 'no per-row stock quotient': a scalar ceiling division builds the staff threshold.

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
| ASSERTED |  | benchmark rows in the PR, author-reported | `ndarray:examples/u64x64` | `f279fe75f4a5` |

**Falsifier**

- Re-run `cargo run --release --example u64x64` with the disable arms (F7/F8) the PR names.
- status: `NOT_RUN`

**No suggestion:** a performance claim outside this review's evidence; no defect follows
