# mask-risc executor v1 — PR3 of the mask-execution arc

> **Status:** ACTIVE (2026-09-14). Filigree plan for PR3; the worker specs in §5 are
> the verbatim briefs. Predecessors: ndarray #306 (PR1, T1 facade), lance-graph #1225
> (PR2, contract in-place forms + `ir.rs` skeleton). Successors: PR4 (lgj-abi
> consumption, `D-MRL-1a`/`D-MRL-0e`), PR5 (join-elision hop probe).
> Companion plans: `cypher-mask-lowering-v1.md` (§7.1 Wave 1 is what this executor is
> held to), `duckdb-to-v3-translation-matrix-v1.md` (§3 gaps G1–G8).

## §0 — What PR3 is, in one sentence

Turn `lance-graph-mask-risc` from a compiling IR skeleton into the **one** borrowing
evaluator above `ndarray::simd`: a `Program` runs over caller-owned `Planes` and
caller-owned `Scratch`, every op is ONE delegation to a T1 facade word, the result is
diffed against a scalar oracle that never touches `ndarray`, on whichever backend
the test binary is built for (`x86-64-v3` in CI, `-v4` locally).

## §1 — Deliverables (D-ids)

| D-id | deliverable | falsifier (disable-verified red-then-green before it ships) |
|---|---|---|
| **D-MRX-0** | ndarray T1 prerequisite: the ten `*_to_mask_under` gated predicates (one per `Pred` variant), shared `pack_under` engine, parity arm `0xAxx` | the can-it-fire count test: 5 live gate words × 4 groups = 20 of 40 closure calls; delete the `gate == 0` skip and it reads 40 |
| **D-MRX-1** | `exec.rs`: `Scratch` (caller-owned, `words × slots`, validated against `Program::scratch_slots`), `execute(&Program, &Planes, &mut Scratch, out: Option<&mut [i32]>) -> Result<Value, ExecError>`; every `MaskOp` is one facade call; `Pred{under: Some}` calls the `_under` member; `Ternlog` dispatches the runtime `imm` to `mask_ternlog::<IMM>` through the generated 256-arm table | F-X1 [corrected 2026-09-14]: the two routings are SEMANTICALLY IDENTICAL, so no differential and no `op_histogram` reading can separate them (`op_histogram` counts IR ops, not facade calls). Its instrument is `count_probe`'s sparse-gate pair — gated vs `pred + and` — measured 3526 ns vs 10594 ns at the same count. A cost measurement, printed, never a gate; F-X3 (tail law): an odd immediate on `n_rows % 64 != 0` then `Count` — remove the tail clear and the count inflates |
| **D-MRX-2** | `reference.rs`: the scalar oracle, `#![forbid]`-style zero `ndarray` imports (a test greps the file), row-at-a-time evaluation of the identical `Program` | F-R1: the oracle module has no `ndarray::` token; F-R2 as RUN: `And` wired to `mask_or` reddens the two-input differential on seeded planes (the `AND2`/`AND3` swap this cell first named was not the disable performed) |
| **D-MRX-3** | `fuse.rs`: `BoolExpr` (leaf / and / or / not / xor over ≤3 distinct leaves) → `MaskOp::Ternlog{imm}` by truth-table evaluation; >3 leaves → left-assoc chain of ternlogs through scratch | F-B1: a 3-leaf `WHERE` yields `mask_passes() == 1`, making the fuser emit two `And`s reads 2; F-B4: the immediate for `a & b & !c` equals `ndarray::simd::ternlog::AND2_ANDNOT`-style value computed independently by bit-serial evaluation |
| **D-MRX-4** | `ternlog_dispatch.rs`: GENERATED 256-arm `match imm { 0 => mask_ternlog::<0>(..), … }` between `GEN-TERNLOG-DISPATCH` markers, generator committed under `crates/lance-graph-mask-risc/tools/`, CI regenerates-and-diffs | the diff gate itself; a test that every `imm` in 0..=255 dispatched equals the bit-serial reference on one random triple |
| **D-MRX-5** | the differential suite: seeded planes at `n_rows ∈ {0,1,63,64,65,130,1000,65_536}`, every `Pred` × gate ∈ {None, Some}, every two-input op, `Not`, 256 ternlog immediates, every `Terminal` | anti-vacuity: the 8 non-`Ne` predicates, UNGATED, satisfy `survivors * 3 < n_rows` at `n_rows >= 3` (the two `Ne` shapes are selective by complement; at `n_rows ∈ {0, 1}` the bound is unsatisfiable — `0 < 0` — so those two counts are execution-only checks). Separately the GATE is pinned selective (`gated < ungated`) at `n_rows >= 64`, or "gated" would be proven against a gate that removes nothing; the suite runs identically under `-Ctarget-cpu=x86-64-v3` (CI) and `-v4` (locally only — there is NO v4 CI arm; every workflow sets v3) |
| **D-MRX-6** | first 64k probe (`examples/count_probe.rs`): `COUNT(alpha & ((A & B) \| C))` as reference / interpreted / fused, equal counts, allocation counter = 0 bytes per execute, plus the F-X1 sparse-gate cost pair | the counting allocator reads 0 after warm-up for `interpreted` and `fused`; the four arms agree on the count |

Out of scope for PR3 (named, not forgotten): `hop.rs` (PR5, needs an edge lane and the
transpose census OQ-6), a strided `Operand` (the gap is in THIS IR — `ndarray::simd`
already ships `ternary_match_strided_to_mask`, `eq_u32_strided_to_mask` and
`masked_strided_group_sum`), `u8/u16/u64` compares (DuckDB matrix G1/G2),
`mask_set_range` (G6), cheap emptiness (G8), the Cypher `mask_lower` seam (Wave 1 of
the Cypher plan consumes this crate; it is not this crate).

## §2 — Dependencies and ordering

1. **D-MRX-0 merges to ndarray `main` FIRST.** lance-graph CI checks out the ndarray
   sibling at its default branch (`rust-test.yml` — no `ref:`), so a PR3 that calls
   `gt_i32_to_mask_under` is red until the ndarray PR is in.
2. PR3 stacks on the same designated branch after #1225 merges; base `main`.
3. `Cargo.toml` gains `ndarray = { path = "../../../ndarray", default-features = false,
   features = ["std"] }` — the SAME coordinates the planner uses (`hpc-extras` not
   needed: `simd_masking_ops` is behind `std` only). Contract dep only if `AlphaMask`
   is touched — it is not in PR3 (`Planes::masks` is `&[&[u64]]`; an `AlphaMask`
   caller passes `words()`).

## §3 — Laws the executor is written under

L1, L2, L4 and L5 are tests. **L3 is a reading of `execute`** — no instrument
counts facade calls — and is marked `[claimed, unverified]` in `exec.rs`'s own
module doc; its three named second shapes are the `_assign` aliasing form, the
odd-immediate tail clear, and `ternlog_self`'s fills.

- **L1 no allocation in `execute`.** `Scratch::new(words, slots)` is the only
  allocation and it is the caller's. A test with a counting global allocator
  asserts 0 bytes across 1000 executes of a 6-op program.
- **L2 no ISA.** No feature gate, no architecture intrinsic, no runtime
  detection macro — a test reads every production module's own source, CODE
  lines only (the law is stated in prose in `lib.rs`, which names the tokens
  it forbids).
- **L3 one delegation per op** `[claimed, unverified]`. Each `MaskOp` arm body
  is a single facade call plus operand resolution, with THREE named second
  shapes: the `_assign` aliasing form, the odd-immediate tail clear (spelled
  once in `clear_tail`), and `ternlog_self`'s fills for `x = f(x, x, x)`.
- **L4 the reference is independent.** `reference.rs` names neither the facade
  crate nor a sibling module that would import it transitively.
- **L5 one materialiser, plus one written-down oracle exemption.**
  `materialize_rows(mask, n_rows) -> Vec<usize>` is the consumer boundary;
  `reference_scratch` is the oracle's independent reading and is NAMED in the
  test rather than missed by it. The check greps every production module for a
  `pub fn` RETURNING an owning collection — a `Vec` parameter is not a
  materialiser.

## §4 — Executor shape (the design the workers are held to)

```rust
pub struct Scratch { words: usize, slots: Vec<Box<[u64]>> }   // caller-owned
pub enum Value { Mask(u16 /*scratch slot*/), Count(usize), Bool(bool),
                 SumI64(i64), OptI32(Option<i32>), Blended }
pub enum ExecError { ScratchTooSmall{need,have},            // the CALLER's buffer
                     ScratchSlotUndeclared{slot,declared},  // the PROGRAM's count
                     ScratchReadBeforeWrite{slot},          // pre-filled scratch: PR5
                     PlaneOutOfRange(u16), PlaneTail(u16), GateAliasesDst{dst},
                     LaneOutOfRange(u16), LaneKind{lane,expected,found},
                     LenMismatch{what,expected,found}, BlendNeedsOut,
                     SumRowBound{n_rows} }
pub fn execute(p: &Program, planes: &Planes, s: &mut Scratch, out: Option<&mut [i32]>)
    -> Result<Value, ExecError>;
```

Operand resolution borrows: `Operand::Plane(i)` → `planes.masks[i]`,
`Operand::Scratch(i)` → the slot (split-borrowed with `split_at_mut` when `dst`
aliases an input — in practice `mem::take` on the boxed slot, which moves a fat
pointer and copies no mask bytes; `dst == a`, and the commutative `dst == b`,
route to the `_assign` facade form, while the non-commutative `dst == b` and
every aliased `Ternlog` go through one `remap_imm`).

`Pred` lowering table (exhaustive, matches `ir.rs`):

| `Pred` | ungated | gated (`under: Some`) |
|---|---|---|
| GtI32 | `gt_i32_to_mask` | `gt_i32_to_mask_under` |
| LtI32 | `lt_i32_to_mask` | `lt_i32_to_mask_under` |
| GeI32 | `ge_i32_to_mask` | `ge_i32_to_mask_under` |
| LeI32 | `le_i32_to_mask` | `le_i32_to_mask_under` |
| EqI32 | `eq_i32_to_mask` | `eq_i32_to_mask_under` |
| NeI32 | `ne_i32_to_mask` | `ne_i32_to_mask_under` |
| EqU32 | `eq_u32_to_mask` | `eq_u32_to_mask_under` |
| NeU32 | `ne_u32_to_mask` | `ne_u32_to_mask_under` |
| MatchU32 | `ternary_match_u32_to_mask` | `ternary_match_u32_to_mask_under` |
| MatchU64 | `ternary_match_u64_to_mask` | `ternary_match_u64_to_mask_under` |

Terminals: `Count` → `popcount` over words (`mask_words` popcount via
`popcount_batch_u64`); `Any` → `mask_any`; `All` → `mask_all(words, n_rows)`;
`MaskedSumI32/MinI32/MaxI32` → `masked_sum_i32/masked_min_i32/masked_max_i32`;
`BlendI32` → `blend_i32` into `out`; `Keep` → `Value::Mask(slot)`.

## §5 — Worker allocation (declared up front)

- **Opus (orchestrator):** this plan; the `exec.rs` operand-aliasing design; every
  gate run centrally (`cargo fmt`, `clippy -D warnings`, `cargo test`) ONCE; the
  differential's disable runs; all commits/pushes; the council; board hygiene.
- **Sonnet W1 — `reference.rs` + `fuse.rs`** (disjoint from W2): from §3/§4 and
  `ir.rs`, no cargo build/check, only `cargo test -p lance-graph-mask-risc -- reference fuse`.
- **Sonnet W2 — `tools/gen_ternlog_dispatch.py` + `ternlog_dispatch.rs`** (generated
  file with markers) + its exhaustive test.
- **Orchestrator — `exec.rs`, `Scratch`, `lib.rs` wiring, `Cargo.toml`, CI lines,
  the 64k probe** (shared files; the mod lines land after W1/W2 return).
- Never Haiku.

## §6 — Board hygiene owed on landing

`STATUS_BOARD.md` rows D-MRX-0..6; `INTEGRATION_PLANS.md` prepend;
`LATEST_STATE.md` inventory delta (mask-risc: executor, oracle, fuser, dispatch);
`AGENT_LOG.md` fan-out entry (orchestrator sole writer); `EPIPHANIES.md` only if a
finding survives the differential (none pre-registered — a plan does not owe an
epiphany); `SUPERSESSION-INDEX.md` regenerated LAST.
