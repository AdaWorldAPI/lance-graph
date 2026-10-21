# 2026-10-21 — D-RFT-1-COMPOSE: typed composition over the DR256 slot coordinate (D-RFT-1)

**Status:** PROBE (lab, `crates/dir-rmap-probe`). TEST-PINNED and MEASURED as stated below; no production ABI changes.
**Claim:** EXO-50 `session_01G22yT6htkcdyXsihxxXdrg/036`, status in `/038`.
**Plan:** `.claude/plans/d-rft-relalg-thought-git-frontier-v1.md` (D-RFT-1); assessment `entries/2026-10-21-d-rft-d-lfa-0-assessment.md`.

## What landed

**`dir-rmap-probe::compose`.** It composes user → group membership (`Page.member`) with the role carriers (`Qr.role_slots`, and `Qr.role` for the slot-major direction).
- Both operands share one coordinate, the 256 slots, and one `Witness`.
- Per coordinate (user `i`, role `r`), the path mask is `member_i ∧ valid_i ∧ carriers_r`.
- Bag is the path count and Distinct is its existence. They are two readings of one computation, never two computations.

**Laws, each TEST-PINNED with a disable that goes red:**

| law | how |
|---|---|
| Stratum comes from a witness, not the caller | Distinct when the left fan (present keys per user) or the right fan (carrier slots per role) is ≤ 1, else Bag |
| A NULL key matches nothing | a key whose payload bit is set but whose presence bit is clear contributes no path (the same model as Quack's validity plane) |
| XOR is refused | an even number of parallel paths would read 0, an answer neither stratum gives |
| Seal mismatch is refused | the two operands were bound to different registers |
| A stale binding is refused | the caller's current `Witness` differs from the binding's |
| The bag sum is checked | `add_bag` refuses a wrap |
| The IAM diamond | two direct groups reaching one role: Bag 2, Distinct 1 |

**Execution.** Every count, existence and mask equality is an r2il procedure. Each is compiled once by `FoldDialect` and replayed through mask-risc `execute_into` over zero-copy row slices (a prepared statement):

| step | procedure | rows |
|---|---|---|
| present keys | `keep(intersect(member_i, valid_i))` | 256 slots |
| support (the user's roles) | `keep(union(role[s] for s in present_i))` | 64 roles |
| path count, support coordinates only | `count_and(member_i, valid_i, carriers_r)` | 256 slots |
| distinct | `count` over the support plane | `users × 64` |

Coordinates outside the support have no path by construction and are never evaluated. The probe has no counting loop of its own. The two pointer tables, of present slots and of supported roles, are the inversion from a plane to its local positions.

**r2il polyfill** (`lance-graph-r2il`):
- `IntXor` (R2IL ordinal 22) lowers to `MaskOp::Xor`, or to `BoolExpr::Xor` when fused.
- `ANY` and `ALL` lower to `Terminal::Any`/`All` and push 1 or 0.
- New procedures: `exists`, `all`, `differ` (`IntXor` then `ANY`, mask equality as a fold), `intersect`, `union`, `keep`, `count_and`.
- 5 r2il disables go red.

## Evidence

- **Independent oracle.** A depth-first walk over the `Prepared` CSRs (nesting restricted to security groups), not `Qr`'s Warshall closure. It is compared lane to lane over every page of `Spec::small()`, with supports compared by `differ`. Anti-vacuity: coordinates with count 1 and with count ≥ 2 both occur.
- **PostgreSQL 16.14** (`examples/rft1_pg_parity.rs`, with a recursive CTE): the per-coordinate lane is equal over 600 users × 64 roles; bag 1421 = `count(*)` and distinct 1374 = `count(DISTINCT (u, role))`. Every 7th user also has a membership row with a NULL group, which matches nothing in PostgreSQL either.
- **Disables.** 7 on compose (valid plane ignored, stratum forced to Bag, seal check, stale check, unchecked bag, support loop truncated, union dropping keys) plus 5 on r2il. All red.

## Measured (`examples/rft1_bench.rs`, release, median of 101 runs, 600 users / 3 pages)

| step / arm | time | allocations per run |
|---|---|---|
| bind (`Qr::bind`) | 46 µs | |
| pages | 39 µs | |
| program prep (`Composer::new`) | 1.4 µs | |
| A — composition (r2il procedures through mask-risc) | 310 µs | 39 |
| C — every coordinate, direct `ndarray::simd` | 131 µs | 0 |
| C+ — A's support-first plan, direct `ndarray::simd` | 14 µs | 0 |
| B — tuple baseline, LAB ONLY, not a candidate | 134–146 µs | 10 |

All arms agree on (bag, distinct) before anything is timed.

**History:** the first wiring ran one r2il fold per (user, role) coordinate and took 17 ms with 78 210 allocations. The fold count was the cause, not the algebra.

## Open

- **Per-execution cost.** A − C+ is about 296 µs for about 2 600 executions, roughly 115 ns of fixed cost per mask-risc `execute_into` on 4-word planes. The plan and the answers are the same, so the gap is execution overhead. UNKNOWN which part dominates (validation, scratch carving, tile loop); not profiled.
- **No count-product terminal.** mask-risc has no terminal that produces many coordinates' counts in one execution (a bit-matrix count product, or a masked positional popcount). The support-first plan is what keeps the execution count down. ndarray has no positional popcount either.
- **Fixture scale.** `Spec::iam(8000)` exceeds 256 security slots (`Refusal::TooLarge`), so the bench runs `Spec::small()`.
- **Not done:** recursion over composed strata (D-RFT-3), a second composition law beyond Count/Exists, and a PostgreSQL arm in CI (the example prints SKIP without a server).
