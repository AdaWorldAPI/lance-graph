# 2026-10-21 — IAM nested groups as a DR256 register (arm Q, dir-rmap-probe)

**Status:** MEASURED, lab only. Nothing admitted. Operator direction (2026-10-21): "IAM nested groups as a DR 256 user 256 group table (global to local register, 'QR code' as group membership matrix)". Answers open decision D3 of the ClassView × DR map (entry `2026-10-21-coresearch-classview-algebra-polyfill.md`) for the case where the security chain fits one register.

## What landed

`crates/dir-rmap-probe/src/qr.rs` (arm Q), `tests/qr_register.rs`, `examples/qr_cost.rs`.

- **Register:** every security group is bound to one of 256 local slots, once per snapshot (global → local). More than 256 security groups is refused (`TooLarge`), never truncated. That is what makes the in-tile closure complete.
- **Nesting:** a 256 × 256 bit tile (child slot → parent slots), closed inside the tile by Warshall (reflexive + transitive).
- **Membership matrix ("QR code"):** users in pages of 256 rows × 256 slot columns. A user's effective groups = its row × `reach`.
- **Roles:** per slot, the grants its reach carries. Per user, they are resolved at bind as a column fold per page: the users holding role `r` are the OR of the columns of the slots whose role set contains `r`.
- **ClassView wiring:** the class declares `effective_roles` computed from `member_of` (`compute_dag`); the DR table maps that position to `Qr::roles_of`. Same shape as `classview_dr.rs`; no new type.
- **Address rule:** every inner loop runs over slot or user coordinates and tests a bit; no bit scan. The ordinal → slot translations happen at bind and are counted (31,470 at 8,000 users). They are frozen-codebook decoding over one snapshot; the contract-level witness is still missing (ADMISSION B).

## Correctness

5 tests. For every user of `Spec::small` (600 users, 174 groups, 164 security groups):
- roles equal dir-sim (arm A), the crepe oracle (arm D) and B+;
- effective security groups match dir-sim's walk slot by slot.

This holds before and after a nesting add plus a membership removal (rebind). Refusals: >256 security groups (`Spec::iam(2000)`), a stale snapshot (bind and ask), a group outside the chain. Anti-vacuity: some users but not all hold roles, and some user gains a group only through nesting.

Disable runs (committed first, each reverted; all red):
- Warshall removed;
- reflexive seed removed;
- truncate instead of refuse;
- bind ignores the witness;
- ask ignores the witness;
- grants not folded through reach;
- page column off by one user;
- last page dropped.

## Measured (release, `examples/qr_cost.rs`, group side of `Spec::small`, every route checked equal to dir-sim first)

| | 8,000 users | 65,000 users |
|---|---:|---:|
| bind (register + closure + roles for every user) | 535–577 µs | FAILED GATE (ms) |
| of which closure + slot roles inside the tile | ≈ 42–60 µs | ≈ 42–60 µs |
| per user: Q roles resolved at bind | 0.8 ns | 0.9–1.5 ns |
| per user: B+ fold | 12.5–16.8 ns | 14.9–18.3 ns |
| per user: Q effective row (M × R, live) | 166–179 ns | 178–183 ns |
| per user: Q roles live (row × slot roles) | 141–215 ns | 146–209 ns |
| prepare (arm B, dir-sim → CSR; pre-existing) | FAILED GATE (ms) | FAILED GATE (ms) |

(Three runs at 8,000 users; the tile-phase split is from a one-off instrumented run.)

## Reading

- **The register itself is cheap:** closing nesting inside the 256 × 256 tile costs tens of µs, independent of the user count.
- **Queries after bind are the cheapest route measured:** ~1 ns per user against ~13–18 ns for the B+ fold.
- **The cost is resolving every user at bind:** about 50 ns per user (membership matrix ≈ 13 ns, page fold ≈ 50 ns, measured by instrumentation at 65,000 users). That passes the µs gate at 8,000 users and fails it at 65,000. Skipping empty columns did not help (within noise; reverted).
- **Live per-user products** (a row times the tile, 164 coordinates tested per user) are 10× slower than the B+ fold. The register pays only when answers are resolved by page, not per row.
- **The arm B prepare** (dir-sim → CSR) is a pre-existing millisecond phase, not introduced here.

## Open

- **Lazy pages:** resolve a 256-user page on first use, so bind is register + closure only and is independent of the user count.
- **Chains over 256 security groups** (`Spec::iam(8000)` has 1,439 ≈ 6 pages): a blocked Warshall over register pages. This is the D-RU-6 tile-recursion territory, where the measured win is narrow.
- **Mail and All chains:** only the security chain is bound.
- **The contract-level frozen-codebook witness** (ADMISSION B), shared with D-CMR-5 and D-RU-6.

## Closeout

`STATUS: measured | OUTCOME: DR256 register reproduces IAM nested-group roles exactly; ~1 ns queries, tens-of-µs closure; bind passes at 8k users, fails the µs gate at 65k | OPEN: lazy pages, >256-group chains, codebook witness`
