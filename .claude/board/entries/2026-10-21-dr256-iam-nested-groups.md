# 2026-10-21 — IAM nested groups as a DR256 register (arm Q, dir-rmap-probe)

**Status:** MEASURED, lab only. Nothing admitted. Operator direction (2026-10-21): "IAM nested groups as a DR 256 user 256 group table (global to local register, 'QR code' as group membership matrix)". Answers open decision D3 of the ClassView × DR map (entry `2026-10-21-coresearch-classview-algebra-polyfill.md`) for the case where the security chain fits one register.

## What landed

`crates/dir-rmap-probe/src/qr.rs` (arm Q), `tests/qr_register.rs`, `examples/qr_cost.rs`.

- **Register:** every security group is bound to one of 256 local slots, once per snapshot (global → local). More than 256 security groups is refused (`TooLarge`), never truncated. That is what makes the in-tile closure complete.
- **Nesting:** a 256 × 256 bit tile (child slot → parent slots), closed inside the tile by Warshall (reflexive + transitive).
- **Membership matrix ("QR code"):** users in pages of 256 rows × 256 slot columns. A user's effective groups = its row × `reach`.
- **Roles:** per slot, the grants its reach carries. Per user, they are resolved per page of 256 users as a column fold: the users holding role `r` are the OR of the columns of the slots whose role set contains `r`.
- **Lazy pages (revision 2):** `Qr::bind` builds only the register (slots, nesting tile, closure, slot roles) and resolves no user. A page is resolved when a question names one of its users (`Qr::page`, `Pages`).
- **ClassView wiring:** the class declares `effective_roles` computed from `member_of` (`compute_dag`); the DR table maps that position to `Qr::roles_of`. Same shape as `classview_dr.rs`; no new type.
- **Address rule:** every inner loop runs over slot or user coordinates and tests a bit; no bit scan. The ordinal → slot translations are counted at bind (450 at 8,000 users: the nesting pairs) and per page (each user's direct groups). They are frozen-codebook decoding over one snapshot; the contract-level witness is still missing (ADMISSION B).

## Correctness

6 tests. For every user of `Spec::small` (600 users, 174 groups, 164 security groups):
- roles equal dir-sim (arm A), the crepe oracle (arm D) and B+;
- effective security groups match dir-sim's walk slot by slot.

This holds before and after a nesting add plus a membership removal (rebind). Refusals: >256 security groups (`Spec::iam(2000)`), a stale snapshot (bind and ask), a group outside the chain. Anti-vacuity: some users but not all hold roles, and some user gains a group only through nesting. A further test checks that bind resolves no page and that two users of one page resolve that page once.

Disable runs (committed first, each reverted; all red):
- Warshall removed;
- reflexive seed removed;
- truncate instead of refuse;
- bind ignores the witness;
- ask ignores the witness;
- grants not folded through reach;
- page column off by one user;
- last page dropped.

Revision 2 disable runs (all red): `Pages` re-resolving on every call, `page` ignoring the witness, a page dropping its last user, a column bit set at the global user index, a user mapped to the wrong page.

## Operator correction (2026-10-21)

Revision 1 resolved every user's roles at bind and measured that. The operator: "Nobody resolves nested groups for 64k users x 1700 groups at once. In real life even with Powershell hashtables that takes 2-12minutes". The unit of work is one team, OU or page asked for on demand, not the population. Revision 2 measures that unit; the eager figure is kept only as a comparison.

## Measured (release, `examples/qr_cost.rs`, group side of `Spec::small`, every route checked equal to dir-sim first)

| | 8,000 users | 65,000 users |
|---|---:|---:|
| bind register (slots + closure + slot roles, no user) | 39.9 µs | 41.3 µs |
| resolve one page (256 users), first touch | 14.7 µs | 13.3 µs |
| per user: Q roles from a resolved page | 0.9 ns | 1.0 ns |
| per user: B+ fold | 12.6 ns | 14.1 ns |
| per user: Q roles live (row × slot roles) | 140.1 ns | 148.1 ns |
| per user: Q effective row (M × R) | 167.8 ns | 258.4 ns |
| every page at once (not a workload; comparison only) | 486.6 µs | FAILED GATE (ms) |
| prepare (arm B, dir-sim → CSR; pre-existing) | FAILED GATE (ms) | FAILED GATE (ms) |

Register 19,108 bytes; one page 10,240 bytes.

## Reading

- **Both on-demand costs are independent of the user count:** the register binds in ≈ 40 µs and a page resolves in ≈ 14 µs at 8,000 and at 65,000 users. Asking about one team costs the register plus that team's page.
- **Queries on a resolved page are the cheapest route measured:** ~1 ns per user against ~13–14 ns for the B+ fold.
- **Live per-user products** (a row times the tile, 164 coordinates tested per user) are 10× slower than the B+ fold. The register pays when answers are read from a resolved page.
- **Eager resolve of the whole population** fails the gate at 65,000 users, as revision 1 measured. Per the correction above, that is not a workload.
- **The arm B prepare** (dir-sim → CSR) is a pre-existing millisecond phase, not introduced here. A page resolve reads that CSR; making the page read the source lanes directly is open.

## Open

- **Pages by OU/team, not by ordinal:** pages are cut at user ordinal / 256. Cutting them by `Dn128` prefix (OU) or team needs a page key and a pointer table of non-empty tiles (the proxy-chained team design, not built).
- **Chains over 256 security groups** (`Spec::iam(8000)` has 1,439 ≈ 6 pages): a blocked Warshall over register pages. This is the D-RU-6 tile-recursion territory, where the measured win is narrow.
- **Mail and All chains:** only the security chain is bound.
- **The contract-level frozen-codebook witness** (ADMISSION B), shared with D-CMR-5 and D-RU-6.

## Closeout

`STATUS: measured | OUTCOME: DR256 register reproduces IAM nested-group roles exactly; register ≈ 40 µs and one 256-user page ≈ 14 µs on demand, both independent of user count; ~1 ns queries | OPEN: OU/team page keys, >256-group chains, CSR-free page read, codebook witness`
