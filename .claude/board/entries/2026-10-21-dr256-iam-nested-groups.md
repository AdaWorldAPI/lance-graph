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

## Masked count over several tiles (revision 3)

Operator direction (2026-10-21): measure amortization by masking over multiple QR tiles, with interlaced row activation by selective tile masking.

A question ("how many users hold role r", or "are effectively in group g") becomes a **slot set**: `Qr::role_slots[r]` (`role` transposed at bind) or `Qr::slots_reaching(g)` (the slots whose reach contains g). It is asked over `k` tiles selected by a **tile mask** (one bit per page), with rows inside each tile selected by a **row mask** (all rows, or interlaced: even rows in even tiles, odd rows in odd tiles). The terminal is a count; no page is resolved and no per-user value is built.

Routes, all checked equal to a page-by-page resolve (`tests/qr_masked.rs`: every role and a sample of groups × 4 tile masks × 4 row masks; scratch is checked left zeroed), measured in `examples/qr_amortize.rs`:

- `page`: resolve each selected page on first touch, then fold its roles over the active rows;
- `warm page`: the same fold over pages already resolved;
- `masked` (`Qr::count_masked`): each needed slot's direct-member list is read once, and every member sets its bit in its tile, branch-free (the tile's selection bit is shifted into the OR); popcount of tile AND row mask;
- `runs` (`Runs::count`): a per-slot tile pointer table (`runs[s][p]..runs[s][p+1]` = slot s's members in tile p; member lists are sorted by user ordinal, so each tile is one run). Built for a slot on first use (`cold`), reused after (`warm`). Only selected tiles' runs are visited, OR'd into one 256-bit row per tile.

65,000 users (254 tiles), all rows. The role is carried by 32 slots (21,042 direct memberships); the group is reached from 26 slots (13,867):

| k tiles | role: page first touch | role: warm page | role: masked | role: runs warm | group: page first touch | group: masked | group: runs warm |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.0 µs | 0.1 µs | 33.1 µs | 0.7 µs | 56.4 µs | 22.8 µs | 0.7 µs |
| 8 | 116.3 µs | 0.5 µs | 34.2 µs | 2.9 µs | 446.6 µs | 22.0 µs | 2.3 µs |
| 32 | 502.0 µs | 2.1 µs | 34.2 µs | 10.3 µs | ms (gate) | 22.0 µs | 8.8 µs |
| 64 | ms (gate) | 4.2 µs | 34.3 µs | 19.8 µs | ms (gate) | 22.0 µs | 16.2 µs |
| 128 | ms (gate) | 8.4 µs | 34.2 µs | 64.6 µs | ms (gate) | 22.0 µs | 44.7 µs |
| 254 | ms (gate) | 16.6 µs | 33.2 µs | 155.0 µs | ms (gate) | 22.1 µs | 124.6 µs |

Runs cold (pointer tables built inside the call): 19–32 µs from k = 1 to 32 for the role. The interlaced row mask gives the same costs within noise: row activation is an AND at the terminal and costs nothing extra. At 8,000 users the masked route is 3.1 µs (role) and 2.1 µs (group) for every k.

### Reading

- **Amortization across tiles:** the masked route's cost is fixed by the memberships of the question's slots (≈ 1.5 ns per membership), not by k. It beats resolving pages one by one from k = 2–4 for a role and from k = 1 for a group, and reaches 100–700× at all tiles.
- **Selective tile masking:** the per-slot tile pointers make the cost proportional to k (≈ 0.3–0.6 µs per tile at these slot counts). They win while few tiles are selected and lose to the masked scan beyond about a quarter of the tiles (k ≈ 64 of 254): per tile they hop across all of the question's member lists, where the masked route reads each list once in order.
- **Warm pages answer role questions cheapest** (their roles are precomputed). Group questions have no precomputed column, so a warm page still pays a row × tile product per user and is slower than either masked route at every k.
- **The branch matters:** with a branch on the tile bit, the masked route ran 19–20 µs at small k and 64–73 µs at k = 128 (member order is random with respect to tiles, so the branch mispredicts at half selection). Branch-free it is a flat ≈ 33 µs. The sparse case is covered by the pointer tables instead.
- **So the choice is by selected fraction:** pointer runs below about a quarter of the tiles, the masked scan above, warm pages for role questions on already-resolved pages. That is a measured policy, not admitted; the crossover depends on slots per question and members per tile.

Disable runs (all red): tile mask ignored (masked; runs), scratch left dirty, empty run per tile, role transposition off by one bit, `slots_reaching` taken upward instead of downward.

## Chained population masking: license groups over all memberships (revision 4)

Operator direction (2026-10-21): filter license groups over all user memberships, as an example of chained population masking by positional filtering.

A **population** (`qr::Pop`) is one bit per user, at tile `u / 256`, row `u % 256`, for every tile. `Qr::population(b, slots)` builds one from a slot set by reading each slot's direct-member list once and scattering by position: `slots_reaching(g)` gives the users effectively in group `g` (nesting included), and `role_slots[r]` gives the holders of role `r`. Populations from one register line up by position, so filters chain as word-wise AND / AND-NOT (`Pop::and`, `Pop::and_not`). The answer is a count (`Pop::count`, `Pop::count_and`, which counts an intersection without building it). `Qr::everyone` is the full population.

Question (`examples/qr_license.rs`): per license group (16), count the users who hold the license, hold role r, and are NOT in location L. Chain: `role AND-NOT location`, then `license AND chain` counted per license. `tests/qr_license.rs` checks all 16 counts, and "no license at all" (`everyone AND-NOT any license`), against dir-sim's own walk per user. Anti-vacuity: the location removes some role holders, the counts differ across licenses, and some users but not all are unlicensed.

| | 8,000 users | 65,000 users |
|---|---:|---:|
| build role population | 2.3 µs | 24.0 µs |
| build location population | 1.3 µs | 9.3 µs |
| build 16 license populations | 11.6 µs | 66.2 µs |
| chain role AND-NOT location | 0.1 µs | 0.3 µs |
| 16 × (license AND chain), counted | 0.3 µs | 2.8 µs |
| whole question, end to end | 15.1 µs | 123.7 µs |
| per-user route over resolved pages (reference) | 363.0 µs | FAILED GATE (ms) |

At 65,000 users the populations read 21,615 + 8,349 + 58,521 memberships; one population is 8,128 bytes (254 tiles × 32 B).

### Reading

- **Building a population costs about 1.1 ns per membership read.** It is the whole cost: chaining and counting are word operations over 254 tiles (0.3 µs per chain step, about 0.18 µs per license count).
- **A population is reusable across questions.** Once built, the 16 license populations answer any other chain (another role, another location, unlicensed users) for the cost of the AND and the count alone.
- **The per-user route** (one effective row per user, then bit tests) pays the 256-slot row × tile product per user, so it is 24× slower at 8,000 users and fails the gate at 65,000.
- Populations cover every tile. Combining them with the tile mask and row masks of revision 3 (selected OUs or teams only) is not measured yet.

Disable runs (all red): AND-NOT acting as AND, `count_and` using OR, `everyone` dropping the last user of a tile, a population leaking slot 0.

## Open

- **Pages by OU/team, not by ordinal:** pages are cut at user ordinal / 256. Cutting them by `Dn128` prefix (OU) or team needs a page key and a pointer table of non-empty tiles (the proxy-chained team design, not built).
- **Chains over 256 security groups** (`Spec::iam(8000)` has 1,439 ≈ 6 pages): a blocked Warshall over register pages. This is the D-RU-6 tile-recursion territory, where the measured win is narrow.
- **Mail and All chains:** only the security chain is bound.
- **The contract-level frozen-codebook witness** (ADMISSION B), shared with D-CMR-5 and D-RU-6.

## Closeout

`STATUS: measured | OUTCOME: DR256 register reproduces IAM nested-group roles exactly; register ≈ 40 µs, one page ≈ 14 µs; masked multi-tile count flat 22–34 µs; 16 license counts through a role AND-NOT location chain in 124 µs end to end at 65k users (chain + counts 3 µs) | OPEN: OU/team page keys, crossover policy, >256-group chains, CSR-free reads, codebook witness`
