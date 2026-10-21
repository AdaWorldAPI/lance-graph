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

## Decision: proxy chains are physical, relationships are semantic (operator, 2026-10-21)

Recorded as given:

> Extension proxy chains stay inside one Dn128 OU prefix. Logical memberOf and nested-group relationships may cross OU boundaries.

- An extension proxy continues a physical team/group chunk within its own OU locality.
- A memberOf or nested-group edge is a logical relationship and may target another OU.
- Cross-OU traversal uses version-attested global-to-local bindings.
- All 256 member slots remain usable. The proxy is separate routing metadata, not a stolen member bit.
- The chunk key must retain stable team/group identity. OU prefix alone is not sufficient.
- A move between OUs invalidates locality mappings, not the group's identity.
- Proxy chains do not replace transitive closure across group chunks.

Not built yet: the proxy chain itself. The probe below binds a team focus by group identity (`TeamFocus::group`, a group ordinal) with its tile mask as locality.

## Team focus, versioned populations, method folds, repair (revision 5)

Operator next probe: bind an OU/team focus without materializing every user; build only the relevant populations; reuse them across several relational method folds; repair them after edits; compare cold and warm with B+. Reuse mask-risc/relalg, no new evaluator.

Built in `crates/dir-rmap-probe/src/pop.rs`:

- **`Pop` carries the witness of its version.** `and`, `and_not`, `or`, `count_and` refuse populations of another version (`Refusal::Stale`). A population is a cache of derived state, rebuilt or repaired from the source rows of the version it claims.
- **`TeamFocus`** binds a group's identity (its ordinal) to the tiles and rows of its effective members. It reads only the member lists of the group and the groups nested in it.
- **`Qr::population_in`** fills a population only in the focus's tiles, through the per-slot pointer tables (`Runs`). The tables carry their version too.
- **Methods:** a `TeamAudit` class declares three methods in `compute_dag` (per-license holders of role r outside location L; unlicensed members; role holders). The DR table binds each to a `lance-graph-relalg` query over the populations lent as `Rel::Plane` (`Pop::plane`, a borrow, no copy), lowered by relalg's existing mask-risc path. All three reuse the same bound populations.
- **`Pop::repair`** brings a population to a new version. It diffs the old and new slot sets (the register is rebound). Members of slots that gained the property are scattered in. Members of slots that lost it, and every user whose direct memberships changed, are re-derived from their direct groups. A user or group with a second supporting path keeps the bit. The repair refuses if the slot binding itself changed. `Edit::RemoveNested` was added to the probe's edit set.

### Correctness (`tests/qr_focus.rs`, 5 tests)

- The three methods equal dir-sim's walk over the team's members (3,000 users, 16 licenses).
- In a narrow focus, populations are empty outside its tiles and equal to the full population inside.
- Over 8 random edit batches (direct add/remove, nesting add/remove among security groups), every repaired population (all roles, the licenses, the location, the team; full and focus-restricted) equals a fresh build.
- **Multi-support falsifier:** a group under two role-carrying parents keeps the role when one nesting edge is removed, and the population is unchanged. When the second edge goes, the group loses the role, but a member with its own direct path through a third group keeps it. Every user is checked against dir-sim.
- Mixing versions is refused: populations, pointer tables, focus, and repair.

Disable runs (all red):
- clear members of a removed slot instead of re-deriving (fails the multi-support test and the edit batches);
- skip touched users;
- skip scattering added slots;
- re-derive from only the last direct group;
- `count_and` ignoring the version;
- `population_in` ignoring the pointer tables' version;
- `population_in` ignoring the focus tiles;
- repair ignoring the population's version.

### Measured (`examples/qr_focus.rs`, release, every route checked equal before timing)

Question: the three methods for the security team with the most direct members, 16 licenses.

| | 65,000 users | 8,000 users |
|---|---:|---:|
| team members / focus tiles / all tiles | 77 / 69 / 254 | 244 / 32 / 32 |
| prepare (arm B CSR, shared with B+) | FAILED GATE (ms) | FAILED GATE (ms) |
| bind register | 52.9 µs | 53.6 µs |
| bind team focus | 0.8 µs | 0.8 µs |
| pointer tables + 18 populations in focus | 143.7 µs | 26.4 µs |
| of which populations with tables warm | 45.2 µs | 14.3 µs |
| 3 methods, relalg (18 queries) | 75.0 µs | 54.8 µs |
| **Q cold**, all but prepare | 326.3 µs | 151.4 µs |
| **Q warm**, version checks + 3 methods, relalg | 75.1 µs | 54.9 µs |
| **Q warm**, version checks + 3 methods, native `Pop` folds | 6.6 µs | 1.2 µs |
| after an edit: rebind register | 51.7 µs | 50.5 µs |
| after an edit: repair 18 populations + focus | 5.7 µs | 5.3 µs |
| after an edit: rebuild focus + tables + populations | 152.1 µs | 27.8 µs |
| B+ prepare + compile | FAILED GATE (ms) | FAILED GATE (ms) |
| B+ per focus user, `groups_of` + `roles_of` | 12.3 µs | 37.2 µs |
| B+ `members_of` populations (licenses + location + team) | 72.3 µs | 6.6 µs |

### Reading

- **Construction vs reuse holds.** Once the populations are bound, the three methods fold natively in 6.6 µs at 65,000 users. That is below B+'s per-user route at 77 members (12.3 µs) and 31× below it at 244 members.
- **relalg has a fixed cost of about 3 µs per query.** It is the same at 8,000 users (125-word planes) and at 65,000. With 18 queries per call it dominates the warm path (75 µs), so through relalg Q warm is slower than B+ per user for a small team. The algebra is reused as asked. The per-query program build is the cost to remove: one program with several counts, or a prepared program per method, kept across calls. Not built.
- **Repair is 27× cheaper than a rebuild** (5.7 vs 152 µs at 65,000 users) and is correct under multi-support. After an edit, rebinding the register (≈ 52 µs) now dominates; an incremental register update is open.
- **The pointer tables are two thirds of the cold population cost.** Each table reads the slot's whole member list, independent of the focus.
- **Tile locality: the focus saves little while user positions are GUID-ordered.** A 77-member team occupies 69 of 254 tiles; at 8,000 users a 244-member team occupies all 32. The tile restriction only pays when users are positioned by OU/team (Dn128 order), which matches the decision above that chunk locality is physical. The fixture has no such ordering.
- **Prepare** (dir-sim → CSR) is still a millisecond phase on both sides. Reading source lanes directly is not admitted (ADMISSION B), so every cold figure above excludes it.
- **Not done:** ACT-R and Thought-Git consuming the same focus and populations. This revision shows only the IAM side of that claim.

## Open

- **Pages by OU/team, not by ordinal:** pages are cut at user ordinal / 256. Cutting them by `Dn128` prefix (OU) or team needs a page key and a pointer table of non-empty tiles (the proxy-chained team design, not built).
- **Chains over 256 security groups** (`Spec::iam(8000)` has 1,439 ≈ 6 pages): a blocked Warshall over register pages. This is the D-RU-6 tile-recursion territory, where the measured win is narrow.
- **Mail and All chains:** only the security chain is bound.
- **The contract-level frozen-codebook witness** (ADMISSION B), shared with D-CMR-5 and D-RU-6.

## Closeout

`STATUS: measured | OUTCOME: DR256 register reproduces IAM nested-group roles exactly; versioned populations bound to a team focus answer three ClassView methods in 6.6 µs warm (native folds) or 75 µs through relalg (≈ 3 µs per query); repair after edits 5.7 µs vs 152 µs rebuild, correct under multi-support | OPEN: relalg per-query program cost, incremental register, Dn128 position order for tile locality, proxy chains, >256-group chains, CSR-free reads, codebook witness, ACT-R/Thought-Git consumers`
