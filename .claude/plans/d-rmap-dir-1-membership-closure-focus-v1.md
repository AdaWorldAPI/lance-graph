# D-RMAP-DIR-1: compiled membership closure vs a 256-slot focus over dir-sim

> **Status: MEASURED, lab only (2026-10-20).** Code: `crates/dir-rmap-probe` (own workspace, CI `dir-rmap-probe.yml`). Nothing is admitted; no production file changed. dir-sim stays authoritative and is read only through its public API. Related: D-RMAP-1/1A (#1617), the `Focus`/`Csr`/`Tile` probe (#1621), D-RPR (#1624), walk_mask (#1634), `ISS-ADMISSION-PREP-BINDING`, the 2026-10-16 no-gather ruling.

## 1. Question

Does `lance-graph-dir-sim` gain more from sharing compiled closure results across users (B+), from reusing a local 256-slot geometry across evaluations (F), or from both? Each optimisation is judged on its own:

- **B:** does a prepared CSR remove the dominant membership-scan cost?
- **B+:** does a compiled group closure add a substantial gain when many users share inheritance?
- **F:** does a reusable 256-slot focus beat B+ once binding and amortisation are counted?

## 2. Arms

| arm | code | what |
|---|---|---|
| A | dir-sim `View` | `groups_transitive`, `members_transitive`, `security_identifiers`, `_through`, `is_member` (authoritative) |
| D | `oracle.rs` | crepe Datalog over the fixture's own integer ids (never dir-sim ordinals) |
| B | `prep::Prepared` | user → direct groups and group → direct users as CSR; nesting child → parents and parent → children as CSR; property planes from dir-sim's own `groups_where` |
| B+ | `prep::Compiled` | B + one closure row per group for each chain (every group, security, mail), up and down, by a fixpoint over the nesting CSR; roles carried by each group's SID chain; incremental insertion of a nesting pair |
| F | `focus::Focus` | B+ closure rows projected onto up to 256 bound groups (4 words per group); reverse map is a direct `u16` array over group ordinals |
| F+users | `focus::FocusUsers` | F + one 4-word focus row per user (the per-user pointer cache, measured so it is not argued) |

Role and location are ordinary groups. A role × location combination is a composite group nested in both, so no new membership semantics exists. "In role R and in location L" is asked of the closure (W5), never stored as a membership.

**Edge source.** dir-sim's membership lanes are `pub(crate)` and this probe changes no production file, so B is built from the same records dir-sim ingested (plus the same change list), translated through dir-sim's public ordinals. A production B would read the lanes; the build times here are an upper bound.

## 3. Fixture

Deterministic (`Spec::iam(n)`): 1,495 groups. These are an all-users group, a company group, 7 divisions, 300 role groups and 40 location groups. There are also 600 role × location composites, 500 resources (a DAG plus 6 two-cycles, 1 in 20 with an unread security flag), 16 license groups and 30 teams (distribution groups, 1 in 5 also security). Grants go on resources, roles, locations and teams. There is a `u8` attribute lane with 12 values. Divisions are mail-enabled but not security, so they cut the SID chain for every user and keep the mail chain.

At 8,000 users this gives 35,314 direct rows, 1,996 nesting pairs and 17 groups with more than 256 direct members. At 65,000 users it gives 263,470 direct rows and 138 such groups.

## 4. Correctness

- `tests/equivalence.rs`, 5 tests on `Spec::small()` (600 users):
  - B, B+ and D equal A on every user and group, for all three chains, roles and direct membership (every user × group pair);
  - F and F+users equal A and D on every role × location question for 16, 64 and 256 focus groups (more than 10,000 questions);
  - after an overlay (membership adds and removals, a nesting pair closing a cycle, a team nested in a role), incremental B+ equals a full recompile, and both equal A and D;
  - a stale witness refuses, and an unfocused group refuses;
  - anti-vacuity: cycles, groups over 256 members, closure passes > 1; the security and mail chains each filter for > 10 % of users and keep groups for > 90 %; most users hold a role; role × location is neither empty nor everything.
- The benchmark asserts, at full scale and before any timing:
  - B+ = A on a 2,000-user sample (500 at 65,000);
  - B+ = D on every membership (79,229 at 8,000; 611,965 at 65,000);
  - roles B+ = A on the sample;
  - F = F+users = B+ on every user and question;
  - W6 and W7 answers equal A.
- **Disable runs** (committed first, each restored with `git checkout`):

| guard removed | result |
|---|---|
| closure: skip unadmitted group `g` | red (2 tests) |
| closure: seed admitted groups only | red (2 tests) |
| incremental insertion: update descendants' down rows | red (overlay test) |
| incremental insertion: refresh SID-chain roles | red (overlay test) |
| witness check | red (refusal test) |
| focus refusal for an unbound group | red (refusal test) |
| oracle: `Sec(p)` on the recursive security rule | red (2 tests) |
| closure: skip unadmitted *parent* | **green**: it is an optimisation only. An unadmitted row is never seeded or written, so ORing it adds nothing. Marked so in the code. |

## 5. Results

**Measurement setup.** Release builds on a 4-core Xeon @ 2.10 GHz with AVX-512, 48 KiB L1d and 2 MiB L2, on a quiet machine. Figures are medians. Per-operation numbers are a timed batch divided by its size. Three runs at 8,000 users agreed within the ranges shown. Arm A is measured on a sample because each call scans every membership row. Cold means after touching 64 MiB of unrelated memory.

### 5.1 Construction (8,000 / 65,000 users)

| step | 8,000 | 65,000 |
|---|---|---|
| dir-sim snapshot (`observe`) | 26 ms | 282 ms |
| B prepare (with GUID translation) | 2.8 ms | 23.8 ms |
| B+ compile (3 chains × up/down) | 1.6 ms | 1.6 ms |
| D crepe, all relations | 76 ms | 920 ms |

### 5.2 Hot workloads, warm, per operation

| workload | A | B | B+ | 8,000: B+ vs A | 65,000: A / B / B+ |
|---|---|---|---|---|---|
| W1 direct membership | 307 ns | 10 ns | (B's CSR) | 31× | 510 / 9 ns |
| W2 user → effective groups | 408–466 µs | 92–124 ns | 32–49 ns | ~9,000–14,000× | 671 µs / 88 / 32 ns |
| W2 emitted as GUIDs | — | — | 299 ns | ~1,400× | 306 ns |
| W3 group → transitive members | 441 µs | 114 ns | 143 ns | ~3,100× | 820 µs / 505 / 515 ns |
| W4 roles via the SID chain | 412–431 µs | 181–192 ns | 13–15 ns | ~28,000–33,000× | 673 µs / 186 / 11 ns |
| W6 team members ∩ attribute (per team) | 460 µs | — | 0.3 µs | ~1,500× | 799 / 2.5 µs |

**A's cost is a full scan per call.** The cost of `members_walk` and `groups_reach` follows the number of membership rows, not the answer: about 0.4 ms at 35,000 rows, about 0.7 ms at 263,000.

### 5.3 W5: role × location, B+ vs F (8,000 users)

One evaluation is one user against every question in the focus.

| focus groups | questions | B+ eval | F eval | F+users eval | F prepare | F+users prepare | break-even vs B+ (F) |
|---|---|---|---|---|---|---|---|
| 16 | 11 | 46–47 ns | 24–25 ns | 7 ns | 22 µs | 113–126 µs | ~1,000 evaluations (measured: equal at N = 1,000; F 1.8× faster at N = 10,000) |
| 64 | 135 | 155–186 ns | 74–88 ns | 57–70 ns | 80–145 µs | 180–279 µs | ~1,000 (F 1.8–2.1× faster at N = 10,000) |
| 128 | 364 | 347–367 ns | 268–282 ns | 229–295 ns | 162–227 µs | 260–327 µs | ~2,000–2,500 (1.3–1.4× at N = 10,000) |
| 237–243 | 600 | 536–558 ns | 533–650 ns | 482–523 ns | 294–334 µs | 395–405 µs | none reproducible: F's evaluation equals or trails B+ |

- At 65,000 users the B+/F figures are within 10 % of these. F+users preparation grows to 582–866 µs, because it holds one row per user.
- Cold single evaluation (256 slots, 600 questions):
  - 8,000 users: B+ 2.6–3.2 µs, F 3.6–4.0 µs;
  - 65,000 users: B+ 4.0 µs, F 3.7 µs.
- **The focus saves only the OR stage** (4 words per direct group instead of 24). Once there are a few hundred questions, the question tests dominate and cost the same in both arms. The 256 × 256 tile was not built: B+ has already compiled the closure, so a tile would recompute it.

### 5.4 Overlay, reconciliation, rebuild (8,000 / 65,000 users)

| step | 8,000 | 65,000 |
|---|---|---|
| W7 all-user reconciliation, A (two versions) | 7.1 s | ~87 s, extrapolated from 500 users |
| W7 same, B+ (two compiled versions; the compiles are 2 × 1.6 ms more) | 0.9 ms | 4.4 ms |
| W8 B full rebuild | 2.7 ms | 23 ms |
| W8 membership edits: rebuild the two membership CSRs, closures untouched | 1.9 ms | 15 ms |
| W8 one nesting pair, incremental closure insertion | 2.1 µs | 2.5 µs |
| W8 B + B+ full recompile | 4.4 ms | 25 ms |
| W8 F (256 slots) rebind | 318 µs | 315 µs |
| W9 snapshot with one user inserted first | every user ordinal moves; the old B+ refuses (witness); snapshot 23 ms + B and B+ rebuild 4.1 ms | 248 ms + 24 ms |

### 5.5 Memory (8,000 users)

| structure | size |
|---|---|
| B (CSR tables and planes) | 214 KiB (1.3 MiB at 65,000) |
| B+ closures | 1.65 MiB: 3 chains × up/down × 287 KiB (one dense chain is 1,495 × 24 words = 281 KiB), independent of user count |
| F (256 slots) | 50 KiB |
| F+users | 300 KiB (2.0 MiB at 65,000) |

## 6. Decisions on the criteria

- **B: yes.** The dominant cost in A is a scan of every membership row on every call. B removes it, with a 31× to ~4,000× gain per call.
- **B+: yes, when users share inheritance.**
  - On top of B: 2–3× for a user's effective groups and 12–15× for roles.
  - All-user reconciliation drops from seconds to about 1 ms.
  - The compile costs 1.6 ms, independent of user count.
  - A nesting change is a 2 µs incremental insertion; a removal is a full recompile.
  - Group → members gains nothing over B, since both read the same direct-user rows.
- **F: not a meaningful general advantage.** Its gain is bounded: up to 2× per evaluation with 64 or fewer focus groups, and only after about 1,000 evaluations reuse one prepared focus, on warm caches. At 128 groups it is 1.3×. Near 256 groups there is no gain, and none when cold. F+users trades about 33 bytes per user for the lowest evaluation cost on small foci; its break-even is about 3,000 evaluations.
- **Answer to the question:** sharing compiled closure results across users is where the acceleration comes from. Reusing a local geometry adds a small, conditional gain on top. A dense global closure is already compact at this directory size: 24 words per row, resident in L2.

## 7. Boundaries and what this does not settle

- **The no-gather rule is not satisfied by B, B+ or F's hot path either**, and this probe does not claim it is. Each reads a CSR row or closure row at an address taken from a user's direct-group list, which is a value in another table. The improvement is in count: a handful of reads per call instead of a scan of every membership row. The co-addressed form would iterate the set bits of a user × group bit row, as #1634's walk_mask does with its adjacency rows. That is the same reads under coordinate addressing, and whether it counts as admissible is part of `ISS-ADMISSION-PREP-BINDING` / the gather decision.
- **Preparation translates addresses** (GUID → ordinal, closure projection onto the focus), counted. It is legal only in the lab until that issue is ruled.
- **The data is synthetic.** The fixture shape is a modelling choice, not a measured directory.
- **Not built:**
  - patching single membership rows (rebuilding the CSRs takes 1.9 ms at 8,000 users);
  - remapping old ordinals after a snapshot rebuild;
  - incremental removal of a nesting pair;
  - a focus that computes its own closure (tile).

## 8. Proposed next step (not authorized)

**D-RMAP-DIR-2:**
- B built inside `Snapshot::build` from the lanes: CSR by user and by group, and the nesting pairs as ordinals.
- B+ closure as a lazily compiled, version-witnessed cache behind the existing `View` methods.
- dir-sim's own tests and this probe's oracle as the equivalence gate.

It waits on two things: a ruling on the hot-path read pattern (§7) and explicit authorization. F is not proposed for production.
