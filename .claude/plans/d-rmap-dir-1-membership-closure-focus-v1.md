# D-RMAP-DIR-1: compiled membership closure vs a 256-slot focus over dir-sim

> **Status: MEASURED, lab only (2026-10-20); round 2 (constraints 1–7) measured the same day, §9.** Nothing is promoted. Code: `crates/dir-rmap-probe` (own workspace, CI `dir-rmap-probe.yml`). Nothing is admitted; no production file changed. dir-sim stays authoritative and is read only through its public API. Related: D-RMAP-1/1A (#1617), the `Focus`/`Csr`/`Tile` probe (#1621), D-RPR (#1624), walk_mask (#1634), `ISS-ADMISSION-PREP-BINDING`, the 2026-10-16 no-gather ruling.

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
| F lazy | `focus::FocusLazy` | F + per-user rows computed on first use (round 2) |
| B+w | bench only | B+ reading only the closure words that hold a question group (question-aware B+, no focus; round 2) |

Round 2: F is bound per chain (all, security, mail), and its preparation is split into slot binding, closure projection and per-user rows, each timed and counted.

Role and location are ordinary groups. A role × location combination is a composite group nested in both, so no new membership semantics exists. "In role R and in location L" is asked of the closure (W5), never stored as a membership.

**Edge source.** dir-sim's membership lanes are `pub(crate)` and this probe changes no production file, so B is built from the same records dir-sim ingested (plus the same change list), translated through dir-sim's public ordinals. A production B would read the lanes; the build times here are an upper bound.

## 3. Fixture

Deterministic (`Spec::iam(n)`): 1,495 groups. These are an all-users group, a company group, 7 divisions, 300 role groups and 40 location groups. There are also 600 role × location composites, 500 resources (a DAG plus 6 two-cycles, 1 in 20 with an unread security flag), 16 license groups and 30 teams (distribution groups, 1 in 5 also security). Grants go on resources, roles, locations and teams. There is a `u8` attribute lane with 12 values. Divisions are mail-enabled but not security, so they cut the SID chain for every user and keep the mail chain.

At 8,000 users this gives 35,314 direct rows, 1,996 nesting pairs and 17 groups with more than 256 direct members. At 65,000 users it gives 263,470 direct rows and 138 such groups.

## 4. Correctness

- `tests/equivalence.rs`, 5 tests on `Spec::small()` (600 users); round 2 brings it to 11, see §9.2:
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

## 6. Decisions on the criteria (round 1; F revised in §9.6)

- **B: yes.** The dominant cost in A is a scan of every membership row on every call. B removes it, with a 31× to ~4,000× gain per call.
- **B+: yes, when users share inheritance.**
  - On top of B: 2–3× for a user's effective groups and 12–15× for roles.
  - All-user reconciliation drops from seconds to about 1 ms.
  - The compile costs 1.6 ms, independent of user count.
  - A nesting change is a 2 µs incremental insertion; a removal is a full recompile.
  - Group → members gains nothing over B, since both read the same direct-user rows.
- **F (round 1, superseded by §9.6):** not a meaningful general advantage. Its gain is bounded: up to 2× per evaluation with 64 or fewer focus groups, and only after about 1,000 evaluations reuse one prepared focus, on warm caches. At 128 groups it is 1.3×. Near 256 groups there is no gain, and none when cold. F+users trades about 33 bytes per user for the lowest evaluation cost on small foci; its break-even is about 3,000 evaluations.
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

## 9. Round 2: the seven constraints (2026-10-20)

Asked: is reusable 256-slot relationship geometry worth anything beyond compiled global closure, and under which workload conditions? Lab only. Nothing is promoted; no production file, canonical SoA or identity authority changes; the no-gather rule stands; `ISS-ADMISSION-PREP-BINDING` stays open.

Setup is as in §5: Xeon @ 2.10 GHz with AVX-512, L1d 48 KiB, L1i 32 KiB, L2 2 MiB, L3 260 MiB, 4 cores. Four full runs at 8,000 users agreed on every pattern below. Figures are the last run. Ranges are given where runs differed by more than 15 %.

### 9.1 What changed in the code

- **F per chain.** `Focus::bind(c, chain, groups, witness)` projects the closure of the chain it is bound to. F is no longer limited to every-group membership.
- **Projection from down rows.** For each focus group, F walks that group's downward closure row and sets its slot in every group the row names. The cost is the size of the focus groups' closures. Round 1 tested every slot of every group's upward row (`project_by_test`, kept as the reference). The two agree on every group. At 64 groups the down-row walk takes 1.6–2.7 µs and the reference 72–82 µs, about 30×. **This is what moved F's break-even**; see §9.5.
- **Split preparation:** slots, then projection, then per-user rows. Eager per-user rows are `FocusUsers`; lazy ones are `FocusLazy`.
- **Incremental refresh.** After a nesting pair, F re-projects the child's downward closure. After membership edits, F recomputes only the edited users' rows. Every refresh takes its witness from the B+ it read.
- **Refusals:**
  - more than 256 groups or a duplicate group;
  - a stale B+;
  - a stale focus, even over a current B+;
  - a group outside the focus.
- **The witness names the snapshot.** It now holds a `Weak<Snapshot>`, compared by pointer. This answers a Codex review finding: two stores both number their first version 0.

### 9.2 Constraint 2: closure correctness, and a refusal for anything incomplete

New tests in `tests/equivalence.rs`, 11 in total:

| test | what |
|---|---|
| `a_focus_on_every_chain_equals_dir_sim_and_the_oracle` | 256 spread groups, every chain, every user × slot. F, eager rows and lazy rows each equal A (dir-sim's own chain walks) and D (crepe). The down-row projection equals the reference. Anti-vacuity: each chain has more than 1,000 positive answers, and the filtered chains have fewer than the all-groups chain. |
| `a_path_through_unfocused_groups_is_represented` | For every chain, at least 20 users whose membership needs two or more hops. The focus holds only the target group, so every intermediate group is outside the focus, and it still answers yes. |
| `a_refreshed_focus_equals_a_fresh_bind_dir_sim_and_the_oracle` | Overlay: membership adds and removals, a nesting pair that closes a cycle, a team nested in a role. The focus refuses before the refresh. After the incremental refresh, every chain, both eager and lazy rows, equals a fresh bind, A and D. |
| `composite_membership_and_policy_conjunction_are_distinct_questions` | Constraint 5, see §9.4. |
| `a_focus_refuses_what_it_cannot_answer_completely` | Refusals: more than 256 groups, a duplicate, a stale B+, an unfocused group, and a stale focus over a current B+. |
| `a_witness_from_another_store_is_refused` | Two stores over the same data share `VersionId(0)` and the same counts, yet still refuse each other. |

Disable runs. The work was committed first, then each guard was removed and the file restored with `git checkout`. Every one went red:

| guard removed | result |
|---|---|
| projection walks the full closure; disabled: walks the focus group and its direct children only (a focus-local closure) | red, 3 tests: the unfocused-path, per-chain and refresh tests |
| refresh re-projects the child's closure | red (refresh test) |
| eager rows recomputed for refreshed users | red (refresh test) |
| lazy cache invalidated on nesting refresh | red (refresh test) |
| bind checks the B+ witness | red (refusal test) |
| bind refuses more than 256 groups | red: panics past the slot range |
| `ask` checks the focus's own witness | **green at first**: the test only asked through a stale B+, so the B+ check refused before the focus check mattered. A current-B+/stale-focus case was added; then red. |
| witness compares the snapshot pointer | red (`a_witness_from_another_store_is_refused`) |

### 9.3 Constraint 1: L1/L2 geometry and the measured working set

**Group-count sweep.** 8,000 users, the same 150 conjunction questions in every row, a focus of 64 groups. The first version of this table let the question count fall with scale, from 658 to 176. That confounded the trend and was caught before any conclusion was drawn. ns per evaluation.

| groups | closure words | one chain up | B+ working set (all users) | F working set | touched per eval: B+ / B+w / F | warm B+ / B+w / F / F+u | hot (one user) B+ / F | cold B+ / B+w / F / F+u |
|---|---|---|---|---|---|---|---|---|
| 1,495 | 24 | 280 KiB | 272 KiB | 45 KiB | 1.0 / 0.8 / 4.8 KiB | 218 / 279 / 74 / 58 | 208 / 60 | 2,103 / 2,290 / 1,121 / 621 |
| 2,895 | 46 | 1.0 MiB | 941 KiB | 82 KiB | 1.9 / 1.2 / 4.8 KiB | 318 / 363 / 76 / 57 | 301 / 61 | 2,161 / 2,364 / 1,170 / 654 |
| 5,695 | 89 | 3.9 MiB | 3.1 MiB | 143 KiB | 3.6 / 1.6 / 4.8 KiB | 344 / 407 / 76 / 57 | 308 / 64 | 2,786 / 3,168 / 1,334 / 703 |
| 11,295 | 177 | 15.3 MiB | 10.1 MiB | 235 KiB | 6.7 / 1.5 / 4.8 KiB | 436 / 466 / 79 / 57 | 372 / 62 | 2,602 / 3,075 / 1,264 / 669 |

What the measurements show, as distinct from what the sizes suggest:

- **B+'s working set passes L2 (2 MiB) between 2,895 and 5,695 groups, and that costs it little.** Warm (users cycled) is within 5–17 % of hot (one user repeated) at every size. B+ grows from 218 to 436 ns because its row grows from 24 to 177 words, not because of misses. A single evaluation reads 1–7 KiB, sequentially, and L3 is 260 MiB.
- **F is flat at 74–79 ns, and F+users at 57 ns, across a 7.5× range of groups.** Its working set stays under 240 KiB, inside L2. The per-evaluation advantage over B+ grows with group count: 2.9× at 1,495 groups, 5.5× at 11,295.
- **Cold, both arms pay a few misses.** F is 1.9–2.2× faster than B+, and F+users 3.4–3.9×.
- **The 8 KiB focus "fits in L1" only for the focus row itself.** What decides the speed is the question set. At 256 focus groups the composite fixture yields 4,312 conjunctions. As 4-word masks that is 135 KiB, beyond L1d. F with masks then takes 8.6–10.1 µs per evaluation, *slower* than B+ at 5.7–6.1 µs. The same focus with each question as two slot bits takes 4.5–4.9 µs. At 64 groups (658 questions, 21 KiB of masks) the mask form is the fastest, at 275–332 ns against 663–676 ns for slot bits. So the cost is set by the question representation crossing L1, not by the 256-slot geometry.
- **B+w (question-aware B+, reading only the needed closure words) never helps.** It reads fewer bytes but gathers scattered words, and is equal to or slower than B+ in every row.

### 9.4 Constraint 5: composite membership vs policy conjunction

Two questions, kept separate in three fixtures (§3 plus `Spec::without_composites()`):

- **W5b, materialized composite** ("is the user in composite C"): one membership bit per question. The question count equals the focus size.
- **W5a, policy conjunction on the composite fixture** ("in role R and in location L"): two bits. The question set is the distinct (role, location) pairs that users hold.
- **W5c, policy conjunction on a fixture without composites** (895 groups, 4,757 pairs): every user holds its role and location directly. The test checks that the conjunction question adds no membership: the oracle's membership count equals the direct and nesting closure.

Neither form adds dir-sim semantics: a composite is an ordinary nested group, and a conjunction is an AND asked of the closure.

Per evaluation, warm, ns, 8,000 users:

| focus groups | W5b composite B+ / F / F+u | W5a conjunction B+ / F / F+u (questions) | W5c conjunction B+ / F / F+u (questions) |
|---|---|---|---|
| 16 | 49 / 24 / 8 | 79 / 33 / 18 (41) | 196 / 36 / 22 (52) |
| 64 | 107–191 / 53–66 / 36–44 | 805–1,188 / 275–332 / 337–440 (658) | 888–914 / 260–308 / 303–408 (644) |
| 128 | 176 / 76–105 / 60–78 | 2,810–2,933 / 2,215–2,449 / 2,291–2,557 (2,101) | 2,728–3,110 / 1,977–2,128 / 2,027–2,049 (1,954) |
| 256 | 320–331 / 192–202 / 189–232 | 5,672–6,101 / 8,605–10,097 / 8,936–10,124 (4,312) | 5,708–5,961 / 7,907–10,366 / 8,370–9,687 (4,208) |

- Composite membership is a cheap question. F's gain over B+ stays at 1.7–2.9× across every focus size, because the number of questions is at most 256.
- Conjunctions multiply questions as the focus grows. F gains 2.4–5.4× up to 64 groups, then the question loop dominates, and past L1 the mask form loses (§9.3).

**At 65,000 users** (one run; the same fixture shape, so more distinct pairs per focus) the pattern holds and sharpens:

- 64 groups (956 conjunctions): F is 3.3× faster than B+ (391 vs 1,291 ns).
- 256 groups (8,160 conjunctions, 255 KiB of masks): F's mask form is 2.1× slower than B+ (23.9 vs 11.1 µs); the slot-bit form is 1.2× faster (9.1 µs).
- Composite membership: F is 1.8–2.8× faster at every focus size.
- Sweep at 1,495 groups: B+ 218 ns, F 70 ns, F+users 57 ns. The F+users table here is 2 MiB, at the edge of L2, and still costs nothing extra per evaluation.

### 9.5 Constraints 3 and 4: preparation accounting and amortization

**Preparation at 64 focus groups, 8,000 users** (W5a, composite fixture). Each phase alone; allocations are counted by a global counting allocator.

| phase | time | allocations | bytes allocated | retained |
|---|---|---|---|---|
| B prepare (CSR tables and planes) | 2.55 ms | 52 | 562 KiB | 214 KiB |
| B+ compile (3 chains × up/down, roles) | 1.57 ms | 7 | 1.65 MiB | 1.65 MiB |
| F focus construction (slots + reverse map) | 0.2 µs | 2 | 3.1 KiB | 3.1 KiB |
| F closure projection (down rows) | 1.6–2.7 µs | 1 | 46.7 KiB | 46.7 KiB |
| F closure projection by test (round 1's method) | 72–82 µs | 1 | 46.7 KiB | 46.7 KiB |
| eager per-user rows, all users | 84–95 µs | 1 | 250 KiB | 250 KiB |
| lazy per-user rows: allocate the cache | 5.4–5.9 µs | 2 | 251 KiB | 251 KiB |
| lazy per-user rows: first touch | 15–18 ns per user | 0 | 0 | 0 |
| refresh after 80 membership edits, eager rows | 0.3 µs | 0 | 0 | 0 |
| one narrow nesting pair (child closure 1 group): B+ insertion / F / F+users | 2.3–3.1 / 0.4 / 2.1 µs | 5 / 0 / 1 | 16 KiB / 0 / 212 B | 8 KiB / 0 / 0 |
| one wide nesting pair (child closure 648 groups): B+ / F / F+users | 10.4 / 33–45 / 143–153 µs | 5 / 0 / 9 | 16 KiB / 0 / 46 KiB | 8 KiB / 0 / 0 |

The hot path of every arm allocates nothing: 0 allocations over 8,000 evaluations per arm, in every W5 row. Eager rows cost 32 B per user. The lazy cache costs the same memory but fills rows at about 16 ns each on first use. It pays off only when a small fraction of users is asked about.

**Amortization.** The B+ compile is shared by every arm, so it drops out of the comparison:

- T_B+(N) = N · e_B+
- T_F(N) = P_bind + P_proj + N · e_F
- T_F+u(N) = P_bind + P_proj + P_users + N · e_F+u
- N* = P / (e_B+ − e_F), when e_B+ > e_F; otherwise there is no crossover.

"Measured" is the smallest N in {1, 2, 5, 10, …, 50,000} at which the directly timed total of F, bind included, is at or below the timed total of B+.

| workload, focus groups | P_bind + P_proj | e_B+ / e_F | N* computed | N* measured | F+users: P_users, N* computed / measured | rebuild per question (bind + project + 1 eval) |
|---|---|---|---|---|---|---|
| W5a, 16 | 1.3 µs | 79 / 33 ns | 27–36 | 50 | 93 µs, 1,539–1,730 / 2,000 | 1.3–1.8 µs |
| W5a, 64 | 2.6–2.8 µs | 805–1,188 / 275–332 ns | 3–5 | 10 | 91–95 µs, 131–201 / 200 | 2.9–3.8 µs |
| W5a, 128 | 4.0–4.9 µs | 2,810–2,933 / 2,215–2,449 ns | 6–9 | 5 | 90–102 µs, 168–271 / 200–1,000 | 6.9–7.0 µs |
| W5a, 256 | 6.5–7.0 µs | 5,672–6,101 / 8,605–10,097 ns | none | none ≤ 50,000 | none | 18.5–26 µs |
| W5b, 16–256 | 1.3–5.1 µs | 49–331 / 24–202 ns | 16–62 | 50–100 | 85–96 µs, 594–2,351 / 1,000–5,000 | 1.2–5.1 µs |
| W5c, 16–128 | 0.6–2.0 µs | 196–3,110 / 36–2,128 ns | 2–4 | 2–5 | 84–93 µs, 89–518 / 100–500 | 0.5–4.5 µs |
| W5c, 256 | 3.3 µs | 5,708–5,961 / 7,907–10,366 ns | none | none ≤ 50,000 | none | 14–17 µs |

- **Reuse over one prepared population pays after 2–100 evaluations.** Round 1 put this at ~1,000. The whole difference is the projection: round 1 tested every group's row for every slot.
- **Rebuilding a focus for every question never pays.** Per question it costs 0.5–26 µs, against 0.05–6 µs for a B+ evaluation, in every row.
- **F+users needs 100–5,000 evaluations**, because it builds 8,000 rows up front. It wins per evaluation only at small foci, and lazy rows cover the case where few users are asked about.
- The gaps between computed and measured N* come from the grid of N and from noise at very small N. The measured value is the one to rely on.

### 9.6 Answer and decisions

**Does reusable 256-slot relationship geometry add anything beyond compiled global closure? Yes, under conditions, and none of it changes B or B+.**

- **Conditions under which F gains:**
  - the questions fit L1 as masks (in this fixture, up to about 650 conjunctions or any number of single-bit composite questions);
  - the focus is reused at least 2–100 times;
  - the directory has more groups than a closure row comfortably covers.

  Then F is 2–5× faster per evaluation warm and about 2× cold (F+users about 4×). The gain grows with group count, from 2.9× at 1,495 groups to 5.5× at 11,295, because F's cost does not depend on the closure row width.
- **Conditions under which it does not gain:**
  - one question per focus, since rebuilding never pays;
  - question sets past L1: about 4,300 conjunctions over 256 groups, where F's mask form is 1.5–1.7× slower than B+ and the slot-bit form only 1.2–1.3× faster;
  - group → members (W3), where F does not apply.
- **The L1 picture.** The nominal 8 KiB tile fits L1 but decides nothing. The working set that matters is the question set plus the focus rows of the users' direct groups (45–235 KiB for every group, in L2). B+'s own working set passing L2 cost it 5–17 % at most here.
- **Round 1's F verdict is revised, not reversed.** The "~1,000-evaluation break-even" was an artifact of the projection method. The remaining caveats are unchanged: the gain is bounded, conditional on reuse and on question shape, and F is still not proposed for production. Promotion was not asked for and is not made.
- **Unchanged:**
  - B is yes;
  - B+ is yes;
  - D-RMAP-DIR-2 is not authorized;
  - the hot-path reads in B, B+ and F are still gathers (§7), and `ISS-ADMISSION-PREP-BINDING` stays open.

### 9.7 What round 2 does not settle

- The data is still synthetic. Real directories may have a different question-to-group ratio, and that ratio decides the conjunction rows.
- Only one machine was measured: a 4-core VM with a 260 MiB L3. On a CPU with a small L3, B+'s past-L2 rows would cost more than measured here.
- Not built: a dense register over the question set itself (for example, questions grouped by role, so one row test answers many), and a lazy focus that projects only the groups a user actually reaches.
