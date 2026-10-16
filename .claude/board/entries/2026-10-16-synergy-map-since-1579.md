# 2026-10-16 — Synergy map since #1579 (merges #1580–#1591)

**Status:** OPEN — exploration map. It decides nothing; the operator chooses.
**Base:** `2026-10-16-synergy-map-since-1011.md` (#1579), which itself revised the
2026-10-11 fleet listing (A–F, X1–X19). This entry revises every point of #1579
and adds what the twelve merges since then opened or closed.
**Delta read:** lance-graph #1580–#1591 (#1580 reached through a second parent).
ndarray, OGAR, ruff: no merge since #1579. ndarray #364 (`eq_u32_via_lane_to_mask`,
D-ASM-11A) and lance-graph #1544 (D-NALP-4 harvest) were both closed unmerged at
17:42 the same day.
**Method:** two Haiku cards (merge map, quotes); every item relied on was re-read
or re-counted on the main thread (ledger in §7).

## 1. The base listing, point by point

### A. `fearless_simd` vs `ndarray::simd`

| point | #1579 | now | evidence |
|---|---|---|---|
| compile-time dispatch; no per-kernel runtime dispatch; v4-on-AVX2 SIGILL; 512-shaped names; SSE tiers; AVX2 integer polyfills | CLOSED | CLOSED | no ndarray merge |
| aarch64 guard | LOOSE | LOOSE | `cpu_guard.rs:318` still: "Covering it needs the kernel's HWCAP bits (`getauxval(AT_HWCAP)`)"; the only `getauxval` call reads `AT_PHDR` (`:465`) |
| generated backends | PARTIAL | PARTIAL | — |
| per-level parity, 18 types uncovered | LOOSE | LOOSE | no ndarray merge |
| MSRV prose | LOOSE | LOOSE | `ISS-NDARRAY-CANONICAL-COORDINATE-COUPLES-FLEET-MSRV` still OPEN (`ISSUES.md:194-196`) |
| one binary everywhere | EXCLUDED | EXCLUDED | — |

### B. "Three places the CPU leaks back"

| point | now |
|---|---|
| the build cares; runtime dispatch | CLOSED |
| consumers inherit no config | CLOSED for lance-graph. **LOOSE for lance-graph-java and sharper:** `native/lgj-abi/.cargo/config.toml:38` hard-codes `-Ctarget-cpu=x86-64-v4`, and its comment (`:2-3`) still says ndarray "pins v3/AVX2" (ndarray's default is `native`); `:34-36` says "one known host". The rules are `native` for host-bound builds and an explicit v4/sapphirerapids tier for shipped artifacts; lgj states neither choice, it hard-codes the second without the guard (X19) |

### C. Accelerate — PARK, unchanged. ### E/F — unchanged.

### D. Unsafe inventory — unchanged (no ndarray merge): missing `// SAFETY:`, CONSOLIDATE / SAFE-API, SAFE-CRATE question, 96 `is_x86_feature_detected!` sites unclassified.

### Findings X1–X19 and TD rows

| id | #1579 | now | evidence |
|---|---|---|---|
| X1, X7, X18 | CLOSED | CLOSED | — |
| X2–X4 ladybug | LOOSE | LOOSE | no ladybug merge |
| X6 | DECIDED (production v4) | DECIDED | — |
| X11 `mul_add` fused claim | LOOSE | LOOSE | `simd.rs:1241` "`F32x8::mul_add` is FUSED on every arm" vs `simd_wasm.rs:197-199` `f32x4_add(f32x4_mul(a,b),c)` without `relaxed-simd`; `simd_wasm.rs:37-40` documents the unfused path, so the two docs contradict each other; the test (`simd.rs:1247`) has no target cfg |
| X16 consumer raw intrinsics | LOOSE | LOOSE, not re-read | — |
| X17 forbidden remedies in records | LOOSE | **PARTLY CLOSED** by #1582: the `rev = "<pinned>"` template in `CARGO-COMPUTE-SUBSTRATE.md` and `cargo-substrate-architect.md` now names `branch = "master"`. Still open: the MSRV issue's pinned-`rev` option; lichess-rs and tesseract-rs comments (no merge there, not re-read) |
| X19 lgj | LOOSE | LOOSE | see B; `check_build_cpu` has zero hits in lance-graph-java `origin/main`; `abi.rs:726-729` still asserts `AVX2 \| AVX512` |
| TD-NDARRAY-SIMD-POPCNT-NATIVE | LOOSE (strike b) | **(b) CLOSED** by #1582 (status line, `TECH_DEBT.md:1911`); (a) objdump the faulting instruction and (c) the `Cargo.toml` comment remain |

## 2. The #1579 dots (D1–D8), revised

| dot | #1579 | now |
|---|---|---|
| **D1** one set algebra, four spellings (relalg `Rel`, D-CMR-1 `Runs`, D-HOLO intervals, quack `fold_ranges`) | LOOSE, hinge D-RA-2 | unchanged; ledger N17 regraded to "waits on D-RA-2" (#1581) |
| **D2** one preimage op for four callers (D-CMR-5, D-ASM-2, D-RA-4, Gremlin `out()`) | LOOSE, H2 proposed it as a fold op | **CLOSED AS A VIOLATION** (#1590): D-CMR-5 and D-RA-4 carry the GATHER banner. A preimage under a mapping reads a position taken from another lane's values. What stays: a hop is a coordinate range and chained hops are range intersections (`fold_ranges`, already shipped); a join is a layout that puts the pair at one coordinate. H2 is withdrawn |
| **D3** read masks × clustering × prefilter | LOOSE (N18, N19) | unchanged |
| **D4** truth: per-row lane terminal vs scalar law (D-NALP-4) | LOOSE; L6 asked to close #1544 | **L6 CLOSED**: #1544 closed 17:42; the question lives in the D-NALP-4 row (`STATUS_BOARD.md:226`). The question itself stays open |
| **D5** label codebooks: one `Binder::code` | LOOSE | **sharper** (§3 S5): #1587 measured the bind surface; D-ASM-1/1b should implement `Binder`, codebook behind `code`, storage per classid (§9 of #1579). Binder implementers today: quack `Users` (test), glove-parity, report (test), sap `CatsBinder`, dir-sim `UserBinder`, dir-sim `IamBinder` (test) |
| **D6** SAG-5 queued on the withdrawn D-SAG-1 | LOOSE | unchanged |
| **D7** two ndarray ids | CLOSED | CLOSED |
| **D8** doc drift against rulings | LOOSE | **PARTLY CLOSED** (#1582: rev template, TD-POPCNT (b), `.cargo/config.toml` oracle note). Still open: X11 doc pair, X19 lgj comment, MSRV issue option, tesseract/lichess comments. `.cargo/config.toml:55-56` ("Anything below v3 still SIGILLs … never build with no target-cpu at all") now sits next to the oracle note (`:56-60`); the two sentences agree but the first predates the avx and v2 arms and reads as if no lower tier existed |

## 3. What the twelve merges added

| PR | arc | what closed | what it opened |
|---|---|---|---|
| #1580 | hydrate `zip` 2 → 9 | workspace resolves again (every 2.x release yanked) | — |
| #1581 | ledger dedup vs #1579 | DOC-WIRE (2), N21, C30 duplicate | DOC-WIRE (1), BAND-RETIRE, O30, O29, D-ASM authorization, D-RA-2 |
| #1582 | doc drift L3 + OpenCog decisions | X17 rev template, TD-POPCNT (b) | `lance-graph-cog` (Apache-2.0 API OpenCog calls), OpenCog = temporary oracle; `TD-ONTOLOGY-REGISTRY-HASH-LOOKUP-REVIEW-1` |
| #1583 | board record of #1582 | — | — |
| #1584, #1585 | D-ASM follow-ups queued: 1b, 8–13 | — | Atomese front end, `learn` measures in arm-discovery, `lance-graph-das` probe, MORK oracle, leapfrog arms, content → NodeGuid (D-ASM-12), thinking facade (D-ASM-13) |
| #1586 | WCOJ co-research map | — | P-1 … P-7 |
| #1587 | WCOJ probes 1–3 | measured: galloping is a middle band; hybrid seek fastest or within noise at every N/m on v4; GYO α-acyclicity; bag oracle (counts) | P-7: the exact triangle mask needs `s_dst[rb[a]] == tc[a]` |
| #1588 | SAP optional CATS fields through validity planes | the "SAP optional fields" candidate from #1587 (`OPTIONAL_CODE_FIELDS = [6, 7, 8, 16, 20, 21]`) | U64 instants still have no `FieldKind` |
| #1589 | board catch-up #1388–#1588 | 115 missing arc entries | open-items ledger (`2026-10-16-board-catchup-open-items.md`) |
| #1590 | VIOLATION banners on gather designs | D-QMB-0…5, D-ASM-11/11A/11B, D-CMR-5, D-RA-4, the WCOJ probe entry: marked, not deleted | "the gather surface that already exists in code is a separate change" |
| #1591 | CE64 grammar wiring (D-CGW-0) + plan D-CGW-1…14 | D-CGW-0: every CE64 field of a parsed triple from a named source; Gadamer gate before counting; Codex P2 ×2 fixed | D-CGW-1…14; the contract amendment (§5 decisions) |

The P-7 conclusion is where the WCOJ arc and the violation ruling meet: the
probe proved that today's ops cannot express the triangle mask without reading
`s_dst` at `rb[a]`, and named a lane-to-lane-through-fk op as the missing piece.
That op is the forbidden shape. The ruling's answer is the layout:
`s_dst[i][a] : rb[i][a]` placed at one coordinate, compared side by side. The
measurements that survive: the hybrid seek (a cursor over a lane's own sorted
keys, no foreign read), GYO acyclicity (a property of the body, not of data),
the bag oracle (counts, a reference answer). The design that replaces D-ASM-11
is not written yet (§5 H4).

## 4. Synergies (the dots between arcs)

**S1 — one fixed register, five reasoners.** D-CGW-1 makes every CE64 field
mean one thing everywhere (the edge is the fixed point, like GMT; it carries no
qualia). That is what makes D-CGW-13 possible: Java OpenNARS and ONA (runners
already pinned and built by the NAL audit), MeTTa (Hyperon + PeTTa, golden
corpus), OpenCog (temporary oracle, #1582) and MORK (D-ASM-10) each project their
native answer onto F, C, the Pearl question, orientation, the inference mantissa
and the certification ladder, with "not comparable" where an oracle has no such
notion. This is the readout of D-ORACLE (`d-oracle-fidelity-matrix-v1.md`), not a
second harness. Order: D-CGW-1, then one fixture through the two NARS runners
that already build, then the rest.

**S2 — the wire gate is admission-as-mask.** D-RPF-4 is blocked on four
decisions, one of them "admission as mask" (open-items ledger). `wire` now runs
exactly that: per (S, P) group an `InterpretiveHorizon` with a `[u64; 4]` root
mask, set algebra only, counting after admission (G5). D-NALP-3's revision fold
is the counting half. The wire probe (741 rows, 593 fusions, 8 suspended, 132
echoes) is measured evidence D-RPF-4 can take for that decision.

**S3 — three devices into one PCS.** #1571 ships the PCS table and says the A2B
CLUT is unmeasured (`ISS-QUALIA-A2B-CLUT-UNMEASURED`). D-CGW-14 adds OpenPSI's
modulators as a device. D-SMD-4 already forbids reading CE64 bits as qualia
axes. All three are the same probe shape: drive named states, record a device
trace, fit device → PCS, hold one trace out. One probe harness serves the engine
telemetry CLUT and OpenPSI; OpenPSI demands stay goals (approved 2026-10-16).

**S4 — orientation has two readers, one table.** D-CGW reads 43..45 as the
Haspelmath dative function for text; `moore_tenant` reads a Moore slot for
nonverbal Cartesian data; D-SMD-2 wants grammar topology from Haspelmath maps.
D-CGW-1 leaves open whether Moore and Haspelmath become one code table. Under the
one-meaning rule they must: either one table with both domains' codes, or the
field holds one of them and the other moves to a row lane.

**S5 — Binder is the one bind surface.** CatsBinder's validity planes (#1588)
show three-valued binding working on a production binder. D-ASM-1b's Atomese
front end should implement `Binder` the same way: an unknown atom name is
`UnknownValue` (the trait never mints), an unbound optional is a validity plane,
and the per-classid label codebook (#1579 §9) sits behind `code`. Share the
trait, never the dictionary (CATS codes are batch-local; atom addresses are
stable).

**S6 — one address minting path.** D-ASM-12 (content → NodeGuid for atoms),
`wire`'s verse markers and `lance-graph-hydrate` all turn content into addresses.
They should share `mint_for` (V3 tail), not grow three hashers. Not checked:
whether `wire` mints at all or only carries markers.

**S7 — `syllogize` breaks the contract D-CGW-1 states.** Ledger N1:
"`syllogize` writes bits 50..52 as `0b111` and 53..63 as 0; `execute` writes W
and 59..63 as 0" — a derived edge is born as Astonishment and inherits no
witness or maturation. Under one meaning per field this is a contract violation,
not a style issue, so it belongs on the D-CGW-1 migration list beside the triad
readers.

**S8 — the triad readers to migrate (census for D-CGW-1).** Calls to
`direction` / `set_direction` / `*_pathological` / `concern_level` on
`CausalEdge64`: `causal-edge/src/edge.rs` 7 (including `compose:808`,
`concern_level:907`), `network.rs:142` (Simpson check), `v2_layout_tests.rs` 4,
`lance-graph-planner/tests/moore_nars16_probe.rs` 3, `cognitive-shader-driver`
examples 4, `staunen_scheduler_probe.rs` 1, `deepnsm-v2/src/wire.rs` 9. The
`moore_tenant.rs:655` hit is `MooreNars16::direction`, a different type. Callers
that pass the triad positionally into `pack(...)` are not caught by this grep and
must be found from `pack`'s call sites.

## 5. Closed ends, loose ends, fruit

### Closed since #1579

1. zip 9: the workspace resolves (#1580).
2. DOC-WIRE (2), N21, C30 (#1581).
3. X17 rev template; TD-POPCNT (b) (#1582).
4. SAP optional CATS fields bind three-valued (#1588).
5. 115 missing arc entries (#1589).
6. Every gather design marked VIOLATION with the correct plans named (#1590);
   D2 / H2 withdrawn with it.
7. D-CGW-0 (#1591); Codex P2 ×2.
8. #1544 and ndarray #364 closed unmerged (L6 done).
9. Decisions given 2026-10-16: 43..45 amended in the contract to orientation
   (readers migrated, own test-first PR); 46..49 stays the mantissa, activation
   becomes a row field; OpenPSI modulators enter qualia, demands do not.

### Low-hanging

| # | item | repo | check |
|---|---|---|---|
| L1 | board records for #1590 and #1591 (arc + LATEST_STATE); D-CGW-1/-9/-14 rows record the decisions | lance-graph | rides the next PR |
| L2 | narrow `simd.rs:1241` to "fused on every arm except wasm without `relaxed-simd`" and cfg the test, or fuse on wasm | ndarray | run the test on the wasm arm first |
| L3 | lgj: say which tier the artifact is (host-bound `native` or shipped v4), fix the "ndarray pins v3" comment, call `check_build_cpu()` in the handshake | lance-graph-java | needs push approval |
| L4 | `.cargo/config.toml:55-56`: name the avx/v2 arms so the sentence stops reading as "nothing below v3 exists" | lance-graph | doc only |
| L5 | STATUS rows: D-ASM-11A notes ndarray #364 closed unmerged | lance-graph | — |

### High-hanging (a session each)

| # | item | why | gate |
|---|---|---|---|
| H1 | **D-CGW-1 contract PR**: 43..45 = orientation in `layout.rs` / `edge.rs` docs; migrate the S8 readers; give the S/P/O pathology triad its own home; fix `syllogize`/`execute` (S7) | approved; unblocks D-CGW-13 and N1 | field-isolation matrix on 43..45; each migrated reader keeps its old answer through the new home; disable runs red |
| H2 | D-CGW-9: activation as a row field, out of 46..49 | approved | every wire edge decodes through the mantissa decoder as the inference it states |
| H3 | D-RA-2, then N17 absorption in the chosen IR | D1 | `¬(x ∧ (x ∨ y))` no wider than written |
| H4 | triangle closure as layout (replaces D-ASM-11): the pair placed at one coordinate, compared side by side | §3 | == datafrog and the bag oracle on F4; no read at a value-derived position |
| H5 | D-CGW-13 first fixture: ONA + Java OpenNARS on one NAL fixture, projected on CE64 | S1 | two disagreeing oracles never project to equal edges outside not-comparable fields |
| H6 | parity groups for the 18 uncovered facade types | A | one flipped scalar op goes red |
| H7 | aarch64 guard via `AT_HWCAP` | A | dotprod build under `qemu-aarch64 -cpu cortex-a53` reports, no SIGILL |

### Heavy-hanging (multi-session, owner or ruling needed)

| # | item | why |
|---|---|---|
| V1 | per-classid label codebook behind `Binder::code` (D-ASM-1/1b) | S5 |
| V2 | D-ASM W1–W3 authorization | ratified, not authorized |
| V3 | D-NALP-5…7 | D4 |
| V4 | SAG-5 rebased on D-CMR | D6 |
| V5 | D-HOLO H8/H9/H11/H12 | pre-registered, not run |
| V6 | D-CGW-14 OpenPSI device profile + the engine CLUT (one harness) | S3 |
| V7 | `lance-graph-cog` + `atomspace-lance` (D-ASM-12/13) | #1582, #1585 |
| V8 | Accelerate split, Kani, bytemuck/zerocopy | PARK |

## 6. Not searched

tesseract-rs, MedCare-rs, q2, HubSPO-rs, spear, a2ui-rs, ladybug-rs, lichess-rs
merges (the PR listing for ndarray, OGAR, ruff, tesseract-rs, lance-graph-java,
MedCare-rs and q2 showed nothing merged since 09:00 except the two closures);
X16; `pack(...)` positional triad callers; whether `wire` mints addresses.

## 7. Harvest ledger (Haiku cards)

| card | task | result | spot check |
|---|---|---|---|
| substrate-mapper | merge map #1580–#1591 | 12 rows, DONE; noted that #1580 is a second-parent merge and that some D-id hits are tails of `TD-` ids | file counts re-run: #1587 = 17, #1588 = 6 |
| quote-harvester | 10 sites | 9/10 at the stated place; Q1 path in my brief was wrong (`src/hpc/cpu_guard.rs` → `src/cpu_guard.rs`), the card found it and said so | `simd.rs:1241` re-read; #1544 / ndarray #364 `closed_at` re-queried; N1 quote re-read in `LEDGER-REVISION-2026-10-14.md:97` |
