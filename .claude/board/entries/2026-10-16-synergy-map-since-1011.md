# 2026-10-16 — Synergy map since the 2026-10-11 fleet listing

**Status:** OPEN — exploration map. It ratifies nothing; the operator chooses.
**Base listing:** ndarray `.claude/knowledge/coresearch-2026-10-11-simd-fleet.md`
(sections A–F, findings X1–X19, fruit tables). This entry revises every point
of it and adds what the merges since then opened or closed.
**Delta read:** first-parent merges since 2026-10-11 — lance-graph 97
(#1468–#1577), ndarray 9 (#355–#363), OGAR 12 (#351–#362), ruff 5 (#121–#125).
**Method:** five Haiku harvest cards (merge maps, board delta, quotes, census),
every relied-on item re-read from source on the main thread; ledger at the end.

## 1. The base listing, point by point

### A. `fearless_simd` vs `ndarray::simd`

| point | 2026-10-11 | now | evidence |
|---|---|---|---|
| compile-time dispatch, one realization per build | CLOSED | CLOSED | unchanged |
| runtime dispatch per kernel | CLOSED | CLOSED | startup-only rule held; `cpu_guard_drift.rs` (#360) adds a build-time warning when a target feature has no guard row |
| v4-on-AVX2 SIGILL | CLOSED (exe + library host) | CLOSED; aarch64 still LOOSE | `cpu_guard.rs:318` still says covering aarch64 needs `getauxval(AT_HWCAP)`; the only `getauxval` call reads `AT_PHDR` (`:465`) |
| fixed 512-shaped names | CLOSED | CLOSED | — |
| no SSE tiers | CLOSED | CLOSED | — |
| AVX2 integer polyfills | CLOSED by measurement | CLOSED | — |
| generated backends | PARTIAL | PARTIAL | unchanged (ternlog only) |
| per-level parity tests | PARTIAL, 18 types uncovered | **still 18 uncovered** | type names in `crates/*parity*` on master: U64x8 45, I8x16 28, U32x16 27, I32x16 20, U8x16 17, F32x16 13, F64x8 10, U8x64 1, U32x4 1; none of the 18 listed types appears |
| MSRV | PARTIAL | CLOSED for the toolchain, LOOSE in prose | fleet CI runs 1.99.0 (tesseract `rust.yml:62`); the prose is stale (§X17) |
| one binary everywhere | EXCLUDED | EXCLUDED | — |

### B. "Three places the CPU leaks back"

| point | now |
|---|---|
| the build cares | CLOSED (#348, #360) |
| consumers do not inherit ndarray's config | **CLOSED for lance-graph**: default `native` (#1505), per-runner `native` resolution keyed into the cache (#1526), production `x86-64-v4` / `sapphirerapids` (#1529–#1531, ndarray #361). Still LOOSE for lance-graph-java (§X19) |
| runtime dispatch covers some kernels | CLOSED |

### C. Accelerate mapping — PARK, unchanged.

### D. Unsafe inventory

| point | now |
|---|---|
| 4 OOB paths, 212 NEEDS-TF, AMX rung | CLOSED |
| missing `// SAFETY:` | LOOSE: raw count on master 974 `unsafe {` vs 393 `// SAFETY:` (grep of literals, not a coverage measure) |
| CONSOLIDATE / SAFE-API / REMOVABLE | LOOSE, mechanical |
| SAFE-CRATE (`bytemuck`/`zerocopy`) | operator question, unchanged |
| aarch64 guard | LOOSE |
| `is_x86_feature_detected!` sites | LOOSE: 96 sites in 17 files (was 95); not yet classified startup / once / per-call |

### E/F. Review issues and SSE follow-ups — unchanged (CLOSED / LOOSE as listed).

### Findings X1–X19

| id | now | evidence |
|---|---|---|
| X1 library host | CLOSED (#355) | `cpu-guard-library` CI job |
| X2–X4 ladybug slices | LOOSE, not touched | no ladybug merge in the delta |
| X6 MedCare/q2 v4 on Railway | DECIDED the other way: production is v4 (rules: Railway ~95 % v4, images built at v4/sapphirerapids); the guard reports the rest | D-PROD-V4 |
| X7 psABI | CLOSED as fact; `config-avx512.toml` is the Sapphire Rapids tier and CI compiles both tiers (ndarray #361) | — |
| X11 `mul_add` not bit-identical | **LOOSE and sharper**: `simd.rs:1241` documents "`F32x8::mul_add` is FUSED on every arm", but `simd_wasm.rs:197-199` computes `f32x4_add(f32x4_mul(a,b),c)` without `relaxed-simd` (two roundings) | the test `f32x8_mul_add_is_fused` carries no target cfg (`simd.rs:1247`), so it would fail on wasm32 without `relaxed-simd`; whether any CI row runs ndarray's lib tests there is NOT checked |
| X16 consumer sweep | LOOSE: `ndarray_bridge.rs` raw `_mm256_*` / `_mm512_*` blocks still present, all under `cfg(not(feature = "ndarray-hpc"))` (`:155`…`:449`) | dead in default builds; delete or keep as the no-ndarray fallback is a decision |
| X17 forbidden remedies in records | LOOSE: `ISS-NDARRAY-CANONICAL-COORDINATE-COUPLES-FLEET-MSRV` still lists a pinned `rev` option and a 1.98 table; `CARGO-COMPUTE-SUBSTRATE.md:218` still shows `rev = "<pinned>"`; lichess-rs `CLAUDE.md:32,60` and `README.md:93-97` say `+1.95`; tesseract `rust.yml:59-60,316` comments say 1.98.1 / "1.95 floor" while the step installs 1.99.0 | docs only |
| X18 two ndarray package ids | LOOSE and wider: OGAR `ogar-encryption` takes `encryption` from ndarray git `branch = "master"`; lance-graph takes ndarray by path; since #1534 lance-graph also takes OGAR by git `main` | PROBE: `cargo metadata` on a lance-graph binary that links `ogar-encryption` / `ogar-auth`, count ndarray ids |
| X19 lgj | LOOSE: `native/lgj-abi/.cargo/config.toml:2-5` still says ndarray "pins v3/AVX2" (it is `native`) and ":35 one known host"; `abi.rs:722-728` still asserts backend ∈ {AVX2, AVX-512}; `check_build_cpu` has 0 hits in lgj-abi | the handshake call from the 2026-10-11 low-fruit table is not done |
| TD-NDARRAY-SIMD-POPCNT-NATIVE | LOOSE: item (b) still proposes runtime detection of VPOPCNTDQ/AMX, which the startup-only rule forbids; the case it describes (host CPUID ≠ compiled features) is what `cpu_guard` now reports | regrade, strike (b) |

## 2. What the delta added — the arcs

| arc | shipped | still open |
|---|---|---|
| **Fold path** (R2IL / SQL polyfill) | `FoldDialect` crate (#1479), procedures + fused mode (#1484), r2il a workspace member, OGAR by git (#1534), core depends on the fold path (#1537), range intersection `fold_ranges` (#1487) | D-SQLP-3 benchmark gates -4/-5/-6; D-NALP-9 blocked on D-SQLP-4 |
| **Source pruning** | D-TELEPORT-0/-1/-2, D-PHOTO-0 (#1484/#1487/#1489) | — |
| **Key match** | D-KEY16 plan (#1475), ndarray parity + probe (#356/#357) | D-KEY16-3/-4 gate verdict not recorded here |
| **Class methods / SAG** | D-CMH-1, G1, G1-4a/4b/6, D-SAG-1 (#1504–#1520), then **withdrawn** to `backup/session-011BMJ-2026-10-12` (#1524); D-CMR set-IR plan (#1532) | D-CMR-1…9 all Queued; **D-SAG-5-1…10 are queued against the withdrawn D-SAG-1** |
| **NAL polyfill** | audit (#1495), D-NALP-1/2 (#1525), revision fold (#1538), exact truth codes in `causal_edge::isa::truth` (#1548) | D-NALP-4 lane-terminal question; -5 … -11 Queued; harvest PR #1544 still open "as a reminder" |
| **relalg** | D-RA-1 (#1546); its rewrite decided 675/2,000 programs in H14d | D-RA-2 maintainer decision; `ISS-RELALG-ABSORPTION-AFTER-DE-MORGAN` |
| **D-HOLO** | H1, H5, H6, H6b, H6c, H14, H14b (withdrawn), H14c, H14d — all measured | H8, H9, H11, H12 pre-registered, not run |
| **Thought lanes** | D-HL-5 probes: slot deltas, lane trie, identity sessions, coverage orientation, givens layer (#1564) | D-HL-1 O1/O2/O3/O6; D-HL-2/-3/-4/-6/-7/-8/-9 |
| **WAL** | D-WAL-1…7 (#1562/#1567); one-WAL ruling D-LNC-7 decided | reader wiring of the version clock; 60-min soak (truncated at 27.5 min) |
| **MeTTa / Datalog** | D-MTP-0/1/1b/6 (#1555), `ruff_metta_spo` (ruff #125), inventory + plan (#1574), council v3 (#1576), NO PINS (#1577); D-MTP-4 fix on fork `main` `e55dc61`, taken by W1's first build | D-ASM-1…6 Queued, not authorized; `TD-METTA-NO-CI-LINE-1`; 26-script golden baseline not built |
| **Build tiers** | native default (#1505), per-runner cache key (#1526), production v4 (#1529–#1531), ndarray CI for both production tiers (#361) | lgj (X19) |
| **IAM / directory** (other sessions) | dir-sim persistence and provisioning (#1473–#1488), RBAC one-subject rule, field masks, authorized walk (#1491, #1556–#1559), OGAR group policy / Graph groups / authorized resolve / sealed stream (OGAR #351–#360) | D-PBL-0…7 Queued |
| **Research docs** | Rubicon, PLN×Shannon, Hyperon parity, ONA assimilation, Haspelmath harness, qualia profile, oracle fidelity (#1492–#1500, #1536, #1543, #1545, #1549, #1570, #1571) | `ISS-QUALIA-A2B-CLUT-UNMEASURED` |

## 3. Closed ends (since 2026-10-11)

1. `cpu_guard` in library hosts, three OOB paths, AMX rung (ndarray #355).
2. Build-time warning for unguarded target features (ndarray #360); cargo-careful fix (#362).
3. Consumer tier policy: native default, cache keyed to resolved flags, production v4 (lance-graph #1505/#1526/#1529–#1531, ndarray #361).
4. Chained gather withdrawn as a key-read design (ndarray #358 → #359); hops are folded masks.
5. Fold path in core (D-FOLD-CORE-a/b); FoldDialect extracted; range intersection lowering.
6. NAL revision as one grouped fold (D-NALP-3); exact deduction/induction/abduction codes (D-NALP-4 scalar half).
7. WAL question answered: `LanceCycleWriter` stays the default with a capped session cache; MemWAL an option (D-LNC-7).
8. H14d: C's own operation measured; Lance masked reads cut rows always, bytes only when survivors are page-local.
9. Hyperon trie panic fixed on the fork and reachable without a pin (D-MTP-4 + #1577).
10. Supervisor respawn backoff (#1540).
11. Ten stale "In PR" board rows set to Shipped in this PR (D-CMR-0, D-TELEPORT-2, D-PHOTO-0, D-KEY16-0/1/2, D-DSV-2, D-DSP-3b, D-LBL-0, D-HPR-6a), each matched to its merge subject.

## 4. Loose ends, and the dots that connect them

**D1 — one set algebra, four spellings.** relalg `Rel` (D-RA-1), the planned
contract set IR `Runs` (D-CMR-1), D-HOLO's interval sets (H6: symbolic cheapest
in all 24 cells) and quack `fold_ranges` (D-TELEPORT-1) all compute ∩ ∪ ∖ over
coordinate runs before a fold. H14d gives the first price: the rewrite decided
34 % of 2,000 programs with zero runs and cut passes 5,898 → 1,065. The hinge is
D-RA-2 ("relalg into the contract set IR, or delete"). Until it is decided the
absorption fix has no agreed home.

**D2 — one preimage serves four callers.** D-MTP-1b measured the hop as the
preimage of a run set under a monotone map at 0.0001–0.0004 ns/row against
1.3–2.4 for the gather. D-CMR-5 (preimage), D-ASM-2 (Datalog existence half),
D-RA-4 (bounded recursion frontier) and Gremlin `out()` all need exactly this
operation. Nobody owns it as a fold op yet; the D-MTP-1b probe is the oracle.

**D3 — read masks, clustering and the Lance prefilter.** H14c/H14d: rows
scanned == |R| always, bytes cut only for clustered survivors (0.19× at 1 %
clustered, 1.00× uniform). D-HL-6 (read mask ∧ thinking mask before decode)
needs exactly an ordinal → row-address projection; D-WAL-6's version clock
already turns a point of view into one row range; D-HL-7 (sparse delta write)
decides where new rows land. Connected: the write layout must cluster by the
apertures that will read it, or masked reads stay row-only. The S3 question
lands here too: Lance's cache holds metadata only (source read, P-CACHE-1), so
on object storage every re-read of data is a new request — clustering cuts
requests, not just bytes.

**D4 — truth.** Exact scalar codes exist (`isa::truth`); revision is a fold;
the Gadamer gate invariants are pinned; D-HL-3 (evidential base, not layer)
and the FirstOfRun rule say independence is a separate contract. Open joint:
D-NALP-4 asks whether populations need a per-row truth lane terminal or can
call the scalar law per survivor. D-NALP-5 (populations) waits on it, and #1544
stays open only as a reminder of it.

**D5 — label codebooks.** Council v3 found no production codebook fit and made
`metta::codebook::LabelCodebook` (u16, mints at `add`) W1 scope, following the
quack `Binder::code` contract (`bind.rs:100-102`), which has no production
impl. D-NALP-6 needs a binder over a reverse label index; D-RA-3 needs labels
resolved through "the label codebook". These three are the same population
(data symbols → u16 addresses) and should be one implementation of
`Binder::code`. D-LBL (concept labels per language in `ogar-vocab`) is a
different population — concept ids, not data symbols — and stays separate.

**D6 — SAG on a withdrawn base.** D-SAG-5-1…10 (rerun SAG parity, ContextSlice,
provenance, policies) are queued against D-SAG-1, which #1524 withdrew. Their
new base is the D-CMR rebuild; until it exists they cannot start.

**D7 — two ndarray identities** (X18) now on two edges: OGAR git master and
lance-graph path, joined by #1534's OGAR-by-git. A binary that links both
compiles two `ndarray` packages.

**D8 — doc drift that contradicts rulings** (X11 fused claim, X17, X19,
TD-POPCNT (b), `lance-graph/.cargo/config.toml:55-56` "anything below v3 still
SIGILLs … never build with no target-cpu" while the avx and v2 arms exist and
the guard reports).

## 5. Fruit

### Low-hanging (≤ 1 hour each)

| # | item | repo | why / check |
|---|---|---|---|
| L1 | status sweep — done in this PR (§3 item 11) | lance-graph | each row matched to its merge |
| L2 | #1575 merge record — done in this PR | lance-graph | — |
| L3 | doc fixes: lgj config comment, lance-graph config `:55-56`, tesseract `rust.yml` comments, lichess `+1.95`, `CARGO-COMPUTE-SUBSTRATE.md:218`, MSRV issue `rev` option, TD-POPCNT (b) | lgj, lance-graph, tesseract-rs, lichess-rs | contradict standing rulings; one PR per repo |
| L4 | run `f32x8_mul_add_is_fused` on the wasm arm (`scripts/masking-parity.sh wasm` or the lib tests under wasmtime); if it fails, either fuse there or narrow the doc to "fused except wasm without `relaxed-simd`" | ndarray | expected 2⁻²⁴, unfused gives 0 |
| L5 | lgj: call `check_build_cpu()` in the `Abi` handshake, raise a Java exception; widen `abi.rs:722-728` to the reported arm | lance-graph-java | the warning path exists since #355 |
| L6 | #1544: close it with an ISSUES row carrying the D-NALP-4 question | lance-graph | a reminder PR is a board row in disguise |

### High-hanging (a session each)

| # | item | why | probe / gate |
|---|---|---|---|
| H1 | decide D-RA-2, then fix the absorption defect in the chosen IR, test first | D1 | the H14d program `¬(x ∧ (x ∨ y))` rewritten no wider than written; disable run red |
| H2 | preimage as a fold op (D-CMR-5) | D2 | == D-MTP-1b oracle; many-to-one refused; then D-ASM-2 uses it |
| H3 | S3 Titan arm on H14d: same R twice, bytes / requests / RSS, uniform vs clustered | D3 | counts, not milliseconds |
| H4 | clustered write layout for one class (D-HL-7 shape), re-run H14d stage 3 | D3 | bytes ≤ 0.25× at 1 % clustered by construction |
| H5 | parity groups for the 18 uncovered facade types | A | disable: flip one scalar `cmpgt` in a new group, it must go red |
| H6 | X18 probe | D7 | `cargo metadata` ndarray id count == 1 |
| H7 | aarch64 guard via `AT_HWCAP/HWCAP2` | A | dotprod build under `qemu-aarch64 -cpu cortex-a53` reports, does not SIGILL |
| H8 | classify the 96 `is_x86_feature_detected!` sites | D | `simd_caps.rs` once, `bitwise.rs`/`bgz17_bridge.rs` per call must classify correctly |

### Heavy-hanging (multi-session, needs an owner or a ruling)

| # | item | why |
|---|---|---|
| V1 | one `Binder::code` implementation for MeTTa, NAL and Datalog labels | D5; decide before D-ASM-1 builds |
| V2 | D-ASM W1–W3 (binder, semijoin existence, bag count, `LanceSpace`) | ratified, not authorized |
| V3 | D-NALP-5…7 (populations, typed surface, OpenNARS parity) | D4 |
| V4 | SAG-5 rebased on the D-CMR rebuild | D6 |
| V5 | D-HOLO H8/H9/H11/H12 | pre-registered, not run |
| V6 | Accelerate core/domain split; Kani; bytemuck/zerocopy | PARK / operator, unchanged |

## 6. Not searched

tesseract-rs, MedCare-rs, q2, HubSPO-rs, spear, a2ui-rs, ladybug-rs merges since
2026-10-11 (only lance-graph, ndarray, OGAR and ruff were mapped); the D-KEY16
probe verdict (ndarray #356/#357 bodies and example output); whether any
lance-graph binary links `ogar-encryption` today; the content of the withdrawn
backup branch beyond its board rows.

## 7. Harvest ledger (Haiku cards)

| card | task | result | spot check |
|---|---|---|---|
| substrate-mapper | lance-graph merge map | 97 rows, DONE | count re-run: 97 |
| substrate-mapper | ndarray / OGAR / ruff merge map | 26 rows, DONE | ndarray first-parent re-run: 9 |
| substrate-mapper | board delta (STATUS/ISSUES/TECH_DEBT/entries) | 203 / 3 / 3 / 51, DONE | ten status rows re-read before editing |
| quote-harvester | 13 stale sites | 12/13; tesseract used `origin/main` from the brief (its branch is `master`) — brief error, re-quoted by hand | Q3, Q5, Q6, Q10, Q13 re-read |
| census-counter | ndarray counters | DONE; one stray `send_message` call with a placeholder, which failed and sent nothing (a miss: out-of-card tool use) | parity type counts re-run with a wider pattern; aarch64 lines re-read |
