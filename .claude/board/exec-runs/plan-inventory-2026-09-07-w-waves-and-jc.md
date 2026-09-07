# W-waves (SIMD) + jc pillars — verification inventory

Repos as of session start: lance-graph `main` @ `aeebfb23`; ndarray branch
`claude/randomized-signature-projection` == `origin/master` tip `b9afcb9b`.
Read-only throughout; no cargo run, no edits, no commits.

Docs read in full: `lance-graph/.claude/knowledge/ndarray-vertical-simd-alien-magic.md`;
`ndarray/.claude/knowledge/vertical-simd-consumer-contract.md` (no lance-graph-side
copy of the consumer-contract doc exists — confirmed absent by `ls`).

---

## W1a primitives

All 5 SHIPPED in `ndarray/src/`, across scalar (`simd_scalar.rs`), AVX-512
(`simd_avx512.rs`), NEON (`simd_neon.rs`), and partially WASM (`simd_wasm.rs`,
i4-unpack + saturating_abs only — no gather/prefetch/popcnt there). No
dedicated AVX2-only file defines these types; `U16x8`/`I8x16`/`U64x8` are each
declared once per arch-file (scalar / avx512 / neon), so the "AVX2" backend in
the contract doc is served by the x86_64 file (`simd_avx512.rs`), not a
separate avx2 module.

| # | TD id | Symbol | Status per TD entry (verbatim) | Found in code | Verdict |
|---|---|---|---|---|---|
| 1 | TD-NDARRAY-SIMD-UNPACK-I4-16D | `I8x16::from_i4_packed_u64`, `I8x16::lane_i8::<N>`, `batch_packed_i4_16<E,F>` | "Status: Open" (`TECH_DEBT.md:1762`) | `simd_scalar.rs:1684,1695,1971`; `simd_avx512.rs:2673,2694,3147`; `simd_neon.rs:2269,2283,2505`; `simd_wasm.rs:819,830` | **SHIPPED, STALE-DOC** (TD says Open, all 4 backends have it) |
| 2 | TD-NDARRAY-SIMD-SATURATING-ABS-I8 | `I8x16::saturating_abs`, `I8x32::saturating_abs` | "Status: Open" (`:1776`) | `simd_scalar.rs:1709,1737`; `simd_avx512.rs:2716,2937`; `simd_neon.rs:2301,2325`; `simd_wasm.rs:841` | **SHIPPED, STALE-DOC** |
| 3 | TD-NDARRAY-SIMD-GATHER | `U16x8::gather_u16`, `palette_lookup_u8x8` | "Status: Open" (`:1790`) | `simd_scalar.rs:1799,1863`; `simd_avx512.rs:2846,2900`; `simd_neon.rs:2356,2423` | **SHIPPED but PARTIAL — STALE-DOC.** The x86_64 ("avx512") impl at `simd_avx512.rs:2846` is, by its own doc comment (`:2832-2836`), "a scalar-loop polyfill (real AVX2 gather via `_mm256_i32gather_epi32` + downcast is tracked as a follow-up optimisation)" — the true-SIMD AVX2 gather the contract doc specified is NOT implemented; only the scalar-correctness-anchor fallback ships on every arch. |
| 4 | TD-NDARRAY-SIMD-PREFETCH | `prefetch_read_t0/t1/t2` | "Status: Open" (`:1803`) | `simd_scalar.rs:1887,1893,1899`; `simd_avx512.rs:3091,3104,3115`; `simd_neon.rs:2445,2462,2477` | **SHIPPED, STALE-DOC** |
| 5 | TD-NDARRAY-SIMD-POPCOUNT-U64 | `U64x8::popcnt`, `U64x8::xor_popcount`, `U64x4::popcnt` | "Status: Open" (`:1817`) | `simd_scalar.rs:1916,1934,1952`; `simd_avx512.rs:2987,3046`; `simd_avx2.rs:2153,2176,2195` (`U64x4`/parity) | **SHIPPED, STALE-DOC** |

All five W1a `TECH_DEBT.md` entries carry `**Status:** Open` verbatim while
every symbol they specify exists in `ndarray/src`. None of the five entries
has been updated to reflect this — a systematic doc-lag, not an isolated slip.

## W1b consumer migrations

Raw-intrinsic census (pattern: `_mm512_|_mm256_|_mm_|core::arch::|vld1q|is_x86_feature_detected|is_aarch64_feature_detected`,
count = matching **lines**, current tree):

| Crate / file | TD id | TD status (verbatim) | Raw-intrinsic lines found | Verdict |
|---|---|---|---|---|
| `crates/holograph/src/hamming.rs` | TD-SIMD-SWEEP-W1 | "Status: Open" | 25 | Open, accurate — not migrated |
| `crates/lance-graph/src/graph/blasgraph/types.rs` | TD-SIMD-SWEEP-W2 (half) | "Status: Open", cites "types.rs (22 raw ops)" | **0** | **STALE-DOC.** `types.rs` has been fully migrated: `hamming_distance_dispatch` (`types.rs:452-467`) now calls `ndarray::hpc::bitwise::hamming_distance_raw(a_bytes, b_bytes)` under `#[cfg(feature = "ndarray-hpc")]` with a scalar `count_ones` fallback otherwise. Zero raw intrinsics remain in this file. |
| `crates/lance-graph/src/graph/blasgraph/ndarray_bridge.rs` | TD-SIMD-SWEEP-W2 (half) | same entry, cites "ndarray_bridge.rs (60 raw ops)" | 66 | **Open, and still uses the exact violation the primitive was built to close** — `_mm512_popcnt_epi64` at lines 465 and 493, where `U64x8::popcnt`/`xor_popcount` (shipped, TD #5) should be consumed instead. |
| `crates/bgz17/src/simd.rs` | TD-SIMD-SWEEP-W3 (half) | "Status: Open" | 22 | Open, accurate. File still hand-rolls `SimdLevel` enum + `detect_simd()` (AP-SIMD-3/8) and both `_mm256_i32gather_epi32` (`:136`) and `_mm512_i32gather_epi32::<2>` (`:223`) directly — none of the shipped `gather_u16`/`palette_lookup_u8x8` primitives are consumed. |
| `crates/bgz17/src/prefetch.rs` | TD-SIMD-SWEEP-W3 (half) | "Status: Open" | 1 | Open, accurate (and slightly better than the TD text: the original aarch64 `_prefetch` call it also cited is already gone — replaced by a documented no-op citing `stdarch_aarch64_prefetch` instability, `prefetch.rs:98-108` — only the x86 `_mm_prefetch` at `:96` remains). |
| `crates/lance-graph-contract/src/mul.rs` (`pub mod i4_eval`, from `:727`) | TD-SIMD-SWEEP-W4 | "Status: Open" (P0) | 64 | Open, accurate — entire raw-intrinsic block (`_mm512_srli_epi64`, `_mm512_cmp*_epi64_mask`, `_mm512_loadu_si512`, etc., `:1086-1279`+) untouched by either W1a-#1 (`from_i4_packed_u64`) or W1a-#2 (`saturating_abs`), though both are shipped and ready to consume. |
| `crates/thinking-engine/src/engine.rs` | TD-SIMD-SWEEP-W5 | "Status: Open" (P3) | 1 (`is_x86_feature_detected!("avx512vnni")`, `:508`) | Open, accurate. The crate already imports `ndarray::simd_amx` (`:160`) and routes the actual matvec dispatch through `simd_amx::matvec_dispatch` (`:475`), but the standalone 3-tier feature-detect at `:503-510` still duplicates raw `is_x86_feature_detected!` rather than delegating tier selection to the polyfill. |

**Totals:** 5 crates, 179 raw-intrinsic-bearing lines currently (holograph 25 +
lance-graph/blasgraph 66 + bgz17 23 + lance-graph-contract 64 + thinking-engine
1). The knowledge doc's cited "158-violation finding" (`E-SIMD-SWEEP-1`,
2026-05-16) is **stale as a live count** — files have grown since (mul.rs in
particular carries many more batch functions than PR #398's original scope).
The one genuine reduction since 2026-05-16 is `blasgraph/types.rs`, which went
from its cited 22 raw ops to 0 by routing through the already-shipped
`ndarray::hpc::bitwise::hamming_distance_raw` — but its sibling file in the
same TD entry, `ndarray_bridge.rs`, still carries all 60+ of its original
violations, so the entry as a whole is half-true, half-stale.

**Net W1b verdict: 0 of 6 consumer-migration TD entries (W1–W5) are closed.**
All 5 primitives they depend on are shipped and ready; none of the 6 files
(types.rs excepted, already independently migrated by a different path) has
been switched over.

## W1.5 items

| # | TD id | Doc status (`ndarray-vertical-simd-alien-magic.md`) | TD-status (verbatim, `TECH_DEBT.md`) | Code reality | Verdict |
|---|---|---|---|---|
| 6 | TD-NDARRAY-SIMD-SIGNATURE-PDE-SWEEP | "SHIPPED" | heading itself says "(W1.5-#6, SHIPPED)"; body: "~~Deferred~~ **SHIPPED 2026-09-02.**" | `ndarray/src/hpc/signature_pde.rs:91` — `pub fn signature_pde_sweep(x: &[Vec<f64>], y: &[Vec<f64>]) -> f64` exists; `sigker/src/kernel.rs:35` imports it (per the doc; not independently re-verified against sigker source in this pass) | **CONFIRMED SHIPPED, TD entry consistent with code.** Lane type is `f64`/`Vec<f64>` (via internal `F64x8`), NOT the `F32x16` originally sketched — both docs already carry this correction. |
| 7 | TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION | Knowledge doc: "SHIPPED (ndarray PR #294)" — `randomized_signature_sweep`/`_sweep_with`/`_step` + `INCREMENT_EPSILON` | **`TECH_DEBT.md:1841` heading: "(W1.5-#7, DEFERRED)"; body `Status: Deferred`** | `ndarray/src/hpc/randomized_signature.rs:244,292` — `randomized_signature_sweep_with<F>` and `pub fn randomized_signature_sweep(path: &[Vec<f64>], matrices: &[f64], biases: &[f64], state_dim: usize) -> Vec<f64>` both exist and compile-checkable by inspection | **CONFIRMED STALE-DOC — exactly the orchestrator's suspicion.** The primitive is shipped in ndarray; `TECH_DEBT.md`'s #7 entry was never updated off "Deferred" even though the knowledge doc (same repo, `.claude/knowledge/ndarray-vertical-simd-alien-magic.md:96-99`) already says "gate open... measured TRUE (2026-09-02)" and the sibling ndarray-side contract doc (`vertical-simd-consumer-contract.md:279-300`) carries a full "⊘ CORRECTED — SHIPPED (ndarray PR #294)" block. Consumer wiring (`sigker/src/randomized.rs:95` `RandomizedSignatureBuilder::encode`) is separately noted in that doc as "NOT yet wired... treat as in-flight" — not independently re-verified here (would require reading sigker source, out of this task's scope). |
| 8 | TD-NDARRAY-SIMD-LYNDON-PACK | Knowledge doc: gate open, "still unbuilt, but NO LONGER GATED" | `TECH_DEBT.md:1854` heading "(W1.5-#8, DEFERRED)"; body `Status: Deferred` | No `lyndon_pack`/`lyndon_unpack_batch`/`I16x16::lyndon_pack` symbol anywhere in `ndarray/src` (0 grep hits) | **CONSISTENT — genuinely open, not stale.** Both docs agree it's unbuilt; TD status "Deferred" matches (though the knowledge doc's own standing note says the gate is open, not blocked — "Deferred" as a TD status word is arguably imprecise but not factually wrong about implementation state). |

**Cross-check against orchestrator's stated priors:** #6 and #7 (PR #293, #294)
ARE merged as claimed — confirmed directly against `ndarray/src/hpc/{signature_pde,randomized_signature}.rs`. #8's scalar prerequisite claim (`sigker::log_signature` exists via lance-graph PR #1150) was **not independently re-verified** in this pass (would require reading `crates/sigker/src/log_signature.rs`, which was out of the grep/read budget spent on the SIMD-primitive side) — flagged in Uncertain. The orchestrator's suspicion that TD's #7 entry is stale-DEFERRED is **CONFIRMED**.

---

## Stale-doc findings

1. **All five W1a `TECH_DEBT.md` entries say "Status: Open"** (`:1762,1776,1790,1803,1817`) while every symbol each specifies is shipped in `ndarray/src/{simd_scalar,simd_avx512,simd_neon}.rs` (line citations in the W1a table above). Not one W1a entry has been updated to "Shipped."
2. **TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION (`TECH_DEBT.md:1841-1850`) says "DEFERRED"** while `ndarray/src/hpc/randomized_signature.rs:244,292` ships `randomized_signature_sweep_with`/`randomized_signature_sweep`, and the workspace's own knowledge doc (`ndarray-vertical-simd-alien-magic.md:96-99`) and the ndarray-side contract doc (`vertical-simd-consumer-contract.md:283-300`) both already record it SHIPPED as of ndarray PR #294. This is a genuine three-way inconsistency: two docs say shipped, one says deferred.
3. **TD-SIMD-SWEEP-W2 (`TECH_DEBT.md:1877-1884`) describes `blasgraph/types.rs` as carrying "22 raw ops"** at "types.rs:440-506" — the file at that path today has **zero** raw-intrinsic lines; `hamming_distance_dispatch` (`types.rs:452-467`) already delegates to `ndarray::hpc::bitwise::hamming_distance_raw`. The entry's other half (`ndarray_bridge.rs`, 66 raw lines including the exact `_mm512_popcnt_epi64` call the shipped `U64x8::popcnt` primitive was built to replace) is still accurate and still Open — so the entry is a stale compound: half done, filed as if neither half were.
4. **The "158-violation finding" cited by the knowledge doc (`ndarray-vertical-simd-alien-magic.md:123`, `E-SIMD-SWEEP-1`) is stale as a live number** — current census across the same 5 crates is 179 lines (see W1b table). This is drift from file growth, not from remediation; only 1 of 6 files (types.rs) shows a real reduction.

---

## jc pillars table

Source: `crates/jc/src/lib.rs` (read in full, 200 lines) + per-module grep for
`PillarResult::deferred` / `pub fn prove`.

| # (as labeled in module doc, `lib.rs:1-29`) | Name | Module | Registered in `run_all_pillars` (`lib.rs:137-183`)? | `prove()` behavior |
|---|---|---|---|---|
| 1 | E-SUBSTRATE-1: bundle associativity @ d=10000 | `substrate.rs` | yes | executes (`substrate.rs:50`) |
| 2 | Cartan-Kuranishi: role_keys ≡ Cartan characters | `cartan.rs` | yes | **`PillarResult::deferred(...)` unconditionally** (`cartan.rs:15-16`, 22-line file total) |
| 3 | φ-Weyl: 144-verb collocation coverage | `weyl.rs` | yes | executes (`weyl.rs:35`) |
| 4 | γ+φ preconditioner: prolongation step reduction | `precond.rs` | yes | executes (`precond.rs:141`) |
| 5 | Jirak Berry-Esseen: weak-dep noise floor @ d=16384 | `jirak.rs` | yes | executes (`jirak.rs:128`) |
| 5b | Pearl 2³ mask-accuracy | `pearl.rs` | yes | executes (`pearl.rs:118`) |
| (no 6) | — | — | — | no module/label numbered "6" exists anywhere in `lib.rs`'s doc header or registry — the sequence explicitly jumps 5→5b→7 |
| 7 | Köstenberger-Stark: inductive mean on Hadamard 2×2 SPD | `koestenberger.rs` | yes | executes (`koestenberger.rs:248`) |
| 8 | Düker-Zoubouloglou: Hilbert-space CLT for AR(1) | `dueker_zoubouloglou.rs` | yes | executes (`dueker_zoubouloglou.rs:181`) |
| 9 | EWA-Sandwich: Σ-push-forward along multi-hop edge paths | `ewa_sandwich.rs` | yes | executes (`ewa_sandwich.rs:271`) |
| 9b | EWA-Sandwich 3D | `ewa_sandwich_3d.rs` | yes | executes (`ewa_sandwich_3d.rs:471`) |
| 10 | Pflug-Pichler: nested-distance Lipschitz on Sigma DN-trees | `pflug.rs` | yes | executes (`pflug.rs:290`) |
| 11 | Hambly-Lyons: signature uniqueness on tree-quotient | `hambly_lyons.rs` | yes | **feature-conditional**: `active::prove()` (`:575`, real math) under `--features hambly-lyons`; **`PillarResult::deferred(...)` by default** (`:737-745`, since `default = []` in `Cargo.toml:26` and `hambly-lyons = ["dep:sigker"]` is opt-in) |

**"11/12 implemented, Pillar 2 deferred" claim: CONFIRMED as literally stated**
in `lib.rs:24-29` ("Pillars 1, 3, 4, 5, 5b, 7-11 are immediately executable
... Pillar 2 (Cartan-Kuranishi) remains deferred"). Counting the 12 vector
entries against that sentence: 11 named as executable (1,3,4,5,5b,7,8,9,9b,10,11
= 11 items) + 1 deferred (2) = 12. Arithmetic checks out.

**One caveat the header doesn't spell out:** "immediately executable" for
Pillar 11 is true only when the crate is built `--features hambly-lyons`.
Under a **plain default-feature build** (`cargo run --example prove_it` with
no `--features`), `hambly_lyons::prove()` also returns a `deferred` result
(`hambly_lyons.rs:738-745`, same shape as `cartan::prove()`) — so a default
`prove_it` run shows **10 executing + 2 deferred**, not "11 executing + 1
deferred," even though the code for Pillar 11 genuinely exists and is not a
stub (unlike Pillar 2, which has no non-deferred path at all — `cartan.rs` is
22 lines total, all comment + the deferred call).

`solver_order.rs` (`crates/jc/src/solver_order.rs`) exists as an explicit
**NEW, separate slot** — its own module doc (`:1-10`) states it is
deliberately NOT "another Pillar 11" (two batteries already use that number
across repos: this crate's `hambly_lyons` = uniqueness, ndarray's
`hpc::pillar::signature` = kernel stability). It is **not** in the
`run_all_pillars()` vector (`lib.rs:137-183` has no `solver_order::prove`
entry) — so it is a 13th battery, outside the 12-pillar registry, run only via
its own example (`w3_battery_sweep.rs`) or directly. Its own `prove()`
(`solver_order.rs:226`) is also feature-gated identically to Pillar 11
(`--features hambly-lyons`; deferred otherwise, `:299-306`).

---

## jc W2–W5 legs

**Numbering note (verified, not conflated with the SIMD W1a/W1b/W1.5 waves
above):** these W-numbers belong to a *different* plan,
`.claude/plans/pillar11-signature-certification-unification-v1.md` (read in
full). Its own waves are W0–W5; **W1 and W4 do NOT live in `crates/jc` at
all** — both are homed in `ndarray/crates/sigker-parity` (an excluded crate),
per the plan's §7 execution-record table and confirmed by an empty grep for
`\bW4\b` anywhere under `crates/jc` in lance-graph. Only W2, W3, W5 (and the
W6 "Theorem 5 lattice leg," an addition beyond the original 0-5 plan) have
in-tree jc artifacts.

| Leg | Certifies | Declared status (plan `pillar11-signature-certification-unification-v1.md` §7, verbatim numbers) | Home | Gate / trigger, quoted |
|---|---|---|---|---|
| W2 | Depth-∞ uniqueness leg (M-2): the Goursat PDE kernel's deviation from the constant/identity path, both forward (tree-equivalence) and converse (non-tree loops) | "Shipped — converse law \|dev/area²−2\| = 0.0077; edge measured at area 2.5e-3, boundary at 2.5e-4" | `jc::hambly_lyons` (`hambly_lyons.rs:201-235`, `active::depth_infinity_leg`), pre-registered via `examples/w2_refinement_sweep.rs` + `examples/w2_area_edge.rs` | Pass conditions in code (`hambly_lyons.rs:616-619`): `pde.forward_max < PDE_FORWARD_EPS` (5.0e-5) `&& pde.area_law_err < PDE_AREA_LAW_TOL` (0.05) `&& pde.edge_dev > PDE_EDGE_MIN_DEV` (1.0e-5) `&& pde.below_edge_dev < PDE_FLOOR` (1.0e-6). Feature-gated `hambly-lyons`. |
| W3 | Solver-order advantage (M-1) + carrier fidelity (M-4): level-2-area-augmented Goursat coefficients vs increment-only, plus the area-domain-vs-kernel-scalar cancellation trap | "Shipped — 29.90x/41.71x advantage, silence exactly 0, kernel-scalar trap demonstrated" | **New pillar slot**, `jc::solver_order` (not "Pillar 11" — see above), `examples/w3_battery_sweep.rs` | Pass condition (`solver_order.rs:237`): `adv1 > 10.0 && adv2 > 10.0 && silence < 1e-12 && area_monotone && trap_fires`. Feature-gated `hambly-lyons`. |
| W5 | PowerSig scalability (M-5) — long-path Goursat solves | "**DEFERRED, trigger measured not fired** — fires at path length ~11585 (memory half); longest in-tree 4609" (plan §7); STATUS_BOARD row identical | No code module yet — only the trigger-check example, `examples/w5_trigger_check.rs` | **Quoted trigger criterion** (`w5_trigger_check.rs:3-7`): "the first real stream whose Goursat solve exceeds 1 GiB or 10 s." Constants in code: `const GIB: f64 = (1u64 << 30) as f64;` `const TIME_TRIGGER_S: f64 = 10.0;`. Memory half computed exactly: `mem_len = (GIB / 8.0).sqrt()` ≈ **11,585** (=√(2^27)). Time half is measured at runtime on the executing machine (not statically knowable from source) — the plan's §7 execution record reports it was measured at **~33,388** on a prior run. `longest_in_tree = jc::hambly_lyons::LONGEST_PATH_POINTS.max(jc::solver_order::LONGEST_PATH_POINTS)` = max(4609, 2049) = **4609** (both constants read directly: `hambly_lyons.rs:211` `pub const LONGEST_PATH_POINTS: usize = 3 * PER_SEG + 1` with `PER_SEG = 1536` → 4609; `solver_order.rs:54` `pub const LONGEST_PATH_POINTS: usize = M + 1` with `M = 2048` → 2049). **Gate state: OPEN (not fired)** — 4609 < 11,585, so even the cheaper (memory) half of the trigger has not been reached; not independently re-run in this pass (would require `cargo run --release ... --example w5_trigger_check`, out of scope for a read-only agent), value taken from the plan doc's own execution-record row, which is itself a prior session's measurement, not a live one from this pass. |
| W6 | "Theorem 5 lattice leg" — the finite-depth Hambly-Lyons certificate (Annals 171(1) 2010 §2.4 Theorem 5, exhaustive lattice-word check) | Not in the original 0-5 plan; appears only inside `hambly_lyons.rs`'s own module doc as an addition ("W6 leg", "the finite-depth certificate") | `jc::hambly_lyons::active::lattice_leg` (`hambly_lyons.rs:363-573`) | Pass condition (`:623-628`): `lat.reduced_merged == 0 && lat.treelike_max_dist < LATTICE_EPS (1e-12) && lat.depth2_false_merges >= 1 && lat.false_merge_unresolved == 0 && lat.false_merge_max_sep_depth <= hl_theorem2_depth(8) && lat.d1_classes == 7`. |

**W1 and W4** (for completeness, not in-tree here): plan §7 records W1 "Shipped
— level-normalized err 3.518e-6, margin 28.4x" and W4 "Shipped — Cholesky over
64 paths + indefinite counterexample; concentration 0.0038 at N=1000," both
homed in `ndarray/crates/sigker-parity` — **not verified against ndarray
source in this pass** (out of the assigned scope, which was lance-graph's jc
crate + ndarray's SIMD primitives specifically).

**W0** — hygiene wave (name-collision/misnomer/stale-note fixes across both
repos), plan §7: "Shipped (lance-graph #1111, ndarray #289)."

---

## jc examples

Full `[[example]]` list from `crates/jc/Cargo.toml` (read in full), with
`required-features`:

| Example | `required-features` |
|---|---|
| `goursat_substrate_probe` | `["hambly-lyons"]` |
| `prove_it` | none |
| `sigma_probe` | none |
| `probe_p1` | none |
| `osint_edge_traversal` | none |
| `splat_to_ewa_bridge` | none |
| `splat_triangle_count` | none |
| `splat_lpa_label_propagation` | none |
| `splat_louvain_modularity` | none |
| `splat_jaccard_adamic_adar` | none |
| `splat_perturbationslernen` | none |
| `ontology_locality_probe` | none |
| `style_table_agreement` | none |
| `substrate_compare` | none |
| `w2_refinement_sweep` | `["hambly-lyons"]` |
| `w5_trigger_check` | `["hambly-lyons"]` |
| `w3_battery_sweep` | `["hambly-lyons"]` |
| `w2_area_edge` | `["hambly-lyons"]` |

**Files present in `crates/jc/examples/` but with NO corresponding
`[[example]]` Cargo.toml entry** (Cargo auto-discovers `examples/*.rs` by
default since `autoexamples` was not seen disabled anywhere in the
`[package]` table read): `l9_loci_real_text.rs`, `partof_isa_vs_palette256.rs`,
`rung_divergence_reliability.rs`, `weather_substrate_reliability.rs`. These
build with default features (no `required-features` gate) since Cargo's
auto-discovery carries no such attribute; not independently confirmed to
compile in this pass (no cargo run performed).

---

## Board rows

`STATUS_BOARD.md` § `pillar11-signature-certification-unification-v1`
(`:205-215`) — matches the plan doc's §7 execution record **exactly**,
including the W5 "DEFERRED, trigger measured not fired... fires at path
length ~11585... longest in-tree 4609" line verbatim. No drift found between
board and plan for this section.

Other `STATUS_BOARD.md` rows naming jc/Pillar/sigker (not all independently
verified — listed for completeness per the task's grep instruction):
- `D-OIF-4` — `ewa_sandwich_x8` in `crates/jc` over `ndarray::simd::F64x8`, status "Queued."
- `D-NXG-11` — "**Blocked** on Pillar-6 σ_step calibration" — note this
  references "Pillar-6," which (per the jc pillars-table finding above) does
  not exist as a jc-registry pillar; this is presumably a different
  numbering scheme (a different plan's own pillar count, e.g.
  `weather-substrate-poc-v2.md` or `dialectic-engine-v1.md`'s "six operator
  pillars," line `:734`) — **not resolved in this pass**, flagged in
  Uncertain.
- `D-MEP-0` — "jc Pillar-6/7 provers green in-checkout" — same "Pillar-6"
  naming, status Queued.
- `TD-JC-CLIPPY-RED-ON-BASE-1` (`TECH_DEBT.md:50`) — heading itself says
  "RESOLVED 2026-09-05 (lint sweep + `jc-proof.yml` clippy step,
  operator-directed after #1181)" — consistent with `AGENT_LOG.md`'s
  2026-09-05 entry describing the same resolution.
- CI: `STATUS_BOARD.md:1160-1164`, `CI-JC` row — `.github/workflows/jc-proof.yml`
  runs `prove_it` on every PR touching `crates/jc/` or `cam.rs`, "In PR"
  status, 5-min timeout — not independently verified against the actual
  workflow file in this pass.

Plans referencing PowerSig/hambly/Pillar-11 beyond the pillar11 plan itself
(found via `grep -rl`, contents not read in this pass):
`thinking-engine-harvest-closure-v1.md`, `h268-probe-wave-v1.md`,
`dismech-causal-replay-v1.md`, `weather-substrate-poc-v2.md`,
`temporal-markov-and-style-classes-v1.md`.

---

## Candidate epiphanies

Facts established this pass, each with citations, suitable for the
orchestrator to promote into a board entry if it chooses:

1. **All five W1a `TD-NDARRAY-SIMD-*` entries in `lance-graph/.claude/board/TECH_DEBT.md` are stale "Open" labels on shipped work.** Every primitive named in TD-NDARRAY-SIMD-UNPACK-I4-16D, -SATURATING-ABS-I8, -GATHER, -PREFETCH, -POPCOUNT-U64 exists in `ndarray/src/{simd_scalar,simd_avx512,simd_neon}.rs` (line numbers in the W1a table above). None of the five TD entries has had its Status line updated.
2. **TD-NDARRAY-SIMD-RANDOMIZED-PROJECTION (#7) is a three-way doc/TD inconsistency, confirmed.** ndarray's own knowledge doc and its consumer-contract doc both record it SHIPPED (ndarray PR #294, `randomized_signature_sweep`/`_sweep_with`/`_step` in `ndarray/src/hpc/randomized_signature.rs:244,292`), but `lance-graph/.claude/board/TECH_DEBT.md:1841-1850` still reads "Status: Deferred." This confirms the orchestrator's stated suspicion exactly.
3. **`gather_u16` (W1a-#3) ships everywhere as a scalar-loop correctness anchor, not as true SIMD gather.** Its own doc comment at `ndarray/src/simd_avx512.rs:2832-2836` states the real AVX2 `_mm256_i32gather_epi32` path "is tracked as a follow-up optimisation" — so the primitive satisfies the API contract (bounds-safe, three backends agree) but not the performance intent the W1a spec described.
4. **`blasgraph/types.rs` has already migrated off raw intrinsics independently of the tracked W1b wave**, via `hamming_distance_dispatch` → `ndarray::hpc::bitwise::hamming_distance_raw` (`types.rs:452-467`), while its sibling `ndarray_bridge.rs` (same TD-SIMD-SWEEP-W2 entry) still carries the exact `_mm512_popcnt_epi64` call the shipped `U64x8::popcnt` primitive exists to replace. The TD entry conflates the two files' states into one still-Open line.
5. **jc's "11/12 implemented" header claim is accurate as literally written, with one unstated caveat:** Pillar 11 (`hambly_lyons`) is only substantively executable under `--features hambly-lyons`; under the crate's own `default = []` feature set it returns the same `PillarResult::deferred(...)` shape as Pillar 2. A plain `cargo run --example prove_it` therefore shows 10 executing pillars, not 11, unless the feature flag is passed.
6. **The jc pillar numbering has no "Pillar 6"** — the module doc's own numbered list (`lib.rs:1-29`) runs 1,2,3,4,5,5b,7,8,9,9b,10,11, and the 12-entry `run_all_pillars()` vector matches that sequence exactly (no gap, no phantom entry). Any reference elsewhere in the board to "Pillar-6" (`D-NXG-11`, `D-MEP-0`) refers to a different plan's own pillar numbering, not the jc crate's registry — a naming collision across documents, unresolved in this pass.
7. **The pillar11-signature-certification-unification-v1 plan's W-numbering is disjoint from the SIMD wave's W1a/W1b/W1.5 numbering** — both use "W1"/"W2"/etc. as labels but for entirely different, unrelated work streams. A session searching "W2" or "W4" without first identifying which plan is meant will find real but unrelated matches (confirmed here: grepping `crates/jc` for `\bW4\b` returns nothing, because that wave lives in `ndarray/crates/sigker-parity`, not lance-graph).
8. **W5's PowerSig trigger is measurably far from firing.** `longest_in_tree` (4609, computed from the two exported `LONGEST_PATH_POINTS` constants) is well under a quarter of the memory-half trigger length (~11,585, computed exactly as `√(2^30/8)`), which is itself the *cheaper* of the two trigger halves per the plan's own prior measurement (time half ~33,388). W5 stays legitimately deferred by its own pre-registered, falsifiable criterion.

---

## Uncertain

- **Sigker source not independently read.** Every claim about `sigker::log_signature` existing (orchestrator's premise for #8's "prerequisite"), `sigker/src/kernel.rs:35` importing `signature_pde_sweep`, and `sigker/src/randomized.rs:95` still running its own scalar loop, is taken from the ndarray-side knowledge docs, not verified by reading `crates/sigker/src/*.rs` directly in this pass. If precision on sigker's own state is needed, a follow-up read of that crate is required.
- **W5's `time_len ≈ 33,388` figure is not a live measurement from this session** — it is quoted from the plan doc's own §7 execution-record row, itself the record of a prior autoattended run (2026-08-31). This agent did not run `cargo run --release ... --example w5_trigger_check` (read-only, no cargo). The memory-half figure (~11,585) IS independently re-derivable from the static constants in `w5_trigger_check.rs` without running anything, and was recomputed here rather than merely copied.
- **"Pillar-6" cross-reference** (`STATUS_BOARD.md` `D-NXG-11`, `D-MEP-0`) is unresolved — could not determine in this pass which document actually owns a "Pillar 6" (candidates not opened: `weather-substrate-poc-v2.md`, `dialectic-engine-v1.md`'s "six operator pillars").
- **The four un-registered jc example files** (`l9_loci_real_text.rs`, `partof_isa_vs_palette256.rs`, `rung_divergence_reliability.rs`, `weather_substrate_reliability.rs`) were confirmed present by `ls` and confirmed absent from `Cargo.toml`'s `[[example]]` list by full read, but were not opened themselves — unknown whether they compile cleanly under default features, and unknown whether Cargo's implicit auto-discovery (rather than an explicit entry) is intentional or an oversight.
- **`neural-debug`, `lance-graph-planner`, and other workspace crates were not grepped for raw intrinsics** — the task scoped the raw-intrinsic census to the 5 crates named in the knowledge doc's per-workload table (holograph, lance-graph/blasgraph, bgz17, lance-graph-contract, thinking-engine); a wider sweep might turn up additional violations outside that named set.
- **The `is_x86_feature_detected!`/`is_aarch64_feature_detected!` grep pattern was included in the raw-intrinsic count** per the task's own symbol list; this inflates the "violation" count slightly relative to a strict "raw SIMD intrinsic call" definition (e.g., `bgz17/src/simd.rs`'s `detect_simd()` calls are architecturally a duplicate-dispatch anti-pattern, AP-SIMD-8, rather than a raw-instruction call per se — both are real violations per the workspace's own `simd-savant` doctrine, just different flavors).
