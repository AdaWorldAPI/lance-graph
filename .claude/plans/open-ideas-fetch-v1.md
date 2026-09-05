# Open-ideas fetch — three stale cards re-derived from the tree (v1)

> **Status:** MEASURED / ready-to-execute — PLANNING ONLY in this PR. Every
> claim below carries file:line from a read of the CURRENT tree (origin/main
> `ac9148f7`, #1171, plus the FF to `afeb0458`). NOT COMPILED, NOT RUN — no
> `cargo` was invoked; every "exists / is wired / has callers" statement is
> structural (grep + read), and §8 names the ones that must be confirmed by
> execution before they are believed.
>
> **READ BY:** the orchestrator that dispatches the Sonnet workers (§7 carries
> the per-D briefs), anyone touching `lance-graph-callcenter::policy`,
> `sigma_propagation`, the `MailboxSoA.sigma` column, or `crates/jc`.
>
> **Origin:** `IDEAS.md` had gone 82 days without an entry (newest 2026-06-15)
> while ~900 PRs merged. Three of its Open cards were picked as the highest
> value; **each one changed shape on contact with the tree**, which is the
> reason this plan exists rather than three direct PRs.
>
> **Confidence:** HIGH on the census; LOW-to-none on anything marked
> DECISION — those are the operator's, stated with a recommendation and
> deliberately not taken.

## 0. The three headline corrections

| card (IDEAS.md) | what the card says | what the tree says |
|---|---|---|
| `IDEA-POLICY-HASH-UDF` (`:1162`) | "policy_hash_v1 UDF **registration**" | Registration is not the blocker. The UDF is bound as an `Arc<ScalarUDF>` **object inside the `Expr`** (`policy.rs:137`), and DataFusion executes the embedded object without any by-name lookup — proof: `register_vsa_udfs` (`vsa_udfs.rs:574`) has **zero callers** in the tree and its UDFs still execute in tests. The blocker is the **body**: `invoke_with_args` returns `NotImplemented` (`policy.rs:~330`). |
| `IDEA-B1-HARDWARE-BACKENDS` (`:1062`) | "AMX/MKL backends for sigma_propagation, waits on ndarray #119/#121" | The kernel is a **2×2 f64** sandwich — three scalars, 12 mul + 6 add (`sigma_propagation.rs:210-227`). An AMX tile is 16×16 bf16/int8; an MKL `dgemm` call costs more than the 18 flops it would do. The idea is **mis-shaped for its own kernel**. The real lever is vertical batching — `F64x8` (8 edges/instruction), whose `Mul`/`Add` already exist on every backend. AND: `ewa_sandwich(` has **zero production call expressions** outside `jc` + the contract (grep, §3.2) — every "caller" is a doc-comment mention. |
| `IDEA-CAUSAL-EDGE-TENSOR-SIDECAR` (`:1126`) | "design a 9-byte sidecar (`CausalEdge64` + 1 byte Σ index) OR Block 14/15" | **The 1-byte index already shipped, as a SoA column** — `BindSpace.fingerprints.sigma: Box<[u8]>` (`bindspace.rs:54-58`), `MailboxSoA.sigma: [u8; N]` (`mailbox_soa.rs:125-133`), `set_sigma` (`:670`), `BackingStoreWrite::set_sigma` both arms (`backing.rs:253-257`). Neither the sidecar nor Block 14/15 is live. **What does NOT exist is the codebook the index points at** — §4. |

## 1. Frozen decisions this plan inherits (cited, not re-derived)

- **Zero-dep contract.** `lance-graph-contract/Cargo.toml` `[dependencies]` is empty by design and says why (a path dep there broke every CI invocation, 2026-07-07). Nothing in this plan adds a dependency to it.
- **All SIMD from `ndarray::simd`** (`simd-savant`; W1a consumer contract, ndarray `.claude/knowledge/vertical-simd-consumer-contract.md`). Consumer code composes typed wrappers; it never writes intrinsics.
- **Bit identity between the jc kernel and the contract copy is CI law** — `jc::ewa_sandwich::tests::the_contract_copy_matches_the_certified_kernel_bit_for_bit` (`ewa_sandwich.rs:~425`, 1000 sampled SPD pairs, `to_bits()`, no tolerance; `jc-proof.yml`). Any batched kernel joins that law or does not land.
- **Internal head pins are absolutely and permanently prohibited** (MedCare-rs `CLAUDE.md`, operator 2026-08-30). A frozen codebook artifact gets a **tag + content invariants**, never a digest gate.
- **Indices, not content** (`I-VSA-IDENTITIES`; `mailbox_soa.rs:128-131`): the codebook stays shared/cold; rows carry the 1-byte reference.
- **The falsifiability rule** (`CLAUDE.md`): every guard needs a can-fire AND a can-stay-silent test on non-trivial inputs; a threshold needs an inertness test; a doc claim is not a behaviour.
- **Sonnet workers: edit-only, no `cargo`, no `git`, disjoint files; the orchestrator compiles once** (`agent-cargo-hygiene.md`; `sonnet-worker-guardrails.md` §1 pasted verbatim into every brief).

## 2. D-OIF-1 — `policy_hash_v1`: give the UDF a body

### 2.1 Evidence

- `crates/lance-graph-callcenter/src/policy.rs:120-140` — `mask_expr` binds `NotYetWiredHashUdf::new()` as an `Arc<ScalarUDF>` inside a `ScalarFunction` expr.
- `:262-335` — the impl: `Signature::any(1, Volatility::Immutable)`, `return_type = UInt64` (already fixed so downstream schemas are stable), `invoke_with_args → Err(NotImplemented("policy_hash_v1 UDF not yet registered — see PR-F1b"))`. The header comment names **FNV-64 as the v1 target**.
- Feature gate: `auth-rls-lite = ["auth-jwt", "query-lite"]`, `query-lite = ["dep:datafusion", "dep:arrow"]` (`Cargo.toml` `[features]`). `datafusion` WITHOUT default features.
- Existing test `redaction_mode_hash_binds_not_yet_wired_udf` (`:598-620`) asserts **plan text only** (`contains("policy_hash_v1")`, `!contains("***REDACTED***")`). It stays green after the fix and is therefore **not** a falsifier of the body. New tests must EXECUTE the plan.
- Execution pattern in this crate: `graph_table.rs:196-229` — `#[tokio::test]`, `MemTable::try_new`, `ctx.register_table`, `df.collect().await`, under `#[cfg(feature = "query")]`. Policy tests today build plans against an **empty** `MemTable` (`policy.rs:530`). `tokio` is already a dev-dep with `rt-multi-thread` + `macros`.
- Array-vs-Scalar handling to copy: `vsa_udfs.rs:88-89` (unwrap) and `:201-204` (length resolution).
- Live docs naming the type: `.claude/patterns.md:89` (`NotYetWiredHashUdf`) — a live inventory, **update in the same commit**. `PR_ARC_INVENTORY.md:5734` and `LATEST_STATE.md:2753` are append-only history — leave.

### 2.2 DECISION D-OIF-1-DEC — hash family and key (operator's; recommendation stated)

| option | what | verdict |
|---|---|---|
| A — FNV-1a-64, unkeyed | what PR #301 named as "v1 target" | **Do not ship as the default.** A masked column is by definition an identifier; unkeyed 64-bit hashing of a low-entropy identifier is reversible by enumeration. It would *look* redacted in every plan and be a lookup table in practice. |
| B — **keyed SipHash-1-3, 128-bit key, in-crate** | ~60 LOC, zero new dependency, deterministic across Rust versions (std's `DefaultHasher` is SipHash-1-3 but **explicitly not stable across releases** and unkeyed — unusable for a persisted pseudonym) | **Recommended.** Same input + same key → same hash (joins still work); different key → different pseudonym space (a deployment's masked values are useless elsewhere). |
| C — SHA-256 truncated to 64 | needs `sha2` (crates.io, no fork) | Acceptable, one more dep for no property B lacks at this width. |

**Key source (part of the decision):** the key must be **bound at rewriter construction**, never read from the environment inside the UDF (the UDF is `Immutable` and must be a pure function of its inputs). Proposed shape: `ColumnMaskRewriter { registry, actor_role, hash_key: [u8; 16] }`; `mask_expr` becomes a method and builds `PolicyHashUdf::new(self.hash_key)`. The **membrane** (the caller that already holds `actor_role`) supplies the key from its own config. Two rewriters, two keys, is the falsifier that proves the key is load-bearing.

### 2.3 Deliverable

- `PolicyHashUdf` (rename; `NotYetWiredHashUdf` is now a false name) in `policy.rs`, same `Signature`/`return_type`, `invoke_with_args` implemented over `ColumnarValue::{Array, Scalar}`:
  - NULL → NULL (Arrow null propagation), never the hash of an empty string.
  - Hashed byte view per type: `Utf8`/`LargeUtf8`/`Utf8View` (UTF-8 bytes), `Binary`/`LargeBinary`/`BinaryView` (bytes), `Boolean` (one byte), `Int8..Int64`/`UInt8..UInt64` (little-endian bytes of the widened `i64`/`u64` — so `Int32(5)` and `Int64(5)` hash equal; document it), `Float32`/`Float64` (IEEE bits of the `f64` widening; `-0.0 ≠ 0.0` — document it), `Date32/64`, `Timestamp(*)` (the underlying integer's bytes).
  - Any other type → `DataFusionError::NotImplemented` naming the type — **loud, and enumerated in the doc comment**, never a silent constant.
- `policy_hash_udf(key) -> Arc<ScalarUDF>` + `register_policy_udfs(ctx, key)` for SQL-text callers, mirroring `register_vsa_udfs` — a convenience, **not** the deliverable (see §0).
- `.claude/patterns.md:89` (`NotYetWiredHashUdf`) updated to the new name.

### 2.4 Pre-registered gates (each red-then-green; the disable is named)

| # | assertion | non-trivial input | disable that must turn it red |
|---|---|---|---|
| G1 | executing a Hash-masked scan over a `MemTable` with ≥3 real rows returns `UInt64`, non-null for non-null inputs | 3 distinct strings | restore `Err(NotImplemented)` |
| G2 | determinism: two separate `SessionContext`s + two rewriters with the SAME key give identical column values | same 3 strings | perturb the key in one rewriter |
| G3 | discrimination: 3 distinct inputs → 3 distinct hashes; and the two equal inputs among 4 → equal hashes | `["a","b","a","c"]` | return a constant |
| G4 | key is load-bearing: two rewriters with DIFFERENT keys give different values for the same input | same string | ignore the key in `invoke` |
| G5 | NULL → NULL; the null count of the output equals the null count of the input | a column with 2 nulls of 5 | hash `""` for nulls |
| G6 | an unsupported type fails LOUD at execute with the type in the message (can-fire); a supported type does NOT (stay-silent) | `List<Int32>` vs `Utf8` | make the fallback return `0` |
| G7 | the old `:598` test still passes unchanged (plan text) — proves the rewrite site did not move | as-is | n/a (regression) |
| G8 | `cargo clippy -p lance-graph-callcenter --features auth-rls-lite -- -D warnings` clean; `cargo test -p lance-graph-callcenter --features auth-rls-lite` green | — | — |

**§8 caveat feeding G1:** `DataFrame::collect()` under `query-lite` (datafusion without default features) is **assumed** to compile — the physical planner is core, `sql` is not needed. If it does not, the execution tests move under `#[cfg(feature = "auth-rls")]` (`= query`) and the plan says so; that is a scope note, not a failure.

## 3. D-OIF-4 / D-OIF-5 — Σ-propagation: the batched kernel and the hop probe

### 3.1 Why "hardware backends" is the wrong shape

- Kernel: `Spd2 { a, b, c }` f64 (`sigma_propagation.rs:113-117`); `ewa_sandwich` = 8 products for `P = M·Σ`, 8 for `R = P·Mᵀ`, then `b = 0.5·(r01 + r10)` (`:210-227`). Byte-identical copy in `jc/src/ewa_sandwich.rs:185-199`.
- AMX: `hpc/amx_matmul.rs` exposes 16×16 tiles (`tile_dpbusd`, `tile_dpbf16ps`, `:288-359`); `simd_amx.rs` exposes u8×i8 VNNI dot/matvec. Neither has a 2×2 f64 shape. MKL `dgemm` on 2×2 is dominated by call overhead. **Rejected for this kernel** — recorded, not deferred.
- The 3×3 analogue exists as a jc probe (`ewa_sandwich_3d.rs`, `Spd3`, `:378`); the n×n idea is `IDEA-PILLAR5PLUS-HIGHER-DIM-SPD`. AMX becomes plausible only at ≥16×16 in **bf16**, and bf16 rounding on the SPD cone can lose PSD — that is its own falsifier and **out of scope** here.

### 3.2 The real lever, and the real gap

- `F64x8` carries `Add`/`Mul` on **every** backend with no FMA in the path: AVX-512 `impl_bin_op!(F64x8, Mul, mul, _mm512_mul_pd)` (`simd_avx512.rs:446-448`), AVX2 (`simd_avx2.rs:973-987`), NEON (`simd_neon.rs:1019-1047`), scalar via `impl_float_type!(F64x8, f64, 8, …)` (`simd_scalar.rs:510`). `mul_add` exists (`:376`) and **must not be used** — it would break bit identity with the scalar kernel.
- So the batched kernel is ordinary consumer code over typed wrappers: 8 edges per lane-op, same 12-mul/6-add order per lane. **No ndarray change is needed; the STOP rule is not triggered.**
- **Home:** `crates/jc` (deps `ndarray`, `Cargo.toml:21`, "MANDATORY"). Not the contract (zero-dep). Name: `ewa_sandwich_x8(m: &[Spd2; 8], sigma: &[Spd2; 8]) -> [Spd2; 8]` plus a slice driver `ewa_sandwich_batch(m: &[Spd2], sigma: &[Spd2], out: &mut [Spd2])` handling the tail with the scalar kernel.
- **The gap that decides the sequencing:** `grep -rn "ewa_sandwich(" crates/` outside `jc/` and `sigma_propagation.rs` returns **nothing**. The four files that mention it — `cognitive-shader-driver/src/bindspace.rs:41`, `mailbox_soa.rs:129`, `perturbation-sim/src/splat.rs:19`, `lance-graph-arm-discovery/src/aerial/codebook.rs:16` — do so in **doc comments**. A faster kernel for a function nobody calls is `E-A-RULED-HOME-NEEDS-A-FIRST-CONSUMER-OR-IT-IS-A-VACANCY-1` verbatim. Therefore D-OIF-4 ships **only paired with D-OIF-5**, whose result decides whether the first consumer (§3.3) is sound.

### 3.3 The named first consumer (NOT built in this plan)

`MailboxSoA::apply_edges` (`mailbox_soa.rs:348-372`) receives `(row, CausalEdge64)` deliveries and today updates only `energy[row]` and `plasticity_counter[row]`; `sigma[row]` is never advanced. The contract's own planned use-site is "B4 shader-driver-sigma-propagate … propagate `sigma_path = ewa_sandwich(...)` along the resonance chain" (`sigma_propagation.rs:78-80`). With a codebook (§4), one hop is: `Σ' = sandwich(M, codebook[sigma[row]])`, then `sigma[row] = nearest(codebook, Σ')`. Two things block wiring it, and this plan does not pretend otherwise:

1. **Where `M` comes from.** A `CausalEdge64` carries no matrix; its v2 spare is **3 bits** (`edge.rs:194, 557` — bits 61-63), so it cannot carry a 256-entry index. Candidates: the *source* row's own `sigma` (M = codebook[σ_src]), or a per-edge-class M. **DECISION D-OIF-5-DEC, operator's.**
2. **Whether re-quantizing to k=256 after every hop keeps concentration.** That is D-OIF-5.

### 3.4 D-OIF-5 — the hop re-quantization probe (jc)

Pre-registered: for hop depth `n ∈ {1, 2, 4, 8, 16}`, over ≥1000 seeded paths, compare `log_norm_growth(seed, Σ_n)` (`sigma_propagation.rs:254`) for (i) exact propagation and (ii) propagation with nearest-codebook re-quantization after each hop, against `pillar_5plus_bound(n)` (`:274-285`) with the existing **1.75×** PASS slack (`:268-272`). **PASS** = re-quantized growth stays inside the slack at every `n` tested. **FAIL** at any `n` = σ-advance in `apply_edges` is unsound at that depth and the consumer is shelved with the number. First arm runs on the probe's own k-means codebook (`sigma_codebook_probe.rs`, `N_EDGES=10_000, K=256, 100 iters, SEED` fixed at `:49-52`) so it does not wait on D-OIF-2's artifact; second arm re-runs on the real codebook when it exists.

### 3.5 D-OIF-4 gates

| # | assertion | disable |
|---|---|---|
| G1 | `ewa_sandwich_x8` is **bit-identical** (`to_bits`) to the scalar contract `ewa_sandwich` on 1000 sampled SPD pairs incl. non-zero off-diagonals (the existing test's anti-vacuity guard) | replace one `*` + `+` with `mul_add` |
| G2 | the slice driver's tail path (len % 8 ≠ 0) is bit-identical too | drop the tail loop |
| G3 | a bench (`jc/examples`) prints ns/edge for scalar vs x8 at N = 65 536; the number is **pinned after measurement**, never predicted | — |
| G4 | `jc-proof.yml` still green (the certified kernel is untouched) | — |

## 4. D-OIF-0 / D-OIF-2 / D-OIF-3 — the Σ codebook that three files point at and none holds

### 4.1 The finding

Every σ byte in the substrate is an index into a codebook that **does not exist**:

- `sigma_propagation.rs:73` — "indexing into a 256-entry static `SigmaCodebook` of `Spd2`".
- `contract/src/splat.rs:308` — "The full SigmaCodebook lives in **lance-graph-cognitive**".
- `arm-discovery/src/aerial/codebook.rs:16-19` — "built and certified **offline by `crates/jc`**".
- `grep -rn SigmaCodebook crates/` → those two doc lines. `crates/lance-graph-cognitive/src` → zero hits. `jc` has the **viability probe** (`sigma_codebook_probe.rs`) and no builder/emitter.

Three claimed homes, zero implementations. The column's writers: `backing.rs:310` (write-shim loop) and one test (`:347`, writes `9`). Readers: two planner examples dumping the byte (`blw_tenant.rs:190`, `blw_rows.rs:289`). Every production row has `sigma = 0`, documented as "untrained / first centroid" (`bindspace.rs:54`) of a codebook with no centroids.

**Two numbers, one claim.** `bindspace.rs:39,57` cite "R²=0.9949 at k=256 (#288)". `arm-discovery/src/lib.rs:13` and `codebook.rs:18` cite "ρ=0.9973". The probe computes **R² in log-Euclidean space** (`sigma_codebook_probe.rs:27, 317`, PASS ≥ 0.99). `0.9973` appears elsewhere as the 3σ coverage constant and the ADC-cosine Spearman band (`probe_adc_cosine_head_to_head.rs:6`) — a different quantity. D-OIF-0 settles which number the codebook claim actually rests on.

### 4.2 Two provenance stories for σ — DECISION D-OIF-2-DEC (operator's)

| story | where it is written | what it implies |
|---|---|---|
| **Fitted** — σ = nearest of 256 k-means centroids over observed Σ | `sigma_codebook_probe.rs` (Lloyd's, affine-invariant Riemannian metric, R² gate) | needs a per-row observed Σ at write time; the codebook is trained once, offline, in jc (float to build, index to use — the CAM-PQ doctrine, `codebook.rs:19-22`) |
| **Declared** — σ = lookup from the typed value's `(PropertyKind, Marking, SemanticType)` + value range | `sigma_propagation.rs:74-77` ("B3 transcode-sigma-assignment") | needs a mapping table, not k-means; the "codebook" is authored, and the R² probe is irrelevant to it |

They are not compatible as the *same* v1. Recommendation: **fitted** for the entries, **declared** only as the assignment *fallback* when no Σ is observable at write time — but that is a design choice with consequences for every consumer, so it is stated and not taken.

### 4.3 D-OIF-0 — re-run the viability probe on the current tree (first, cheap, unconditional)

`cargo run -p jc --example sigma_probe` (the `[[example]] sigma_probe`, `jc/Cargo.toml`). Record R², the PASS verdict, and the recommendation string it prints (`sigma_codebook_probe.rs:342-360`) in `AGENT_LOG` + this plan's §9. Reconcile 0.9949 vs 0.9973 in one sentence with the source of each. ~seconds of compute; no decision depends on skipping it.

### 4.4 D-OIF-2 — the codebook TYPE (contract) + BUILDER/EMITTER (jc) + INSTANCE home

- **Type, in the contract (zero-dep, LE law):** `SigmaCodebook([Spd2; 256])`, `from_le_bytes(&[u8; 256·24])` / `to_le_bytes`, `entry(u8) -> &Spd2`, `nearest(&Spd2) -> u8` (affine-invariant metric via the existing `Spd2::{log_spd, sqrt, pow, eig}` `:130-176`), and a **boundary check** on load — every entry `is_spd(eps)` (`:187`), per the module's own contract that SPD is checked "at boundaries (codebook load, runtime gate)" (`:108-109`). 6 KiB.
- **Builder, in jc:** the k-means the probe already runs, refactored so the emitter writes the frozen table (float to build). Emits **LE bytes + a tag**. **No digest pin, no digest gate** — identity = `K == 256` ∧ all-SPD ∧ `det > 0` ∧ the tag. The operator law on internal pins is absolute.
- **Instance home:** the contract carries the TYPE; the loaded INSTANCE must live where files can be read and where `set_sigma` is called — `cognitive-shader-driver` (owner of `MailboxSoA.sigma`), **not** `lance-graph-cognitive` (which the splat proxy names but which holds nothing). Recommendation only; the crate that owns the column should own the loader.
- Gates: G1 `from_le_bytes(to_le_bytes(cb)) == cb` bit-exact; G2 a non-SPD entry is **refused** at load (can-fire) and a valid table is not (stay-silent); G3 `nearest` returns the entry's own index for each of the 256 entries (identity), and a perturbed entry returns its unperturbed neighbour (discrimination); G4 the emitted table from the probe's fixed SEED is byte-stable across two runs (determinism).

### 4.5 D-OIF-3 — the first writer of a non-zero σ (gated on D-OIF-2-DEC)

Whichever provenance the decision picks, the first writer is the ingest/write path that today calls `BackingStoreWrite::set_sigma` in its loop (`backing.rs:310`) with a constant. Falsifier: after ingesting a batch with ≥2 genuinely different Σ shapes, `sigma` holds ≥2 distinct non-zero values (can-fire) and a batch of identical shapes holds one (stay-silent). **Not built until the decision lands**; the plan carries both shapes' one-line specs so the brief is a paste.

## 5. What this plan REJECTS, with the reason on record

| item | verdict | why |
|---|---|---|
| AMX / MKL for the 2×2 sandwich | rejected | shape mismatch: 3 scalars vs 16×16 tiles; call overhead ≫ 18 flops (§3.1) |
| 9-byte `CausalEdge64 + u8` sidecar | superseded | the u8 landed as a SoA column (§0); a second projection of the same byte is `SECOND-PROJECTION` |
| SchemaSidecar Block 14/15 | superseded | same |
| Σ index inside the `u64` | impossible | v2 spare is 3 bits (`edge.rs:557`) |
| wiring σ-advance into `apply_edges` | deferred, gated | needs D-OIF-5 PASS **and** D-OIF-5-DEC (the M source) |
| bf16 ≥16×16 SPD on AMX | out of scope | own falsifier (PSD under bf16 rounding) — file it as an idea, do not build |
| a CI gate on IDEAS.md staleness | rejected | a ledger that is legitimately quiet would fail it — a gate that fires on quiet carries no information (`E-ANTI-EIGENVALUE-…`). A dated census line in `LATEST_STATE` instead. |

## 6. Sequencing

```
D-OIF-0  probe re-run (jc, seconds)            ── unconditional, first
D-OIF-1  policy_hash_v1 body                   ── independent; needs D-OIF-1-DEC (hash + key)
D-OIF-4  ewa_sandwich_x8 (jc)  ┐
D-OIF-5  hop re-quant probe   ┘ paired         ── D-OIF-5 arm 1 needs no artifact
D-OIF-2  SigmaCodebook type + builder + home   ── needs D-OIF-2-DEC (fitted vs declared)
D-OIF-3  first σ writer                        ── needs D-OIF-2 + D-OIF-2-DEC
D-OIF-6  IDEAS ledger: 4 entries + 3 flips     ── THIS PR
```

Three decisions stand between the plan and the workers: **D-OIF-1-DEC**, **D-OIF-2-DEC**, **D-OIF-5-DEC**. D-OIF-0, D-OIF-4, D-OIF-5 (arm 1) and D-OIF-6 need none.

## 7. Worker briefs (Sonnet, edit-only; the orchestrator compiles once)

Every brief starts with `sonnet-worker-guardrails.md` §1 **verbatim**, then: *Read `.claude/board/AGENT_LOG.md` first; do NOT write it — leave your record in your own tag-file. Do NOT run `cargo` or `git`. Do not claim it compiles or that tests pass — you did not run them. Your file scope is exactly the files named; another worker owns everything else.*

- **W-1 (D-OIF-1)** — files: `.claude/patterns.md:89` (`NotYetWiredHashUdf`), `crates/lance-graph-callcenter/src/policy.rs`. Spec: §2.3 + the decided hash/key from D-OIF-1-DEC. Tests: §2.4 G1-G7 as `#[tokio::test]` under `auth-rls-lite`, each with its disable named in a comment. STOP if `DataFrame::collect` does not resolve under `query-lite` — report, do not change features.
- **W-4 (D-OIF-4)** — files: `crates/jc/src/ewa_sandwich.rs` (append only below the existing kernel), `crates/jc/examples/ewa_sandwich_bench.rs` (new). Spec: §3.2; **no `mul_add`, ever**; same op order per lane as `:185-199`. Tests: §3.5 G1-G2 with the disable named. STOP if any test needs a tolerance — that is a bug, not a tolerance.
- **W-5 (D-OIF-5)** — files: `crates/jc/src/sigma_hop_requant_probe.rs` (new), `crates/jc/src/lib.rs` (one `pub mod` line + one pillar-table entry, `lib.rs:150-185` shape). Spec: §3.4. Output: the table `n → (exact growth, requantized growth, bound, PASS/FAIL)` printed by `prove`.
- **W-2 (D-OIF-2, after D-OIF-2-DEC)** — files: `crates/lance-graph-contract/src/sigma_propagation.rs` (type + loader + boundary check, zero deps), `crates/jc/src/sigma_codebook_probe.rs` (emitter arm). Spec: §4.4, gates G1-G4.
- **W-0 (D-OIF-0)** — no edits: the orchestrator runs `cargo run -p jc --example sigma_probe` itself and records the result (§4.3). Not a worker task.

Orchestrator gates after the fleet lands: `cargo fmt -p <crate>`, `cargo clippy -p <crate> --all-targets -- -D warnings` per touched crate (`-p`, never `--all` — `tesseract-rs`'s lesson), `cargo test -p <crate>` with the feature sets named above, plus the two board gates and `supersession_index.py` regenerated LAST.

## 8. Claims in this plan that are STRUCTURAL and must be confirmed by execution

1. `DataFrame::collect()` compiles under `query-lite` (datafusion sans default features). Fallback named in §2.4.
2. `F64x8` `Mul`/`Add` are non-FMA on **all four** backends — read from `impl_bin_op!` (AVX-512) and the operator impls (AVX2/NEON); the scalar macro body was not read line-by-line. G1 of §3.5 is what proves it.
3. `cargo run -p jc --example sigma_probe` runs on the default feature set (it is not `required-features`-gated in `Cargo.toml:44-56`; `goursat_substrate_probe` is).
4. The probe's k-means codebook is exposable to D-OIF-5 arm 1 without a public API change — if not, W-5 gets a one-line `pub(crate)` accessor in `sigma_codebook_probe.rs` and its scope grows by that file.

## 9. Results ledger (append as they land)

- D-OIF-0: _(not yet run)_
