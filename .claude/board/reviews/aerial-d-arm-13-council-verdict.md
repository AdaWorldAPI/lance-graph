# Council verdict — D-ARM-13 (Aerial+ Rust transcode)

> 3-savant brutal review convened 2026-05-30 on branch `claude/jolly-cori-clnf9`
> (the Stage-D ratification ensemble applied to the ARM *code*). Background
> Opus agents; two were write-denied at the tool level and returned their
> reviews inline (captured below); iron-rule-savant persisted its own file
> (`aerial-d-arm-13-iron-rule-savant.md`).

## Consolidated verdict: **LAND-with-revision**

| Savant | Verdict | Headline |
| --- | --- | --- |
| brutally-honest-tester | **HOLD pending fixes** (0 P0, 2 P1) | "loads through the same loader" is split-true; Jirak floor missing |
| iron-rule-savant | **LAND-with-revision** | code yields (doesn't violate) every iron rule; the Jirak *doc-comment* writes a cheque the code can't cash |
| dto-soa-savant | **LAND-with-revision** | no parallel store / no fifth column; contract-homing drift is the real risk |

**No P0. Code is sound** — independently re-verified by the tester: `cargo test`
35/35, `--no-default-features` 17/17, clippy `-D warnings` clean, `#![forbid(unsafe_code)]`,
all 9 modules registered, `exclude` standalone done right (AP3/AP4/AP6/AP7 clean).
Every revision below is **prose / honesty / tech-debt**, not logic.

## Converged action ledger (all applied this commit unless noted)

| # | Finding (who) | Severity | Fix | Done |
| --- | --- | --- | --- | --- |
| 1 | "loads through the *same* loader / byte-compatible" overstates — `lance_graph::parse_triples` **accepts** `implies`, but `ruff_spo_triplet::from_ndjson` **rejects** it (closed vocab) until D-ARM-SYN-1 (tester P1, dto-soa F2, iron-rule) | P1 | Downgrade to "shape-compatible" + precise loader caveat in `ndjson.rs`, `translator.rs`, `lib.rs`, `README.md` | ✅ |
| 2 | Jirak floor doc-comment claims an upstream gate that doesn't exist; D-ARM-7 Queued (iron-rule #1, tester P1) | P1 | Make `rule::passes` doc honest ("not implemented; MUST NOT wire to live SpoStore until D-ARM-7"); file **ISSUE ARM-JIRAK-FLOOR** | ✅ |
| 3 | Contract-homing drift — local `CandidateRule` disagrees with D-ARM-2 (`n:u32` vs `WindowMetadata`); "shape identical" promise already false (dto-soa F3) | P1 | Honest `rule.rs` module+struct docs; file **TD-ARM-CARRIER-FORK** w/ the `pub use`-when-D-ARM-2-lands path | ✅ |
| 4 | "bit-identical weights" overclaims portability (iron-rule secondary, tester P2) | P2 | Footnote: intra-platform reproducibility, not bitwise-portable determinism (`lib.rs`, `README`) | ✅ |
| 5 | `max_antecedent ≥ 2` recovery untested; reproducibility asserts bit-exact f32 (tester P2) | P2 | Logged in TD-ARM-CARRIER-FORK; add coverage when D-ARM-3 lands the real multi-antecedent path | ⏶ logged |
| — | **Refuted:** `fmt_f32` drift vs `serde_json` — tester ran the full [0,1] grid: **0 mismatches** (Rust `{}` already shortest-round-trip; `1`→`1.0` is exactly what it patches) | — | no action | ✅ refuted |

## Cross-agent agreements (high confidence)

- **D-ARM-SYN-1 is the one genuine blocker** for the "same ruff loader" story: `ruff_spo_triplet::Predicate` is closed and `from_str`/`from_ndjson` hard-reject `implies`. All three flagged it; correctly council-gated (not silently patched).
- **D-ARM-7 (Jirak) is a doctrinal — not compile — dependency.** Enforce D-ARM-7 *before* any D-ARM-5 that consumes this proposer (D-ARM-5 = where `(f,c)` first meets live `revision` + `SpoStore`).
- **The determinism firewall is structurally intact** (excluded, std-only, seeded, proposal-only output). `arm_to_nars`'s `c = m/(m+k)` round-trips into the canonical `TruthValue::revision` as evidence `w = m` exactly; `expectation` is byte-identical to `spo::truth::TruthValue::expectation` — no rival revision kernel (iron-rule, I-SUBSTRATE-MARKOV check).
- **`NarsTruth` is a transport handle, not a parallel truth store** — the proof is it carries *no* revision algebra (dto-soa F1).

## Adjacency (council consensus)

D-ARM-13 = the **D-ARM-9 Aerial leg pulled in-process** (supersedes the plan §14 Python-IPC deferral per user directive). Hardest tie: **D-ARM-7 (Jirak)**. Pre-stages **D-ARM-1/D-ARM-2** (contract carriers — re-export when they land). Partially lands **D-ARM-4** (`translator`, now ahead of its own spec). Feeds **D-ARM-5** (hypothesis test) and the **D-ARM-10/11** codegen gates. Spawns **D-ARM-SYN-1/2/3**. Realises **E-DISCOVERY-CODEGEN-BRACKET-1** and **E-INTERPRET-NOT-STORE-1** at the shape level — qualified: truth scale genuinely unified, triple shape unified-by-convention, contract home not yet unified.
