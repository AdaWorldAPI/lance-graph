# IRON_RULE_SAVANT review — D-ARM-13 (Aerial+ Rust transcode)

> **Subject:** `crates/lance-graph-arm-discovery/` on branch `claude/jolly-cori-clnf9`
> **Council question:** does this candidate violate an iron rule or doctrinal invariant — especially **I-NOISE-FLOOR-JIRAK** by claiming significance the n-bound doesn't support?
> **Reviewer lens:** substrate-veto angle. Any VIOLATES is an automatic REJECT for the council.
> **Reviewed:** 2026-05-30. Read-only doctrine pass; no cargo, no edits.

---

## Verdict: **LAND-with-revision**

No iron rule is *violated* by the code as written. The crate is honest about what it is (an upstream, seeded, feature-gated *proposer*) and it does **not** make any statistical-significance claim in code — so I-NOISE-FLOOR-JIRAK has nothing to bite on yet. But the crate ships a **doc-comment promissory note** (`rule.rs:70-71`) that asserts a Jirak floor exists "upstream of this one," and **that gate does not exist anywhere in the crate or its callers**. That is not a violation today; it is a *primed* violation — the moment this proposer is wired to a live `SpoStore` (D-ARM-5) without D-ARM-7 landing first, plan §11.1's named failure (substrate calcifies on noise) fires. The revision required before LAND is narrow and is spelled out in Finding 1.

This is **not a HOLD**, because:
- the crate is `exclude`d from the workspace (root `Cargo.toml:46`, under `exclude`, confirmed absent from `members`),
- it is std-only / zero-dep (no external `use` beyond `crate::`/`super::`/`std::`),
- its output type is `CandidateRule` / `CandidateTriple` — a *proposal*, never a committed triple, with no `SpoStore` write path in this crate,

so nothing it produces can reach the deterministic compile path or the SPO store *from within this PR*. The firewall is structurally intact at the crate boundary. The HOLD would only be warranted if this crate itself performed the Stage-C revision; it does not (Stage C lives in `lance-graph`, D-ARM-5, still Queued).

---

## Per-iron-rule findings

### I-NOISE-FLOOR-JIRAK — **YIELDS (with a mandatory pre-wire condition)** — the big one

**What the rule demands:** any noise-floor or statistical-significance claim on this substrate must cite Jirak 2016 weak-dependence rates, never classical IID Berry-Esseen (`CLAUDE.md:311-343`). The plan elevates this to MANDATORY at Stage A: §4 "This is not optional," §2.1 termination criterion `n × support ≥ JIRAK_MIN_EVIDENCE`, §11.1 risk, §15 invariant table.

**What the code actually does — the gates, concretely:**

- `rule.rs:73-75` `CandidateRule::passes(min_support, min_confidence)` gates ONLY on the two classical ARM floors: `self.support >= min_support && self.confidence >= min_confidence`. No Jirak term.
- `extract.rs:172` is the single emission gate in the Aerial proposer: `if rule.passes(params.min_support, params.min_confidence) { out.push(rule); }`. Classical-only.
- `extract.rs:78` the apriori pre-prune also uses `params.min_support` only.
- `ExtractParams` (`extract.rs:46-56`) carries `min_support: 0.01`, `min_confidence: 0.5` — the plan §2.1 classical defaults — and **no** `jirak_*` field.
- `aerial/mod.rs:103-115` `AerialProposer::mine()` / `next_batch()` returns `extract_rules(...)` directly. No post-filter.
- **Grep confirms:** the token `jirak` appears **zero times** in `crates/lance-graph-arm-discovery/src/`. There is no `jirak` module, no `JIRAK_MIN_EVIDENCE` const, no significance-deviation test against the independence null.

**Is this a VIOLATES?** No — and the distinction is load-bearing for the council. I-NOISE-FLOOR-JIRAK is triggered by *making a significance claim* ("N σ above noise floor," a calibrated threshold presented as principled). **This crate makes no such claim.** `support`/`confidence` are honest descriptive statistics measured on the window (`encode.rs:165-184`), and `passes()` is explicitly documented as the *classical* gate. The doc-comment even names the rule and says the Jirak floor is separate. A descriptive statistic with no significance assertion does not invoke Berry-Esseen, so there is nothing to mis-cite. **YIELDS.**

**BUT — the doc-comment writes a cheque the codebase can't cash.** `rule.rs:68-71` states verbatim:

> "The Jirak-bound significance floor (`I-NOISE-FLOOR-JIRAK`) is a *separate*, stricter gate applied **upstream of this one** — see the module-level docs of the proposer."

Following that pointer: the proposer module docs (`aerial/mod.rs:17-24`) describe the determinism rationale and say output "is gated by the downstream ratification council" — they do **not** describe, implement, or reference any Jirak upstream gate. The synergy doc (`aerial-arm-ruff-spo-codegen-synergies.md:171, §4`) repeats the same claim ("the `I-NOISE-FLOOR-JIRAK` floor (D-ARM-7, separate gate) keeps the thin tail from ever being emitted"). **D-ARM-7 is Queued, not Shipped** (`STATUS_BOARD.md:630`). So the "upstream gate" the comment promises is a *future deliverable that does not yet exist*. The code therefore documents a safety property it does not possess.

This is the precise risk plan §11.1 names: "Stage A emits rules at high rates that scrape past `MIN_CONFIDENCE` but below the Jirak floor; Stage C's revise weights them in; the substrate calcifies on weak signal." With `min_confidence` defaulting to 0.5 (`extract.rs:53`) and the NARS confidence mapping `c = m/(m+k)` saturating toward 1 as `m = support×n` grows (`translator.rs:57-58`), a thin-but-frequent spurious correlation at a 200K window produces a *high-c* candidate — exactly the "weak signal the revise cannot down-weight fast enough" case. The Jirak floor is the only thing in the design that stops it, and it is absent.

**Required revision before LAND (any one of these closes it):**
1. **Preferred:** weaken the `rule.rs:70-71` doc-comment from a present-tense assertion ("is a separate gate applied upstream") to a forward reference ("**will be** gated upstream by D-ARM-7 `jirak`; until that lands, `passes()` is the *only* gate and this proposer MUST NOT be wired to a live `SpoStore`"). Make the absence honest. Cheapest; no code.
2. Add the `jirak` floor field to `ExtractParams` now (even if the default is a permissive `0.0` placeholder that cites D-ARM-7 as the source of the real bound), so the gate *site* exists in `extract.rs:172` and D-ARM-7 only has to supply the number. Removes the "primed violation" entirely.
3. Add a crate-level `// DOCTRINE:` note + a `#[test]` canary that asserts the proposer is not exported for SPO ingestion until a Jirak gate is present (a compile-time tripwire mirroring plan §11.1's canary).

**Council bottom line on the big one:** the *finding the council asked me to test* — "does this claim significance the n-bound doesn't support?" — **the code does not; the doc-comment claims a floor that doesn't exist.** Land only after the comment is made honest (or the gate site is stubbed). Do **not** let D-ARM-5 (the live-`SpoStore` round-trip) reference this proposer until D-ARM-7 lands. The exclusion + proposal-only output is what downgrades this from HOLD to LAND-with-revision.

### I-SUBSTRATE-MARKOV — **YIELDS** (verified against the canonical revision)

**What I checked:** does the truth mapping silently diverge from `lance_graph::graph::spo::truth::TruthValue::revision` (`truth.rs:57-72`) and thereby break the Chapman-Kolmogorov/NARS revision invariant?

**The two functions are different operations and must not be confused:**
- The crate's `arm_to_nars` (`translator.rs:53-63`) is a **single-observation constructor**: it turns one ARM rule's raw stats into an *initial* `(f, c)`. `f = confidence` (P(Y|X)); `c = m/(m+k)`, `m = support×n`, `k=1.0`.
- The canonical `TruthValue::revision` (`truth.rs:58-72`) is a **two-belief combinator**: `w_i = c_i/(1-c_i)`, `w = w1+w2`, `f = (w1·f1+w2·f2)/w`, `c = w/(w+k)`, `k=1.0`.

**The crate does NOT reimplement revision** — and that is correct. It stops at producing the `(f,c)` that a *downstream* Stage C (D-ARM-5, in `lance-graph`, not this crate) feeds into the canonical `revision`. So there is no parallel/divergent revision kernel here to violate the semigroup property. `translator.rs:27-30` is explicit about this intent ("Deliberately *not* a re-implementation of the SPO store's `TruthValue`... honours `E-SOA-IS-THE-ONLY`").

**Algebraic consistency check (the subtle part):** for the downstream `revision` to be coherent, the crate's `c` must be the inverse of revision's confidence→evidence map. Revision recovers evidence as `w = c/(1-c)`. The crate sets `c = m/(m+k)` ⇒ `c/(1-c) = m/k = m` (at `k=1`). So a candidate carrying `c` round-trips into `revision` as evidence weight `w = m = support×n` — i.e. the crate's "evidential mass" IS exactly the NARS evidence count the revision arithmetic expects. **They agree by construction.** No divergence.

**The one place the crate touches the SPO truth algebra directly** is `NarsTruth::expectation` (`translator.rs:43-45`): `c·(f−0.5)+0.5`. This is **byte-identical** to `TruthValue::expectation` (`truth.rs:48-50`) and is asserted by `translator.rs:200-207`. Same `TruthGate` will gate an ARM-mined rule and a structurally-extracted triple. Correct.

Bundle math (vsa_bind / vsa_bundle / d=16384 / concentration-of-measure) is entirely untouched — this crate never enters that path. **YIELDS.**

### I-VSA-IDENTITIES — **YIELDS** (clean separation; no register touched)

ARM operates strictly on identity-typed atoms and never superposes content:
- `Item { feature: u32, category: u32 }` (`rule.rs:17-23`) is a pure identity pair — small ints, `Eq + Hash + Ord`, the Test-0 "natural ID/enum" register the rule blesses.
- `CandidateTriple { s, p, o, f, c }` (`translator.rs:86-98`) is `(s,p,o)` identity IRIs + scalar truth — never a fingerprint.
- The autoencoder's `Vec<f32>` one-hot input (`encode.rs:90-98`) and decoder probabilities (`autoencoder.rs:96-124`) are **a learned compressor's internal activations**, not a VSA carrier. They are never `vsa_bundle`d, never written to a `BindSpace` column, never persisted. They live and die inside one `extract_rules` call.
- **No `Vsa16kF32` / `Binary16K` / CAM-PQ / palette-codebook / quantized register appears anywhere in the crate** (no such import; std-only). `arm_to_nars` and the ndjson emitter touch only `f32` scalars + `String` IRIs.

The four tests are respected by virtue of the crate using the register (HashMap/enum-style `Item`) for exact-match work rather than reaching for VSA at all. There is no superposition of content here to destroy. **YIELDS.**

### I-LEGACY-API-FEATURE-GATED — **YIELDS (NA-leaning)**

No v1/v2 layout reclaim is in play. The crate defines no packed bitfield accessor (`pack`/`unpack`/`with_*`/`set_*`) over a versioned layout; `CausalEdge64` is not touched. The `aerial` feature (`Cargo.toml:34-36`, `lib.rs:63-64`) gates the *autoencoder module*, not a v1 accessor — and it is gated cleanly: `#[cfg(feature = "aerial")] pub mod aerial;` with the carriers (`rule`/`translator`/`ndjson`/`encode`) compiling feature-free, so a deployment can take the truth/triple contract without the AE. That is the *correct* application of feature-gating doctrine (isolate the heavy/nondeterministic path), not the anti-pattern (silent semantic divergence under a flag). No AP1 alias. **YIELDS.**

---

## Determinism boundary — enforced or merely documented?

**Structurally enforced at the crate boundary, but "seeded ⇒ reproducible" is NOT "deterministic" and the docs overclaim by one word.**

What IS enforced (genuine, not vibes):
- The crate is in `exclude` (root `Cargo.toml:46`), confirmed **absent** from `members`. The nondeterministic AE cannot enter the `lance-graph` compile graph — verified, not asserted.
- Output is `CandidateRule` — a proposal. There is **no `SpoStore` write, no codegen call, no ratification-bypass** path in the crate. The only sink is `ndjson::to_ndjson` (a `String`), which a *downstream* loader must choose to ingest. Nothing nondeterministic crosses into compile output from here.
- The AE is seeded through one `Rng` stream (`aerial/mod.rs:84`, `rng.rs` SplitMix64) covering weight init, denoising mask, and epoch shuffle. `reproducible_from_seed` (`mod.rs:194-205`) and `training_is_reproducible_from_seed` (`autoencoder.rs:337-347`) assert bit-identical rules / weights for same seed+data+build.

Where the claim is too strong — **"seeded ⇒ reproducible" ≠ "deterministic across platforms":**
- The forward/backward pass is single-threaded fixed-order f32 accumulation (`autoencoder.rs:100-117, 199-232`) — good, that removes *reduction-order* nondeterminism **within one binary**.
- But the AE uses f32 **transcendental / libm** functions whose last-ULP results are **not guaranteed identical across platforms, libm versions, or `-ffast-math`-style codegen**: `tanh` (`autoencoder.rs:106`), `exp` (`autoencoder.rs:246`), `ln` (`autoencoder.rs:263`; `rng.rs:51`), `cos` + `sqrt` (`rng.rs:51-52`). A one-ULP difference in `tanh` early in training, amplified over 500 epochs of SGD, can flip a borderline `p > τ_c` reconstruction-probe decision (`extract.rs:155`) and thus *add or drop a rule* on a different machine. So same-seed reproducibility holds **on the same platform/build**, which is all the tests check and all the audit story needs — but the lib-doc phrase "Same seed, same data, and same hyper-parameters give **bit-identical** weights and identical rules" (`lib.rs:39-42`) is true only intra-platform. It should say "bit-identical *on a fixed platform/toolchain*."

Why this is acceptable (not a HOLD): the determinism that the doctrine *requires* is determinism of the **compile path downstream of the ratification firewall** — `op_emitter` Rust codegen, `ruff_python_codegen`. This crate sits **upstream** of that firewall (synergy doc §5 table; plan §0 Stage-D firewall). A proposer is *allowed* to be nondeterministic precisely because the council + hypothesis-test re-derive and ratify before anything compiles. Platform-variance in the proposer changes *which candidates get proposed*, not *what compiles* — and the council is the gate that absorbs that. So "seeded for auditability, fenced behind exclusion + proposal-only output" is the correct posture. The only fix needed is the one-word doc honesty (`lib.rs:41` "bit-identical" → "bit-identical on a fixed platform").

**Determinism verdict:** boundary is real and enforced by construction; the reproducibility *claim* is mildly overstated (cross-platform f32 transcendental variance) and should be footnoted. Folds into the LAND-with-revision doc pass.

---

## Adjacency / epiphany (which D-ARM-* this most depends on or blocks)

D-ARM-13 is the **transcode of D-ARM-9's Aerial leg pulled in-process** (the plan §14 had deferred the autoencoder to Python-over-IPC; the user directive superseded that — see `AGENT_LOG` top entry + `STATUS_BOARD.md:636`). Its hardest dependency is **D-ARM-7 (Jirak)**: not a compile dependency, but a *doctrinal* one — D-ARM-13's own doc-comments (`rule.rs:70`, synergy §4) promise the Jirak floor that only D-ARM-7 supplies, and plan §11.1 makes that floor the sole defense against the calcify-on-noise failure. **D-ARM-13 should be treated as blocked-for-SPO-ingestion-purposes by D-ARM-7**, even though it ships green standalone. It also pre-stages **D-ARM-1/D-ARM-2** (the contract carriers): `rule.rs:8-9, 84` and `translator.rs` are deliberate *local mirrors* of the planned `lance-graph-contract::{CandidateRule, Proposer}` — when D-ARM-1/2 land, this crate should re-export them rather than keep the duplicates, or the AP-style "two CandidateRule shapes" drift the plan warns against (§3.2 seam) sets in. It most directly **blocks/feeds D-ARM-5** (hypothesis-test): D-ARM-5 is where this proposer's `(f,c)` actually meets `TruthValue::revision` and the live `SpoStore` — and D-ARM-5 is exactly the wave where the missing Jirak gate becomes load-bearing, so **the sequencing constraint is firm: D-ARM-7 before D-ARM-5-consuming-this-proposer.** The epiphany I see, and endorse: the transcode correctly realizes `E-DISCOVERY-CODEGEN-BRACKET-1` (Aerial = the runtime-data frontend of the three-frontend/one-substrate/two-codegen bracket; synergy doc §0) and respects `E-INTERPRET-NOT-STORE-1` (ARM rules are one interpretation projection, emitted as ndjson, never the canonical store). The genuinely useful *new* finding surfaced here is the **predicate-vocabulary gap** (D-ARM-SYN-1): `ruff_spo_triplet::Predicate` is a closed set with no `Implies`/`CoOccursWith`, so the very ndjson this crate emits (`DebugProjector` → `"implies"`, `translator.rs:131-135`) cannot yet load through the canonical `parse_triples` loader — a real, deliberate ontology decision the council must take before the bracket closes end-to-end. That gap is correctly flagged as council-gated rather than silently patched, which is itself doctrine-respecting.

---

## Summary table

| Rule | Verdict | One-line rationale |
|---|---|---|
| I-NOISE-FLOOR-JIRAK | YIELDS* | code makes no significance claim; but `rule.rs:70-71` doc asserts an upstream Jirak gate that does not exist (D-ARM-7 Queued) — make the comment honest before LAND, and never wire to live SpoStore (D-ARM-5) until D-ARM-7 lands |
| I-SUBSTRATE-MARKOV | YIELDS | `arm_to_nars` is a single-obs constructor, not a rival revision; `c=m/(m+k)` round-trips into canonical `TruthValue::revision` as `w=m` exactly; `expectation` is byte-identical; bundle math untouched |
| I-VSA-IDENTITIES | YIELDS | operates on identity `Item{feature,category}` + `(s,p,o)` IRIs; AE activations are a compressor's internals, never bundled/persisted; no Vsa/CAM-PQ/quantized register touched |
| I-LEGACY-API-FEATURE-GATED | YIELDS | no v1/v2 layout reclaim; `aerial` feature gates the module cleanly (carriers compile feature-free), not a silently-divergent v1 accessor |

\* the `*` is the LAND-condition: one honest-doc edit (plus the `lib.rs:41` "bit-identical" → "bit-identical on a fixed platform" footnote) and a firm D-ARM-7-before-D-ARM-5 sequencing note. No code logic change required; the firewall (exclusion + proposal-only output) holds today.
