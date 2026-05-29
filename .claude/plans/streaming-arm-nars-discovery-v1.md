# streaming-arm-nars-discovery-v1 — Streaming association-rule discovery → NARS revision → ratified SPO triples → deterministic codegen

> **Status:** PROPOSAL / integration plan. Spec only; **no code in this plan**.
> **Authored:** 2026-05-29 (session continuation on `claude/activate-lance-graph-att-k2pHI`, post-PR #433 + #434 merges).
> **Supersedes nothing; integrates / sequences:**
>   - `unified-soa-convergence-v1.md` (PR #434 — the ONE SoA all consumers read; this plan adds the *upstream proposer* leg)
>   - `odoo-business-logic-blueprint-v1.md` (the typed Odoo SoA + L-doc projection — extraction proposer)
>   - `odoo-source-extraction-v1.md` (the AST proposer that backs `OdooConfidence::Extracted`)
>   - `style_recipe.rs` + `op_emitter.rs` (Phase 1 + Phase 2 of the SoA→SoC codegen — the *downstream* consumer of ratified rules)
>
> **Anchored to (FINDING-grade):** `I-NOISE-FLOOR-JIRAK` (Jirak-bound thresholds, not classical Berry-Esseen), `I-VSA-IDENTITIES` (bundle identities, never content; ARM extracts identity-rules from content stats), `I-SUBSTRATE-MARKOV` (the NARS revision arc IS the Markov trajectory), `E-SOA-IS-THE-ONLY` (proposers write to the one SoA via SpoBuilder; never a parallel DTO), `E-BATON-1` (cross-mailbox state is the discrete owned baton — discovery batches ride this contract), `E-INTERPRET-NOT-STORE-1` (triplet substrate admits domain-owned interpretation; ARM IS a domain interpretation projection).
>
> **Papers anchored to:**
>   - Karabulut, Groth, Degeler — *Neurosymbolic Association Rule Mining from Tabular Data* (arxiv 2504.19354v1, Apr 2025). Aerial+ as one possible Stage-A proposer. Source: https://github.com/DiTEC-project/aerial-rule-mining (workspace-fork at AdaWorldAPI/aerial-rule-mining; outside MCP allowlist as of 2026-05-29).
>   - Abreu, Cruz, Guerreiro — *Ontology-Driven Model-to-Model Transformation of Workflow Specifications* (arxiv 2511.13661v1, Nov 2025). Independent confirmation of the externalize-interpretation-not-code doctrine — § "from code-centric to ontology-driven" is the direct mirror of our triplet-substrate position.
>
> **Owns the answer to:** *"in a perfect world we would need rules discovery and stream proprietary data through NARS reasoning and stream 20.000-200.000 [records per window] and try to determine co-correlation into deterministic rule candidates and do hypothesis testing against facts and edges."*

---

## 0. Executive summary (one screen)

The Odoo SoA → Foundry SoC pipeline shipped today (PR #433: `style_recipe.rs`; this branch: `op_emitter.rs`) is the **deterministic downstream codegen leg**. Its inputs are typed `OdooEntity` SoA records that came from two **proposer legs** today:

1. **Curated (L-doc projection)** — humans translate prose to const data. High confidence, low throughput.
2. **Extracted (AST walk)** — `tools/odoo-blueprint-extractor` parses `/home/user/odoo` Python ORM source. Deterministic, lossless for what's in the AST. Bounded by what the source explicitly states.

Both proposers feed the same SPO substrate. Both are **bounded by the literal artifact** — neither can surface co-correlations that emerge only in *runtime data* (parquet rows, transaction streams, invoice history). The paper *MOST OF THE BUSINESS LOGIC LIVES IN THE DATA, NOT THE SCHEMA*: ARM (Aerial+ or the classical FP-Growth lineage) is the missing **third proposer leg** that mines runtime tabular data for `(X → Y)` rules with NARS-compatible truth `(frequency, confidence)`.

The plan opens a new crate `lance-graph-arm-discovery` that streams 20K-200K rows per window through:

```
parquet/stream
   │
   ▼  Stage A — proposers (parallel feeds)
   ┌──────────────────────────────────────────────────────────────┐
   │  A1. Streaming pair-stats     [deterministic, default trunk] │
   │      sufficient statistics per (item_i, item_j) over window  │
   │      → support, confidence, Jirak-bound significance        │
   │  A2. Aerial+ neural-symbolic [optional, behind feature flag] │
   │      autoencoder + reconstruction-probe rule extraction      │
   │      → support, confidence (paper Algorithm 1)               │
   └──────────────────────────────────────────────────────────────┘
   │
   ▼  Stage B — translator (ARM truth → NARS truth)
   ┌──────────────────────────────────────────────────────────────┐
   │   confidence (ARM)        → frequency  (NARS f ∈ [0, 1])     │
   │   support × window_size   → confidence (NARS c ∈ [0, 1))     │
   │   → CandidateTriple { s, p, o, truth: TruthValue, origin }   │
   └──────────────────────────────────────────────────────────────┘
   │
   ▼  Stage C — hypothesis test (against SpoStore + EdgeColumn)
   ┌──────────────────────────────────────────────────────────────┐
   │   for each candidate:                                        │
   │     prior = SpoStore::lookup(s,p,o)                          │
   │     match  prior                                             │
   │     ─ Some(t) → SpoStore::revise(t, candidate.truth)          │
   │                  (NARS revision; widens confidence, weights    │
   │                   frequency by inverse variance)              │
   │     ─ Contradiction(prior, c, candidate, c') →                 │
   │                 commit a Contradiction edge per The Click     │
   │     ─ None  → queue for ratification (Stage D)                │
   └──────────────────────────────────────────────────────────────┘
   │
   ▼  Stage D — ratification (epiphany-brainstorm-council gate)
   ┌──────────────────────────────────────────────────────────────┐
   │   threshold-survive candidates with `origin = ArmDiscovery`  │
   │   pass through the 5-savant council before becoming a triple │
   │   the codegen path consumes. NEVER auto-promote — discovery  │
   │   is the Epiphany branch of The Click, not the Commit branch.│
   └──────────────────────────────────────────────────────────────┘
   │
   ▼  Stage E — codegen (op_emitter consumes ratified)
   ┌──────────────────────────────────────────────────────────────┐
   │   `op_emitter::emit_op_dispatch` takes ratified triples that │
   │   carry `OdooConfidence::Ratified` (new variant) and emits    │
   │   `RECIPE_* ` consts + per-kind Op slices. Determinism rests │
   │   on the ratification gate — Stage D is the firewall between │
   │   nondeterministic proposers and deterministic compile path. │
   └──────────────────────────────────────────────────────────────┘
```

Total new crates: **1** (`lance-graph-arm-discovery`). Total new deliverables: **D-ARM-1 … D-ARM-9**. Two corrections to PR #434's unified-SoA plan (§7 of this doc): an extra `discovery_arc` SoA column and a `discovery_origin: u8` provenance byte.

**No code in this PR. No cargo invoked.**

---

## 1. Context — the two papers and the workspace

### 1.1 Paper anchor — Aerial+ (Karabulut, Groth, Degeler, 2025)

Neurosymbolic ARM. The problem they solve is *rule explosion*: classical exhaustive miners (FP-Growth, HMine) emit O(2^k) rules over k features. Their approach:

1. One-hot encode rows into transaction vectors.
2. Train an **under-complete denoising autoencoder** with softmax-per-feature and BCE loss (paper §3.2). The latent representation compresses feature co-occurrence.
3. **Extract rules by exploiting reconstruction** (paper §3.3, Algorithm 1). For antecedent candidate `X`: mark its categories at probability 1, uniform elsewhere, forward-pass; for every category `Y` with `p_Y > τ_c` and `min_X p_X > τ_a`, emit rule `X → Y` carrying support and confidence.

Their result on five UCI datasets: 2–10× fewer rules, full data coverage, equal or higher confidence vs FP-Growth. On Spambase→CORELS: **1,409 rules vs 275,003 at higher accuracy in 5 s vs 1,258 s.**

The critical observation for us is *not* the neural part — that's a compressor optional for our typed-and-sparse domain. The critical observation is the **truth definition** (paper §2, verbatim):

> "An association rule X → Y is said to have support level s if s% of transactions in D contains X ∪ Y. The confidence of a rule is the conditional probability that a transaction containing X also contains Y."

This maps to `lance_graph::graph::spo::TruthValue::new(f, c)` with **no impedance mismatch**:

- ARM **confidence** = P(Y|X) → NARS **frequency** `f` (degree to which the implication holds)
- ARM **support × window_size** evidential weight → NARS **confidence** `c` (how much evidence backs `f`)

An Aerial+ candidate rule lifts straight into `SpoBuilder::build_edge` as a `(s, p, o, f, c)` quad with full NARS revision semantics. The reconstruction-probe pattern (`unbind` + cleanup against codebook) is the continuous twin of CLAUDE.md's `likelihood = vsa_cosine(unbind(bundle), codebook_fp)` thresholded by resonance.

### 1.2 Paper anchor — Ontology-driven M2M (Abreu, Cruz, Guerreiro, Nov 2025)

Independent confirmation of our position. Their pipeline: proprietary JSON workflow defs → semantic lifting via RML → RDF triples + DL reasoner over BBO ontology → BPMN 2.0 generation via Camunda Model API. Result: 92 BPMN diagrams from 69 JSON inputs, 94.2% success, 404 ms/file, **fully deterministic CI pipeline**.

The load-bearing quote (paper §4, "From a code-centric instantiation to an ontology-driven method"):

> "An initial, code-centric prototype implemented a direct JSON → Java → BPMN pipeline. Although feasible end-to-end, IBPM rapidly accumulated special-case handlers as the specification evolved (e.g., button patterns, conditional targets, multi-instance conventions), requiring code edits rather than configuration and limiting portability beyond a specific engine/version. The ontology-driven approach externalizes mapping knowledge into ontologies and RML rules."

This is **verbatim** the position our triplet substrate + `derive_style_recipe` + `op_emitter` enforces — externalize interpretation into ontology + declarative rules, not code. The paper's failure mode (5.81% — dynamic/time-based behavior absent from static JSON) is *exactly* our Stage-2 dark-atom gap (Money/Quantity/ApplyRate/EmitAmount/Event/FiscalCtx not yet lit because the extractor doesn't populate `return_kind`/`semantic_role`).

The two papers bracket our architecture:

- **Paper 1 (BPMN M2M)** validates the **downstream** codegen thesis — triples → deterministic generated artifact.
- **Paper 2 (Aerial+)** supplies the missing **upstream discovery** leg — tabular data → truth-carrying SPO candidates.

Both externalize symbolic artifacts from proprietary/non-symbolic sources via **thresholded extraction**. Both converge on **SPO + NARS truth** as the invariant middle. That convergence is the candidate `E-DISCOVERY-CODEGEN-BRACKET-1` epiphany; this plan is the implementation surface that operationalizes it.

### 1.3 Workspace position

Today's proposer legs into the SPO substrate (and into the typed `OdooEntity` SoA that drives `style_recipe.rs` + `op_emitter.rs`):

| Proposer | Source | Confidence | Throughput |
|---|---|---|---|
| `D-ODOO-BP-1b` (curated) | `.claude/odoo/L*.md` prose | `OdooConfidence::Curated` | 1-2 entities/hour (human) |
| `D-ODOO-EXT-2` (extracted) | Python AST over `/home/user/odoo` | `OdooConfidence::Extracted` | ~10s entities/sec (one-shot batch) |
| **THIS PLAN** (ArmDiscovery) | parquet streams / runtime tabular data | `OdooConfidence::ArmDiscovered` (new) | 20K-200K rows/window, continuous |

The third leg is the only one that surfaces co-correlations emerging from *runtime behavior* — invoices that consistently route through Fiscal Position A when partner country B AND product category C, observable in years of `account.move` rows but absent from `account_fiscal_position.py`. The codegen path already exists; this plan supplies its missing input.

---

## 2. The five-stage pipeline (detailed)

### 2.1 Stage A — proposers (parallel, fan-in)

Two proposer types share the **same output shape** — a `CandidateRule { antecedent: Vec<Item>, consequent: Vec<Item>, support: f32, confidence: f32, n: u32 }` where `n` is the window size. The output of either feeds Stage B identically. Both are gated behind feature flags in the new crate so deployments can pick `arm-pair-stats` only, `arm-aerial` only, or both.

#### A1. Streaming pair-stats (default trunk, deterministic)

The cheap branch. Per-window sufficient statistics over `(item_i, item_j)` pairs and (optionally) triples up to a fixed antecedent bound `a` (paper-default `a=2`):

```text
For each row in window:
  for each item i in row.items:
    counts[i] += 1
    for each item j in row.items where j > i:
      pair_counts[i, j] += 1
      // optional: triples if a >= 3
      for each item k in row.items where k > j:
        triple_counts[i, j, k] += 1
After window closes:
  for each (i, j) with pair_counts[i,j] >= MIN_SUPPORT_COUNT:
    support     = pair_counts[i,j] / n
    confidence  = pair_counts[i,j] / counts[i]
    emit CandidateRule { i → j, support, confidence, n }
```

**Properties:**

- Fully deterministic — same input → same candidates.
- Memory bound: `O(k² + k³)` for k items at `a ≤ 3`. For k = 200 features, k³ = 8 M counters; with `u32` counters that's 32 MB / window. Fits.
- Throughput: one pass per row, SIMD-amenable; 200 K rows/window at 10 µs/row = 2 s/window on one core.
- **The Jirak-bound significance** (§4) is what filters the noise floor — it's not just `support ≥ threshold`, it's `support ≥ jirak_lower_bound(n, weak_dependence_index)`.

This is the cornerstone proposer. Aerial+ is optional fan-in.

#### A2. Aerial+ neural-symbolic (optional, behind `aerial` feature flag)

For high-dimensional sparse data where pair-stats memory blows up (k > 500), or where 3+ antecedents matter and triple-counts won't fit, optionally fan in an Aerial+-style proposer. Implementation is **out of crate** — we don't bring the autoencoder into Rust. Instead:

1. The Python reference (DiTEC-project/aerial-rule-mining) runs as a separate process.
2. It writes `CandidateRule` records as NDJSON to a Unix socket or stdin pipe consumed by the Rust crate.
3. Cargo `aerial` feature gates the IPC client.

This keeps the autoencoder out of the deterministic compile path (it's a *second proposer*, not the trunk) and avoids a heavy ONNX/Burn dep in `lance-graph`.

#### Termination criterion (both proposers)

A proposer emits a rule only if all of:

- `support ≥ MIN_SUPPORT` (configured per-feed; default 0.01)
- `confidence ≥ MIN_CONFIDENCE` (default 0.5)
- `n × support ≥ JIRAK_MIN_EVIDENCE` (the weak-dependence-aware floor; see §4)

Below any of these, the candidate is **silently dropped** — never crosses Stage A→B boundary.

### 2.2 Stage B — translator (ARM truth → NARS truth)

A pure function `arm_to_nars(rule: &CandidateRule) -> TruthValue`. The mapping:

```rust
// In lance-graph-arm-discovery::translator
pub fn arm_to_nars(rule: &CandidateRule) -> TruthValue {
    // ARM confidence is P(Y|X) — directly maps to NARS frequency.
    let frequency = rule.confidence.clamp(0.0, 1.0);
    // Evidential mass: support × window_size gives us the count of supporting
    // observations; NARS confidence = m / (m + k) where k is the personality
    // constant (default k=1 in NAL-9). Larger m → confidence → 1.
    let m = (rule.support * rule.n as f32) as u32;
    let k = NARS_PERSONALITY_K; // configured per-feed; default 1.0
    let confidence = (m as f32) / (m as f32 + k);
    TruthValue::new(frequency, confidence)
}
```

The translator also wraps the rule's `(antecedent, consequent)` into the `(subject, predicate, object)` triple shape via a domain-supplied projector (per-feed config: an Odoo feed projects `(model_name, predicate_name, value)`; an SBM feed projects different).

### 2.3 Stage C — hypothesis test (against SpoStore + EdgeColumn)

The most consequential stage. For each `CandidateTriple { s, p, o, truth, origin }` produced in Stage B:

```rust
match spo_store.lookup(s, p, o) {
    // Prior exists with truth t_prior; NARS revise:
    Some(t_prior) => {
        let revised = TruthValue::revise(&t_prior, &candidate.truth);
        spo_store.update(s, p, o, revised);
        // Stamp a CausalEdge64 emission on the witness arc (per OQ-11.2 from #434);
        // the emission's confidence_u8 + inference_mantissa records this revision.
        mailbox.emit(CollapseGateEmission::from_revision(t_prior, candidate.truth, revised));
    }

    // No prior. Two paths:
    None if candidate.truth.expectation() < CONTRADICTION_THRESHOLD => {
        // Low-expectation novel candidate → just queue for ratification.
        ratification_queue.push(candidate);
    }
    None => {
        // High-expectation novel candidate → check for inversion contradictions.
        // (Looking for prior (s, p, ¬o) or similar negation patterns.)
        match spo_store.find_negation(s, p, o) {
            Some(t_inverse) => {
                // Committed contradiction per The Click — preserve, don't overwrite.
                spo_store.commit_contradiction(
                    Triple::new(s, p, o, candidate.truth),
                    Triple::new(s, p, negate(o), t_inverse),
                );
            }
            None => {
                // Genuine novelty — queue for ratification.
                ratification_queue.push(candidate);
            }
        }
    }
}
```

**Critical invariants:**

- The test reads from `SpoStore` via the existing `lance_graph::graph::spo::SpoBuilder` surface; no parallel index. Per `E-SOA-IS-THE-ONLY`.
- Revisions emit `CausalEdge64` rows; that emission IS the audit log (§11.2 of unified-soa-convergence-v1: "No separate revision log column.").
- Contradictions are NEVER overwritten — they're committed alongside per The Click ("Opinions are committed contradictions preserved, not resolved.").
- Novel candidates ENTER the ratification queue — they don't auto-promote.

### 2.4 Stage D — ratification (epiphany-brainstorm-council gate)

Already exists (PR #433 shipped `epiphany-brainstorm-council` + 5 savant cards). The ratification queue contents are exactly the workload the council was built for: domain-specific candidate findings that need pre-merge multi-perspective vetting.

For each candidate in the queue, the panel runs:

- `iron-rule-savant` — does the candidate violate I-NOISE-FLOOR-JIRAK by claiming significance the n-bound doesn't support?
- `dto-soa-savant` — does the triple fit one of the four BindSpace columns (Fingerprint/Qualia/Meta/Edge), or does it pretend to introduce a fifth?
- `cascade-impact-savant` — landing this rule changes which file?
- `prior-art-savant` — does an existing `E-…` or `K-…` (knowledge doc) already state this?
- `creative-explorer-savant` — what's the inverse / dual / second-order implication?

LAND verdict → `OdooConfidence::Ratified` stamp → triple becomes available to `op_emitter` Stage E.

The ratification queue itself is a `lance_graph_arm_discovery::RatificationQueue` — a bounded ring buffer (default 1024 candidates) backing onto Lance for persistence. Council runs are *not* automatic — they're queued for human-triggered batches (per session), per the council's design.

### 2.5 Stage E — codegen (op_emitter consumes ratified)

Already exists (this branch: `op_emitter.rs`). The only extension: `op_emitter::bucket_corpus` filters its input by `confidence ≥ OdooConfidence::Ratified` (a partial order — `Curated > Extracted > Ratified-via-ARM > Conjecture`). ArmDiscovered candidates that haven't passed Stage D never reach `op_emitter`.

This is the firewall. The deterministic codegen path stays deterministic because Stage D is the gate.

---

## 3. Crates and deliverables

### 3.1 New crate: `lance-graph-arm-discovery`

Location: `crates/lance-graph-arm-discovery/`. Sits alongside `lance-graph-ontology` in the workspace. Dependencies:

```toml
[dependencies]
lance-graph-contract = { path = "../lance-graph-contract" }  # TruthValue, CausalEdge64
arrow = "58"                                                  # window batches over RecordBatch
parquet = "58"                                                # parquet input feed
thiserror = "2"

[features]
default = ["arm-pair-stats"]
arm-pair-stats = []                                          # default trunk
arm-aerial = ["dep:tokio", "dep:serde_json"]                # IPC client for Aerial+ subprocess

[dependencies.tokio]
version = "1"
features = ["rt", "net", "io-util", "macros"]
optional = true

[dependencies.serde_json]
version = "1"
optional = true
```

**Public surface:**

```rust
// src/lib.rs (zero-dep beyond contract + arrow/parquet)
pub mod proposer;       // Stage A — pair-stats + (feature) aerial IPC client
pub mod translator;     // Stage B — arm_to_nars
pub mod hypothesis;     // Stage C — SpoStore round-trip + revision/contradiction
pub mod queue;          // Stage D queue — ratification buffer
pub mod feed;           // window / batch / projector configuration
pub mod jirak;          // Stage A threshold helpers (Jirak-bound)

// Re-exports
pub use proposer::{CandidateRule, Proposer, PairStatsProposer};
pub use translator::{arm_to_nars, CandidateTriple};
pub use hypothesis::{HypothesisTest, HypothesisOutcome};
pub use queue::{RatificationQueue, QueueEntry};
pub use feed::{Feed, FeedProjector, WindowSize};
```

Module-by-module deliverables in §3.3.

### 3.2 Touchpoints in existing crates

| Crate | Change | D-id | Risk |
|---|---|---|---|
| `lance-graph-contract` | Add `OdooConfidence::ArmDiscovered` + `OdooConfidence::Ratified` variants (or generalize to a `ProvenanceTier` enum). | D-ARM-1 | LOW — additive |
| `lance-graph-contract` | Add `pub trait Proposer { fn next_batch(&mut self) -> Vec<CandidateRule>; }` so the discovery crate is dependency-injectable. | D-ARM-2 | LOW — trait surface |
| `lance-graph` | `SpoBuilder::revise` already exists in `graph::spo::truth`; verify it preserves Contradiction semantics. | D-ARM-3 | LOW — verification only |
| `lance-graph-ontology` | `op_emitter::bucket_corpus` filters by `confidence ≥ Ratified`. | D-ARM-4 | LOW — one-line filter + test |
| `lance-graph-ontology` | New `style_recipe` rule (Rule 8): when entity is `ArmDiscovered`-backed, recipe acquires `DAtom::Compute` weight 2 (provisional). | D-ARM-5 | MED — opens recipe-rule pacing |
| `unified-soa-convergence-v1` (PR #434) | Two corrections proposed — separate `discovery_arc` column + `discovery_origin: u8` byte. See §7. | D-ARM-6 | MED — touches OQ-11.2 + OQ-11.5 |

### 3.3 New-crate module deliverables

| D-id | Module | Scope | Lines | Conf | Status |
|---|---|---|---|---|---|
| **D-ARM-1** | `lance-graph-contract` | `ProvenanceTier::{Curated, Extracted, ArmDiscovered, Ratified, Conjecture}` enum + comparison ordering | 50 | HIGH | Queued |
| **D-ARM-2** | `lance-graph-contract::proposer` | `pub trait Proposer { fn next_batch(...) }` + `CandidateRule` + `WindowMetadata` | 100 | HIGH | Queued |
| **D-ARM-3** | `lance-graph-arm-discovery::proposer::pair_stats` | Streaming pair-stats over RecordBatch; `a ∈ {1, 2, 3}` antecedent bound; emits `CandidateRule` | 400 | HIGH | Queued |
| **D-ARM-4** | `lance-graph-arm-discovery::translator` | `arm_to_nars` + `CandidateTriple` carrier + projector trait + Odoo `FeedProjector` impl | 200 | HIGH | Queued |
| **D-ARM-5** | `lance-graph-arm-discovery::hypothesis` | Round-trip against `SpoStore`; revision, contradiction commit, queue-for-ratification | 350 | MED | Queued |
| **D-ARM-6** | `lance-graph-arm-discovery::queue` | `RatificationQueue` ring buffer + persistence shim (Lance optional) | 200 | MED | Queued |
| **D-ARM-7** | `lance-graph-arm-discovery::jirak` | Jirak-2016 weak-dependence threshold helpers (n^(p/2-1) bound; p ∈ (2, 3]); cites EPIPHANIES § FORMAL-SCAFFOLD | 150 | HIGH | Queued |
| **D-ARM-8** | `lance-graph-arm-discovery::feed` | `Feed` + `FeedProjector` + window-size config; Odoo `account.move` projector as example | 250 | MED | Queued |
| **D-ARM-9** | `lance-graph-arm-discovery::proposer::aerial_ipc` | NDJSON-over-Unix-socket IPC client (feature-gated `arm-aerial`) | 200 | MED | Queued |
| **D-ARM-10** | `lance-graph-ontology::op_emitter` | One-line filter `confidence ≥ Ratified` + 2 tests | 30 | HIGH | Queued |
| **D-ARM-11** | `lance-graph-ontology::style_recipe` | Recipe rule 8: ArmDiscovered backing adds `DAtom::Compute` weight 2 (provisional, ratification-gated) | 80 | MED | Queued |
| **D-ARM-12** | benches + an end-to-end test feed | Synthetic parquet fixture; bench window-throughput; round-trip test through stages A-E with a small council | 400 | MED | Queued |

Total: ~2,400 LOC. About one-third of `lance-graph-ontology`'s `odoo_blueprint` size.

---

## 4. Thresholds — the Jirak grounding (I-NOISE-FLOOR-JIRAK)

CLAUDE.md's iron rule on weakly-dependent fingerprint bits applies here directly:

> **Classical IID Berry-Esseen is WRONG for this system.** Use **Jirak 2016** (arxiv 1606.01617, Annals of Probability 44(3) 2024–2063, "Berry-Esseen theorems under weak dependence") for any noise-floor or statistical-significance claim. Rate: `n^(p/2-1)` for `p ∈ (2,3]`, `n^(-1/2)` in L^q for `p ≥ 4`.

For ARM, the question is: at window size n with weakly-dependent transaction items, what's the *minimum* observed support s* at which we can claim `s_observed - s_true > δ` is significant?

Classical (wrong for our domain): for IID Bernoulli items, Berry-Esseen gives the σ threshold as `s* ≥ z · sqrt(s(1-s)/n)` with `z ≈ 1.96` at 95%.

Jirak (correct): the threshold scales as `n^{-1/(p/2-1)}` with `p` the moment characterizing dependence (p=4 → `n^{-1/2}`, p=2.5 → `n^{-0.25}`). For ARM items with shared categorical encoding and partial-order purchase dependence (the canonical weak-dependence pattern), p ≈ 3.0 is a reasonable default, giving `n^{-1}` decay — much stricter than the IID `n^{-1/2}`.

**Operational consequence for D-ARM-7:** the `jirak` module exposes:

```rust
pub fn jirak_significance_threshold(
    window_size: u32,
    p_moment: f32,                  // dependence index; default 3.0
    confidence_alpha: f32,          // significance level; default 0.05
) -> f32;
```

with the default conservative threshold of `(window_size as f32).powf(-1.0 / (p_moment / 2.0 - 1.0))`. A `CandidateRule` survives Stage A only if its support deviation from the null (independence) exceeds this bound.

**This is not optional.** Without Jirak grounding, the ARM discovery proposer leaks low-confidence rules into the SPO store at a rate the SpoStore::revise NARS revision cannot down-weight fast enough; the substrate calcifies on noise. With Jirak grounding, the proposer's false-positive rate aligns with the Markov-chain noise floor the rest of the substrate operates against.

Cross-ref: `I-NOISE-FLOOR-JIRAK` (CLAUDE.md), Jirak 2016 (arxiv 1606.01617), `.claude/board/EPIPHANIES.md` [FORMAL-SCAFFOLD] pillar 4.

---

## 5. Throughput regime — 20K-200K rows/window

The user's stated regime: "stream 20.000 - 200.000 [records per window]." Implications:

| Window size n | Per-pass time (200 features, 1 core) | Memory peak (counters) | Use case |
|---|---|---|---|
| 20,000 | ~0.2 s | k² × 4 B = 160 KB | sub-second iteration; in-session experimentation |
| 50,000 | ~0.5 s | 160 KB | typical CI batch over a week of Odoo `account.move` |
| 100,000 | ~1.0 s | 160 KB | typical month of mid-volume client |
| 200,000 | ~2.0 s | 160 KB | typical month of large client; the upper bound |

Memory is dominated by k² pair counters not n; n only controls per-row pass cost. For k = 200 (typical Odoo entity feature count after one-hot encoding), the trip is comfortably in-memory at any window size in the stated range.

For k = 1000+ (a denormalized multi-entity feed: `account.move` ⨝ `account.move.line` ⨝ `res.partner`), pair-counter memory blows to 4 MB and triple-counter memory to 4 GB. At that point:

- Drop `a` from 3 to 2 (pair-only): 4 MB fits.
- Or partition by entity-prefix: each shard runs k ≈ 200; merge candidates downstream.
- Or enable the `arm-aerial` feature: the autoencoder compresses k effectively for the high-dim sparse case.

The bench in D-ARM-12 will pin these numbers against a representative synthetic Odoo feed. The throughput claim is **bounded by the FeedProjector's row-decode cost**, not the pair-stats inner loop.

**Streaming, not batch:** the window is a sliding ring of `n` recent rows, not a one-shot batch. As new rows enter, old rows leave; the counters are incrementally updated (add new row's pair contributions, subtract leaving row's). This preserves the steady-state "near-real-time discovery" property the user named.

---

## 6. Mailbox SoA touchpoint — where this plugs into unified-SoA (PR #434)

The hypothesis-test stage (§2.3) is where ARM-discovery meets the **one little-endian SoA**. Concretely:

### 6.1 What ARM-discovery WRITES to the mailbox SoA

For every revision (Stage C path `Some(t_prior)`), the proposer emits **one** `CollapseGateEmission` (`CausalEdge64`-backed) onto the per-mailbox witness arc. Per `E-BATON-1` the emission carries:

- 13-byte base + 10 bytes/baton; one baton per revision; total wire cost = 23 bytes.
- `confidence_u8` stamps the post-revision NARS `c`.
- `inference_mantissa` stamps the post-revision NARS `f` (i4 signed mantissa per the 2026-04-21 layout).
- Source identity (the proposer that emitted it) goes in the new `discovery_origin: u8` byte proposed in §7.

For every contradiction commit (Stage C path with negation), the proposer emits **two** linked emissions — one for each side of the contradiction, with a back-pointer linking them. Per The Click: contradictions preserved, never resolved.

For every novel-candidate queue push, the proposer emits **zero** mailbox writes — the candidate sits in the (separate) `RatificationQueue`. Only post-ratification does it land on the mailbox SoA.

### 6.2 What ARM-discovery READS from the mailbox SoA

For every candidate triple, the hypothesis-test reads the prior `TruthValue` from `SpoStore::lookup`. That lookup IS a read against the `EdgeColumn` (`[CausalEdge64; N]`) per row — the existing surface, no new column needed.

### 6.3 No new column REQUIRED (with one caveat)

The unified-SoA columns shipped in D-MBX-A1 (lines 67-83 of `mailbox_soa.rs`):

```rust
pub edges: [CausalEdge64; N],
pub qualia: [QualiaI4_16D; N],
pub meta: [MetaWord; N],
pub entity_type: [u16; N],
```

These cover everything ARM-discovery needs for steady-state operation. **The caveat:** if we want to track multiple in-flight candidate rules per row (e.g. "this row is consistent with hypothesis H1, H2, but contradicts H3"), the existing `edges: [CausalEdge64; N]` arc is a single ring per row; multiple candidate streams compete for the same arc. That's what §7's `discovery_arc` proposal addresses.

For v1 (this plan), we ship without `discovery_arc` and live with single-arc contention — the proposer batches candidates so contention is bounded. If contention becomes the dominant cost in benches (D-ARM-12), v1.1 adds the column.

---

## 7. Corrections proposed to unified-soa-convergence-v1 (PR #434)

Reviewed the unified-SoA plan + handover. Two specific corrections to fold in via a follow-up PR after this discovery plan is ratified. Both are SPEC corrections — they don't invalidate the plan, they refine OQ defaults.

### 7.1 Correction 1 — `discovery_arc: [u32; D]` column, separate from `edges`

**Status of OQ-11.2 in #434:** "Witness arc width `W`? Plan default: W = 16 (~64 B/row at u32 handles). Needs user ratification before D-MBX-A3 lands."

**My add:** the `W=16` arc-handle column is for **belief-state arc emissions** — the cumulative trace of `CausalEdge64`-stamped revisions on the canonical mailbox state. It is NOT designed for tracking **in-flight discovery candidates** that haven't yet committed to a revision.

In ARM-discovery, a single window of 200K rows may produce thousands of candidate rules; each candidate touches multiple rows in hypothesis-test. If a row participates in K candidate hypotheses concurrently, the existing `edges` arc would either overflow at K > 16 or arbitrarily evict candidates.

**Proposal:** carve a parallel `discovery_arc: [u32; D]` column, D = 8 (default). Sits next to `edges` but rotates on a different cadence — `discovery_arc` rotates per window (every 200K rows), `edges` rotates per Commit/Prune. The new column also gets `discovery_arc_head: u8` per row (rotation index).

**Cost:** D × 4 = 32 bytes/row, plus 1 byte head. Total +33 B/row. At 1 M mailbox rows in a typical persistent mailbox = +33 MB. Acceptable.

**Trade-off:** more memory, cleaner separation between ratified-state-arc and candidate-stream-arc. Without this split, the proposer is forced to either rate-limit (degrading throughput) or share the `edges` arc (polluting the audit trail).

**This is a D-ARM-6 deliverable, separate from D-MBX-A3 to avoid blocking #434's roll-up.**

### 7.2 Correction 2 — `discovery_origin: u8` provenance byte

**Status of OQ-11.5 in #434:** "SoA version field width? Plan default: u16 at layout root; no per-column version stamps in v1."

**My add:** the SoA-root `version: u16` tracks layout-schema version. It doesn't tell a downstream consumer **which proposer produced the evidence currently sitting in any given row's `edges` arc.** Today, downstream consumers must assume "any extracted/curated entity"; once ARM-discovery starts emitting, that assumption breaks.

**Proposal:** add `discovery_origin: [u8; N]` column. Bit fields:

```text
discovery_origin (u8):
  bits 0-1 : ProvenanceTier (00=Curated, 01=Extracted, 10=ArmDiscovered, 11=Ratified)
  bits 2-3 : proposer id    (00=AstWalker, 01=PairStats, 10=Aerial, 11=Other)
  bits 4-7 : reserved (16 future proposers)
```

**Cost:** N bytes per mailbox = 1 KB for N = 1024. Negligible.

**Trade-off:** consumers (op_emitter, council, hypothesis-test) can filter on origin without consulting a parallel registry. Without this byte, the council's `prior-art-savant` can't tell whether a triple's prior support came from human curation (high prior) or from a chain of ARM-discovered revisions (re-revisable).

**This is also a D-ARM-6 sub-deliverable; defers to #434's D-MBX-10 for SoA-root-version semantics but adds the per-row origin byte alongside.**

### 7.3 Non-correction — `Vsa16kF32` deprecation stays untouched

I have nothing to add to OQ-11.4 ("CLAUDE.md `Vsa16kF32` doctrinal update"). The deprecation is correctly scoped: `Vsa16kF32` is a local-bundle compute carrier, not a cross-boundary state. ARM-discovery does NOT reach for `Vsa16kF32` — it operates on typed `CandidateRule` records and typed `(s, p, o)` triples. The bundle math (Markov ±5, role-key binding) is unaffected because ARM doesn't enter that path.

---

## 8. Deliverables — consolidated table

| D-id | Title | Crate | Lines | Conf | Status | Blocks / Depends on |
|---|---|---|---|---|---|---|
| **D-ARM-1** | `ProvenanceTier` enum + ordering | `lance-graph-contract` | 50 | HIGH | Queued | Blocks ALL other D-ARM-* |
| **D-ARM-2** | `Proposer` trait + `CandidateRule` carrier | `lance-graph-contract` | 100 | HIGH | Queued | Blocks D-ARM-3, D-ARM-9 |
| **D-ARM-3** | Pair-stats proposer (default trunk) | `lance-graph-arm-discovery::proposer::pair_stats` | 400 | HIGH | Queued | Depends on D-ARM-1/2/7; blocks D-ARM-12 |
| **D-ARM-4** | ARM-truth → NARS-truth translator | `lance-graph-arm-discovery::translator` | 200 | HIGH | Queued | Depends on D-ARM-1/2 |
| **D-ARM-5** | Hypothesis test + revision + contradiction commit | `lance-graph-arm-discovery::hypothesis` | 350 | MED | Queued | Depends on D-ARM-4; blocks D-ARM-12 |
| **D-ARM-6** | Ratification queue + `discovery_arc`/`discovery_origin` corrections to #434 | `lance-graph-arm-discovery::queue` + mailbox SoA cols | 200 + spec | MED | Queued | Depends on PR #434 D-MBX-A3 merge; blocks D-ARM-12 |
| **D-ARM-7** | Jirak-bound significance helpers | `lance-graph-arm-discovery::jirak` | 150 | HIGH | Queued | Blocks D-ARM-3 |
| **D-ARM-8** | Feed + FeedProjector config + Odoo example projector | `lance-graph-arm-discovery::feed` | 250 | MED | Queued | Depends on D-ARM-2; blocks D-ARM-12 |
| **D-ARM-9** | Aerial+ IPC client (feature-gated) | `lance-graph-arm-discovery::proposer::aerial_ipc` | 200 | MED | Queued | Optional; depends on D-ARM-2 |
| **D-ARM-10** | `op_emitter::bucket_corpus` ratification filter | `lance-graph-ontology::op_emitter` | 30 | HIGH | Queued | Depends on D-ARM-1 |
| **D-ARM-11** | `style_recipe` rule 8 for ArmDiscovered backing | `lance-graph-ontology::style_recipe` | 80 | MED | Queued | Depends on D-ARM-1 |
| **D-ARM-12** | End-to-end test + bench | `lance-graph-arm-discovery::tests` + benches | 400 | MED | Queued | Depends on D-ARM-3..8 |

**Total:** ~2,410 LOC, 12 deliverables, 1 new crate, 2 spec corrections.

---

## 9. Execution order

```text
Wave 1 — Contract (D-ARM-1, D-ARM-2) — additive contract trait + provenance enum
  ├─ ships as one PR; one Sonnet agent; ~150 LOC + 5 tests
  └─ blocks everything below; lands first

Wave 2 — Jirak (D-ARM-7) — pure math, no IO
  ├─ ships as one PR; one Sonnet agent; ~150 LOC + reference tests against Jirak 2016 worked examples
  └─ blocks Wave 3a

Wave 3a — Pair-stats proposer (D-ARM-3) — the default trunk
  ├─ ships as one PR; one Sonnet agent (Opus reviewer); ~400 LOC + 15 tests
  └─ blocks D-ARM-12

Wave 3b — Translator (D-ARM-4) — pure function
  ├─ ships as one PR; one Sonnet agent; ~200 LOC + 10 tests
  └─ Wave 3a and 3b parallel

Wave 4 — Hypothesis test (D-ARM-5) — SpoStore round-trip
  ├─ ships as one PR; Opus agent (multi-source: SpoStore + EdgeColumn + The Click semantics); ~350 LOC + 12 tests
  └─ depends on Waves 3a, 3b

Wave 5a — Queue + SoA corrections (D-ARM-6) — spec PR for #434 follow-up + queue impl
  ├─ ships as TWO PRs; one for spec follow-up against #434 (council-reviewed); one for queue impl
  └─ depends on PR #434 D-MBX-A3 landing

Wave 5b — Feed + projector (D-ARM-8) — DI config surface
  ├─ ships as one PR; one Sonnet agent; ~250 LOC + 8 tests
  └─ parallel with Wave 5a

Wave 6 — op_emitter + style_recipe rule (D-ARM-10, D-ARM-11) — downstream gates
  ├─ ships as one PR; trivial; one Sonnet agent; ~110 LOC + 4 tests
  └─ depends on D-ARM-1

Wave 7 — Aerial+ IPC (D-ARM-9, OPTIONAL) — feature-gated fan-in
  ├─ ships when user signals demand; one Sonnet agent; ~200 LOC
  └─ optional; not blocking

Wave 8 — End-to-end test + bench (D-ARM-12)
  ├─ ships as one PR; Opus agent (multi-source: all prior waves); ~400 LOC + bench
  └─ depends on Waves 1-6 (or 1-7 if Aerial+ included)
```

**Estimated calendar:** 6-8 sessions if executed serially with the disciplined "no cargo in agents" rule. Main thread runs cargo verifies after each wave merges to main.

---

## 10. Open questions

| # | Question | Default proposal | Blocks |
|---|---|---|---|
| **OQ-ARM-1** | What's the default window size for steady-state operation? | 100K rows, configurable per Feed | D-ARM-3, D-ARM-8 |
| **OQ-ARM-2** | What's the Jirak `p_moment` default for Odoo tabular data? | p = 3.0 (gives `n^{-1}` decay; conservative) | D-ARM-7 |
| **OQ-ARM-3** | What's the NARS personality constant `k` in `arm_to_nars`? | k = 1.0 (NAL-9 default) | D-ARM-4 |
| **OQ-ARM-4** | Should `RatificationQueue` persist across sessions, or be in-memory only? | In-memory v1; persist behind `--persist-queue` flag in v2 | D-ARM-6 |
| **OQ-ARM-5** | Antecedent bound `a`: hard-cap at 2, allow 3, or higher? | Hard-cap at 2 for pair-stats trunk; Aerial+ subprocess can go higher | D-ARM-3 |
| **OQ-ARM-6** | Contradiction commit shape — single `Contradiction` edge type, or symmetric pair? | Symmetric pair (one CausalEdge per side, back-pointer between) | D-ARM-5 |
| **OQ-ARM-7** | Do we need a `discovery_arc` column from day-one, or live with `edges` contention in v1? | Defer to v1.1 — measure contention in D-ARM-12 bench first | D-ARM-6 |
| **OQ-ARM-8** | What's the right policy for inverse-fingerprint contradiction detection in Stage C? Hash collision rate? | Cite `I-NOISE-FLOOR-JIRAK`; concrete cutoff TBD in D-ARM-5 | D-ARM-5 |
| **OQ-ARM-9** | How do council ratification verdicts flow back into the queue? Webhook? Manual session trigger? | Manual session trigger (per-session council batch); no webhook v1 | D-ARM-6 |
| **OQ-ARM-10** | Should Aerial+ IPC be a separate crate or a feature inside `lance-graph-arm-discovery`? | Feature flag inside; promote to separate crate if it grows large | D-ARM-9 |

---

## 11. Risks

### 11.1 Risk — proposer leaks low-confidence rules into SPO

**Failure mode:** Stage A emits rules at high rates that scrape past `MIN_CONFIDENCE` but below the Jirak floor; Stage C's revise weights them in; the substrate calcifies on weak signal.

**Mitigation:** D-ARM-7 (Jirak helpers) is mandatory in the threshold path. D-ARM-3's emission gate routes through `jirak_significance_threshold` BEFORE checking the user's `MIN_CONFIDENCE` config. A canary test in D-ARM-12 asserts no rule with `support × n < jirak_min_evidence(...)` ever crosses the proposer boundary.

**Confidence:** HIGH — the iron rule is already named; this plan inherits it.

### 11.2 Risk — Stage D ratification becomes the bottleneck

**Failure mode:** discovery emits 100s of novel candidates per day; the council can't keep up; the queue saturates; high-quality candidates wait behind low-quality.

**Mitigation:** the queue is bounded (default 1024); overflow drops *oldest* candidates (FIFO with priority bias). The council can run on prioritized batches (highest expectation first). v2 may add a triage pre-filter (the `cascade-impact-savant` runs solo on every queue entry; full panel only on those that survive cascade-impact > 0.5).

**Confidence:** MED — depends on observed discovery rate; D-ARM-12 bench will inform.

### 11.3 Risk — contradiction commit semantics drift from The Click

**Failure mode:** Stage C's contradiction path "commits" the contradiction but the EdgeColumn/SpoStore doesn't actually have a contradiction-edge type; "preserve" decays into "drop one side."

**Mitigation:** D-ARM-5 includes an audit pass against `lance_graph::graph::spo::truth` — verify the contradiction primitive exists at the truth level. If not, surface a follow-up to `lance-graph-contract` to add `ContradictionEdge` to the EdgeColumn taxonomy. Block D-ARM-5 on that contract addition.

**Confidence:** MED — depends on current state of `spo::truth::Contradiction`; needs a verification pass in Wave 4.

### 11.4 Risk — `lance-graph-arm-discovery` becomes a dumping ground

**Failure mode:** the new crate accumulates one-off projectors, ad-hoc feeds, and "just one more proposer" extensions; it loses focus.

**Mitigation:** strict scope discipline — the crate's only public surface is Proposer, CandidateRule, CandidateTriple, HypothesisTest, RatificationQueue, Feed, FeedProjector. Domain-specific projectors live in their own crates (e.g. `lance-graph-odoo-feed` if Odoo grows beyond an example). The crate's invariants are documented in its README; PR template requires a `Touches public surface? YES/NO` checkbox.

**Confidence:** MED — requires sustained governance discipline.

### 11.5 Risk — windowed pair-stats over-counts dependent observations

**Failure mode:** sliding-window updates count the same row's contribution twice across overlap, inflating support; subtraction on exit is correct in expectation but introduces variance the Jirak floor doesn't bound.

**Mitigation:** D-ARM-3 implements non-overlapping windows by default. Sliding (overlapping) windows are a feature flag with a tagged Jirak adjustment. Bench in D-ARM-12 tests both.

**Confidence:** HIGH — same fix as any windowed-statistics implementation; well-trodden.

---

## 12. Success criteria

**Quantitative:**

- D-ARM-3 (pair-stats) sustains ≥ 100K rows/window/sec on a 16-core machine for k = 200 features (bench in D-ARM-12).
- D-ARM-5 (hypothesis test) round-trips a candidate triple through revision in < 10 µs against an in-memory SpoStore of 1 M triples.
- D-ARM-12's end-to-end test pipeline (synthetic feed → proposer → translator → hypothesis-test → ratification → op_emitter) completes for 200K rows + 50 candidate triples in < 5 s.
- Zero candidate rules pass Stage A with `support × n < jirak_min_evidence(...)`.

**Qualitative:**

- Council ratification verdicts (Stage D) provide observable evidence that the discovery leg surfaces *non-obvious* co-correlations the AST extraction missed (e.g. a partner-country / product-category / fiscal-position triple invariant that is implicit in 5 years of `account.move` history but never spelled out in `account_fiscal_position.py`).
- The corrections proposed in §7 are absorbed by #434's follow-up PR without contention; OQ-11.2 and OQ-11.5 close.
- `lance-graph-arm-discovery` keeps its public surface stable across the v1.0 → v1.1 transition (no breaking API change required by the `discovery_arc` column landing).

**Doctrinal:**

- The substrate stays lossless. ARM-discovered triples enter the SPO store via `SpoBuilder` only, never bypass it.
- The compile path stays deterministic. Stage D's ratification gate is the firewall; nothing nondeterministic crosses into `op_emitter`.
- The Click is preserved. Novel candidates ARE the Epiphany branch; ratified candidates become Commits; contradictions are preserved as committed contradictions.
- `I-NOISE-FLOOR-JIRAK` is the floor, not a guideline. The proposer's noise floor IS Jirak; nothing leaks below it.

---

## 13. Cross-refs

| Doc | Section | Relation |
|---|---|---|
| `unified-soa-convergence-v1.md` (PR #434) | §11.1 — One SoA, never transformed | This plan's writes go through `SpoBuilder` → mailbox SoA, never a parallel DTO. |
| `unified-soa-convergence-v1.md` | §11.2 — witness IS belief-state arc | Stage C revisions emit `CausalEdge64` onto the witness arc. |
| `unified-soa-convergence-v1.md` | §11.6 — nine half-baked consumers | This plan adds an **upstream proposer node** to the architecture; doesn't touch any of the nine. |
| `unified-soa-convergence-v1.md` | OQ-11.2 + OQ-11.5 | §7 of THIS plan supplies the spec defaults: `discovery_arc D=8`, `discovery_origin u8`. |
| `odoo-business-logic-blueprint-v1.md` | D-ODOO-BP-1g (JITson recipes) | ArmDiscovered triples flow into the same recipe path; Stage E IS the JITson hand-off. |
| `odoo-source-extraction-v1.md` | EXT-* deliverables | Sibling proposer leg; same downstream substrate; different upstream source. |
| `style_recipe.rs` (PR #433) | derive_style_recipe rules 1-7 | This plan proposes rule 8 (D-ARM-11) for ArmDiscovered backing. |
| `op_emitter.rs` (this branch, pre-merge) | `bucket_corpus`, `emit_op_dispatch` | This plan proposes the one-line ratification filter (D-ARM-10). |
| CLAUDE.md `I-NOISE-FLOOR-JIRAK` | iron rule | §4 of this plan operationalizes it as the Stage A threshold. |
| CLAUDE.md `I-SUBSTRATE-MARKOV` | iron rule | The NARS revision arc IS the Markov-chain trajectory; ARM doesn't perturb the Markov property. |
| CLAUDE.md `I-VSA-IDENTITIES` | iron rule | ARM-discovery operates on identity-typed `(s,p,o)` triples; never bundles content. |
| CLAUDE.md `E-BATON-1` | epiphany | Stage C emissions are batons; cross-mailbox propagation rides existing baton handoff. |
| CLAUDE.md "The Click" | doctrine | Novel candidates → Epiphany; revised priors → Commit; conflicts → committed Contradiction. |
| EPIPHANIES `E-INTERPRET-NOT-STORE-1` (PR #433 council-ratified) | epiphany | ARM is *one* interpretation projection of the lossless triplet substrate; multiple projections can coexist. |
| EPIPHANIES candidate `E-DISCOVERY-CODEGEN-BRACKET-1` (this session, council-pending) | candidate | The two papers (Aerial+ + ontology M2M) bracket our architecture: discovery upstream, codegen downstream, SPO+NARS middle. |
| Paper — Karabulut, Groth, Degeler 2025 (arxiv 2504.19354v1) | §2 (truth definitions) | Direct mapping ARM → NARS truth. |
| Paper — Karabulut, Groth, Degeler 2025 | §3.3 + Algorithm 1 | Aerial+ rule extraction; D-ARM-9 IPC client interface mimics this output shape. |
| Paper — Abreu, Cruz, Guerreiro 2025 (arxiv 2511.13661v1) | §4 ("from code-centric to ontology-driven") | Independent confirmation of the externalize-interpretation doctrine. |
| Jirak 2016 (arxiv 1606.01617) | Theorem 2.1 | The weak-dependence Berry-Esseen rate that D-ARM-7 cites for the noise floor. |

---

## 14. What this plan does NOT cover

- **Reverse ARM** (mining BPMN diagrams back into proprietary source) — explicitly out of scope. The Abreu et al. paper §7.1 names this as future work; we inherit that scope.
- **Multi-feed coordination** — running 3 parallel feeds (Odoo invoices + MedCare encounters + WoA workflows) and synthesizing across them is v2 work. v1 is single-feed.
- **Continuous council** — running the council in a polling loop instead of session-triggered batches is explicitly deferred (OQ-ARM-9 default). The council was designed for human-in-the-loop verification; automating it without a contract surface change would dilute the gate.
- **GPU-accelerated pair-stats** — v1 is CPU-only. SIMD via existing `ndarray::simd_soa.rs` (when D-MBX-7 lands) is the first acceleration step. GPU is v2+.
- **Aerial+ Rust port** — the autoencoder stays in Python. D-ARM-9 only specifies the IPC client. A future plan may add a Burn/Candle port if the IPC overhead becomes the bottleneck (unlikely at 100K rows/window).

---

## 15. Invariants this plan inherits

| Invariant | Source | How this plan respects it |
|---|---|---|
| **I-NOISE-FLOOR-JIRAK** | CLAUDE.md iron rule | Mandatory Stage A threshold via D-ARM-7. |
| **I-VSA-IDENTITIES** | CLAUDE.md iron rule | ARM operates on `(s,p,o)` identity-typed triples; content stats inform the truth value, never the bind. |
| **I-SUBSTRATE-MARKOV** | CLAUDE.md iron rule | NARS revision IS Chapman-Kolmogorov-respecting; bundle math untouched. |
| **I-LEGACY-API-FEATURE-GATED** | CLAUDE.md iron rule | Aerial+ IPC behind `arm-aerial` feature; default trunk (pair-stats) is feature-clean. |
| **E-SOA-IS-THE-ONLY** | EPIPHANIES PR #434 | Writes go through `SpoBuilder` → mailbox SoA, no parallel DTO. |
| **E-BATON-1** | EPIPHANIES PR #418 | Stage C emissions cross mailbox boundaries as discrete owned batons (`CollapseGateEmission`). |
| **E-INTERPRET-NOT-STORE-1** | EPIPHANIES PR #433 | ARM is one interpretation projection; never stored back into triples; deterministically re-derivable. |
| **E-NORMALIZED-ENTITY-1** | EPIPHANIES 2026-05-28 | `CandidateTriple` carrier is a typestate over `Proposed → Tested → Ratified` — same shape pattern. |
| **The Click P-1** | CLAUDE.md | Novel candidates → Epiphany branch; revisions → Commit branch; contradictions → preserved as committed. |
| **AGI-as-glove** | CLAUDE.md Stance | No new traits-on-the-side; `Proposer` lives in `lance-graph-contract` next to the other domain contracts. |

---

## 16. Sequencing against unified-soa-convergence (PR #434)

This plan is **strictly additive** to PR #434. The dependency direction:

```text
PR #434 (unified-SoA convergence) — landed
   │
   │  D-MBX-A1 columns (edges, qualia, meta, entity_type)
   │  D-MBX-A2/A3 (BindSpace gap + witness-arc handle) — pending
   │
   ▼
this plan (ARM discovery) — proposes
   │
   │  Wave 1-4 land regardless of D-MBX-A3 (uses existing edges arc)
   │  Wave 5a (D-ARM-6 corrections) depends on D-MBX-A3 having landed
   │
   ▼
Future v1.1 — discovery_arc column + multi-feed coordination
```

If #434's D-MBX-A3 lands BEFORE this plan's Wave 5a, the corrections fold cleanly. If this plan's Wave 4 lands FIRST (using the existing `edges` arc only), the discovery leg is operational; Wave 5a just adds room.

No blocking dependency in either direction. Both plans can progress in parallel; the SoA contract is the integration surface.

---

## 17. Decision log

| Date | Decision | Rationale |
|---|---|---|
| 2026-05-29 | Author plan as v1, additive to PR #434 | Discovery is *upstream* proposer; doesn't touch the SoA contract layer. |
| 2026-05-29 | Pair-stats is the default trunk, Aerial+ is fan-in | Determinism preferred; neural compression earns its keep only at high k. |
| 2026-05-29 | Aerial+ stays in Python via IPC | Avoids ONNX/Burn dep in `lance-graph`; preserves the determinism boundary. |
| 2026-05-29 | Reject `ProvenanceTier::Auto` (auto-ratification) | Council gate is non-negotiable per The Click. |
| 2026-05-29 | Reject continuous council polling | Human-in-the-loop is the design; auto-poll dilutes the gate. |
| 2026-05-29 | Defer `discovery_arc` column to v1.1 | Live with single `edges` arc contention until bench (D-ARM-12) measures. |
| 2026-05-29 | Add `discovery_origin: u8` byte in v1 | Cheap (N bytes); council `prior-art-savant` needs it to triage. |
| 2026-05-29 | Hard-cap antecedent bound `a ≤ 2` in pair-stats | Memory bound; `a = 3` requires Aerial+ subprocess fan-in. |
| 2026-05-29 | Symmetric pair contradiction commit (OQ-ARM-6) | Matches existing `CausalEdge64` symmetry; back-pointer is natural. |

---

## 18. Provenance

Authored by main thread during session-continuation on `claude/activate-lance-graph-att-k2pHI` after PR #433 (style_recipe + epiphany council) merged and PR #434 (unified-SoA convergence) merged. Both papers (Karabulut 2025; Abreu 2025) shared by user in this session; integration emerges from cross-reading them against the existing op_emitter pipeline.

No subagents spawned for this plan. Council recommended to ratify candidate epiphany `E-DISCOVERY-CODEGEN-BRACKET-1` before this plan moves to Wave 1.

End of `streaming-arm-nars-discovery-v1.md`.
