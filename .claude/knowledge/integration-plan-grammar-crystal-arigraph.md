# Integration Plan — Grammar × Crystal × AriGraph

> **Scope:** the contract-layer grammar/crystal additions + the in-tree
> AriGraph unbundling hooks. Consolidates epiphanies from the 2026-04-18
> session into a single shipping plan.
> **Status:** contract additions in progress on branch
> `claude/teleport-session-setup-wMZfb`.

---

## The Epiphanies (what we learned this session)

### E1 — Grammar-tiered routing is the right architecture

DeepNSM + rule tables handle 90–99 % of parses **locally** at sub-10 µs.
LLM fallback only fires on the 1–10 % remainder, and when it does, it
sees a **surgical ambiguity ticket**, not the full sentence. This inverts
the default LLM-first pipeline: local grammar is the hot path, LLM is
an exception handler.

### E2 — Grammar-heavy languages are *easier*, not harder

Finnish (15 cases), Hungarian (18), Turkish agglutination, etc. encode
slot semantics directly in morphology. `-ssa` → Lokal "in"; `-sta` → Lokal
"from"; `-lla` → Adessive "at/by". A case-ending lookup table replaces
parse ambiguity. German *Wechsel* prepositions that are ambiguous in
English become unambiguous once the Finnish translation of the same
entity is in the bundle.

### E3 — FailureTicket as structured LLM handoff

When the local parser fails, it emits a `FailureTicket` containing:

- partial parse (resolved vs unresolved tokens, coverage score)
- attempted NARS inference + recommended next inference
- 2³ causal-trajectory ambiguity mask (SPO role swap space)
- TEKAMOLO slot fillings so far
- list of Wechsel ambiguities with candidate roles

The LLM never reads the sentence again — it resolves exactly the slots
the parser couldn't.

### E4 — Cross-linguistic superposition via VSA bundling

Google / Wikidata / Wikipedia the same entity in EN + FI + RU + TR,
parse each, and **XOR-bundle the fingerprints**. Where English leaves a
Wechsel ambiguous, Finnish case morphology collapses it. The superposed
fingerprint carries the disambiguation "for free" — no LLM call needed.

### E5 — Markov ±5 context chain with replay

Each sentence crystal is surrounded by ±5 neighbors in a
[`ContextCrystal`]. When a token parse is ambiguous, the chain is
**replayed** (forward or backward) with the ambiguous branch set each
way. The branch that keeps NARS confidence coherent wins.

### E6 — NARS reasons about grammar

Grammar resolution is reasoning, not pattern matching. The seven NARS
inferences map onto grammar decisions:

| Inference              | Grammar use                                 |
|------------------------|---------------------------------------------|
| Deduction              | Rule-clear case ending                      |
| Induction              | Generalize across similar sentences         |
| Abduction              | Best explanation for surface form           |
| Revision               | Update belief from new evidence             |
| Synthesis              | Bind cross-domain signals                   |
| Extrapolation          | Extend known pattern to novel input         |
| Counterfactual Synthesis | "What if the Wechsel were X?"              |

### E7 — SentenceCrystal umbrella

The crystal hierarchy unifies previously-separate objects:

```
SentenceCrystal   — one parsed sentence + triples + TEKAMOLO
ContextCrystal    — Markov ±5 window around a sentence
DocumentCrystal   — full document
CycleCrystal      — one cognitive cycle (observe → act → feedback)
SessionCrystal    — full conversation / agent session
```

All share the `Crystal` trait (hardness, revision_count, crystallized_at,
fingerprint, truth).

### E8 — CrystalFingerprint polymorphism + lossless passthrough

Four native forms, all mutually translatable:

| Variant          | Size  | Role                                         |
|------------------|-------|----------------------------------------------|
| `Binary16K`      | 2 KB  | Compact semantic (Hamming similarity)        |
| `Structured5x5`  | 3 KB  | Rich native form (5×5×5×5×5 cells + quorum)  |
| `Vsa10K_I8`      | 10 KB | lancedb-native VSA (int8)                    |
| `Vsa10K_F32`     | 40 KB | lancedb-native VSA (f32)                     |

**Key correction from session:** the 10 K variants are **not** "wire only."
lancedb famously supports 10 000-D VSA natively. The passthrough is
**lossless bundling** between Structured5x5 ↔ VSA10K, not a format
conversion. Structured5x5 is the native rich form; VSA10K is native
storage.

### E9 — 5^5 cells + 5D quorum

Axes: Element × SentencePosition × Slot × NarsInference × StyleCluster.
3125 cells in 3 KB. The optional `Quorum5D` carries per-axis consensus
∈ [0, 1] — tells us on which dimension the crystal is contested.

### E10 — Episodic memory unbundles when hardened

Young crystals live bundled (cheap, lossy). When NARS revision drives
hardness past [`UNBUNDLE_HARDNESS_THRESHOLD`] (≈ 0.8), the crystal
**unbundles** into individually-addressable facts in episodic memory.
Cold facts **re-bundle** to reclaim space. This is self-tuning storage
driven by actual evidential weight.

### E11 — AriGraph is the substrate

In-tree at `crates/lance-graph/src/graph/arigraph/`: 4,696 LOC, 7 files,
zero todos, transcoded from Python `AdaWorldAPI/AriGraph` (which remains
as the harvesting repo upstream). Handles episodic memory + triplet
graph + retrieval + sensorium + orchestrator + xai_client + language.
The crystal unbundling hooks land here.

### E12 — Demo matrix is already wireable

Three demos share the same substrate, different ingestors:

```
Chess    (precision)  ruci + lichess-bot + AriGraph + contract + cockpit
Wikidata (scale)      wikidata-ingest → AriGraph, 14.4 GB in RAM
OSINT    (applied)    spider-rs + reader-lm + DeepNSM + AriGraph
```

---

## What Ships in the Contract PR

**Module layout added to `crates/lance-graph-contract/src/`:**

```
grammar/
  mod.rs               — exports + LOCAL_COVERAGE_THRESHOLD
  ticket.rs            — FailureTicket, PartialParse, CausalAmbiguity
  tekamolo.rs          — TekamoloSlots, TekamoloSlot enum
  wechsel.rs           — WechselAmbiguity, WechselRole enum
  finnish.rs           — FinnishCase (15), suffix → case lookup,
                         TEKAMOLO hints, cross-lingual role mapping
  inference.rs         — NarsInference (7), inference → StyleCluster
  context_chain.rs     — ContextChain, ReplayRequest, ReplayDirection

crystal/
  mod.rs               — Crystal trait, CrystalKind, TruthValue,
                         UNBUNDLE_HARDNESS_THRESHOLD
  fingerprint.rs       — CrystalFingerprint enum (4 variants),
                         Structured5x5, Quorum5D,
                         bundle_vsa10k_f32 / unbundle_structured_from_vsa10k
  sentence.rs          — SentenceCrystal + Triple
  context.rs           — ContextCrystal (Markov ±5)
  document.rs          — DocumentCrystal
  cycle.rs             — CycleCrystal (anchor-tagged)
  session.rs           — SessionCrystal
```

**Invariants:** zero deps (contract crate invariant preserved), serde
kept out of types by project convention, tests in-file.

**Estimated diff:** ~1,100 LOC net new types + ~60 LOC tests.

## What Ships in the AriGraph PR (follow-up)

On `crates/lance-graph/src/graph/arigraph/episodic.rs`:

```rust
impl EpisodicMemory {
    /// Scan all bundled crystals; unbundle each whose hardness exceeds
    /// UNBUNDLE_HARDNESS_THRESHOLD into individually-addressable facts.
    pub fn unbundle_hardened(&mut self) -> UnbundleReport { ... }

    /// Unbundle exactly one crystal identified by fingerprint.
    pub fn unbundle_targeted(&mut self, fp: &CrystalFingerprint)
        -> UnbundleReport { ... }

    /// Opposite direction: compact cold facts back into a bundle.
    pub fn rebundle_cold(&mut self, cutoff: u64) -> RebundleReport { ... }
}

pub struct UnbundleReport { pub crystals_unbundled: u32, pub facts_emitted: u64, ... }
pub struct RebundleReport { pub facts_compacted: u64, pub crystals_created: u32, ... }
```

**Estimated diff:** ~200 LOC on episodic.rs + tests.

## Out of Scope (for these PRs)

- PGN ingester and chess vertical (separate plan in
  `chess-nars-vertical-slice.md`).
- Wikidata ingest (`wikidata-spo-nars-at-scale.md`).
- OSINT pipeline (`osint-pipeline-openclaw.md`).
- DeepNSM grammar hooks — the contract defines types, DeepNSM wiring is
  a follow-up so downstream can consume `FailureTicket` at the parser
  boundary.
- lance-graph-cognitive grammar/ triangle integration — it lives in a
  different crate; the contract types are the shared vocabulary.

## Order of Operations

1. Contract PR: grammar/ + crystal/ modules, zero-dep, compiling, tested.
2. Open with absolute PR # (referenced as PR #NNN once assigned).
3. AriGraph PR: episodic.rs unbundle/rebundle, wire to the contract's
   `CrystalFingerprint` + `UNBUNDLE_HARDNESS_THRESHOLD`.
4. Follow-ups (separate sessions): DeepNSM parser emits `FailureTicket`;
   cognitive grammar/ triangle consumes the unified types; Finnish +
   cross-linguistic bundling as the first concrete consumer.
