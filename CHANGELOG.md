# Changelog — Format Switches and Architectural Corrections

> **Purpose:** Track every change to core data-format choices (VSA
> variants, fingerprint widths, algebra selections, role-key layouts)
> and the architectural reasoning behind them. This is the canonical
> place to answer "when did X format change and why?"
>
> **Scope:** Format-level changes only. Per-PR feature history lives
> in `.claude/board/PR_ARC_INVENTORY.md`. Per-deliverable status lives
> in `.claude/board/STATUS_BOARD.md`. Architectural iron rules live
> in `CLAUDE.md` § Substrate iron rules.
>
> **Format:** Reverse chronological. Each entry: date, change,
> rationale, affected files, cross-refs. Superseded entries stay —
> they are the format arc.

---

## 2026-04-21 — CORRECTION: Revert `Vsa10k = [u64; 157]` bitpacked + XOR

**Status:** Cleanup shipped on branch `claude/vsa-switchboard-cleanup-2026-04-21`, commit `cd5c049...`

**What was removed:**

- `contract::grammar::role_keys::Vsa10k` type alias
- `contract::grammar::role_keys::VSA_ZERO` constant
- `contract::grammar::role_keys::RoleKey::{bind, unbind, recovery_margin}` methods
- `contract::grammar::role_keys::{vsa_xor, vsa_similarity}` free functions
- `contract::grammar::role_keys::{word_slice_mask, slice_matching_bits}` helpers
- `deepnsm::content_fp` (new module, deleted entirely)
- `deepnsm::markov_bundle` (new module, deleted entirely)
- `deepnsm::trajectory` (new module, deleted entirely)
- `deepnsm` feature flag `grammar-10k`
- 8 orphan tests that covered the deleted methods

**Why:** Session 2026-04-21 introduced `Vsa10k = [u64; 157]` as a
bitpacked binary VSA carrier with GF(2)/XOR algebra. This was a
Frankenstein — it never matched the rest of the stack:

- `ndarray::hpc::vsa` defined `VSA_WORDS = 157` but nothing in
  lance-graph consumed it before this session.
- Existing `crystal::fingerprint::Vsa10kF32 = Box<[f32; 10_000]>`
  (40 KB) was the canonical VSA carrier with element-wise multiply/
  add algebra (`vsa_bind`/`vsa_bundle`/`vsa_cosine`).
- `Binary16K = Box<[u64; 256]>` (2 KB) was the Hamming-comparison
  format — different purpose (comparison, not bundling).
- The session's 5-role "lossless superposition" test passed because
  of slice-isolation, not because XOR bundling was lossless. With
  true shared-space f32 multiply/add, losslessness comes from f32
  dynamic range, a completely different mechanism.

Two prior tech debt entries from 2026-04-19 had already flagged this:

- P1: "VSA substrate renaming: Vsa10k* → Vsa16k* + float framing"
- P0: "FP_WORDS = 157 (not 160); SIMD remainder loops remain"

Both were ignored when the session introduced the bitpacked format.
The cleanup reverts to pre-session state for the affected files.

**Verification:** 167 contract tests + 46 deepnsm tests pass. Full
workspace `cargo check` clean.

**Cross-refs:**
- `.claude/knowledge/vsa-switchboard-architecture.md` — three-layer
  architecture (carrier + domain catalogues + content stores)
- `CLAUDE.md § I-VSA-IDENTITIES` — iron rule: VSA operates on
  identities, not content
- `.claude/board/EPIPHANIES.md` — CORRECTION-OF entries dated 2026-04-21
- `.claude/board/TECH_DEBT.md` — new P0 entry "D5 deepnsm files built
  on wrong VSA substrate"

---

## 2026-04-21 — PROPOSED: Vsa10k → Vsa16k coordinated rescale (not yet shipped)

**Status:** Queued as P1 tech debt. Blocked on coordinated cross-repo PR.

**What:** Rename + rescale the float VSA carrier family:

| Before | After |
|---|---|
| `Vsa10kF32 = Box<[f32; 10_000]>` (40 KB) | `Vsa16kF32 = Box<[f32; 16_384]>` (64 KB) |
| `Vsa10kI8 = Box<[i8; 10_000]>` (10 KB) | `Vsa16kI8 = Box<[i8; 16_384]>` (16 KB) |
| — (new) | `Vsa16kBF16 = Box<[bf16; 16_384]>` (32 KB) — AMX-accelerated |
| — (new) | `Vsa16kF16 = Box<[f16; 16_384]>` (32 KB) — Apple M-series / ARMv8.2+ |

**Why:** See `.claude/knowledge/vsa-switchboard-architecture.md` §
"When to Use Which Format." Key wins:

1. `Binary16K` (16,384 bits) ↔ `Vsa16kF32` (16,384 f32) passthrough
   becomes 1:1 lossless instead of "1.6 bits per dim" approximation.
2. +60% role capacity (Johnson-Lindenstrauss bound: d / log(N)
   orthogonal items).
3. Power-of-2 role slice boundaries possible: SUBJECT[0..4096),
   PREDICATE[4096..8192), OBJECT[8192..12288), remaining 4096 for
   TEKAMOLO/NARS. Aligns with AVX-512 SIMD loads.

**NOT a SIMD-alignment fix for floats.** Both 10,000 and 16,384 f32
are AVX-512-clean (`/ 16` exact). The 157→160 u64 alignment debt was
a bitpacked-binary concern only.

**Cost:** +24 KB per vector. Working sets still fit RAM for Animal
Farm (40K sentences × 64 KB = 2.5 GB); heavy for OSINT at 1M+ scale
(quantize to `Vsa16kI8` or `Binary16K` for persistence).

**Blockers:**
- Coordinated ndarray PR: `VSA_DIMS = 16_384`, `VSA_WORDS = 256`
- lance-graph-contract PR: rename `Vsa10k*` → `Vsa16k*`, rescale
  arrays, sandwich constants, passthrough functions, role-key slices
- Recalibrate hand-tuned σ thresholds (or replace with Jirak-derived
  bounds per the P1 probe)

**Affected files (estimate):** ~40 LOC ndarray, ~200 LOC contract,
~28 `.claude/*.md` docs for cross-ref cleanup.

**Cross-refs:**
- `.claude/board/TECH_DEBT.md` 2026-04-21 entry "Vsa10k→Vsa16k
  coordinated rescale"
- `cross-repo-harvest-2026-04-19.md` H6
- Original P1 debt from 2026-04-19

---

## 2026-04-21 — NEW: Three-layer VSA switchboard architecture (documentation)

**Status:** Documentation shipped. Code implementation pending D5
rewrite on correct substrate.

**What:** Formalize the three-layer separation that was conflated
before:

**Layer 1 — Switchboard carrier (domain-agnostic, in `crystal/`):**
- One set of types: `Vsa16kF32` / `BF16` / `F16` / `I8`, `Binary16K`
- One algebra: `vsa_bind` (multiply), `vsa_bundle` (add), `vsa_cosine`
- Transitions: `Vsa16kF32 ↔ Binary16K` (sign-binarize), `Vsa16kF32 ↔
  Vsa16kI8` (quantize), `Vsa16kF32 ↔ Vsa16kBF16` (precision reduce)

**Layer 2 — Domain role catalogues (per-domain):**
- `grammar/role_keys.rs` — SUBJECT, PREDICATE, OBJECT, MODIFIER,
  TEMPORAL, KAUSAL, MODAL, LOKAL, INSTRUMENT, BENEFICIARY, GOAL,
  SOURCE, 15 Finnish cases, 12 tenses, 7 NARS inferences
- `persona/role_keys.rs` (future) — MODAL, AFFECTIVE, TONE, REGISTER,
  STANCE, GENDER, AGE_GROUP, FORMALITY
- `callcenter/role_keys.rs` (future) — INTENT, SENTIMENT,
  AGENT_ACTION, URGENCY, ESCALATION_TRIGGER
- `archetype/role_keys.rs` (future) — unified with existing palette
  archetypes (256/plane in bgz17) + VoiceArchetype (16 channels)

Each catalogue allocates disjoint `[start:end)` slice ranges of the
carrier. Catalogue only — no algebra. Methods live on the carrier.

**Layer 3 — Content stores (per-domain, NOT VSA):**
- `thinking_styles/*.yaml` — style configs (NARS priority, morphology
  tables, TEKAMOLO priority)
- `persona/*.yaml` (future) — persona slots, prompts, behavior rules
- `callcenter/intents.yaml` (future) — intent definitions, resolution
  flows, escalation rules
- `TripletGraph` — SPO facts with NARS truth + Pearl 2³ mask
- `EpisodicMemory` — episode snapshots with ±5 context

**Why:** Pre-cleanup, the session put Layer-1 algebra (bind/unbind)
on Layer-2 RoleKey struct in grammar/ — conflating the carrier with
a specific domain. The switchboard framing separates these:
**carrier + algebra is universal; role catalogues are per-domain;
content lives in named stores with O(1) lookup.**

**Cross-refs:**
- `.claude/knowledge/vsa-switchboard-architecture.md` — full three-
  layer architecture + four tests before VSA + CAM vs VSA matrix
- `CLAUDE.md § I-VSA-IDENTITIES` iron rule

---

## 2026-04-21 — NEW iron rule: I-VSA-IDENTITIES

**Status:** Shipped in `CLAUDE.md` § Substrate iron rules.

**What:** VSA operates on IDENTITY fingerprints that POINT TO content.
Never on content's bitpacked/quantized register itself.

**The register-loss problem:** XOR-bundling (or any superposition)
of CAM-PQ codes, quantized indices, or sign-binarized fingerprints
destroys the mapping from bits back to codebook entries.

**The four tests before VSA** (applied in order; first failure
short-circuits):

- **Test 0 — register laziness:** Does this thing have a natural
  name / ID / enum variant? If yes, use the register.
- **Test 1 — bundle size:** Is N ≤ √d / 4 ≈ 32 at 16K dim?
- **Test 2 — role orthogonality:** Are role keys mutually orthogonal?
- **Test 3 — cleanup codebook:** Is there a known codebook to match
  against after unbind?

**Consequences:** CAM-PQ and VSA are separate tools. Switching
requires decompression, not mixing. Persona / callcenter / archetype
/ thinking-style catalogues all follow the identity-fingerprint +
YAML-content pattern.

**Cross-refs:**
- `CLAUDE.md § I-VSA-IDENTITIES` — full rule text
- `.claude/knowledge/vsa-switchboard-architecture.md` § Identity vs Content

---

## 2026-04-20 (pre-cleanup) — `CrystalFingerprint` variants established

**Status:** Unchanged. This is the correct pre-existing format.

**Variants:**

| Variant | Layout | Bytes | Purpose |
|---|---|---|---|
| `Binary16K` | `Box<[u64; 256]>` | 2 KB | Hamming similarity comparison (popcount-fast) |
| `Structured5x5` | `Box<[u8; 3125]>` + optional `Quorum5D` | 3 KB | Named-cell introspection (Element × Position × Slot × Inference × Style) |
| `Vsa10kI8` | `Box<[i8; 10_000]>` | 10 KB | Quantized VSA (lancedb-native, i8) |
| `Vsa10kF32` | `Box<[f32; 10_000]>` | 40 KB | **Lossless VSA bundling** (f32 multiply/add) |

**Algebra** (in `crystal/fingerprint.rs`):
- `vsa_bind(a, b) = a[i] * b[i]` — element-wise multiply (self-inverse for ±1 bipolar)
- `vsa_bundle(vecs) = Σ v[i]` — element-wise add (lossless within f32 dynamic range)
- `vsa_superpose(vecs, weights)` — weighted add
- `vsa_cosine(a, b) = dot(a, b) / (||a|| · ||b||)` — similarity

**Transitions:**
- `Vsa10kF32 ↔ Binary16K` — 1:1 bit-to-dim lossy passthrough (~1.6 bits/dim)
- `Vsa10kF32 ↔ Vsa10kI8` — lossless up to i8 precision
- `Structured5x5 ↔ Vsa10kF32` — 3,125 cells + 5 quorum floats, lossless roundtrip

**This was the correct substrate all along.** The 2026-04-21 session
reverted cleanup above restores use of this substrate for all VSA
workloads.

---

## References

- `CLAUDE.md` — iron rules (I-SUBSTRATE-MARKOV, I-NOISE-FLOOR-JIRAK,
  I-VSA-IDENTITIES)
- `.claude/knowledge/vsa-switchboard-architecture.md` — three-layer
  architecture + decision matrices
- `.claude/knowledge/encoding-ecosystem.md` — full encoding map
  (MANDATORY before any codec/encoding work)
- `.claude/board/TECH_DEBT.md` — open format debts (Vsa10k→Vsa16k
  rename, 157→160 SIMD, Jirak thresholds)
- `.claude/board/EPIPHANIES.md` — format-related epiphanies with
  "why this dilutes" warnings
- `.claude/plans/categorical-algebraic-inference-v1.md` — Five Lenses
  meta-architecture (algebra-level, carrier-independent)
