# Epiphanies — Append-Only Log (date-prefixed)

> **APPEND-ONLY.** Every epiphany, realization, correction, or
> "aha" moment gets a dated entry here so nothing gets lost between
> sessions. Reverse chronological (newest first). Never delete an
> entry; correct via a new entry that cites the old one.
>
> **Format invariant:** every entry begins with a `## YYYY-MM-DD —`
> header. A CONJECTURE / FINDING / CORRECTION-OF label is optional
> but encouraged. Body is short: one paragraph + optional
> cross-reference. Long material goes in a dedicated knowledge
> doc; the epiphany here is the **pointer + one-line claim**.
>
> Mutable field: `**Status:**` line (FINDING / CONJECTURE /
> SUPERSEDED) is the only thing in an entry that can be updated.
> Everything else is immutable.

---

## How to use

**When a new insight surfaces** — stop, prepend an entry with today's
date at the top of the "Entries" section below. One paragraph. If
the full idea needs more room, create a dedicated knowledge doc
and reference it from the epiphany entry.

**When an old epiphany is wrong** — prepend a new entry labeled
`CORRECTION-OF YYYY-MM-DD <title>` and update the old entry's
`**Status:**` line to `SUPERSEDED by <new-entry>`. Never edit the
old body.

**When reading the log** — top N entries are the recent thinking;
deeper entries are the accumulated substrate. Everything is there.

---

## Prior art (pre-existing epiphany collections — do not duplicate)

These files already hold numbered epiphany sets from earlier work.
New epiphanies go in **this file** with date prefix; the files below
stay as historical references.

| File | Contents |
|---|---|
| `linguistic-epiphanies-2026-04-19.md` | E13–E27 (Chomsky hierarchy, Σ10 Rubicon, sigma_rosetta, Markov living frame, resonanzsiebe, method grammar, 4D hashtag glyph, membrane, verbs as productions) |
| `cross-repo-harvest-2026-04-19.md` | H1–H14 (Born rule, phase-tag threshold, interference truth, Grammar Triangle ≡ ContextCrystal(w=1), NSM ≡ SPO axes, FP_WORDS=160, Mexican-hat, Int4State, Glyph5B, Crystal4K, teleport F=1, 144-verb, Three Mountains) |
| `integration-plan-grammar-crystal-arigraph.md` | E1–E12 (grammar-tiered, morphology-easier, FailureTicket, cross-lingual superposition, Markov ±5, NARS-about-grammar, crystal hierarchy, sandwich, 5D quorum, episodic unbundle, AriGraph substrate, demo matrix) |
| `session-capstone-2026-04-18.md` | 8 epiphanies from 2026-04-18 session (four-pillar inheritance, CMYK/RGB qualia, vocabulary IS semantics, WorldMapRenderer, Σ hierarchy maps to crate boundaries, proprioception as ontological self-recognition, BindSpace+cycle_fingerprint as latent episodic, two-frame DTO) |
| `crystal-quantum-blueprints.md` | Crystal mode vs Quantum mode split (bundled Markov SPO chain vs holographic residual) |
| `endgame-holographic-agi.md` | 5-layer stack, 12-step holographic memory loop, three-demo matrix |
| `fractal-codec-argmax-regime.md` | Orthogonal research thread — MFDFA on Hadamard-rotated coefficients as fractal-descriptor leaf |

## Governance

- **APPEND-ONLY.** Immutable body per entry.
- **Mutable:** `**Status:**` line only (FINDING / CONJECTURE /
  SUPERSEDED by <date-title>).
- **Corrections APPEND as new dated entries.** The old entry's
  Status changes to SUPERSEDED.
- **`permissions.ask` on Edit** (same rule as `PR_ARC_INVENTORY.md`
  / `LATEST_STATE.md` — rewriting history prompts for approval;
  Write for append stays unprompted).

---

## Entries (reverse chronological)

## 2026-04-19 — Mandatory epiphanies log (this file)

**Status:** FINDING

Every epiphany from prior sessions lived in separate doc (E1–E12
here, H1–H14 there, E13–E27 somewhere else). No single place to
append a new one. This file is the unified target going forward.
Old files stay as historical substrate; new insights land here with
date prefix. Cross-reference: `BOOT.md`, `CLAUDE.md`, `cca2a/
concepts.md` — all four bookkeeping files now plus this one.

## 2026-04-19 — Cold-start tax is solvable with three mandatory reads

**Status:** FINDING

A new session on non-trivial workspace burns 20–30 turns rediscovering
what's shipped. Three files (`LATEST_STATE.md`, `PR_ARC_INVENTORY.md`,
`.claude/agents/BOOT.md`) + SessionStart hook closes the gap to
3–5 turns. Proven by PR #211. Savings per cold-start: ~$15–35 of
Opus. See `.claude/skills/cca2a/SKILL.md` for the full pattern.

## 2026-04-19 — 10,000-D f32 VSA is lossless under linear sum

**Status:** FINDING

Earlier framing of "Vsa10kF32 is wire-only passthrough" was wrong.
10K × 32 = 320 K bits of capacity ≫ any single signal; orthogonal
role keys give exact unbundle. **10K f32 is native storage**, not
passthrough. lancedb famously supports 10K-D VSA natively. Cross-ref:
PR #209 refactor.

## 2026-04-19 — Signed 5^5 bipolar is lossless; unsigned / bitpacked is lossy

**Status:** FINDING

Negative cancellation on bipolar cells is VSA-native; opposing cells
at the same sandwich dim cancel on bundling. Unsigned 5^5 saturates
under accumulation (lossy). Binary bitpacked commits to 0/1 via
majority vote (lossy). CAM-PQ projection is distance-preserving
(lossless cross-form). Cross-ref: PR #209 sandwich layout.

## 2026-04-19 — VSA convention is `[start:stop]` contiguous slices, not scattered bits

**Status:** FINDING

Role keys own disjoint contiguous slices of the 10K VSA space —
SUBJECT=[0..2000), PREDICATE=[2000..4000), etc. Binding into one
slice does not contaminate another. Scattered-bit role encoding
(early draft) was the wrong pattern. Cross-ref: PR #210 D6
role_keys.rs.

## 2026-04-19 — Finnish object marking is Nominative/Genitive/Partitive, NOT Accusative

**Status:** FINDING (CORRECTION-OF an earlier Latinate transplant)

Prior draft wrote Finnish "Accusative `-n/-t` → Object" which is
a Latinate transplant. Finnish object marking actually uses:
Nominative (plural), Genitive `-n` (total singular), Partitive
`-a/-ä` (partial / negated). True Accusative is only for personal
pronouns (`minut`, `sinut`, `hänet`, `meidät`, `teidät`, `heidät`).
Each language gets its native case terminology.
Cross-ref: `grammar-landscape.md` §4.1.

## 2026-04-19 — Morphology-rich languages are easier, not harder

**Status:** FINDING

Finnish 15 cases → 98%+ local coverage. English (word order only) →
85% (WORST case). Case endings directly encode TEKAMOLO slots;
morphology commits grammatical role at the morpheme level,
eliminating the inference English needs. Cross-ref:
`grammar-tiered-routing.md` §Morphology Coverage Table.

## 2026-04-19 — Markov ±5 is the context upgrade to NARS+SPO 2³+TEKAMOLO

**Status:** FINDING

Pre-Markov reasoning unit = sentence. Post-Markov = trajectory.
NARS doesn't reason about "this sentence"; it reasons about "this
sentence in this flow." The context dimension is the whole point.
Cross-ref: `integration-plan-grammar-crystal-arigraph.md` E5.

## 2026-04-19 — Grammar Triangle IS ContextCrystal at window=1

**Status:** FINDING

Two parallel architectures turn out to be the same thing at
different window sizes. Triangle emits `Structured5x5` with S/O
collapsed + only t=2 populated; ContextCrystal populates all 5
axes. Unification. Cross-ref:
`cross-repo-harvest-2026-04-19.md` H4,
`ladybug-rs/docs/GRAMMAR_VS_CRYSTAL.md`.

## 2026-04-19 — NSM primes map directly to SPO + Qualia + Temporal axes

**Status:** FINDING

The 65 Wierzbicka primes aren't orthogonal to SPO — they ARE an
SPO encoding. I/YOU/SOMEONE → Subject; THINK/WANT/FEEL →
Predicate; SOMETHING/BODY → Object; GOOD/BAD → Qualia.valence;
BEFORE/AFTER → Temporal; BECAUSE/IF → Causality via Markov flow.
DeepNSM + Structured5x5 already speak NSM's vocabulary.
Cross-ref: `cross-repo-harvest-2026-04-19.md` H5.

## 2026-04-19 — Chomsky hierarchy isomorphism with Pearl rungs and Σ tiers

**Status:** FINDING

Type-3 Regular = Pearl rung 1 = Σ1–Σ2 = DeepNSM FSM (LLM token
prediction lives here). Type-2 CF = rung 2 = Σ3–Σ5 = SPO 2³. Type-1
CS = rung 3–4 = Σ6–Σ8 = Markov ±5 + coref + counterfactual. Type-0
TM = rung 5 = Σ9–Σ10 = LLM escalation only. The 90–99% local /
1–10% LLM split is the Chomsky-hierarchy boundary between
context-sensitive-decidable and Turing-complete-undecidable. The
split is mathematically principled, not arbitrary.
Cross-ref: `linguistic-epiphanies-2026-04-19.md` E13, E26.

## 2026-04-19 — Grindwork vs accumulation is the subagent model split

**Status:** FINDING

Grindwork (single-source mechanical: write-file-from-spec, grep,
list paths) → Sonnet. Accumulation (multi-source synthesis:
harvest across repos, combine N docs, trace architecture) → Opus.
Cheaper tiers produce shallow outputs under accumulation; quality
drop is visible. Never Haiku.
Cross-ref: `CLAUDE.md §Model Policy`.

## 2026-04-19 — Zipball-for-reads is ~20× cheaper than MCP-per-file

**Status:** FINDING

`mcp__github__get_file_contents` drops the full file into context
and recharges on every subsequent turn. Zipball to `/tmp/sources/`
+ local grep lands only the grep output (typically 2–10 KB) vs
50 KB per file per turn. 95% savings on cross-repo harvest turns.
MCP stays for writes (PR creation, comments).
Cross-ref: `CLAUDE.md §GitHub Access Policy`.

---

(append new epiphanies above this marker; format: `## YYYY-MM-DD — <title>`)

## 2026-04-19 — Prompt↔PR ledger is 10⁷× cheaper than code grep
**Status:** FINDING
**Scope:** @workspace-primer domain:bookkeeping

To answer "what did we ship for topic X":

- **Grep across code:** ~100 MB of Rust across N crates, ~25M tokens of context, minutes of agent turns.
- **Grep the ledger:** one `grep X .claude/board/PROMPTS_VS_PRS.md` returns `<prompt file> | #N <title>`. ~25 tokens, sub-second.

Seven orders of magnitude cheaper. The pairing **prompt-file ↔ PR** is the
minimum addressable record of "this artifact was built to answer this
brief" — the hyperlink that replaces re-discovery by full-text scan.

The line is mechanical bookkeeping (Haiku-level, no synthesis). The
value accumulates on every subsequent "what about X" query thereafter:
ledger-first, code-never-unless-necessary.

Cross-ref: PR #213 (lance-graph, 41 prompts × merged PRs), PR #110
(ndarray, 25 prompts × merged PRs). Both shipped in ~90s on a dumb
enumerate+match+append loop. No code reads, no MCP, no synthesis.

## 2026-04-19 — Code-arc knowledge loss is 30-50% of session tokens (ambient)
**Status:** FINDING
**Scope:** @workspace-primer domain:bookkeeping

Empirical (per user, 2026-04-19): **30-50% of session tokens** burn on
rediscovering what code paths exist, what was tried, what got reverted,
what decisions led to the current shape. This is **orthogonal** to the
20-30-turn cold-start tax — it's the *ambient* loss across every query,
every subagent spawn, every refactor.

The ledger closes three channels at once:

| Channel | Before | After | Discount |
|---|---|---|---|
| Cold-start (once per session) | 20-30 turns | 3-5 turns | ~6× |
| Find-code (per query) | ~25M tokens (grep codebase) | ~25 tokens (grep ledger) | 10⁷× |
| **Ambient arc knowledge (every turn)** | **30-50% of session budget** | **~0%** | **2×-eternal** |

All three channels collapse to two text-file reads: PROMPTS_VS_PRS.md +
PR_ARC_INVENTORY.md. The second file is read only when arc detail is
needed (Knowledge Activation trigger), so the routine cost is 0.

Cross-ref: PRs #211-213 (CCA2A + board split + ledger). `.claude/BOOT.md`
cold-start tax. `EPIPHANIES.md` 10⁷× finding above.

## 2026-04-19 — Vector (10⁴ cells) vs Matrix (10⁸ cells): don't conflate
**Status:** FINDING
**Scope:** @workspace-primer @container-architect domain:vsa domain:memory

Entirely different objects, four orders of magnitude apart. Calling them
both "10,000 VSA" was category error.

| Object | Shape | Cells | Bytes (BF16) | Purpose |
|---|---|---|---|---|
| **16K-D wire vector** (intentional) | 1 × 16,384 | **10⁴** | 32 KB | one lossless fingerprint for wire / Markov bundle / crystal / holographic |
| **10K × 10K glitch matrix** (unintentional) | 10,000 × 10,000 | **10⁸** | 200 MB | nothing — imported debris from outdated ladybug-rs / bighorn |

The 100-million-cell matrix is ~10,000× bigger than the 10,000-cell
vector. They share only a numeric coincidence in one dimension; the
semantics, cost, and lifecycle are completely unrelated.

**Consequence for the rename PR:**

- `Vsa10kF32` → `Vsa16kBF16` migration is about the VECTOR (cheap,
  per-row, ≤32 KB).
- The 10k × 10k MATRIX deletion is a separate P0 cleanup independent
  of the substrate rename.
- Any future ledger / knowledge-doc / plan entry describing 10k-D
  HDC must specify VECTOR explicitly. "10,000-D HDC" alone is
  ambiguous — spell out "16,384-cell wire fingerprint" or "10,000-cell
  lossless wire vector" to preclude the matrix reading.

Cross-ref: TECH_DEBT "CORRECTION-OF ... 10k × 10k GLITCH MATRIX"
(2026-04-19). IDEAS REFINEMENT-2 (HDC = FP16/BF16, not FP32).

## 2026-04-19 — Working-set invariant: hot structures must fit in L3
**Status:** FINDING
**Scope:** @container-architect @cascade-architect @truth-architect domain:memory domain:codec domain:performance

Typical server L3 cache = 32-96 MB (AMD EPYC, Intel Xeon). Any hot-path
structure exceeding this size incurs DRAM latency (~100 ns) on every
miss vs L3's ~12 ns — an 8× penalty per access that compounds in
inner loops. **This is true regardless of storage capacity** — LanceDB
can hold terabytes, but what the CPU touches per cycle must fit L3.

The codec stack is architected around this invariant:

| Working structure | Size | L3 verdict | Role |
|---|---|---|---|
| Container `[u64; 256]` Hamming | 2 KB | ✓ 16,000× | Popcount fingerprint |
| 16K-D BF16 wire vector | 32 KB | ✓ 1,000× | HDC point, Markov bundle |
| 256 × 256 u8 distance table (bgz-tensor) | 64 KB | ✓ L1 | Archetype attention |
| 1024 × 1024 f32 | 4 MB | ✓ | Per-role slot |
| 4096 × 4096 u8 CAM-PQ palette | 16 MB | ✓ upper edge | Centroid distance |
| **10,000 × 10,000 f32 glitch matrix** | **400 MB** | **✗ 12× over** | **None — delete** |
| 16K × 16K BF16 | 512 MB | ✗ | Never build |
| 100K × 100K anything | ≥10 GB | ✗ | Sparse-only or CAM-PQ |

**Rule for hot tables:**

- Dense square matrices: cap at `sqrt(L3_BUDGET / cell_size)` on a side.
  At 32 MB budget, f32 cells → ~2,900 × 2,900; BF16 → ~4,000 × 4,000;
  u8 → ~5,700 × 5,700.
- Wider-than-L3 tables must be projected, quantized, or made sparse
  (CSR / HyperCSR / palette-indexed) before entering a hot path.
- 1-D vectors are cheap — a 16K-D BF16 row is 32 KB, thousands
  cache-resident simultaneously. The limit binds on 2-D dense, not 1-D.

The codec compression chain (full planes 16 KB → ZeckBF17 48 B →
Base17 34 B → PaletteEdge 3 B → CAM-PQ 6 B → Scent 1 B) exists so that
any intermediate table stays L3-resident regardless of population size.
The 10K × 10K glitch matrix violates this at the root.

Cross-ref: EPIPHANIES "Vector (10⁴ cells) vs Matrix (10⁸ cells)"
(2026-04-19). TECH_DEBT "Ladybug 10k × 10k GLITCH MATRIX" (2026-04-19).
docs/CODEC_COMPRESSION_ATLAS.md is the chain spec.

## 2026-04-19 — SUPERSEDES 2026-04-19 "Vector vs Matrix" + "L3 working-set invariant"
**Status:** SUPERSEDED (downgrade both)

Both prior entries restate invariants the workspace has known for months:

- L3 working-set cap → already the design principle behind the full
  codec chain (full planes → ZeckBF17 → Base17 → Palette → CAM-PQ → Scent).
  See `docs/CODEC_COMPRESSION_ATLAS.md`, not an EPIPHANIES entry.
- Vector-vs-matrix category distinction → trivially true, never a
  point of ambiguity in the workspace proper.

**What's actually true:**

The 10k × 10k glitch matrix exists because nobody touched the
stone-age ladybug-rs / bighorn code after it was imported. The import
itself was migration desperation — closing loose ends on the cognitive
stack before a release, not a considered architectural choice. No
one re-validated the imports against the L3 invariant because the
imports were expected to be rewritten or deleted later.

The correct framing is **legacy-hygiene debt**, not new knowledge.
Action: delete-on-touch when someone has bandwidth, not a design
principle waiting to be learned.

Downgrading both prior entries to SUPERSEDED to keep the FINDING log
clean for actual findings.

## 2026-04-19 — Fractal leaf probe NEGATIVE: w_mfs is per-tensor, not per-row
**Status:** FINDING (valid negative)
**Scope:** @cascade-architect @container-architect domain:codec domain:fractal

Probe ran on Qwen3-8B (safetensors BF16, shard 1, layer 0):

| Tensor | Rows probed | w_mfs mean | w_mfs CoV | H mean | Verdict |
|---|---|---|---|---|---|
| gate_proj | 100 of 12288 | 0.504 | **0.190** | 0.519 | ✗ flat |
| k_proj | 100 of 1024 | 0.506 | **0.197** | 0.514 | ✗ flat |

Gate was CoV(w_mfs) > 0.3. Both tensors at ~0.19 — below threshold.

**Interpretation:** after Hadamard rotation, Qwen3 weight rows are
near-white-noise (H ≈ 0.5). All rows share the same multifractal
shape; the discriminating signal is amplitude (σ) and sign pattern,
not fractal structure. Fractal descriptor per-row reduces to σ_energy
alone = 2 bytes BF16, already captured by TurboQuant's log-magnitude.

**Consequence:** 7-byte FractalDescriptor per-row doesn't crack the
argmax wall. TurboQuant/PolarQuant (per-coordinate sign + log-mag)
remains the correct argmax-regime codec. The `compute_mfdfa_descriptor`
module (PR #216) stays useful as an analysis tool and per-TENSOR
characterisation metric — but not as a per-row compression codec.

**Roadmap update:** Steps 3-6 from fractal-codec-argmax-regime.md
are gated-out by this negative. Step 2 (the module) is shipped and
valid. The FractalDescriptor leaf concept retires as a per-row codec
candidate; the 7-byte budget goes back to I8-Hadamard or PolarQuant.

Cross-ref: `.claude/knowledge/fractal-codec-argmax-regime.md`
§ Honest Uncertainty (predicted this outcome). PR #216 (module +
probe shipped).

## 2026-04-19 — CORRECTION-OF fractal leaf probe: measured magnitude, missed phase
**Status:** CORRECTION

Prior entry reported the probe as a valid negative. **That was the wrong
probe.** Per user (2026-04-19): "The point is to encode phase by doing
fractal encoding."

What MFDFA-on-coefficients measures:
- Multifractal width w, Hurst H, fractal dimension D of the |coefficient|
  magnitude distribution across scales. These are envelope statistics.

What this MISSED:
- **The sign pattern S** of Hadamard-rotated coefficients is the phase.
- Two rows with identical |c_i| distribution can have completely different
  sign patterns → completely different inner products against queries.
- Magnitude statistics are flat across rows (CoV 0.19) because trained
  weights share the envelope; what differs per-row is the phase sequence.

Correct probe: **fractal structure of the sign sequence** post-Hadamard.
- Count sign-flips per window at scales s ∈ {4, 8, 16, …, n/4}.
- Measure scaling of flip density: D_phase = log(flips) / log(scale).
- Per-row CoV(D_phase) is the real gate. Expected to be LARGE because
  sign patterns encode distinct interference directions per row.

Original prompt (fractal-codec-argmax-regime.md) DID include "sign
pattern S" as a LEAF component. The MFDFA module (PR #216) covers only
(D_mag, w, σ, H_mag) — it's half the descriptor. The other half
(phase fractal / sign-flip scaling) is still unshipped.

**Gate still open.** Fractal leaf as argmax codec is not proven wrong;
only the magnitude-only variant is. A sign-sequence fractal probe is
the actual test.

Action:
- `fractal_descriptor` stays `lab`-gated (correct call — unproven).
- Next probe: sign-sequence multifractal on same Qwen3 rows. If
  CoV(D_phase) > 0.3 → revisit the leaf codec with phase encoding.
- Prior "NEGATIVE" finding is scope-corrected: "magnitude-only fractal
  leaf is flat" — phase-fractal leaf unmeasured.

## 2026-04-19 — Fractal codec ICC measurement: DEFINITIVELY NEGATIVE (magnitude-only)
**Status:** FINDING (measured via endpoint psychometry)
**Scope:** @cascade-architect domain:codec domain:psychometry

Ran codec_rnd_bench.rs with FractalDescOnly + FractalPlusBase17 wired
as candidates. Population: q_proj L0 of Qwen3-8B [4096×4096], N=128
rows. Ground truth = pairwise cosines in f32.

**Results (ICC_3_1 is the argmax-regime metric):**

| Codec | Bytes | ICC_3_1 | Pearson r | Spearman ρ |
|---|---|---|---|---|
| Passthrough (baseline) | 0 | **1.0000** | 1.0000 | 1.0000 |
| Base17 (golden-step 17-d) | 34 | **0.0240** | 0.0742 | 0.0466 |
| **Fractal-Desc (4-D mag)** | 7 | **−0.9955** | 0.0160 | 0.0012 |
| **Fractal + Base17 blend** | 41 | **−0.4879** | 0.0748 | 0.0409 |

**Key readings:**

1. **Fractal-Desc alone anti-correlates with ground truth (ICC ≈ −1).**
   Not noise — genuinely inverse ranking. The 4-D (D, w, σ, H) descriptors
   are near-constant across rows (CoV 0.19 from earlier probe), so
   pairwise "cosine" in descriptor space is essentially noise ~0.5
   against a ground-truth distribution with heavy tails — the rank
   statistic inverts against true cosine magnitudes.

2. **Fractal ADDED to Base17 ACTIVELY HURTS it.** Base17 alone: 0.024.
   Blend 0.75*Base17 + 0.25*Fractal: −0.488. The fractal component
   doesn't just fail to add signal — it contaminates the Base17 signal.
   A codec gating system must be able to *reject* bad auxiliary
   features, not blend them.

3. **Note on Base17 at ICC 0.024 on q_proj:** confirms Invariant I2
   (near-orthogonality of Qwen3 attention projections at 1024-d+
   dimension). Base17's 17-d projection loses almost everything on
   q_proj specifically — consistent with the 67-codec sweep finding
   that i8-Hadamard at ~9 B/row is the argmax-regime leader, not
   Base17.

**Consequence for the fractal codec line of research:**

- **Magnitude-only fractal leaf is empirically dead** on q_proj at
  Qwen3 scale. Measurement complete via endpoint ICC_3_1 — no longer a
  conjecture, no longer a "wrong probe" question.
- **Phase-encoding variant (sign-sequence fractal) remains UNMEASURED.**
  Infrastructure is now wired: swap the encoding inside
  FractalDescOnly to compute fractal statistics of the sign pattern
  (flips-per-scale) and re-run. One function body change.
- **Fractal-interpolation-between-Base17-anchors** (the round-trip
  codec idea) is also still unmeasured — requires implementing
  `decode(anchors, desc) -> Vec<f32>` to feed through the bench.
  The blending approach (current FractalPlusBase17) is NOT the same
  thing; it mixes scores post-hoc rather than reconstructing the row.

**Lab gate holds.** Everything stays behind `--features lab`. Main
builds don't link fractal_descriptor. No leak risk.

Cross-ref: fractal-codec-argmax-regime.md, EPIPHANIES 2026-04-19
CORRECTION (fractal measured magnitude not phase), IDEAS 2026-04-19
"Fractal codec validation path", PR commits fc386bb / afe67e1 /
48f781e / 18c53e0.

Wall time of the full 60+ codec bench: 13 min. Downloaded: 0 B (used
cached Qwen3-8B shard from the earlier probe). Deterministic.

## 2026-04-19 — Phase-fractal codec also NEGATIVE — row-level fractal discrimination dead
**Status:** FINDING (measured via endpoint psychometry)
**Scope:** @cascade-architect domain:codec domain:psychometry

Ran codec_rnd_bench.rs with both magnitude-fractal AND phase-fractal
candidates. Same population (Qwen3-8B q_proj L0, N=128, pairwise cosines).

**Measurements (ICC_3_1 is the argmax-regime metric):**

| Codec | Bytes | ICC_3_1 | Pearson r |
|---|---|---|---|
| Passthrough baseline | 0 | **1.0000** | 1.0000 |
| Base17 (34 B anchors) | 34 | 0.0240 | 0.0742 |
| Fractal-Desc (4-D magnitude) | 7 | **−0.9955** | 0.0160 |
| **Fractal-Phase (5-D flip density)** | 5 | **−0.9972** | −0.0074 |
| Fractal + Base17 blend | 41 | −0.4879 | 0.0748 |
| Phase + Base17 blend | 39 | −0.4982 | 0.0742 |

**Key finding:** BOTH orthogonal axes of row-level fractal statistics
are flat across Qwen3 q_proj rows after Hadamard rotation.

- Magnitude envelope (D, w, σ, H): near-constant — confirmed by
  ICC ≈ −1.
- Sign-flip density profile at 5 scales: ALSO near-constant — ICC
  slightly worse at −0.9972.

**Implication:** Invariant I2 (near-orthogonality of Qwen3 rows at
1024/4096-d) means once rows are Gaussian-ish post-Hadamard, every
row-level summary statistic looks identical. Only the SPECIFIC
coordinate-by-coordinate sign/magnitude assignment discriminates, and
that cannot compress below ~full sign pattern (~1 bit/coord, ~512 B
for a 4096-d row).

**Fractal-leaf line of research is closed** for row-level-statistic
compression. Three probes completed, all negative:
  1. CoV(w_mfs) ≈ 0.19 (first cheap probe, 100 rows)
  2. ICC_3_1(Fractal-Desc) = −0.9955 (magnitude, 4-D, 128 rows)
  3. ICC_3_1(Fractal-Phase) = −0.9972 (phase, 5-D, 128 rows)

**Still-open variant (unmeasured):** fractal-interpolation-between-
Base17-anchors for ROUND-TRIP codec. That approach stores full
Base17 (17 golden-step anchors = near-full phase signature at those
points) + fractal shape params to guide interpolation BETWEEN
anchors. Doesn't rely on row-level fractal statistic discrimination.
Requires implementing `FractalCodec::decode(Base17, Descriptor)` via
IFS and registering as candidate. Unbuilt.

**Wall times:**
- First bench (2 fractal candidates): 782 s (13 min)
- Second bench (4 fractal candidates): 1354 s (22.5 min)
- Delta: ~9.5 min for 2 more candidates on 128 rows × 60+ codec sweep.

**Codec R&D sweep state post-finding:** I8-Hadamard at ~9 B/row
remains the argmax-regime leader. Fractal leaf is not on the
Pareto frontier; do not pursue row-level-statistic compression
further. Focus codec research on either:
  - Full sign-pattern preservation schemes (~512 B/row minimum).
  - Round-trip IFS from Base17 anchors (unmeasured, novel).
  - Different underlying orthogonal bases (SVD-per-group instead of
    shared Hadamard) — different basis might give different
    row-level statistics, but I2 says near-orthogonality is generic.

Cross-ref: commits 0f635e6 (phase variant), 18c53e0 (first ICC run),
fractal-codec-argmax-regime.md, EPIPHANIES 2026-04-19 prior entries.

## 2026-04-20 — Zipper codec WORKS — Hadamard sign-flip invariance was the fractal bug
**Status:** FINDING (measured via endpoint psychometry, 3 populations)
**Scope:** @cascade-architect domain:codec domain:psychometry

Ran codec_rnd_bench.rs with ZipperPhaseOnly + ZipperFull added. Three
populations on Qwen3-8B L0 (N=128, pairwise cosines, 1037 s wall).

**Root-cause diagnosis (confirmed by user, validated by measurement):**

All prior fractal descriptors (magnitude + phase) were **sign-flip
invariant**. MFDFA variance is invariant under negation; sign-flip
density is invariant under bit-flip. So WHT(−x) produces IDENTICAL
descriptor to WHT(x), giving cos(x, −x) = 1.0 from the codec but −1.0
from ground truth. THIS is what produced the ICC = −0.999. Not "codec
produces noise", but "codec collapses opposite rows" → perfect
ranking inversion against ground truth.

**Zipper fix:** sample ACTUAL SIGN BITS at φ-stride positions instead
of derived flip-density. Under negation, every phase bit flips →
phase_bits XOR all-ones → cosine → −1.0. Invariance broken; codec
preserves the sign relationship that ground truth measures.

**Results (ICC_3_1 across three populations):**

| Codec | Bytes | k_proj | gate_proj | q_proj |
|---|---|---|---|---|
| Passthrough (baseline) | 0 | 1.000 | 1.000 | 1.000 |
| Base17 | 34 | 0.007 | 0.012 | 0.024 |
| Fractal-Desc (magnitude) | 7 | **−0.999** | **−0.999** | **−0.996** |
| Fractal-Phase (flip density) | 5 | **−0.999** | **−0.999** | **−0.997** |
| **Zipper-Phase** | **8** | **0.050** | **0.049** | **0.097** |
| **Zipper-Full** | **64** | **0.129** | **0.107** | **0.203** |

**Key readings:**

1. **Zipper-Phase at 8 B BEATS Base17 at 34 B on every population.**
   2× to 4× higher ICC at 1/4 the storage. The φ-stride anti-moiré
   principle works for phase encoding.
2. **Zipper-Full at 64 B achieves top-5 recall 0.6 on q_proj** (Base17:
   0.0). The codec retrieves correct nearest-neighbors on 60% of
   queries — real reconstructive signal, not just ranking.
3. **Not yet competitive with I8-Hadamard leader (~9 B, ICC ~0.9).**
   Zipper-Full is a Pareto-meaningful new point but still ~4× off the
   leader on ICC. Room for improvement:
   - Wider phase stream (128 or 256 active bits)
   - φ-permute morph on the 64-bit scale (user's earlier suggestion)
   - Different phase/magnitude blend weights (current 0.5/0.5)
   - SVD-per-group basis instead of Hadamard
4. **Magnitude stream has signal.** Going phase-only (8 B) → full
   (64 B) adds 2-3× ICC on each population. The halo positions at
   φ²-stride carry non-redundant information vs phase at φ-stride.

**Architectural confirmations:**

- Aperiodic (X-Trans) sampling works as theorized — anti-moiré
  property preserves discriminative information across the Hadamard
  butterfly.
- Zeckendorf non-adjacent Fibonacci indices produce non-colliding
  strides without hand-tuning (φ vs φ² satisfied this naturally).
- Matryoshka single-container truncation works (8 B → 64 B via
  reading more of the same descriptor).

**Explicit constants locked (per user):**

  PHASE_ACTIVE_BITS    = 64  (per bgz17 halo signal-bit range)
  MAG_ACTIVE_SAMPLES   = 56
  ZIPPER_BYTES         = 64  (8 B phase + 56 B i8 magnitude)

Cross-ref: commits 7740759 (implementation), 6999106 (architecture
doc). bgz17 container design "family zipper" concept in
phi-spiral-reconstruction.md — empirically validated at last.
