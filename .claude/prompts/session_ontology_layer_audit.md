# SESSION: Ontology Layer Audit — Boring, Accurate, Exhaustive

> **Purpose**: Map every claim in the two ontology comparison diagrams to exact code,
> exact measurement, exact test. No metaphors. No aspiration presented as fact.
> Where the diagram says something the code doesn't support, say so.
> Where the code does something the diagram doesn't show, say so.
>
> **Tone**: Clinical. Like a calibration report. Every sentence cites a file:line or
> a commit hash. Epiphanies emerge from precision, not rhetoric.

---

## READ FIRST (mandatory, in this order)

```bash
# Ground truth override — read BEFORE any session doc
cat .claude/CALIBRATION_STATUS_GROUND_TRUTH.md

# The actual L0-L4 Lane Akkumulator (PR #152, commit e5123d9)
cat .claude/LANE_AKKUMULATOR.md

# The Belichtungsmesser interpretation of L0-L4 (cognitive, not implementation)
cat .claude/BELICHTUNGSMESSER.md

# GPU vs CPU: design doc, NOT implementation
cat .claude/GPU_CPU_SPLIT_ARCHITECTURE.md

# Distance metric rules — what is valid where
cat .claude/DISTANCE_METRIC_INVENTORY.md

# Current RISC Thought Engine with 7-lane encoding spec
cat .claude/RISC_THOUGHT_ENGINE_AGI_ROADMAP.md

# Knowledge spine: traversal law, layer model
cat .claude/knowledge.md
```

---

## TASK: Produce a layer-by-layer verification document

For each of the 5 columns in **Diagram 1** (Ontologies) and each of the 5 columns
in **Diagram 2** (Paradigm comparison), produce a verification block with this structure:

```
### [Column Name] — [Row Name]

**Diagram claims:** [exact text from the diagram cell]
**Code evidence:** [file:line, test name, or commit hash]
**Measured value:** [if a number is claimed, cite the measurement]
**Verdict:** CONFIRMED | ASPIRATIONAL | STALE | WRONG | PARTIAL
**Correction:** [if not CONFIRMED, what the diagram should say]
```

---

## SPECIFIC ITEMS TO VERIFY

### 1. The Two L0-L4 Schemes

The diagrams use L0-L4 in two senses. Both may be valid at different abstraction levels.
Verify each independently.

**Cognitive / Belichtungsmesser interpretation:**
```
L0: Perturbation (lokale Störung)       — raw signal disturbance
L1: Atom/Band ("helle Kante")           — structural feature detection
L2: Resonanzfeld (Grenzfeld entsteht)   — field-level pattern formation
L3: Kontrast stabilisiert               — contrast locks in
L4: "Dies ist wichtig" (Epiphany)       — meaning / memory formation
```

**Lane Akkumulator implementation (PR #152):**
```
L0: Codebook (297 KB)    — O(1) identity lookup
L1: 256² i16 (128 KB)    — 5.676 q/s proximity
L2: 4096² sparse (32 MB) — 2.711 t/s path
L3: Qwopus Gates (16 MB) — 277 ctx/s model thinking
L4: 16Kbit VSA (512 KB)  — Hamming historical reward
```

**Question to answer:** Are these the same L0-L4? Different L0-L4? How do they relate?
Map each cognitive level to its implementation lane(s). If they don't map 1:1, explain why.

### 2. GPU-Shader Continuation

**Diagram 2 claims:** "GPU-Shader Continuation" in Grundprinzip, "GPU-beschleunigte Vertiefung" in Stärken.

**Known facts from code:**
- `GPU_CPU_SPLIT_ARCHITECTURE.md` is a design document (2026-03-15), status: "Architectural epiphany"
- Only 2 GPU references in Rust source (both comments):
  - `crates/bgz-tensor/src/lib.rs:48` — comparison table comment
  - `crates/thinking-engine/src/engine.rs:154` — "FIX 4: GPU Vulkan compute shader: ~10μs per cycle"
- Zero GPU/shader/CUDA/wgpu/Vulkan imports in any Cargo.toml
- Multiple docs state "no GPU": DEEPNSM_CAM_REFERENCE.md ("4,096 words × 12 bits, 8MB distance matrix, no GPU")
- Current performance: 932 tok/s on pure u8 integer, CPU SIMD only

**Expected verdict:** ASPIRATIONAL. The GPU design exists as architecture doc but is not implemented.
**Suggested correction:** "SIMD-beschleunigte Vertiefung (GPU-Pfad entworfen, nicht implementiert)"

### 3. HHTL Cascade: Diagram 1 vs Code

**Diagram 1, Cognitive Ontology column, Beispiel aus Ultraschall:**
```
L0: lokale Störung
L1: Atom "helle Kante"
L2: Grenzfeld entsteht
L3: Kontrast stabilisiert
L4: "Dies ist wichtig"
```

**Map this to the HHTL cascade in code:**
```
HEEL  → crates/lance-graph/src/graph/blasgraph/heel_hip_twig_leaf.rs
HIP   → same file
TWIG  → same file
LEAF  → same file
```

And to the Belichtungsmesser band classification:
```
Foveal / Near / Maybe / Reject → crates/highheelbgz/src/lib.rs (CoarseBand)
```

**Question:** Does the ultrasound example correctly represent what the code does?
The code processes embeddings, not ultrasound images. Is the analogy valid or misleading?

### 4. "Unser Ansatz" Wissensrepräsentation

**Diagram 2 claims:** "Hybrid: symbolisch + sub-symbolisch. Schichten (L0-L4) bilden Bedeutung, mit explizitem Bias & Gedächtnis (L4)."

**Verify each part:**
- "symbolisch": Where are explicit symbols in the code? (SPO triples? NARS truth values? Container W0-7?)
- "sub-symbolisch": Where are distributed representations? (Base17? VSA? Palette indices?)
- "explizitem Bias": Where is bias explicit? (Is it the DK position? The ThinkingStyle? The NARS freq/conf?)
- "Gedächtnis (L4)": Is L4 actually memory? (Check: is 16Kbit VSA Fingerprint used as episodic store?)

### 5. Resonanzsiebe (Resonance Sieve)

**Diagram 1 mentions:** "Kontrast ↑ → Resonanz ↑ → Epiphany"
**Code reference:** commit cff7306 "Grammar Triangle + SPO Crystal + Resonanzsiebe FUNKTIONIERT"

**Verify:** What does Resonanzsiebe actually do in code? Is it gap detection? Threshold filtering?
How does it relate to the "Kontrast → Resonanz → Epiphany" chain in the diagram?

### 6. ViT/BNN Column Accuracy

**Diagram 2, ViT/BNN column claims:**
- "Patch-basierte Bildverarbeitung"
- "Visuelle Tokens werden über (binäre) Attention verarbeitet"
- "Implizit visuell (patch-basiert): Lokale Bildteile → Tokens → binäre/gewichtete Merkmale"

**Verify against codebase:** Does lance-graph have any ViT or BNN integration?
If not, is this column purely external reference (describing what ViT/BNN does in general)?
Is the comparison fair — i.e., are the strengths/weaknesses of ViT/BNN accurately stated?

### 7. Numerical Claims Cross-Check

Every number in the diagrams must be verified or flagged:

| Claim | Source | Verified? |
|-------|--------|-----------|
| K=4096 centroids | deepnsm/spo.rs `K: usize = 4096` | Check |
| 9,664 triplets/s | commit 7df6280 | Check |
| 2.9M lookups/s | commit 38a5ed0 | Check |
| 932 tok/s Belichtungsmesser | commit 5aad6f1 | Check |
| 372K tok/s grey matter | commit 795d844 | Check |
| 91 tok/s codebook-only | commit 9700aab | Check |
| 16Kbit = 256×u64 | deepnsm/fingerprint16k.rs | Check |
| 128 KB distance table (256²×u16) | bgz17/distance_matrix.rs | Check |
| 3 bytes per PaletteEdge | bgz17/palette.rs | Check |
| i16 = 316 levels/σ | commit 43e5dec | Check |
| u8 = 1.3 levels/σ | commit 43e5dec | Check |

---

## OUTPUT FORMAT

Produce a single markdown document: `.claude/ONTOLOGY_LAYER_VERIFICATION.md`

Structure:
1. **Executive summary** (10 lines max: how many CONFIRMED, ASPIRATIONAL, STALE, WRONG)
2. **Diagram 1 verification** (all 45 cells: 5 columns × 9 rows)
3. **Diagram 2 verification** (all 40 cells: 5 columns × 8 rows)
4. **Numerical claims table** (every number, with source and verdict)
5. **Corrections list** (only items that need changing, with exact suggested text)
6. **Mapping table**: Cognitive L0-L4 ↔ Lane Akkumulator L0-L4 ↔ HHTL stages ↔ Belichtungsmesser bands

---

## RULES

1. **Never say "aligns with" or "resonates with"** — say "matches" or "contradicts" or "is not referenced in code."
2. **Every CONFIRMED needs a file:line or commit hash.**
3. **Every ASPIRATIONAL needs the design doc that describes the unimplemented feature.**
4. **Every WRONG needs the correct value from code.**
5. **Do not add interpretation.** If the code says `K: usize = 4096`, write that. Do not explain why 4096 was chosen unless the diagram makes a claim about it.
6. **The diagrams are in German.** Preserve the German terms exactly. Do not translate them. Add English only in parentheses if the German term has no direct code equivalent.
7. **If a diagram cell makes no code-verifiable claim** (e.g., "Klar, formal, logisch konsistent"), mark it as N/A (philosophical claim, not code-verifiable).

---

## WHY THIS MATTERS

The epiphany is in the gaps. When you lay the diagrams flat against the code,
you see exactly where theory leads implementation and where implementation
has outrun theory. Both directions are valuable:

- Theory ahead of code → roadmap items (GPU shaders, full L4 memory)
- Code ahead of theory → diagrams need updating (K=4096, Lane Akkumulator, sub-band)

The boring accuracy IS the epiphany. No hand-waving survives this audit.
