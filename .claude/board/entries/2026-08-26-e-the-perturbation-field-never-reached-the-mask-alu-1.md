### E-THE-PERTURBATION-FIELD-NEVER-REACHED-THE-MASK-ALU-1

**Audited 2026-08-26** (session review of the p64 ⟷ cognitive-shader-driver
seam; operator analysis ratified against source, every claim file:line).

**The finding.** `StreamDto` / `PerturbationDto` / `BusDto` do NOT form an ALU
chain between p64 and the driver — they form an **adapter/transport seam**,
and the dense field is dropped at it:

```
StreamDto.codebook_indices → ingest_codebook_indices → BindSpace   (transport)
PerturbationDto.energy: Vec<f32>          ← stays behind, UNCONSUMED
PerturbationDto.top_k  → dispatch_from_top_k
                       → filter e > SCAN_WORTHY_ENERGY
                       → min(idx)..=max(idx) → ColumnWindow          (window!)
ShaderBus → BusDto → dispatch_busdto → Binary16K + qualia + meta   (projection)
```

`{7, 42, 900}` becomes *scan rows 7..901*, not the mask `{7, 42, 900}` — a
control-plane window heuristic, not mask-native execution. Meanwhile p64 IS
the mask surface (`edge_to_layer_mask` → `[u64;64]` / `[[u64;64];8]`,
`StyleParams{layer_mask, combine, contra, density_target}`) and is **DTO-blind**:
none of the three names appears in `p64-bridge`.

**The 4096 ≟ 4096 question — design intent recovered (operator, 2026-08-26).**
The two spaces were DESIGNED as one: the 64×64 field was meant as the COCA
codebook LUT (4096 vocabulary), with the SPO 2³ decomposition as its 8×
amortization and meta-awareness carried on `CausalEdge64` through the NARS
decomposition — which is WHY CE64 spends 3×8 = 24 bits on S/P/O. The p64
address `S/4 × O/4` (`edge_to_block`: 256×256 SO space → 64×64 blocks of 4)
was intended to land on the same 4096 places the codebook indexes.

That upgrades the question, it does not answer it: intent is not identity, and
the CODE has never proven `energy[i] ↔ cell[i/64][i%64] ↔ (S/4, O/4)`. Two
4096-place spaces designed as one can still have drifted into different
universes across the format history. Wiring on the count (or on the recovered
intent) alone would be representation-before-generator (7a's named error).
The mapping must be PROVEN — and the intent now supplies the concrete
hypothesis the probe tests:

```
CONTROL     top_k → min/max ColumnWindow            (today's arm, the baseline)
EXPERIMENT  energy field → proven 64×64 mask → p64 ALU → combine/contra/style
SABOTAGE    permute the energy↔cell addressing
```

If the sabotage permutation yields the same result, the spatial assignment
carries no information and the ALU hypothesis dies. Gate:
`ISS-PERTURBATION-P64-ADDRESS-IDENTITY-UNPROVEN`.

**Why this matters beyond hygiene:** this seam is very plausibly the piece of
the recovered resonance architecture (grounding-descent plan §7h) that existed
on paper and never made the last metre to the mask ALU — the dense
resonance field exists, the mask ALU exists, and the join between them
collapsed to top-k → window during implementation.

**Two code defects fixed alongside (same PR):**
1. `dto.rs` claimed `PerturbationDto IS f64[4096]` twice — the field is
   `Vec<f32>`, wrong scalar AND wrong shape, and `from_energy_f32` accepts
   arbitrary slice lengths unchecked. Docs corrected; behaviour untouched.
2. `busdto_to_binary16k` silently aliased any index ≥ 16384 onto a foreign
   bit via `% width_bits`. A hard `assert!` was tried FIRST and refuted by
   the existing corner-corpus test: `u16::MAX` is a pinned legal transport
   value (the headline round-trips losslessly via `qualia[9]`, never via the
   plane). Correct semantics: **skip, never alias** — an out-of-plane index
   sets NO bit, a documented loss in the same class as supporters with
   energy ≤ `SUPPORT_ENERGY` (likewise unrecoverable from the plane).
   Anti-aliasing + plane-edge tests added; before the fix, recovery reported
   a phantom top_k index `1` no producer ever emitted. The same wrap in
   `ingest_codebook_indices` is DOCUMENTED, not changed (lab-only
   gRPC/serve surface).

**The lost address contract, reconstructed (operator, same session).** The
4096 equality is not coincidence but a broken wire: the intended stack was

```
COCA codebook, 4096 atoms → 12-bit address → 6 bit | 6 bit → 64×64 LUT
    → superposition/perturbation over the field
    → NARS 2³ decomposition (8 relational READINGS of the same field —
      the 8× amortization: same address, different reading; proto-ClassView)
    → S/P/O result → CausalEdge64 bits 0..23 (3×8 SPO — the fossil that
      survives verbatim in the v2 layout) → meta-awareness / MUL
```

Today's `edge_to_block` (`S/4 × O/4` over the 256×256 SO space) is plausibly
a LATER re-use of the same 64×64 geometry, not the original codebook
addressing. `PerturbationDto` itself is the tell: `energy` sized by
`CODEBOOK_SIZE = 4096` plus `top_k: [_; 8]` — though whether the 8 echoes
the 2³ decomposition or is merely "top eight" is NOT proven and needs
historical code/docs. Lineage note: the COCA-as-tokens usage entered around
the deepnsm-v2 → paperless-rs / tesseract-rs arc (~PR −10..−25 from here).

The audit question for the HISTORY (now in the ISSUES entry): what was the
canonical map `codebook_id[0..4095] ↔ (row, col)[0..63]²` —
(A) row-major `id>>6, id&63`, (B) Morton deinterleave 12→6+6, or (C) a
codebook-specific permutation? A row-major cast would be bijective and test
green even if the truth was Morton — and still spatially scramble the field.
A grep for `>>6`/`&63`/Morton in thinking-engine, p64-bridge, and deepnsm
finds NO mapping in code today: the wire is fully torn, both ends live.

**Where the idea actually went (operator, same session): UP, into the
intake/token layer — not lost, migrated.** The old path
(COCA 4096 → 64×64 → SPO/NARS 2³ → CE64 → shader) has a modern successor:

```
PDF/image → tesseract-rs (crates/tesseract-paperless) → DocIr regions
  → ONE versioned tokenization receipt (#1017: one span tokenization serves
    Tantivy + DeepNSM-v2 + forward-prediction, no per-consumer retokenize)
  → COCA / academic lexical identity (18,559 real surface forms; DeepNSM-v2
    leaves the last basins deliberately empty rather than padding)
  → DeepNSM-v2 WordId + Pos on the 256:256 token rail (#798 broke the 4096
    ceiling: 16-bit vocabulary, 65,536 addresses — Alice used 7,675)
  → SPO → NARS/causal → CausalEdge64 / witness / alpha
```

So 4096 was never "the semantic population" — it was the first workable
working surface. The forward question is therefore NOT "wire the old DTOs
into p64" and NOT "squeeze 18k/65k back into 4096". p64's modern role is a
**WORKING SET / local ALU surface**: attention/current span/current basin
selects ≤ 4096 active relations, and THOSE live on the 64×64 mask field —
"4096 things addressable simultaneously in this local resonance field", not
"4096 concepts exist". The open design question, sharper than the wiring one:
**where between `tesseract-paperless → token receipt → DeepNSM-v2` does a
small-enough active set first arise to justify a 64×64 mask-native ALU?**


**Addendum, same session — can helix24 (Fisher-2Z LUT) carry the perturbation
energy? Gated YES (idea, unprobed).** The alignment is threefold and each leg
is already measured elsewhere today:

1. **The energy values are similarity-shaped.** `PerturbationDto.energy` is
   produced as `next[j] += table[i][j] · energy[i]` over the 4096×4096 u8
   distance table (`thinking-engine/src/engine.rs`, scale `1/(255−floor)`) —
   correlation-derived, non-negative. Fisher z is the variance-stabilising
   axis for exactly this shape, and
   `E-THE-FOUR-READINGS-OF-ONE-HELIX-CARRIER-…-1` measured 2Z as EXACTLY
   uniform as a LUT-over-field axis (r distorts 204×, y 11.6×) — and the
   perturbation field IS a LUT over a field.
2. **helix24's weakness is irrelevant here, its strength is the point.**
   The pole-penalty entry measured helix24 (`ResidueEdge`, 3 B) as bounded
   (∝ y) where helix48's polar byte diverges (∝ 1/r), winning on INDEX and
   losing on ANGLE. Perturbation energy needs index/magnitude, never angle —
   the carrier's losing axis is unused.
3. **The geometry already meets in the middle.** The engine's own compute
   plan ("Option B: Block Tiling / Rollrasen") tiles the 4096×4096 table
   into **64×64 blocks** with the energy chunk register-resident — the p64
   working-field geometry is already the engine's tiling unit. A 64×64
   working set with per-cell helix24 energy is 12 KB (vs 16 KB f32), in the
   units (2Z = geodesic ρ) that compose across families — and additive
   thresholds in z-space make the SCAN_WORTHY/SUPPORT pair scale-free.

Gates, in order: (a) the DOMAIN proof — accumulated energy is not bounded to
the arctanh domain a priori; the normalisation (converged fixed point? per-cycle
renorm?) must be measured before any `atanh` touches it (`Similarity::CLAMP_EPS`
exists, but clamping is a bandage, not a domain proof); (b) sequenced BEHIND
`ISS-PERTURBATION-P64-ADDRESS-IDENTITY-UNPROVEN` — an energy register on a
field whose addressing is unproven inherits the scramble; (c) zero-copy law —
helix24 as THE carrier of the working-set cell or as a transport codec, never
a second store beside the f32.

The intake seam already practices the same derive-at-the-address doctrine
this arc keeps confirming: DocIr has document identity (no minted source_id),
Region has span identity (no span_id), the token receipt exists (no
per-consumer retokenize), COCA/DeepNSM give lexical identity (no second text
world). The DTO path audited above reads, in hindsight, like a temporary
bypass built while that token/resident-state infrastructure did not yet
exist.
3. The two energy thresholds are now NAMED (`SCAN_WORTHY_ENERGY = 0.01`,
   `SUPPORT_ENERGY = 0.0`) with cross-referencing docs: worth-scanning is
   deliberately stricter than worth-recording-as-support. Distinct roles,
   not drift — the comment is the receipt.

Receipts: `engine_bridge.rs` (seam + fixes), `thinking-engine/src/dto.rs`,
`p64-bridge/src/lib.rs` (`edge_to_block` S/4×O/4), `busdto_bridge_test.rs`
(plane contract + new falsifier legs).
