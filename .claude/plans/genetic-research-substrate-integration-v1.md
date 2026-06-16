# Genetic Research Substrate Integration v1 — implementation plan

> **Type:** PROPOSAL / integration plan. Companion to:
>   - `docs/GENETIC_RESEARCH_VIA_STACK.md` — the pattern-recognition hand-off explaining *why*.
>   - `.claude/handovers/2026-06-16-genetic-research-headstone-exploration.md` — the destination-state synthesis explaining *what complete looks like*.
>   - `.claude/plans/3DGS-genetics-4x4-fanout-plan.md` — the predecessor exploratory plan (4×4 lane interpretation for static representation). This plan extends with **dynamics + counterfactual lane**.
> **Status:** initial proposal. 10 deliverables (D-GEN-1..10) + 3 gating probes. No code shipped yet.
> **Boundary:** representation + research tooling only. Not a clinical genomics product plan. No medical/diagnostic claims.

---

## 0. Architectural decisions locked (do not re-litigate)

Per the substrate's §0 anti-invention guardrail (lance-graph #496) and the no-new-variant contract test (#500):

1. **No new `ValueSchema` enum variant.** Genetic-research rows ride `Full` or `Compressed` presets; specialisation is via `classid → ClassView` mint.
2. **No new `EdgeCodecFlavor` enum variant.** Genomic edges ride **`Pq32x4`** (`TurbovecResidue` tenant has shipped storage) or **`CoarseOnly`** (1-byte palette, no separate residue slab). `CoarseResidue` is **BLOCKED** until the operator mints its dedicated `ValueTenant` (`TD-COARSERESIDUE-NO-VALUE-TENANT`, `.claude/board/TECH_DEBT.md:40`) — pairing it with `Full` or `Compressed` today leaves the signed-4-bit residue unaddressable.
3. **No C++ source vendored into any `-rs` adapter crate.** htslib / bcftools / etc. stay upstream; harvested via the `ruff_cpp_spo` pattern (cross-repo handover at `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-spo-handover.md`) when the harvester ships.
4. **Counterfactuals stay in their own lane.** `InferenceType::Counterfactual` mantissa = -6 is the mechanical enforcement. **NEVER written as observed SPO.**
5. **No new substrate primitives.** Every deliverable in this plan is **wiring** of existing primitives. If a deliverable feels like it needs a new substrate type, it's the wrong shape — escalate to the 5-specialist drift-catching pass before writing code.
6. **Closed-vocab discipline applies.** New genetic predicates land in `ruff_spo_triplet::Predicate` under the `predicate_count_locked_at_N` gate (`AdaWorldAPI/ruff` PR #5 pattern).

---

## 1. Deliverables

### D-GEN-1: `crates/adapter-genetics-experimental` scaffold

**What:** new crate sibling to `lance-graph-callcenter`. Defines `GenomicSubstrate` trait analogous to `OcrProvider` (lance-graph #498) — the engine-agnostic seam that any provider (1000-Genomes / GIAB / clinical / synthetic) implements. Locks the 4×4 lane interpretation from `3DGS-genetics-4x4-fanout-plan.md` (`lane0: sequence coordinate / lane1: motif covariance / lane2: expression / methylation / lane3: time / sample / lineage`).

**Where:** `crates/adapter-genetics-experimental/`. Cargo.toml deps on `lance-graph-contract` + `ruff_spo_triplet` (via cross-repo) + `serde` (no `clang` / `noodles` / heavy deps in the trait crate).

**Lift:** ~1 day. Pure interface + locked-shape test (the `ruff_ruby_spo` PR #4 discipline).

**Test gate:** `genomic_substrate_trait_compiles_and_has_default_impl`.

### D-GEN-2: FASTA + VCF parsers (host `noodles-*`)

**What:** opt-in feature flags `fasta`, `vcf` that host `noodles-fasta` and `noodles-vcf` (the pure-Rust bioinformatics ecosystem). Map `noodles::vcf::Record` → `lance_graph_contract::NodeRow` via a `VcfRecordTranscoder` analogous to `LayoutBlock::to_node_row` (lance-graph #498's `2fa7fcb0`).

**Where:** `crates/adapter-genetics-experimental/src/fasta.rs` + `vcf.rs`.

**Lift:** ~3 days. The `noodles` crate is mature; the transcode follows the OCR `LayoutBlock → NodeRow` pattern almost verbatim.

**Test gate:** `vcf_record_transcodes_to_node_row_with_canonical_classid`.

### D-GEN-3: k-mer → CAM-PQ 48-bit fingerprint

**What:** `fn kmer_fingerprint(seq: &[u8], k: usize) -> Cam6x8` that computes the CAM-PQ 6-subspace × 256-centroid fingerprint of a k-mer-frequency vector. Use bgz17 11/17 stride for k-mer sampling (anti-moiré, see `BGZ17_ELEVEN_SEVENTEEN_RATIONALE.md`).

**Where:** `crates/adapter-genetics-experimental/src/kmer.rs`.

**Lift:** ~half-day. CAM-PQ codec is shipped (`crates/lance-graph/src/cam_pq/`); this is just feeding it.

**Test gate:** `kmer_fingerprint_is_deterministic` + `bgz17_stride_visits_all_17_residues`.

### D-GEN-4: Reference genome NodeGuid addressing mint

**What:** pin the `classid → ClassView` registry entries for genomic classes:
- `classid 0x0001_0001` = `Cell::Generic`
- `classid 0x0001_0002` = `Cell::RAS_pathway_node`
- `classid 0x0001_0003` = `ViralIntegrationSite`
- `classid 0x0002_0001` = `GenomicPosition` (HEEL = organism / HIP = chromosome / TWIG = position-prefix)
- `classid 0x0002_0002` = `Variant`
- `classid 0x0003_0001` = `Gene`
- `classid 0x0003_0002` = `Transcript`
- `classid 0x0003_0003` = `Protein`

(Specific classid values to be ratified by an OGAR mint pass — these are placeholders pinning the slot.)

**Where:** `crates/adapter-genetics-experimental/src/class_mint.rs` + sync with `lance-graph-ontology` registry.

**Lift:** ~1 day. Mostly registry entries + per-class `ClassView` declarations that ride existing `ValueSchema` presets.

**Test gate:** `genomic_classids_ride_existing_presets_no_new_variant` (mirrors lance-graph #500's `ocr_schema_fit_rides_existing_preset_no_new_variant`).

### D-GEN-5: GO / Reactome / ClinVar Pattern D hydrate_*() glue

**What:** add three `hydrate_*()` glue functions over the shipped `OwlHydrator` / `MetaStructureHydrator` (Pattern D — Meta-Structure Hydration, `crates/lance-graph-ontology/src/hydrators/mod.rs:1-57`). **No new trait** — the substrate already ships `hydrate_dolce` / `hydrate_owltime` / `hydrate_provo` / `hydrate_qudt` / `hydrate_schemaorg` / `hydrate_skos` / `hydrate_fibo_fnd` / `hydrate_fibo_be` / `hydrate_odoo` / `hydrate_zugferd` / `hydrate_skr03` / `hydrate_skr04` as the proven pattern. Per `mod.rs` line 19: *"Each per-ontology hydrator is data + ~50 LOC of glue, never a bespoke crate."*

- `hydrate_go(reg, source)` → Gene Ontology OBO/OWL via `OwlHydrator` with the `G` slot keyed for biological process / molecular function / cellular component.
- `hydrate_reactome(reg, source)` → Reactome pathway hierarchy (~2500 pathways) via `MetaStructureHydrator` declaring `inherits_from` for pathway-subpathway containment.
- `hydrate_clinvar(reg, source)` → ClinVar clinical-significance annotations as variant SPO edges with `Provenance::ClinicalCurated = (0.98, 0.95)` calibration.

**Where:** `crates/lance-graph-ontology/src/hydrators/go.rs` + `reactome.rs` + `clinvar.rs`. Each file is *data + glue*: pick the parser, declare the `G` slot, name the parent, whitelist the cascade edge IRIs — mirrors the shipped `dolce.rs` / `provo.rs` / `skos.rs` shape.

**Lift:** ~1 week (GO + Reactome are large; ClinVar is straightforward).

**Test gate:** `go_term_count_matches_reference_release` + `reactome_pathway_hierarchy_round_trips` + `clinvar_provenance_calibration_matches_curated_pair`.

### D-GEN-6: §14 oracle adapter for variant caller comparison

**What:** `VariantCallerOracle` — a `OracleSubstrate` impl that compares two variant callers' output (e.g. GATK HaplotypeCaller vs. DeepVariant) on the same input, with `Provenance` annotations identifying which caller contributed which call. Uses `rubicon::oracle::compare_normalised` directly.

**Where:** `crates/adapter-genetics-experimental/src/oracle.rs`.

**Lift:** ~3 days. The §14 oracle is shipped; this is the genetics-domain mapping.

**Test gate:** `caller_comparison_finds_known_discordant_calls_in_giab_subset`.

### D-GEN-7: KRAS G12D 1024-cell counterfactual fan-out simulation

**What:** the dynamics-axis flagship. Build a `MailboxSoA<1024>` representing a 32×32 cellular lattice; instantiate KRAS-G12D-vs-WT as `SplitPoles` at one row; fan-out via `consume_firing` over 100 cycles. Compare the observed lane (G12D propagates → MAPK cascade → tumor-suppressor loss) against the counterfactual lane (WT, no propagation). Run `revise_if_minority_wins` at cycle 100; verify the observed-lane victory aligns with published KRAS-G12D oncogenic-transformation rates.

**Where:** `crates/adapter-genetics-experimental/src/sim/kras_propagation.rs` + `tests/kras_g12d_counterfactual_fanout.rs`.

**Lift:** ~2 weeks. The substrate primitives are shipped; this is the integration + calibration against published outcomes.

**Test gate:** `kras_g12d_propagation_outpredicts_wt_counterfactual_at_cycle_100` (with measured tolerance).

### D-GEN-8: DeepNSM genetic-language reader probe

**What:** point the existing `crates/deepnsm` sentence-level AriGraph reader at codon-triplet sequences. Treat each codon as a `NsmPrime`; verify `SentenceTransformer64` produces a `Sentence64 { p64, cam, spo_hint }` for DNA the same way it does for English. Compare reading-state convergence on protein-coding vs. non-coding sequences.

**Where:** `crates/adapter-genetics-experimental/examples/genetic_language_probe.rs` (analogous to `lance-graph-arm-discovery/examples/coreference_rung_probe.rs`).

**Lift:** ~1 week. Mostly fixture construction (codon-triplet NsmPrime mapping) and validation.

**Test gate:** `deepnsm_genetic_language_p64_consistency_protein_coding_vs_noncoding`.

### D-GEN-9: Histology splat extension — per-splat genomic profile

**What:** mint `classid` for `HistologySplatWithGenomicProfile`. The splat carries:
- `key (16 B)`: spatial position in the histology slide.
- `edges (16 B)`: 12 in-family neighboring splats + 4 out-of-family (vascular / immune).
- `value (480 B)`: `Full` ValueSchema preset, with `Fingerprint` tenant carrying the Σ₁ SEED of the resident cell's expression vector and `HelixResidue` tenant carrying orientation.

**Where:** `crates/adapter-genetics-experimental/src/histology_bridge.rs`. Reuses the splat-native ultrasound substrate (3DGS arc) verbatim.

**Lift:** ~3 days. The splat substrate is shipped; this is the genomics-side extension.

**Test gate:** `histology_splat_carries_genomic_profile_at_existing_preset_no_new_variant`.

### D-GEN-10: §14 oracle benchmark against GIAB truth set

**What:** run the substrate's variant-calling pipeline (D-GEN-2 + D-GEN-3 + D-GEN-6) against the Genome in a Bottle Consortium truth sets (HG001-HG004). Report sensitivity / precision / F1 with `Provenance` calibration for each call.

**Where:** `crates/adapter-genetics-experimental/benches/giab_benchmark.rs`.

**Lift:** ~1 week. Most of this is data wrangling.

**Test gate:** `giab_hg002_f1_meets_published_minimum_for_call_set_v4_2_1`.

---

## 2. Gating probes (before any FINDING-grade claim)

Per the `lance-graph` PR #500 discipline (probes spec before measured claims):

### PROBE-CHAODA-1000G

**What it gates:** the claim *"CHAODA detects novel variants without trained classifier."*

**Method:** build CLAM tree on 1000-Genomes Phase 3 feature vectors (AF, depth, strand bias, neighbourhood entropy, conservation score). Compute CHAODA anomaly scores on a held-out test set containing known novel singletons. Measure ROC-AUC against ground truth.

**Pass condition:** ROC-AUC ≥ 0.85 on novel-singleton detection (calibrated against published CADD / REVEL baselines).

**Implementation:** ~3 days. The CLAM + CHAODA kernels are shipped; this is fixture + scoring.

### PROBE-KRAS-COUNTERFACTUAL-DET

**What it gates:** the claim *"KRAS counterfactual fan-out is deterministic."*

**Method:** run D-GEN-7's KRAS G12D 1024-cell simulation twice with identical seeds. Bit-compare the final `MailboxSoA.energy[]` arrays and the counterfactual-lane edge counts.

**Pass condition:** bit-exact match across two runs (the substrate's no-randomness invariant applies; any deviation indicates an unmarked f32 nondeterminism).

**Implementation:** ~2 days; included in D-GEN-7's test scope.

### PROBE-CAM-PQ-VS-BLAST

**What it gates:** the claim *"CAM-PQ 48-bit fingerprint approximates sequence similarity."*

**Method:** on a held-out RefSeq subset, compare CAM-PQ Hamming-distance rankings against BLAST e-value rankings. Compute Spearman ρ and ICC (the substrate's reliability stats from ndarray #218).

**Pass condition:** Spearman ρ ≥ 0.7 against BLAST e-value top-100 rankings, and ICC ≥ 0.6. (The Σ₁ SEED preserves 94% of Jina semantic similarity on language; biological-sequence similarity may calibrate differently.)

**Implementation:** ~1 week. Includes the protein-language-model embedding pipeline (ESM/ProtBERT load via existing GGUF loader + ndarray AMX int8 GEMM at 197 GMAC/s).

---

## 3. Sequencing

| Phase | Deliverables | Cumulative time |
|---|---|---|
| **P1** | D-GEN-1, D-GEN-2, D-GEN-3, D-GEN-4 | ~2 weeks |
| **P2** | D-GEN-5, PROBE-CHAODA-1000G | ~3 weeks |
| **P3** | D-GEN-6, D-GEN-7, PROBE-KRAS-COUNTERFACTUAL-DET | ~6 weeks |
| **P4** | D-GEN-8, D-GEN-9, PROBE-CAM-PQ-VS-BLAST | ~8 weeks |
| **P5** | D-GEN-10 | ~9 weeks |

**Minimum viable hand-off:** end of P1. The adapter crate compiles, FASTA/VCF round-trip, k-mer fingerprints are computable, the class mint pins the genomic classid range. That's the *"shape locked"* milestone analogous to `ruff_ruby_spo`'s locked-shape test in PR #4 — enough for a genomics-domain session to pick up and continue without architectural drift.

**The flagship deliverable:** D-GEN-7 (KRAS counterfactual fan-out). When that ships and PROBE-KRAS-COUNTERFACTUAL-DET passes, the dynamics axis is proven and the substrate is differentiated from every standard bioinformatics tool.

---

## 4. Open questions for the operator

1. **Tesseract-style corpus pin:** which 1000-Genomes / GIAB / ClinVar release version pins the corpus? Pin one before P2 starts.
2. **Class IDs:** the placeholder `0x0001_0001` etc. need an OGAR mint pass before being committed to the registry.
3. **D-GEN-7 calibration target:** which published study's KRAS-G12D oncogenic-transformation rate is the canonical pass target? (Suggested: Hancock et al. 2002 or a more recent meta-analysis; operator to confirm.)
4. **Ontology release pinning:** GO release cadence is monthly; ClinVar is weekly; Reactome is quarterly. Per-hydrator pinning strategy?
5. **Clinical-vs-research boundary:** this plan stays on the research-tooling side (the existing `3DGS-genetics-4x4-fanout-plan.md` line: *"no medical/diagnostic claims are made"*). The §14 oracle's variant-caller comparison is research benchmarking, not clinical decision-support. Confirm that holds.

---

## 5. Cross-references

- **Pattern-recognition hand-off:** `docs/GENETIC_RESEARCH_VIA_STACK.md` — the *why* doc for a domain expert.
- **Headstone:** `.claude/handovers/2026-06-16-genetic-research-headstone-exploration.md` — the destination-state synthesis.
- **Predecessor plan:** `.claude/plans/3DGS-genetics-4x4-fanout-plan.md` — static representation, 4×4 lane interpretation. This plan extends with dynamics + counterfactual.
- **Cross-repo C++ harvester (relevant if htslib transcoding wanted):** `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-spo-handover.md` + `AdaWorldAPI/ruff/.claude/handovers/2026-06-16-ruff-cpp-headstone-exploration.md`.
- **Upstream substrate context:** `lance-graph` PR #491 (entropy × energy framing) + PR #494 (EntropyRung / Quadrant / nars_entropy) + PR #495 (3-byte EdgeRef witness; reliability ⊥ causality empirically) + PR #496 (ValueSchema presets + §0 anti-invention) + PR #498 (GUID decode→read-mode keystone; helix Signed360; OCR→NodeRow transcode template) + PR #500 (rebaseline + no-new-variant contract test).
