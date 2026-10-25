# Research shelf inventory: takeaways, evidence and next falsifiers

**Audited:** 2026-10-25. **Scope:** 90 existing Markdown files in this `.arxiv/` research shelf. This index is editorial synthesis of their contents, **not** a new primary-source verification, executed benchmark, scientific review or authorization to implement.

## Editorial finding: family resemblance, not one Markdown schema

Many notes use **source and read scope → punchline or source-grounded findings → lance-graph transfer hypothesis → falsifiers / kill conditions**. However, the actual headings vary (`Punchline`, `Harvest`, `Hypothetical transfer`, `Testables`, `Decision boundary`). Some are only abstract-level bookmarks; others are surveys, multi-paper distillations, full-length technical analyses or source transcriptions. A file under `.arxiv/` is **not necessarily an arXiv paper**: journal articles, conference papers, teaching notes, software and philosophical sources are also included.

Do **not** retroactively reformat the notes. For future notes, prefer a light consistent header: **Source + version/DOI; type/read grade; evidence status; one-sentence takeaway; source finding; proposed transfer; verified existing seam (if any); falsifier/oracle/kill condition; related notes and PRs**. Keep `READ-FULL / EXCERPT / ABSTRACT / BOOKMARK / SYNTHESIS / SOURCE TRANSCRIPTION` distinct from `PROPOSED / TESTED / IMPLEMENTED / REJECTED`. The original note owns its specific evidence grade.

**Index status:** Every takeaway and follow-up below is a concise *editorial interpretation*, not an assertion that the suggested experiment has run. The priority is a **proposed** triage: **P0** correctness/cross-note gate; **P1** focused experiment; **P2** deeper reading or optional comparison. Follow-up marked **READ FURTHER** is intentionally **not** a completed source review.

## Cross-note experiments: seven high-leverage continuations

| Gate | Combined insight | Adversarial control / decision |
|---|---|---|
| **X1. Typed-law admission** | DialEgg + egglog + Ruler + eggcc + 3LA + Sutra | Preserve SQL K3 UNKNOWN, Bag/Distinct, i8 rounding, FP signed zero/NaN and effect order. Only extract after lawful typing and independent oracle. |
| **X2. Focus economics** | DR256/QR256 + 256-rule normalization + NNUE-style delta + cache/SoA | Compare compile + bind + gather + execute + verify on 1/8/64/256 reuses, warm/cold, sharing/nonsharing, stale version. |
| **X3. Evidence provenance** | Wang + Jøsang + CE64 + detached revision | Duplicate observation with distinct event ID but same evidence root; reject double counting; prove fission/undo when legal. |
| **X4. Virtual intervention ledger** | Pearl/Halpern + counterfactual graphs + SOPD + CARE + world models | Replay divergent virtual futures from one frozen generation, including overdetermined causes; virtual output must not self-certify or commit. |
| **X5. Metacognitive action** | NAL-9 + ONA + self-model review + SAG | Disable trace, shuffle events and inject do-nothing action; look for an actual downstream policy difference. |
| **X6. Cross-language semantic binding** | Haspelmath + DeepNSM + Hyperon/MeTTa + typed DAG | Keep ambiguous alternatives, scoped source provenance, and unsupported-operator refusals. |
| **X7. Information gain versus purpose** | PLN/BALD + active view selection + Rubicon + reward misspecification | Compare entropy/surprise with utility on wrong-model and predictable-but-bad controls. |

**Sequencing hypothesis:** X1 first (tiny, read-only proof of correct law boundaries), then X2 (explicit end-to-end cost); X3–X7 are separate research threads, not dependent on introducing a universal new IR. The repository README currently documents distinct Cypher/DataFusion and Quack/mask-risc paths, **not** a shipped unified compiler.

## Complete per-note inventory

All 90 original notes are linked below. **Takeaway** summarizes the document, **Next** is an unexecuted suggested falsifier or reading step.

### A. Mathematical compiler, RelAlg and proof extraction

| Note | Takeaway | Next follow-up | Priority |
|---|---|---|:---:|
| [guarded kleene algebra verified control flow excerpt](1907.05920_guarded_kleene_algebra_verified_control_flow_excerpt.md) | Guarded Kleene algebra can reason about control flow only under admitted laws. | Test equivalence with a separate finite-state evaluator. | P1 |
| [oopsla byods ascent custom relations dr excerpt](2023_oopsla_byods_ascent_custom_relations_dr_excerpt.md) | Ascent custom relations do not automatically imply a DR/SoA runtime. | Compare admitted relations with coordinate witnesses. | P2 |
| [3la formal interface flexible matching application validation](2026-10_3la_formal_interface_flexible_matching_application_validation.md) | Formal software/hardware interfaces can validate transformations. | Specify minimal typed R2IL/relalg validity contract. | P0 |
| [eggcc effect safe extraction statewalk r2il map](2026-10_eggcc_effect_safe_extraction_statewalk_r2il_map.md) | Effectful e-graph extraction must preserve state order. | Reject reordered aliasing writes and stale bind reads. | P0 |
| [four russians m4ri reverse lookup excerpts](2026-10_four_russians_m4ri_reverse_lookup_excerpts.md) | Four Russians tables can amortize binary elimination. | Benchmark rank and invertibility against scalar GF(2). | P1 |
| [qr256 gaussian egglog rotation falsifier paths](2026-10_qr256_gaussian_egglog_rotation_falsifier_paths.md) | DR/QR tiles are typed masks, not proofs or global identities. | Run 256-rule tests with generation and dirty-state guards. | P0 |
| [tatlock babble smallproofs chassis lakeroad glenside](2026-10_tatlock_babble_smallproofs_chassis_lakeroad_glenside.md) | Library learning, proof size and target instruction cost are distinct. | Compare law admission, minimization and extraction separately. | P1 |
| [tatlock publication survey mathematical compiler index](2026-10_tatlock_publication_survey_mathematical_compiler_index.md) | Mathematical-compiler papers form a research map, not one implementation. | Use as navigation; select smallest unmet falsifier first. | P1 |
| [tatlock relational ematching ruler enumo harvest](2026-10_tatlock_relational_ematching_ruler_enumo_harvest.md) | Relational e-matching and Ruler/Enumo propose laws, not soundness. | Verify generated rules under typed independent oracle. | P0 |
| [zayed dialegg semantic bridge epiphany breadcrumbs](2026-10_zayed_dialegg_semantic_bridge_epiphany_breadcrumbs.md) | DialEgg offers a dialect bridge for guarded equality exploration. | Round-trip unknown operation and one lawful matrix chain. | P0 |
| [zucker kleene algebra kat derivatives relalg excerpt](2026-10_zucker_kleene_algebra_kat_derivatives_relalg_excerpt.md) | Kleene/KAT derivatives offer symbolic normalization only within matching semantics. | Find SQL bag/NULL/effect counterexamples. | P1 |
| [gaussian greedy reversible circuit excerpt](2201.06508_gaussian_greedy_reversible_circuit_excerpt.md) | Reversible circuit synthesis has independent algebraic gate costs. | Compare Gaussian vs greedy gate counts with inverse replay. | P2 |
| [egglog equality saturation datalog excerpt](2304.04332_egglog_equality_saturation_datalog_excerpt.md) | Egglog combines Datalog facts and equality classes; neither is physical-row identity. | Guard law domains, source generations and termination fuel. | P0 |
| [selective yannakakis smash mask first thought coordinates](2502.20233_selective_yannakakis_smash_mask_first_thought_coordinates.md) | Yannakakis-style reductions can cut intermediate join work. | Compare exact multiplicity and read costs with mask-first strategy. | P1 |
| [sutra tensor normal form exact approximate boundary](2605.02337_sutra_tensor_normal_form_exact_approximate_boundary.md) | Discrete Kleene-polynomial parity cannot justify continuous logic rewrites. | Sweep fractional inputs for monotonicity and associativity failures. | P1 |
| [regularize irregularity into one path](2609.22106_regularize_irregularity_into_one_path.md) | Regularize sparse irregularity into existing operations where possible. | Account for preparation and prove no extra hot-path operator. | P1 |
| [quantum gaussian reversible proof excerpt](2609.28901_quantum_gaussian_reversible_proof_excerpt.md) | Reversible Gaussian steps need recorded algebraic witnesses. | Replay inverse operations without extrapolating quantum speedups. | P2 |
| [gaussian partial pivot growth warning excerpt](2610.08701_gaussian_partial_pivot_growth_warning_excerpt.md) | Partial-pivot Gaussian elimination can exhibit large growth. | Use scaled numerical fixtures and higher-precision reference. | P1 |
| [distributed datalog wcoj hypercube bookmark](2610.12692_distributed_datalog_wcoj_hypercube_bookmark.md) | Hypercube WCOJ suggests alternatives for distributed relational work. | READ FURTHER: methods before skew/communication tests. | P2 |
| [chordal codes elimination order excerpt](2610.17934_chordal_codes_elimination_order_excerpt.md) | Chordal elimination order can reduce fill under chordality assumptions. | Check graph chordality and perfect elimination before timing. | P2 |

### B. Numerical contracts, quantization and hardware

| Note | Takeaway | Next follow-up | Priority |
|---|---|---|:---:|
| [narrow storage wide accumulate round once](1905.12322_narrow_storage_wide_accumulate_round_once.md) | Store narrowly, accumulate widely and narrow once. | Exhaust rounding, overflow and repeated CE64 accumulation. | P1 |
| [tanh velocity factors are not nars evidence](2008.02078_tanh_velocity_factors_are_not_nars_evidence.md) | Tanh/atanh numeric identities do not encode NARS evidence counts. | Compare numeric equality separately from revision semantics. | P1 |
| [deiser hyperbolic areafunctions atanh domain primary excerpt](2023_deiser_hyperbolic_areafunctions_atanh_domain_primary_excerpt.md) | atanh has strict domain and singular endpoints. | Validate palette256 endpoint, clamp and special-code policy. | P1 |
| [2024-11-10 singh tanh activation is not confidence](2024-11-10_singh_tanh_activation_is_not_confidence.md) | Tanh activation is not epistemic confidence. | Require evidence model beyond successful activation parity. | P1 |
| [yuan hardware aware rust nnue cache delta masks](2026-10_yuan_hardware_aware_rust_nnue_cache_delta_masks.md) | Hardware-aware SoA and delta updates can change focus economics. | Compare full and incremental recompute under cache churn. | P1 |
| [special values are declared not computed](2209.05433_special_values_are_declared_not_computed.md) | Special float codes are policies, not incidental arithmetic. | Exhaust NaN, infinity and saturation encodings. | P1 |
| [shared scale lives outside the element](2310.10537_shared_scale_lives_outside_the_element.md) | Shared block quantization scale is lossy and axis-sensitive. | Measure transpose and rotation reconstruction error. | P1 |
| [stagnation is a rounding property](2603.24161_stagnation_is_a_rounding_property.md) | Stochastic rounding avoids stagnation in expectation, not per-run determinism. | Measure deterministic replay and seeded error over repeated updates. | P1 |
| [correlation needs symmetry breaking](2605.01487_correlation_needs_symmetry_breaking.md) | Fold local correlations before reconstructing states requiring symmetry breaking. | Compare fold-first/reconstruct-first under controlled noise. | P2 |
| [path invariance and explicit boundaries](2608.00546_path_invariance_and_explicit_boundaries.md) | Representation invariance requires explicit degeneracy boundaries. | Run metamorphic path-parity with singular fixtures. | P1 |

### C. NARS evidence and metacognitive replay

| Note | Takeaway | Next follow-up | Priority |
|---|---|---|:---:|
| [fusion needs to know whether evidence is independent](0712.1182_fusion_needs_to_know_whether_evidence_is_independent.md) | Cumulative fusion assumes independent evidence; dependent reports need averaging. | Check duplicate evidence roots, fission and dogmatic limits. | P0 |
| [revision is not bayesian conditioning](1303.1517_revision_is_not_bayesian_conditioning.md) | Bayesian conditioning differs from provenance-aware evidential revision. | Build dependent/independent-root counterexamples. | P0 |
| [wang nars41 truth revision resource control primary excerpt](2000_wang_nars41_truth_revision_resource_control_primary_excerpt.md) | NARS truth inference and resource control are different contracts. | Pin the tutorial rules against scalar oracles. | P1 |
| [wang confidence is evidence stability not a 95pct interval](2001_wang_confidence_is_evidence_stability_not_a_95pct_interval.md) | Confidence expresses evidence stability, not a 95% interval. | Test weight conversion and source-root independence. | P0 |
| [wang two numbers preserve direction and stability of evidence](2009_wang_two_numbers_preserve_direction_and_stability_of_evidence.md) | Belief direction and support stability need two numbers. | Test information lost in one-scalar codecs. | P1 |
| [self in nars requires internal experience and mental operations](2018_self_in_nars_requires_internal_experience_and_mental_operations.md) | NARS self-models require internal experience and mental acts. | Demonstrate a traced self-action changing later inference. | P1 |
| [isaev hammer nars agi20 weighted revision primary excerpt](2020_isaev_hammer_nars_agi20_weighted_revision_primary_excerpt.md) | NARS AGI20 operator equations need explicit weighting assumptions. | Reproduce weighted revision with independent formulas. | P1 |
| [nal1 to nal9 is a capability ladder not nine cycles](2026-08_nal1_to_nal9_is_a_capability_ladder_not_nine_cycles.md) | NAL-1…9 are expressive capability levels, not nine machine cycles. | Define one operational fixture per supported level. | P1 |
| [ce64 hypothesis layers research map](2026-10_ce64_hypothesis_layers_research_map.md) | CE64 encodings raise distinct questions on evidence, precision, semantics and control. | Assign independent oracle and owner to each open Q1–Q6. | P0 |
| [inverted temporal ablation reasoner oracles](2026-10_inverted_temporal_ablation_reasoner_oracles.md) | Temporal ablation and source-independent reasoner oracles are essential. | Run reasoner parity on held-out temporal cases. | P1 |
| [nars evidence nal metacognition distillation index](2026-10_nars_evidence_nal_metacognition_distillation_index.md) | ALU precision does not prove NARS truth, independence or metacognition. | Pin separate tests for each claim. | P1 |
| [nars self epistemic needs evidence primary excerpts](2026-10_nars_self_epistemic_needs_evidence_primary_excerpts.md) | Internal monitoring and evidence roots must not be conflated. | Check revision and self-correction against external witnesses. | P1 |
| [software awareness ona observability metamorphic holdout excerpts](2026-10_software_awareness_ona_observability_metamorphic_holdout_excerpts.md) | Observability must survive holdouts and negative controls. | Inject no-op, shuffled logs and test leakage traps. | P1 |
| [software awareness ona ruff research excerpts](2026-10_software_awareness_ona_ruff_research_excerpts.md) | Codegraph observability is not evidence of subjective self-awareness. | Require inspected traces to affect later control choices. | P2 |
| [johansson hammer machine psychology aarr ona primary excerpt](2026_johansson_hammer_machine_psychology_aarr_ona_primary_excerpt.md) | Machine psychology mechanisms must distinguish core NARS from extensions. | Reproduce a behavioral task with external-validity limits. | P2 |
| [meta uncertainty bayesian model comparison vs ce64 mul](2210.07278_meta_uncertainty_bayesian_model_comparison_vs_ce64_mul.md) | Model-comparison uncertainty is not a CE64 confidence multiplier. | Compare model uncertainty and belief uncertainty explicitly. | P2 |
| [metacognition requires replayable self models not just logs](2503.13467_metacognition_requires_replayable_self_models_not_just_logs.md) | Metacognition requires replayable self-diagnosis that changes control. | Ablate recalled trace and observe downstream behavior. | P0 |
| [executable narsese needs runtime validation not label guessing](2604.18873_executable_narsese_needs_runtime_validation_not_label_guessing.md) | Symbolic labels require runtime Narsese validation to count as correctness. | Parse, execute and report answer plus execution success. | P1 |
| [undated nal9 mental operations are executable events not a ninth op](undated_nal9_mental_operations_are_executable_events_not_a_ninth_op.md) | NAL-9 is an executable feedback loop, not an extra opcode. | Prove inspect-diagnose-act changes future operations. | P1 |

### D. Causal intervention, planning and information acquisition

| Note | Takeaway | Next follow-up | Priority |
|---|---|---|:---:|
| [counterfactual causation can be bounded not guessed](1301.3898_counterfactual_causation_can_be_bounded_not_guessed.md) | Causal queries sometimes admit bounds rather than a point probability. | Return identifiable bounds under unobserved confounders. | P0 |
| [actual causality needs contingencies not just deletion](1608.07879_actual_causality_needs_contingencies_not_just_deletion.md) | Actual causality needs contingencies, not single-edge deletion. | Add overdetermination and alternative-context probes. | P0 |
| [openpsi rubicon care wisdom frontier source excerpts](2026-10_openpsi_rubicon_care_wisdom_frontier_source_excerpts.md) | CARE, OpenPsi and Rubicon separate proposal, appraisal and commitment. | Ablate each stage in one decision trace. | P1 |
| [rubicon gollwitzer volition source excerpts](2026-10_rubicon_gollwitzer_volition_source_excerpts.md) | Rubicon divides goal deliberation from volitional execution. | Compare frozen goal intent against execution scheduler. | P1 |
| [wisdom frontier active inference delta and misspecification](2026-10_wisdom_frontier_active_inference_delta_and_misspecification.md) | Information seeking fails when the learned model or utility is misspecified. | Test wrong-model active inference against actual task benefit. | P1 |
| [minimal counterfactual edits are dependency witnesses](2102.03322_minimal_counterfactual_edits_are_dependency_witnesses.md) | Minimal counterfactual graph edits can witness dependencies. | Check graph edits for minimality and reproducible prediction flip. | P1 |
| [explanations need both necessity and sufficiency](2202.08816_explanations_need_both_necessity_and_sufficiency.md) | Necessity and sufficiency are separate explanation criteria. | Test retain-only and remove-only graph masks. | P1 |
| [intervene on information flow not only nodes](2205.08000_intervene_on_information_flow_not_only_nodes.md) | Causal intervention may alter edge information flow without deleting nodes. | Compare mediator-path interventions with blunt node ablation. | P1 |
| [true thinking score decorative cot ablation bookmark](2510.24941_true_thinking_score_decorative_cot_ablation_bookmark.md) | Displayed reasoning traces may be decorative rather than causally necessary. | READ FURTHER: pin benchmark protocol, then ablate trace vs answer. | P2 |
| [uncertainty chooses the next observation](2602.15355_uncertainty_chooses_the_next_observation.md) | Uncertainty can drive selection of the next informative observation. | Test budgeted selection against random and wrong-model controls. | P0 |
| [counterfactual search must exceed edge deletion](2603.04209_counterfactual_search_must_exceed_edge_deletion.md) | Counterfactuals include feature and edge additions as well as deletions. | Test a bounded edit grammar and genuine minimality. | P1 |
| [world model as replayable intervention algebra](2606.05925_world_model_as_replayable_intervention_algebra.md) | World models separate observations, prediction and intervention branches. | Replay two virtual futures without permitting self-certification. | P0 |
| [synthesize programs gate commit](2606.14581_synthesize_programs_gate_commit.md) | CARE illustrates program proposal separated from deterministic commit gates. | Inject malformed candidates and unauthorized-write refusals. | P0 |
| [prediction may update belief not purpose](2607.21273_prediction_may_update_belief_not_purpose.md) | Prediction rewards may accidentally replace utility with predictability. | Test dark-room and predictable-but-bad choices. | P0 |
| [necessity is measured by counterfactual disruption](2607.21573_necessity_is_measured_by_counterfactual_disruption.md) | Counterfactual disruption quantifies necessity, not mere salience. | Test temporal perturbations with reference necessity scores. | P1 |
| [no quorum without coordinate binding](2608.11034_no_quorum_without_coordinate_binding.md) | Replica majority is meaningless without aligned coordinates. | Test identity binding, tied votes and inconclusive cases. | P0 |
| [detached revision preserves visited states](2608.16333_detached_revision_preserves_visited_states.md) | Counterfactual repairs must not rewrite the observed trajectory. | Contrast detached student-state replay with contaminated rollout. | P0 |
| [projection closure finds epistemic boundaries](2608.20836_projection_closure_finds_epistemic_boundaries.md) | Cross-view reconstruction residuals can reveal epistemic boundary changes. | Check shuffled, aligned and changed-regime views. | P1 |

### E. Graphs, grammar, Hyperon and agent context

| Note | Takeaway | Next follow-up | Priority |
|---|---|---|:---:|
| [haspelmath-deepnsm-ce64-harness](2026-10_haspelmath-deepnsm-ce64-harness.md) | Semantic-map grammar needs executable ambiguity-preserving harnesses. | Run DeepNSM lexical alternatives with pinned source verses. | P1 |
| [haspelmath semantic maps grammar abi notes](2026-10_haspelmath_semantic_maps_grammar_abi_notes.md) | Crosslinguistic meaning geometry suggests grammar mappings, not a universal ABI. | Test typed verb/affordance map on language counterexamples. | P2 |
| [hyperon metagraph folds chronomorphisms primary excerpts](2026-10_hyperon_metagraph_folds_chronomorphisms_primary_excerpts.md) | Metagraph folds and chronomorphisms suggest compositional operators. | Translate one fixed Hyperon fold with oracle parity. | P1 |
| [hyperon metta to lance graph parity map](2026-10_hyperon_metta_to_lance_graph_parity_map.md) | MeTTa equivalence needs a versioned semantic projection matrix. | Close HY-Q gaps with an upstream evaluator probe. | P0 |
| [opencog pln shannon attention research map](2026-10_opencog_pln_shannon_attention_research_map.md) | PLN truth and Shannon uncertainty supply different scheduling signals. | Contrast decisions under a shared action budget. | P1 |
| [pln shannon bald surprisingness primary excerpts](2026-10_pln_shannon_bald_surprisingness_primary_excerpts.md) | BALD disagreement, Shannon entropy and surprisingness differ. | Build samples that reverse their priority rankings. | P1 |
| [sag1509 information metagraph optimization excerpts](2026-10_sag1509_information_metagraph_optimization_excerpts.md) | Information gain and metagraph folds suggest appraisal but not policy proof. | Measure utility with cost-matched surprise controls. | P1 |
| [sag1509 nars hyperon reflexive appraisal excerpts](2026-10_sag1509_nars_hyperon_reflexive_appraisal_excerpts.md) | NARS and Hyperon reflexive appraisal have distinct semantics. | Demonstrate whether self-appraisal changes subsequent choice. | P1 |
| [sag1509 source oracles falsifier matrix](2026-10_sag1509_source_oracles_falsifier_matrix.md) | Source citations are not executable oracle results. | Turn the SAG matrix into explicit pass/fail fixtures. | P0 |
| [sag agent attention active selection excerpts](2026-10_sag_agent_attention_active_selection_excerpts.md) | Relevant repository attention is selection, not maximal context. | Benchmark graph search under equal token/call budgets. | P1 |
| [sag ona internal experience source excerpts](2026-10_sag_ona_internal_experience_source_excerpts.md) | ONA internal experience depends on executable event semantics. | Pin upstream revision and trigger a causal self-event. | P1 |
| [sag repo graph context source excerpts](2026-10_sag_repo_graph_context_source_excerpts.md) | Structural repository links can outperform lexical lookups for some edits. | Measure code-location and edit success with equal retrieval budget. | P1 |
| [opencog hyperon 100 page reading bookmark](2310.18318v1_opencog_hyperon_100_page_reading_bookmark.md) | Hyperon monograph is queued reading, not a full source audit. | Read pinned sections before making parity claims. | P2 |
| [hologram is a graph theorem matches must preserve witnesses](2408.10592_hologram_is_a_graph_theorem_matches_must_preserve_witnesses.md) | Geometry theorem graphs need source-witnessed bindings beyond fast masks. | Check structure and attributes with separate graph matcher. | P1 |
| [graph attributes structure matching gasm bookmark](2409.20212_graph_attributes_structure_matching_gasm_bookmark.md) | Attribute-aware graph matching may influence canonical address design. | READ FURTHER: full methods, then adversarial collision fixtures. | P2 |
| [semantic maps top down crosslinguistic graph bookmark](2412.01423_semantic_maps_top_down_crosslinguistic_graph_bookmark.md) | Crosslinguistic semantic-map topology could constrain grammar mappings. | READ FURTHER: test structural universals against counterexamples. | P2 |
| [superposition capacity breaks two hop retrieval](2606.24948_superposition_capacity_breaks_two_hop_retrieval.md) | Reversible binding does not guarantee multi-hop superposition retrieval. | Compare two-hop capacity against exact DR addressing. | P1 |

### F. Identity, policy and provenance

| Note | Takeaway | Next follow-up | Priority |
|---|---|---|:---:|
| [iam access paths temporal provenance primary excerpts](2026-10_iam_access_paths_temporal_provenance_primary_excerpts.md) | IAM access paths require temporal and policy provenance. | Test grants, expired roles, nesting and why-access witnesses. | P1 |
| [iam negative provenance authority and agentic policy excerpts](2026-10_iam_negative_provenance_authority_and_agentic_policy_excerpts.md) | IAM denial needs negative provenance and external authority. | Test conflict/why-not cases, no invented absence proofs. | P1 |

### G. Philosophical context (not implementation evidence)

| Note | Takeaway | Next follow-up | Priority |
|---|---|---|:---:|
| [dialectics algorithmic determinations excess information grounds](2512.17373_dialectics_algorithmic_determinations_excess_information_grounds.md) | Dialectical determination offers conceptual hypotheses, not executable proofs. | Translate one idea to a falsifiable operator or keep contextual. | P2 |
| [hung vo 2025 the dialectic of ai hegel geist machine freedom](hung_vo_2025_the_dialectic_of_ai_hegel_geist_machine_freedom.md) | Hegelian AI progression is conceptual framing, not subjectivity evidence. | Record operationalizable claims separately from interpretation. | P2 |
| [plevrakis 2024 can ai be a subject like us hegelian approach](plevrakis_2024_can_ai_be_a_subject_like_us_hegelian_approach.md) | A speculative analysis of AI subjecthood is a philosophical argument. | Keep source transcription distinct from technical validation. | P2 |
| [van tuinen 2020 philosophy in the light of ai hegel or leibniz](van_tuinen_2020_philosophy_in_the_light_of_ai_hegel_or_leibniz.md) | Hegel/Leibniz parallels frame ontology questions, not benchmarks. | Extract testable propositions only where meaningful. | P2 |

## Maintenance and decision discipline

1. Add new papers as new notes with pinned primary source/version, explicit read grade, exact supported claim, separate project hypothesis and an independent kill test.
2. Add a row here on intake. If two documents share a falsifier, cross-link and run **one** canonical oracle instead of duplicating a speculative subsystem.
3. Link the actual test output / PR / pinned commit when a follow-up runs; only then promote `PROPOSED` to `TESTED`. A merged adjacent PR is not proof that a literature transfer was implemented.
4. For source transcriptions, preserve provenance and copyright/source boundaries; this index does not stand in for reading or reusing the entire text.
5. Preserve strict refusal fences: coordinate identity and source versions, bag/K3 semantics, exact versus approximate arithmetic, independent evidence roots, side-effect order, external commit authorization.
6. Re-audit coverage after each new file; source or project status may become stale after this snapshot.

**Coverage check:** 90 unique entries, 90 existing notes, 7 research groups. No omitted file in the snapshot. No existing note modified.
