# SPEC v1.1 (premise-audited) — CE64 / ogar-loco hardening: H1, H2, H5, H6, H7

Source of the items: stockfish-rs `docs/NARS-RECIPE-LOCO-PALETTE-INVENTORY-V1.md`
§ "Concrete → /5plus3" (PR #21). H3 is deferred to the coresearch council
(its premise is open: no recipe names a subject edge). H4 (ARE/ZCF/HKF
wire-or-delete) is an operator decision and is NOT in scope.

Panel declaration: default 5 (prior art, iron rules, code truth, cascade
impact, different views) + default 3 (overclaim, dilution-collapse, firewall).

## 1. FROZEN DECISIONS

F1. CE64 v2 is the default layout (`causal-edge/Cargo.toml` `default =
    ["causal-edge-v2-layout"]`); v1 is an opt-out compat build that must keep
    compiling (`default-features = false`).
F2. I-LEGACY-API-FEATURE-GATED (lance-graph CLAUDE.md): one name must not
    silently mean two things across feature flags; v1 paths route through the
    canonical mapping or are documented no-ops.
F3. D-PEARL-IO-0 (board entry 2026-10-07-pearl-io-0, TEST-PINNED): only bits
    40..42 select the Pearl operator; the mantissa and 59..63 are forbidden as
    opcodes (F11, F12). Nothing here may make the mantissa an opcode.
F4. Counterfactual's v2 mantissa is −6 (`edge.rs` `to_mantissa`), already the
    contract's tag on the replay path (`chain_counterfactual.rs:253`).
F5. ogar-loco engine owns exactly IF, IF_ELSE, REPEAT, WHILE, REPEAT_UNTIL;
    every other control byte is REFUSED with `RunError::UnhandledControlFlow`,
    never handed to a dialect (`interpret.rs:18-32`, and the precedent test
    `a_body_less_control_byte_is_refused_by_the_engine_not_offered_to_the_dialect`
    `interpret.rs:~1273`).
F6. mask-risc A2: "a mask bit is never a weight"; EWA ranks WHERE, never WHAT.
F7. Falsifiability rule (lance-graph CLAUDE.md): every guard ships a
    can-fire AND a can-stay-silent test; disable-runs red-then-green.
F8. No model identifier in any committed artifact; append-only board.

## 2. INPUT INVENTORY

H1 — two vocabularies over `0x90..=0xB1`:
- OGAR `crates/ogar-loco/src/nars.rs`: `NarsVocabulary` (`:133-170`),
  `domain_stack_arity` Datapath 2 / Gate 1 / Control 1 (`:142-147`),
  `domain_body_refs` 0 (`:153`), no value codebook; free fns `recipe_at`
  (`:94`), `tier_of` (`:109`), `is_mask_op` (`:115`), `byte_of_id` (`:121`).
  Consumers outside nars.rs: none found in OGAR src/tests/examples
  (grep `nars::`), none in lance-graph (grep `NarsVocabulary`).
- lance-graph `crates/lance-graph-ogar/src/recipe_vocab.rs`:
  `RecipeVocabulary` (`:138`), arity 1 for every recipe (test `:383`),
  `domain_value_codebook` = prefix-scoped attention-focus codebook (`:171`,
  test `:392`). Consumers: `recipe_vocab.rs`,
  `examples/recipe_dispatch_bridge_probe.rs`. Depends on ogar-loco via git
  `branch = "main"` (`lance-graph-ogar/Cargo.toml:137`).
H2 — PROC_DEF:
- `lib.rs:590` `PROC_DEF = FnIndex(0x83)`; `vocabulary.rs:247` body_refs 1,
  `:218` stack_arity 0.
- `interpret.rs:157` `CONTROL_BAND = 0x01..=0x1F`; `:178` `is_engine_control`;
  `:308-321` `run_body` hands every non-band byte to `dialect.call`.
- Doc promise: `interpret.rs:25` lists PROC_DEF as refused.
H5 — "masking ops" label: `nars.rs:2, :13, :18-21, :105-116`. The 9 Datapath
  recipes include TCA, MCT, MPC, SDD, HPM, SSAM whose kernels in
  `lance-graph-contract` `recipe_kernels.rs` operate on `ThoughtCtx` f32
  scalars / `Vec<f32>` candidates, not on mask lanes.
H6 — counterfactual encoding:
- `causal-edge/src/network.rs:171-213` `CausalNetwork::counterfactual`, `:211`
  `set_inference(InferenceType::Abduction) // counterfactual = abduction`.
  No caller outside the file (grep `.counterfactual(` hits only
  `learning/src/scm.rs`, a different SCM method). No test pins Abduction.
- `edge.rs:528-539` `set_inference`: v2 writes `to_mantissa()` (4 bits);
  v1 writes discriminant (3 bits).
H7 — doc drift: `causal-edge/src/lib.rs:9-15` and `edge.rs:138-150` show only
  the v1 layout (temporal 52..63).

## 3. PROPOSED RESOLUTION (committed)

H1. (PREMISE-SPLIT: two questions — where the mint lives, and whether
    Datapath arity 2 is a fact. It is not: `nars.rs:47-51,139-141` call it
    POLICY.) `RecipeVocabulary` is the ONE `FnSpec` mint for recipe bytes (arity 1 —
    the operand is the attention-focus address prefix — plus the prefix-scoped
    codebook). `NarsVocabulary`'s `impl Vocabulary` is DELETED from OGAR.
    `nars.rs` keeps only the id↔byte map and the bucket classifier
    (`recipe_at`, `byte_of_id`, `tier_of`). Rationale: the arity-2 Datapath
    claim is asserted from bucket, not from any kernel ("these numbers are this
    module's policy", `:140`); the arity-1 claim is test-pinned against a
    consumer. Landing order: OGAR PR first; lance-graph-ogar follows `main`
    automatically and needs no edit unless it imports the deleted impl (it
    does not).
H2. The engine owns every SHARED-CORE byte that either sits in
    `CONTROL_BAND` or references a body: `is_engine_control(f) :=
    CONTROL_BAND.contains(f.0) || (f.0 < DOMAIN_FLOOR && body_refs(f) > 0)`.
    PROC_DEF then reaches `run_branching` and is refused with
    `UnhandledControlFlow` (its existing `other =>` arm). The rule is
    layout-shaped, not a per-byte list (honours the module's own anti-drift
    note). Domain bytes (≥ `DOMAIN_FLOOR`) stay the dialect's even if a domain
    declares body refs. Doc `:25` becomes true; no doc change needed.
H5. (Premise audit: "masking op" is a KERNEL property; the label was
    derived from the catalogue's execution-tier bucket.) Fix the LABEL, incl.
    the file header `nars.rs:1-2, :13`. `nars.rs` stops calling Datapath recipes
    "masking ops": `is_mask_op` is renamed `is_datapath` (old name kept one
    release as `#[deprecated]` alias — but it has no consumers, so delete
    outright is acceptable; spec chooses DELETE since no consumer exists),
    and the module doc says Datapath = "uniform per-cycle tier", with the
    measured note that 6 of the 9 kernels are scalar ThoughtCtx arithmetic,
    not mask ops. Mask migration of those kernels is coresearch O-work, not
    this change.
H6. (RE-ASKED after premise audit: PREMISE-SPLIT.) The mantissa carries two
    concepts here: the inference rule that produced the edge (read as an
    instruction by `forward`, `edge.rs:~686`) and the rung-3 provenance tag
    (−6, `chain_counterfactual.rs:253`). The rung itself already has its own
    carrier, `ReasoningBand::Counterfactual` (`layout.rs:~331`; band and −6
    pinned as independent, `tests/v2_layout_tests.rs:896-925`).
    Committed resolution (this council): `CausalNetwork::counterfactual`
    ADDS `.with_reasoning_band(ReasoningBand::Counterfactual)` to its result
    (v1: documented no-op, `edge.rs:1301`). The mantissa write is UNCHANGED in
    this change; the comment `// counterfactual = abduction` is replaced by
    one stating the mantissa records the rule, the band records the rung.
    Tests: v2 band == Counterfactual after `counterfactual`; silence twin: a
    `causal_query` hit is not banded Counterfactual.
    DEFERRED to coresearch (Q3b): whether the −6 provenance tag on the replay
    path should also move to the band, and what `forward` must do when a
    weight edge carries −6.
H7. Both doc sites show the v2 layout as primary (S 0..7, P 8..15, O 16..23,
    F 24..31, C 32..39, mask 40..42, direction 43..45, mantissa 46..49 i4,
    plasticity 50..52, W 53..58, Epi5 59..63) and keep the v1 table under a
    "v1 (default-features = false)" heading. Doc-only.

## 4. NON-GOALS
- H3 (C-byte sink) — premise open, coresearch.
- H4 (Demonstrations) — operator decision.
- Migrating any recipe kernel to mask algebra — coresearch.
- Implementing PROC_DEF execution — needs its own falsifier per F5.
- Changing `forward`'s destruction of W / Epi5 — D-CE64-LOOP-0 open item.

## 5. PRE-REGISTERED GATES
G1 OGAR: `cargo test -p ogar-loco` green; new test
   `proc_def_is_refused_by_the_engine_not_offered_to_the_dialect` (anti-vacuity:
   asserts body_refs(PROC_DEF)==1 and 0x83 ∉ CONTROL_BAND) red under the old
   predicate, green under the new; silence twin: a domain byte ≥ 0x90 still
   reaches the dialect; and a domain vocabulary cannot raise body_refs on a
   byte < DOMAIN_FLOOR (test).
G2 OGAR: clippy -D warnings, fmt clean; no remaining `impl Vocabulary for
   NarsVocabulary`; grep `masking op` in nars.rs returns 0.
G3 lance-graph: `cargo test -p causal-edge` green under default AND
   `--no-default-features`; new counterfactual test disable-verified.
G4 lance-graph: `cargo test -p lance-graph-ogar` green against OGAR branch.
G5 Board: one lance-graph `.claude/board/entries/` entry + entries_index.py
   --write; inventory doc (stockfish-rs) H-rows updated with PR links.

## 6. PER-SAVANT QUESTION SETS

S1 prior art: (1) Is any of H1/H2/H5/H6/H7 already fixed or decided in a board
  entry, ISSUES row, or open PR? (2) Is there an existing ruling that NARS
  recipe arity is 2 for Datapath? (3) Was "counterfactual = abduction" ever an
  operator ruling? (4) Does an existing ID cover the PROC_DEF gap?
S2 iron rules: (1) Does H6 violate I-LEGACY-API-FEATURE-GATED under v1?
  (2) Does H2's predicate let a domain vocabulary forge engine control? (3)
  Does any change make the mantissa an opcode (F3)? (4) Does H1's deletion
  break append-only or a published API rule? (5) AP1-AP9 hits?
S3 code truth: verify every file:line in §2 as CODED/CLAIMED/ABSENT; in
  particular (1) `run_body` passes 0x83 to the dialect; (2) no consumer of
  `NarsVocabulary`/`is_mask_op`/`tier_of` exists in OGAR, lance-graph,
  stockfish-rs, MedCare-rs, a2ui-rs; (3) no test pins Abduction for
  counterfactual; (4) `run_branching`'s fallback arm refuses PROC_DEF; (5)
  `Vocabulary` exposes `body_refs` to the interpreter.
S4 cascade: list every file/test/doc/board row that must change for each H,
  split mandatory vs follow-up; cross-repo order (OGAR before lance-graph?).
S5 different views: strongest alternative reading for H1 (should NarsVocabulary
  win?), H5 (fix kernels instead?), H6 (keep Abduction and make the replay
  path −1?); and one second-order consequence per item. No redesigns.

Output contract per savant: ≤10 findings, each `(Q#, CONFIRMS | VIOLATES |
GAP | PRIOR-ART-AT | RISK, file:line, ≤2 sentences)`. Read-only.
