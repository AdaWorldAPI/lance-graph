# D-SMD: semantic Morton grammar / qualia-masked DO falsifier plan

**2026-10-13 | DRAFT | research + implementation handoff, no authorization to change ABI or action permissions.**
Companions:
- docs/research/semantic-morton-qualia-masked-do-rfc.md
- .arxiv/2026-10_haspelmath_semantic_maps_grammar_abi_notes.md
- Existing: #1496, #1497, #1498, #1500, #1501, #1509 and docs/architecture/ce64-semantic-upper-half.md

## One-sentence implementation target

Build and falsify a *versioned, cross-linguistic semantic graph* projected into an 8/12/16-bit **pure Morton** cascade, with **Moore8 local semantic-pressure probes**, separately typed **qualia-conditioned candidate masks**, and an **OGAR-authorized DO boundary**, reusing ClassView/Quack/R2IL/mask-risc/ndarray and leaving the physical register contract unchanged.

## Hard boundaries (all steps)

1. DO NOT repurpose the fourth Morton nibble for qualia. 8 bits -> 16×16; 12 bits -> 64×64; 16 bits -> 256×256. Each nibble refinement adds a 4×4 grid of children. Prefix property is a fixed convention; test it.
2. DO NOT declare the 144 slots universally real without checking the actual source table, its axes and linguistic data. Distinguish 12×12 predicate families, VerbRoleTable/TAM and Palette256; do not assume one is the other.
3. DO NOT rewrite current CE64 semantics. Refer to actual layout.rs and doc/architecture source. Bits 43..52 are occupied; treat novel Moore/qualia readings as versioned external projections.
4. DO NOT build a new executor, inventory, identity model or authority registry. Prefer ClassView/HotPlug, OGAR/OGIT, Quack and R2IL to a fresh stack.
5. DO NOT interpret unknown authorization or unread semantic content as allow, and do not let Moore pressure change committed goals.
6. DO NOT claim objective access to subjective qualia; operationalize internal self-state monitoring + controlled causal influence on decisions.
7. DO NOT use language “universal grammar,” “holographic” or “resonance” as benchmark conclusions before independent experiments.
8. No silent codebook renumbering / mapping reinterpretation after persistence. New observations add graph versions; migrations are explicit.

## Phase 0 — audit existing truths (D-SMD-0, documentation first)

Inspect actual paths/symbols/versions in current HEAD:
- 12×12 predicate/role tables, palette IDs and their actual semantic axes.
- Morton HHTL or other address codecs; verify nibble ordering, parent/child convention and level boundaries with scalar examples.
- Existing Moore tenant, 8-neighbor neighborhood implementation, Staunen/prediction/verification/plasticity.
- ClassView / WideFieldMask / HotPlug / schema seal; Quack and mask-risc lowering; R2IL FoldDialect.
- OGAR MethodDecl/ActionDef and actual DO authority path, scope and missing approvals.
- CE64 vs documentation split of encoding, reading, operation.
- Existing research plans D-NALP / D-HL / Rubikon / PL(N)/Shannon, current branch vs merged truth.

Output: a compact **source inventory with links and line ranges** plus a mismatch ledger. Every planned signature must be marked existing or proposed. No implementation before this audit is reviewed.

## Phase 1 — scalar reference (D-SMD-1)

Write a minimal *isolated scalar test harness*, not a public API first:
- Known Morton prefix vectors for XY in [0,255]^2, nibble MSB->LSB coarsening.
- Bijective interleave/deinterleave 16-bit addresses; 8-bit/12-bit prefix consistency.
- All 144 candidate palette cells mapped into **unique** 8-bit positions, 112 explicitly reserved/unassigned; generate deterministic fixtures and mapping seal.
- Out-of-bounds, reserved, invalid reading, unknown schema and legacy address tests.
- Stable semantic-node-ID roundtrip independently of Morton position.
- 2D grid baseline and unquantized graph adjacency baseline.

Kill if parent/child arithmetic fails or a codebook upgrade changes legacy semantic IDs without a migration/adapter.

## Phase 2 — grammar topology (D-SMD-2)

From Haspelmath maps, construct a tiny reviewed provenance corpus; label **grammatical functions** separately from word forms, and tag unresolved/ambiguous cases. Include:
- English *to* / French *à* across purpose/direction/recipient/experiencer.
- Indefinite usage across English/Russian.
- Reflexives/middle/reciprocal.
- Comitative/instrumental/cause/source.
- A diachronic extension case.

Split held-out languages/functions before fitting edges; no test contamination. Compare:
1. lexical-string identity,
2. flat 144-category dictionary,
3. witnessed graph with no spatial embedding,
4. grid embeddings: row-major/random/Morton/spectral or learned,
5. higher-rank/sparse projection where neighborhoods conflict.

Measure language coverage, false merging, out-of-map rate, connected-region violations, structural graph distortion and nonlocal edges.

Kill the canonical 12×12 wheel claim if measured topology is not cyclical or two-dimensional; keep the graph and use a versioned imperfect projection if still useful.

## Phase 3 — geometric semantic pressure (D-SMD-3)

Use identical source data and same number of candidate evaluations across these schedulers:
- FIFO,
- evidence/confidence priority,
- Shannon entropy from an explicit probability model,
- calibrated expected information gain,
- Moore8 pressure from embedding neighbors,
- direct semantic graph-neighbor pressure,
- blended Moore8 + explicit nonlocal edges.

For the Moore8 term define a stable directional mask, signed weights and boundary handling. Run “missing predicate bridge,” distractor, topological bottleneck and false-friend probes. Report oracle-supported discoveries per decoded byte, per evaluation, elapsed time and false-positive edges. Hold inference rules constant (see #1497).

Kill pressure locality if it has no measured advantage or promotes irrelevant high-entropy distractors.

## Phase 4 — qualia in the control loop (D-SMD-4)

Establish a versioned **QualiaReading** with externally testable/internal-monitoring signals, not a subjective-experience assertion. Candidate axes: valence, urgency/arousal, agency, volition, novelty, resolution/care. Freeze the operational definitions and avoid covertly reading existing CE64 bits as these axes.

With goal, evidence, policy, rule set, observation and computational budget held fixed:
- baseline ignores qualia,
- full qualia-conditioned mask,
- each dimension removed independently,
- scrambled-state ablation,
- adversarial perturbation,
- duplicated-evidence control,
- unknown/unread state handling.

Measure selection stability, behavioral differences, goal completion, preference/constraint violations and explainable intervention witnesses. Demonstrate state updates lead to **predicted** action-selection changes, not just attractive labels.

Storage comparisons:
- dense 12×12×u8 (144 bytes);
- signed 12×i4 and 12×i4 outer-product factors (96 bits = 12 bytes);
- rank-2/3 with sparse corrections;
- 32-bit two 12-bit Boolean factor masks + 8-bit nuance;
- separate per-occurrence 24/32-bit qualia carriers.

Mandatory rank-1 counterexample: activate only (r0,c0) and (r1,c1), while (r0,c1) and (r1,c0) must remain inactive. A single Cartesian-product Boolean mask cannot represent this; rank-1 real factorization likewise cannot represent the diagonal 2×2 matrix.

Kill compressed representations that drop critical semantic distinctions or create unauthorized DO candidates.

## Phase 5 — ClassView -> masked DO (D-SMD-5)

After audit, build only missing adapters:
- Source utterance/construction -> witnessed function candidates/ambiguity.
- Semantic codebook -> stable semantic IDs and versioned Morton projections.
- Typed ClassView/WideFieldMask -> Quack and R2IL matching selection.
- Masked candidate -> OGAR ActionDef validation (authority, scopes, idempotency, approval).
- Execution/witness -> CE64 observed causal-evidence update, NOT inferred causal status from graph proximity.
- Error path: malformed language adapter, missing mapping, unsigned codebook, unsatisfied authority -> reject or explicit unresolved.

Test method dispatch independently of physical semantics. Insist on version-sealed method binding and no promotion of planning to actuation.

## Phase 6 — “Borg-able” and “holographic” falsifiers (D-SMD-6)

- **Assimilation:** add a held-out language and compare map coverage before/after without changing past examples or contaminating old semantic IDs. Show language-specific residual mappings.
- **Resonance:** measure overlapping activation neighborhoods beyond a flat translation dictionary; counterexample languages must not be merged.
- **Holography:** hide portions of expression-to-function evidence and reconstruct held-out links/roles using distributed overlapping witnesses. Require a measurable benefit over nearest-neighbor lookup and simple graph propagation; otherwise use the more accurate term “semantic projection.”
- **Foveation:** compare coarse→fine Morton exploration with flat search at equal decode/compute budget, no global dense scan.
- **Safety:** adversarial semantic aliasing, cross-tenant collisions, provenance forgery and DO bypass attempts.

## Concrete reviewer acceptance checks

- [ ] Exact repository source inventory and stated unknowns, no fabricated methods.
- [ ] Scalar Morton encode/decode proof vectors and roundtrip, including boundaries.
- [ ] Persistent semantic IDs decoupled from spatial coordinates, versioned projection.
- [ ] 144 mapping and 112 remaining coarse cells treated explicitly.
- [ ] Haspelmath functions and per-language regions grounded in cited figures, plus held-out cases.
- [ ] Dense, rank-1 and sparse-corrected representations compared with diagonal counterexample.
- [ ] Equal-budget Shannon/EIG/Moore attention and ablation controls.
- [ ] Qualia observation/control probes vs scrambled/baseline signals.
- [ ] Read/semantic/goal/qualia/action masks distinct and unknown-safe.
- [ ] Only OGAR authorizes actual DO; CE64 causal claims earned by intervention and witness.
- [ ] Quack/R2IL parity demonstrates actual matching semantics, not only numeric output.
- [ ] No change to current byte/ISA layout, no CE64 reinterpretation, no unreviewed register override.
- [ ] All purported “universals” report negative evidence and falsifier outcomes.

## Copyable follow-up prompt for another coding session

> Starting at latest AdaWorldAPI/lance-graph main, read docs/research/semantic-morton-qualia-masked-do-rfc.md and .arxiv/2026-10_haspelmath_semantic_maps_grammar_abi_notes.md, then this plan, the source-of-truth CE64 layout and linked #1496/#1497/#1498/#1500/#1501 plans. First produce exact, line-linked inventory of the actual 12×12 palette and Moore tenant, ClassView/HotPlug, OGAR ActionDef and Morton/HHTL address codecs. Record discrepancies. Do not change hardware bits, introduce a second executor or assert universal grammar. After the audit, implement only an independent scalar Morton 8/12/16-bit reference plus test vectors and semantic-node-ID/versioned-projection fixtures in a separate narrowly scoped PR. Run unit/property tests, report failing counterexamples, and stop if canonical semantic-map geometry cannot be supported by evidence. Subsequent phases require separate sign-off.

**This plan is intentionally staged; the present PR ships research/documentation only.**
