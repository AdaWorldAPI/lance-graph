# Convergence Architect — Expansion Pattern Catalogue

> **READ BY:** `convergence-architect` (canonical); also any meta-Opus
> reviewer scanning for alignment opportunities between subsystems;
> any planner (PP-N) asking whether a proposed migration has hidden
> 0-friction properties before committing to its cost estimate.
>
> **Status:** FINDING-SEEDED (each EP below is grounded in at least one
> concrete workspace instance observed in sprint-11, sprint-12, or the
> sprint-13 preflight; no EP is purely theoretical).
>
> **Predecessors:**
> - `CLAUDE.md §The-Click` — the core invariant that all EPs orbit
> - `.claude/knowledge/iron-rules-doctrine.md` (PP-2) — the four-axis
>   framing that EPs must align to before promotion
> - `.claude/board/sprint-log-13/preflight-meta-review-opus.md` W-Meta-Opus
>   §2 — per-planner table with EP-class observations
> - `.claude/agents/convergence-architect.md` (PP-14) — the agent card that
>   activates this catalogue
>
> **Maintenance:** APPEND-ONLY within §2 (the EP catalogue). New EPs are
> appended after EP8; existing entries are never edited (corrections append
> as a dated annotation). Promoted EPs are annotated but not deleted.

---

## §1  What the Convergence-Architect Sees

The four iron rules in `CLAUDE.md §Substrate-level iron rules` bound the
current substrate:

| Iron rule | Axis | What it forbids |
|---|---|---|
| `I-SUBSTRATE-MARKOV` | Substrate operator | Non-commutative binding in transition kernels |
| `I-NOISE-FLOOR-JIRAK` | Statistical model | Classical Berry-Esseen on weakly-dependent bits |
| `I-VSA-IDENTITIES` | Data semantics | Superposing content registers instead of identity fingerprints |
| `I-LEGACY-API-FEATURE-GATED` | API version | v1 accessors silently writing into v2 reclaim zones |

These four rules CLOSE four axes. But the workspace continues to evolve:
new subsystems appear, new sprint deliverables land, new CSI observations
accumulate. Each time the workspace grows, there are new places where two
things that look like two things are actually one thing — and the boundary
between them was always meant to vanish.

**The gap between the four iron rules and the next iron rule is where
0-friction boundaries live.**

The convergence-architect's scan is a systematic attempt to find that gap:
to look at every pair of subsystems, DTOs, carriers, or sprint deliverables
and ask: "Does an algebra already exist that collapses the boundary between
these to a no-op? If not, is there a structural reason why — or has the
alignment simply not been noticed yet?"

When the alignment IS noticed, it produces one of three outcomes:

1. **OPPORTUNITY-NOW:** the alignment is immediately actionable as a sprint
   deliverable (0-friction migration, single-primitive wiring, etc.).
2. **WORTH-EXPLORING-SOON:** the alignment is plausible but needs a probe
   (a grep, a benchmark, or a paper search) before committing.
3. **IRON-RULE CANDIDATE:** the alignment, once named, turns out to be a
   structural guarantee that should constrain all future implementers —
   i.e., the pattern is substrate-level, has N >= 3 instances, and a
   backing citation exists or can be found.

The catalogue below names 8 expansion patterns (EP1..EP8) observed or
seeded from the workspace's sprint-11/12/13 preflight history. The list
is neither exhaustive nor closed. EP9+ will emerge from future sprint scans.

---

## §2  Expansion-Pattern Catalogue EP1..EP8

Each entry follows the canonical shape:
- **Name** — short, memorable identifier
- **Shape** — the abstract structural setup
- **Workspace instance** — at least one concrete cite (file:line or PR/CSI)
- **Grep target** — how to find similar patterns elsewhere
- **Why it's a 0-friction boundary** — the algebraic alignment
- **Promotion track** — what would make this an iron rule

---

### EP1 — 0-Friction Baton Handover

**Name:** 0-friction baton handover

**Shape:** Two consecutive processing cycles C1 and C2 where
`C1.emitted_fp == C2.expected_input_fp`. The cursor that terminates C1 is
algebraically identical to the cursor that initializes C2. Passing the
cursor IS the handoff; no allocation, no translation, no bridging struct.

More precisely: if C1 produces output with fingerprint `fp_out` and C2
consumes input expecting fingerprint `fp_in`, and if `fp_out == fp_in` by
construction (both are computed from the same role-indexed identity in the
same VSA carrier), then the boundary is a no-op. The baton handoff costs
nothing because the runner's hand and the relay line are the same geometry.

**Workspace instance:**
`ShaderDispatch.input_fingerprint == prior_cycle.emitted_cycle_fingerprint`
in `crates/cognitive-shader-driver/src/driver.rs`. The dispatch cycle
emits a `cycle_fingerprint` that indexes into the next dispatch's
`input_fingerprint` slot. The cursor closes the loop without allocation.
See `CLAUDE.md §The-Click` for the canonical formulation (the Markov ±5
trajectory braid is the multi-cycle generalization of this single-handover
pattern).

**Grep target:**
```bash
# Find all fingerprint identity comparisons across cycle boundaries:
grep -rn "cycle_fingerprint\|emitted_fp\|input_fingerprint" crates/
# Find all dispatch cycle closures:
grep -rn "fn dispatch\|fn emit\|fn handoff" crates/cognitive-shader-driver/
```

**Why it's a 0-friction boundary:** the role-indexed identity fingerprint
is computed once (at VSA bind time) and shared across cycle boundaries
by construction (I-VSA-IDENTITIES: VSA operates on identity fingerprints
that POINT TO content, never on content itself). Two cycles that share a
role-indexed identity fingerprint share a pointer to the same content —
the boundary between them is a pointer comparison, not a copy.

**Promotion track:** N=2 so far (ShaderDispatch + the CLAUDE.md §The-Click
Markov ±5 formulation). Needs N=3 (a third concrete cycle-to-cycle handoff
in the codebase that exhibits the same property) + a backing citation
(Markov Chapman-Kolmogorov guarantee, per I-SUBSTRATE-MARKOV, is already
an iron rule — the question is whether the specific cursor-close-loop
pattern deserves its own rule or is a corollary of I-SUBSTRATE-MARKOV).

---

### EP2 — Algebraic-Operator Reuse Across Layers

**Name:** Algebraic-operator reuse across layers

**Shape:** Two layers L1 and L2 implement the same algebraic operation OP
independently (with different names, in different crates). Wiring them to
the same primitive eliminates the translation cost AND ensures the Markov
guarantee propagates across layers (since it's the same operator in both
places, not two operators that happen to be similar).

Formally: if `L1.op_A(x, y) == vsa_bundle(x, y)` and
`L2.op_B(x, y) == vsa_bundle(x, y)`, then routing both through the same
`crystal::fingerprint::vsa_bundle` call site means any proof about
`vsa_bundle` (associativity, commutativity in expectation, CK consistency)
applies to both L1 and L2 without re-proving.

**Workspace instance:**
`CollapseGate::Bundle` in `crates/lance-graph-planner/src/strategy/collapse_gate.rs`
and `vsa_bundle` in `crates/crystal/src/fingerprint.rs`. The planner's
collapse gate performs a saturating element-wise add over VSA bundles; the
crystal substrate's `vsa_bundle` is the same saturating element-wise add.
Per I-SUBSTRATE-MARKOV, the CK property holds for `vsa_bundle` by
construction — wiring `CollapseGate::Bundle` to call `vsa_bundle` directly
makes the planner Markov-correct AND zero-overhead per the iron rule.

**Grep target:**
```bash
# Find all bundle/superposition operations in the planner:
grep -rn "Bundle\|superpose\|bundle\|saturating_add" crates/lance-graph-planner/
# Find all vsa_bundle call sites in crystal:
grep -rn "vsa_bundle\|vsa_bind\|vsa_cosine" crates/
```

**Why it's a 0-friction boundary:** the two operators are definitionally
equal. Collapsing them eliminates a layer of indirection and aligns the
planner's semantics with the crystal substrate's proven guarantees
(I-SUBSTRATE-MARKOV). The cost of the collapse is one import; the benefit
is proof-transitivity across layers.

**Promotion track:** this EP is a direct corollary of I-SUBSTRATE-MARKOV.
It does not need its own iron rule — it IS the implementation consequence
of the existing rule. The right action is to flag every divergent bundle
operation in the codebase as a P1 finding (route to brutally-honest-tester)
rather than promoting EP2 separately.

---

### EP3 — Role-Keyed Identity Migration as One-Liner

**Name:** Role-keyed identity migration as one-liner

**Shape:** A planned migration from type A to type B appears expensive
(struct rewrite, data migration, test matrix), but the actual cost approaches
one line because the role-keyed identity fingerprints that TYPE B requires
already exist at TYPE A's usage sites. The migration is not "rewrite the
data" but "tell the compiler that what you already have is the right type."

Formally: if `A.field_X` is a role-keyed identity fingerprint in
`Vsa16kF32` (per I-VSA-IDENTITIES, Layer 2: domain role catalogues), and
if `B` is a new type that requires a role-keyed identity fingerprint in
`Vsa16kF32` for the same role, then the migration is:
`let b = B { fp: a.field_X };` — one line, zero allocation, zero new
computation.

**Workspace instance:**
`WitnessIndexHashMap` → real CAM-PQ wiring via bind-then-compress (D-CSV-16,
PP-5 spec `.claude/specs/pr-sprint-13-witness-cam-pq.md`). The migration
from the placeholder `WitnessIndexHashMap` to a real CAM-PQ index is one
line because the role-keyed identity fingerprints already exist in
`WitnessCorpus`. The CAM-PQ codec's `bind-then-compress` path accepts
the same `Vsa16kF32` identity fingerprints that the WitnessCorpus already
carries. See also CSI-15 (CamPqIndexPlaceholder → WitnessIndexHashMap
rename in PR #390) for the immediate predecessor migration.

**Grep target:**
```bash
# Find all placeholder types that may be role-keyed identity migration targets:
grep -rn "Placeholder\|Stub\|Todo\|WitnessIndex" crates/
# Find all bind-then-compress / CAM-PQ wiring patterns:
grep -rn "bind_then_compress\|cam_pq\|CamPq\|CamCodec" crates/
```

**Why it's a 0-friction boundary:** I-VSA-IDENTITIES guarantees that
role-keyed identity fingerprints in `Vsa16kF32` are the canonical Layer-2
currency of the workspace. Any type that accepts such fingerprints can be
wired to any other type that emits them without translation — the shared
type IS the interface. A migration between two types that both use this
currency costs only the rename; the data was already in the right format.

**Promotion track:** this EP is a direct consequence of I-VSA-IDENTITIES.
Like EP2, it may not need its own iron rule — it IS the migration-cost
corollary of the existing rule. Promotion would be appropriate if a pattern
of N >= 3 "unexpectedly expensive migrations that turned out to be one-liners
because role-keyed fingerprints already existed" is observed. Current N=1
(D-CSV-16). Watch sprint-13 close.

---

### EP4 — Carrier-as-Grammar

**Name:** Carrier-as-grammar

**Shape:** A struct S whose field set F = {f_1, ..., f_N} can be bijectively
mapped to a set of grammatical roles R = {r_1, ..., r_N} such that each
field embodies EXACTLY one grammatical role. When this bijection exists, the
struct IS the grammar — not a container that carries grammar, but a reified
grammar where field layout encodes cognitive/linguistic role.

The canonical instance: the `Think` struct from `CLAUDE.md §The-Click` maps
directly to TEKAMOLO (the German mnemonic for the temporal/causal/manner/
local/object order). Trajectory = Subject (what is being thought about).
Awareness = Modal (how confidently). Free_energy = Kausal (why this thought).
Resolution = Predicate (what it concludes). Graph = Lokal (where in fact-
space). Episodic = Temporal (when, relative to prior). The DTO carries
cognition the way a photon carries electromagnetism — not as payload, as
identity.

**Workspace instance:**
`crates/lance-graph-contract/src/thinking.rs` — the `ThinkingStyle` struct
and its field modulation axes map to the TEKAMOLO roles per
`CLAUDE.md §The-Click`: "Parsing text and parsing thought use the same
role-indexed slices — because thinking about a sentence and thinking about
thinking use the same algebraic substrate." See also PP-4 Think methods spec
(`.claude/specs/pr-sprint-13-think-methods.md`) §2.1: "This is NOT yet the
eight-field doctrinal Think" — the PP-4 spec identifies the minimum-viable
migration step toward the carrier-as-grammar property.

**Grep target:**
```bash
# Find Think struct candidates (multi-field structs with VSA/awareness fields):
grep -rn "struct Think\|trajectory.*Vsa\|awareness.*Param\|free_energy.*Free" crates/
# Find TEKAMOLO role annotations in existing code:
grep -rn "TEKAMOLO\|Subject\|Predicate\|Temporal\|Kausal\|Lokal" crates/ .claude/
```

**Why it's a 0-friction boundary:** when the struct's field layout IS the
grammatical role inventory, there is no translation layer between the DTO
and the grammar. `think.trajectory` does not map to Subject — it IS Subject.
The boundary between "data container" and "cognitive grammar" collapses to
zero because the carrier's field indices ARE the role-indexed VSA slices
(per §The-Click: SUBJECT[0..4K), PREDICATE[4K..8K), etc.).

**Promotion track:** WORTH-EXPLORING as iron rule. The carrier-as-grammar
invariant is a form of I-VSA-IDENTITIES (operate on identity fingerprints,
not content) applied at the struct level rather than the bit level. A
fifth iron rule "I-CARRIER-AS-GRAMMAR: a struct whose fields are role-
indexed identity fingerprints MUST NOT be used as a generic container for
unrelated data — its field layout IS its API contract" would be consistent
with the four-axis framing if the "data semantics" axis can accommodate
a struct-level extension. Needs meta-Opus review.

---

### EP5 — Multiple Iron Rules Collapse to a Meta-Pattern

**Name:** Multiple iron rules collapse to a meta-pattern

**Shape:** N existing iron rules share a common abstract shape S. The shared
shape S is itself a meta-rule: it predicts what axis the (N+1)-th iron rule
will occupy. This prediction is an OPPORTUNITY to look for patterns along
the predicted axis BEFORE they become defects.

In the current workspace: the four iron rules (per iron-rules-doctrine.md
§1) pin four axes: substrate operator, statistical model, data semantics,
API version. These four axes are orthogonal — each one is a different KIND
of substrate guarantee. The meta-pattern predicts: a fifth iron rule will
occupy a fifth axis (possibly memory ordering, ABI-stability under codebook
rebase, or workspace-member/exclude consistency).

**Workspace instance:**
`.claude/knowledge/iron-rules-doctrine.md` §1 axis table (PP-2 output,
2026-05-16) + §5.5 "Candidates currently under N >= 3 watch." The doctrine
itself is the meta-pattern artifact — PP-2 surfaced it explicitly and named
the axis table that makes the prediction testable.

**Grep target:**
```bash
# Find all candidate fifth-axis patterns in TECH_DEBT and sprint-logs:
grep -rn "ABI.stability\|memory.order\|workspace.*member\|board.hygiene" \
  .claude/board/ .claude/knowledge/
# Find all iron-rule citations to see which axes are already bound:
grep -rn "I-SUBSTRATE-MARKOV\|I-NOISE-FLOOR\|I-VSA-IDENTITIES\|I-LEGACY" \
  .claude/ crates/
```

**Why it's a 0-friction boundary:** the meta-pattern collapses the work of
"find the next iron rule by accident" to "look at the §5.5 watch list and
run the §3 promotion checklist." The boundary between "we don't know what
the next iron rule will be" and "we have a structured prediction about which
axis it occupies" collapses to one read of iron-rules-doctrine.md §1.

**Promotion track:** EP5 is the doctrine-discovery EP. It does not produce
a new iron rule itself — it produces the PREDICTION that constrains where
to look for the next one. The promotion track is: (a) confirm which of the
§5.5 candidates reaches N >= 3, (b) run the §3 checklist on that candidate,
(c) promote via the ceremony. EP5 itself is retired by the existence of
iron-rules-doctrine.md — the meta-pattern has been named.

---

### EP6 — Sprint-N Bug Becomes Sprint-(N+1) Feature

**Name:** Sprint-N bug becomes Sprint-(N+1) feature

**Shape:** A CSI-N defect observation from sprint-N's meta-review is
rephrased positively: the invariant that SHOULD have held (but didn't,
causing the defect) becomes a new iron-rule candidate or expansion
opportunity for sprint-(N+1). The defect reveals the correct invariant by
negation; the convergence-architect names the positive form.

Formally: if CSI-N = "pattern P was violated, causing defect D," then the
positive framing is "invariant NOT-P is a 0-friction boundary: when NOT-P
holds, D cannot occur." The question for the convergence-architect is:
"Does NOT-P already hold at other sites in the codebase? Could we make
NOT-P hold universally at zero cost?"

**Workspace instance:**
CSI-19 D-CSV-* numbering drift (sprint-13 preflight-meta-review-opus.md §3).
The defect: planners chose IDs independently without a coordinator, causing
cross-spec references to point at the wrong deliverables. The positive
framing (EP6): "An ID-coordination invariant — every D-CSV-N and OQ-CSV-N
is assigned by a single coordinator pass before any planner references it —
is a 0-friction boundary if baked into the preflight worker prompts.
The cost of adding the coordinator pass is ~15 LOC in worker prompts; the
cost of NOT having it is the CSI-19 class of drift per sprint."

A second instance: the `lib.rs` orphan-module pattern (AP4 in
codex-p1-anti-patterns.md, CSI-8 from sprint-11). The defect: new .rs
files not registered in lib.rs. The positive framing: "a module-registration
invariant — every new .rs file is paired with a lib.rs entry in the same
commit — is 0-friction because the registration cost is one line per file."
This became a process discipline in worker-template-v2.md §5.1.

**Grep target:**
```bash
# Find all CSI observations across sprint-logs:
grep -rn "CSI-[0-9][0-9]*" .claude/board/sprint-log-*/
# Find all anti-pattern entries that have a positive-framing opportunity:
grep -rn "was violated\|causing defect\|missed\|orphan\|drift" \
  .claude/knowledge/codex-p1-anti-patterns.md
```

**Why it's a 0-friction boundary:** a defect that recurs (N >= 2 in
sprint-logs) already has an observed-instance count approaching the iron-rule
threshold. Rephrasing it positively converts the defect-hunting budget into
a prevention budget — and prevention is cheaper than detection per sprint.

**Promotion track:** EP6 is a meta-pattern about iron-rule discovery
(same class as EP5). Individual EP6 applications produce specific OPPORTUNITY-
NOW items (e.g., "add coordinator pass to preflight prompts"); they do not
promote EP6 itself. EP6 is retired when the workspace has a systematic CSI→
positive-framing review step baked into every meta-review template.

---

### EP7 — Cross-Repo Synergy Without API Changes

**Name:** Cross-repo synergy without API changes

**Shape:** Two repos R1 and R2 both independently implement or call a runtime
detection singleton S. Wiring them to the SAME singleton eliminates both the
duplication AND the divergence risk (if R1 and R2 each have their own S,
they can give different answers about the same hardware — a category of
silent inconsistency).

The 0-friction property: if S is already a singleton in R1 (e.g., computed
once at process startup via `once_cell` or `std::sync::OnceLock`), and if
R2 can call R1's S directly via a dep-injection or a feature flag without
changing R2's public API, then the wiring is: add the dependency + call the
existing API. Zero new logic, zero API change.

**Workspace instance:**
`ndarray`'s `simd_caps()` singleton at
`/home/user/ndarray/src/hpc/simd_caps.rs` + `lance-graph`'s planned SIMD-i4
dispatch (D-CSV-13b, PP-6 spec `.claude/specs/pr-sprint-13-simd-i4.md`).
Both subsystems need to detect AVX-512 / NEON capability at runtime. The
ndarray `simd_caps()` singleton is the proven pattern (cited in PP-6 §2:
"Follows the ndarray `simd_caps()` proven pattern"). Wiring lance-graph's
dispatch to call the same singleton eliminates independent AVX-512 detection
in two repos.

**Grep target:**
```bash
# Find all runtime SIMD detection patterns in both repos:
grep -rn "simd_caps\|is_x86_feature_detected\|detect_features\|AVX512" \
  /home/user/ndarray/src/ crates/
# Find all cross-repo dep wiring points:
grep -rn "ndarray.*feature\|ndarray-hpc\|ndarray.*dep" crates/*/Cargo.toml
```

**Why it's a 0-friction boundary:** the detection singleton is a
pure-read operation with no side effects and no API surface change. Two
repos that share a singleton share not just the result but also the test
coverage — any test that exercises `ndarray`'s `simd_caps()` path implicitly
covers `lance-graph`'s dispatch path. The boundary between two detection
systems collapses to one shared read.

**Promotion track:** N=1 (SIMD detection). Needs N=3 (two more cross-repo
singleton opportunities) before EP7 approaches iron-rule territory. The
axis would be "cross-repo singleton consistency" — a new axis not covered by
the current four (which are all within a single codebase's algebra). This
would be a genuinely new fifth axis if promoted.

---

### EP8 — Doctrine-Promotion as Concentration-of-Mass

**Name:** Doctrine-promotion as concentration-of-mass

**Shape:** A body of N individual sprint decisions {d_1, ..., d_N}, each
made independently with some cognitive overhead, can be collapsed into one
iron rule I such that all future instances of the same decision are resolved
by reading I instead of re-deriving. The "concentration of mass" is the
reduction in per-decision cognitive overhead from O(reasoning-from-scratch)
to O(read-the-rule).

This EP is the meta-pattern behind all iron-rule promotions. The 0-friction
property is: once an iron rule is written, every future impl worker,
planner, and reviewer resolves the covered decision in O(1) — the rule text
IS the decision. The boundary between "figure out the right answer" and "the
right answer is already written" collapses to one read.

**Workspace instance:**
`I-LEGACY-API-FEATURE-GATED` promotion from sprint-11 E-META-10 through
sprint-12 CSI-18 to the iron-rules-doctrine.md §2.4 entry. Before the
promotion: every PR that touched a feature-gated bit layout required a
human reviewer to independently reason about v1-vs-v2 accessor semantics.
After the promotion: the rule text in CLAUDE.md covers the decision; the
brutally-honest-tester (PP-13) enforces it mechanically. Four instances in
PR #383 triggered the promotion; the concentration-of-mass is every future
PR that touches a feature-gated layout.

See also iron-rules-doctrine.md §5.4 "The line between style and iron" —
the promotion threshold (N >= 3 instances + substrate-level consequence) is
the formal definition of when individual decisions concentrate into a rule.

**Grep target:**
```bash
# Find all recurring-decision patterns in sprint-log meta-reviews:
grep -rn "same class\|same pattern\|recur\|repeated\|again in" \
  .claude/board/sprint-log-*/
# Find all EPIPHANIES entries that are approaching iron-rule threshold:
grep -rn "E-META\|E-SUBSTRATE\|E-ORIG" .claude/board/EPIPHANIES.md
```

**Why it's a 0-friction boundary:** the boundary between "need to reason
every time" and "read the rule once" is a promotion ceremony. The ceremony
is defined (iron-rules-doctrine.md §3); the cost is bounded (~10 LOC PR +
meta-Opus review); the payback is unbounded (every future instance). The
highest ROI EP is always EP8 when the promotion threshold has been reached.

**Promotion track:** EP8 IS the promotion track. It does not promote itself
— it IS the mechanism by which all other EPs promote. EP8 is retired only
if the iron-rules-doctrine.md §3 promotion ceremony is replaced by a
fundamentally different governance mechanism.

---

## §3  When NOT to Spawn the Convergence-Architect

The convergence-architect is a synthesis agent. Synthesis is expensive (Opus
model, multi-source reads). Do not spawn when:

| Situation | Route instead |
|---|---|
| "Is there a bug in this impl?" | PP-13 brutally-honest-tester |
| "Does this baton handoff have a type mismatch?" | PP-15 baton-handoff-auditor |
| "Has the plan drifted from the spec?" | PP-16 preflight-drift-auditor |
| "Where is type X defined?" | Explore subagent (Sonnet grindwork — single-source search) |
| "What does function F do?" | Read + Grep (no synthesis needed) |
| "Run the tests and tell me if they pass" | Bash + worker-template-v2.md §6 |
| "Is this correct?" without cross-subsystem scope | Any single-purpose agent or main-thread read |
| No alignment hypothesis to test — just curious | Save tokens; wait for a concrete pre-plan trigger |

The anti-trigger list is important because synthesis without a hypothesis
wastes Opus budget. The convergence-architect should be spawned WITH a
candidate alignment in mind (or with an explicit request to scan for one)
— not as a "general review" agent.

---

## §4  Verdict Semantics

### OPPORTUNITY-NOW (>= 80% confidence ROI in current sprint)

The alignment exists. The work is bounded. The sprint ROI is clear enough
that a planner can add a D-CSV-N entry without a probe phase.

**What raises confidence above 80%:**
- The two subsystems already share a concrete type (same struct, same enum)
  that makes the alignment mechanically verifiable (not just algebraically
  plausible).
- A grep confirms the pattern at 3+ call sites, not just in the description.
- An existing iron rule directly predicts the alignment (e.g., I-VSA-IDENTITIES
  predicts that any two sites that already use Vsa16kF32 can be wired to
  share identity fingerprints — OPPORTUNITY-NOW for any such pair found).

**Action:** planner adds as D-CSV-N with the convergence-architect cited as
originator. If the orchestrator overrides (decides not to include), the
override reason is recorded in the sprint-log meta-review.

### WORTH-EXPLORING-SOON (40-80% — queue as OQ-CSV-N)

The alignment is plausible but needs one or more of:
- A targeted grep to confirm the pattern at the proposed site.
- A benchmark to confirm the performance claim (e.g., "shared singleton
  reduces detection latency by X%").
- A paper search to confirm the algebraic backing (e.g., "does this
  bundling operator satisfy associativity in the relevant sense?").
- A truth-architect review for proposals that touch HHTL / claims-without-
  probes / new layer proposals.

**Action:** planner adds OQ-CSV-N with the probe question specified. The
convergence-architect is NOT called again for the probe — the probe is a
Sonnet grindwork task (single-source search or benchmark run). The
convergence-architect re-evaluates only when the probe returns its result.

### DROP-WITH-RATIONALE (< 40%)

The apparent alignment is shallow. Common reasons:

- **Type mismatch at the boundary:** the two systems use the same abstract
  concept but different concrete representations (e.g., both use "fingerprint"
  but one uses `[u64; 256]` Hamming and the other uses `[f32; 16384]`
  cosine — not the same algebra).
- **Hidden cost:** the migration from A to B appears zero-cost but requires
  touching N call sites with incompatible semantics (e.g., the "role-keyed
  identity" is present but the role catalogue is different between the two
  systems — a SUBJECT key in grammar/role_keys.rs is not the same slice as
  a SUBJECT key in persona/role_keys.rs).
- **Theoretical alignment, not practical alignment:** the two operations
  are the same algebra in theory but diverge at the representation boundary
  (e.g., `MergeMode::Xor` and `vsa_bundle` are both "combine" operations
  but per I-SUBSTRATE-MARKOV, Xor is NOT a Markov-respecting transition
  kernel — the algebras diverge at exactly the substrate-level consequence
  that would make the boundary matter).

**Action:** append to the DROP log in this section (below) with a one-line
WHY. Future agents consulting this doc can skip the re-derivation.

**DROP log (append new entries below; APPEND-ONLY):**

| Date | EP class | Apparent alignment | Why it resists collapse |
|---|---|---|---|
| (none yet — catalogue is new as of 2026-05-16) | — | — | — |

---

## §5  Workflow Integration

The convergence-architect occupies the pre-plan slot in the CCA2A loop.
The four-agent quality lifecycle with arrows:

```
PRE-PLAN DIVERGENCE
  convergence-architect  <--- hunts 0-friction alignment opportunities
       |
       | OPPORTUNITY-NOW items --> planner bakes into D-CSV-N
       | WORTH-EXPLORING-SOON  --> planner queues as OQ-CSV-N
       |
       v
PRE-SPAWN CONVERGENCE
  preflight-drift-auditor (PP-16)  <--- catches plan/spec drift
       |
       v
DURING-IMPL BOUNDARY CHECK
  baton-handoff-auditor (PP-15)  <--- catches boundary mismatches
       |
       v
POST-IMPL CODE REVIEW
  brutally-honest-tester (PP-13)  <--- catches codex-class bugs
       |
       v
COMMIT + META-REVIEW
  W-Meta-Opus  <--- cross-cutting honest review of the wave
```

The four agents are complementary, not competitive. A pattern that the
convergence-architect names as OPPORTUNITY-NOW becomes an alignment that the
baton-handoff-auditor monitors for correct implementation and the
brutally-honest-tester verifies at the code level. When an alignment is
named, the three critic agents can enforce it; until it is named, they
cannot — enforcement presupposes articulation.

---

## §6  Maintenance Protocol

### Append-only EP catalogue

The §2 EP catalogue is APPEND-ONLY within each EP entry. New EPs are added
after EP8 with incrementing IDs (EP9, EP10, ...). Corrections to an existing
EP entry are added as a dated annotation below the original text:

```markdown
> **CORRECTION 2026-XX-XX:** [correction text] — [agent or reviewer]
```

No EP entry is deleted. Promoted EPs are annotated:

```markdown
> **PROMOTED 2026-XX-XX → iron rule I-<NAME> (sprint-N PR #NNN)**
```

Reversed EPs (convergence observation turned out to be a defect pattern)
are annotated:

```markdown
> **REVERSED 2026-XX-XX → see codex-p1-anti-patterns.md AP-N**
> The apparent alignment was [description]; the actual defect was [description].
> Future agents: the boundary resists collapse because [reason].
```

### Promotion to doctrine via §6 ceremony

When an EP reaches the promotion threshold (N >= 3 instances + substrate-
level consequence + backing citation), follow the ceremony in
`convergence-architect.md §6`:

1. Flag in OPPORTUNITY-NOW report as `PROMOTION-READY`.
2. Draft iron-rules-doctrine.md §3 checklist (all boxes ticked).
3. Submit to meta-Opus review in next sprint-log.
4. Iron-rule PR adds §2.N to iron-rules-doctrine.md + CLAUDE.md.
5. Annotate the EP as PROMOTED (see above).

### Reversal track

If a WORTH-EXPLORING probe returns evidence that the alignment is actually
a defect pattern (the boundary resists collapse because it SHOULD resist
collapse — two systems that appear to share algebra but must NOT be wired
together), the EP is reversed and the finding goes to
codex-p1-anti-patterns.md as a new AP entry. The brutally-honest-tester
(PP-13) is notified via the AP entry so it can grep for the pattern in
future diffs.

This reversal track is rare but important. The convergence-architect's job
is to name alignments; the critic agents' job is to test them. When a test
fails, the failure data flows BACK to this catalogue so future scans do not
re-derive the same dead end.

---

## §7  Cross-References

- **`convergence-architect.md`** (PP-14, `.claude/agents/`) — the agent
  card that activates this catalogue. §2 quick-reference table; §6 promotion
  ceremony; §7 sibling agent links.
- **`CLAUDE.md §The-Click`** — the core invariant. Every EP that touches
  VSA substrate cites §The-Click.
- **`CLAUDE.md §Substrate-level iron rules`** — I-SUBSTRATE-MARKOV /
  I-NOISE-FLOOR-JIRAK / I-VSA-IDENTITIES / I-LEGACY-API-FEATURE-GATED.
  EPs that are corollaries of existing iron rules are noted in Promotion Track.
- **`.claude/knowledge/iron-rules-doctrine.md`** (PP-2) — the four-axis
  framing. Every EP5/EP8 observation cites §1 axis table.
- **`.claude/knowledge/codex-p1-anti-patterns.md`** (PP-13 companion) —
  the defect catalogue. EP6 produces positive framings of AP entries; EP
  reversals produce new AP entries.
- **`.claude/agents/brutally-honest-tester.md`** (PP-13) — post-impl
  pre-commit gate. Enforces alignments that convergence-architect names.
- **`.claude/agents/baton-handoff-auditor.md`** (PP-15) — during-impl
  boundary mismatch detector. Monitors EP1 handover patterns specifically.
- **`.claude/agents/preflight-drift-auditor.md`** (PP-16) — pre-spawn
  plan/spec drift detector. Monitors whether OPPORTUNITY-NOW items baked
  into a plan survive the preflight-to-spawn transition.
- **`.claude/board/EPIPHANIES.md`** — FINDING accumulation register.
  The gap between the last FINDING and the current iron-rule frontier is
  exactly where EP-class opportunities live.
- **`.claude/board/sprint-log-13/preflight-meta-review-opus.md`** —
  W-Meta-Opus §2 per-planner table: EP4 (PP-4 Think-as-carrier), EP3
  (PP-5 CAM-PQ one-liner migration), EP6 reversal signal (CSI-19
  D-CSV-numbering drift).
- **`.claude/board/sprint-log-13/oq-catalog.md`** — the OQ-CSV-N catalogue.
  WORTH-EXPLORING-SOON outputs from convergence-architect propose new entries.

---

## §8  One Sentence That Should Survive Any Refactor

**When two carriers share an algebra you did not notice, the boundary
between them was always meant to vanish.**

---

*Authored W-Sprint-13-PP-14 (Opus agent, main-thread), 2026-05-16.
Sources: user request 2026-05-16; CLAUDE.md §The-Click and iron rules;
iron-rules-doctrine.md (PP-2); preflight-meta-review-opus.md (W-Meta-Opus
§2 EP-class observations); brutally-honest-tester.md (PP-13, sibling
structure mirror); codex-p1-anti-patterns.md (PP-13 companion, catalogue
structure mirror); BOOT.md Knowledge Activation Protocol.*
