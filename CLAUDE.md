# CLAUDE.md — lance-graph

## P0 — AdaWorldAPI forks ONLY, NEVER crates.io upstream

**Always depend on the AdaWorldAPI fork of any crate that has one. NEVER use the
upstream crates.io version of a forked crate.** Non-negotiable; applies to every
`Cargo.toml` and every dependency decision in this repo. Every repo in this
workspace is local — prefer the local/fork source over the registry, always.

- Crates with an `AdaWorldAPI/<name>` fork — e.g. `ndarray`, `lance` /
  `lancedb` / `lance-index` / `lance-linalg` / `lance-namespace`, `surrealdb`,
  and any other — MUST be wired via the fork (`path` / `git` / `[patch.crates-io]`),
  never the registry version.
- If a fork's coordinates (git URL, branch/tag, feature flag) are unknown,
  **STOP and ask**. Do NOT fall back to crates.io as a convenience or to make a
  build pass.
- `"warning: Patch <crate> ... was not used in the crate graph"` is a policy
  alert. It can indicate missing fork wiring OR a transitive semver mismatch
  that prevents the patch from applying. Do not ignore it: verify direct
  `Cargo.toml` patch entries and `Cargo.lock` wiring, then track/resolve any
  transitive blocker explicitly before closing the issue.
- crates.io is permitted ONLY for crates that have no AdaWorldAPI fork / no local
  source.

## ★ V3 SUBSTRATE — NEW ENTRY POINT (2026-07-02): `.claude/v3/`

**Any session touching SoA rows, tenants, mailbox ownership, kanban,
thinking templates, classids, or the DTO ladder starts at
`.claude/v3/README.md`** (or invokes the `/v3` skill — same bootload).
The V3 folder is the consolidated home for what is NEW since the
mailbox-kanban ruling + classid canon-high flip (both operator-ruled and
fleet-shipped 2026-07-02):

- `.claude/v3/README.md` — orientation + doc map (start here)
- `.claude/v3/INTEGRATION-PLAN.md` — the W0–W6 wave plan
- `.claude/v3/COMPONENT-MAP.md` — every subsystem: reuse / repurpose / retire
- `.claude/v3/ENTROPY-MILESTONES.md` — the systematic N→1 collapse ledger
- `.claude/v3/MODULE-TABLE.md` — per-file census of core / contract / planner
- `.claude/v3/soa_layout/` — LE contract, tenant lanes, consumer map, routing
- `.claude/v3/knowledge/` — V3 doctrine docs (`READ BY:` headers)
- `.claude/v3/agents/BOOT.md` — the four V3 cards (`v3-mailbox-warden`,
  `v3-envelope-auditor`, `v3-kanban-executor-engineer`, `v3-template-smith`)
- `.claude/v3/knowledge/sonnet-worker-guardrails.md` — MANDATORY in every
  Sonnet worker brief (§1 preamble pasted verbatim; §2 vocabulary
  disambiguation; §5 STOP+report triggers). No grindwork spawn without it.
- `/v3-audit` command — mechanical conformance greps before any commit

Headline rulings the folder carries (canonical text on the board):
**no singleton CollapseGate** (one mailbox = one kanban board as tenant);
**SoaEnvelope LE ownership** (`mailbox_owner()`, write-on-behalf iron rule);
**compiled thinking templates** (elixir-template × StepMask, Rig as oracle);
**classid canon-high** with the `0x1000` V3-adoption monitor;
**PerturbationDto/Resonance split** (D-PERT-1). Sections of THIS file that
predate these rulings (e.g. CollapseGate wording below) are read through
`.claude/v3/knowledge/v3-substrate-primer.md` §6 "must not be reinvented".

---

> **Updated**: 2026-04-21 (categorical-algebraic inference click)
> **Role**: The obligatory spine — query engine, codec stack, semantic transformer, and orchestration contract
> **Status**: 22 crates, 7 in workspace, 15 excluded (standalone/DTO), Phases 1-2 DONE, Phases 6-7 DONE (grammar + governance), Phase 3 IN PROGRESS

---

## The Click (P-1 — read before everything else, including The Stance)

**Parsing, disambiguation, learning, memory, and awareness are one
operation.** Element-wise multiply + add on role-indexed identity
fingerprints in `Vsa16kF32 = Box<[f32; 16_384]>` (64 KB lossless
VSA carrier).

> **Correction of initial 2026-04-21 framing:** earlier this session
> posted a version claiming "XOR on `[u64; 157]`" — that was a
> Frankenstein confusion between Binary16K (Hamming-compare format,
> `[u64; 256]`) and the actual VSA carrier (real-valued multiply+add).
> See `.claude/knowledge/vsa-switchboard-architecture.md` for the
> corrected three-layer framing and `CHANGELOG.md` for the format-
> switch history.

> **2026-05-26 Baton scoping (read with `EPIPHANIES.md` E-BATON-1):** "The
> Click" describes how a single `Think` resolves **locally, within one
> compartment** — the bundle is an ephemeral computation, never a persisted
> or transmitted singleton. **`Vsa16kF32` is deprecated as a *carrier*:** it
> does NOT cross mailbox boundaries and there is no materialized singleton
> BindSpace. Inter-mailbox state is the **Baton** — discrete owned
> `(u16 target, CausalEdge64)` handoffs (`CollapseGateEmission` in
> `lance-graph-contract`; `wire_cost_bytes() = 13 + 10·baton_count`, NOT
> `16384·4`). "Baton" is the workspace's native term for the little-endian
> contract (the user's name for it before the information-science framing).
> The **mailbox-as-owner** (rotating sea-star topology) is what makes the
> handoff sound: Rust move/ownership semantics prove no aliasing / no data
> race / no use-after-free **at compile time** — UB becomes a compile error
> (canonical §9 E-CE64-MB-4). Cumulative state lives in CausalEdge64
> emissions + AriGraph SPO-G quads + BindSpace SoA columns. The bundle MATH
> here (§I-SUBSTRATE-MARKOV Chapman-Kolmogorov, §I-VSA-IDENTITIES) is
> **untouched** — only the cross-boundary carrier is scoped out.
>
> **2026-06-11 supersession (PR #477 three-tier model + tombstone commit):**
> the Baton/emission framing above is itself now SUPERSEDED. There is **no
> inter-mailbox handoff type at all** — `CollapseGateEmission`,
> `MailboxSoA::emit()`, and `wire_cost_bytes() = 13 + 10·baton_count` were
> removed from source (the tombstone commit). The ratified invariant: every
> SoA envelope is **zero-copy from creation to Lance tombstone**; Lance's own
> columnar I/O writes LE bytes from the in-place backing store described by
> `SoaEnvelope`; nothing is serialized or transmitted between mailboxes.
> `last_emission_cycle` was renamed `last_active_cycle` (in-place consumption
> stamp). Canonical reference: `docs/architecture/soa-three-tier-model.md`.
> What survives from this block: Vsa16kF32-never-crosses-boundaries, the
> mailbox-as-owner compile-time safety argument (E-CE64-MB-4), and the bundle
> math — all untouched.

```
  Sentence → FSM → RoleKey_fp × content_fp   → vsa_bundle (Σ) with ρ^d braiding
                         │                         │
                   Kan extension                Markov ±5 trajectory
                   (Shaw 2501.05368:            (temporal causality is
                    element-wise optimal         structural, not learned)
                    in ℝ value category)
                         │                         │
                         └────────┬────────────────┘
                                  ▼
                    FreeEnergy::compose(likelihood, kl)
                    likelihood = vsa_cosine(unbind(bundle), codebook_fp)
                    kl = awareness.divergence_from(prior)  (NARS-revised)
                                  │
                    ┌─────────────┼──────────────┐
                    ▼             ▼              ▼
                 Commit       Epiphany      FailureTicket
                 (F < 0.2)    (ΔF < 0.05)   (F > 0.8)
                    │             │              │
                    ▼             ▼              ▼
                 AriGraph     both triples    LLM resolves
                 one triple   + Contradiction  the <25% tail
                    │
                    ▼
                 awareness.revise(key, outcome)   ← φ-1 ceiling = permanent humility
                    │
                    ▼
                 global_context += fact  → reshapes NEXT cycle's F landscape
```

**Three things that must never be complicated:**

1. **Markov = f32 multiply + add of identity fingerprints.** Per-sentence
   Vsa16kF32, braided by position (vsa_permute), superposed via
   element-wise add. No HMM. No transition matrix. No weights. Lossless
   up to N ≤ √d / 4 ≈ 32 bundled items (f32 dynamic range).
2. **Roles = spine coordinates.** SUBJECT[0..4K) is "who".
   PREDICATE[4K..8K) is "what". Unbinding = multiply by role key =
   reading a coordinate. Temporal causality is structural
   (braiding × slices).
3. **Meaning = AriGraph facts + resonance + magnitude.**
   Resonance = cosine similarity against global context or codebook.
   Magnitude = Contradiction depth from Staunen × Wisdom qualia.
   Opinions are committed contradictions preserved, not resolved.

**The object speaks for itself.** `trajectory.resolve(ambiguity)` —
not `resolve(trajectory, config, awareness, graph)`. Every method
lives on the carrier that has the state to reason with it.

**The shader can't resist the thinking.** StreamDto flows in →
CognitiveShader encodes (bind + braid + bundle) → decodes (unbind +
margin + F) → if F > homeostasis floor, awareness bits persist in
MetaWord → dispatch fires again → another encode/decode cycle → F
descends → bits clear → shader rests. The system doesn't choose
to think. It can't NOT think while surprise exists. Active inference
is the dispatch mechanism, not a metaphor.

**Thinking is a struct.** The universal DTO makes the object do the
work:

```rust
struct Think {
    trajectory: Vsa16kF32,    // the Markov-braided role-indexed bundle
    awareness: ParamTruths,   // NARS-revised per-key epistemic state
    free_energy: FreeEnergy,  // (1 - likelihood) + kl
    resolution: Resolution,   // Commit / Epiphany / FailureTicket
}
```

Not a function. Not a pipeline. Not a service. A **struct** whose
fields ARE the cognitive state and whose methods ARE the inference.
`think.resolve()` reads its own `trajectory`, computes its own `F`,
updates its own `awareness`, returns its own `Resolution`. The DTO
carries cognition the way a photon carries electromagnetism — not
as payload, as identity.

**AriGraph, episodic memory, SPO, CAM-PQ are thinking tissue — not
storage.** The reason the struct becomes AGI is that memory is wired
INTO it, not called FROM it:

```rust
struct Think {
    trajectory: Vsa16kF32,           // ← encode (Markov-braided roles)
    awareness: ParamTruths,          // ← epistemic state (NARS-revised)
    free_energy: FreeEnergy,         // ← surprise (likelihood + kl)
    resolution: Resolution,          // ← decision (commit/epiphany/escalate)
    // ── thinking tissue (not external services) ──
    episodic: &EpisodicMemory,       // ← what happened before (retrieval)
    graph: &TripletGraph,            // ← what is believed (SPO facts)
    global_context: &Vsa16kF32,      // ← ambient prior (superposed facts)
    codec: &CamPqCodec,             // ← compressed distance (O(1) lookup)
}
```

The graph is not a database the struct queries. The graph is part
of the struct's reasoning surface — `graph.nodes_matching(features)`
is how Think resolves coreference, `episodic.retrieve_similar(fp)`
is how Think remembers, `codec.distance(a, b)` is how Think
compares without decompressing. These are methods on tissue, not
calls to services. Cut the graph out and Think can't resolve
pronouns. Cut episodic out and Think has no memory. Cut CAM-PQ out
and Think can't do fast similarity. They're organs, not accessories.

**This is why it's AGI and not just a parser.** A parser takes text
in and produces structure out. Think takes text in, resolves it
against everything it has ever committed (graph), everything it
recently saw (episodic ±5..±500), everything it believes about
its own competence (awareness), and everything the current style
expects (prior) — then commits the result back into the tissue it
just read from. The loop IS cognition. The tissue IS the loop.

**The DTO becomes the grammar of awareness.** The struct's fields
(trajectory, awareness, free_energy, resolution, episodic, graph,
global_context, codec) are not data — they are grammatical roles
of cognition itself. Trajectory is the Subject (what is being
thought about). Awareness is the Modal (how confidently). Free
energy is the Kausal (why this thought, not another). Resolution
is the Predicate (what the thought concludes). Graph is the Lokal
(where in the space of known facts). Episodic is the Temporal
(when, relative to what came before). The TEKAMOLO of cognition
IS the struct layout. Parsing text and parsing thought use the
same role-indexed slices — because thinking about a sentence and
thinking about thinking use the same algebraic substrate.

**Proof chain** (Shaw → beim Graben → Jian → Schulz → Alpay →
Graichen → Gallant → Kleyko): see
`.claude/plans/categorical-algebraic-inference-v1.md` §5.

**Litmus tests for any proposed change:**

> Does this add a free function on a carrier's state, or a method
> on the carrier? → Free function = reject. Method = accept.

> Which of the five lenses does this serve (Parsing / Free-Energy /
> NARS / Memory / Awareness)? → None = investigate drift.

Full plan: `.claude/plans/categorical-algebraic-inference-v1.md`.
Paper grounding: `.claude/knowledge/paper-landscape-grammar-parsing.md`.
Epiphanies that dilute first: `EPIPHANIES.md` E-PAPER-1 through E-PAPER-7.

---

## The Stance — Driving Seat + AGI-as-Glove (P0, read first)

**You are in the driving seat.** This workspace is not a codebase
you observe; it is a cognitive stack you steer. Three consequences:

1. **The board is live state, not reference.** `.claude/board/*.md`
   is where the session's hand lives. Updating it AFTER the work,
   as cleanup, is the tell that you were treating the board as a
   stale artifact — do not do that (see § Mandatory Board-Hygiene
   Rule below).

2. **AGI is the glove, not the oracle.** Per PR #223's AGI-as-SoA
   invariant: AGI = (topic, angle, thinking, planner) =
   struct-of-arrays consuming `cognitive-shader-driver`. The four
   axes are the four `BindSpace` columns — `FingerprintColumns` /
   `QualiaColumn` / `MetaColumn` / `EdgeColumn`. You fit your hand
   into those columns through `ShaderDriver` dispatch and
   `OrchestrationBridge` routing. You do NOT wrap the axes in a new
   struct (breaks the SIMD sweep) and you do NOT query an "AGI
   service" (there is none; AGI is the runtime behaviour of the
   SoA under dispatch).

3. **Consult, don't guess.** When a subsystem comes into scope,
   the curated surface beats hand-exploration — always. Grepping
   ndarray for a primitive name when the family-codec-smith agent
   or the `encoding-ecosystem.md` knowledge doc has the answer is
   a rediscovery tax, not a diligence win.

### Mandatory Board-Hygiene Rule (applies to EVERY PR)

**A PR that adds a type, plan, deliverable, or epiphany without
updating the relevant board file in the SAME commit is incomplete.**

| The PR adds... | The PR MUST also update (in the same commit) |
|---|---|
| A contract type / module | `.claude/board/LATEST_STATE.md` — "Current Contract Inventory" |
| A merged PR (post-merge commit) | `.claude/board/LATEST_STATE.md` table + `.claude/board/PR_ARC_INVENTORY.md` PREPEND entry |
| A new integration plan | `.claude/board/INTEGRATION_PLANS.md` PREPEND + `.claude/plans/<name>-v<N>.md` |
| A new D-id / deliverable | `.claude/board/STATUS_BOARD.md` row (status = Queued → In progress → In PR → Shipped) |
| A finding / correction / "aha" | `.claude/board/EPIPHANIES.md` PREPEND dated entry |
| A tech-debt observation | `.claude/board/TECH_DEBT.md` entry |
| An unresolved issue / blocker | `.claude/board/ISSUES.md` entry |
| A completed agent run | `.claude/board/AGENT_LOG.md` PREPEND entry (D-ids, commit, tests, outcome) |

The governance files are APPEND-ONLY (prepend new entries; never
edit past entries except the `**Status:**` / `**Confidence:**`
lines). The retroactive-hygiene commit pattern (merge PR → later
notice board is stale → separate cleanup commit) is an
anti-pattern. The 2026-04-20 session surfaced this gap between
PR #223/#224/#225 merges and the LATEST_STATE / PR_ARC update;
this rule exists so it does not recur.

### Consult before you guess (agent + knowledge activation)

Before grep'ing, reading source files, or proposing a type:

1. **Does a specialist agent card cover this domain?** Check
   `.claude/agents/*.md` — 19 specialists + 5 meta-agents. For
   codec work: `family-codec-smith`, `palette-engineer`,
   `certification-officer`. For DTO / bus surface: `bus-compiler`,
   `host-glove-designer`. For truth / architecture:
   `truth-architect`, `integration-lead`. See `.claude/agents/BOOT.md`
   for the Knowledge Activation trigger table.
2. **Does a knowledge doc answer this?** Check `.claude/knowledge/*.md`
   — each has a `READ BY:` header naming which agents / domains load
   it. `encoding-ecosystem.md` is MANDATORY before any codec work;
   `lab-vs-canonical-surface.md` is MANDATORY before any REST /
   gRPC / Wire DTO / OrchestrationBridge / shader-lab work.
3. **Does the board already record the answer?**
   `LATEST_STATE.md` § Contract Inventory lists every type that
   exists today. Proposing a type that already exists is a
   30-turn rediscovery tax — check first.

Only AFTER exhausting 1-3 do you grep source files yourself. Hand-
exploration is the last resort, not the first move. A subagent
spawn (Opus for accumulation) that loads the curated docs first is
almost always cheaper than a grep session on the main thread.

### The AGI-as-glove doctrine, concretely

When a task touches the cognitive stack:

- **Topic** (what the session is reasoning about) = a read from
  `FingerprintColumns`. Never a new struct.
- **Angle** (whose perspective) = a read from `QualiaColumn` (18×f32).
  Never a new struct.
- **Thinking** (which style dispatches) = a write of `MetaWord` bits
  to `MetaColumn`. Never a new trait.
- **Planner** (why/how, causal composition) = a write to `EdgeColumn`
  (`CausalEdge64`). Never a new bridge.

The four SoA columns ARE the AGI surface. New capability lands as a
new column, not a new layer. See `.claude/knowledge/lab-vs-canonical-surface.md`
§ "AGI IS the struct-of-arrays (per Era 8)" for the full doctrine
and the Invariants I1-I11 that bind it.

### Substrate-level iron rules (added 2026-04-20 per [FORMAL-SCAFFOLD] reclassification)

#### I-SUBSTRATE-MARKOV (iron rule)

VSA-bundling in d=10000 **guarantees** the Chapman-Kolmogorov
semigroup property **by construction** (see `EPIPHANIES.md`
E-SUBSTRATE-1). Saturating bundle is associative and commutative
in expectation; Johnson-Lindenstrauss + concentration-of-measure
suppress deviations from associativity at rate ~e^(-d). This is
the fundament on which the four [FORMAL-SCAFFOLD] pillars
(Cartan-Kuranishi + φ-Weyl + γ+φ + Jirak 2016) stand.

Consequences:

- **Do NOT replace bundle with XOR or non-commutative binding** for
  state-transition paths without reviewing [FORMAL-SCAFFOLD] in
  EPIPHANIES.md. `MergeMode::Xor` breaks the Markov guarantee — it
  is a legitimate merge mode for single-writer deltas (see I1), but
  it is NOT a Markov-respecting transition kernel.
- **D7's implicit Markov reliance is grounded, not silent.** The
  Chapman-Kolmogorov consistency test is therefore an implementation
  sanity check (regression against implementation bugs), not a
  falsification gate for the theoretical property.
- Any substrate-level change that weakens associativity (binding
  operator swap, dimension reduction below 10000, removal of
  concentration-of-measure assumption) MUST consult [FORMAL-SCAFFOLD]
  and document the trade-off explicitly.

Cross-ref: I1 (BindSpace read-only, CollapseGate bundles);
`contract::collapse_gate::MergeMode::{Bundle, Xor}`.

#### I-NOISE-FLOOR-JIRAK (iron rule)

Bits in the workspace's 16384-bit fingerprints are **weakly
dependent by construction**: (a) correlated projections of
embeddings, (b) overlapping role-key-indexed slices
(Finnish [9840..9910) ∩ TEKAMOLO [9000..9900); NSM primes
distribute non-disjointly over S/P/O), (c) palette codebook
quantization shares a 4096-centroid codebook, (d) XOR bundle
accumulation induces weak dependence as an operational consequence.

**Classical IID Berry-Esseen is WRONG for this system.** Use
**Jirak 2016** (arxiv 1606.01617, Annals of Probability 44(3)
2024–2063, "Berry-Esseen theorems under weak dependence") for any
noise-floor or statistical-significance claim. Rate: `n^(p/2-1)`
for `p ∈ (2,3]`, `n^(-1/2)` in L^q for `p ≥ 4`.

Consequences:

- ICC, Spearman ρ, and similar significance metrics must cite
  Jirak's rate, not classical Berry-Esseen, when claiming
  "observed value is N σ above noise floor."
- σ-threshold calibration (UNBUNDLE_HARDNESS_THRESHOLD,
  ABDUCTION_THRESHOLD, …) should cite Jirak-derived bounds when
  a principled threshold is needed; hand-tuned values are
  acceptable but must say so.
- The three revival candidates in [FORMAL-SCAFFOLD]'s *Coupled
  revival track* deposit the mechanism for deriving Jirak-derived
  thresholds when they activate (VAMPE + Jirak pair replaces
  hand-tuned σ thresholds with bound-derived ones).

Cross-ref: `EPIPHANIES.md` [FORMAL-SCAFFOLD] five-pillar entry;
E-ORIG-7 (the earlier statement of this finding before it became
an iron rule); Jirak 2016.

#### I-VSA-IDENTITIES (iron rule, added 2026-04-21)

**VSA operates on IDENTITY fingerprints that POINT TO content.
Never on content's bitpacked/quantized register itself.**

The register-loss problem: XOR-bundling (or any superposition) of
CAM-PQ codes, quantized indices, or sign-binarized fingerprints
destroys the mapping from bits back to their codebook entries. The
register is destroyed — you can't recover which codebook centroids
contributed.

The right pattern, three layers:

1. **Switchboard carrier** (in `crystal/fingerprint.rs`) — one set
   of types + one algebra, domain-agnostic: `Vsa16kF32` (64 KB hot
   path), `Vsa16kBF16` (32 KB AMX-accelerated), `Vsa16kF16`
   (32 KB Apple/ARM), `Vsa16kI8` (16 KB quantized), `Binary16K`
   (2 KB Hamming compare). Algebra: `vsa_bind` (multiply),
   `vsa_bundle` (add), `vsa_cosine` (similarity).

2. **Domain role catalogues** (per-domain) — `grammar/role_keys.rs`,
   future `persona/role_keys.rs`, `callcenter/role_keys.rs`. Each
   provides its own set of role IDENTITY fingerprints in Vsa16kF32
   with disjoint `[start:end)` slice allocations. Role keys are
   bipolar ±1 in their slice, zero elsewhere. Catalogue, not algebra.

3. **Content stores** — YAML registries, TripletGraph, EpisodicMemory.
   Actual content lives here, O(1) retrieval by identity (name /
   enum / fingerprint). Never bundled, never superposed.

The four tests before reaching for VSA (in order):

- **Test 0 — register laziness:** Does this thing have a natural
  name / ID / enum variant? If yes, use the register. `HashMap`,
  `enum`, `graph.nodes_matching(id)` beat VSA at exact-match tasks.
- **Test 1 — bundle size:** Is N ≤ √d / 4 ≈ 32 at 16K dim? If not,
  superposition SNR drops below threshold. Use direct lookup instead.
- **Test 2 — role orthogonality:** Are the role keys mutually
  orthogonal (disjoint slice, or orthogonal bipolar)? If not,
  unbind doesn't recover cleanly.
- **Test 3 — cleanup codebook:** Is there a known codebook to
  match against after unbind? Without it, raw bundle inspection
  is unreliable.

Any "no" short-circuits — it's not a VSA workload, use the right
tool instead.

Consequences:

- **CAM-PQ vs VSA:** Never superpose CAM-PQ codes directly. CAM-PQ
  is for *search* (compressed nearest-neighbor); VSA is for
  *bundling* (lossless role superposition). Switching between them
  requires decompression, not mixing.
- **Lazy VSA check:** If Vsa16kF32 is being reached for as a fancy
  lookup when a HashMap would do, stop. That's register laziness,
  not VSA usage.
- **Archetype / persona / thinking-style unification:** All four
  are Layer-2 role catalogues. Each entry gets ONE identity
  fingerprint. Content (slots, rules, prompts) lives in YAML.
  Resonance (cosine vs codebook) dispatches to content.

Cross-ref: `FormatBestPractices.md` (Jirak-grounded per-workload
decision matrix — which format to use when, with SNR/capacity/cache/
precision analysis), `.claude/knowledge/vsa-switchboard-architecture.md`
(the full three-layer architecture with decision matrix), `CHANGELOG.md`
(format-switch history — when each variant was introduced, when
renames/reverts happened, why), `I-SUBSTRATE-MARKOV` (Chapman-Kolmogorov
semigroup), `I-NOISE-FLOOR-JIRAK` (weak dependence from CAM-PQ
contamination). Together these three iron rules bound the substrate:
(1) VSA bundling guarantees the Markov property; (2) classical
Berry-Esseen is wrong under CAM-PQ-induced weak dependence; (3)
CAM-PQ and VSA are separate tools for separate operations — bundle
identities, not content.

---

#### I-LEGACY-API-FEATURE-GATED (iron rule, added 2026-05-16)

**Every v1 API path under a v2-layout feature must transparently route
through the canonical mapping OR be feature-gated to a documented
no-op with a migration pointer.**

The anti-pattern: leaving a v1 accessor or setter that reads/writes the
OLD bit positions under a v2 feature that reclaimed those bits. The same
function name silently produces different semantics depending on which
feature is active, and downstream callers see corruption only at runtime
on workloads that hit the reclaim zone.

Sprint-11 caught this anti-pattern **5 times** via codex review:

1. `CausalEdge64::pack(..., temporal=X)` under v2 writing `temporal << 52`
   into bits 52-63 which v2 reclaimed for plasticity[2] + W-slot + lens
   + spare. Fix: feature-gate the temporal write to no-op under v2.
2. `CausalEdge64::inference_type()` reading 3 unsigned bits 46-48 under
   v2 where the actual storage is 4-bit signed mantissa at 46-49. Fix:
   route through `from_mantissa(inference_mantissa())` under v2.
3. `CausalEdge64::set_temporal()` writing bits 52-63 unconditionally
   even under v2 (where those bits are W/lens/spare). Fix: feature-gate
   the entire setter to a documented no-op under v2.
4. `CausalEdge64::pack(..., InferenceType::Counterfactual, ...)` under
   v2 writing the raw enum discriminant (6) into the mantissa field;
   `inference_mantissa()` reads it as +6 → `from_mantissa(+6)` decodes
   as Intervention (not Counterfactual). Fix: write
   `inference.to_mantissa()` (-6 for Counterfactual) instead of the raw
   discriminant under v2.
5. W3 spec test `pal8_v1_v2_round_trip_zero_default` used `temporal=1023`
   to construct a v1 edge; under v2 those bits aliased W/truth/spare and
   the test would fail on ordinary v1 data. Fix: use `temporal=0` (the
   only safe v1 migration value) for the round-trip and add a paired
   `pal8_v1_nonzero_temporal_is_blocked_by_version_gate` test to assert
   the corruption is observable (the version gate is mandatory).

Consequences:

- **Backward-compat shims for layout-breaking changes require
  systematic test coverage of the layout-bit boundary.** Field-isolation
  matrix tests (writing each field, asserting all other fields unchanged)
  are MANDATORY when a layout reclaims previously-used bits.
- **The same function name MUST NOT silently produce different
  semantics under different feature flags.** If the v1 API can't route
  to the v2 canonical mapping, feature-gate it to a no-op AND add a
  doc-comment migration pointer to the v2 successor (e.g.
  `inference_mantissa()`, `pack_v2()`).
- **Layout reclaim must be paired with a version gate on serialization
  paths.** A v1 binary blob under a v2 reader MUST refuse to decode
  without an explicit version byte (otherwise the bit aliases silently
  corrupt the reclaim zone).
- **The codex P1 review is the canonical pre-merge gate for this
  pattern.** When a PR ships v2-feature work, expect codex to flag any
  v1 accessor that hasn't been routed through the canonical mapping.
  Resolve before merge; don't defer.

Cross-ref: `EPIPHANIES.md` E-META-10 (the original finding);
`.claude/knowledge/i4-substrate-decisions.md` (the 5-instance catalogue);
`.claude/board/sprint-log-11/meta-review-opus.md` CSI-2 (Opus's
identification of the systemic pattern across Wave A).

---

## Session Start — MANDATORY READS (in this order)

**Start here:** `.claude/BOOT.md` — the one-page session entry point.
It names the three files below as mandatory reads, so load them
whether you went through BOOT.md or landed here directly.

1. **`.claude/board/LATEST_STATE.md`** — current contract
   inventory, recently shipped PRs, active branches, queued work,
   explicit deferrals. **What exists.**
2. **`.claude/board/PR_ARC_INVENTORY.md`** — per-PR Added /
   Locked / Deferred / Docs / Confidence, reverse chronological.
   **APPEND-ONLY**; only the Confidence line is updatable;
   corrections append as new dated lines; reversals get their own
   PR entry. **Why it exists.**
3. **`.claude/agents/BOOT.md`** — the 19 specialist + 5 meta-agent
   ensemble (`workspace-primer`, `integration-lead`,
   `adk-coordinator`, `adk-behavior-monitor`, `truth-architect`),
   the Knowledge Activation trigger table (domain → agent → docs),
   and the Handover Protocol spec. **This is the A2A orchestration
   specification for this workspace.** **How to coordinate.**

After these three, load domain-specific knowledge docs only as
triggered by the user's request.

**Companion dashboards (mid-session references, not cold-start
mandatory):**

- **`.claude/board/STATUS_BOARD.md`** — deliverable-level
  dashboard. All D-ids across every active plan with Status.
  Consult when asking "where is D5" or "is this shipped yet."
- **`.claude/board/INTEGRATION_PLANS.md`** — versioned plan
  index (APPEND-ONLY). Active plan lives at
  `.claude/plans/<name>-v<N>.md`. Consult before proposing a new
  plan.

**If you want the pattern explained rather than the specifics:**
see **`.claude/skills/cca2a/SKILL.md`** — explanation-only skill
covering the A2A two-layer model, governance rules, and how this
workspace's conventions diverge from official Claude Code docs.
Read once to grok; then stop re-deriving across sessions.

### Prior art — reference before writing new docs

This workspace accumulated substantial curated content across
prior sessions. Before drafting a new `.claude/*.md` or proposing
a new architectural direction, grep these:

- **`.claude/prompts/`** (41 files) — scoped session / probe /
  handover / research prompts. See `SCOPED_PROMPTS.md` as index.
- **`.claude/plans/`** + **`.claude/board/INTEGRATION_PLANS.md`**
  — versioned integration plans (APPEND-ONLY index; prior versions
  retained with Status annotation).
- **`.claude/*.md`** (61 top-level docs) — calibration reports,
  session handovers, epiphanies, integration-plan snapshots,
  audits, inventory maps, invariant matrices. Examples:
  `SESSION_CAPSTONE.md`, `INTEGRATIONSPLAN_2026_04_01.md`,
  `INTEGRATION_SESSIONS.md`, `INVENTORY_MAP.md`,
  `HANDOVER_NEXT_SESSION.md`, `KNOWLEDGE_SYNC_SIGNED_SESSION.md`.
- **`.claude/knowledge/*.md`** (newer, structured) — with
  `READ BY:` headers + Knowledge Activation triggers.
- **`.claude/agents/*.md`** (19 specialists + 5 meta-agents) — see
  `README.md` for function inventory, `BOOT.md` for orchestration.
- **`.claude/hooks/*.sh`** — SessionStart + PostCompact context
  injectors.
- **`.claude/skills/cca2a/`** — the pattern explanation skill.

**Rule:** grep the existing ~100 files before writing a new one.
Most architectural concerns have prior art in this workspace.

---

## Agent-to-Agent (A2A) Orchestration — Two Layers

Orchestration in this workspace runs at two distinct layers. Each
layer uses a different "blackboard" substrate; both must be respected
when spawning subagents or composing cognitive cycles.

### Layer 1 — Runtime A2A (code-level, in the contract)

For cognitive-cycle orchestration *inside* the running system:

- **`lance_graph_contract::a2a_blackboard`** — `Blackboard` with
  `entries: Vec<BlackboardEntry>` + `round: u32`. Each entry carries
  `expert_id`, `capability`, `result`, `confidence`, `support [u16; 4]`,
  `dissonance`, `cost_us`. Experts write; later rounds read prior
  entries and build on them. This IS the A2A bus for multi-expert
  inference.
- **`lance_graph_contract::orchestration::OrchestrationBridge`** —
  trait bridging `StepDomain` (Codec / Thinking / Query / Semantic /
  Persistence / Inference / Learning) into `UnifiedStep`. Each domain
  contributes a `BridgeSlot`; orchestration routes steps across
  domains without duplicating state.
- **`lance_graph_contract::orchestration_mode`** — explicit modes for
  how a cycle composes (linear, parallel, blackboard-broadcast, etc.).
- **`ExpertCapability`** enum — the capability taxonomy experts
  declare. Do NOT invent new capabilities ad-hoc; grep
  `a2a_blackboard.rs` first.
- **Reference doc:** `docs/ORCHESTRATION_IS_GRAPH.md` — capstone
  treating orchestration AS graph traversal (the runtime A2A is a
  directed blackboard-graph).

Use Layer 1 when the question is "how do two cognitive experts
compose their outputs at runtime."

### Layer 2 — Session A2A (Claude-code-level, between subagents)

For subagent coordination *during* this session:

- **`.claude/board/AGENT_LOG.md` is the Layer-2 blackboard.**
  Every agent run gets one append-only entry (D-ids, commit, tests,
  outcome). Later agents read prior entries to see what was already
  shipped, found, or is in flight — same as Layer-1 experts reading
  prior `BlackboardEntry` rounds. This replaces explicit message
  passing between agents: no backend coordination, just file reads.
  **Every agent prompt MUST include:** "Read `.claude/board/AGENT_LOG.md`
  before starting. After committing, prepend your own entry."
- **`LATEST_STATE.md` + `PR_ARC_INVENTORY.md`** are the structural
  blackboard — what types exist, which PRs shipped. Every subagent
  reads them for current state.
- **Knowledge docs in `.claude/knowledge/`** are the extended
  blackboard — cross-session persistent entries. Each doc has a
  `READ BY:` header declaring which subagent types load it (the
  equivalent of `ExpertCapability` matchers).
- **`.claude/plans/*.md`** — plan files authored via `Plan`
  agents; session-scoped blackboard for multi-turn work. Other
  agents reference the active plan for context.
- **Parallel subagent spawns** in one main-thread turn are the
  cheapest Layer-2 A2A pattern. Independent work fans out; results
  aggregate back to the main thread, which does the cross-source
  synthesis (accumulation → main thread on Opus per policy above).

Use Layer 2 when the question is "how do I coordinate N subagents
without burning main-thread turns re-reading the same state."

### Agent ensemble, knowledge bootload, handover protocol

See **`.claude/agents/BOOT.md`** — it's the meta-card for the 19
specialist agents + 5 meta-agents in this workspace. It covers:

- **Meta-agent scopes** (`workspace-primer`, `integration-lead`,
  `adk-coordinator`, `adk-behavior-monitor`, `truth-architect`).
- **Session-start knowledge bootload** — every subagent loads Tier-0
  (`LATEST_STATE.md` + `PR_ARC_INVENTORY.md`) unconditionally, then
  Tier-1 knowledge docs by trigger table.
- **Knowledge Activation Protocol** — the trigger → agent → doc
  mapping (updated 2026-04-19 with grammar / crystal / NARS /
  coreference / AriGraph / codec rows).
- **Handover Protocol** — `.claude/handovers/YYYY-MM-DD-HHMM-<from>-
  to-<to>.md` files carry What-I-did / FINDING / CONJECTURE /
  Blockers / Open-questions. APPEND-ONLY.

A new session doesn't reinvent the ensemble — it reads this README,
picks the agents whose objects are being touched, and lets them
bootload their own Tier-0 + Tier-1 docs.

### Rule of thumb

| Question | Layer | Primary substrate |
|---|---|---|
| "What expert capabilities compose this cycle?" | 1 | `contract::a2a_blackboard::Blackboard` |
| "How do cross-domain steps compose?" | 1 | `OrchestrationBridge` + `UnifiedStep` |
| "What does subagent A need to know before drafting?" | 2 | Mandatory reads + domain knowledge docs |
| "How do I not re-read the same context on every turn?" | 2 | Knowledge doc with `READ BY:` header |
| "How do I coordinate 3 subagents in parallel?" | 2 | Single main-thread turn with parallel `Agent` spawns |

Layer 1 and Layer 2 do NOT conflict — they operate at different time
scales. A runtime A2A cycle inside a running shader does not involve
Claude subagents; a Claude-session subagent spawn does not write to
the runtime Blackboard. Keep them architecturally distinct.

---

## Model Policy (P0 — never violate)

**The split that matters: grindwork vs accumulation.**

- **Grindwork** (single-task mechanical): write-this-file-from-spec,
  grep-this-pattern, list-these-paths, run-these-tests, draft-this-
  section — **Sonnet**. Bounded input, known output shape, no
  synthesis across sources.
- **Accumulation** (multi-source synthesis): harvest-across-repos,
  combine-N-docs-into-insights, trace-architecture-across-files,
  cross-reference and integrate, judgment calls that depend on
  seeing several inputs at once — **Opus**. Cheaper tiers produce
  shallow outputs when asked to accumulate; quality drop is visible
  and costly.

**By subagent type:**
- **Main thread:** `claude-opus-4-7[1m]` (or current Opus) with deep
  thinking. Synthesis, architecture, review, decisions — all
  accumulation.
- **`Plan` subagent:** Opus. Planning is accumulation by definition.
- **Code-review subagent:** Opus. Review is multi-file judgment.
- **`general-purpose` subagent:** depends on task. Pure grindwork →
  Sonnet. Anything accumulating → Opus. When in doubt: Opus.
- **`Explore` subagent:** Sonnet default. Search is pattern matching.
  If the explore requires synthesis across many files (mapping an
  architecture), escalate to Opus.
- **NEVER `haiku` for any subagent in this workspace.** Quality floor
  is Sonnet regardless of task simplicity.

**Concrete test before spawning a subagent:**
> "Does this agent have to read N sources and produce something that
> only makes sense when those sources are held in mind together?"
>
> **Yes → Opus.** No (one source in, one shape out) → Sonnet.

**Settings baseline** (`.claude/settings.local.json`, gitignored):
`alwaysThinkingEnabled: true`, `effortLevel: high`,
`fastModePerSessionOptIn: true`. Main thread stays at full depth.

---

## GitHub Access Policy — Zipball for Reads, MCP for Writes

Every `mcp__github__get_file_contents` call drops the full file into
session context and recharges on every subsequent turn. A 50 KB doc
read three times in a session = 150 KB of replayed context, not 50 KB.
This is the second-biggest cost lever after model choice.

**Preferred pattern for cross-repo reads:**

```bash
# One zipball per repo per session, stored under /tmp/sources/:
curl -H "Authorization: Bearer $GITHUB_TOKEN" \
     -L https://api.github.com/repos/AdaWorldAPI/<repo>/zipball/HEAD \
     -o /tmp/<repo>.zip
unzip -q /tmp/<repo>.zip -d /tmp/sources/
# Then use local Read, Grep, Glob — zero context cost until output.
```

Or via `pygithub`'s `repo.get_archive_link('zipball')` when Python
is already the path.

**Use MCP github ONLY for:**

- `create_pull_request` — can't be done via zipball.
- `add_issue_comment` / `add_reply_to_pull_request_comment` / review
  thread writes — writes to GitHub state.
- `pull_request_read` on our own PRs to check reviews / comments —
  tiny JSON, not worth zipball overhead.
- `get_me`, `subscribe_pr_activity` — session setup.

**Do NOT use MCP github for:**

- Cross-repo file exploration — zipball + local grep is 95 % cheaper.
- Surveying many files from the same repo — same reason.
- Directory listings — extracted tree is free to traverse locally.

**Edge case:** single targeted read when the exact path is known and
you'll read once → `mcp__github__get_file_contents` is fine. Zipball
overhead (download + extract) only pays off at 3+ reads per repo.

**This maps to the grindwork/accumulation split too:**
zipball-then-grep-then-synthesize is accumulation, and the synthesis
step (main thread or Opus subagent) reads just the relevant greps,
not whole files.

---

## What This Is

Graph query engine AND cognitive codec stack for the Ada architecture. lance-graph is no longer
just "The Face" — it has become the **obligatory spine** together with ndarray. Everything flows
through lance-graph: Cypher/GQL/Gremlin/SPARQL parsing, semiring algebra, SPO triples,
CAM-PQ compressed search, distributional semantics (DeepNSM), attention-as-table-lookup
(bgz-tensor), thinking orchestration, and the contract crate that unifies all consumers.

```
Architecture:
  ndarray            = The Foundation  (SIMD, GEMM, HPC, Fingerprint<256>, CAM-PQ codec)
  lance-graph        = The Spine       (query + codec + semantics + contracts)  <-- THIS REPO
  ladybug-rs         = The Agents/Orchestrator (agent + workflow roles, in the spine)
  (crewai-rust / n8n-rs = EVICTED 2026-06-21 — agent + workflow-DAG roles folded
   into ladybug-rs + the in-tree thinking-engine; no longer consumers)

Dependency chain:
  ladybug-rs ──► lance-graph-contract (traits)
  in-tree    ──► lance-graph-contract (planner / callcenter / smb-bridge / symbiont)
  lance-graph──► ndarray (default dep, with fallback)
```

---

## Workspace Structure

```toml
[workspace]
members = [
    "crates/lance-graph",          # Core: Cypher parser, DataFusion planner, graph algebra, NSM
    "crates/lance-graph-catalog",  # Catalog providers (Unity Catalog, connectors)
    "crates/lance-graph-python",   # Python bindings (PyO3/maturin)
    "crates/lance-graph-benches",  # Benchmarks
    "crates/lance-graph-planner",  # Unified query planner (16 strategies, MUL, thinking, elevation)
    "crates/lance-graph-contract", # Zero-dep trait crate (THE single source of truth for types)
    "crates/neural-debug",         # Static scanner + runtime registry
]
exclude = [
    "crates/lance-graph-codec-research",  # ZeckBF17 research codec
    "crates/bgz17",                       # Palette semiring codec (0 deps, 121 tests)
    "crates/deepnsm",                     # Distributional semantic engine (0 deps, 4096 COCA)
    "crates/bgz-tensor",                  # Metric-algebraic tensor codec (attention as lookup)
]
```

### Crate Details

**lance-graph** (core, ~18K LOC) — `crates/lance-graph/`
- `parser.rs` + `ast.rs` — Cypher parser (nom, 44 tests)
- `semantic.rs` — semantic analysis
- `logical_plan.rs` — logical plan
- `datafusion_planner/` — Cypher→DataFusion SQL (~6K LOC: scan, join, predicate pushdown, vector, UDF, cost)
- `graph/spo/` — SPO triple store (truth, merkle, semiring, builder, 30 tests)
- `graph/blasgraph/` — GraphBLAS sparse matrix (~5K LOC: 7 semirings, CSR/CSC/COO/HyperCSR, HHTL, cascade)
- `graph/neighborhood/` — neighborhood search (CLAM, scope, zeckf64)
- `graph/metadata.rs` — MetadataStore (Arrow RecordBatch CRUD)
- `graph/graph_router.rs` — Three-backend graph router (blasgraph/DataFusion/palette)
- `nsm/` — DeepNSM DataFusion wiring (tokenizer, parser, encoder, similarity, UDF scaffold)
- `nsm_bridge.rs` — NSM→SPO mapping with NARS truth values + Arrow export (13 tests)

**lance-graph-planner** (10,326 LOC) — `crates/lance-graph-planner/`
- 16 composable strategies: CypherParse, GqlParse, GremlinParse, SparqlParse, ArenaIR, DPJoinEnum, RuleOptimizer, HistogramCost, SigmaBandScan, MorselExec, TruthPropagation, CollapseGate, StreamPipeline, JitCompile, WorkflowDAG, ExtensionPlanner
- `thinking/` — 12 styles, NARS dispatch, sigma chain (Ω→Δ→Φ→Θ→Λ), semiring auto-selection
- `mul/` — Meta-Uncertainty Layer (Dunning-Kruger, trust qualia, compass, homeostasis, gate)
- `elevation/` — dynamic elevation L0:Point→L5:Async (cost model that smells resistance)
- `adjacency/` — Kuzu-style CSR/CSC substrate with batch intersection
- `physical/` — CamPqScanOp, CollapseOp, TruthPropagatingSemiring
- `api.rs` — Planner + CamSearch + Polyglot detection

**lance-graph-contract** (1,076 LOC, ZERO DEPS) — `crates/lance-graph-contract/`
- `thinking.rs` — 36 ThinkingStyles, 6 clusters, τ addresses, FieldModulation, ScanParams
- `mul.rs` — SituationInput, MulAssessment, DkPosition, TrustTexture, FlowState, GateDecision
- `plan.rs` — PlannerContract trait, PlanResult, QueryFeatures, StrategySelector
- `orchestration.rs` — OrchestrationBridge trait, StepDomain, UnifiedStep, BridgeSlot
- `cam.rs` — CamCodecContract, DistanceTableProvider, IvfContract
- `jit.rs` — JitCompiler, StyleRegistry, KernelHandle
- `nars.rs` — InferenceType(5), QueryStrategy(5), SemiringChoice(5)

**deepnsm** (standalone, ~2,200 LOC, 0 deps) — `crates/deepnsm/`
- Replaces transformer inference: 680GB model → 16.5MB, 50ms/token → <10μs/sentence
- 4,096-word COCA vocabulary (98.4% English coverage)
- 4096² u8 distance matrix from CAM-PQ codebook
- 512-bit VSA encoder: XOR bind + majority bundle (word order sensitive)
- 6-state PoS FSM → SPO triples (36-bit packed)

**bgz-tensor** (standalone, ~1,300 LOC, 0 deps) — `crates/bgz-tensor/`
- Attention via table lookup: Q·K^T/√d → table[q_idx][k_idx] in O(1)
- Weight matrix 64MB → Base17 136KB → 256 archetypes 8.5KB → distance table 128KB
- AttentionSemiring: distance table (u16) + compose table (u8) = multi-hop in O(1)
- HHTL cascade: 95% of pairs skipped

**bgz17** (standalone, ~3,500 LOC, 0 deps, 121 tests) — `crates/bgz17/`
- Palette semiring codec, PaletteMatrix mxm, PaletteCsr, Base17 VSA
- SIMD batch_palette_distance, TypedPaletteGraph, container pack/unpack

---

## ndarray Integration Policy

**ndarray is the default dependency.** lance-graph should always use ndarray for:
- `Fingerprint<256>` (canonical type, replaces standalone `NdarrayFingerprint` mirror)
- SIMD dispatch via `simd_caps()` singleton
- CAM-PQ codec (ndarray implements `CamCodecContract`)
- CLAM tree (ndarray has 46 tests, full build+search+rho_nn)
- BLAS L1/L2/L3 via MKL/OpenBLAS/native backend
- ZeckF64 (ndarray is canonical, 3 copies need dedup)
- HDR cascade search
- JIT compilation via jitson/Cranelift

**Fallback without ndarray**: When the `ndarray-hpc` feature is disabled, lance-graph falls
back to its standalone implementations in `blasgraph/ndarray_bridge.rs`. This is for:
- Minimal builds (CI, wasm, embedded)
- Downstream consumers who don't need HPC compute
- Compilation without the full ndarray dependency tree

```toml
# In crates/lance-graph/Cargo.toml:
[dependencies]
ndarray = { path = "../../../ndarray", optional = true, default-features = false }

[features]
default = ["unity-catalog", "delta", "ndarray-hpc"]
ndarray-hpc = ["dep:ndarray"]
```

---

## Build Commands

```bash
# Check workspace (default: with ndarray)
cargo check

# Check without ndarray (fallback mode)
cargo check -p lance-graph --no-default-features

# Run core tests
cargo test -p lance-graph

# Run planner tests
cargo test -p lance-graph-planner

# Run contract tests
cargo test -p lance-graph-contract

# Run standalone codec tests (fast, no network)
cargo test --manifest-path crates/bgz17/Cargo.toml
cargo test --manifest-path crates/deepnsm/Cargo.toml
cargo test --manifest-path crates/bgz-tensor/Cargo.toml

# Run all workspace tests
cargo test

# Python bindings
cd crates/lance-graph-python && maturin develop
```

---

## Current Status (2026-03-28)

### What's DONE
- **Phase 1** (blasgraph CSC/Planner): DONE — CscStorage, HyperCsrStorage, TypedGraph, 7 semirings, SIMD Hamming
- **Phase 2** (bgz17 container/semiring): DONE — 121 tests, PaletteSemiring, PaletteMatrix, PaletteCsr, Base17 VSA
- **Unified planner**: DONE — 16 strategies, MUL assessment, thinking orchestration, dynamic elevation, CAM-PQ operator
- **Contract crate**: DONE — zero-dep trait crate with canonical types for all consumers
- **Polyglot parsing**: DONE — Cypher, GQL (ISO 39075), Gremlin, SPARQL → same IR
- **DeepNSM**: DONE — distributional semantic engine, 4096 COCA vocabulary, 512-bit VSA
- **bgz-tensor**: DONE — attention as table lookup, AttentionSemiring, HHTL cascade
- **NSM bridge**: DONE — NSM→SPO mapping, NARS truth values, Arrow 57 RecordBatch, 13 tests
- **Deep audit**: DONE — 8 inventory documents in docs/ (see docs/UNIFIED_INVENTORY.md)

### What's IN PROGRESS (Phase 3)
- [ ] Wire ndarray as default dep (Cargo.toml change + `ndarray-hpc` feature flag)
- [ ] Replace `NdarrayFingerprint` with `ndarray::hpc::fingerprint::Fingerprint<256>`
- [ ] Dedup ZeckF64 (3 copies → 1 canonical in ndarray)
- [ ] Wire CAM-PQ: ndarray codec → lance-graph UDF → planner operator
- [ ] Contract adoption: planner + n8n-rs + crewai-rust depend on contract crate
- [ ] Move bgz17 from `exclude` to `members` with `bgz17-codec` feature flag
- [ ] Consolidate nsm/ module to thin wrappers over crates/deepnsm/

### What's OPEN (Phase 4+)
- [ ] Wire planner strategies to lance-graph core (actual parser, not regex)
- [ ] Wire elevation to execution with timing feedback
- [ ] GraphRouter 3-backend routing (DataFusion + palette + blasgraph)
- [ ] n8n-rs OrchestrationBridge implementation
- [ ] End-to-end integration test (query → thinking → plan → execute → result)

### Test Summary
- bgz17: **121 passing** (standalone)
- lance-graph core: **~100+ passing** (SPO + Cypher + semirings + NSM bridge)
- lance-graph-planner: **~15 passing** (strategy selection, truth propagation, collapse gate)
- lance-graph-contract: **~15 passing** (thinking styles, τ addresses, modulation)
- Total: **250+ passing**

---

## Key Dependencies

```toml
# Verified against Cargo.lock 2026-06-14. The lance family moves in lockstep at =7.0.0 (PR #445).
arrow = "58"
datafusion = "53"
lance = "=7.0.0"          # exact-pinned: lancedb 0.30.0 transitively requires lance =7.0.0
lance-linalg = "=7.0.0"
lancedb = "=0.30.0"       # 0.30 → lance 7 → object_store 0.13.2 (0.29 → lance 6 → object_store 0.12)
ndarray = { path = "../../../ndarray" }  # AdaWorldAPI fork, default, optional fallback
nom = "7.1"
snafu = "0.8"
deltalake = "0.32"  # optional
```

---

## Architecture Notes

### The Codec Stack (see docs/CODEC_COMPRESSION_ATLAS.md)
```
Full planes (16Kbit, ρ=1.000) → ZeckBF17 (48B, ρ=0.982) → Base17 (34B, ρ=0.965)
  → PaletteEdge (3B, ρ=0.937) / CAM-PQ (6B, varies) → Scent (1B, ρ=0.937)
```

### The Thinking Pipeline (see docs/THINKING_ORCHESTRATION_WIRING.md)
```
YAML card → 23D vector → ThinkingStyle(36) → FieldModulation(7D) → ScanParams
  → MUL assessment → NARS type → semiring selection → 16 strategies → elevation
```

### The Semiring Inventory (see docs/SEMIRING_ALGEBRA_SURFACE.md)
- 7 HDR semirings (blasgraph): XorBundle, BindFirst, HammingMin, SimilarityMax, Resonance, Boolean, XorField
- 1 SPO truth semiring (spo/): HammingMin with frequency/confidence
- 1 palette semiring (bgz17): PaletteCompose with 256×256 distance table
- 1 planner semiring (planner): TruthPropagating with NARS deduction/revision
- 1 attention semiring (bgz-tensor): AttentionTable + ComposeTable
- 5 contract semiring choices: Boolean, HammingMin, NarsTruth, XorBundle, CamPqAdc

### Type Duplication (see docs/TYPE_DUPLICATION_MAP.md)
- Fingerprint/BitVec: 4 copies (ndarray canonical)
- ZeckF64: 3 copies (ndarray canonical)
- ThinkingStyle: 4 copies (contract canonical, not yet adopted)
- Base17: 3 copies (bgz17 canonical)
- NARS InferenceType: 3 copies (contract canonical)
- CSR adjacency: 5 implementations (different purposes)

---

## Cross-Repo Dependencies

```
WHO DEPENDS ON lance-graph-contract:
  in-tree        — planner, callcenter, smb-bridge, symbiont (the BINDING consumers)
  ladybug-rs     — Planner + CamPq + OrchestrationBridge
  (crewai-rust / n8n-rs — EVICTED 2026-06-21, operator-directed; superseded by
   ladybug-rs + the in-tree thinking-engine. NO LONGER binding consumers — the
   contract evolves without a crewai/n8n multi-repo bump. Scattered "replaces
   crewai-rust's X" doc-comments are historical provenance; StepDomain::{Crew,N8n}
   stay reserved-dormant. See EPIPHANIES E-CREWAI-N8N-EVICTED.)

WHO WE DEPEND ON:
  ndarray        — Fingerprint, CAM-PQ, CLAM, BLAS, ZeckF64, HDR cascade, JIT
  lance          — columnar storage, versioning, vector search
  datafusion     — SQL query engine
  arrow          — columnar memory format

SIBLING REPOS:
  /home/user/ndarray/       — The Foundation (BLAS, Fingerprint, CAM-PQ, CLAM, jitson)
  /home/user/crewai-rust/   — The Agents (agent cards, thinking styles, YAML templates)
  /home/user/n8n-rs/        — The Orchestrator (workflow DAG, step routing, compiled styles)
  /home/user/kuzudb/        — Reference (column-grouped CSR adjacency model)
```

---

## Knowledge Base (agents read these before working)

```
.claude/knowledge/signed-session-findings.md     — BF16 tables, gate modulation, quality checks
.claude/knowledge/phi-spiral-reconstruction.md   — φ-spiral, family zipper, stride/offset, Zeckendorf, VSA
.claude/knowledge/primzahl-encoding-research.md  — prime fingerprint, Zeckendorf vs BF16 vs prime encoding
.claude/knowledge/bf16-hhtl-terrain.md           — BF16-HHTL correction chain, 5 hard constraints, probe queue
.claude/knowledge/zeckendorf-spiral-proof.md     — φ-spiral proof (scope-limited, see header before citing)
.claude/knowledge/two-basin-routing.md           — Two-basin doctrine, representation routing, pairwise rule, attribution
.claude/knowledge/encoding-ecosystem.md          — MANDATORY: full encoding map, synergies, read-before-write checklist
.claude/knowledge/frankenstein-checklist.md       — Composition failure modes (VibeTensor §7), boundary test matrix
.claude/knowledge/lab-vs-canonical-surface.md     — MANDATORY before touching REST/gRPC/Wire DTO/endpoint/shader-lab (prevents "add another REST endpoint" hallucination)
.claude/knowledge/autoattended-multiagent-pattern.md — MANDATORY before planning a wave with ≥4 parallel workers; 4-savant taxonomy (PP-13/14/15/16), worker iron rules, atomic-consolidation pass
.claude/knowledge/ndarray-vertical-simd-alien-magic.md — MANDATORY before writing SIMD code in any consumer crate OR filing a PR against `adaworldapi/ndarray` `src/simd_*`; canonical reference for the wave W1a (5 ndarray primitives) + W1b (5 consumer migrations) + W1.5 (3 sigker primitives, gated on jc Pillar 11) plan
.claude/knowledge/core-first-transcode-doctrine.md — MANDATORY before any C++→Rust transcode / codegen / AST-DLL / "port Tesseract" / DO-adapter work; the Core-first inversion (a generated layer is only as clean as the OGAR Core it targets) + the OGAR movable-parts assume-contract + the unicharset adapter-parity falsifier. Carried by the core-first-architect / adapter-shaper / core-gap-auditor ensemble.
.claude/knowledge/ogar-consumer-preflight.md     — MANDATORY before any consumer (woa-rs/medcare-rs/smb-office-rs) classid/bridge/codebook work; the consumer-side spellbook (pull via *Port::class_id / contract::ogar_codebook::canonical_concept_id; NEVER construct a *Bridge or copy the codebook). Inverse of OGAR's SURREAL-AST-TRAP-PREFLIGHT; operational mirror of docs/CONSUMER-BRIDGE-DEPRECATION.md (#589/#590).
.claude/CALIBRATION_STATUS_GROUND_TRUTH.md       — OVERRIDE: read BEFORE any SESSION_*.md
.claude/PLAN_BF16_DISTANCE_TABLES.md             — 5-phase plan for BF16 distance tables
.claude/TECHNICAL_DEBT_SIGNED_SESSION.md          — 56% useful, 44% bypass (honest review)
.claude/CODING_PRACTICES.md                       — 6 patterns from EmbedAnything + quality checks
```

## Knowledge Activation (MANDATORY)

**P0 Rule: `.claude/knowledge/encoding-ecosystem.md` must be read BEFORE any
codec, encoding, distance, compression, or representation work.** This is the
map of all 8+ encoding representations, their crate locations, their invariants,
their synergies, and their FINDING/CONJECTURE status. Never guess architecture.

**P0 Rule: `.claude/knowledge/lab-vs-canonical-surface.md` must be read BEFORE
any work that mentions REST, gRPC, Wire DTOs, `/v1/shader/*` endpoints, the
shader-lab binary, `OrchestrationBridge`, `UnifiedStep`, codec research ops,
or "external API".** The canonical consumer surface is `UnifiedStep` via
`OrchestrationBridge` — the REST/gRPC server and per-op Wire DTOs are
LAB-ONLY scaffolding. Adding another `/v1/<thing>` endpoint is the
Kahneman-Tversky System-1 easy path and is nearly always wrong; extending
the canonical bridge is the System-2 correct move. See the Decision
Procedure in that doc before writing a single new handler.

**Best practice (P0 for transcode work): `.claude/knowledge/core-first-transcode-doctrine.md`
must be read BEFORE any C++→Rust transcode, codegen, AST-DLL, "port Tesseract",
or DO/DTO-adapter work.** The Core-First inversion: a generated layer (AST /
adapters / codegen'd Rust) is only ever as clean as the Core it targets, so the
**OGAR Core is shaped first, deliberately** — never treated as the residue of
"what we couldn't codegen." Emit **thin classid-keyed adapters that ASSUME the
Core** (identity = `classid`; state = SoA value tenants per the #511
`SoaMemberSpec` calibration; relations = `EdgeBlock`;
composition/inheritance = `classid → ClassView`; invocation = `UnifiedStep`),
and treat the `ruff_cpp_spo` SPO harvest (`has_function` / `inherits_from` /
`virtually_overrides`) as the **ClassView method-resolution manifest** — the
harvest and the codegen are two halves of ONE system, not orthogonal. **Never**
build a parallel Tesseract-rs object model, **never** let an adapter carry its
own state (a Core gap → *extend the Core deliberately*, never hack the adapter),
and **never** force intrusive/stateful methods into the adapter mold (route them
to raw-pointer hand-port — Frankenstein flattening guard). Holds for
mechanical/data-shaped leaf methods only; CONJECTURE until
`PROBE-OGAR-ADAPTER-UNICHARSET` runs byte-parity green. Carried by the
`core-first-architect` / `adapter-shaper` / `core-gap-auditor` ensemble.

**Best practice (consumer-side): `.claude/knowledge/ogar-consumer-preflight.md`
is the consumer mirror of OGAR's SurrealQL-AST trap.** Read it BEFORE any consumer
crate (woa-rs / medcare-rs / smb-office-rs) pulls a classid, migrates off a
`*Bridge`, or touches the OGAR codebook. The consumer trap is the *inverse* of the
producer trap: re-implementing the Core locally — a `*Bridge` construction, a local
`*_CODEBOOK` copy, a parallel registry — instead of pulling via
`*Port::class_id` (spine) / `contract::ogar_codebook::canonical_concept_id`
(membrane, BBB-safe), then stamping the OGAR#95 app prefix. Five questions, 90
seconds. Carried by `integration-lead` / `baton-handoff-auditor` /
`core-first-architect`.

Every `.claude/knowledge/` document has a `READ BY:` header listing which agents
MUST load it before producing output in that domain. When a knowledge trigger fires
(see `.claude/agents/BOOT.md § Knowledge Activation Protocol`), the relevant
knowledge docs are loaded BEFORE the agent responds.

**Critical process rule:** `.claude/knowledge/bf16-hhtl-terrain.md` contains a
probe queue with CONJECTURE/FINDING status for each architectural claim. Any agent
proposing changes to HHTL cascade, γ+φ placement, Slot D/V layout, or bucketing
strategy MUST check the probe queue first. If the relevant probe is NOT RUN, the
next deliverable is the probe, not more synthesis.

**Insight update cycle:**
```
Claim → Probe defined (pass/fail criteria) → Probe written (example file)
→ Probe run → Result recorded in knowledge doc → CONJECTURE promoted to FINDING or corrected
```
No knowledge doc should contain unmarked conjectures. Label everything.

## In-Session Orchestration Discipline

**P0 Rule: Read before Write, always.** Before calling `Write` on any path
that may already exist, run `Read` (or `git status` for committed files).
The `Edit` tool is the default for modifying existing files; `Write` is only
for new files or genuine full rewrites the user explicitly asked for. This
rule applies to the model's own prior commits in the same session — "I just
wrote this" is not a license to overwrite it without checking state.

**Diagnostic signature of the failure mode:** a `git diff` showing `~N
insertions / ~N deletions` on a file of size N — same magnitude, same shape,
virtually every line different — means the file was *regenerated from prompt*
instead of *built from state*. If you see this on your own commit, you just
overwrote committed work. Revert with `git restore <path>` and use `Edit`
for any genuine refinement.

**Tool-reach reminder for deferred tools.** `AskUserQuestion`, `TodoWrite`,
`WebSearch`, `WebFetch` are namechecked in the Claude Code system prompt but
sit behind `ToolSearch` in Opus 4.6 — their schemas are not loaded by default.
Do not treat "not in the current schema list" as "not available." Reach for
`ToolSearch` with `select:<name>` when the situation calls for them
(multi-step tracking, user clarification on ambiguous denials, web lookups).

**Upstream regression filed:**
[anthropics/claude-code#46861](https://github.com/anthropics/claude-code/issues/46861)
documents the pattern observed in this workspace (Opus 4.6 reproducer: post-
commit `Write`-over-self without `Read`; deferred-tool reach failure for the
tools listed above). Treat that issue as the canonical reference if a future
session asks why this section exists. The 172/171 diff that triggered the
filing was on `.claude/prompts/arxiv.md` in this repo, branch
`claude/risc-thought-engine-TCZw7`.

## Model Registry (Jina v5 is ground truth anchor)

```
Model            Base       Tokenizer      Vocab    Hidden  Act   Status
─────            ────       ─────────      ─────    ──────  ───   ──────
Jina v5 small    Qwen3      Qwen3 BPE      151K    1024    silu  GROUND TRUTH (safetensors + ONNX on disk)
Reranker v3      Qwen3      Qwen3 BPE      151K    1024    silu  SAME BASE as v5 (listwise, cross-encoder)
ModernBERT-large OLMo       OLMo BPE       50K     1024    gelu  ONNX on disk (GeGLU, code-friendly)
BGE-M3           XLM-R      SentencePiece  250K    1024    gelu  baked u8 lens (multilingual)
Jina v3          XLM-R      SentencePiece  250K    1024    gelu  baked u8 lens (LEGACY, not ground truth)
Qwopus 27B       Qwen2      Qwen2 BPE      151K    5120    silu  305 tables in Release v0.1.2
Reader-LM 1.5B   Qwen2      Qwen2 BPE      151K    —       —     in Release, not baked

CRITICAL: Reranker v3 = Qwen3 base (NOT v2 XLM-RoBERTa).
  Same tokenizer as Jina v5. Same architecture. Same silu gate.
  Baked reranker lens was built from Qwen2 GGUF → needs rebuild with Qwen3 tokens.

Tokenizer sharing:
  Qwen3 BPE (151K): Jina v5, Reranker v3
  Qwen2 BPE (151K): Qwopus, Reader-LM (DIFFERENT from Qwen3!)
  XLM-RoBERTa (250K): Jina v3, BGE-M3 (LEGACY)
  OLMo (50K): ModernBERT
```

## Thinking Engine (crates/thinking-engine/)

```
Core:        engine.rs, bf16_engine.rs, signed_engine.rs, role_tables.rs
Composition: composite_engine.rs, dual_engine.rs, layered.rs, domino.rs
Calibration: cronbach.rs, ground_truth.rs, reencode_safety.rs (x256 proven)
Encoding:    spiral_segment.rs, prime_fingerprint.rs (VSA bundle perturbation)
Patterns:    pooling.rs, builder.rs, auto_detect.rs, tokenizer_registry.rs
Sensors:     jina_lens.rs, bge_m3_lens.rs, reranker_lens.rs, sensor.rs
Cognition:   cognitive_stack.rs, ghosts.rs, persona.rs, qualia.rs, world_model.rs
Bridge:      l4_bridge.rs, bridge.rs, semantic_chunker.rs, tensor_bridge.rs

Examples:
  jina_v5_ground_truth.rs    — end-to-end pipeline (tokenize → ground truth)
  end_to_end_signed.rs       — BF16/i8 smoke test (CDF collapse confirmed)
  dual_signed_experiment.rs  — u8 vs BF16 comparison across 3 lenses
  calibrate_lenses.rs        — Spearman ρ + ICC (real Qwen3 tokenizer)
  stream_signed_lens.rs      — 5-lane encoder from GGUF (BF16 stream)
  stream_hdr_lens.rs         — u8 CDF HDR lens from GGUF

Data on disk (gitignored, download from HF or Releases):
  jina-v5-onnx/              model.safetensors (1.2 GB) + model.onnx (2.3 GB) + tokenizer
  modernbert-onnx/           model.onnx (1.5 GB) + tokenizer
  jina-reranker-v3-BF16-5lane/ 5-lane encoding (u8/i8/γ+φ, 1.1 MB)
```

## Documentation Index

```
docs/UNIFIED_INVENTORY.md              — Master index, LOC census, crate map, dep graph
docs/TYPE_DUPLICATION_MAP.md           — Every duplicated type with file:line precision
docs/CODEC_COMPRESSION_ATLAS.md        — Full→ZeckBF17→BGZ17→CAM-PQ→Scent chain
docs/THINKING_ORCHESTRATION_WIRING.md  — End-to-end thinking pipeline (9 layers)
docs/SEMIRING_ALGEBRA_SURFACE.md       — All 15 semirings across all repos
docs/ADJACENCY_SYNERGY_MAP.md          — 6 CSR models, BLAS×PQ×BGZ17, KuzuDB
docs/METADATA_SCHEMA_INVENTORY.md      — 4096 surface, 16Kbit, Lance persistence, Arrow
docs/INTEGRATION_DEBT_AND_PATHS.md     — Strengths, weaknesses, epiphanies, 7 paths
docs/ORCHESTRATION_IS_GRAPH.md         — Capstone: orchestration AS graph traversal
docs/CONSUMER_WIRING_INSTRUCTIONS.md   — How to consume lance-graph-contract
```

## Session: AutocompleteCache + p64 Convergence (2026-03-31)

### New in lance-graph-planner
- `src/cache/` — 7 modules, 39 tests:
  - `kv_bundle.rs`: HeadPrint=Base17 (from ndarray), AttentionMatrix 64×64/256×256
  - `candidate_pool.rs`: ranked candidates, Phase (Exposition→Coda)
  - `triple_model.rs`: self/user/impact × 4096 heads, DK, Plasticity, Truth=NarsTruth
  - `lane_eval.rs`: Euler-gamma tension, DK-adaptive, 4096-head evaluation
  - `nars_engine.rs`: SpoHead, Pearl 2³, NarsTables (causal-edge hot path), StyleVectors
  - `convergence.rs`: AriGraph triplets → p64 Palette layers → CognitiveShader
  - `kv_bundle.rs`: VSA superposition store
- `src/strategy/chat_bundle.rs`: AutocompleteCacheStrategy (Strategy #17)
- `src/serve.rs`: Axum REST server, OpenAI-compatible /v1/chat/completions
- `AUTOCOMPLETE_CACHE_PLAN.md`: full implementation plan with 6 agent scopes

### New in bgz-tensor
- `src/hhtl_cache.rs`: HHTL cache with RouteAction (Skip/Attend/Compose/Escalate), HipCache k=64
- `src/hydrate.rs`: --download/--reindex/--verify with feature flags (qwen35-9b/27b-v1/v2)
- `data/manifest.json`: SHA256 for all 41 shards
- GitHub Release v0.1.0-bgz-data: 41 bgz7 assets, 685 MB

### Dependencies
- lance-graph-planner → ndarray (hardware: Base17, read_bgz7_file)
- lance-graph-planner → causal-edge (protocol: CausalEdge64, NarsTables)
- lance-graph-planner → p64 + p64-bridge + bgz17 (convergence highway)

### Architecture Rules
- ndarray = hardware acceleration (SIMD, no thinking logic)
- causal-edge = protocol (CausalEdge64, NarsTables = precomputed NARS as lookup tables)
- lance-graph-planner = thinking (NarsEngine, AutocompleteCache, Styles)
- p64 = convergence point (both repos meet, no circular deps)
- AriGraph (lance-graph core) cannot be planner dep (circular) — use p64 convergence instead

### 18 Papers Synthesized
EMPA, InstCache, Semantic Caching, C2C, ContextCache, Krites, Thinking Intervention,
ThinkPatterns, Thinkless, Holographic Resonance, DapQ, Tensor Networks, PMC Attention Heads,
LFRU, Illusion of Causality, NARS Same/Opposite, KVTC, CacheSlide.
All findings in `.claude/knowledge/session_autocomplete_cache.md`.

### Benchmark
611M SPO lookups/sec. 17K tokens/sec. 388 KB RAM. 100% information preservation.


## CANON — Minimal SoA node + zero-fallback ladder (locked 2026-06-13)

> Operator-locked this session. Append-only. Wrapper `NodeGuid` (#480) is audited
> against THIS, never the reverse. No RFC-9562 ceremony in the hot key.

**Node = 4096 bit = 512 byte:** `key(16) | edges(16) | value(480)`.

**Key (16 byte, little-endian):**
```
 0..4   classid   (u32)   8 hex — prefix-routable; default 0x0000_0000
 4..6   HEEL      (u16)   ┐
 6..8   HIP       (u16)   ├ 3 cascade tiers (HHTL path)
 8..10  TWIG      (u16)   ┘
10..13  family    (u24)   ┐ trailing 6 bytes = basin-local key
13..16  identity  (u24)   ┘ (masked load after the trie binds the prefix)
```
`local_key()` = bytes 10..16 (family ++ identity), the only discriminator once
the prefix is resolved.

**Edge block (16 byte):** 12 in-family + 4 out-of-family, one byte per slot.
Canonical, NOT mandatory — always reserved (zeroed when unused), never shrunk; a
class opting out of edges is resolved via `classid → ClassView`, never by a stride
change.

**Zero-fallback ladder (monotonic — zero = fall through to the broader default):**
- `classid == 0x0000_0000` → default class, no prefix routing (dormant)
- `family  == 0x00_0000`    → default basin, no neighborhood grouping (dormant)
- ⇒ while both are zero, `identity` (24 bit) ALONE discriminates — the bootstrap
  address. 16.7M identities per default basin; mint a non-zero family to expand.

**RESERVE, DON'T RECLAIM:** a zero tier means *not consulted*, never *compacted
away*. classid(4B) and family(3B) keep fixed offsets so a non-zero mint later
wakes routing/basin binding with ZERO `ENVELOPE_LAYOUT_VERSION` change. Mint path
must `debug_assert` identity uniqueness while in the default basin.

**No UUID ceremony:** no version nibble, no variant bits, no namespace/kind. The
dash-groups are the only semantics; the tail is plain `family|identity`.

Reference impl: `canonical_node.rs` (`NodeGuid` / `EdgeBlock` / `NodeRow`, with
`const _` size asserts at 16/16/512 and `is_default_class` / `is_unbasined` /
`is_bootstrap_address` guards).
