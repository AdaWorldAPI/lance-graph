# Technical Debt Log — Open + Paid (double-entry, append-only)

> **Append-only ledger** for knowingly-deferred work: TODOs, shortcuts,
> workarounds, unsafe assumptions, missing probes, hardcoded
> thresholds, stubs, and anything else we shipped with intentional
> debt. Debt moves Open → Paid by status-flip; rows are NEVER
> deleted.
>
> **Purpose:** separate from `ISSUES.md` (bugs) and `IDEAS.md`
> (speculation). Tech debt is **code that works but we know
> something better is owed**. This file is where we admit it.

---

## Double-entry discipline

Same pattern as `ISSUES.md`:

1. **Open Debt** — known shortcut or deferral, captured at shipping
   time.
2. **Paid Debt** — when the shortcut is replaced with the proper
   implementation, append here with PR anchor + Status flip on the
   original Open entry to `Paid YYYY-MM-DD`.

The Open entry stays in its section forever (chronology). The Paid
section accumulates retirements for audit.

---

## Governance

- **Append-only.** Never delete a row.
- **Mutable fields:** `**Status:**` and `**Payoff:**` lines only.
- **`permissions.ask` on Edit** (same rule as other bookkeeping
  files).

## Cross-references

- `ISSUES.md` — an issue may become tech debt when knowingly
  deferred rather than fixed.
- `IDEAS.md` — a rejected idea that was "the better way" often
  leaves behind a debt entry documenting the compromise shipped
  instead.
- `PR_ARC_INVENTORY.md` — which PR introduced the debt + which PR
  paid it.
- `STATUS_BOARD.md` — debt items that block deliverables are
  cross-referenced from the D-id row.
- `EPIPHANIES.md` — an epiphany often retroactively turns something
  into tech debt (the old approach is now known to be suboptimal).

---

## Kanban Format (priority + scope on every entry)

Every debt item carries:
- **Priority** — `P0` must-pay-before-next-phase / `P1` pay-soon /
  `P2` eventual / `P3` keep-tracked-but-low.
- **Scope** — which agent / deliverable / domain owes it:
  `@<agent-name>`, `D<N>`, `domain:<tag>`.

Ticket tag: `[P2 @truth-architect D10 domain:grammar]`. Same
filter discipline — agents pull their own debt by `@`-mention.

## Open Debt

(Seeded with known deferrals from recent PRs. New items PREPEND
with today's date.)

```
## YYYY-MM-DD — <short title>
**Status:** Open
**Priority:** P0 | P1 | P2 | P3
**Scope:** @<agent> D<N> domain:<tag>
**Introduced by:** PR #NNN
**Payoff estimate:** <rough LOC / time>

<one paragraph: what shortcut was taken, why, what the proper fix
looks like, any blocking dependencies>

Cross-ref: <file:line / deliverable D-id / epiphany entry>
```

## 2026-04-24 — Frankenstein blast radius on branch `claude/read-claude-md-jh51O` (Vsa10k / L3 / 157 confusion)

**Status:** Open
**Priority:** P0
**Scope:** @integration-lead @truth-architect domain:vsa domain:plans
**Introduced by:** this branch, commits `468357d`, `7a60c42`, `585f8b0`, `489911b` (plus inherited from earlier `dfcf246` / PR #210 / PR #242-#243)
**Payoff estimate:** plan-doc surgery (1 day) + contract rename already scoped in IDEAS.md (see 2026-04-19 entries)

Session on `claude/read-claude-md-jh51O` ran out of context before cleanup. Next session needs these breadcrumbs.

### Root errors (three clustered, one session)

1. **"L3" misread.** User said "L3" = CPU cache hardware budget (~16 MB). Session interpreted as cognitive-stack Layer 3 and built a fabricated precision-tier architecture on top.
2. **`Vsa10k BF16` fabrication.** No such type exists. The real types are `CrystalFingerprint::Vsa10kI8` (10 KB) and `Vsa10kF32` (40 KB legacy) — variants of `CrystalFingerprint`, not precision tiers of `Vsa10k`.
3. **`Vsa10k = [u64; 157]` inherited confusion.** 157-word bit-packed labelled "10K VSA" is the pre-existing error (source: `dfcf246`). Bit-packed never uses 10,000 — uses 16,384 (powers of 2). Only the high-dim numeric 40 KB / 80 KB forms legitimately carry 10K-D framing, and only for grammar ±5 wire bundling. Workspace already filed the rename in `IDEAS.md` 2026-04-19 entries (157 → 256 for binary; Vsa10k* → Vsa16k* for VSA floats); this session ignored the filed correction.

### Blast radius — plans written this session (needs surgery)

| File | Sections | Verdict |
|---|---|---|
| `.claude/plans/callcenter-membrane-v1.md` | §§ 14-17 (commit `468357d`) | Mixed. §14 cold-storage + §15 VSA-UDF-dispatch concept good; concrete sizes/type names poisoned. §16 persona-as-function (32 atoms × 16 weights, PersonaSignature 56-bit) good. §17 four-way multiply + ONNX-replaces-Chronos good; tensor-shape annotations poisoned. |
| `.claude/plans/callcenter-membrane-v1.md` | § 18 (commit `7a60c42`, lines 1139-1224) | **Delete whole section.** Three-tier precision table, father-grandfather compression, CK-safety proof — all built on fabricated `Vsa10k BF16` and L3-as-cognitive-layer. |
| `.claude/plans/callcenter-membrane-v1.md` | line 822 (edited `489911b`) | Already partially fixed. Still references "Fingerprint<256>" as if distinct from Vsa10k — verify correctness after rename. |
| `.claude/plans/unified-integration-v1.md` | DU-3 precision-note (lines 196-201, commit `7a60c42`) | **Delete paragraph.** References §18 that must go. DU-3 body (5 UDFs signature) kept but delegates must use canonical `RoleKey::bind/unbind` from PR #243. |
| `.claude/plans/unified-integration-v1.md` | DU-1 ONNX classifier (`468357d`) | Good core idea. `recent_fingerprints: Tensor[N, 16384]` tensor-shape comment at line 86 needs review against the 16K binary vs 16K-D float substrate split (see IDEAS.md 2026-04-19 CORRECTION). |
| `.claude/plans/unified-integration-v1.md` | DU-2 Archetype ECS bridge | Good. Mapping table (Entity=PersonaCard / World=Blackboard / Tick=CollapseGate fire) is sound. |
| `.claude/plans/unified-integration-v1.md` | DU-4 `rationale_phase: bool` (commit `a05979e`) | Shipped correctly to `CognitiveEventRow`. No change needed. |

### Blast radius — source (this session did NOT modify these; flagging for rename sweep)

- `crates/lance-graph-contract/src/grammar/role_keys.rs` — `VSA_WORDS = 157`, `Vsa10k = [u64; 157]`. Originates `dfcf246` (pre-PR #210). Already in rename sweep per IDEAS.md `CORRECTION-OF 2026-04-19 FP_WORDS`.
- `crates/deepnsm/src/{content_fp,markov_bundle,trajectory,context,encoder}.rs` — all consume `Vsa10k` / `VSA_WORDS = 157`. Added in PR #243 (`c6e69c4`). Follows whichever direction the rename goes.
- `CLAUDE.md` P-1 section — references `[u64; 157]` and `trajectory: Vsa10k`. Prose; updates after source rename.

### Blast radius — vsa_udfs.rs (commit `585f8b0`)

`crates/lance-graph-callcenter/src/vsa_udfs.rs` (628 lines) has three wrong operations — flagged separately for the canonical-delegation pass:

- `unbind_op`: set-bit-counting on role-indexed slice → returns fraction. WRONG — should delegate to `RoleKey::unbind` + `recovery_margin` (shipped in `79ac8f6` / PR #242).
- `bundle_op`: implements XOR, named "vsa_bundle". WRONG NAME — XOR is `MergeMode::Xor` (single-writer delta); violates **I-SUBSTRATE-MARKOV** iron rule if used on state-transition paths. Rename to `vsa_xor` (honest) or implement true CK-safe bundle.
- `braid_at_op`: cyclic word rotation. WRONG — should use `vsa_permute` (PR #209 reference braid).

### Good ideas salvageable (architectural, not in poisoned sections)

Verbatim user framings from this session, recorded so they don't get lost:

1. **Two callcenter modes coexist.** (a) DataFusion polyglot query mode — addresses the 4096 / 0xFFF command space (SPARQL / SQL / Cypher / GQL / NARS); Redis-like UDF access. (b) VSA blackboard orchestration — roles bind work orders into accumulator; CollapseGate XOR flushes into Markov trajectory; new empty ledger starts.
2. **VSA lazy-buffer pattern.** Bind events → accumulate in VSA → CollapseGate XOR flush → Markov trajectory → scalar `CognitiveEventRow` projection → external consumer. The accumulator is the lazy buffer; CollapseGate is the bell.
3. **Kitchen analogy.** Work orders bind into VSA (order sheet) → CollapseGate = bell → callcenter/waiter picks up scalar `CognitiveEventRow` → external consumer gets ticket.
4. **BBB iron rule already in contract source** (`external_membrane.rs:10`): `Self::Commit` MUST NOT contain `Vsa10k`, `RoleKey`, `SemiringChoice`, `NarsTruth`. Any plan proposing these cross the gate is invalid by construction.
5. **Grammar ±5 wire format is the ONLY legitimate 10,000-D / 10K-D / 16K-D float carrier.** 40 KB (f32) / 80 KB (u8 5-lane) per the IDEAS.md rename plan. Bit-packed fingerprint substrate is separate: `Container<[u64; 256]>` = 16,384 bits = 2 KB, Hamming-only, NOT VSA.
6. **Chronos → ONNX replacement** (DU-1): full 288-class `(ExternalRole × ThinkingStyle)` product prediction vs Chronos 1D scalar. Grounding sound; tensor-shape details need re-verification post-rename.
7. **Archetype ECS bridge** (DU-2): `VangelisTech/archetype` sits ABOVE callcenter; Entity = PersonaCard, World = Blackboard, Tick = CollapseGate fire. Thin adapter crate `lance-graph-archetype-bridge`. Mapping table intact.

### Prior art this session ignored

- `.claude/board/IDEAS.md` 2026-04-19 entries: "FP_WORDS = 256 (supersede the 160 plan)" + "CORRECTION-OF 2026-04-19 FP_WORDS = 256" + "REFINEMENT-OF 2026-04-19 CORRECTION-OF FP_WORDS scope". Explicit governance ban: *"There shall be zero occurrences of '10,000-D binary VSA' / '10,000-bit VSA' in any workspace file."* This session wrote multiple such occurrences.
- `.claude/board/TECH_DEBT.md` 2026-04-19 `FP_WORDS = 157 (not 160)` entry (exists above this row once this appends).
- `.claude/board/TECH_DEBT.md` 2026-04-19 `VSA substrate renaming: Vsa10k* → Vsa16k* + float framing` entry.

### Archetype / Chronos — breadcrumbs the plan docs do NOT yet carry

Flagging here what's in this session's conversation context but not in `unified-integration-v1.md` or `callcenter-membrane-v1.md`. Budget honesty: recording only what I can attribute to this session's conversation — not reconstructing.

**Archetype — name collision is undocumented.** "Archetype" in DU-2 refers exclusively to the external `VangelisTech/archetype` Rust ECS crate (simulation layer above callcenter). But the user explicitly noted mid-session: *"What is internal is the person archetypes"* — pointing at `crates/thinking-engine/src/persona.rs` (internal VSA-bound cognitive archetypes). These are two distinct objects sharing the word "archetype":

| Sense | Lives in | Role | Covered in plans? |
|---|---|---|---|
| External ECS Archetype | `VangelisTech/archetype` crate | Simulation tick layer ABOVE callcenter | Yes — DU-2 |
| Internal persona-archetype | `thinking-engine/src/persona.rs` | VSA-bound cognitive identity INSIDE substrate | **No** — only contract-side `PersonaCard` (metadata routing) is named, not the internal archetype |

Next session: add an explicit disambiguation paragraph to DU-2 and to `callcenter-membrane-v1.md` § 16 ("Persona as function") noting that `PersonaCard` (contract, metadata) and `thinking-engine::persona::*` (internal, VSA-bound archetype) are different objects; the callcenter sees the former, never the latter. The "32 atoms × 16 weights" formulation in § 16 sits at the metadata/compression layer — whether it corresponds to or is distinct from the internal archetype is an **open question** (I don't have confident attribution for this from this session's conversation).

**Chronos — what's captured vs what's missing.** Replacement rationale ("1D scalar → 288-class `(ExternalRole × ThinkingStyle)` product"; "ONNX infra already justified by Jina v5 ONNX on disk"; "task = classification, not time-series forecasting") is captured in § 17 of callcenter-membrane-v1.md and DU-1 of unified-integration-v1.md. I do NOT hold confident additional Chronos-specific content from this session's brainstorming that could be added without fabrication. Flagging so the next session knows: if the $200 session's memory of the brainstorm contains richer Chronos material, that's new information to add, not something this session held and failed to record.

**Archetype × Chronos interplay — NOT captured anywhere.** The user grouped "archetype and chronos" together as joint brainstorming content. The plans treat them as independent deliverables (DU-1 Chronos-replacement, DU-2 Archetype-bridge). Whether they compose (e.g. Archetype tick driving ONNX-classifier-as-style-oracle at each tick, feeding Blackboard rounds) is **not documented**. Candidate composition: `ArchetypeTick → UnifiedStep → CollapseGate fire → ONNX classifier predicts next (role, style) → next tick's PersonaCard` — but this is my conjecture, not session-attributable, so flagging as an **open design question** for the next session rather than writing it into a plan.

### ONNX > Chronos — what's in plans vs what's missing

§ 17 of `callcenter-membrane-v1.md` (commit `468357d`) holds a 6-criterion "Why ONNX over Chronos" table: **Output** (1D scalar vs 288 logits) · **Task type** (forecasting vs classification) · **Training** (pre-trained vs Lance E-DEPLOY-1 corpus) · **Precision** (style axis vs role × thinking product) · **Infra** (separate model vs `ort` crate justified by Jina v5 ONNX on disk) · **Fit** (partial vs full product).

**Gap:** the table enumerates where Chronos LOSES but not where Chronos WOULD LEGITIMATELY WIN. User flagged "Chronos only useful for XYZ" as a piece of brainstorm content not captured. I do NOT hold session-attributable XYZ content. Reasoning from first principles (NOT session-attributable, flag if reused): Chronos is appropriate when the task is genuinely temporal forecasting — predicting next F-value N cycles ahead, predicting style-drift onset, predicting gate-commit-rate over a rolling window, auxiliary time-series heads on the Lance internal_cold timeline. These would compose with (not replace) the ONNX classifier's instantaneous prediction. The next session should either fill the XYZ from their own brainstorm recall, or explicitly reject Chronos across all use cases.

### Archetype / persona / thinking-style modeling — epiphany candidates not yet in EPIPHANIES.md

§ 16 and § 17 hold modeling insights that sit in plan docs but have NOT been promoted to dated entries in `.claude/board/EPIPHANIES.md`:

- **Four-way multiply = architecture** (§ 17): `(persona 288 × style 36 × stage 2 × learned-dynamics)` ≈ 20 736 × oracle configurations; F-descent IS the automatic architecture search over this space; misaligned configurations are dropped by the CollapseGate predicate. Epiphany framing: *free-energy minimisation over the four-way product = neural-architecture search without an outer optimiser*.
- **Persona as function** (§ 16): 32 cognitive atoms × 16 weightings = 16^32 addressable persona space, compressed to 56-bit PersonaSignature. YAML runbooks are macro scaffolding for the context loop, not persona identity. Epiphany framing: *persona identity is a coordinate in atom-space, not a YAML artefact; the YAML is a program on the context loop*.
- **MM-CoT stage split = faculty asymmetry** (§ 17 row): `rationale_phase: bool` maps to `FacultyDescriptor::is_asymmetric()` (inbound_style ≠ outbound_style). Epiphany framing: *MM-CoT rationale-vs-answer split is NOT a new axis — it reuses the existing faculty asymmetry from the contract*.

**Already in EPIPHANIES.md:** `E-DEPLOY-1` (commit `5dbdf25`) captures the Supabase-shape thinking-extension 9-dim joint epiphany. The three candidates above are NOT yet prepended there. Board-hygiene rule from CLAUDE.md requires EPIPHANIES entries in the same commit as the plan additions — this session violated that rule for § 16 / § 17.

Next session action: prepend three dated EPIPHANIES.md entries using the framings above, cross-referenced to § 16 / § 17 / commit `468357d`. Do NOT write additional modeling content I don't hold; if the next session has richer brainstorm recall, let them author from that rather than inheriting my reconstruction.

### Correction plan for next session (P0 order)

1. **Delete** `callcenter-membrane-v1.md` § 18 (lines 1139-1224).
2. **Delete** `unified-integration-v1.md` DU-3 precision-note (lines 196-201).
3. **Append** EPIPHANIES.md entry with the three-cluster root-error summary (L3 / Vsa10k-BF16 fabrication / 157 inheritance), referencing this TECH_DEBT row.
4. **Fix** `vsa_udfs.rs` three operations to delegate to canonical `RoleKey::bind/unbind`, `vsa_permute`, and an honest XOR name.
5. **Update** `LATEST_STATE.md` § Current Contract Inventory to include `FacultyRole` + `FacultyDescriptor` (added this session in `2a4a245`, never board-logged — same-commit hygiene rule violation).
6. **Proceed with** the IDEAS.md Vsa10k* → Vsa16k* rename sweep OR scope-limit it per the REFINEMENT entry (grammar / quantum / ladybug allow-list) — that is its own PR, not a cleanup prerequisite.

Cross-ref: EPIPHANIES.md entries needed; IDEAS.md 2026-04-19 rename-sweep entries; PR #242 (`defe928`) The Click categorical-algebraic click; PR #243 (`c6e69c4`) D5 Trajectory; this branch commits `468357d` / `7a60c42` / `585f8b0` / `489911b` / `a05979e` / `2a4a245`.

---

### Seeded from PRs #204–#211

## 2026-04-19 — Contract `ContextChain::coherence_at` returns 0 for non-Binary16K variants
**Status:** Open
**Priority:** P2
**Scope:** @resonance-cartographer @container-architect D4 domain:grammar
**Introduced by:** PR #210
**Payoff estimate:** ~80 LOC + tests

D4 shipped with Hamming-based coherence on the `Binary16K` variant
only; other `CrystalFingerprint` variants (`Structured5x5`,
`Vsa10kI8`, `Vsa10kF32`) return 0 as a zero-dep fallback. Cosine
coherence on the f32 variants would unlock richer disambiguation
but requires adding a minimal linear-algebra shim without breaking
the zero-dep invariant of the contract.

Cross-ref: `crates/lance-graph-contract/src/grammar/context_chain.rs`.

## 2026-04-19 — CausalityFlow has 3/9 TEKAMOLO slots; modal/local/instrument + beneficiary/goal/source deferred
**Status:** Open
**Priority:** P1
**Scope:** @integration-lead @truth-architect D1 domain:grammar
**Introduced by:** PR #208 + #210 (deliberate deferral)
**Payoff estimate:** 6 new `Option<String>` fields in
`lance-graph-cognitive/src/grammar/causality.rs` + tests

Full thematic-role inventory needs 6 more slots on `CausalityFlow`.
Deferred per user decision; D2 ticket emission and D3 triangle
bridge map only the 3 existing slots for now. Phase 2 work is
consistent with 3/9; Phase 3 may benefit from the extension.

Cross-ref: `grammar-landscape.md` §3, `STATUS_BOARD.md` D1 row.

## 2026-04-19 — Named-Entity pre-pass (NER) is the biggest OSINT blocker; stubbed out
**Status:** Open
**Introduced by:** architectural choice (all PRs)
**Payoff estimate:** dedicated PR, ~800 LOC new crate / subsystem

COCA 4096 has zero coverage of proper nouns (Altman, Anthropic,
Riyadh). Every unknown entity falls through to hash-bucket
collisions in the SPO graph. Grammar work proceeds without NER;
OSINT pipeline is blocked on this.

Cross-ref: `grammar-tiered-routing.md` §C8,
`STATUS_BOARD.md` Research threads section.

## 2026-04-19 — FP_WORDS = 157 (not 160); SIMD remainder loops remain
**Status:** Open
**Introduced by:** architectural choice (ndarray::hpc::vsa)
**Payoff estimate:** coordinated ndarray + lance-graph change,
~30 LOC in ndarray + 0 in lance-graph if field naming is stable

160 u64 = 10,240 bits (SIMD-clean for AVX-512 / AVX2 / NEON), zero
remainder loop in every SIMD pass. Current 157 u64 has 5-word
scalar tail. Performance delta is measurable but not critical yet.

Cross-ref: `cross-repo-harvest-2026-04-19.md` H6.

## 2026-04-19 — Abduction-threshold for unbundle-to-graph is hand-picked (0.88)
**Status:** Open
**Introduced by:** #208 (inherits PR design)
**Payoff estimate:** empirical calibration run on real corpus +
~30 LOC threshold parameterization

NARS Abduction confidence threshold for promoting facts into the
triplet graph is hand-picked at 0.88. Miscalibration on a specific
corpus (e.g. Animal Farm) would compound errors. Calibration is
pending D10 validation harness.

Cross-ref: `integration-plan-grammar-crystal-arigraph.md` E8,
`STATUS_BOARD.md` D10 row.

---

## Paid Debt

(No debt paid at initial commit. When an Open entry is retired,
APPEND here with same title + PR anchor.)

```
## YYYY-MM-DD — <same title as Open entry> (from YYYY-MM-DD)
**Status:** Paid YYYY-MM-DD
**Payoff:** PR #NNN (commit SHA) — <one-line description>

<verbatim original Open paragraph>

Cross-ref: <same + PR link>
```

---

## How to use this file

**When shipping with a known shortcut** — prepend to **Open Debt**
with `**Status:** Open` + `**Introduced by:** PR #NNN` +
`**Payoff estimate:**`. One paragraph describing what's owed.

**When paying debt** — append to **Paid Debt** with the same title
+ date anchor + `**Status:** Paid YYYY-MM-DD` + `**Payoff:** PR
#NNN`. Flip the Open entry's Status to `Paid YYYY-MM-DD`.

**When debt becomes irrelevant** (e.g. the feature it blocked got
abandoned) — flip Open Status to `Moot YYYY-MM-DD`. Keep the row.

Nothing is lost. Every shortcut has a trail from introduction to
payoff (or abandonment).

## 2026-04-19 — VSA substrate renaming: Vsa10k* → Vsa16k* + float framing
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @truth-architect D6 D0 domain:vsa domain:codec
**Introduced by:** architectural choice (all PRs referring to "10K VSA")
**Payoff estimate:** 1 contract-crate PR + 1 audit-sweep PR across
~28 `.claude/*.md` files (21 lance-graph + 7 ndarray).

The VSA substrate is misnamed and mis-scaled throughout the workspace:

1. **Naming:** `Vsa10kF32`, `Vsa10kI8` in
   `lance-graph-contract::crystal::CrystalFingerprint` should be
   `Vsa16kF32`, `Vsa16kI8`.
2. **Scale:** 10,000-D is a legacy narrower width; move to 16,384-D
   (64 KB lossless f32, 32 KB BF16, 80 KB u8-5-lane, 160 KB BF16-5-lane).
3. **Role-key slices:** `grammar::role_keys` addresses `[0..10000)`;
   re-scale to `[0..16384)` proportionally, keeping slices disjoint.
4. **Framing:** eliminate every occurrence of "10,000 binary VSA" — it
   collapses the 2 KB Hamming fingerprint (Container) with the
   ≥64 KB float VSA substrate. They are different objects.

Cross-ref: `.claude/board/IDEAS.md` CORRECTION-OF entry (2026-04-19).
Audit list of 28 files affected: `grep -rn "10000\|10,000\|Vsa10k\|10 000-D" .claude/`.

## 2026-04-19 — Ladybug 10000-D VSA import caused 700-1100 MB memory blowup
**Status:** Open
**Priority:** P1
**Scope:** @container-architect @integration-lead domain:vsa domain:memory
**Introduced by:** PRs #200-#203 (ladybug-rs / bighorn imports —
cognitive crate + CognitiveShader + BindSpace + adaptive codecs)
**Payoff estimate:** LanceDB zero-copy mmap migration + sparse/lazy
materialization audit + VSA scale decision (10k → 16k adds ~60% per
row: 40 KB → 64 KB at f32, worse memory not better).

Observed (per user, 2026-04-19): importing ladybug-rs at 10,000-D VSA
pushed runtime memory to **700-1,100 MB**. Arithmetic: at 40 KB/row
(Vsa10kF32), this is ~17-27 K live fingerprints in RAM
simultaneously. Pathologies:

1. **Migrating to 16,384-D makes it worse** — 64 KB/row × same
   population = 1.1-1.8 GB. 5-lane encoding (80 KB u8, 160 KB BF16) is
   worse still. Wider substrate without a storage-layer fix inflates
   the problem.

2. **Storage-layer fix is mandatory** — LanceDB `FixedSizeList<Float32,
   N>` natively supports mmap zero-copy. Hot rows stay OS-cached; cold
   rows pay page-fault but not RAM. Target runtime memory: ≤ 100 MB
   working set regardless of row count.

3. **Population audit** — before the 16k rename, audit which consumers
   keep VSA fingerprints live in RAM vs fetch-on-demand. Working set
   should be bounded by cache policy, not row count. Candidates for
   fetch-on-demand: Markov ±5 trajectory (rarely revisited), AriGraph
   episodic (LRU-evictable), VSA table snapshots.

4. **Sparse representation gap** — most VSA role-bundles have a few
   active slices at a time. A sparse encoding (indices-of-nonzero or
   Structured5x5 middle cells only) could drop per-row to a few KB for
   the common case. Only the "full field" queries need the dense
   f32 representation.

**Gate before 16k rename PR:** measure peak RAM on a representative
workload (Animal Farm D10 corpus when it lands) with the current 10k
substrate. If the fix is "mmap only hot rows", the substrate switch
is safe. If the fix requires sparse representation, the rename needs
to redesign the storage contract.

**Cross-ref:** IDEAS CORRECTION-OF + REFINEMENT-OF entries
(2026-04-19). PRs #200-#203 introduced the memory footprint.

## 2026-04-19 — CORRECTION-OF 2026-04-19 Ladybug 700-1100 MB blowup — it's a 10k × 10k GLITCH MATRIX, not HDC population
**Status:** Open
**Priority:** P0
**Scope:** @container-architect @integration-lead domain:memory domain:cleanup
**Introduced by:** PRs #200-#203 (ladybug-rs / bighorn imports — carried
over code that allocates a dense 10,000 × 10,000 structure)
**Payoff estimate:** identify + delete the single glitch allocation.
No migration, no redesign, no substrate change.

The prior entry framed the 700-1,100 MB blowup as a per-row HDC
population cost (17-27 K live fingerprints at 40 KB/row). **This was
wrong.** Per user (2026-04-19):

**There is no 10,000 × 10,000 matrix we actually want.** The memory
blowup came from a dense 10k × 10k structure imported as a glitch
from outdated ladybug-rs / bighorn code. Arithmetic:

- 10,000 × 10,000 × f32 = **400 MB** (single allocation)
- Plus cognitive-stack state + other working memory → 700-1,100 MB.

**Revised fix:**

1. **Identify the glitch** — grep ladybug-imported modules (cognitive
   crate, CognitiveShader, BindSpace, CollapseGate, adaptive codecs)
   for any dense `[[T; 10000]; 10000]`, `Vec<Vec<T>>` of that shape,
   `FixedSizeList<T, 100000000>`, or 400 MB-scale buffer. High-probability
   candidates: a token-token distance matrix, co-occurrence matrix, dense
   attention matrix, or K=10000 CLAM centroid distance table that was
   imported intact without trimming to the workspace's actual scale.
2. **Delete it.** It has no consumer in lance-graph proper (grammar /
   crystal / quantum use 1-D 10k-width HDC vectors, never square
   matrices).
3. **Verify** with peak-RAM measurement on a minimal workload.

**Invalidates:**

- Prior "16k rename makes memory worse" analysis — the per-row HDC
  math was sound but irrelevant to this specific blowup.
- Mmap zero-copy requirement for HDC population — still good hygiene
  but not the fix for the 700-1,100 MB observation.
- Sparse encoding as a Structured5x5-alternative — still architecturally
  useful but unrelated to the glitch.

The 16k rename + f32 → BF16 migration proceed independently of this
debt item. This one is a one-shot deletion.

## 2026-04-19 — SUPERSEDES all "stoneage import" / ladybug-refactor entries — retire ladybug-rs entirely
**Status:** CLOSED (via architectural decision)
**Scope:** @integration-lead @workspace-primer domain:architecture
**Decision:** per user (2026-04-19), ladybug-rs is retired. Migration
target is **ada-rs + lance-graph exclusively**. Ladybug-rs becomes
read-only historical reference; no maintenance, no refactor, no
integration obligation. Harvest-only.

**Consequences (ledger cleanup):**

- "Ladybug 700-1100 MB memory blowup" (glitch matrix): the matrix
  lives in ladybug-import code that is itself scheduled for removal.
  Deletion still P0 **only if** it ends up compiled into the current
  ada-rs / lance-graph binaries; otherwise it goes away with the
  import archival. Downgrade to P2, gate on "does cargo tree show the
  ladybug dep?"
- "Vsa10k → Vsa16k rename sweep" (2026-04-19): scope tightens to
  **ada-rs + lance-graph only**. Ladybug's internal types don't get
  renamed — they get left alone in the archive.
- "Ladybug import refactor resistance" table: obsolete. Don't refactor
  ladybug code; if a pattern is useful (PhaseTag, BindSpace,
  CognitiveShader, adaptive codec), reimplement cleanly in the target
  repo against canonical `Fingerprint<256>`.
- `CLAUDE.md` architecture diagram: "ladybug-rs = The Brain" line is
  stale; ada-rs + lance-graph now carry both the Brain and the Spine.
  Update in a follow-up PR.

**Repos in the canonical stack (post-retirement):**

```
ndarray            = The Foundation  (SIMD, GEMM, HPC, Fingerprint<256>, CAM-PQ)
lance-graph        = The Spine       (query + codec + semantics + contracts)
ada-rs             = The Brain       (BindSpace, SPO server, 4096 surface, cognitive shader)
crewai-rust        = The Agents      (agent orchestration, thinking styles)
n8n-rs             = The Orchestrator (workflow DAG, step routing)
```

(Was 5 repos; still 5, with ada-rs taking ladybug-rs's Brain slot.)

**Cross-ref:** prior entries "Ladybug 700-1100 MB memory blowup",
"Vsa10k* → Vsa16k* rename sweep", "CORRECTION-OF ... 10k × 10k GLITCH
MATRIX" — all carry this decision as their closing context.
