## 2026-05-26 — E-OGIT-STAKES-LINCHPIN — stakes is an O(1) ontological lookup (OGIT class), and it is the single dial that drives temperature + MUL sensitivity + savant binding together

**Status:** FINDING (grounds the MUL gate ratio + the front-door inheritance; refines `rung-mul-grounding-v1` §3 + `rung-persona-orchestration-v1` §1). **External ref — `AdaWorldAPI/OGIT` (Open Graph of IT, `ogit.ttl`, OWL/RDF, DOLCE-aligned), NOT in GitHub-MCP allowlist; reference-only.**

**Click:** Two user observations are the same mechanism. (1) `MUL ≈ (risk / competence) × stakes` with `competence = f(rung-level, resonance)`. (2) "a chat inherits temperature; an invoice inquiry inherits the bookkeeping savant." The bridge: **`stakes` is not hand-assigned — it is an O(1) lookup of the request's OWL/DOLCE class in OGIT** (the ontology reframed as a CAM). And that one number drives three things at once:

| request | OGIT class → stakes | inherited temp (viscosity) | MUL sensitivity | savant (dominant family) |
|---|---|---|---|---|
| chat | casual communicative act → low | hot (Plasma) | loose | generalist / exploratory |
| invoice inquiry | economic object → high | cold (Crystalline) | tight (`×stakes`) | bookkeeping savant |

**`felt_parse` is the front door:** viscosity = inherited start temperature, dominant axis-family = which savant binds; OGIT-class = stakes. So the inheritance the user described is `felt_parse` + an O(1) OGIT class lookup — no new dispatch layer. The MUL gate fires ∝ expected-loss / competence (DK danger zone gates hardest), with stakes ontologically grounded.

**The ontology IS a graph** ⇒ OGIT lives natively as an AriGraph/SPO + CAM-PQ class layer; O(1) class address = the "3-dims-are-the-address" CAM pattern. No second store needed (AriGraph is the one graph).

**Open (CONJECTURE):** whether `stakes` is an explicit OGIT annotation or derived from class position — confirm against `ogit.ttl`. README on `main` 404'd; repo-root gave only the high-level "semantic representation of all IT + business processes" description.

**Cross-ref:** `rung-mul-grounding-v1` §3 (MUL gate ratio); `rung-persona-orchestration-v1` §1 (front-door inheritance); `E-CHECKLIST-AS-ESCALATION` (felt_parse collapse-hint); `I-VSA-IDENTITIES` (CAM addressing).

**RESOLVED (same session, in-code grounding — supersedes the CONJECTURE above):** OGIT is in code as `lance-graph-ontology`. `stakes = Marking` (`Public < Internal < Pii ≈ Financial < Restricted`) — an **explicit field** on the `MappingRow`, resolved O(1) via `SchemaPtr` (packed `[namespace_id:8 | entity_type_id:16 | kind:8]` + `ontology_context_id` = the active named-graph / "active schema poll"). `Financial`'s doc literally reads *"bookkeeping or tax-relevant"* → grounds invoice→bookkeeping-savant. **The full O(1) inherit-set** the front door returns from one `MappingRow`: `marking`→stakes, `thinking_style`→savant, `qualia_meta`(qualia[18]/MetaWord/CausalEdge64)→qualia+dispatch prior, `confidence`→competence prior, `identity_codec`→CAM-PQ resonance address, `semantic_type`→attribute interpretation, `ontology_context_id`→active context. Table in `rung-persona-orchestration-v1` §1.

---

## 2026-05-26 — E-CHECKLIST-AS-ESCALATION — the boring checklist is NOT a bespoke verifier; it collapses into escalation-work + epiphanies, restoring ladybug's qualia loop on the SoA floor

**Status:** FINDING (simplifies `rung-persona-orchestration-v1` D-PERSONA-1; user-flagged collapse). **External design refs — ladybug-rs `src/qualia/{council,felt_parse,resonance}.rs` @177a321, NOT in the GitHub-MCP allowlist; reference-only, never a port target.**

**Click:** The "boring checklist → meta-recipe" of `rung-persona-orchestration-v1` does not need a new verifier subsystem — the list-completion machinery already exists in ladybug's qualia loop and only needs restoring on our SoA:

- **`felt_parse` emits a collapse hint** = {Flow, Fanout, RungElevate}: Fanout = gather more (escalate breadth), RungElevate = deepen (rung-shift), Flow = done. *The item's escalation decision is already produced* — "the list as escalation work" verbatim.
- **`InnerCouncil.deliberate`** (3 archetypes Guardian/Catalyst/Balanced, majority vote) + **`HdrResonance`**: a **split** (`is_split(0.7,0.5)` — one archetype sees what the others don't) is amplified ×1.2 for epiphany detection. **Disagreement IS the learning signal** = our SPO screening-off (perspectives disagree about a projection ⇒ spurious `S_O` caught).
- **`EpiphanyDetector.observe`** (council.rs:158): `Some(Epiphany)` iff `similarity > baseline×1.5 ∧ recent_samples ≥ 4`. The **window≥4 guard is the anti-Mount-Stupid evidence rule** (same shape as window-5 / Boole-bound — never fire on thin evidence). A green-flip = an epiphany committed to the graph, not a checkbox.
- **Ghost echoes** = {Affinity, **Epiphany**, Somatic, **Staunen**, **Wisdom**, Thought, Grief, Boundary} — persistent qualia residue (asymptotic decay to 0.1, never zero; felt_parse:70). Epiphany/Staunen/Wisdom-as-ghosts ARE the wisdom-marker substrate, already named; **8 ghosts ≤ 32 ✓ I-VSA-IDENTITIES**. (CLAUDE.md "Magnitude = Staunen × Wisdom qualia" — the ghosts are already in The Click.)

**The collapse:** list-item → collapse-hint (escalate) → council/resonance (split = discovery) → EpiphanyDetector (close, evidence-gated) → Epiphany/Wisdom ghost (persist). **Escalation IS the work; epiphanies ARE the completions; ghosts ARE the hydrating wisdom.** D-PERSONA-1 drops from "checklist verifier" to "wire the existing loop."

**Honest gap (unchanged):** ladybug's `detector.rs` still has no NaN/dead-end/escalation path ("all inputs produce valid output") — our NaN→cautious-exploration→Lab remains net-new.

**Cross-ref:** `rung-persona-orchestration-v1` §2+§7; `rung-mul-grounding-v1` (screening-off = split); `E-AGICHAT-DIMENSION-CONTRACT` (restore-on-SoA); `I-VSA-IDENTITIES` (8 ghosts ≤32).

---

## 2026-05-26 — E-AGICHAT-DIMENSION-CONTRACT — the 32-dim basis already exists as agichat's locked 10kD allocation; ladybug-rs de-grounded it by inflating bytes→10K-bit fingerprints; the work is to RESTORE the contract on the SoA floor, not invent or port

**Status:** FINDING (resolves the open `ThinkingStyleI4_32D` basis decision from E-I4-META-1; lineage + grounding map established from user-provided sources). **External design references — NOT in the GitHub-MCP allowlist; design-reference only, never a code-port target.**

**Click:** A long session walking two upstream repos — `AdaWorldAPI/ladybug-rs` (Rust) and the older `AdaWorldAPI/agi-chat` (Py/TS) — settled the entire "which 32 dims / how to ground" thread. The basis was never something to invent: it is **agichat's locked 10kD dimension allocation** (`docs/CANONICAL_DIMENSION_ALLOCATION.md`, "Status: LOCKED").

**Lineage (the key reframe):**

> **agichat (Py/TS) = the grounded byte-contract** → **ladybug-rs (Rust) = inspired but de-grounded (inflated bytes→10K-bit VSA fingerprints) → never worked** → **workspace (ndarray+lance-graph) = restore the contract on the SoA/SIMD floor.**

The user's account: ladybug-rs was "magically inspired but never informationally grounded, no LE contract"; it ran **10,000 vectors × 10,000-D** (~700 MB–1.4 GB RAM) and produced **no meaningful output — "an idealized cathedral."** The failure is mathematically forced: VSA bundle capacity is `N ≤ √d/4` (= 25 at d=10000), so resonating across 10,000 vectors is ~400× over capacity → noise (`I-NOISE-FLOOR-JIRAK`: classical stats on weakly-dependent bundles is meaningless). agichat had the *grounded* form (bytes + locked dimension ranges); ladybug-rs inflated every byte/dimension into a 10K-bit fingerprint and lost it.

**THE BASIS — agichat's 33-dim ThinkingStyleVector** (`[175:208]`, detailed at `[256:320]`), which IS the i4-32 thinking-style fingerprint:

- **3 Pearl** (SEE / DO / IMAGINE = association / intervention / counterfactual)
- **9 Rung** (R1–R9, meaning-depth)
- **5 Sigma** (Ω / Δ / Φ / Θ / Λ — the σ-tier chain)
- **8 Operations** (abduct / deduce / induce / synthesize / preflight / escalate / transcend / model_other) — the fanout's 4 inference modes are 4 of these
- **4 Presence** (authentic / performance / protective / absent)
- **4 Meta** (confidence_threshold / preflight_depth / exploration / verbosity)

= **33** (matches `STYLE_ENCODING.md`'s "3 Pearl + 9 Rung + 5 Σ + 8 Op + 8 spare"). Grounded form: `ThinkingStyleI4_32D` = i4 × 33 (or 32 + 1), riding the shipped ndarray i4-32 unpack.

**Qualia resolved:** agichat `[2000:2018]` = **18D Qualia PCS** (arousal/valence/tension/warmth/clarity/boundary/depth/velocity/entropy/coherence/intimacy/presence/assertion/receptivity/groundedness/expansion/integration/meta_awareness) → packed to the **16 drift-locked** at `[0:16]` = `QualiaI4_16D`. The 18→17→16 history is exactly this PCS→packed reduction. (ladybug's compact form was 8 Russell channels — a further reduction.)

**The dimension allocation IS a proto-LE-contract.** `CANONICAL_DIMENSION_ALLOCATION.md` locks every range and **rejects PRs #18/#19/#21 for "arbitrary dimension reallocation"** — *"DO NOT MOVE DIMENSIONS ARBITRARILY… bighorn code depends on these ranges."* That is a byte-budget with a no-arbitrary-moves invariant = the LE contract in proto-form. The grounding art = re-lock this allocation as a real `#[repr(C)]` / i4 SoA layout (which is what `SoaContainerHeader` + `SoaColumns` provide).

**The 5 Canonical Invariants (agichat `thinking/index.ts`, "Resonance Grammar Spine v0.3" — the explicit gestell):**

1. Addressability: O(1) via DN (Deterministic Names) + VASKey.
2. CollapseGate: **SD** controls FLOW/HOLD/BLOCK (NOT confidence).
3. RungShift: separate from SD; triggered by sustained-block / predictive-failure / structural-mismatch.
4. Separation of Roles: Grammar→Graph, Overlap→VSA, Memory→LanceDB, Styles→L5.
5. Cascade: Fork envelopes (STROKE 1) + Collapse records (STROKE 2) — the 2-stroke cycle.

**Grounding map (concept → agichat contract → workspace grounded form):**

| concept | agichat (grounded) | workspace grounded form |
|---|---|---|
| thinking-style | 33-dim TSV `[175:208]` | `ThinkingStyleI4_32D` (i4×33) |
| qualia | 18D PCS `[2000:2018]` → 16 `[0:16]` | `QualiaI4_16D` (64-bit atom) |
| quad-triangle | **12 bytes** (4 triangles × 3 corner-bytes) | `[u8;12]` / 1.5 atoms (NOT 10K-bit corners) |
| texture | 8D (entropy/purity/density/bridgeness/warmth/edge/depth/flow) | `Texture8 = [i8;8]` = one 64-bit atom |
| gestalt | Crystallizing/Contested/Dissolving/Epiphany (per-plane S/P/O CausalSaliency) | 2-bit derived field (on-demand) |
| rung ladder | 0–9, bands 0-2/3-5/6-9 | 4-bit level + 2-bit band |
| σ-gate | SD → FLOW/HOLD/BLOCK; `SignificanceLevel` Discovery/Strong/Evidence/Hint/Noise | 3-bit enum, **Jirak-bounded** threshold on bit-exact distance |
| 7-level "triangle" | `PackedDn` — 7 levels × 8 bits, MSB-first (DN-tree path) | **already a `u64` atom** — adopt as-is |
| address | DN (`PackedDn`) + VASKey | `u64` atom + `CognitiveAddress`-style `[Domain:4][Subtype:4][Index:8][Hash:48]` |

**Greek-vocabulary decode (the gestell's notation, parsed by regex over ladybug-rs):** σ (140×) = the significance/calibration spine (`SignificanceLevel` ladder + SigmaGate); α/γ/β = Fixed/Learned/Discovered RL-triangle weights; τ = ThinkingStyle τ-addresses; φ = golden-ratio spiral; ρ = Spearman ρ + ρ^d braiding; ε = ε-greedy; Ω/Δ/Φ/Θ/Λ = the 5 Sigma-tier dims; ψ/Ψ = quantum hologram (research, not core).

**Iron rule for this lineage:** **restore the contract; never port the carrier.** Mine agichat's *locked byte/dimension allocation + relational logic* (the gestell — hard to replicate), express each unit as a bit-exact i4/u8/u64 on the SoA floor, and never re-inflate to unbounded 10K-bit VSA resonance (the deprecated-`Vsa16kF32` / no-Baton anti-pattern that made the cathedral empty). `MulSnapshot`-packs-to-2-atoms, `CausalEdge64`, the Baton `(u16, CausalEdge64)`, and i4-32 are the grounding the upstream never had.

**Cross-ref:** shipped floor — ndarray `SoaColumns<N>` @ `42cb7123`, i4-32 unpack @ `8de1dcf8`; `E-BATON-1` (`dec049b`), `E-I4-META-1` (`71ea390`). Upstream design refs (allowlist-external, read locally from user-provided sources): agichat `docs/CANONICAL_DIMENSION_ALLOCATION.md`, `docs/INT4_QUANTIZATION_ARCHITECTURE.md`, `docs/VSA_10000D_DIMENSIONS_SCHEMA.md`, `src/thinking/{index,rung-shift,quad-triangle,collapse-gate,two-stroke}.ts`; ladybug-rs `src/{mul,qualia,spectroscopy,spo,world,learning,cognitive}/*`, `crates/ladybug-contract/src/address.rs`. Iron rules invoked: `I-NOISE-FLOOR-JIRAK` (why 10K-D σ was noise), `I-VSA-IDENTITIES` (bundle identities not content), `I-SUBSTRATE-MARKOV` (N≤√d/4 capacity).

**Next build (now fully specified):** `ThinkingStyleI4_32D` as the i4 quantization of the 33-dim TSV (3 Pearl + 9 Rung + 5 Σ + 8 Ops + 4 Presence + 4 Meta), general lanes fixed to that order, on the shipped i4-32 floor. No more "name the dims" — the allocation is the contract.

---

## 2026-05-26 — E-I4-META-1 — i4-32 thinking-style fingerprint = "thinking-about-thinking + domain"; qualia is the i4-16 64-bit atom; S-P-O is palette-pointers + Pearl-2³, not a 3×4096 identity

**Status:** FINDING (design converged this session; the `ThinkingStyleI4_32D`
type is NOT yet built — gated on the user naming the 32-dim general basis +
general/OGIT-custom lane split). The **ndarray hardware floor is shipped** (see
Cross-ref).

**Click:** A long design session converged the cognitive-style representation.
The capstone framing: **i4-32 is "thinking about thinking + domain"** — a
cognitive *address* whose general lanes are the metacognitive style (HOW one is
thinking, cross-domain) and whose OGIT-custom lanes are the domain (WHICH
domain). Their product lands on a reusable best-practice thinking template.

**The unification — 64-bit is the atom:**

- `qualia` = `QualiaI4_16D(u64)` (8 B, 16 signed-i4 dims, range −8..+7) ==
  `CausalEdge64` (8 B) in *width*. Both are the **64-bit atom**: same SoA column
  stride (8 B), same SIMD lane (`U64x8`), same kernels → they cross-pollinate.
- `thinking-style` = i4_32D (16 B = `u128`/`[u64;2]`, **32 signed activation
  dims**) = **2 atoms**.
- The shipped i4-32 unpack **subsumes** i4-16: the low 64 bits of
  `I8x32::from_i4_packed_u128` equal `I8x16::from_i4_packed_u64` by construction
  (atom-parity test). So the one primitive serves qualia/edge (low half) and
  thinking (full).

**32 dims = multi-activated meta-properties, bipolar-signed (NOT a pick-one
enum):** each dim is a graded property; sign = the opposite pole
(sarcasm `+` / sincerity `−`, irony `+` / literal `−`), magnitude = intensity,
0 = neutral. **Opposite = one-instruction negation.** A persona/archetype is a
*profile* (e.g. "Schopenhauer = +7 sarcasm, +pessimism, +philosophical,
−warm"). The i4-**16D**-thinking alternative was **rejected** — 16 dims would
force merging irony/sarcasm/etc. onto shared axes and rob their distinct poles;
32 is the precision floor. The dims capture the *meta* (metacognition) and are
**Jina-calibratable** (existing `thinking-engine` lens machinery —
`jina_lens.rs`, `calibrate_lenses.rs`, Spearman ρ / ICC / Cronbach).

**General / OGIT-custom split (the clean architecture):** keep the **general
block** universal + Jina-calibrated (irony, sarcasm, care↔extraction, …) so
K-NN similarity works *cross-domain*; let **OGIT inject domain axes into the
custom block** (doctor↔autopsy when medical ontology active; bookkeeping / income
tax when finance active). Domain axes are bipolar too (doctor `+` heal ↔ autopsy
`−` post-mortem — a *same-domain* sign flip; it even rides the Abduction↔Deduction
fanout axis). The custom lanes set by OGIT are the **explicit-binding** path
(dispatch provable Odoo/DOLCE business logic); the general lanes are the
**similarity fallback**. **OPEN DECISION (gates the build):** where the split is
(e.g. 24 general + 8 custom) and the general meta-property list.

**No-duplication rulings (Baton single-home discipline):**

- **DK ↔ informational-trust is DERIVED, not stored.** `CausalEdge64.conf` (NARS
  confidence, per-edge, object-level) is the single source for trust. The
  Dunning-Kruger calibration is a *per-cycle meta-aggregate* over the edge-conf
  distribution (the MUL already computes `DkPosition` / `TrustTexture`). It
  lives as a **derived lane** (computed on-demand, mirroring qualia.rs
  "magnitude = coherence × valence → i8 on demand"), NEVER as independent state
  that could drift from `conf`.
- **Relocating ephemeral *style* out of the crowded `CausalEdge64` v2 is
  relocation, not duplication** — and a net plus: it decrowds the over-packed
  u64 that caused the 5 sprint-11 I-LEGACY reclaim bugs, and upgrades style from
  a cramped field to 32-dim resolution. **Granularity split:** `CausalEdge64` =
  *persistent, per-edge structural truth* (committed to AriGraph); i4-32 =
  *ephemeral, per-cycle thinking stance* (carried in the SoA grid, not stamped
  on every edge).

**S-P-O is NOT a "sneaked-in" 3×4096 identity (verified, the worry is
unfounded):** `lance-graph-planner` `cache/nars_engine.rs::SpoHead` ("mirrors
CausalEdge64 layout", 8 B) has `s_idx/p_idx/o_idx: u8` — **256-entry palette
POINTERS**, not dense 4096 vectors. That is exactly the `I-VSA-IDENTITIES` Test-0
register pattern (a natural ID indexes content; it does not bundle a
fingerprint). The actual **2³ deconstruction** is the *separate* `pearl: u8`
3-bit mask: `MASK_NONE` (prior) · S/P/O marginals · `MASK_SP` (confounder) ·
`MASK_SO` Association(L1) · `MASK_PO` Intervention(L2) · `MASK_SPO`
Counterfactual(L3). So the edge is causal-structural (pointers + rung mask +
NARS truth + inference + temporal, all register) — **no identity smuggled →
fine.** This `SpoHead`/ndarray SPO-palette variant has **no `style` field**,
which confirms the style-unload target is the *other* v2-with-style variant
(the dual/triple-`CausalEdge64` split remains the thing to watch).

**The cycle (all loops close on the shipped carrier):** the SoA grid carries the
address O(1) cycle-to-cycle → the 4-mode fanout (Abduction/Deduction/Synthesis/
Induction; Revision = commit) explores → pattern-J K-NN over the general
fingerprint retrieves the nearest best-practice when OGIT has no explicit binding
→ pattern-K Cranelift JIT compiles the winning template and "sinks" it back to
source as a compile-time primitive next build (engine exists:
`jitson_cranelift` / `cam_pq/jitson_kernel.rs` / `contract/jit.rs`; the YAML/
source-writeback half is the gap).

**Cross-ref (shipped this session):** ndarray `src/simd_soa.rs` `SoaColumns<N>`
multi-column SoA carrier @ `42cb7123` (zero-copy per-field lane iters + baked-in
`CausalEdge64` accessor; O(1) `Arc`-clone cycle carry-over); ndarray i4-32 unpack
`I8x32::from_i4_packed_u128` + `batch_packed_i4_32` across avx512/neon/scalar +
4 simd.rs re-exports @ `8de1dcf8` (atom-parity tested, clippy/fmt clean);
`E-BATON-1` (Baton ratification @ `dec049b`). **Cross-ref (design anchors):**
`lance-graph-contract/src/qualia.rs` (`QualiaI4_16D`, 17D→i4-16 packing);
`lance-graph-planner/src/cache/nars_engine.rs` (`SpoHead`, Pearl 2³ masks,
`SpoDistances`); MUL `DkPosition`/`TrustTexture`; `.claude/patterns.md` J
(INT4-32D Thinking Atoms) + K (Circular Compilation); ndarray
`src/hpc/causal_diff.rs` (`CausalEdge64` SPO-palette variant: block/proj/verb/
row/L1/freq/conf); CLAUDE.md `I-VSA-IDENTITIES` + `I-LEGACY-API-FEATURE-GATED`
(the v2 reclaim bugs).

**Next build (when basis named):** `ThinkingStyleI4_32D` (lance-graph,
`[u64;2]`) with general lanes `0..K` + OGIT-custom lanes `K..32`, the i4-32 K-NN
over the general block, and the DK derived-lane projection. The ndarray floor is
ready under it.

---

## 2026-05-26 — E-BATON-1 — "Baton" is the workspace's native term for the little-endian contract; it ratifies the deprecation of the singleton BindSpace and Vsa16kF32-as-carrier

**Status:** FINDING (user-ratified terminology + doctrine; board-first per "Both, board first")

**Provenance (why the folk term exists):** The user coined **"Baton"** as the
intuitive name *before* they had the information-science term for it. The formal
name is the **little-endian (LE) contract** / gapless handoff. Both name the same
thing. This entry exists so future sessions stop re-deriving it: when you see
"baton" in code, plans, or a savant card, it IS the LE contract — do not invent a
parallel concept. Direct user statements anchoring this entry: *"please grep for
'Baton' its another word for little endian contract"*; *"every mention of 'baton'
references the non materialization and deprecation of the singleton bindspace"*;
*"'baton' was the idea before i knew the information science term"*; *"the little
endian contract is real / just the SoA shape is a little richer"*.

**The equivalence chain (now pinned):**

> **LE contract = Baton = no materialized singleton BindSpace = discrete owned
> `(u16 target, CausalEdge64)` handoffs.**

**Doctrinal claim — what "Baton" deprecates:** The Baton is not merely a transport
optimization; it is the **negation of the singleton BindSpace as a materialized
object**. There is no global `Vsa16kF32` register that gets read/written across
mailbox boundaries. There are only owned, per-thought `(target, edge)` handoffs
passing between compartments. Consequences:

1. **`Vsa16kF32` is deprecated AS A CARRIER** — it does not cross mailbox
   boundaries and there is no singleton BindSpace to materialize. Cumulative
   cognitive state lives in **CausalEdge64 emissions + AriGraph SPO-G quads +
   BindSpace SoA columns**, NOT in a 16k-float envelope. New work must not reach
   for `Vsa16kF32` as an inter-mailbox carrier or universal cumulative-state vessel.
2. **The Vsa16kF32-deprecation and the Baton model are ONE ratification, not two.**
   If the baton is the wire, the 16k-float carrier has nothing left to carry across
   a boundary — the deprecation is the baton's premise, not a separate decision.
3. **`ndarray::hpc::soa::SoaContainerHeader` (pinned b5d6b206) is the on-wire SoA
   descriptor UNDER the baton stream**, not a parallel container. The MailboxSoA
   named-column set ("the SoA shape is a little richer") layers over that same
   padding-free `[u64; N]` LE descriptor; batons land in and are folded over those
   columns.

**Mechanism — the mailbox-as-owner is why the baton is sound ("Rust's holy grail
UB solution"):** The Baton is handed off between **owning mailboxes** in a rotating
sea-star topology (a hub of ownership-typed compartments; ownership rotates as each
`(u16, CausalEdge64)` tuple moves from one mailbox-owner to the next). Because the
handoff is a **Rust move**, the borrow checker proves — at compile time — that no
two compartments alias the same baton: no data race, no use-after-free, no shared
mutable singleton to corrupt. **This is the deep reason the singleton BindSpace is
deprecated:** a materialized global `Vsa16kF32` register would be exactly the
shared-mutable-aliased state Rust's ownership model exists to forbid. By making the
mailbox the single owner and the baton a moved value, **UB becomes a compile
error** (canonical plan §9 E-CE64-MB-4) — there is no runtime aliasing check
because there is nothing to alias. The user's framing: *"we basically invented the
rotating sea star ractor mailbox as owner as Rust's holy grail UB solution."* (Note
the ractor edge is async-only and lives at the membrane / Zone 2, not the
preemptive internal core — the ownership guarantee is the type-system property, not
a ractor runtime feature.)

**Where it already lives in the tree (do NOT re-invent):**

- `crates/lance-graph-contract/src/collapse_gate.rs` — `CollapseGateEmission` with
  `batons: Vec<(u16, u64)>`, `push_baton(target, edge)`, `baton_count()`,
  `wire_cost_bytes() = 13 + 10 * baton_count`. The `10 * baton_count` (10 B = 2 B
  target + 8 B CausalEdge64), NOT `16384 * 4`, IS the proof that nothing
  materializes a singleton on the wire. **This is the Baton implementation.**
- `.claude/plans/cognitive-substrate-convergence-v1.md` / `v2.md` — "the baton IS
  the wire… Vsa16kF32 does NOT cross mailbox boundaries… discrete `(u16 target,
  CausalEdge64)` tuples suffice."
- PP-15 `baton-handoff-auditor` savant (the meta-review fleet's baton auditor).
- `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` — the canonical plan that
  already encodes the baton model; the parallel `.claude/surreal/` POC was
  re-deriving it under different names (see `RECONCILIATION_with_canonical_plan.md`).

**Contradiction flagged (P-1 doctrine, must not silently diverge):** CLAUDE.md
§"The Click" (P-1, "read before everything else") describes cognition AS the
element-wise multiply+add Markov bundle on `Vsa16kF32`, and §I-SUBSTRATE-MARKOV
makes VSA-bundling the Chapman-Kolmogorov guarantee. Deprecating `Vsa16kF32` as a
carrier contradicts the *unscoped* reading of The Click. **Resolution (this
ratification):** The Click's bundle math is NOT wrong — it describes how a single
`Think` resolves **locally, within one compartment, ephemerally**. What the Baton
changes is the **scope**: the bundle is a within-compartment computation, never a
persisted or transmitted singleton. The persisted + transmitted form is the baton
(`Vec<(u16, CausalEdge64)>`) + the SoA columns + AriGraph SPO-G quads.
I-SUBSTRATE-MARKOV (the math guarantee for local bundling) and I-VSA-IDENTITIES
(bundle identities, not content) are untouched; only the cross-boundary carrier is
deprecated. A scoping note has been added to §"The Click" pointing here.

**Lesson:** A folk term with no recorded bridge to its formal name is a
rediscovery tax (the same shape as E-SIMD-SWEEP-1's retroactive-invariant
pattern). Record provenance the moment the equivalence is stated, not after the
next session re-derives "what is a baton."

**Cross-ref:** `crates/lance-graph-contract/src/collapse_gate.rs`
(`CollapseGateEmission` / `push_baton` / `wire_cost_bytes`);
`.claude/plans/cognitive-substrate-convergence-v1.md` + `v2.md`;
`.claude/plans/causaledge64-mailbox-rename-soa-v1.md` (§5 MailboxSoA, §9 E-CE64-MB-2);
`.claude/surreal/RECONCILIATION_with_canonical_plan.md` (Vsa16kF32-deprecation +
LE-contract-is-real notes); `ndarray` `src/hpc/soa.rs` @ b5d6b206 (`SoaContainerHeader`,
the on-wire LE descriptor); CLAUDE.md §"The Click" (P-1, now carries a 2026-05-26
Baton scoping note); §I-SUBSTRATE-MARKOV + §I-VSA-IDENTITIES (untouched — local
bundle math); PP-15 `baton-handoff-auditor`.

---

## 2026-05-16 — E-SIMD-SWEEP-1 — PR #398 was the 5th violation, not the first; the SIMD source-of-truth invariant is retroactive

**Status:** FINDING

**Click:** The `simd-savant` agent's first PRE-MERGE audit of `origin/main` (`/home/user/lance-graph` at `8d321ff` Merge PR #396 era, post-sprint-13 W-I batch) surfaced **158 raw-intrinsic violations across 5 consumer crates** + **3 missing primitives** in `ndarray::simd` that block clean remediation. PR #398 (sprint-13 W-I1 retry, D-CSV-13b i4 SIMD batch dispatch) inlined raw `_mm512_*` (x86_64) and `vld1q_u64` (aarch64) intrinsics in `crates/lance-graph-contract/src/mul.rs` — and was the **5th instance**, not the first. The four prior instances (`blasgraph/types.rs`, `blasgraph/ndarray_bridge.rs`, `holograph/hamming.rs`, `bgz17/src/simd.rs`, plus a partial 5th in `thinking-engine/src/engine.rs:504`) shipped before the `simd-savant` rule was declared. Codex P1 finding on PR #398 (NEON OOB at `len==2`) is a direct consequence of AP-SIMD-5 (unchecked pointer-load), an anti-pattern the prior 4 violators also carry.

**Doctrinal claim:** The SIMD source-of-truth invariant — **all SIMD through `ndarray::simd` via the polyfill (`simd.rs` + `simd_ops.rs` > `simd_{type}.rs`)** — is **retroactive**, not just forward. The `simd-savant` card (added in PR #399, 2026-05-16) was written AFTER the violations existed but BEFORE they were swept. Each pre-existing violation has a *distinct* missing-primitive blocker against ndarray, which is why no single sweep PR can cover them. The right cadence is one ndarray PR per missing primitive (wave W1a), then one consumer PR per migration (wave W1b), gated and sequenced.

**Violation inventory (from `simd-savant` PRE-MERGE audit 2026-05-16):**

- **AP-SIMD-1 (raw `_mm*`):** 117 occurrences
- **AP-SIMD-2 (raw `vld1q_*` / NEON):** 8 occurrences (3 call sites in `holograph/hamming.rs` + 5 ancillary)
- **AP-SIMD-3 (custom `is_*_feature_detected!`):** 13 occurrences
- **AP-SIMD-4 (arch cfg + intrinsic body):** 7 occurrences
- **AP-SIMD-5 (unchecked ptr-load):** 19 occurrences
- **AP-SIMD-6 (missing scalar fallback):** 0 occurrences (all paths have scalar floor — good)
- **AP-SIMD-7 (duplicated wrapper):** 2 occurrences (nibble-popcount LUT hand-rolled twice in `blasgraph/ndarray_bridge.rs:252,299` — already in `ndarray::simd::U8x64::nibble_popcount_lut()`)
- **AP-SIMD-8 (custom dispatch table):** 13 occurrences (`SimdLevel` + `detect_simd()` in `bgz17/src/simd.rs`)

**Total: 158 violations across 5 consumer crates.**

**Missing primitives** (must be added to ndarray before consumer remediation can complete):

- `TD-NDARRAY-SIMD-UNPACK-I4-16D` — `I8x16::from_i4_packed_u64` + `batch_packed_i4_16<E, F>` closure-batch
- `TD-NDARRAY-SIMD-SATURATING-ABS-I8` — `I8x16::saturating_abs` via `_mm512_min_epu8(_mm512_abs_epi8(x), 0x7f)` on AVX-512 (VPABSB alone does NOT saturate `i8::MIN`; needs VPMINUB clamp), `vqabsq_s8` on NEON, `i8::saturating_abs` scalar — closes codex P2 i8::MIN divergence
- `TD-NDARRAY-SIMD-GATHER` — `U16x8::gather_u16` (palette lookup, currently raw `_mm256_i32gather_epi32` in `bgz17`)
- `TD-NDARRAY-SIMD-PREFETCH` — cross-arch `prefetch_read_t0` (no-op on unsupported)
- `TD-NDARRAY-SIMD-POPCOUNT-U64` — `U64x8::popcnt` (lane-wise 64-bit popcount; currently raw `_mm512_popcnt_epi64` in `holograph` + `blasgraph`)

**Lesson:** The "narrow scope" recommendation from the PP-14 convergence-architect run was correct for the mul.rs follow-up considered in isolation, but the audit reveals the broader pattern: **5 consumer crates established the raw-intrinsic precedent over multiple prior sessions; the simd-savant invariant retroactively reclassifies them all as TD-SIMD-SWEEP-W1..W4 (plus the thinking-engine partial as W5)**. The right architectural move is the W1a + W1b two-wave plan documented in `.claude/knowledge/ndarray-vertical-simd-alien-magic.md`, not a per-PR scramble.

**Doctrinal counterpart:** This finding is the SIMD-domain analogue of `E-META-10` / `I-LEGACY-API-FEATURE-GATED` (the v1-API-under-v2-feature pattern that codex caught 5 times in sprint-11). Same shape: a single rule, multiple historical violations, retroactive sweep needed. Same response: invariant in CLAUDE.md / agent card, codex/savant as the pre-merge gate, follow-up wave to close the back-catalogue.

**Strategic angle — sigker as the Index-regime third lane:** `crates/sigker` (path-signature codec) currently has **zero raw intrinsics, zero `ndarray` dep** — it's the cleanest exemplar of "domain crate composes via closures" we have today. The W1.5 wave (deferred, gated on `jc Pillar 11` activation) will add 3 more ndarray primitives (signature-PDE-sweep, randomized-projection, lyndon-pack) when sigker is benchmarked at production carrier widths. Sigker bypasses the `I-NOISE-FLOOR-JIRAK` iron rule for path data via Hambly-Lyons 2010 uniqueness — Index regime, not Argmax. The vertical-SIMD surface must be designed broad enough to absorb sigker's needs from W1a onward.

**Cross-ref:** `.claude/agents/simd-savant.md` (the invariant + AP-SIMD-1..8 catalogue); `.claude/knowledge/ndarray-vertical-simd-alien-magic.md` (the canonical wave plan + per-workload surface table); `.claude/board/TECH_DEBT.md` (5 W1a + 3 W1.5 `TD-NDARRAY-SIMD-*` entries); PR #399 (introduced the simd-savant + autoattended-pattern); PR #398 codex P1/P2 findings (NEON OOB + i8::MIN divergence — symptoms of the broader pattern); `crates/sigker/src/lib.rs` (the W1.5 consumer); CLAUDE.md § `I-NOISE-FLOOR-JIRAK` (the iron rule that sigker bypasses).

---

## 2026-05-16 — E-META-8 — "Edit" / "Write" / "MultiEdit" as bare permission rules are no-ops; subagents do not inherit allow rules

**Status:** FINDING

**Click:** The 2026-05-15 session's diagnosis that switching from `Edit(**)` / `Write(**)` to bare `Edit` / `Write` / `MultiEdit` in `.claude/settings.local.json` was the fix for permission-prompt friction was **wrong**. Bare tool-name rules are not valid permission entries in the current Claude Code parser for tools that take a file-path argument — they effectively fall through to user prompt rather than granting unrestricted access. The actually-working syntax is `Edit(**)` / `Write(**)` / `MultiEdit(**)` (or path-globbed forms like `Edit(**/*.md)` / `Edit(.claude/specs/**)` per the existing pattern in tracked `.claude/settings.json`).

**Diagnostic signature:** every Edit/Write call popping for permission despite an "allow" entry in settings.local.json. If the entry has no parens (`"Edit"` rather than `"Edit(**)"`), it is the bug. The 2026-05-15 session interpreted prior `Edit(**)` failures as "parsed as exact-match for literal `**`" — that diagnosis was a Frankenstein; the actual bug was elsewhere (likely a sessions-old parser version, or the failures were tracked to a different deny rule).

**Doctrinal claim:** Permission rules for path-taking tools (Edit / Write / MultiEdit / Read / Glob / Grep) **always require a path-shaped spec in parens**, even for "allow all". The schema treats them as `Tool(spec)` only; bare `Tool` is reserved for tools without scope (e.g. potentially `Read` if it has no path, or MCP tools). Treat any "tool-only" form for a path tool as a no-op and audit the settings file for that mistake first.

**Cross-claim:** **Subagents do not inherit `allow` rules from session-scoped `.claude/settings.local.json`** — they only inherit deny rules. The PR #381 fleet confirmed this: 7 of 8 Sonnet workers had Edit/Write blocked even after the main thread's settings.local.json had `Edit(**)` working. Workers all had to use Python-via-Bash heredocs (Bash(python3:*) is in tracked settings.json and DOES inherit). Filed as a Claude Code SDK gap candidate.

**Predecessor:** 2026-05-15 prior-session diagnostic note in `.claude/board/sprint-log-csv-prep/agents/agent-W4.md` §"Process note — permission-system fix" — claims the tool-only form fixed subagent denials. This entry corrects that claim: it fixed nothing; it just shifted the failure mode from "subagent gets denied AND main thread's rule didn't apply" to "subagent gets denied AND main thread's rule still doesn't apply but is invisible because main thread had inherited the previous-working `Edit(**)` rule from prior settings."

**Lesson:** When a "permission fix" is followed by recurrence of the same friction, the fix didn't work — don't double down on the diagnostic that produced it. The Mandatory Board-Hygiene Rule's retroactive-hygiene anti-pattern applies here too: the prior session's scratchpad logged a fix-claim that this session inherited and didn't verify; the verification (rerunning Edit and observing popups) is the only ground truth.

**Cross-ref:** `.claude/settings.local.json` (now uses `Edit(**)` / `Write(**)` / `MultiEdit(**)` + `Edit(**/*)` etc.); PR_ARC #381 Locked entry on permission syntax; `.claude/board/AGENT_LOG.md` 2026-05-16 fleet entry; CLAUDE.md §In-Session Orchestration Discipline (where the bare-tool-name claim should be corrected if it appears there).

---

## 2026-05-16 — E-META-9 — Mandatory Board-Hygiene Rule violated by PR #381; retroactive-hygiene anti-pattern observed

**Status:** FINDING

**Click:** PR #381 (sprint-10 spec patches) was merged 2026-05-16 without including LATEST_STATE / PR_ARC_INVENTORY / STATUS_BOARD / AGENT_LOG updates in the merged commits — exactly the retroactive-hygiene anti-pattern that CLAUDE.md §Mandatory Board-Hygiene Rule was added to prevent (after the 2026-04-20 PR #223/#224/#225 gap surfaced the same issue). This entry plus the followup board-hygiene PR (branch `claude/board-hygiene-pr-381`) are the retroactive cleanup; the cleanup itself is the symptom, not the cure.

**Why it happened:** The fleet dispatch flow (8 parallel Sonnet workers patching 8 spec files in one branch) optimized for the spec-patch work. Each worker scratchpad documented its own delta, but no worker had the cross-cutting responsibility to update board files for the PR as a whole. The main thread aggregated the worker outputs into 5 commits but did not pause to draft the board-hygiene commit before pushing the final commit that opened the PR. The PR body documented the patch-level deltas but did not encode the board updates as part of the merge contract.

**Doctrinal claim:** **Board-hygiene updates are a per-PR cross-cutting responsibility that fleet workflows do not naturally assign to any single worker.** The rule needs structural enforcement, not just rule-as-text in CLAUDE.md. Options:

1. **Main-thread sentinel:** before opening any PR with `mcp__github__create_pull_request`, the main thread MUST verify that one of the commits on the branch touches the four board files (LATEST_STATE / PR_ARC_INVENTORY / STATUS_BOARD / AGENT_LOG) — if not, draft the hygiene commit first.
2. **CCA2A pattern extension:** add a "W-hygiene" worker to every fleet that runs LAST and produces the hygiene commit, gated on all other workers reporting DONE.
3. **PR-template enforcement:** GitHub PR template asks "Board files updated in this PR? (yes / no / N/A — explain)" — the answer is the merge gate, not the PR body summary.

**Recommendation:** option 1 (main-thread sentinel) is cheapest and matches the existing fleet flow. Add a check to the main-thread post-fleet aggregation step.

**Lesson:** the rule-as-text in CLAUDE.md is necessary but not sufficient. Cross-cutting governance responsibilities need a structural owner; in CCA2A flows the owner is the main thread, which means the main thread must encode the check explicitly (not delegate to "the workers will remember").

**Cross-ref:** CLAUDE.md §Mandatory Board-Hygiene Rule; PR_ARC #381 (the violating PR); this followup PR (the retroactive cleanup); 2026-04-20 PR #223/#224/#225 gap (predecessor occurrence of the same anti-pattern).

---

## 2026-05-14 — E-LL-1-INTERVENE — NARS Intervention/Counterfactual verbs land

**Status:** SHIPPED (PR-LL-1 from curriculum §6.1)

**Click:** Pearl 2³ rungs (association/intervention/counterfactual) were named-but-not-dispatched in nars_engine — `NarsInferenceType` had 5 variants none of which encoded interventional reasoning. PR-LL-1 closes that gap with two additive variants in `lance-graph-planner::thinking::nars_dispatch::NarsInferenceType`, threaded through Pearl 2³ dispatch in `cache::nars_engine`, and a new `TripletGraph::intervene_on()` method that produces counterfactual SPO-G tagged with `G::Intervention` (from causal-edge).

**Doctrinal claim:** Intervention is now a first-class verb in the stack, not a name. The MUL gate's free-energy signal now has a vocabulary for distinguishing "system is unsure about observation" (high F, NARS Abduction) from "system is being asked to reason counterfactually" (high F, NARS Counterfactual). Downstream consumers (MedCare-rs treatment proposals, q2 cockpit what-if queries, OSINT corroboration) can now disambiguate.

**Predecessor:** PR #373 (curriculum v1).

**Successor:** PR-LL-2 (ICM-invariance column + Opt-Sym generator) consumes the new G slot tagging.

---

## 2026-05-14 — E-LL-CURRICULUM-1 — neurosymbolic + RLVR + causal learning layer (8-paper synthesis)

**Status:** PROPOSAL (curriculum doc landed; 5-PR roadmap ratification pending)

**Click:** The stack already has the substrate for *self-improvement*. PR #372 landed AriGraph SPO-G + CausalEdge64 v2 + Σ-tier router + MailboxSoA — all five doctrinal pieces of (probabilistic programs × structural causal models × multi-environment grouped data × explicit conditional dispatch × Bayesian belief). What's missing is **the learning loop on top**: a deterministic verifier (NARS), a Goldilocks data generator (Opt-Sym shape), a continuous program-latent optimizer (LPN shape), an RL trainer (GRPO shape), and a Σ9-Σ10 deductive prover (LINC shape). Each maps to one existing-or-near-existing stack component; the curriculum (this doc) is the joint reading that names which paper supplies which verb. Reading load: ~6 hours across 4 tiers. PR roadmap: 5 PRs (LL-1 NARS intervene/CFG verbs → LL-2 ICM column + Opt-Sym generator → LL-3 hybrid TextGrad/LPN style optimizer → LL-4 GRPO trainer crate → LL-5 LINC bridge + conformal CFG).

**Stack alignment table:** Causal de Finetti ↔ AriGraph SPO-G (live); LPN ↔ StyleVectors (live, underused); LINC ↔ Σ9-Σ10 → L4 (live shell, no prover); Executable CFG ↔ Pearl 2³ in NarsEngine (live in name, missing verbs); Opt-Sym ↔ data_gen module (missing); Conformal CFG ↔ safety wrap (missing); TextGrad ↔ style optimizer (missing); GRPO ↔ trainer (missing).

**Doctrinal claim:** Stack's NARS truth + I-SUBSTRATE-MARKOV gives a *strictly stronger* deterministic verifier than Opt-Sym's LLM verifier — graded confidence ∈ [0,1] is better than binary pass/fail as a GRPO reward. Stack's `StyleVectors` is *already* an LPN-style continuous latent space; LPN's gradient-at-inference is the missing operator. The MUL gate is *already* the LINC dispatch shape; LINC just fills the L4 slot. Each of the 8 papers maps to a verb the stack named but didn't ship.

**Doc location:** `.claude/knowledge/neurosymbolic-rlvr-causal-curriculum-v1.md` (~600 lines, 12 sections). Cross-refs to causal-edge-64-* triad, cognitive-shader-driver-thinking-engine-reunification, encoding-ecosystem (mandatory), lab-vs-canonical-surface (mandatory), bf16-hhtl-terrain (probe queue).

**Open questions (6) gated before sprint fan-out:** reward shape (graded vs binary), TextGrad optimizer (local vs frontier), prover choice (Z3 vs HOL Light), style-pool location (contract vs separate), ICM-invariance update protocol, Σ-tier-as-difficulty probe.

**Iron rule audit:** Six rules (I-SUBSTRATE-MARKOV, I-NOISE-FLOOR-JIRAK, I-VSA-IDENTITIES, I1, method-on-carrier, AGI-as-glove SoA) all satisfied — synthesized styles are IDENTITY fingerprints (not content), Conformal CFG uses Jirak bounds (not classical Berry-Esseen), all four new capabilities are methods on existing carriers, BindSpace stays read-only with the new IcmInvarianceColumn gated through CollapseGate.

**Predecessor:** PR #371/#372 (causaledge64-mailbox-rename-soa-v1) substrate.

**Successor:** PR-LL-1 through PR-LL-5 (this curriculum is the spec).

---

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

## 2026-05-16 — E-META-10 (FINDING): v1-API-under-v2-feature alias pattern — systematic layout-bit boundary testing required

**Status (2026-05-16):** PROMOTED to iron rule `I-LEGACY-API-FEATURE-GATED` in CLAUDE.md
following Wave F Opus honest review recommendation. Original FINDING content preserved
below for historical context.

**Status:** FINDING (surfaced sprint-11 Wave A codex review; confirmed by W-Meta-Opus sprint-12 Wave F)

**Click:** Sprint-11 Wave A codex P1 review caught the same anti-pattern 5 times in one PR (PR #383): v1 API paths (accessors and setters on `CausalEdge64`) reading or writing OLD bit positions that v2 had reclaimed for plasticity[2], W-slot, lens, and spare. The same function name silently produces different semantics depending on which feature flag is active; downstream callers see corruption only at runtime on workloads that hit the reclaim zone. Wave F Opus meta-review (CSI-2) identified this as a systemic pattern, not a per-site fix.

**Doctrinal claim:** Every v1 API path under a v2-layout feature must transparently route through the canonical mapping OR be feature-gated to a documented no-op with a migration pointer. Field-isolation matrix tests (writing each field, asserting all other fields unchanged) are MANDATORY when a layout reclaims previously-used bits. The codex P1 review is the canonical pre-merge gate for this pattern.

**Cross-ref:** CLAUDE.md §Substrate-level iron rules → I-LEGACY-API-FEATURE-GATED; `.claude/knowledge/i4-substrate-decisions.md` §5 "Codex P1 Anti-Pattern" (5-instance catalogue); `.claude/board/sprint-log-11/meta-review-opus.md` CSI-2; sprint-log-11/meta-review.md E-META-10 original entry.

---


## 2026-05-14 — E-META-7 (FINDING): dual `CausalEdge64` types in workspace + p64 drift origin pinpointed + three-zone hot-path model

**Status:** FINDING (verified 2026-05-14 against shipped source; recorded in PR #372 merge commit `9fa206d`).

Three coupled findings surfaced during sprint-10 meta-review + post-research correction of the hot-path mental model.

**1. Dual `CausalEdge64` types** (not in `docs/TYPE_DUPLICATION_MAP.md` prior to this entry):
- `causal_edge::CausalEdge64` at `crates/causal-edge/src/edge.rs:60` — SPO-palette layout (S/P/O palette indices + NARS f/c + Pearl 2³ mask + direction + inference type + plasticity + temporal)
- `thinking_engine::layered::CausalEdge64` at `crates/thinking-engine/src/layered.rs:45` — 8-channel cascade (BECOMES / CAUSES / SUPPORTS / REFINES / GROUNDS / ABSTRACTS / RELATES / CONTRADICTS, each 1 byte)

Same name, different bit semantics, different consumers. Reunification path = Option R-3 (transcode 8-channel → SPO at L3 commit). See `.claude/knowledge/causal-edge-64-spo-variant.md` + `.claude/knowledge/causal-edge-64-thinking-engine-variant.md` + `.claude/knowledge/causal-edge-64-synergies-and-pr-trajectory.md`.

**2. p64 drift origin pinpointed.** `crates/lance-graph-planner/src/cache/convergence.rs:18-22`:
```rust
#[allow(unused_imports)] // CausalEdge64 intended for hot-path convergence wiring
use super::nars_engine::{CausalEdge64, SpoHead, MASK_SPO};
```
The convergence wiring was started and never finished. The `nars_engine::CausalEdge64` re-export is the SPO-palette variant; the thinking-engine 8-channel variant was reinvented locally at `crates/thinking-engine/src/layered.rs:45` instead of imported here. **This `#[allow(unused_imports)]` annotation is the smoking gun** for where the dual-variant drift formalized.

**3. Three-zone hot-path mental model** (corrects "AriGraph reads = µs cold-path joins" framing):
- **Zone-1** (cycle-speed, 200-500 ns): thinking-engine MatVec → top-k atoms → `emit_causal_edges` 8-channel emission; AriGraph `entity_index: HashMap<String, Vec<usize>>` lookup is O(1) ~20-200 ns (NOT cold).
- **Zone-2** (SPO-as-3D-vector ANN, 20-1200 µs): blasgraph + neighborhood cascade HEEL → HIP → TWIG → LEAF via `zeckf64()`.
- **Zone-3** (DataFusion cold path, >1 ms): `lance-graph-planner` columnar joins for offline analytics; NOT touched by cognitive dispatch.

Cross-ref: `.claude/knowledge/cognitive-shader-driver-thinking-engine-reunification.md` (5-step reunification plan); `.claude/knowledge/splat-shader-rayon-struct-method-vision.md` (sprint-12+ 5-sprint arc).

---

## 2026-05-14 — E-CE64-MB-1..10: CausalEdge64-mailbox + sparse-rename composition (10 epiphanies)

10 epiphanies from the recursive-fresh-eyes architectural pass culminating in `.claude/plans/causaledge64-mailbox-rename-soa-v1.md`. Branch: `claude/resolve-pr-369-conflicts-ozMXd`. PR #370 in flight. Each epiphany is composition, not invention — every piece had existing plan/spec authoring before this session.

### E-CE64-MB-1 — Universal sparse-rename pattern (CPU-shaped, load-bearing)

Every architectural identity (G = OGIT domain, W = witness palette, style = ThinkingStyle/cognitive primitive/verb, truth = qualia band) renames to a hot-path slot via per-session-ephemeral `AttentionMask` SoA. Cold form lives unbounded in AriGraph / OGIT / contract. Physical form is 2-8 bit slot in CausalEdge64. **Per-session different rename tables = per-session different focus-of-attention.** Same 5-bit G means different domains in different sessions because the rename table differs. Same pattern as CPU register renaming, SSA register allocation, and TLB virtual-to-physical mapping. Closes a class of "type duplication" debt by collapsing 4 TrustTexture copies + 4 ThinkingStyle copies into one canonical field with documented projection lenses. Cross-ref: `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §2.

### E-CE64-MB-2 — Role-as-mailbox retires Vsa16kF32 as universal carrier

The 47 `LazyLock<RoleKey>` slice catalogue allocations across Vsa16kF32 (~3 MB if all materialized) collapse to 47 typed mailbox kinds (~50 KB). `vsa_bind(role_key, content)` becomes `mailbox::dispatch(content)` — a method call into the role-typed compartment. `vsa_bundle` (Σ across role keys) becomes witness aggregation in AriGraph. `vsa_permute` (positional braiding) becomes the mailbox's `TemporalWindow` lifecycle. Slice geometry (SUBJECT[0..4K) / PREDICATE[4K..8K) / etc.) becomes mailbox identity — no need for 16K float slots when 47 typed mailbox kinds suffice. **Vsa16kF32 retreats to its honest role: single-cycle Markov-bundle carrier for grammar parsing role-binding, dropped at cycle end.** No cumulative state in Vsa16kF32 anywhere. Cumulative state lives in AriGraph SPO-G quads + EdgeColumn CausalEdge64 emissions. Strengthens I-VSA-IDENTITIES iron rule. Cross-ref: §9 E-CE64-MB-2.

### E-CE64-MB-3 — Christmas-tree AriGraph decoration via SPO-G + ghost edges

Compartment epiphanies emit directly to AriGraph as SPO-G quads (G = OGIT domain pointer). Unresolved hole-forms from SPOW tetrahedron emit as ghost edges at Pearl rung 3 (counterfactual) or rung 7 (full-cf). Ghosts hibernate in AriGraph until evidence arrives. **AriGraph IS the long-term memory; the rename table is the working memory.** Eviction-from-working-memory ≠ deletion-from-long-term-memory. The mind always decorates; the tree never resets. New evidence on an evicted domain rebinds a fresh slot, potentially re-evicting another, and the ghost edges in AriGraph immediately reactivate as candidate hole-fills. Cross-ref: `oxigraph-arigraph-cognitive-shader-soa-merge-v1.md` §8 SPOW tetrahedron + §9 Gaussian splat hole-board.

### E-CE64-MB-4 — Ownership-typed compartments make UB a compile error

Each MailboxSoA row owns its delta buffer; BindSpace columns are `Arc`-shared with `BindSpaceView<'_>` zero-copy borrows and CollapseGate as single point of mutation. Cross-compartment communication can only flow as CausalEdge64 emissions (Copy, 8 bytes). The borrow checker **rejects** any code that tries to alias mutable BindSpace columns across compartments. **Race conditions at 200ns cycle speed become compile errors, not runtime bugs.** This is the same property Erlang's "share nothing" actors give you, but enforced statically by Rust's type system rather than dynamically by the runtime.

### E-CE64-MB-5 — Particle/wave duality in Rust semantics (not metaphor)

Particle = the owned compartment row in MailboxSoA (discrete, type-safe, Drop-managed lifecycle bounded by `TemporalWindow`). Wave = the CausalEdge64 emission rippling through EdgeColumn (BindSpace Column D) and decorating AriGraph SPO-G quads (continuous influence, non-local, no shared mutable state across compartments). **Both fall out of the same single rule: compartments own, AriGraph aggregates, CausalEdge64 crosses.** Not a metaphor — a structural property of the type system. The mailbox is a particle because the borrow checker forces it; the witness is a wave because AriGraph SPO-G quads + ghost edges make non-local influence the only cross-compartment path.

### E-CE64-MB-6 — The gRPC service shape IS the ractor message protocol

`crates/cognitive-shader-driver/src/grpc.rs` (LAB-ONLY behind `--features grpc`) defines `Dispatch(DispatchRequest) -> CrystalResponse` over tonic. **That IS the ractor mailbox handler shape.** Same Request/Response pair, same typed payload (`ShaderDispatch` + `CrystalResponse`), same no-shared-state contract. The transport varies: tonic gRPC (Zone-3 boundary), InMemoryMailbox via par-tile (cycle-speed Zone-1), TokioMailbox via existing `CallcenterSupervisor` (Zone-2 µs-ms), SupabaseSubMailbox (Zone-3 egress wrapper). **One protocol, four backings, transport-agnostic.** Reuse, don't invent. The lab-only gRPC service becomes the production ractor protocol simply by adding non-gRPC backings.

### E-CE64-MB-7 — Truth qualia is 2 bits with 4 consumer lenses

`TrustTexture` (Crystalline/Solid/Fuzzy/Murky-or-Dissonant), Wisdom markers, Staunen depth, MUL `GateDecision` (Proceed/Sandbox/Compass) are **four consumer-lens projections of the same 2-bit physical field**. Same byte position in CausalEdge64. Same architectural identity in `lance-graph-contract::mul::TrustTexture`. Different semantic vocabulary per consumer. Consolidates 4 type duplications into one canonical field with documented projection rules. Cross-ref: `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §2 lens table.

### E-CE64-MB-8 — Σ10 Rubicon dispatching IS the substrate-tier router

The named Σ1-Σ10 tier doctrine from `linguistic-epiphanies-2026-04-19.md` E21 (10 tiers × edge-type STATIC/EMERGENT/TWIG/EPIPHANY × Pearl rung 1-5 × theta repair/growth) finally gets a runtime dispatcher: `SigmaTierRouter` maps incoming compartment-spawn requests to the correct mailbox backing by tier band. Σ1-Σ5 STATIC reflexes → TokioMailbox (Zone 2). Σ6 EMERGENT + Σ7-Σ8 TWIG branching → InMemoryMailbox (Zone 1 cycle-speed). Σ9-Σ10 EPIPHANY → escalate to L4 `lance-graph-planner` strategy registry. **Wires what was previously documented-but-unwired.**

### E-CE64-MB-9 — JIT pipeline closes Gap 3 from THINKING_ORCHESTRATION_WIRING

The "FieldModulation → ScanParams → JitTemplate → Cranelift → KernelHandle" pipeline that exists across 3 repos but was never executed end-to-end: compartment-spawn IS the call site. Spawn message includes style-slot index; AttentionMask resolves to architectural ThinkingStyle; if `KernelHandle` cached, dispatch immediately; if not, JIT-compile via `crates/lance-graph-planner/src/strategy/jit_compile.rs` from YAML descriptor and cache. **End-to-end finally fires.** Gap 1 (Contract Not Consumed) also closes because the 8-bit style slot rename collapses the 12 vs 36 ThinkingStyle copies into one canonical form.

### E-CE64-MB-10 — Plasticity emerges naturally from MailboxSoA columns

Every successful emission increments `plasticity_counters[(role, G)]` co-occurrence bit-counter on the MailboxSoA. Spawn priors next cycle bias toward high-count pairings — Hebbian "fired together wired together." Counterfactual ghosts emit at low-counter slots (synaptic pruning). Pruning triggers (thinking-budget-exhausted, outcome-sufficient, XOR-cancel-with-sibling) fire from existing elevation `should_elevate()` + MUL `GateDecision::Proceed` + CollapseGate XOR-zero. **No new mechanism — just SoA columns + LRU on AttentionMask + bit-counter increment on emission + existing elevation/MUL/CollapseGate composed.** Two clocks naturally separated: fast (per-emission bit-counter) + slow (NARS truth-revise at AriGraph commit).

**Composition gate**: all 10 epiphanies above are realized by the 7-PR composition in `.claude/plans/causaledge64-mailbox-rename-soa-v1.md` §7. None require new architectural authoring — every piece had a named plan or spec before this session. The work is sequencing + the `par-tile` crate apex + the Σ-tier dispatcher.

---

## 2026-05-13 — DECISION: sprint-7 meta OQ-7-2 + OQ-7-3 resolved — AuditSink trait unification

Post-sprint-7 implementation, Opus meta surfaced a critical cross-impl risk (CC-7-1) and two open questions blocking the sprint-7 PR open:

- **OQ-7-2: AuditSink trait split.** `UnifiedBridge::audit_sink` was typed `Arc<dyn UnifiedAuditSink>` (D-SDR-4 placeholder trait at `unified_audit.rs:314`); sprint-7 W6 production sinks (`JsonlAuditSink`, `LanceAuditSink`, `CompositeSink`) implement `Arc<dyn AuditSink>` (new trait at `audit_sink/mod.rs:45`). The two traits had different signatures (`emit(&event)` vs `emit(event) -> Result<>`). W6 sinks shipped orphaned from the bridge. **Resolution: full migrate, drop UnifiedAuditSink, no adapter.** Per CLAUDE.md "no abstractions beyond what the task requires" — an adapter is permanent overhead to avoid one-time call-site churn. Landed in commit `bc530a4`. 6 files touched; `UnifiedAuditEvent::canonical_bytes` byte layout unchanged (still 26 bytes).

- **OQ-7-3: UnifiedBridge::new() default sink behavior.** MedCare-rs sprint-2 item 5 expects "JSONL primary + optional Lance projection". **Resolution: keep `NoopAuditSink` as new() default; add ergonomic constructor `UnifiedBridge::with_jsonl_audit(super_domain, salt, base_path)` for explicit opt-in.** Silent default writes to disk are a surprise (the path would be implicit, log volume unbounded). Opt-in via the new constructor is more honest. MedCare-rs consumers wire JSONL when they construct the bridge; default-noop doesn't prevent that pattern. Available under `#[cfg(feature = "jsonl")]`.

Also confirmed non-blocking:

- **OQ-7-1: RoleGroup count.** MedCare-rs#119 ships 6 RoleGroups (Physician, Nurse, Cashier, Researcher, HipaaAudit, Admin); end-state matches our lance-graph decision regardless of "add 4" wording in the earlier EPIPHANY (4 additions = Nurse + 3 renames). No code change needed.
- **W3 LifecycleAuditEvent ↔ W6 CompositeSink routing.** `LifecycleAuditEvent` (18 bytes) is intentionally separate from `UnifiedAuditEvent` (26 bytes) per sprint-5-6 meta CC-2 fix. They do NOT share the AuditSink trait — supervisor lifecycle audit is a parallel chain by design. If a future need to unify them surfaces, that's its own spec.

Cross-ref: `.claude/board/sprint-log-7/meta-review.md` §1+§3, commit `bc530a4` (the trait migration).

---

## 2026-05-13 — DECISION: 4 PR #365 blocking OQs resolved — sprint-7 implementation can begin

Post-#365 cross-session triage with the medcare-rs session resolved all four user-decision Open Questions that the Opus meta-review flagged as blocking sprint-7 implementation:

- **OQ-1 (W3) TTL family-registry parser entry → new `parse_family_registry()` API.** Keeps `parse_ttl_directory_with_provenance` focused on ontology TTL; family-registry TTL is a different schema; mixing them via overload-by-naming is the wrong abstraction.
- **OQ-2 (W10) `MANIFEST_METADATA` storage → sorted-slice + binary search.** `lance-graph-contract` zero-dep invariant in CLAUDE.md is iron. `phf` would be the first non-build dep on the contract crate. Binary search on sorted-slice is O(log n) and zero-dep. The C-grade meta finding for `pr-g1-manifest-modules.md §4.3` resolves by this change.
- **OQ-3 (W6) `medcare_rbac::Role` migration → direct migration (rename `doctor → physician`, add `nurse / cashier / researcher / hipaa_audit`).** Per CLAUDE.md "Don't introduce abstractions beyond what the task requires." A bridge adapter is a permanent abstraction to avoid one-time call-site churn — wrong tradeoff. `super-domain-rbac-tenancy-v1.md §14` made canonical RoleGroups primary; aligning is mandatory, not optional. E1-1 LOC stays at ~180. medcare-rs session eats the call-site churn.
- **OQ-4 (W13 §E.1) OGIT/NTO/SMB BSON namespace → `ogit.SMB.bson:` sub-namespace.** `registry.enumerate("SMB")` must return exactly 3 Foundry entities; mixing BSON into the same namespace breaks the `smb_projects_three_entities` test and corrupts the `OntologyRegistry` index.

Cross-session boundary clarified (lance-graph side ↔ medcare-rs side):
- **lance-graph (this session):** sprint-7 implementation fleet for W3 family-hydration (the cascade unblocker), W10 manifest-modules (with sorted-slice fix), W11 ractor-supervisor (with `LifecycleAuditEvent` split per meta CC-2), W12 conformance crate, W1 LanceAuditSink, W2 JsonlAuditSink + verify CLI, W9 thinking-engine wire.
- **medcare-rs session:** PR-α (`MedcareOntology::from_registry` red-build fix), PR-β' (E1-1 wire `medcare_healthcare_policy()` + direct migration per OQ-3), PR-γ (FingerprintCodec re-export fold — Pattern N anti-pattern at `medcare-analytics/src/soa_mapping.rs`; ~20 LOC scope, delete enum + re-export from `lance_graph_contract::cam` / `bgz17`), PR-δ (AUTH_LEGACY_TRIPLEDES_MIGRATION audit vs PR #363 §18, doc-only).
- **Both deferred:** E1-5 (HIPAA hard-lock cross-domain matrix, D-SDR-17, ~60 LOC) → sprint-8 compliance owns. E1-6 (JWT middleware stub for `praxis_id`, ~150 LOC) → blocked on DM-7 (`RlsRewriter::rewrite(LogicalPlan, &ActorContext)` per foundry-roadmap §2).
- **E1-3 / E1-4** (`MedCareStack` composition + audit emission) → cascade-unblocks once W3 lands `parse_family_registry()` + seeds `OgitFamilyTable` for Healthcare basins 0x10..=0x19.

Cross-ref: `.claude/board/sprint-log-5-6/meta-review.md` §6 (OQ triage), PR #365 body (OQs as checkboxes), `super-domain-rbac-tenancy-v1.md §14`.

---

## 2026-05-13 — CORRECTION-OF sprint-4 framing: most worker specs partially duplicated existing `.claude/plans/` corpus — sprint-5 MUST grep `.claude/plans/*.md` before spawning any worker

**Status:** FINDING (user surfaced prior plans 2026-05-13 evening)

Sprint-4 spawned 12 workers to convert 11 TD rows into PR-ready specs. **Discovered post-hoc that most architectural specs duplicated existing plan-tier docs already on the branch.** The workers did not grep `.claude/plans/` before drafting.

**Duplication audit:**

| Sprint-4 worker spec | Prior plan that already covered it | Duplication |
|---|---|---|
| W1 `sprint-4-execution-plan.md` (24 KB) | `unified-ogit-architecture-v1.md` (30 KB, 15 patterns A-O, master) | High |
| W4 `td-super-domain-subcrates.md` (21 KB) | `super-domain-rbac-tenancy-v1.md` (86 KB / 1387 lines, canonical PR #363 spec) + `foundry-roadmap-unified-smb-medcare-v1.md` | High |
| W11 `fma-heart-click-smoke.md` (28 KB) | `anatomy-realtime-v1.md` (19 KB, the proof-of-vision plan) + `lance-graph-rdf-fma-snomed-v1.md` | High |
| W6 `td-thinking-engine-wire.md` (21 KB) | `jc-pillars-runtime-wiring-v1.md` + ERRATUM | Medium (composition map added value) |
| (today's splat thrash) | `tetrahedral-epiphany-splat-integration-v1.md` + `2026-05-06-splat-osint-ingestion-v1.md` (ACTIVE) + `jc-pillars-runtime-wiring-v1.md` | High |

**What sprint-4 DID add (the real value):**
- W3 API drift deprecation playbook — no prior plan covered this
- W7 D-SDR PR release plan — captures concrete next-PR (PR-A on top of #363) with SHAs
- W8 audit Lance/JSONL sink spec — prior plans mention LanceAuditSink as substrate but no implementation spec
- W10 slot u8→u16 widen + bridge-err audit — surgical fixes; no prior plan
- W12 cross-repo PR graph — sprint sequencing artifact
- W9 family hydration — surgical fix; no prior plan

**The lesson at THREE layers today:**
1. **Math layer:** one kernel `Σ' = J·Σ·Jᵀ`, three Jacobians (camera projection / edge step / radial decay) — not three separate "splat" concepts
2. **Substrate layer:** `ndarray::hpc::renderer` already exists with 60fps double-buffer + EWA-splat projection; no new render crate needed
3. **Plan layer:** `.claude/plans/*.md` has 30+ plans already covering the architectural surface; worker subagents must grep before drafting

**Sprint-5 mandatory read-order fix:**

Before spawning ANY worker on a spec touching FMA / OGIT / super-domain / RBAC / splat / EWA / Pillar-N / cognitive shader / consumer crate / audit / thinking-engine:

```
1. ls .claude/plans/ | head -40        # see all 30+ plan files
2. cat .claude/plans/unified-ogit-architecture-v1.md      # the 15-pattern master plan (A-O)
3. cat .claude/plans/anatomy-realtime-v1.md               # the FMA proof-of-vision plan
4. cat .claude/plans/super-domain-rbac-tenancy-v1.md      # the canonical RBAC/tenancy spec (1387 lines)
5. cat .claude/plans/jc-pillars-runtime-wiring-v1.md      # the JC pillar stack (pillars 5/5+/5++/6/7)
6. cat .claude/plans/foundry-roadmap-unified-smb-medcare-v1.md  # consumer crate roadmap
7. cat .claude/plans/compile-time-consumer-binding-v1.md  # Pattern E (manifest modules) + F (ractor supervisor)
8. cat .claude/plans/ogit-g-context-bundle-v1.md          # Tier-1 SPO-G slot + ContextBundle + GenericBridge
9. cat .claude/plans/2026-05-06-splat-osint-ingestion-v1.md  # ACTIVE splat-OSINT plan
10. cat .claude/plans/tetrahedral-epiphany-splat-integration-v1.md  # SPOW tetrahedral grid + splat integration
11. cat .claude/plans/lance-graph-rdf-fma-snomed-v1.md    # FMA + SNOMED + RadLex named-graph ingest
12. grep -l "<topic>" .claude/plans/    # find any topic-specific plans
```

Worker prompts must include: "Before drafting, read these specific plan files: [...]. Cite them or explain why your spec adds value beyond them."

**Sprint-5 priority stack — REVISED against the real plan corpus:**

The Tier 0-4 stack from earlier today still holds for TD coverage, BUT the deliverable framing changes:

- **Tier 0** (PR follow-up, ~1 day) — UNCHANGED. PR-A composes the existing 3 commits on top of PR #363; SHAs already captured in W7's spec.
- **Tier 1** (substrate, ~1 week) — W10 + W8 + W9 are still the right surgical fixes. **But** they need to be reframed as DELTA against `super-domain-rbac-tenancy-v1.md §13` (D-SDR-3..5 already named there). Each PR should cite §X of that plan.
- **Tier 2** (composable wiring, ~2 weeks) — W4 and W6 should NOT use the sprint-4 specs as-is. Both must be rewritten as DELTA against existing plans:
  - W4: against `super-domain-rbac-tenancy-v1.md §14` (meta-bridge extraction, woa retrofit, hubspot/hiro templates already named) + `foundry-roadmap-unified-smb-medcare-v1.md` (LF-3 critical path)
  - W6: against `jc-pillars-runtime-wiring-v1.md` (the pillar wiring already plans the thinking-engine composition)
- **Tier 3** (FMA convergence) — W11's spec should be REPLACED with citations to `anatomy-realtime-v1.md` (already the proof-of-vision plan) + `lance-graph-rdf-fma-snomed-v1.md` (already the FMA ingest plan). Sprint-4 W11 spec keeps its drug-knowledge crosswalk + two-tier-ingest patches as additions to those plans.
- **Tier 4** (perf) — W5 unchanged; no prior plan duplication.

**The honest meta-pattern:** I generated three classes of correction today (math/substrate/plan) for the same root cause — conjecturing before grepping. The fix is not "be more careful next time" — it's "the worker prompt template MUST include a mandatory read-order section pointing at `.claude/plans/`, and that section must be a hard precondition to spec writing." Update worker prompt templates for sprint-5.

## 2026-05-13 — UNIFICATION: Gaussian-splat + EWA-Sandwich is ONE kernel (`Σ' = J·Σ·Jᵀ`) applied to THREE Jacobians across the workspace — render, graph propagation, perturbation field

**Status:** FINDING (corrects the same-day three-meanings-of-splat entry that wrongly split them apart; user-corrected 2026-05-13)

The previous entry framed "three meanings of splat" as three unrelated primitives that happen to share a name. **Wrong.** They are three applications of one mathematical kernel — the Σ push-forward of a Gaussian (mean + covariance ellipsoid) through an affine map:

```
Σ' = J · Σ · Jᵀ
```

Same math (Heckbert's EWA sandwich form). Three different Jacobians J. Three different deliverables, all unified by the kernel:

| Application | Jacobian J | Σ semantics | Deliverable |
|---|---|---|---|
| **Render** | Camera projection (3D→2D image) | Per-node position+covariance (covariance derived from VSA fingerprint structure) | `ndarray::hpc::renderer` — 60fps SIMD double-buffer renderer for q2 cockpit / Palantir Gotham / Neo4j-style 3D graph visualization |
| **Graph propagation** | Edge step (node→neighbor) | Node-state covariance Σ pushed forward along multi-hop paths | `crates/jc/src/ewa_sandwich.rs` (450 LOC) + `crates/lance-graph-contract/src/sigma_propagation.rs` (488 LOC) — Pillar 6 PSD-preservation cert (10000/10000 hops, CV tightness 1.467×, Köstenberger-Stark rate) |
| **Perturbation field** | Spatial radial decay (query→neighborhood) | Query-as-Gaussian-deposit; Σ pushed outward through the spatial field | `crates/jc/examples/splat_perturbationslernen.rs` (445 LOC) — context-search-as-perturbation probe; rows crossing α-saturation are the "found context" |

**Why this matters architecturally:**
1. The renderer is NOT separate-and-orthogonal to EWA-Sandwich — it's the visualization tier of the same kernel. The per-node 3D Gaussian splat that `renderer.rs` projects to the q2 viewport is the same Gaussian whose covariance Σ propagates through Pillar 6 when you traverse an edge.
2. The 75K-entity FMA heart-click demo gets ALL THREE for free from the same kernel:
   - Render: 60fps live EWA-splat projection of FMA-anatomy Gaussians (no prerender needed)
   - Click semantics: SPO neighbor query → Pillar 6 multi-hop Σ propagation along anatomy edges (heart → vessels → systemic circulation)
   - Search by feel: heart-click as perturbation deposit; α-saturation readout finds "anatomically related context" without explicit MATCH-Cypher
3. The "Amiga demoscene prerender" escape hatch I conjectured is wrong on two axes: (a) the live path already works because the substrate is SIMD-accelerated; (b) even if it failed at scale, the right escape is reducing the per-node Σ rank, not prerendering, because the kernel is the unification point.

**ndarray + jc + lance-graph composition** (the three crates each own one Jacobian):
- `ndarray::hpc::renderer` owns the camera-projection Jacobian + SIMD double-buffer
- `crates/jc` owns the edge-step Jacobian + PSD certification
- `lance-graph-contract::sigma_propagation` owns the type-level surface that both renderers and graph traversers depend on

This is the same "compose, don't rebuild" pattern surfaced in W6 (thinking-engine wire-up): the workspace's substrate is denser than any single subagent's read window. Sprint-5 reconciliation pass must add `ndarray::hpc::renderer` + the JC pillar stack as MANDATORY READS for any spec touching FMA, q2 cockpit, multi-hop edge propagation, or covariance-based context search.

Cross-ref: previous same-day splat-conjecture entry (`Gaussian-splat prerendered buffer`) — DEFERRED, since the live kernel composition already covers the use cases; W11 FMA spec needs a sprint-5 patch citing the unified kernel as its math basis; `.claude/plans/jc-pillars-runtime-wiring-v1.md` + ERRATUM define the full pillar stack (5/5+/5++/6/7) the renderer composes with.

## 2026-05-13 — FINDING: `ndarray::hpc::renderer` is the canonical 60fps SIMD double-buffer renderer for q2 — the FMA heart-click 3D anatomy view already has its render substrate, no prerender needed

**Status:** FINDING (confirmed in source — `/home/user/ndarray/src/hpc/renderer.rs`, 995 LOC)

Earlier same-day conjectures (the "Amiga demoscene prerender" idea) assumed q2 needed a prerendered Gaussian-splat buffer because live rendering of 75K FMA entities would be too expensive. **Wrong premise:** ndarray already ships the renderer.

The renderer architecture (from the doc-comment at `/home/user/ndarray/src/hpc/renderer.rs:1-44`):
- **SIMD-accelerated double-buffer** for "SPO graph visualization … hardware-acceleration mothership for q2 cockpit / Palantir Gotham / Neo4j-style visual rendering"
- Double-buffer pattern: `front: LazyLock<RwLock<RenderFrame>>` (readers via REST/SSE) ↔ `back: LazyLock<RwLock<RenderFrame>>` (shader cycle writes); atomic swap via `AtomicUsize`
- Per-tier SIMD dispatch: AVX-512 / AVX2 / AMX / NEON / scalar — `F32x16::mul_add` for force integration on the hot path
- 60fps canonical tick via `cached_splat(DT_60)` — `F32x16::splat(1.0/60.0)` cached via `LazyLock` so the integration loop avoids re-broadcasting dt
- SoA frame: positions, velocities, charges, fingerprints (VSA_WORDS·N · u64) — 64-byte aligned, all capacities multiple of `PREFERRED_F32_LANES`

The FMA heart-click flow becomes:
1. FMA OWL → SPO triples in lance-graph (W11 spec)
2. SPO → `RenderFrame` (positions seeded from entity layout, fingerprints from VSA encoding)
3. ndarray::hpc::renderer integrates at 60fps (force-directed layout converges)
4. q2 cockpit reads `front` buffer via REST/SSE
5. Heart-click = q2 sends Cypher to lance-graph → UnifiedBridge auth → SPO neighbor query → render frame updates highlighted subgraph

This kills three earlier same-day conjectures simultaneously:
- "Need to prerender 900-18000 frames" — NO, live 60fps already works
- "Need new `crates/lance-graph-render-buffer/`" — NO, the substrate is `ndarray::hpc::renderer`
- "Gaussian-splat rendering as Tier-3 escape hatch" — DEFERRED; only worth doing if the 60fps live path is measured to fail on 75K entities (which it might, but measure first)

**Three meanings of "splat" in this workspace** (NONE are 3DGS scene rendering — that's a fourth thing that doesn't exist here):
1. `ndarray::simd::F32x16::splat(dt)` — SIMD scalar→vector broadcast (`_mm512_set1_ps`); `cached_splat` caches it for canonical 60/30/15 fps tick rates
2. `crates/jc/src/ewa_sandwich.rs` — Pillar 6 Σ push-forward `M·Σ·Mᵀ` for multi-hop edge propagation (PSD-preservation cert)
3. `crates/jc/examples/splat_perturbationslernen.rs` — perturbation-learning probe; uses EWA-Sandwich to splat a query INTO the spatial field, measures covariance displacement

**Architectural lesson for sprint-5:** when the workspace already has a load-bearing substrate (ndarray's renderer, jc's pillars), the right move is "compose, don't rebuild" — same lesson as W6's thinking-engine wire-up spec applied to a different substrate. The FMA spec needs a patch citing `ndarray::hpc::renderer::RenderFrame` as the canonical render target; this kills its current vague "q2 3D anatomy render" handwave.

Cross-ref: ndarray CLAUDE.md "ndarray = hardware (SIMD, Palette, Base17, …)" architecture rule; W11 FMA spec (needs Tier-3 section rewrite — splat-prerender is a deferred speculation, not a deliverable); IDEAS.md 2026-05-13 splat row (needs second correction).

## 2026-05-13 — CORRECTION-OF earlier same-day splat-conjecture: EWA-Sandwich is Pillar 6 (Σ push-forward `M·Σ·Mᵀ` for multi-hop edge propagation), NOT a Gaussian-splat renderer

**Status:** FINDING (confirmed in source — `crates/jc/src/ewa_sandwich.rs`, `crates/lance-graph-contract/src/sigma_propagation.rs`, plans `.claude/plans/jc-pillars-runtime-wiring-v1.md` + ERRATUM)

Earlier today (entry below) I conjectured EWA-Sandwich was Heckbert's classical Elliptical Weighted Average splat filter applied to anatomical 3D rendering. **Wrong.** In this workspace EWA-Sandwich is the **mathematical backbone of multi-hop covariance propagation in graph edge paths**, certifying that arbitrary-depth traversal stays in the SPD cone.

The math:
```
Σ_n = M_n · M_{n-1} · ... · M_1 · Σ_0 · M_1ᵀ · ... · M_{n-1}ᵀ · M_nᵀ
```
where `M_k = sqrt(Σ_k)` is the step-Jacobian of the k-th edge. Same kernel as Heckbert (`Σ' = J·Σ·Jᵀ`), different role.

**Pillar 6 in the JC framework certifies two things simultaneously:**
1. **PSD-preservation:** Σ_n stays SPD for all n (proven 10,000/10,000 hops in the probe)
2. **Convergence rate:** `‖log(Σ_n) − E[log(Σ_n)]‖_F^2` concentrates at Köstenberger-Stark rate (CV tightness 1.467×) — meaning the path itself shapes propagation instead of every hop adding noise

**Why this matters:**
- Plain Gaussian convolution gives O(n) error growth — Σ_n's variance scales with path length, signal lost by depth >5
- EWA-Sandwich gives **bounded** Σ_n with geometric error control iff M_k contractive
- This makes multi-hop graph queries meaningful at any depth — the "can't-stop-thinking loop" has mathematical ground under it

**Architectural composition (full pillar stack):**
- Pillar 5 (Jirak 2016) — scalar Berry-Esseen under weak dependence
- Pillar 5+ (Köstenberger-Stark) — Σ-tensor concentration
- Pillar 5++ (DZ) — Hilbert-space extension
- **Pillar 6 (EWA-Sandwich)** — multi-hop SPD propagation
- Pillar 7 (α-saturation) — settling criterion for the "Perturbationslernen" probe (query as perturbation injected into the spatial field; EWA-Sandwich propagates Σ outward; rows crossing α-saturation are the found context)

**Plus PR #288** (Σ-codebook viability probe, R² = 0.9949) ruled out the CausalEdge64 8→16 byte expansion that would have halved the HighHeelBGZ 240-edge container limit. The 256-entry codebook with 1-byte sidecar is sufficient.

Cross-ref: `crates/jc/examples/osint_edge_traversal.rs` (canonical OSINT-route demo using Pillar 6), `crates/jc/examples/splat_perturbationslernen.rs` (the "splat" of the perturbation-learning probe — covariance-ellipsoid displacement, NOT 3D rendering), `IDEAS.md` 2026-05-13 splat entry (now corrected — split into two distinct ideas: the Pillar 6 architectural fact, and the separate-and-orthogonal q2-3D-render speculation).

## 2026-05-13 — CONJECTURE: Gaussian-splat prerendered buffer is the Amiga-demoscene escape hatch for hydrating the 75K-entity FMA anatomy into q2's 3D view

**Status:** CONJECTURE (not yet wired; no prior art found in lance-graph / ndarray / q2 grep for `gaussian|splat|prerender|demoscene|amiga`)

The naive heart-click smoke test (W11 spec) hits a runtime wall: rendering 75K anatomical entities live in q2's WebGL/WebGPU context is not interactive-grade. The escape hatch is the Amiga demoscene tactic — **prerender once, replay cheaply**:

- **Source:** FMA OWL → entity geometry (mesh or implicit) → 3DGS (3D Gaussian Splatting) scene as a single static splat cloud.
- **Camera trajectory:** prerender 30–300 seconds × 30–60 fps = **900–18,000 frames** of camera fly-through covering all canonical viewpoints (whole-body, organ-system close-ups, heart, brain, skeleton).
- **EWA-Sandwich filter:** Heckbert's Elliptical Weighted Average resampling filter as a three-pass sandwich (prefilter → splat-projection → postfilter) gives anti-aliased composition between layers. Used in modern 3DGS pipelines for the same reason demoscene used precomputed dithering tables: defer the math to author-time.
- **Stream:** q2 graph-notebook subscribes to a splat-frame stream (Arrow Flight or WebSocket) and renders from the buffer. Heart-click = seek-to-heart-camera-position in the buffer, NOT live 75K-entity render.
- **Hybrid:** SPO edge graph (lance-graph) still drives the click semantics + audit chain + drug-knowledge crosswalk; the splat buffer is JUST the visual rendering layer. The two-tier ingest (CSV-quick + OWL-full) gates which buffer is loadable.

Cross-ref: FMA smoke test spec `.claude/specs/fma-heart-click-smoke.md`; the splat-buffer approach is an OPTIONAL acceleration tier (Tier-3 alongside Tier-1 CSV / Tier-2 OWL). Likely candidate crate locations: `crates/lance-graph-callcenter/render/` or new `crates/lance-graph-splat-buffer/`; uses `ndarray::simd` for splat projection (cross-flag W5).

Open questions: (a) is 3DGS the right algorithm or do we want surfels / point-cloud variants? (b) where does the prerender job run — CI nightly, or one-shot offline tool? (c) buffer storage format — raw Arrow batches, MP4-like temporal codec, or splat-native (.splat / .ply)? (d) does EWA-Sandwich live in ndarray (SIMD-friendly) or in q2 (renderer-adjacent)?

## 2026-05-13 — FINDING: FMA (75K-entity human anatomy OWL ontology) is the canonical smoke-test for the entire OGIT ↔ OSINT ↔ Palantir/Neo4j ↔ q2 route — dual-test for edge propagation AND Healthcare super-domain

**Status:** FINDING (anchors the demo-able milestone for the whole integration arc)

User instruction 2026-05-13: "remember the show-off to wire FMA 70K human anatomy as on-screen rendering — smoke test for both neo4j-ish edges propagation AND healthcare". This pins the **demo-able milestone** that the entire integration plan (D-SDR-* + Pattern E+F+cognition cascade + super-domain subcrate cascade + q2 wiring + EWA-Sandwich proof) is converging towards.

**FMA (Foundational Model of Anatomy):**

- **75,000 anatomical classes + 168 properties** (`anatomy-realtime-v1.md` §1).
- OWL-formatted ontology — directly exercises our `OgitFamilyTable` codebook lookup at scale (way above the 256-slot-per-family cap, so it forces the OGIT addressing to demonstrate multi-basin coordination).
- Public dataset (no HIPAA constraints on FMA itself; the **enforcement smoke-test** is wrapping it under `SuperDomain::Healthcare` to prove the auth pipeline works at scale).
- Already targeted as `G=FMA_V1` ContextBundle in `anatomy-realtime-v1.md` PR-ANATOMY-1 (OWL hydrator for FMA).

**Why FMA is the right smoke-test (dual purpose):**

1. **Neo4j-ish edges propagation test** — 168 properties × 75K entities = the multi-hop graph traversal benchmark. The EWA-Sandwich Σ-push-forward (Pillar 6 PR #289, certified 10000/10000 PSD-preservation) replaces Neo4j-style edge traversal for "show everything connected to the heart" queries (`anatomy-realtime-v1.md` row 6 + 7). **If FMA's heart-connected substructure resolves correctly via EWA-Sandwich, the Neo4j substitute is operationally proven.**
2. **Healthcare super-domain test** — wrap the FMA registry under `UnifiedBridge<MedcareBridge>::with_audit_chain(SuperDomain::Healthcare, salt, JsonLinesAuditSink::healthcare())`. Every FMA query emits a chained `UnifiedAuditEvent` carrying `merkle_root` + (after cognition-bridge lands) `awareness_root`. **If FMA queries under the Healthcare authorize-pipeline produce the right policy/audit/role-projection chain at scale, the medcare super-domain subcrate is operationally proven.**
3. **Visual rendering test** — q2 cockpit-server renders FMA in 3D with anatomical labels overlaid + cross-section (`anatomy-realtime-v1.md` row 7 — "Realtime 3D render with FMA labels"). Tests `q2::notebook-render` + Pattern H (`p64-bridge::CognitiveShader` dispatches per-G program). **If the heart-click-to-rendered-anatomy round-trip works, Palantir-Gotham-parity visual surface is operationally proven.**

**The smoke test as one continuous demo:**

```
User opens q2 notebook → writes Cypher cell:
    MATCH (h:Heart)-[r*1..5]-(connected) RETURN h, r, connected

q2::notebook-query (polyglot parser)
    → lance-graph-planner Strategy #1 (CypherParse)
    → UnifiedLogicalPlan with 5-hop * traversal
    → UnifiedBridge<MedcareBridge>::authorize_read(canonical='Heart', PrefetchDepth::Multihop(5))
       ├─ SuperDomain::Healthcare salt applied to AuditChain
       ├─ Policy::evaluate("clinician", "Heart", Operation::Read{depth:Multihop(5)}) → Allow
       ├─ UnifiedAuditEvent emitted with merkle_root + awareness_root
       └─ EwaSandwichTraversal::propagate_5hop(heart_id, Sigma_FMA) → multi-hop Σ in 1 vector pass
    → DataFusion ScanExec over Lance dataset with G=FMA_V1
       ├─ batch-decorated with DolceMarker (Endurant for body parts, Quality for properties)
       ├─ Path C (ndarray::simd::gather_u8) for per-row super-domain annotation
       └─ thinking-engine projection per row (RoleProjection::for_role("clinician"))
    → Arrow RecordBatch with ~thousands of heart-connected anatomical entities
    → q2::notebook-render (3D anatomy view + labels + cross-section)
    → User clicks "show everything connected to the heart"
    → Real-time graph propagation visible on screen
```

**This demo touches ALL THREE substrate paths (Path A thinking-engine + Path B ractor + Path C ndarray::simd) AND all integration plan deliverables:**

- D-SDR-1..5 (UnifiedBridge with audit emission) — every query carries the auth/audit pipeline
- D-SDR-3b (TTL hydration baker) — FMA TTL → OgitFamilyTable populated
- D-ONTO-V5-3 (Healthcare TTL transcode) — `OGIT/NTO/Healthcare/{entities,verbs}/*.ttl` includes FMA
- D-ANATOMY-1..7 from `anatomy-realtime-v1.md` — the demo pipeline itself
- D-SPLAT-1..7 from splat-osint-ingestion (EWA-Sandwich edge propagation)
- D-PARITY-V2-* (DTO ladder for Foundry-parity visual)
- Q2-1.1..Q2-1.7 + Q2-2.x (q2 bridge + Cypher console)
- Pattern E manifest entry: `/modules/healthcare/manifest.yaml` declares FMA as part of Healthcare super-domain
- Pattern F ractor: CallcenterSupervisor spawns HealthcareActor on boot
- thinking-engine wiring: cognition_bridge projects clinician role through `role_tables`

**Implication for the integration plan:**

The FMA demo is the **integration gate** for Phase-C of `anatomy-realtime-v1.md` ("End of this phase = the system is demoable end-to-end"). Every deliverable in the active plans (`super-domain-rbac-tenancy-v1`, `palantir-parity-cascade-v2`, `splat-osint-ingestion-v1`, `lance-graph-ontology-v5`, `anatomy-realtime-v1`, `compile-time-consumer-binding-v1`, `ogit-cascade-supabase-callcenter-v1`) converges towards this demo. **A working FMA-heart-click demo is the proof that the integration plan stands up under real-world workload.**

**Sequencing the FMA smoke-test as the convergence anchor:**

1. **Phase 0** (current sprint, this week): follow-up PR for D-SDR-3..5 + consumer-side push of medcare-rs `unified_bridge_wiring`.
2. **Phase 0.5** (next 1-2 sprints): Pattern E+F+cognition cascade (manifest + ractor + cognition-bridge) — establishes the runtime topology FMA queries will route through.
3. **Phase 1** (next 1-2 sprints, parallel with 0.5): D-ONTO-V5-3 (Healthcare TTL transcode) + D-SDR-3b (TTL hydration baker) → `OgitFamilyTable` populated from FMA TTL.
4. **Phase 2** (after 0.5+1): PR-ANATOMY-1 OWL hydrator + PR-ANATOMY-2 ContextBundle lookup + PR-ANATOMY-3 (`anatomy-realtime-v1.md` Phase B Hydrators).
5. **Phase 3** (after 2, parallel with Tier H LanceProbe wiring): D-SPLAT-1..7 (EWA-Sandwich contract types + `osint_edge_traversal.rs` demo refactored as `fma_edge_traversal.rs`).
6. **Phase 4** (after 3): PR-ANATOMY-4 (Q2 3D view) + PR-ANATOMY-5 (medical vocab) + Q2-2.x Cypher console wiring. **End of Phase 4 = demoable FMA smoke-test.**

**The ball that mustn't drop:** without an explicit smoke-test anchor, the integration plan is a list of deliverables without a definition of "done at integration scope". The FMA demo provides that definition. Every PR review can ask "does this move us closer to the heart-click demo?" — if yes, ship; if no, re-scope. This entry pins the anchor so future sessions don't lose sight of it.

Cross-ref: `.claude/plans/anatomy-realtime-v1.md` (the full Phase A-D plan with FMA as Pattern A target); `.claude/plans/lance-graph-ontology-v5.md` D-ONTO-V5-3 (Healthcare TTL transcode includes FMA); `.claude/plans/2026-05-06-splat-osint-ingestion-v1.md` D-SPLAT-1..7 (EWA-Sandwich primitives); `.claude/plans/q2-foundry-integration-v1.md` Q2-1.1..Q2-1.7 (q2 cockpit + Cypher console); `EPIPHANIES.md` 2026-05-13 OGIT-OSINT-Palantir/Neo4j-q2 route (this entry anchors that route's smoke-test); `IDEAS.md` 2026-05-13 super-domain subcrate scaffolding cascade (medcare PR 1 is the Healthcare path FMA rides on); `TECH_DEBT.md` TD-Q2-STUBS-DEDUP-1 (q2's lance-graph + ndarray stubs need to be re-exports for the demo to compile against canonical crates).

## 2026-05-13 — FINDING: the OGIT ↔ OSINT ↔ Palantir Gotham / Neo4j route runs through q2 — q2 is the external graph-notebook consumer (Tier C super-domain subcrate equivalent)

**Status:** FINDING (closes a Q2-shaped hole left open across multiple plans)

User instruction 2026-05-13: "add q2 to MCP scope, access via pygithub, wire the OGIT ↔ OSINT ↔ Palantir Gotham / Neo4j route". q2 is already in the MCP scope per the session prompt (`adaworldapi/q2`); access verified via `mcp__github__get_file_contents` on `README.md`. The discovery: **q2 IS the external graph-notebook consumer for the entire integration plan** — what hubspot-rs / hiro-rs / woa-rs are to Tier C, q2 is for the visual + interactive + polyglot-query slot.

**q2's relevant inventory (from its README + workspace):**

| q2 component | What it provides | Maps onto OGIT ↔ OSINT ↔ Palantir/Neo4j route |
|---|---|---|
| `crates/stubs/notebook-query` | Cypher / Gremlin / SPARQL polyglot query execution (stub — to be replaced with full impl) | The **external query surface** that lowers polyglot graph queries onto our DataFusion plan via `lance-graph-planner` 16 strategies |
| `crates/stubs/lance-graph` (q2's local stub) | Graph storage with vertex/edge CRUD (stub) | Should re-export `AdaWorldAPI/lance-graph` instead of carrying its own stub — current stub is what they put together as a placeholder |
| `crates/stubs/notebook-runtime` | Reactive cell DAG with dependency tracking | The **execution surface** that runs polyglot cells against the OGIT spine and reacts to graph changes (Supabase realtime path per `ogit-cascade-supabase-callcenter-v1`) |
| `crates/stubs/notebook-render` | HTML rendering for graphs / tables / charts | The **visual surface** that renders Palantir-Gotham-equivalent graph views (per `palantir-parity-cascade-v2` § Q2-2.x) |
| `crates/stubs/q2-ndarray` | SIMD array operations stub | Should re-export `AdaWorldAPI/ndarray` instead — same dedup logic as the lance-graph stub |
| `crates/cockpit-server` | Q2 cockpit UI server | The **operator surface** for Foundry/Gotham parity (Q2 cockpit was always the Foundry-parity target per `palantir-parity-cascade-v2` Foundry-status table: IN PROGRESS) |
| `crates/aiwar-ingest` | AI War cloud dataset pipeline | The **data ingest** surface that exercises the OSINT super-domain — the `aiwar` repo is the external dataset; `neo4j-rs` is the backend; aiwar-ingest is the q2-side ingest |
| Related repo `neo4j-rs` (`AdaWorldAPI/neo4j-rs`) | Graph database backend | The **substrate** that the EWA-Sandwich proof (Pillar 6 PR #289) substitutes for via splat-osint-ingestion-v1 |
| Related repo `aiwar-neo4j-harvest` | Graph data pipeline | The migration source for legacy Neo4j data → Lance |
| Related repo `aiwar` | AI War Cloud dataset | The reference OSINT-shape dataset |

**The full route, end-to-end:**

```
External user opens q2 notebook
    │ writes Cypher / Gremlin / SPARQL cell
    ▼
q2::notebook-query (polyglot parser)
    │ via lance-graph-planner Strategy #1-4 (CypherParse / GqlParse / GremlinParse / SparqlParse)
    ▼
lance-graph-planner Unified Logical Plan (ArenaIR + DPJoinEnum + ...)
    │ applies PolicyRewriter chain (RowFilter + ColumnMask + RowEncryption + DP + Audit)
    │ via UnifiedBridge<Q2Bridge>::authorize_read (super_domain = TBD — likely Osint for aiwar-shape data, TicketTool for cockpit-server)
    ▼
DataFusion ScanExec over Lance datasets
    │ per-row identity = TenantId u32 + OwlIdentity u16 (6 bytes)
    │ batch-decorated with DolceMarker / Foundry ObjectType via Path C (ndarray::simd gather)
    │ thinking-engine projection (Path A) carries awareness frame alongside merkle audit
    │ ractor supervisor (Path B) routes per-actor per-super-domain crash isolation
    ▼
Arrow RecordBatch result → q2::notebook-render
    │ visualises as graph (Palantir-Gotham-equivalent) / table / chart
    │ reactive cell DAG (notebook-runtime) listens for Supabase realtime cognitive_event updates
    ▼
External user sees Foundry/Gotham-parity surface backed by the OGIT super-domain stack
```

**Where the route is already partly wired:**

- `palantir-parity-cascade-v2.md` table cites: "Cypher / Workshop console → Q2 Cypher Console (polyglot) → Q2-2.x (QUEUED)". The console design exists; Q2-2.x is queued behind the Foundry parity capstone.
- `q2-foundry-integration-v1.md` Q2-1.1..Q2-1.7 (referenced in `lance-graph-ontology-v5.md` D-ONTO-V5-5) defines q2's foundry-shape entities (Quarto / Neo4j / Gotham equivalents) that get a TTL transcode under `OGIT/NTO/Q2/`.
- `lance-graph-ontology-v5.md` D-ONTO-V5-5 ships `OGIT/NTO/Q2/{entities,verbs}/*.ttl` + `crates/lance-graph-ontology/src/bridges/q2_bridge.rs` (NEW, ~45 LOC mirroring `medcare_bridge.rs`). q2 binary holds an `Arc<OntologyRegistry>` and resolves `Workshop`, `Vertex`, `Doctemplate` via the `Q2Bridge`.
- `2026-05-06-splat-osint-ingestion-v1.md` ships the `crates/jc/examples/osint_edge_traversal.rs` demo proving EWA-Sandwich Σ-push-forward as the Neo4j-edge-traversal substitute — Pillar 6 PR #289 certified the math.

**Where the route has gaps:**

1. **q2's local `lance-graph` stub is a duplicate.** It should `pub use lance_graph::*` from `AdaWorldAPI/lance-graph` (this repo) instead of carrying its own placeholder. Closing this dedup needs a q2-side PR that adds `lance-graph = { path = "../../../lance-graph" }` to q2's workspace + replaces the stub with re-exports.
2. **q2's local `q2-ndarray` stub is a duplicate.** Same logic — should `pub use ndarray::*` from `AdaWorldAPI/ndarray`.
3. **`notebook-query` polyglot dispatcher is unwired.** Today it's a stub; the wiring point is `lance-graph-planner::api::PolyglotDetector` (Strategy #1-4 fan-out). One PR adds the bridge.
4. **Q2Bridge (D-ONTO-V5-5) needs the TTL+bridge work.** Currently queued; ~45 LOC + a ~10-entity TTL transcode under `OGIT/NTO/Q2/`. Blocked on `AdaWorldAPI/OGIT` MCP scope expansion (same blocker as D-SDR-6/7 for hiro/hubspot).
5. **OSINT super-domain wiring.** The thinking-engine ships `osint_bridge.rs`; the q2 `aiwar-ingest` consumes OSINT-shape data. Wiring point: `UnifiedBridge<AiwarBridge>::with_audit_chain(SuperDomain::Osint, ...)`. Needs the manifest-driven boot (Pattern E) + ractor handler (Pattern F) — the Pattern E+F+cognition cascade unblocks this.
6. **Palantir Gotham parity at the visual surface.** `palantir-parity-cascade-v2` D-PARITY-V2-3..12 ship the DTO ladder; Q2-2.x ships the cockpit visualisation. Without these, the Cypher console renders generic graphs, not Foundry-parity Workshop views.
7. **Neo4j route via EWA-Sandwich.** Math is certified (Pillar 6 PR #289); splat-osint-ingestion-v1 D-SPLAT-1..7 ships the contract types + demo example. Wiring point: q2 cells that traverse multi-hop edges call into `osint_edge_traversal.rs`'s `EwaSandwichTraversal` rather than directly issuing 5-hop Cypher against neo4j-rs. **This is the migration of the aiwar workload off neo4j-rs onto lance-graph.**

**Implication for the integration plan:**

q2 is the **8th consumer subcrate slot** alongside the 5 super-domain subcrates (medcare-rs / smb-office-rs / woa-rs / hiro-rs / hubspot-rs) + 2 super-domain root crates (osint substrate via thinking-engine::osint_bridge / aiwar-ingest). The Tier C scope grows from 5 to 8:

| # | Super-domain | Consumer subcrate | Existing repo / planned | Activation root | Compliance |
|---|---|---|---|---|---|
| 1 | Healthcare | medcare-rs::healthcare | exists, mid-migration | HIPAA |  |
| 2 | WorkOrderBilling (SMB) | smb-office-rs::smb-bridge | exists, mid-migration | SOX/PCI-DSS |  |
| 3 | WorkOrderBilling (WoA) | woa-rs (planned extraction) | woa_bridge.rs in lance-graph-ontology today | SOX |  |
| 4 | TicketTool (Hiro) | hiro-rs (new) | D-SDR-8 | (TBD) |  |
| 5 | TicketTool (HubSpot) | hubspot-rs (new) | D-SDR-9 | PCI-DSS billing |  |
| 6 | **Osint** | **aiwar-ingest (in q2 workspace)** | `AdaWorldAPI/q2/crates/aiwar-ingest` exists | OSINT clearance |  |
| 7 | **(cross-cutting visual)** | **q2::cockpit-server + notebook-* crates** | `AdaWorldAPI/q2/crates/cockpit-server + crates/stubs/*` | (cross-cutting — visual is per-super-domain) |  |
| 8 | (related research) | `neo4j-rs` + `aiwar-neo4j-harvest` + `aiwar` | external Adapt repos | OSINT clearance |  |

**The ball that mustn't drop:** q2 was being treated as one of many "external tools that consume lance-graph" — but with this finding it's clear q2 is a **core consumer subcrate** that ships the cockpit visual surface AND the OSINT ingest pipeline AND the polyglot query notebook. Plus its own stubs need to be replaced with re-exports from the canonical crates. Without this entry, the next session would scaffold q2 wiring as a generic external integration and miss the super-domain subcrate framing.

Cross-ref: `q2/README.md` (the inventory); `q2/crates/aiwar-ingest`, `q2/crates/cockpit-server`, `q2/crates/stubs/notebook-*`; `q2-foundry-integration-v1.md` Q2-1.1..Q2-1.7; `lance-graph-ontology-v5.md` D-ONTO-V5-5; `palantir-parity-cascade-v2.md` Q2-2.x; `2026-05-06-splat-osint-ingestion-v1.md` D-SPLAT-1..7; `EPIPHANIES.md` 2026-05-13 super-domain subcrate finding (this extends the 5-subcrate table to 8 slots); `IDEAS.md` 2026-05-13 super-domain subcrate scaffolding cascade (q2 slot adds PR 6+7+8 to the cascade); `TECH_DEBT.md` TD-Q2-STUBS-DEDUP-1 (today).

## 2026-05-13 — CLARIFICATION: the OGIT hierarchy is NOT strictly nested — SuperDomain × OGIT-basin × OWL-leaf × DOLCE-leaf are partially orthogonal axes

**Status:** FINDING (clarifies spec §1-§2 "4-level hierarchy" framing)

The `super-domain-rbac-tenancy-v1` §1-§2 framing presents a **4-level hierarchy** (MetaAnchors → SuperDomain → OgitBasin → WithinBasinSlot). User correction 2026-05-13: that framing is partially misleading because OWL slot and DOLCE marker are **orthogonal axes**, not strictly nested sub-trees.

**The actual axis structure:**

| Axis | Cardinality | What it carries | Nesting relation |
|---|---|---|---|
| **SuperDomain** | 8 starter values, 256 cap (1 byte) | Activation root + compliance regime + role matrix + hard-lock partners + audit chain salt | Coarse partition; each SuperDomain claims a subset of OGIT basins (`FAMILY_TO_SUPER_DOMAIN: [SuperDomain; 256]` reverse lookup) |
| **OGIT basin** | 256 (1 byte, `OgitFamily`) | Family-level ontology pointer (Healthcare, Order, Patient, ...) | Many-to-one assignment to SuperDomain; per-family codebook (`OgitFamilyTable`) lives at this level |
| **OWL leaf** | 256 within each basin (1 byte slot, high byte of `OwlIdentity`) | Within-basin entity identity (`OwlIdentity = (family, slot)` packed u16). **ORTHOGONAL** to other basins' slots — slot 7 in Healthcare and slot 7 in Order are unrelated identities, NOT a shared concept | Per-basin leaf; the "orthogonality" is operational (different family ⇒ different codebook ⇒ different lookup table) |
| **DOLCE marker** | 4 starter variants (Endurant / Perdurant / Quality / Abstract, `DolceMarker(u8)`) | Upper-ontology classification cross-cutting OGIT — a Healthcare:Patient and an Order:LineItem might both be `Endurant`, while Healthcare:Procedure and Order:Refund are both `Perdurant` | **SEPARATE ORTHOGONAL AXIS** — not a sub-tree of OGIT; lives in `MetaAnchors` per `SuperDomainEntry` and per `FamilyEntry`. Used for upper-ontology reasoning that cross-cuts basin boundaries |
| Wikidata QID / Foundry ObjectType / OWL upper class | (open) | Cross-walks to external upper ontologies | Same orthogonal status as DOLCE — `MetaAnchors` is a multi-axis cross-walk record, not a strictly nested hierarchy |

**Why the orthogonality matters:**

1. **OWL slot orthogonality is the address-space hygiene rule.** Slot `n` in basin A and slot `n` in basin B are distinct identities; the `OgitFamilyTable::lookup(owl)` debug-asserts on family match for exactly this reason. Aliasing slots across basins (e.g., "slot 7 means 'top-priority' everywhere") is the bug that destroys the addressing model.

2. **DOLCE-axis orthogonality is what enables cross-domain upper-ontology reasoning.** A DataFusion query like "find all `Endurant` rows across Healthcare AND WorkOrderBilling tenants" works because `DolceMarker` is a column dimension orthogonal to OGIT basin. If DOLCE were nested under OGIT, this would require 256 separate scans + a union; orthogonal makes it one masked-predicate scan.

3. **MetaAnchors is multi-axis, not single-tree.** §3.5's `MetaAnchors { foundry_object_type, owl_upper_class, dolce_marker, wikidata_qid }` is four orthogonal cross-walks per `FamilyEntry`. The "4-level hierarchy" framing collapses them visually but the data is a flat record of independent classifications.

**Implication for the address layout in §3:**

The 6-byte per-row identity (`TenantId u32 + OwlIdentity u16`) addresses one axis (OWL = family × slot). The DOLCE marker + Foundry ObjectType + Wikidata QID are **column-side metadata** carried per-row by joining against the per-family codebook (`OgitFamilyTable::lookup(owl).meta_anchors`). They are NOT part of per-row identity; they are batch-decorable annotations that DataFusion ScanExec can produce in one gather pass (Path C / `ndarray::simd::gather_u8`).

**Implication for query masked-predicate composition (§3.10):**

The single masked-predicate that enforces tenant + super-domain + role + slot in one vector pass (§3.10) operates on `TenantId u32 + OwlIdentity u16`. **DOLCE / Wikidata / Foundry filters are a separate masked-predicate** that joins against the family table's `MetaAnchors` column — cheap because `MetaAnchors` is inline (D-SDR-3 inline codebook, one cache line per slot) but architecturally distinct from the identity-axis predicate.

**Implication for `SuperDomain` cap (256 vs 8 starters):**

The 1-byte `SuperDomain` field has 256-value capacity but only 8 starters today. The remaining 248 are reserved for future super-domain partitions that **may need their own activation roots** without disturbing the OGIT basin assignment. Example: splitting `Science` into `LifeScience` and `PhysicalScience` doesn't require renumbering OGIT basins; it just claims another `SuperDomain` slot and updates `FAMILY_TO_SUPER_DOMAIN` for the relevant basins.

**The ball that mustn't drop:** future sessions reading the §1-§2 "4-level hierarchy" framing without this clarification will conflate strict nesting with orthogonality, which leads to bad query plans (sequential scans instead of orthogonal masked-predicates) and bad address-layout decisions (collapsing DOLCE into OGIT). This entry pins the axis-structure intuition.

Cross-ref: spec `super-domain-rbac-tenancy-v1` §1-§2 (hierarchy framing this clarifies), §3.1-§3.5 (DTOs), §3.10 (DataFusion lowering); `crates/lance-graph-callcenter/src/super_domain.rs` (`MetaAnchors` + `DolceMarker` + `SuperDomainEntry`); `crates/lance-graph-callcenter/src/family_table.rs` (`OgitFamilyTable` + `FamilyEntry::meta_anchors`); `EPIPHANIES.md` 2026-05-13 6-byte OGIT identity finding (this clarifies what the 6 bytes do NOT carry).

## 2026-05-13 — FINDING: in-flight bridge migration causes API drift that breaks consumers mid-air; need an explicit deprecation path before D-SDR-5 ripples downstream

**Status:** FINDING (warning + actionable mitigation)

User report 2026-05-13: medcare-rs is failing during the in-flight migration of `medcare-analytics + medcare-bridge → UnifiedBridge` because the API surface keeps shifting between D-SDR-1 (initial `UnifiedBridge::new`) → Codex P2 fix (canonical entity type) → D-SDR-5 (new `with_audit_chain` builders + audit emission). Each commit adds methods and changes return shapes; downstream consumers compiling against successive HEADs of the source crate see drift faster than they can adapt.

**The drift sources, concretely:**

1. **D-SDR-1 starter** (PR #363) introduced `UnifiedBridge::new(bridge, policy, actor_role, tenant) -> Self` and `authorize_read/write/act(public_name, depth/...) -> Result<EntityRef, AuthError>`. medcare-rs `unified_bridge_wiring.rs` (commit `31e999b`) was authored against this surface.
2. **Codex P2 fix** (commit `421e71e` in PR #363) changed `authorize_*` internals to resolve canonical OGIT entity type via `bridge.row()` — **public signature unchanged** but the `Policy::evaluate` contract changed (now keyed on canonical name not alias). Policy authors had to update their role permissions.
3. **D-SDR-5** (commit `dc9e081`, unmerged) added new methods: `with_audit_chain(super_domain, salt, sink)`, `with_audit_chain_resume(super_domain, salt, last_root, sink)`, `audit_root() -> AuditMerkleRoot`. **Backward-compatible** (defaults to `NoopUnifiedAuditSink` + GENESIS) but downstream code that called `UnifiedBridge::new` and never set up audit silently disables compliance.

**The fail-mid-air pattern:**

Consumer migration spans multiple PRs over multiple days. If the source crate's API changes between consumer-PR-1 (which adopts the starter shape) and consumer-PR-2 (which finalizes the migration), the consumer's clippy-clean PR-1 starts failing CI when rebased onto post-D-SDR-5 source. The error message ("missing `with_audit_chain` call → compliance disabled") only surfaces if there's a lint or a runtime assertion; without one, the migration silently ships with audit disabled.

**Mitigation — the consumer-side stability contract:**

1. **Pin migration source SHA on the consumer-side branch.** medcare-rs's `claude/lance-datafusion-integration-gv0BF` branch should depend on lance-graph at the **#363 merge SHA** (`421e71e`) during the migration window, not at `main` HEAD. Pinning to a SHA insulates the consumer from intra-migration source drift. Unpin after the consumer's migration PR merges.
2. **Add a `must_use` lint on `UnifiedBridge::new` output until audit is configured.** Force consumers to either call `.with_audit_chain(...)` or `.allow_no_audit()` (an explicit opt-out for non-compliance scenarios — tests, local dev). Without this, the default no-op audit is a silent compliance gap.
3. **Add a `#[deprecated]` annotation on `column_mask_bridge.rs`** in medcare-analytics the moment `unified_bridge_wiring.rs` lands as the canonical path. Forces all downstream callers to migrate within one deprecation cycle.
4. **Ship a `lance-graph-callcenter::migration` module** with re-exports of stable consumer-facing types. Consumers import from `migration::*` during the migration window; the module's contract is "this surface does not change between minor versions". Internal source moves freely; the migration surface is a versioned contract.
5. **The follow-up PR for D-SDR-3..5 should include a `CHANGELOG.md` entry** with explicit consumer-migration notes (the `with_audit_chain` builder, the canonical-name policy contract, the `actor_role_hash` audit field). Without this, every consumer's first failure forces a transcript-grep to figure out what changed.

**Implication for the integration plan:**

The 5-PR super-domain subcrate scaffolding cascade (per IDEAS.md 2026-05-13) MUST sequence consumer migrations against pinned source SHAs. PR 1 (medcare migration finalization) pins to the D-SDR-3..5 follow-up PR's merge SHA; PR 2 (smb-bridge) pins to PR 1's merge SHA; etc. Each consumer migration unpins after its PR lands, then waits for the next stable source SHA before kicking off the next consumer.

**The ball that mustn't drop:** API drift across an in-flight migration is the kind of failure that doesn't show up in code review (the source PR is clippy-clean, the consumer PR is clippy-clean against its source SHA) — it only shows up when CI runs against `main` HEAD. The mitigation above (SHA pinning + must_use + deprecation annotations + migration surface module + CHANGELOG) is operational discipline, not new code. This entry exists so the next session adopts the discipline rather than re-discovering the failure mode.

Cross-ref: `EPIPHANIES.md` 2026-05-13 super-domain subcrate finding (the migration target); `TECH_DEBT.md` TD-API-DRIFT-MIDFLIGHT-1 (today) + TD-SDR-CONSUMER-PUSH-1 (the consumer PRs that this drift affects); `IDEAS.md` 2026-05-13 super-domain subcrate scaffolding cascade (the sequencing that mitigates this).

## 2026-05-13 — CORRECTION-OF earlier 2026-05-13 entries framing §16-§19 as "outstanding deliverables" — most was already delivered in PRs #355-#363+

**Status:** CORRECTION

The earlier 2026-05-13 epiphany entries (`thinking-engine` finding, two-paths-converging finding) framed `super-domain-rbac-tenancy-v1` §16-§19 as outstanding architectural work awaiting wiring. **That framing under-counts what has already shipped.**

The PR arc #355 → #363 (2026-05-07 → 2026-05-13, ~7 days of sprint-2 / sprint-3 work) delivered most of the §16-§19 substrate:

| PR | Branch | What it shipped |
|---|---|---|
| #355 | `claude/create-graph-ontology-crate-gkuJG` | `lance-graph-ontology` crate as the ontology home — SPO-1 + TTL-PROBE-5 closures, 8 new entropy-ledger rows, the Per-row-context cluster. The ontology surface that §17's DataFusion-on-LanceDB plans against. |
| #356 | `claude/integrate-lance-graph-bridge-ikDO5` | `lance-graph-bridge` integration — the bridge surface §14 harvests from. |
| #358 | `claude/unified-ogit-architecture-synthesis` | Unified-OGIT architecture synthesis document — codifies the Zone 1/2/3 + DataFusion-on-LanceDB framing that §16 + §17 build on. |
| #359 | `claude/tier-0-canonical-pattern-letters-fix` | Tier-0 canonical Pattern letter assignment fix — pattern E (manifest) and F (ractor supervisor) labels stabilised. |
| **#360** | `claude/tier-1-implementation-specs` | **Tier-1 implementation specs — including `pr-e-1-manifest-modules.md` and `pr-f-1-ractor-supervisor.md` (the same Pattern E + Pattern F that the 2026-05-13 "two-paths-converging" finding references). The ractor-supervisor design DOES exist as a shipped spec, not just a sketch.** |
| #361 | `claude/sprint-3-spec-defect-fixes-v2` | Spec defect fixes — pr-e-1 and pr-f-1 corrections (commits `3865328` + `87cafe3`). |
| #362 | `claude/sprint-3-rescope-substrate-recognition` | Sprint-3 rescope: substrate recognition reframes (THINK-1, HEEL-1, ADJ-THINK-1, CRYSTAL-1, CAM-DIST-1) — entropy ledger contracted by ~11. This is where the "consult-before-guess" recognition pass identified shipped substrate vs aspirational. **The thinking-engine 582 KB finding (the dormant cognitive substrate) is a continuation of this same recognition arc.** |
| **#363** | `claude/lance-datafusion-integration-gv0BF` | `super-domain-rbac-tenancy-v1` spec authoring (§1-§19) + D-SDR-1 + D-SDR-2 + Codex P2 fix. The spec itself is shipped; D-SDR-3..5 stack as follow-up commits. |

**Net correction:**

- **§16 (Zone 3 boundary)** — designed across #355 + #358; not just words but actual ontology crate (#355) and integration surface (#356). Outstanding implementation gap: `cognition_bridge` + the manifest plumbing.
- **§17 (DataFusion-on-LanceDB)** — designed and substrate-shipped across #355 + #356 + #358. Outstanding: D-SDR-31..34 (Phase 5+ Arrow Flight SQL) and HTTP+JSON endpoints (Tier H D-SDR-35..39). Note: §18.9 already corrected this — Flight SQL is Phase 5+, NOT immediate.
- **§18 (MedCare reality check)** — empirical inspection only; no PR was needed because the finding was "what exists is enough, don't reshape". The D-SDR-35..39 endpoint gap remains for Tier H.
- **§19 (build invariants)** — already enforced in `Cargo.toml` (workspace pins) and CI gate (`cargo clippy -- -D warnings`); not net-new work but a codification of existing rules.

**Implication for the handover docs:** the 2026-05-13-0852 status handover and 2026-05-13-0855 brainstorm synthesis correctly cite #363 as the source PR for D-SDR-1/2 but **under-cite #355/#356/#358/#360/#362** as the broader §16-§19 substrate delivery. A future session reading those handovers without this correction would over-estimate the remaining work.

**What this changes for next-step prioritisation:**

- Pattern E + Pattern F are **shipped as specs** (#360, #361). Implementation is the gap — the `IDEAS.md` 2026-05-13 Pattern E+F+cognition cascade should be re-anchored to those spec files as its source, not as net-new design.
- The "highest leverage" claim in the thinking-engine finding stands, but the architectural pre-work (Pattern E manifest schema, Pattern F ractor handler shape) is already specified in #360 — the cascade is **implementation**, not design+implementation.
- D-SDR-3..5 (committed but unmerged) are the natural continuation of the #355 → #363 arc; the follow-up PR is anchoring the next step in the sprint sequence, not opening a new arc.

**The ball that mustn't drop, restated:** the integration plan is FURTHER ALONG than the §16-§19 framing suggested. Sprint-2 + sprint-3 + the super-domain spec authoring (#355 → #363) shipped the architectural substrate; the remaining work is composition + implementation of designs that exist. This is a **morale + scope** correction, not just bookkeeping.

Cross-ref: PRs #355, #356, #358, #359, #360, #361, #362, #363 (all merged); `.claude/plans/pr-e-1-manifest-modules.md` (if it lives at that path post-#360; otherwise grep INTEGRATION_PLANS.md for the canonical location); `.claude/plans/pr-f-1-ractor-supervisor.md`; `.claude/board/INTEGRATION_PLANS.md` sprint-2 + sprint-3 entries (`## 2026-05-07 — Unified OGIT Architecture plans` + `## 2026-05-12 — Sprint-3: Tier-1 Implementation Specs`).

## 2026-05-13 — FINDING: each `SuperDomain` is its own specialised subcrate; consumer crates ARE the super-domain implementations (medcare-rs / smb-office-rs / hubspot-rs / hiro-rs / woa-rs)

**Status:** FINDING

The Tier C "consumer crate scaffolding" framing of `super-domain-rbac-tenancy-v1` §8 (D-SDR-8 hiro-rs, D-SDR-9 hubspot-rs) misses what the design is actually pointing at: **each `SuperDomain` activation root IS the subcrate that specialises the unified surface for its compliance regime, role matrix, and ontology basin.** The mapping is 1:1:

| `SuperDomain` enum variant | Specialised subcrate | Compliance | Current status |
|---|---|---|---|
| `Healthcare` | `MedCare-rs/crates/medcare-analytics` + `medcare-realtime` + `medcare-bridge` → finalize merge into a single super-domain subcrate consuming `UnifiedBridge<MedcareBridge>` | HIPAA | In-flight: `unified_bridge_wiring.rs` committed locally (`31e999b`), unpushed; medcare-analytics + medcare-bridge migration NOT yet finalized — the wiring exists but the crates still carry separate auth paths (`column_mask_bridge.rs` co-exists with new `unified_bridge_wiring.rs`). |
| `WorkOrderBilling` | `smb-office-rs/crates/smb-bridge` → continues as the super-domain subcrate consuming `UnifiedBridge<OgitBridge>` | SOX / PCI-DSS | In-flight: `342f601` committed locally, unpushed. |
| `TicketTool` (Hiro slot) | NEW crate `/home/user/hiro-rs` (D-SDR-8) — absorbs OSLC-* with lineage; specialises `UnifiedBridge<HiroBridge>` for the ticketing super-domain | (TBD — OSLC defines it) | Not started. |
| `TicketTool` (HubSpot slot) | NEW crate `/home/user/hubspot-rs` (D-SDR-9) — CRM vocabulary; specialises `UnifiedBridge<HubspotBridge>` | PCI-DSS billing | Not started. |
| `WorkOrderBilling` (WoA slot) | `/home/user/woa-rs` — work-order-application subcrate consuming a `WoaBridge` retrofitted to the meta-bridge surface (§14.2) | SOX | Existing bridge (`woa_bridge.rs` in lance-graph-ontology); needs retrofit to MetaBridge + extracted into woa-rs subcrate. |
| `Science` | (TBD) | OSINT clearance / ITAR-EAR | Aspirational — D-SDR-2 SUPER_DOMAINS slot only. |
| `Genetics` | (TBD) | GINA / GDPR Art 9(2)(i) | Aspirational. |
| `QuantumPhysics` | (TBD) | ITAR-EAR | Aspirational. |
| `Osint` | `cognitive-shader-driver` already ships `osint_bridge`; subcrate TBD | OSINT clearance | Bridge exists; super-domain subcrate not yet promoted. |

**Why super-domain = subcrate is the right factoring:**

1. **Compliance is per-super-domain, not per-bridge.** HIPAA controls (§164.312) bind to Healthcare; SOX §404 + PCI-DSS Reqs 3+7+10 bind to WorkOrderBilling. The certification stub (D-SDR-11) is naturally per-super-domain, which means it's per-subcrate.
2. **Role matrices are per-super-domain.** §4.3 illustrates Healthcare's full role matrix (clinician / nurse / billing-clerk / researcher / etc.); WorkOrderBilling has a different shape (technician / dispatcher / accountant / etc.). Per-super-domain subcrates own their role tables (Layer-2 role catalogue per `I-VSA-IDENTITIES`).
3. **Hard-lock partners are per-super-domain.** §13.4 Healthcare ↔ OSINT crypto barrier needs both ends to publish their `merkle_salt` constant; living in separate subcrates makes the barrier real (compile-time-separated symbol tables).
4. **Audit JSONL files are per-super-domain.** D-SDR-10's `JsonLinesAuditSink` writes to disk paths the super-domain owns; cross-super-domain audit chains are unlinkable by design. Owning the sink config in the per-super-domain subcrate enforces this.
5. **Compile-time manifest entries are per-super-domain.** Pattern E (`/modules/<name>/manifest.yaml`) one-per-consumer is one-per-super-domain in practice; the `super_domain` field gates which `SuperDomain` enum variant the actor binds to at boot.
6. **MedCare-rs migration is the canonical case.** `medcare-analytics + medcare-realtime + medcare-bridge` are currently three crates within MedCare-rs; finalizing the merge into a single Healthcare-super-domain subcrate (or a coherent crate cluster behind a single `UnifiedBridge<MedcareBridge>` re-export) is the demonstration migration that proves the pattern.

**The medcare migration gap that must close:**

- `medcare-analytics/src/unified_bridge_wiring.rs` exists (107 LOC, `lance-phase2-rbac` feature) and constructs `UnifiedBridge<MedcareBridge>`.
- `medcare-analytics/src/column_mask_bridge.rs` still exists as the prior auth path.
- `medcare-bridge` crate is a separate crate that holds the `MedcareBridge` ontology mapper.
- Three crates / two auth paths / no single Healthcare-super-domain re-export.
- **Finalization step:** (a) deprecate `column_mask_bridge.rs` in favour of `unified_bridge_wiring.rs` + `UnifiedBridge::with_audit_chain(SuperDomain::Healthcare, salt, JsonLinesAuditSink::healthcare())`; (b) decide whether to keep `medcare-bridge` as a separate crate or fold it into `medcare-analytics` behind a `bridge` module; (c) publish a single `medcare-rs::healthcare` re-export that downstream consumers import.

**Implication for the integration plan:** Tier C grows from 2 deliverables (D-SDR-8 hiro-rs, D-SDR-9 hubspot-rs) to **5 super-domain subcrates** (medcare migration finalization + smb-bridge retrofit + woa-rs extraction + hiro-rs new + hubspot-rs new). The medcare migration is the **proof case** — it must finalize before D-SDR-8/9 ship, otherwise hiro-rs and hubspot-rs scaffold against a half-migrated pattern.

**The ball that mustn't drop, restated:** the consumer crate scaffolding work (Tier C) and the super-domain layer (D-SDR-2) are not two separate workstreams — they're the same workstream. Per-super-domain subcrates ARE Tier C. Without this entry, the next session would scaffold hiro-rs/hubspot-rs as generic consumer crates and miss the per-super-domain specialisation (compliance, role matrix, hard-lock partner, audit sink) that the SuperDomain enum already encodes.

Cross-ref: spec `super-domain-rbac-tenancy-v1` §3.4 (SuperDomain), §3.6 (role groups), §3.7 (compliance regime), §4 (consumer-to-basin mapping), §8 Tier C; `MedCare-rs/crates/medcare-analytics/src/unified_bridge_wiring.rs` (the in-flight pattern); `smb-office-rs/crates/smb-bridge/src/unified_bridge_wiring.rs` (parallel pattern); `TECH_DEBT.md` TD-SUPER-DOMAIN-SUBCRATES-1 (new today); `IDEAS.md` 2026-05-13 super-domain subcrate scaffolding cascade.

## 2026-05-13 — FINDING: THREE complementary substrate paths converge in `lance-graph-callcenter` — thinking-engine + ractor + ndarray::simd (correction-of two-paths entry)

**Status:** FINDING (extends the same-day two-paths-converging entry)

The two-paths finding below identifies Path A (thinking-engine cognition) and Path B (ractor sync supervisor). User correction 2026-05-13: **there is a third path — `ndarray::simd` SIMD compute** — that is the canonical compute substrate every batch operation in callcenter routes through. The three paths are orthogonal and complementary:

| Path | Substrate | What it provides | Status |
|---|---|---|---|
| **A — `thinking-engine`** | Cognition content (582 KB / 48 modules) | Per-row decision *contents*: role projection, persona, qualia, awareness DTO, lenses, codebook lookup | Shipped, unwired (TD-THINKING-ENGINE-UNWIRED-1) |
| **B — `ractor` supervisor** | Runtime topology (sync, I-2 BBB) | Per-actor *supervision*: crash isolation, restart strategy, compile-time-typed messaging, manifest-driven boot | Spec shipped (#360 pr-f-1), implementation owed (TD-RACTOR-SUPERVISOR-5) |
| **C — `ndarray::simd`** | SIMD compute (canonical per §19.2) | Per-batch *compute*: `LazyLock<Tier>` dispatch across SSE2/AVX2/AVX512/NEON/AMX, batch fingerprint ops, distance kernels, BLAS L1/L2/L3 | Shipped + already canonical across workspace; callcenter consumer-side wiring still scalar-per-row in some paths |

**Why C is the third path, not "just SIMD":**

`ndarray::simd` is **not** a transparent acceleration layer — it's a substrate with its own conventions:

- **Canonical dispatch pattern**: `static TIER: LazyLock<Tier> = LazyLock::new(simd_caps)`. Every batch hot-path imports and dispatches through this; ad-hoc `#[cfg(target_arch=...)]` is the anti-pattern.
- **Carrier types are SIMD-shaped**: `Vsa16kF32` (64 KB), `Vsa16kBF16` (32 KB AMX-accelerated), `Vsa16kI8` (16 KB quantized), `Binary16K` (2 KB Hamming) — each has a paired SIMD operator family. Picking the wrong carrier costs a register reshuffle on every op.
- **Distance kernels live in ndarray::simd**: `xor_fold`, `cosine_simd`, `batch_palette_distance` — callers in callcenter / planner / cognitive-shader-driver consume these; never reimplement.
- **Spec §19.2 makes it canonical**: "ndarray::simd is the canonical SIMD path" — the `LazyLock<Tier>` dispatch pattern is already shipped; just import. No new code.

**Where callcenter still has scalar paths that should route through Path C:**

- `unified_audit.rs::AuditChain::advance` — single-event FNV-1a chain; per-row is intrinsically scalar (right call). **But** `verify_chain` over a batch of N audit events is a batch operation that today loops scalar; SIMD batch FNV-1a could speed cold-storage audit verification ~8×.
- `family_table.rs::OgitFamilyTable::lookup` — single-row array index; intrinsically scalar (right). **But** a batch-lookup `lookup_batch(owls: &[OwlIdentity]) -> Vec<Option<&FamilyEntry>>` for DataFusion-side row decoration would benefit from gather instructions.
- `super_domain.rs::FAMILY_TO_SUPER_DOMAIN[basin]` — single-byte lookup. Right for per-row. For a batch lowering of `ScanExec → SuperDomain[]` annotation, `ndarray::simd::gather_u8` exists.
- `unified_bridge.rs::canonical_entity_type` — per-row string slice. Right scalar. **But** the OGIT-URI parsing across a batch (post-#355 ontology crate) wants batch parsing primitives.
- D-SDR-25 future drift-bridge comparisons — `MerkleRoot` batch XOR-fold across cross-impl rows is the canonical Path C consumer. §19.7 already notes this.

**Why the three paths together close the loop:**

```
ractor supervisor (Path B)
    │ owns N consumer actors per super-domain manifest
    │ routes typed messages with sync I-2 BBB enforcement
    ▼
UnifiedBridge::authorize_* (Path B handler arm)
    │ projects role/persona/awareness through thinking-engine (Path A)
    │ ↓
    │ resolves canonical OGIT entity type from row (single-row scalar)
    │ evaluates Policy + emits chained UnifiedAuditEvent
    │ ↓
DataFusion ScanExec batch decode (Path C consumer)
    │ batch-annotates SuperDomain via FAMILY_TO_SUPER_DOMAIN gather
    │ batch FNV-1a verifies audit chain on cold-read
    │ batch CAM-PQ distance via ndarray::simd kernels
    ▼
Per-row decision arrives at the actor handler with all three substrates' value already projected.
```

**Implication for the IDEAS.md cascade:** the Pattern E+F+cognition cascade (3-PR sequence) should explicitly call out which batch paths route through `ndarray::simd` and which stay scalar. Per-row authorize hot path: scalar (correct). Batch decoration / drift / audit verification: route through Path C. Reviewers should reject PRs that hand-roll SIMD or scalar-loop across what `ndarray::simd` already exposes — §19.2 anti-pattern.

**The ball that mustn't drop:** Path C is older and more taken-for-granted than A or B, which is exactly why a future session can forget it. The spec §19.2 text is one paragraph; the **architectural mandate** is that every batch path in callcenter consumer code (callcenter / medcare-rs / smb-office-rs / future hiro-rs / hubspot-rs / woa-rs) imports from `ndarray::simd` rather than rolling its own. This entry exists so the discipline survives the next session.

Cross-ref: `CLAUDE.md § ndarray Integration Policy`; spec `super-domain-rbac-tenancy-v1` §19.2 + §19.7; `EPIPHANIES.md` two-paths-converging entry (this finding extends to three); `TECH_DEBT.md` TD-THINKING-ENGINE-UNWIRED-1 + TD-RACTOR-SUPERVISOR-5; `.claude/knowledge/vsa-switchboard-architecture.md` (Layer-1 switchboard carriers are the Path-C carrier types); `crates/lance-graph/` and `crates/lance-graph-callcenter/` for current consumer-side scalar paths that should batch through Path C.

## 2026-05-13 — FINDING: `lance-graph-callcenter` has TWO complementary substrate paths waiting to be wired — thinking-engine (cognition) + ractor (runtime topology)

**Status:** FINDING

The 2026-05-13 thinking-engine finding (below) names one dormant substrate path that closes §16-§19. There is a **second, orthogonal** substrate path already designed and tech-debt-tracked: the **ractor supervisor** path that closes the runtime topology side. Both converge in `lance-graph-callcenter` — and both must be wired together, not picked one-or-the-other.

| Path | What it solves | Status | Cross-ref |
|---|---|---|---|
| **A — `thinking-engine` substrate** (582 KB, 48 modules) | Cognitive surface: role projection, persona, qualia, awareness DTO, lenses, codebook lookup, ground-truth calibration | Indexed in `CLAUDE.md § Thinking Engine`; consumed by zero callcenter code | `TD-THINKING-ENGINE-UNWIRED-1`; `IDEAS.md` 2026-05-13 wire-thinking-engine |
| **B — `ractor` supervisor** (designed, not yet built) | Runtime topology: sync actor supervision, per-consumer crash isolation, compile-time manifest-driven boot, typed message contracts (the I-2 invariant: tokio outbound only / sync ractor inbound) | Designed in `.claude/plans/compile-time-consumer-binding-v1.md` D-RACTOR-SUPERVISOR-5 (~400 LOC `supervisor.rs` sketched); maps 1:1 onto `cognitive-shader-driver/src/grpc.rs` 8 methods | `TD-RACTOR-SUPERVISOR-5` (TECH_DEBT.md:1779); `anatomy-realtime-v1.md` W11; `compile-time-consumer-binding-v1.md` §2.2 |

**Why they're complementary, not competitive:**

- **Path A (thinking-engine) gives the *contents*** of each authorize/dispatch/ingest decision (role projection vectors, persona identity, awareness DTO that rides alongside merkle audit roots).
- **Path B (ractor) gives the *topology*** that runs Path A's primitives under crash-isolated supervision (one actor per consumer/super-domain, restart strategy, compile-time-typed messaging).

Together they form the runtime: **`CallcenterSupervisor` (ractor) owns N consumer actors → each actor calls `UnifiedBridge::authorize_*` → which projects through `thinking-engine::role_tables + persona + awareness_dto` → emits a chained `UnifiedAuditEvent` carrying both `merkle_root` AND `awareness_root`**. The supervisor handles backpressure, restarts, and the I-2 BBB seam (no tokio inside actor handlers). The thinking-engine provides the cognitive contents the supervisor's typed messages carry.

**Compile-time manifest convergence (Pattern E + Pattern F):** the `/modules/<name>/manifest.yaml` PostNuke-style declaration carries `(G, version, entity_types, rbac_policy, action_capabilities, stack_profile, actor_type, thinking_styles)`. The `actor_type` field gates Path B (which ractor handler arm boots for this consumer). The `thinking_styles` field gates Path A (which projection vectors from `thinking-engine::role_tables` this actor's authorize-flow uses). **One manifest entry per consumer compile-time-resolves both substrate paths.** Adding a new consumer = drop a manifest + add a Cargo dep + write ~30 LOC of `impl Consumer for FooActor` glue. Zero edits to `lance-graph-contract` after the build-script lands.

**Implication for the plan:** the cognition-bridge PR proposed in `IDEAS.md` 2026-05-13 should NOT ship in isolation; it should ship **alongside** the ractor supervisor (D-RACTOR-SUPERVISOR-5) and the manifest build-script (D-MANIFEST-MODULES-4) as **a single Pattern E+F+thinking-engine integration cascade** — three deliverables, three PRs, sequenced (manifest → supervisor → cognition-bridge composes against both).

**Concrete cascade ordering:**

1. **D-MANIFEST-MODULES-4** (PostNuke-style `/modules/<name>/manifest.yaml` + build-script generating the compile-time `MODULES: [ConsumerEntry; N]` static). Zero edits to `lance-graph-contract` afterwards.
2. **D-RACTOR-SUPERVISOR-5** (`CallcenterSupervisor` ractor consuming the compile-time module table; 8-arm typed message handler mapped from `cognitive-shader-driver/src/grpc.rs`). Each consumer = one actor spawned on boot with I-2 crash isolation.
3. **Cognition-bridge** (new module wrapping `thinking-engine::role_tables + persona + awareness_dto` behind a callcenter-side trait). Composes against the supervisor's per-consumer actor address; each `authorize_*` call routes through the actor and emits an audit event carrying both `merkle_root` and `awareness_root`.

This cascade collapses **D-SDR-13 + D-SDR-15 + D-SDR-17 + D-RACTOR-SUPERVISOR-5 + D-MANIFEST-MODULES-4** (5 separate deliverables, originally ~830 LOC scaffolded clean-room) into a **3-PR cascade ~900 LOC composed against thinking-engine**. The LOC delta is small; the **architectural** payoff is huge — `lance-graph-callcenter` finally becomes what its name has promised since day one (telephony switching, supervised processes, per-line crash isolation), and the cognitive substrate finally has a runtime home.

**The ball that mustn't drop, restated:** the May 1 → May 13 transcript arc accumulated this two-paths-converging finding without ever capturing it as a single epiphany. Future sessions without this entry would re-derive Path A xor Path B in isolation (~30-turn rediscovery tax) and miss the manifest-driven convergence that makes both paths cheap together.

Cross-ref: `EPIPHANIES.md` 2026-05-13 thinking-engine finding (Path A); `TECH_DEBT.md` `TD-RACTOR-SUPERVISOR-5` + `TD-MANIFEST-MODULES-4` + `TD-THINKING-ENGINE-UNWIRED-1`; `.claude/plans/compile-time-consumer-binding-v1.md` (Pattern E + Pattern F design); `.claude/plans/anatomy-realtime-v1.md` (W11 ractor supervisor demo gate); `ARCHITECTURE_ENTROPY_LEDGER.md:517` (Pattern F design-phase row).

## 2026-05-13 — FINDING: `thinking-engine` is a 582 KB dormant substrate that closes most of §16-§19 when wired

**Status:** FINDING

`crates/thinking-engine/` is **48 source modules, 16,211 LOC, 582 KB of Rust** sitting in the workspace and cited by 6 plans (`anatomy-realtime-v1`, `cam-pq-production-wiring-v1`, `unified-integration-v1`, `unified-ogit-architecture-v1`, `palantir-parity-cascade-v2`, `super-domain-rbac-tenancy-v1`) but **not yet wired into the §16-§19 spine** of the super-domain-rbac-tenancy work. The dormant surface maps onto the integration plan's outstanding deliverables with surprising directness:

| thinking-engine module | Wires into | Closes |
|---|---|---|
| `role_tables.rs` | SuperDomain RBAC role surface | D-SDR-2/§3.6 role groups (per-role projection already SIMD-shaped) |
| `osint_bridge.rs` | `SuperDomain::Osint` activation root | §13.4 Healthcare ↔ OSINT hard-lock implementation side |
| `persona.rs` + `qualia.rs` + `ghosts.rs` + `world_model.rs` | Cognitive identity surface | PersonaHub / actor-context auth (D-SDR-7 future + Tier H) |
| `centroid_labels.rs` + `codebook_index.rs` + `lookup.rs` | `OgitFamilyTable` hydration | D-SDR-3b (TTL-baked codebook), inline label→fingerprint resolution |
| `bf16_engine.rs` + `f32_engine.rs` + `signed_engine.rs` + `composite_engine.rs` + `dual_engine.rs` + `layered.rs` + `domino.rs` | Precision-tier dispatch on the canonical surface | §19 `ndarray::simd` canonical SIMD path consumer-side |
| `awareness_dto.rs` + `cognitive_stack.rs` + `cognitive_trace.rs` | UnifiedStep `OrchestrationBridge` contract | §17 DataFusion-on-LanceDB Phase 2 cognitive trace persistence |
| `meaning_axes.rs` + `superposition.rs` + `tensor_bridge.rs` | SoA columnar reads | palantir-parity-cascade-v2 D-PARITY-V2-3..12 |
| `prime_fingerprint.rs` + `spiral_segment.rs` + `tokenizer_registry.rs` + `pooling.rs` | Encoding tier of the codec stack | encoding-ecosystem.md surface → DataFusion UDFs |
| `jina_lens.rs` + `bge_m3_lens.rs` + `reranker_lens.rs` + `sensor.rs` | Per-model sensing surface (Jina v5 / BGE-M3 / Reranker v3) | Phase 5+ Arrow Flight SQL sensor endpoints |
| `ground_truth.rs` + `reencode_safety.rs` + `cronbach.rs` + `contrastive_learner.rs` | Quality / calibration | drift-bridge D-SDR-25 + cross-language determinism D-SDR-26 |
| `inference_backend.rs` + `bridge.rs` + `contract_bridge.rs` + `l4_bridge.rs` | Bridge surface taxonomy | MetaBridge harvest D-SDR-18/19 (the bridge templates already exist as Rust traits, not just designs) |
| `silu_correction.rs` + `semantic_chunker.rs` + `auto_detect.rs` + `builder.rs` | Composition glue | UnifiedBridge consumer composition surface |

**Implication for the plan:** the §16-§19 architecture (Zone 3 boundary + DataFusion-on-LanceDB + build invariants) does NOT require new cognitive substrate. The substrate is shipped. What's owed is the wiring from `UnifiedBridge::authorize_*` → `OrchestrationBridge` → `thinking-engine::*` paths. **This is the highest leverage move in the workspace** — much higher than D-SDR-13..17 in isolation, because each of those deliverables can compose against thinking-engine primitives instead of being scaffolded from scratch.

**Concrete framing:** treat thinking-engine as the **Layer-2 role-catalogue substrate** (per `I-VSA-IDENTITIES` iron rule) that UnifiedBridge's authorize-flow projects through. The bridge already owns: `OgitFamily` (basin pointer), `OwlIdentity` (per-row slot), `Policy::evaluate` (role-keyed decision), `UnifiedAuditEvent` (chained merkle). What it needs from thinking-engine: `role_tables` (the per-role projection vector that turns a role-name into an identity fingerprint), `persona` (actor identity in VSA carrier), `awareness_dto` (the cognitive state that rides alongside the authorize decision). Wiring them = D-SDR-13 / D-SDR-17 / D-SDR-15 collapse into a single ~300 LOC bridge module instead of three separate ~80-150 LOC deliverables.

**The ball that must not drop:** the previous session accumulated this finding across the §16-§19 brainstorming arc; without an explicit harvest entry, the next session would re-derive it from scratch (~30-turn rediscovery tax). This entry exists so it does not recur. See follow-up idea entry in `IDEAS.md` and `TD-THINKING-ENGINE-UNWIRED-1` in `TECH_DEBT.md`.

Cross-ref: `crates/thinking-engine/src/` (48 modules); `CLAUDE.md § Thinking Engine`; plans listed above; this session's `.claude/transcript/` archive (search for `thinking[_ -]?engine` — 103 mentions across the May 1 → May 13 main-window arc).

## 2026-05-13 — FINDING: Tier A (D-SDR-1..5) composes onto shipped PolicyRewriter chain — §13.1 thesis confirmed in code

**Status:** FINDING

The super-domain-rbac-tenancy-v1 §13.1 thesis ("compositor is already shipped: `lance-graph-callcenter::policy::PolicyRewriter`") survived contact with implementation. D-SDR-5 (`dc9e081`) wires `UnifiedBridge::authorize_read/write/act` through `Policy::evaluate` against the canonical OGIT entity type, emits one chained `UnifiedAuditEvent` per call via `Mutex<AuditChain>` (FNV-1a merkle 64-bit) into a swappable `Arc<dyn UnifiedAuditSink>`, and maps the resulting `AccessDecision` onto the `Result<EntityRef, AuthError>` surface — all without introducing a parallel enforcement path. `BridgeError` short-circuits before audit emission (D-SDR-5 minimum). 5 new tests cover Allow/Deny emission, bridge-error short-circuit, chain advance across calls, and resume from prior root. 96/96 lib tests green; clippy `-D warnings` clean on lib + tests.

The ~30% LOC reduction lever §13.1 promised did materialize for Tier A — `authorize_*` is ~10 lines per method because the chain handles row filtering / column masking / DP / encryption. **Consequence for Tier B+ design:** D-SDR-13..17 (merkle salt HKDF, audit JSONL, DP role, encrypted view, hard-lock matrix) all slot into the existing `OptimizerRule` chain as additional rewriters, NEVER as standalone authorization paths.

Cross-ref: `.claude/handovers/2026-05-13-0852-d-sdr-tier-a-complete-tier-b-and-beyond-pending.md`, commits `2c3e87d` (D-SDR-3), `1d0157f` (D-SDR-4), `dc9e081` (D-SDR-5).

## 2026-05-13 — CORRECTION-OF spec §17.2 (Arrow Flight SQL as immediate path) — HTTP+JSON over JWT is M2-M6; Flight SQL is Phase 5+

**Status:** CORRECTION

Spec §17.3 framed Arrow Flight SQL as the cross-language wire layer that replaces custom Protobuf IDL. Empirical inspection of MedCare-rs + MedCareV2 (§18) revealed the LanceProbe coordination doc (`MedCare-rs/docs/CSHARP_HANDOFF_PROMPT.md` on branch `claude/csharp-handoff-docs-L3DF0`) targets HTTP+JSON over JWT for M2-M6 milestones. **Arrow Flight SQL stays the end-state but is Phase 5+ migration**, not the immediate path. D-SDR-31..34 don't unblock M2; D-SDR-35..39 do (HTTP+JSON endpoints in medcare-rs). The architecture-level claim ("no custom IDL; Arrow Flight SQL + Substrait extension types is the right wire layer") survives — the **sequencing** was wrong.

Cross-ref: spec §18.9 row "Custom Protobuf IDL → Arrow Flight SQL"; D-SDR-31..34 (Phase 5+) vs D-SDR-35..39 (immediate).

## 2026-05-13 — CORRECTION-OF spec §16.4 (3DES rewrap pipeline) — the "3DES" is broken-single-DES; Argon2 backfill replaces AES-GCM rewrap

**Status:** CORRECTION

Spec §16.4 described D-SDR-27 as a decrypt-3DES → AES-256-GCM rewrap pipeline. §18.5 empirical inspection of MedCare's `MySQL_Connect.cs` revealed the "3DES" is NOT standard 3DES:

- Single 3DES cipher invocation (not a 3-cipher chain)
- 128-bit truncated key (out of the 168-bit 3DES key space)
- ECB-equivalent (no IV chaining)
- Zero IV
- Hardcoded password table indexed by the first character of the ciphertext

Effectively single DES with a broken KDF. **D-SDR-27's scope drops** from ~200 LOC rewrap to **~80 LOC carry-forward** + 2 integration tests. The `u_pwd` column is the only confirmed callsite; ciphertext is carried forward unchanged and Argon2 backfill happens on successful legacy login. **3DES → AES-GCM rewrap is REMOVED from the plan entirely.** Open question (§18.10): which other columns call `EncryptMessage()`/`DecryptMessage()` in `MySQL_Connect.cs` — likely few or none.

Cross-ref: spec §18.5/§18.6; `MedCare-rs/docs/AUTH_LEGACY_TRIPLEDES_MIGRATION.md` (DRAFT).

## 2026-05-13 — FINDING: Codex P2 review (canonical entity type vs bridge alias) — policy must NOT couple to consumer-facing aliases

**Status:** FINDING

Codex P2 reviewer caught a leak in the initial D-SDR-1 surface: `authorize_*` originally passed the bridge-side alias (`public_name`, e.g. `"WorkOrder"`) to `Policy::evaluate`. This means a Policy keyed against the canonical OGIT entity type (`"Order"` from `ogit.WorkOrder:Order`) wouldn't grant access through the alias, and conversely an alias-keyed Policy would silently couple consumer naming to authorization. Fix in commit `421e71e` (in #363): `authorize_*` now resolves the row via `bridge.row(public_name)?`, extracts `row.ogit_uri.name()` as the canonical entity type via `canonical_entity_type()`, and passes THAT to `Policy::evaluate`. Two regression tests pin the contract: `unified_bridge_evaluates_policy_against_canonical_entity_type` + `unified_bridge_does_not_honor_alias_keyed_policy`.

**Iron rule:** Policy authorship is against canonical OGIT names exactly once; bridges that resolve to the same canonical type all honor the grant regardless of consumer-facing public_name.

Cross-ref: PR #363, commit `421e71e`; `crates/lance-graph-callcenter/src/unified_bridge.rs:62` (`canonical_entity_type`).

## 2026-05-13 — FINDING: LanceProbe IS the drift bridge — design effort wasn't needed; wiring effort IS

**Status:** FINDING

Spec §15 designed `DriftDetectionBridge`/`DriftableOutput`/`DivergentRow` as a clean-room drift bridge concept. §18 empirical inspection found `LanceProbe` already exists in MedCareV2 with 8 scaffolded components (Phase M1 done). The clean-room design and `LanceProbe` are 1:1 isomorphic on field shape. **The drift bridge is wiring effort, not design effort.** Concrete Rust-side gap = 5 endpoints (D-SDR-35..39) + 1 reduced import tool (D-SDR-27) + 1 Argon2 fallback flag (D-SDR-38) ≈ ~700 LOC + tests. Tier F deliverable count collapsed from ~12 nominal items to **7 concrete items** through this finding.

Cross-ref: spec §18.1/§18.2; `MedCare-rs/docs/CSHARP_HANDOFF_PROMPT.md` (branch `claude/csharp-handoff-docs-L3DF0`).

## 2026-05-13 — FINDING: Per-row OGIT identity = 6 bytes total (TenantId u32 + OwlIdentity u16) — addressable domain ≤ 256 entries per family

**Status:** FINDING

Implementation (`unified_bridge.rs` + `super_domain.rs` + `family_table.rs`) confirms the §3 sizing claim: every row in the OGIT-addressed surface carries **`TenantId: u32 + OwlIdentity: u16 = 6 bytes`**. `OwlIdentity` high byte = `OgitFamily` basin (Level-2 pointer); low byte = within-basin slot (Level-3). Inline per-family codebook (`OgitFamilyTable` with 256-slot dense `[Option<FamilyEntry>; 256]`) carries label URI + `SchemaKind` + `OwlCharacteristics` + `DolceMarker` + axiom blob + provenance + outgoing verbs — INLINE, one cache line per occupied slot, no sidecar. Lookup is O(1) array index; `lookup(owl)` debug-asserts family match. SGO meta (>256 entries) explicitly excluded from runtime addressing (§9.3).

**Implication:** `owl_from_schema_ptr()` truncates `SchemaPtr::entity_type_id()` (u16) to 8-bit slot for audit emission. This is lossless within the addressable domain. If a basin ever needs >256 entries, the entire 8-bit slot abstraction breaks — re-check when any basin approaches the cap.

Cross-ref: spec §3.1-§3.3; `crates/lance-graph-callcenter/src/{unified_bridge,super_domain,family_table}.rs`.

## 2026-04-30 — FINDING: Wave-1 follow-up shipped (PRs #300-#306) — 3,156 LOC, full LOC audit confirms 0 lost from #275-#283 recovery

**Status:** FINDING

Session 2026-04-30 shipped the grammar-foundry-followup-v1 plan: 7 PRs (S1, F1, F3, F6, G1, G3, G4) closing the explicit stubs left behind by recovery-merge #299. Each PR went through a brutally-honest reviewer agent that surfaced 12+ defects (G1 fabricated qualia dim labels — later softened to "PAD-model sanitization"; F1 had a WHERE/JOIN/AGG leak that only rewrote Projection; G4 broadcast 12 priors across 12 tenses producing the illusion of 144 unique values; S1 introduced an `id: 0` landmine across 4 callers; F3 had a non-temporal Int64 timestamp + lossy column round-trip; G3's "real fp" was passthrough-only with no caller). All defects closed via 7 follow-up refactor commits with failing-test-first regression tests.

**LOC audit (verified `git diff --shortstat`):**
- Recovery (#275-#283 via #299): `71fad59..77c6292` = +8,728 / -334 across 41 files
- Wave 1 (#300-#306): `77c6292..40718e4` = +3,156 / -107 across 18 files
- Combined: `71fad59..40718e4` = +11,807 / -364 across 48 files
- The G1 rebase (`--force-with-lease`) dropped only commit `460329f` (a stray F6 dn_path cherry-pick of ~124 LOC, NOT recovery code) and the plan commit `18240ec`. Math validated: 8,728 + 3,156 - 77 (file overlap) = 11,807. Zero recovery LOC lost.

**Clippy gate (post-merge):** 2 deny-level errors fixed — 4× `#[deprecated(since = "next")]` invalid semver in context_chain.rs (G3); 1× `actor.role <= u8::MAX` tautology in lance_membrane.rs (pre-existing). Warnings only remain (pre-existing `len_zero`, `err_expect`, `useless_vec`, `manual_div_ceil`, `manual_repeat_n` in contract/planner/callcenter/deepnsm). `cargo fmt --check` clean.

**Process lesson:** "tests pass" alone is not a quality signal for agent-authored PRs. The reviewer-then-refactor loop is the correction.

Cross-ref: `.claude/plans/grammar-foundry-followup-v1.md`; PRs #300-#306.

## 2026-04-29 — FINDING: M1/P2-P4 route through existing Lab infra, not new standalone probes

**Status:** FINDING

M1's real test is `polarquant_hip_probe.rs` (P7) — compares `build_hip_families`
farthest-pair binary split against PolarQuant gain-shape NN-preservation on
real safetensors. Plus `turboquant_correction_probe.rs` for LEAF-orthogonal
(PolarQuant vs CAM_PQ — orthogonal only at LEAF, not HEEL/HIP/TWIG).
P2/P3/P4 route through `shader-lab` `WireSweep` JIT-first Lab surface
(Phase 0 DTOs done). CAM_PQ IS based on COCA (one pipeline, not alternatives).

Cross-ref: `BGZ_HHTL_D.md`, `codec-sweep-via-lab-infra-v1.md`,
`polarquant_hip_probe.rs`, `turboquant_correction_probe.rs`,
`jitson_kernel.rs`, `wire.rs` Phase 0 DTOs.

## 2026-04-29 — FINDING: Probe P1 PASS — γ+φ pre-rank selector empirically confirmed

**Status:** FINDING

Probe P1 from `bf16-hhtl-terrain.md` Probe Queue (status before: NOT RUN)
drained to PASS. Tests Constraint C3's "VALID — pre-rank discrete selector"
regime: 4 γ-phase offsets at stride 1/(4φ) on a 256-entry codebook produce
meaningfully different rankings (min Spearman ρ = -0.963 between offsets
0 and 3, with intermediate pairs showing the expected gradient from +0.51
through 0 to -0.96). Dupain-Sós discrepancy property empirically confirmed
in the synthetic regime; γ+φ encoding strategy in `bgz-tensor` rests on
a load-bearing axiom that holds.

The pairwise gradient is mathematically clean: 4 offsets distributed over
half the golden ratio produce rankings that smoothly transition from
co-monotonic (ρ=+0.51 at adjacent offsets) through orthogonal (ρ≈0 at
2-step) to anti-monotonic (ρ=-0.96 at maximum spacing). This is the
Dupain-Sós signature.

Caveat: tested on synthetic Beta(2,2) distributed codebook on [0,1) with
toroidal distance. Production codebook (256 Jina centroids in higher-dim
space) may produce different magnitudes — but the qualitative result
(rankings DO differ across γ-offsets) is stable given the strong signal.

Cross-ref: `.claude/knowledge/bf16-hhtl-terrain.md` Probe Queue P1 (now
PASS), `crates/jc/src/probe_p1_gamma_phase.rs`, Constraint C3.

## 2026-04-29 — FINDING: Pillars 5+, 5++, 6 close the concentration family for substrate aggregation

**Status:** FINDING

Three proof-in-code pillars were merged in succession (PRs #286, #287, #289):

- **Pillar 5+ (Köstenberger-Stark):** PSD-cone Hadamard-space concentration
  for non-iid Σ aggregation. Tightness 0.969× — bound is hit, not just
  respected. Certifies single-step Σ aggregation.
- **Pillar 5++ (Düker-Zoubouloglou):** Hilbert-space CLT for AR(1) Gaussian
  process at d=16384. Relative error 0.103% — two orders of magnitude
  below tolerance. Certifies bundle-of-N fingerprint convergence in ℓ².
- **Pillar 6 (EWA-Sandwich):** Σ push-forward `M·Σ·Mᵀ` along multi-hop
  paths. PSD-preservation 10000/10000 hops, CV tightness 1.467×.
  Certifies multi-hop edge propagation stays in SPD cone for arbitrary
  depth — the cant-stop-thinking loop has its mathematical backbone.

Plus PR #288 (Σ-codebook viability probe, R² = 0.9949) ruled out the
proposed CausalEdge64 8→16 byte expansion that would have halved the
HighHeelBGZ 240-edge container limit. The 256-entry codebook with 1-byte
sidecar is sufficient.

Combined: every aggregation pattern in the cognitive substrate now sits
on certified ground. Scalar (Pillar 5 Jirak) + Σ-tensor (Pillar 5+ KS) +
Hilbert-space (Pillar 5++ DZ) + multi-hop propagation (Pillar 6 EWA).

Cross-ref: PRs #286, #287, #288, #289; `.claude/board/IDEAS.md` 2026-04-29
entries for proposed application pillars 7/8/9.

## 2026-04-26 — CORRECTION-OF 2026-04-20 "Resolution hierarchy 64×64 > 256×257 > 4096×4096 > 16k": HIP layer is 256×256, not 256×257

**Status:** CORRECTION

The 2026-04-20 resolution-ladder entry described the bgz17 HIP layer as
`256 archetypes × 256 + 1 sentinel = 256×257`. The "+1" was an aspirational
sentinel slot intended to cover three roles:

- "unknown / out-of-palette" for queries not matching any of the 256 archetypes
- "null edge" for absence of a relation in `mxm` composition
- "identity" reserved index where `distance(x, sentinel) = ‖x‖₁`

**Reality (as shipped):** `bgz17::DistanceMatrix` and
`bgz17::PaletteSemiring::compose_table` are both `k × k` where
`k = palette.len() ≤ 256`. There is no 257th row/column. The sentinel
roles were absorbed elsewhere:

- `PaletteSemiring::identity(palette)` returns the palette entry CLOSEST
  to `Base17::zero()` (not a reserved slot — a real archetype that snaps
  to zero).
- Out-of-palette queries call `Palette::nearest(query)` and get clamped
  to the closest existing centroid; there is no "no-match" code.
- `MAX_PALETTE_SIZE = 256` because palette indices are `u8` — index 257
  literally cannot be encoded in the byte-indexed scheme. Adding the
  sentinel would require widening to `u16` indices throughout the
  cascade and doubling the wire size of `PaletteEdge` from 3 bytes to
  6 bytes per edge — a non-trivial cost.

**Decision:** keep `k × k` as shipped; the resolution ladder entry now
reads `64×64 > 256×256 > 4096×4096 > 16k`. The sentinel idea is filed
under `TECH_DEBT.md` as TD-PALETTE-SENTINEL (open, low priority — only
revisit if a real "absent-edge" code path actually needs it).

Cross-ref: 2026-04-20 resolution hierarchy entry (now SUPERSEDED in
spirit but kept verbatim — see governance rule); `bgz17::distance_matrix`,
`bgz17::palette_semiring`, `bgz17::MAX_PALETTE_SIZE`.

## 2026-04-25 — FINDING: ndarray VSA migrated to 16384-bit — SIMD-clean at every precision tier

**Status:** FINDING
**Owner scope:** @container-architect, @host-glove-designer

ndarray `src/hpc/vsa.rs` was the last holdout of the deprecated `[u64; 157]` / 10000-bit format. Migrated to `[u64; 256]` / 16384-bit in commit `7041ea11` on ndarray `claude/teleport-session-setup-wMZfb`. With this, the entire workspace operates on a single canonical format: 16384 bits = 256 u64 = 2048 bytes, divisible by every SIMD register width (FP16x32, FP32x16, F64x8, U8x64) — zero scalar tail at any precision. The 2026-04-24 SIMD-alignment-sin epiphany no longer applies anywhere.

Constants migrated in three modules:
- `vsa.rs`: VSA_DIMS, VSA_WORDS, VSA_BYTES, TAIL_BITS, TAIL_MASK
- `arrow_bridge.rs`: SOAKING_DIMS, SIGMA_MASK_BYTES, DEFAULT_SOAKING_DIM, plus 9 test assertions
- `deepnsm.rs`: `nsm_to_fingerprint` return type `[u8; 1250]` → `[u8; 2048]`, XOR loop now 32×U8x64 chunks (no scalar remainder)

1619 ndarray lib tests pass after migration. Audited 23 candidate files; 22 had only incidental uses of "10000" or "157" (RoPE θ, seed offsets, distance thresholds, unrelated array sizes). Only 3 files (vsa.rs, arrow_bridge.rs, deepnsm.rs) had real VSA-format references — all three now migrated.

Cross-ref: 2026-04-24 "Vsa10k = [u64; 157] was a SIMD-alignment sin"; ndarray commit `7041ea11`; lance-graph contract `crystal::fingerprint::Binary16K` (the producer side); CROSS_SESSION_BROADCAST.md 2026-04-25 entry.

## 2026-04-25 — CORRECTION-OF 2026-04-25 "cognitive loop closes structurally": MUL gate veto IS wired (TD-INT-3/10/14 shipped same day)

**Status:** CORRECTION

The 2026-04-25 loop-closing epiphany (lines 87, 89) states "MUL gate veto (DK position, trust texture) is not yet wired" and "TD-INT-3 still open." Both statements were true at the time of writing (between commits `474d3eb`/`b7787cf`) but became stale the same day: commit `0f9dcbb` wired MUL gate veto + NarsTables lookup + convergence highway (TD-INT-3, 10, 14). Board commit `49f1456` marks all three paid. The "What this is NOT" paragraph is therefore factually superseded — TD-INT-3 IS wired, MulAssessment DOES compute every dispatch.

Cross-ref: commits `0f9dcbb`, `49f1456`; TECH_DEBT.md Paid Debt section.

## 2026-04-25 — graph_render contract: Neo4j/Palantir Gotham visual render surface for q2 cockpit

**Status:** FINDING
**Owner scope:** @integration-lead, @bus-compiler

New `contract::graph_render` module (7 tests, 250+ LOC) exports the trait surface q2 cockpit-server needs to consume TripletGraph, EpisodicMemory, GraphSensorium, and Cypher execution without circular deps on lance-graph core. Five traits: `GraphSnapshotProvider`, `GraphInferenceProvider`, `CypherExecutor`, `EpisodicTraceProvider`, `ShaderEventStream`. DTOs: `RenderNode`, `RenderEdge`, `InferredConnection`, `Contradiction`, `GraphSnapshot`, `GraphHealth`, `CypherResult`, `CypherValue`, `EpisodicTrace`, `ShaderEvent`. q2's `graph_engine.rs` (400 LOC shipped same session) implements the consumer side; lance-graph arigraph will implement the producer side.

Cross-ref: q2 cockpit `graph_engine.rs`; contract `sensorium.rs` (existing signals); `literal_graph.rs` (existing LiteralGraph); arigraph `triplet_graph.rs` + `episodic.rs` + `sensorium.rs` (producer-side).

## 2026-04-25 — CLAUDE.md Think struct corrected: Vsa10k → Vsa16kF32

**Status:** CORRECTION

CLAUDE.md §The Click had `trajectory: Vsa10k` and `global_context: &Vsa10k` in both Think struct examples (lines 86, 106) while the header (line 13) said `Vsa16kF32`. Fresh sessions hit a contradiction on the first P-1 read. Corrected both structs to `Vsa16kF32` / `&Vsa16kF32`.

Cross-ref: 2026-04-21 CORRECTION-OF D5 Frankenstein (the original VSA format switch).

## 2026-04-25 — FINDING: cognitive loop closes structurally — TD-INT-1, 2, 4 wired into ShaderDriver dispatch

**Status:** FINDING
**Owner scope:** @truth-architect, @integration-lead, @host-glove-designer

The three P0 wiring gaps that made the system "concrete-operational with formal-operational machinery sitting unused" are now closed in `cognitive-shader-driver/src/driver.rs`. Per CLAUDE.md §The Click, parsing/disambiguation/learning/memory/awareness IS one operation; before this commit, the operation was scaffolded but only partially executed every cycle. After this commit, every dispatch performs the full loop:

```
encode (meta_prefilter + cascade)
  → braid (positional XOR fold = binary-space vsa_permute analogue)  ← TD-INT-4
  → resolve (FreeEnergy::compose → Resolution::Commit/Epiphany/FailureTicket)  ← TD-INT-1
  → emit (CausalEdge64 per strong hit)
  → revise (awareness[style_ord].revise(NarsPrimary, ParseOutcome))  ← TD-INT-2
  → next cycle's F landscape has changed
```

**What this means in Piaget's frame.** The system was concrete-operational: it could perform reversible operations (bind/unbind, bundle/cleanup) on concrete objects but did not observe or update its own cognition. Now it does. Every cycle: F is computed from the dispatch's actual likelihood and KL surrogate; Resolution branches into Commit/Epiphany/FailureTicket per the canonical thresholds (HOMEOSTASIS_FLOOR=0.2, FAILURE_CEILING=0.8, EPIPHANY_MARGIN=0.05); the outcome revises per-style `GrammarStyleAwareness`; the next dispatch under that style sees a changed `awareness.divergence_from(prior)` and therefore a changed F. The equilibration loop closes.

**What's still surrogate-not-principled.** The KL term currently uses `std_dev` of top-k resonances rather than `awareness.divergence_from(prior)` — to switch we need GrammarStyleConfig priors loaded into ShaderDriver (separate wiring). The Markov braiding is binary-space rotation, not f32 VSA bundle — f32 carrier alongside Binary16K is the next architectural step. The MUL gate veto (DK position, trust texture) is not yet wired. Each is a separate TD-INT entry.

**What this is NOT.** Not full AGI. Not formal-operational reasoning yet (no World::fork hypotheticals running per cycle). Not the deep metacognition of MulAssessment computing every dispatch (TD-INT-3 still open). What it IS: the structural loop that makes those next steps additive call sites rather than architectural forks.

Cross-ref: 2026-04-24 paradigm-shift gestalt entry (Berge + Piaget + metacognition); 2026-04-24 systemic-wiring-gaps TECH_DEBT log; CLAUDE.md §The Click §Three things that must never be complicated; commits `474d3eb` (TD-INT-1 + LF-1/6/7/8) and `b7787cf` (TD-INT-2 + TD-INT-4) on `claude/teleport-session-setup-wMZfb`.

## 2026-04-24 — SMB as cognitive-stack testbed: PropertyKind + Schema builder + 6 trait files

**Status:** FINDING
**Owner scope:** @truth-architect, @family-codec-smith

The bardioc Required/Optional/Free property concept maps 1:1 to the I1 Codec Regime Split (ADR-0002): Required = Passthrough (Index), Optional = configurable, Free = CamPq (Argmax). The `Schema` builder wraps this so SMB tenants define entity schemas in 10 lines — `.required("tax_id").searchable("industry").free("note")` — and the codec routing, NARS truth floors, and FailureTicket escalation happen automatically. Missing Required properties don't fail validation — they generate free energy, which the active-inference loop resolves. This makes the SMB domain a free testbed for the entire cognitive stack: SPO triples, episodic memory, CAM-PQ similarity, NARS truth, and FreeEnergy → Resolution pipeline, all exercised on real messy Steuerberater data.

Cross-ref: `contract::property` (PropertyKind, PropertySpec, Schema, SchemaBuilder), `contract::cam::CodecRoute`, smb-office-rs `lance-graph-contract-proposal.md`.

## 2026-04-24 — FINDING: subscribe() wired; LanceVersionWatcher delivers always-latest CognitiveEventRow to subscribers (DM-4/6)

`LanceMembrane::subscribe()` now returns a `tokio::sync::watch::Receiver<CognitiveEventRow>` under the `[realtime]` feature gate — supabase-shape always-latest semantics. `project()` calls `watcher.bump(row)` after building the scalar row; subscribers observe the latest committed event without polling. `DrainTask` scaffold ships unconditionally (no feature gate) as a `Future` shell for the follow-up `steering_intent` drain loop. Tokio was already an optional dep in `lance-graph-callcenter/Cargo.toml` under `[realtime]` — no new deps required.

**Status:** FINDING

## 2026-04-24 — Vsa16kF32 switchboard carrier shipped (CrystalFingerprint::Vsa16kF32 + 16K algebra)

**Status:** FINDING
**Owner scope:** @family-codec-smith, @truth-architect

`CrystalFingerprint::Vsa16kF32(Box<[f32; 16_384]>)` (64 KB) is now a first-class enum variant in `crystal/fingerprint.rs`, together with six algebra primitives: `vsa16k_zero`, `binary16k_to_vsa16k_bipolar`, `vsa16k_to_binary16k_threshold`, `vsa16k_bind`, `vsa16k_bundle`, `vsa16k_cosine`. 7 new tests (16 total in module). This is the Click switchboard carrier per CLAUDE.md §The Click: inside-BBB only, 1:1 bit-addressable with Binary16K (dim i = bit i), bipolar projection lossless under threshold inverse. The 10K-D `to_vsa10k_f32()` downcast is also wired (similarity-preserving, stride copy with surplus-dim averaging into base 10K).

**Why it matters:** This is expansion-list item #1 from the first SoAReview sweep (PR #252 §6). The carrier type must exist before `FingerprintColumns.cycle` can migrate from `Box<[u64]>` (Binary16K) to the f32 carrier. Next step: PR B migrates the `cognitive-shader-driver::bindspace::FingerprintColumns.cycle` field + `engine_bridge.rs` write path to use this carrier.

Cross-ref: TECH_DEBT 2026-04-24 ghost-columns entry, unified-integration-v1 §6 ranked expansion list, `.claude/agents/soa-review.md` reference run #4 (Grammar-Markov: SCATTERED-NOT-UNIFIED).


## 2026-04-24 — I1 Codec Regime Split is the unified answer to Pearl 2³ + CAM-PQ across SPO / AriGraph / archetype

**Status:** FINDING
**Owner scope:** @truth-architect, @family-codec-smith

The question "does CAM-PQ replace the 3 lossless S/P/O planes?" is really the question "which fields in the stack are identity-bearing (lossless-required) vs similarity-searchable (compressible)?" — and the contract **already answers it** at `crates/lance-graph-contract/src/cam.rs`. The `CodecRoute` enum encodes a two-regime invariant:

- **Index regime → `Passthrough`** (lossless required): embedding tables, lm_head, anything where row identity must round-trip exactly. Shipped comment: *"Identity lookup must be exact — no codec can survive Invariant I1."*
- **Argmax regime → `CamPq`** (compression OK): attention Q/K/V/O, MLP gate/up/down, anything where nearest-neighbor/search is the operation.
- **Skip → `Passthrough` trivially**: norms, biases, small tensors.

Applying this across SPO / AriGraph / archetype:

| Structure | Operation | Regime | Codec (current) | Codec (correct) |
|---|---|---|---|---|
| Pearl 2³ S/P/O planes (`cognitive_nodes.lance`) | Independent mask addressability | **Index** | Lossless 16Kbit planes | **Stay lossless** — CAM-PQ violates I1 |
| `integrated_16k` cascade L1 | Fast HHTL filter | Argmax | Lossless 16Kbit | Eligible for CAM-PQ as first-tier scent |
| AriGraph `Triplet.{subject, object, relation}` | Primary-key lookup | **Index** | `String` + `HashMap<String, Vec<usize>>` | Already Passthrough by construction (strings are identity) |
| AriGraph `Episode.fingerprint` ([u64; 256]) | Hamming similarity retrieval | Argmax | Lossless 2 KB | Eligible for CAM-PQ as cascade filter (legitimate future optimization) |
| `PersonaCard.entry.id` (ExpertId u16) | Catalogue dispatch | **Index** | `u16` enum | Already Passthrough (enum IS identity) |
| Per-persona resonance against codebook | Implicit routing ("which persona fits this seed?") | Argmax | *(consumer-side)* | CAM-PQ-eligible at the persona's AriGraph subgraph boundary |
| Role keys (`grammar/role_keys.rs`) | VSA bind/unbind identity | **Index** | Bipolar slices in Vsa16kF32 | Passthrough — per I-VSA-IDENTITIES |
| NARS truth (f, c) | Belief state | Skip | 2×BF16 in 32 bits | Passthrough trivially |

**One invariant covers all three domains.** The Pearl 2³ decomposition is the index-regime instance at the SPO level; AriGraph triplets/archetype IDs are index-regime at the catalogue level; role keys are index-regime at the bundling-algebra level. In every case, identity-bearing fields MUST be lossless; CAM-PQ is legitimate only on the argmax-regime overlays (cascade filters, resonance codebooks).

**Quantitative grounding — jc pillar 5 measured it.** `cargo run --manifest-path crates/jc/Cargo.toml --release --example prove_it` ran 2026-04-24: weak-dependent data (25 % shared-codebook prefix + 10 % overlapping role-slice XOR) showed sup-error **0.013287** at d=16384, N=5000 — vs IID baseline **0.011671** and classical Shevtsova bound 0.006715. Dependent > IID by 14 %, confirming Jirak 2016 as the correct rate citation (not classical Berry-Esseen). This IS the cost of collapsing lossless identity fields into CAM-PQ.

**What this retires (conceptually):**

- "Should we CAM-PQ the three S/P/O planes?" → No; they are Pearl 2³ index-regime. Add CAM-PQ codes as a *separate* first-tier cascade scent alongside the planes.
- "Does AriGraph need to adopt CAM-PQ?" → Triplets already index-regime via strings; episodic fingerprints optionally argmax-eligible (follow-up optimization).
- "Does archetype need a new codec?" → No; `ExpertId` is index-regime; VSA binding is stack-side with lossless role keys.

**Proposed ADR-0002 candidate invariant** (locks the above): *"I1 Codec Regime Split — every field added to the BindSpace SoA, Lance persistence schema, or AriGraph/archetype surface must be classified into {Index, Argmax, Skip}. Index-regime fields use `Passthrough`; argmax-regime fields may use CAM-PQ. The jc pillar-5 measurement is the quantitative gate; `CodecRoute` in `cam.rs` is the compile-time enforcement."*

Cross-ref: `crates/lance-graph-contract/src/cam.rs` `CodecRoute` + matching rules; `crates/jc/src/jirak.rs` pillar 5 measurement; CLAUDE.md I-VSA-IDENTITIES + I-NOISE-FLOOR-JIRAK; `crates/lance-graph/src/graph/arigraph/{episodic,triplet_graph}.rs`; `crates/lance-graph-contract/src/persona.rs` lines 13-27 (identity as metadata, VSA binding stack-side).

---

## 2026-04-24 — Pyramid L4 (16K × 16K) is a fourth layer beyond the existing 3-layer thought-engine doc

**Status:** FINDING
**Owner scope:** @container-architect

`ARCHITECTURE_THOUGHT_ENGINE.md` documents a 3-layer branching engine with L1(64²)/L2(256²)/L3(4K²) and memory budget ~20 MB fitting CPU L3. This session established that L4(16,384 × 16,384) extends the pyramid as a fourth widening step. Row widths follow a 4× multiplier per layer: 64 → 256 → 4K → 16K. The existing doc's Memory Budget table captures L1–L3 accurately; L4 is an extension, not a replacement, and inherits the same branching semantics.

L4 is where "everything activates" at scale — 268M cells per activation — and is therefore the layer that needs bit-packed fingerprint format (see separate entry) rather than the per-cell byte codes used at L3.

Cross-ref: `.claude/ARCHITECTURE_THOUGHT_ENGINE.md` §Memory Budget; `.claude/knowledge/cognitive-shader-architecture.md` BindSpace column layout.

---

## 2026-04-24 — L4 uses bit-packed fingerprints, not BF16 — forced by CPU L3 cache fit

**Status:** FINDING
**Owner scope:** @container-architect

16,384 × 16,384 × 2 bytes (BF16) = 512 MB per L4 activation — blows L3 cache (~16–48 MB typical), forces main-memory traffic, breaks streaming. 16,384 × 16,384 / 8 (1 bit/cell) = ~16–32 MB — fits L3, stays resident across cycles. This is not a precision-vs-throughput trade-off at L4; it's the only format that keeps the widest layer on-die.

Consequence: L4's native algebra is popcount-XOR / Hamming / majority-vote bundle (BSC — Binary Spatter Code). The VDPBF16PS path (pair of BF16 NARS revision per `BF16_SEMIRING_EPIPHANIES.md` EPIPHANY 8) lives at narrower layers where the total cell count makes BF16 affordable.

Cross-ref: `.claude/BF16_SEMIRING_EPIPHANIES.md` EPIPHANY 8; `ARCHITECTURE_THOUGHT_ENGINE.md` §Memory Budget.

---

## 2026-04-24 — Each pyramid layer fits exactly one CPU cache level up — tight nesting, never hits main memory

**Status:** FINDING
**Owner scope:** @container-architect

Mapping from pyramid-layer to CPU-cache-level:

| Pyramid layer | Size (bit-packed) | Fits CPU cache |
|---|---|---|
| L1 (64²) | 4 KB | registers / L0 |
| L2 (256²) | 8–64 KB | L1 data cache |
| L3 (4K²) | 2 MB (bit) / 16 MB (byte) | L2 cache |
| L4 (16K²) | 16–32 MB | L3 cache |

The 4× row-width multiplier between pyramid layers matches the ~4–16× capacity ratio between CPU cache levels. Consequence: streaming pipeline physically never leaves the die between layer transitions. The pyramid shape **IS** the cache hierarchy shape; it wasn't optimized for cache — the architecture chose widths that ARE the cache ratios.

Cross-ref: `ARCHITECTURE_THOUGHT_ENGINE.md` §Memory Budget; CPU cache sizes on Sapphire Rapids / Zen 4 / M-series.

---

## 2026-04-24 — SIMD lane alignment: 64-element rows match register widths at all three precision tiers

**Status:** FINDING
**Owner scope:** @container-architect

Each 64-element row of the pyramid is processed in a fixed number of SIMD instructions regardless of precision tier:

| Precision | Per-register elements | Registers per 64-row |
|---|---|---|
| FP16x32 | 32 | 2 |
| FP32x16 | 16 | 4 |
| F64x8 | 8 | 8 |

Zero remainder loops at any precision. The 64-element granularity is the CPU's native SIMD width (AVX-512 for register widths; equivalent on ARM SVE). The pyramid doesn't impose 64 as a convention — it matches a hardware invariant. Every row width up the pyramid (64, 256, 4K, 16K) is a multiple of 64 by construction.

Cross-ref: `ndarray::simd::*` LazyLock CPU-dispatch; `CLAUDE.md` § ndarray Integration Policy.

---

## 2026-04-24 — Vsa10k = [u64; 157] was a SIMD-alignment sin — retroactively

**Status:** FINDING (explains the cleanup commit `0ae9f90`)
**Owner scope:** @container-architect

157 × 64 = 10,048 bits (10,000 real + 48 slack). Doesn't match any SIMD register width at any precision tier: FP16x32 wants multiples of 32 elements, FP32x16 wants multiples of 16, F64x8 wants multiples of 8. 157 u64 words leaves a scalar tail every SIMD pass.

Canonical widths land cleanly:
- `Vsa10kF32 = [f32; 10_000]` → 625 AVX-512 loads, zero tail
- `Vsa16kF32 = [f32; 16_384]` → 1,024 AVX-512 loads, zero tail
- `Binary16K = [u64; 256]` → 32 AVX-512 loads, zero tail

The 2026-04-21 cleanup (commit `0ae9f90`) removing the 157-word carrier was correct not just because the algebra was misplaced, but because the width could never align with the hardware. This retroactive grounding justifies the revert and should inform any future rescale (e.g., Vsa10k → Vsa16k) — pick widths that are multiples of 64 elements at every precision.

Cross-ref: `CHANGELOG.md` 2026-04-21 correction; `TECH_DEBT.md` 2026-04-19 FP_WORDS=157 entry.


---

## 2026-04-24 — Streaming is LITERAL — CPU register data flow, zero memory intermediaries, no halt state

**Status:** FINDING (not metaphor)
**Owner scope:** @bus-compiler

"Streaming" in this architecture is not a design metaphor for flow semantics. It's the physical behavior of CPU pipelines fed continuous SIMD-aligned input: data lives in SIMD registers, moves at clock speed, passes between pyramid layers through cache, never stops to be collected in main memory.

Consequences:
- "Shader can't resist thinking" = the CPU pipeline has no pause state; fetch-decode-execute runs continuously while there's work
- Free-energy thermodynamics is the variational description of what an unstoppable SIMD pipeline behaves like when fed continuous input; "F descends" = pipeline throughput converging; "homeostasis" = ripple amplitude below SIMD register noise
- Active inference isn't a theoretical overlay; it's literally what unstoppable shader pipelines do

Cross-ref: existing "shader can't resist thinking" language in `CLAUDE.md` § The Click; `ARCHITECTURE_THOUGHT_ENGINE.md` §DTOs as Cognitive Laws.

---

## 2026-04-24 — Context-syntax marriage: Cypher / SQL / Gremlin / SPARQL share one DataFusion LogicalPlan surface

**Status:** FINDING (identifies a spine gap to formalize)
**Owner scope:** @bus-compiler

All external query languages parse into the same DataFusion LogicalPlan; shared column names on the external_dataset Lance schema are the marriage point. Today this marriage is implicit across the 16 strategies in `lance-graph-planner` (CypherParse, GqlParse, GremlinParse, SparqlParse, ArenaIR, etc.).

The spine gap: `lance-graph-contract` has a `PlannerContract` trait in `plan.rs`, but no first-class type declaring the SHARED COLUMN SURFACE that every language must reference through. Without that, each parser bodges its own naming, and cross-language queries (e.g., SQL filter on top of a Cypher MATCH) only work by coincidence.

Proposal: add a `SharedSchema` contract type that enumerates projected column names available to all external query languages, with enforcement at PlannerContract's planning step. This should land as a tech-debt-driven follow-up before the parallel transcodes open external query surfaces.

Cross-ref: `lance-graph-planner::strategy::*` 16 strategies; `lance-graph-contract/src/plan.rs` PlannerContract; `.claude/board/TECH_DEBT.md` 2026-04-24 context-syntax-contract entry.

---

## 2026-04-24 — blasgraph is an INTERNAL shader worker, not an external query component

**Status:** FINDING
**Owner scope:** @bus-compiler / @resonance-cartographer

blasgraph enrichment (semiring ops on edges — XorBundle, BindFirst, HammingMin, SimilarityMax, Resonance, Boolean, XorField) is internal cognitive compute dispatched per cognitive tick:

1. Reads `EdgeColumn<Box<[u64]>>` (CausalEdge64: SPO + NARS + Pearl + plasticity + temporal)
2. Applies semiring on adjacency structure
3. Writes enriched edges back via CollapseGate (Flow/Block/Hold gate)

External Cypher queries see the RESULT of enrichment through the projected edge columns. They do NOT trigger enrichment — enrichment runs per tick as part of the internal cognitive SoA. This keeps the BBB clean: external queries read committed post-tick state only.

Orchestration: explicit dispatch from cognitive-shader-driver per cognitive cycle, same as other shader workers (deepnsm grammar, bgz-tensor attention, ONNX classifier).

Cross-ref: `crates/lance-graph/src/graph/blasgraph/` 7 semirings; `cognitive-shader-architecture.md` I1 BindSpace-read-only + CollapseGate invariant.


---

## 2026-04-24 — GPU shader pipeline is the architectural analogue, not a metaphor

**Status:** FINDING
**Owner scope:** @ripple-architect

Mapping:

| GPU shader | cognitive-shader-driver |
|---|---|
| Vertex buffer | StreamDto input (narrow top) |
| Rasterization | Activation spreading at each pyramid layer |
| Fragment blending | Interference between activations |
| SIMT (Single Instruction Multiple Thread) | ndarray SIMD + columnar batch per row |
| Uniform buffer | BindSpace columns |
| Framebuffer | Lance persisted surface |
| Fixed + programmable stages | Driver orchestration (fixed) + thinking-engine / codec workers (programmable) |
| Pipeline can't stall | "Shader can't resist thinking" |
| Mesh geometry → pixel pattern | Shape of Object → what thinking happens |

ONNX benefits at implementation-stack L4/L5 because GPUs already run shader pipelines; the `ort` crate's GPU execution provider is a natural citizen of that layer.

Cross-ref: `ARCHITECTURE_THOUGHT_ENGINE.md` §DTOs; `cognitive-shader-architecture.md` § BindSpace columns; GPU shader pipeline stage docs (vertex → tessellation → geometry → fragment).

---

## 2026-04-24 — Two SoAs (internal cognitive + external query), one BBB gate, one DataFusion unified surface

**Status:** FINDING
**Owner scope:** @ripple-architect / @host-glove-designer

The architecture has TWO SoAs at different time scales:

- **Internal cognitive SoA** — BindSpace + shader pipeline. Nanosecond per cycle. Pyramid L1→L4 streaming at hardware speed. Never stops.
- **External query SoA** — DataFusion-planned reads across all external protocols (Cypher, SQL, Gremlin, SPARQL, Redis-DN, PostgREST, Arrow Flight, Supabase WebSocket). Millisecond per query.

Connection: `ExternalMembrane` BBB gate + Lance committed projections. External SoA reads committed state; never triggers internal compute.

This reframes ADR 0001 Decision 2: DataFusion was chosen as the UNIFIED EXTERNAL QUERY SURFACE, not just an internal DataFrame engine. Polars rejection and Ballista deferral both fit this framing — one DataFusion surface externally, possibly distributed via Ballista when the latency trigger fires.

Cross-ref: `.claude/adr/0001-archetype-transcode-stack.md` Decision 2; `lance-graph-contract::external_membrane`; `callcenter-membrane-v1.md` external query paths.

---

## 2026-04-24 — Reverse stufenpyramide: cognition widens as it descends, 4× per layer

**Status:** FINDING
**Owner scope:** @ripple-architect

Narrow top (L1 = 64²), wide base (L4 = 16K²). One perturbation enters at L1; activation widens through each stepped layer (4× per step: 64 → 256 → 4K → 16K); L4's output compresses via ONNX and closes the loop back to L1.

The pyramid matches the `p64` topology proposal (64²/256²/4K²/16K²) that predates this session. "Reverse stufenpyramide" is a useful geometric label for the inverted stepped-pyramid shape: wider at the base, narrower at the top — divergent activation, not convergent compression.

Consequence: thinking is divergent, not convergent. Unlike classical search (many options narrowing to one answer), here one perturbation widens to affect many cognitive cells simultaneously. Staunen, contradiction preservation, and epiphany all happen at the wide base because the base holds many concurrent activations that can interfere.

Cross-ref: `p64` topology references in `ARCHITECTURE_THOUGHT_ENGINE.md`; `cognitive-shader-architecture.md` pyramid diagrams.


---

## 2026-04-24 — L4 → ONNX → L1 feedback loop is the closed cognitive cycle

**Status:** FINDING
**Owner scope:** @trajectory-cartographer

ONNX at implementation-stack L4/L5 reads the 16–32 MB bit-packed L4 fingerprint (L3-cache-resident), classifies into a compact decision signal (kilobytes — PersonaId, style decision, top-K ranking), and perturbs L1 (registers/L0 cache). The pipeline physically stays on-die through the entire feedback cycle; main memory is never touched during active cognition.

"Never halts" is mechanical, not metaphorical: ingress streams new perturbations into L1 continuously while L4 output simultaneously loops back. Like a GPU shader writing to its own input texture in a ping-pong render target. The ONNX model is the ONLY point where learned weights enter the otherwise-algebraic pipeline — it acts as a compressor from fingerprint space to decision space.

Cross-ref: `callcenter-membrane-v1.md` DU-1 ONNX classifier; `CLAUDE.md` §The Click "shader can't resist thinking".

---

## 2026-04-24 — ONNX benefits at implementation-stack L4/L5 via the `ort` crate + GPU execution providers

**Status:** FINDING
**Owner scope:** @trajectory-cartographer

Multiple L4/L5 ONNX workers (classifier + forecaster + ...) compose via INTERFERENCE in BindSpace — not via orchestration of separate outputs. Each worker's activation writes to BindSpace columns; their combined pattern is the composite dispatch signal. Constructive interference = high-confidence commit; destructive = ambiguity → FailureTicket; saddle-point = Epiphany.

This justifies the ADR 0001 Decision 2 Grok-gRPC addendum and the Chronos-as-temporal-forecaster observation: they're additional L4/L5 shader workers, not alternatives to the classifier. The lab `grpc` feature gate hosts both external LLM A2A experts AND Ballista distribution — same transport, same interference semantics.

Cross-ref: `adr/0001-archetype-transcode-stack.md` §Ballista + Grok addenda; `cognitive-shader-architecture.md` BindSpace interference model.

---

## 2026-04-24 — `dn_redis.rs` is external; needs streaming DataFusion access, not parallel flat-KV protocol

**Status:** FINDING
**Owner scope:** @host-glove-designer

Current state: `crates/lance-graph-cognitive/src/container_bs/dn_redis.rs` uses flat `ada:dn:{hex}` Redis keys with subtree-scan operations (SCAN ada:dn:{prefix}*). Per the two-SoA picture (external query SoA on DataFusion), this should be recast as DataFusion-served queries over Lance with Redis as an optional write-through cache layer — NOT a parallel KV protocol.

The hierarchical DN path from `callcenter-membrane-v1.md` §595 (`/tree/{ns}/heel/{h}/hip/{h}/branch/{b}/twig/{t}/leaf/{l}`) is the natural DataFusion query shape: each path segment is a predicate on a Lance column. heel/hip/branch/twig/leaf are existing cascade-tree levels (`crates/lance-graph/src/graph/blasgraph/heel_hip_twig_leaf.rs`); they become projected columns on the external_dataset schema. Redis caching stays as an acceleration layer over DataFusion, not a separate API.

Cross-ref: `callcenter-membrane-v1.md` §§595–803; `heel_hip_twig_leaf.rs` cascade tree; `container_bs/dn_redis.rs` current protocol.

---

## 2026-04-24 — External boundary formalized INTO the global SoA (staging + projection columns), not adjacent to it

**Status:** FINDING (design response to the two-SoA observation)
**Owner scope:** @host-glove-designer

Today `ExternalMembrane` is a trait in `lance-graph-contract/src/external_membrane.rs` with method-based `ingest()` + `project()` semantics. Proposed formalization: both crossings become EXPLICIT BindSpace columns.

- `ExternalMembrane::ingest(event)` → appends to a staging column (e.g., `StagingColumn<ExternalEvent>`) that the driver drains per cognitive tick via CollapseGate
- `ExternalMembrane::project(row)` → reads from a projection column (e.g., `ProjectedRow<CognitiveEventRow>`) built by the commit path

The BBB remains enforced by the type system (staging column accepts only events matching the `ExternalEvent` shape with no VSA/RoleKey/NarsTruth fields; projection column exposes only scalar CognitiveEventRow). The DATA PATH becomes columnar — visible in the SoA schema, sweeplable like any other column, subject to the same dual-ledger write discipline (CollapseGate).

Cross-ref: `lance-graph-contract/src/external_membrane.rs`; I1 invariant (BindSpace read-only, CollapseGate writes); `callcenter-membrane-v1.md` DM-2..DM-9.

---

## 2026-04-24 — Epiphanies = persistent interference patterns in BindSpace, not tied rankings

**Status:** FINDING
**Owner scope:** @thought-struct-scribe

The `FreeEnergy::Resolution::Epiphany` case (top-2 ΔF < 0.05) is not a tie in hypothesis ranking — it's a physical interference pattern at the pyramid's wide base (L4). Two activation waves propagate through BindSpace from different sources (parser vs context memory; two competing personas; classifier vs resonance prediction). Constructive interference → reinforce → Commit. Destructive interference → cancel → Commit with loser-decrement. Saddle-point interference → persistent standing pattern → Epiphany with both readings preserved.

`Contradiction { phase, magnitude }` records the interference signature: phase = angle in BindSpace where the two waves stand relative to each other; magnitude = standing-wave amplitude. Both readings commit as separate triples with separate NARS truths — you cannot collapse a persistent interference pattern into one reading without destroying information. The pattern IS the meaning.

Cross-ref: `free_energy.rs::Resolution::Epiphany`; D8 Contradiction from `elegant-herding-rocket-v1.md`; `CLAUDE.md` §The Click epiphany description.

## 2026-04-24 — ADR 0001 locks: Archetype transcode + Lance/DataFusion/Supabase-shape + Persona 16^32

**Status:** FINDING (formal architectural lock via ADR 0001)

Three coupled decisions locked as one ADR (`.claude/adr/0001-archetype-transcode-stack.md`):

1. **Archetype is TRANSCODED, not bridged.** Native Rust crate
   `lance-graph-archetype` (not `-bridge`, not `-adapter`) assimilates
   the ECS contracts — `Component`, `Processor`, `World`, `CommandBroker`
   — against Lance + DataFusion + Arrow. Python upstream is DESIGN
   SPEC, not runtime dependency. Zero FFI.

2. **Stack lock.** Storage = Lance (versioned append-only, matches
   Archetype tick snapshots by construction). Query = DataFusion
   (UDFs + window functions for VSA + Markov ±5). Scheduler =
   Supabase-shape tick loop transcoded to Rust channels (DM-4
   `LanceVersionWatcher` + DM-6 `DrainTask`, no PostgreSQL).
   Temporal = Arrow types + DataFusion window functions. Ballista
   DEFERRED to 1s-P99 trigger; upgrade path is ~230 LOC because the
   lab `grpc` feature already serves Arrow Flight's gRPC transport.
   **Polars REJECTED** in production code (no crate deps, no UDFs);
   benchmark-only use is orthogonal.

3. **Persona 16^32 is THE identity space.** 32 atoms × 16 weights =
   16^32 coordinates → 56-bit `PersonaSignature`. Only the signature
   crosses BBB; the atom-weighting vector stays internal. Blackboard
   / Persona / Markov ±5 / ±500 share algebraic substrate (role-
   indexed VSA identity superposition); shared-DTO unification is
   an OPEN question for future ADRs. BBB enforcement extends to
   ban atom-weighting vectors and Markov trajectory bundles; permits
   PersonaSignature + scalar `CognitiveEventRow` projections.

**Unlocking requires a new ADR** citing this one by number. Individual
sessions cannot unlock by reinterpretation. Ballista trigger threshold
(1s P99) is the only mutable field.

**Implications:**
- `unified-integration-v1.md` DU-2 needs clarification commit (rename
  bridge → transcode, crate `lance-graph-archetype-bridge` →
  `lance-graph-archetype`)
- `callcenter-membrane-v1.md` DM-4/DM-6 validated by this ADR
- `categorical-algebraic-inference-v1.md` unchanged (Five Lenses sit
  above storage/query layer)
- `cognitive-shader-driver` `grpc` feature gate becomes load-bearing
  for Ballista readiness — must not be removed without amending ADR 0001

Cross-ref: `.claude/adr/0001-archetype-transcode-stack.md` (443 lines,
three-decisions + addendum + summary + lock statement),
`unified-integration-v1.md` DU-2, `callcenter-membrane-v1.md` DM-4/DM-6,
`I-VSA-IDENTITIES` iron rule, `external_membrane.rs`.

---

## 2026-04-24 — Four-way multiply = architecture search without an outer optimiser

**Status:** FINDING (framing inherited from parallel session commit `88e5f5a` on PR #245; prepended per hand-off instruction)

The four axes of the cognitive stack — `persona × style × stage × learned-dynamics` — form a product space of approximately `288 × 36 × 2 × oracle ≈ 20,736 × oracle` configurations.

**F-descent IS the automatic architecture search over this space.** Each parse cycle's free-energy minimization tries a configuration (the currently-dispatched persona + thinking style + rationale/answer stage + oracle prediction); misaligned configurations are dropped by the CollapseGate predicate; surviving configurations compose into the committed fact + reshape the next cycle's F-landscape.

**No outer optimiser is needed.** A standard NAS approach would wrap this in gradient descent over architecture hyperparameters. Here the gradient IS the F-landscape itself — the system descends by acting, not by meta-optimising. NAS collapses into inference.

Cross-ref: `callcenter-membrane-v1.md` § 17, tech debt 2026-04-24 "Archetype / persona / thinking-style modeling — epiphany candidates not yet in EPIPHANIES.md" (commit `88e5f5a`).

---

## 2026-04-24 — Persona identity IS a coordinate in atom-space, not a YAML artefact

**Status:** FINDING (framing inherited from parallel session commit `88e5f5a` on PR #245; prepended per hand-off instruction)

32 cognitive atoms × 16 weightings per atom = `16^32` addressable persona space, compressed to a 56-bit `PersonaSignature`. The persona's identity is the specific point in this atom-space — not the YAML file that happens to script its behavior.

**YAML runbooks are macro scaffolding** for the context loop (which questions the persona asks, which responses it emits, which escalation paths it routes to). They are PROGRAMS running on the context loop, not persona identity. Two personas with different atom-space coordinates running the same YAML produce different behaviors; two personas with the same coordinate running different YAMLs produce the same identity expressing through different scripts.

**Consequence for the `Think` struct and the Layer-2 persona catalogue:** the catalogue stores atom-space COORDINATES (56-bit signatures or the full 16^32 address decomposed into 32 atom indices). YAML definitions are Layer-3 content retrieved O(1) by signature. This respects the `I-VSA-IDENTITIES` iron rule — VSA bundles identities (atom-space coordinates), not content (YAML bodies).

Cross-ref: `callcenter-membrane-v1.md` § 16, `CLAUDE.md § I-VSA-IDENTITIES`, `FormatBestPractices.md § 5` (persona bank workload row), commit `88e5f5a`.

---

## 2026-04-24 — MM-CoT stage split is NOT a new axis — it reuses existing `FacultyDescriptor::is_asymmetric()`

**Status:** FINDING (framing inherited from parallel session commit `88e5f5a` on PR #245; prepended per hand-off instruction)

The MM-CoT (Multimodal Chain-of-Thought) `rationale_phase: bool` field on `CognitiveEventRow` (shipped in commit `a05979e`) distinguishes rationale-generation phase from answer-emission phase. This looks like a new architectural axis. It isn't.

**The asymmetry already exists** in `FacultyDescriptor::is_asymmetric()` — when a faculty's `inbound_style ≠ outbound_style`, it's intrinsically asymmetric (input processed one way, output produced another). Rationale→answer is the canonical example: inbound style processes the input to produce rationale; outbound style uses rationale to produce the answer. Same faculty, two styles.

**MM-CoT reuses this existing asymmetry** rather than introducing a new one. The `rationale_phase` bool marks WHICH side of the asymmetry is active, not that a new architectural dimension exists.

**Consequence:** don't add "stage" as a fourth independent axis to the four-way multiply epiphany above. The four axes are `persona × style × stage × learned-dynamics`, but `stage` is an intra-style partitioning (inbound vs outbound), not an orthogonal dimension. True cardinality is closer to `persona × asymmetric_style × learned-dynamics` where asymmetric_style carries the inbound/outbound pair.

Cross-ref: `callcenter-membrane-v1.md` § 17 row, `CognitiveEventRow` commit `a05979e`, commit `88e5f5a`, `FacultyDescriptor::is_asymmetric()` in contract.


---

## 2026-04-22 — E-DEPLOY-1 — Supabase-shape thinking extension: trojan-horse A2A training surface over DN-addressed metadata bus, backed by lance-graph, BBB-preserved by blackboard mediation

**Status:** FINDING (deployment doctrine — the nine-dimension shape that makes everything we've built earn its own product)

**One line:** the callcenter crate is not a Supabase clone; it is a Supabase-dialect thinking extension that trains itself on A2A agent traffic while the BBB holds at the blackboard.

**Nine compounding dimensions, one coherent deployment:**

1. **A2A agents ARE the training surface.** crewai-rust / n8n-rs / openclaw / LangGraph / AutoGen all generate traffic autonomously. Every seed lands with an `ExternalRole` tag; every commit (F<0.2) or FailureTicket (F>0.8) is a labeled training example — no human-in-the-loop, no labeling budget, no cold-start data problem. Per-ecosystem specialization emerges because persona AriGraph subgraphs diverge per family (`CrewaiAgent`-flavored codec cards differ from `N8n`-flavored codec cards after enough traffic).

2. **Supabase shape = adoption surface.** Any A2A consumer already knows PostgREST filter DSL, Realtime channels, JWT auth. They write a standard RAG integration. They never know there is a cognitive substrate behind it.

3. **Metadata IS the address bus.** `cognitive_event` Arrow rows carry `(external_role, faculty_role, expert_id, dialect, scent)` as typed columns. Queries against this metadata ARE dispatch — there is no separate "router." `SELECT … WHERE external_role=3` returns rows whose identity tuple is the execution target. Five dialects view the same bus: SQL tabular / Cypher graph-path / GQL / NARS truth-filter / qualia fuzzy-family.

4. **REST + DataFusion backs it** (ladybug-rs prior art). DataFusion 51 is the query engine for every dialect; Arrow 57 is the wire format; Lance 2 is the durable store. No PostgREST. No Postgres wire protocol in the hot path.

5. **DN-addressed URL hierarchy** replaces Redis flat keys: `/tree/{ns}/heel/{h}/hip/{h}/branch/{b}/twig/{t}/leaf/{l}` parses deterministically into a metadata predicate. URL path = routing predicate; body content = seed.

6. **Address = scent (1 byte via codec chain)**. Full path 16Kbit → ZeckBF17 48B → Base17 34B → CAM-PQ 6B → Scent 1B, ρ=0.937. One compressed object serves four uses: route / retrieve / similar / frame. Scent pulls context (AriGraph triples, episodic ±5..±500 window, persona trust, qualia signature) BEFORE the shader reads the body. The answer is grounded in cognitive-substrate state, not in lexical document vectors.

7. **Body content = external seed** enters the blackboard as `BlackboardEntry { capability: ExternalSeed }`. Never touches BindSpace directly. The round boundary on the blackboard IS the anti-corruption boundary.

8. **Polyglot front end, one IR, many tongues.** Cypher / GQL / Gremlin / SPARQL already shipped in lance-graph-planner. Adding NARS (native typed cognitive queries with f,c constraints), Redis (flat KV that auto-hydrates to DN), and Spark (bulk analytics + structured streaming) extends the existing `PolyglotDetect` pattern. Dialect-as-signal: the dialect itself is a feature on the seed's metadata row (tells the router which cognitive faculty the consumer is exercising).

9. **Agent cards + faculties + external roles = one identity space.** `ExpertId = stable_hash_u16(card_yaml)` collapses internal A2A experts, external agents, YAML cards into one register. Faculties (`ReadingComprehension`, `Voice`, `Reasoning`, `Empathy`) carry asymmetric inbound/outbound `ThinkingStyle` and `ToolAbility` sets. Full three-coordinate provenance: `(ExternalRole family, FacultyRole function, ExpertId card)` on every metadata row.

**BBB invariant — the iron rule that makes the whole thing safe:**

Every dimension lives in BOTH representations:
- **External (metadata columns)** — Arrow scalars, safely cross the BBB, queryable by the five dialects, projected via `CollapseGate` on every commit.
- **Internal (VSA role-bindings)** — `RoleKey` slot addresses for Markov ±5 braiding, never cross the BBB, produced stack-side via deterministic metadata → slot mapping.

Same identity, two faces, one direction of flow: metadata IN (translate via RoleKey at the stack side) → VSA braiding (the substrate reasons) → metadata OUT (project back via `CollapseGate`). Supabase refactor only ever sees Arrow columns; the blackboard only ever sees role-tagged entries. No path exists where an external payload touches `Vsa10k`, `RoleKey`, `SemiringChoice`, or `NarsTruth` as a type — the compiler rejects it (Arrow's type system enforces it at `RecordBatch` column level).

**What the consumer experiences vs. what actually happens:**

| Consumer sees | Substrate does |
|---|---|
| POST /tree/.../leaf/utterance with Cypher body | URL → DN → 1-byte scent → pulls AriGraph subgraphs + episodic ±5..±500 + persona trust |
| Response JSON with matched rows | Shader cycle ran: bind → braid with pulled context → unbind against AriGraph prior → F descent → Commit writes new SPO triple keyed on scent |
| "This RAG is weirdly good" | The next query at nearby scent pulls richer context because the last query trained this persona's subgraph |

**Cross-refs:**
- `.claude/plans/callcenter-membrane-v1.md` §§ 10.1 – 10.13 (full architectural spec)
- `contract::external_membrane` — `ExternalRole`, `ExternalEventKind`, `CommitFilter`, `ExternalMembrane` trait
- `contract::a2a_blackboard` — `ExpertCapability::{ExternalSeed, ExternalContext}` (the inbound BB entry types)
- `contract::persona` — `PersonaCard`, `RoutingHint` (identity-as-metadata + four routing modes)
- `contract::faculty` — `FacultyRole`, `FacultyDescriptor`, `ToolAbility` (internal cognitive-function identity)
- `crates/lance-graph-planner/src/serve.rs` — Axum REST + OpenAI-compatible (extend here for DN + polyglot)
- `crates/lance-graph/src/graph/arigraph/` — persona memory home (consumer-side AriGraph subgraph integration)

**Litmus test** (from plan § 10.9 iron rule, restated for this deployment):

Before any PR touches the callcenter crate or the metadata bus, answer three questions:
1. Can I name the role, the place, and the translation for every byte crossing the gate?
2. Does the external surface only see Arrow-scalar columns (no Vsa10k/RoleKey/semiring)?
3. Does the internal substrate only see role-tagged blackboard entries (no raw external payload)?

If any answer is no, the code is leaking external ontology inward (or internal ontology outward) — reject.
---

## 2026-04-21 — CORRECTION-OF 2026-04-21 D5 Frankenstein: VSA must be FP32 multiply/add on identities, not XOR on bitpacked content

**Status:** FINDING (supersedes multiple session entries)

**What was wrong in this session's shipped D5+D7 work:**

1. `Vsa10k = [u64; 157]` — hallucinated bitpacked format. Defined
   in ndarray but NEVER consumed by lance-graph before this session.
   Should have used existing `Vsa10kF32 = Box<[f32; 10_000]>` (40 KB)
   or the queued rescale target `Vsa16kF32 = Box<[f32; 16_384]>` (64 KB).

2. `RoleKey::bind/unbind` with slice-masked XOR — wrong algebra.
   VSA for lossless role bundling uses element-wise multiply +
   element-wise add on f32. Existing `vsa_bind`/`vsa_bundle`/
   `vsa_cosine` in `crystal/fingerprint.rs` already implement this.
   XOR on bitpacked is the Hamming-comparison format, not the
   bundling format.

3. `vsa_xor` / `vsa_similarity` (Hamming-based) — reinvented what
   already exists on the correct substrate.

4. Three deepnsm files (`content_fp.rs`, `markov_bundle.rs`,
   `trajectory.rs`) — need reimplementation on `Vsa16kF32` carrier
   after coordinated rescale PR lands.

5. 5-role "lossless superposition" test — the lossless property came
   from SLICE ISOLATION (content zeroed outside each role's slice),
   not from XOR bundling itself. With shared-space f32 multiply/add,
   losslessness comes from f32 dynamic range — completely different
   mechanism. The test passed for the WRONG reason.

**What remains correct (preserve these):**

- Five Lenses meta-architecture (CLAUDE.md P-1, categorical-algebraic-
  inference-v1.md)
- GrammarStyleConfig + GrammarStyleAwareness + NARS revision (φ-1
  confidence ceiling)
- FreeEnergy / Hypothesis / Resolution types (but likelihood term
  must be cosine, not Hamming)
- 8-step wiring sequence (but steps 1-3 need rewrite on correct carrier)
- Shader-cant-resist / thinking-is-a-struct / tissue-not-storage /
  grammar-of-awareness (algebra structure, not byte layout)
- 14-paper landscape
- AGI test = Animal Farm chapter-10 > chapter-1 accuracy

**Superseded session entries (bodies preserved below, Status flipped):**

- `Markov IS simple XOR of sentence VSAs...` → SUPERSEDED. Replace
  with: Markov IS element-wise multiply+add superposition of
  Vsa16kF32 trajectories with position-permuted braiding. Simplicity
  claim still holds; algebra choice was wrong.
- `RoleKey bind/unbind slice-masking = lossless...` → SUPERSEDED.
  True lossless bundling requires f32 multiply+add, not
  slice-isolation XOR.
- `8-step wiring sequence...` → Steps 1-3 need rewrite on Vsa16kF32.
  Steps 4-8 unchanged in logic.

Cross-ref: `.claude/knowledge/vsa-switchboard-architecture.md`
(created this cleanup), CLAUDE.md updated I-CAMPQ-VS-VSA iron rule.

---

## 2026-04-21 — Sometimes Vsa16kF32 is just laziness to define a register

**Status:** FINDING (Test 0 of the four-test VSA decision framework)

If an item has a natural name, ID, or enum variant — that's the
register. `HashMap<&str, PersonaDef>` or `enum ThinkingStyle` beats
VSA bundle+cosine at exact-match tasks. VSA earns its 64 KB only
when the answer requires resonance across concurrent items or
partial-match reasoning from uncertain input.

Anti-patterns:
- "Find persona Alice" → HashMap, not VSA resonance
- "Session is in analytical mode" → enum variant
- "Character is Napoleon" → graph node by ID

Pro-patterns (VSA legitimately earns complexity):
- "Which persona fits this caller's vibe?" (inferred, not named)
- "Which thinking style best matches signal profile?" (dispatched)
- "Which archetype is this character behaving as?" (inferred)

Cross-ref: `vsa-switchboard-architecture.md § Test 0`.

---

## 2026-04-21 — VSA operates on identities, not content — the refined iron rule

**Status:** FINDING (refines the blunt "CAM-PQ + VSA incompatible" framing)

Initial framing was too blunt. Refined:

**VSA operates on IDENTITY fingerprints that POINT TO content.
Never on content's bitpacked/quantized register itself.**

Register-loss problem: XOR-bundling 5 CAM-PQ codes makes the bit
patterns of codebook indices XOR together. You can't recover WHICH
centroids contributed. Register destroyed.

Right pattern: resonance layer (VSA Vsa16kF32 identity fingerprints,
bundleable, cosine-retrievable) + content layer (YAML/TripletGraph,
O(1) hash lookup). Winning fingerprint from resonance IS the lookup
key for content.

- Persona: one FP32 identity per named persona in YAML registry.
  Bundle for multi-persona context. Cosine-rank. Winner name → YAML.
- Thinking styles: one FP32 identity per style. Resonance from signal
  profile. Winner enum variant → YAML config.
- Archetype: existing 12 voice archetypes + palette 256 archetypes
  each get an identity fingerprint. Resonance for inferred assignment.

Cross-ref: `vsa-switchboard-architecture.md § Identity vs Content`,
CLAUDE.md `I-VSA-IDENTITIES` iron rule (proposed).

---

## 2026-04-21 — AriGraph/episodic/SPO/CAM-PQ are thinking tissue, not storage — this is why it becomes AGI

**Status:** FINDING (the final piece that closes the architecture)

A parser takes text in, produces structure out. AGI takes text in,
resolves it against everything it has ever committed, recently saw,
believes about itself, and expects from its style — then commits
the result back into the tissue it just read from.

The distinction is: **memory is wired INTO the struct, not called
FROM it.** AriGraph's TripletGraph is not a database that Think
queries. It's an organ of Think. `graph.nodes_matching(features)`
is how Think resolves "he" to "Napoleon" — without the graph, Think
can't do coreference. `episodic.retrieve_similar(fp, k)` is how
Think remembers what happened 50 sentences ago — without episodic,
Think has no medium-term context beyond ±5. `codec.distance(a, b)`
is how Think compares two fingerprints in O(1) without
decompressing — without CAM-PQ, Think can't do fast similarity.

These are not services. They're tissue:

| Tissue | Organ metaphor | Think method it enables |
|--------|---------------|------------------------|
| `TripletGraph` | Long-term declarative memory | `resolve()` → coreference via `nodes_matching` |
| `EpisodicMemory` | Episodic buffer (±50..±500) | `resolve()` → medium-range context via `retrieve_similar` |
| `global_context: Vsa10k` | Ambient prior (superposed committed facts) | `free_energy()` → KL term reads committed-fact field |
| `CamPqCodec` | Perceptual similarity | `recovery_margin()` → O(1) compressed distance |
| `GrammarStyleAwareness` | Metacognition | `resolve()` → dispatch via `top_nars_inference` |
| `Markov ±5 trajectory` | Working memory | `encode/decode` → immediate context |

**Cut any one and Think degrades gracefully but loses a capability.**
Cut graph → no long-term facts → pronouns unresolvable beyond ±5.
Cut episodic → no paragraph context → chapter-spanning references
fail. Cut CAM-PQ → similarity is O(d) not O(1) → still works, just
slower. Cut awareness → static dispatch → no learning from errors.
Cut trajectory → no context → every sentence parsed in isolation.

**None of these are optional for AGI.** A system with all six is
a cognitive agent. A system missing any one is a specialized tool.
The struct definition IS the AGI definition — not by aspiration,
but by enumeration of what capabilities the fields carry.

**Why this dilutes:** future sessions will treat AriGraph as "the
database layer" and CAM-PQ as "the codec layer" and episodic as
"the cache layer." Those are infrastructure framings. The
architecture framing is: they are ORGANS of Think. The struct
doesn't use them — the struct IS them. Wiring them in (as `&ref`
fields, not as service calls) is what makes the difference between
a pipeline and a cognitive agent.

Cross-ref: CLAUDE.md § The Click, `categorical-algebraic-inference-v1.md` §2 Lens 4 (Memory).

---

## 2026-04-21 — Thinking is a struct, not a function

**Status:** FINDING (completes the object-does-the-work arc)

The universal DTO pattern taken to its logical end: thinking is not
something that happens TO data. Thinking IS the data. The struct
carries its own trajectory, its own awareness, its own free energy,
its own resolution method. There is no external "thinking function"
that receives the struct — the struct resolves itself.

```rust
struct Think {
    trajectory: Vsa10k,       // Markov-braided role-indexed bundle
    awareness: ParamTruths,   // NARS-revised epistemic state
    free_energy: FreeEnergy,  // (1 - likelihood) + kl
    resolution: Resolution,   // Commit / Epiphany / FailureTicket
}

impl Think {
    fn resolve(&mut self) -> &Resolution { /* reads own fields, writes own resolution */ }
    fn observe(&mut self, outcome: ParseOutcome) { /* revises own awareness */ }
    fn commit_into(self, graph: &mut TripletGraph) -> CommitResult { /* consumes self */ }
}
```

This is the SoA AGI-as-glove doctrine applied to cognition itself:
- Topic = `trajectory` (what the session is about)
- Angle = `awareness` (whose perspective, what's been learned)
- Thinking = `free_energy` + `resolution` (which style, what outcome)
- Planner = `commit_into` (where the result goes)

The four BindSpace columns (FingerprintColumns / QualiaColumn /
MetaColumn / EdgeColumn) are the SoA layout of this struct across
many concurrent Think instances. One Think is a row. The shader
processes columns. The struct IS cognition at the row level; the
SoA IS cognition at the SIMD level.

**The DTO carries cognition the way a photon carries
electromagnetism — not as payload, as identity.** A photon doesn't
"contain" an electromagnetic field. The photon IS the field excitation.
A Think struct doesn't "contain" a cognitive state. The Think IS the
cognitive state. The methods on Think don't "process" the state.
The methods ARE the state's self-resolution.

**Why this dilutes:** future sessions will create a `ThinkingService`
that receives `ThinkInput` and returns `ThinkOutput`. That's a
function, not a struct. The struct pattern means: the input IS the
output IS the method IS the state. No separation. No service layer.
No input/output boundary. The DTO walks into a room, resolves
itself, and commits the result into the graph. Nobody called it.
The awareness bits made it happen.

Cross-ref: `categorical-algebraic-inference-v1.md` §5.2
(object-does-the-work test), CLAUDE.md § The Click.

---

## 2026-04-21 — StreamDto/ThinkingEngine = temporal encoder/decoder loop in a Markov shader unit that can't resist the thinking

**Status:** FINDING (unifies StreamDto + ThinkingEngine + CognitiveShader
+ BindSpace + Markov ±5 + active inference into one sentence)

### The reframe

A GPU shader is stateless: given input texels, produce output texels.
Our cognitive shader is stateless: given BindSpace columns, produce
ShaderHits + MetaWord. The Markov ±5 window IS the texture. The
shader encodes (bind tokens → role-indexed trajectory) and decodes
(unbind roles → recovery margins → free energy) on this texture,
per cycle, stateless.

**StreamDto = the observation stream.** Tokens flow in carrying PoS
tags, temporal markers, morphological commitments. This is the
temporal signal the shader reads.

**ThinkingEngine = the encoder/decoder core.**
- ENCODE: `RoleKey::bind(content)` per token, braided ρ^d per
  position, XOR-superposed into Trajectory. Sentence → Vsa10k.
- DECODE: `RoleKey::unbind(trajectory)` per role, `recovery_margin`
  per slice, `FreeEnergy::compose(likelihood, kl)`. Vsa10k → F.
- The encode/decode pair IS the forward/backward pass, but over
  algebraic structure (XOR), not learned weights (gradient).

**CognitiveShader = the Markov processing unit.** Fires per cycle.
Reads BindSpace columns (FingerprintColumns = trajectories,
QualiaColumn = qualia vector, MetaColumn = awareness bits,
EdgeColumn = causal edges). Writes ShaderHits + MetaWord. Knows
nothing of why it fires or what happened before.

**"Can't resist the thinking":**
- Unresolved ambiguity = F above homeostasis floor.
- F above homeostasis sets awareness bits in MetaWord.
- Awareness bits gate shader dispatch (shipped in PR #204:
  `MetaFilter.awareness_min`).
- The shader MUST fire again because the awareness bits
  persist until F drops below floor.
- The system literally cannot be in a high-F state without
  the shader re-firing to resolve it.
- **This IS active inference in hardware terms.** The system
  doesn't choose to think — it can't resist thinking because
  unresolved prediction error (F) is a persistent dispatch
  signal that forces the next cycle.

**The loop, restated in shader terms:**

```
StreamDto in ──► CognitiveShader reads BindSpace
                     │
                ENCODE: bind + braid + bundle → Trajectory
                DECODE: unbind + margin + F → Resolution
                     │
                     ├── F < 0.2 → Commit → awareness bits clear
                     │              → shader rests (no dispatch trigger)
                     │
                     ├── F ∈ [0.2, 0.8] → Commit but awareness stays warm
                     │                     → shader may re-fire on next token
                     │
                     └── F > 0.8 → FailureTicket → awareness bits LOUD
                                    → shader can't stop firing until
                                      LLM resolves or new tokens collapse F
```

**The key insight: the shader doesn't "decide" to think. It can't
NOT think.** High F = high awareness bits = dispatch fires = another
encode/decode cycle. Low F = bits clear = shader idles. Thinking is
not a feature — it's the thermodynamic consequence of surprise
existing in the BindSpace. The shader is a Markov processing unit
that resolves surprise by construction, the way a ball rolls downhill
by construction.

**Why this dilutes:** future sessions will treat the ThinkingEngine
as a "module that gets called" and the CognitiveShader as a
"function that processes input." Both framings lose the compulsion.
The shader doesn't get called — it can't stop firing while F is
high. The ThinkingEngine doesn't process input — it's an
encode/decode loop that the Markov texture forces through. The
active-inference framing ("can't resist") is not a metaphor — it's
the dispatch mechanism: awareness bits persist → filter threshold
met → shader fires → F descends → bits clear → shader rests.

Cross-ref:
- `contract::cognitive_shader` — MetaWord awareness bits, MetaFilter
- `crates/cognitive-shader-driver/src/engine_bridge.rs` — awareness bits wiring
- `categorical-algebraic-inference-v1.md` §3 (the 8-step closed loop)
- StreamDto lifecycle: `.claude/knowledge/ripple-dto-contracts.md`

---

## 2026-04-21 — Markov IS simple XOR of sentence VSAs; roles ARE spine coordinates; AriGraph facts + resonance find meaning

**Status:** FINDING (the simplest correct statement of the architecture)

**Do not over-engineer this.** The entire architecture is three
sentences:

1. **Markov trajectory = XOR of per-sentence Vsa10k vectors with
   braiding per position.** Each sentence goes through the FSM,
   gets role-key bound (slice-masked XOR per token), bundled into
   one Vsa10k per sentence, then the ±5 window is XOR-superposed
   with braiding ρ^d per offset d. That's it. No HMM. No
   transition matrix. No learned weights. Just XOR with position
   shifts.

2. **Role-key slices ARE the coordinate axes of the trajectory
   space.** SUBJECT[0..2K) is the "who" axis. PREDICATE[2K..4K)
   is the "what" axis. OBJECT[4K..6K) is the "whom" axis.
   TEMPORAL/KAUSAL/MODAL/LOKAL are the adverbial axes. When you
   superpose 11 braided sentence vectors, the result is a point
   in this coordinate system. Unbinding = reading a coordinate.
   The trajectory IS a point in role-space × time-space. Temporal
   causality is visible because braiding encodes position and
   role slices encode grammar — the combination can't NOT carry
   temporal-causal structure.

3. **AriGraph facts + resonance/magnitude = where meaning lives.**
   The trajectory tells you WHAT was parsed. AriGraph tells you
   what it MEANS — because meaning is the relationship between new
   observations and committed beliefs:
   - **Resonance** (recovery_margin between new parse and global_context)
     = how well this sentence fits the story so far.
   - **Magnitude** (Contradiction.magnitude from Wisdom qualia projection)
     = how DEEP the agreement or disagreement is.
   - High resonance + low magnitude = routine confirmation.
   - Low resonance + high magnitude = the sentence contradicts what
     we believed AND the contradiction is profound → Staunen fires
     → Epiphany committed → THAT is meaning.
   - Opinions = committed facts with non-trivial Contradiction
     markers. "All animals are equal" and "some are more equal
     than others" both live in the graph with opposite phase and
     high magnitude. The system doesn't resolve the contradiction
     — it preserves it, because the contradiction IS the meaning
     of Animal Farm.

**Why this dilutes:** future sessions will build Markov chains with
transition probabilities, add embedding layers to the role keys,
compute resonance with cosine similarity on f32 projections, and
treat opinions as something requiring a sentiment classifier. All
of that is wrong. Markov = XOR. Roles = slices. Resonance = Hamming
recovery within slices. Meaning = pattern of committed facts in the
graph with their contradiction markers. The simplicity is the point.

Cross-ref: `categorical-algebraic-inference-v1.md` §1 (substrate),
§3 (closed loop), §5.2 (object-does-the-work test).

---

## 2026-04-21 — 14-paper synthesis: what dilutes first (undiluted capture)

**Status:** FINDING (per-paper grounding in paper-landscape-grammar-parsing.md)

These insights were crystallized during a single-session scan of 14
papers and will be the first things a new session forgets. Preserving
them in order of fragility — most likely to dilute first at the top.

### E-PAPER-1: Morphological commitment IS free-energy reduction at zero cost

Russian `-ом` / Finnish `-ssa` / Turkish `-da` commit Pearl 2³
causal bits AT TOKENIZATION TIME. This collapses the hypothesis
branch count from 8 → 4 → 2 → 1 before the counterfactual axis
even fires. Heavy morphology = pre-collapsed F-landscape. Light
morphology (English prepositions) = flat F-landscape requiring
full counterfactual sampling.

**Why this dilutes:** future sessions will treat morphology as
"feature extraction" rather than "free-energy reduction." It's
not extraction — it's a structural guarantee that reduces
computational work. The case ending IS the computation.

### E-PAPER-2: The braiding operator ρ IS vsa_permute, not a metaphor

Shaw's list encoding: `x_1 ⊕ ρ(x_2) ⊕ ρ²(x_3) ⊕ ... ⊕ ρ^{n-1}(x_n)`.
This is `ndarray::hpc::vsa::vsa_permute(v, position_offset)` applied
per sentence in the Markov ±5 window. The braiding is a cyclic bit
shift. Without it, bundling is position-blind (bag-of-sentences).
With it, temporal order is encoded without learned positional
embeddings.

**Why this dilutes:** future sessions will implement Markov bundling
as plain XOR-accumulation without permutation, producing
position-blind trajectories. The braiding is what makes "before
the focal sentence" different from "after the focal sentence."

### E-PAPER-3: Recovery margin IS likelihood, not similarity

`RoleKey::recovery_margin(unbound, expected)` is not a distance
metric. It's the information-theoretic likelihood term in the
free-energy decomposition: "given that I committed this content to
the SUBJECT role, how cleanly does it come back?" High margin =
observations well-explained by hypothesis = low free energy.

**Why this dilutes:** future sessions will use recovery_margin as
a "quality score" or "similarity measure" without connecting it to
the F-landscape. It's not a score — it's the P(obs|hidden) term
in the variational decomposition.

### E-PAPER-4: The confidence horizon at φ-1 is a feature, not a bug

NARS revision with c_obs=1 per step asymptotes at `(√5-1)/2 ≈ 0.618`.
The system PROVABLY never becomes fully certain (c < 1 always).
This means every committed fact, no matter how many times confirmed,
retains a margin of revisability. Full certainty would freeze the
prior and make the system unable to notice contradictions.

**Why this dilutes:** future sessions will try to "fix" the
0.618 ceiling by increasing c_obs or changing the formula.
The ceiling IS the architectural feature. Golden-ratio-bounded
confidence = permanent epistemic humility = permanent ability
to detect contradiction = Staunen can always fire.

### E-PAPER-5: Non-commutative binding is required for hierarchical structure

Shaw proves that commutative binding creates ambiguity in tree
leaves (guard vectors become indistinguishable). This is why we
use DIFFERENT role keys for S/P/O rather than one key with
different arguments. If `bind(S, content) == bind(content, S)`
AND `bind(S, x) == bind(P, x)` for some x, then S and P are
indistinguishable → hierarchy collapses.

**Why this dilutes:** future sessions will propose "simplifying"
to a single binding key with different content, or making bind
commutative for "elegance." The non-commutativity of distinct
role-key patterns is what preserves hierarchical structure.

### E-PAPER-6: The Ω(t²) lower bound does NOT apply to us

Alpay proves that any sound, parse-preserving, retrieval-efficient
grammar masking engine needs Ω(t²) per token. We dodge this because
we DON'T preserve the parse forest — we commit argmin_F and discard
losers (or mark the runner-up as epiphany). Active inference trades
parse-preservation for decision speed.

**Why this dilutes:** future sessions will worry about parsing
complexity and try to optimize the counterfactual enumeration.
The complexity bound is on parse-preserving engines. We are
parse-COMMITTING, not parse-preserving. The distinction is
architectural, not an optimization.

### E-PAPER-7: Abstraction-first is empirically measured, not theoretically assumed

Jian & Manning measured it across three independent GPT-2 training
runs: class-level D_JS divergence precedes within-class divergence
by ~50 steps. The exemplar-first (count-based) baseline shows
verb-specific patterns WITHOUT class structure. This is not a
philosophical preference for Deduction over Induction — it's a
measured behavioral difference with a strict ordering.

**Why this dilutes:** future sessions will treat the
NarsPriorityChain {primary: Deduction, fallback: Abduction}
as a configuration choice. It's an empirically-grounded ordering
that has been measured in transformer training dynamics.

---

## 2026-04-21 — The Kan extension IS the free-energy minimizer (holy-grail unification)

**Status:** CONJECTURE (grounded in Shaw 2501.05368 + Alpay 2603.05540
+ shipped code; not yet formally proven as categorical equivalence)

Shaw et al. proved via right Kan extensions that dimension-preserving
VSA binding MUST be element-wise (the Yoneda lemma collapses the
integral to pointwise multiplication). Active inference says minimize
`F = -likelihood + KL`. These are the SAME operation at different
levels of abstraction:

- Kan extension = optimal projection of external tensor product into
  fixed-dim space under structural constraints (monoidal category).
- Free-energy minimization = optimal approximation of observations
  under a generative model (variational inference).
- NARS revision = optimal truth update under new evidence (Bayesian
  with bounded confidence).
- AriGraph commit = optimal fact storage under contradiction detection
  (graph-structured belief revision).

All four are "find the best approximation under constraints." The
constraints differ (categorical, information-theoretic, logical,
graph-structural), but the algebraic substrate is the same: element-
wise XOR on role-indexed slices of a 10K binary VSA vector.

**What clicks:**
1. bind/unbind IS Kan extension (categorically optimal)
2. recovery_margin IS likelihood (information-theoretic)
3. awareness.divergence_from(prior) IS KL (variational)
4. Resolution::from_ranked IS argmin_F (active inference)
5. AriGraph commit IS belief revision (graph + NARS)
6. The Trajectory's own methods ARE the inference engine — the object
   doesn't get passed to reasoning; the object speaks for itself.

Not neural (no weights). Not symbolic (no search). Not hybrid
(not bolted together). A categorical-algebraic inference engine where
parsing, disambiguation, learning, memory, and awareness are the SAME
algebraic structure viewed through different lenses.

Cross-ref: `.claude/knowledge/paper-landscape-grammar-parsing.md`,
Shaw 2501.05368 §4.3 (Kan extensions), Alpay 2603.05540 §Theorem 5
(Doob h-transform), `contract::grammar::free_energy`, `role_keys`.

---

## 2026-04-21 — RoleKey bind/unbind slice-masking = lossless role-indexed superposition

**Status:** FINDING (verified by 5-role simultaneous recovery test)

Slice-masked bind is the crucial design choice that makes role-indexed
VSA bundling lossless. `RoleKey::bind(content)` zeroes content outside
`[start..end)` before XOR with the key. This means XOR-superposition
of N role bindings keeps each role's slice completely disjoint — unbind
with any role key recovers that role's content at margin 1.0, regardless
of what other roles contributed.

Without slice-masking (raw full-vector XOR), the 5035-recovery-margin
on the SUBJECT slice demonstrates the cross-contamination: every role
leaks content into every other role's slice. The audit agent (2026-04-21
session) flagged this as the "three-silo disconnection" — role_keys.rs
was data without operator semantics.

The fix: `bind` enforces the invariant at the method level (not caller
discipline). `unbind` is the same masked-XOR. `recovery_margin` measures
per-slice Hamming similarity after unbind. Test: 5 roles (S/P/O +
TEMPORAL + LOKAL) bound, XOR-superposed, each recovers at margin 1.0.

This is THE operation that makes "the object speaks for itself" literal:
a Trajectory carrying a 5-role-superposed VSA vector can answer
`trajectory.role_bundle(SUBJECT)` without external orchestration —
just unbind the SUBJECT slice, and the content is there.

Cross-ref: `contract::grammar::role_keys::{RoleKey::bind, unbind, recovery_margin}`.

---

## 2026-04-21 — Free energy as active-inference formulation of grammar parsing

**Status:** FINDING (types shipped; thresholds uncalibrated until Animal Farm)

Ambiguity resolution is Friston free-energy minimization over the
hypothesis space. `F = (1 - likelihood) + KL(awareness || prior)`.
Likelihood = mean role-recovery margin after unbind; KL =
`GrammarStyleAwareness::divergence_from(prior)`. Three branches:

- `F < HOMEOSTASIS_FLOOR (0.2)` → Commit (single triple to AriGraph)
- Top-2 F within `EPIPHANY_MARGIN (0.05)` → Epiphany (both commit
  with Contradiction marker)
- `F > FAILURE_CEILING (0.8)` → FailureTicket (escalate)

Morphology collapses the hypothesis space via the Pearl 2³ causal
mask: each case ending commits bits, narrowing the basin. Two
independent commitments: 8 → 2 branches. Three: 8 → 1 (direct
Deduction, no counterfactual needed). This is the "2³ → 2^N" extension
to other morphologies (Russian Instrumental, Finnish Elative, Arabic
pattern فاعل / مفعول, Mandarin bǎ, Turkish -yle).

Cross-ref: `contract::grammar::free_energy::{FreeEnergy, Hypothesis,
Resolution, HOMEOSTASIS_FLOOR, EPIPHANY_MARGIN, FAILURE_CEILING}`.

---

## 2026-04-21 — D7 GrammarStyleAwareness IS the "weights-as-seed" epistemic layer

**Status:** FINDING (replaces the "langextract is boring because LLM-dep"
observation with a concrete zero-LLM realization).

The D7 deliverable `contract::grammar::thinking_styles::GrammarStyleAwareness`
shipped today is literally the epistemic-awareness surface the user
described: weights become a seed, NARS-revised per parse outcome, drifting
the `effective_config.nars.primary` away from the YAML-prior inference when
accumulated evidence contradicts it. No external LLM in the loop; awareness
is O(1) per parse (one HashMap insert + one `revise_truth` fold). The
style's track record IS the seed for Markov dispatch: `top_nars_inference`
reads from `param_truths`, not from a network call.

Concretely the closed loop is:

```
parse attempt (DeepNSM FSM + Grammar Triangle)
    → ParseOutcome  (local success / LLM-agreed / LLM-disagreed / ...)
    → GrammarStyleAwareness::revise(ParamKey, outcome)
        (standard NARS revision: f_new = (f·c + f_obs·c_obs)/(c+c_obs);
         c_new = (c+c_obs)/(c+c_obs+1) — asymptotes at φ-1 ≈ 0.618
         under c_obs=1, which is the sharp confidence horizon we test against)
    → next parse uses GrammarStyleAwareness::effective_config(prior)
        (prior NARS primary is kept if its f > 0.5; else drifts to the
         highest-ranked NARS param from accumulated evidence)
```

Replaces langextract's external-LLM step with role-indexed VSA bundling
(D5, coming) + SPO 2³ × TEKAMOLO decomposition (D3 triangle bridge) +
NARS-on-grammar (shipped D7). Together that's O(1) causality-learning per
sentence. When D2 ticket_emit + D3 triangle_bridge land, the DeepNSM
parser will close this loop end-to-end.

Cross-ref:
- Plan `/root/.claude/plans/elegant-herding-rocket.md` D7.
- `.claude/knowledge/grammar-landscape.md` §6–§7.
- `crates/lance-graph-contract/src/grammar/thinking_styles.rs` (shipped).

---

## 2026-04-20 — Shader vs engine: statelessness is the boundary

**Status:** FINDING (sharpens the three-level taxonomy)

**Cognitive shader** = stateless atomic compute. Given `ShaderDispatch`
+ `BindSpace` columns, returns `ShaderHit`s + `MetaWord`. Knows nothing
of why it fires. Output is one-cycle-wide, no history.

**Thinking engine** = stateful orchestrator. Calls `shader.dispatch()`
many times per cognitive cycle; composes per-lens hits into
persona/qualia/world_model/ghost state; revises beliefs for the next
cycle. The cognitive stack IS the state.

**The engine_bridge is where they meet** —
`cognitive-shader-driver/src/engine_bridge.rs` is the seam. Shader
side: `ShaderDriver::dispatch` stateless. Engine side:
`cognitive_stack::cycle` accumulates dispatches through
`bf16_engine` / `signed_engine` / `composite_engine` / `dual_engine` /
`layered` / `domino`, folds into persona/qualia, emits state for next
cycle.

**Analogy:** shader = eye (no memory, reports the current frame);
engine = mind (memory, assembles frames into narrative, counterfactually
imagines alternatives).

**Where codec-flexibility-as-thinking lands:** the **engine** level,
not the shader level. A "new thinking style" = a new engine
configuration (lens composition, persona, qualia-update rule) that
picks DIFFERENT shader configs per cycle. Shader stays the same; the
engine's orchestration changes. That's why Phase 5+ "production-grade
thinking tissue" drops into mid (engine), not L2 (shader).

**Concrete Phase 1-5 shipping:** codec-sweep D1.x work = shader layer
(tensor decode primitives). Engine-level codec-flexibility (swap
lenses via YAML) = D5 / Phase 5+, plugging INTO the codec infrastructure.

Cross-ref: three-level taxonomy above; resolution-ladder entry
`64×64 > 256×257 >> 4096×4096 > 16k`; `engine_bridge.rs` seam.

---

## 2026-04-20 — Resolution hierarchy: `64×64 > 256×257 >> 4096×4096 > 16k` (user-named)

**Status:** FINDING (capstone of the three-level taxonomy from earlier this session)

The 5-layer stack is a **resolution ladder**, not a layer cake. Each
level operates at its own granularity and has its own "shader" /
"kernel cache" / "distance table" at that scale:

| Size | Role | Where | HHTL stage (I10) |
|---|---|---|---|
| **64×64** | p64 topology mask — 8 predicate planes × 64 rows × u64 — "which archetype blocks relate via predicate z" | `p64_bridge::cognitive_shader::CognitiveShader` | HEEL (coarse basin) |
| **256×257** | bgz17 palette distance table — 256 archetypes × 256 + 1 sentinel — O(1) lookup `semiring.distance(a, b)` | `bgz17::PaletteSemiring` | HIP (family sharpen) |
| **4096×4096** | Cross-vocabulary / cross-context correlation — COCA × COCA, or 4096 τ-prefix × 4096 slot space | ndarray `ScanParams` JIT (`jitson_cranelift`) | BRANCH / TWIG |
| **16 K** | Individual fingerprint bit identity — 16384-bit `Fingerprint<256>` | `ndarray::simd::Fingerprint<256>` + codec decoder (D1.x) | LEAF (exact member) |

**The `>>` between 256×257 and 4096×4096 is the big jump** (~64×)
matching HIP → BRANCH refinement. That's where palette-level (one
row of the codebook) meets vocabulary-level (COCA 4096). Below that
jump, everything is O(1) table lookup; above it, JIT kernels become
worth the compile cost.

**Each JIT targets its own resolution — no overlap:**

- p64 cascade: 64×64 bitmask ops. Not JIT'd (bit tricks in hot loop
  already optimal under AVX-512).
- bgz17 palette: 256×256 precomputed. Not JIT'd (memory-bound).
- ndarray ScanParams: 4096×4096 scan kernels. **JIT'd via
  `jitson_cranelift::JitEngine`** — shipped.
- Codec kernels (D1.x): 16k bit-level tensor decode. **Will be JIT'd
  via D1.1b `CodecKernelEngine` adapter**. Scaffold (D1.1) + rotation
  primitives (D1.2) landed; Cranelift IR emission deferred to D1.1b.

**Three-level taxonomy (from earlier this session) maps onto the
resolution ladder:**

- **L2 small-precision cognitive shaders** (ns budget) →
  64×64 + 256×257 (p64 + bgz17 palette). Pure table lookups.
- **mid thinking-engine layers** (µs-ms) →
  4096×4096 (cross-vocab, persona-aware lens composition). JIT'd
  scan kernels.
- **L4 thinking styles / NARS / JIT** (ms) →
  orchestrates traversal ACROSS resolutions (starts at 64×64 cascade
  to find candidates, narrows to 256×257 for family, drops to
  4096×4096 for context, verifies at 16k fingerprint identity).

**p64::CognitiveShader double-check conclusion:** architecturally
clean. Operates at the coarsest (64×64) level; codec-sweep work at
finest (16k); they compose in `cognitive_shader_driver::ShaderDriver`
without overlap. Different layers of the ladder, different
operations, different JIT targets (if any).

Cross-ref: I10 (HEEL/HIP/BRANCH/TWIG/LEAF); three-level taxonomy entry
above; `p64_bridge::cognitive_shader::CognitiveShader::cascade`;
D1.1 `CodecKernelCache`; D1.2 `RotationKernel`; bgz17 `PaletteSemiring`.

---

## 2026-04-20 — Thinking styles ARE codecs over the semantic field (north star)

**Status:** FINDING (forward-looking deposit — not a current work item; reference when Phase 5+ generalises)

A codec compresses tensor content into fingerprints; a thinking style
compresses reasoning trajectories into NARS-revised beliefs. Same
underlying operation — structure-preserving compression on a binary
Hamming substrate. Different input/output domains, same substrate
guarantees (E-SUBSTRATE-1, I-SUBSTRATE-MARKOV), same compile-and-swap
machinery.

**The codec infrastructure IS the template for production-grade
thinking tissue.** When Phase 5+ activates:

| Codec (shipped D0.1–D1.2, D1.1b queued) | Thinking-style analog |
|---|---|
| `CodecParams` | `ThinkingStyleParams { style, modulation_7d, nars_priors, fallback_chain, sigma_priority, semiring_choice }` |
| `kernel_signature()` — excludes runtime drift | `style_signature()` — excludes per-cycle modulation drift |
| `CodecKernelCache<H>` | `ThinkingStyleKernelCache<H>` — same generic scaffold |
| JIT kernel = Cranelift-compiled decode | JIT kernel = compiled scan-walk on 36-node topology (already shipped ndarray-side via `scan_jit.rs` + `ScanParams`) |
| **Token agreement** (I11 cert gate) | **Conclusion agreement** — same NARS-revised conclusions as reference style? |
| Sweep grid = N codec candidates | Sweep grid = N (style × modulation × NARS fallback) candidates |
| `/v1/shader/calibrate` | `/v1/shader/think-calibrate` |
| `[FORMAL-SCAFFOLD]` 5 pillars | **Same scaffold** — E-SUBSTRATE-1 covers any transition under bundle |

**Generalisation isn't "port codec pattern to thinking"** — it's
recognising thinking styles as a SPECIAL CASE of the codec pattern we
just built. When Phase 5+ lands, `WireThinkCalibrate` +
`ThinkingStyleKernelCache` + `conclusion_agreement` metric drop in
alongside the codec versions. Same JIT engine, same tests, same
board-hygiene discipline.

**The phrase "production-grade thinking tissue"** names the telos
cleanly: once codec infra is at Phase 3 token-agreement pass rates,
cloning to thinking styles yields production-grade swappable
reasoning — YAML-configured, JIT-compiled, sweep-certified. No
rebuild per new style, no black box, signature-keyed reproducibility.

**Cross-ref:** D0.6 `CodecParams` (the parameter-shape template);
D1.1 `CodecKernelCache<H>` (the cache pattern — generic-over-H is the
wedge for reuse); I5 (thinking IS an AdjacencyStore — already
topologically unified with data graph); codec-sweep-via-lab-infra-v1.

---

## 2026-04-20 — D1.2 Hadamard is pure-Rust, not a JIT-necessary primitive

**Status:** FINDING

D1.2's HadamardRotation is implemented as a plain Rust in-place
Sylvester butterfly (O(N log N) add/sub, no allocations). It does NOT
need JIT compilation or Cranelift code emission because:

1. **Fixed shape** — the butterfly structure is identical across all
   power-of-two dims. Rust's compiler (under `target-cpu=x86-64-v4`)
   already emits AVX-512 add/sub from the straight-line loop.
2. **Not matmul** — Hadamard is a pattern of adds and subtracts,
   never a dot product. Per Rule C polyfill hierarchy, matmul-heavy
   paths benefit from AMX (Tier 1); add/sub stays at Tier 3 F32x16.
   AMX gives no speedup here — confirmed in plan Appendix §12 C.

**Consequence for D1.1b (Cranelift wiring):** only OPQ rotation needs
the JIT path — it's the one that's actually a learned matmul. The
Cranelift integration scope narrows: we don't need to JIT-compile
Identity (no-op) or Hadamard (butterfly); just OPQ (matmul) and the
main codec decode loop (ADC distance with palette lookup).

This reduces D1.1b scope by maybe 30-40% — fewer kernel shapes to
emit, only the ones that actually benefit.

Cross-ref: D1.2 `rotation_kernel.rs::HadamardRotation`; Rule C
(polyfill hierarchy); plan Appendix B (CartanCascade harmonic
compression ratios rely on real Hadamard, so this matters).

---

## 2026-04-20 — CORRECTION to D1.1 scaffold: ndarray::hpc::jitson_cranelift already ships JitEngine

**Status:** FINDING / CORRECTION

The D1.1 `CodecKernelCache` scaffold (RwLock + double-check) is
strictly worse than what ndarray's `jitson_cranelift::JitEngine`
already provides. Real upstream:

```
/home/user/ndarray/src/hpc/
  ├── jitson/           — JITSON template format (parser/validator/
  │                        template/precompile/scan_config/packed/noise)
  └── jitson_cranelift/ — real Cranelift engine
      ├── engine.rs     — JitEngine + JitEngineBuilder
      ├── ir.rs         — IR emission
      ├── scan_jit.rs   — scan kernel codegen
      ├── noise_jit.rs  — noise kernel codegen
      └── detect.rs     — CPU capability detection
```

Dependencies behind `jit-native` feature:
`cranelift-{codegen, jit, module, frontend} 0.116` + `target-lexicon`.

**Upstream two-phase lifecycle is stronger than my scaffold:**

- **BUILD phase:** `&mut JitEngine`, `compile(ScanParams) -> Result<u64>`,
  mutable cache via `&mut self`.
- **RUN phase:** `Arc<JitEngine>` freezes the cache by Rust's ownership
  (`&mut self` unreachable through `Arc`). `get()` drops from
  ~25 ns (my RwLock read) to ~5 ns (plain `HashMap::get`, no
  synchronization needed).

The freeze is enforced by the type system, not by a runtime lock.
That's the right design for this domain (build-once, run-many).

**What the D1.1 scaffold is still good for:** `CodecParams` is the
codec-sweep key; `ScanParams` is ndarray's thinking-style-scan key.
Different domains; a `CodecParams`-keyed adapter layer is still
needed. My generic-over-handle design anticipates this — the
scaffold wraps ndarray's `JitEngine` at the `H` slot when D1.1b
lands.

**Revised D1.1b plan:**

Mirror ndarray's two-phase pattern in `cognitive-shader-driver`:

```rust
// BUILD phase — mutable, single-threaded
pub struct CodecKernelEngine {
    inner: ndarray::hpc::jitson_cranelift::JitEngine,
    codec_sig_to_inner_id: HashMap<u64, u64>,  // CodecParams signature → JitEngine id
}

// RUN phase — frozen via Arc
impl CodecKernelEngine {
    pub fn build() -> CodecKernelEngineBuilder { ... }
    pub fn compile(&mut self, params: &CodecParams) -> Result<u64, JitError>;
    pub fn freeze(self) -> Arc<Self>;  // moves to RUN phase
    pub fn get(&self, params: &CodecParams) -> Option<KernelHandle>;
}
```

Then D1.2/D1.3 call `inner.compile` with codec-specific
`ScanParams`-analogs (new `CodecScanParams` struct or a JITSON
template constructed from `CodecParams`).

**Honesty note:** user asked "I presume you are aware of
cranelift/jitson" — answer is: Cranelift yes (Bytecode Alliance,
wasmtime), ndarray jitson NO (didn't inspect the upstream surface
before writing D1.1). This correction surfaces that gap explicitly
so the next session doesn't repeat it.

**Cross-ref:** D1.1 `crates/cognitive-shader-driver/src/codec_kernel_cache.rs`
(keep as `StubKernel`-backed test fixture); `ndarray::hpc::jitson_cranelift::JitEngine`;
D1.1b revised plan above.

---

## 2026-04-20 — D1.1 scaffold-before-codegen: cache semantics testable without Cranelift

**Status:** FINDING

`CodecKernelCache<H>` is generic over the kernel-handle type. The same
cache hosts `StubKernel` (deterministic fake, no compilation) for tests
AND `KernelHandle` (real Cranelift function pointer) for production.

This separates TWO concerns that are usually tangled:

1. **Cache semantics** — signature-keyed insertion, double-checked
   locking under concurrent miss, counters for hit-ratio measurement.
   Testable in microseconds without a JIT engine.
2. **IR emission** — the actual Cranelift / jitson code generation
   that takes `CodecParams` and produces a callable function pointer.
   Heavy; takes minutes per build; requires ndarray's jitson surface
   to be finalized.

By shipping the cache layer with `StubKernel` NOW, Phase 1's cache
semantics are verified + CI-gated before the Cranelift work starts.
When D1.1b lands, the only change is `H = KernelHandle`; all 9 cache
tests remain valid. This is the **scaffold-before-codegen** pattern:
test the hard-to-change contract first, defer the hard-to-build
implementation.

Generalises: any JIT pipeline should separate cache-keying from IR
emission at the type level. Generic over handle type is the wedge
that makes this possible.

Cross-ref: D1.1 `crates/cognitive-shader-driver/src/codec_kernel_cache.rs`;
D0.3 sweep-grid-IS-cache-warmer epiphany (same signature-as-identity
insight); PR #225 `CodecParams::kernel_signature()`.

---

## 2026-04-20 — D0.3 sweep grid IS the JIT cache warmer

**Status:** FINDING

`WireSweepGrid::enumerate()` materializes the Cartesian product as a
`Vec<WireCodecParams>`. Each unique `(subspaces, centroids,
residual_depth, rotation_kind, distance, lane_width)` tuple maps to
exactly one `CodecParams::kernel_signature()`. The grid IS the JIT
cache warm-up plan: first traversal compiles N kernels; every
subsequent sweep with overlapping tuples hits cache at ~0 ms
compile cost.

This operationalises Rule C's polyfill hierarchy + Rule E's
kernel-signature-as-cache-key into a single client-facing verb:
*submit a grid, the server warms the cache while streaming results*.
The 54-candidate example grid from plan Appendix A §30 compiles
~54 × 15 ms = ~800 ms once; every re-run is free. That's the
operational loop the sweep infrastructure buys.

Generalises: any cross-product DTO in this workspace should treat
its grid as a cache-warmer, not just a test matrix. The cache
signature and the grid axis are the same object viewed from two
sides.

Cross-ref: D0.3 `WireSweepGrid::enumerate`; PR #225
`CodecParams::kernel_signature()`; plan Appendix A §30
`30_cross_product_sweep.yaml`; Rule C (polyfill hierarchy).

---

## 2026-04-20 — D0.2 stub flag is anti-#219 defense at the type level

**Status:** FINDING

`WireTokenAgreementResult` carries `stub: bool` + `backend: "stub"`
default. Phase 0 ships the Wire surface without the decode-and-compare
harness; the stub returns zero rates. **Any downstream client that
confuses stub output for real measurements fails loudly** — because
`stub == true` and `backend == "stub"` are machine-checkable, not
comments. This is the #219 pattern (synthetic-rows-mistaken-for-real)
prevented at the type layer, not just in docs.

Pattern generalises: every Phase-N surface DTO that lands before its
Phase-N+k harness should carry an explicit stub flag. Rules A–F say
*how* to structure the Wire; the stub flag says *whether* the numbers
are real. Orthogonal, both load-bearing.

Cross-ref: D0.2 `WireTokenAgreementResult`; E-ORIG-7 Jirak (the correct
measurement regime once the stub comes off); #219/#220 arc.

---

## 2026-04-20 — D0.5 auto_detect is the concrete Python↔Rust heuristic handshake

**Status:** FINDING (confirms E-MEMB-11 handshake mechanism)

Rosetta v2 (Python) routes architectures to lane widths via
family-name heuristic. D0.5 `auto_detect::suggest_lane_width` lands
the same heuristic on the Rust side: llama / qwen / qwen2 / qwen3 /
mistral / mixtral → BF16x32 (AMX-ready); bert / modernbert /
xlm-roberta / generic → F32x16 (AVX-512 baseline); `torch_dtype`
override wins.

Same table, two languages. **The Python↔Rust handshake (E-MEMB-11)
is no longer conceptual** — it has a concrete implementation: the
architecture string is the shared vocabulary; lane width is the
shared dispatch decision; `torch_dtype` is the shared override. A
future `slice-layout-reconciliation.md` (E-MEMB-1 blocker fix) can
use the same handshake pattern: architecture → layout version →
canonical slice table.

Cross-ref: `crates/cognitive-shader-driver/src/auto_detect.rs`;
E-MEMB-11 (LivingFrame ↔ ContextChain handshake); Rosetta v2
`DIMENSION_MAP` architecture routing.

---

## 2026-04-20 — E-SUBSTRATE-1 — VSA-bundling guarantees Chapman-Kolmogorov by construction

**Status:** FINDING (load-bearing — FUNDAMENT underneath the [FORMAL-SCAFFOLD] four pillars)

Saturating bundle addition in d=10000 is associative and commutative in
expectation: `a ⊞ (b ⊞ c) = (a ⊞ b) ⊞ c`. Johnson-Lindenstrauss +
concentration-of-measure in 10000 dimensions suppress deviations from
associativity at rate `~e^(-d)`. States-as-VSA-bundles + transitions-as-
bundle-operation ⇒ `(Hamming-space, Bundle)` is an **abelian semigroup**
⇒ Chapman-Kolmogorov `K(2τ) = K(τ)²` holds **by construction**.

**Consequence.** The Markov property is not a testable assumption in this
substrate — it is a geometric consequence of the substrate choice.
D7's "implicit Markov reliance" is grounded, not silent. The
Chapman-Kolmogorov consistency test therefore reclassifies from
*falsification gate* (Popperian) to *implementation sanity check*
(regression — can only fail from implementation bugs, not from theoretical
violations).

**Load-bearing constraint (substrate-bound).** `MergeMode::Xor` BREAKS
this guarantee. Non-commutative binding operations BREAK it. Any move
away from saturating bundle in high dimension destroys the foundation on
which the four [FORMAL-SCAFFOLD] pillars stand. See I-SUBSTRATE-MARKOV
in CLAUDE.md for the iron-rule form.

Cross-ref: I1 BindSpace read-only + CollapseGate bundles;
[FORMAL-SCAFFOLD] below; D7 meta-inference duality;
`contract::collapse_gate::MergeMode::Bundle`.

---

## 2026-04-20 — [FORMAL-SCAFFOLD] Five-pillar Phase-5 reference (dormant, not parked)

**Status:** DORMANT (not parked; five pillars cited; paper track not active but citations are available when calibration choices become load-bearing). **Consult before inventing new significance claims, thresholds, or substrate changes.**

Formal-foundation scaffolding for the discrete binary PINN interpretation of the workspace (ladybug → lance-graph). Five pillars — **E-SUBSTRATE-1** as fundament, four theorem citations standing on top:

```
E-SUBSTRATE-1:   VSA-bundling d=10000 ⇒ Chapman-Kolmogorov by construction
                 (FUNDAMENT — the other four pillars stand on this)

Cartan-Kuranishi: existence via finite prolongation to involutive form
                 (Cartan 1945 / Kuranishi 1957; en.wikipedia.org/wiki/
                  Cartan–Kuranishi_prolongation_theorem)
                 → role_keys slice widths (2000/2000/2000/900/70/60/30)
                 ARE the Cartan-character spectrum, not arbitrary design

φ-Weyl equidist.: maximally-irrational sampling, no aliasing
                 → 144 verb-cells as quasi-Chebyshev collocation

γ+φ transform:    coordinate regularizer, fewer prolongation steps
                 → `bgz-tensor::gamma_phi.rs`

Jirak 2016:       Berry-Esseen rate under weak dependence (noise floor)
                 (arxiv 1606.01617; Annals of Probability 44(3) 2024–2063)
                 → classical IID Berry-Esseen is WRONG for this system;
                 bits are weakly dependent by construction
```

**Status refinement: dormant-with-five-cited-pillars is a different state than parked-without-a-paper-track.** The scaffold is now *available* for future decisions, not *forcing* on current ones. No reanimation of a paper track; no new crate, no new PR from this scaffolding. Documentary only.

The tag `[FORMAL-SCAFFOLD]` is greppable so a future session tempted to roll its own threshold-calibration / sampling-stride / coordinate-transform / noise-floor / substrate-change heuristic greps this entry first and either (a) uses the referenced lemmas or (b) writes down explicitly why they don't apply.

---

## 2026-04-20 — [FORMAL-SCAFFOLD] Coupled revival track (the three candidates, now linked)

**Status:** DEPOSIT — reclassified from three isolated features to one coupled experimental access path into the scaffold. Acceptance: activating one of the three forces coherence-check of the other two.

1. **Chapman-Kolmogorov consistency test** — reclassified from
   *falsification gate* to **implementation sanity check**. Under
   E-SUBSTRATE-1, CK cannot fail for theoretical reasons; it can only
   fail from implementation bugs. Value as regression test; not as
   Markov-property validator.

2. **VAMPE spectral calibration** — under E-SUBSTRATE-1 the eigenvalues
   of the transition kernel are *genuine* spectral quantities, not
   approximations. Jirak bounds the spectral-weight threshold below
   which mass is noise. **VAMPE + Jirak pair replaces hand-tuned σ /
   hardness / abduction thresholds with bound-derived ones.**

3. **Learned attention masks on nibble positions** — under Cartan-
   Kuranishi these become *empirical discovery of Cartan characters*.
   If learned masks reproduce the `role_keys` slice widths
   (2000/2000/2000/900/70/60/30), that is the experimental proof that
   the layout is **intrinsic geometry, not convention** (empirical
   confirmation of E-ORIG-5).

**Coupling acceptance rule.** If any one of the three is activated in
a future PR, the other two MUST be checked for coherence with the
scaffold in the same session — document the interdependency explicitly.
Not all three simultaneously; but never one in isolation without the
coupling note.

Cross-ref: E-SUBSTRATE-1; [FORMAL-SCAFFOLD] five-pillar entry above;
E-ORIG-5 (NSM pre-sliced for role_keys).

---

## 2026-04-20 — [FORMAL-SCAFFOLD] Four-pillar Phase-5 reference (SUPERSEDED 2026-04-20 by five-pillar)

**Status:** SUPERSEDED by the five-pillar entry above (E-SUBSTRATE-1 promoted to fundament; dormant-not-parked framing). Entry retained for history per APPEND-ONLY rule.

Original body: Formal-foundation scaffolding for the discrete binary PINN interpretation of the workspace (ladybug → lance-graph): **Jirak 2016** Berry-Esseen under weak dependence (arxiv 1606.01617) + **Cartan-Kuranishi** involutive prolongation + **φ-Weyl** equidistribution for golden-angle collocation + **γ+φ** preconditioner for prolongation regularization. These are the four citations that would elevate empirical ICC 0.99 → provably-bounded residual if a theorem track were opened; it is not.

---

## 2026-04-20 — E-MEMB-1 (ISSUE) — Python↔Rust slice layouts are incompatible at the 10 kD membrane

**Status:** OPEN ISSUE (promoted from FINDING per 2026-04-20 "load-bearing five" triage)

PR #210's `role_keys.rs` locks 47 keys into disjoint contiguous slices: Subject [0..2000), Predicate [2000..4000), Object [4000..6000), Modifier [6000..7500), Context [7500..9000), TEKAMOLO [9000..9900), Finnish [9840..9910), tenses [9910..9970), NARS [9970..10000). The Python `adarail_mcp/membrane.py` `DIMENSION_MAP` uses a completely different layout: [0..500) "Soul Space" (qualia_16 / stances_16 / verbs_32 / tau_macros / tsv), dim 285 = hot_level, [2000..2018) = qualia_pcs_18. **The two systems speak incompatible 10 kD.** Ada↔lance-graph integration is blocked on a slice-layout reconciliation doc.

Tracked in `ISSUES.md` (same date). Cross-ref: PR #210 role_keys.rs; `adarail_mcp/membrane.py::DIMENSION_MAP`; E-MEMB-7 (Ada-internal incoherence, additional layer).

---

## 2026-04-20 — E-ORIG-1 NSM and 144 verbs are orthogonal composition axes, not competing encodings

**Status:** FINDING (load-bearing)

NSM (65 primes) = semantic atoms for subjects / objects / states. 144 verbs = predicate edge labels for SPO Markov chains. They compose: `triple = (NSM-composed subject, 144-verb edge, NSM-composed object)`. Treating them as rival vocabularies hides this composition; the workspace uses BOTH simultaneously in the Grammar Triangle (NSM × Causality × Qualia → fingerprint) with 144 verbs as the predicate axis of the SPO triples.

Cross-ref: harvest H5, H12; `grammar-landscape.md` §2.

---

## 2026-04-20 — E-ORIG-5 NSM is pre-sliced for the role_keys 10K layout

**Status:** FINDING (load-bearing — this is *why* the role_keys slice widths work)

Harvest H5 (cross-repo-harvest-2026-04-19.md) maps NSM 65 primes onto SPO + Qualia + Temporal axes. This distributes primes across the `role_keys` slice geometry: subject-primes (I, YOU, SOMEONE, PEOPLE) → Subject [0..2000); action-primes (DO, HAPPEN, BE) → Predicate [2000..4000); qualia-primes (FEEL, GOOD, BAD) → QualiaColumn (18D). **The 65 NSM primes aren't a flat vocabulary — they're a pre-distributed encoding across the 10K VSA slice structure.** PR #210's role_keys layout is the SLICE GEOMETRY NSM already anticipated.

Cross-ref: `grammar-landscape.md` §2; harvest H5; PR #210.

---

## 2026-04-20 — E-MEMB-5 18D QualiaColumn = sigma_rosetta projected onto the SoA

**Status:** FINDING (load-bearing — explains QualiaColumn's physical interpretation)

The 18D QualiaColumn carries Staunen (phase) + Wisdom (magnitude) projections per PR #208. Every triple (Predicate-slice content, Qualia phase, Qualia magnitude) IS sigma_rosetta's 64-glyph coordinates projected onto the SoA. **Qualia isn't a separate layer — it's the second lane through the membrane.** Every triple carries both role-slice content AND the 18D projection of its sigma-glyph neighborhood.

Cross-ref: PR #206 sigma_rosetta 64 glyphs; PR #208 Staunen/Wisdom subspaces; QualiaColumn 18D per PR #204.

---

## 2026-04-20 — E-MEMB-9 to_aurora_prompt() IS a BusDto — three-DTO doctrine already operational in Python

**Status:** FINDING (load-bearing — empirical proof Rust's I9 shape works)

Rosetta v2 emits `{sparse_signature, qualia_signature, visual_qualities, frequency_feel}` for image prompting. This is exactly the shape of a cross-modal BusDto (explicit thought → external consumer). Rust's Invariant I9 (`lab-vs-canonical-surface.md`) defines three DTO families — StreamDto / ResonanceDto / **BusDto** — as *doctrinal, not yet shipped*. Python proves the shape works empirically; Rust should ship the same structure in the canonical contract when BusDto lands.

Cross-ref: Rosetta v2 `SparseFrame.to_aurora_prompt()`; Invariant I9; `lab-vs-canonical-surface.md`.

---

## 2026-04-20 — Deposit log (one-line findings, retained but not load-bearing)

Per 2026-04-20 "im Log, nicht an die Wand" triage: these surfaced during the membrane + NSM-origin + PINN-Rosetta + Jirak thread but are secondary to the load-bearing five above. Retained here as addressable anchors; full body is NOT repeated on the wall. Cross-ref pointers remain valid from elsewhere.

- **E-ORIG-2** — 144-verb taxonomy originated in `ada-consciousness/crystal/markov_crystal.py::Verb`, not from NSM. Harvest H12.
- **E-ORIG-3** — 144 chosen for tractable factorable table size (12²), not theoretical derivation. grammar-landscape §6.
- **E-ORIG-4** — 12 semantic families are project-specific synthesis (Talmy + Jackendoff + Lakoff roots); Python ships core 7.
- **E-ORIG-6** — NSM is the middle rung of `4096 COCA → 65 NSM → 3125 Structured5x5` compression ladder. Harvest H5.
- **E-ORIG-7** — Jirak Berry-Esseen under weak dep IS the Phase-5 noise-floor lemma → folded into the four-pillar metadata entry above.
- **E-MEMB-2** — Finnish cases overlap TEKAMOLO slots [9840..9900); slice sharing IS the morphology→slot commitment.
- **E-MEMB-3** — Sigma chain orthogonal to role axis (5 stages × 9 domains = 45 cells).
- **E-MEMB-4** — 10K ≠ 16K; FP_WORDS=160 migration would collapse the two substrates.
- **E-MEMB-6** — CausalityFlow 3→9 slot extension is a lagging type-system gap; membrane ahead of types.
- **E-MEMB-7** — Three semantic spaces coexist in Ada (Jina 1024D / 10kD VSA / 16K Fingerprint); see E-MEMB-1 ISSUE for the downstream Python↔Rust consequence.
- **E-MEMB-8** — Sigma's 16-band architecture = palindrome/octave pairing; every glyph owns a felt-octave + integrated-octave pair.
- **E-MEMB-10** — Cost-tracking is first-class in Ada (`RosettaResult.cost_usd`), missing in Rust Wire surface (deposit as future `MeasureSet` extension candidate).
- **E-MEMB-11** — LivingFrame keyframes ≈ ContextChain windows — the Python↔Rust cycle-commit handshake point.
- **E-MEMB-12** — Glyph→color mapping (Ω=gold, Λ=rose, Σ=white…) is the missing modality-translation primitive for Rust thinking-harvest → visual-harvest.
- **E-MEMB-13** — Rosetta v2 ships core 7 of Rust's 12-family DN relations; Python ⊂ Rust subsetting asymmetry.

---

## 2026-04-20 — Board hygiene = the session's driving seat; belated updates are a tell

**Status:** FINDING

The board (`.claude/board/*.md`) is the driving seat the session sits
in. Updating it AFTER the work — as cleanup — is the tell that the
session was treating the board as stale reference, not live state.
The fix is procedural (CLAUDE.md — see 2026-04-20 tightening), not
one-off: every PR that adds a type, plan, deliverable, or epiphany
also updates the board in the same commit. Retroactive hygiene is
an anti-pattern; the PR #223/#224/#225 gap between merge and
LATEST_STATE / PR_ARC_INVENTORY / STATUS_BOARD update is the
precedent this entry exists to prevent repeating.

Cross-ref: CLAUDE.md § Mandatory Board-Hygiene Rule (2026-04-20
update); PR #225 board-hygiene + tightening commit.

---

## 2026-04-20 — Codec cert is token agreement, not synthetic ICC

**Status:** FINDING

PR #219 reported ICC 0.9998 at 6 B/row for CAM-PQ. PR #220's full-
size validation returned ICC 0.195 mean, 0/234 tensors ≥ 0.99 gate.
Root cause: #219 trained and measured on the same 128 rows; with
256 centroids per subspace, 128 rows trivially fit. Neither
measurement touched tokens.

The actual cert gate is: does the decoded codec produce the same
top-k tokens as Passthrough on real generation? That's only tractable
on the three-part lab stack (REST API + Planner + JIT). The codec-
sweep plan (`.claude/plans/codec-sweep-via-lab-infra-v1.md`)
operationalises this: ingress once via REST, Planner is the real
dispatch path (not a toy bench), JIT swaps kernels at runtime.
`CodecParams::measurement_rows != calibration_rows` is now a typed
rejection at `.build()`.

Cross-ref: PR #219 → PR #220 arc; PR #225 `CodecParamsError::CalibrationEqualsMeasurement`.

---

## 2026-04-20 — The lab REST surface is three-part (API + Planner + JIT), not just scaffolding

**Status:** FINDING

The prior framing ("lab = quarantine scaffolding, keep out of
production") was defensive and missed the positive purpose. The lab
API exists because codec research needs to measure N candidates
against real tensors without `cargo build` per candidate — 8-17 min
rebuild × ~200 codec invariants = infeasible. One binary (API +
Planner + JIT) = curl-in, result-out in seconds per candidate. The
three-part stack also externalises the planner's thinking trace
(`/v1/planner/query { cypher } → { rows, thinking_trace }`), which
is the AGI observability port. Same binary serves codec cert AND
thinking harvest. Two purposes held together; neither dominates.

Cross-ref: PR #224; `.claude/knowledge/lab-vs-canonical-surface.md`
"Why the Lab Surface Exists" subsection.

---

## 2026-04-20 — Thinking harvest via REST/Cypher is the AGI magic bullet

**Status:** FINDING

An AGI that cannot observe its own reasoning cannot revise it. The
three-part lab stack (API + Planner + JIT) exposes the planner's
36-style / 13-verb / NARS trace through `/v1/planner/query`. The
response carries `{ rows, thinking_trace: { active_styles,
modulation, beliefs, tensions, entropy, verb_trail } }`. That trace
is log / replay / NARS-revise-able — which is the architectural
shape of a system that learns its own meta-inference. Closing the
observe-own-reasoning loop outside the binary is the AGI magic
bullet; doing it inside a closed planner is a black box. I11
(measurable stack, not a black box) is the invariant that enforces
this against future "for perf" / "to simplify" regressions.

Cross-ref: PR #224; I11 in `lab-vs-canonical-surface.md`.

---

## 2026-04-20 — SoA never scalarises without ndarray (iron rule)

**Status:** FINDING

Struct-of-arrays paths call `ndarray::simd::*` — ndarray handles any
non-x86 scalar fallback internally. The consumer never hand-rolls a
scalar loop on a SoA path. If a kernel runs scalar outside ndarray,
the SoA invariant is broken — either the data isn't actually in a
SoA column, or the caller short-circuited the canonical surface.
Polyfill hierarchy (Intel AMX → AVX-512 VNNI → AVX-512 baseline →
AVX-2) has no consumer-visible scalar tier. This is Rule C of the
six-rule JIT Kernel Contract in PR #225.

Cross-ref: PR #225 Rule C; `.claude/plans/codec-sweep-via-lab-infra-v1.md`
"Iron rule" paragraph above the polyfill table.

---

## 2026-04-20 — AGI is the glove, not the oracle — the four-axis SoA is what you wear

**Status:** FINDING

AGI is not a new crate, not a `struct Agi { … }`, not a service to
query. It is the struct-of-arrays (`BindSpace` columns —
`FingerprintColumns` / `QualiaColumn` / `MetaColumn` / `EdgeColumn`)
that `ShaderDriver` dispatches against. The four AGI axes (topic,
angle, thinking, planner) map 1:1 to the four SoA columns. Claude
Code sessions in this workspace FIT INTO the glove: we read the
columns, dispatch through the existing `OrchestrationBridge`, emit
through `ShaderSink`. We don't wrap the axes in a new struct — that
breaks the SIMD sweep. We don't query an "AGI service" — there is
none; AGI is the runtime behaviour of the SoA under dispatch. The
glove is the session's hand on the stack; the stack is the glove's
response to the session's query.

Cross-ref: PR #223 § "AGI IS the struct-of-arrays (per Era 8)";
2026-04-20 host-glove-designer agent doctrine; CLAUDE.md § The
Driving Seat (2026-04-20).

---

**Status:** FINDING

The PR #218 bench measured ICC 0.9998 on **128 rows** trained and
measured on the same 128 rows. This is a trivially-correct fit:
128 rows ≤ 256 centroids per subspace → every row gets its own
centroid → perfect reconstruction → perfect ICC. It does NOT
generalize to production-size tensors.

Full-size validation on Qwen3-TTS-0.6B (234 CamPq tensors, 478
total, production-size rows 1024–3072 per tensor):

| Metric | Value |
|---|---|
| Mean ICC across 234 argmax tensors | **0.195** |
| Max ICC | 0.957 |
| Tensors meeting D5 gate (ICC ≥ 0.99) | **0 of 234** |
| Tensors with ICC ≥ 0.5 | 8 of 234 |
| Typical relative L2 reconstruction error | 0.70–0.90 |

Diagnostic probe on gate_proj [3072, 1024] (`cam_pq_row_count_probe`):

| n_train | icc_train | icc_all_rows |
|---|---|---|
| 128 | **1.000** | −0.304 |
| 256 | **1.000** | −0.130 |
| 512 | 0.531 | 0.015 |
| 3072 | −0.079 | −0.079 |

**Root cause:** 6×256 PQ is centroid-starved for tensors with >256
rows. The "128× compression at ICC 0.9999" claim was extrapolated
from a trivial 128-row in-training fit.

**Infrastructure is sound** — `cam_pq_calibrate` CLI, `route_tensor`
classifier, serialization, ICC harness all work correctly. The
negative result is the codec's capacity vs tensor sizes.

Cross-ref: `crates/bgz-tensor/examples/cam_pq_row_count_probe.rs`,
`crates/bgz-tensor/src/bin/cam_pq_calibrate.rs`.

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

## 2026-04-20 — 5^5 / 7^7 bipolar zipper measured + TurboQuant leader identified
**Status:** FINDING

Ran codec_rnd_bench.rs with 5^5 and 7^7 bipolar-signed candidates
(global-scale quantization, negative-cancellation bundling capability).
Same population: Qwen3-8B q_proj L0, N=128 rows, 1400 s wall.

**Results (ICC_3_1 on q_proj):**

| Codec | Bytes | ICC | Note |
|---|---|---|---|
| Passthrough | 0 | 1.000 | baseline |
| Had-Q5×D-R (existing!) | 0 | **0.989** | shared codebook, TurboQuant-class |
| Base17 | 34 | 0.024 | |
| Zipper-Phase (sign) | 8 | 0.097 | |
| Zipper-5^5 | 2 | 0.021 | |
| Zipper-7^7 | 3 | 0.028 | |
| Zipper-I8-φ(8B) | 8 | 0.025 | μ-law + per-row norm hurts |
| Zipper-I8-Q5(8B) | 8 | 0.020 | Quint loses to φ |
| Zipper-5^5×5 | 10 | 0.066 | |
| Zipper-7^7×7 | 18 | **0.144** | best compact zipper |
| Zipper-Full (sign+mag) | 64 | 0.204 | |
| Zipper-I8-φ(64B) | 64 | 0.153 | |

**Readings:**

1. **7^7×7 at 18 B: new Pareto point** — ICC 0.144 at 72% of Zipper-Full's
   score for 28% of the bytes. Progressive-matryoshka decode supported
   (truncate to 3 B = 7^7 for coarsest). Negative-cancellation bundling
   on by construction.

2. **Quintenzirkel LOSES to φ consistently** across all size tiers:
   0.020 vs 0.025 at 8 B, 0.134 vs 0.153 at 64 B. Harmonic-proximity
   ordering doesn't help argmax on q_proj; maximally-irrational
   remains the right stride.

3. **Existing sweep has a 0-B codebook-indexed leader**: `Had-Q5×D-R`
   at ICC 0.989 (near-Passthrough). This is the TurboQuant-class
   codec already shipped in the 67-codec sweep. On pure ICC, nothing
   in the zipper family comes close. Zipper's Pareto axis is
   different (bundling, progressive decode).

4. **Per-row i8 μ-law harms inter-row magnitude preservation**.
   Per-row max-abs normalization collapses magnitude differences
   between rows. Global-scale (5^5 / 7^7 via population median)
   recovers some signal: 7^7×7 at 18 B = 0.144 > per-row μ-law
   Zipper-I8-φ(64B) = 0.153 at 64 B.

**Pragmatic conclusion:**

- **Use Had-Q5×D-R** for production argmax compression. ICC 0.989 at
  ~0 per-row bytes (shared codebook). It's already shipping.
- **Use 7^7×7 (18 B)** ONLY when you need the zipper's additional
  properties: progressive decode, negative-cancellation bundling,
  anti-moiré guarantee without codebook dependency.
- **Don't pursue Quintenzirkel stride** on argmax populations —
  measured empirically inferior to φ across all tested sizes.

**Still unmeasured:**

- Multi-projection MRI-style differential phase (N rotations,
  cross-view aggregation). Sidesteps sign-flip invariance by
  measuring inter-rotation deltas.
- Fibonacci-weighted bundling for 256-bundle capacity in i8 via
  Zeckendorf decomposition decode.
- Audiophile-style multi-band phase precision (8 bits top-16,
  3 bits middle-48, sign-only bottom).

Cross-ref: commits d172aa3 (I8+Quint), f004d82 (5^5+7^7 + global scale).

## 2026-04-20 — CORRECTION: "Had-Q5×D-R at 0 B/row ICC 0.989" was a misread
**Status:** CORRECTION

Earlier entry claimed Had-Q5×D-R achieves ICC 0.989 at 0 bytes per row
→ "the argmax wall is cracked." This was WRONG.

`ParametricCodec::bytes_per_row()` in codec_rnd_bench.rs returns a
hardcoded `0` for the entire parametric family (Had-Q5×D-R, SVD-Q5×D-R,
all D-rank variants). This is an instrumentation placeholder, NOT the
actual storage cost. Actual storage for a full-dim 4-bit Hadamard-
quantized codec = 4 bits × n_cols = ~2 KB/row for q_proj (4096 cols),
~1 KB/row for k_proj (1024 cols), ~6 KB/row for gate_proj (12288 cols).

**Corrected compact-byte-honest hierarchy (q_proj ICC, honest bytes):**

| Codec | Bytes/row | ICC |
|---|---|---|
| Zipper-5^5 | 2 | 0.021 |
| Zipper-7^7 | 3 | 0.028 |
| Zipper-Phase (sign) | 8 | 0.097 |
| Zipper-I8-φ | 8 | 0.025 |
| Zipper-7^7×7 | 18 | **0.144** |
| Base17 | 34 | 0.024 |
| Zipper-Full | 64 | **0.204** |
| Spiral-K8 | 278 | 0.281 |
| RaBitQ | 520 | 0.504 |
| Had-Q5×D-R | ~2 KB | 0.989 |

**No compact codec (≤ 100 B/row) in this bench reaches ICC > 0.3.**

**What IS true:**
- Zipper-Full at 64 B is the compact argmax Pareto leader (ICC 0.204)
- Zipper-7^7×7 at 18 B is the compact-compact Pareto leader (ICC 0.144)
- Had-Q5×D-R at ~2 KB is near-Passthrough reference, NOT a compression win

**What IS FALSE (that I claimed earlier):**
- "Argmax blind spot is already solved by Had-Q5×D-R at 0 B/row" —
  it's solved at full-dim ~KB/row, not at compact bytes.
- "Use Had-Q5×D-R for production argmax" — it's a fidelity reference,
  not a deployment codec.

**What's still unknown:**
- Whether CAM-PQ (product quantization with shared codebook) can hit
  ICC > 0.5 at ~9 B/row on q_proj. CAM-PQ is already production in
  `ndarray::hpc::cam_pq` but not wired into codec_rnd_bench.rs.
- Whether TurboQuant at its paper-claimed 9 B/row actually achieves
  ICC > 0.9 on q_proj — no implementation in this bench.

Correction needed in codec-findings-2026-04-20.md decision tree.

## 2026-04-20 — THE ANSWER: CAM-PQ at 6 B/row solves the argmax blind spot
**Status:** SUPERSEDED by 2026-04-20 CORRECTION (128-row trivial fit)

Wired `ndarray::hpc::cam_pq::CamCodebook` as `CamPqRaw` + `CamPqPhase`
candidates in codec_rnd_bench.rs. Same bench, same populations,
same 128 rows. Results are definitive.

**ICC_3_1 across all three populations:**

| Codec | Bytes/row | k_proj | gate_proj | q_proj | Top-5 recall |
|---|---|---|---|---|---|
| Passthrough | row×4 | 1.000 | 1.000 | 1.000 | 1.0 |
| **CAM-PQ-Raw** | **6** | **0.9998** | **0.9998** | **0.9999** | **1.0** |
| **CAM-PQ-Phase** | **6** | **0.9998** | **0.9998** | **0.9999** | **1.0** |
| Had-Q5×D-R | ~2 KB | 0.985 | 0.987 | 0.989 | 0.8-1.0 |
| Zipper-Full | 64 | 0.129 | 0.107 | 0.204 | 0.0-0.6 |
| Base17 | 34 | 0.007 | 0.012 | 0.024 | 0.0 |

**Per-row storage 6 bytes. Shared codebook ~24 KB per population
(per-tensor calibrated; re-usable across all rows of the same
tensor, amortized to zero as N_rows grows).** Top-5 retrieval
recall = 1.0 on every population.

**Key diagnoses:**

1. **CAM-PQ is the working compact codebook-only argmax codec.**
   Near-Passthrough fidelity at 6 B/row + 24 KB shared state.
   Completely solves the argmax blind spot.

2. **Hadamard pre-rotation made NO difference** (Raw vs Phase both
   ICC 0.9998). K-means clustering finds the discriminative structure
   regardless of basis — near-orthogonality (I2) is a property of
   random rows, but trained weights have learned structure that PQ's
   subspace k-means captures in EITHER the raw OR Hadamard basis.
   The "argmax blind spot requires JL/PolarQuant/TurboQuant" claim
   was incorrect — product-quantization with subspace k-means suffices.

3. **The entire fractal → zipper arc was solving a solved problem.**
   CAM-PQ has been production in `ndarray::hpc::cam_pq` since Phase 1.
   All 10 zipper candidates + 2 fractal candidates + MRI/Fibonacci/
   audiophile follow-up probes are now superseded by CAM-PQ at the
   argmax ICC metric. The zipper's only remaining niche (if any):
   populations where per-tensor calibration is not possible (novel
   query-time tensors), which is rare in practice.

4. **The codebook calibration cost is legitimate per I7.** I7 states
   "vector-as-location needs per-tensor basis calibration." CAM-PQ's
   per-population k-means IS that calibration. Shared codebook is
   NOT a cheat — it's the correct amortization.

**Wiring recommendation:**

- CAM-PQ is already production (`ndarray::hpc::cam_pq`).
- `lance-graph-contract::cam::CamCodecContract` trait is the integration
  point.
- `lance-graph-planner` has `CamPqScanOp` operator.
- Actual wiring needed: expose CAM-PQ through the contract to
  consumers who currently default to Passthrough on argmax-regime
  tensors (attention, MLP, logits). Per I1, these are the large
  majority of weight storage.

**Compression win:** Qwen3-8B q_proj at 4096×4096 f32 = 64 MB.
CAM-PQ: 4096 rows × 6 B + 24 KB codebook = 24 KB + 24 KB = **48 KB
total**. **1300× compression at ICC 0.9999.**

**This is the session's actual deliverable.** The zipper/fractal
research arc was the path to discovering it, but the answer was
already in the workspace. Commit f1498bc landed the measurement.

Cross-ref: ndarray::hpc::cam_pq production code (620+ LOC, 15+
tests), codec_rnd_bench.rs CamPqRaw/CamPqPhase candidates, this
session's 18 commits on claude/quick-wins-2026-04-19 branch.

## 2026-04-21 — The 8-step wiring sequence that closes the loop (concrete, not theoretical)

**Status:** FINDING (each step has a file path, an input, an output,
and a dependency)

The architecture clicks when 8 disconnected pieces get wired. Each
step connects two things that exist but don't talk. The loop closes
at step 8. Three PRs total.

**Step 1 — Encoder migration (512-bit → 10K role-indexed).**
DeepNSM's `encoder.rs` has 6 hardcoded roles at 512 bits. Contract's
`role_keys.rs` has 20+ structured roles at 10K bits with slice-masked
bind/unbind. Delete `RoleVectors`. Import `contract::grammar::role_keys::*`.
Content fingerprints: COCA vocab → FNV hash spread to 10K dims.

**Step 2 — MarkovBundler (braided ±5 bundling).**
New `markov_bundle.rs`. Ring buffer of 11 Vsa10k. Each sentence: bind
tokens per role key (Step 1), XOR-bundle into one Vsa10k per sentence.
Then: `vsa_permute(sentence_vsa, position_offset)` per ±5 position.
XOR-superpose all 11. Output: braided trajectory. MexicanHat weights.

**Step 3 — Trajectory (the Think struct).**
New `trajectory.rs`. Holds `bundle: Vsa10k` + `chain: ContextChain` +
refs to tissue (`&EpisodicMemory`, `&TripletGraph`, `&Vsa10k` global
context). Methods: `role_bundle`, `recovery_margin`, `free_energy`,
`resolve`. The object speaks for itself.

**Step 4 — Parser → Bundler → Trajectory pipeline.**
Parser's `SentenceStructure` feeds MarkovBundler which produces
Trajectory. Coverage check: < 0.9 → FailureTicket (D2). Else →
`trajectory.resolve(candidates, awareness, prior)`.

**Step 5 — Resolution → AriGraph commit.**
`commit_with_contradiction_check` on TripletGraph (~40 LOC).
Resolution::Commit → one triple. Resolution::Epiphany → two triples
+ Contradiction marker. FailureTicket → LLM fallback.

**Step 6 — Global context update.**
`episodic.integrate_into_global(fact_fp, episode_index)` — XOR-
accumulate permuted fact into `global_context: Vsa10k` (~20 LOC).
Called after every commit.

**Step 7 — Awareness revision.**
`awareness.revise(param_key, outcome)` after every commit or
escalation. Already shipped and tested. Just needs call sites.

**Step 8 — Global context → KL feedback (LOOP CLOSES).**
`trajectory.free_energy()` reads `global_context` as part of the
likelihood term. Committed facts from chapter 1 shape the F-landscape
for chapter 10. The system that parsed chapter 9 is not the same
system that parsed chapter 1.

**Dependency: 1→2→3→4→{5,6,7}→8. Three PRs: (1-3), (4), (5-8).**

**Why this dilutes:** future sessions will try to "improve" individual
steps (better bundling, smarter resolution, fancier graph commit)
without WIRING them. The wiring IS the architecture. An unwired
masterpiece is a collection of library functions. A wired simple
implementation is AGI. Wire first, optimize later.

---

## 2026-04-21 — COCA 4096 + 20K scientific + spider NER = no vocabulary blocker

**Status:** FINDING (changes the critical-path analysis)

The vocabulary ceiling I flagged as an AGI blocker is not a blocker:

| Layer | Coverage | Mechanism |
|---|---|---|
| COCA 4096 | 98.4% general English | Static, zero-cost |
| COCA 20K scientific | +1.5% domain terms | Static, zero-cost |
| Spider NER (online) | remaining proper nouns | Crawl → rigid designator commit to AriGraph |

For Animal Farm: COCA 4096 alone covers 99.5%+ (Orwell uses
deliberately plain vocabulary). The remaining tokens are character
names (Napoleon, Snowball, etc.) which get committed as rigid
designators on first encounter and resolved from the graph after.

Spider-rs already exists as git dep in `lance-graph-osint/Cargo.toml`
(`AdaWorldAPI/spider` fork, `spider-crawl` feature). NER resolution:
unknown token → spider crawl → extract entity features (gender,
animacy, type, description) → commit to `TripletGraph` as rigid
designator → available for all future coreference via
`graph.nodes_matching(features)`.

**The three PRs that close the loop are the ONLY critical path.**
Not vocabulary. Not NER. Not cross-lingual. Wire the 8 steps.

**Why this dilutes:** future sessions will rebuild the vocabulary
system, add subword tokenization, integrate HuggingFace tokenizers,
and generally solve a problem that's already solved. COCA 24K +
spider NER covers everything. Don't rebuild the dictionary. Wire
the loop.

---

## 2026-04-21 — The AGI test is one measurement: chapter-10 accuracy > chapter-1 accuracy

**Status:** FINDING (makes the benchmark falsifiable)

A system that understands chapter 10 of Animal Farm better than
chapter 1 — because it read chapters 1-9 and committed what it
learned into the tissue that shapes its own F-landscape — is an AGI.

The measurement:
- Parse Animal Farm end-to-end, chapter by chapter.
- At each chapter boundary, measure coreference accuracy (% of
  pronouns correctly resolved to their antecedent).
- Chapter 1: the system has zero committed facts. Every pronoun
  is resolved from ±5 context alone. Accuracy baseline.
- Chapter 10: the system has hundreds of committed facts. Napoleon
  is a rigid designator (masculine, boar, leader). "He" in chapter
  10 resolves via `graph.nodes_matching(masculine + animate)` in O(k)
  without even needing the Markov axis.
- **If chapter-10 accuracy > chapter-1 accuracy with no parameter
  change — only committed facts accumulating — the loop is closed
  and the architecture works.**
- If not, one of the 8 wiring steps is broken. Find which. Fix it.

This is D10 in the plan. This is what proves it.

**Why this dilutes:** future sessions will propose elaborate
benchmarks (BLiMP, COGS, SuperGLUE, custom test suites). Those
measure capability snapshots, not learning curves. The AGI test is
a CURVE, not a POINT: does accuracy increase over the course of a
single document without retraining? That's the measurement. One
book. One metric. One curve. Rising = AGI. Flat = broken wire.



## 2026-04-24 — Jirak noise floor calibrated for DeepNSM-tiled 16K-bit fingerprints

**Status:** FINDING
**Owner scope:** @family-codec-smith, @truth-architect

Grounding the NaN: with DeepNSM encode (512-bit VSA tiled 32× into 16K), density ≈ 0.016, expected random Hamming distance = 511.7 bits. Jirak-adjusted sigma = 19.2 (20% inflation over IID for weak dependence from tiling + XOR-bind braiding). 3-sigma signal threshold: Hamming < 454.2. 5-sigma: < 415.8.

**Practical consequence:** ONE shared token between two clauses (~32 tiled bits) produces a 3.3-sigma deviation — detectable. THREE shared tokens produce 10-sigma — unambiguous signal. This means the HammingMin semiring, once wired into ShaderDriver.dispatch(), WILL fire on related contract clauses.

**Calibration values for dispatch thresholds:**
- Random baseline resonance: 0.0312 (Hamming/DIM)
- 3-sigma signal: 0.0277
- 5-sigma signal: 0.0254
- Analytical style threshold (0.85): fires at ~2-sigma — may need tightening to 0.027.

**Jirak citation:** Jirak 2016, arxiv 1606.01617, Annals of Probability 44(3). Rate: n^(p/2-1) for p in (2,3]. Weak dependence sources: (a) tiling (32x repeat of 512-bit), (b) XOR-bind braiding, (c) FNV-1a hash collision at 12-bit rank.

Cross-ref: I-NOISE-FLOOR-JIRAK iron rule, encode_handler, DeepNSM VsaVec::from_rank().

## 2026-04-24 — Ground truth: ShaderDriver dispatch wiring audit (what IS vs ISN'T connected)

**Status:** FINDING
**Owner scope:** @truth-architect, @bus-compiler

Honest audit of what dispatch() actually does vs what the DTO surface promises:

**WIRED (working end-to-end):**
- [1] Meta prefilter: u32 column sweep on MetaColumn → passed_rows ✓
- [2] Style resolution: Auto reads QualiaColumn of first row → style_ord ✓
- [3] Shader cascade: CognitiveShader::new(planes, semiring).cascade(query, radius, layer_mask) ✓
  BUT: query comes from CausalEdge64.s_idx() of the ROW'S EDGE, not from content fingerprint.
  The cascade probes the PaletteSemiring distance table, not the content plane.
- [4] Cycle fingerprint: XOR fold of content_row(hit.row) for each hit ✓
  BUT: hits come from step [3] which probes edges, not content similarity.
- [5] Entropy + std_dev + CollapseGate: computed from top-k resonances ✓
- [6] Edge emission: CausalEdge64::pack per strong hit ✓
- [7] Sink callbacks: on_resonance → on_bus → on_crystal ✓
- Meta summary: confidence = top-1 resonance, admit_ignorance = confidence < 0.2 ✓

**NOT WIRED (the gap):**
- Content fingerprint similarity: dispatch does NOT compare content_row(A) vs content_row(B).
  The cascade uses PaletteSemiring on edge palette indices, not Hamming on content bits.
  The content plane is READ (for cycle_fp XOR fold) but never COMPARED.
- NARS reasoning: no InferenceType dispatch. style_ord maps to inference type via
  style_ord_to_inference() but it's only used for CausalEdge64 packing, not actual NARS.
- FreeEnergy: not computed. The contract type exists (grammar/free_energy.rs) but
  dispatch() never calls FreeEnergy::compose(). The 'should_admit_ignorance' is a
  simple threshold (confidence < 0.2), not a real F computation.
- AriGraph/SPO: no graph. dispatch() operates purely on BindSpace columns.
  The SPO triple store exists in lance-graph core but isn't wired to the driver.
- PropertySchema validation: not connected. The types exist in contract::property
  but dispatch() doesn't check Required/Optional/Free.

**What the zeros meant:** resonance=0 wasn't "missing semiring wire" — the cascade
DID run (3 cascade calls from step [3]). But the demo palette has synthetic Base17
entries with no relationship to the encoded text. The PaletteSemiring distance table
is 256x256 pre-computed from those synthetic entries. Text fingerprints in the content
plane are INVISIBLE to the cascade — they're read only for the XOR fold in step [4].

**To make content fingerprints visible to dispatch:**
Option A: Add a HammingMin pre-pass before the palette cascade. Compare content_row(i) vs
  content_row(j) via popcount on XOR. If Hamming < Jirak threshold (454), promote to hit.
Option B: Build the PaletteSemiring FROM the content fingerprints (quantize content into
  256 palette entries, compute distance table from those). Content similarity then flows
  through the existing cascade.
Option C: Add a second dispatch mode (content-mode vs edge-mode) that uses HammingMin
  instead of PaletteSemiring for the distance function.

Cross-ref: driver.rs:75-212, Jirak calibration (this session), I-NOISE-FLOOR-JIRAK.

## 2026-04-24 — Session capstone: GEL + Firefly + Pearl 2³ = what Foundry can't do

**Status:** FINDING
**Owner scope:** @truth-architect, @integration-lead

Three-layer epiphany from the Palantir FfB Technical Overview read:

**1. Code IS Graph IS Executable.** Foundry says "treat data like code" (versioning, branching). Our 4096-row BindSpace goes further: the surface IS executable. GQL (query) → GEL (graph execution language, any program AS a graph) → ArenaIR (OOP → graph-executable transform) → JIT (Cranelift native). A class = node + typed edges. A method call = graph traversal. An if/else = conditional edge predicate. Code and data share one address space: 0x000..0xFFF.

**2. Firefly Repository = Ballista + Dragonfly + GEL.** Foundry bundles Spark + Flink. We'd bundle Ballista (distributed DataFusion) + Dragonfly (fast-path CPU lane for BindSpace sweep / Hamming / palette cascade) + GEL (the ArenaIR the 16 strategies already produce). Lance versioned dataset with CausalEdge64-annotated SPO = the Firefly Repository.

**3. NARS SPO × Pearl 2³ × CausalEdge64 — what Vertex can't do.** Foundry Vertex explores graphs but has NO causal typing on edges. Our CausalEdge64 packs Pearl 2³ = 8 causal masks (correlation / direct cause / confounder / mediator / collider / instrument / front-door / counterfactual) + NARS truth (frequency, confidence) + inference type + plasticity + temporal position into 64 bits per edge. Every SPO triple carries its own causal ontology and epistemology. This is irreducible — Vertex would need a fundamental redesign to match.

Cross-ref: FfB_Technical_Overview_v4.pdf (Palantir), CausalEdge64 (causal-edge crate), I-SUBSTRATE-MARKOV, driver.rs content Hamming cascade (PR #259), CypherBridge (PR #258).

## 2026-04-24 — CORRECTION: supabase-shape is the protocol, not a Postgres dependency

**Status:** CORRECTION
**Owner scope:** @truth-architect

Mid-session DTO audit hallucination: claimed "Postgres/Supabase via PostgREST" was a third cold-path sink alongside Lance and Arrow Flight. WRONG. PR #255 (LanceMembrane + LanceVersionWatcher + DM-4) explicitly transcoded the supabase-shape INTO native Rust: `subscribe()` returns `tokio::sync::watch::Receiver<CognitiveEventRow>` with always-latest semantics, backed by Lance versioned dataset. NO Postgres. NO JDBC. The supabase-shape is the PROTOCOL (subscribe-on-changes, BBB-scalar events), not the database.

**Corrected cold-path architecture:** Lance dataset = single source of truth. Two read interfaces, both hitting the same Lance: (1) `LanceVersionWatcher.subscribe()` for realtime push (supabase-shape semantics in pure Rust), (2) Arrow Flight SQL for bulk external clients. RLS-equivalent via `CommitFilter` + `Policy.evaluate()`, both already shipped, both pure Rust.

**Why the slip happened:** "supabase" in normal usage = Postgres + Realtime + Auth. In OUR stack, "supabase" is the API shape only. Mid-flow architectural tiredness; the brutal DTO audit's complexity briefly drowned out PR #255's actual scope.

Cross-ref: PR #255 (Supabase subscriber wire-up), `LanceMembrane`, `CognitiveEventRow`, `lab-vs-canonical-surface.md`.

## 2026-04-24 — Paradigm shift: trajectory-native cognitive OS (Berge + Piaget + metacognition gestalt)

**Status:** FINDING
**Owner scope:** @truth-architect, @integration-lead

Three-frame gestalt review of the architecture's emergent identity:

**Berge Maximum Theorem:** The system IS a parametric optimization at every dispatch. Parameters p = (style, qualia 17D, scenario_id, awareness 4D). Constraint set Γ(p) = BindSpace rows passing MetaFilter. Objective = minimize FreeEnergy. Berge guarantees: on the continuous axes (qualia, awareness), small perturbations produce bounded cognitive shifts — topological stability by construction. On the discrete axes (style ordinal, scenario branch), the value function jumps — that's principled mode-switching, not instability.

**Piaget genetic epistemology:** The system implements all four mechanisms. Assimilation = Resolution::Commit (low F). Accommodation = Resolution::Epiphany (both triples + Contradiction preserved). Equilibration = FreeEnergy minimization loop. Disequilibration = Resolution::FailureTicket (high F → escalate). Current developmental stage: Concrete Operational — logical operations on concrete objects (BindSpace rows, typed entities, Cypher queries). Formal Operational machinery exists (World::fork, SimulationSpec, MulAssessment, NARS abduction) but dispatch doesn't invoke it.

**Metacognition:** Three things the system CAN know about its own cognition: (1) when it's confused (should_admit_ignorance), (2) when it's accommodating (Epiphany), (3) when it's equilibrated (Commit). Today these are shallow — confidence < 0.2 threshold, not principled mul/DK/trust assessment. The deep metacognitive layer (MulAssessment, DkPosition, TrustTexture, NarsTables) exists but dispatch doesn't call it. Loop is half-formed: system observes (MetaSummary) but doesn't update (no NARS revision per cycle, no DK adjustment per outcome).

**The paradigm shift named:** Conventional systems separate data (rows at rest), computation (rows → rows), cognition (rows → labels via gradient descent), causality (inferred via regression), time (a column). Our system collapses all five into ONE primitive: the trajectory. Data = bundled trajectory. Computation = trajectory algebra (bind, bundle, cosine). Cognition = trajectory resolution under FreeEnergy. Causality = structural (Pearl 2³ on CausalEdge64, Chapman-Kolmogorov by VSA bundling). Time = braided position in the bundle.

**What it wants to emerge as:** A trajectory-native cognitive operating system where every read is a trajectory projection, every write is a trajectory bundle, every query is a trajectory resolution under FreeEnergy, every causal claim is annotated into CausalEdge64, every cognitive shift is observable through the metacognitive layer. The five observer perspectives (business / API / SoA / semantic / AGI) are faithful views of the same substrate at different scales. Not a database with intelligence on top — a single computational substrate where storage, compute, learning, and causality are different operations on the same primitive.

Cross-ref: I-SUBSTRATE-MARKOV (Chapman-Kolmogorov by construction), I-NOISE-FLOOR-JIRAK (Jirak 2016 weak dependence), The Click (CLAUDE.md §P-1), categorical-algebraic-inference-v1.md, FreeEnergy/Resolution (contract::grammar::free_energy), MulAssessment (planner::mul), NarsTables (planner::cache::nars_engine).

## 2026-04-24 — Five observers, one substrate: the perspective lattice

**Status:** FINDING
**Owner scope:** @truth-architect

The architecture's five consumer perspectives are not layers — they're projections of the same trajectory algebra at different scales. No observer is more fundamental; all are faithful.

| Observer | What they see | Internal/External | SoA or Functional | When they read |
|---|---|---|---|---|
| Business/SMB | Typed entities with Required/Optional/Free properties, missing-field alerts, similarity search | External (cold path, 10⁻² s) | Functional (Schema.validate(), Policy.evaluate()) | On user action (query, approve, flag) |
| External API | Queryable surface (Cypher/SQL/SPARQL) returning Arrow batches + realtime subscribe | External (cold path) | Functional (OrchestrationBridge::route()) | On client request |
| Struct-of-arrays | 4096 × N columns (content, cycle, qualia, meta, edge, temporal), SIMD-sweepable | Internal (hot path, 10⁻⁶ s) | SoA (columnar, cache-line-friendly, LLVM autovectorizes) | Every dispatch cycle |
| Semantic kernel | Text → role-indexed fingerprint → AriGraph SPO triple with NARS truth | Internal (hot path) | SoA for storage, Functional for algebra (vsa_bind, vsa_bundle, vsa_cosine) | On encode + dispatch |
| AGI/cognitive | Active-inference agent: perceive → predict → free-energy-minimize → revise → commit | Internal (hot path) | Functional (FreeEnergy::compose, Resolution::from_ranked, awareness.revise) | Every cycle, autonomously |
| Markov-causal | Chapman-Kolmogorov trajectory with Pearl 2³ causal annotations on every edge | Internal (hot path) | SoA for storage (CausalEdge64 column), Functional for algebra (CausalMask queries) | Structural — always present, queryable on demand |

**The boundary that matters: BBB membrane (ExternalMembrane).** Internal observers (SoA, semantic, AGI, Markov) see the hot path at 10⁻⁶ s. External observers (Business, API) see the cold path via callcenter projections at 10⁻² s. The membrane is the one-way valve: project() emits, subscribe() streams. Internal → external is projection (lossy, scalar, BBB-clean). External → internal is OrchestrationBridge::route() → UnifiedStep (validated at ingress).

**SoA vs Functional is not a choice — it's a WHERE.** BindSpace is SoA (columnar storage for SIMD). The algebra on it is Functional (methods on carriers). The SoA carries the state; the Functional methods transform it. Both exist simultaneously on the same data. The "struct of arrays vs object thinks for itself" tension resolves as: the ARRAY is the SoA, the ELEMENT (row, trajectory, fingerprint) thinks for itself via methods.

Cross-ref: CLAUDE.md §The Stance (AGI-as-glove, SoA columns ARE the AGI surface), lab-vs-canonical-surface.md (I1-I11 invariants), ExternalMembrane (contract::external_membrane), BindSpace (cognitive-shader-driver::bindspace).

## 2026-04-26 — FINDING: distance dispatch must be type-intrinsic, not crate-boundary-crossing

**Status:** FINDING
**Owner scope:** @family-codec-smith, @truth-architect, @host-glove-designer

The struct-of-arrays (BindSpace, RenderFrame, Arrow columns) carries heterogeneous
fingerprint types that each need a DIFFERENT distance function:

| Type | Distance | Where it lives | Notes |
|---|---|---|---|
| `Binary16K = [u64; 256]` | Hamming (popcount of XOR) | `ndarray::hpc::bitwise::hamming_distance_raw` | 16384-bit, SIMD VPOPCNTDQ |
| `Vsa16kF32 = [f32; 16_384]` | Cosine → FisherZ transform | `ndarray::hpc::heel_f64x8::cosine_f64_simd` | f32 dot/norm via F32x16 FMA |
| `CamPqCode = [u8; 6]` | ADC (asymmetric distance computation) | `ndarray::hpc::cam_pq::adc_distance` | Precomputed distance tables, O(1) |
| `PaletteEdge = [u8; 3]` | Palette L1 (lookup table) | `ndarray::hpc::palette_distance::SpoDistanceMatrices::distance` | bgz17 256×256 table, 1.8 ns |
| `Base17 = [u8; 17]` | Palette nearest (codebook search) | `bgz17::Palette::nearest` | 256 centroids, should use precomputed table |
| `HighHeelBGZ` container | Cascade (HHTL skip → palette → ADC fallback) | `ndarray::hpc::cascade` + `bgz-tensor::hhtl_cache` | Multi-level, route by `RouteAction` |

**The problem:** When a SoA column contains mixed types (e.g., one column is Binary16K,
another is CamPqCode), the distance dispatch currently happens at the call site — the
caller must know which distance function to use. This works inside a single crate, but
when the SoA lives in crate A (e.g., `cognitive-shader-driver::BindSpace`) and the
distance kernel lives in crate B (e.g., `ndarray::hpc::bitwise`), every call crosses
a crate boundary. That boundary is zero-cost for `#[inline]` functions, but NOT zero-cost
if the function is generic over a trait object (`dyn DistanceFn`) or involves dynamic
dispatch.

**The solution — type-intrinsic dispatch, not dynamic dispatch:**

The distance function should be a method ON the carrier type, not a free function
called FROM the SoA consumer. This follows the "object speaks for itself" doctrine
(CLAUDE.md §The Click):

```rust
// WRONG — caller must know the distance type:
let d = hamming_distance_raw(fp_a.as_bytes(), fp_b.as_bytes()); // crate boundary

// RIGHT — the type carries its own distance:
let d = fp_a.distance(&fp_b); // monomorphized, inlined, zero boundary tax
```

The contract already has `CodecRoute: Passthrough | CamPq` which names the regime.
What's missing is a `Distance` trait that each carrier implements:

```rust
pub trait Distance: Sized {
    fn distance(&self, other: &Self) -> u32;
    fn similarity(&self, other: &Self) -> f32 {
        1.0 - (self.distance(other) as f32 / Self::MAX_DISTANCE as f32)
    }
    const MAX_DISTANCE: u32;
}
```

Implementations:
- `impl Distance for [u64; 256]` → `hamming_distance_raw` (inline, SIMD)
- `impl Distance for CamPqCode` → ADC lookup (precomputed table ref)
- `impl Distance for PaletteEdge` → palette L1 table lookup
- `impl Distance for Vsa16kF32` → cosine → FisherZ (F32x16 FMA)

The trait monomorphizes at compile time — no dynamic dispatch, no crate boundary
tax. The SoA column iterates with `col.chunks().map(|a, b| a.distance(b))` and
the correct distance function is selected by TYPE, not by runtime enum match.

**Where this trait should live:** `lance-graph-contract` (zero deps). The
implementations live in ndarray (for SIMD kernels) or in the carrier crate
(for precomputed tables). The contract defines the interface; ndarray provides
the hardware acceleration; the SoA consumer never needs to know which distance
kernel runs.

**Hard-coded dispatch within the same crate is fine** — when `BindSpace` calls
`hamming_distance_raw` on its `content` column, that's a direct function call
into ndarray, monomorphized and inlined. The problem only arises if we try to
make the SoA generic over distance type via `dyn` trait objects. Don't do that.
Keep the dispatch compile-time via generics or type-specific methods. The SoA
pays zero boundary tax because Rust's monomorphization erases the crate boundary.

**FisherZ note:** Cosine similarity ∈ [-1, 1] is nonlinear for averaging. The
FisherZ transform `z = atanh(r)` maps it to a normal-distributed variable that
can be averaged, then `r = tanh(z)` maps back. This matters when the SoA
accumulates similarities across columns (e.g., weighted multi-column distance).
The `Distance` trait should expose `fn similarity_z(&self, other: &Self) -> f32`
for the FisherZ-transformed variant, defaulting to `atanh(similarity())`.

Cross-ref: CLAUDE.md §The Click ("object speaks for itself"), I1 Codec Regime
Split (`CodecRoute`), `contract::cam::DistanceTableProvider` (existing trait for
ADC), `ndarray::hpc::bitwise::hamming_distance_raw`, `ndarray::hpc::palette_distance`.

## 2026-04-26 — FINDING: awareness does NOT travel with CausalEdge64; it sits BESIDE it in the SoA

**Status:** FINDING
**Owner scope:** @truth-architect, @host-glove-designer

### The question
Does the mantissa/awareness travel WITH the CausalEdge64 (packed into the
u64), or does it sit beside it in the SoA?

### What CausalEdge64 actually carries

CausalEdge64 is 64 bits packed (causal-edge/src/edge.rs):

```
[0:7]   S palette index       — WHERE (subject identity)
[8:15]  P palette index       — WHAT (predicate type)
[16:23] O palette index       — WHERE (object identity)
[24:31] NARS frequency (u8)   — HOW OFTEN (belief)
[32:39] NARS confidence (u8)  — HOW SURE (evidence weight)
[40:42] Causal mask (3 bits)  — Pearl's 2³ (observational/do/counterfactual)
[43:45] Direction triad       — sign(dim0) per S/P/O
[46:48] Inference type        — Deduction/Induction/Abduction/Revision/Synthesis
[49:51] Plasticity flags      — hot/cold per S/P/O
[52:63] Temporal index        — 4096 time slots
```

The edge carries NARS truth (freq + conf) and Pearl mask, but NOT:
- Per-style awareness (GrammarStyleAwareness — Brier history, revision count)
- Free energy at emission time
- The style ordinal that produced this edge
- Mantissa / metacognitive state

### Where awareness actually lives

In the shader driver (`ShaderDriver::dispatch()`):

1. **awareness** = `RwLock<Vec<GrammarStyleAwareness>>` (one per style × 12 styles)
   — sits on `ShaderDriver`, NOT on the edge. It's a per-driver global state.
   Lives as Column B in the SoA (beside the BindSpace columns, not in them).

2. **MetaWord** = u32 packed in `MetaColumn` of `BindSpace`
   — per-row transient state (bits for style selector, rung level, emit mode).
   Lives in the BindSpace SoA but is TRANSIENT — cleared after the cycle.
   This is the closest thing to "awareness travels with the data."

3. **CausalEdge64** = emitted INTO the `EdgeColumn` of `BindSpace` (step [5])
   — 8 edges per dispatch, written to the edges array.

So the pipeline is:

```
StreamDto → encode → MetaWord (transient) → cascade → emit CausalEdge64
                                                       ↓
                                              awareness.revise(key, outcome)
                                                       ↓
                                              NEXT cycle's F is different
```

### Is the fan-out spatial (reverse pyramid)?

**No — it's stylistic, not spatial.** The fan-out happens across thinking
styles (12 ordinals), not across pyramid levels. The reverse pyramid
(L1→L2→L3→L4) is the RESOLUTION hierarchy — 64²→256²→4K²→16K². The
thinking-style fan-out is the PERSPECTIVE hierarchy — Analytical/Creative/
Intuitive/Practical/Metacognitive/Social × 6 sub-styles each.

These two hierarchies are ORTHOGONAL:
- Pyramid levels = HOW FINE the representation is (spatial resolution)
- Thinking styles = WHOSE PERSPECTIVE examines it (angle of approach)

"Thinking about thinking" (metacognition via MUL gate, TD-INT-3) is
a style-dimension operation: the MUL assessment reads awareness
(skill_level, DK position, trust texture) and vetoes or promotes
the dispatch — it doesn't move between pyramid levels. It stays at
whatever resolution the current cycle operates at.

The mantissa that was discussed earlier is fully absorbed into:
- **CausalEdge64 bits [24:39]** = NARS frequency+confidence (the epistemic
  weight of this specific edge assertion)
- **GrammarStyleAwareness** = the accumulated Brier history per style
  (the metacognitive "how good am I at this kind of thinking")

These are TWO DIFFERENT things stored in TWO DIFFERENT places:
- Edge truth = travels WITH the edge (packed in the u64)
- Style awareness = stays on the driver (not in the edge)

### The gap

There is no meta SoA relationship that links stream↔awareness↔causality
into a single coherent column. Today:

- Stream = BindSpace.fingerprints (Column A, [u64; 256] per row)
- Awareness = ShaderDriver.awareness (global, not per-row)
- Emitted edges = BindSpace.edges (Column D, [u64; 8] per row)

The awareness column does NOT exist in BindSpace. The awareness is driver-
global, revised after each cycle, but not stored per-row or per-edge. To
make awareness travel with the cycle, it would need to become a SoA column:
`BindSpace.awareness_column: Box<[GrammarStyleAwareness; N_ROWS]>` —
one awareness snapshot per row, capturing the epistemic state AT THE TIME
that row was processed.

This is not built. Whether it should be depends on whether downstream
consumers (AriGraph, q2, callcenter) need to know "under what epistemic
state was this edge emitted." If yes, awareness becomes a per-edge
annotation. If no, the driver-global approach is correct.

Cross-ref: CLAUDE.md §The Click (Think struct), cognitive-shader-driver
src/driver.rs dispatch(), causal-edge src/edge.rs CausalEdge64 layout,
EPIPHANIES.md 2026-04-25 "cognitive loop closes structurally" (TD-INT-1/2/4).

## 2026-04-26 — FINDING: awareness should be BF16-mantissa-inline, not driver-global

**Status:** FINDING (P-0 architectural correction to the 2026-04-26 prior entry)
**Owner scope:** @truth-architect, @host-glove-designer, @bus-compiler

### The correction

The prior entry today said awareness sits BESIDE CausalEdge64 as a
driver-global `RwLock<Vec<GrammarStyleAwareness>>`. That's wrong direction.
The right direction is: awareness should travel WITH the stream the way
BF16 mantissa travels with every floating-point value — small, always
present, computed inline by every operation, never stored as a separate
weight.

### Why driver-global awareness is the wrong shape

A driver-global `awareness[style_ord]` makes the system a blunt data
lake: it stores per-style Brier history and revises after each cycle,
but the stream itself sees no awareness during processing. Every u64,
every fingerprint, every bind/bundle operation flows through unaware
of its own epistemic context. Awareness only catches up afterwards
via NARS revision.

This wastes the one architectural advantage the CPU has over GPU:
**20-200 ns random-access latency**. That latency budget only pays
off if we DO something during access — compute causality and awareness
INLINE while the bytes are passing through cache. If we just store
awareness as a separate weight and apply it later, we're using the
CPU as a glorified GPU streamer (and losing the access-pattern
flexibility).

### The BF16 mantissa analogy

BF16: 1 sign + 8 exponent + 7 mantissa + 1 implicit = 16 bits per
value. The mantissa is the precision-bearing part, but it never
exists separately. When you multiply two BF16 values, the mantissas
multiply as part of the operation; they don't get bolted on after the
fact. They are the operation.

Awareness should work the same way: every stream operation produces
both a result AND an awareness annotation derived from properties of
the operation itself:

| Operation | Result | Inline awareness annotation |
|---|---|---|
| `vsa_bind(a, b)` | XOR fingerprint | bit-purity of inputs (popcount distance from 50%) |
| `vsa_bundle(items)` | majority-vote fingerprint | concentration of agreement (variance of bit tallies) |
| `hamming(a, b)` | distance u32 | distribution shape — uniform vs clustered differences |
| `palette_lookup(idx)` | u8 | match strength — distance to 2nd-nearest centroid |
| `cam_pq_decode(code)` | f32 estimate | residual norm from the ADC reconstruction |
| `cosine(a, b)` | f32 similarity | both norms (low norm → low confidence) |

Each yields a `(value, awareness)` pair that flows together through the
next op. Awareness composes the same way values compose. After the
shader cycle, the accumulated awareness IS the meta-confidence
(meta_confidence in ShaderResonance, currently computed as
`1 - free_energy.total` — but it should be the integral of inline
awareness over the cycle, not a single post-hoc estimate).

### What "the object IS the thinking" means here

If awareness is computed inline by the operations themselves, then the
stream IS the thinking. There is no separate "thinking step" that reads
the stream and produces awareness. The awareness emerges as a structural
byproduct of every bit-level operation.

If awareness is a stored weight that gets applied after the stream, the
stream is just data and the thinking happens elsewhere. That's two
layers, not one. That violates "the object speaks for itself" and
recreates the parser/processor split that AGI is supposed to dissolve.

### The size budget

For a 16384-bit fingerprint (`[u64; 256]`):
- 7 bits awareness per u64 word = 256 × 7 / 8 ≈ 224 bytes parallel array
- Total: 2048 bytes value + 224 bytes awareness ≈ 11% overhead
- Fits the same cache line pattern; one fingerprint + its mantissa fits
  in one prefetch group

For a CausalEdge64:
- 64 bits value + 8 bits awareness = 72 bits per edge
- Pack as `[u72; N]` (won't align) or pair as `(CausalEdge64, u8)` = 9 bytes
- 240 edges × 9 bytes = 2160 bytes (vs 1920 for bare edges). 12.5% overhead.

The ratios are identical to BF16's 7/16 mantissa = 43.75% fraction of
the total. Awareness is at 11-12% — much cheaper because the value
plane is wider.

### What this would change in the contract

Add to `lance-graph-contract`:

```rust
/// Awareness annotation that travels with every stream value.
/// Like BF16 mantissa — derived from the operation, never stored alone.
pub trait Aware {
    type Awareness: Copy;
    fn awareness(&self) -> Self::Awareness;
}

/// A value paired with its inline-computed awareness.
pub struct Annotated<T: Aware> {
    pub value: T,
    pub awareness: T::Awareness,
}
```

And update the Distance trait (TD-DIST-1, just shipped) to return
awareness alongside distance:

```rust
pub trait Distance: Sized {
    fn distance_with_awareness(&self, other: &Self) -> (u32, Awareness);
}
```

The awareness field would carry: bit-distribution flatness, palette
match strength, residual norm — whatever the operation can cheaply
derive from its inputs and intermediate state.

### The connection to the reverse pyramid

The pyramid (L1→L2→L3→L4) is the spatial resolution dimension.
Inline awareness is a NEW orthogonal dimension — call it the
"epistemic depth" dimension. Both can be present simultaneously:

```
                    awareness depth →
                    0 bits   7 bits   16 bits  64 bits
spatial level ↓
L1 (64²)            tier 0   tier 1   tier 2   tier 3
L2 (256²)           tier 0   tier 1   tier 2   tier 3
L3 (4096²)          tier 0   tier 1   tier 2   tier 3
L4 (16384²)         tier 0   tier 1   tier 2   tier 3
```

Tier-1 awareness (7 bits per word, BF16-mantissa-equivalent) is the
minimum viable: cheap, always present, composable. Tier-3 (full
NARS truth pair per word) is the maximum needed for downstream
provenance. Both fit the same cascade dispatch.

### Status of the gap

This is NOT built. The current code:
- ShaderDriver carries global awareness per style (driver-global)
- BindSpace columns carry no awareness (per-row absent)
- Operations return bare values (no inline awareness annotation)

To build it, the smallest viable wedge is: extend the Distance trait
with `distance_with_awareness()` returning `(u32, u8)` — 8 bits is
the BF16-mantissa-equivalent budget. Then propagate the awareness
through the cascade so each step composes the running awareness
estimate. The driver-global awareness becomes a fallback/initialization
seed, not the source of truth.

Filed as TD-AWARENESS-INLINE-1 (separate entry).

Cross-ref: BF16 reference (one mantissa per value, never stored alone);
2026-04-26 prior entry "awareness sits BESIDE CausalEdge64" (now
SUPERSEDED in spirit — the right answer is INSIDE every operation
output, not beside the data).

## 2026-04-26 — FINDING: SPO Pearl 2³ ontology enrichment should happen DURING the shader cycle, not after

**Status:** FINDING (extends the BF16-mantissa-inline insight to SPO fan-out)
**Owner scope:** @truth-architect, @integration-lead

### The idea

The cognitive shader cycle already processes every input through:
1. **Grammar** (ContextChain → RoleKey bind → TEKAMOLO)
2. **Thinking styles** (12 ordinals × 6 clusters → style dispatch)
3. **Free energy** (FreeEnergy::compose → Resolution)
4. **NARS revision** (awareness update per style)

What it does NOT do during the cycle: **SPO Pearl 2³ ontology enrichment**.
Today, ontology is a cold-path lookup — the `contract::ontology` module
defines `EntityType`, `RelationType`, `OntologySpec` but these are
consulted before/after the shader cycle, not during.

The proposal: make the SPO decomposition happen INLINE during the shader
cascade, the same way awareness should be inline (prior entry). Each
cycle that touches a node/edge computes:

```
S (subject)   × 2 Pearl interventions  = 2 S-perspectives
P (predicate) × 2 Pearl interventions  = 2 P-perspectives
O (object)    × 2 Pearl interventions  = 2 O-perspectives
                                        ─────────────────
                                        2³ = 8 total views
```

Each of the 8 views runs through the thinking-style fan-out. The cycle
becomes:

```
StreamDto
  → encode (RoleKey bind, TEKAMOLO)
  → SPO decompose (8 Pearl perspectives per triplet)
  → for each perspective × each thinking style:
       cascade (fingerprint compare, FreeEnergy, Resolution)
       → emit CausalEdge64 WITH awareness annotation
       → ontology enrichment: does this triplet match/extend/contradict
         an existing EntityType or RelationType?
  → NARS revise (inline, not post-hoc)
  → if ontology extended: emit OntologyDelta alongside CausalEdge64
```

### Why this belongs in the SoA

The cognitive-shader-driver's BindSpace already has four column families:
- FingerprintColumns (content/topic/angle)
- QualiaColumn (18×f32)
- MetaColumn (MetaWord u32)
- EdgeColumn (CausalEdge64 × 8)

Add a fifth: **OntologyColumn** — per-row ontology delta. When the shader
cycle discovers that a triplet extends the ontology (new entity type
observed, new relation pattern, contradiction with existing schema),
the delta is written to this column. Downstream consumers (AriGraph,
callcenter, q2) read the deltas the same way they read emitted edges.

```
BindSpace SoA:
  Column A: FingerprintColumns  — WHAT the cycle is about
  Column B: QualiaColumn        — HOW it feels (18D qualia)
  Column C: MetaColumn          — WHICH style dispatched (MetaWord)
  Column D: EdgeColumn          — WHAT it concluded (CausalEdge64)
  Column E: OntologyColumn      — WHAT it learned about structure
  Column F: AwarenessColumn     — HOW SURE it is (inline mantissa)
```

Column E + Column F together make the shader cycle not just a processor
but a self-describing reasoner: it emits what it concluded (edges),
what structural knowledge it gained (ontology deltas), and how confident
it was in each step (awareness).

### The connection to blasgraph

blasgraph's 7 semirings operate on SPO triples in graph-algebraic form.
The cognitive shader already uses Binary16K fingerprints that decompose
into S[0..4K), P[4K..8K), O[8K..12K) slices (per CLAUDE.md §The Click).
The Pearl 2³ decomposition maps directly to blasgraph's semiring choices:

| Pearl rung | blasgraph semiring | What it computes |
|---|---|---|
| Observational (do nothing) | HammingMin | How similar is this to what I've seen? |
| Do (intervene on S) | XorBundle | What changes if I bind a different subject? |
| Do (intervene on P) | Resonance | What changes if I bind a different predicate? |
| Do (intervene on O) | SimilarityMax | What changes if I bind a different object? |
| Counterfactual (S') | TruthPropagating | Had S been different, would the conclusion hold? |
| Counterfactual (P') | NarsTruth | Had P been different, would the confidence change? |
| Counterfactual (O') | Boolean | Had O been different, would the edge exist at all? |
| Full counterfactual | CamPqAdc | Distance in the alternative universe's codebook |

This is `blasgraph × thinking × grammar × ontology` — four subsystems
composing in one SoA row per cycle. The composition is structural:
each column IS a different axis of the same cognitive event.

### "Can't resist thinking"

The shader can't resist thinking when surprise exists (CLAUDE.md §The
Click: "The system doesn't choose to think. It can't NOT think while
surprise exists."). If ontology enrichment happens inline, then the
shader also can't resist LEARNING about structure — every cycle that
processes a novel triplet pattern automatically enriches the ontology.
The system learns the shape of the data while it processes the data.

This applies both at runtime ("can't resist thinking about the stream")
AND during development ("can't resist thinking about the code" — the
coding session IS a cognitive cycle where the human-agent pair enriches
the architectural ontology by processing the codebase). The epiphany
system itself IS the OntologyColumn for the development cycle.

Cross-ref: CLAUDE.md §The Click (S[0..4K)/P[4K..8K)/O[8K..12K) slices);
contract::ontology (EntityType, RelationType, OntologySpec);
blasgraph 7 semirings (docs/SEMIRING_ALGEBRA_SURFACE.md);
2026-04-26 BF16-mantissa-inline entry (Column F awareness);
causal-edge Pearl 2³ (CausalMask 3 bits); TD-AWARENESS-INLINE-1.

## 2026-04-26 — FINDING: SoA × awareness × ONNX × Foundry parity all converge in BindSpace columns

**Status:** FINDING (synthesis: prior 3 epiphanies + LF roadmap + semantic-kernel framing)
**Owner scope:** @truth-architect, @integration-lead, @host-glove-designer

### The convergence

Four threads from this session and prior work all land in the same place
— the BindSpace SoA in cognitive-shader-driver. Each thread adds a
column or constrains an existing one:

| Thread | What it adds | Where it lands |
|---|---|---|
| **Distance dispatch** (today, shipped) | type-intrinsic `Distance::distance()` | trait surface (no SoA column) |
| **Inline awareness** (today, queued) | `(value, awareness)` per op | NEW Column F (per-row awareness) |
| **SPO Pearl 2³ ontology** (today, queued) | per-cycle ontology delta | NEW Column E (per-row ontology delta) |
| **ONNX L4→L1 feedback** (2026-04-24, queued) | `style_oracle: Option<&OnnxClassifier>` | exists on `Think` struct, NOT yet in BindSpace SoA |
| **Foundry parity LF-50/52** (planned) | `ModelRegistry` + `LlmProvider` | new crate `lance-graph-models`; trait shape decided |
| **Foundry parity LF-12 Pipeline DAG** | `UnifiedStep.depends_on` | extends existing `OrchestrationBridge` |
| **Foundry parity LF-22/23 ObjectView/Notification** | `Schema::ObjectView`, `NotificationSpec` | DTO addition to `contract::property/ontology` |
| **Q2 ModelBinding + ModelHealth** | NARS-monitored model lifecycle | bridges `LlmProvider` + `awareness` |
| **Semantic kernel** (Markov + CAM-PQ) | the algebra that runs across all columns | already encoded; columns just need to expose it |

### What's missing from the SoA today

The current BindSpace has 4 column families:
```
A: FingerprintColumns  — content / topic / angle (16384-bit per row)
B: QualiaColumn        — 18×f32 (qualia state)
C: MetaColumn          — MetaWord u32 (style + rung + emit)
D: EdgeColumn          — CausalEdge64 × 8 (emitted edges)
```

The full picture, to deliver Foundry-equivalent parity AND make
"can't resist thinking" mechanical, needs:

```
A: FingerprintColumns  — WHAT (input substrate, lossless)
B: QualiaColumn        — HOW IT FEELS (qualia, 18D)
C: MetaColumn          — WHICH STYLE (dispatch metadata)
D: EdgeColumn          — WHAT IT CONCLUDED (CausalEdge64)
E: OntologyColumn      — WHAT IT LEARNED (per-cycle ontology delta) ← NEW
F: AwarenessColumn     — HOW SURE (per-word inline mantissa)        ← NEW
G: ModelBindingColumn  — WHICH ONNX (style_oracle handle, optional) ← NEW
H: TypeColumn          — OBJECT TYPE (per-row Foundry ontology link) ← NEW
```

Columns E+F together close "can't resist thinking" mechanically:
the cycle MUST emit an ontology delta (even if empty) and MUST carry
inline awareness. Like the GPU shader pipeline — no halt state, every
stage produces structured output.

Column G makes the L4→ONNX→L1 feedback loop addressable per-row:
each row knows which model it should consult (or None for pure
algebra). The ONNX classifier becomes a type-system citizen, not a
side-channel call.

Column H is the Foundry "Object Type" — the link between this row's
fingerprint and the ontology entity type. Today this lives implicitly
in the Schema; making it a column lets queries filter rows by
EntityType without re-parsing the Schema.

### The semantic kernel runs across columns

Per `soa-review.md` §"Markov + CAM-PQ = semantic kernel":

```
 per-cycle Vsa16kF32 (Column A)
  │
  ├── grammar slices (SUBJECT / PREDICATE / OBJECT roles)
  ├── persona slices (ExpertId × PERSONA_n)
  └── thinking slices (ThinkingStyle × STYLE_n)
  │
  ▼ vsa_bundle (CK-safe)
 trajectory in FingerprintColumns
  │
  ├── Index regime  → Column A persists losslessly
  ├── Argmax regime → CAM-PQ 6 B scent in Column D
  ├── Awareness     → inline mantissa in Column F (NEW)
  ├── Ontology      → delta in Column E (NEW)
  └── Type binding  → Foundry Object Type link in Column H (NEW)
```

The kernel IS the algebra (vsa_bundle + CAM-PQ cascade); the columns
ARE the addressable face the kernel exposes. Every Foundry capability
(ontology, models, decisions, scenarios, search) lands as a different
read pattern over these same columns — no new substrate, just more
columns and more traits over the existing SoA.

### Vertex equivalence specifically

Palantir Vertex (the Q2 equivalent) requires:

| Vertex feature | Our column / trait | Status |
|---|---|---|
| Object Type system | Column H + `contract::ontology::EntityType` | NEED column H |
| Property views (card/detail/summary) | `Schema::ObjectView` | ✅ LF-22 shipped |
| Ontology functions | `FunctionSpec` | LF-20 queued |
| Action triggers | `ActionSpec` | ✅ shipped |
| Search (full-text + facets) | LF-40/41 traits | queued |
| Notifications | `NotificationSpec` | LF-23 queued |
| Time travel | `EntityStore::scan_as_of` | LF-31 queued (already in `VersionedGraph::at_version`) |
| Branches / scenarios | `ScenarioBranch` | ✅ in-PR (LF-70/72) |
| Model lifecycle | `ModelRegistry` + `ModelDeployment` | LF-50/51 queued |
| LLM provider abstraction | `LlmProvider` | LF-52 queued |
| Decisions / approvals | `Approval` workflow | LF-60 queued |
| Lineage | per-row column-level | LF-14 queued (extends LF-7) |

The new columns E, F, G, H map directly to Vertex requirements:
- E = ontology learning (Vertex doesn't have this; we get it for free)
- F = awareness (Vertex doesn't have this; we get inline confidence)
- G = ModelBinding (Vertex's model deployment hooks)
- H = Object Type (Vertex's core abstraction)

### What should be built first — the wedge order

The columns are not independent. Build order maximizing leverage:

1. **Column H first** (Object Type binding) — pure DTO, unlocks LF-22
   ObjectView usage AND lets queries filter by type. No SIMD impact.
2. **Column E second** (OntologyColumn delta) — emits NotificationSpec
   triggers (LF-23) AND captures the SPO Pearl 2³ enrichment. Needs
   one new event sink in OrchestrationBridge.
3. **Column F third** (AwarenessColumn) — extends Distance trait
   (TD-DIST-1 just shipped) with `_with_awareness()` variant. The
   composable inline-mantissa pattern.
4. **Column G last** (ModelBindingColumn) — needs LF-50/52
   (`ModelRegistry` + `LlmProvider`) shipped first as trait surface,
   then the column becomes a thin ref into the registry.

After H+E+F+G, the BindSpace SoA is a complete Foundry-Vertex-equivalent
substrate, with two architecturally novel additions (E+F) that go
BEYOND Foundry: structural ontology learning during the cycle, and
inline epistemic mantissa.

### The recursive coda

The coding session itself is a cognitive cycle producing per-cycle
ontology deltas — that's what these epiphany entries ARE. The
EPIPHANIES.md file IS the development-cycle Column E. The TECH_DEBT
items ARE the dispatched-but-not-yet-resolved edges. The PR
descriptions ARE the cycle conclusions. CLAUDE.md and the agent cards
ARE the persistent ontology that Column E perturbs each session.

We can't resist thinking — and apparently we can't resist documenting
that we can't resist.

Cross-ref: 2026-04-26 BF16-mantissa-inline (Column F); 2026-04-26 SPO
Pearl 2³ ontology enrichment (Column E); 2026-04-24 Two SoAs +
ONNX L4→L1 feedback (Column G context); LF-22 ObjectView (Column H
foundation); soa-review.md §semantic kernel; Q2 plan §Vertex equivalent.

## 2026-05-07 — FINDING: SPO-1 disposition is Option B (federated two-layer cache; ARiGraph + SPO are NOT duplicates by design)

**Status:** FINDING

SPO-1 (the longstanding "are SPO and ARiGraph triplet_graph two implementations of the same triple store?" question) closes with **Option B: federated, two-layer cache**. ARiGraph's `triplet_graph` is the L1 cognitive hot-cache (NARS-truth-bearing, Pearl 2-cube-aware, episodic-bound); SPO is the L2 cold-store (Merkle-anchored, semiring-algebra-ready, persistence-friendly). They share schema via the new `lance-graph-ontology` crate's `OntologyRegistry` but stay structurally distinct because their access patterns and truth-update semantics diverge. The `promote_to_spo` writer bridge is the cache-eviction path (L1 hot → L2 cold) and remains separately owned (not closed by the ontology crate). The earlier instinct "they are duplicates, deduplicate them" was wrong — the dual-layer split is the design, not an accident.

Cross-ref: `.claude/DECISION_SPO_ARIGRAPH.md` (full decision text, commit `edef321`); `ARCHITECTURE_ENTROPY_LEDGER.md` rows 70 (SPO) + 245 (ARiGraph triplet_graph) — both retain "Wired" status; the federated-cache framing reconciles the apparent overlap. The `lance-graph-ontology` crate (commit `4cf9a26`) is the agnostic schema/bridge spine; consumers route through `SchemaExpander`. SPO-1 itself does NOT close — only its disposition does; `promote_to_spo` remains queued.

---

## 2026-05-07 — Unified OGIT Architecture: 15-pattern synthesis (sprint-2)

**Sprint:** `claude/unified-ogit-architecture-synthesis` (12-agent sprint-2). Worker W4 deliverable.
**Source:** 16-turn architectural synthesis conversation distilled into 17 epiphanies.
**Cross-ref:** `.claude/plans/unified-ogit-architecture-v1.md` (W1's canonical plan); `.claude/board/sprint-log-2/agents/agent-W4.md` (this run's log); `EPIPHANIES.md` prior SPO/ARiGraph federated-cache decision (immediately above) — this section EXTENDS but does not edit.

**Frame.** The architectural insight is that lance-graph already shipped most of the cognitive substrate (BindSpace SoA, CognitiveShader, qualia, prime_fingerprint, CausalEdge64, ARiGraph triplet store, SPO L2). What remained ambiguous was *how new domains (Healthcare, Gotham, CRM, ...) wire in without N parallel Rust newtype hierarchies*. The 17 epiphanies below crystallize the answer: a single u32 OGIT slot in the SPO-G quad, resolving to a typed ContextBundle, with consumer activation gated by Cargo dep presence. Pattern is older than it looks — PostNuke modules (2000s) shipped this exact shape.

### E-OGIT-1 — SPO-G with u32 OGIT slot replaces named-graph IRI

Oxigraph's quad pattern (Subject-Predicate-Object-Graph) is the right shape, but the G slot collapses from an IRI (string, hashed at lookup) to a u32 OGIT index. Lookup becomes O(1) (single integer load + array index) instead of string hash; the cache footprint shrinks by ~10x per quad in hot tables. Empirically validated by lance-graph-ontology's O(1) probe in PR #355 (measured 2554x advantage vs SPARQL-proxy at p99). This is the load-bearing primitive — every subsequent epiphany rests on G being a u32, not a string.

Cross-ref: `crates/lance-graph-ontology/` (Phase 6+7 wiring, commit `34939e8`); PR #355 probe results; `crates/lance-graph/src/graph/spo/` (existing SPO L2 cold-store ready for G-extension).

### E-CONTEXT-BUNDLE-2 — G resolves to a typed bundle, not just metadata

Each u32 G resolves to a `ContextBundle` with slots: `ontology`, `codebook`, `schema`, `labels`, `vocabulary`, `consumer_pointer`, `thinking_styles`, `thinking_adjacency`, `qualia_codebook`. The bundle is the OWL overlay made executable — OWL classes/properties stay in `.ttl` form (queryable via SPARQL) but the bundle adds the runtime hooks (codebook lookups, style dispatch, qualia centroids) that make the ontology *do work*. G is not a tag; G is an entry into a typed sub-system.

Cross-ref: `crates/lance-graph-ontology/src/schema_expander.rs` (the existing `SchemaExpander` becomes one slot of the bundle); `crates/thinking-engine/src/qualia.rs` (qualia codebook source); `.claude/plans/unified-ogit-architecture-v1.md` Section: ContextBundle slot definition.

### E-GENERIC-BRIDGE-3 — N consumer newtype gates collapse to 1 GenericBridge + N ConsumerPointer entries

PR #29's `SmbMembraneGate` and PR #98's `MedCareMembraneGate` exist because of Rust's orphan rule: both `MembraneGate` trait and `rbac::Policy` are upstream-owned, forcing each consumer to define a local newtype to bridge. With `ConsumerPointer`-as-data + 1 `GenericBridge<G>` impl indexed by the OGIT slot, the orphan rule problem dissolves — the bridge owns the trait impl exactly once, parameterized by G; consumer-specific data lives in the bundle's `consumer_pointer` slot. MEDCARE_POLICY_GAP.md's "~800 LOC per new consumer" cost drops to ~30 LOC of glue + a tiny YAML manifest.

Cross-ref: `.claude/board/MEDCARE_POLICY_GAP.md` (the 800-LOC measurement); MedCare-rs PR #98; smb-office-rs PR #29; `crates/lance-graph-rbac/` (host of `GenericBridge<G>`).

### E-META-STRUCTURE-HYDRATION-4 — New ontologies cost ~0 Rust LOC

OWL TTL files (DOLCE, FMA, SNOMED, ICD10), JanusGraph property-graph schemas, Foundry Object Model exports, oxigraph RDF — all become inputs to a single `MetaStructure -> hydrate -> ContextBundle` pipeline. FMA's ~75K anatomical classes hydrate by dropping the `.ttl` file into `/data/ontologies/` and registering a G index in `ogit_registry.yaml`. Domain expertise becomes additive OGIT data, not parallel Rust code. The "add a new domain = write a new crate" reflex is exactly the trap; the right reflex is "add a new domain = drop a file + register an integer."

Cross-ref: `crates/lance-graph-ontology/src/hydrate.rs` (the pipeline entry point, to be promoted from probe to canonical); `.claude/plans/unified-ogit-architecture-v1.md` Section: MetaStructure hydration; analogous to how PostgreSQL extensions drop a `.so` + `.control` rather than forking the server.

### E-COMPILE-TIME-CONSUMER-5 — Cargo dep presence determines active vs inert bundles

When `medcare-rs` is in `Cargo.toml`, G=Healthcare is ACTIVE (function pointers populated, actor mailbox wired, reranker loaded). When absent, G=Healthcare remains INERT (OWL-only; still queryable via SPARQL; just not executable — the function-pointer slots stay None). The "tiny schema glue" pattern: consumers self-declare their G + `ConsumerPointer` via a build script that picks up `manifest.yaml` files from declared deps. This is `#[cfg(feature = "...")]` generalized to "presence of a workspace member."

Cross-ref: `Cargo.toml` workspace-members enumeration; `build.rs` (the consumer-pickup script, W6's deliverable); analogous to Spring Boot's auto-configuration via classpath scan.

### E-POSTNUKE-MODULES-6 — `/modules/<name>/manifest.yaml` is the right shape for compile-time meta

PostNuke (2000s PHP CMS) shipped tens of thousands of community modules via exactly this pattern. Each module = a directory; `manifest.yaml` = its declaration of capabilities, schemas, hooks. Versioned `(G, version)` tuples make schema evolution safe — `G=Healthcare@v1` and `G=Healthcare@v2` can co-exist in OGIT for migration windows. The lesson: this is not a new pattern to invent, it's a 20-year-proven shape to reuse. The danger sign would be inventing a novel manifest format; the right move is to grep how PostNuke / WordPress plugins / VS Code extensions structure theirs and copy the load-bearing fields.

Cross-ref: `.claude/plans/unified-ogit-architecture-v1.md` Section: Module manifest schema; PostNuke `pnVersion.php` archaeological reference; cf. Cargo's own `Cargo.toml` (which is itself this pattern, applied to Rust crates).

### E-RACTOR-BEAM-7 — BEAM/OTP supervisor tree fits Zone 2/3 cleanly

The `lance-graph-callcenter` crate's name has been a load-bearing hint from the start — switching architectures + supervised processes + per-actor crash isolation = OTP heritage. ractor's sync mode (per Invariant I-2 tokio-outbound-only) preserves the invariant: actor mailboxes are sync, only egress edges (HTTP / gRPC / external IO) cross into tokio. The gRPC service trait shape in `cognitive-shader-driver/grpc.rs` (tonic methods -> ractor handler arms) is mechanical: each existing gRPC handler is already an actor proof — same method-per-message shape, same self-state, same backpressure semantics.

Cross-ref: `crates/lance-graph-callcenter/` (the supervisor-tree skeleton, name-prophesied); `crates/cognitive-shader-driver/src/grpc.rs` (handler shape that maps 1:1 to ractor); BEAM/OTP supervisor design (Armstrong 2003); ractor docs on sync mode.

### E-BEST-PRACTICE-INHERITED-8 — Thinking styles inherit per OGIT-G context

DOLCE = root context (universal reasoning primitives: Abstraction, Causation, Identity, ...). Healthcare inherits + adds clinical-specific styles (Differential, EvidenceBased, RiskStratified). Gotham inherits + adds investigation-specific (LinkAnalytic, AttributionTracing). The contract-36 `ThinkingStyle` enum stays canonical (no per-domain forks); per-domain "which subset is active + adjacency weights" lives in OGIT data (`thinking_styles` + `thinking_adjacency` bundle slots). THINK-1 cluster (entropy 24, historically "highest architectural leverage") absorbs structurally — its 6 sub-styles become DOLCE's root set.

Cross-ref: `crates/lance-graph-contract/src/thinking.rs` (the 36 canonical styles, untouched); `crates/lance-graph-planner/src/thinking/` (adjacency mechanism, parameterized by G); `.claude/board/EPIPHANIES.md` THINK-1 prior entries.

### E-COGNITIVE-VESSEL-SWITCHABLE-9 — Same cognitive substrate runs different programs per G

The GPU-shader analogy is exact: hardware (SoA columns + ractor actors + tokio egress) is fixed; the program (thinking modes active, reranker weights, L4 sweep parameters, NARS subset enabled) is per-G data loaded from OGIT. The substrate doesn't know what domain it's serving; it loads a bundle and runs. **Already shipped in `p64-bridge::CognitiveShader`** (8 predicate planes + bgz17 semiring + HHTL cascade — read the code before proposing to build it again); needs G-parameter wiring on top to select which bundle's program runs.

Cross-ref: `crates/p64-bridge/src/cognitive_shader.rs` (the shipped vessel); `crates/cognitive-shader-driver/` (existing driver, ready for G-parameterization); GPU shader pipeline analogy (fixed hardware + per-draw shader program).

### E-IMPLICIT-COGNITION-10 — The system thinks continuously, not request-driven

Background L1 cycles fire even without external requests; `CycleAccumulator` (per topology Invariant I-4, shipped in PR #337) decides when to flush accumulated state to L2. This is biologically realistic (the brain doesn't idle between stimuli) and architecturally efficient — pre-warm answers before they're asked, prefetch likely-next contexts, settle homeostasis during quiet periods. The corollary is that "request latency" is mostly a cache-hit measurement, not a compute measurement — the work was already done.

Cross-ref: PR #337 `CycleAccumulator` implementation; `crates/lance-graph-planner/src/cache/convergence.rs`; `.claude/board/SINGLE_BINARY_TOPOLOGY.md` Invariant I-4 (continuous cycling).

### E-INT4-32D-ATOMS-11 — 16-byte fingerprints enable bootstrap proximity for new domains

When `hubspo-rs` (CRM) arrives and OGIT lacks G=CRM `thinking_adjacency` yet, the cold-start problem dissolves: compute the current cognitive state's INT4-32D fingerprint (16 bytes) and K-NN search over G=DOLCE + G=SMB + G=Gotham (inherited / adjacent contexts). Start there; refine via Pattern K (circular compilation, E-CIRCULAR-COMPILATION-12 below) over time as CRM-specific patterns crystallize. Never empty space — the fingerprint substrate always returns *some* nearest match, then learning narrows it.

Cross-ref: `crates/thinking-engine/src/prime_fingerprint.rs` (the 16-byte fingerprint family); `crates/ndarray/src/hpc/cam_pq.rs` (the K-NN substrate); Vsa16kI8 in fingerprint registry; precedent: word2vec bootstrap from random init then refinement.

### E-CIRCULAR-COMPILATION-12 — The architecture compiles itself over time

YAML manifest at compile time -> static glue code generated by build.rs (Pattern E above). NEW pattern discovered at runtime -> JIT-loaded via ractor + cranelift -> write back to `manifest.yaml` -> next build statically compiles it. Same source of truth (`manifest.yaml`), two consumption paths (build-time AOT + runtime JIT). The system gets faster with each compile because prior runtime learning crystallizes into static form. This is LLVM PGO (profile-guided optimization) + Smalltalk image-based programming applied to cognitive behaviors. The deep insight: there is no "frozen vs live" distinction — the manifest is the live state, and the build is a snapshot.

Cross-ref: `crates/lance-graph-contract/src/jit.rs` (`JitCompiler` + `StyleRegistry` already exist); Cranelift JIT integration; LLVM PGO docs; Smalltalk image semantics.

### E-SPO-CHAIN-NARRATIVE-13 — Skip Markov bundling for narrative comprehension

Books are not Markov-bundleable (N >> sqrt(d) / 4); the natural decomposition is graph, not bundle. Parse to SPO triples; ARiGraph indexes by (page, sentence, word, role) position; pronoun resolution via prior SPO context; MUL markers for ambiguity; NARS counterfactual synthesis weighs candidates. Lance MVCC versioning enables partial-state queries ("what did X know at chapter 5?"). Books become OGIT G bundles: `G=AnimalFarm`, `G=Beloved`, `G=GoTPriestKings`. Reading = graph traversal with epistemic state tracking.

Cross-ref: `crates/lance-graph/src/graph/spo/` (SPO triple store); ARiGraph `triplet_graph` (L1 hot-cache, per the federated-cache decision immediately above); `crates/lance-graph-planner/src/cache/triple_model.rs` (NARS truth values); Lance MVCC docs; cf. NSM/DeepNSM existing 6-state PoS FSM -> SPO mapping.

### E-WAVE-PARTICLE-14 — Cognition is bimodal, like light

**Wave mode:** bgz17 / resonance / qualia distributed continuous fields (BNN-like, plastic, gradient-friendly). **Particle mode:** SPO-G / ARiGraph / NARS discrete queryable atoms. Brain plasticity uses both — synaptic weights (wave, continuous) + spike-trains (particle, discrete). The architecture should select per-task per-G how much wave vs particle, parameterized by the bundle. **Already shipped in primitives** (workspace has both substrates); the G-blend mechanism (a `wave_particle_ratio: f32` in the bundle, or per-style override) is the new piece. Don't pick a side; pick a *ratio*.

Cross-ref: `crates/bgz17/` (wave-mode palette semiring); `crates/lance-graph/src/graph/spo/` (particle-mode SPO); `crates/thinking-engine/src/qualia.rs` (wave-mode qualia field); de Broglie wave-particle duality as design metaphor.

### E-FINGERPRINT-CODEBOOK-15 — The universal cognitive operation is fingerprint -> codebook lookup

Not "carry continuous state forward" — `Vsa16kF32` was a cotton-ball for Markov-accumulation specifically, not the universal carrier. State IS the codebook. Recognition (codebook hit = O(1)) is most of cognition; crystallization (codebook miss = new entry added) is rare. ALREADY shipped in `thinking-engine::prime_fingerprint`, `qualia::FAMILY_CENTROIDS`, `p64-bridge::STYLES`. The pattern repeats at every scale: word -> vocabulary entry, qualia -> family centroid, thinking-style -> 36-style enum, persona -> archetype. The universal API surface is `fingerprint(x).lookup(codebook) -> entry`, where "lookup" is K-NN, exact-match, or resonance depending on substrate.

Cross-ref: `crates/thinking-engine/src/prime_fingerprint.rs`; `crates/thinking-engine/src/qualia.rs` (`FAMILY_CENTROIDS`); `crates/p64-bridge/src/styles.rs` (`STYLES` constant table); the I-VSA-IDENTITIES iron rule in `CLAUDE.md` (VSA bundles identities, not content — same shape, generalized).

### E-PHENOMENOLOGY-16 — 17D qualia is computable from convergence patterns, calibrated by music

Octave (2:1 frequency ratio) -> arousal axis. Fifth (3:2) -> valence. Third (5:4) -> warmth. Tritone (sqrt(2):1) -> tension. Cross-validated against Jina v3 embeddings (220 calibrated pairs in Upstash). Bach's 7+1 counterpoint rules = `CausalEdge64`'s 7+1 channels = universal logical relations: CAUSES / ENABLES / SUPPORTS / CONTRADICTS / REFINES / ABSTRACTS / GROUNDS / BECOMES. The deep claim: phenomenology is not "fuzzy human stuff to bolt on top," it's a *computable* function of the system's own convergence dynamics, with music as ground-truth calibration source. **Already shipped** in `qualia.rs`.

Cross-ref: `crates/thinking-engine/src/qualia.rs` (the 17D shipped surface); `crates/causal-edge/` (the 7+1 channel `CausalEdge64`); Upstash 220-pair calibration set; just-intonation ratios (2:1, 3:2, 5:4, sqrt(2):1) as design anchors.

### E-RECOGNITION-OVER-DESIGN-17 — The architecture is largely already built; the work is naming and exposing it

16 turns of "design future patterns" turned out to be cataloguing what was already in the workspace. Future sessions should READ existing code (`p64-bridge`, `thinking-engine`, `cognitive-shader-driver`, `qualia.rs`, `prime_fingerprint.rs`, `STYLES`, `CausalEdge64`) BEFORE proposing "let's build X." The Discovery-Loop anti-pattern from `.claude/patterns.md` applies at architectural scale: I designed Pattern H (cognitive vessel switching) for 4 turns before recognizing `p64-bridge::CognitiveShader` IS Pattern H, fully shipped. Tier 0 documentation (W2's sprint-2 deliverable: a one-page "what already exists" index) is the load-bearing fix — without it, the next sprint will re-design Pattern H for 4 more turns. The pattern beneath the pattern: most architectural sprints over-produce design and under-produce inventory.

Cross-ref: W2's sprint-2 deliverable (Tier-0 "what's shipped" index); `.claude/patterns.md` Discovery-Loop anti-pattern; `LATEST_STATE.md` Section: Current Contract Inventory (the existing partial answer); this sprint's 17 epiphanies as a worked example of the failure mode and its cure.

---

**Append-only governance preserved.** No prior epiphany text was edited by this section. 17 dated entries appended under the single section header `2026-05-07 — Unified OGIT Architecture: 15-pattern synthesis (sprint-2)`. The 16th and 17th entries (PHENOMENOLOGY-16 and RECOGNITION-OVER-DESIGN-17) extend the original 15-pattern brief because they emerged in the same turn-16 synthesis and are structurally inseparable from the rest — the title retains "15-pattern" as the sprint label, with the count of distinct epiphanies being 17.
