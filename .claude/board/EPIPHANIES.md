## 2026-06-17 — E-ODOO-EXTRACT-DEEP-READS — the consumer-side recompute-DAG probe surfaces a corpus enrichment ask, not a ClassView gap

**Status:** FINDING (consumer-side, ratifying gap on producer side). `od-ontology::RecomputeDag` shipped at [`AdaWorldAPI/odoo-rs@d8a270d`](https://github.com/AdaWorldAPI/odoo-rs/commit/d8a270d) (540 LOC, 8 tests, clippy-clean) topologically sorts the `MethodKind::Compute` subset of the slice 1 / slice 2 corpora cleanly. The hand-found `_compute_amount.md` MISSED-1 P0 cycle (`move._compute_amount` → `line.reconciled` → `line._compute_amount_residual`) is **structurally invisible** because the Odoo extractor's `reads_field` for `@api.depends('line_ids.reconciled')` emits only the **first hop** (`account_move.line_ids`) — not the **leaf** (`account_move_line.reconciled`). One sibling triple per `@api.depends` leaf reads, same wire shape, same provenance source, would land the catch.

**The wishlist (consumer-canonical):** [`AdaWorldAPI/odoo-rs UPSTREAM_WISHLIST.md`](https://github.com/AdaWorldAPI/odoo-rs/blob/main/crates/od-ontology/specs/UPSTREAM_WISHLIST.md). The P0 section names the ask + carries the PROBE FINDING block under it; the probe's two findings are baked in (slice 2 compute acyclic; full graph carries a legitimate `_onchange_*` cooperative loop — filtered out by `MethodKind`).

**Why this matters here.** The ClassView design session is NOT blocked by this ask. The consumer correctly framed the recompute DAG as needing the *machinery* (which is now shipped consumer-side) AND the *data* (which the extractor owns). The reframe matters: the consumer's wishlist had originally listed this as P0 ASK to the ClassView session; the probe demonstrates that one extractor change is sufficient, no ClassView interface required. Producer-side action: when re-running the extractor on `/home/user/odoo`, lift the second AST level of `@api.depends('a.b')` strings into a sibling `reads_field` triple resolved through the relation's `comodel_name`. See `.claude/knowledge/odoo-extractor-wishlist-from-od-ontology-v1.md` § 3 for the concrete shape.

**Companion ratifications** (also 2026-06-17, also from the producer side): `ruff#19` (Rails STI → `inherits_from`, making `inherits_from` cross-language canonical for C++ + Rails + a natural P1 ask for Odoo `_inherit`) and `ruff#21` (`Predicate::ValidationKind` for per-attribute typed validation, a P2 ask for Odoo `@api.constrains`). Both recorded in the wishlist's "2026-06-17 update" section.

## 2026-06-17 — E-MATERIALIZED-AWARENESS-2 — the driver wire is live (provenance-only); the four vocabularies are one 2-axis structure

**Status:** FINDING (shipped on branch `claude/materialize-awareness-f34-loop`): the `cognitive-shader-driver` now runs the `materialize` F→34→F loop + the ndarray HHTL `fork_decision` as a **side analysis** per cycle, recording `MaterializeProvenance` on `ShaderCrystal`. **Provenance-only — the gate/emit/persistence path is byte-for-byte unchanged** (operator decision 2026-06-17). 2 driver tests + 638 contract lib green.

**The unification, now grounded in shipped types.** The entropy×energy `Quadrant`, Csikszentmihalyi's `mul::FlowState(challenge,skill)`, Friston model-vs-surprise, and the Staunen↔Wisdom ladder are **one 2-axis structure**. CHALLENGE = surprise (`free_energy` / orthogonal leaf-residue magnitude); SKILL = engagement (`confidence`+`ATTEND_GAIN` / in-domain codebook capacity). The driver's observable→`ThoughtCtx` mapping is faithful (sd←std_dev exact, confidence←1−F = the driver's own `demonstrated_competence`, dissonance←|felt−demonstrated| = the Dunning-Kruger gap). The HHTL fork is the **anxiety escape**: Anxiety (challenge≫skill) at leaf → `ForkDomain` (mint a new classid = Friston model-switch); Flow → resolve in-domain; Boredom → Commit. Cross-repo: ndarray ships the fork math (PR #221 merged, `entropy_ladder::{residue_surprise, fork_decision, ForkAction}`), lance-graph drives it.

**[H]/[S] joints (unchanged from E-…-1, now with the wire built):** the fork challenge is a **`std_dev` dispersion proxy (CONJECTURE)** with a std_dev-calibrated floor/σ — the real orthogonal `CoarseResidue` magnitude is not yet surfaced into the cycle; HHTL cascade depth is stubbed `depth==max⇒leaf`; the σ-threshold awaits a Jirak-derived bound (`I-NOISE-FLOOR-JIRAK`). Promoting these to [G] = the next gated wire (surface real residue + cascade depth; optionally let the provenance feed a future gate path once measured to help).

## 2026-06-16 — E-MATERIALIZED-AWARENESS-1 — awareness materializes iff it is *causal in dispatch*; the closed `F→34→F` loop is the reduction-to-practice (and the falsifier)

**Status:** FINDING for the criterion + loop (reduction-to-practice **shipped**: `lance-graph-contract::materialize`, 6 tests green, zero-dep/offline). The broader "this is the system's awareness" reading stays **[NOVEL — probe-gated]**: no prior art by construction (ours — the 2³-rung→NARS-candidate→34-tactic dispatch loop), validity established by the perturbation probe, not citation.
**Confidence:** High on the criterion + the shipped loop; the wire to the *real* substrate (driver-side `ThoughtCtx::from_live` + version-diff provenance) is the gated next step.

**The criterion (falsifiable).** *Awareness materializes iff perturbing the surprise/free-energy signal changes which tactic fires.* If dispatch is invariant to the awareness state, the awareness is a **dead label** — "awareness that can never materialize." `materialize::awareness_is_causal(base, lo_f, hi_f)` is the predicate; the test `awareness_free_energy_is_causal_in_dispatch` is the green falsifier, and `non_awareness_fields_are_inert` is its specificity control (candidates/beliefs must NOT steer dispatch).

**The wire that was missing (now built).** The 34 tactics (`recipe_kernels`, the canonical "34" — the ndarray `hpc/styles/*` set is divergent/registry-less and is NOT canonical) were dispatch *targets* with no selector and an open loop (they ran only in an example against a toy ctx; the driver loop ran the *12* threshold ordinals, leaving the rich 34 inert). `materialize` adds: (a) **`select_tactic`** — awareness→id, with `free_energy` (surprise) as the **primary** axis so dispatch tracks awareness by construction; (b) **`materialize`** — the closed loop: select → `Tactic::run` (folds `delta_conf`) → settle the gate (dispersion/contradiction decay) → recompute surprise → re-dispatch; **rest is reached** when the CollapseGate is in FLOW (`sd<SD_FLOW`) **and** residual surprise falls below `HOMEOSTASIS_FLOOR` (0.2) — a cool gate with unresolved surprise is not rest. For a *firing* chain this is guaranteed: attending decays dispersion and raises confidence each fired step, so both `sd` and surprise descend monotonically; a *blocked* tactic ends the run (re-dispatch of an unchanged state cannot unblock it). "The shader can't resist the thinking" made literal. The settle/attend updates fire only on a tactic that actually fired (review #515: a blocked tactic must not fake progress; `free_energy` stays the primary dispatch axis even under contradiction — `dissonance` is a lower-weight secondary, not an override).

**Prior-art positioning (not competitors — background for the disclosure).** NOTEARS / PCMCI / DCDI / ICP / SEA are adjacent observational/interventional *discovery* methods (arXiv 1803.01422 / 1702.07007 / 2007.01754 / 1501.01332 / 2402.01929); our loop does not *discover* a DAG — it dispatches reasoning over recorded/candidate structure and lets NARS revise. **Operating boundary respected, [G]:** Janzing-Schölkopf (0804.3678) — Shannon-symmetric, colliders-only observationally; full orientation needs mechanism asymmetry (so dispatch never claims identified orientation, only revisable candidates).

**Open / next.** The shipped loop runs on the in-memory `ThoughtCtx`; wiring it to the live shader (build `ThoughtCtx` from `FreeEnergy`/`MulAssessment`/hits in `driver.rs`, fold the trace into the SoA EdgeColumn / version-diff "what-fired-why" provenance) is the gated driver-side step. The materialization probe is the acceptance test for that wire too.

## 2026-06-16 — E-TRANSCODE-EXEC-LADDER-1 — the Core-First transcode has a 3-rung execution ladder (codegen → two-tier compile → elixir-tissue over surreal/kanban/odoo), and rungs 2–3 land on already-shipped substrate

**Status:** CONJECTURE (operator forward-design). v1 is the shipped doctrine; v2/v3 are gated on `PROBE-COMPILE-TWO-TIER` + `PROBE-SURREAL-TISSUE-SWAP` (both in `core-first-transcode-doctrine.md`), themselves floored by the v1 `PROBE-OGAR-ADAPTER-UNICHARSET`.
**Confidence:** Medium — the substrate each rung lands on is shipped and cited (`contract::kanban`, `contract::jit`, `surreal_container`, `E-SUBSTRATE-IS-THE-SCHEDULER`); the two NEW edges (Odoo→kanban ingest, AST-DLL tissue hot-swap) are unbuilt.

**Context.** The Core-First Transcode Doctrine (knowledge doc, captured this session) framed transcode v1: thin classid-keyed adapters target the OGAR Core, bodies codegen'd at build. The operator then extended it along the *execution model* — how a body is compiled and where it lives — across two more rungs.

**The ladder.**
- **v1 — Core-first codegen.** Bodies generated once at build, targeting `canonical_node` / `classid → ClassView` (#498).
- **v2 — two-tier compile.** ONE Elixir-shaped adapter source, TWO backends: **existing → compile-time** (Askama→Jinja analogy: a `defadapter!` proc-macro monomorphises to Rust, zero runtime cost), **new → JIT** (jitson/Cranelift). Not greenfield: `contract::jit` already defines `JitCompiler::compile(JitTemplate) → KernelHandle`, compiled by ndarray jitson/Cranelift, cached by n8n-rs `CompiledStyleRegistry`.
- **v3 — elixir-tissue over a fixed Core.** Core stays immutable; the DO-shaped business logic is **replaceable tissue** (BEAM hot-swap heritage — the deep reason Elixir is the right syntax to steal, not mere ergonomics) living in the **AST-DLL**, persisted + served + hot-swapped via SurrealDB's API; a **Kanban orchestration** reacts to **Odoo shapes** and dispatches the tissue. `contract::kanban`'s header already *names this seam verbatim*: planner emits `KanbanMove` → ractor drives the `KanbanColumn` → `surreal_container` projects the columns as the kanban view, carried as `UnifiedStep{step_type:"kanban.*"}` (`StepDomain::Kanban`). `E-SUBSTRATE-IS-THE-SCHEDULER` already has the substrate emit the schedule via surreal LIVE.

**Why this matters.** The transcode's "holy grail" (Core-first thin adapters) was framed as a build-time codegen story. The execution-model ladder shows the SAME invariant survives JIT and hot-swap: whether a body is macro-monomorphised, Cranelift-JIT'd, or swapped from the SurrealDB AST-DLL, it is STILL a thin adapter targeting the OGAR Core, and a tissue adapter needing state the Core can't hold is STILL a Core gap → EXTEND-CORE. The execution model changes; the iron guard does not. And the two ambitious rungs are ~85% shipped substrate (kanban contract, jit contract, surreal_container, scheduler epiphany) + exactly two unbuilt edges.

**The two unbuilt edges (the honest scope).**
1. **Odoo→kanban ingest** — map Odoo model records / stage transitions / automated-action triggers into `UnifiedStep{step_type:"kanban.*"}` / `KanbanMove`. No bridge exists today (`AdaWorldAPI/odoo` is the Python ERP, local at `/home/user/odoo`).
2. **AST-DLL tissue store + hot-swap over SurrealDB** — persist codegen'd/JIT'd DO adapter bodies in `surreal_container`, serve + hot-swap via the surreal API, gated by `kanban.rs`'s read-only-projection / commit ruling. Conceptually supported by the crate's `view`/`read`/`write` split; the swap-without-Core-rebuild parity is unprobed.

**Lesson.** When an operator's forward vision arrives as "vN is X + Y + Z", the high-value move is to *locate each clause on shipped substrate before treating it as new work* — here three of the four load-bearing pieces (kanban seam, JIT tier, surreal projection) already existed and only needed naming + two ingest edges. Capturing the ladder (vs. only v1) prevents the v2/v3 framing from being re-derived next session, and the cited symbols make the "already shipped" claim checkable.

**Cross-ref.** `core-first-transcode-doctrine.md` § "The execution-model ladder (v1 → v2 → v3)" (the canonical statement + the two new probes); `E-SUBSTRATE-IS-THE-SCHEDULER` (the v3 reactive tier this extends); `contract::kanban` / `contract::jit` / `surreal_container` (the shipped substrate); `tesseract-rs-ast-dll-codegen-v1` (the v1 codegen plan whose bodies become the v2/v3 tissue).

---

## 2026-06-16 — E-UNBLOCK-CASCADE-1 — three independent fork/contract landings collapsed onto the same `MailboxSoaView` seam, closing four queued deliverables in one commit

**Status:** FINDING.
**Confidence:** High — every claim is grounded in shipped code (`hhtl::from_guid_prefix`, `MailboxSoaOwner for MailboxSoA<N>`, `LanceVersionScheduler`, `SurrealMailboxView`) and a single-pass `cargo test` sweep (632 + 86 + 5 + 4 green).

**Context.** Three landings hit the workspace within ~24 h:
- PR #498 (`feat(contract): GUID decode→read-mode keystone + helix Signed360 right-size + OCR→NodeRow transcode`) — surfaced `NodeGuid::decode() → GuidParts` + `classid_read_mode`.
- `AdaWorldAPI/surrealdb` PR #34/#35/#36/#37 → main at `3aa6ab9` (lance/lancedb/object_store pins reconciled) — closed the `BLOCKED(C)` on `surreal_container`.
- The cherry-pick of `jolly-cori-clnf9` commit `463d71b` (the +149 LOC `MailboxSoaOwner` impl for `MailboxSoA<N>`) had been the integrated-cognitive-planner-v1 Seam #3.

**The find.** All three meet at exactly **one trait surface**: `lance_graph_contract::soa_view::MailboxSoaView`. Each landing made a DIFFERENT impl viable on the same boundary:
- **`MailboxSoA<N>` (cognitive)** — the in-process owner+view, so the Rubicon loop runs in-RAM (was only on `jolly`).
- **`SurrealMailboxView<'a>` (surreal-side view)** — D-PG-6's read glove, now buildable end-to-end via the fork's `kv-lance` backend.
- **`NiblePath::from_guid_prefix`** — the ontology-side keystone follow-up of #498's `classid → ReadMode` LazyLock: a deterministic 20→16 nibble fold that satisfies the routing-prefix `is_ancestor_of` invariant the LE contract names.

`LanceVersionScheduler` (D-MBX-9-IN core impl) sits one layer up and consumes ANY `V: MailboxSoaView` — so a single OUT-direction wrapper drives all three impls without case-splitting. The trait's read-only-by-design (`MailboxSoaView` has no mutator method) is the structural enforcement of `kanban.rs:1-21`'s "surreal=project-read-only, callcenter=commit" ruling; the SurrealQL adapter NOT importing `MailboxSoaOwner` is the compile-time tripwire if a future drift tries to mutate through the projection.

**Why this matters.** Four "still BLOCKED" rows from the most recent unblock-list synthesis (last sync turn) all collapse onto a SINGLE commit, because the substrate already had the shape — only three independent dep/code landings had to converge. The pattern:
- Substrate trait designed once + multiple implementors (no `Box<dyn>` in hot paths — generic `V: MailboxSoaView` everywhere).
- Read-only-by-trait-design = compile-time enforcement of the architectural ruling (no need for a runtime "you can't write through this" guard).
- A typed `BlockedColdBuild` error variant lets a heavy dep wire-up (surrealdb cold build) be deferred without breaking the contract-side adapter — the surface ships, the integrator flips it on in their branch.

**Lesson.** When three plans cite the same trait surface as their unblock dependency, the first session that lands ANY one of the implementors should ALSO ship the trait-impl shape for the others (even as a stub returning a typed error). This collapses N independent post-unblock follow-ups into 1 commit's worth of trait engineering. The cost is ~50 LOC of stub + a typed error variant; the benefit is N − 1 fewer post-merge commits per queued plan.

**Cross-ref.** Identity-architecture v1 §3 (the bijection-width problem); polyglot-container-query-membrane-v1 §D-PG-6 (the Rubicon kanban VIEW); integrated-cognitive-planner-v1 §2 (Seams #1–#6, the additive-only convergence); `E-SUBSTRATE-IS-THE-SCHEDULER` (the bidirectional kanban subscription); `kanban.rs:1-21` (the read-only ruling the trait shape enforces).

---

## 2026-06-16 — E-OCR-PLAN-DRIFT-1 — the #497 OCR-transcode plans drifted from the substrate in 6 ways; 2 were showstoppers

**Status:** FINDING (5-specialist framing — cascade-architect / family-codec-smith / palette-engineer / dto-soa-savant / truth-architect, each read the merged plans + source in full).
**Confidence:** High — every claim cited plan file:line vs current source file:line; convergent across ≥4 lenses for the load-bearing ones.

**Context.** #497 (Tesseract→tesseract-rs transcode plan family, 7 design docs) and #498 (helix `Signed360` + GUID keystone) merged within hours. The #497 plans were authored against the pre-#498 branch, so they reason against a substrate that shifted under them. Five specialists framed the merged plans against the post-#498 architecture.

**The two showstoppers:**
1. **The "reversible without a hash" rationale is false in code** (truth-architect). The migration's headline — "OCR text reconstructs from residue + codebook, no string column" — has no support: `deepnsm/vocabulary.rs` maps `rank→&str` via a stored table, every decode entry point takes a *known* rank as input, and there is no `residue→rank` inverse (helix encode is lossy). The "reversible residue" was a renamed stored string-table keyed by index — the very thing it claimed to avoid.
2. **The "Morton-tile stacked-pyramid perturbation-shader cascade" does not exist** (cascade-architect). 0 hits in either repo; Morton is explicitly *rejected* for Hilbert (`linalg/hilbert.rs:50`). Three deliverables (D-OCR-52, the reconstruction round-trip, the whole soa-centroid synthesis plan) were built on a fabricated subsystem name.

**Convergent drift (≥4 lenses):**
- Plans argue "HelixResidue is 48 B, category-wrong, don't use it" — #498 made it **6 B** (a stored `Signed360` place index), which IS the keep-the-index design the plan wanted. Every byte budget was dead (Full 154→112, carve `[32,186)`→`[32,144)`).
- D-OCR-50 (`LayoutBlock::to_node_row`) already SHIPPED in #498 — described as future work.
- §0 tripwires: `ValueSchema::Ocr` (5th preset variant = anti-invention violation; ride Full/Compressed); `Meta` u64 overloaded 5 ways (split → Energy/Plasticity/residues/content-store); `TurbovecResidue` is the *edge* codec (rank-only fidelity) — wrong carrier for glyph→word (use DeepNSM CamCodes).
- HHTL Doc→Page→Block→Line→Token onto HEEL/HIP/TWIG+family is a *coherent address-trie, NOT a Frankenstein* (family-codec + cascade) — but it spends the similarity-basin semantics on layout, so OCR nodes must be `classid`-marked as layout-addressed.

**Disposition.** All 7 plans corrected (rebaselined to #498; Morton purged → real primitives `framebuffer::build_mipmap_pyramid` / `splat3d/depth_cascade` / CAKES; reversibility reframed to identity→content-store + codebook-as-repair-signal; §0 tripwires fixed; master critical-path fixed per the open CodeRabbit Major). Unmeasured claims (int8-exact LSTM, bit-reproducible diff, 200k-LOC 1:1 layout) gated behind 4 probes in `ocr-probes-v1.md` (OCR-RT/DET/POST/SCHEMA); **OCR-SCHEMA shipped as a contract test** proving OCR rides an existing preset (no new `ValueSchema` variant).

**Lesson.** When two PRs touch the same substrate within hours, the later merge silently invalidates the earlier plan's premises. Plans citing sizes/budgets/file:line must be rebaselined the moment a substrate PR lands — and "reversible / never-stored" claims must be PROVEN against the actual decode path before becoming a migration's rationale.

---

## 2026-06-13 — E-TURBOVEC-AMX-WRONG-TOOL-1 — AMX accelerates the operation TurboQuant deliberately removed

**Status:** FINDING (benchmarked; AVX-512+VNNI host, `amx_available=false`).
**Confidence:** High — measured, with a mechanistic explanation that holds across the tier ladder.

**The finding.** turbovec (Google TurboQuant, arXiv 2504.19874) was brought
onto the spine as `crates/lance-graph-turbovec` (excluded standalone, path-deps
the AdaWorldAPI turbovec + ndarray forks). Its scan was *also* expressed as a
batched int8 GEMM through `ndarray::simd::matmul_i8_to_i32` (the polyfill that
ships AMX `TDPBUSD` → AVX-512 VPDPBUSD → AVX-VNNI → scalar). Measured
(`n=20 000, dim=512, k=10, 4-bit`):

| kernel | ns/query | recall@10 |
|---|---|---|
| native nibble-LUT ADC (AVX-512BW) | 76 073 | 0.785 |
| polyfill int8 GEMM (VPDPBUSD-zmm) | 866 899 | 0.764 |
| scalar reference | 6 267 279 | — |

The polyfill GEMM is **11.4× slower** than the native LUT, and native is 82×
faster than scalar. **Mechanism:** TurboQuant's design *trades the matmul away*
— LUT-ADC is an O(1) table gather per coordinate; the GEMM does the full
`dim`-length dot per (query,vector) pair. AMX is a tile *matrix-multiply* unit,
so it accelerates exactly the operation TurboQuant removed. The AMX tile (256
MAC/instr, ~4× VNNI) would bring the polyfill from 11.4× → ~3× slower — still a
loss. **A gather is not a matmul; no tile engine makes it one.**

**Consequences.**
- Keep the native LUT kernel as turbovec's production path. The polyfill is
  retained only as (a) proof the index is `ndarray::simd`-clean / AMX-ready and
  (b) a measured baseline. AMX is the right tool only where the workload is
  genuinely matmul-shaped (e.g. an exact-rerank LEAF over a tiny survivor set).
- Generalises the I-VSA-IDENTITIES register lesson to *kernels*: match the SIMD
  primitive to the algorithm's operation, not to peak MAC/instr. "Ship AMX via
  dispatch" is correct *plumbing* (the polyfill does ship it), but plumbing
  doesn't make the wrong-shaped op fast.
- The genuinely promising turbovec⇄bgz-tensor wiring is NOT AMX: it is a
  Belichtungsmesser σ-gated block reject on the LUT scan (turbovec has only a
  heap-min prune, no statistical threshold). See
  `crates/lance-graph-turbovec/KNOWLEDGE.md` §3B.

Cross-ref: `crates/lance-graph-turbovec/KNOWLEDGE.md` (full synergy map +
reproduce); `ndarray::hpc::amx_matmul::matmul_i8_to_i32` (the 4-tier ladder);
I-NOISE-FLOOR-JIRAK (the σ-threshold path inherits the Jirak obligation).

## 2026-06-12 — E-OUTER-BOUNDARY-IS-ORM-1 — there is only one boundary, and it is ontology-mediated

**Status:** FINDING (PR #487 tombstone commit makes this source-true; OGAR class + `SoaEnvelope` + Lance columnar I/O is the realized triangle).
**Confidence:** High — every prior candidate inner boundary has now been removed or recast as ownership transfer; no surviving call-site asserts otherwise.

**The reframing.** `CollapseGateEmission` looked like a wire format for an
inter-mailbox seam. It was, in fact, the workspace's last hand-written DTO
asserting an **inner** boundary that does not exist. Between mailboxes there is
only ownership transfer (Rust move semantics — `E-CE64-MB-4` makes UB a compile
error); within a mailbox there are only in-place bytes (`SoaEnvelope` geometry).
The only real boundary is the **outer** one — where the SoA meets persistence
and meaning — and that boundary is **ontology-mediated, not DTO-mediated**.

**This is exactly the ORM pattern.**

| ORM | This substrate |
|---|---|
| Table schema | OGAR class (label, fields, tools, templates) |
| Column mapping | `SoaEnvelope` + `ColumnDescriptor` (byte geometry) |
| Active record | register-bank slice wrapped by the class view |
| SQL writer | Lance columnar I/O (writes LE bytes from the in-place store) |
| Hand-rolled row DTO | `CollapseGateEmission` — **the anti-pattern** |

In an ORM you don't write a bespoke struct per table-crossing; the mapping
derives the persisted shape from the schema. Here likewise: the OGAR class
supplies the semantics, the envelope supplies the geometry, Lance does the
writing — and any independent carrier struct at that seam is **schema drift by
definition** (per #477's "every DTO is a derived view of an OGAR class"). The
emission type was not just unused; it was a second, ontology-bypassing
description of data the class layer already described.

**Why `MailboxId` / `MergeMode` / `GateDecision` survive.** They are vocabulary
*of* the ontology side — addressing (which register bank), merge policy (how
overlapping writes compose), gate decision (apply / block / hold). They are
not parallel descriptions of row data; they are the operational verbs the
ontology binds.

**Consequences (the test for any future PR):**

- **Inner seams are ownership transfers**, never carrier types. If a proposed
  type looks like "DTO crossing from mailbox A to mailbox B", it is wrong by
  construction — the SoA is the DTO; the move is the crossing.
- **The outer seam has exactly one description.** OGAR class on one side,
  `SoaEnvelope` on the other, Lance in between. A new "wire format" at this
  seam is the same anti-pattern by a different name — propose a new column or
  a class-template specialization instead.
- **Hand-rolled active-record structs are ORM-bypass.** If you find yourself
  serializing-then-deserializing fields the class already names, the class
  template + `ClassView` + `FieldMask` is the right reach.

**Cross-ref:** PR #477 (three-tier model + "what does NOT exist" table); PR
#487 (tombstone commit — emission artifacts removed; `consume_firing` is the
in-place ownership successor); `docs/architecture/soa-three-tier-model.md`
(register-file analogy + ORM mapping); `E-OGAR-NORTHSTAR-1` (the class spine
this boundary binds to); `I-LEGACY-API-FEATURE-GATED` (legacy carrier paths
must route to the canonical mapping or be removed — the tombstone is removal).
## 2026-06-10 — E-PROBE-MANTISSA-1 — golden-mantissa centroid placement measured: beats uniform-random on coverage AND pile-up; PHASE-1 bit-exactness green

**Status:** FINDING (probes run first-hand: `crates/helix/tests/probe_mantissa_fill.rs`, 4/4 green)
**Confidence:** High — three independent baseline seeds, golden wins all, both metrics, both sample counts.

Wave-0 receipts (per `OGAR/docs/INTEGRATION-TEST-PLAN.md` §1; the
volumetric-field-edge proposal's first gate):

- **PROBE-MANTISSA-FILL GREEN.** Shipped `HemispherePoint::lift` (azimuth
  `n·φ`, equal-area `r=√u`) placing k implicit centroids over a 256×256
  tile (16×16 in-disk bins) vs seeded uniform-random on the same disk
  support: k=256 → golden occupied **192** vs random 141–150, max-bin
  **3** vs 5–6 (≈half the pile-up); k=1024 → occupied **208** vs 205–206,
  max-bin **7** vs **11**; zero empty interior bins (r ≤ 0.9) at k=1024.
  The "golden mantissa places implicit centroids pairwise-uniformly" leg
  of the volumetric/field-edge proposal is now measured, not asserted.
- **PROBE-PHASE-1 GREEN.** `CurveRuler` regeneration is bit-exact across
  independent constructions (20 (path,depth) pairs incl. `u64::MAX`), and
  the stride-4-over-17 arc is a full permutation from every one of the 17
  offsets — the D-QUANTGATE-mandated integer phase walk holds.

Remaining gates before the VolumetricField edge-layout leaves `[H]`:
PROBE-ATTN-EDGE (LUT weight ↔ edge-strength ρ vs Pflug anchors),
PROBE-SPLAT-PSD (EWA Σ composition), PROBE-CASCADE-SPARSITY (HHTL
skip-ratio ≥90% on the volumetric workload). Canon pin: `OGAR/CLAUDE.md`
schema-driven block (PR #51).

## 2026-06-09 — E-MINT-TRACE-1 — the live mint is already global (registry.rs:476); the "namespace-local" doc is stale; dedup is net-new; the bijection IS the dedup

**Status:** FINDING (traced, ratified: `entity_type` = global shared template id)
**Confidence:** High (read the mint, not the doc comment)

**Trace before change paid twice.** (1) `namespace.rs:12` documents `entity_type_id` as "dense **within the namespace**" — but the actual mint is `registry.rs:476 entity_type_id = (rows.len()+1)`: **global append-order across all namespaces**. The doc comment is stale; the GLOBAL semantics DECISION-2/3 want are already the live behavior. (2) It corrected this session's own claim, minutes old: the registry is **not** template-deduped — every append mints a fresh id (`enumerate_first_with_entity_type_id` is defensive, not reuse evidence). Frugal dedup + the `entity_type↔NiblePath` pairing are net-new.

**Blast radius traced benign:** ~16 `entity_type_id()` readers store-as-column-value or compare; none dense-index an array BY entity_type. Global/sparse ids break nothing. Dedup consequence: per-id row lookup becomes namespace-ambiguous ⇒ resolve by `(namespace, entity_type)`.

**The synthesis that shrinks Phase B:** the bijection IS the dedup. One pair table `NiblePath ↔ entity_type` in the registry: path present ⇒ reuse the template id (new row, new namespace); absent ⇒ mint fresh (monotone, never reused) + record the pair. The pair table is simultaneously the template registry, the dedup index, and the bijection witness the round-trip test proves. Moves 1+2 of the Phase B seam are one mechanism.

**Process lesson (generalizes):** doc comments describe intent at write-time; the mint line is the contract. For any "is this id local or global / dense or sparse" question, read the assignment site and grep for dense-indexing consumers before believing prose.

**Cross-ref:** identity-architecture plan DECISION-3 + Phase B grounded seam (CORRECTION block); E-OGAR-NORTHSTAR-1 (Status updated); I-LEGACY-API-FEATURE-GATED (the positional `contract/ontology.rs:85` helper is the v1 path to gate).

## 2026-06-09 — E-ANCESTRY-TRINITY-1 — NiblePath::is_ancestor_of is ONE bit-shift read three ways: subClassOf = supervision-edge = north-star template specialization

**Status:** FINDING (cross-session convergence — OGAR/SurrealDB session + identity-contract session, independently)
**Confidence:** High

**The convergence.** A parallel CCA2A session (OGAR / nexgen op-surreal-ast / SurrealDB RecordId) pulled #480 and independently re-derived the OGAR↔lance-graph membrane as **"the registry mint of `(entity_type, NiblePath)` per class"** — exactly DECISION-2 (OGAR mirror) committed from this side in #481. Two sessions, opposite directions, same membrane.

**The new synthesis it surfaces:** `NiblePath::is_ancestor_of` (a single HHTL bit-shift on the GUID routing prefix) is simultaneously THREE relations:
- **OWL `subClassOf`** (ontology inheritance) — OGAR-AST-CONTRACT §1.
- **OTP supervision edge** (ractor parent-routing / delegation through `OrchestrationBridge`) — the other session's "supervisor-edge is now [G] mechanical" finding.
- **North-star template specialization** (a domain class descends from its shared template) — E-OGAR-NORTHSTAR-1.

They are the SAME relation: the north-star template hierarchy IS the routing/supervision hierarchy IS the subClass hierarchy — one bit-shift, three names. Consequence: reusing a template (inherit + switch namespace), being-supervised-by, and being-a-subclass-of are the same arithmetic; there is no separate routing structure to maintain.

**Coordination:** the OGAR session is on #480 (Phase A); #481 carries the OGAR-side answer it needs — OGAR = OGIT mirror, immutable ClassIds, north-star spine, `namespace`=domain. Its proposed `D-IDENT` paired-note + `D-IDENTITY-PIN` should absorb the `namespace`=domain + north-star framing on next pull.

**Cross-ref:** E-OGAR-NORTHSTAR-1; E-IDENTITY-WHITEBOX-1; identity-architecture DECISION-2 + north-star guard; `hhtl.rs::is_ancestor_of`.

## 2026-06-09 — E-OGAR-NORTHSTAR-1 — ontology cache = OGAR mirror with a reusable north-star template spine (namespace specializes, entity_type is shared)

**Status:** DECISION (OGAR mirror RATIFIED via decision-gate; north-star template model RATIFIED 2026-06-09 "frugal it is"; `entity_type` = GLOBAL shared template id RATIFIED via decision-gate)
**Confidence:** High (both halves ratified; the live mint is global append-order across namespaces)

**Two decisions, one architecture.**

(1) **OGAR mirror (ratified).** The ontology cache's source of truth is OGAR — a one-way mirror of OGIT (+ OWL / Wikidata class-backbone / HHTL) with an append-only immutable ClassId space (protobuf-field-number discipline: mint once, never renumber, tombstone deprecations). Chosen for OWNERSHIP + dissolving the upstream dependency — and, pre-production, immutable ClassIds upgrade NodeGuid from "stable within an OGIT version" to "stable forever." Explicitly NOT a drift fix: content-drift for existing entities does not exist once the cache is mapped from a source (Stefan's correction, twice). The mirror buys ownership, not drift-immunity.

(2) **North-star template spine (recommended model).** The ClassId space is NOT a flat domain×shape explosion. `entity_type`/`NiblePath` is a SHARED, DOLCE-rooted SHAPE template (small spine, reused across domains); `namespace:u8` selects the domain (healthcare / Odoo / WoA-rs / OpenProject-nexgen-rs / OWL / Wikidata). A domain reuses a template by default (switch namespace, inherit the field-set), specializes via NiblePath-descent + FieldMask delta, and mints a new ClassId only for a genuinely novel shape.

**It's the intended reading of the NodeGuid octet split, not new machinery.** `namespace:u8 | entity_type:u16 | kind:u8` already separates domain from shape; `FieldMask + inherit` (parent-OR-delta, class_view.rs) already IS template-reuse-with-delta; `NiblePath::is_ancestor_of` already IS template→specialization ancestry; `dolce_category_id` already roots the spine. The mechanism exists; the curated template ontology + domain→template mappings are the OGAR / Phase B content.

**Frugality is double:** (a) the ClassId space is shape-sized (templates), not domain×shape-sized — fits u16 with room; (b) the shape codebook / palette / shape_hash is encoded ONCE per template and shared 256 ways, and cross-domain alignment is free (same entity_type ⇒ same shape). Reusable templates compose WITH immutability (they ARE the immutable spine) — frugality and stability reinforce, they don't trade off. Per-domain precision is preserved by the FieldMask delta, so "lazy" here is DRY, not sloppy; the only real cost is curation (the template boundaries), which is OGAR's editorial job.

**Phase B becomes:** stand up OGAR as the OGIT mirror + north-star template registry; seed entity_type↔NiblePath from it; the build-time round-trip proves the bijection. The surrealdb-coords blocker (N8 / Phase H) is unrelated and remains.

**Cross-ref:** identity-architecture plan DECISION-2 + the north-star guard; E-IDENTITY-WHITEBOX-1 (NodeGuid composition); I-VSA-IDENTITIES (closed template vocabulary interns; Wikidata's open instance mass stays content, never a ClassId).
## 2026-06-10 — E-WHP-BIPOLAR-1 — bipolar phase makes the perturbation pyramid a Walsh-Hadamard transform on VSA (deterministic, quantum-shaped, classical)

**Status:** FINDING (operator-pinned in `OGAR/CLAUDE.md`; crystallized both sides).
**Confidence:** High that receipts already exist (VSA bind/bundle is the iron-rule
algebra; `Vsa16kF32` is bipolar; helix `CurveRuler` is the bit-exact integer phase
generator); CONJECTURE on synthesis-as-Walsh-pyramid until WHP-1..4 land.

When the deterministic phase from the §6 perturbation pin is made **signed (±1)** —
one bit per (addr, level) — the cascade IS the Walsh-Hadamard transform of the
address tree, carried on the workspace's existing VSA-bipolar algebra: signs compose
by XOR (= `vsa_bind`), magnitudes compose by `vsa_bundle` (Chapman-Kolmogorov-
respecting per `I-SUBSTRATE-MARKOV`). Each cell is a Walsh-resonance superposition
recoverable by role-key unbind; `I-VSA-IDENTITIES` Test 1 (N ≤ √d/4 ≈ 32) IS the
substrate's uncertainty principle. **Roundtrip bit-exact** because phase is
generated from the address, not stored — Walsh-Hadamard is self-inverse up to
scale. "Schrödinger's cat in a glass box": superposition is real, identity
recoverable by key, no measurement randomness.

**TWO-ALGEBRA RULE (load-bearing):** sign = XOR; magnitude = bundle, NEVER
`MergeMode::Xor` (breaks Markov; the named anti-pattern is PP-13 P1-1 "raw-XOR
ordering as 'nearest'"). Sign side preserves the write-back data-flow rule
(single-target gated XOR is allowed); magnitude side preserves Parseval
(L2 conservation → "top gaussian preserved", not Kombinatorik-style selection).

Honest fences: "quantum-like" is the bundling algebra, not measurement randomness
(no headline drift); bipolar = 1-bit phase (multi-bit stacks above when measured to
be needed); Parseval requires the bundle, not just XOR. Probes WHP-1..4 land before
any L2-conservation claim ships. Full treatment: ndarray
`guid-prefix-shape-routing.md` §4b; policy mirror:
`guid-canon-and-prefix-routing.md` §7; canon: `OGAR/CLAUDE.md`.

## 2026-06-10 — E-CANON-GUID-1 — the canonical GUID's dash-groups are the cascade; routing/quorum crystallized before dilution

**Status:** FINDING (canon operator-pinned in `OGAR/CLAUDE.md`; crystallization docs landed both sides)
**Confidence:** High on the canon + receipts; CONJECTURE (probes named) on the new surfaces

The operator-pinned canonical identity is HEX-counted — it IS the GUID:
`classid(8)-HEEL(4)-HIP(4)-TWIG(4)-[basin·leaf(6)+identity(6)]`; the UUID
dash-groups ARE the cascade delimiters. Key-of-key-value: node = key(128) +
value(3968) = 4096 bits — the key routes/resolves/compares/scopes/names with
zero value decode; Lance compresses the value freely (compression never costs
addressability). 3×4 uniform tiers (`tier = nibble >> 2`); RFC 9562 = wrapper
concern (wrappers adapt to canon, never the reverse). Centroid-tile reading
[H]: path = 6 bytes = CAM-PQ 6×256; per-class codebooks scoped by class
routing prefix (longest-prefix wins), 4⁴-hierarchical so nibble prefixes =
centroid ancestry. Crystallized: this repo's policy side at
`.claude/knowledge/guid-canon-and-prefix-routing.md`; ndarray's mechanism
side at `ndarray/.claude/knowledge/guid-prefix-shape-routing.md` (the
`PrefixShapeTable` + φ-quorum anti-eigenvalue-theater contract, with the
PP-13 casebook as the named failure catalog). `contract::quorum` (#411
scaffold) is the named landing spot for the quorum certificate. Probes:
ROUTE-1, QUORUM-1, PHI-1, PYR-1, CODEBOOK-44, HILBERT-L4 (blocker).

## 2026-06-09 — E-IDENTITY-WHITEBOX-1 — structured identity + round-trip converts the substrate from black-box to CI-falsifiable

**Status:** FINDING (Phase A landed: `identity::NodeGuid` composed, 15 tests green)
**Confidence:** High

**The synthesis:** a structured 128-bit immutable identity (UUIDv8 = the HHTL
nibble-address formalized + namespaced) PLUS the `roundtrip_eq` guarantee turn the
substrate from a black box into a CI-falsifiable surface. Two structural witnesses:
the **bijection** (`entity_type ↔ NiblePath`, proven eineindeutig at build time)
and **`roundtrip_eq`** (a lossy projection fails CI, not code review). The
round-trip whitens the PLUMBING (identity / LE byte-contract / member-vs-container)
— the darkest, most-expensive-bug layer; it does NOT whiten semantics (needs
ground-truth corpora) or the lossy codec (needs ρ-certification). Those keep their
own witnesses; the trade is "vigilance over the substrate" → "a test over the substrate".

**Compose, don't re-invent (Agent A sweep):** the 128-bit identity space is empty
(no committed `u128`/`Uuid`/`[u8;16]`-as-id), but every GUID field already exists as
a committed scalar — `SchemaPtr.packed` (ns8|entity_type16|kind8) ⊕ `NiblePath`
prefix ⊕ truncated `StructuralSignature` ⊕ local. `NodeGuid` is their composition.

**Eineindeutigkeit (ratified):** `entity_type:u16` is the canonical exact class
identity; `NiblePath` is the bijective DERIVED view (a *truncated* prefix can't be
the identity — deep classes collide past it, `hhtl.rs`). The registry mints the
pair 1:1; a build-time bijection round-trip proves it; the GUID prefix-consistency
invariant (`prefix == niblepath_of(entity_type)[..N]`) catches drift.

**Free side-effects:** the structured GUID is also (a) a KV key via the existing
`EntityKey(&[u8])` (smb-bridge already length-branches it), and (b) a **quadkey** —
Lance ORDER BY the NiblePath gives subtree range-scans = zone-map-pruned byte
ranges, no index (ADR-024 "HHTL prefix establishes a frame").

**Landed:** `lance_graph_contract::identity::NodeGuid` (D-IDENTITY-1 / Phase A) +
`NiblePath::from_packed`. 599 contract lib tests (+15), clippy `-D` clean, fmt clean.
Plans: `identity-architecture-exists-vs-needs-v1.md` (exists-vs-needs map),
`cognitive-write-roundtrip-substrate-v1.md` (the round-trip mechanism).

## 2026-06-06 — E-DEINTERLACE-TWO-SCALES — deinterlace is one operation at two scales; no-cross-cycle-lag = byte-scale deinterlace

**Status:** FINDING (source-grounded; `temporal.rs` PR #468 confirms row-scale; byte-scale is a documented gap)
**Confidence:** High

**The synthesis:** temporal causality in the SoA system must be enforced at two
independent scales that share the same monotonic clock:

```text
Row/query scale  →  HLC tick + DependsClosure  →  temporal.rs::deinterlace()  (SHIPPED, PR #468)
Byte/column scale → SoaEnvelope::cycle() stamp → MailboxSoA Arc-swap COW      (GAP — plan written)
```

Both are the SAME operation — "sort by the causal clock and project the result
into the reader's reference frame" — but at different granularities.

**Row scale (PR #468 confirms):**
`temporal.rs:18-20` defines the standing wave correctly: "merge-sort by HLC
tick and every field's row lands on one timeline. The result IS the standing
wave / kanban SoA." The `deinterlace()` function + `EpistemicMode` (Strict /
Aware / Retro) + `DependsClosure` implement this. 8 tests pass.

**Byte scale (current gap):**
Nothing in the codebase prevents a reader from holding column data from SoA
cycle N and cycle N+1 in the same SIMD sweep. The `SoaEnvelope::cycle()` stamp
exists but is not enforced as a snapshot barrier.

**The fix (plan: `cycle-coherent-soa-snapshot-v1.md`):**
Arc-swap COW at column granularity in `MailboxSoa::advance_phase`:
1. Writer increments `cycle`, then swaps the `Arc<[u8]>` of each mutated column.
2. Reader snapshots all column Arcs under one cycle stamp (lock-free retry).
3. The resulting `MailboxSoaSnapshot { cycle, cols }` is structurally coherent.

**The boundary:**
`MultiLaneColumn` in ndarray stays layout-only. The Arc-swap policy lives in
lance-graph's `MailboxSoa`. ndarray does not learn that cycles exist.

**The clock is one clock:**
`SoaEnvelope::cycle()` (byte scale) and `QueryReference::ref_version` (row
scale) are the same monotonic sequence. Threading `snapshot.cycle` into
`QueryReference` closes the loop: row-scale and byte-scale deinterlace use
the same clock.

**Standing wave clarification (Q3 probe result):**
The "standing wave" is NOT a compute recurrence. It is the deinterlaced
projection over Lance versions — provided by Lance versioning itself (O(1)
90° lookup). Do not implement a standing wave in compute.

**Cross-ref:**
- PR #468 (`temporal.rs`) — row-scale (SHIPPED)
- PR #477 (`soa_envelope.rs`) — envelope contract (IN REVIEW)
- `.claude/plans/cycle-coherent-soa-snapshot-v1.md` — byte-scale fix plan
- `docs/probes/q3-standing-wave-falsification.md` — probe confirming no wave in compute

---

## 2026-06-04 — E-AUDIT-RETENTION-CAVEAT — substrate-b consumer doc Lance-versions-as-audit claim was overstated; corrected to retention-policy-gated (codex P1 on #465)

**Status:** CORRECTION (codex P1 on PR #465, 2026-06-04; merged + immediate follow-up correction per the no-silent-edit discipline — the FIX appends; the original epiphany E-SUBSTRATE-B-CAPABILITY-ROADMAP stands as the corrected reference now reads).

**The overclaim (now corrected in `.claude/knowledge/old-stack-capability-parity.md`):** §2.1 said *"Immutable audit = append-only by construction — versions never disappear; the log IS the audit trail."* §5.1 followed up with *"Three OLD components collapse to one ... consumers should NOT introduce separate stores."*

**The reality codex caught:** Lance 7.0+ exposes `Dataset::cleanup_old_versions` and `lance.auto_cleanup.*` settings. Old versions CAN be removed for storage management — the version log is therefore **not guaranteed immutable without explicit retention policy**. Consumers following the doc's guidance to drop their separate audit store could see historical audit reads disappear after cleanup.

**The corrected framing:**
1. **Audit is retention-policy-gated**, not by-construction-immutable. For audit-class workloads, retention must be configured (disable auto-cleanup, tag versions, OR route to a separate append-only sink).
2. **Regulatory-grade audit** ("cannot be deleted, cannot be manipulated") requires a separate signed write-once sink — substrate-b doesn't claim to replace it.
3. **The collapse is two-and-a-half, not three.** Historisation + TSDB collapse outright; audit is conditional on retention policy + workload class (non-regulatory: yes with retention; regulatory: no, external sink still required).

**Why this matters for the substrate-b shape:** the three-primitives codification (E-SUBSTRATE-B-CAPABILITY-ROADMAP) survives — the multi-purpose-Lance-versions claim is still load-bearing. What changes is the audit guarantee + the consumer-guidance default ("introduce no separate store"): now reads "introduce no separate store *for non-regulatory audit, with retention configured*; regulatory audit remains a separate concern."

**Cross-ref:** PR #465 (merged) + the follow-up correction PR; `.claude/knowledge/old-stack-capability-parity.md` §2.1 + §5.1 (corrected); codex P1 finding (audit retention outside prunable Lance versions).

---

## 2026-06-04 — E-SUBSTRATE-B-CAPABILITY-ROADMAP — three load-bearing NEW-stack primitives codified; consumer integration shape documented

**Status:** FINDING (substrate-b consumer integration pattern, codified after the OGAR / surrealdb / ractor / lance-graph correspondence work converged on three structural primitives, 2026-06-04).

**Three NEW-stack primitives substrate-b consumers must internalise** (now codified in `.claude/knowledge/old-stack-capability-parity.md`):

1. **Lance versions are a multi-purpose primitive.** One primitive serves three capabilities a substrate-b consumer would otherwise build separately: `checkout_version(V)` = point-in-time query (Historisation); the version log = time-series; append-only immutability = signed audit. Consumers should NOT introduce separate stores for these three.

2. **Per-element auth = palette256 + Hamming popcount on `Binary16K`.** The hot-path auth primitive is bit-op-per-element via the per-vertex `_effectiveReaders` / `_effectiveDevices` bitmap. Materialised on write; checked on read via Hamming popcount / bit-intersection; uncached / immediate-effect by construction. ACL changes at version V are in effect at every read at version >= V; consumers should NOT introduce an auth cache.

3. **ractor Actor + Lance-version-as-state-machine = the Rubicon phase machine.** A substrate-b actor models its lifecycle as a typed state enum on a ractor `Actor`; state-enter side-effect fires the Lance commit at the Decision state; events arriving before Decision are deferred; per-state timeouts route through `MessagingErr::Saturated`. The actor's state history IS the Lance version log on its dataset; no separate state-machine event store needed.

**Two consumer-side patterns** that fall out of these primitives:
- The `LanceVersionWatcher` (in-proc event bus) uses `std::sync` per the I-2 invariant — tokio is reserved for Layer-3 outbound sinks. A consumer that wires tokio for in-process subscription violates I-2 and reproduces the bug `version_watcher.rs`'s 2026-05-06 plan correction note already records as fixed upstream. This is a `hollow-wire-failure-modes.md` failure-mode magnet.
- The migration endpoint contract (`POST /v1/{entity,edge,traverse,query,graphql,audit}` + `WS /v1/stream` + `POST /v1/dispatch`) is the substrate-b dual-stack ground-truth surface — same workload replayed against substrate-b AND the system being replaced; §14 acceptance gate produces a per-endpoint verdict.

**Cross-ref:** `.claude/knowledge/old-stack-capability-parity.md` (new); `.claude/knowledge/lab-vs-canonical-surface.md` (companion); `.claude/knowledge/hollow-wire-failure-modes.md` (companion); `AdaWorldAPI/lance-graph#452/#453/#454/#455/#456/#457/#458` (merged contributions); `AdaWorldAPI/surrealdb#35/#36` (kv-lance feature + Lance backend struct); `AdaWorldAPI/ractor#1` (`MessagingErr::Saturated`); `AdaWorldAPI/OGAR#5/#6/#7/#8` (carrier shipping).

---

## 2026-06-03 — E-HELIX-NDARRAY-MANDATORY — `helix` ndarray wiring: optional `path` → mandatory `git` (codex P2 + "ndarray is mandatory") — an optional path dep is a clean-checkout trap

**Status:** FINDING (codex P2 on #460 + user directive, 2026-06-03; fix verified — 63 unit + 6 doctests green with mandatory ndarray, clippy -D warnings + fmt clean; the git source was patched to the local `master` checkout for the in-sandbox build, github fetch deferred to CI).

**The trap (codex P2):** an *optional* `path` dependency does NOT make the default build self-contained. Cargo reads every dependency manifest (including optional ones) during resolution to build the lockfile, so `ndarray = { path = "../../../ndarray", optional = true }` makes a clean checkout WITHOUT the sibling fail (`failed to read .../ndarray/Cargo.toml`) *before* feature selection. The "default build needs no ndarray checkout" claim was therefore false.

**The fix (two directives converge):** (1) codex wants the wiring clean; (2) the user — "ndarray is mandatory for lance-graph" (it is "The Foundation"). So helix now takes ndarray as a **mandatory, non-optional git dependency**: `ndarray = { git = "https://github.com/AdaWorldAPI/ndarray.git", branch = "master", default-features = false, features = ["std"] }`. A git source resolves the manifest remotely (no sibling-checkout needed); non-optional drops the `ndarray-hpc` feature entirely; `simd.rs` is now single-impl (always `ndarray::simd`, no scalar fallback — ndarray does its own AVX-512/AVX2/scalar dispatch internally).

**Why git, not `[patch]`, and no cycle:** helix is standalone (own `[workspace]`, root `exclude`), so it resolves ndarray independently of the lance-graph workspace's path-based ndarray — no source-unification needed (the workspace's `[patch.crates-io] ndarray` is separately known-ineffective: the fork's 0.17.2 can't semver-satisfy the lance-index crates.io 0.16.1; PR_ARC ~line 2081). The fork is self-contained — only internal subcrate path deps (`crates/p64`, `crates/fractal`, `ndarray-rand`, `crates/ndarray-gen`) which travel with the clone — and has **no back-dependency on lance-graph**, so the git dep introduces **no import cycle**.

Cross-ref: PR_ARC #459 Correction; codex P2 #460; `crates/helix/Cargo.toml`; `jc::weyl` (local-const precedent).

---

## 2026-06-03 — E-HELIX-OVERLAP — the `helix` Place/Residue codec is ~80% re-derivation of existing (in places CERTIFIED) primitives; shipped standalone by user directive, overlap documented not hidden

**Status:** FINDING (placement check via `encoding-ecosystem.md` + repo grep, 2026-06-03; crate shipped on claude/gallant-rubin-Y9pQd — 61 unit + 6 doctests green on the default zero-dep build AND under `--features ndarray-hpc`). User directive: "create crate crates/helix … scoped only to crate, self resolving" → after the overlap was surfaced via `AskUserQuestion`, the user ratified **Standalone helix** (vs compose-existing vs rename).

**The overlap (don't-reinvent ledger):** the Fisher-Z/arctanh→i8 quantiser already exists as `bgz-tensor::projection::Base17Fz` + `bgz-tensor::fisher_z::FamilyGamma` (CERTIFIED ρ≥0.999, 21 roles, 1.7B); the golden-spiral azimuth proof as `jc::weyl::prove()` (1-D, Ostrowski 2/N); stride-4 coupling in `thinking-engine::reencode_safety` + `highheelbgz`; the EULER_GAMMA hand-off in `jc::precond` + `bgz-tensor::euler_fold`; 256-palette/L1 endpoints in `bgz17::palette`. **Genuinely new:** the equal-area `√u` hemisphere placement (`r=√u`, `Y=√(1−r²)`) and the PLACE/RESIDUE doctrine (zero prior hits). Consolidation path (when helix graduates from clean-room): re-export `FamilyGamma` behind a feature, route coupling through the canonical `(i·11)%17`/stride-4 zipper, feed `ResidueEdge` endpoints into the existing HIP/TWIG CAKES tier. Tracked as TD-HELIX-OVERLAP-1.

**Two corrections folded in:** (1) `KNOWLEDGE.md` referenced `const::simd::{GOLDEN_RATIO,EULER_GAMMA,E}` — that path does NOT exist; canonical is `std::f64::consts` (ndarray does not wrap them). helix defines local consts (`E` from `std` since it is stable; `GOLDEN_RATIO`/`EULER_GAMMA` as literals since not stable on every toolchain, mirroring `jc::weyl`'s local `PHI_INV`). (2) The name `helix` collides with a planner plan-doc "Helix" (Foundry time-series histogram) that leaned *away* from a crate — `crates/helix` is free, but flagged. **Iron rule respected:** stride-4-over-17 is coprime → full permutation; the banned Fibonacci-mod-17 (misses {6,7,10,11}) is NOT used. **Open (CONJECTURE, probe NOT RUN):** helix int8 endpoint fidelity vs the naive-u8 floor gate (≥0.9980 Pearson) — a fidelity-vs-ground-truth probe is owed before promotion past clean-room status.

Cross-ref: `crates/helix/KNOWLEDGE.md` (§ Overlap & Consolidation), `encoding-ecosystem.md`, `jc::weyl`, `bgz-tensor::{projection,fisher_z}`, TD-HELIX-OVERLAP-1, the `E-READ-NOT-GREP` consult-before-guess doctrine.

---

## 2026-06-01 — E-NARS-FIGURE-CAPSTONE — NAL syllogism resolution hardwired on CausalEdge64 like Pearl 2³: the figure = which SPO palette term two edges share → rule → conclusion edge

**Status:** SHIPPED (`causal-edge::syllogism`, branch claude/jolly-cori-clnf9; 14 tests v2 / 13 v1, the new file clippy- + fmt-clean). User: "the syllogism resolution needs to be hardwired similar to SPO 2^3 rung decomposition … using causaledge64 and wiring EW64" + "NAL syllogism notation is the missing capstone for glueing all 3 reasoning methods with the 10-rung ladder and the JITson cranelift templates vs elixir."

**The capstone:** `CausalEdge64::forward()` composes two edges POSITIONALLY with a *pre-set* inference type — it never asks WHICH TERM they share. NAL syllogism IS that question. `figure(other)` resolves it by integer SPO-palette equality — the hardwired analogue of the Pearl 2³ mask (O(1), branch-minimal, no float on the structural path):
- `o1==s2` → Chain → **Deduction** ⊢ s1→o2
- `s1==o2` → ChainRev → **Deduction** ⊢ s2→o1
- `s1==s2` → SharedSubject → **Induction** ⊢ o1→o2
- `o1==o2` → SharedObject → **Abduction** ⊢ s1→s2
- same statement (`s1==s2 ∧ o1==o2`) → None (that is Revision, not a syllogism).
`syllogize(other)` emits the conclusion `CausalEdge64` (outer terms + canonical NARS truth + signed v2 mantissa Ded +1 / Ind +2 / Abd −1 + AND-ed Pearl mask). Firewall: integer term-match PROPOSES the figure; the deterministic NARS truth-function ADDRESSES. Truth math is byte-identical to `ndarray::hpc::nars` (hardware) + `forward()` (protocol) — NOT a new truth type.

**The "don't reinvent" catch (E-READ-NOT-GREP in action):** user flagged "we have 34+ opennars vocabulary, it just needs to be wired." Reading-first found it all already present — vocabulary (`cognitive_codebook::{NarsInference 10, NarsCopula 12}` with the `{M-->P,S-->M}` notation in comments + `CognitiveAddress` + fingerprints), the tested engine (`planner::nars_engine::nars_infer`, 9 rules / 17 tests), the canonical truth (`ndarray::hpc::nars::NarsTruth` + 7 truth-fns), the atoms (`atoms.rs` Operation lane: abduct/deduce/induce/synthesize), the wire (`causal_edge::InferenceType` signed mantissa). The speculative deduction/induction/abduction I had started adding to `contract::exploration::NarsTruth` were a 3rd copy AND mislabeled (induction⇄abduction swapped vs the canonical engine) — **REVERTED.** The gap was never the truth math; it was the FIGURE decomposition. Next (gated): wire EW64 `EdgeRef`→`CausalEdge64`→`syllogize` across the ≤4 hot edges in cognitive-shader-driver; cranelift/elixir dual-compile of the figure table.

Cross-ref: `E-READ-NOT-GREP`, the firewall doctrine, CausalEdge64 v2 mantissa, `episodic_edges` (EW64), `atoms` Operation family, the Pearl 2³ mask.

---

## 2026-06-01 — E-RESEARCH-COUNCIL-PROPOSE-VALIDATE — 8 LLM/float semantics papers, council-firewall-filtered: the "PROPOSE (float, offline) / ADDRESS (integer, hot)" doctrine independently re-derived from 7 of 8; 1 ADOPT-NOW, 3 integer validators, 1 adversarial foundation-probe

**Status:** FINDING (5-agent research council, 2026-06-01; ALL papers READ IN FULL per `E-READ-NOT-GREP` — A1 read 1311 lines / A5 read 1517+1695, via Read not grep). Doc: `research-council-semantics-papers-2026-06.md`.

**The corroboration:** all 8 papers are LLM/float-based; the council extracted only the deterministic/integer/offline kernel from 7 and cleanly SKIP'd the 1 with no quarantine seam (segmentation = unconditional firewall trap, float-all-the-way-down). **The workspace's "similarity PROPOSES (float, offline, upstream) / CAM ADDRESSES (integer, hot, deterministic)" doctrine was independently re-derived by 4 of 5 reviewers from 7 different papers** — strong external evidence the core architecture is right.

**The slate (firewall-filtered):**
- **ADOPT-NOW (offline-buildable):** SemDiD (2506.23601) → a `head2head::WinnerCriterion::Repulsion` — repulsion-from-nearest-rival (VSA-overlap, cosine→Hamming) + ε-quality-floor + harmonic combiner; training-free, integer, self-contained in `contract::head2head`. *(A4)*
- **Shared operator (A4):** "retain a candidate iff its hypervector is far from incumbents under a quality floor" — mount on head2head (arbitration, SemDiD) AND EW64 admission (LaMAR's "novelty beats volume" — refines the just-merged `promote`/`coldest`).
- **3 integer VALIDATORS:** Legality Score (A1, prime-reduction purity, deepnsm-side, offline) · `⟨u,v⟩` cognitive-load (A2, CAM-PQ similarity self-explanation) · footprint `{→,←,∥,#}` (A5, aerial→DOLCE ordering validator).
- **ADVERSARIAL PROBE (highest-value foundation test):** Kozlowski (2508.10003) — non-orthogonality is *signal*; the hard 4096-basin + ρ=0.9973 CAM-PQ partition **may discard the entangled low-rank semantics**. Distinction (A3): role-disjointness (binding) is fine; *content* orthogonality is the challenge. Runnable test: antonym-direction interference vs CAM-PQ mis-addressing; pair with SAFARI's Weyl Semantic-Shift auditor over OGIT/DOLCE (tests #444 98.6%).
- **SKIP:** segmentation (2412.08671) — firewall trap.

**Process:** the council read full text (no grep/head/tail) per `E-READ-NOT-GREP`; the reading-path TEST caught a missing-poppler blocker before dispatch (extracted PDFs→.txt via pymupdf). Cross-ref: `E-READ-NOT-GREP`, the firewall doctrine, `head2head`, `episodic_edges`, DeepNSM.

---

## 2026-06-01 — E-READ-NOT-GREP — judgment-critical review agents must READ full files, not grep/sed/head/tail; fragments invalidate judgment

**Status:** IRON RULE (process; user-stated 2026-06-01). Across the EW64 council + the 3 prior relayed sessions, every wrong framing came from narrating off grep/head fragments rather than reading the type. The council's R3 even found a grep-induced mis-citation (`edge.rs:750 concern_level` reads `direction()`, not `PlasticityState`). **RULE:** when a review/council/grounding agent's VERDICT depends on a type's semantics, its brief MUST instruct it to READ the relevant files in full (the Read tool), NOT grep/sed/head/tail. A fragment seen out of context produces a confident-but-wrong judgment — **grep is for LOCATING, reading is for JUDGING.** Baked into the agent-brief template (`autoattended-multiagent-pattern.md` §Rule 7). Cross-ref: `E-BASIN-NOT-EDGE-PLASTICITY` (the conflation grep-fragments produced).

---

## 2026-06-01 — E-BASIN-NOT-EDGE-PLASTICITY — the 4th-strike object conflation: per-basin `Heel.plasticity` (a NARS-confidence COOLING knob, not on the EW64 hot path) is NOT coarse edge-plasticity; the Hebbian edge weight is per-plane `PlasticityState` (gated) or the MRU slot-order (shipped)

**Status:** FINDING (5-agent council-resolved + orchestrator source-verified, 2026-06-01). R4(critic) + my own full-file reads killed the "compose `Heel.plasticity` × MRU" resolution that spec §6/§8 (relayed from 2 prior sessions) walked into. **Verified by reading the source:** (a) `MailboxSoaView` (the EW64 hot path) has **NO** Heel/plasticity column (only `energy`/`edges_raw`/`meta_raw`/`entity_type`) — Heel is unreachable from the EW64 edges; (b) `Heel.plasticity`'s only writer is `revise_truth()`, which **COOLS** as NARS confidence rises (`high_heel.rs:252` "Cool plasticity as confidence rises"; `is_frozen = plasticity==0 && conf>0.8`) — **opposite polarity** to Hebbian fire→hot; (c) `HighHeelBGZ.edges` are `CausalEdge64` u64s, EW64 slots are `EdgeRef` — different encodings, no index map. So "compose Heel × MRU" is a **phantom join of anti-correlated signals on the wrong edge set.** **RESOLUTION:** coarse strength = the MRU slot-order (#447, shipped); the real per-edge Hebbian weight = per-plane `PlasticityState` co-fire (GATED, phase B); no Heel, no new field. **The 4th strike** (after CausalEdge64-lens / per-plane-axis / Heel-vs-PlasticityState): same-word-different-**OBJECT** — "plasticity" names BOTH a cold-path basin cooling knob AND a hot-path per-edge Hebbian state; they don't compose. Cross-ref: D-ATOM-4/RawEdge (shipped), spec §9, `E-READ-NOT-GREP`.

---

## 2026-06-01 — E-EW64-STRENGTH CORRECTION — "W15 0..3 plasticity" is `high_heel::Heel` (128-byte container field), NOT the 64-bit `CausalEdge64`; the 64-bit edge's plasticity is the 3-bit-per-plane `PlasticityState`

**Status:** CORRECTION (factual, Plan-agent-grounded against source 2026-06-01). Refines `E-EW64-STRENGTH-IS-CE64-PLASTICITY`'s mechanism claim; does NOT change D-EW64-2 (MRU slot-order strength stores NO plasticity — it stands regardless).

The original said "strength = co-addressed `CausalEdge64` plasticity (W15 0=frozen..3=hot)." Source check: the W15 byte-3 scalar 0..3 is `high_heel::Heel` (`high_heel.rs:168`, a **128-byte container field**), NOT the 64-bit `CausalEdge64`. The 64-bit edge's plasticity is `causal-edge::PlasticityState` (`plasticity.rs`) = **3 bits, one per S/P/O plane** (heat/freeze/hot_count), not a 0..3 scalar. "Co-addressed CE64 plasticity" must pick ONE model — a **USER decision**. Consequence: the **plasticity-WRITE co-fire op is GATED** on 3 counts — (1) `causal-edge` doesn't build offline (anstream uncached); (2) the Heel-scalar-vs-PlasticityState-per-plane mismatch needs user design; (3) it hits `I-LEGACY-API-FEATURE-GATED` (the v1 PLAST_SHIFT=49 vs v2=50 boundary codex caught 5× in sprint-11). D-EW64-2's MRU (slot-order = recency, no stored weight) is unaffected. The Hebbian "wire together" weight-bump waits on the user's plasticity-model decision. Cross-ref: D-EW64-2/3/4, `causal-edge::edge.rs:471/483` (v2-gated getter/setter), `I-LEGACY-API-FEATURE-GATED`.

---

## 2026-05-31 — E-EW64-STRENGTH-IS-CE64-PLASTICITY — EW64's "stronger immediate edges" need no new field (strength = co-addressed CE64 plasticity + MRU slot-order); the surreal LIVE wingman is the designed orchestrator but GATED + optional

**Status:** FINDING (design resolution, register-lazy). User-stated 2026-05-31 ("episodicwitness64 needs the stronger immediate edges; wingman orchestration in surrealdb in same substrate is an option"). Resolves the strength gap in `EpisodicEdges64`; weighs the surreal-same-substrate option. Feeds the queued spec `episodic-witness64-ce64-prefetch.md`.

**The gap (grounded):** `EpisodicEdges64` = 4×`EdgeRef{family:u8, local:u16}` (`episodic_edges.rs`) — NO strength / weight / recency field. "Stronger immediate edges" has no home today.

**The register-lazy resolution (no 16-bit EdgeRef change):** EW64 already **shares CE64 low-40 bits (co-address)** — each EW64 edge co-addresses a `CausalEdge64` whose **plasticity** (`high_heel` W15 `0=frozen..3=hot`; v2 plasticity[2]) IS the edge strength. Two complementary orderings, both free:
- **strength = co-addressed CE64 plasticity** — the Hebbian weight is already there; EW64 inherits it by co-address.
- **the 4 slots = an MRU hot set** — slot 0 = strongest/most-immediate; fire → promote to slot 0 (strengthen); age → demote → evict slot 3 to the cold connectome. Slot ORDER is the ranking. No new field.

**The wingman = the surreal LIVE scheduler (already `E-SUBSTRATE-IS-THE-SCHEDULER`):** witness fires → Lance append → surreal LIVE → promote the EW64 edge + prefetch the next into the SoA; the mailbox runs no planner loop. "Same substrate" = SurrealDB holds the **cold connectome** (all edges as graph RELATE + strength/recency) AND orchestrates the hot EW64 refresh (LIVE). The **hot 4-edge EW64 stays in the SoA** (resident, deterministic, zero-copy); SurrealDB is the cold+reactive tier, never the hot compute.

**The option, weighed:** surreal-same-substrate is the DESIGNED goal and the right fit for cold+wingman — but GATED on `surreal_container` (OQ-11.6: fork dep + Lance 6 pin) and OPTIONAL: the design is substrate-free, so a **LanceDB-LIVE trigger is the fallback** (surreal is the goal, not a dependency). Adopting surreal is a free upgrade when OQ-11.6 clears, not a blocker.

**Honest state + the unblocked next:** this is the `E-ARIGRAPH-IS-AN-ISLAND` gap — EW64/`SpoWitness64` = 0 code symbols; the Lance→surreal→kanban subscription unbuilt; `HotWitness` = `todo!()`. The surreal side is blocked (OQ-11.6). The UNBLOCKED, firewall-clean, offline-testable next = the **contract-side EW64 strength/ordering atom**: the hot-4 MRU semantics + CE64-plasticity-mirror strength on `EpisodicEdges64` (no fork), deferring the surreal wingman to wire when OQ-11.6 clears. **Firewall:** EW64 stores opaque `(family,local)` + co-addressed CE64; surreal would store opaque handles + strength — never COCA. Cross-ref: `E-SUBSTRATE-IS-THE-SCHEDULER`, `E-EW64-IS-PREDICTIVE-PREFETCH`, `E-PLANNING-IS-WHITE-MATTER`, `E-ARIGRAPH-IS-AN-ISLAND`, `episodic-witness64-ce64-prefetch.md` (queued spec), OQ-11.6.

---

## 2026-05-31 — E-PLANNING-IS-WHITE-MATTER — the 64k mailboxes are GREY matter (compute); planning lives in the WHITE matter (the CE64/EW64 plasticity connectome), not in OTP/BEAM scheduling

**Status:** FINDING (architecture reframe; unifies existing Hebbian/plasticity findings under the grey/white lens — the *mechanisms* are already on the board, the *framing* of planning is the new part). User-stated 2026-05-31 ("it doesn't make sense to have 64k OTP BEAM Erlang multithreading when you don't recognize the potential as grey vs white matter and BNN what fires together wires together"). Extends the language-network map (`E-ARCUATE-CONDUCTION`) into the cognitive substrate. Answers: "what can the mailbox SoA do about planning."

**Grey vs white:**
- **Grey matter (compute / neurons)** = the 64k mailboxes (per-mailbox SoA: Fingerprint/Qualia/Meta columns + the `Think` compute) AND the PFC executive (`lance-graph-planner`: MUL / elevation / strategies — goal-setting + suppression).
- **White matter (connectome / axonal tracts)** = the CE64 (causal) + EW64 (episodic) EDGE columns + **plasticity** (`high_heel` W15 u8 `0=frozen..3=hot`; v2 `CausalEdge64` plasticity[2]; `sensorium.plasticity_flux`). `arcuate.rs` is the first explicit *named* tract.

**Planning is a white-matter phenomenon, not OTP scheduling:**
- A plan = a trajectory through the mailbox population. The white matter encodes which trajectories are **myelinated** (high plasticity = well-worn = automatic).
- "Fire together → wire together" (already on board: `E-EW64-IS-PREDICTIVE-PREFETCH`, `plasticity_counters`, the prefetch spine): executing a path increments edge plasticity → consolidates it into **procedural memory** (a habit/skill).
- Planning = bias toward myelinated paths (exploitation) + the **spreader** recruiting adjacent low-plasticity edges when the goal isn't reached (exploration, OQ-11.1/§11.5) + **prefetch** making the next step resident before it's asked. NOT a DAG computed by `KanbanMove`/`VersionScheduler`.
- **Reframe:** `KanbanMove` / `VersionScheduler` / ractor = grey-matter process COORDINATION (necessary plumbing), **not the planner**. The planner IS the plasticity-weighted EW64/CE64 connectome, under PFC (MUL) bias + `head2head::SupportSpread` action-selection.

**Why "64k OTP/BEAM concurrency" misses it:** Erlang treats processes as isolated units with explicit message passing (supervision trees). A BNN treats them as a connected population where the CONNECTIONS carry the computation. The brain is mostly white matter; a 64k-grey-node system with a thin connectome can compute in parallel but can't PLAN — planning lives in the wiring. The lever is the connectome (EW64/plasticity), not more grey-matter concurrency.

**Honest state (the mechanism is DESIGN, not built):** A3 `witness_arc` MISSING; the Hebbian spreader radius/decay TBD (OQ-11.1, D-MBX-A4); `plasticity_counters` described not built; the prefetch spine = the unbuilt EW64 reactive seam ("every link shipped, the chain open at the joints"). The grey/white lens UNIFIES these and reframes the planner; the **buildable seam = the plasticity update + spread on the SoA EdgeColumn** (white-matter growth). Cross-ref: `E-EW64-IS-PREDICTIVE-PREFETCH`, `E-ARCUATE-CONDUCTION` (first tract), §11.5 plasticity-spreaders, OQ-11.1/11.2, `head2head`, `sensorium.plasticity_flux`, `high_heel` W15.

---

## 2026-05-31 — E-ARCUATE-CONDUCTION — the stack has conduction aphasia: Broca+Wernicke intact, the arcuate cable (disambiguator_glue) carries no signal (the producer gap) — closing it IS the next wire

**Status:** FINDING (diagnosis, grounded in source). Extends `E-BROCA-WERNICKE-HIPPO` to the full distributed language network (doc § "the full language network"). Names the single highest-value wire.

**The diagnosis:** `disambiguator_glue` IS the **arcuate fasciculus** — the Broca↔Wernicke cable (`Trajectory` → contract `context_chain`, `disambiguator_glue.rs:65`). It is *shipped*. But `MarkovBundler::push` is never called by `pipeline.rs`, so no `Trajectory` is produced → the cable carries no signal. Broca (projection: `parser`→SPO + `markov_bundle`) and Wernicke (comprehension: `comprehension.rs` + COCA similarity) each work in isolation; only the connection between them is dead. **Clinical signature matches conduction aphasia exactly:** production + comprehension intact, *repetition* (routing production through to comprehension) fails. This is not a missing organ — it is a severed-but-present cable.

**The fix (next wire):** `pipeline → MarkovBundler::push → Trajectory → disambiguator_glue → context_chain (±5 replay) → comprehension router`. Closes the producer gap (`OQ-ARC-PRODUCER` already resolved the substrate = 16384-dim role-indexed `Trajectory`) AND the ±5 ambiguity-resolution wire in one flow.

**Other landmarks placed (full map in doc):** PFC = MUL + free-energy gate + global_context (WIRED planner-side, **not connected to the language faculty**); temporal-lobe semantic = COCA 4096² distance + DOLCE; angular gyrus = `vocabulary` + `nsm_primes` (word↔concept; metaphor = aerial cross-cohort). **Modality boundary (honest N/A):** auditory cortex / motor cortex / supramarginal phonology have no counterpart — DeepNSM is text+COCA, not audio. Do NOT build phonology. Cross-ref: `E-BROCA-WERNICKE-HIPPO`, `E-ENGLISH-BIFURCATES`, `disambiguator_glue.rs`, `context_chain.rs`, three-Markovs (#2 = the MarkovBundler wave).

---

## 2026-05-31 — E-BROCA-WERNICKE-HIPPO — the language stack is THREE separable faculties (projection ≠ comprehension ≠ memory); the witness lifecycle IS consolidation (a story aging into a fact)

**Status:** FINDING (architecture SoC; the faculty separation is enforced in code as of this commit). The consolidation arc (story→fact) within it is CONJECTURE (unbuilt/unmeasured). User-stated 2026-05-31 ("Markov bundler should be separate as the projection, while the sentence resolution is literal text comprehension with ambiguity resolution without tokens … we're sitting on a Broca and Wernicke and hippocampus"). Refines `E-ENGLISH-BIFURCATES`; doc § "the three faculties".

**Three faculties, never fused:**
- **Broca = projection / syntax:** PoS-FSM → SPO (`parser.rs`) + the role-superposed MarkovBundler **wave** (`markov_bundle.rs`→`Trajectory`); the basin/literal split (`arcs.rs::split_arcs`). *Assembles + projects structure.*
- **Wernicke = comprehension / resolution:** literal text comprehension over the **tokenless** COCA distributional space (4096 ranks + 4096² distance, NOT BPE); ambiguity resolution (±5 = contract `context_chain`); the fact/story router (`comprehension.rs`, reads `SentenceStructure` per-triple). *Resolves meaning.*
- **Hippocampus = episodic memory + consolidation:** the story-arc (`EpisodicEdges64`, ±5→±500) + crystallisation into semantic (neocortex = DOLCE). *Remembers + consolidates.*

**The spaghetti this corrects (concrete):** the first slice (`9af7f15`) put the fact/story router as a method on `Trajectory` — fusing the Wernicke decision onto the Broca projection carrier. Corrected here: the router moved to `comprehension.rs` reading the **comprehended** `SentenceStructure` (tokenless, per-triple); `Trajectory` keeps only `split_arcs`. Projection and resolution never share a carrier. deepnsm lib 95 green (arcs 2 + comprehension 4), default-clippy-clean.

**The consolidation insight (genuinely new — refines the bifurcation):** `WitnessTable`'s `spo_fact_ref None→Some→tombstone` IS hippocampal→neocortical **systems consolidation**. A story-arc witness accumulates in episodic memory, crystallises (`Some` = committed fact), then the episodic witness prunes (tombstone). **An aged story becomes a fact.** So fact-landing has TWO sources: the input fork (atemporal SPO → DOLCE) AND consolidation (a temporal story aged over ±500 → DOLCE). The bifurcation is not only an input switch — it is also a maturation path. `OQ-CONSOLIDATION`: is ±500 the trigger and `None→Some` the crystallisation? (net-new, unbuilt).

**Firewall:** Broca+Wernicke = deepnsm (English, upstream); Hippocampus+neocortex = downstream/agnostic. Only the `Landing{fact,story}` bit crosses — a boolean, not COCA. Cross-ref: `E-ENGLISH-BIFURCATES`, `E-EPISODIC-CLOSURE` (the three lifecycle structures the hippocampus owns), three-Markovs (#2 hybrid = the MarkovBundler projection wave).

---

## 2026-05-31 — E-ENGLISH-BIFURCATES — English deconstructs into BOTH fact-landings and story-arcs; the temporal marker is the router, the splat is the literal→basin resolver, ±5..500 is the missing wire

**Status:** CONJECTURE (architecture synthesis; assembles shipped parts + names the missing wires — end-to-end unbuilt/unmeasured). User keystone 2026-05-31 ("English can become both landing as facts and/or as story arc … enough moving parts to create the holy Grail"). Capstone that ties the four world-spine threads into one engine. Doc: `english-fact-story-bifurcation-grail-v1.md`.

**The keystone — English SPO bifurcates by temporality:**
- **atemporal SPO → FACT-LANDING** → aerial 10000² splat resonance proposes the OWL/DOLCE class → CAM confirms → **frozen identity** (DOLCE/OGIT, never moves). "a dog is a mammal."
- **temporal SPO → STORY-ARC** → ±5 coreference (`context_chain`) threads it → `EpisodicEdges64` basin (`family==0`) → `WitnessTable` accumulate-then-prune. "the dog ran to the park."
- **The router already exists in the sensor:** DeepNSM emits `SentenceStructure{triples, modifiers, negations, TEMPORALS}` (`parser.rs:57-66`). The `temporals` field IS the fact/story switch — WIRED today, read by nothing. Smallest net-new piece.

**The splat IS the literal→basin resolver (the piece the basin/literal duality was missing).** literal-arc = many COCA pointers (surface, redundant); basin-arc = the one DOLCE class (declared, exact). The 10000² gaussian splat lands a literal cluster on its basin: similarity PROPOSES (float, offline, jc-certified ρ=0.9973 → frozen integer codebook), CAM CONFIRMS. = the **semantic-landing** resolver, distinct from ±5 coreference (local) and head2head (angle). Corrects OQ-RESOLUTION-TREE: the "resolution tree" is THREE resolvers at three scales, not one mechanism.

**It IS E-EPISODIC-CLOSURE's three lifecycle structures, routed by temporality:** FACT → frozen identity (DOLCE/CAM, never moves); STORY-recent → within-session CLAM (±5, the only mover); STORY-old → cross-session append-index (±500 tail). The bifurcation is not a new structure — it is the rule that picks WHICH of the three an English SPO lands in. So "±5..500" = hot CLAM aging into the cold append-index, the two episodic structures already named.

**The "missing wire" (user-named): ±5.** DeepNSM emits SPO but its own markov does NOT connect to the contract-side `context_chain` ±5 replay-resolver. Latent defect surfaced: DeepNSM has TWO disconnected, dimensionally-incompatible mechanisms — a 512-bit `ContextWindow` (LIVE, `pipeline.rs:199`) and a 16384-dim `MarkovBundler` (DEAD — no producer; `content_fp` test-only). Three wires open: (1) DeepNSM SPO → `context_chain` ±5; (2) the temporal router (read `temporals`, route, net-new); (3) ±5→±500 tier (hot CLAM → cold append-index, net-new). Already free: `WitnessTable` ships the accumulate→prune lifecycle verbatim; `context_chain` ships the ±5 replay.

**Firewall HELD (GoBD-with-Rumi guard, end-to-end):** language/COCA stays UPSTREAM in DeepNSM (core has 0 deepnsm dep); both destinations AGNOSTIC (DOLCE class, episodic basin = opaque handles, never `rank:u16`); float lives only offline in jc, online is integer; similarity proposes, identity addresses, never swapped. The ~4096 story-basins ≠ COCA-4096 (independent 12-bit `local`; OQ-BASIN-COUNT confirmed distinct).

**Honest state:** DeepNSM SPO+temporals WIRED (102 tests); aerial splat→DOLCE SHAPE wired (42 tests; producer in ndarray; end-to-end CONJECTURE); ±5 `context_chain` WIRED contract-side; `EpisodicEdges64`+`WitnessTable` WIRED (#446); routing + 3 wires = net-new. ~5 tested shapes, 3 missing wires, 1 net-new router. **First buildable slice (firewall-safe):** `Trajectory::split_arcs → (BasinArc, LiteralArc)` in deepnsm (names the duality at the `disambiguator_glue` seam; gives the dead `MarkovBundler` a producer; English-side only). **Promoting probe:** does temporal-routed, English-sourced SPO landing reproduce #444 locality (98.6% intra-basin) on the fact path? PASS ⇒ CONJECTURE→FINDING. Cross-ref: `E-EPISODIC-CLOSURE`, `E-ARM-JC-RESOLVES-BOTH-SEAMS`, three-Markovs taxonomy, `splat-codebook-aerial-wikidata-compression.md`, `owl-dolce-hhtl-compartments-aerial-fed.md`.

---

## 2026-05-31 — E-EPISODIC-CLOSURE — the episodic spine closes on three lifecycle-separated structures; compression IS the bounded horizon (not a codec)

**Status:** FINDING (architecture; converged 2026-05-31, grounded in cognitive-risc/faiss-homology/wikidata-hhtl docs + AriGraph 2407.04363 + #444 probe).

1. **Three structures by lifecycle:** frozen identity = OGIT palette + CAM (never moves); cross-session index = Lance append-only version log = pseudo-radix (append + immutable pointer => stable addressing, no rebalance); within-session = CLAM over an ephemeral KV (the only thing that moves).
2. **EW64 = AriGraph episodic edges** (not a CE64 lens): basin + multiple edges; intra-basin (~98.6%) inherited ~0 bits; cross-family (~1.4%) = 4-bit nibble into the OGIT-class palette (identities inherited, never on the edge). Shipped: EpisodicEdges64 (D-EW64-1).
3. **Compression IS the bounded horizon:** a research = a free-energy descent resting at the homeostasis floor; awareness (MUL residual-F) = the stop; 256 inputs -> <32 clusters; 4096-64k/KV = shock-absorber headroom. Lever = horizon-shortening (arbiter quality), not a codec. Bitmask doubles as attention mask; ViewAngle (D-VIEW-1) selects the inherited view-schema.

Firewall held: identity exact (CAM/OGIT), stories flexible (CLAM/discovery), never swapped. Plan: episodic-risc-spine-v1.md.

---


## 2026-05-31 — FINDING (PROBE RESULT, measured): ontology partition-locality SURVIVES on real ontologies — locality 98.6%, max fan-out 3 (<=16), Q=0.325 ⇒ 16-bit local refs + <=16 family frontier are REAL (on real data, NOT yet Wikidata)

**Status:** FINDING (measured, not asserted). Probe `crates/jc/examples/ontology_locality_probe.rs` run on the on-disk ontologies (DOLCE-Ultralite, schema.org, Odoo, PROV-O, QUDT, OWL-Time) — the falsifier for the delta-card/inherited-nothingness addressing claim (probe #1 of `delta-card-addressing-integration-map.md`). PASS.

**Measured numbers (1170 classes, 1224 subClassOf edges, 33 top-basins):**
- **LOCALITY = 98.61%** (1207/1224 edges intra-basin) — the map's "~90% local" claim survives and EXCEEDS it.
- **FAN-OUT max = 3** (≤16 ✓); histogram: 1121 classes have exactly 1 parent-basin, 15 have 2, 1 has 3, 33 are roots. ⇒ no class needs more than 3 distinct family pointers; the ≤16 frontier has huge headroom.
- **MODULARITY Q = 0.3246** (>0.3 = clear community structure; Newman modularity of the basin partition).

**What it proves / doesn't:** on REAL frozen-ISA ontology structure, the 16-bit LOCAL reference + the ≤16 family-cohort frontier are real — most subClassOf references stay inside one top-basin, partition locality is genuine. HONEST CAVEAT (in the probe's own verdict): measured on real ontologies (~10³ classes), **NOT Wikidata** (~10⁸); same KIND of structure, smaller scale. The Wikidata P279 run remains the open probe (gated on a real dump, not on disk). Promotes the addressing-locality CONJECTURE to FINDING *on real ontologies*; the Wikidata-scale claim stays CONJECTURE.

**Falsifies a worry:** had locality been low or fan-out high, the cheap-local-reference + inherited-nothingness scheme would degrade to mostly-far pointers (the scheme's main risk, flagged in the #442 review + the integration map). It didn't — the partition is real. Cross-ref: `delta-card-addressing-integration-map.md` (probe #1), `agnostic-lazy-world-spine.md`, `jc/examples/splat_louvain_modularity.rs` (the modularity machinery reused), `wikidata-lazy-spine-hydration-v1.md` (D-LWS-8 probe harness).

## 2026-05-31 — FINDING (SoC correction): markov_soa IS AriGraph (the cold-path Markov chain promoted to the hot-path SoA); AriGraph is agnostic & NOT necessarily English — the language layer (DeepNSM/COCA) stays UPSTREAM and never reaches into the hot graph

**Status:** FINDING + done (move shipped; AriGraph version unverified-offline — core does not build in the sandbox). Corrects the premature `deepnsm::markov_soa` placement (`e0a5049`, now deleted) AND its own first framing (the "inject a COCA distance as an alternative" error — that would be the GoBD-with-Rumi mistake).

**markov_soa IS AriGraph — not "a projector that lives in AriGraph."** AriGraph is a Markov chain in the **cold path**; `markov_soa` is that same chain **promoted to the hot path** (the per-mailbox SoA). Same object, agnostic nature, hot instead of cold. Particle/wave: EW64 / the `CausalEdge64` W-slot → witness arc = the **particle** (discrete, addressable, exact); the windowed projection = the **wave** (accumulated resonance). Both ARE AriGraph. Now `crates/lance-graph/src/graph/arigraph/markov_soa.rs`.

**AriGraph is agnostic AND NOT necessarily English — the deeper SoC step.** AriGraph holds SPO from ANY source (business, GoBD, Wikidata, English text); its agnosticism is structural — the SoA row is three **opaque `u16` ranks** carrying no language. The match metric is **AriGraph's OWN `cam_pq::DistanceTables`** (the graph's native semantic distance), injected as `Fn(u16,u16)->u8` so the projector names no encoding. **The language layer stays UPSTREAM in DeepNSM and never reaches into the hot graph:** DeepNSM / COCA-4096 / grammar templates are the *English-language input sensor* — they scan flat data (usually English), parse it, EMIT SPO into AriGraph, and **MUST stay English** (the grammar templates get messy the instant they aren't). **Injecting a COCA/language distance into the hot-path graph would be the GoBD-with-Rumi error** — running a *language* lens over an *agnostic* graph. The injected distance is AriGraph's cam_pq, NOT a language table. SPO *can* be English (when DeepNSM produced it) but the SoA/AriGraph mailbox-view is never *forced* into a language. Reuse DeepNSM by it FEEDING AriGraph upstream, never by core calling into it (core has 0 deepnsm dep — the dep graph enforces this).

**Status of the code:** `SpoRanks{s,p,o:u16}` (opaque) + `SoaWavePrimer` + `WaveProjection::best_guess_match(injected dist)` — 4 tests written (determinism, clamp+proximity, injected-distance match, empty=0); **unverified-offline** (lance-graph core's lance/datafusion/arrow deps don't fetch in the sandbox — compile-verify on a full checkout). Truly-correct home = inside the EW64-in-SoA seam (P1+P2); this is the agnostic wave-projector that seam will host. Cross-ref: three-Markovs FINDING (#2 hybrid), EW64-reactive-seam, `witness_corpus.rs` (AriGraph native cam_pq), `soa_view.rs::MailboxSoaView`.

## 2026-05-31 — FINDING (taxonomy, standing definition): the THREE Markovs — one word, three ranked uses; the deterministic CE64→EW64 chain is the line between grounded and praying

**Status:** FINDING (standing definition, user-stated 2026-05-31). The canonical disambiguation of "Markov" in this stack. Anchors `markov_soa` (#2), the EW64 reactive-seam (#1 plumbing), and the deprecated VSA-substrate (#3). Ranked by epistemic grounding.

**The three Markovs:**

1. **Context-chain building (THE substrate, deterministic).** Mailbox chaining through the `CausalEdge64` W-slot → `EpisodicWitness64` arc — walk the witness references. Fully deterministic, exact, addressable; the arc IS the chain, no bundle. This is *the* Markov — reasoning traverses it; it is truth. (= the EW64 reactive seam; P1+P2 plumbing below.)

2. **Hybrid+ autocomplete (deterministic spine + leashed fuzz).** #1's deterministic chain PLUS a fuzzy accumulated witness-bundle as **speculative autocomplete** — the bundle *proposes* the next mailbox, the chain *confirms or refutes*. Deterministic spine + fuzzy proposer on top; a wrong guess is cheap (cheap reprioritization, never a wrong answer). (= `deepnsm::markov_soa`, shipped `e0a5049`, + the grail-fold experiment, P3 below.) **Invariant: #2 is only ever #2 while its fuzz stays leashed to #1's chain — an UNLEASHED bundle degrades into #3 by definition.**

3. **"Sink in and pray" (the error).** Old VSA bundle-as-Markov: ceiling-bound superposition, opaque vector, hope-based readout — **NOT deterministically grounded** like #1. The black box the whole thread rejected; deprecated for reasoning (the "every GGUF would already be VSA-fp32" disproof: 30 years, planetary compute, nobody adopted it as a substrate ⇒ it lost). If materialized at all, signed base-5 packed, never fp32, and never as #3.

**The line (one sentence):** **#1 is the chain. #2 is the chain PLUS a guess it must confirm. #3 is the guess WITHOUT a chain to confirm it.** The presence/absence of the deterministic CE64→EW64 chain underneath is the entire distinction. The firewall is NOT fuzzy-vs-exact — it is **"fuzz a chain confirms (#2, legit) vs fuzz nothing confirms (#3, error)."**

**Dependency ordering (gate before grail — wire first, then experiment):**
- **P1 — AriGraph → SoA.** Implement the `HotWitness` `todo!()` scaffold (`witness_tombstone.rs`, D-ATOM-5: calcify/from_hot/tombstone-persist/WitnessLink-verify); episodic/semantic edges become SoA-resident. (`E-ARIGRAPH-IS-AN-ISLAND`: "Ee→EW64(hot)+WitnessCorpus(cold)").
- **P2 — EW64 in `MailboxSoaView`.** Define `EpisodicWitness64` + add `fn episodic_witness(&self) -> &[EpisodicWitness64]` to the view — following the EXISTING deferred-accessor pattern (`soa_view.rs:71` "add `fn qualia()` when the first consumer needs it"). The CE64→EW64 arc becomes a readable, addressable SoA column.
- **P3 — the grail experiment (CONJECTURE, gated, Jirak-baselined).** ONLY after P1+P2: fold CE64 or EW64 **deterministically** into a VSA projection; **measure** recoverable best-guess signal vs the black-box baseline (Jirak floor, `I-NOISE-FLOOR-JIRAK`). PASS ⇒ "deterministic-arc-fold autocomplete" (the grail) — promote; FAIL ⇒ stays a #2 proposer. You cannot fold what isn't wired (P1+P2), and you cannot tell signal from prayer without the baseline. P3 is a NEW deliverable DOWNSTREAM of the EW64 seam spec — not part of it (no scope creep).

**Cross-ref:** `markov_soa.rs` (#2, COCA+CAM-PQ, no cosine), EW64-reactive-seam FINDING (#1 plumbing), substrate-decision FINDING (#3 deprecation + signed-base-5 carrier note), `witness_tombstone.rs::HotWitness` (D-ATOM-5), `soa_view.rs:71` (the accessor pattern), `I-VSA-IDENTITIES`, `I-NOISE-FLOOR-JIRAK`.

## 2026-05-31 — FINDING (substrate decision, CONVERGED): explicit 32k SPO-W IS the substrate; VSA16k is a strictly-fuzzy PROPOSER (cognitive priming) via COCA+CAM-PQ — never cosine, never truth

**Status:** FINDING (user-stated + judged 2026-05-31, decisive). Reasoning-substrate decision FIRM; the proposer role is the legitimate home (CONJECTURE on whether it carries signal above noise). This entry CONVERGED across several refinements this session — it supersedes two earlier framings of mine: (a) "VSA = per-cycle experience/soul-print vector" (wrong scope), (b) "keep DeepNSM as a parallel universe" (DeepNSM migrates too). Shipped artifact: `crates/deepnsm/src/markov_soa.rs` (commit e0a5049).

**The truth path is deterministic, full stop.** A whole book = **~32k exact SPO-W triplets in context** (SPO + the `CausalEdge64` W-slot witness), via the mailbox reference-pointer table (`TripletGraph` → `SpoStore` L2 cold columnar, `spo_bridge.rs`) + per-mailbox NARS awareness. Exact, deterministic, CAM-addressed, **zero hallucination, zero embedding, zero bundle**. Holding the explicit 32k is worth **categorically more** than any fuzzy bundle: the explicit form is addressable (each triplet retrievable by CAM), lossless, reasoning-capable (CE64/EW64 traverse, NARS-revise, counterfactual-test), provenance-bearing (the W) — a bundle is none of these. Capacity math seals it: ~√d/4 ≈ 32 recoverable items at d=16384; a book is ~32k = 1000× over → a whole-book bundle is recovery-noise. "String theory": the 32k are the full configuration; a bundle is one projected shadow — you can't do physics in the shadow.

**VSA16k's legitimate role = a strictly-fuzzy PROPOSER (cognitive priming), firewall-gated.** The fundamental ERROR was never the fuzziness — it was the *posture*: a black box **praying for meaning** (opaque vector, cosine, hope). The irony: the SAME fuzziness is *correct* one layer over, in the discovery/proposer layer (faiss-homology / `I-VSA-IDENTITIES`: similarity lives ONLY there, never in addressing/reasoning). As a proposer it **proposes where-to-look / what-this-resembles** ("this feels like a Sicilian with a pinch of death trap"), the exact 32k SPO-W **always confirms**, and a wrong guess = cheap reprioritization, never a wrong answer (honest approximation, not praying). Test that separates sin from virtue: "if this number is wrong, what breaks?" — reasoning: the answer (catastrophe); discovery: you prefetched the wrong region and exact-confirm corrects you (self-healing). = **cognitive priming**: System-1 prime (VSA proposer) → System-2 calculate (32k SPO-W). The free-energy loop the stack already runs: prior = the prime, evidence = the triplets.

**The match is DeepNSM's OWN machinery — NOT cosine.** COCA-4096 vocabulary + the CAM-PQ 4096² u8 word-distance matrix via `SimilarityTable::lookup_u8` + grammar heuristics. `markov_soa.rs` makes this explicit: a SoA ±window → the addressable list of COCA-rank SPO triplets + full provenance (which rows, what proximity); `best_guess_match` = nearest-triplet CAM-PQ similarity. The triplets stay addressable (no superposition kills the register). Zero new dep (consumes `contract::soa_view::MailboxSoaView` through DeepNSM's existing contract dep). 5 tests incl. `best_guess_match_uses_cam_pq_not_cosine`.

**DeepNSM migrates too (not a standing parallel universe):** its NSM→SPO FSM *produces triplets* → folds INTO the deterministic SPO substrate; its `markov_bundle` (512-bit XOR/majority) is the same projection idea → deferred-research bucket. Nothing remains as a parallel reasoning substrate.

**Aerial synergy (grounded seam):** `markov_soa` is the **within-cohort** proposer (one book's ±window); aerial is the **cross-cohort** proposer (mines X→Y over many observations). They share one firewall + one distance idea — DeepNSM's CAM-PQ u8 word-distance is exactly aerial's `CodebookDistance::distance(a,b)->u32` shape, so a ~20-line adapter lets aerial probe DeepNSM's COCA semantic space. Both feed the EW64 "fire together → wire together" prefetch from opposite ends. Queued (not built): the `markov_soa`→`CodebookDistance` adapter D-id (crosses deepnsm↔arm-discovery; own slice).

**CONJECTURE + probe:** does the CAM-PQ match over windowed rank-triplets carry recoverable best-guess signal above the Jirak noise floor (`I-NOISE-FLOOR-JIRAK`)? PASS ⇒ promote as an Aerial discovery-lens; FAIL ⇒ research-only. **Reconciliation note:** CLAUDE.md "The Click" + `I-SUBSTRATE-MARKOV` still frame the VSA bundle AS the reasoning substrate — superseded (reasoning = deterministic SPO-W; VSA = fuzzy proposer); CLAUDE.md edit deferred to a deliberate doc pass, this board finding is the authoritative ledger meanwhile. Cross-ref: `delta-card-addressing-integration-map.md`, EW64-reactive-seam finding, `I-VSA-IDENTITIES`, `markov_soa.rs`, aerial `CodebookDistance`.

## 2026-05-31 — FINDING (integration gap, SHOCK): EW64 is the unbuilt REACTIVE SEAM — Markov is the basis, predictive-prefetch is the Meta, "fire together → wire together"; every link shipped, the chain is open at the joints

**Status:** FINDING (named integration gap + behavioral spec; user-stated 2026-05-31). Refines `E-EW64-IS-PREDICTIVE-PREFETCH` + `E-AERIAL-FEEDS-EW64-PREFETCH` + `E-ARIGRAPH-IS-AN-ISLAND` (2026-05-30). Decision: spec the WHOLE reactive seam (one spec), spawn AFTER the running probe wave consolidates (second wave).

**The layering (corrected — I had it inverted):** **Markov (the `CausalEdge64` W-slot → EW64 witness arc) is the BASIS** — the substrate fact of which witnesses fired in sequence ("fire together"). **Predictive-prefetch is the META** — the emergent behavior on the Markov basis: because they fired together, prefetch the next before it's asked ("wire together"). So EW64 is not an optimization layered onto reasoning — **the prefetch IS the wiring IS the learning** (Hebbian, literally): aerial mines co-occurrence offline ("fire together"), EW64 prefetches it online ("wire together"), the surviving arc is the learned structure. One mechanism, three names by timescale.

**The reactive spine (the keystone, previously missed): `Lance update = the witness pointer = the SurrealDB kanban subscription trigger`.** A witness fires → Lance fragment append (the update IS the witness pointer materializing) → SurrealDB LIVE subscription on that table fires → the kanban (`KanbanMove`, #437) advances a mailbox phase → EW64 prefetches the aerial-predicted next arc into the SoA → the shader finds it already resident. The update, the pointer, and the trigger are the **same event** — the "wire together" propagates THROUGH the storage layer as the prefetch signal. This is why EW64 shares CE64 low-40 bits (co-address), why kanban is in `contract`, why surreal_container is a transparent SoA view — all built to be links of ONE chain, and the chain is the thinking.

**The SHOCK (the diagnosis):** every individual link exists and tests green, but **the chain is open at the joints** — the island-archipelago failure (`E-ARIGRAPH-IS-AN-ISLAND` verbatim: "Ee→EW64(hot prefetch)+WitnessCorpus(cold)" is the unwired task). Shipped: `CausalEdge64`, `WitnessTable<64>`, `ReasoningWitness64` (splat.rs:78), `KanbanMove`+SoA view (#437), aerial X→Y (#436/#438/#443). Scaffold-only: `HotWitness` (witness_tombstone.rs:70, `todo!()` bodies). **Unbuilt: `EpisodicWitness64`/`SpoWitness64` (arc `pr-ce64-mb-4`) — 0 code symbols — AND the Lance-LIVE→Surreal→kanban subscription.** It's the most expensive kind of gap: invisible in green suites (every crate passes; the system doesn't *do* the thing) because the **integrating seam was never built**. EW64 is not a type to add — it's the seam that closes the reactive loop: contract-atom (shares CE64 bits) + the Lance→Surreal→kanban subscription + materializing `HotWitness`'s `todo!()`s. Three links, one chain.

**Next (queued, second wave):** one spec `.claude/specs/episodic-witness64-ce64-prefetch.md` covering the whole seam (contract EW64 atom CE64-mirrored, shares low-40 SPO bits, `SpoWitness64` alias `pr-ce64-mb-4`; the Lance-LIVE→Surreal-subscribe→kanban wiring contract; `HotWitness` materialization), impl phased/gated (surrealdb+ractor are heavy/cross-crate; firewall + D-ARM-7 hold). Also LE-1: EW64's second role = syntactic-coreference pointer (relative pronoun → antecedent pointer, not a bundle; register-laziness). Cross-ref: `E-EW64-IS-PREDICTIVE-PREFETCH`, `E-AERIAL-FEEDS-EW64-PREFETCH`, `E-ARIGRAPH-IS-AN-ISLAND`, `E-ARIGRAPH-PAPER-GROUNDS-CE64-EW64` (LE-1); `delta-card-addressing-integration-map.md`; #437 kanban; `splat.rs::ReasoningWitness64`, `witness_table.rs`, `witness_tombstone.rs::HotWitness`, `arigraph/{episodic,witness_corpus}.rs`.

## 2026-05-31 — FINDING (capstone synthesis): the DELTA-CARD world-spine — card = surprise, deck = expectation; key and value compress by the same delta-over-frozen-archetype move

**Status:** CONVERGED VISION (8-turn design synthesis; primitives shipped, consolidation + delta-card value model NEW, claims labelled + probed). Full map: `delta-card-addressing-integration-map.md`. Supersedes the scattered addressing fragments in `agnostic-lazy-world-spine.md`.

**The one idea:** a card stores the *surprise*, the deck stores the *expectation*; **meaning = deck ⊗ delta**. Everything — recipe, Wikidata entity, address, sentence-mailbox — is a small delta over an inherited frozen archetype, reconstructed on demand. This IS the free-energy framing (`CLAUDE.md` F = (1−likelihood)+kl): archetype = prior, delta = prediction-error, **bit-width = residual surprise**. It applies to BOTH halves of a row — the **key** (address) and the **value** (content) compress identically.

**Cookbook (value side):** `recipe = inherited(region×season×persona) + 8–16 delta bits` (texture/sweet/sour/salty/veg-axis, 2b each). The 16-bit card is meaningless until resolved against its deck. Boundary: the delta carries the *compressible profile*; irreducible specifics (quantities, novel steps, fusion) are stored values / forks (generator-vs-derivable split).

**Addressing chain (key side):** (1) **partition-as-address, schema-as-deck** — the address is *location not a stored column* (Quartettkarten: the card is *in* the Auto box, doesn't carry category=Auto); the 256-ary OWL/DOLCE subClassOf directory encodes upper bits in the path, OGIT holds the lookup once, the row stores ~0 address/schema/label bits. (2) **27-bit truthful floor** — 113M → ⌈log₂⌉ = 27 bits irreducible (the QID already ≈ 2²⁷; classes can't shrink *identity*) — but partition-as-address makes the 27 bits FREE per-row (`(path<<offset)|row_index`). (3) **sparse radix range-delegation** — don't build 256⁴ files (2.6% occupied); a path-compressed trie of occupied branch points (≈ class count, KB–MB); skew absorbed by 38× headroom (cohort≠class). (4) **frozen ISA, no rebalance** — DOLCE/FIBO/GoBD/OGIT are compiled constants, leaves append-only ⇒ a compiled perfect hash, the rebalancer is DELETED; schema bumps are version-gated amortized upgrades (I-LEGACY-API-FEATURE-GATED).

**Frame model (x264/265 capstone) = Lance fragment-versioning, not new machinery:** I-frame = frozen radix + compacted base fragment; P-frame = appended entities + CLAM-clustered arrivals (adaptive INSIDE the delta); B-frame = the RISC compose-cache; GOP compaction = re-emit keyframe = the amortized upgrade = the one moment validated similarity FREEZES into structure. Resolves frozen-vs-adaptive (CLAM proposes in the delta, keyframe never moves); tradeoff = read-amplification bounded by GOP length.

**RISC compose-not-materialize (edge side):** store generators (~N), derive ≤7-hop closure via `ComposeTable`/`mxm` — dissolves the hub problem (hubs reached by composing forward generators, never store inbound back-edges); generators=continuant/cold, composed paths=occurrent/evictable-KV (one ontology-derived eviction policy); new surface = per-predicate composability flag.

**Two trees, never confused (iron-rule guard):** frozen ontology radix = the address (exact, CAM); CLAM/CHESS adaptive manifold tree = proposes/validates the partition + delta placement (similarity, discovery-only, offline). Adaptive proposes, frozen ships.

**Scale identities (all the same powers of two):** 6-bit cohort(64) ⊂ 16-bit book(64K SPO; Bible ~32k = half) ⊂ 18-bit hot envelope(256K concurrent = both books + a Wikidata window) ⊂ 32-bit world(4.3B, cold/lazy). Reasoning = traversing the CE64 W-slot → WitnessTable/EpisodicWitness64 arc (NOT the 16384 VSA bundle, retired legacy). Reading = accumulating SPO mailboxes + their arc.

**Probes (falsifiers, measure before freeze):** (1) Louvain modularity + CLAM on the real P279+edge graph → is locality ~90%, what's the natural fan-out, which hubs to compose; (2) delta-card residual histogram per cohort → does 8–16 bits reconstruct truthfully; (3) ≤7-hop reachability hit-rate + compose-cache churn vs stored-edge baseline → sets GOP cadence. Gate: D-ARM-7 (`jc::jirak`) before writing a live store. The one missing runtime piece: the `NiblePath`-keyed tiered hydration manager. Cross-ref: `delta-card-addressing-integration-map.md`, `agnostic-lazy-world-spine.md`, all shipped primitives (#441/#442/#438/#443, CausalEdge64, ComposeTable, CLAM, Lance fragments).

## 2026-05-31 — FINDING (north-star vision): the AGNOSTIC LAZY WORLD-SPINE — Wikidata as a foveated tiered substrate; one address unifies ontology+memory+space

**Status:** NORTH-STAR VISION (living). Addressing + compression + late-resolution are BUILT; the runtime tiered-hydration layer is NOT. Doc: `agnostic-lazy-world-spine.md`. The goal the D-ARM-13/14 + D-CLS + Wikidata-HHTL arc serves.

**The goal:** compress Wikidata into a **lazy-loading spine** — tiny resident skeleton + **foveated, blasgraph-adjacent, on-demand hydration** (sharp at the reasoning fovea, periphery coarse; Google-Maps tile prefetch of the adjacent basin). One **unified allocation address** = `NiblePath` = ontology position **=** memory arena **=** (at the leaf) spatial coordinate.

**The tiering (this session's closing synthesis):** COLD `Lance columnar` ◄─`NiblePath`─► HOT `mailbox SoA register` (#437, label-free `&[T]`) ◄── SEMANTIC `OGIT/DOLCE cache` (C2 resolve-not-store). Three reframings complete it: **(1)** the cold path SPLITS — DataFusion rows/cols joins are SLOW and serve **business-SQL ground truth ONLY** (off the hot path); HHTL hydration is **address-based, not join-based** (`NiblePath` → Lance read → CAM/palette/`blasgraph`, O(1)). The `GraphRouter` routes HHTL→fast-address, SQL→DataFusion. **(2)** DOLCE → a **1-bit residence policy**: continuant (Endurant/Quality/Abstract = permanent, persist) vs occurrent (Perdurant = temporary, evictable — the Baton/event traffic). The ontology's top split IS the cache policy; `dolce_id 0..3` stays cache-resolvable, eviction keys on the derived bit. **(3)** AriGraph SPO + labels → **agnostic SoA register + late-hydrated labels** (the #441 C2 "class flies above the SoA" doctrine, wholesale): structure hot+agnostic, semantics a cache overlay keyed by address ⇒ representation compartmentalized (basins), cheap (resolve-not-store + lazy), agnostic (register is meaning-free).

**Invariants held:** CAM-exact + similarity-only-in-discovery (the view/address are exact; the φ-spiral leaf is a coordinate, not a fuzzy index — faiss iron rule); keep `dolce_id 0..3` (derive the 1 bit, don't drop the 4-facet axis); the SoA stays agnostic forever (never cache a label in the register — core inv #1 / C2).

**Markov is the CE64→EW64 arc, NOT the 16384 VSA bundle (retired legacy).** `witness_table.rs`: "the chain of W-references across edges forms a Markov-style belief-update arc through episodic-reference vectors." Traversal walks W-references backward without dereferencing the full SPO store per hop — native, integer, exact, cheap. So **reasoning = traversing the CausalEdge64 W-slot → WitnessTable/EpisodicWitness64 arc + SPO**, not bundling a fingerprint; the 16384 carrier survives ONLY as the discovery-layer (aerial/splat) similarity carrier, never on the reasoning hot path. ⇒ **reading a text = accumulating SPO mailboxes + their CE64/EW64 arc** (no embedding, no forward pass); ambiguity resolved by counterfactual testing (`recipe_kernels`: world⊗factual⊗counterfactual, divergence=popcount, on the scenario-only Counterfactual channel). A **250-page book ≈ 4–5k sentences ≈ ~4096 SPO mailboxes** + counterfactual overhead = one bounded cohort (the per-cohort `WitnessTable<64>`, 6-bit W-slot, walks inside the cohort). **Pointer-width = corpus-size identity:** 6-bit W-slot = 64 (the cohort) ⊂ **16-bit (in EpisodicWitness64) = 65,536 ≈ 64K SPO = one BOOK** (Bible ~32k sentences = half; novel ~4–5k = ~7%) ⊂ 32-bit `mailbox_ref` = 4.3B (the world-spine, Wikidata ~115M). 64K is exactly the documented mailbox-envelope lower bound (witness_table.rs "64K–256K", plan §10). Pick the pointer width, you've picked the horizon: cohort ⊂ book ⊂ world. **Address vs hot working-set (the 256K payoff):** Wikidata (~115M) is 32-bit-ADDRESSED (cold spine, lazy, never resident); the documented **256K (2¹⁸) is the concurrent HOT mailbox envelope**. You foveate Wikidata, so 256K holds whole corpora + a hydrated Wikidata slice at once: Bible (~32k) + LOTR (~28-30k) = ~62k ≈ one 16-bit corpus, leaving ~190k (3× headroom) for the Wikidata reasoning window. ⇒ cross-corpus grounded reasoning ("Frodo ↔ biblical archetype, grounded in Wikidata") fits in one hot context BECAUSE the spine stays cold. Bounded hot context, unbounded cold spine. Full nesting: 6-bit cohort(64) ⊂ 16-bit book(64K) ⊂ 18-bit hot envelope(256K) ⊂ 32-bit world(4.3B, cold).

**Bit budget + addressing:** the resident agnostic row shrinks 16384 → ~4096 bits (HHTL address carries class+label inheritance; qualia-i4-16D 64 + thinking-i4-32D 128 + CausalEdge64-with-W-slot + EpisodicWitness64 + presence/class fit). The address can be brutal: **byte-aligned 256⁴ = 2³² ≈ 4.3 B** — the 4-byte CAM-PQ code IS the address = class+label key = palette-distance key (vs 64K² = 4.3 B shallow, 4096³ = 69 B headroom). Fan-out frozen append-only once chosen.

**The one missing runtime piece:** a `NiblePath`-keyed **tiered hydration manager** (hot mailbox-SoA ↔ cold Lance, foveated `RouteAction` prefetch, perm/temp eviction, late labels). CONJECTURE to probe: the Poincaré φ-spiral leaf encoding. Gate: D-ARM-7 (`jc::jirak`) before any hydrated rule writes a live store. Cross-ref: `agnostic-lazy-world-spine.md`, `owl-dolce-hhtl-compartments-aerial-fed.md`, `wikidata-hhtl-load.md`, #437/#441/#442/#443, `crates/jc`.

## 2026-05-31 — FINDING: D-CLS (#441) ↔ D-ARM-14 (#438) converge on Wikidata-HHTL — the second-domain falsifier reuses the class-meta-DTO 1:1 (cross-session synthesis)

**Status:** FINDING (cross-session reconciliation, confirmed by the D-ARM-14 session). Anchors the Wikidata-HHTL arc on the merged D-CLS machinery; no parallel layer grows.

**The convergence:** 439/440/441 (D-CLS) and 436/438 (D-ARM-13/14) are convergent, not conflicting. The Wikidata "D-CLS triple" `(class_id, shape_hash, presence_bitmask)` = `(ClassId(u16), StructuralSignature, FieldMask)` — ALL #441 types. wikidata facet-bitmask = `FieldMask`; shape = `signature()`'s structural-hash (generalised from `OdooEntity` to any entity); presence = `FieldMask`.

**The firewall split (clean territory, no collision):**
- **Proposer side (the D-ARM-14 session's lane):** `arm-discovery::aerial` stays the zero-dep PROPOSER — similarity in discovery only, skeleton-only predicates, emits `(s,p,o,f,c)`. Validated by the proposer-layering FINDING (67903a8): aerial FEEDS the AST hub, is not the hub.
- **Hub side (this/D-CLS session's lane):** `contract`/`ontology` own `FieldMask`/`signature`/`ClassView` + the new `contract::hhtl::NiblePath` router. "The hub side owns contract/ontology."

**This slice (D-WIKI-HHTL-1):** `contract::hhtl::NiblePath` — the 16ⁿ Abstammung bucket router #438's wiring doc names as "downstream." DOLCE-agnostic (`basin: u8`). basin `0..3` = `dolce_id::{ENDURANT=0,PERDURANT=1,QUALITY=2,ABSTRACT=3}` (#441 cache u8), which ALSO matches arm-discovery's discovery-side `DolceCategory::basin()` ordering (#438) — both sides agree on the nibble WITHOUT either embedding the enum (OD-DOLCE: resolve through the cache, b31464d). + `FieldMask::inherit` (mask-inherits-as-delta; multi-parent = facet bit in the same mask, NOT a 2nd path). 4 teeth-tests; 501 contract lib green; zero-dep preserved.

**One proposer-side alignment — NOT this session (the D-ARM-14 session owns it, its own branch):** `aerial::ontology::DolceCategory` currently hardcodes `dolce:…` IRIs; ratified pattern = emit `dolce_id` u8, resolve the IRI late from cache (proposer stores no semantics). `basin()` already matches `dolce_id` ordering → alignment, not rework. Recorded here for cross-session visibility only.

**#438 council verdict (4f381a8):** no code action items — it reviewed a `discovery_origin` byte-grammar plan, not the crate; fixed a stale Wave-1 citation, escalated 2 spec decisions (tier-set + proposer-id width) to ISSUES, queued D-CHESS-BRINGUP-1.

**Cross-ref:** #441 (D-CLS), #438 (D-ARM-14 P1), 67903a8 (proposer-layering), b31464d (OD-DOLCE), `splat-codebook-aerial-wikidata-compression.md`, `wikidata-hhtl-load.md`, `contract::hhtl`.
## 2026-05-31 — FINDING: ANY OWL/DOLCE domain compartmentalizes into the same 16ⁿ HHTL, fed by aerial — the class-meta-DTO is the universal substrate; domains differ in content, never structure (D-ARM-14 Phase 2)

**Status:** FINDING (per-layer domain-independence is shipped/proven on 2 domains; full-scale universality is CONJECTURE — each new domain is a falsifier). Doc: `owl-dolce-hhtl-compartments-aerial-fed.md`. PR: D-ARM-14 Phase 2 (`claude/jolly-cori-clnf9-darm14-p2`).

**The generalization (the user's "potential"):** medicine (SNOMED/FMA), finance (FIBO), geography, law, Odoo, and all of Wikidata are NOT bespoke loaders — each yields the SAME four things and lands as the SAME `(ClassId, StructuralSignature, FieldMask)` row: a **basin** (DOLCE `dolce_id` 0..3), a **nibble path** (`NiblePath`, P279 subClassOf descent), a **`FieldMask`** (property presence), a **`StructuralSignature`** (FNV-1a shape-family). Domain enters only as *content* (the property-id set + the corpus), never as *structure*. This is `cognitive-risc-classes.md` N4 operationalised — "chess + Odoo + Wikidata-anatomy through one Class+SoA+HHTL+CAM, no special-case."

**Why it holds (proof chain, not hope):** every layer is domain-agnostic by construction — `NiblePath` takes a `basin: u8` with zero DOLCE knowledge (#442); `FieldMask` is one-bit-per-property (#441); `StructuralSignature` is label/QID/domain-independent (Odoo #441 + Wikidata #442 BOTH collapse structurally-identical classes to one family); `ClassView` is a trait both `RegistryClassView` (Odoo) and `WikidataClassView` (Wikidata) impl unchanged. DOLCE is resolved late from the cache (`basin = dolce_id`, no enum). The **aerial feed** is the part that makes each domain self-populating: similarity is an injected integer `CodebookDistance` (the 10000² splat, jc-certified), so a new domain is fed by pointing aerial at (a) a domain splat codebook + (b) a domain row corpus — `tests/wikidata_landing.rs` is the template; swap the fixture, **no new hub code**. Phase 2 proved this end-to-end (splat → recover 6 DOLCE basins → land on real `FieldMask`+signature; film≡tv collapse, human⊂person inherit).

**Compartmentalization rule:** ONE tree axis (Abstammung/P279), DOLCE facets seed root nibbles 0x0..0x3 (0x4..0xF reserved). The DOMAIN is an ORTHOGONAL compartment (namespace/facet tag), NOT a second path — cross-domain multi-typing is a facet bit in the same `FieldMask`, like bat = mammal-path + flight-bit. Open: reconcile the OGIT *byte*-basins (0x10..) with the `NiblePath` *nibble*-basin before a multi-domain load.

**Scale-freezes (carry from the #442 review):** `NiblePath` silent truncation at depth-16 (deep medical/bio chains collide), `StructuralSignature` u32 (birthday-collision among shape-families across many domains), the DOLCE nibble/byte addressing. None bites the 2 curated domains; all bite at full multi-domain scale — name each as a conscious freeze before the load that needs it. Gated on D-ARM-7 (`jc::jirak`) before any domain's feed writes the store.

**Cross-ref:** `owl-dolce-hhtl-compartments-aerial-fed.md`, `wikidata-hhtl-load.md`, `ogit-owl-dolce-ontology-compartments.md`, `splat-codebook-aerial-wikidata-compression.md`; #441 (D-CLS), #442 (Wikidata-HHTL), #438 + Phase 2 (aerial feed); the D-CLS↔D-ARM-14 convergence FINDING; `cognitive-risc-classes.md` N4.

## 2026-05-31 — FINDING (research): arm-discovery is a PROPOSER that FEEDS the SPO-AST, not the SPO-AST itself — using it AS the AST would conflate proposer↔hub + push similarity into addressing

**Status:** FINDING (read-only research, answering "can lance-graph-arm-discovery be the SPO-AST?"). No code; prevents a layering mistake.

**Core-doc SPO-AST = the HUB** (cognitive-risc-core "AST is the hub"): one canonical AST; Elixir + OWL/DOLCE/OGIT/Odoo all lower INTO it; a move/rule/inference = a **guarded rewrite over SPO state**, same node shape across domains; lowers OUT to SurrealQL + planner candidates.

**arm-discovery = the upstream PROPOSER leg** (verified, lib.rs:4 "the upstream proposer"): emits flat `CandidateRule`s (antecedent→consequent associations via codebook-probe; rule.rs `Proposer` trait), tagged `ArmDiscovered`. NO AST node type exists in the crate (grep: encode/codebook are distance/probe, not trees). Predicates are skeleton-only (`rdf:type`/`subClassOf`, ontology.rs); `Implies`/`CoOccursWith` NOT in vocab yet (D-ARM-SYN-1 deferred, ndjson.rs:16).

**Verdict — NO, not as the SPO-AST; YES as a feeder:**
- **Why not the AST:** (1) it's a proposer, not the hub — core-doc: "business logic is just one proposer's candidates… same candidate object, differing only by discovery_origin." AstWalker (OWL/Odoo) is a DIFFERENT proposer; both feed the hub. Using arm-discovery AS the AST conflates the proposer layer with the hub layer. (2) its codebook-probe is SIMILARITY/lossy (ANN-shaped); the faiss-homology iron rule: "similarity lives ONLY in the proposer/discovery layer, never in addressing/structure." An AST node is structure → must be exact (CAM), not similarity. (3) it emits flat rules, not a guarded-rewrite TREE.
- **What it legitimately does (the synergy, already mapped in aerial-arm-ruff-spo-codegen-synergies.md):** `CandidateRule` → `ruff_spo_triplet::Triple{s,p,o,f,c}` (needs `Implies`, D-ARM-SYN-1) = ONE input stream to the hub, the `ArmDiscovered`-provenance candidates. arm-discovery is the runtime-data proposer; ruff_spo_triplet is the triple contract it emits into; the AST hub consumes that + AstWalker + the existing `lance-graph LogicalOperator` polyglot IR.

**The actual SPO-AST gap:** the guarded-rewrite AST node type does NOT exist yet. It would live in contract (or a new IR module), be CAM-addressable (exact identity, zero-float — classes.md CAM invariant), and CONSUME candidates from {arm-discovery (ArmDiscovered), AstWalker (Extracted), LLM (conjecture), the polyglot LogicalOperator}. The `discovery_origin` u8 (core-doc, ISA-width-at-risk) is exactly the proposer-tag that lets them coexist as one candidate object.

**Cross-ref:** cognitive-risc-core "AST is the hub" + "business logic is just one proposer's candidates" + discovery_origin u8; faiss-homology-cam-pq "similarity proposer-only, never addressing"; `arm-discovery/src/{lib.rs:4,rule.rs}` (proposer); `aerial-arm-ruff-spo-codegen-synergies.md` (the feed mapping + D-ARM-SYN-1 Implies); `ruff_spo_triplet::Triple` (the emit contract); lance-graph `LogicalOperator` (the polyglot IR, a sibling hub-input).
## 2026-05-31 — SHIPPED D-CLS-RENDER + PLANNED Wikidata-HHTL (the N4 second-domain falsifier)

**Status:** D-CLS-RENDER SHIPPED-in-PR; Wikidata-HHTL = PLANNED (next arc, the classes.md:N4 falsifier).

**D-CLS-RENDER (shipped):** `ClassView::render_rows(class,mask) -> Vec<RenderRow{label,predicate}>` — the off-bits-skipped render surface (classes.md:49). Presence-ONLY (C2): a row appears iff its bit is set; the mask never changes meaning. Template-agnostic — an askama per-class template iterates these rows; the engine (F3=askama, OD ratified) is the deferred render-crate Wave (the plan's Wave-4 multi-agent, not a solo slice). Review->fix: clippy -D warnings caught a doc-list-indent lint (line starting with `+` read as a bullet) -> reflowed. 497 contract lib green; class_view clippy+fmt clean. Completes the CONTRACT side of the XML-parse stack: SoA=doc, ObjectView=XSD, ClassView=parser, FieldMask=presence, render_rows=the XSLT output rows.

**Wikidata-HHTL — the next arc (planned, why it matters):** classes.md N4 = "don't freeze the SoA schema until >=2 genuinely different domains run through it"; faiss-homology closed-picture = "chess + Odoo + Wikidata-anatomy all run through the same Class+SoA+HHTL+CAM with no special-case." Odoo is domain-1 (D-CLS shipped). **Wikidata HHTL is the SECOND-DOMAIN FALSIFIER** that proves FieldMask/StructuralSignature/ClassView are universal, not Odoo-cosplaying. DIRECT REUSE: wikidata facet-bitmask = my `FieldMask`; shape = my `signature()`; (class_id, shape_hash, presence_bitmask) persisted row = exactly the D-CLS triple.

**The right SMALLEST slice (not the 115M load — that's a streaming pipeline):** the wikidata-hhtl doc's bring-up = the **HHTL 16^n nibble router + OWL/DOLCE facet template**. Concretely shippable + reuses D-CLS: (a) `HhtlPath` (16^n nibble sequence = the P279/subClassOf Abstammungs-path, the ONE tree axis; bit-shift routing, O(1)); (b) the OWL-construct -> HHTL-form table as a typed mapping (subClassOf->path, closed-ObjectProperty->facet-bit=my FieldMask, DatatypeProperty->Quartett slot, disjointWith->collision-free additive); (c) mask-inherits-along-path-as-DELTA (the leaf stores only its increment over the parent — prevents the union disease). zero-dep contract types + a tiny falsifiable test (a 2-level P279 path + a facet, assert the leaf mask = parent-delta). DEFER: the streaming json.gz loader, the 115M scaling, the reasoning Derived store.

**Cross-ref:** D-CLS arc (FM/RES/SIG/AUDIT/RENDER, the reused machinery); wikidata-hhtl-load.md (the pipeline); faiss-homology-cam-pq.md (facet u64 = SoA facet column = FieldMask; HHTL=IVF cell); classes.md N3/N4 + :49; cognitive-risc-core (HHTL=Schedule-layer bucket router).
## 2026-05-31 — SHIPPED-in-PR: D-CLS-AUDIT — curated-corpus shape-family audit (classes.md:42 CONFIRMED on real data, falsifiably) + clippy fix

**Status:** SHIPPED-in-PR (#440 D-CLS Wave-2 input). Extends D-CLS-SIG from one lane to the full curated corpus.

`class_signature` gains: `curated_entities()` (concat all 15 l1..l15 ENTITIES = 64 curated consts), `corpus_summary() -> FamilySummary{entity_count, family_count, largest_family}`, and the FALSIFIABLE teeth-test `discovered_taxonomy_collapses_entities_to_fewer_families` — over the REAL 64 curated entities, asserts `family_count < entity_count` (classes.md:42 "entities are ~dozens of shape-families" — CONFIRMED on actual data, not asserted on a fixture; the test would FAIL/surface if the signature were too fine or the claim false for this corpus) + largest-family >=2 members. This is the Wave-2 discovery input: run it, get the named shape-families table (the human/spec-owner naming step).

**Review->fix (the ///-pipeline working again):** clippy `-D warnings` flagged an unused `FieldMask` import at `class_resolver.rs:32` (from D-CLS-RES — it was only used in the test module via super::*). Fixed: moved the import into `#[cfg(test)] mod tests`. This would have FAILED the CI clippy gate (the same gate that bit M1) — caught + fixed pre-push. 6 class_signature + 240 ontology lib green; my files clippy-clean (remaining warnings are pre-existing oxrdf/cargo-toml/ndarray, not mine).

**The D-CLS arc, end-to-end now:** D-CLS-FM (contract FieldMask+ClassView) -> D-CLS-RES (resolver over live cache) -> D-CLS-SIG (signature + object_view bit-basis) -> D-CLS-AUDIT (the corpus discovery, falsifiable). Remaining: name the ~N families (Wave-2 human step over corpus_summary output); the askama render crate (project() -> off-bits-skipped HTML); the registry by_entity_type_id O(1) index.

**Cross-ref:** D-CLS-SIG (extended); #440 plan Wave-2; classes.md:42-44 (discovered taxonomy, now falsifiably confirmed); the M1 clippy-erasing-op lesson (CI -D warnings gate — caught the unused import the same way).
## 2026-05-31 — SHIPPED-in-PR: D-CLS-SIG — class_signature (the HONEST discovered-taxonomy; structural-hash group-by, not aerial-cluster vaporware)

**Status:** SHIPPED-in-PR (#440 D-CLS). The corrected D-CLS-2 + D-CLS-3 — replaces the brutal-review-killed "Aerial+ clusters entities" vaporware with the deterministic group-by classes.md:43 actually prescribes ("group-by-on-structural-hash OR Aerial+").

`lance-graph-ontology::odoo_blueprint::class_signature` (pure const analysis, no hot path):
- **`signature(&OdooEntity) -> StructuralSignature(u32)`** — FNV-1a (mirrors the workspace `style_recipe::fnv1a_recipe` idiom) over the canonicalized structural tuple: `[kind_disc, field-kind-histogram x6, method-kind-histogram x5, has_state_machine]`. Deterministic + NAME-INDEPENDENT (groups by structure, not model name). Two entities with the same signature ARE the same shape-family (classes.md:43 discovered taxonomy).
- **`object_view(&OdooEntity) -> ObjectView`** — derives the per-class field-SET (the real FieldMask bit-basis): field position i = declared-field i = stable bit i (N3 append-only); primary_label = first textual field; template by size (<=4 Card else Detail); capped at FieldMask::MAX_FIELDS(64). **This FILLS the supplied-placeholder the class_resolver (D-CLS-RES) took** — the two slices now compose: signature→family, object_view→the bit-basis RegistryClassView consumes.
- **`shape_families(&[OdooEntity]) -> Vec<(sig, members)>`** (BTreeMap-sorted deterministic) + **`audit(...)`** rows.

**Honesty (brutal-review corrections folded in):** deterministic group-by, NOT aerial-cluster (aerial mines RULES not entity-clusters — confirmed no clustering entry point); scoped to the curated l-lane consts (not the false-66); FNV collisions are intentional (same structure→same family). 4 teeth-tests over REAL l1::ENTITIES (determinism, name-independence, bit-basis derivation, group-completeness). 238 ontology lib green; clippy+fmt clean.

**Composes the D-CLS arc:** D-CLS-FM (contract FieldMask+ClassView) ← D-CLS-RES (ontology resolver over the cache) ← D-CLS-SIG (the field-set + shape-families the resolver needs). Next: run the audit over all curated l-lanes → name ~10-15 families (human/spec-owner step); the askama render crate consuming project(); the registry by_entity_type_id O(1) index.

**Cross-ref:** D-CLS-FM/D-CLS-RES (the arc this completes the input for); #440 plan §D-CLS-2/3; brutal-review verdict (aerial-cluster→group-by correction); classes.md:41-44 (discovered taxonomy) + :48 (delta bitmask); `style_recipe::fnv1a_recipe` (the hash idiom reused).
## 2026-05-31 — SHIPPED-in-PR: D-CLS-RES — class_resolver (ontology-side impl ClassView; the meta-DTO flies over the LIVE OGIT cache)

**Status:** SHIPPED-in-PR (#440 D-CLS). Makes the contract `ClassView` trait LIVE — the "OGIT hashtable single-lookup → class meta-lookup" upgrade, done.

`lance-graph-ontology::class_resolver::RegistryClassView<'a>` impls `contract::class_view::ClassView` over a borrowed live `OntologyRegistry`: `class_id → shape`. DOLCE resolved LATE from the cache (`enumerate_first_with_entity_type_id(class) → MappingRow → ogit_uri → classify_odoo`), never stored on the row (OD-DOLCE "use the ontology cache" ratified). `dolce_id::{ENDURANT,PERDURANT,QUALITY,ABSTRACT}` = the stable u8 ids the contract trait returns (contract has no DOLCE enum; consumer maps back). Dep-inversion: contract owns vocabulary, ontology owns answers (ontology already deps contract).

**Honest scope (no fabrication):** resolves class existence + DOLCE + template from the live cache; the per-class field-SET (ObjectView, the bit-basis) is SUPPLIED, not enumerated — a MappingRow is a single entity's leaf row with no field-list. Field enumeration = the deferred D-CLS structural-signature audit (scope: 64 curated consts). No field-set fabricated.

**Review→fix caught a real perf gap (the ///-review-fix pipeline working):** `registry::enumerate_first_with_entity_type_id` is an O(n) row-scan + FULL MappingRow clone — called per dolce_category_id would be O(n)-with-heavy-clone per render. Fixed at MY layer: per-class `RefCell<HashMap>` memo (DOLCE is stable per class → scan once, not per call) + documented the underlying registry `by_entity_type_id` index as a deferred registry slice. No registry edit (collision avoidance). 4 teeth-tests (incl memo-stability) + 234 ontology lib green; clippy+fmt clean.

**Next (deferred D-CLS):** the structural-signature audit (64 curated OdooEntity → ObjectView field-sets + shape-family group-by-on-structural-hash, NOT aerial-cluster); the registry `by_entity_type_id` O(1) index; the askama render crate consuming `project()`.

**Cross-ref:** D-CLS-FM (the contract trait this implements); #440 plan; OD-DOLCE ratification (cache-resolves); `registry::enumerate_first_with_entity_type_id` (the O(n) gap, memoized); `hydrators::dolce_odoo::classify_odoo` (the live DOLCE resolution); classes.md:39 (resolve-not-store).
## 2026-05-31 — SHIPPED-in-PR: D-CLS-FM — class_view (FieldMask + ClassView meta-DTO; the class flies ABOVE the agnostic SoA)

**Status:** SHIPPED-in-PR (#440 D-CLS contract foundation). The XML-parse framing made real, OD-gates ratified.

`contract::class_view` (zero-dep): `ClassId(u16)` (reuses soa_view::class_id width, OD-CLASSID-WIDTH) + `FieldMask(u64)` presence bitmask (of/with/has/count, C2 presence-NEVER-semantics, N3 stable append-only positions) + **`ClassView` resolver TRAIT** (`fields`/`template`/`dolce_category_id`/`field_label`/`project`) + `ClassProjection` iterator. EXTENDS the existing `ontology::ObjectView`/`FieldRef`/`DisplayTemplate` (the per-class ordered field set = the bit basis), does NOT duplicate.

**The architecture (user's framing):** OGIT today = hashtable single lookups (uri→row); the class = a META lookup (class_id→shape: ordered fields+labels+template+bit-basis), composing many leaf lookups. XML map: SoA row=XML doc (agnostic bytes), ObjectView=XSD, ClassView=parser+schema, FieldMask=which optional elements present, askama=XSLT. **Classes fly as a meta-DTO ABOVE the SoA so the SoA stays agnostic — zero labels in the bytes; labels/template/DOLCE resolved LATE from the OGIT cache at projection** (classes.md:39 resolve-not-store; core inv #1 nothing-semantic-in-register). C2 falls out free: bit=presence (on SoA, structural); bit→field→label=resolution (above SoA, semantic).

**Layering (dep-inversion like MailboxSoaView):** contract=agnostic surface (FieldMask + ClassView trait, zero-dep); ontology=implements ClassView (the parser, resolves labels from OGIT hashmap — DOLCE-from-cache per OD ratification); render=consumes project()+template, skips off-bits. 3 teeth-tests: presence-bits, meta-DTO-projects-above-agnostic-(class,mask), late-label-resolution. 496 contract lib green; clippy+fmt clean.

**Next (deferred, the D-CLS waves, P0-corrected):** ontology-side `impl ClassView` over OntologyRegistry (the resolver/parser); D-CLS-2 structural-signature audit (scope: 64 curated consts, NOT the false-66/real-381 — brutal P0-1); D-CLS-3 deterministic group-by-on-structural-hash (NOT aerial-cluster vaporware); FieldPositionTable freeze-append-only-on-first-emit (AP2); the render crate (askama). class_id stays a discriminator OUTSIDE the CAM content layer.

**Cross-ref:** #440 plan; OD-gate ratifications (2026-05-31); REVIEW VERDICT P0/AP2; `ontology::ObjectView` (extended); `soa_view::class_id` (#437, reused); classes.md:39/48/49 (resolve / delta-bitmask / off-bits-skip); MailboxSoaView (the dep-inversion precedent #437).
## 2026-05-31 — OD-GATES RATIFIED (spec owner) for odoo-classes-bitmask-render-v1 (#440): DOLCE-from-cache (dissolves the 6-vs-4), ClassId u16 reuse-existing, kind+class_id both DTO-views, askama

**Status:** RATIFICATION (spec owner = user, 2026-05-31). Unblocks the plan's Wave-0 pre-conditions, WITH the review-verdict P0 corrections folded in. Two answers reframe the plan, not just answer it.

- **OD-TEMPLATE-ENGINE → askama** (compile-time, classes.md:72 lean). Per-class `.html.j2` compiled into the new excluded render crate. RATIFIED as-planned.
- **OD-CLASSID-WIDTH → ClassId(u16), REUSE the existing accessor.** Wrap/align `lance-graph-contract::soa_view::class_id() -> &[u16]` (N1 hook, #437) — do NOT mint a colliding 2nd newtype (the brutal-tester P0-2). Discriminator stays OUTSIDE the CAM content-hash layer (I-VSA-IDENTITIES; never hashed-as-content). RATIFIED with the reconciliation constraint.
- **OD-DOLCE-CANONICAL → DISSOLVED: "use the ontology cache."** The spec owner's ruling reframes it: DolceCategory is NOT a new canonical contract enum to pick among 4 — it is a **resolved attribute of the OGIT class in the ontology cache** (`OntologyRegistry` MappingRow). The 6-vs-4 variant mismatch (the brutal-tester P0-3) EVAPORATES: there is no enum beauty contest; the cache resolves class_id → DOLCE category as one more field. The 4 duplicate enums become CONSUMERS of the cache's answer, not competitors. D-CLS-1 changes from "pick canonical + 4-way From-map" to "the cache is the source; the local enums read from it." (Honors classes.md:39 "the meta-DTO resolves; it does not store" + classes.md:2 "CAM, not ANN" identity-resolution.)
- **OD-CLASSID-VS-ENTITYKIND → DISSOLVED: "it's a DTO; Odoo becomes a custom view."** The spec owner's ruling: class_id and kind are NOT competing discriminators to reconcile — the **meta-DTO RESOLVES both as views over the same row**. Odoo's `kind {Model,Transient,Abstract}` is just one more projected field; class_id is another. So they COEXIST not by compromise but because neither is special — both are DTO-resolved views (classes.md thesis: "Odoo becomes a custom view"). No replace-vs-coexist tradeoff; the DTO subsumes the question.

**Unifying principle (both rulings):** the meta-DTO RESOLVES (logic), it does not STORE (state) — classes.md:39. DOLCE-from-cache and kind-as-view are the same move: resolve attributes over the row via the ontology cache, don't bake competing canonical enums/discriminators into the bytes.

**Wave-0 status:** all 4 OD gates resolved. Still BLOCKING on the review-verdict P0 fixes BEFORE agent spawn: (P0-1) 381 entities not 66 — scope decision; (P0-2) reuse soa_view::class_id (now ratified); (P0-3) DOLCE-from-cache (now dissolves it); (P1) D-CLS-3 → deterministic group-by-on-structural-hash not aerial-cluster (aerial mines rules); (AP2) FieldPositionTable freeze-append-only-on-first-emit. With these folded in, Wave-1 (D-CLS-2 audit / D-CLS-4 render skeleton / D-CLS-5 ClassId-reuse) is spawnable.

**Cross-ref:** #440 plan §2 (OD gates) + §4 (D-CLS); the 2026-05-31 REVIEW VERDICT (P0 errors); `soa_view.rs:61` (existing class_id, reuse); `lance-graph-ontology/registry.rs` (OntologyRegistry = the DOLCE-resolving cache); classes.md:39 (resolve-not-store) / :13 (one class_id keys three things).
## 2026-05-31 — REVIEW VERDICT (council+brutal) of odoo-classes-bitmask-render-v1 (#440): HOLD — 3 P0 factual errors + 1 vaporware deliverable; plan must NOT spawn agents as-written

**Status:** FINDING / plan-correction (5-savant-council + 3-brutal review the plan's §2 + user requested, 2026-05-31). The merged plan #440 is sound in DOCTRINE (classes.md §"Jinja=classes+presence bitmask" verbatim) but has load-bearing factual errors vs on-disk source. Corrections required before Wave-1 spawn. Plan file is append-only governance — these corrections are dated board entries, NOT edits to the plan.

**P0 ERRORS (verified against source, file:line):**
1. **Entity count is 381, NOT 66.** 64 curated consts in `odoo_blueprint/l{1..15}.rs` + **317 `EXT_*`** in `odoo_blueprint/extracted/` (compiled: `mod.rs:77 pub mod extracted;`). D-CLS-6 adds `class_id` to the SHARED `OdooEntity` struct → ALL 381 back-fill, not 66. D-CLS-2/9 "iterate the 66" is ~6× off — either scope to curated-64 explicitly (defensible: extracted/mod.rs:5 "additive, curated stays canonical") OR own all 381. Decide; don't silently ignore 317.
2. **`class_id` ALREADY EXISTS** — `lance-graph-contract/src/soa_view.rs:61` `fn class_id(&self)->&[u16]` + `class_id_at` (the N1 hook shipped #437, aliases entity_type u16). D-CLS-3/5's new `ClassId(u16)` newtype COLLIDES in the same crate. Must RECONCILE with the existing accessor (newtype wrapping the same u16, or extend the accessor), not mint a second.
3. **Canonical `DolceCategory` has 6 variants, NOT 4** — `cognition/entity.rs:87`: Endurant/Perdurant/**AbstractObject**/Quality/**Region**/**Other**. NO variant named `Abstract`. D-CLS-1's From-test (plan:105 "preserves Endurant/Perdurant/Quality/Abstract") references a non-existent variant → won't compile. Also `entity.rs:88` has a live `// Stage 2 expands this enum` TODO = actively-edited (AP1 collision). The OD-DOLCE-CANONICAL default-lean must map all 6 (incl Region/Other/AbstractObject) + callcenter `Abstract`→`AbstractObject` + `Unknown`→`Other`.

**P1 VAPORWARE:** D-CLS-3 "Aerial+ structural-hash → 10-15 shape-families" — aerial only exposes `mine()->Vec<CandidateRule>` (association rules, antecedent→consequent), NO entity-clustering/group-by entry point (`aerial/mod.rs:69`; grep cluster/group_by/shape_family = zero code). "Point Aerial+ at Odoo to cluster entities" CANNOT be done with the cited crate. Downgrade D-CLS-3 to a deterministic **group-by-on-structural-hash** (the plan's own risk-register fallback) — that's real and matches classes.md:43 ("group-by-on-structural-hash OR Aerial+").

**IRON-RULE (council, YIELDS-WITH-AP):** No iron rule violated. (a) presence≠facet-code wall HOLDS — `FieldMask(u64)` presence-of-field is correctly distinct from faiss-homology facet-u64 (CAM closed-range codes); keep the wall explicit in D-CLS-7 doc (never AND-ed/superposed). (b) **AP2 landmine:** D-CLS-5/7 FieldPositionTable must **freeze append-only on first emit**, NOT recompute the bit positions from the field-union each build (a re-audit reordering members → bit N changes meaning → old masks misread; N3 + I-LEGACY-API-FEATURE-GATED). Add a golden position-stability test. (c) u16 ClassId fine — discriminator stays OUTSIDE the CAM content-hash layer (never hashed-as-content/superposed). (d) zero-dep boundary sound (arm-discovery local-newtype+TryFrom, no contract dep).

**SHIPPABLE-ONCE-RATIFIED (brutal cut):** D-CLS-2 (signatures, read-only — FIX count to 381 or scope-to-64), D-CLS-4 (render-crate skeleton, isolated), D-CLS-5 (ClassId — ONLY after reconciling soa_view::class_id). PREMATURE until P0 fixed: D-CLS-1 (6≠4 variant map), D-CLS-3 (vaporware→group-by), D-CLS-6/7/8/9 (inherit the 381 + ClassId-dup).

**The 4 OD gates remain blocking (surfaced to user this turn):** OD-DOLCE-CANONICAL (now: map 6 variants), OD-CLASSID-WIDTH (u16 lean OK), OD-CLASSID-VS-ENTITYKIND (coexist lean OK), OD-TEMPLATE-ENGINE (askama lean OK). No agent spawns until ratified.

**Cross-ref:** #440 (the plan); `odoo-classes-bitmask-render-v1.md` §2 (OD gates) / §4 (D-CLS deliverables); `soa_view.rs:61` (existing class_id); `cognition/entity.rs:87` (6-variant DolceCategory); `aerial/mod.rs:69` (mine, not cluster); classes.md:43/50/65 (discovered-taxonomy / presence-not-semantics / N3); iron-rule AP2; the post-#438 4-savant council (#440 `4f381a8`).
## 2026-05-30 — SHIPPED-in-PR: M1 keystone — `Tactic::requires() -> ThoughtMask` (the latent checklist made data; reliability = coverage, extraction not construction)

**Status:** SHIPPED-in-PR #439 (D-MBX-A6-P3-M1). The panel-recalibrated keystone of reliability-checklist-arc-v1, built autonomously.

`contract::recipe_kernels` now has: `ThoughtField` (8-field enum, stable bit positions, append-only per N3) + `ThoughtMask(u8)` (zero-dep bitmask: `of`/`has`/`len`/`is_empty`/`covered_by`) + **`Tactic::requires(&self) -> ThoughtMask` NON-defaulted** — all 34 tactics declare which ThoughtCtx fields their `apply` reads (audited from the bodies: Cr→beliefs, Tcp→candidates+sd, Mcp→confidence+free_energy, Rte→free_energy+rung, …; the 4 algebraic constant-only tactics Are/Zcf/Icr/Hkf legitimately empty). `covered_by` (`required & known == required`) IS the reliability-coverage gate in miniature.

This realizes the creative-explorer M1 insight: reliability is a DECLARED ACCESSOR, not a constructed gate — the checklist was latent in 34 apply() bodies, now reified as data. Makes P1(coverage)/P7(reconcile)/P11(class_id→checklist) DERIVED. Non-defaulted = no silent-empty theater (the council's no-op warning); teeth-test `requires_masks_are_varied_not_a_constant_stub` FAILS on a copy-paste/empty stub (asserts exactly-4-empty + ≥8 distinct masks). 9 recipe_kernels tests + full contract lib green; non-defaulted method safe (the 34 are the only Tactic impls workspace-wide).

**Cross-ref:** reliability-checklist-arc-v1 RECALIBRATION (M1 keystone); E-TEMPLATE-IS-CHECKLIST-IS-DATOMS (requires() = the executable checklist); E-RELIABILITY-IS-CHECKLIST-COVERAGE (covered_by = the gate); E-RELIABILITY-NOT-VALIDITY; `recipe_kernels.rs` ThoughtField/ThoughtMask/Tactic::requires.
## 2026-05-30 — RECALIBRATION (3-agent panel) of reliability-checklist-arc-v1: keystone is M1 `Tactic::requires()->AtomMask` (extraction not construction); P2 needs a corpus it lacks; P5 is P0-blocked; P3/P4 are AP6 theater; P10 off-arc

**Status:** FINDING / plan recalibration (cascade-impact + brutally-honest-tester + creative-explorer, catalyst mode 2026-05-30). Recalibrates `.claude/plans/reliability-checklist-arc-v1.md`. Convergent across all three.

**THE UNLISTED KEYSTONE (creative-explorer M1 — reframes the menu):** the 34 `recipe_kernels::Tactic` impls EACH already declare a latent checklist — every `apply()` reads a DIFFERENT subset of `ThoughtCtx`'s 8 fields (Cr→beliefs, Tcp→candidates+sd, Mcp→confidence+free_energy) but NONE reifies it as data. So reliability is NOT a gate to construct — it's a missing accessor: **`Tactic::requires(&self) -> AtomMask`** (one default method). Build that, and P1 (coverage) / P7 (reconcile) / P11 (class_id→checklist) become DERIVED, not built. "Reliability is already declared 34 times, just not read back — the work is EXTRACTION, not construction." `Tactic::requires()` and the domain `coverage()` are the SAME op at two altitudes (kernel required-fields ↔ `OdooStyleRecipe.atoms`), one bitmask over one basis.

**CORRECTIONS to my menu (verified by source, file:line):**
- **P2 (probe) — the witness corpus DOES NOT EXIST** (grep empty in planner/consumer-conformance). P2's first sub-task = BUILD the corpus. It's still rank-1 (it can honestly return "gate is cosmetic, don't build P1") but it's not free.
- **P5 is P0-BLOCKED, not S-M:** `atoms::I4x32::pack`/`unpack` are BOTH `todo!()` (atoms.rs:83,88); the 32-vs-33 (carrier 32 lanes vs CANONICAL_ATOMS 33) is an unresolved BLOCKED fork (atoms.rs:39-43). argmax-over-I4x32 sits on a panicking carrier + undecided dim. Do NOT build until the dim fork resolves.
- **P3/P4 = AP6 dead-surface THEATER as scoped:** `try_advance_phase` has ZERO production callers (only FakeSoa); `resolve_style`/`reliability_of` have ZERO production callers; `plan()` returns input unchanged. Emitting from plan() repeats the theater the council just caught. P4 = MailboxSoaOwner impl with no ractor caller = AP6.
- **P1 DUPLICATES `recipes::Coverage`** (recipes.rs:54, SPO-2³, 3 variants) — minting a 4th Coverage = drift. Reframe P1 as M1's derived accessor, not a new enum.
- **P6 is COSMETIC:** recipe.rs has no `pub mod recipe` in lib.rs — never compiles in. Deleting already-invisible dead code = wash, not a deliverable. (Downgrades my earlier "delete it" instinct to "leave it / opportunistic.")
- **P9 lance bump is STILL BLOCKED** (lancedb 0.29.0 transitively pins lance =6.0.0) — NOT free-to-land as the menu claimed.
- **P10 polyglot = OFF-ARC** (different thread) — DROP from this arc.

**MISSING items the panel adds:**
- **M1** `Tactic::requires() -> AtomMask` (the keystone above).
- **M2 — checklist-COMPLETENESS auditor (unknown-unknown finder):** static-scan each Tactic's declared `requires()` (M1) vs the fields its `apply()` actually touches (neural-debug already does static scanning); mismatch = a dark requirement nobody listed. The ONLY item that finds missing checklist ITEMS (not missing evidence) — the hot-path complement to the cold Stockfish validity gate.
- **M3 — version-arc-as-kanban scheduler wire** (E-SUBSTRATE-IS-THE-SCHEDULER): make the substrate emit the schedule; P3's KanbanMove is the latent hook.

**RECALIBRATED ORDER (replaces the menu's open sequence):**
1. **M1** — `Tactic::requires() -> AtomMask` default method + a teeth-test (assert distinct Tactics declare distinct masks; fails if all-same/empty). Pure contract, zero-dep, extraction not construction. THE keystone.
2. **P2 + its corpus** — build a small witness/recipe corpus, then probe: does coverage state (M1-derived) change a Rubicon terminal vs DAG-only? Honest pass/fail; can return "cosmetic, stop."
3. **M2** — completeness auditor over M1's masks (static, reuses neural-debug). Finds unlisted-requirement gaps.
- DEFER behind P2-pass: P1-as-derived-coverage, P11 O(1) index. DROP/leave: P5 (P0-blocked), P6 (cosmetic), P10 (off-arc), P9 (still blocked). GATE P8 on P2. P3/P4 only after a real consumer exists (no theater).

**Brutal's non-theater-now set was {P2,P9,P11}; creative's minimal-core was {P5,M1}; cascade's first-3 was {P2,P6,P1}.** Synthesis: **M1 is the keystone all three implicitly point at** (cascade's P1, creative's M1, brutal's "P1 dups Coverage→reframe") — start M1, then P2-with-corpus, then M2. P5 explicitly deferred (P0). 

**Cross-ref:** `.claude/plans/reliability-checklist-arc-v1.md` (the menu, now recalibrated); E-TEMPLATE-IS-CHECKLIST-IS-DATOMS + E-RELIABILITY-IS-CHECKLIST-COVERAGE (M1 is their executable form); `recipe_kernels::{Tactic,ThoughtCtx}` (the latent checklists); `recipes::Coverage` (the dup P1 must not repeat); `atoms.rs:83,88,39-43` (P5 todo!()+32/33 block); neural-debug (M2's static scanner); E-SUBSTRATE-IS-THE-SCHEDULER (M3).

---

## 2026-05-30 — E-TEMPLATE-IS-CHECKLIST-IS-DATOMS — the NARS/elixir reasoning template, the per-domain checklist, and the Odoo D-Atoms are ONE object; reliability = the template's required atoms are LIT (known). #433 already built half of it.

**Status:** FINDING — major unification (user 2026-05-30: "the checklist IS the NARS/elixir reasoning templates; in Odoo terms tax classes, billable hours, account type, etc."). Collapses three things I'd treated as separate, and re-connects #433 (which I'd mis-filed as "unrelated Odoo codegen").

**The unification (one object, three faces):**
- **NARS/elixir reasoning template** = the EXECUTABLE form (the recipe/`Tactic` that fires; recipe_kernels = "the Elixir-like recipe layer").
- **Per-domain checklist** = the COVERAGE form (the required items to evaluate) — `E-RELIABILITY-IS-CHECKLIST-COVERAGE`.
- **Odoo D-Atoms** = the INSTANCE form (the actual domain fields). A template DECLARES its required inputs → those required inputs ARE the checklist → the checklist items ARE the domain's evaluable fields. Not three systems — one.

**#433 ALREADY BUILT HALF OF THIS (grounded, file:line — I mis-filed it as unrelated):**
`lance-graph-ontology/src/odoo_blueprint/style_recipe.rs`:
- `enum DAtom` (:116) = the checklist-item catalogue, as REAL variants: `FiscalCtx`, `Money`, `ApplyRate` (VAT/currency rate, `OdooSemanticRole::Tax`), `Quantity`, `EmitAmount`, `Compute`, `Validate`, `Onchange`, `Event`, `Action`, `Entity`, `Law` (`regulation_iri` = UStG §12 / EU VAT). ⇒ EXACTLY the user's "tax classes / billable hours / account type" — domain fields that must be evaluated.
- `OdooStyleRecipe { method_id, atoms: Vec<(DAtom,u8)>, regulation_iris, return_kind, recipe_id }` (:209) = the TEMPLATE-AS-CHECKLIST: `atoms` = which checklist items this template REQUIRES (+weights). `recipe_id` = FNV-1a content-address over sorted atoms → equivalent templates collapse (CAM dedup).
- **The 5-lit / 6-dark atom split (#433 honest-flag)** = the knowns/unknowns COVERAGE BITMASK, already present: 5 atoms fire today (Entity/Compute/Validate/Onchange/Action — kind-driven = KNOWN); 6 dark (Money/Quantity/ApplyRate/EmitAmount/Event/FiscalCtx — gated on Stage-2 extractor = UNKNOWN until populated). lit-vs-dark IS the coverage bitmask. "Atoms flip from stage2→must set when Stage-2 lands" = checklist boxes going from unknown→known.

**So reliability = required atoms LIT:** a template's `atoms: Vec<(DAtom,u8)>` are the required checklist; an instance's populated Odoo fields LIGHT them; reliability-to-Commit = `required_atoms ⊆ lit_atoms` (the `required & known == required` AND-test, E-RELIABILITY-IS-CHECKLIST-COVERAGE). Plan = named dark atoms remain (the Stage-2 gap); Prune = required atom unsatisfiable.

**Elixir open/closed maps exactly** (recipe.rs E-LADDER §2): add-FIELD = data (a new checklist box / D-Atom population — no recompile); add-TEMPLATE = structure (a new required atom-set / OdooStyleRecipe — register it). The hot-load split IS the checklist-vs-template-evolution split.

**Consequence / correction:** #433's `OdooStyleRecipe` is NOT "unrelated Odoo codegen" (my earlier filing in the 3-recipe-module finding) — it's the DOMAIN-INSTANCE face of the reasoning-template-as-checklist. The cross-domain generalization: a Rails/medical/chess frontend writes its OWN `style_recipe.rs` (its own D-Atom catalogue = its own checklist) over the same triplet shape (#433 doc: "A Rails frontend writes its own style_recipe.rs"). So D-Atoms are the per-domain checklist; the NARS reasoning template is the domain-agnostic shape; reliability-coverage is uniform across domains.

**Build implication:** the cheap checklist gate (E-RELIABILITY-IS-CHECKLIST-COVERAGE) does NOT need a new checklist type — it READS `OdooStyleRecipe.atoms` (required) vs the instance's lit-atom bitmask (known). First cut: a `coverage(required: &[DAtom], lit: bitmask) -> CoverageState{Covered|Gap(dark)|Unsatisfiable}` over the existing D-Atom catalogue, generalized to any domain's atom enum. Reuses #433 wholesale.

**Cross-ref:** #433 `odoo_blueprint/style_recipe.rs` (DAtom :116 / OdooStyleRecipe :209 / 5-lit-6-dark); E-RELIABILITY-IS-CHECKLIST-COVERAGE (coverage = reliability); E-RELIABILITY-NOT-VALIDITY (Stockfish audits checklist COMPLETENESS = unknown-unknowns); recipe_kernels "Elixir-like layer"; recipe.rs E-LADDER-SERVES-MAILBOX §2 (open/closed); the 3-recipe-module finding (CORRECTED: OdooStyleRecipe = the domain-instance face, not unrelated); cognitive-risc-classes (class_id→checklist, HHTL-inherited).

---

## 2026-05-30 — E-RELIABILITY-IS-CHECKLIST-COVERAGE — the cheap RISC alternative to psychometric calibration: reliability = (required rungs/checklist-items COVERED) over a knowns/unknowns SoA bitmask, AND-tested in one cycle. No float, no corpus.

**Status:** FINDING + BUILD-DIRECTION (user 2026-05-30: "cheap and efficient alternative — 10-layer rungs ladder + a checklist per domain of what needs evaluation; reasoning has a normalized set of info with validation across knowns/unknowns as SoA"). The cheap structural alternative to E-CALIBRATE-RELIABILITY-PSYCHOMETRICALLY — complementary, not competing.

**The reframe:** reliability becomes STRUCTURAL + PRIOR, not a post-hoc statistic. Instead of measuring Cronbach α over a corpus, make it COVERAGE of a normalized evaluation set:
- **10-rung ladder = the normalized DEPTH axis.** `pearl_rung: u8` (1..=9, +0) ALREADY exists on `ThoughtCtx` (recipe_kernels.rs:36-37), proprioception, world_map, cognitive_shader (0..9), SPO triplet, CausalEdge64; `recipes::Recipe.tier` = Sun et al. reasoning-ladder difficulty. Doctrine: E-LADDER-SERVES-MAILBOX. The ladder is real + threaded; this REUSES it.
- **Per-domain checklist = the EVALUATION axis (x).** Each domain (class_id) declares WHICH rungs/items must be evaluated. The checklist is `class_id`-keyed and inherited along the HHTL path (like labels/columns/templates — the cognitive-risc-classes triangle), so domains don't hand-roll it.
- **knowns/unknowns as SoA = a presence bitmask** (= cognitive-risc-classes N3 "stable per-class bitmask, append-only, bit=field-N-populated"). Reliability-to-Commit = `required & present == required` — a SIMD batch-AND popcount over the SoA column, ONE cycle (the 0xFFF/facet-AND efficiency).

**Why it's better-fit than calibration here:** (1) NO float, NO offline corpus, NO calibration pass — just bitmask coverage. (2) DISSOLVES the threshold problem the iron-rule-savant flagged: no 0.2/0.8/0.15/0.35 to calibrate OR Jirak-bound — the Rubicon 3-way maps onto COVERAGE STATE not a magnitude: Commit = required checklist covered; Plan = named known-UNKNOWNS remain (re-deliberate to fill them); Prune = checklist cannot be satisfied. (3) It's the `class_id`→checklist projection — one more payoff off the discriminator the SoA already needs (N1).

**knowns vs unknowns = the enumerable-gap axis (MUL/Dunning-Kruger):** a *known-unknown* = an unchecked box you can NAME (→ Plan); the checklist makes unknowns ENUMERABLE — you can't be confidently-wrong about a box you know is empty. This is reliability-as-coverage doing the epistemic-humility work the φ⁻¹ ceiling did, but structurally + cheaply.

**Honest tension (not glossed):** a checklist only covers known categories — an UNKNOWN-UNKNOWN (a required-but-UNLISTED item) is invisible to coverage. That is EXACTLY the cold-path VALIDITY gate's job (E-RELIABILITY-NOT-VALIDITY): the bring-up test (chess/Stockfish, domain ≥2) FALSIFIES the checklist — finds the box nobody listed. So coverage = cheap HOT reliability gate; Stockfish/oracle = cold validity gate that audits checklist COMPLETENESS. Complementary to the psychometric path: use checklist-coverage for the hot per-cycle gate; reserve Cronbach/ICC for offline auditing whether the checklist items themselves cohere.

**Cheap build (vs the calibration build):** add a `class_id`-keyed per-domain checklist (which rungs/items required) + a `coverage` bitmask column on the SoA (knowns); the Rubicon gate = `required & known == required` AND-test + popcount for the Plan/gap signal. Reuses rung (exists), bitmask (N3 exists), class_id (N1, #439 hook exists). NO new float, NO thinking-engine dep — lighter than the psychometric slice. Sequence: this is the DEFAULT cheap gate; psychometric calibration is the heavier offline audit when a domain's checklist itself is in question.

**Cross-ref:** E-RELIABILITY-NOT-VALIDITY (reliability vs validity split — this is the cheap reliability gate, Stockfish stays the validity gate); E-CALIBRATE-RELIABILITY-PSYCHOMETRICALLY (the heavier alternative this undercuts for the hot path); E-LADDER-SERVES-MAILBOX (the rung doctrine); cognitive-risc-classes N1 (class_id) + N3 (stable per-class presence bitmask) + the HHTL-inherited checklist; `recipe_kernels.rs:36 ThoughtCtx.rung`; `recipes::Recipe.tier`; MUL/Dunning-Kruger (known-unknown enumeration); iron-rule-savant VIOLATES-I-NOISE-FLOOR-JIRAK (dissolved, not calibrated).

---

## 2026-05-30 — E-CALIBRATE-RELIABILITY-PSYCHOMETRICALLY — replace the hand-tuned Rubicon (f,c)/SD thresholds with MEASURED psychometric reliability (Cronbach α / ICC / Spearman / Pearson) — the existing crates, applied brutally to the gate

**Status:** FINDING + BUILD-DIRECTION (user 2026-05-30: "be brutal and use psychometry calibration"). Follows directly from E-RELIABILITY-NOT-VALIDITY: if (f,c) is a RELIABILITY coefficient, calibrate it with real reliability statistics, don't hand-tune it. Resolves the iron-rule-savant's VIOLATES-I-NOISE-FLOOR-JIRAK (uncited 0.2/0.8/0.15/0.35 thresholds).

**The existing psychometric machinery (grounded, file:line):**
- `thinking-engine/src/cronbach.rs` — `cronbach_alpha(items:&[&[f32]]) -> f32` (TESTED; α-identity=1.0 test) + `CronbachResult`/`cronbach_analysis` with the canonical bands (>0.90 excellent/redundant, 0.70-0.90 acceptable, <0.70 poor, <0.50 unacceptable) + `variance_agreement_scores`. Its OWN doc: "replaces the BF16 ±0.008 heuristic with empirical cross-model test" — i.e. it ALREADY swapped a hand-tuned threshold for a psychometric one. We do the same to the Rubicon gate.
- `jc/src/probe_p1_gamma_phase.rs::spearman_rho(&[usize],&[usize]) -> f64` (rank correlation; identity=1/reverse=-1 tested).
- `thinking-engine/examples/codebook_pearson.rs` (Pearson); `calibrate_lenses.rs` (Spearman ρ + ICC); `reencode_safety.rs`/`ground_truth.rs` (the calibration family).
- bgz-tensor calibration suite: `bin/cam_pq_calibrate.rs`, `quality.rs`, `variance_audit.rs`, `similarity.rs`.

**The brutal move:** the Rubicon RELIABILITY gate (Evaluation→{Commit|Plan|Prune}) thresholds — currently hand-tuned `(f,c)` expectation + CollapseGate SD (FLOW<0.15/HOLD/BLOCK>0.35) + recipe_kernels 0.2/0.8 + the style confidence_thresholds (Skeptical 0.95 in learning/cognitive_styles) — get CALIBRATED, not guessed:
- **Cronbach α** = internal consistency of the witness arc / multi-recipe / multi-lens measurement. A belief is reliable-enough-to-Commit when the items (recipe outcomes / lens distances / witness emissions) cohere (α above a measured band), not when c>0.8 by fiat. Per-mailbox or per-cohort α over the CausalEdge64 (f,c) emission arc.
- **ICC** = inter-rater (inter-recipe / inter-mailbox) agreement — the cross-mailbox consensus the a2a_blackboard quorum needs.
- **Spearman/Pearson** = does the reliability ranking track an external criterion (the validity bridge — Spearman vs Stockfish/oracle ranking; this is where reliability MEETS validity, measured not assumed).

**Float-boundary doctrine preserved (your CAM-PQ rule):** psychometric calibration is OFFLINE FLOAT (Cronbach/ICC/Pearson over a corpus, in thinking-engine — heavy, std), emitting a FROZEN threshold artifact the HOT path reads as an integer/const. Same shape as jc certifies the codebook offline → aerial reads online. So the calibrated reliability bands replace the hand-tuned consts; the hot Rubicon gate stays integer/cheap.

**Dependency boundary (respect):** contract + planner are ZERO-/light-dep and do NOT dep thinking-engine. Calibration lives at the cognitive-shader-driver / thinking-engine layer (the bridge, thinking-engine behind a feature gate). The CONTRACT carries only the calibrated threshold CONSTANT (a number with a citation); the calibration PRODUCES it. So: thinking-engine calibrates (offline) → emits bands → contract const (cited) → planner/Rubicon reads. No new contract dep.

**This makes the R-GATE probe rigorous:** instead of "do Analytical vs Creative differ" (necessary-not-sufficient), the probe becomes "compute Cronbach α / ICC on a real witness-trace corpus per style; does the MEASURED reliability band change the Commit/Plan/Prune outcome?" — a psychometric, citable pass/fail, satisfying I-NOISE-FLOOR-JIRAK (Jirak-bounded where the band needs a significance floor).

**Honest scope:** this is a DIRECTION (the calcs exist + the boundary is clear), not a built slice. First cut would be a thinking-engine `reliability_calibration` that runs cronbach_alpha over a recipe/witness corpus and emits the bands; then cite them in the contract const that replaces 0.2/0.8/0.15/0.35. Sequenced after the R-GATE probe proves the gate is non-cosmetic on a live trace.

**Cross-ref:** E-RELIABILITY-NOT-VALIDITY (why calibrate reliability); iron-rule-savant VIOLATES-I-NOISE-FLOOR-JIRAK (the uncited thresholds this fixes); I-NOISE-FLOOR-JIRAK (Berry-Esseen significance floor for the bands); `thinking-engine/cronbach.rs` (the α impl + "replaces hand-tuned heuristic" precedent); `jc/probe_p1_gamma_phase.rs spearman_rho`; bgz-tensor cam_pq_calibrate/quality; faiss-homology-cam-pq (offline-float→online-integer boundary); R-GATE probe; recipe_kernels SD_FLOW/SD_BLOCK; learning/cognitive_styles confidence_threshold.

---

## 2026-05-30 — FIX (council follow-through): StyleStrategy de-theatred — resolve_style decodes the 23D vector; reliability_of is the R-GATE measurable; plan() honestly labeled pure-passthrough

**Status:** SHIPPED-in-PR #439 (fixes the theater the brutally-honest-tester caught in D-MBX-A6-P3a). Probe-first per reviewers; reliability-not-validity framing per E-RELIABILITY-NOT-VALIDITY.

Three honest fixes to the no-op #439 shipped:
1. **`resolve_style` now DECODES the 23D style vector** (idx 4=analytical/3=creative/0=depth, the `selector.rs::style_alignment` convention) → dominant-axis ThinkingStyle. Kills the constant-`DEFAULT_STYLE` bug (recipe selection was identical for every query). NOTE: 23D planner vector, NOT the contract i4-32D `style_vector`/`StyleRecipe` surface (separate, deferred).
2. **`reliability_of(style, ctx) -> f32`** — the R-GATE MEASURABLE: runs style-selected recipe Tactics over a ThoughtCtx, returns accumulated confidence ∈[0,1]. RELIABILITY (settledness), NOT validity (external/post-commit) per E-RELIABILITY-NOT-VALIDITY. Pure: no plan mutation, no commit.
3. **`plan()` honestly labeled pure-passthrough** — computes reliability, emits NOTHING (no faked KanbanMove the planner can't build pre-A6-overhaul). The dead-store theater is gone; the comment now states the truth.

**Probe-first (reviewers' rule honored):** test `r_gate_reliability_varies_by_style` is the R-GATE probe written BEFORE any Rubicon gate field — asserts Analytical vs Creative select distinct mechanisms (TruthAwareInference vs StructuralDivergence) so a style-conditioned gate is non-cosmetic. test `resolve_style_decodes_the_23d_vector_not_constant_default` proves the bug is fixed. test `plan_is_pure_passthrough_until_emit_edge_lands` asserts plan stays None (no theater). 5 style_strategy tests + full planner lib green; fmt-clean (ran BEFORE commit).

**Still deferred (honestly, NOT shipped):** the emit edge (plan()→KanbanMove) gated on the D-MBX-A6 planner-output overhaul; truth-gating the Rubicon transition (only if R-GATE proves it changes an outcome on a real witness trace — the in-test mechanism-distinctness is necessary-not-sufficient; the full probe needs a live trace); `try_advance_phase` still has no production `MailboxSoaOwner` impl (separate cognitive-shader-driver slice).

**Cross-ref:** E-COUNCIL-SYNTHESIS (the theater this fixes); E-RELIABILITY-NOT-VALIDITY (the measurable's framing); #439 D-MBX-A6-P3a; `style_strategy.rs` (resolve_style/reliability_of); `selector.rs:137` (23D convention); D-MBX-A6 (the emit-edge home).

---

## 2026-05-30 — E-RELIABILITY-NOT-VALIDITY — the substrate's NARS (f,c) "truth" measures RELIABILITY (consistency/settledness/consensus), not VALIDITY (ground-truth correspondence); validity is conferred externally at/after the Rubicon Commit. "Truth" is a wisdom marker in disguise.

**Status:** FINDING (user-stated 2026-05-30, epistemic reframe). Corrects the "truth gate" mislabel propagated by the truth-architect council angle + this session.

**The measurement-theory answer (validity vs reliability):**
- NARS `(f,c)` is reliability machinery: `confidence c = w/(w+k)` = amount of accumulated agreeing evidence relative to horizon k = a RELIABILITY COEFFICIENT (same family as Cronbach's α — workspace ships `thinking-engine/cronbach.rs` + ICC; both reliability stats; I-NOISE-FLOOR-JIRAK Berry-Esseen = reliable-above-noise = reliability too). `frequency f = w+/w` = consensus value IN EXPERIENCE. `expectation = c·(f−0.5)+0.5` = reliability-weighted consensus.
- It does NOT measure VALIDITY (correspondence to ground truth). NARS is non-axiomatic / experience-grounded BY CONSTRUCTION — it measures whether the system's own witnessing COHERES (reliability), never whether it's SO (validity). The chess bring-up test is the tell: "ground truth is a Stockfish call away" — the substrate emits high-(f,c) GM-FLAVORED candidates (reliable/consensus-strong); STOCKFISH is the validity oracle. The substrate can be confidently, consistently WRONG.
- **Reliability is necessary-but-not-sufficient for validity** (classic psychometrics). The substrate cannot tell a reliable-true from a reliable-false belief from the inside. ⇒ "truth hasn't been validated yet" is exact.

**Why "wisdom marker in disguise":** Wisdom (Staunen×Wisdom qualia, The Click) = epistemic SETTLEDNESS + humility ("how well do I hold this", never "is it true"). Confidence with the φ⁻¹ ceiling ("permanent humility", c<1 always) IS a wisdom measure. Labeling it "truth" is a category slip — it's the wisdom/reliability axis wearing truth's name. "Crystallized knowledge committed in the end" = reliability accumulates → Rubicon COMMIT calcifies it into a durable fact → only post-commit/externally is validity assessable. COMMIT ≠ VALIDATION; commit = crystallization of reliability; validation is downstream.

**Design consequence (splits the gate I/truth-architect mislabeled into TWO at two clocks):**
- RELIABILITY gate (HOT, Evaluation→Commit): `(f,c)` expectation + CollapseGate SD `gate_state()`. Plan/Prune = reliability-too-low/contradictory; Commit = crystallize a reliable belief. Shader-speed.
- VALIDITY gate (COLD, AFTER commit): external oracle — Stockfish (bring-up) / GoBD audit / reciprocal A→B,B→A (recipe SDD#32) / FailureTicket→LLM (F>0.8). Cold-store-speed. = the two-clock decoupling, epistemically named.
- So `Evaluation→{Commit|Plan|Prune}` gates on RELIABILITY (settled enough to crystallize), NOT truth. A committed fact's validity is still PENDING the cold/external check.

**Corrects R-GATE probe:** it must probe RELIABILITY thresholding, not "truth": does style-conditioned reliability threshold change the CRYSTALLIZATION outcome (Skeptical demands higher c → more Plan/re-deliberate; Creative crystallizes at lower c)? Pass = ≥1 differing terminal. Validity is OUT of this probe — it's the separate post-commit external gate (the Stockfish bring-up IS the validity gate, already planned).

**Cross-ref:** E-COUNCIL-SYNTHESIS (the truth-architect "truth gate" this corrects); The Click (Staunen×Wisdom, φ⁻¹ ceiling = permanent humility, FailureTicket); cognitive-risc-core bring-up test (Stockfish = validity oracle); `thinking-engine/cronbach.rs` (reliability stat); I-NOISE-FLOOR-JIRAK; `spo/truth.rs TruthValue`; recipe SDD#32 (reciprocal validation); Rubicon Commit = calcify; R-GATE probe.

---

## 2026-05-30 — COUNCIL SYNTHESIS (catalyst, 7 savants): #439 StyleStrategy is PASSTHROUGH THEATER; the real target = thinking-style → PlannerDTO(=KanbanMove) → truth-gated Rubicon scheduling. Honest correction + the wiring map.

**Status:** FINDING (council-catalyzed synthesis 2026-05-30; 7 savants: iron-rule, dto-soa, creative-explorer, cascade-impact, prior-art, brutally-honest-tester, truth-architect). Corrects the overstated D-MBX-A6-P3a commit. The council ENRICHED (not gated) — it surfaced theater I shipped + the real design.

**HONEST CORRECTION (brutally-honest-tester, confirmed by source):** my #439 `StyleStrategy` commit "wires thinking-styles as the planning substrate" OVERSTATED a no-op:
- `StyleStrategy::plan()` runs `recipe_kernels` then DISCARDS the `Outcome` + mutated `ThoughtCtx`; returns `input` byte-identical (dead-store). Emits NO KanbanMove/Candidate.
- The test asserts only `out.is_ok()` — green on a no-op = AP6 theater; nothing asserts the plan changed.
- `resolve_style()` discards `ctx.thinking_style`, always returns DEFAULT_STYLE → recipe selection is CONSTANT for every query.
- `try_advance_phase` shipped-but-DEAD (only FakeSoa calls it; no real `MailboxSoA: MailboxSoaOwner`). `KanbanMove.exec` write-only (always Native, never read). Libet −550ms only in tests.
⇒ #439's contract types (KanbanColumn DAG, try_advance_phase) are SOUND in isolation; the StyleStrategy "integration" is two disjoint islands with no connecting edge. Recorded as TECH-DEBT, not silently left.

**DON'T REVIVE recipe.rs (4 savants unanimous):** iron-rule = VIOLATES-I-NOISE-FLOOR-JIRAK (uncited 0.2/0.8/0.1 thresholds) + 32-vs-33 dim hazard; dto-soa = FIFTH-COLUMN-VIOLATION (3rd/4th parallel "style" representation); creative-explorer = DELETE it, the keystone is the argmax decode; cascade = full revive is CROSS-CRATE (jit::StyleRegistry trait ext hits 17 deps + reopens atoms.rs pack/unpack todos + reroutes shipped #439). recipe.rs `StyleRecipe` is dead/`todo!()`/unexported.

**THE KEYSTONE (creative-explorer + prior-art, the real target):** the canonical style→schedule pipeline is ONE arrow, NO StyleRecipe:
`I4x32 (composition) → argmax → ThinkingStyle (identity, 6-bit, τ) → cluster()→Mechanism (selection) → tau()→KernelHandle (exec)` ; the deferred `I4x32→argmax→ThinkingStyle` decode (style_strategy.rs:95) IS the unifying keystone the session circled. Four altitudes = four existing SoA columns (topic/angle/thinking/planner), not a 5th layer.

**"PlannerDTO" is DRIFT (prior-art):** no canonical type — it's PlanResult / PlanInput / Candidate+KanbanMove unnamed. Do NOT mint a new PlannerDTO. Canonical home = D-MBX-A6 ("planner output = KanbanMove"). The triangle thinking-style→PlannerDTO→Rubicon is A6's STATED PREMISE; 2 of 3 edges shipped (#437/#439), missing edge = `Outcome→KanbanMove` emit (the A6-P3 NEXT node).

**TRUTH-GATED RUBICON (truth-architect, the missing epistemic layer):** `try_advance_phase` gates on DAG legality ONLY, never truth (f,c) — yet the SoA owner exposes `edges_raw()` (CausalEdge64 f/c) at the transition site and ignores it. Deciding predicates exist UNUSED: `TruthGate::passes(expectation)`, `CausalEdge64::counterfactual_ready` (ZERO callers). Evaluation→{Commit|Plan|Prune} = the epistemic 3-way (commit-iff-high-conf / prune-iff-low / plan-iff-contradictory) maps 1:1 onto expectation bands. Style→threshold link exists in the WRONG crate (`learning/cognitive_styles.rs` fp[18] confidence_threshold: Skeptical 0.95). `ThoughtCtx.gate_state()` (FLOW/HOLD/BLOCK) is the natural Planning→{CognitiveWork|Prune} gate but lives on ThoughtCtx, unjoined to edges_raw() f/c — TWO truth registers, unreconciled.

**THE NEXT BUILD (synthesized, replaces P3b):** the real A6-P3 slice = make StyleStrategy ACTUALLY schedule:
1. Implement `resolve_style`: decode `ctx.thinking_style` i4-32D vec → argmax `ThinkingStyle` (the keystone; kills the constant-DEFAULT_STYLE bug).
2. `StyleStrategy::plan()` must EMIT — thread the ThoughtCtx outcome into a `KanbanMove` (or `PlanResult` field), and a test asserting the plan CHANGED by style (Skeptical≠Creative output). Kills the passthrough theater.
3. Truth-gate the transition: thread `expectation()`/`gate_state()` into the Evaluation→terminal choice.
**PROBE FIRST (both reviewers):** R-GATE — does threading expectation() change ANY column outcome vs DAG-only on a fixed witness trace (Skeptical 0.95 vs Creative 0.6)? Pass = ≥1 differing terminal; Fail = cosmetic. Do NOT add a threshold field until the number exists. Probe → then wire.

**Cross-ref:** #439 (D-MBX-A6-P3a, corrected); D-MBX-A6 (KanbanMove output overhaul, the canonical home); `style_strategy.rs:95,136`; `soa_view::try_advance_phase`; `contract::{thinking(tau/argmax), atoms(I4x32/CANONICAL_ATOMS[33]), recipe_kernels(gate_state), recipes}`; `causal-edge counterfactual_ready`; `spo/truth.rs TruthGate`; `learning/cognitive_styles.rs`; recipe.rs (do-not-revive); E-VERSION-ARC / E-SUBSTRATE-IS-THE-SCHEDULER.

---

## 2026-05-30 — FINDING (via #433 ref): three recipe modules; `contract::recipe::StyleRecipe` is the CANONICAL i4-32D style↔atom↔JIT home but is STALE+ORPHANED (unblocked yet never migrated/exported). StyleStrategy (#439) correctly built on the LIVE recipes/recipe_kernels — adjacent, not wrong.

**Status:** FINDING (grounded, prompted by user "check 433"). Tech-debt + a scoped follow-up; NOT a #439 defect.

**Three distinct recipe modules in contract — disambiguated:**
1. `contract::recipe.rs` (SINGULAR) — **"Composition layer: thinking-style recipes"**: `StyleRecipe { name, weights:&[(Atom,i8)], composition:Option<I4x32> }` + `PersonaRecipe` (+β/thresholds) → `KernelHandle` (Cranelift). The canonical **atoms(i4-32D) → StyleRecipe → PersonaRecipe → JIT** ladder (E-LADDER-SERVES-MAILBOX §2). "JIT target is the recipe, not the per-atom dot"; "Elixir-style open/closed hot-load split". = EXACTLY the i4-32D-style + Cranelift-template substrate the user named.
2. `contract::recipes.rs` (PLURAL) — the 34 reasoning-TACTIC catalogue (RTE/HTD/…).
3. `contract::recipe_kernels.rs` — the 34 executable `Tactic` kernels.
   (+ a 4th, unrelated: `lance-graph-ontology::odoo_blueprint::OdooStyleRecipe` — #433's Odoo-codegen D-Atom fingerprint, renamed from StyleRecipe to avoid THIS collision.)

**The rot (the real find):** `recipe.rs`'s blocker D-ATOM-1 **HAS LANDED** — `contract::atoms` is real now (`I4x32` struct, `Atom`, `CANONICAL_ATOMS:[Atom;33]`). But `recipe.rs` was NEVER migrated: still uses `I4x32Stub=[i8;32]`/`AtomStub=u8` forward-stubs, AND is **NOT exported in lib.rs** (`pub mod recipe;` absent; only atoms/recipes/recipe_kernels exported). So `StyleRecipe`/`PersonaRecipe` is **dead code**: defined, stub-blocked despite the blocker resolving, unexported, zero consumers. Classic "blocker cleared, dependent never updated" debt — the kind #433's `prior-art-savant`/`dto-soa-savant` gate exists to catch.

**Impact on D-MBX-A6-P3a (#439) — adjacent, not wrong:** `StyleStrategy` wired to the LIVE `recipes`+`recipe_kernels` (reasoning tactics). That was the correct shippable choice (those are exported + tested; `recipe.rs` is not). It is NOT the canonical i4-32D `StyleRecipe` home, and it does not yet use `atoms::I4x32`/`PersonaRecipe`. So #439 stands as-is (correct, green); the canonical-home migration is a SEPARATE slice, not a #439 fix.

**Scoped follow-up (proposed, NOT in #439):** "D-MBX-A6-P3b — revive recipe.rs": migrate `I4x32Stub`→`atoms::I4x32`, `AtomStub`→`atoms::Atom`, export `pub mod recipe;` in lib.rs, then route `StyleStrategy` through `StyleRecipe` (style = i4-32D atom composition) → `PersonaRecipe` → the τ/jit `KernelHandle`. THAT closes the real i4-32D-style → Cranelift-template loop the user named. The epiphany-brainstorm-council (#433) is available here as a CATALYST (not a gate): invoking dto-soa-savant + creative-explorer would ENRICH the i4-32D revive with angles, not authorize it. P3b can proceed without it; the council is an amplifier to reach for, not a checkpoint to clear.

**Also from #433 (process) — CORRECTED 2026-05-30:** the epiphany-brainstorm-council is a CATALYST for new insight (spawn 4-7 Opus savant lenses to GENERATE divergent reframings / cross-impact), NOT a guardian of what may be recorded. Its LAND/REVISE/REJECT is synthesis output, not permission. So this session's tee-prepends are NOT 'pending approval' at any gate — they are deposited priors. The council is something to INVOKE to enrich an idea (esp. the derived ones: E-FIREFLY-*, E-0xFFF, E-POLYGLOT-*), never a checkpoint they must pass. (Supersedes the earlier 'council-gated' framing in this same entry.)

**Cross-ref:** #433 (style_recipe + epiphany-brainstorm-council + 5 savant cards); `contract::recipe.rs` (StyleRecipe/PersonaRecipe, stale); `contract::atoms` (D-ATOM-1 landed, #411); `contract::recipes`/`recipe_kernels` (what #439 uses); `OdooStyleRecipe` (ontology, the renamed-to-avoid-collision sibling); E-LADDER-SERVES-MAILBOX §2; D-MBX-A6-P3a (#439).

---

## 2026-05-30 — SHIPPED-in-PR: D-MBX-A6-P3a — StyleStrategy (thinking-style planning substrate wired into the planner)

**Status:** SHIPPED-in-PR #439 (builds on A6-P1 #437 / A6-P2). First live cut of D-MBX-A6-P3 consumer wiring.

`lance-graph-planner` now CONSUMES the contract cognitive substrate (it referenced none before): new `strategy::style_strategy::StyleStrategy` (#18 in `default_strategies()`) resolves the active `ThinkingStyle` → `cluster()` → `cluster_mechanism()` → selects which of the 34 `recipe_kernels::Tactic` fire over a `ThoughtCtx` built from `PlanContext` markers (`free_will_modifier`→temperature). The **style selects the recipe** (cluster→mechanism), not a hardcoded id list — and carries `style.tau()` (the JIT macro address, grounds `ExecTarget::Jit`). Mirrors `mul::escalation` (thin planner module over zero-dep contract). Planner already deps contract; NO new dep edge; contract stays zero-dep (no circular-dep, the AriGraph trap avoided).

Verified: 3 new tests (analytical→truth-aware selection; every cluster→mechanism total over RECIPES; plan() passes through) + 192 planner lib green; rustfmt-clean; rebased onto main post-#438 (arm-discovery, no collision).

**Deferred (the rest of A6-P3, per the design map):** i4-32D `thinking_style` vec → argmax `ThinkingStyle` decode; `Outcome`→`Candidate`/`KanbanMove` adapter; `tau`→`JitTemplate`→Cranelift `KernelHandle` compile (the real ExecTarget::Jit path); recipe-outcome→membrane commit via `CognitiveOpKind::MetaWordCommit` (OrchestrationBridge, never callcenter→planner); pre-recipe-fire RBAC (today pre-commit only); `class_id`→`recipe_id` resolver (ontology gap).

**Cross-ref:** `planner::strategy::style_strategy`; `contract::{thinking(tau/cluster),recipes,recipe_kernels}`; `mul::escalation` precedent (#411); `ExecTarget::{Jit,Elixir}` (#439); `OntologyRegistry::attach_thinking_style` (registry.rs:311, the existing class→style seam); D-MBX-A6-P3.

---

## 2026-05-30 — RE-CENTER: thinking-styles ARE the planning substrate — i4-32D style → τ address → JITson/Cranelift template → KernelHandle; recipes are what styles SELECT. (corrects the recipe-centric framing of the build target above)

**Status:** BUILD-GRADE correction (user 2026-05-30: "thinking styles are the most important planning substrate… i4-32D thinking styles and jit/JITson cranelift compiler templates"). Re-centers the default-recipe build target one entry up: styles are the dispatcher, recipes are the tactics dispatched. Grounded by grep (file:line).

**The full planning-substrate pipeline — ALL SHIPPED, just unwired into the mailbox planner:**
- `contract::thinking` — **36 thinking styles / 6 clusters**, `ThinkingStyle::ALL:[_;36]`, each mapped to a **τ (tau) macro address** for JIT (`thinking.rs:19`): Analytical τ0x40-4F, Creative τ0xA0, Empathic τ0x80, Direct τ0x60, Exploratory τ0x20, Meta τ0xC0. Plus `StyleCluster`(6)→`PlannerCluster`(4) via `to_planner_cluster()`, `FieldModulation`, `ScanParams`. The i4-32D form = the compact SIMD selection vector (32×i4 = 16B, same width family as CausalEdge64/QualiaI4_16D — hot-path AND-able).
- `contract::jit` — closes the loop: `JitTemplate` (JITSON JSON) → `JitCompiler::compile` (ndarray Cranelift engine) → `KernelHandle{fn_ptr:*const u8}` (native code, Send+Sync) → `StyleRegistry` kernel cache (param_hash → cached kernel). n8n-rs implements `CompiledStyleRegistry`; ndarray = the jitson engine; lance-graph produces templates (`jitson_kernel.rs`).
- `contract::recipes`/`recipe_kernels` — the 34 tactics styles SELECT (the prior build-target entry).

**The pipeline (one line):** `i4-32D thinking style → τ address → JitTemplate(JITSON) → Cranelift compile → KernelHandle(native fn_ptr) → cached by StyleRegistry`; the style ALSO selects which recipe/Tactic runs over ThoughtCtx/SoA.

**This grounds the #439 ExecTarget variants concretely:** `ExecTarget::Jit` = the τ→template→Cranelift→KernelHandle path (compiled). `ExecTarget::Elixir` = the recipe_kernels interpreted "Elixir-like" layer (the un-JITted fallback). The two exec targets I shipped now have their actual machinery identified: JIT = jit.rs, Elixir = recipe_kernels.rs.

**Re-centered build target (corrects the recipe-centric framing):** the mailbox planner's default planning substrate = **thinking-style selection → τ → (cached KernelHandle | recipe Tactic) → over the SoA**. Styles dispatch; τ addresses; JITson compiles; recipes are the tactic catalogue. The minimal first slice should wire STYLE SELECTION first (the planner picks a ThinkingStyle → cluster → ScanParams), then recipe/kernel dispatch — not recipes in isolation. The Opus design map (in flight) was launched recipe-centric; re-center it on thinking+jit at synthesis.

**Gaps to verify in the map:** does the planner already use ThinkingStyle (it had its own 12; contract says it maps via `to_planner_cluster()`)? Is `StyleRegistry`/`JitCompiler` wired to a real Cranelift engine (ndarray) or stub? Is the τ-address→template path built or spec? (jit.rs says lance-graph `jitson_kernel.rs` produces templates — verify it exists.)

**Cross-ref:** `contract::thinking` (36 styles/τ/i4-32D/FieldModulation); `contract::jit` (JitTemplate/JitCompiler/KernelHandle/StyleRegistry); `contract::recipes`+`recipe_kernels`; `ExecTarget::{Jit,Elixir}` (#439); the prior "default recipes" build-target entry (this re-centers it); D-MBX-A6-P3; ndarray jitson/Cranelift engine.

---

## 2026-05-30 — BUILD TARGET: default recipes for the mailbox planner — the DTOs already exist, only the WIRING is missing

**Status:** BUILD-GRADE (user-directed 2026-05-30: "we need default recipes for our mailbox planner… the DTOs are already there to wire"). Graduated from brainstorm to concrete. Grounded by grep (file:line below); Opus design map in flight.

**The finding — everything exists, nothing is wired:**
- `contract::recipes` — `Recipe` struct + `RECIPES:[Recipe;34]` (RTE/HTD/RCR/… id 1..=34, each with tier/mechanism/bucket/spo2cubed/substrate) + `recipe(id)`/`recipe_by_code`/`by_mechanism`/`causal` lookups. = the DEFAULT RECIPE CATALOGUE, locked.
- `contract::recipe_kernels` — the `Tactic` trait + `ThoughtCtx` (sd/free_energy/dissonance/temperature/confidence/rung/candidates/beliefs) + `Outcome`; "the Elixir-like recipe layer: 34 hot-dispatchable units, registry-routed by id." = the EXECUTABLE default recipes.
- **GAP (verified):** `lance-graph-planner` references NEITHER `recipes` nor `recipe_kernels` nor `Tactic` anywhere. The mailbox planner has NO default recipes wired. This is the build.
- **Consumers ready (DTOs exist):** `callcenter::{OntologyDto/EntityTypeDto/PropertyDto/LinkTypeDto/ActionTypeDto, MembraneRegistry, LanceMembrane, cognitive_bridge_gate, policy, rls}` = the committing membrane; `ontology::{OntologyRegistry, MappingRow, SchemaPtr, enumerate_first_with_entity_type_id}` = OGIT classes on SoA; `rbac::{Policy, Role(can_read/can_write/can_act), Operation, evaluate, AccessDecision}` = the gate.

**The chain to wire:** recipe (contract `Tactic`) → candidate (planner `CandidatePool`) → rbac gate (`rbac::evaluate`) → ontology class resolve (`OntologyRegistry` by entity_type_id) → membrane commit (`callcenter::LanceMembrane`). Recipe substrate + consumer DTOs exist; the SEAMS between them are the gap.

**This is the "Elixir-like template" layer made real** (connects the GEL/ExecTarget::Elixir thread): recipe_kernels is literally documented as "the Elixir-like recipe layer." So the default-recipe wiring IS the planner-strategy/template work, grounded in shipped code — not the speculative PlanPacket. Centered on AST/SoA markers (ThoughtCtx = our substrate markers), per "the mailbox understands AST."

**Maps to D-MBX:** part of / sibling to D-MBX-A6-P3 (planner consumer wiring). Awaiting Opus map for: dependency directions (keep contract zero-dep; circular-dep risk like AriGraph↔planner), the minimal first slice (RecipeStrategy in default_strategies vs default_recipes() selector), and the rbac-gate placement (pre-recipe vs pre-commit).

**Cross-ref:** `contract::recipes`/`recipe_kernels` (recipe catalogue + Tactic); `planner::{traits,api,cache/candidate_pool,strategy/mod}`; `callcenter::{lance_membrane,ontology_dto,cognitive_bridge_gate}`; `ontology::registry`; `rbac::{policy,role,access}`; D-MBX-A6-P3; `ExecTarget::Elixir` (#439); recipe_kernels "Elixir-like" doc.

---

## 2026-05-30 — BRAINSTORM: E-FIREFLY-PACKET-IS-THE-LOCATION-TRANSPARENT-PLAN — (adjacent inspiration) backport firefly's packet-executor as the in-mailbox PLANNING SUBSTRATE; the same packet distributes cross-server via ONE gRPC hop (BEAM location transparency)

**Status:** BRAINSTORM / adjacent-inspiration (user framing 2026-05-30: "just brainstorming, you might only find an adjacent inspiration"). NOT a ratified decision — a candidate direction to remember, not a committed build plan. Downgraded from an over-eager "DECISION" label. The forks/slice below are sketch options, not a sequenced commitment.

**The backport:** the mailbox's PLANNING SUBSTRATE = firefly's packet-flows-through-a-node-graph executor. A plan = a graph of plan-nodes; planning/candidate-generation = a packet hopping through it (validate→transform→persist shape), TTL-bounded, accumulating `trace[]`. Replaces a bespoke planner-tick loop with the firefly model. Each hop = a phase transition = a version commit (`E-VERSION-ARC-IS-THE-KANBAN`); the `trace[]` = the witness arc (R4); `ExecTarget` (#439) = how each node executes; transport = orthogonal.

**The punchline (why it's elegant): distribution = ONE gRPC packet.** Because the planning substrate IS packet-based, the in-mailbox plan state is ALREADY a complete serializable execution context (`FireflyPacket`: 80B LE header + 1250B resonance + ctx + trace + ttl). Cross-server distribution needs NO new abstraction: serialize the packet once → gRPC to another server instance → it deserializes and continues the hop. firefly's `hop(current,next)` is already location-transparent (target = an address; a routing table decides local vs remote). = **BEAM location transparency** (send to a PID; local-or-remote is just transport) — and exactly what ractor embodies (Boxed-local / Serialized-remote).

**Reconciles with `E-RACTOR-WANTS-TOKIO-NOT-GRPC` (NOT a contradiction):**
- INSIDE hop (local node) = move the struct, zero-serialize (Tokio/in-mem). gRPC-slower-than-Tokio STILL holds; you never pay it locally.
- OUTSIDE hop (crosses a server) = serialize ONCE → gRPC (or cluster-TCP / Flight). Pay serialization only at the boundary, amortized (two-clock decoupling). The packet is distribute-ready BY CONSTRUCTION; "only a gRPC packet" is true because the planning DTO already speaks packets.
This realizes the long-standing inside/outside "location-transparency" synergy (the §1.1 candidate) concretely: ONE packet, two transports.

**Honesty / nuance:** "only a gRPC packet" is true at the abstraction level; the resonance is ~1.25KB + ctx, so a cross-server hop has real serialize cost — paid only at the boundary, not per local hop. The win is no NEW distribution layer, not zero cost.

**Design forks to decide BEFORE building (the slice gates on these):**
1. **DTO:** extend `KanbanMove` into a `PlanPacket` (header + SoA-resonance ref + op + trace + ttl), or a new contract type beside it? (KanbanMove is the per-move record; PlanPacket is the hopping execution context — likely a NEW type that CARRIES a KanbanMove per hop.)
2. **Address width:** 0xFFF (12-bit, inside, cache-aligned) vs firefly's 256-bit SHA at the durable/cross-server boundary (`E-0xFFF` / firefly divergence). Likely: 0xFFF local, SHA-CAM at the gRPC hop (mirrors witness-materialization-at-commit).
3. **Payload by-ref vs by-value:** inside, the packet must reference the SoA (R1 "never serialize the SoA") — carry a `MailboxSoaView` handle / row-range, NOT a copied resonance. Only the gRPC hop snapshots. (This is the R1 + R5 hot/cold-snapshot rule applied to the packet.)
4. **Serialize format for the gRPC hop:** protobuf (tonic) vs firefly's LE-packed header + hex. (gRPC/tonic already in the lab surface; promote at the genuine boundary.)
5. **Routing table:** where local-vs-remote node resolution lives (the pointer table / mailbox index).

**Minimal first slice (proposed, fork-1/3 dependent):** a zero-dep `contract::plan_packet::PlanPacket` — routing header (src/tgt address, ttl, seq, flags) + an op + a `witness_chain_position` trace + a BORROWED SoA reference (not owned resonance), with `hop()` (local, move) and a `to_wire()`/`from_wire()` boundary (the only serialize point). Carries a `KanbanMove` per phase. Local hop = struct move; `to_wire` only at a remote hop. Unit-tested with a fake routing table (local vs remote). Defers the actual gRPC service (lab/outside) and the node-graph executor (planner crate) to follow-ups.

**Maps to D-MBX:** this IS the A6-P3 planning-substrate design (was "bespoke planner loop") + the distribution story for D-MBX-9's cross-server case. Sequenced after the address-width decision (relates P-B canonical 0xFFF type).

**Cross-ref:** `/home/user/firefly/rust/src/dto/packet.rs` (hop/pack_header — the location-transparent primitive); `E-FIREFLY-IS-GEL-OUTSIDE-PROTOTYPE`; `E-RACTOR-WANTS-TOKIO-NOT-GRPC`; `E-VERSION-ARC-IS-THE-KANBAN`; `ExecTarget`/`KanbanMove`/`MailboxSoaView` (#437/#439); RISC core inv 4/5/7 (hot-cold, snapshot, two-clock); D-MBX-A6-P3 / D-MBX-9.

---

## 2026-05-30 — E-FIREFLY-IS-GEL-OUTSIDE-PROTOTYPE — adaworldapi/firefly is a runnable GEL substrate + the OUTSIDE-transport prototype; its FireflyPacket = the serialized cross-process Baton; transport is Redis-mRNA (NOT gRPC)

**Status:** FINDING (read firefly source 2026-05-30, cloned read-only to /home/user/firefly — PUBLIC repo, NOT in the 16-repo authorized scope; do not push to it). User: "my toy Ballista executing gRPC packets with 0xFFF as transport."

**What firefly IS:** a runnable (Railway-deployed) "Universal Executable Substrate" — `BOOT.md`: "doesn't compile code, it RUNS compiled graphs"; per-language compilers (RUBBERDUCK/Ruby, PYTHONIC, JAVELIN/Java, RUSTLER/Rust) all emit "1.25KB Hamming nodes." **= GEL made concrete** (`E-GEL-IS-THE-GRAPH-SUBSTRATE`): any language → uniform graph node → executable substrate. Firefly is a parallel, smaller, deployed prototype of exactly what the D-MBX arc builds inside lance-graph.

**Transport CORRECTION (honest):** the actual transport is **Redis streams + mRNA** (`docs/MRNA_TRANSPORT.md`, `rust/src/dto/packet.rs`), NOT gRPC. `FireflyPacket` routes via Redis `XADD`/`XREAD` on `firefly:node:{hash}` streams. "gRPC packets" in the user's framing = the intent/analog (RPC-style cross-process packets); the code uses Redis-stream mRNA. Either way it is the OUTSIDE (cross-process, serialized) layer — confirming `E-RACTOR-WANTS-TOKIO-NOT-GRPC`: inside=Tokio-Baton zero-serialize, outside=serialized packet (firefly's choice = Redis-mRNA; ractor's = cluster-TCP; lab = gRPC/Flight).

**Concept-for-concept map (firefly ↔ lance-graph), the striking convergence:**
- `FireflyPacket` (80B header + 1250B resonance + JSON ctx) = the **serialized cross-process Baton** (`CollapseGateEmission` is the in-process LE tuple; FireflyPacket is its outside-the-process form).
- **80B LE routing header** (src/tgt = 32B SHA256 node addr, ttl u8, priority, flags u16, sequence u32, CRC32) = the "0xFFF as transport" layer — ADDRESSES route, payload rides. (firefly uses 256-bit SHA addresses, not 12-bit 0xFFF — a divergence: full content-hash vs compact 12-bit; see open Q.)
- **1250B = 10,000-bit Hamming, 4 zones**: Content[0:3000] / Process[3000:6000] / Qualia[6000:8000] / Context[8000:10000]. = the SAME 4-signature decomposition as BindSpace SoA column families (content/edge/qualia/meta). Independent convergence.
- `ttl` + `hop()` (decrement, append to `trace[]`) = the **witness arc / belief-state chain** (R4); TTL=0 → dead-letter = absorbing **Prune** terminal.
- Redis consumer-group scaling rules: **VALIDATE/TRANSFORM parallel (pure), PERSIST single (ordering)** = EXACTLY the hot-path-parallel vs commit-gate-single-writer split (RISC core invariant 4) — independently arrived at.
- stream-length **backpressure** = RISC core invariant 8.
- **Storage Trinity** (LanceDB vectors / DuckDB facts / Kuzu graph) = lance-graph's unify-on-Lance (firefly splits 3 engines where lance-graph + surrealdb-kv-lance unifies; a real architecture fork to note).

**Why it matters for the arc:** firefly is the **OUTSIDE-transport reference implementation** + a working GEL executor. It validates (by independent convergence) the hot/cold split, the address-routes-payload-rides packet shape, the 4-zone resonance, and the witness-as-trace. ExecTarget-wise it is a distributed executor (Ballista-shaped): a `KanbanMove{exec=Distributed}` would lower to a FireflyPacket on the outside path.

**Open questions / divergences to reconcile (NOT decided):**
1. **Address width:** firefly = 256-bit SHA256 node address; lance-graph 0xFFF = 12-bit aligned address. Full content-hash (collision-proof, big) vs compact 12-bit (cache-aligned, codebook-bounded). Which at which layer? (likely 0xFFF inside / SHA-CAM at the durable/cross-process boundary, mirroring the CAM-materialization-at-commit rule.)
2. **Transport:** Redis-mRNA (firefly) vs ractor-cluster-TCP vs Arrow-Flight-CAM — three OUTSIDE options; pick per deployment.
3. **Storage:** firefly trinity (3 engines) vs lance-graph unify-on-Lance. Reconcile or keep firefly as the polyglot-frontend ingest tier feeding the unified substrate.
4. firefly is OUTSIDE authorized scope — keep as read-only reference; do not push.

**Cross-ref:** `/home/user/firefly/{BOOT.md,rust/src/dto/packet.rs,docs/MRNA_TRANSPORT.md,docs/ARCHITECTURE.md}`; `E-GEL-IS-THE-GRAPH-SUBSTRATE`; `E-RACTOR-WANTS-TOKIO-NOT-GRPC`; `E-0xFFF-IS-ONE-ALIGNED-ADDRESS`; CollapseGateEmission Baton; RISC core invariants 4/8; RUBBERDUCK (github.com/AdaWorldAPI/rubberduck, the Ruby→node compiler).

---

## 2026-05-30 — FINDING: E-RACTOR-WANTS-TOKIO-NOT-GRPC — local ractor is a Box<dyn Any> pointer-move over Tokio mpsc (zero serialize); gRPC is strictly slower and is LAB-ONLY. CAM/0xFFF-over-Flight is the cross-PROCESS path only

**Status:** FINDING (grounded in ractor + cognitive-shader-driver source, 2026-05-30). Answers the user's transport question: does ractor want gRPC, or is it slower than Tokio?

**ANSWER: For in-process mailboxes, Tokio mpsc beats gRPC by construction — gRPC is never the local transport.** Proof from ractor source:
- **Local (default, NO `cluster` feature):** a message is `BoxedMessage { msg: Box<dyn Any + Send> }` (`ractor/src/message.rs:62-63`), moved through a Tokio mpsc and recovered via `m.downcast::<Self>()` (`message.rs:102`). **Zero serialization** — a heap-boxed pointer move + downcast. Blanket `impl<T: Any+Send> Message for T` (ractor CLAUDE.md) = any Rust type rides as-is, zero boilerplate.
- **Cluster (`cluster` feature):** the SAME enum is replaced by `SerializedMessage { args: Vec<u8> }` (`message.rs:30-57`) → serialize → TCP via `ractor_cluster` NodeServer/NodeSession. Note: ractor's OWN distributed transport is **raw TCP serialization, NOT gRPC.**

**Why gRPC is strictly slower locally:** gRPC = protobuf encode + HTTP/2 framing + socket syscall + decode. The local Tokio path SKIPS all of it (pointer move). gRPC only earns its cost at a **process/node boundary** — and even there ractor's native answer is TCP serialization, not gRPC. So: never gRPC for same-process mailboxes; serialization (TCP or Flight) only when crossing a process.

**The gRPC/Ballista/Flight + CAM/0xFFF occurrences (the user's first point):** these live in `cognitive-shader-driver/src/{grpc.rs,wire.rs}` — **LAB-ONLY** (`lab-vs-canonical-surface.md:52,54`: "Test-transport convenience… NEVER in production binary"), feature-gated (`Cargo.toml`: `grpc.rs` required-features=["grpc"], `tonic optional=true`). CAM-encoded-over-Flight/Ballista = the cross-PROCESS / distributed-DataFusion movement path, exactly where serialization is unavoidable anyway, and where 0xFFF/CAM codes are the COMPACT wire form (12-bit address / 6-byte CAM-PQ code instead of full vectors — Flight/Ballista carry the addresses, not the payloads). So CAM-over-Flight is the RIGHT tool for its layer (inter-process), and Tokio-Baton is the right tool for the hot in-process layer.

**Maps to the inside/outside duality (this session's stance):** INSIDE (in-process, hot) = Tokio mpsc + Baton (`CollapseGateEmission` LE tuple), zero-serialize pointer move — the kanban hot path. OUTSIDE (cross-process, distributable) = ractor `cluster` TCP serialize OR CAM/0xFFF-over-Flight/Ballista for DataFusion-federated movement. gRPC sits with OUTSIDE+LAB, never INSIDE. Decision: keep ractor on local Tokio for the mailbox swarm; reserve serialized transport (prefer ractor_cluster TCP or Flight-CAM) for genuine process boundaries; gRPC stays lab-only.

**Cross-ref:** `ractor/src/message.rs` (Boxed vs Serialized); ractor CLAUDE.md (blanket Message impl, cluster=TCP); `lab-vs-canonical-surface.md:52-54` (grpc/wire LAB-ONLY); `E-VERSION-ARC-IS-THE-KANBAN` (inside/outside); the earlier hot-path Tokio-cost finding; faiss-homology-cam-pq (CAM = compact address, the right Flight payload).

---

## 2026-05-30 — CORRECTION: E-0xFFF-IS-ONE-ALIGNED-ADDRESS — the "4 distinct 4096s" are ONE deliberate 0xFFF (12-bit) address space, aligned for efficiency; not a magic-number coincidence. Corrects E-POLYGLOT-4096-IS-CONJECTURAL

**Status:** CORRECTION (supersedes the "4096 = 4 distinct unrelated magic numbers, no canonical surface" claim in `E-POLYGLOT-4096-IS-CONJECTURAL`) + FINDING (user-stated 2026-05-30).

**The correction (I was wrong):** I labeled the recurring 4096 as four unrelated coincidental magic numbers. It is **ONE deliberate 0xFFF / 12-bit address space**, and the surfaces were **aligned to it for efficiency** — not coincidence. The polyglot path addresses by 0xFFF; 4096 = 2^12 is the address width, so:
- deepnsm COCA vocab (4096 words = 12-bit rank),
- AutocompleteCache attention topology (64×64 = 4096 heads),
- BindSpace addressing (16 prefixes × 256 = 4096 slots),
- the COCA² distance matrix
…all land on **the same 12-bit index ON PURPOSE**, so a parsed symbol / vocab rank / attention head / SoA slot **co-index without translation**. The "incidentally also 4096" IS the alignment tell.

**Why aligned (the efficiency):** one 0xFFF address space = no remap between stages (parse → vocab → head → SoA share the index); fits a mask, shift-addressable / AND-testable in a cycle (same "5 facets ≈ one u64, SIMD batch-AND over the SoA column" efficiency as wikidata-hhtl-load). This is the CAM addressing invariant (cognitive-risc-classes): content → fixed-width address → every layer indexes by it. 0xFFF is that fixed width.

**What STILL stands from the prior finding:** the **end-to-end frontend→0xFFF→SoA wiring is still partial** — the IR is string-keyed (`ast.rs label: String`); the planner doesn't yet resolve labels→12-bit rank; the parser→4096 join (`cache/convergence.rs`) is the half-built p64-drift terminus. BUT the alignment means wiring it is CHEAP (shared address, not a translation layer) — the design intent was always one address space; only the resolution call is missing. So: address space = unified-by-design (corrected); resolution path = not-yet-called (stands).

**Consequence for P-A / P-B:** P-B ("unify the 4096 surface") is NOT "merge 4 unrelated things" — it's "**declare the 0xFFF address width as ONE canonical 12-bit type** and have the surfaces name it instead of re-spelling 4096". Lighter than I framed it. And the polyglot conformance loop (P-A) can additionally assert that equivalent queries resolve to the same 0xFFF address (once resolution is wired), not just the same LogicalOp shape.

**Cross-ref:** corrects `E-POLYGLOT-4096-IS-CONJECTURAL` + `E-POLYGLOT-TWO-IR-ROUTES`; cognitive-risc-classes (CAM addressing, fixed-width); wikidata-hhtl-load (facet u64 / SIMD batch-AND efficiency); `cache/convergence.rs` (the 0xFFF join, partial); faiss-homology-cam-pq (exact-address addressing).

---

## 2026-05-30 — FINDING: E-POLYGLOT-TWO-IR-ROUTES — the 4 dialect parsers build IR via TWO different routes (in-strategy vs ArenaIR), converging on one LogicalOp arena that NOTHING asserts they agree on; 4096-in-planner = the AutocompleteCache, joined via convergence.rs

**Status:** FINDING (grounded source read 2026-05-30, per user pointer to planner/datafusion polyglot path). Sharpens `E-POLYGLOT-4096-IS-CONJECTURAL` with the real per-parser reality.

**The 4 polyglot parsers are at two IR-construction routes (arena.push counts verified):**
- **In-strategy transpilers (push LogicalOp directly):** `gremlin_parse.rs` (37 `arena.push`; full step→IR: ScanNode/IndexNestedLoopJoin/RecursiveExtend/Aggregate…) + `sparql_parse.rs` (31).
- **Feature-detect → shared ArenaIR route (0 push in strategy):** `cypher_parse.rs` (0; the REAL nom parser lives in `lance-graph/src/parser.rs`, then `strategy/arena_ir.rs` (#2) builds the arena) + `gql_parse.rs` (0; explicitly "delegate to CypherParse for shared syntax… ArenaIR will build the plan from detected features", gql_parse.rs:157-159 — sets only feature flags + `estimated_complexity`).

⇒ Both routes are SUPPOSED to land on the same `ir/logical_op.rs LogicalOp` arena, but **NOTHING asserts the two routes (or the 4 dialects) produce equivalent IR for equivalent queries.** That is exactly where silent divergence hides — and exactly what the polyglot IR-conformance loop (P-A) would catch. Concretely actionable: GQL today only flips feature bits; whether ArenaIR reconstructs the same arena Gremlin emits directly is UNTESTED.

**The 4096↔polyglot join (corrected):** the 4096 in THIS crate = the **AutocompleteCache attention topology** (64×64 = 4096 interdependent heads: `cache/{triple_model,kv_bundle,lane_eval,candidate_pool,mod}.rs`), NOT a query codebook and NOT the deepnsm COCA-4096. The link parser→4096 runs: dialect parser → LogicalOp IR → `cache/convergence.rs` ("AriGraph triplets → 8 predicate layers × 64×64 = 4096 heads → CognitiveShader"). convergence.rs is the JOIN — and it is the same module flagged half-built (p64 drift `#[allow(unused_imports)]`, `E-ARIGRAPH-IS-AN-ISLAND`). So "polyglot 4096 in datafusion/planner" = parsers (wired, 2 routes) → IR (wired) → convergence→4096-heads (PARTIAL/dead terminus).

**Datafusion side:** IR → `lance-graph/src/datafusion_planner/` → Spark-dialect unparser (SQL = backend, confirmed). The DataFusion LogicalPlan is the EXECUTION lowering of the same LogicalOp the dialects converge on (the ExecTarget::Native path).

**Revised P-A (the shippable slice), now precise:** polyglot IR-conformance harness asserting (1) the 4 dialects produce equivalent `LogicalOp` for a hand-written equivalent-query corpus, AND (2) the two routes (in-strategy vs ArenaIR) agree — closing the GQL/Cypher-via-ArenaIR vs Gremlin/SPARQL-direct gap. Attaches to `lance-graph-consumer-conformance`. This is the floor that grounds GEL's query-language slice (`E-GEL-IS-THE-GRAPH-SUBSTRATE`).

**Cross-ref:** `E-POLYGLOT-4096-IS-CONJECTURAL`; `E-GEL-IS-THE-GRAPH-SUBSTRATE`; `E-ARIGRAPH-IS-AN-ISLAND` (convergence.rs p64 drift); `strategy/{gremlin,sparql,cypher,gql}_parse.rs`; `strategy/arena_ir.rs`; `ir/logical_op.rs`; `cache/convergence.rs`; `datafusion_planner/`.

---

## 2026-05-30 — CORRECTION + FINDING: E-GEL-IS-THE-GRAPH-SUBSTRATE — GEL = Graph Execution Language (any-language→graph, BEAM-analogous); NOT a query dialect. Corrects "GEL absent" in E-POLYGLOT-4096-IS-CONJECTURAL

**Status:** CORRECTION (supersedes the "GEL/EdgeQL = absent dialect" line in `E-POLYGLOT-4096-IS-CONJECTURAL`) + FINDING (user-stated 2026-05-30: GEL is the user's own coinage, not EdgeDB's rebrand).

**The correction:** I mislabeled "GEL" as EdgeDB's EdgeQL→GEL rebrand and filed it under "absent query dialect." WRONG. **GEL = Graph Execution Language** — a graph SUBSTRATE for representing ANY language in graph form, analogous to how any code lowers to OTP/BEAM/Erlang. GEL is NOT a frontend dialect (not P-C/P-D); it is the IR/substrate layer that frontends lower INTO, represented as an executable graph.

**The BEAM analogy (exact, and ALREADY partially built here):**
- BEAM: Erlang/Elixir/Gleam/LFE → BEAM bytecode; BEAM provides actors (processes) + preemptive scheduling + supervision trees + message passing.
- GEL: any language → graph representation; the graph IS executable.
- **ractor IS the BEAM actor model ported to Rust** — mailboxes = BEAM processes, `lance-graph-supervisor` = supervision tree, `CollapseGateEmission` batons = message passing. So GEL's execution engine = the ractor-over-SoA path being wired (D-MBX-A6 / kanban lifecycle). Not hypothetical.

**Where GEL sits in the stack (it is the substrate everything converges on):**
- **Lowering rule = "AST is the hub"** (cognitive-risc-core): any language → one canonical AST → SPO/graph. GEL = the GRAPH FORM of that hub. `logical_plan.rs LogicalOperator` (query IR) is ONE SLICE of GEL (query languages only); full GEL = any language, not just query.
- **Runtime = `ExecTarget::Elixir`** (shipped #439) literally names "execute on the BEAM-analogous path." The exec-target axis already enumerates GEL's backends.
- **Execution trace = the version arc** (`E-VERSION-ARC-IS-THE-KANBAN` + `E-SUBSTRATE-IS-THE-SCHEDULER`): graph executes → phase commit → Lance version → scheduler fires next. That IS a graph-execution-language running, with the version arc as its trace.
- **State = the SoA** (`E-SOA-IS-THE-ONLY`): GEL nodes/edges = SPO + CausalEdge64 over the one MailboxSoA.

**Prior-art / naming collision to reconcile:** `holograph/src/width_32k/schema.rs` uses "GEL" = "global execution layer" — genuinely adjacent (Global/Graph EXECUTION Layer). Reconcile whether that is the seed, a collision, or unrelated before formalizing the GEL name in code.

**Implication for the polyglot sequence:** GEL reframes it. P-A (IR-conformance loop) still grounds the QUERY-language slice of GEL. But the broader GEL goal = "any language → graph" is the SUBSTRATE the whole D-MBX arc already realizes (IR + SPO + MailboxSoA + ractor lifecycle + ExecTarget + version arc). GEL is the NAME for the convergence, not a new component to bolt on. Do NOT build a "GEL parser"; the work is recognizing/consolidating that the existing IR+SoA+ractor IS GEL, then widening the lowering beyond query languages (LE-4 Odoo/OWL, Elixir templates, etc. = other-session lowerings into GEL).

**Cross-ref:** `E-POLYGLOT-4096-IS-CONJECTURAL` (corrected); cognitive-risc-core "AST is the hub"; `ExecTarget::Elixir` (#439); `E-VERSION-ARC-IS-THE-KANBAN`; `E-SUBSTRATE-IS-THE-SCHEDULER`; ractor=BEAM-actor-model; `holograph/.../schema.rs` (GEL prior-art).

---

## 2026-05-30 — E-POLYGLOT-4096-IS-CONJECTURAL — "loop through 4096 0xFFF polyglot mapping" is NOT wired: 4096 is 4 distinct magic numbers, and frontend->4096->SoA does not exist (string-keyed IR end-to-end)

**Status:** FINDING (grounded read 2026-05-30, file:line agent map). Honest correction of the "loop through 4096 polyglot" framing — labels what exists vs what is conjectural.

**What EXISTS (wired):**
- **Polyglot frontends → ONE IR:** Cypher/GQL/Gremlin/SPARQL all converge on `logical_plan.rs:19 LogicalOperator` (arena IR). Real Cypher parser `lance-graph/src/parser.rs`; the other three are thin Strategy adapters (`planner/src/strategy/{cypher,gql,gremlin,sparql}_parse.rs`) with affinity scoring. Extension shape = Parser + Strategy + affinity arm + 1-line registry (`strategy/mod.rs:51`).
- **SQL = BACKEND only** (Cypher→DataFusion→Spark unparser, `datafusion_planner/`, `spark_dialect.rs`); NOT a frontend.
- **NARS = truth/inference only** (`spo/truth.rs` (f,c); planner `cache/nars_engine.rs`; CausalEdge64 mantissa); NO Narsese parser.
- **GEL/EdgeQL = ABSENT** (the one "GEL" hit = "global execution layer", unrelated).
- **Conformance harness EXISTS** (`lance-graph-consumer-conformance/src/harness.rs` A1-A10) but tests consumer BRIDGES, not dialect IR-equivalence.

**What is CONJECTURAL (NOT wired) — the honest gaps:**
1. **"4096 surface" is 4 DISTINCT uses of the magic number, no canonical type:** (a) deepnsm COCA vocab 12-bit `VOCAB_SIZE=4096` (`deepnsm/vocabulary.rs:19`); (b) AutocompleteCache 64×64=4096 attention heads (`planner/cache/triple_model.rs`); (c) BindSpace 16 prefixes×256 slots (`docs/METADATA_SCHEMA_INVENTORY.md`, design only); (d) COCA² distance matrix. (Palette codebook is 256, NOT 4096.) ⇒ "loop through 4096 0xFFF" is NOT a single enumerable surface today — unification is a prerequisite task, not a loop.
2. **frontend→4096→SoA path DOES NOT EXIST.** The IR is string-keyed end-to-end (`ast.rs:14` `label: String`); the planner never calls `vocabulary.tokenize(label)`; DataFusion scans by string table name; SPO uses FNV-1a hash keys, not 4096 ranks. DeepNSM (the only thing that resolves to 4096) runs PARALLEL to the graph planner, unintegrated.

**The REAL shippable slice (grounded, no conjecture):** the **polyglot IR-conformance loop** — assert the 4 wired dialects produce the SAME `LogicalOperator` for equivalent queries. Attaches to the existing conformance harness; needs only a hand-written equivalent-query corpus + deep-IR-equality. This GROUNDS the polyglot claim (currently untested) and is the prerequisite floor before any 4096 wiring. (`E-SOA-IS-THE-ONLY` / AST-is-the-hub: prove the hub agrees before resolving it to addresses.)

**Sequenced follow-ups (each its own slice, ratification-gated):**
- P-A: polyglot IR-conformance loop (SHIPPABLE NOW — grounds the hub).
- P-B: unify the 4096 surface = declare ONE canonical 12-bit address type (contract/ndarray); collapse the 4 uses onto it. (Prereq for any "loop through 4096".)
- P-C: NARS-as-frontend (Narsese parser → LogicalOperator) — the dialect where query=inference; closes the loop to the (f,c) truth model already present.
- P-D: frontend→4096→SoA = add `label_rank: Option<u16>` to the IR, wire vocab resolution at plan time, SoA column access by rank (OOV fallback). The conjectural path, last + biggest.
- (SQL-as-frontend, GEL: lower priority; SQL is already a backend, GEL absent.)

**Cross-ref:** cognitive-risc-core "AST is the hub"; `logical_plan.rs:19`; `deepnsm/vocabulary.rs:19`; `lance-graph-consumer-conformance`; `ExecTarget` (#439, the backend axis this frontend axis pairs with).

---

## 2026-05-30 — E-SUBSTRATE-IS-THE-SCHEDULER — surreal's time-series/LIVE over the version arc is a cheap planner→execution scheduler firing back INTO the mailbox; the substrate↔mailbox loop is bidirectional

**Status:** FINDING (architectural, user-stated 2026-05-30). Return-path complement of `E-VERSION-ARC-IS-THE-KANBAN`. Together they close the loop.

**The loop is bidirectional:**
- **OUTBOUND (mailbox → surreal), free:** `advance_phase` commit = Lance version = kanban move; surreal LIVE-subscribes (`E-VERSION-ARC-IS-THE-KANBAN`).
- **INBOUND (surreal → mailbox), this finding:** surreal's time-series + LIVE/scheduled query IS a **scheduler**. A surreal scheduled/LIVE event fires back into the mailbox as the next `advance_phase` trigger. The mailbox does NOT run its own planner-tick loop — **surreal schedules it, cheaply**, off the same `versions()` stream.

**This completes the GitHub homology (never one-way):**

| GitHub | Substrate ↔ mailbox |
|---|---|
| push commit → PR | mailbox commit → version arc (outbound, free) |
| CI / scheduled workflow fires → acts on PR | surreal LIVE/scheduled event fires → drives mailbox's next phase (inbound) |
| Actions runner = scheduler | surreal time-series = scheduler |

**Architectural wins:**
1. **planner→execution edge, done cheaply.** Planning precipitates a kanban move (a version); surreal's scheduler watches the arc and fires the execution tick back. `ExecTarget` (#439) = HOW it runs; the surreal event = WHEN. Mailbox stays a pure state machine (`try_advance_phase`, #439); surreal = clock + planner-dispatch.
2. **Two-clock decoupling (RISC core invariant 7) for free:** hot shader speed (mailbox SoA mutation) vs cold scheduler cadence (surreal time-series) — decoupled by construction, same `versions()` stream read both directions. No separate scheduler infra.
3. **`ExecTarget::SurrealQl` made literal:** a scheduled SurrealQL query is BOTH the trigger AND a valid execution backend — the planner→execution path can live entirely in the substrate scheduler for that target. (Native/Jit/Elixir targets: surreal fires the trigger, the mailbox runs the backend.)

**Consequence for D-MBX:** the planner→execution wiring (part of D-MBX-A6-P3 + D-MBX-8 Σ10-commit→ractor-START) gains a substrate-native option — surreal-scheduled tick instead of an in-process planner loop. Still gated by surreal_container fork (OQ-11.6) for the surreal side; the contract side (`ExecTarget`, `try_advance_phase`, `MailboxSoaView`) is already shipped/in-PR and backend-agnostic.

**Open (implementation):** surreal scheduled-query vs LIVE-trigger as the fire mechanism; backpressure when the scheduler outruns the hot path (RISC core invariant 8 — shed by ⟨f,c⟩). Architecture is substrate-native; wiring waits on OQ-11.6.

**Cross-ref:** `E-VERSION-ARC-IS-THE-KANBAN`; surrealdb #31 (Timeline over `Dataset::versions()`); `ExecTarget`/`try_advance_phase` (#439); D-MBX-8/9/A6-P3; cognitive-risc-core invariants 7 (two-clock) + 8 (backpressure).

---

## 2026-05-30 — E-VERSION-ARC-IS-THE-KANBAN — the mailbox's Lance version timeline IS the kanban arc, for free; consume it like a GitHub CI/PR subscription (push, not poll)

**Status:** FINDING (architectural simplification, user-stated 2026-05-30). Grounded in surrealdb #31 substrate fact + Lance versioning. Reframes D-MBX-9.

**The insight:** since kv-lance is native (surrealdb #31: one `MergeInsert`/commit = one Lance dataset version), a mailbox's **`Dataset::versions()` timeline IS its kanban arc — it falls out of the substrate for FREE.** Each `MailboxSoaOwner::advance_phase` commit = one new Lance version = one kanban move. No separate kanban update mechanism is built; the version stream IS it.

**The consumption pattern = a GitHub CI/PR subscription (the exact homology this session ran):**

| GitHub PR | Mailbox kanban |
|---|---|
| commits pushed to a PR | phase-transition commits to the mailbox's Lance dataset |
| CI/review events | new Lance versions appear |
| `subscribe_pr_activity` → pushed to subscriber | surreal LIVE query over `versions()` → kanban updates pushed |
| react per event, never poll | consumer reacts per version, never rebuilds |
| append-only PR timeline | append-only version arc (= witness/belief-state arc, R4) |

**Two consequences:**
1. **Witness pointer (R4 / EW64) = `(mailbox_id, lance_version)`** — a pointer into the substrate's OWN version arc. "Cheap AriGraph witness pointer" is now concrete + free; the AriGraph episodic edge ("happened at the same time") = "happened at version V". The KanbanMove record (incl. `libet_offset_us`, `exec`, `witness_chain_position`) rides as Lance commit/version metadata.
2. **D-MBX-9 collapses** from "build a kanban view structure" to "**LIVE-subscribe to the mailbox version stream**" — surreal time-series view over `Dataset::versions()` + a LIVE query = the Rubicon kanban. The `Timeline` read surface over `Dataset::versions()` already built in surrealdb #31 IS this. The `MailboxSoaView` borrow trait (#437) is the per-version read lens.

**The "in-mailbox arc" framing:** each mailbox owns its own version arc (the versions of its SoA dataset/fragment). The arc is in-mailbox; the kanban is the cross-mailbox time-series view of those arcs. SurrealDB time-series consumes it.

**Open (implementation, not architecture):** true push (surreal LIVE query) vs cheap-poll of `versions()` — both honor the pattern; LIVE is the goal. Still gated by surreal_container fork (OQ-11.6) for the surreal-side view — but the design is now substrate-free, so D-MBX-9 is a subscription wiring, not a build.

**Cross-ref:** surrealdb #31 (kv-lance native + Timeline over `Dataset::versions()`); `E-SOA-IS-THE-ONLY` R3/R4; D-MBX-9; `KanbanMove`/`MailboxSoaView` (#437); `is_absorbing` (#439, the cycle-end commit = a terminal version); LE-3.

---

## 2026-05-30 — D-MBX-A6-P2 landed (contract): Rubicon lifecycle enforcement + ExecTarget strategy tag

**Status:** SHIPPED-in-PR (contract slice). Builds on `#437` (D-MBX-A6-P1).
- `KanbanColumn::next_phases()` + `can_transition_to()` — the Rubicon lifecycle DAG (Planning→CognitiveWork→Evaluation→{Commit|Plan|Prune}; Planning→Prune Libet veto; Plan→Planning re-deliberate; Commit/Prune absorbing). Lifecycle enforcement is now a contract-level, testable invariant.
- `KanbanColumn::is_absorbing()` — distinguishes cycle-END columns (Commit/Prune, tombstone now) from `Plan` (terminal decision but re-deliberates). The ractor driver tombstones iff absorbing; LE-3 cycle-end commit/SLA hooks here.
- `MailboxSoaOwner::try_advance_phase()` — checked default: validates the edge, returns `KanbanMove` or `RubiconTransitionError` (no mutation on illegal). The ractor lifecycle driver uses this.
- `ExecTarget` {Native|Jit|SurrealQl|Elixir} — the planner JIT-adjacent execution-target (strategy) tag; now a field on `KanbanMove` (resolves the `#437` deferred NOTE; size still ≤16 B). Distinct from the planner's 16 composable *planning* strategies.
- Zero-dep preserved; 489 contract lib tests (+4); planner/shader-driver/supervisor cargo-check clean.
Next: ractor MailboxSoA owner-impl + planner emit (candidate generation) — the consumer side.
**Cross-ref:** `#437`; D-MBX-A6; `E-DUPLICATION-IS-INTRINSIC-VS-TEMPORAL`; LE-3 (Rubicon commit).

---

## 2026-05-30 — E-DUPLICATION-IS-INTRINSIC-VS-TEMPORAL — the SoA "duplicate" intentionally separates intrinsic awareness from the temporal belief-state arc (= AriGraph semantic/episodic = CE64/EW64 = SoA1:SoA2)

**Status:** FINDING / design ruling (user-confirmed 2026-05-30). Answers "are you duplicating intentionally?" — YES.

The SoA is FIXED (frozen byte-shape) ⇒ the "duplicate" is the SAME shape instantiated twice, which is what makes `SoA1:SoA2` superposition well-defined. The cut it encodes (one separation, four names):
- **intrinsic awareness** (the *now*): semantic CE64 + gestalt/qualia (`awareness_dto::ResonanceDto`) — atemporal "what resonates now".
- **temporal belief-state arc** (the *over-time*): episodic EW64 witness arc (`chain_position`) — "how the belief came to be, across observations".
= AriGraph **Es (semantic) / Ee (episodic)** duality = **CE64 / EW64** = **SoA1 : SoA2** (shader superposes them, Hebbian) = energy-field / gestalt `ResonanceDto` layering. So `TD-RESONANCEDTO-DUP-1` is NOT a dedup target — it's the intrinsic/temporal separation; the fix is to NAME it (e.g. `ResonanceField` vs `GestaltResonance`), not collapse it.

## 2026-05-30 — LOOSE-ENDS (documented per user "don't die token wall"; NOT yet built)

**LE-1 — EW64 as syntactic-coreference witness pointer (DeepNSM > Markov > grammar).** Wire `deepnsm` (PoS FSM → SPO) → Markov VSA context → grammar heuristics so that a **relative pronoun (Relativpronomen) is NOT bundled** into the VSA trajectory but instead gets a **witness pointer** (EW64) to its antecedent. Rationale: I-VSA-IDENTITIES "register laziness" — exact-match coreference is a POINTER, not a bundle (bundling destroys the register). ⇒ EW64 gains a SECOND role beyond aerial-prefetch: a **syntactic-Markovian context** = episodic memory pointing at BOTH hot witness mailboxes AND cold-path SPO. (Consumers of the grammar/Markov path: `contract::grammar::role_keys`, `deepnsm`, the Vsa16k Markov substrate.)

**LE-2 — cold-path SPO + cold-path AriGraph UNIFY.** Now that EW64 is a *cheap AriGraph witness pointer*, the two cold stores belong together: ONE cold store = semantic SPO (Es) + episodic witness arcs (Ee), EW64 as the link. Resolves the parallel-SPO-store debt (`F-WIRE-DTO-DUP-MAP`/`E-ARIGRAPH-IS-AN-ISLAND`) on the cold side. Matches AriGraph paper (one graph G = Vs,Es,Ve,Ee).

**LE-3 — mailbox-cycle-end Rubicon commit decision.** At the END of a mailbox cycle, the LAST Rubicon kanban state (terminal Commit/Plan/Prune, `KanbanColumn`) = a DECISION that: (a) **commits SPO-W (the EW64 witness) to the cold path** (= AriGraph calcify / `witness_tombstone::calcify` — currently dead `todo!()`, D-ATOM-5), AND (b) for business-logic mailboxes, **commits to SLA + goalstate**. This is the commit gate at Rubicon. Wires `MailboxSoaOwner::advance_phase(Commit)` → cold materialization + SLA/goalstate update.

**LE-4 — Odoo + OWL business-logic action substrate = OTHER SESSION.** The SLA/goalstate business-logic commit (LE-3b) + Odoo/OWL as the business-logic *action* substrate is explicitly deferred to a separate session. Documented here so it is not lost; do NOT build it in this arc.

**BindSpace consumers (singleton→per-mailbox migration surface, grep 2026-05-30):** contract `{cognitive_shader,splat,lib}.rs`; planner `{cache/convergence,lib}.rs`; cognitive-shader-driver `{driver,wire,engine_bridge,mailbox_soa,proposal,serve,wire_dto,spo_bridge,cognitive_shader,cognitive_shader_dispatch,spo_witness,cam}.rs`. The singleton still threads through driver/engine_bridge/wire*; `spo_witness.rs` is the existing witness seam to reconcile with EW64. (Ref D-MBX-3/5: kill `BindSpace::zeros(4096)` singleton; migrate consumers to per-mailbox MailboxSoA.)

**Cross-ref:** `E-ARIGRAPH-PAPER-GROUNDS-CE64-EW64`; `E-AERIAL-FEEDS-EW64-PREFETCH`; `E-ARIGRAPH-IS-AN-ISLAND`; `F-RESONANCEDTO-IS-LAYERED-NOT-DUP`; D-MBX-3/5/A5; I-VSA-IDENTITIES.
## 2026-05-30 — E-DISCOVERY-ORIGIN-HOME-IS-ARIGRAPH-BRIDGE — the `discovery_origin` byte's natural home is the AriGraph hot↔cold bridge column, not the mailbox-SoA byte — surfaced by council prior-art savant 2026-05-30

**Status:** FINDING (R2 council verdict, 2026-05-30, recorded in `AGENT_LOG.md`). Documentation-only; no code or plan modified. Cross-ref: `E-ARIGRAPH-IS-AN-ISLAND` (the prior finding the options doc author missed), `E-ARIGRAPH-PAPER-GROUNDS-CE64-EW64` (CE64 = AriGraph semantic edge; EW64 = AriGraph episodic edge), `.claude/plans/post-438-integration-options-v1.md` §3 (the author's bias the council attacked).

The 2026-05-30 council convening on the post-#438 integration options surfaced an integration target the options-doc author **completely omitted**: the AriGraph hot↔cold bridge. R2 cited `E-ARIGRAPH-IS-AN-ISLAND` + `D-REUNIFY-1/2/3` — the unwired bridge between `arigraph/triplet_graph.rs` (semantic Es) / `arigraph/witness_corpus.rs` (episodic Ee) and the hot SoA (CE64/EW64 cols) / cold Lance store. Per `E-ARIGRAPH-PAPER-GROUNDS-CE64-EW64`, CE64 IS an AriGraph semantic edge; EW64 IS an AriGraph episodic edge; the witness arc IS Ee.

**The reframe:** if `discovery_origin` widens to `u16`, its natural carrier is **a column on the AriGraph bridge**, not the mailbox-SoA byte. Three reasons:
1. Provenance is a property of WHERE-A-FACT-CAME-FROM = what the AriGraph semantic edge encodes (CE64's `discovery_origin` slot lives natively there).
2. The mailbox-SoA byte is a hot-path performance compromise that bakes provenance into the row-level layout; the bridge is the architectural seam where hot meets cold and provenance MUST be materialized per witness-materialization invariant (cognitive-risc-core #5).
3. The wikidata spec's `Derived` tier is *already* declared as living on the reasoning store via `provenance=Derived` column (wikidata-hhtl-load.md:25). The same column home works for the other tiers; the byte is then a hot-path projection of the bridge column, not the source of truth.

**Author's options-doc miss:** all 8 options framed `discovery_origin` as a mailbox-SoA byte question. None considered the bridge-column framing. The kind of architectural drift the prior-art-savant role exists to catch.

**Consequence:** OD-1 (6-bit vs u16) is partially settled by this. If the byte is a projection of a bridge column, the column can be u16 (or any width) without binding the hot-path byte. The byte becomes a cache, not the ISA.

---

## 2026-05-30 — E-ARIGRAPH-PAPER-GROUNDS-CE64-EW64 — AriGraph (arXiv 2407.04363v3) IS the source: its semantic-edge/episodic-edge duality grounds CE64/EW64; the episodic edge ("happened at the same time") IS the witness arc

**Status:** FINDING (read the AriGraph paper, Anokhin et al. AIRI, 2026-05-30). User's "first is AriGraph!!!" — this is the canonical design source for the semantic+episodic arc.

AriGraph world model G = (Vs, Es, Ve, Ee):
- **Es = semantic edges = SPO triplets** `(v, rel, u)` (semantic memory) ⇒ **CE64 = an AriGraph semantic edge** (SPO palette + NARS f/c, 64-bit packed).
- **Ee = episodic edges = `(observation-vertex v_e^t, all SPO triplets E_s^t extracted at t)`** — "connect all triplets that happened at the same time" ⇒ **EW64 = an AriGraph episodic edge = the witness arc.** The board's `witness_chain_position` / belief-state arc IS AriGraph's episodic edge. The CE64/EW64 pairing = AriGraph's semantic/episodic duality, EXACTLY. Grounded, not ad-hoc.
- **"happened at the same time" = Hebbian "fire together".** aerial+ (PR `#436`) GENERALIZES it: offline ARM mines co-occurrence `X→Y` across many observations; AriGraph records per-observation co-occurrence online. Both feed EW64's predictive prefetch (`E-AERIAL-FEEDS-EW64-PREFETCH`).
- **Retrieval = semantic search (Contriever embedding similarity, depth d/width w BFS, Alg 2) + episodic search (relevance-weighted top-k episodic vertices; rel = n_i/max(N_i,1)·log(max(N_i,1))).** Per iron rules: Contriever FLOAT similarity is discovery-layer-only; CAM exact + HHTL facet-AND are the addressing layer. Episodic search = retrieving witness arcs.
- **Ariadne loop = cognitive-RISC 5-layer stack:** AriGraph (semantic+episodic content) = Substrate; retrieval→working-memory + planning (sub-goals) = Compilation; the plan = Schedule (kanban); ReAct decision = Execution (shader); goal/agent = Producer.

**lance-graph impl vs paper:** faithful PORT — `arigraph/triplet_graph.rs` (Es semantic), `witness_corpus.rs`/`episodic.rs` (Ee/Ve episodic), `orchestrator.rs` (Ariadne loop), `retrieval.rs` (semantic search), `sensorium.rs`. But it is an ISLAND (`E-ARIGRAPH-IS-AN-ISLAND`): semantic/episodic edges NOT wired to the hot SoA (CE64/EW64 cols) or cold Lance store. **Wiring task = Es→CE64(hot)+TripletGraph/SpoStore(cold); Ee→EW64(hot prefetch)+WitnessCorpus(cold); under load/store + witness-materialization discipline.**

**Wikidata scale (wikidata-hhtl-load.md):** AriGraph semantic memory at 115M-entity scale = HHTL/CAM — P279 subClassOf DAG = the ONE 16^n tree axis; OWL/DOLCE closed ranges = facet bitmasks (SIMD batch-AND over the SoA facet column = the semantic-search accelerator); CAM shape-dedup; en+de cols; Derived reasoning store (transitive closures) orthogonal+CAM-indexed. 120GB→~38GB structural. = how AriGraph's Vs/Es scale to a world KG.

**Cross-ref:** arXiv 2407.04363v3; `E-EW64-IS-PREDICTIVE-PREFETCH`; `E-AERIAL-FEEDS-EW64-PREFETCH`; `E-ARIGRAPH-IS-AN-ISLAND`; wikidata-hhtl-load + faiss-homology-cam-pq + cognitive-risc-{core,classes}; `arigraph/{triplet_graph,witness_corpus,episodic,orchestrator,retrieval}.rs`.

---

## 2026-05-30 — F-RESONANCEDTO-IS-LAYERED-NOT-DUP — the two ResonanceDto are two abstraction layers (energy-field Ψ vs gestalt-awareness), a name collision masking a MISSING INTEGRATION, not a copy

**Status:** FINDING (both defs read 2026-05-30; reframes `TD-RESONANCEDTO-DUP-1` from "dedup" to "disambiguate + integrate").

- **`thinking-engine/src/dto.rs:59`** = RAW ENERGY FIELD (Ψ interference): `energy: Vec<f32>` (codebook[4096]) + `cycle_count`/`converged`/`top_k[8]`; `from_energy`/`entropy`/`active_count`. Low-level signal in the StreamDto→ResonanceDto→BusDto→ThoughtStruct speed-zone bus.
- **`thinking-engine/src/awareness_dto.rs:21`** = GESTALT + USER MODEL: `hdr: HdrResonance` (3D S/P/O) + `gestalt_state` (Crystallizing/Contested/Dissolving/Epiphany) + `dissonance`/`n_resonant`/`total_energy` + inferred `user_style: ThinkingStyle`/engagement/valence/depth/confidence; `from_superposition`.

⇒ **NOT a true duplicate** — same name, two layers (raw energy vs gestalt-awareness). machete/`TD-RESONANCEDTO-DUP-1` flagged a NAME COLLISION masking a layering. User's read confirmed: it's "just missing the integration."

**Missing integration (the actionable chain):** energy-field (Ψ) → gestalt-awareness → **+ qualia** (today a SEPARATE `QualiaDto`, integrated only at `MomentDto`) → **selection of i4-32D thinking-styles / atoms / strategies**. The awareness ResonanceDto stops at a COARSE `user_style` (3 variants) and does NOT drive the full i4-32D layer (`contract::atoms::I4x32` 33-atom TSV, `contract::recipe_kernels` 34 tactics, `contract::thinking` 36 styles) nor STRATEGY selection (elixir / jit / JITson-Cranelift templates). "thinking about thinking" = NARS meta-layer (`planner::mul` Meta-Uncertainty / Dunning-Kruger + `user_model_confidence`) above the selection.

**Resolution (reframes the TD):** NOT pick-a-winner-and-delete. (1) Disambiguate names (`ResonanceField` for Ψ energy vs `GestaltResonance` for the awareness model). (2) Wire the integration onto the ONE SoA ("ResonanceDto IS the SoA", PR #353): MailboxSoA columns ALREADY carry the substrate — `qualia:[QualiaI4_16D;N]` + `meta:[MetaWord;N]`(thinking/awareness bits) + `entity_type`(class_id) + `edges`(CausalEdge64). resonance→gestalt→qualia→i4-32D-style→strategy converges onto SoA columns + the contract atom/recipe/thinking surface, NOT a third struct. The "dedup" IS an integration onto the SoA — the "one SoA never transformed" convergence.
**Cross-ref:** `TD-RESONANCEDTO-DUP-1` (reframed); `F-WIRE-DTO-DUP-MAP`; SoA-DTO ledger (PR #353); `contract::{atoms,recipe_kernels,thinking,qualia,mul}`.

---

## 2026-05-30 — E-AERIAL-FEEDS-EW64-PREFETCH — aerial+'s mined X→Y associations ARE the predictive-prefetch table EW64 consumes; aerial learns "fire together", EW64 "wires together"

**Status:** CONJECTURE / design (user-stated 2026-05-30). Sweet-spot wiring for the EpisodicWitness64 follow-up; closes the "prefetch WHAT, from where" gap in `E-EW64-IS-PREDICTIVE-PREFETCH`.

EW64 = CPU-style predictive prefetch co-issued with CE64 into the shader. Open question was "prefetch what." **Aerial+ (#436) discovers it:**
- **aerial (discovery / slow / nondeterministic, upstream of firewall):** mines `X→Y` association rules with data-derived `(f,c)` — the Hebbian "when antecedent X fires, consequent Y co-occurs" ("fire together").
- **EW64 (hot / prefetch):** when CE64(X) activates in the SoA, EW64 prefetches the episodic-witness pointer for the aerial-predicted Y (its witness arc), resident BEFORE the shader needs it ("wires together").
- aerial's `(f,c)` → EW64's prefetch confidence; un-ratified aerial rules never reach EW64 (firewall). EW64 shares CE64's low-40 SPO bits so antecedent/consequent co-address (superposition).
⇒ **EW64 type design:** payload = witness-arc pointer to the predicted consequent + confidence/recency lens; populated from the RATIFIED aerial association table; co-issued with CE64.
**Cross-ref:** `E-EW64-IS-PREDICTIVE-PREFETCH`; `E-AERIAL-IS-THE-DISCOVERY-PROPOSER`; `aerial-arm-ruff-spo-codegen-synergies.md`; pr-ce64-mb-4 `SpoWitness64`.

---

## 2026-05-30 — E-ARIGRAPH-IS-AN-ISLAND — AriGraph's cross-crate wirings are nominal/orphaned; the hot→cold bridge is dead; machete corroborates

**Status:** FINDING (read-only audit 2026-05-30, file:line-cited, Opus agent). Pre-existing, NOT introduced by PR #437.

AriGraph (`crates/lance-graph/src/graph/arigraph/`) is almost entirely standalone:
- **Nominal wirings (zero impls):** contract `graph_render`/`sensorium`/`persona` declare provider traits AriGraph is "the producer" for — but those traits have NO impl anywhere; AriGraph's own `GraphSensorium` doesn't implement the contract trait.
- **Dead hot→cold bridge:** `graph/witness_tombstone.rs` (calcify→Tombstone→WitnessLink, the D-ATOM-5 hot→cold snapshot machinery) is ALL `todo!()` AND not declared in `graph/mod.rs` (orphaned/uncompiled). `planner/src/cache/convergence.rs:22-27` p64 drift CONFIRMED dead (`CausalEdge64`/`SpoBase17`/`DistanceMatrix` `#[allow(unused_imports)]`; per-head edge-emission never built; nothing calls `run_convergence`/`update_planes` on a real path). `planner/src/serve.rs` orphaned (`mod serve;` declared nowhere). Contract `counterfactual.rs`/`quorum.rs`/`recipe.rs` exist on disk but NOT in `lib.rs` (the causal-edge→AriGraph `EpisodicEdge` bridge lives in counterfactual.rs, `todo!()` + BLOCKED).
- **Parallel/bypassed:** `lance-graph-osint` has its `lance-graph`(AriGraph) dep COMMENTED OUT — feeds convergence from its own `extractor::Triplet`, never AriGraph.
- **machete corroborates:** `lance-graph` flags `lancedb` (cold-path Lance persistence unwired — matches dead witness_tombstone), `bgz17`/`bgz-tensor` (codec cascade unconnected) unused; `cognitive-shader-driver` flags `prost` (gRPC/serve unwired — matches orphaned serve.rs).
- **Design sound where built:** `spo_bridge::promote_to_spo` is the one LIVE load/store gate; HotWitness/WitnessCorpus/EpisodicMemory snapshot-by-value (no arena pointers) — honor hot→cold-copy by design, just unbuilt. NO live-code violation (bridge is dead, not wrong).
- **Two parallel witness vocabularies:** AriGraph `WitnessEntry`/`WitnessLink` (u64 mailbox_id placeholder) vs PR #437 R4 `witness_chain_position` (contract `MailboxId`) — not unified.

**Action:** record as tech-debt; reconnection (D-ATOM-5 witness_tombstone, p64 convergence terminus, witness-vocab unification) is large + separate. Do NOT strip the machete-flagged deps blindly — cold/codec/serve wiring is intended-but-unbuilt, not truly dead deps.
**Cross-ref:** `F-WIRE-DTO-DUP-MAP`; `E-SOA-IS-THE-ONLY`; cognitive-risc-core invariants 4/5.

---

## 2026-05-30 — E-AERIAL-IS-THE-DISCOVERY-PROPOSER — #436 aerial+ is the runtime-data + class-discovery proposer feeding the ONE SPO/class substrate through the ratification firewall; class_id is its SoA landing

**Status:** FINDING (synergy synthesis, post-#436 rebase, 2026-05-30).

#436 shipped `crates/lance-graph-arm-discovery` (Aerial+ ARM transcode). Synergies with this arc + Cognitive-RISC:
- **Aerial+ = a PROPOSER** in the RISC `discovery_origin` ISA (`ArmDiscovered` tier): a mined association, an AST-walk step, an LLM conjecture = the SAME candidate object differing only by `discovery_origin` (core invariant 9; proposers dumb, Rubicon arbitrates). Aerial is nondeterministic (seeded) → stays UPSTREAM of the ratification council (determinism firewall).
- **Emits SPO+NARS `Triple{s,p,o,f,c}`** into `ruff_spo_triplet` (mirrors `odoo_ontology::OntologyTriple`). Gap: closed predicate vocab rejects `Implies`/`CoOccursWith` (D-ARM-SYN-1; ratification = the determinism firewall, not the brainstorm-council — see correction at top of file). Same SPO substrate the cognitive SoA packs (CausalEdge64 SPO palette + f/c) ⇒ aerial candidates → ratification firewall → CausalEdge64/SPO → kanban (D-MBX-A6) → shader.
- **Aerial ALSO discovers CLASSES** (shape-families): cognitive-risc-classes — taxonomy DISCOVERED "via group-by-on-structural-hash or Aerial+"; splat→aerial→Wikidata discovers OWL/DOLCE+ HHTL classes+basins. ⇒ aerial is the discovery engine behind the `class_id` the SoA needs (the "ontology classes wired into the SoA" ask). Float lives OFFLINE in `jc` (Jirak-Cartan certified 256-codebook); aerial addresses it ONLINE with integer codes (CAM-PQ doctrine).
- **SPO-vocabulary debt (extends F-WIRE-DTO-DUP-MAP):** ≥4 parallel SPO-triple types (AriGraph `TripletGraph`, `ruff_spo_triplet::Triple`, `odoo_ontology::OntologyTriple`, aerial `CandidateTriple`, osint `extractor::Triplet`) — "one SoA never transformed" wants ONE; unification is the convergence work.
- **class_id landing (shipping now):** the SoA's class discriminator IS the existing `entity_type: [u16; N]` (= OGIT `EntityTypeId`); expose it as `MailboxSoaView::class_id()` (N1 freeze hook). Metadata resolves one layer up via `lance-graph-ontology::OntologyRegistry` (perf gap: add O(1) `by_entity_type_id` index; today O(n) `enumerate_first_with_entity_type_id`).
**Cross-ref:** `aerial-arm-ruff-spo-codegen-synergies.md`; `splat-codebook-aerial-wikidata-compression.md`; cognitive-risc-{core,classes,faiss-homology}; PR #437 `MailboxSoaView`.
## 2026-05-31 — E-LANCE7-OBJECTSTORE-SURREALDB — the lance 6→7 bump is what *aligns* object_store with the surrealdb fork; the fork's `kv-lance` `=6.0.0` pins were already self-contradictory against its own object_store 0.13

**Status:** FINDING (deps; verified against crates.io dep graphs + a lock-only `cargo update`, no compile). User directive: "let's do 7 + 0.3 but we need to test surrealdb."

Bumping lance `=6.0.0 → =7.0.0` + lancedb `=0.29.0 → =0.30.0` (lockstep, 7 crates) looked like "ride the head" — but **testing surrealdb showed it's a *coherence fix*, not a version chase.** The AdaWorldAPI/surrealdb fork's `surrealdb-core` declares `kv-lance = ["dep:lance", "dep:lance-index", "dep:lancedb", …]` with `lance / lance-index = "=6.0.0"` + `lancedb = "=0.29.0"`, **yet its workspace already runs `object_store = "0.13.0"`.** The contradiction: lance 6.0.x requires `object_store ^0.12.3`; only lance 7.0.0 moves to `^0.13.2`. So the fork's `=6.0.0` pins could never coexist with its own object_store 0.13 — **the fork is already shaped for lance 7.**

Evidence (crates.io dep API + lock-only resolve): lancedb `0.29.0 → lance =6.0.0`, `0.30.0 → lance =7.0.0` (no lancedb release ever pinned `=6.0.1` — kills the old TD-LANCE-6.0.1-PIN path); lance 6.0.x → `object_store ^0.12.3`, lance 7.0.0 → `^0.13.2`; resolved lock = lance 7.0.0 / lancedb 0.30.0 / object_store 0.13.2 / arrow 58.3.0 / datafusion 53.1.0 (arrow + datafusion **unchanged** across 6→7). **Lesson: a "test the integration" instinct caught a latent diamond the dep-version-only view missed — `object_store`, not `lance`, was the real coupling.** lance-7 API-level compile is unverified in-sandbox (no `protoc`; lance-encoding build-dep) → CI gates it. Cross-ref: TECH_DEBT TD-SURREALDB-KVLANCE-LANCE7; root `Cargo.toml` RESOLVED(A2/B2); PR #445 (lance-graph), companion PR on adaworldapi/surrealdb.

---

## 2026-05-30 — E-ARM-JC-RESOLVES-BOTH-SEAMS — aerial's two open seams (the distance oracle AND the D-ARM-7 Jirak floor) both resolve to `crates/jc`; jc PROVES the splat codebook, aerial USES it to discover the DOLCE skeleton that compresses Wikidata

**Status:** FINDING (architecture; seams concrete, end-to-end pipeline is CONJECTURE). User framing: "gaussian-splat spatial blasgraph top-k 10000×10000 … for OWL/DOLCE+ SPO HHTL classes and basin via aerial+ to deterministically compress Wikidata … adjacent to JC Jirak[-Cartan] with EWA-sandwich gaussian splat."

The de-float (E-ARM-PROBE-IS-CODEBOOK-TOPK) left aerial with two open seams: a production `CodebookDistance` oracle, and the unimplemented D-ARM-7 Jirak significance floor (ISSUE ARM-JIRAK-FLOOR). **Both resolve to the same crate — `crates/jc` (Jirak-Cartan)** — and the grounding is already in the workspace, not invented:

- **Oracle ← jc.** `jc::ewa_sandwich{,_3d}` (Pillars 9/9b) certify the Gaussian-splat Σ-push-forward `J·W·Σ·Wᵀ·Jᵀ` for `ndarray::hpc::splat3d` — i.e. the 10000² BLASGraph spatial top-k that *builds* the codebook is correct. `jc::sigma_codebook_probe` is **literally where ρ=0.9973 comes from**: it measures a 256-codebook capturing the Σ-distribution at R²≥0.99 in log-Euclidean space. `jc::pflug` (Pillar 10) certifies the CAM-PQ/HHTL tree quantization is Lε-faithful (compression is faithful, not lossy-by-surprise). The frozen `[u32; dim²]` table feeds aerial's existing `MatrixDistance`/`CodebookDistance` seam — no new aerial dependency.
- **Jirak floor ← jc.** `jc::jirak` (Pillar 5) is the weak-dependence Berry-Esseen engine: classical IID is wrong here (I-NOISE-FLOOR-JIRAK), the correct rate is `n^(p/2-1)`. D-ARM-7's significance gate derives its threshold from it.

**The float boundary is the punchline.** float lives ONLY in jc's offline build+certify (k-means, SPD/EWA math, Berry-Esseen sup-error); it runs once, emits a frozen integer artifact, and is never on aerial's online path. That is the CAM-PQ doctrine end-to-end — *build the codebook offline (float OK), address it online with integer codes* — and it makes the whole "deterministically compress Wikidata" claim float-free at runtime. Downstream: aerial discovers the OWL/DOLCE+ SPO HHTL classes + basins (the `specs/wikidata-hhtl-load.md` skeleton: P279/P31 DAG + basin assignment, DOLCE as the axis template), the Jirak floor decides which rules are significant enough to persist, and codebook-HHTL is the 16ⁿ bucket router. Full map: `.claude/knowledge/splat-codebook-aerial-wikidata-compression.md`. Cross-ref: jc pillars 5/9/9b/10, `wikidata-hhtl-load.md`, `ogit-owl-dolce-ontology-compartments.md`, `3DGS-HHTL-datalake-traversal-plan.md`, E-ARM-PROBE-IS-CODEBOOK-TOPK.

---

## 2026-05-30 — E-ARM-PROBE-IS-CODEBOOK-TOPK — Aerial+'s reconstruction probe IS a codebook top-k; the autoencoder was a float approximation of a lookup the substrate already does exactly; de-floating it dissolves the determinism firewall

**Status:** FINDING (de-float shipped on `claude/jolly-cori-clnf9`, 28/28 tests, zero f32 in the discovery path). User-directed: "neither cam_pq nor any other crate uses (or should) float … all is deterministic [a,b] codebook distance, ρ=0.9973 spearman."

The first D-ARM-13 transcode reproduced Aerial+ literally — an `f32` denoising autoencoder. That was a substrate regression: this stack addresses by **exact codebook CAM, never float similarity** (`faiss-homology-cam-pq.md`; `I-VSA-IDENTITIES` — CAM-PQ codes are for *search*, integer, never superposed as float). The autoencoder's reconstruction probe ("mark the antecedent, read off the high-probability consequents") is, mechanically, a **nearest-neighbour query** — and the **palette256 distance table** answers that *exactly and in integers* at ρ=0.9973 vs cosine. So the float net was a slow, seed-dependent approximation of a lookup the substrate already performs.

The de-float (codebook-probe backend):
- autoencoder (`f32` weights) → injected integer `CodebookDistance` oracle (palette256; zero-dep trait, real impl is `bgz17::PaletteDistanceTable` / BLASGraph splat top-k / HDR-popcount);
- reconstruction probe → codebook top-k from the antecedent within `θ` (integer);
- softmax ranking → integer distance ranking; support/confidence → integer counts + ppm cross-multiply gates;
- `(f,c)` `f32` → `TruthU8` (= CausalEdge64 `confidence_u8` + i4 mantissa); seeded RNG → deleted.

**The payoff is structural, not cosmetic.** Float was the *only* source of nondeterminism, so removing it dissolves three things at once: (1) the "seeded ≠ bitwise-deterministic" caveat closes — the probe is bitwise-identical on every target; (2) Aerial is no longer a *nondeterministic fan-in* that must hide behind the ratification gate / out of the compile path — it joins the **deterministic trunk** beside pair-stats (D-ARM-3); (3) **D-ARM-9** (Python-IPC to isolate nondeterminism) is fully moot — there is no nondeterminism to isolate. The ratification council still governs *promotion to the SPO store*, but the firewall is now about ratification, not float. General rule this crystallises: **a "learned" similarity that the codebook reproduces at ρ≈1.0 should never be reached for as float — the codebook IS the learned-once model, frozen and integer.** Code: `crates/lance-graph-arm-discovery/src/aerial/{codebook,extract,mod}.rs`. Cross-ref: `faiss-homology-cam-pq.md`, `cognitive-risc-classes.md` (CAM-not-ANN), `I-VSA-IDENTITIES`, the prior float transcode's review (`.claude/board/reviews/`).

---

## 2026-05-30 — E-DISCOVERY-CODEGEN-BRACKET-1 (realised) — the Aerial+ transcode is the runtime-data frontend of a bracket whose substrate + codegen legs ALREADY EXIST in the ruff fork; the ruff SPO predicate vocabulary is the only missing seam

**Status:** FINDING (type-level, grounded in source read 2026-05-30) + CONJECTURE (the three D-ARM-SYN wiring deliverables). Author-stated; the three wirings pass the determinism-firewall ratification (nondeterministic stays upstream) — NOT the brainstorm-council, which is a catalyst not a gate (see top-of-file correction).

The two-paper bracket (`streaming-arm-nars-discovery-v1.md`: Aerial+ discovery upstream, Abreu M2M codegen downstream, SPO+NARS middle) is not a future aspiration — **its substrate and both codegen legs are already implemented in `adaworldapi/ruff`** and `lance-graph-ontology`. Transcoding Aerial+ to Rust (D-ARM-13) made this concrete:

1. **One substrate, three frontends.** `ruff_spo_triplet::Triple { s, p, o, f, c }` *"mirrors `lance_graph::graph::spo::odoo_ontology::OntologyTriple` field-for-field"* and its ndjson is *"exactly what `parse_triples` reads."* Three proposer frontends converge on it: `ruff_python_dto_check` (static Python AST → `Extracted`/Authoritative), `ruff_ruby_spo` (Rails scaffold), and **`lance-graph-arm-discovery` (Aerial+, runtime data → `ArmDiscovered`)**. The first two are bounded by the literal artifact; only the ARM leg surfaces co-correlations that exist solely in runtime rows.

2. **Two codegen legs, one thesis.** `op_emitter.rs` (ratified SoA → deterministic Rust dispatch) and `ruff_python_codegen::round_trip` (AST → deterministic Python) are the same externalise-interpretation thesis the Abreu paper validates, in two target languages. Both sit downstream of the ratification firewall.

3. **The truth scale is already shared.** Aerial's `arm_to_nars` produces the exact `(f, c)` that `TruthValue::new` and `ruff_spo_triplet::Provenance::truth()` carry, and `NarsTruth::expectation()` reimplements `TruthValue::expectation` so one `TruthGate` covers mined and extracted facts alike.

4. **The one missing seam (the actionable finding).** `ruff_spo_triplet::Predicate` is a **closed vocabulary** (`rdf:type, has_function, emitted_by, depends_on, reads_field, raises, traverses_relation`) and `from_ndjson` **hard-rejects** anything else — and **none of them is an implication/association relation.** An `X → Y` ARM rule therefore cannot flow through that loader until `Implies`/`CoOccursWith` is added (a *deliberate* ontology change per that crate's own doc). This is D-ARM-SYN-1; it gates SYN-2 (the `CandidateRule → ModelGraph` adapter) and SYN-3 (the `ArmDiscovered` truth calibration below the codegen gate).

**Determinism boundary (unchanged, reaffirmed):** Aerial+ is the only nondeterministic node in the bracket. The transcode keeps it standalone, seeded (`aerial::Rng`), behind the `aerial` feature, and emitting a `CandidateRule` *proposal* — never a committed triple. Promotion is the council's job. Full map: `.claude/knowledge/aerial-arm-ruff-spo-codegen-synergies.md`. Code: `crates/lance-graph-arm-discovery/`. Cross-ref: `E-INTERPRET-NOT-STORE-1`, `E-SOA-IS-THE-ONLY`, `I-NOISE-FLOOR-JIRAK`, Karabulut 2025 §2/§3.3, Abreu 2025 §4.
## 2026-05-30 — E-EW64-IS-PREDICTIVE-PREFETCH — EpisodicWitness64 is a CPU-style predictive prefetch co-issued with CausalEdge64 into the cognitive shader; (4x4)^4 L1-4 derives from cortex/hippocampus; "fire together wire together" = CE64/EW64 SoA1:SoA2 superposition

**Status:** CONJECTURE / design (user-stated 2026-05-30). Design basis for the queued `EpisodicWitness64` (`SpoWitness64`, pr-ce64-mb-4) follow-up.

- **EW64 = predictive prefetch.** Like a CPU prefetcher pulls a cache line *before* the instruction needs it, `EpisodicWitness64` predictively prefetches the episodic context (the witnessed SPO arc) so that when CE64 + EW64 enter the cognitive shader *together*, the episodic prior is already resident. EW64 anticipates; CE64 is the causal "instruction"; they are co-issued.
- **"What fires together wires together" = CE64/EW64 SoA1:SoA2 superposition** (Hebbian). Superposing the two SoAs in the cognitive shader (the inside / zero-copy path) IS the wiring step — co-activation of CE64 (causal) + EW64 (episodic) over the same CAM address space strengthens the binding. Grounds the earlier "SoA1:SoA2 superposition inside cognitive-shader-driver" as Hebbian plasticity.
- **(4x4)^4 (L1-4) derives from cortex + hippocampus.** The shader's 4-level (4x4)^4 block layering (L1-4) is brain/hippocampus-derived (cortical microcircuit / hippocampal indexing), not arbitrary tiling. Ground the exact L1-4 ↔ cortical-layer / hippocampal-subfield mapping when speccing EW64.
- **Type implication:** EW64 shares CE64's low-40 SPO+NARS bits (co-addressing for superposition) + an episodic/prefetch lens (recency, salience, witness-arc pointer); prefetched ahead of the cycle; CE64+EW64 issued as a pair.

**Cross-ref:** `E-SOA-VIEW-IS-A-BORROW`; pr-ce64-mb-4 `SpoWitness64`; `causal-edge/src/edge.rs` (CE64 layout to mirror); "The Click" (AriGraph/episodic = thinking tissue); D-MBX-A3 (witness-arc handle column).

---

## 2026-05-30 — F-WIRE-DTO-DUP-MAP — ResonanceDto is the duplicated one (2 defs in thinking-engine); StreamDto/BusDto single-def cross-crate Wire DTOs; P64 = convergence crate

**Status:** FINDING (grep audit 2026-05-30, user-requested DTO hunt).
- **ResonanceDto — DUPLICATED:** two defs, BOTH in thinking-engine — `awareness_dto.rs:21` AND `dto.rs:59` (41 uses thinking-engine + 3 cognitive-shader-driver). = existing `TD-RESONANCEDTO-DUP-1` (Deferred → fold into D-MBX-2). Per the SoA-DTO ledger "ResonanceDto IS the SoA" — dedup converges both onto the one SoA shape, not pick-a-winner.
- **StreamDto — single def** (`thinking-engine/src/dto.rs:40`; 5 + 3 uses). LAB Wire DTO (input carrier into CognitiveShader).
- **BusDto — single def** (`thinking-engine/src/dto.rs:120`; 54 + 48 + 1 uses). Heavily cross-crate bus transport DTO.
- **P64 — convergence crate** (`p64-bridge` 17, bgz-tensor 8, planner 6, shader-driver 5, osint 1); no single `P64` type — it's the convergence point (CLAUDE.md: "p64 = where both repos meet, no circular deps").

**Action:** ResonanceDto dedup stays `TD-RESONANCEDTO-DUP-1` (tied to D-MBX-2 SoA convergence — dedup onto the one SoA per "one SoA never transformed"). StreamDto/BusDto are NOT duplicated; cross-crate spread = expected LAB Wire surface. No new action unless dedup is prioritized now.
**Cross-ref:** `TD-RESONANCEDTO-DUP-1`; SoA-DTO entropy ledger (PR #353); `lab-vs-canonical-surface.md`.

---

## 2026-05-30 — E-SOA-VIEW-IS-A-BORROW — the transparent SoA view surrealdb needs is a zero-dep contract *borrow trait*, not a DTO; the read/owner split makes "view is read-only" structural

**Status:** FINDING (derived; subject to `epiphany-brainstorm-council` per PR #433). Builds on the author-stated R1 ("one SoA never transformed") + R4 (witness-as-pointer) rulings in `E-SOA-IS-THE-ONLY` — those pre-exist and are not council-gated; this is the contract-shape consequence.

The planner⟷ractor⟷surrealdb wiring (user-requested 2026-05-30) is realized WITHOUT a new DTO family (`lab-vs-canonical-surface.md` §"Decision Procedure"): extend the canonical `OrchestrationBridge`/`UnifiedStep` surface (`StepDomain::Kanban` + `"kanban."` prefix) + add `kanban::{KanbanColumn, KanbanMove}` + a zero-dep borrow trait `soa_view::MailboxSoaView` returning `&[T]` column slices. The borrow trait is the key move: it lets the in-RAM `MailboxSoA<N>` (ractor-owned, in cognitive-shader-driver), a surreal kv-lance view, and the planner all read the SAME bytes through one vocabulary — the dependency-inversion pattern already used by `PlannerContract`/`OrchestrationBridge`, so the contract stays zero-dep (it cannot name `MailboxSoA<N>` from another crate without a dep). The `MailboxSoaView` (read) vs `MailboxSoaOwner` (mutate `advance_phase`) split makes "the view is read-only" a *structural* guarantee (surreal implements only the read half) — honoring R1 by type, not convention.

**Cross-ref:** `E-SOA-IS-THE-ONLY` (R1/R4 origin); `lab-vs-canonical-surface.md` §Decision Procedure; `unified-soa-convergence-v1.md §5+§8.4` (D-MBX-A6); LATEST_STATE Contract Inventory 2026-05-30.
## 2026-05-30 — E-DISCOVERY-ORIGIN-WIDTH — the discovery_origin byte is over-subscribed in BOTH fields, and the Jirak threshold has a 3-place reciprocal bug

**Status:** FINDING (verified on-disk, file:line cited). Documentation-only; no code/plan changed. Full detail: `.claude/knowledge/discovery-origin-provenance-reconciliation-v1.md`. Author-stated; not council-gated.

Surfaced answering the user's "I don't know what is correct" about discovery_origin / ProvenanceTier, after they supplied the canonical `cognitive-risc-*` specs.

1. **`discovery_origin` exists in ZERO `.rs` files** — only 7 `.claude/` docs. The WAL has not hardened around it; the "ISA-ossification trap" window is OPEN, fix cost = markdown edit, not migration.
2. **proposer-id width is wrong everywhere, monotonically.** Committed plan §7.2 = 2 bits (4 slots, full); #434 review = 3 bits (8); canonical core spec = 3 bits but declares it insufficient and says widen to **6 bits (64) or u16**. The committed version is the most-wrong.
3. **ProvenanceTier is ALSO over-subscribed (nobody had flagged this).** Six distinct tier names exist across the corpus — Curated/Extracted/Conjecture/ArmDiscovered/Ratified/Derived — but every layout gives the field 2 bits = 4 slots. My own ARM plan contradicts itself: §7.2 enumerates 4, D-ARM-1 enumerates 5. Code today (`mod.rs:450 OdooConfidence`) has only {Curated, Extracted, Conjecture}. The canonical specs disagree with *themselves* (core: ArmDiscovered/Ratified; wikidata: Derived).
4. **Jirak threshold reciprocal bug, verified in 2 of 3 places.** Correct rate `n^{-(p/2-1)}` (matches CLAUDE.md `I-NOISE-FLOOR-JIRAK` and the plan's own worked examples + blockquote line 375). Plan line 381 (prose) and line 393 (pseudocode `powf(-1.0/(p/2-1))`) write the reciprocal `n^{-1/(p/2-1)}`. At n=1e5, p=2.5 the bug makes the floor ~1e-20 instead of ~0.056 — i.e. silently disables the noise floor the iron rule calls "not optional." Also: default `p=3.0` sits exactly at the classical Berry-Esseen crossover, so the "stricter than IID" claim is false at the default; use `p≈2.5`.

**Open decisions (user's, not resolvable by citation):** OD-1 6-bit vs u16; OD-2 fate of Conjecture + treat Derived as a separate reasoning-provenance axis; OD-3 code/spec divergence on Conjecture. None applied — held for the user.

---

## 2026-05-29 — E-SOA-IS-THE-ONLY — there is ONE SoA, never transformed; mailbox SoA mutation IS the hot path; Libet −550 ms anchors the Rubicon kanban in surrealkv-on-lance; SPO-W witness is a *pointer* via the belief-state arc

**Status:** CONJECTURE / design (user-stated 2026-05-29, post-PR-#433). Records five layered rulings; details in `bindspace-singleton-to-mailbox-soa-v1.md` §11.1–§11.5. Author-stated; not council-gated.

1. **One SoA, never transformed (§11.1).** The mailbox SoA is the *single* universal carrier across the stack — never re-encoded into a different shape. Only three operations are allowed on it: (a) cognitive-shader thinking (hot path), (b) cold-path read/write to LanceDB, (c) AriGraph Markov-chain context building. *Any change in any mailbox SoA = the only hot-path activity.* Today's `crates/lance-graph` containers are *cold-path-adjacent thinking* — only accidentally aligned. Realignment (`D-MBX-7`): lance-graph containers ≡ `MailboxSoA` layout ≡ `ndarray::simd_soa.rs` alignment, unlocking **1.4–4.2× SIMD acceleration**. Nice-to-have today; **hard prerequisite** for the SurrealDB transparent view (§2.7).

2. **Mailbox = full BindSpace, reinvented as LE; witness = belief-state arc (§11.2).** Carry everything BindSpace had, but as LE-contract types (never `Vsa16kF32`). The **witness IS the per-row arc of `CausalEdge64` emissions** (`CollapseGateEmission` arc) — that arc *implicitly* documents NARS revision because every emission stamps `confidence_u8` + `inference_mantissa`; reading the arc IS the `(frequency, confidence)` trace. **No separate revision log column.** D-MBX-A1 columns (`edges`/`qualia`/`meta`/`entity_type`) landed; D-MBX-A2 closes remaining BindSpace-expressivity gaps; D-MBX-A3 adds the witness-arc handle column.

3. **Libet −550 ms anchors the Rubicon (§11.3).** Concretises `E-RUBICON-RACTOR` with wall-clock: pre-decisional = counterfactual deliberation; **commit at t = −550 ms** (Libet readiness-potential anchor) = Σ10 commit = ractor START of the actional phase; Libet "free won't" = pre-(−550 ms) CollapseGate veto / ghost-tier preempt; post-actional → ractor STOP → tombstone. The **Rubicon kanban lives in `surrealkv`-on-lance** (a *view* over leading LanceDB, §2.7), columns = the 4 action phases; ractor lifecycle transitions = kanban moves. (`D-MBX-8` timing anchor; `D-MBX-9` kanban view.)

4. **SPO-W witness is a *pointer*, not stored data (§11.4).** The witness is an arc-handle into the belief-state-arc array; the pointer lives equivalently in the **mailbox SoA**, the **kanban row**, or the **mailbox index**. **The SoA itself decides** whether to commit a belief as a fact-with-witness into (a) other mailboxes (inter-mailbox baton handoff carrying the pointer) or (b) cold-path facts (LanceDB SPO-G calcification linking back to the arc). No storage redundancy: the witness lives where the arc lives. (`D-MBX-A5`.)

5. **Counterfactual Staunen × Wisdom = plasticity spreaders (§11.5).** When a mailbox is pre-Rubicon (counterfactual phase), high Staunen × Wisdom *spreads* plasticity beyond the focal row (Hebbian spread). Hot-path-only — the spread is a mailbox-SoA mutation, never a side channel. Radius/decay TBD. (`D-MBX-A4`.)

**Refinements (2026-05-29, same session):**

- **§11.3 kanban refined.** The Rubicon kanban has **4 explicit columns**: **Planning** (ractor mailbox owned SoA, counterfactual) → **Cognitive work** (Σ10 commit + actional SoA mutation) → **Evaluation of goalstate** (post-actional reflection on witness arc) → **Commit · Plan · Prune** (terminal 3-way decision: calcify / re-deliberate / ghost-preempt). Supersedes the earlier 4-Heckhausen-phase mapping in this entry.
- **§11.4 sharpened.** "Witness in other mailboxes" means a *pointer into the AriGraph episodic Markov chain* — the temporal sequence of mailbox states *is* the episodic memory substrate (CLAUDE.md "The Click": *"AriGraph, episodic memory, SPO, CAM-PQ are thinking tissue — not storage"*). No parallel episodic structure exists.
- **§11.6 — the "half-baked nine" all consume THE same SoA from A-Z.** AriGraph · Markov-grammar Vsa16kF32 substrate · BindSpace · `lance-graph` cold containers · `lance-graph-planner` · `cognitive-shader-driver` · `lance-graph-callcenter` · `lance-graph-ontology` (read-only, AS IS) · thinking-styles/atoms — every one consumes THE single SoA byte layout at every boundary it crosses. The SoA gets a **version byte at the layout root** so older bytes stay readable after schema upgrade (`D-MBX-10`, governed by `I-LEGACY-API-FEATURE-GATED`). For SurrealDB to provide a transparent (zero-copy) view, versioning aligns with the **Lance 6.0.1 / LanceDB 0.29 / DataFusion 53** stack — only one bump pending: `lance =6.0.0 → =6.0.1` patch (`D-MBX-11`); arrow/datafusion/lancedb already on target. The kanban/ractor lifecycle requires a **lance-graph-planner DTO surface overhaul** (`D-MBX-A6`) re-expressing planner DTOs as operations on the SoA + 4-phase kanban transitions.

**Cross-ref:** `E-BATON-1`, `E-MAILBOX-IS-BINDSPACE`, `E-RUBICON-RACTOR` (Heckhausen + Libet origin), `E-LADDER-SERVES-MAILBOX` (§5 ghost preempt = Libet veto, §6 tombstone), `I-VSA-IDENTITIES`, `I-LEGACY-API-FEATURE-GATED` (governs the SoA version gate), `causaledge64-mailbox-rename-soa-v1` §10 + E-CE64-MB-8 (Σ10 router), `linguistic-epiphanies-2026-04-19.md` E21 (Σ10 Rubicon tier doctrine), D-CSV-10 (`SigmaTierRouter` shipped #388), `bindspace-singleton-to-mailbox-soa-v1` §11.1–§11.6 (full ruling + D-MBX-A2/A3/A4/A5/A6/7/8/9/10/11/12 + OQ-11.1–11.8); `epiphany-brainstorm-council` (PR #433 pre-merge gate — bypassed here because rulings are author-stated).

---

## 2026-05-28 — E-NORMALIZED-ENTITY-1 — `NormalizedEntity<Stage>` is the single typed carrier holding the four-way inheritance chain (Odoo → OGIT → OWL → DOLCE → FIBU/FIBO); stage advancement is typestate, not method calls on a context

**Status:** FINDING (architectural unification). Drives `normalized-entity-holy-grail-v1`. The carrier is a typed lens into a `MailboxSoA` row — it does NOT own the four cognitive columns; the mailbox does. The 4-way inheritance slots (`odoo`/`ogit`/`owl`/`dolce` + optional `fibu`) populate as stages advance; phantom-typed `Stage` parameter forbids out-of-order traversal at compile time. Consumers chain ON the carrier (`entity.resolve_ogit(ctx).hydrate_owl(ctx)...`), never reach into its internals.

---

## 2026-05-28 — E-OP-FIVE-VERBS-1 — only five universal verbs span the unification: `resolve` (Odoo → OGIT) · `hydrate` (OGIT → OWL) · `classify` (OWL → DOLCE) · `align` (DOLCE → FIBU/FIBO) · `think` (the op-chain over the normalized carrier)

**Status:** FINDING. Every business operation is a special case of one of the five. The temptation to add a sixth verb (`reconcile`, `report`, `aggregate`) is a sign the operation should compose multiple `think` steps, not add to the algebra. The five-verb closure parallels the Vsa16k algebra closure (`bind`/`bundle`/`cosine`) — small algebra, large surface.

---

## 2026-05-28 — E-OP-THREE-CALLSITES-1 — `Op<I,O>` is one trait with three call sites (`apply` cold · `apply_stream` warm · `apply_soa` hot), one set of const data shared across all three; same heuristic, three execution speeds

**Status:** FINDING. The cold path runs the kernel once per entity (DataFusion-routed). The warm path maps it over a `Stream` (per-element flow-controlled). The hot path is a SoA-swept SIMD kernel (JIT-compiled from the same const data). The Op's `kind()` is its register-layer identity per `I-VSA-IDENTITIES`; the shader dispatches on kind. Consumers don't pick the call site — the transaction context does.

---

## 2026-05-28 — E-TRANSACTION-CONTEXT-1 — three typed transaction shapes own commit + Baton epoch + Lance version policy: `Interactive` (eager cascade, live Lance, sync DFS) · `Bulk` (epochal flush, per-batch snapshot, async) · `Periodisch` (JIT chain, frozen Lance, iterate-to-fixed-point)

**Status:** FINDING. The corollary of `E-OP-THREE-CALLSITES-1`: the three call sites map 1:1 to the three contexts; the consumer's typed enclosure (`woa.interactive { ... }` / `woa.bulk { ... }` / `woa.periodisch { ... }`) picks the call site, the cascade traversal mode, the Lance version pinning, and the Baton epoch boundary. Same chain shape inside; different commit discipline outside.

---

## 2026-05-28 — E-CASCADE-AS-EDGECOLUMN-1 — dependency cascade collapses Odoo's six overlapping mechanisms into ONE typed graph on `EdgeColumn`; transaction context picks the traversal discipline

**Status:** FINDING (conjecture pending Stage-2 enumeration). Odoo encodes cascade in: (1) `@api.depends` strings, (2) `@api.constrains` post-write hooks, (3) SQL FK `ondelete`, (4) `base.automation` server actions, (5) `_inherits` field forwarding, (6) implicit model cascades (mail-thread auto-subscribe, tax-tag aggregation). We unify all six as `CausalEdge64` rows on the mailbox's `EdgeColumn` with a `CascadeKind` discriminant. Traversal mode (sync DFS / async batched / JIT-fixed-point) comes from the transaction context. The six-into-one collapse is the structural improvement over Odoo's prior art.

---

## 2026-05-28 — E-ODOO-AS-PRIOR-ART-1 — Odoo solved the three regimes (interactive / bulk / periodisch) 15 years ago via `@api.depends` strings + `env.context` flags + lock-date wizards; we re-encode the same decomposition as compile-time typed boundaries

**Status:** FINDING. Odoo got the decomposition right — three SLAs, three commit disciplines, three cascade modes. What hurts in Odoo is the ENCODING: stringly-typed dependency declarations, runtime-evaluated context flags, multi-screen close wizards. Failure mode is runtime drift. Our re-encoding (typestate + Op-with-three-call-sites + typed transaction contexts) makes the same three regimes fail at COMPILE time. The decomposition is borrowed; the failure timing is the improvement.

---

## 2026-05-28 — E-CONSUMER-CANNOT-INTERPRET-1 — business heuristics MUST be expressible as SIMD-amenable const data; regex / hand-rolled `if line.account.code.starts_with("84")` is structurally banned because the chain does not expose that primitive

**Status:** FINDING (iron-rule candidate). The structural ban is the point: today consumers CAN write hand-rolled business logic because the type system permits it; CodeRabbit/Codex will not catch it. Post-migration, the only way to "check an SKR range" is `chain.chk_data(SkrAccountInRange::new(8400..=8499))` where `SkrAccountInRange` is a typed Op the shader dispatches against `SKR03_CHART`. Hand-rolled regex becomes a MISSING FUNCTION, not a code-review finding. Pairs with `I-VSA-IDENTITIES`: consumer-side interpretation is identity-layer drift.

---

## 2026-05-28 — E-NO-AUTOMATIC-REGIME-PICK-1 — the cognitive shader does NOT autonomously choose between hot / warm / cold execution; the consumer's typed transaction context does (correction of the earlier "shader picks based on flow rate + surprise" framing)

**Status:** FINDING (mid-session correction). The cute framing — "shader picks hot vs cold based on flow rate" — conflates three SLA regimes that have genuinely different correctness requirements: interactive MUST see live data (no frozen snapshot), periodisch MUST NOT see writes after fiscal cutoff (frozen point-in-time). A shader that switches modes based on flow pressure can silently break either invariant. The consumer's enclosing typed context (`Interactive` / `Bulk` / `Periodisch`) is the only authority for which mode is correct; the shader executes within whichever mode the context dictates. Pairs with the SoA-as-AGI doctrine: AGI is the SHADER'S behaviour under SoA dispatch, but the regime under which that dispatch runs is the CONSUMER'S typed declaration.

---

## 2026-05-28 — E-CODEBOOK-INHERITS-FROM-OGIT — every identity (entities, savants, atoms, ontology classes, regulation rules, accounts) lives as a codebook entry inherited from OGIT; LE-byte SoA per mailbox stores the codes; bitpacked u64 is a desperation-bucket fallback; the SoA doesn't guess

**Status:** FINDING (architectural correction, supersedes the role-key-as-canonical interpretation of `I-VSA-IDENTITIES`; drives the v2-step codebook foundation in `contract::callcenter::ogit_uris`).

**Click (the 2026-05-28 user-given doctrine, distilled across four messages):**

1. **`Vsa16kF32` deprecated** as cross-boundary carrier (matches CLAUDE.md "The Click" 2026-05-26 baton-scoping update). Inter-mailbox state is the `(u16, CausalEdge64)` baton; ephemeral in-mailbox compute may still use `Vsa16kF32`.
2. **Every data lives as LE-byte SoA per mailbox** (`E-MAILBOX-IS-BINDSPACE`). Row content = codes; never raw bits.
3. **Codebook for everything — including the semantic ontology graph.** Entities, savants, atoms, ontology classes, regulation rules, accounts (Kontenrahmen positions) all get codebook entries.
4. **Inherited from OGIT, because the SoA doesn't guess.** Identities are deterministic — `OntologyRegistry` resolves OGIT URI → stable row index. No hashing, no FNV-seeded random bits, no autogenerated IDs.
5. **Bitpacked u64 (RoleKey slices in `Binary16K`) is a desperation-bucket fallback.** Useful only when codebook lookup is unavailable (ephemeral Hamming compare); not canonical identity.
6. **The codebook is RICH, not flat** — Kontenerkennung is the canonical example: each entry carries a parent chain (SKR03 → SKR04 → custom) + NARS-truth confidence per link + multi-dimensional dispatch over `(business type × transaction type × form × regulation × law × entity × product)`.
7. **The audit query is the load-bearing read**: *"how were similar transactions / entities handled before · what patterns are required by regulation · what currently exists · what needs audit vs what is confidently repeating existing patterns"* — answered by composing episodic memory + AriGraph SPO-G audit-witness chain + regulation-ontology codebook + NARS confidence threshold + Pearl 2³ on the emitted `CausalEdge64`. High confidence → repeat (no audit); below floor → escalate.

**Mechanism (the codebook layer, full shape):**

- **Identity:** OGIT URI under `https://ogit.adaworldapi.com/<domain>#<Name>` (e.g. `callcenter/savants#FiscalPositionResolver`). `OntologyRegistry::resolve(uri)` → stable row index (u32 codebook code).
- **Inheritance:** typed parent chain per entry, NARS-truth-weighted per link (deduction down a chain, revision across siblings).
- **Multi-dim applicability:** each entry carries selectors over (business × transaction × form × regulation × law × entity × product); dispatch resolves the cross-product match.
- **Regulation as ontology:** legal/regulatory patterns (HGB, GoB, AO, UStG, IFRS, GoBD) are themselves OGIT URIs with inheritance + applicability; floor-of-compliance is a typed predicate over them.
- **Audit threshold:** NARS confidence on a `CausalEdge64` emission < `audit_floor` → escalate (LLM resolves the <25% tail per CLAUDE.md "The Click"); ≥ floor → confident pattern repetition committed to AriGraph SPO-G as both the action and its audit witness.
- **Storage:** LE-byte SoA columns per mailbox store the codebook codes (u32 row indices). No bitpacked planes. The `Baton` (`(u16, CausalEdge64)`) carries the code across mailbox boundaries.

**Fix (what lands today + what's queued):**

- **Today (this commit):** `contract::callcenter::ogit_uris` ships the canonical OGIT URI per savant + `savant_ogit_uri(id) → &'static str` lookups (8 tests). `contract::callcenter::role_keys` stays compiled as the documented desperation-bucket fallback (module doc updated). `D-ODOO-BP-1b` Wave 1 (L1–L5) already shipped 4008 lines of typed `OdooEntity` content (`e4c747a`); Wave 2 (L6–L10) running in parallel.
- **Queued (separate D-ids):** (a) `savants.ttl` under `data/ontologies/ogit/callcenter/` + `OntologyRegistry::hydrate_from_*` wiring for the 25 savants; (b) Kontenerkennung-style inheritance + NARS confidence per link (`CodebookEntry { uri, parent: Option<u32>, link_truth: NarsTruth, applicability: ... }`); (c) regulation-ontology codebook (HGB / GoB / AO / UStG / IFRS / GoBD as OGIT URIs); (d) audit-threshold dispatch in `cognitive-shader-driver` (NARS confidence vs `audit_floor` → `CausalEdge64` Pearl 2³ + AriGraph SPO-G witness OR escalation Baton).

**Lesson (generalizes):** "consult before guess" (CLAUDE.md §"Driving Seat") extends beyond grepping the codebase — it extends to **identity itself**. The SoA doesn't guess: every row's identity comes from a deterministic codebook lookup (OGIT → registry → row index). Inventing identities (FNV-seeded random bits in a slice, hash of a name) is the same anti-pattern as inventing types that already exist — both are 30-turn rediscovery taxes. The fix is the same: route every identity through `OntologyRegistry`.

**Cross-ref:** `E-MAILBOX-IS-BINDSPACE` (LE-byte SoA per mailbox), `E-BATON-1` (`(u16, CausalEdge64)` cross-boundary state), `I-VSA-IDENTITIES` (Layer-2 catalogue doctrine — clarified: catalogue is OGIT-resolved, NOT bitpacked random bits), `E-SAVANT-COMPOSITION-1` (Reasoner trait surface was wrong — codebook foundation is the canonical identity layer the typed compositions reach for), CLAUDE.md "The Click" 2026-05-26 baton-scoping update (Vsa16kF32 deprecated as carrier), CLAUDE.md INTEGRATION_PLANS line 470 (Pillar 1: "OGIT as universal SPO-G lingua franca with `ontology_context_id: u32` per named graph"), `.claude/plans/odoo-savant-reasoners-v2.md` (v2 reshape — Group F per-savant compositions now compose over OGIT-codebook identities, not RoleKey slices), `.claude/plans/odoo-business-logic-blueprint-v1.md` (BP-1b content layer is OGIT-compatible via `model_name` → `classify_odoo`/`hydrate_odoo`).
## 2026-05-28 — E-SAVANT-COMPOSITION-1 — the `Reasoner` trait surface (D-ODOO-SAV-4, PR #420) is the wrong shape: savants are typed compositions over `CausalEdge64` + `Tactic` + `callcenter/role_keys`, not service impls

**Status:** FINDING (architectural correction, drives `odoo-savant-reasoners-v2`).

**Click:** v1's "MED on dispatch shape" caveat resolved on review against CLAUDE.md "P-1 The Click" + "P0 AGI-as-glove". v1 shipped `Reasoner` trait + 4 `*Reasoner` impls (`CustomerCategoryReasoner`, `NextBestActionReasoner`, `PostingAnomalyReasoner`, `OtherReasoner`) + `SavantConclusion` struct + `SavantSuggestion` enum + `build_conclusion(savant, ctx)` free function. **Three doctrine litmus tests in CLAUDE.md name this verbatim**: (1) "new capability lands as a new column, not a new layer" — `Reasoner` is a new layer; (2) "free function on a carrier's state = reject" — `build_conclusion(savant, ctx)` is the named anti-pattern; (3) "wrap the axes in a new struct = breaks the SIMD sweep" — `SavantConclusion`/`SavantSuggestion` duplicate `CausalEdge64`. The substrate v1 should have composed instead is already shipped: **`CausalEdge64`** (the 64-bit causal neuron with SPO palette + NARS truth + Pearl 2³ + inference mantissa — *the conclusion IS the emitted edge*), **`Tactic` trait + 34 kernels** (PR #411 ratified explicitly as "the Elixir-like recipe layer ... later fronts the real fingerprint substrate via cognitive-shader-driver with no change to the 34 call sites"), **33-TSV atoms** (PR #411 `contract::atoms::CANONICAL_ATOMS`), **role-key catalogues** (`I-VSA-IDENTITIES` names `callcenter/role_keys.rs` as the future Layer-2 home).

**Mechanism (right shape):** each savant = (role-key identity in `callcenter/role_keys.rs` — one Vsa16kF32 slice per savant) + (typed composition of `Tactic` dispatches over `ThoughtCtx`) + (`CausalEdge64` emission spec: SPO palette + Pearl 2³ + v2 signed mantissa). No new trait. No new conclusion type. No new dispatch service. The shader runs the composition over the SoA columns; the savant's "decision" rides in the emitted `EdgeColumn` row. Per `E-BATON-1`: cross-boundary state IS the `(u16, CausalEdge64)` baton, so the conclusion *literally is* the baton's edge.

**Fix:** `odoo-savant-reasoners-v2` plan + INTEGRATION_PLANS PREPEND + STATUS_BOARD rows (D-ODOO-SAV-5a..5e) + this epiphany — landed in the same commit per board hygiene. v2 keeps v1 compiling under `legacy-reasoner` feature with `#[deprecated]` migration pointers (per `I-LEGACY-API-FEATURE-GATED`) until woa-rs migrates its `Reasoner::reason()` call sites to `SavantPattern` resolution. Removal in a follow-up PR after the migration.

**Lesson (generalizes):** the "consult before guess" rule in CLAUDE.md §"Driving Seat" (3) names this failure mode — *"Grepping ndarray for a primitive name when the family-codec-smith agent or the `encoding-ecosystem.md` knowledge doc has the answer is a rediscovery tax, not a diligence win."* v1 was four trait impls shipped before reading PR #411 (`recipe_kernels.rs` + `Tactic` trait), `causal-edge/edge.rs` (the canonical edge primitive), or `vsa-switchboard-architecture.md` (the three-layer Layer-2 catalogue doctrine). Every one of those contained the answer to "where does the savant's dispatch live." The codex P1 review on PR #420 was the surface signal; this is the underlying diagnosis. Going forward: a plan with "MED on dispatch shape" should ship with the review pass done *before* the implementation D-id, not as a follow-up.

**Cross-ref:** PR #420 (v1 ship), PR #411 (33-TSV atoms + 34-tactic `Tactic`), PR #418 + `E-BATON-1` (mailbox-owned SoA, CE64 as cross-boundary state), `causal-edge/edge.rs` (v2 layout, signed mantissa), `lance-graph-contract::recipe_kernels` (the `Tactic` trait + 34 kernels), CLAUDE.md "P-1 The Click" + "P0 AGI-as-glove" litmus tests, `I-VSA-IDENTITIES` (Layer-2 role catalogues), `I-LEGACY-API-FEATURE-GATED` (deprecation discipline), `.claude/plans/odoo-savant-reasoners-v1.md` (the shipped v1) → `.claude/plans/odoo-savant-reasoners-v2.md` (this reshape).
## 2026-05-27 — E-AUDIT-1 — `with_jsonl_audit` jsonl-feature build break: a default-feature `cargo check` masks feature-gated error-type mismatches

**Status:** FINDING

**Click:** `UnifiedBridge::with_jsonl_audit` (added in PR #366 as the OQ-7-3 opt-in constructor, `#[cfg(feature = "jsonl")]`) was typed `-> std::io::Result<Self>`, but its body is `JsonlAuditSink::new(...)?` and `JsonlAuditSink::new -> Result<Self, AuditError>`. `AuditError` carries `Io(#[from] std::io::Error)` (the io→AuditError direction) but there is **no** reverse `From<AuditError> for std::io::Error`, so the `?` could not coerce `AuditError` into the declared `io::Error` return — E0277. The **default-feature `cargo check` skips this path entirely** (the constructor is feature-gated), so the break only surfaced when CI built `--features jsonl`.

**Fix (commit `ea2a378`, branch `claude/activate-lance-graph-att-k2pHI`):** one line — return the honest error type `Result<Self, crate::audit_sink::AuditError>`. Two equivalent ~3-line forms existed: (1) change the return type to `Result<Self, AuditError>` [taken — "W2's instinct"]; (2) add a crate-wide `impl From<AuditError> for std::io::Error` and keep the `io::Result` signature. Form 1 chosen: the form-2 coercion would lossily flatten `ChannelFull` / `Serialize` / `SchemaMigration` / `Lance` / `Arrow` into `io::Error::other`, lying about the failure class for every future caller. Zero callers depended on the old signature (grep across all cloned repos: only the def + one doc-comment mention), so the signature change broke nothing. `cargo check/test -p lance-graph-callcenter --features jsonl` clean (137 tests).

**Lesson (generalizes):** a `#[cfg(feature = …)]` fn whose body uses `?` across two error types is invisible to a default-feature `cargo check`. Any crate with optional-feature error paths needs a CI matrix that builds each feature (or `--all-features`), else these E0277s ship to whoever first enables the feature — here, the still-queued MedCare-rs sprint-2 item 5 that consumes this constructor. And: error-type honesty beats `From`-coercion convenience — return the real domain error rather than a lossy `into()` to a narrower std type.

**Cross-ref:** PR #366 (constructor introduction, OQ-7-2/7-3 locks); `audit_sink/mod.rs::AuditError`; TD-SDR-AUDIT-PERSIST-1 in TECH_DEBT.md (the JSON-as-debt reframe — orthogonal to this signature fix).
## 2026-05-27 — E-RUBICON-RACTOR — the Σ10 Rubicon commit IS the Heckhausen action-phase crossing; ractor start/stop = crossing/closing the Rubicon; Libet "free won't" = the pre-commit veto; the kanban is a SurrealDB VIEW over leading LanceDB storage

**Status:** CONJECTURE / design-grounding. Names the psychological origin of the *already-shipped* Σ10 Rubicon doctrine. **Provenance note:** "Σ10 Rubicon" is canonical and implemented (`SigmaTierRouter` Rubicon-resonance dispatch, `D-CSV-10` shipped PR #388; origin `linguistic-epiphanies-2026-04-19.md` E21), but **"Libet" and "Heckhausen" appear nowhere in the board/code/transcripts** — that grounding was a different session or verbal, recorded here now.

**The model.** Heckhausen/Gollwitzer's **Rubicon model of action phases**: pre-decisional *deliberation* (motivational, reversible) → **crossing the Rubicon** (intention formed = volitional, *irreversible* commitment) → pre-actional → *actional* → post-actional *evaluation*. **Libet**: the readiness potential precedes conscious decision; conscious will's role is the **veto** ("free won't") — abort before the act.

**The mapping (orchestration = ractor start/stop):**
- **Deliberation (pre-decisional)** = a mailbox accumulating energy, **no commit** — reversible; the `MailboxSoA.energy` integrates baton receipts (`apply_edges`) below threshold.
- **Crossing the Rubicon** = the **Σ10 commit**: `ΔF < threshold AND resonance > Rubicon-bar` (`SigmaTierRouter`, D-CSV-10) → **ractor START of the actional phase** — irreversible: the baton emits (`CollapseGateEmission`), the row's contents flow through L3 to AriGraph/SPO (calcify). "Opinions are committed contradictions preserved" = post-Rubicon irreversibility.
- **Libet veto ("free won't")** = the CollapseGate **pre-commit abort window**: a mailbox can be preempted/vetoed *before* crossing (ghost-tier preemptible to zero — `E-LADDER-SERVES-MAILBOX` §5). The veto is the only "free will" lever; the readiness potential (energy ramp) is mechanical.
- **Post-actional evaluation → ractor STOP / die** → tombstone-witness persists (`E-LADDER-SERVES-MAILBOX` §6). Start=spawn-at-crossing, stop=evaluate-and-die is the ractor outer-swarm lifecycle (`D-PERSONA-5`, async at the boundary; inner Click stays sync).

**The kanban = a SurrealDB VIEW over leading LanceDB.** The Rubicon action-phase board (deliberation | crossed/intention | actional | evaluated) is a **SurrealDB view projecting LanceDB rows** — moving a thought across columns = ractor start/stop transitions. **LanceDB is the leading storage (source of truth, append-only/versioned); SurrealDB-on-`kv-lance` is a view/query surface over it, never a separate store** (corrects any "SurrealDB-on-Lance is the cold tier" framing — the cold tier is LanceDB; SurrealDB is one view over it). See `bindspace-singleton-to-mailbox-soa-v1` §2.7.

**Cross-ref:** `linguistic-epiphanies-2026-04-19.md` E21 (Σ10 Rubicon tier doctrine), `causaledge64-mailbox-rename-soa-v1` §10 + E-CE64-MB-8 (SigmaTierRouter = substrate-tier router), `cognitive-substrate-convergence-v*` (D-CSV-10 Rubicon-resonance, #388), `E-LADDER-SERVES-MAILBOX` (§1 ractor outer-swarm sync/async, §5 ghost preempt = veto, §6 tombstone), `E-MAILBOX-IS-BINDSPACE` + `bindspace-singleton-to-mailbox-soa-v1` §2.7 (LanceDB-leading / SurrealDB-view), `D-PERSONA-5` (ractor outer-swarm runtime). **Open:** whether to add a `linguistic-epiphanies` entry naming the Heckhausen/Libet origin alongside E21 (awaits go).

---

## 2026-05-27 — E-MAILBOX-IS-BINDSPACE — the singleton `Arc<BindSpace>` dissolves *onto* mailboxes: `MailboxSoA<N>` *becomes* the per-mailbox ephemeral thoughtspace (BindSpace surrogate), it is NOT copied per mailbox

**Status:** CONJECTURE / design-ruling (migration spec authored this session; NOT yet implemented). Extends `E-BATON-1` + `E-CE64-MB-4` downward into the column layout. Plan: `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md`.

**The correction:** the ask began as *"MailboxSoA has an individual **copy** of the BindSpace."* That is still a singleton (N synchronized copies = the aliasing problem `E-CE64-MB-4` kills). The ruling: **there is no global address space to copy.** `MailboxSoA<N>` *is* the BindSpace for the life of one think-arc — each mailbox **owns** its per-row SoA columns (born in the mailbox, die with it), and the shared singleton `Arc<BindSpace>` is **dissolved**, not sharded.

**Current singleton (to migrate):** `crates/cognitive-shader-driver` — `ShaderDriver.bindspace: Arc<BindSpace>` (`driver.rs:56`); one `Arc::new(BindSpace::zeros(4096))` in `bin/serve.rs:29`; per-row read/write surface in `engine_bridge.rs`. `BindSpace` (`bindspace.rs:234`) carries a **64 KB/row `Vsa16kF32` `cycle` plane** (`FLOATS_PER_VSA=16384`) — 256 MB across the 4096-row singleton.

**Column migration (full map in the plan §3):** `cycle` `Vsa16kF32` plane → **DROP** (ephemeral local compute, never a column — `E-BATON-1`); `content/topic/angle` dense planes → **reference** (CAM-PQ code ≤6 B), not own (`I-VSA-IDENTITIES`); `edges`(`CausalEdge64`)/`qualia`(i4-16D)/`meta`(u32)/`entity_type`(u16) → **own** in `MailboxSoA`; `temporal`/`expert` → fold into `CausalEdge64`/mailbox identity (OQ-2); **`ontology: Arc<OntologyRegistry>` STAYS shared** (cold Zone-2, read-only). Per-row hot footprint drops ~71.6 KB → ~24–30 B.

**Why it's safe:** mailbox *moves* batons in (`apply_edges`) and *moves* emissions out (`CollapseGateEmission`), so the borrow checker proves no shared-mutable aliasing of the thoughtspace — the guarantee the `Arc<BindSpace>`+`CollapseGate` "read-only by convention" only enforced by discipline becomes a **compile error** (`E-CE64-MB-4`).

**Gated staging (plan §6):** S1 add columns behind `mailbox-thoughtspace` feature → S2 move `engine_bridge` surface onto mailbox rows → S3 driver holds a sea-star of mailboxes (kill the 4096 singleton in `serve.rs`) → S4 death→SPO+Lance tombstone-witness → S5 delete `BindSpace`+`cycle` plane. Gated on `D-CE64-MB-1-impl` (par-tile) + `PR-NDARRAY-MIRI-COMPLETE`; S5 blocked on the CLAUDE.md "The Click" / `Vsa16kF32` doctrinal update (OQ-4, already flagged in `surreal/RECONCILIATION`).

**Refinement (same session, 2026-05-27):** the per-mailbox SoA *is* **THE little-endian contract** — singular, and the **same SoA layout runs the whole vertical with no boundary re-encode**: cognitive-shader-driver `MailboxSoA` → `lance-graph-contract` LE types (`CausalEdge64`/`QualiaI4_16D`/`MetaWord`/`SoaColumns`/`entity_type`) → lance-graph storage (Lance columns / tombstone-witness); `ShaderCrystal.persisted_row` is a pointer to the same SoA row, not a serialized copy (plan §2.5). **The Ontology is NOT in the SoA and stays AS IS** — lazylock (`registry.rs:39 LazyLock<NamespaceRegistry>`) + the `ontology_dictionary` Lance **cache** (`lance_cache.rs`, TTL-sourced, drop-and-rebuild; its own header already says "BindSpace … never lands here") (plan §4). **DTO audit (plan §2.6):** `p64-bridge` already conforms (maps `CausalEdge64`/`ThinkingStyle` straight to palette, no re-encode — the template); the legacy re-encode seam is `engine_bridge.rs` `bind_busdto`/`unbind_busdto`/`busdto_to_binary16k` (collapse in S2); `thinking-engine` DTOs survive only as the `StreamDto` ingress adapter + thin read-projections (`BusDto`/`ThoughtStruct`) over the mailbox SoA; `ResonanceDto.energy` *is* `MailboxSoA.energy` (the two `ResonanceDto` defs are `TD-RESONANCEDTO-DUP-1`, **deferred**). **Hot/cold (plan §2.7):** the SoA extends past RAM — `ThoughtStruct` is *later also a transparent view into the SurrealDB ThoughtStruct container table(s)*, same SoA layout hot (mailbox) or cold (container), no RAM↔storage re-encode. Hot ceiling **~64k–256k thoughts** (64k ≈ 300–600 MB ⇒ ~6 KB/thought, dominated by the content/topic/angle Hamming planes that stay hot — dropping only the 64 KB `Vsa16kF32` plane is what makes the working set fit; **resolves OQ-1**). New deliverable D-MBX-6.

**Cross-ref:** `E-BATON-1`, `E-CE64-MB-4`, `E-LADDER-SERVES-MAILBOX` (§6 tombstone-witness), `I-VSA-IDENTITIES`, `I-LEGACY-API-FEATURE-GATED` (feature-gate the v1 accessors during S1–S4), `E-CONTRACT-NO-SERIALIZE` (compile-time handshake, no membrane serialize — same byte layout to disk), `causaledge64-mailbox-rename-soa-v1` (§5 MailboxSoA), `cognitive-substrate-convergence-v1` (D-CSV-7 shipped accumulator), `TD-RESONANCEDTO-DUP-1`.

---

## 2026-05-27 — E-CONTRACT-NO-SERIALIZE-2 — correction to E-CONTRACT-NO-SERIALIZE (below): the audit event never leaves the inside; "serialize at the membrane" was the wrong half — audit is not membrane traffic at all

**Status:** FINDING (sharpens the entry directly below; user correction via the question "why should the audit event go outside?"). The §1 half of the prior entry — *contracts compile types, never serialize; build-time serde codegen is fine* — stands. This entry replaces its "outer membrane's job" framing (board is append-only, so the prior entry is left intact and corrected here).

The audit event does **not go outside**, and there is no reason for it to. It is a **cognitive-compliance witness** — a merkle-chained event (`merkle_root` / `prev_merkle`) that **calcifies into SPO + a Lance columnar tombstone** (cf. `E-LADDER-SERVES-MAILBOX` §6). It is **examined in place** — lance-graph *is* a query engine, so HIPAA §164.312(b) "audit review" is a query against the witness, not an export to a SIEM. The merkle chain is the tamper-evidence; no external append-only file is needed for integrity.

- **No JSON by default.** `JsonlAuditSink` / `with_jsonl_audit` are the legacy "ship logs to Splunk" pattern this stack obsoletes — not a sanctioned boundary.
- **The audit sink is inner, not membrane.** It belongs with the SPO/Lance tissue, behind the membrane — never in the outer client-facing layer. "Emit via the membrane sink" (prior entry's "Correct shape") was wrong.
- **Off-box durability / external-auditor copies are an infra concern** — replicate the durable Lance/merkle artifact, or do a deliberate on-request export *action* at the storage edge. Egress as an explicit act on the artifact, never the sink's standing behavior, never the client membrane.

So the lance-graph-side direction stands but with a corrected target: relocate the concrete `JsonlAuditSink` out of `lance-graph-callcenter` as an at-most-optional export adapter; callcenter keeps only the `AuditSink` trait + `UnifiedAuditEvent` type; the **canonical sink is the SPO/Lance witness projection**, not a JSON file at the membrane.

Cross-ref: medcare-rs `CLAUDE.md` commitment #7 (corrected in MedCare-rs #159); prior entry `E-CONTRACT-NO-SERIALIZE` below.

---

## 2026-05-27 — E-CONTRACT-NO-SERIALIZE — a contract crate is a compile-time handshake (shared types + traits), NOT an outer serialization boundary; JSON emission belongs at the membrane, never on the BBB/contract surface

**Status:** FINDING (architectural vow, user-stated 2026-05-27 via the medcare-rs consumer session; recorded for the next session that touches the audit-sink / bridge surface — no lance-graph code change in this entry).

A contract / BBB-tier crate (`lance-graph-contract`, `lance-graph-ontology`, `lance-graph-callcenter`) exists to make producer and consumer **compile against the same types + traits** — a handshake vow. It must not turn values into JSON or any wire/file format. Serialization is the **outer membrane**'s job (on the medcare consumer side that is `medcare-realtime`, the Supabase/Foundry-equivalent boundary).

On the audit surface specifically:

- **Contract-appropriate (keep):** the `AuditSink` trait (the vow — "hand me typed events, I decide what to do") and the `UnifiedAuditEvent` type (the shared shape both sides build against).
- **Violation (the smell this vow names):** `JsonlAuditSink` + `UnifiedBridge::with_jsonl_audit` living in `lance-graph-callcenter`. A contract-tier crate emitting JSONL is acting as a serialization boundary.
- **Correct shape:** `UnifiedBridge` already takes `with_audit_chain(super_domain, salt, Arc<dyn AuditSink>)` — the trait, the handshake. The concrete JSON-emitting sink should be supplied by the membrane. Proposed direction: relocate the concrete `JsonlAuditSink` out of callcenter into a membrane/sink crate; callcenter keeps only the `AuditSink` trait + `UnifiedAuditEvent` type.
- **NOT a violation:** build-time serde codegen — e.g. `lance-graph-contract/build.rs` parsing `modules/*/manifest.yaml` (serde_yaml) to *generate* Rust types (#412). That IS "compile types"; the crate's runtime `[dependencies]` stay serde-free.

Cross-ref: medcare-rs `CLAUDE.md` commitment #7 (consumer-side record of the same vow). The medcare bridge-audit path currently leans on callcenter's JSON sink; the gate path correctly emits via the membrane sink — reworking the bridge path to match is tracked on the medcare side, not done here.

---

## 2026-05-27 — E-LADDER-SERVES-MAILBOX — the escalation ladder serves the *mailbox*, not the persona; atoms (bipolar I4-32D) are the bottom layer, measured by *quorum*, with split-poles preserved as a counterfactual mantissa; AriGraph is ephemeral-hot in the mailbox and calcifies to cold SPO + a Lance tombstone-witness

**Status:** CONJECTURE / design-synthesis (a session design arc, anchored to four FINDING-grade iron rules below; NOT yet implemented). Refines `rung-persona-orchestration-v1` (the *name* "persona" is demoted — see §1 of this entry). Supersedes nothing; extends `E-CHECKLIST-AS-ESCALATION` (D-PERSONA-1) downward (atoms) and outward (mailbox lifecycle).

**Grounded anchors (FINDING):** `I-VSA-IDENTITIES` (persona = Layer-2 catalogue; Test 0 register-laziness; bipolar ±1 role keys; Test 2/3 orthogonality+cleanup), `E-BATON-1`/Baton-scoping (mailbox-as-owner, bundle ephemeral, no persisted singleton), `I-LEGACY-API-FEATURE-GATED` (`CausalEdge64` 4-bit signed mantissa @46-49, `InferenceType::Counterfactual = to_mantissa() = −6`), The Click (Staunen×Wisdom = Contradiction magnitude; "opinions are committed contradictions preserved, not resolved"; `awareness.revise`; F<0.2 Commit / ΔF<0.05 Epiphany / F>0.8 FailureTicket).

---

### §1 — The ladder serves the mailbox; "persona" was mis-centered

`rung-persona-orchestration-v1` framed the escalation→epiphany→ghost ladder as serving a *persona*. **Wrong primary object.** Per `I-VSA-IDENTITIES`'s own unification — *"Archetype / persona / thinking-style … are Layer-2 role catalogues; each entry gets ONE identity fingerprint; content lives in YAML; resonance dispatches to content"* — **persona is a dispatch policy, not a container.** The owned unit is the **mailbox** (mailbox-as-owner, sea-star). The ladder is *mailbox* machinery; a persona only decides *what to fan out as mailboxes* and *where the temperature/β knob sits*. The D-PERSONA-1 **types are already mailbox-shaped** (`Checklist`, `CollapseHint`, `InnerCouncil`, `GhostEcho`) — so this is a reframe + (pending) rename, not a rebuild. Three personas = three policies over one substrate:

| persona | β / temperature | fan-out pattern |
|---|---|---|
| business | cold (exploit) | business-logic checkboxes → supervised mailboxes, **respawn-if-failed** (bounded) |
| chat | warm (explore) | episodic persona-modeling + self-state-awareness **over witness-arcs** (never a persisted self-singleton — E-BATON-1) |
| OSINT | **annealing** (hot→cold) | self-generated hypothesis-mailboxes; cross-style synthesis = the Layer-1 `a2a_blackboard` driven *autonomously*; provenance→NARS-confidence gates the Rubikon (untrusted sources never commit as fact); preserve `dissonance`, never average |

Constraint: checkbox-as-mailbox fan-out + respawn lives at the **outer swarm boundary** (ractor, async), the inner Click stays sync — do not double-mailbox (E-BATON-1). Respawn needs a bounded supervision policy (N retries → `FailureTicket`), or crash-loop.

### §2 — Three layers: atoms → thinking styles → persona recipes

- **Atoms** — the bottom layer. The current `contract::thinking` "36 ThinkingStyles" are **demoted to atoms**, encoded **I4-32D**: 32 *bipolar* dimensions → **64 poles** (32−, 32+) = 16 bytes/vector. The atom set is the orthogonal basis + cleanup codebook (satisfies `I-VSA-IDENTITIES` Test 2/3, which 36 ad-hoc style-fingerprints did *not*).
- **Thinking styles** (Kant, Schopenhauer, bookkeeping-savant) = I4-32D **compositions** over atoms.
- **Persona recipes** = compositions of styles + thresholds, purpose, β.

**JIT placement:** atoms + styles stay **I4-32D data**; the **persona recipe** is what gets templated into a Cranelift `KernelHandle` (`contract::jit` `StyleRegistry`). A 32-D i4 dot is one SIMD sequence — Cranelift overhead only amortizes at the fused-recipe level. Add-atom = data; add-style/persona = template (Elixir-style hot-load open/closed split).

**Pole budget (user's allocation, with the atom-kind caveat):** the named axes that are genuinely *bipolar-continuous* — trust↔DK (one calibration axis, **not** two — see §3), wisdom↔Staunen (= temperature, §4), plasticity (rigid↔plastic), 6 hardwired business-logic dichotomies (FIBU/GoBD §6b) — are correct atoms. **But NARS inference-type / strategy / semiring are *categorical selectors*, not bipolar magnitudes** (`contract::nars` = `InferenceType(5)` + `QueryStrategy(5)` + `SemiringChoice(5)` + Pearl 2³ ≈ the "24"). By Test 0 (register-laziness) those belong in an **enum/register that gates which atoms fire**, not as poles. Allocating ~24 of 32 dims to NARS likely miscategorizes discrete selectors as continuous atoms. NARS *truth* (frequency, confidence) IS continuous → atoms; NARS *type* → register. **OPEN:** the atom-basis derivation (ICA/PCA over the 36 / theory-driven from the 6 clusters / hybrid) is the load-bearing unsolved design step. (i4 precision floor: documented tradeoff, cite `FormatBestPractices.md` Jirak; SIMD path gated on MANDATORY `ndarray-vertical-simd-alien-magic.md`.)

### §3 — The crux: a dichotomy needs a *quorum* to project, and a split is not averaged

A bounded dichotomy does not yield its projection for free. To place a measurement between two poles you need a **quorum**; a *split* quorum means the projection is **contested, not merely noisy**. Every one of the 32 axes inherits this — the universal cost of bipolar structure. The quorum machinery already exists: **`InnerCouncil` `is_split(0.7,0.5)` + ×1.2 split-amplify** (3-archetype vote) and the Layer-1 **`a2a_blackboard` `support[u16;4]` + `dissonance`** (wide quorum). Therefore **each I4-32D atom value is a quorum *output*, i.e. a pair `(I4 position, quorum-confidence)` = NARS truth `(frequency, confidence)` applied per axis.** A split is held as a **Contradiction, never averaged** (averaging contested state = laundering false confidence; the cardinal OSINT sin).

### §4 — wisdom↔Staunen = temperature (control axis, self-regulated)

This axis is *not* a measured feature like trust/DK — it is **sampling temperature** (Wisdom = low-temp/sharp/exploit; Staunen = high-temp/diffuse/explore), the same knob as the EFE explore/exploit β. It is **self-regulated by free energy** (thermostat): high surprise → Staunen → hot → wide sampling; F descends → Wisdom accrues → cools → commits. **This retroactively explains the `WisdomMarker` 0.1 floor** built in D-PERSONA-1: that floor *is the minimum temperature* — the φ-1 "permanent humility" ceiling means cognition never anneals to absolute zero. Distinct from **plasticity** (update-rate): you can run hot-but-rigid or cold-but-plastic, so both keep separate dims. Open layout question: temperature as a flat peer dim, or a **meta-atom read first** that sets the sampling sharpness for unbinding the other 31 (one-pass vs two-stage I4 sweep).

### §5 — Split resolution = counterfactual mantissa (replaces quorum-tiering)

On `is_split`, do **not** widen the quorum through tiers (too complex). Instead **commit the majority pole and fork the minority pole into a counterfactual-testing mailbox, retained as an episodic mantissa.** "Mantissa" is literal: the minority pole is a single `CausalEdge64` **−6 (Counterfactual) nibble** in the episodic witness — the road-not-taken costs **4 bits**, not a replay buffer (committed pole = coarse exponent / direction; counterfactual = fine mantissa / "could-have-been"). This is the mechanical form of *"contradictions preserved, not resolved"*, satisfies the counterfactual-stays-in-a-separate-lane rule (Counterfactual-tagged, never written as observed SPO truth), and IS the rung-ghost's counterfactual-learning fuel. **Loop closes via revision:** the counterfactual mailbox is **ghost-tier** (preemptible to zero, tests only on β headroom); if its test later beats the committed pole (lower F), that is a **NARS `awareness.revise`** on the original axis — the road-not-taken reopens and can overturn the verdict. **Staging:** v1 commit-majority/drop-minority → v2 deposit the −6 mantissa (contradiction-honesty for 4 bits, no spawn) → v3 full counterfactual-mailbox + revision loop. Spawn gated on dissonance/Staunen > threshold; ghost Staunen-keyed GC prunes counterfactuals that never pay.

### §6 — AriGraph: ephemeral-hot in the mailbox, calcified-cold in SPO + Lance tombstone-witness

AriGraph is **not a persisted singleton graph** (E-BATON-1). Its live/episodic form is **ephemeral inside the mailbox** (the working hot path). When a fact stabilizes it **calcifies into the SPO ontology** (cold, persistent). When the ephemeral mailbox dies (sea-star spawn→die→merge), its **witness persists as a tombstone in Lance**, linking the calcified cold fact back to the mailbox that committed it. This is one **compression hierarchy down the codec atlas**:

```
hot (full-fidelity ephemeral AriGraph, in mailbox)
 → calcified semantic  (SPO-G quads, cold, Lance)            — "what is believed"
 + tombstone witness   (Lance, compressed ~Scent/Base17)     — "what happened / who committed it"
 + counterfactual residue (CausalEdge64 −6 mantissa, 4 bits) — "the road not taken"
```

Fallout: because Lance is append-only/**versioned**, the tombstone layer *is* the audit trail — GoBD/provenance falls out of the substrate by construction (`E-FIBU-GOBD-BY-CONSTRUCTION`, §6b), not as bolted-on logging. The one thing to nail is **link integrity**: the calcified SPO fact needs a durable back-pointer to its tombstone, and the tombstone must outlive the mailbox — Lance versioning is the right home for both.

---

**Proposed deliverables (NOT yet queued — pending greenlight):** D-ATOM-1 atom-basis derivation + I4-32D layout; D-ATOM-2 style/persona Cranelift recipe templates; D-ATOM-3 quorum-projection `(position, confidence)` per axis; D-ATOM-4 counterfactual-mantissa v2 (deposit) then v3 (mailbox+revision); D-ATOM-5 AriGraph hot→calcify→tombstone wiring.

**To-verify (cross-refs asserted from this session's dialogue, confirm against board before relying):** the `WitnessCorpus` deliverable D-id (CAM-PQ-indexed witness + salience decay) and the `SigmaTierRouter` Rubicon-resonance Σ-tier D-id were cited in-conversation as the homes for the tombstone-witness and the Rubikon admission gate respectively — confirm exact ids in `STATUS_BOARD.md` / `PR_ARC_INVENTORY.md`.

**Still pending separately (NOT folded here):** the substrate-Markov **re-scope** (substrate Markov = guarantee for *unsolicited materialization* only; episodic Markov is the governing transition account) awaits the `[FORMAL-SCAFFOLD]` dependency check (do the four pillars need substrate Markov as the *transition account* or only as the *guarantee*?) before it can be written as an `I-SUBSTRATE-MARKOV` amendment. The `rung-persona-orchestration-v1` → mailbox-centric **rename** also awaits explicit go (touches D-ids).

**Cross-ref:** `I-VSA-IDENTITIES`, `E-BATON-1`, `I-LEGACY-API-FEATURE-GATED`, `E-CHECKLIST-AS-ESCALATION` (D-PERSONA-1: `InnerCouncil`/`EpiphanyDetector`/`GhostEcho`/`WisdomMarker`), `E-FIBU-GOBD-BY-CONSTRUCTION` (§6b business atoms + GoBD audit), `E-OGIT-STAKES-LINCHPIN` (stakes→temperature→savant, the front-door inheritance), The Click (Staunen×Wisdom, Resolution thresholds, `awareness.revise`); `contract::thinking` (36→atoms), `contract::jit` (`StyleRegistry`/`KernelHandle`), `contract::mul` (i4 SIMD eval, `DkPosition`/`TrustTexture`), `contract::nars` (5/5/5 selectors → register), `contract::a2a_blackboard` (`support`/`dissonance` quorum); `FormatBestPractices.md`, `ndarray-vertical-simd-alien-magic.md` (MANDATORY before the I4 SIMD path).

**CORRECTION (2026-05-27, append-only) — §2 atom framing was wrong; superseded by `E-AGICHAT-DIMENSION-CONTRACT` + `.claude/knowledge/atom-basis-inventory.md`:** §2 said "the 36 `contract::thinking` styles demote to atoms (I4-32D, 32 bipolar dims / 64 poles)" — **retracted.** The atom basis is the **LOCKED 33-dim TSV** (3 Pearl + 9 Rung + 5 Σ + 8 Operations + 4 Presence + 4 Meta), NOT derived and NOT the 36 styles. Corrected layering: **atom = one lane** of that TSV (smallest unit, bare-metal, not human-legible); **style = one i4 vector over the atoms** (the molecule — the 36 `ThinkingStyle` ids resolve to such vectors); **persona = composition of styles**. Atoms are **not SIMD** — execution stack is **atoms → `cognitive-shader-driver` → SIMD**; the atom layer holds the carrier + catalogue and dispatches through the driver. Business is an **OGIT-inherited sidecar**, not an atom. The OO style/persona object layer (D-ATOM-2) is the actual metacognition; atoms are the bytes it rides. Code: `contract::atoms::CANONICAL_ATOMS` (locked 33). (This append-only correction follows the workspace's Storno-as-append rule — the wrong §2 is preserved, not edited.)

---

## 2026-05-26 — E-RIGID-RULES-OPEN-DOORS — rigidity belongs to the rules (the HOW, stakes-gated), never to the stance toward a door opening (the WHETHER-to-welcome, baseline-positive); and the welcome is *learned* per topic×texture, not naive

**Status:** FINDING / stance-correction (rebalances the rigidity emphasis of `E-FIBU-GOBD-BY-CONSTRUCTION`; adds the learned-texture policy). Refines `rung-persona-orchestration-v1` §9.

**Click:** We *are* SPO — a "door opening" is a new viable triple/edge, a 2³ projection screening-in, or an `EpiphanyDetector` fire. Two axes that must not be conflated:

1. **rule-rigor** scales with stakes/`Marking` (`Financial`→hard `Soll=Haben`/GoBD/immutability) — the **HOW**.
2. **door-welcome valence** is a baseline-positive stance toward novelty, **stakes-independent** — the **WHETHER**.

Stakes gate the rigor, never the welcome. A bookkeeper is strict on the books *and* glad a new client walked in. So: a door opening is a **rewarded epiphany** (wisdom-marker grows; Affinity/Epiphany/Staunen ghosts brighten), not merely permitted; the MUL gate evaluates rigorously without being a sour bouncer; even a rule-`NO` carries no hostility to the door. Rigidity everywhere → no epiphanies; openness everywhere → can't hold money. **Rigid HOW, happy WHETHER.** (Corrects a drift: the FIBU/GoBD commits over-weighted rigidity.)

**The welcome is learned, not naive.** *If in doubt, the agent fingerprint learns over time, per topic, which `TrustTexture` (Murky/Dissonant/Fuzzy/Clear, `mul/trust.rs`) means **don't touch** and which means **engage** — and vice-versa.* The learned `topic × texture → touch/avoid` policy lives in the wisdom-marker (cold-path / `CrystalCodebook`; content keyed by fingerprint, not bundled — I-VSA-IDENTITIES). Decision ladder: (1) hard rule → follow (rigid); (2) no rule, learned policy exists → follow it; (3) in doubt (no rule / thin history / Murky) → cautious-exploration + Lab, and record the outcome to grow the policy. Young fingerprint = rules + cautious-exploration; mature fingerprint = *taste*. The learning IS the calibration-gap closing.

**Cross-ref:** `rung-persona-orchestration-v1` §9; `rung-mul-grounding-v1` §1 (calibration gap, experience curve); `E-FIBU-GOBD-BY-CONSTRUCTION` (the rigidity rebalanced); `E-CHECKLIST-AS-ESCALATION` (EpiphanyDetector = door opening; ghosts); `mul/trust.rs` (`TrustTexture`).

---

## 2026-05-26 — E-FIBU-GOBD-BY-CONSTRUCTION — German Finanzbuchhaltung is already partly in-code; GoBD legal compliance falls out of the substrate's pure-engine + digested-rules + append-only + Storno-as-append invariants — not a bolt-on

**Status:** FINDING (corrects the "FIBU is net-new" assumption; refines `rung-persona-orchestration-v1` §6b + D-PERSONA-6).

**Click:** "FIBU" (Finanzbuchhaltung) is **not net-new** — the Financial subtree is DACH-first and developed in-code: `contract::grammar::role_keys` has the German SMB catalogue (`KUNDE/SCHULDNER/MAHNUNG/RECHNUNG/DOKUMENT/BANK/FIBU/STEUER`, `FIBU_KEY` @[13072..13584)); `contract::tax.rs` has a **pure `TaxEngine`** (`collect(rule_bundle, period, entries)`, nondeterminism = `Err`), the **`fibu_entry`** RecordBatch (`booking_code, amount, tax_rate`), DACH `Jurisdiction {De, At, Ch}`, and a versioned + 32-byte **digested** `RuleBundle`; `SKR04` is in the foundry roadmap; DATEV/GoBD/BaFin are pre-flagged regulated-tenant triggers (`lf-integration-mapping-v1` LF-80/81). So the Odoo harvest **extends** this (`l10n_de`: SKR03/04→`booking_code`, USt→`RuleBundle`, DATEV→wire), it does not invent it.

**The convergence:** German bookkeeping law **GoBD** (*Unveränderbarkeit / Festschreibung / Nachvollziehbarkeit* — immutable, audit-traceable, deterministic books) **falls out of the substrate by construction**, not as a compliance layer:

| GoBD requirement | substrate invariant that already provides it |
|---|---|
| deterministic books | **pure `TaxEngine`** (nondeterminism = `Err`) |
| audit checksum / rule provenance | **digested `RuleBundle`** (32-byte digest, versioned) |
| Unveränderbarkeit (immutability) | **append-only** postings + boards + CausalEdge64 move-semantics |
| correction = reversal, not edit | **Storno-as-append** = *"committed contradictions preserved, not resolved"* (CLAUDE.md) |

So at `Financial`/FIBU stakes the MUL gate's hard invariants are: **Soll = Haben**, **GoBD immutability** (Storno-append, never edit), **SKR account validity**, **deterministic tax**. Storno is exactly the workspace's append-only-correction pattern (this very entry corrects a prior assumption by *appending*, not editing).

**Cross-ref:** `rung-persona-orchestration-v1` §6b + D-PERSONA-6; `contract::tax.rs`, `contract::grammar::role_keys` (FIBU_KEY); `foundry-roadmap-unified-smb-medcare-v1` (FiBu/SKR04); `lf-integration-mapping-v1` LF-2/LF-80; `E-OGIT-STAKES-LINCHPIN` (marking=Financial→stakes).

---

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

## 2026-05-28 — E-SURREAL-POC-UNANNOTATED-SUPERSEDURE — the `.claude/surreal/` POC docs read as if SurrealDB is the persistence target; the 2026-05-27 ruling that LanceDB is the *leading* store and SurrealDB is one *view* over it is recorded only in `E-RUBICON-RACTOR` + plan `bindspace-singleton-to-mailbox-soa-v1` §2.7 — readers landing in `.claude/surreal/` first do not see it

**Status:** FINDING / navigability (no architectural decision; no code change). The supersedure itself is the FINDING-grade work of #418 + `E-RUBICON-RACTOR`; this entry records that the older POC surfaces are **unannotated** with that pointer, which is a discoverability gap, not a correctness gap.

**The unannotated surfaces (read-only audit):**
- `.claude/surreal/RECONCILIATION_with_canonical_plan.md:20` — row *"SurrealDB-on-Lance persistence | Zone 2 `lance-graph-callcenter` + AriGraph SPO-G quads"* still names SurrealDB as the persistence home.
- `.claude/surreal/cognitive-substrate.md` — substrate framing predates the LanceDB-leading ruling.
- `.claude/surreal/01_deps_substrate.md` … `12_clean_writer_invariants.md` — the 12-task POC reads as if SurrealDB is the persistence target; `surreal_container` (SurrealDB-on-`kv-lance`) framing.

**The current ruling these surfaces lack a pointer to:** *"LanceDB is the leading store / source of truth (append-only, versioned). SurrealDB is one VIEW over it (the Rubicon kanban), never a store."* — `EPIPHANIES.md` `E-RUBICON-RACTOR` + `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md` §2.7 (on PR #418 head, branch `claude/splat3d-cpu-simd-renderer-MAOO0`).

**Lowest-risk fix (NOT done in this handover):** add a non-mutating pointer file (e.g. `.claude/surreal/SUPERSEDURE_NOTE_2026-05-27.md`) that names `E-RUBICON-RACTOR` + plan §2.7 as the current ruling. The append-only governance leaves the existing lines intact; the pointer file is the discoverability surface.

**Practical consequence (do not let this drift back):** `surreal_container` (BLOCKED A/B/C/D — fork dep + Lance 6 pin) is **optional**, not on the critical path for D-MBX-6. D-MBX-6's hot/cold transparent view uses the LanceDB cold tier directly; the SurrealDB kanban is a *second* view over the same LanceDB rows.

**Cross-ref:** `E-RUBICON-RACTOR` (current ruling), `E-MAILBOX-IS-BINDSPACE` (§2.7 of the plan it gates), `E-BATON-1` (the LE-contract anchor), `.claude/plans/causaledge64-mailbox-rename-soa-v1` (driver plan that subsumes the surreal POC), `.claude/handovers/2026-05-28-1200-pr-418-419-surreal-mailbox-baton-plan-map.md` §5 (the audit source for this entry).
