# Entropy Milestones — the systematic N→1 collapse ledger

> An entropy milestone is a point where **N parallel representations of one
> concept collapse to 1 canonical representation**, with (a) the N named,
> (b) the canonical survivor named, (c) a **mechanical gate** that proves the
> collapse is complete (a grep, a range count, a parity test — never a
> claim). Aliases during transition don't count against the collapse; they
> are the mint-forward tax and retire on their own proof.
>
> Sources: the 2026-07-02 mapping fleet's entropy_candidates + the operator
> ruling arc. Status: SHIPPED / IN-FLIGHT / QUEUED / RULING-NEEDED.

## Status: FINDING (ledger; append rows, update Status only)

---

## Shipped / in-flight collapses

| # | N representations | → Canonical survivor | Mechanical gate | Status |
|---|---|---|---|---|
| M1 | 2 classid split orders (canon-low legacy vs canon-high) | `CLASSID_ORDER = CanonHigh` + `_LEGACY` read aliases | corpus scan: zero old-form rows → alias retirement; adoption 100% → P4 (ONE two-metric range-count tool, W6a) | SHIPPED (#628 fleet); gate tool QUEUED |
| M2 | 2 concepts sharing the `ResonanceDto` name (mechanical Ψ field vs perspectival Three-Mountains) | Ψ → `PerturbationDto` (D-PERT-1); perspectival keeps the name | rename lands w/ deprecated alias; grep `dto::ResonanceDto` = alias-only; blast radius confirmed 7 files + 1 comment | QUEUED (small, mechanical) |
| M3 | implicit/undeclared write ownership | ONE stamp: `SoaEnvelope::mailbox_owner()` + cast pairing | `/v3-audit` check 4 = zero unstamped online writes; warden OWNED verdicts | stamp SHIPPED; enforcement gated on W1 batch writer |
| M4 | singleton `BindSpace` sink + per-mailbox `MailboxSoA` coexisting | `MailboxSoA<N>` per mailbox | parity test (mailbox_soa.rs:1145) green + engine_bridge cutover → **W7 deletes bindspace.rs**; grep `Arc<BindSpace>` = zero | IN-FLIGHT (successor shipped; cutover pending W1) |
| M5 | cross-mailbox carriers (Vsa16k-as-carrier, Baton, CollapseGateEmission, emit()) | zero-copy envelope; Lance columnar I/O = the only byte writer | resurrection grep (`/v3-audit` check 2) = tombstone comments only | SHIPPED (PR #477 tombstone) |
| M6 | 3+ tail shapes read ad-hoc | classid-keyed `ReadMode` registry {tail_variant, value_schema, edge_codec} | every reader routes `classid_read_mode()`; DEFAULT documented TEMPORARY | SHIPPED (registry); DEFAULT retirement open |

## Queued collapses (fleet-discovered, gates defined)

| # | N representations | → Canonical survivor | Mechanical gate | Status |
|---|---|---|---|---|
| M7 | 2 column-geometry systems (`SoaEnvelope` trait [zero production impls] vs `VALUE_TENANTS` table + `MailboxSoaView/Owner`) sharing ColumnDescriptor types by convention | ONE: production types implement SoaEnvelope, or SoaEnvelope is re-scoped as the spec/descriptor surface (ruling) | grep `impl SoaEnvelope for` ≥1 production type, or the trait doc names its non-trait role; W1 wiring decides | RULING-NEEDED (feeds W1) |
| M8 | 4 near-duplicate thinking engines (u8/BF16/i8/f32 — same 7-method API) | one generic/enum-dispatched engine (BuiltEngine already half-unifies) | the 4 structs become thin type aliases/params; parity suite green across dtypes | QUEUED |
| M9 | 5+ `ThinkingStyle` copies (contract 36 canonical; thinking-engine 12 is a NEW uninventoried copy; +3 known ledger entries) | contract `thinking.rs` 36-style taxonomy | grep non-contract ThinkingStyle defs = re-exports only; duplication ledger row closed | QUEUED (blocks StepMask catalogue work) |
| M10 | 2 compiled-dispatch stacks (jit.rs n8n-era StyleRegistry [orphaned] vs ExecTarget::Elixir recipe_kernels [exercised]) | the W3 template stack (elixir-template triple + StepMask) | jit.rs either implements against the template stack or retires; ExecTarget::Jit path documented | QUEUED (W3) |
| M11 | 2 kanban-phase representations (SymbiontBoard bare `phase` field vs `KanbanTenant` bytes) | tenant-shaped everywhere (the W2a board-as-tenant type + lane) | POC reads/writes phase through the tenant; grep bare-field phase = zero outside tests | QUEUED (W2a) |
| M12 | 2 budget concepts (elevation `PatienceBudget` per-strategy vs the −550_000 µs Libet anchor per-cycle) | one budget allocator (elevation extended as the 550 ms scheduler, W2d) | elevation reads/writes the Libet anchor; doc cross-ref both ways | QUEUED (W2d) |
| M13 | 2 "OGAR action" concepts (elixir_template::OgarAction enum vs contract `action::ActionDef`+CapabilityExecutor RBAC gate) | keep BOTH (different jobs) — collapse the NAME ambiguity via explicit disambiguation in every doc/brief | guardrails §2 row (done) + compiled-templates.md disambiguation; grep unqualified "OGAR action" in briefs = zero | IN-FLIGHT (doc-side done this PR) |
| M14 | 3 `BindSpace` concepts (shader-driver singleton SoA; ladybug 8:8 dispatch table; graph/spo/merkle store) + stale "BindSpace" prose | scoped names; shader singleton RETIRES (M4); prose sweep to MailboxSoA vocabulary | grep bare `BindSpace` in doc comments describing per-mailbox mechanisms = zero | QUEUED (doc sweep + W7) |
| M15 | 2 `GateDecision` types (mul::{Flow,Hold,Block} live kanban gate vs collapse_gate::{gate,merge} write-merge) — the documented GATE-1 clash + 3 "CollapseGate" vocabulary users | rename the write-merge one (e.g. `WriteMergeGate`) and/or the dispersion op (`ResonanceDispersionGate`); mul::GateDecision keeps the name | grep `GateDecision` resolves to ONE type per import; cycle_accumulator GATE-1 note closed | RULING-NEEDED (rename choice) |
| M16 | 3 independently-invented fail-closed deferral patterns (RuntimeError::NotImplemented / EquivalenceClass::Failure-note / CompileError::NotImplemented) | one documented convention (not a new abstraction) | a §-note in compiled-templates.md naming the pattern; new deferrals cite it | QUEUED (doc-only) |
| M17 | 2 control-flow vocabularies with a FALSE 1:1 claim (graph-flow NextAction×6 vs template linear-only) | honest mapping: Step↔Task + ogar_name()↔Task::id(); control flow closed by StepMask/ControlSignal (W3a/b) | adapter tests replay a template with WaitForInput/End/GoTo semantics; compiled-templates.md corrected (this PR) | IN-FLIGHT (doc corrected; code W3) |
| M18 | 2 lifecycle vocabularies (planner sigma chain Ω→Δ→Φ→Θ→Λ vs KanbanColumn 6 phases) with no documented mapping | a documented mapping (or an explicit "orthogonal" ruling) | one table in mailbox-kanban-model.md; both modules cross-ref it | RULING-NEEDED |
| M19 | duplicated consumer routes (per multi-anchor AST: N duplicated routes = one canonical concept) | 1 canonical concept (classid) + N ClassView skins | the duplication anchor's vote report per consumer; mints reviewed | QUEUED (per-consumer, W5; detector = E-RUFF-ODOO-MULTI-ANCHOR-AST) |
| M20 | 64-bit awareness cramming (CausalEdge64 3/4-bit mantissa) alongside facet payloads | 96-bit facet payloads, classview-lens-selected (E-V3-CLASSVIEW-FOCUS-LENS) | no NEW CausalEdge64 bit-field semantics (review gate); residual role ruling | IN-FLIGHT (ruling recorded; scoping [H] open) |
| M21 | 3 dep-free hand-copies of the 16-byte NodeGuid LE encoder (q2 cpic, q2 fma/converge.rs, woa-rs erp/canon.rs — each "byte-identical", zero shared code) | one zero-dep `canon-node-bytes` extraction all three import | byte-parity test vs contract NodeGuid; grep local encoders = imports only | QUEUED (W5) |
| M22 | 2 divergent q2 OSINT V3 bakes (crates/osint-bake canon-high 0x0700_0000 vs data/osint-v3 STALE pre-flip 0x1000_0700 dual-GUID scheme) | one canon-high bake against osint_classview.rs's 0x0700/0x0701 reservation | re-bake; grep pre-flip forms in q2 data/ = zero (or dual-alias-read only) | QUEUED (W5; latent until a reader assumes canon-high) |
| M23 | 2 write-path doctrines coexisting (owner-stamped V3 writes vs smb-office-rs `LanceConnector::upsert` — the ONE live online consumer write, no stamp/classid/envelope) | all online consumer writes routed through the batch-writer cast | consumer-map §2 table shows zero ORPHAN-WRITE rows; warden green fleet-wide | QUEUED (W5 first live migration; medcare-soa writer BORN stamped as the prevention half) |

## The meta-rule (why this ledger exists)

Every milestone above was found EITHER by an operator ruling OR by the
duplication/parallel-representation smell — never by feature pressure. The
V3 migration is, structurally, this ledger: the substrate is done when the
QUEUED column is empty and every gate is green. New candidate rows: append
below with the same four columns; a row without a mechanical gate is not a
milestone, it's a wish.

Cross-ref: COMPONENT-MAP.md (per-subsystem evidence), INTEGRATION-PLAN.md
(waves that execute the rows), board E-* entries (canonical ruling texts),
docs/TYPE_DUPLICATION_MAP.md (the pre-V3 duplication ledger M9 extends).
