# temporal-markov-and-style-classes-v1 — the ratified 2026-07-10 cognition arc

> **Status:** ACTIVE (operator-ratified 2026-07-10: "other than that the
> proposed design is great" — caveats accepted as gates).
> **Authority:** board entries `E-MARKOV-TEMPORAL-STREAM-1`,
> `E-THINKING-STYLES-ARE-CLASSES-1`, `E-ORCHESTRATION-ORGANS-1`,
> `E-ACK-IS-THE-KANBAN-TRIGGER-1` (canonical text there; this file is the
> execution plan). Probe-first: NO mechanism is ripped out before its probe.

## Track A — Markov as temporal stream; VSA demoted to its niche

Ruling: the singleton-BindSpace VSA substrate is NOT the default. Markov (as
in DeepNSM) = a `temporal.rs` sorted stream over grammar-resolver
ambiguities, processed live/granularly in thinking — the version-range read
(`QueryReference::at(v, rung)` + deinterlace) generalizes the ±5 VSA braid
(any window, per-reader rung, no N≤32 SNR ceiling, auditable). Episodic
basins = `part_of:is_a` rails (le-contract L1–L3) per E-BASIN-IS-A-NODE;
episodic axis = Lance versions (OGAR D-DELTA mapping made primary). The
L4 `6× palette256:palette256` tenant carries the Morton 2bit×2bit 4×4
inverse-pyramid perturbation cascade (instantiates the OGAR
perturbation-encoding pin: magnitude-only storage, palette envelope,
4⁴ hierarchical codebooks; distance = 256×256 LUT). Anchor numbers:
4096 COCA vocab = CAM codebook (4096² u8 LUT) = 64×64 gridlake sweet spot.
VSA keeps the I-VSA-IDENTITIES four-test niche (≤32 lossless role
superposition within one compartment); the bundle ALGEBRA is untouched —
the Markov TRAJECTORY moves to the stream, where the semigroup property
holds exactly (this strengthening recorded as the [FORMAL-SCAFFOLD]
consult; Click supersession #3 in CLAUDE.md).

| D-id | Deliverable | Gate |
|---|---|---|
| D-MTS-1 | Markov-as-stream parity probe: grammar disambiguation on the DeepNSM corpus — temporal version-range resolution vs the VSA ±5 braid (accuracy + latency, same fixtures) | MUST run green before any VSA-path removal; truth-architect reviews |
| D-MTS-2 | L4 palette² shader fidelity: certification battery (Pearson/Spearman/Cronbach) vs the 0.96–0.998 anchors; representation engineered FIRST (E-V3-CODEC-FIDELITY lesson — Base17-on-raw-Jina \|ρ\|=0.32) | certification-officer battery green |
| D-MTS-3 | Hierarchical-4⁴ vs flat-256 codebook fidelity (OGAR F11-adjacent named test) — the 2bit×2bit cascade's prefix rigor stands or falls on it | ρ vs the 0.9973/0.965 anchors |
| D-MTS-4 | M4 target sharpened: singleton BindSpace cutover targets MailboxSoA + temporal stream + palette tenants (not "MailboxSoA carrying the same VSA planes") | rides M4's existing parity gate |
| D-MTS-5 | Pythagorean-comma vertical-quorum probe (E-COMMA-QUORUM-1): comma-offset vs aligned 4×4 pyramid — inter-level correlation/rank preservation; aligned must collapse, comma must hold L independent witnesses; quantized as a coprime integer walk (D-QUANTGATE, CurveRuler precedent); significance via Jirak | gates the no-materialization claim (64×64…256k×256k never dense) AND the E-COMMA-REPLAY-1 sub-gates: (i) replay determinism — any level regenerated from (GUID, envelope) twice is bit-identical, write-time-independent; (ii) a never-before-computed level passes quorum-independence on first projection; scale selector = wide-classview-style selection (existing-machinery search before any new mask type) |

**Addendum (operator nudge, 2026-07-10):** the per-level cognitive-mantissa
carrier in the shader is `thinking_engine::layered::CascadeChannels8(u64)`
(the renamed thinking-engine twin; collapses into `causal_edge::CausalEdge64`'s
signed mantissa slot) — D-MTS-2's design names it explicitly. Vertical
level offsets follow the comma progression (generated from the address,
never stored); per-level content = palette-quantized magnitude envelope only.

## Track B — Thinking styles as classes (domain:appid:classview)

Ruling: styles move entirely under the classid umbrella. A style class
resolves (never copies — classes only PROJECT the existing SoA view):
`WideFieldMask` (attention projection over the SoA row: SPO / Pearl 2³
rung set / CausalEdge / qualia), `StepMask` (plan projection over the
compiled template), the rung-ladder set, the `ActionDef`+`KausalSpec`
best-practice DO arm (executed by the unified ActionHandler — the
symbiont-as-actionhandler is replaced by this), and the i4-32D modulation
vector (kept; masks discretize selection, not modulation). Per-cycle casts
reference the style classid → attention is replayable from the address
(zero bytes stored). Internal thinking = cognition-domain concepts ×
`AppPrefix::Core (0x0000)` custom half — NOT classid `0x0000_0000` (the
zero-fallback ladder owns it); `0xFFFF` only via a batched mint that
updates `classify_form` in the same batch. Dispatch stays MetaWord bits
(AGI-as-glove) — a `StyleClass` trait anywhere is the drift signal. The
hot-plug registry seat is pre-reserved ("thinking-styles is one entry
away", E-HOTPLUG-GENERIC-1).

| D-id | Deliverable | Gate |
|---|---|---|
| D-TSC-1 | M9 ThinkingStyle dedup (5+ copies → contract taxonomy) | BLOCKS all other Track-B rows (a 6th copy otherwise) |
| D-TSC-2 | Batched cognition-domain mint in OGAR (+ scanner reconciliation if 0xFFFF reserved) | OGAR allocation batch, never solo; COUNT_FUSE |
| D-TSC-3 | Masks + rung set + KausalSpec as class-record properties (codebook-scoping precedent) | envelope-auditor if any byte lands; else contract-only |
| D-TSC-4 | W6c coexistence re-ruling: catalogue shares the custom half with the PERMANENT 0x1000 marker (D-CCF-4 rescinded) — operator ruling needed | ESCALATE to operator; not assumed |

## Track C — Orchestration organs (already executed 2026-07-10)

rs-graph-llm = consumer workflow shell (BBB-clean, contract-only) + internal
slow path (oracle/HITL); rig = oracle-frequency proposer (ratchet-shrinking);
surrealdb = storage + read glove + `ExecTarget::SurrealQl` lowering — NEVER
orchestration (W2c re-scoped). Interconnect = ONE kanban board/WAL; the
ack-pump (`BatchWriter::ack_and_propose`) makes orchestration self-updating:
cast is fire-and-forget, the Lance ack proposes the next move, the driver
never waits (StreamDto can't-stop-thinking). Ownership: `run_cycle` takes
explicit `on_behalf` (owner ≠ classid, rs-graph-llm `8ef18b9`).

| D-id | Deliverable | State |
|---|---|---|
| D-ORG-1 | `BatchWriter::ack_and_propose` self-pumping loop + probes | SHIPPED this arc (2 tests green) |
| D-ORG-2 | W2c re-scope: symbiont arm = storage/read-glove only | STATUS_BOARD row updated this arc |
