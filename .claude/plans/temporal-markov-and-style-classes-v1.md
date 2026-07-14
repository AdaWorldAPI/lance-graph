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
| D-MTS-5 | Pythagorean-comma vertical-quorum probe (E-COMMA-QUORUM-1): comma-offset vs aligned 4×4 pyramid — inter-level correlation/rank preservation; aligned must collapse, comma must hold L independent witnesses; quantized as a coprime integer walk (D-QUANTGATE, CurveRuler precedent); significance via Jirak | **MEASURED GREEN 2026-07-10** (`perturbation-sim/examples/comma_quorum.rs`): comma N_eff 11.00/12 vs strict 1.00 / unit 2.49 / rational 3.92; both E-COMMA-REPLAY-1 sub-gates PASS (bit-identical any-order replay; fresh level-12 +0.83 witnesses at max\|ρ\|=0.156); 82 KB vs ~69 GB dense — never materialized. Measured boundary condition: N_eff(comma) = min(L, spectral participation of the detail) — broadband residues get the latent granularity, concentrated content saturates early (regime-B ceiling 2.55). Canonical: E-COMMA-QUORUM-MEASURED-1 |

**Addendum (operator nudge, 2026-07-10):** the per-level cognitive-mantissa
carrier in the shader is `thinking_engine::layered::CascadeChannels8(u64)`
(the renamed thinking-engine twin; collapses into `causal_edge::CausalEdge64`'s
signed mantissa slot) — D-MTS-2's design names it explicitly. Vertical
level offsets follow the comma progression (generated from the address,
never stored); per-level content = palette-quantized magnitude envelope only.

**Technique input queued (operator, 2026-07-10, from the stockfish-rs
sibling arc):** `stockfish-rs/.claude/knowledge/stockfish-pext-morton-adjacency.md`
— read while harvesting real Stockfish source. Honest finding: Stockfish
has NO Morton/pyramid/palette/comma code (verified, zero hits); what DOES
transfer is the `Magic` struct's `pext`/`pdep` gather→table-lookup→scatter
pattern (`attacks.h:130-159`) as a candidate AVX-512/BMI2 fast path for
the 2bit×2bit 4×4 tile pack/unpack (same hardware primitive family the
standard Morton bit-interleave trick uses — different mask content), plus
the three-tier CPU-family dispatch lesson (BMI2 PEXT is known-slow on
some AMD families; ship a portable fallback, don't trust ISA presence
alone) as independent confirmation of the `ndarray::simd` multi-tier
dispatch discipline already in force here. CONJECTURE until D-MTS-2
actually builds and measures the tile primitive this way.

> **Addendum (operator, 2026-07-10, second wave — three rulings):**
> (1) **E-STYLE-FAMILY-VS-RUNBOOK-1** — 12 = abstract FAMILIES for
> orchestration; 36 = literal NARS RUNBOOKS; runbooks seed the rung ladder;
> rs-graph-llm chaining consumes runbooks as its replayable unit; runbooks
> later feed the elixir-LIKE notation (compiled templates, M10). D-TSC-1's
> canonical 12-type is therefore `StyleFamily` with `default_runbook()` /
> `ThinkingStyle::family()`.
> (2) **E-THINKING-TENANTS-V3-1** — thinking tenants migrate to V3
> (**D-TTV-1**); old CausalEdge64 kept as perturbation baseline.
> (3) **D-MTS-6** — measure the smaller-CE64 × comma awareness curve vs
> that baseline (the bits-as-projection-axis twin of E-COMMA-REPLAY-1).
> **MEASURED GREEN 2026-07-10** (`comma_awareness.rs`): k*=1 vs aligned
> k*=4; the comma lattice buys ≈ log₂(12) effective bits; D-MTS-6b
> (driver-integrated fixture) gates any real CE64 shrink — see
> E-COMMA-AWARENESS-MEASURED-1.
> Execution spec for D-TSC-1: `.claude/plans/dtsc1-thinkingstyle-dedup-spec-v1.md`.

> **Addendum (operator, 2026-07-10, third wave — D-MTS-1 context-building
> design input):** the **AriGraph context, V3-TENANT-SHAPED, is most
> probably essential for Markov-chain context building while streaming
> text.** The ingest legs feeding it: **`lance-graph-arm-discovery`**
> (the Aerial+ transcode — float-free association-rule discovery via the
> palette256 integer distance oracle, ρ=0.9973, → `{s,p,o,f,c}` ndjson →
> SPO loader) and **DeepNSM** (text→SPO via the 6-state PoS FSM; the
> **64×64 = 4096 COCA vocabulary IS the CAM index codebook**, 4096² u8
> LUT). The representational comparison the D-MTS-1 spec MUST weigh for
> the streamed context items:
>
> - **CAM-PQ `6×8 = 48-bit` codes** — 6 subspaces × 1 byte = 6 bytes =
>   exactly the OGAR GUID path (HEEL+HIP+TWIG): the ADDRESS-side coding.
> - **`6× palette256:palette256` = 12 bytes = ONE V3 tenant** — 6
>   subspaces × an X:Y byte PAIR each (the 256×256 centroid-tile
>   reading, le-contract L4 carve): the VALUE-tenant-side coding —
>   exactly one content-blind facet payload.
>
> The likely resolution (to be probed, not assumed): key addresses,
> tenant carries — the 48-bit path codes route/dedupe the streamed SPO
> items, while the palette² tenant is the projection surface the
> perturbation shader attends per cycle. This joins D-TTV-1 (the
> AriGraph-context lane is a thinking tenant) and instantiates the L4
> carrier named in E-MARKOV-TEMPORAL-STREAM-1.

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

## Addendum (session evidence, 2026-07-12) — the chess-signature arc as the evidence layer under this spine

NOT a ruling change — session evidence that VALIDATES Track A/B with measured
numbers, plus honestly-graded conjectures. Full arc: EPIPHANIES
`E-CHESS-SIGNATURE-ARC-1`; the V3 transfer + rejected rhymes + the one new
method: `E-CHESS-ARC-TO-V3-TRANSFER-1`. Grades explicit ([G]/[H]/[S]);
probe-first honored.

**1. Thinking is a moving wave — the wave IS this plan's temporal.rs Markov
stream [G/H].** Turk & Polson, "Chess Signatures of Play" (arXiv 2606.18544,
June 2026), model a game (a thought) as a multivariate PATH; the Lyons
signature is its reparametrization-invariant featurization, discriminating
info in the LÉVY AREAS (order-interaction). This is external evidence FOR
Track A's ruling (`E-MARKOV-TEMPORAL-STREAM-1`): the Markov trajectory lives
on the temporal.rs sorted STREAM, not the VSA braid — because the
discriminating structure is in the wave's ORDER, which means/aggregates
DESTROY. Measured: every averaged chess style-measure landed ~1.1–1.3×
chance; the order-preserving stream is the correct carrier. The workspace
already owns the featurizer: `sigker::signature_truncated` (Chen–Lyons),
certified by jc pillar #11 (`hambly_lyons.rs`) against the exact tree-like-
equivalence uniqueness theorem the paper's identifiability rests on.

**2. Sender→receiver meta-awareness lives in the PREDICATE (edge), not the
nodes [H, strong].** Chess personality measured **98% INTER (interaction) /
2% INTRA (trait)** — identity is the opponent-response, not a stored vector.
In SPO that is S(self)—**P(interaction)**—O(other): the **CausalEdge /
EdgeColumn (P)** carries the adaptive identity; `FingerprintColumns` (S/O
nodes) carry the ~2% invariant. The mailbox **cast→response IS
sender→receiver**. Consequence for Track B's `WideFieldMask` (which projects
SPO / Pearl 2³ rung set / CausalEdge / qualia over the SoA row): the
**CausalEdge projection is the measured seat of meta-awareness** — weight it
accordingly. This is also why the constant-vector chess probes failed: they
fit an INTRA model to a 98%-INTER process.

**3. Three reasoning surfaces — and which one carries identity [H].** The
`Think` struct resolves against three tissues; the chess arc measured their
division of labor. **Episodic basins** (`part_of:is_a` rails, le-contract
L1–L3) = averaged CONTEXT — the mean, measured NOT to identify (comfort-basin
probe 1.1×). **SPO 2³ rung-decomposed facts** = what is BELIEVED (Pearl-
laddered; the chess awareness ladder L0-egocentric/L1-perspective/L2-gestalt +
the counterfactual clause ARE Pearl rungs 1/2/3, `E-SF-AWARENESS-OPPONENT-ARC-1`).
**The temporal.rs wave** = the moving TRAJECTORY, where the signature — and
identity — lives (delayed-gratification-sac rate 9.70× reputation-correct, in
the TAIL not the mean). So: **basins GROUND, facts ANCHOR, the WAVE
IDENTIFIES.** Validates Track A third-wave addendum (AriGraph context,
V3-tenant-shaped, essential for Markov context-building) with the sharpening
that the CHAINING (the wave), not the BASIN (the mean), is the discriminating
surface.

**4. Macro/wide-angle vs tele = WideFieldMask breadth vs version-range depth
[S — CONJECTURE, PROBE UN-RUN].** The spatial wide-angle (attention over the
current SoA row = `WideFieldMask`) vs the temporal tele (k-ply version-range
deinterlace depth, `QueryReference::at(v, rung)`) is a candidate reasoning-mode
dial, RHYMING with NARS breadth-vs-depth inference. **Graded [S]: the chess
D-SF-FILTER probe (spatial-field read vs k-ply rollout read, switched on the
contact/wedge signal) was queued but NEVER RUN** — this is a conjecture, NOT
evidence, and must not calcify as fact. Probe it before wiring.

**5. The needle discipline for the L4 perturbation shader [G].** When the
discriminating signal is a RARE event, bundling (`CollapseGate
MergeMode::Bundle`, and the L4 palette² shader accumulation) destroys it — a
mean's SNR is bounded by the event rate (a ~1% needle cannot survive
averaging; measured across every aggregate probe). Route the needle to
anomaly-detection (ndarray CLAM/CHAODA); `MergeMode::Bundle` is correct for
the Markov-context BASIN, wrong for the discriminating rare event. This is
I-VSA-IDENTITIES stated as a merge-mode selection rule for the shader.

**6. Epiphany = non-tree Lévy sweep [H, probe-gated].** D-EPIPHANY-SIG-1
(carved in `E-CHESS-ARC-TO-V3-TRANSFER-1`): the Lévy area of the `Think`
belief-trajectory signature distinguishes real belief-update (non-tree, area
swept) from circular rumination (tree-like, collapses) — the operator's
`epiphany ↔ NARS` link. In-tree parts: `sigker` + jc/hambly_lyons + Think/NARS
+ temporal.rs. Probe-first; stays [H] until run. Cross-ref, not repeated here.

**Rejected rhymes (anti-dilution, do NOT build):** ✗ "the perturbation shader
IS the signature transform" (different algebras: VSA bind+braid+bundle vs
iterated integrals — braiding-by-position rhymes with order-encoding but is
[S] until measured); ✗ "personality = the QualiaColumn"; ✗ treating the ~1.3×
aggregate nulls as signal. Full text: `E-CHESS-ARC-TO-V3-TRANSFER-1`.
