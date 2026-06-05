# Handover — BindSpace dissolution architectural delta

> **From:** bardioc runtime-session (2026-06-05)
> **To:** lance-graph session (fresh, when spun up)
> **Canonical doc:** `bardioc/BINDSPACE_DISSOLUTION_HANDOVER.md` (+ bardioc PR #18) — read that for the full ~440-line delta. This note is the pointer + the headlines per the handovers protocol.

## What I did

- Walked the architectural arc for BindSpace dissolution in a runtime-session
  conversation (2026-06-05).
- Captured the durable delta as `BINDSPACE_DISSOLUTION_HANDOVER.md` in
  bardioc (PR #18) + an append-log entry in bardioc's
  `CROSS_SESSION_COORDINATION.md`.
- Pushed this thin handover note here per the `.claude/handovers/` protocol
  so a fresh lance-graph session lands on it via the standard bootload.
- Did **NOT** execute the migration plan (this session's domain;
  out of bardioc's MCP scope + needs lance-graph board-hygiene grounding).

## FINDING

- **Vsa16kF32 has three real loads, two with replacements + one research-only:**
  - Markov context-chain bundling → **standing wave** (`lance-graph-planner::temporal`, shipped PR #468).
  - BBB inner/outer ontology helper → **Rubicon** (`bardioc/rubicon/`, shipped bardioc PR #17).
  - 5^5^5 holograph experiment → `sentence_crystal` + `holograph` crate, **research artifact**, NOT migration scope.
- **The 16k width is a storage constant**, not a cognition constant
  (literally one Lance "book" register at 64 KB = 16384 × f32). Cognition
  has no opinion on the number; storage does. The cycle plane dies (E-BATON-1);
  the 16k width survives as the book-aligned storage I/O unit.
- **Post-P64, the columns are already L1-L4 cascade-shaped** (per plan §10.5:
  64 / 256 / 4096 / 16384 = AVX-512 register / AMX tile row / 4 KiB page /
  L1d cache). The LE contract isn't *adding* a lossless invariant — it's
  naming a property the columns already have.
- **MailboxSoA D-MBX-A1 inherits 4 of 5 BindSpace columns as the same
  LE-contract types** (`edges` / `qualia` / `meta` / `entity_type` —
  same `CausalEdge64` / `QualiaI4_16D` / `MetaWord` / `u16`). G5 + G6
  in the engine_bridge function matrix are **1-line retargets through
  typed accessors**, not rewrites. **The migration is value-preserving
  (no conversion), NOT layout-guaranteed byte-identical** —
  `CausalEdge64` is `pub struct CausalEdge64(pub u64)` *without*
  `#[repr(transparent)]` in `crates/causal-edge/src/edge.rs:117`, so a
  slice-reinterpretation pattern (`&[u64] as &[CausalEdge64]` via
  `transmute`) is **NOT** supported. Use typed accessors throughout
  (`mb.set_edge(row, e)`, `mb.set_qualia(row, q)`, etc.); the function
  matrix in the canonical doc §5 already does this. Adding
  `#[repr(transparent)]` to `CausalEdge64` is a separate concern in
  `crates/causal-edge/` and would unlock zero-copy slice patterns if
  ever wanted later.
- **The 2026-05-27 plan is still authoritative.** Its §3 column map and §6
  gated steps are correct; this delta narrows scope and adds context.

## CONJECTURE

- Adding the `content/topic/angle` Hamming planes to `MailboxSoA` (still gated
  on D-MBX-A2) unlocks G1 ingress + G4 cycle-half — that's the gating gap
  before S1-S4 can ship.
- **Radix-trie addressing** (PackedDn generalised to variable-length paths)
  composes with Lance's append-only / standing-wave subtree-scan-at-`v_ref`.
  OGAR identities (`ogit-erp/sale.order/42`) are already prefix-rich; Rubicon's
  `LanceMembrane::commit_event` keyed on `inv.object_instance` becomes a trie
  append, the audit log = `traverse_subtree("ogit-erp/sale.order/42/", *)`.
- **Qualia trie codebook with Quintenzirkel rotational invariance** is a
  follow-on D-MBX (compression-only, independent of the singleton
  dissolution). Frozen-set + circle-of-fifths progression structure →
  8 B → 1-2 B per row; task-scoped activation via `QualiaScope` analogous
  to OGAR PR #28's `_effectiveReaders` (tax masks `arc/numbness/*`; invoice
  masks `arc/fear-of-death/*`; therapy = full codebook).
- **Layer correction**: `Staunen` and `Wisdom` are PLASTICITY STAGES, not
  qualia archetypes. They live on `plasticity_counter: [u8; N]` and its
  derivatives, not in `qualia`. The CLAUDE.md "Staunen × Wisdom qualia"
  framing was a misnaming (see canonical doc §8). They CORRELATE with
  qualia-trie-miss on genuinely novel input but neither implies the other.
- **The trie-miss IS the Staunen signal** — side effect of a lookup you
  were doing anyway, with structural context (which prefix you missed at)
  a flat hash collision-set couldn't give. Cheaper than BLAKE3-per-row for
  novelty detection.
- **Zero-copy audit** is the residual discipline after S1-S4 ship. Fires
  at cross-boundary copies, re-encode membranes (canonical bad pattern:
  `engine_bridge.rs:262-267` f32→i4→column), Arrow translations on hot-path
  data, DTO-as-payload-when-could-be-view. Does NOT fire at transient
  bundle compute inside one think-arc (E-BATON-1 carve-out).
- **64 MB working set fits one TiKV region** (default ~96 MB, splits at
  ~144 MB). The deferred cross-server HLC tick gets its natural home —
  Raft log = tick sequence, region leadership = single-writer authority
  for tick advancement, region splits = the only event needing
  reconciliation across leaders. No custom protocol.

## Blockers

- **D-MBX-A2** (add Hamming planes + ratify temporal/expert decisions on
  `MailboxSoA`) is the gating gap. S1-S4 cannot land until A2 ships.
- **No active lance-graph session today** — this handover restores state
  when one starts.

## Open questions

- **Bucket 2** (`lance-graph-cognitive::storage::bind_space::BindSpace`,
  ~20 sites across `cypher_bridge.rs` / `spo/sentence_crystal.rs` /
  `fabric/{executor,zero_copy}.rs` + `learning/{scm,feedback}.rs`) is a
  SEPARATE migration arc per `docs/BINDSPACE_MIGRATION_GAP.md` (the
  ladybug-style Container/CogRecord/PackedDn port). Confirm before
  treating it as singleton scope.
- **Bucket 3** (`sentence_crystal` + `holograph` crate) is the 5^5^5
  research artifact (cube-with-Schnittpunkten geometry mentally /
  sandwich layers in IT reality). Stays as research; no action needed
  unless someone tries to make it a carrier (then I-VSA-IDENTITIES fires).
- **Qualia trie codebook timing** — independent of the singleton
  dissolution; can ship before or after. The 8 B → 1-2 B compression rung
  earns its keep when the codec-stack rotation makes room.
- **StreamDto / ResonanceDto / BusDto DTO collapse** (plan §2.6) is
  mostly mechanical post-S2; worth a doc pass when the time comes.
- **D-MBX-7** (lance-graph container layout = MailboxSoA layout =
  `ndarray::simd_soa.rs` aligned) per plan §11.1 stays as the
  prerequisite for the §2.7 D-MBX-6 (zero-copy SurrealDB view).

## Sources

- **Canonical** (read first): `bardioc/BINDSPACE_DISSOLUTION_HANDOVER.md`
  (bardioc PR #18).
- **The plan** (still authoritative): `.claude/plans/bindspace-singleton-to-mailbox-soa-v1.md`.
- **Cross-session ledger**: `bardioc/CROSS_SESSION_COORDINATION.md` —
  2026-06-05 append-log entry mirrors this handover.
- **Prior art**: `docs/BINDSPACE_MIGRATION_GAP.md` (bucket-2 ladybug arc),
  `CLAUDE.md` "The Click" + 2026-05-26 Baton scoping note (Vsa16kF32
  framing, deprecated as a carrier).
- **Rubicon binding** (the BBB replacement): `bardioc/rubicon/` (Phases
  1-5 shipped, bardioc PR #17 merged); `bardioc/MIGRATION_SPINE.md` for
  the spine view; `bardioc/STANDING_WAVE_ARCHITECTURE.md` for the
  deinterlace engine (= the standing-wave replacement for Vsa16kF32's
  Markov-bundling load).
- **OGAR docs** affected: none — the contract surface does not change.
  `QualiaScope` (if/when it lands) is a class-level annotation analogous
  to `ContextDepends`; no breaking change.

## Punch list for execution (if you pick this up)

1. Ratify **D-MBX-A2** — add `content/topic/angle: [[u64; 256]; N]` Hamming
   planes to `MailboxSoA<N>`; decide on `temporal` (fold into
   `CausalEdge64.temporal` or keep `[u64; N]`) and `expert` (subsume into
   `mailbox_id` or keep optional column).
2. Ship **S1** — G5 + G6 retargets in `engine_bridge.rs`
   (`write/read_qualia_*` + `persist_cycle` minus cycle plane). 1-line
   edits behind a feature gate; both `&BindSpace` and `&MailboxSoA`
   overloads alive in parallel.
3. Ship **S2** — G4 dissolve. Replace `dispatch_busdto`/`unbind_busdto`/
   `busdto_to_binary16k` with a `BusDto::view_over(&MailboxSoA, row)` +
   inverse. `engine_bridge.rs` shrinks ~600 → ~150 LOC.
4. Ship **S3** — G1 reshape. `ingest_codebook_indices` becomes a thin
   sensor adapter per plan §2.6 StreamDto fate.
5. Ship **S4** — delete `Arc<BindSpace>` from `ShaderDriver`; drop the two
   `BindSpace::zeros(4096)` allocations in `bin/serve.rs:29` +
   `bin/grpc.rs:29`. Singleton dissolved; plan §5 reached.
6. Then: zero-copy audit pass per the canonical doc §9; D-MBX-A6 (planner
   DTO overhaul) per plan §11.6.4; optional qualia trie codebook follow-on.
