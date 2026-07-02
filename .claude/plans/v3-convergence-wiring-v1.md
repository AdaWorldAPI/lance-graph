# V3 Convergence Wiring — Plan v1 (wire, don't invent)

> **Status:** ACTIVE (2026-07-01). Operator: "I'm all in for your ideas — document
> and PR so the other sessions benefit and converge." Model split in effect:
> Sonnet 5 grindwork / Fable 5 decisions+nuance.
>
> **The organizing finding** (`EPIPHANIES.md`
> `E-V3-TENANTS-ALREADY-EXIST-WIRE-DONT-INVENT`): every layer of the V3
> substrate already contains its own solution — the gaps are unwired seams,
> not missing machinery. This plan is the seam list, ordered by
> value-per-line, each item either a probe or a wiring of EXISTING types.
> §0 anti-invention holds throughout: no new `ValueSchema` variants, no new
> carriers, no parallel registries.

## D-ids

| D | Layer | Deliverable | Status |
|---|---|---|---|
| D1a | reasoning ladder | `RungLevel::{from_u8, elevate, de_elevate, pearl_level, causal_mask_bits}` + `RungElevator` (zero-dep, contract) — "elevates on sustained BLOCK" as a pure policy over `GateDecision` + the P2/P3-certified mask algebra | **Shipped this session** |
| D1b | reasoning ladder | Driver wiring: thread a `RungElevator` through the cycle loop (replace `driver.rs` `ctx.rung = 1` proxy); dedup `wire.rs`/`grpc.rs` u8→rung matches through `RungLevel::from_u8` | In progress (Sonnet) |
| D2 | AriGraph wave | P6 probe: `markov_soa::best_guess_match` driven by a real 256×256 palette-table distance — the wave uses the SAME certified metric as the particle chain (P1–P3) | In progress (Sonnet) |
| D3 | render / q2 | P7 probe (q2-side, spec in §3): ClassView bitmask → askama render; rendered fields == masked tenants | Queued (q2 push-gated; spec ready) |
| D4 | registry | One-row registry: codebook row seeds `{tail, value_schema, edge_codec, bitmask, template}`; read-mode parity fuse next to COUNT_FUSE | Planned (§4) |
| D5 | tiles / q2 | Nibble-hierarchy falsifier: FNV `cascade3` bytes have no nibble ancestry → HHTL routing on bake mints is tier-granular only | Recorded (`ISS-Q2-CASCADE3-...`, §5) |
| D6 | process | Negative-existence claims require an exhaustive-search declaration (worker rule) | **Shipped this session** (knowledge doc) |
| D7 | orchestration | rig / rs-graph-llm FailureTicket loop — the only fully unwired angle | Deferred frontier (§7) |

Carve certification (both `ValueSchema` presets matrix-covered) and the
`markov_soa` verification shipped immediately before this plan —
see EPIPHANIES `E-V3-TENANTS-ALREADY-EXIST-WIRE-DONT-INVENT`.

## §1 The two-SoA-worlds doctrine (decision, no code this round)

Canonical `NodeRow` (AoS, 512 B) and `MailboxSoA<N>` (true columnar) stay TWO
worlds; **Lance's columnar I/O is the reconciler** (the tombstone write IS the
AoS→SoA shred). Runtime bridging happens through `MailboxSoaView`'s per-row
deferred accessors (`energy_at`, `edge_block_at`, `hhtl_path_at`) — both worlds
satisfy those honestly; the `&[T]` column borrows are the columnar OWNER's
privilege. **Rule for future consumers:** write against per-row access unless
you provably own columns. Do NOT build an in-RAM AoS→SoA mirror — that would be
the serialization-in-the-hot-path anti-pattern (ADR-022 in spirit).

## §2 Rung ladder = dispatch policy over certified mask algebra (D1)

No new math. `RungLevel` 0–9 already names the Pearl boundary
(`Counterfactual = 6`). The mapping shipped in `cognitive_shader.rs`:

- rung 0–2 → Pearl L1 (Association) → mask `O = 0b001` (**convention**, pending probe)
- rung 3–5 → Pearl L2 (Intervention) → mask `PO = 0b011` (**P3-certified**)
- rung 6–9 → Pearl L3 (Counterfactual) → mask `SPO = 0b111` (**P3-certified**)

`RungElevator` (threshold 2, hand-tuned per `I-NOISE-FLOOR-JIRAK` disclosure):
sustained BLOCK elevates, sustained FLOW relaxes toward the dispatched base,
HOLD resets streaks without creep. The driver threads it per cycle; the
provenance rung is the elevator's live level, not the `= 1` proxy.

## §3 P7 render probe spec (D3 — q2-side, ready to execute when q2 opens)

In q2 (askama lives there; lance-graph has none by design — render is the
consumer's side of the membrane):

1. Take one OSINT-V3 `NodeRow` (mint via
   `mint_for(classid_read_mode(CLASSID_OSINT_V3).tail_variant, …)`).
2. Resolve `ClassView`/`ReadMode` → `ValueSchema::Cognitive.field_mask()`.
3. Render through a minimal askama template that iterates ONLY mask-present
   tenants (the bitmask IS the focus of attention).
4. Assert: the rendered field set == exactly the mask's tenant set — no
   phantom fields, no dropped fields; a `Compressed`-class row renders a
   DIFFERENT field set through the same template machinery.

Pass ⇒ `E-RENDER-IS-CLASS-BITMASK-ASKAMA-NOT-SEMIRING` gets its anchor.
Fail ⇒ the doctrine needs revision BEFORE more render code accretes.

## §4 One-row registry (D4 — the structural one)

Today three surfaces must agree and nothing forces it: `ogar_codebook`
(concept ids), `BUILTIN_READ_MODES` (8 hardcoded entries), `ClassView`.
Target shape (NO new types — a wiring):

1. **Phase A (lance-graph only):** a `classid → ReadMode` *derivation default*
   from the codebook row's domain (e.g. every `0x07XX` defaults to the OSINT
   read-mode family) with the explicit `BUILTIN_READ_MODES` entries as
   overrides. Fuse test: every classid in `BUILTIN_READ_MODES` has a codebook
   domain route (`classid_concept_domain != Unassigned` after masking the
   gen-marker) — the read-mode parity fuse, sibling of COUNT_FUSE.
2. **Phase B (cross-repo, with OGAR):** when OGAR emits vocab into the compiled
   binary, it emits read-mode axes with it; `BUILTIN_READ_MODES` becomes the
   bootstrap fallback. Gated on the operator (OGAR is halt-adjacent this
   session) and on the osint-codebook decision.

Litmus for every step: *is this a new layer?* → reject; *is this the codebook
row resolving to more of what already exists?* → proceed.

## §5 Nibble-hierarchy falsifier (D5 — recorded, q2-side)

q2 `cpic::cascade3` derives tier bytes from FNV-1a over cumulative prefixes:
sibling-shared BYTES per tier (sound), but hash bytes carry no NIBBLE ancestry
— the OGAR canon's 4⁴-hierarchical-codebook condition (`is_ancestor_of` =
centroid containment) does NOT hold below whole-byte granularity on these
mints. Falsifier spec: take two DNs sharing a 3-deep prefix, show
`common_prefix_depth` at nibble granularity is ~random beyond the shared-byte
boundary. Consequence if confirmed: HHTL routing over bake mints must clamp to
tier granularity, OR the cascade generator moves to a hierarchical codebook.
Recorded in `ISS-Q2-CASCADE3-NIBBLE-ANCESTRY` (board).

## §6 What stays deferred and why

- **D7 rig/rs-graph-llm loop:** first place determinism ends; probe-first
  treatment when the operator opens it. Everything below is now integer-green
  (contract 763+, core 925, planner 204, arigraph 124).
- **osint `0x0700` reconciliation:** operator decision; the two-id-spaces
  reading (classid space vs concept-vocabulary space, aliasing in the lo u16)
  is on `ISS-OSINT-SYSTEM-ROOT-SLOT-VIOLATION` as Option C.
- **classid human-readable reorder (`0x07:01::1000`):** DEFERRED-by-design,
  hands off until post-V3, implemented as the one flippable split-order.

## §7 Session-A2A notes for the next session

- Build works in-sandbox now: crates.io is proxy-allow-listed, root
  `[patch.crates-io] ndarray` is the local sibling path, `protoc` via apt.
  `cargo test -p lance-graph` is ~2 min cold.
- Model split: Sonnet 5 = grindwork (bounded input, known shape); Fable 5 =
  accumulation/decisions/nuanced plans. Same test as the old Opus policy.
- Every negative-existence claim ("X does not exist / is never called")
  requires an exhaustive-search declaration (D6; born from a truncated-grep
  false claim this session, corrected on the board).
