# SoA migration diff resolution — 2026-06-13

> **Type:** meta-resolution (audit + supersession map), not a new plan.
> **Audience:** any session touching the BindSpace dissolution, the SoA envelope ABI, the identity architecture, or the cycle-coherent snapshot work after PR #490.
> **Scope:** harmonises the SoA / BindSpace / identity plan family against the shipped reality post-#487/#489/#490 + the operator-pinned canon (OGAR/CLAUDE.md P0). Names every drift, pins the supersession, and lists the open work that survives. Does not declare new deliverables; does not retroactively rewrite plan bodies.
> **Trigger:** the bardioc cross-session driver opened PR #490 ("#489 is canonical"), which retired the Phase-A UUIDv8 wrapper. With the canonical NodeGuid layout now live, multiple plans hold framing that no longer matches code or canon. Resolution doc lands the diff in one place.

---

## 1. Canon anchors (the truth all plans must match)

| Anchor | Where | Says |
|---|---|---|
| Operator GUID canon | `OGAR/CLAUDE.md` P0 | 16-byte key = `classid · HEEL · HIP · TWIG · family · identity` (8·4·4·4·6·6 hex); RFC-WAIVED ("the GUID is NOT RFC-stamped"); 3×4 uniform tiers; tier-of-level = `level >> 2`; codebook scoping = classid prefix; *"wrappers adapt to the canon, never the reverse."* |
| Doc-lock | `lance-graph/CLAUDE.md` (4ea6ac92, #488 line-of-commits) | Same layout; reserve-don't-reclaim discipline for the zero-fallback ladder; canon does not pre-pay RFC ceremony in every key. |
| Code form | `crates/lance-graph-contract/src/canonical_node.rs` (#489 + #490) | Locked layout matches canon group-by-group; `Display` impl renders self-describing 8-4-4-4-12 hex; size-asserts at 16/16/512; 12+4 `EdgeBlock`; 480-byte `value` slab on `NodeRow`. |
| Three-tier model | `docs/architecture/soa-three-tier-model.md` (#477) | **No emission, no inter-mailbox handoff type.** SoA is zero-copy from creation to Lance tombstone. `MailboxSoaOwner` mutates; `MailboxSoaView` reads. Lance's columnar I/O writes LE bytes from the in-place store. |
| Tombstone commit | #487 | `CollapseGateEmission` + `MailboxSoA::emit()` deleted from source; `last_emission_cycle → last_active_cycle`; `consume_firing(row) -> bool` is the in-place successor. |
| Phase-A retirement | #490 | `identity::NodeGuid` (UUIDv8 wrapper) deleted; `IDENTITY_LAYOUT_VERSION` removed; `lance_graph_contract::NodeGuid` now re-exported from `canonical_node`. |

---

## 2. Plan inventory and status

The SoA / BindSpace / identity plan family, post-#490.

| Plan file | Intent (one line) | Status after #490 | Land-PRs |
|---|---|---|---|
| `bindspace-singleton-to-mailbox-soa-v1.md` (2026-05-27) | Dissolve `Arc<BindSpace>` singleton → per-mailbox `MailboxSoA<N>` | **Partially-superseded.** §3 column-by-column map authoritative; D-MBX-A1 columns shipped in #386; §2.6 DTO inventory and §5 sequencing predate the #477 emission tombstone (which made some §5 details moot). | #386 (D-MBX-A1 columns), #477 (rename + tombstone), #487 (emit() removed) |
| `unified-soa-convergence-v1.md` + addendum (2026-05-29) | "Five layered rulings" for SoA end-to-end, never re-encoded | **Authoritative as doctrine, stale on stack pins.** §1 rulings unchanged (anchor for E-SOA-IS-THE-ONLY). §4.2 stack table predates lance→7.0.0 / lancedb→0.30.0 bumps. Addendum (#486) partially addresses; this doc names the residual. | #434 (landed), #486 (addendum), #487/#488/#489/#490 (incremental impl) |
| `identity-architecture-exists-vs-needs-v1.md` (2026-06-09) | Compose `NodeGuid` from existing scalars (UUIDv8) | **§N1 layout fully-superseded by canon.** The proposed `namespace + entity_type + kind + niblepath_prefix + shape_hash + local + RFC version/variant + layout_version` layout did NOT ship. Canon's `classid + HEEL + HIP + TWIG + family + identity` (no ceremony, no shape_hash, no layout_version) won. §N5 entity_type↔NiblePath bijection: **shipped** in #484. §N3 SoaEnvelope impls: **still queued** (only `TestEnvelope` implements). | #480 (Phase-A, retired), #484 (bijection), #489 (canon code form), #490 (canon wired + Phase-A retired) |
| `cognitive-write-roundtrip-substrate-v1.md` (2026-06-11) | Cold-path write as `TripletProjection` + roundtrip_eq gate | **Still-authoritative.** Doctrine intact; blocked on `SoaEnvelope` impls for the canonical row layout. | None yet shipped |
| `cycle-coherent-soa-snapshot-v1.md` | Arc-swap COW snapshot at column granularity | **Still-authoritative.** Trait shape (`SnapshotProvider::Column`, generic `MailboxSoaSnapshot<C>`) ratified by #487 CodeRabbit-Critical fix. No implementor yet. | None yet shipped |
| `singleton-to-snapshot-nudge-v1.md` (PR #478) | Every shared-mutable singleton → per-owner SoA + Arc-swap | **Still-authoritative.** Codebook-vs-runtime-state rule unchanged. D-SNGL-3 (AttentionMatrix `unbundle_from`) still queued. | #478 (plan, no code) |
| `polyglot-container-query-membrane-v1.md` (PR #484) | SurrealQL + DataFusion + Cypher membrane over HHTL | **Research-only / superseded in spirit.** Author self-flagged in #484 body: *"superseded in discussion by the self-describing-key convergence (the class-in-key makes the cold path already a graph; no membrane needed)."* | None |
| `causaledge64-mailbox-rename-soa-v1.md` | `MailboxSoA<N>` shape + cycle-stamp rename | **Mostly shipped.** D-CSV-1/2/5a/7 done; the rename is in #477. | #383/#384/#386/#477 |
| `bindspace-columns-v1.md` | Original column-map prior to dissolution | **Superseded by `bindspace-singleton-to-mailbox-soa-v1.md`.** Kept for history. | (historical) |

---

## 3. Concept-level diff (shipped vs planned)

### 3.1 Retired (planned, but never shipped or deleted post-ship)

| Concept | Plan reference | Disposition |
|---|---|---|
| `CollapseGateEmission` | bindspace-singleton-to-mailbox-soa-v1 §5; causaledge64-mailbox-rename §3 | **Deleted #487.** Replaced by in-place `MailboxSoA::consume_firing(row)`. |
| `MailboxSoA::emit()` | bindspace-singleton-to-mailbox-soa-v1 §3 | **Deleted #487.** Same successor. |
| "Baton" as a `(u16 target, CausalEdge64)` carrier type | bardioc handover §5; CLAUDE.md 2026-05-26 block | **Never materialised.** Three-tier model (#477) ratifies "no inter-mailbox handoff type at all." `MailboxId`/`MergeMode`/`GateDecision` survive as concepts in `collapse_gate.rs`. |
| `wire_cost_bytes() = 13 + 10·baton_count` | CLAUDE.md 2026-05-26 block | **Gone with the carrier (#477).** |
| `Vsa16kF32` as a *carrier* | unified-soa-convergence-v1 §2.2; bardioc handover §1 | **Deprecated as carrier; still allocated as `BindSpace.fingerprints.cycle`.** Plan §0 says "deprecated"; code at `bindspace.rs` still holds the 64KB plane. Dissolution lands at S4. |
| `identity::NodeGuid` (UUIDv8 wrapper) | identity-architecture-exists-vs-needs-v1 §N1 | **Deleted #490.** Pre-production with zero in-tree consumers per #480's own body; canon-incompatible per "wrappers adapt to the canon, never the reverse." |
| `IDENTITY_LAYOUT_VERSION` | identity-architecture-exists-vs-needs-v1 §N1 | **Deleted #490.** Canon does not pre-pay an in-band layout version in every key. |
| `SHAPE_HASH_BITS` / `LOCAL_BITS` consts | identity-architecture-exists-vs-needs-v1 §N1 | **Deleted #490.** Canon real-estate reclaimed: HIP+TWIG occupy the bytes the wrapper had spent on `shape_hash`. |
| `niblepath_prefix` slot in NodeGuid | identity-architecture-exists-vs-needs-v1 §N1 | **Not in canon.** Canon's HEEL is 4 nibbles uniform (not a truncated prefix of a `NiblePath`); HIP+TWIG are 4 more nibbles each (uniform 3×4). |
| RFC 9562 version=8 / variant=0b10 bits | identity-architecture-exists-vs-needs-v1 §N1 | **Not in canon.** Canon: *"RFC 9562 is a WRAPPER format, and wrappers adapt to the canon, never the reverse — any boundary that genuinely requires RFC-valid UUIDs owns that adaptation at its membrane and pays it explicitly."* |

### 3.2 Renamed (plan vs code label drift)

| Plan label | Code label | Where |
|---|---|---|
| `last_emission_cycle` | `last_active_cycle` | mailbox_soa.rs (PR #477 rationale: "there is no emission; the stamp marks in-place consumption") |
| `MailboxSoA::emit()` | `MailboxSoA::consume_firing(row)` | mailbox_soa.rs (PR #487 successor; zero-copy in-place stamp + energy reset) |
| `advance_phase` | `try_advance_phase` (Result-returning) | soa_view.rs (PR #487, returns `Result<KanbanMove, RubiconTransitionError>`) |
| `identity::NodeGuid` | `canonical_node::NodeGuid` (re-exported as `lance_graph_contract::NodeGuid`) | #490 |

### 3.3 Shipped but never planned (genuinely new in code)

| Concept | Code site | Notes |
|---|---|---|
| Zero-fallback ladder on `NodeGuid` | canonical_node.rs:11-18 | classid==0 ⇒ default class (dormant); family==0 ⇒ default basin (dormant); identity alone discriminates in the bootstrap state. "Reserve, don't reclaim" — same discipline that `EdgeBlock` extends to row-layout level. |
| `EdgeBlock` (12 in-family + 4 out-of-family) | canonical_node.rs:120-127 | Canonical, not mandatory: 16 bytes always reserved (zeroed when unused). Opt-out is registry-resolved via `classid → ClassView`, never by changing the row stride. The row-layout analogue of the key-side zero-fallback ladder. |
| `NodeRow { key:16, edges:16, value:480 } = 512 byte` | canonical_node.rs:129-140 | The Lance row the `MailboxSoaOwner` owns and the `MailboxSoaView` reads. Value slab is class-resolved; energy/meta/qualia/entity_type/materialised CausalEdge64/helix residue/fingerprint/class extensions all land here when the class's ClassView declares them. |
| `impl Display for NodeGuid` | canonical_node.rs:135-150 (#490) | Self-describing `{classid:08x}-{heel:04x}-{hip:04x}-{twig:04x}-{family:06x}{identity:06x}` per canon's *"every printed GUID is self-describing at sight."* LE in-memory bytes fold through the accessors so hex print is canon-ordered regardless. |
| Three-tier model formal split | docs/architecture/soa-three-tier-model.md (#477) | Tier 1 = MailboxSoA (hot, zero-copy creation→tombstone); Tier 2 = KanbanColumn / Rubicon (sole secondary, sole-mutator via `try_advance_phase`); Tier 3 = OGIT ontology + OGAR classes (inherited, O(1) via HHTL/`NiblePath` prefix). |
| `SoaEnvelope` trait (envelope-level LE contract) | soa_envelope.rs (#477) | Stable column ordering, fixed row stride, cycle stamp, `ENVELOPE_LAYOUT_VERSION = 1`, `verify_layout()` gate (catches stride mismatch, column overlap, column-past-stride, packet size mismatch, version skew). |

### 3.4 Status discrepancies (plan says X, code says Y)

| Item | Plan says | Code says |
|---|---|---|
| **D-MBX-A1 columns** (edges/qualia/meta/entity_type on MailboxSoA) | "Ship S1" | **Shipped** (PR #386, mailbox_soa.rs:73-94 with named setters/getters) |
| **D-MBX-A2** (Hamming planes + temporal/expert) | Gating gap | **Still queued** — MailboxSoA<N> has no Hamming columns yet (mailbox_soa.rs:51-108) |
| **S2** (G4 dissolve / engine_bridge ~600→~150 LOC) | Queued behind A2+S1 | **Not shipped** — engine_bridge.rs ~34KB unchanged |
| **S4** (delete `Arc<BindSpace>`) | Final step | **Not shipped** — driver.rs:56 still has `pub(crate) bindspace: Arc<BindSpace>`; both `bin/serve.rs:29` and `bin/grpc.rs:29` still call `BindSpace::zeros(4096)` |
| **`SoaEnvelope` impl for MailboxSoA** | Implied substrate | **No real implementor exists.** Only `TestEnvelope` in `soa_envelope.rs` tests (lines 254-310). MailboxSoA does not impl `SoaEnvelope`. |
| **`MailboxSoaOwner` / `MailboxSoaView` for MailboxSoA** | Ratified trait split (three-tier doc) | **No real implementor exists.** Traits defined; MailboxSoA does not implement them. |

---

## 4. BindSpace dissolution sequence status (post-#490)

The bardioc handover named `D-MBX-A2 → S1 → S2 → S3 → S4` sequencing. Status as of 2026-06-13 19:54Z (post-#490):

| Step | Deliverable | Status | Blocker / next move |
|---|---|---|---|
| **D-MBX-A2** | Hamming planes (`content`/`topic`/`angle: [[u64; 256]; N]`) + temporal/expert decisions on `MailboxSoA<N>` | ✗ Queued | Type work; OQ-11.2 default W=16 implies ~6 KB/thought addition |
| **S1** | G5 + G6 retargets (qualia I/O + `persist_cycle` minus cycle plane) | ✗ Queued, partially-trivial | A2 not strictly required for qualia/edges/meta retargets; engine_bridge still routes through BindSpace |
| **S2** | G4 dissolve (BusDto → row-view, `dispatch_busdto`/`unbind_busdto`/`busdto_to_binary16k` deleted; engine_bridge ~600 → ~150 LOC) | ✗ Queued | A2 + S1 — content-plane writes need Hamming columns |
| **S3** | G1 reshape (`ingest_codebook_indices` → thin sensor adapter; StreamDto becomes a true membrane-edge type) | ✗ Queued | S2 |
| **S4** | Delete `Arc<BindSpace>` field on `ShaderDriver` + the two `BindSpace::zeros(4096)` allocations + `bindspace.rs:234` type itself | ✗ Queued | S3 |

**Honest assessment:** the canonical row layout is now live in `canonical_node.rs` (#489/#490), but **MailboxSoA<N> has not yet been migrated to use `NodeRow` row-strided storage**. The plan's S2 ~600→~150 LOC engine_bridge shrink is the next high-leverage move, and it needs A2 first.

---

## 5. LE-contract violations still on the books

Per bardioc handover §9 zero-copy audit, "bytes don't translate at any boundary." Sites that still violate:

| Violation | Location | Status |
|---|---|---|
| f32 → `from_f32_17d` → i4 re-encode at the qualia sensor membrane | `engine_bridge.rs` (handover named line 262-267; verify exact range post-rebases) | ✗ Still present. Canonical bad pattern: the f32 should die at the sensor membrane; everything past it should already be i4 bytes. |
| `Vsa16kF32` as cross-boundary persistent state | `bindspace.rs` `FingerprintColumns.cycle` (the 64 KB plane) | ✗ Still allocated. Plan says "ephemeral local compute only" (E-BATON-1 carve-out); code persists it at singleton level. Dissolves at S4. |
| DTO-as-owned-`Vec<T>` where the DTO could be a `&[T]` view over a SoA column | engine_bridge.rs BusDto/StreamDto/ResonanceDto payloads | ✗ Still present. G4 dissolve (S2) addresses this. |
| Two same-name `CausalEdge64` types | causal-edge / thinking-engine | ✗ Disambiguation PR called out in #487 body as "not in scope — needs its own PR." |
| `OntologyRegistry` linear-scan vs the doc claim of "O(1) index" | `ontology` crate | ✗ Called out in #477 docs/probes/particle-soa-envelope-audit.md. Replace with HHTL radix-trie lookup; evaluate `entity_type` redundancy. |

**Not violated** (audit checked):
- Cognitive-shader-driver does not do Arrow translation on the hot path. Arrow lives only in the ontology cold path (legit per handover §9's "calcified cold knowledge" carve-out).
- The new `NodeGuid::Display` impl (#490) is LE-clean: bytes are folded through field accessors that read LE, so hex print is canon-ordered regardless of in-memory byte order.

---

## 6. The Staunen/Wisdom-as-DIKW-climb-position correction

**Source:** bardioc handover §8 (2026-06-05) + operator image relays (2026-06-13: the 8-rung extended pyramid and the KM Cognitive Pyramid) + the [DIKW pyramid](https://en.wikipedia.org/wiki/DIKW_pyramid) canonical lineage.

**Canonical DIKW (the lineage to anchor against):** four vertically-stacked layers — `Data → Information → Knowledge → Wisdom` — bridged by three integration arrows:

| Arrow | Lifts | Operation |
|---|---|---|
| **Processing** | Data → Information | organize, label, transform |
| **Cognition** | Information → Knowledge | interpret, contextualise, find relevance |
| **Judgment** | Knowledge → Wisdom | visualise, analyse, apply, anticipate |

The KM Cognitive Pyramid variant adds rung-specific markers (PIRs at Knowledge, CCIRs at Wisdom, EEI/EEFI at Information) and two orthogonal axes that run *along* the climb: **Reactionary↑Anticipatory** (decision behaviour — Data-level reactions are reactionary, Wisdom-level decisions are anticipatory) and **high↓low decision risk** (raw data drives risky decisions, wisdom drives safer ones). Extensions above Wisdom — `Self-Actualization → Universal Knowledge → Transcendence` (8-rung extended) or `Shared Understanding` (KM variant) — are post-DIKW; classical DIKW tops out at Wisdom.

**Where `Staunen` and `Wisdom` actually sit:**

- **Wisdom IS a DIKW rung** — the canonical apex of the four-layer pyramid. Not a marker on a horizontal axis. Anticipatory; Know Why; low decision risk; the result of Judgment applied to Knowledge.
- **Staunen is not a DIKW rung** — it's the *phenomenological encounter* at or below the Data layer, where stimulus overflows current frameworks and Processing hasn't yet integrated it. Reactionary; Know What (or not yet); high decision risk; pre-integration wonder.

So `Staunen` and `Wisdom` are **vertical endpoints of the DIKW climb axis** — the climb itself IS the discipline↔entropy gradient. Up the climb: discipline accumulates, entropy decreases. Down the climb: stimulus arrives, entropy is high, discipline is yet-to-accumulate. They are *not* two qualia archetypes (which is what `lance-graph/CLAUDE.md` line ~120 currently treats them as) and not horizontal opposites of one axis (which is how my prior framing of this section read).

**Substrate mapping** (the canon-aligned implementation):

- The DIKW climb counter IS `plasticity_counter: [u8; N]` on `MailboxSoA<N>` (saturating u8 per W6 §4.4, incremented on every accepted edge). High saturation = rehearsed climb = Wisdom-positioned firings. Low saturation + recent `last_active_cycle` recency = encounter without rehearsal = Staunen-positioned firings.
- Staunen and Wisdom are **derivatives** over `plasticity_counter` + `last_active_cycle` + classid-prefix-resolved codebook hit-rate, never their own column. The substrate computes them on demand from the lifecycle columns it already owns.
- They are **orthogonal** to the qualia codebook (which names the *archetype* the firing snaps to — what the affective state IS, independent of where the firing sits on the DIKW climb). A Wisdom-positioned firing can have any qualia archetype; a Staunen-positioned firing can too. The two layers correlate only in the special case of genuinely-novel input (Staunen + codebook trie-miss tend to co-occur), but neither implies the other.

**Bipolar / two-algebra connection:** the canon's **TWO-ALGEBRA RULE** — *"sign = XOR (`vsa_bind`); magnitude = `vsa_bundle`, NEVER raw-XOR-on-magnitudes"* (OGAR/CLAUDE.md P0) — applies to the climb axis. The DIKW climb direction (up = Judgment composing Knowledge into Wisdom; down = Processing decomposing Information back to Data) is the **signed** side (one bit per rung-transition, XOR-composed across rungs to give the cumulative direction of the climb step). The *intensity* of the firing at any given rung is the magnitude side (`vsa_bundle`, Markov-respecting). The Walsh-Hadamard cascade pyramid on the address tree IS the substrate that carries this — top-gaussian preserved rung-to-rung per Parseval, anti-moiré via the helix `CurveRuler` stride-4-over-17 walk.

**Cascade-tier connection:** the canon's three tiers (HEEL · HIP · TWIG, each u16 = 4 nibbles = 4 climb steps) match the three DIKW transitions (Processing / Cognition / Judgment). The classid prefix (8 hex = 32 bits) carries the codebook scope; the HEEL/HIP/TWIG tiers carry the rung-position; family/identity carry the basin-local content. This is a clean substrate decomposition of the DIKW climb — *not coincidentally,* the canon's 3×4 uniform tiers are the same shape as DIKW's 3 transitions + 4 layers.

**Status in lance-graph:** **NOT YET CORRECTED.**

- `lance-graph/CLAUDE.md` line ~120 still reads: *"Magnitude = Contradiction depth from Staunen × Wisdom qualia."* Treats them as qualia archetypes; misses the DIKW-climb framing entirely.
- The §11.5 *"high Staunen × Wisdom spreads plasticity to adjacent rows"* mechanism reads correctly only when those markers come from the climb-position lifecycle layer (derived from `plasticity_counter` + `last_active_cycle` + codebook hit-rate), not the qualia codebook.
- No PR has landed to fix the framing.

**Action:** flagged here as `TD-CLAUDE-MD-STAUNEN-MISNAME`; pending CLAUDE.md maintenance pass. Three specific edits:

1. *"Magnitude = Contradiction depth from Staunen × Wisdom qualia"* → *"Magnitude = Contradiction depth across the DIKW climb axis (Data → Information → Knowledge → Wisdom). Staunen marks pre-integration encounter at/below the Data rung; Wisdom marks the apex. Both are derived from `plasticity_counter` + `last_active_cycle` + codebook hit-rate, NOT qualia archetypes."*
2. §11.5 rephrasing of "Staunen × Wisdom" → "DIKW-climb dynamics" or equivalent, with the substrate derivation made explicit.
3. Add a short DIKW-anchor sub-section to "The Click" that names the canonical four-layer lineage + the three bridging operations (Processing / Cognition / Judgment) and maps them onto the cascade tiers (HEEL → Processing-class, HIP → Cognition-class, TWIG → Judgment-class) so the cascade's purpose is self-describing.

Not in scope of this resolution doc (which is plan-side, not CLAUDE.md-side); flagged on the punchlist.

---

## 7. Resolved punchlist (post-#490)

Prioritised; each item names the smallest follow-up PR that closes it.

1. **Errata stubs on affected plans** (this PR) — append a 1-2 line errata block on:
   - `bindspace-singleton-to-mailbox-soa-v1.md` (pointing here for D-MBX-A1 shipped status + rename + emission-tombstone)
   - `identity-architecture-exists-vs-needs-v1.md` (pointing here for §N1 canon-supersession + §N5 bijection-shipped)
   - `unified-soa-convergence-v1.md` (pointing here for §4.2 stack-pin drift)
   - `polyglot-container-query-membrane-v1.md` (ratify research-only)

2. **NodeRow speaking accessors + LE round-trip + SoaEnvelope binding** (next PR, lance-graph) — `NodeRow::key_bytes()` / `edges_bytes()` / `value_bytes()` / `as_le_bytes()` / `from_le_bytes()`; const `ColumnDescriptor` table for the three top-level slots; `impl SoaEnvelope for NodeRowPacket<'a>` (borrowed slice wrapper, zero-copy). The bridge between the canonical row layout and the envelope ABI.

3. **MailboxSoA speaking setters** (same next PR) — `set_energy(row, value)`, `add_energy(row, delta)`, `set_plasticity(row, value)`, `set_threshold(value)`, `set_current_cycle(value)`, `set_last_active_cycle(row, value)`. Currently only the D-MBX-A1 column setters exist (`set_edge` / `set_qualia` / `set_meta` / `set_entity_type`); the original energy/plasticity columns are written by direct field access. Names should match the migration map's vocabulary so call sites self-describe.

4. **D-MBX-A2** (separate PR, gating) — Hamming planes + temporal/expert decisions on `MailboxSoA<N>`. The unblock for S1-S4.

5. **CLAUDE.md Staunen × Wisdom misnaming fix** (CLAUDE.md maintenance pass, separate) — rewrite the §"The Click" `Magnitude` line + §11.5 to read the markers from the plasticity-lifecycle layer.

6. **engine_bridge re-encode audit** (after D-MBX-A2 lands) — Close `engine_bridge.rs` qualia f32→i4 re-encode + the DTO-as-owned-Vec sites flagged in handover §9.

7. **bardioc handover sync** (separate bardioc PR) — `BINDSPACE_DISSOLUTION_HANDOVER.md` adds a post-2026-06-05 appendix reflecting #477/#487/#489/#490 shipped, this resolution doc's diff results.

8. **OntologyRegistry linear-scan → HHTL radix-trie** (separate PR, lance-graph) — Called out in #477 audit; classid-prefix-scoped lookup per canon's *"codebook scoping = the class routing prefix."*

9. **Two `CausalEdge64` types disambiguation** (separate PR, lance-graph) — Called out in #487 body.

---

## 8. What this doc does NOT do

- Does not create new plans or deliverables (every item in §7 lands in its own PR).
- Does not declare new architectural decisions. The canon (OGAR/CLAUDE.md P0) is the canon. The three-tier model (#477) is the three-tier model. This doc only reports what the diff IS.
- Does not retroactively rewrite plan bodies. Each affected plan gets a brief errata note pointing here (item §7.1).
- Does not edit `lance-graph/CLAUDE.md` (item §7.5 is a separate maintenance pass).
- Does not touch any code (item §7.2/§7.3 are the next PR).

---

## 9. Cross-references

- OGAR/CLAUDE.md P0 — the canon (operator-pinned).
- `lance-graph/CLAUDE.md` — doc-lock; `4ea6ac92` made it canonical.
- `bardioc/BINDSPACE_DISSOLUTION_HANDOVER.md` — the 2026-06-05 handover this doc extends.
- `docs/architecture/soa-three-tier-model.md` — #477 ratified model.
- `crates/lance-graph-contract/src/canonical_node.rs` — code form of the canon.
- `crates/lance-graph-contract/src/soa_envelope.rs` — envelope-level LE contract.
- `crates/cognitive-shader-driver/src/mailbox_soa.rs` — the D-MBX-A1 MailboxSoA.
- PR #470 (handover propagation), #477 (three-tier), #478 (singleton nudge), #480 (Phase-A NodeGuid, retired), #482 (GUID canon), #484 (bijection + D-IDENTITY-2), #487 (emission tombstone), #489 (canonical_node code form), #490 (wire-in + Display + Phase-A retirement).
