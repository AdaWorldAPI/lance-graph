# B1 — Semantic identity + storage substrate inventory (read-only, 2026-10-10)

Checkouts: `/home/user/_ro/lance-graph` @ `1573eb4c` (Merge #1464), `/home/user/_ro/OGAR`, `/home/user/_ro/ndarray` (main).
Grades: **VERIFIED-IN-CODE** (read at file:line), **CLAIMED-IN-DOC** (only prose says so), **ABSENT** (closed search space named).
Paths below are relative to `lance-graph/crates/lance-graph-contract/src/` unless prefixed.

## 0. What #1464 / #1465 changed (the reading contract)

- #1464 (merged, `1573eb4c`) files: board entry + plan + `aiwar.rs`, `canonical_node.rs`, `nan_projection.rs`, `ocr.rs`, `soa_graph.rs`.
  - W1 additive `_with` / `_plugged` forms that take a `ReadMode`/`ResolvedReading` as a parameter
    (`ocr::LayoutBlock::to_node_row_with`, `aiwar::aiwar_node_rows_with`, `soa_graph::{project_snapshot_with, nearest_anchor_with}`,
    `nan_projection::project_energy_nonfinite_plugged`). Plan claims 4/4 disable runs red (CLAIMED-IN-DOC: plan W1 bullet).
  - W4 doc-only deprecation of `classid_read_mode` + `CLASSID_*_V3` (VERIFIED-IN-CODE `canonical_node.rs:1839-1857`: doc "Deprecated, kept"; no `#[deprecated]` attribute).
  - `classid_read_mode` STILL falls through to `ReadMode::DEFAULT` (V1 tail, `Full`) for any unknown classid — `canonical_node.rs:1850-1857` + `BUILTIN_READ_MODES` `:1773-1819`. W6 (remove the V1 default) is OPEN.
- #1465 (open, branch `ccr-f5674497-orfkox`, file `crates/lance-graph-ogar/src/lib.rs`): adds the five canon domains to `concept_override`
  (`0x0701→OSINT_V3, 0x0A01→FMA_V3, 0x0E01→CPIC_V3, 0x0101→PROJECT_V3, 0x0202→ERP_V3`) and routes "reading-only canon domains"
  (refused by the capability join with `UnknownClassid|NoCapabilitiesFor`) with the palette seats. Tests in the patch:
  `canon_domains_read_from_the_plug_as_the_classid_table_reads_them`, `a_canon_domain_next_to_capability_ids_activates_and_reads_its_domain`. NOT on main.

## 1. OGAR authority + concept registration

| item | where | contract | grade |
|---|---|---|---|
| OGAR `CODEBOOK` (authoritative, PRIVATE const) | `OGAR/crates/ogar-vocab/src/lib.rs:1154` | `(&str, u16)` concept → `0xDDCC` id; "ids only arrive, never move" | VERIFIED-IN-CODE |
| `canonical_concept_id` / `canonical_concept_name` | `ogar-vocab/src/lib.rs:1904`, `:1926` | name↔id, 1:1 | VERIFIED-IN-CODE |
| `class_ids` const module | `ogar-vocab/src/lib.rs:1966` (+ `class_ids::ALL`, used by `resolve_concept_row` `capability_registry.rs:360`) | compile-time const per concept; tests `constants_match_codebook`, `every_codebook_entry_has_a_const` (CLAIMED in doc at lib.rs:1953-1956; names not re-opened) | VERIFIED (module) |
| `resolve_hotplug(consumer, classids, covered)` | `ogar-vocab/src/capability_registry.rs:432-478` | checks minted → ≥1 capability per id → expected executor → coverage both directions; returns `(concepts, capabilities)`. **Carries no reading.** | VERIFIED-IN-CODE |
| `PortSpec` + `APP_PREFIX` / `classview()` / `class_id()` | `ogar-vocab/src/ports.rs:51-120`; prefixes 0x0000..0x000C at `:99,142,221,292,343,428,497,574,630,652,678,728` | per-vendor render skin in the LOW u16 | VERIFIED-IN-CODE; **deprecated as "fossil" by D-HPR rule 5** (plan `v3-mandatory-hotplug-reading-v1.md` rule 5, CLAIMED-IN-DOC; no code marker in ports.rs found) |
| ruff → OGAR mint | `OGAR/crates/ogar-from-ruff/src/mint.rs:71` `mint_graph::<P: PortSpec>` | classid via the port's prefix; test `mint_is_port_agnostic_same_concept_different_render_prefix` (`:386`) | VERIFIED-IN-CODE |
| Contract mirror `ogar_codebook::CODEBOOK` | `lance-graph-contract/src/ogar_codebook.rs:584` | wire mirror, zero-dep; doc says "Carries the two domains … project-mgmt 0x01XX, commerce/ERP 0x02XX" but the test pins 0x04,0x05,0x08,0x09,0x0B,0x0C ids too (`:999-1027`) → doc line is stale | VERIFIED-IN-CODE (doc stale) |
| Contract `canonical_concept_id` | `ogar_codebook.rs:806` | used by `hotplug::verify_against_mirror` (`hotplug.rs:647`) | VERIFIED-IN-CODE |
| `AppPrefix` enum | `ogar_codebook.rs:301-394` (`prefix`, `from_prefix`, `render`) + `render_classid` `:401` = `compose_classid(concept, prefix)` (concept HIGH, prefix LOW) | membrane copy of PortSpec prefixes; test `app_prefixes_match_ogar_allocation_table` `:1057` — **hand-mirrored values, no cross-repo compile link** (contract is zero-dep) | VERIFIED-IN-CODE |
| classid split/compose | `ogar_codebook.rs:441-573` (`ClassidOrder::CanonHigh`, `classid_canon`, `classid_custom`, `classid_canon_compat`, `flip_classid`, `classid_app_prefix`, `classid_concept`) | canon HIGH since 2026-07-02 flip; tests `:1137,1183,1210,1221` | VERIFIED-IN-CODE |

Limitations: drift guard between OGAR CODEBOOK and contract mirror is a set of hard-coded `assert_eq!` spot checks (`ogar_codebook.rs:999-1027`), not a full table compare — VERIFIED-IN-CODE. A runtime cross-check exists only at activation time (`hotplug::verify_against_mirror`, `hotplug.rs:647`).

## 2. HotPlug / Activation / ResolvedReading / SlabReading (post-#1464 main)

All in `hotplug.rs` (1328 lines). Module doc `:23-48` names the ONE resolution path: SPOG context (`spog_tenants::graph_of(key)`) × OGAR `Activation` × `SlabDeclaration` × this build (`SlabReading::from_tag`, `ENVELOPE_LAYOUT_VERSION`).

| item | line | contract | writer / lifetime |
|---|---|---|---|
| `HotPlug {consumer, classids: &'static [u16], covered}` | `:54-63` | one `const` per consumer | consumer, compile-time |
| `Activation {concepts, capabilities, read_modes (private), register_laws (private)}` | `:73-139` | no `Default` by design; `read_mode_for` fails closed `NoReadingFor` (`:198-205`); `declared_readings` audit slice (`:216`) | built only by an authority via `Activation::new` (`:144`); owned, per plug |
| `with_register_laws` / `register_law_for` | `:162-180` | law per concept, fails closed `NoRegisterLawFor` | **no production caller anywhere** — grep over `_ro/{lance-graph,OGAR,ndarray}` *.rs finds only hotplug.rs itself + register128.rs doc links (closed space: those three checkouts) → OgarAuthority never declares a law, so `bind_signed_register` can never succeed in production. ABSENT (production) |
| `SlabReading` `#[repr(u8)]` | `:231-256` | `Facet96=0`, `Register128=1`, `RegisterI4x32(law)`, `RegisterI8x16(law)`; tags `from_tag` `:267-293` (0,1, 2..=7 = carving×law, else `UnknownSlabReading`), `to_tag` `:297-312` | declared by the slab's metadata envelope ("beside the slab") |
| `SlabDeclaration {reading, value_schema, layout_version}` | `:322-334` | physical facts only; NO concept/classid | writer records what it used |
| `ResolvedReading {concept, read_mode, slab: Option<SlabReading>}` | `:338-348` | `Copy+Eq+Hash`, resolve once per population | |
| `Activation::resolve_for_context(concept, Option<&SlabDeclaration>)` | `:370-397` | OGAR must know concept → else `NoReadingFor`; no decl → OGAR reading, `slab:None`; decl → layout must == `ENVELOPE_LAYOUT_VERSION` (=2, `soa_envelope.rs:54`) else `SlabLayoutVersion`; **declared value_schema overrides, tail+edge_codec stay OGAR's** | |
| `resolve_tenant_reading(key, slab)` | `:407-413` | `graph_of(key)` then above | |
| `ResolvedReading::bind_register128(rails)` | `:434-455` | requires slab `Register128` + schema has every rail tenant | only caller outside contract: `crates/jc/tests/register128_bounded_stats.rs:91,197` (test) |
| `bind_signed_register(&Activation, rails)` | `:480-523` | slab carving+law must equal concept law | no caller outside hotplug tests (grep closed as above) |
| `ActivationDrift` | `:526-617` | named bangs incl. `MirrorDrift` | |
| `mirror_disagreement` / `verify_against_mirror` | `:628-650` | authority ids vs contract mirror | |
| `CapabilityAuthority` trait | `:722` | `activate(&HotPlug) -> Result<Activation, ActivationDrift>` | implemented by `lance-graph-ogar::OgarAuthority` (`crates/lance-graph-ogar/src/lib.rs:535`) |

OgarAuthority reading derivation (main): `plug_readings` `lance-graph-ogar/src/lib.rs:452-462` = every plugged id → `concept_override(id).unwrap_or(ReadMode::PLUG_AND_PLAY_V3)`; `concept_override` `:476-486` on main has ONLY `0x1717` (blockly palette, `Bootstrap`). #1465 (open) adds the 5 canon domains.

Tests in hotplug.rs (names, `grep -n "fn .*()"`): `socket_shape_round_trips_through_a_trait_object` :752, `an_empty_table_activates_but_still_refuses_to_invent_a_reading` :825, `resolution_is_deterministic` :880, `one_slab_under_two_contexts_resolves_per_context` :897, `an_unknown_context_fails_closed` :922, `an_unsupported_physical_reading_fails_closed` :939, `an_undeclared_slab_keeps_inherited_behaviour` :984, `a_historical_slab_survives_a_class_migration` :1002, `payload_bytes_do_not_participate` :1042, `one_resolution_serves_a_whole_population` :1066, `signed_rails_need_a_signed_carving_and_an_agreeing_law` :1152, `a_register_takes_its_concept_from_the_context_never_the_payload` :1256, `a_concept_missing_from_the_mirror_is_named_drift_not_silence` :1299. External: `jc/tests/register128_bounded_stats.rs`.

Observed limitations:
- `SlabDeclaration` has no serializer/parser anywhere — it is a Rust struct only; tag u8 decoding exists (`from_tag`) but no envelope writes it (see §(c)).
- `resolve_for_context` takes the declared value_schema verbatim; it does not check that the OGAR tail/edge codec is compatible with the declared reading.
- `layout_version` must equal exactly `2`; any future bump makes every prior declaration unreadable (no range).

## 3. ClassView, EdgeCodecFlavor, CascadeShape — how a class "selects a reading"

- `ClassView` trait: `class_view.rs:1065`, keyed by `ClassId = u16` (`:54`, the concept half). Selection accessors, all DEFAULT-bodied:
  - `edge_codec_flavor` `:1271` → `CoarseOnly`; `rail_carving` `:1289`; `cascade_shape` `:1334` → `G3D4`; `band_reading` `:1354` (legacy); `epistemic_reading` `:1369` → `None` (refuses); `value_schema` `:1397` → `Full` ("TEMPORARY POC", doc says revert to Bootstrap before merge — still Full).
  - Implementors (closed space: `_ro/lance-graph/crates`, `_ro/OGAR/crates`, `q2`, `MedCare-rs`): 16 `impl ClassView for` sites (ontology RegistryClassView, WikidataClassView, sap CatsSchema, OGAR OgarClassView, q2 OsintClassView, MedCare 5 views, tests). **None overrides `value_schema`, `edge_codec_flavor` or `cascade_shape`** (grep `fn (value_schema|edge_codec_flavor|cascade_shape)(&self` → only the trait itself). VERIFIED-IN-CODE / overrides ABSENT.
  - Test: `value_schema_default_is_full_temporary_poc` `class_view.rs:2157`.
- `EdgeCodecFlavor {CoarseOnly=0, CoarseResidue=1, Pq32x4=2}` `canonical_node.rs:836-849`; also a field of `ReadMode` `:1584-1593`. `EdgeBlock = EdgeFacet([u8;16])` `:720`, `:816` — bytes only, no typed accessors on the row.
- `CascadeShape {G6D2, G4D3, G3D4}` `facet.rs:801-819`; `FacetCascade {facet_classid:u32, tiers:[FacetTier;6]}` `facet.rs:94-101` (`FacetTier{lo,hi}` `:38`), a reading over a borrowed `[u8;16]`.
- **Observed overlap (not named in #1464's residue list):** `value_schema` and `edge_codec` are resolvable two ways — `ClassView::value_schema/edge_codec_flavor(classid u16)` and `ReadMode` via `Activation`. The #1464 plan names only `classid_read_mode`/`BUILTIN_READ_MODES` as the second path; the ClassView accessors are dormant (no overrides) but still classid-keyed reading selectors. The facet reading of the 12-byte payload itself (G6D2/G4D3/G3D4 grouping) is selected ONLY by `ClassView::cascade_shape`, which has no plug-path equivalent in `ReadMode`.

## 4. FieldMask / WideFieldMask

- `FieldMask(pub u64)` `class_view.rs:70`; `MAX_FIELDS=64`; `from_positions` `:84` ignores (does not fold) positions ≥64; `with`, `has`; doc: "Presence, NEVER semantics (C2)"; N3 append-only positions.
- `WideFieldMask(WideRepr::{Small(u64)|Wide(Box<[u64]>)})` `:243-251`; `from_positions(&[u8])` `:310` (so positions ≤255), `full_for(field_count)` `:360`, `EMPTY`, and/or/minus, manual `PartialEq/Hash` representation-independent `:617-640`, `WideMaskCapError` `:649`.
- Two distinct bit-spaces share the type: (i) class-field presence over `ObjectView` fields (ontology, rbac, sap, wikidata_hhtl `presence_mask` `lance-graph-ontology/src/wikidata_hhtl.rs:77`), (ii) **ValueTenant presence** — `ValueSchema::field_mask()` `canonical_node.rs:1431-1495`, where bit = `ValueTenant` discriminant. Nothing in the type distinguishes them.
- Consumers of the tenant-presence space: `ValueSchema::has/tenant_bytes` (`:1499-1517`), `hotplug::bind_register128/bind_signed_register` (`hotplug.rs:446,511`), `causal_witness::project/elected(&WideFieldMask)` (`causal_witness.rs:415,444` — here the mask is over 24 loci, a third bit-space).
- Tests: `field_mask_is_presence_bits` `class_view.rs:2032`, `field_mask_inherit_is_nondestructive_union` `:2067`, `execute_defaults_fires_wide_positions_past_64` `:1881`.

## 5. ValueTenant / VALUE_TENANTS / ValueSchema (canonical_node.rs)

- Rule (`:986-991`): "Stable, append-only positions … a tenant's presence bit and its byte offset in VALUE_TENANTS never move once instances persist, and retired tenants are never reused. The discriminant IS the FieldMask bit position." Compile-asserted contiguous + discriminant==index (`:1371-1395`).
- Layout (row offsets; value-slab offset = row−32):

| # | tenant | bytes | row off | kind | ValueSchema membership |
|---|---|---|---|---|---|
| 0 | Meta | 8 | 32 | U64 | Cog, Full |
| 1 | **Qualia** (`QualiaI4_16D`, 16×signed i4) | 8 | 40 | U64 | Cog, Full |
| 2 | MaterializedEdges (4×CE64) | 32 | 48 | U64×4 | Full |
| 3 | Fingerprint | 32 | 80 | U8 | Cog, Comp, Full |
| 4 | HelixResidue (Signed360) | 6 | 112 | U8 | Comp, Full |
| 5 | TurbovecResidue | 16 | 118 | U8 | Comp, Full |
| 6 | Energy f32 | 4 | 134 | F32 | Cog, Full |
| 7 | Plasticity | 4 | 138 | U32 | Cog, Full |
| 8 | EntityType | 2 | 142 | U16 | Cog, Comp, Full |
| 9 | Kanban | 8 | 144 | U64 | Cog, Full |
| 10-12 | Frozen/Learned/ExploreStyle | 12 each | 152/164/176 | U8 | Full |
| 13 | Tekamolo (Facet96) | 16 | 188 | U8 | Full |
| 14 | **CausalWitness** (Facet96, G24N4 signed i4 loci, EXPERIMENTAL) | 16 | 204 | U8 | Full |
| 15 | EpisodicBasin | 32 | 220 | U8 | Full |
| 16-17 | Register0/1 (Register128, no classid) | 16 each | 252/268 | U8 | Full |
| 18 | Nars16x8 | 16 | 284 | U16×8 | Full |
| 19 | MoorePalettePairs | 16 | 300 | U8×16 | Full |
| 20 | MooreNars16 | 16 | 316 | U16×8 | Full |

  Full ends at row 332 → `Full.tenant_bytes() == 300` of 480 (pinned by the test message `canonical_node.rs:2773`). Compile asserts `:1520-1522` (Full ≤ 480, Full covers every tenant, Bootstrap empty).
- `ValueSchema {Bootstrap=0 (#[default]), Cognitive=1, Compressed=2, Full=3}` `:1407-1419`.
- **BoardAggregates reservation**: prose only, "re-based by ordinal" now to **21** (`:1274-1283, :1307, :1322, :1341`). No enum variant, no descriptor. CLAIMED-IN-DOC.
- Limitation: tenants 10-20 exist only in `Full`; the `Cognitive`/`Compressed` presets never gained them, so a domain reading `OSINT_V3`/`FMA_V3` cannot bind Register128 rails (`RegisterRailAbsent`).

## 6. Moore tenants (moore_tenant.rs)

- `MOORE_LANES=8` `:34`; `MooreSlot` `:42-59` (NW..SE, `offset()` `:93`); `MooreNars16(u16)` `:121`: Pearl3 b0-2, Energy4 b3-6, Plasticity3 b7-9, Polarity1 b10, Epi5 b11-15; `from_ce64(edge, polarity)` `:199` (reads CE64 40..42, 46..49, 50..52, 59..63); `ce64_upper(witness)` `:213` (writes those + witness<<53, direction 43..45 = 0); `epistemic(slot)` `:177` decodes Epi5 as `Epi5Gen::V1` directly.
- `MooreTenantView::new(&[u8;480])` `:290` (`nars16`, `palette_pair`, `moore_nars16`); `MooreTenantMut::new(&mut [u8;480])` `:328` (`set_nars16`, `set_palette_pair`, `set_moore_nars16`, `lift_ce64` `:370-405`, all-or-nothing, refuses mixed witness / reserved Epi5).
- Status per module doc `:16-27`: physical layout RATIFIED, MooreNars16 field reading CANDIDATE; Nars16x8 reading undeclared.
- Own tests `:413-800`: `a_layout`, `b_existing_tenants_preserved`, `c_new_tenant_isolation`, `d_le_round_trip`, `e_moore_slot_identity`, `g_exhaustive_moore_nars16_fields`, `g_ce64_bit_mapping`, `h_direction_refusal`, `i_witness_ownership`, `k_zero_fallback`, `l_codec_mutations_are_caught`.
- **Every call site outside the module** (closed space: `.rs` in `_ro/{lance-graph,OGAR,ndarray}`, `q2`, `MedCare-rs`, `a2ui-rs`, `tesseract-rs`, `lance-graph-java`, `odoo-rs`, `openproject-nexgen-rs`):
  - `lance-graph-planner/tests/moore_tenant_isa_equivalence.rs:13,86-95` — TEST: `MooreTenantMut::lift_ce64` (writer), `MooreTenantView::moore_nars16` + `ce64_upper` (reader).
  - `cognitive-shader-driver/tests/moore_palette_pairs_lut.rs:14,42-69` — TEST: `set_palette_pair` (writer), `palette_pair` (reader), `ValueTenant::MoorePalettePairs.value_offset()`.
  - `perturbation-sim/examples/moore_observable_first.rs:84,293`, `MoorePertubationsfeld.rs:33,84,336` — EXAMPLES: use `MooreSlot` enum only, never the tenant bytes.
  - `lance-graph-planner/tests/moore_nars16_probe.rs:112` — defines its OWN local `struct MooreNars16(u16)` (probe predating the contract type), does not import the contract one.
  - `lance-graph-contract/src/lib.rs:129` `pub mod moore_tenant;` (re-export only).
  - **No production reader or writer.**
- Limitations: View/Mut take a raw `&[u8;480]` — no `ResolvedReading`/`ValueSchema` gate, so they "read" tenants 18-20 even on a `Cognitive`/`Compressed` row where those bytes are not materialised (unlike Register128, which is bind-gated). `epistemic()` bypasses the `(classid, rail) → Epi5Reading` declaration that `epistemic_state5` requires (module doc `epistemic_state5.rs:42-49`).

## 7. NodeRow / NodeGuid / EdgeBlock / FacetCascade / SoaEnvelope

- `NodeGuid([u8;16])` `canonical_node.rs:35`, `#[repr(C, align(16))]`; `classid()` `:270`; `new` `:230` (V1 `family u24 · identity u24`), `new_v2` `:542`; `mint_for(tail_variant, …)` `:404-460` — V1 → `new`; V2/V3 → `new_v2` (**V3 and V2 mint the same byte layout**; the only difference is the tail label), and without `guid-v2-tail` the V2/V3 arm PANICS instead of degrading to V1 (doc `:386-402`). `NodeGuid::read_mode()` `:327` = `classid_read_mode(self.classid())` — a carrier-method entry to the deprecated path; its doc does NOT carry the #1464 "Deprecated, kept" note (only `classid_read_mode` does, `:1839`).
- `NodeRow {key: NodeGuid, edges: EdgeBlock, value: [u8;480]}` `:892-896`, `#[repr(C, align(64))]`; size asserts 16/16/512 `:899-901`; `NODE_ROW_COLUMNS` `:930`, `NODE_ROW_STRIDE=512` `:952`, `GUIDS_PER_NODE=32` `:970`.
- `EdgeFacet([u8;16])` `:720`, `EdgeBlock = EdgeFacet` `:816` (bytes only; `.facet()` projection); `EdgeCodecFlavor` `:836`.
- `FacetCascade` `facet.rs:94` (`facet_classid: u32` native-endian + 6 tiers); `CascadeShape` `facet.rs:801`.
- `SoaEnvelope` trait `soa_envelope.rs:170-215`: `LAYOUT_VERSION = ENVELOPE_LAYOUT_VERSION (=2, :54)`, `columns`, `row_stride`, `n_rows`, `cycle`, `mailbox_owner()` (default 0 = bootstrap), `as_le_bytes`, `row_le`, `column_le`. `NodeRowPacket<'a>` `canonical_node.rs:1890-1908` implements it for `&[NodeRow]`.
- `spog_tenants::graph_of(addr)` `spog_tenants.rs:41-43` = `classid_canon(addr.classid())` — **the concept IS read from the key's canon half**; what #1464 forbids is deriving the *reading* from the classid value.
- Persisted form in the wild: q2 `cockpit-server/src/osm_lance.rs:447-473` writes rows as Lance `FixedSizeBinary(512)` with schema metadata `soa:envelope_layout_version`, `soa:row_stride`, `soa:row_carving = "key:0..16|edges:16..32|value:32..512"`, `soa:endianness`, `soa:classid`, `soa:slab_digest`, `soa:source` (`:70-81`). Same keys in `lance-graph/crates/lance-graph/examples/soa_to_lance.rs` and `tests/soa_verbatim.rs`. **No value-schema or slab-reading key.**

## 8. CausalEdge64 + EpistemicState5

- Layout constants `causal-edge/src/layout.rs`: S 0, P 8, O 16, FREQ 24, CONF 32, CAUSAL(Pearl3) 40, DIR(S/P/O sign triple) 43, INFER (signed i4 mantissa) 46-49, PLAST 50-52, W(witness, 6 bit) 53-58, TRUTH/topology 59-60, spare/certification 61-63; `V1_TEMPORAL_SHIFT=52` `#[deprecated]`.
- `CausalEdge64(pub u64)` `causal-edge/src/edge.rs:180`, canonical bytes `to_le_bytes` `:207`; doc `:165-167`: "Nothing in-tree persists a v2 edge as bytes today". `forward(weight, compose_s/p/o)` `:668-690` decodes the op from `weight.inference_mantissa()` (v2) and faults on unexecutable codes (`IsaFault::Unsupported`). `revision` `:723-733` = truth only (f,c), keeps every other field of `self` including bits 53-63.
- `isa::truth` `causal-edge/src/isa.rs:208-264`: `evidence_weight` (saturates at c≥0.999; known NaN defect when both saturate, pinned in `tests/ce64_isa_golden.rs`), `deduction`, `induction`, `abduction`, `revision → Option` (None if ws ≤ EPSILON), `synthesis` (mean, "not a NARS rule").
- `epistemic_raw5` / `with_epistemic_raw5` `edge.rs:1091,1114` (v2) and stub `:1322,1327`; `edge_v3.rs:261`.
- `EpistemicState5` `lance-graph-contract/src/epistemic_state5.rs:316`: `raw5 = topology | certification<<2`, `Topology2` `:125` (Direct/IndirectKnown/IndirectUnknown/Unknown), `Certification3` `:173` (0 Open..5 Causes, 6/7 reserved), `decode(Epi5Gen, raw5)` `:340`, `Epi5Reading::project` `:425`, `Epi5Declarations` `:465` keyed `(ClassId, RailAxis)`. Doc `:42-49`: "Declared, never inferred". 12 `#[test]`s in module.
- causal-edge tests: `tests/ce64_isa_contract.rs`, `ce64_isa_golden.rs` (+ `ce64_legacy_capture.txt` golden capture), `ce64_op_contract.rs`, `d_rpf_conf_0.rs`, `d_rpf_p5_beta.rs`, `t2_pearl_subset_mask.rs`, `t4_learn_plasticity_patterns.rs`, `src/v2_layout_tests.rs`.
- Limitation: `MooreNars16::epistemic` (`moore_tenant.rs:177`) decodes bits without any `Epi5Declarations` lookup — a second, undeclared reading path for the same 5 bits.

## 9. Palette256 / PairAddress / DistanceLut

- `PaletteState(pub u8)`, `PairAddress(pub u16)` (`(left<<8)|right`, row-major, explicitly NOT Morton), `PaletteLut<'a>` (64 KiB law), `PALETTE_LUT_LEN` — `lance-graph/crates/cognitive-shader-driver/src/palette_perturbation.rs:28-80`. **Not in the contract.** Contract carries only `pub type Palette256Pair = (u8, u8);` `lance-graph-contract/src/awareness_facet.rs:32` and the MoorePalettePairs tenant ("#1336 PairAddress orientation", `canonical_node.rs:1162-1168`). Test: `cognitive-shader-driver/tests/moore_palette_pairs_lut.rs`.
- `helix::DistanceLut {table: Vec<u16>}` `lance-graph/crates/helix/src/distance.rs:18`, `linear()` (|a−b|, metric) `:24`, `from_floor(&RollingFloor)` `:37`; 256×256×u16 = 128 KB. bgz17 has its own `Palette` (`bgz17/src/palette.rs:14`).

## 10. Register128 and its readings (register128.rs)

- `Register128(pub [u8;16])` `:56`; only interpretation the crate provides: `words()` → 4×LE u32 `:67` (`from_words` `:79`).
- `RegisterRails::{One,Two}` → `tenants()` `[Register0]` / `[Register0, Register1]` `:96-118`; `RegisterLanes` (concept + rails, `get/set` on `NodeRow`) `:121-180`, built only by `bind_register128`.
- `RegisterCarving::{I4x32 (32×i4, nibble k=lo/hi of byte k/2), I8x16}` `:189-205`.
- `RegisterLaw` `:224-237`: `RelativeOffset` ("a relative address in a local window … An address, not a strength"), `AxisPosition` ("signed position on a declared semantic axis"), `Support` ("positive supports, zero unresolved, negative falsifies"). Doc: Epistemic state deliberately absent.
- `SignedRegisterLanes` `:277` (`read_i4x32/read_i8x16/write_*` `:358-430`, checked carving+law on every call); `SignedRegisterRefusal` `:241`.
- Tests (12): `i4x32_round_trips_every_value_in_every_dim` :455, `i8x16_round_trips_and_dim_k_is_byte_k` :487, `the_same_bytes_differ_under_the_two_carvings` :515, `a_different_law_is_refused_and_writes_nothing` :539, `a_different_carving_is_refused` :574, `an_out_of_range_i4_is_refused_before_writing` :602, `signed_writes_stay_inside_their_rail` :628, `the_register_is_exactly_one_declared_rail` :662, `words_round_trip_little_endian` :676, `the_rails_touch_no_other_tenant` :687, `writing_a_rail_leaves_every_other_byte_alone` :722, `register_writes_are_counted_per_tenant` :755. External: `jc/tests/register128_bounded_stats.rs` (unsigned only).
- Consumers of signed carvings: none outside tests (no authority declares a law, §2).

## 11. Answers

### (a) Sanctioned path for "a class selects how its bytes are read" after #1464

Plan rule 2 (`.claude/plans/v3-mandatory-hotplug-reading-v1.md`, CLAIMED-IN-DOC, operator DECISION): *"**One resolution path.** How a row is read (tail, value schema, edge codec, `Facet96` vs `Register128`) comes from the plug (`HotPlug` → `CapabilityAuthority::activate` → `Activation::resolve_for_context`) plus the slab's own `SlabDeclaration`. Nothing else selects a reading. Lockstep tables (a central list every consumer must be added to) are deprecated."*

Code (VERIFIED-IN-CODE `hotplug.rs:30-48`): *"[`Activation::resolve_for_context`] composes them into one [`ResolvedReading`] or a named [`ActivationDrift`]. - **An explicit declaration wins for reading existing data.** … - **No declaration means inherited behaviour**, not "this is Facet96" … - **No per-concept permission table.**"* and `canonical_node.rs:1839-1844`: *"a classid-keyed reading is the legacy path. New code takes its reading from the plug (`Activation::read_mode_for` / `Activation::resolve_for_context`), and the contract's `_with` / `_plugged` forms accept it as a parameter."*

Caveats (VERIFIED-IN-CODE): (1) on main the canon domains (0x0701/0x0A01/0x0E01/0x0101/0x0202) still get their non-default reading only via `classid_read_mode`; the plug gives them `PLUG_AND_PLAY_V3` (`Full`) until #1465 lands. (2) The 12-byte payload grouping (`CascadeShape`) and the Epi5 reading have NO plug-path field — still `ClassView::cascade_shape` / `Epi5Declarations` keyed by classid. (3) `ClassView::value_schema`/`edge_codec_flavor` remain a classid-keyed selector (dormant: no implementor overrides). (4) `NodeGuid::read_mode()` still routes to the legacy lookup without a deprecation note. (5) Register laws: path exists in type, no authority supplies one.

### (b) Tenants 18-20 outside their own module + tests

Closed search space: all `*.rs` under `_ro/lance-graph`, `_ro/OGAR`, `_ro/ndarray`, `q2`, `MedCare-rs`, `a2ui-rs`, `tesseract-rs`, `lance-graph-java`, `odoo-rs`, `openproject-nexgen-rs` (excluding `target/`), patterns `MooreTenantView|MooreTenantMut|MooreNars16|MoorePalettePairs|Nars16x8|lift_ce64|ce64_upper|MOORE_LANES|MooreSlot|ValueTenant::(Nars16x8|MoorePalettePairs|MooreNars16)`.
- Read/written ONLY by: `moore_tenant.rs` (+its tests), and two TEST files in other crates — `lance-graph-planner/tests/moore_tenant_isa_equivalence.rs` (lift_ce64 write, moore_nars16 read) and `cognitive-shader-driver/tests/moore_palette_pairs_lut.rs` (set/get palette pair). `perturbation-sim` examples import `MooreSlot` only (no tenant bytes). `canonical_node.rs` declares them (+ its own layout tests, `ValueSchema::Full`).
- **ABSENT in production code.** Implicit touch: any code iterating `VALUE_TENANTS` or using `Full.tenant_bytes()` is affected by their existence — notably `weather-poc/src/canonical.rs:26-30` (`weather_value_offset()` = end of last tenant) and `tenant_counter.rs:27` (`N_TENANTS = VALUE_TENANTS.len()` = 21).
- `Nars16x8` (18) has NO accessor-level reader besides `MooreTenantView::nars16`/`set_nars16`, and no test outside `moore_tenant.rs` touches it at all.

### (c) Serialized artifacts carrying FieldMask bits 18-20, SlabReading tags 2..=7, or the 300-byte tenant_bytes

Closed search: (i) files `*.soa *.lance *.bin *.snap *.golden* *.arrow *.parquet *.bso2` and dirs `fixtures|golden|testdata|data` in `_ro/lance-graph` (found: `python/examples/test_kg/*.lance` (KG example, not NodeRow), `probes/weather-p1/jc_input.bin` (2,400,016 B = 16 + 2.4M, not a multiple of 512 → not NodeRows), `crates/weather-poc/data/field_manifest_v1.tsv`, others are ontology/model data); (ii) `*.json *.tsv *.toml *.yaml *.txt` in `_ro/lance-graph`, `q2`, `MedCare-rs` for tenant/reading names → none; (iii) Lance schema metadata keys (`soa:*`) in `_ro`, `q2`, `MedCare-rs` → only `envelope_layout_version`, `row_stride`, `row_carving`, `endianness`, `classid`, `slab_digest`, `source` (+mtime/len) — no value schema, no FieldMask, no slab-reading tag; (iv) `.to_tag()` / `SlabReading::from_tag` callers → none outside hotplug.rs.
- **FieldMask bits 18-20:** never persisted; only computed in-memory by `ValueSchema::Full.field_mask()`. ABSENT.
- **SlabReading tags 2..=7 (and 0/1):** no serializer, no envelope field, no fixture. `SlabDeclaration` is constructed only in tests (`hotplug.rs` tests, `jc/tests/register128_bounded_stats.rs:33,190`). ABSENT.
- **300-byte tenant_bytes:** appears only as test assertions (`canonical_node.rs:2773,2860,2977`; `moore_tenant.rs:438`). **One live dependency worth flagging:** `weather-poc` bakes 512-B rows to any `io::Write` (`canonical.rs:180-215`) with the W1 weather image at `weather_value_offset()` = `Full.tenant_bytes()` = 300 (`:262,292` tests pin `== Full.tenant_bytes()`). The offset is *designed* to move when tenants are appended ("append-only tenants added ahead of weather move this extension automatically", `:5-7`), so a weather bake written before tenants 16-20 existed (value-slab offset 220 before Register0/1, 252 before tenants 18-20) is unreadable at today's 300 with no version marker beyond `ENVELOPE_LAYOUT_VERSION=2` (unchanged by those mints). No such bake is committed in the searched repos (closed space above). This is a real exception to the "offsets never move once instances persist" reading of `:986-991`: weather's own region is outside VALUE_TENANTS and moves by construction.

### (d) Signed readings: value semantics vs coordinate/mask/opcode semantics

| reading | where | signed form | semantics (per its own doc) | class |
|---|---|---|---|---|
| Qualia | `qualia.rs:175` `QualiaI4_16D(u64)`, tenant 1 | 16×i4 −8..7 | bipolar chroma channels; `magnitude()` | **VALUE** |
| I4x32 | `atoms.rs:74` | 32×i4 | thinking-style weighting; sign = bipolar pole | **VALUE** |
| RegisterLaw::AxisPosition | `register128.rs:229-232` | i4/i8 | signed position on a semantic axis (fox +7) | **VALUE** (axis coordinate in meaning-space, a magnitude) |
| RegisterLaw::Support | `register128.rs:233-235` | i4/i8 | evidence stance | **VALUE** |
| EdgeCodecFlavor::CoarseResidue | `canonical_node.rs:843-845` | per-dim signed 4-bit residue | quantization residual | **VALUE** |
| CausalWitness (G24N4) | tenant 14, `causal_witness.rs:201`; doc `canonical_node.rs:1057-1073` | 24×i4 | "each nibble is a context pointer — a signed ±8 window offset … never a strength/magnitude (loci, not values)" | **COORDINATE** |
| RegisterLaw::RelativeOffset | `register128.rs:225-228` | i4/i8 | "a relative address in a local window … An address, not a strength" | **COORDINATE** |
| HelixResidue `Signed360` | tenant 4 | 48-bit signed sphere | full-sphere position | **COORDINATE** |
| MooreSlot offsets | `moore_tenant.rs:93` | `(i8,i8)` | neighbour displacement | **COORDINATE** |
| CE64 INFER mantissa | `causal-edge/src/layout.rs:15-24` | i4 | sign = chain direction, \|v\| = NARS rule index | **OPCODE** (enum), not value, not coordinate |
| MooreNars16 `Energy4` | `moore_tenant.rs:110,149` | raw nibble of CE64 46..49 | named "Energy" but documented as "the raw signed-mantissa nibble, CE64 bits 46..49" = the INFER opcode field | **OPCODE mislabelled as energy** — naming hazard |
| CE64 DIR (43..45) | layout.rs | 3 sign bits | S/P/O sign triple | **MASK/direction** |
| EpistemicState5 | bits 59..63 | unsigned product code | topology × certification | **MASK-like product code** (not signed) |

Mask semantics proper (`FieldMask`, `WideFieldMask`, Pearl3 causal mask, presence bits) are all unsigned; none of the signed readings is a mask.
