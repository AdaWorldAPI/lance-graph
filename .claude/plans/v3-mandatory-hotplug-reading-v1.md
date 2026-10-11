# v3-mandatory-hotplug-reading-v1 — the reading comes from the plug, never from the classid value

**Status:** ACTIVE (D-HPR-0..6). DECISION by the operator, 2026-10-10. W0 (this plan + the ruling record) lands first; W1 onward are code.

## The ruling (operator, 2026-10-10, verbatim)

> "Yes V3 mandatory, everything else in regards to capabilities hotplug.rs plug and play > ogar pattern making sure lockstep is deprecated"
>
> "Any concept is immutable. That's why Ontologies exist. Different languages are a label. The concept needs to be immutable adress"

Earlier the same day: *"only the hotplug.rs and Ontology slab metadata envelope define if 32 + 96 is in storage or 128"* and *"1000 V3 Marker is a fossil too"* (plan `cross-glove-business-parity-v1.md` §C.9.3).

What it means, in rules:

1. **V3 is mandatory.** No new key is minted with a V1 tail, and no reading falls back to V1 silently.
2. **One resolution path.** How a row is read (tail, value schema, edge codec, `Facet96` vs `Register128`) comes from the plug (`HotPlug` → `CapabilityAuthority::activate` → `Activation::resolve_for_context`) plus the slab's own `SlabDeclaration`. Nothing else selects a reading. Lockstep tables (a central list every consumer must be added to) are deprecated.
3. **A concept is an immutable address.** Its id never changes meaning. This is what ontologies are for.
4. **Names in different languages are labels of one concept, not separate concepts.** "Stundenzettel", "TimeSheet", "Zeiterfassung" and "TimeEntry" are labels; the concept is the address (here `0x0103` / `ogit.WorkOrder:TimeSheet`). A label table maps names to the address; it never mints.
5. **The classview fossils are deprecated, not evicted** (operator, 2026-10-10: *"1000 isn't hurting, just document and Mark it as deprecated. No need to force evict"*). The `0x1000` V3 marker and the per-app render prefixes `0x0000`–`0x000C` stop carrying meaning for new work. Keys, constants and callers that already use them keep working; nothing is migrated by force.

## Where things stand (measured 2026-10-10 on `origin/main` `feb21972`)

**The plug path already exists.** `lance-graph-ogar::plug_readings` (`lib.rs:452`) gives every plugged concept `ReadMode::PLUG_AND_PLAY_V3` unless `concept_override` (`lib.rs:477`) names an exception. `Activation::read_mode_for` fails closed (`NoReadingFor`), never defaulting.

**The second path, which stops selecting new readings** (⊘ the heading read "The second path that must go", which contradicts rule 5; regraded 2026-10-11 — the path stays as the reader of stored legacy keys, it stops being where a NEW reading comes from): `canonical_node::classid_read_mode(classid)` (`canonical_node.rs:1850`; `:1813` when written) looks the full `u32` up in `BUILTIN_READ_MODES` (`:1773`; was `:1743`). It has two faults:

- the V3 reading is chosen by the `0x1000` marker in the classview bits (the `*_V3` keys);
- an unknown classid falls through to `ReadMode::DEFAULT`, which is a **V1** tail. That is exactly the silent V1 fallback the 2026-09-07 ruling forbade on the plug path.

**Callers of `classid_read_mode` / the `*_V3` constants** (`grep -rn`, then each file opened):

| repo | file | use |
|---|---|---|
| lance-graph | `contract/src/ocr.rs:105,124` | `to_node_row(classid, identity)` reads schema + tail by classid |
| lance-graph | `contract/src/aiwar.rs:81,119` | mints OSINT rows on `CLASSID_OSINT_V3`, tail by classid |
| lance-graph | `contract/src/nan_projection.rs:167` | the non-`_resolved` variants look the schema up by classid |
| lance-graph | `contract/src/soa_graph.rs:204,457` | `DomainSpec` consts on `*_V3` classids, tail by classid |
| lance-graph | `contract/src/hhtl.rs`, `selection.rs`, `ogar_codebook.rs`, `classid_scan.rs` | tests / docs / legacy-form classification |
| lance-graph | `callcenter/src/graph_table.rs`, `deepnsm-v2/src/promote.rs`, `lance-graph/tests/canonical_witness_identity_probe.rs`, `contract/tests/v3_mint_reachability_probe.rs` | mint or read on `*_V3` |
| lance-graph | `lance-graph-ogar/src/lib.rs:735` | a test pinning that the old path yields V1 for an unplugged id |
| q2 | `osint-bake/src/bin/{fma,body}.rs`, `tools/body-soa-wire` | bake FMA on `CLASSID_FMA_V3`, tail by classid |
| q2 | `geo/src/bso2.rs` | `CLASSID_GEO_V3 = 0x0F01_1000` |
| q2 | `cockpit-server/src/osint_classview.rs:782` | a literal `0x0701_1000` key in a fixture |
| MedCare-rs | `medcare-cohorts/src/graph_feed.rs:25`, `spog_masks.rs` | `CLASSID_CPIC_V3`; mask tests over `0x0701_1000` |
| OGAR | `ogar-osm` (`CLASSVIEW_V3_SUBSTRATE`), `ogar-ro`, `ogar-dismech`, `ogar-loco` tests | mint concepts at classview `0x1000` |

**Persisted data carrying `0x1000` keys:** q2's FMA / body `.soa` bakes (`0x0A01_1000`), osm slabs (`0x0F01_1000`), the MedCare medication lane (`CPIC_V3`). These must stay readable (rule 5).

## Waves

- [x] **W0 — D-HPR-0: record.** This plan, the ruling (`entries/2026-10-10-v3-mandatory-hotplug-reading.md`), `INTEGRATION_PLANS.md`, `STATUS_BOARD.md`.
- [x] **W1 — D-HPR-1: contract takes the reading as a parameter.** `ocr`, `aiwar`, `nan_projection`, `soa_graph` get variants that take a `ReadMode` (or a `ResolvedReading`) instead of looking the classid up. `DomainSpec` carries its reading. The classid-lookup forms stay, documented as legacy, until W3. Additive: no consumer breaks.
  - Falsifier: a test per module that the new form yields the reading it was handed, including one where the handed reading differs from what `classid_read_mode` would return (so a caller that silently re-looks-up fails).
  - **Shipped:** `ocr::LayoutBlock::to_node_row_with`, `aiwar::aiwar_node_rows_with`, `soa_graph::{project_snapshot_with, nearest_anchor_with}`, `nan_projection::project_energy_nonfinite_plugged` (resolves each run through `Activation::resolve_tenant_reading`; an unplugged concept is `NoReadingFor`). Tests use classid `0x1718_0000`, which the lookup reads as V1 / `Full`; each is red when its form is changed to re-look-up (disable runs, 4/4).
- [x] **W2 — D-HPR-2: the canon domains get their reading from the authority.** The domain value models (`OSINT`/`PROJECT`/`ERP` Cognitive, `FMA`/`CPIC` Compressed) move into `concept_override`, keyed by concept, all on a V3 tail. `OgarAuthority` then answers for those concepts with no classview involved.
  - **Shipped:** `lance-graph-ogar::concept_override` arms `0x0701`/`0x0A01`/`0x0E01`/`0x0101`/`0x0202` → `ReadMode::{OSINT,FMA,CPIC,PROJECT,ERP}_V3`. Test `canon_domains_read_from_the_plug_as_the_classid_table_reads_them` pins each equal to `classid_read_mode(CLASSID_*_V3)`; red with an arm removed. No consumer plugs these concepts today (`grep` of every `HotPlug` in lance-graph, OGAR, MedCare-rs, q2, odoo-rs, openproject-nexgen-rs, HubSPO-rs, spear), so no live reading changes.
  - Falsifier: `read_mode_for(concept)` equals today's `classid_read_mode(<V3 classid>)` for each canon domain, so the move changes no reading.
- [ ] **W3 — D-HPR-3: consumers move.** lance-graph callcenter / deepnsm-v2 / probes, then q2 osint-bake + geo, MedCare cohorts. Each takes its reading from its `Activation`. ⊘ "Then `classid_read_mode` gets `#[deprecated]`" is struck: W4 and rule 5 rule the attribute out (it fails every caller under `-D warnings`).
  - **Census 2026-10-11 (synergy ledger):** MedCare has 0 callers. q2's callers are offline bake mints on `CLASSID_FMA_V3`, which can take the reading when next touched. deepnsm-v2 `promote::key_at` is blocked on a concept mint (its corpus has no OGAR concept), not on code. q2 geo (`osm_node` 0x0F01) cannot get a plug reading at all: the reading is gated behind the geo capability table, which expects `osm-soa-bake`. The geo and deepnsm cases need the rule "reading is not a capability" (O3).
- [x] **W4 — D-HPR-4: document the fossils as deprecated.** Done in `lance-graph-contract`: the five `CLASSID_*_V3` constants and `classid_read_mode` carry a "Deprecated, kept" doc note. A doc note, not `#[deprecated]`: the attribute would turn every existing caller red under `-D warnings`, which is the forced migration the ruling rules out. OGAR (`ogar-osm` `CLASSVIEW_V3_SUBSTRATE`, ro, dismech, loco) and q2 `geo` get the same note when next touched; existing mints stay.
- [ ] **W5 — D-HPR-5: legacy reads stay.** Stored `0x1000` and pre-flip keys keep resolving through `BUILTIN_READ_MODES`; no forced removal. A slab may additionally declare its reading (`SlabDeclaration`), which wins for its own data. Removing a table key is optional and only after nothing reads it.
  - Holds today. **Caveat (2026-10-11):** a `SlabDeclaration` overrides `value_schema` and the register reading only, never the key tail (`hotplug.rs:231-257`, `:392`). It cannot protect a V1/V2-tail slab. q2's `osint_scene.soa` is one: V2 tails under a classid whose builtin reading is V1. It is read by its own projection and must not be re-read through the table.
- [ ] **W6 — D-HPR-6: V1 cannot be reached by default.** `ReadMode::DEFAULT` stops being the fallback for an unknown classid (it becomes `NoReadingFor`, as on the plug path), and `guid-v3-tail` stops being a feature gate.
  - Split (2026-10-11). **W6a**: un-gate `guid-v3-tail`. It is default-on and no fleet consumer disables it, so only `--no-default-features` builds change; there the 5 V3 fossils read `DEFAULT` today (coresearch probe P-X3). **W6b**: an unknown classid errors. This is a behaviour change, and both q2 and MedCare have already hit the V1 default once, so it needs operator approval.
  - **Decided (operator, 2026-10-11: "Recommended accepted"): go, staged.** 1. W6a. 2. W3 moves the remaining mint paths onto plug readings. 3. A check that every stored artifact's classid resolves. 4. Only then W6b.
- [x] **W7 — D-HPR-7: storage readings leave the concept key** (O3, decided (a)). **Shipped (#1474):** the five domain arms are removed from `concept_override`. The domain models differed from `PLUG_AND_PLAY_V3` only in `value_schema`, which a `SlabDeclaration` overrides, so no new mechanism was needed. Test `canon_domains_read_plug_and_play_and_their_slab_restores_the_domain_model`. The five domain storage readings move out of `concept_override` (keyed by concept) into the slab's `SlabDeclaration` or the owning consumer's plug. A consumer plugging a shared concept then reads `PLUG_AND_PLAY_V3`. Falsifier: a plug of `0x0202` by a non-Odoo consumer reads `PLUG_AND_PLAY_V3`, and Odoo's own slab still reads `ERP_V3`.
- [x] **W8 — D-HPR-8: a plug that asks only for a reading succeeds** (decided). **Shipped:** `lance-graph-ogar` `OgarAuthority::activate`, tests `a_plug_covering_nothing_activates_for_its_reading_only` (SAP glove on `0x0103`, a non-executor on `0x0901`) and `a_plug_that_covers_a_capability_is_still_refused_without_its_table`. The glove-parity pin now says both glove plugs activate for their reading; claiming a capability on `0x0103` is still `NoCapabilitiesFor`. Reading is not a capability. A plug with `covered` empty gets readings only, also for a concept that has a capability table. A plug that names a capability still needs the table. Falsifier: `[0x0103]` with `covered: &[]` → `Ok` and `read_mode_for(0x0103) == PLUG_AND_PLAY_V3`; with `covered: &["anything"]` → refused as today.

## Open questions (operator)

- **O1.** What classview should a NEW mint of a canon domain use instead of `0x1000`: `0x0000` (shared canonical core), or an app-owned value? Not blocking: existing mints keep `0x1000`.
- **O2.** Does WoA's `ogit.WorkOrder:TimeSheet` resolve to `0x0103`, with "Stundenzettel" as its label, instead of today's `TimesheetActivity` (plan `cross-glove-business-parity-v1.md` §C.9.5)? Rule 4 settles that the German name is a label; it does not settle which WoA row the label names.

- **O3 (2026-10-11).** Since W2, `concept_override` keys storage readings by **concept** (`0x0101` → `PROJECT_V3`, `0x0202` → `ERP_V3`, `0x0A01` → `FMA_V3`). Those three u16s are also code-minted consumer appid heads (OpenProject, Odoo, q2). So any consumer that plugs `commercial_document` (aliased by the WoA, SMB, Odoo and HubSpo ports) gets the ERP storage reading, not `PLUG_AND_PLAY_V3`. This is latent: no plug names these concepts today. **Decided 2026-10-11: (a)** → W7 (D-HPR-7). Options were:
  - (a) key storage readings by a consumer-owned value;
  - (b) move the heads off occupied concept slots;
  - (c) document that plugging these concepts means that domain's storage model.

  Related, **decided 2026-10-11: yes** → W8 (D-HPR-8). The question was: should a plug that asks only for a reading (`covered` empty) succeed for a concept that has a capability table? It would unblock the SAP/Odoo `0x0103` plugs, q2 geo and deepnsm-v2. Record: `entries/2026-10-11-coresearch-labels-address-lifetime.md` (X6), `entries/2026-10-11-synergy-ledger-classid-hotplug.md`.

## What this plan does not do

- It does not re-key stored data. Rule 5 keeps old keys readable.
- It does not change the `domain:appid` layout (§C.9.4 audit: do not reverse `render_classid`).
