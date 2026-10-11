# Synergy ledger — classid reading, hotplug, cross-glove parity (2026-10-11)

**Status:** OPEN — revision of the 24-point listing (P1–P24) against source in
lance-graph, OGAR, OGIT, odoo-rs, q2, MedCare-rs and the fleet. Eight scouts
read the code (113 items; 108 shown, 5 private-repo items omitted). One
adversarial re-reader per scout checked every item against its cited lines:
106 hold, 7 were refuted or corrected (5 of them shown). The
completeness critic was not run (stopped to save tokens), so the ranking below
is the main thread's.

> **⊘ Corrected 2026-10-11 (operator, plan `concept-labels-hotplug-v1.md`):** capabilities on a
> concept carry no list of consumers. Each consumer's plug declares what it covers, and the
> expected-executors check in OGAR's `resolve_hotplug` is a lockstep residue to deprecate.
> The executor-list framing of `ogar-p18-capability-table-0x0103` and
> `odoo-capability-0x0103-consumer-tension` is superseded.

## Leverage ranking (open items)

**Low fruit (hours, no operator needed):**
1. `contract-ogar-reading-only-partition-vs-0x0103` — probe "reading is not a
   capability" in `lance-graph-ogar`. A plug with `covered` empty succeeds for
   a concept that has a capability table. This unblocks the SAP/Odoo `0x0103`
   plugs, q2 geo and deepnsm-v2 (rated high because it changes activation; the
   probe itself is hours).
2. `contract-slab-declaration-has-no-tail` — one doc sentence in `hotplug.rs`:
   a slab declaration wins for `value_schema` only, never the tail.
3. `fleet-sibling-plugs-skip-reading-socket` — sibling plug tests call
   `resolve_hotplug` directly and never assert a reading. Add a test-only
   `OgarAuthority.activate` check.
4. `fleet-lance-graph-ogar-lint-fmt-gate` — clippy and fmt lines for
   `lance-graph-ogar` in CI (6 pre-existing clippy errors to clear first).
5. `glove-p19-cats-validity-plane` — Quack already has `BoundField.validity`;
   `CatsBinder` can serve optional fields instead of refusing them.
6. `glove-p9-…` — D-XGP-6's gate is answer equality, not byte equality.
7. Board hygiene: the `board-*` rows (duplicates, retire wording, stale cells).

**High (a PR with a probe, or one operator answer):**
- Coresearch O3 (X6): `concept_override` keyed by concept collides with three
  appid heads. Operator: (a) / (b) / (c).
- `glove-p8-report-catalog-from-basis` (D-XGP-5) and
  `glove-draft-has-no-range-or-group` (Quack Draft has only Eq/Ne and
  Count/Rows).
- `ogar-p18-capability-table-0x0103` together with
  `odoo-capability-0x0103-consumer-tension`. Operator: are capabilities on a
  shared concept declared per concept or per consumer?
- `ogar-billable-work-entry-no-customer-edge` and the P20 work date — an OGAR
  `Class` change.
- `odoo-hr-timesheet-harvest-missing`, `odoo-analytic-model-source-unvendored`.

**Heavy (cross-repo or blocked on data):**
- `contract-deepnsm-corpus-has-no-concept` (a concept mint),
  `glove-p6b-…` (WoA rows), `glove-p10-sap-behaviour-oracle` (no trace),
  `ogit-workorder-dialect-divergence`, `odoo-storage-writer-would-persist-0x0002`,
  `q2-medcare-geo-plug-unreachable`, D-HPR-6b (operator).

## Ledger

`holds` = the re-reader's verdict. Refuted items keep the scout's text, and
the correction follows it.

| id | status | fruit | ref | item |
|---|---|---|---|---|
| `board-p14-iss-regrade` | open | low | P14 | ISS-CLASSID-READ-MODE-IS-A-SECOND-RESOLUTION-PATH: 'retire the 0x1000 keys' contradicts two rulings; basis and caller list predate D-HPR |
| `board-p17-status-cells-stale` | open | low | P17 | STATUS_BOARD cells stale vs merged PRs; D-XGP-2 shipped in #1455, not #1460 |
| `contract-board-regrade-iss-plan-status` | open | low | P14, P17 | ISSUES top entry, plan 'Where things stand', and STATUS_BOARD D-HPR-2 all carry text the 2026-10-10 no-eviction refinement superseded (plus drifted line numbers) |
| `contract-domainspec-v3-consts-are-the-bake-readers` | open | low | P24 | soa_graph DomainSpec consts still key on CLASSID_*_V3 and `project_snapshot` filters rows by `key.classid() == domain.classid` — this IS the stored-bake population W5 protects; by-ruling it stays, only a doc note is missing — **corrected:** The cited lines are accurate: soa_graph.rs:91/117/134/155 hold the V3 consts; the :259/:414 filters are exact classid matches; q2 body.rs:44/81, fma.rs:72-73/144/177, body-soa-wire:27/65 and geo/src/bso2.rs:22/148-154 read as quoted; MedCare graph_feed.rs:25 h… |
| `contract-hpr3-legacy-reader-inventory` | open | low | P1 | D-HPR-3 (W3): every caller of classid_read_mode / CLASSID_*_V3 in lance-graph classified; in-repo W3 is nearly empty, the real population is q2/MedCare |
| `contract-hpr5-legacy-keys-stay` | open | low | P2 | D-HPR-5 (W5): stored 0x1000 / pre-flip keys keep resolving via BUILTIN_READ_MODES — true; but 'a SlabDeclaration may override for its own data' only covers value_schema, never the tail |
| `contract-ogar-crate-has-no-clippy-gate` | open | low | P15 | lance-graph-ogar: CI runs `cargo test` only for the workspace-excluded crate; no clippy line exists, so the reported pre-existing clippy errors cannot be caught |
| `fleet-lance-graph-ogar-lint-fmt-gate` | open | low | P15 | lance-graph-ogar has no clippy and no fmt gate in lance-graph CI; the existing TD row is stale and q2 has already been broken by it |
| `glove-p19-cats-validity-plane` | open | low | P19 | Validity plane for optional CATS fields: Quack already ships the mechanism (BoundField.validity + Filter::sql_where), and only the CATS side is missing |
| `glove-p9-lgj-convergence-gate-is-answers-not-bytes` | open | low | P9 | D-XGP-6: lowering_convergence.rs exists and already covers count + grouped sum. The plan/board gate 'byte-equality' contradicts the file's own scope (it compares answers), and a canonical-bound case needs a dependency decision. |
| `ogar-label-mechanism` | open | low | new | Q1: OGAR has a label DTO but no class-level i18n label table; the TTL lift parses rdfs:label and drops it |
| `ogar-lo-u16-two-readings` | open | low | new | NEW: OGAR's ledger and ports.rs still define the lo u16 as the APP_PREFIX render skin (+0x0006 reserved for q2), while the 2026-10-10 ruling makes it 64k app-owned mask space and calls the render prefixes a likely fossil — OGAR carries no entry for the ruling |
| `ogar-p11-fossil-notes` | open | low | P11 | P11: no 0x1000 'fossil' note in ogar-osm/ro/dismech/loco or in ports.rs; ogar-osm::geo_identity_classid is a LIVE producer of 0x0F01_1000, not just a test literal — **corrected:** The core of P11 holds and every cited line reads as claimed: osm.rs:60-67 documents CLASSVIEW_V3_SUBSTRATE as 'the V3 marker form the codebook already uses' with no fossil wording; ogar-ro:424-428 and ogar-dismech:526-531 use 0x1000 only as a test app_prefix;… |
| `ogar-p20-work-date-attribute` | open | low | P20 | P20/Q3: no temporal-role pattern exists in OGAR; the convention on promoted classes is a typed attribute (date/datetime) — BillableWorkEntry has only `billable`, and 9 of its 12 edge targets (incl. Duration) are unminted |
| `ogar-p23-ogit-tables-ordinals` | open | low | P23 | P23: OGAR #350 OgitTables landed; its TypeId is a per-load row ordinal, not an address — TimeSheet sits ~789th of 791 with WorkOrder the last of 79 domains, so almost any tree edit shifts it |
| `ogar-p5-timesheet-alias-and-label` | open | low | P5 | P5: Stundenzettel→0x0103 already landed as a port alias; what is missing is the OGIT name `TimeSheet`, the global normalizer, and the label path |
| `ogit-minuten-unit-and-rounding` | open | low | P6 | P6 caveat from OGIT: minuten is raw integer minutes; WoA bills in 15-minute rounded-up units |
| `ogit-p23-ogittables-ordinals` | open | low | P23 | P23: OGAR #350 landed; TimeSheet gets a u16 TypeId without a vocab mint, but it is a load-local ordinal over a moving tree, not an address |
| `ogit-p5-woa-hours-row-identity` | open | low | P5 | O2/P5: the WoA hours row in OGIT is ogit.WorkOrder:TimeSheet; OGAR pins 0x0103 on TimesheetActivity, which OGIT does not model |
| `q2-medcare-p1-q2-mint-callers` | open | low | P1 | P1 (q2 part): every q2 classid_read_mode call is an offline bake MINT path on CLASSID_FMA_V3; no q2 reader calls it |
| `q2-medcare-p11-q2-geo-doc` | open | low | P11 | P11 (q2 part): q2 geo and body docs still treat classview 0x1000 as meaningful, and two of them recommend minting a new contract constant |
| `q2-medcare-p24-stored-population` | open | low | P24 | P24 revised: q2's stored FMA/OSINT bakes are mostly PRE-FLIP forms, not 0x0A01_1000; no q2 or MedCare code reads them through soa_graph or classid_read_mode |
| `contract-hpr6-no-v1-default` | open | high | P3 | D-HPR-6 (W6): unknown classid → ReadMode::DEFAULT (V1 / Full / CoarseOnly) today; making it an error touches 4 production signatures and ~8 anti-vacuity pins; guid-v3-tail is already default-on and its off-arms are untested dead code |
| `glove-p6a-duration-sap-lens-needs-unit-ruling` | open | high | P6 | Converted grade for duration, SAP half: needs a canonical unit, then a derived minutes lens in CatsBatch::bind. It does not need WoA. — **corrected:** The gap is real: SAP hours_logged→duration is Hypothesized, the two basis notes assume different units, and the SAP side does not need WoA to be graded. But three things do not survive a re-read. (1) The basis.rs citations :183-187 and :244-248 point past the… |
| `glove-p8-report-catalog-from-basis` | open | high | P8 | D-XGP-5 report Catalog: the shipped precedent exists (report boundary::Catalog + a Catalog-backed Quack Binder test). The named input type CanonicalBasis does not exist, and the cross-glove gate is blocked on a second population. |
| `ogar-billable-work-entry-no-customer-edge` | open | high | new | NEW: BillableWorkEntry has no billing-party/customer edge, but WoA's TimeSheet is booked against a mandatory Customer — the WoA glove's one mandatory FK has no canonical slot |
| `ogar-p18-capability-table-0x0103` | open | high | P18 | P18: a capability declaration on 0x0103 = one new ogar-vocab domain table (ActionDefs whose object_class ends in the literal `billable_work_entry`, expected executors = every glove, appended to domain_tables) — with the per-table gate forcing one shared capability set |
| `q2-medcare-p3-evidence` | open | high | P3 | P3 evidence: both consumers have already hit the V1 default for an unregistered classid; neither calls the lookup on such keys, so NoReadingFor would break no current caller in q2 or MedCare |
| `fleet-capability-tables-crm-and-billable` | open | heavy | P18 | OGAR has four capability tables (ocr, geo, healthcare, document); with no CRM table and no 0x0103 table, HubSPO's HOT_PLUG and the SAP/Odoo glove plugs all report NoCapabilitiesFor |
| `glove-p10-sap-behaviour-oracle` | open | heavy | P10 | SAP behaviour oracle (D-XGP-7): SIMAF holds no CATS trace; the only BAPI 'behaviour' in it is a unit-test mock. Stays blocked. |
| `glove-p18-capabilities-on-0x0103` | open | heavy | P18 | Capabilities on BILLABLE_WORK_ENTRY: an OGAR domain table would end NoCapabilitiesFor(0x0103), but the authority's own rule says to declare capabilities only once a consumer implements them, and the gloves implement none |
| `glove-p20-no-temporal-role` | open | heavy | P20 | BillableWorkEntry has no temporal role (confirmed at OGAR b4f5a36). The SAP date conversion is already done; only the authority field is missing. |
| `glove-p6b-duration-cross-glove-woa-rows` | open | heavy | P6 | Cross-glove duration anchor (SAP hours×60 == WoA TimeSheet.minuten): blocked on WoA rows and a WoA lane binder, and neither exists here |
| `odoo-xgp4-analytic-pivot` | open | heavy | P7 | D-XGP-4 Odoo analytic-line mapping + behaviour: what odoo-rs now has vs still lacks |
| `ogit-p20-work-date` | open | heavy | P20 | P20 from OGIT: no vendor-neutral work-date concept; WoA's datum is undefined in TimeSheet.ttl and defined twice elsewhere with other meanings |
| `board-0x1000-retirement-wording-contradicted` | new | low | new | Three copies of 'its retirement is the plan's P4 operator checkpoint' and other pre-refinement retire/decide wording contradict two rulings |
| `board-d-hpr-0-entry-contradicts-plan` | new | low | new | D-HPR-0 ruling entry keeps 'retire/remove' and wave texts the refinement and the plan contradict |
| `board-d-hpr-2-codex-p1-unrecorded` | new | low | new | D-HPR-2 shipped text omits the codex-P1 behaviour change: canon domains now activate reading-only |
| `board-d-hpr-4-render-prefix-note-missing` | new | low | new | STATUS_BOARD D-HPR-4 says render prefixes are documented deprecated; only the 0x1000 constants are |
| `board-d-xgp-entries-stale-opens` | new | low | new | D-XGP entries and LATEST_STATE keep OPEN lines and claims later sections closed or corrected |
| `board-dup-fossil-block-x3` | new | low | new | Same 25-line classview/fossil block verbatim in entry, plan §C.9.3 and the knowledge doc; layout ruling in 4 places |
| `board-dup-render-audit-summary` | new | low | new | Plan §C.9.4 re-lists the audit findings in different wording beside the canonical entry |
| `board-dup-second-resolution-path` | new | low | new | 'classid_read_mode is a second resolution path' stated six times with three different caller lists |
| `board-dup-woa-block-x2` | new | low | new | WoA anchors finding verbatim twice; O2 restated in five places |
| `board-epiphanies-july-0x1000-vs-v3-mandatory` | new | low | new | EPIPHANIES row E-V3-DUAL-SCHEMA-0x1000-IS-PERMANENT-1 (CURRENT) disagrees in meaning with D-HPR-0; neither cites the other |
| `board-integration-plans-and-plan-headers-stale` | new | low | new | INTEGRATION_PLANS and the cross-glove plan header contradict the rulings and the shipped code |
| `board-latest-state-pr-arc-missing` | new | low | new | LATEST_STATE / PR_ARC_INVENTORY carry none of the arc's merged PRs; three LATEST_STATE entries still say 'unmerged' |
| `board-missing-d-ids` | new | low | new | Work the plans name as next or open has no D-id or STATUS_BOARD row; plan D-id range omits D-XGP-8; D-XGP-4 gate is already satisfied |
| `board-plan-v3-w3-vs-w4` | new | low | new | Plan v3 contradicts itself (W3 '#[deprecated]' vs W4 'not #[deprecated]'; 'must go' vs 'kept'); D-HPR-3 gate is eviction-shaped |
| `board-stale-line-cites` | new | low | new | Line citations into lance-graph-ogar/src/lib.rs drifted after #1465 |
| `contract-slab-declaration-has-no-tail` | new | low | new | NEW: the slab envelope cannot declare a key tail — SlabReading has Facet96/Register128/signed-register variants only, so 'the slab wins for existing data' stops at the value slab |
| `contract-w4-doc-note-gaps` | new | low | new | NEW: W4 doc notes landed on the five CLASSID_*_V3 consts + classid_read_mode only; NodeGuid::read_mode, the DomainSpec consts, and two stale Cargo.toml feature comments still describe the legacy path as current |
| `fleet-contract-reach-matrix` | new | low | new | How a lance-graph-contract change on main reaches each sibling: immediately for most, frozen for openproject (a lock at a commit no longer on main) and a2ui (a lock behind the D-HPR/D-V3M commits) |
| `fleet-expected-executors-unplugged` | new | low | new | OGAR names expected executors for geo ('osm-soa-bake') and document ('paperless-kv') that no consumer crate in the fleet declares a HotPlug for — **corrected:** The geo half is refuted. 'osm-soa-bake' is a real crate in the sibling repo openstreetmap-website-rs (Cargo.toml:13 `name = "osm-soa-bake"`), which q2 consumes (q2 Cargo.lock:7297; test-suite.yml:58). Its src/capability.rs:43-113 sets CONSUMER = "osm-soa-bake"… |
| `fleet-glove-parity-trigger-gap` | new | low | new | glove-parity CI does not re-run when lance-graph-ogar changes, has no push or schedule trigger, and resolves OGAR at main HEAD with no lockfile |
| `fleet-guid-v3-tail-default-everywhere` | new | low | new | guid-v3-tail is a contract default and no fleet consumer disables defaults, so un-gating it (P3) changes no sibling's feature set, and q2's comment saying v3 is off is wrong |
| `fleet-hotplug-adoption-matrix` | new | low | new | Which siblings already follow the plug pattern: HotPlug in MedCare, HubSPO and tesseract; RbacPlug in MedCare, HubSPO and spear; a2ui consumes a binding; q2, odoo-rs and openproject use classid tables or PortSpec only |
| `fleet-main-main-coupling-no-cron` | new | low | new | lance-graph and OGAR depend on each other's main HEAD with no lockfile and no scheduled run; only spear and tesseract-rs detect sibling drift daily |
| `fleet-no-ci-openproject-odoo` | new | low | new | openproject-nexgen-rs and odoo-rs have no CI, so their classid pins and builds are checked nowhere (and MedCare's own CI header says it may be inert) |
| `fleet-q2-wasm-needs-nightly` | new | low | new | q2's two WASM crates need nightly (`#![feature]`, `-Z` rustflag, build-std), but the only toolchain file pins stable 1.99.0 |
| `fleet-report-ogar-no-ci` | new | low | new | lance-graph-report-ogar (workspace-excluded, path-deps OGAR) has 4 tests and no CI step |
| `fleet-sibling-plugs-skip-reading-socket` | new | low | new | All three sibling HotPlug consumers activate through `resolve_hotplug`, which returns no readings, instead of `CapabilityAuthority::activate`, which carries `read_mode_for` |
| `fleet-spear-dual-contract-source` | new | low | new | spear pulls the contract and lance-graph-rbac from two sources (git main and sibling path) with no [patch] to unify them |
| `fleet-toolchain-pin-bypass-and-prose-drift` | new | low | new | Two lance-graph jobs float on `stable` instead of the 1.99.0 pin; several sibling workflows restate the version or describe the wrong one |
| `glove-basis-grades-not-wired-to-canonical-front` | new | low | new | The authority basis (basis::*_FIELDS grades, keyed by ClassView names) and the canonical-name front (CanonicalBinder, keyed by OGIT URIs in OntologyRegistry) are disconnected |
| `glove-billable-ne-keeps-unknown` | new | low | new | The only Converted claim (billing_indicator→billable) is three-valued for '=' only: 'billable <> true' selects BILLABLE_UNKNOWN rows |
| `glove-ogit-datum-polysemous` | new | low | new | OGIT ogit.WorkOrder:datum is one property URI defined twice with different meanings, and TimeSheet reuses it with no definition of its own |
| `glove-sap-baseline-waves-untracked` | new | low | new | sap-glove-quack-v1 W1–W6 are unbuilt and have no STATUS_BOARD rows; W1's PR-D targets the superseded glove::GlovePivot |
| `glove-xgp3-gate-names-unbuilt-controls` | new | low | new | STATUS_BOARD D-XGP-3 (Shipped) carries the gate 'N1/N2/N5/N6 red without the conversion', but none of N1–N7 exists as a test |
| `odoo-analytic-behavior-classification-inputs` | new | low | new | Prefix classifier misfiles on_change_unit_amount, and the manifest lists 4 of the 8 analytic methods |
| `odoo-bundle-graph-ids-wrong` | new | low | new | ODOO_BUNDLE_ID 0x0002 / FIBO 0x0007 mix the app-prefix space into OGIT g-slots (real: ODOO 50, FIBOFND 20) |
| `odoo-nav-vendor-version-skew` | new | low | new | Vendored Odoo nav files come from two upstream versions |
| `odoo-project-task-user-aliases` | new | low | new | OdooPort has no labels for project.project / project.task / res.users / res.company, though hr.employee is aliased |
| `odoo-stale-preflip-docs` | new | low | new | odoo-rs docs still describe the pre-flip order and deleted APIs |
| `odoo-xgp4-handoff-stale` | new | low | new | D-XGP-4 handoff is superseded twice and still asks for app_prefix 0x0002 — **corrected:** The staleness is real, but the headline claim is wrong. The app_prefix=0x0002 row (:58) sits in handoff §2, and the handoff's own correction block (:10) already supersedes §1, §2, §5 and §8.1-2, so the handoff does not 'still ask' for app_prefix. What is actua… |
| `ogit-alias-name-sets-disagree` | new | low | new | Three disjoint name sets for one concept: OGIT local name, WoaPort aliases, canonical_concept promotions |
| `ogit-german-label-in-ogit` | new | low | new | A German label can be written in OGIT today only as a plain RDF language-tagged rdfs:label; OGIT has no precedent or convention for it |
| `ogit-timesheet-attribute-drift` | new | low | new | OGIT TimeSheet carries 7 of the WoA model's 14 fields; start/end times and the pause marker are missing |
| `q2-medcare-contract-cargo-doc-drift` | new | low | new | Cross-dimension observation: lance-graph-contract Cargo.toml says V2/V3 mints fall back to V1 without features; the code panics instead |
| `q2-medcare-hotplug-model` | new | low | new | Is MedCare's HOT_PLUG the model for SAP/Odoo plugs? For the capability side yes; for the reading side no |
| `q2-medcare-osint-scene-v2-tail-under-v1-registry` | new | low | new | q2 runtime OSINT scene is stored with a V2 tail under classids whose builtin reading is V1, so BUILTIN_READ_MODES answers wrongly for it |
| `contract-ogar-reading-only-partition-vs-0x0103` | new | high | new | NEW synergy with P18: lance-graph-ogar routes only the five hard-listed canon storage domains around the capability join; 0x0103 (BILLABLE_WORK_ENTRY) goes to the join and `NoCapabilitiesFor` refuses the whole SAP/Odoo plug |
| `glove-draft-has-no-range-or-group` | new | high | new | Quack Draft expresses only Eq/Ne and Count/Rows, so every canonical fold (date range, sum duration by …) is hand-built from bound Cols |
| `odoo-analytic-model-source-unvendored` | new | high | new | No analytic-line model .py in odoo-rs, so no ClassView ordinals, source pins or decorator kausal |
| `odoo-hr-timesheet-harvest-missing` | new | high | P7 | hr_timesheet employee_id/project_id/task_id are not harvested anywhere; methods keyed on a non-model subject |
| `ogit-label-readers-drop-tags` | new | high | new | ogar-from-schema reads rdfs:label and dcterms:description into EntityDecl, but drops language tags and extra labels, and into_class never carries either into Class |
| `ogit-workorder-dialect-divergence` | new | high | new | NTO/WorkOrder is written in a dialect the rest of OGIT and OGAR's reader do not use |
| `q2-medcare-cpic-local-guid` | new | high | new | q2 cpic crate keeps a parallel local NodeGuid (V1 u24 tail) and uses the classid low u16 as a kind ordinal |
| `contract-deepnsm-corpus-has-no-concept` | new | heavy | new | NEW: deepnsm-v2 `promote::key_at` is a production mint that cannot move to the plug path — its corpus has no OGAR concept, so it borrows CLASSID_OSINT_V3 for the V3 tail only |
| `odoo-capability-0x0103-consumer-tension` | new | heavy | P18 | A capability table on shared 0x0103 that names one executor refuses every other glove |
| `odoo-storage-writer-would-persist-0x0002` | new | heavy | new | The unwired od-server lance sink would persist concept\|\|0x0002 keys that classid_read_mode reads as V1 DEFAULT |
| `odoo-xgp4-mapping-ownership-split` | new | heavy | new | Where the Odoo field mapping lives: plan says odoo-rs, shipped code says lance-graph glove-parity |
| `ogit-timesheet-customer-has-no-role` | new | heavy | new | WoA books hours against a Customer (mandatory); BillableWorkEntry has no customer or billing-party edge |
| `q2-medcare-geo-plug-unreachable` | new | heavy | new | q2 geo mints under osm_node 0x0F01 but can never obtain a plug reading: the reading is gated by capability-consumer identity |
| `q2-medcare-p2-slab-declaration-cannot-override-tail` | new | heavy | P2 | P2 revised: a SlabDeclaration can override value_schema only, not the tail, so it cannot protect stored V1/V2-tail rows under a concept the plug reads as V3 |
| `board-d-hpr-5-is-status-quo` | closed | low | P2 | D-HPR-5 'Queued' describes behaviour that already holds: BUILTIN_READ_MODES still resolves and SlabDeclaration already wins on the plug path |
| `board-p13-render-classid-reversal` | closed | none | P13 | Reversing ogar_codebook::render_classid: audited and ruled out; every record agrees |
| `fleet-toolchain-sweep-1-99` | closed | none | P22 | Fleet toolchain sweep to Rust 1.99.0 is complete in all ten repos; lance 13 / lancedb 0.40 pinned where used |
| `odoo-analytic-views-vendored` | closed | none | P7 | Analytic parent views vendored + region corpus regenerated (#42, #43) |
| `odoo-toolchain-1-99` | closed | none | P22 | odoo-rs on Rust 1.99.0 (toolchain + Dockerfile) |
| `ogar-q4-no-read-mode-consumers` | closed | none | new | Q4: OGAR has zero code consumers of lance-graph's classid_read_mode / CLASSID_*_V3 / ReadMode family and never enables guid-v3-tail explicitly — nothing to migrate in W3/W6 on the OGAR side |
| `q2-medcare-p1-medcare-no-callers` | closed | none | P1 | P1 (MedCare part): MedCare has zero classid_read_mode calls; its V3 mints already pass TailVariant::V3 directly |
| `q2-medcare-p22-toolchain` | closed | none | P22 | P22 confirmed for q2 and MedCare: Rust 1.99.0 and lance 13 / lancedb 0.40; MedCare pins exact, q2 floats |
| `fleet-coderabbit-cap` | excluded | none | P16 | CodeRabbit org spending cap (1 review/hour) — external, non-code |
| `fleet-hubspo-spear-not-billable-gloves` | excluded | none | new | HubSPO-rs and spear have no billable-work concept (0x0103 / TimeSheet), so there is no fourth or fifth glove |
| `odoo-0x0002-render-prefix-pins` | excluded | none | P12 | odoo-rs pins the fossil render prefix 0x0002 in the classview bits |
| `ogit-no-other-timeentry-concept` | excluded | none | new | No OGIT time-entry concept outside NTO/WorkOrder for BILLABLE_WORK_ENTRY to converge on |
| `q2-medcare-osint-classview-literal` | excluded | none | P1 | q2 osint_classview 0x0701_1000 literal is a test; the production reader of stored OSINT V3 keys reads only the concept half |

Five MedCare-rs-internal items (private repository) are left out of this public ledger. The only MedCare fact used above: it has 0 `classid_read_mode` callers.
