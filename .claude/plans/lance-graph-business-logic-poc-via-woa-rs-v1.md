# Lance-Graph Business-Logic POC via woa-rs — v1

> **Author:** main thread (Opus 4.7), session 2026-05-21 (branch `claude/activate-lance-graph-att-k2pHI`)
> **Status:** Active (Draft)
> **Scope:** Consolidating integration plan that sequences the three consumer-plan harvests (`lance-graph-in-{woa-rs,smb-office-rs,medcare-rs}-v1`) + the `unified-bridge-consumer-migration-v1` substrate work into a single P0/P1/P2-prioritised POC roadmap, with **woa-rs as the customer-visible target** and PR-5 (XRechnung) as the POC milestone. This plan does not introduce new D-ids — it references existing ones from the four predecessor plans, locks in priorities, and names the cross-plan dependencies.
> **Path:** `.claude/plans/lance-graph-business-logic-poc-via-woa-rs-v1.md`
> **Confidence:** HIGH on the POC framing (per the four predecessor plans' session-appended refinements §8-§13 across the three consumer plans + §4 of the unified-bridge plan); MED on parallel-substrate dependencies (the convergence-v1 / RBAC / callcenter-membrane work timing is outside this plan's control); HIGH on the codegen-bucket force multiplier (per RFC v02-006 13-bucket coverage of all 660 woa routes).
> **Predecessors:**
>   - `unified-bridge-consumer-migration-v1.md` (§4.5 CAM bar codes, §4.6 read-only spine, §4.7 per-OGIT storage)
>   - `lance-graph-in-woa-rs-v1.md` (§9 hot/cold, §10 OGIT-in-sea-orm, §11 rewarding target, §12 codegen-bucket integration, §13 POC framing)
>   - `lance-graph-in-medcare-rs-v1.md` (§8 parallelbetrieb shipped + MongoDB alt cold path, §9 quickest substrate-complete target)
>   - `lance-graph-in-smb-office-rs-v1.md` (§7 empty + canonical UnifiedBridge template source)
>   - `super-domain-rbac-tenancy-v1.md` (Tier A D-SDR-1..5; the RBAC substrate the consumer plans bind to)

---

## 1 — Why this exists

Six months of lance-graph substrate work — 22 crates, VSA bundling, CAM-PQ codec, 36 thinking styles, 16-strategy planner, NARS inference, AriGraph triplet store, 10+ ontology hydrators, 5 tenant bridges, 4 substrate iron rules, ATT NLSpec activation, multi-sprint A2A pattern, parallelbetrieb reconciler, MongoDB bridge, sea-orm DTOs, OGIT TTL — has shipped **zero end-user visible outcomes**. Every architectural decision has been bet-the-substrate without customer-visible payoff demonstrating return (per `lance-graph-in-woa-rs-v1.md` §13.1).

Three consumer integration plans landed earlier in this session, each with a different role: SMB ships the canonical UnifiedBridge wiring template + XRechnung pattern (`lance-graph-in-smb-office-rs-v1.md` §7); MedCare ships the parallelbetrieb reconciler shell + parity-dashboard pattern (`lance-graph-in-medcare-rs-v1.md` §8); woa-rs absorbs both and adds the customer-visible web-UI surface (`lance-graph-in-woa-rs-v1.md` §11). The three are complementary; **only woa-rs has a paying customer in the loop** (Stefan, per the WoA glossary).

This plan locks the sequencing across the three consumer plans into one ranked roadmap where the **first POC slice (woa-rs PR-5, XRechnung visible reward) is the moment lance-graph stops being substrate-only and starts being a customer-deliverable system**. Everything after PR-5 is post-POC iteration on a proven integration.


## 2 — The POC milestone

**The first POC slice is `woa-rs` PR-5 (XRechnung visible reward).** It is the moment lance-graph stops being substrate-only and starts being a customer-deliverable system. PR-5 ships:

- One route handler in the `pdf_render` bucket (per RFC v02-006): `POST /vorgaenge/<wid>/rechnung/xrechnung`.
- One customer-visible artefact: an EN16931-conformant ZUGFeRD/Factur-X invoice (XML + embedded-PDF/A-3).
- One end-to-end existence proof of:
  - The OGIT spine (hydrated at boot per `unified-bridge-consumer-migration-v1.md` §4.3).
  - The CAM bar-code substrate (`OwlIdentity` addressing a real Customer row per §4.5).
  - The hydrators (`hydrate_zugferd` + `hydrate_zugferd_rules` + `SchematronHydrator` + `XsdHydrator`) producing a valid invoice from a real workorder.
  - The unified bridge's 4-stage authorize over the customer's tenant + role.
  - The sea-orm cold-path read + Lance hot-path projection (per `lance-graph-in-woa-rs-v1.md` §9).

This maps 1:1 to **Phase 9 of `erp_foundry_hhtl_ontology_distillation.md`** ("First Foundry-style projection: `fibo:Transaction` Object Type with end-to-end ingest path") — viewed from the consumer-side instead of the substrate-side. The two framings converge: the substrate plan calls it "first Foundry-style projection"; this POC plan calls it "first customer-visible artefact."

External validation surfaces — what makes the POC observably "working":

| Surface | What "POC working" looks like |
|---|---|
| Stefan's browser | `/vorgaenge/<wid>/rechnung/xrechnung` returns a downloadable Factur-X PDF + raw XML side-channel. |
| Stefan's accountant | The XRechnung file imports cleanly into DATEV / X-Rechnung official validation tools (`https://xeinkauf.de/xrechnung/`). The Factur-X PDF renders correctly in any conformant viewer. |
| lance-graph's substrate ledger | `.claude/board/AGENT_LOG.md` carries entries naming PR-5's "first real-load test" for each substrate primitive exercised: hydrators (`hydrate_zugferd`, `hydrate_dolce`, `hydrate_provo`, `hydrate_qudt`, `hydrate_skr03/04`), unified-bridge stages (chinese-wall + super-domain + role-group + slot redaction), planner strategies (TBD if PR-5 uses Cypher; otherwise direct sea-orm), Lance projection columns, CAM-PQ codec (TBD — XRechnung itself doesn't need similarity search, but reading-Customer-by-bar-code does), Schematron + XSD rule sets. |

## 3 — Priority ranking

Per `lance-graph-in-woa-rs-v1.md` §13.5, the cross-plan priorities are:

| Priority | Plan / phase | LOC + days | Reasoning |
|---|---|---|---|
| **P0** | `lance-graph-in-woa-rs-v1.md` Phase 0-3 + PR-5 (XRechnung visible reward) | ~7-8 days net (per §12.6 refined estimate) | First end-user-visible existence proof of the 6-month lance-graph substrate. POC milestone. |
| **P1** | `lance-graph-in-woa-rs-v1.md` PR-6 (parity dashboard) + PR-7 (RLS unification via codegen-bucket pivot) | ~3-4 days | Second + third visible-reward milestones; PR-7 alone upgrades ~317 routes via one codegen edit (per §12.4). |
| **P1** | `lance-graph-in-medcare-rs-v1.md` Phase 1-3 (D-LGMC-1 build fix → D-LGMC-4 unified-bridge constructor → D-LGMC-2/3 RLS close) | ~5-7 days | Engineering-completeness POC; complements PR-5 by showing the substrate also works for a HIPAA-regulated transcode (not just a web app). |
| **P1** | `lance-graph-in-smb-office-rs-v1.md` Phase A-B (D-LGSMB-1/2 SmbBridge upstream + D-LGSMB-3 type-param swap + Phase B TTL authoring + role groups) | ~7 days | Template-source completion; ships canonical `SmbBridge` so the unified-bridge surface stabilizes across all consumers. Unblocks D-UB-5 sister deliverable. |
| **P2** | All three plans' Phase 4-5 (Cypher / SPARQL / CAM-PQ opt-in) | ~10-18 days total | Post-POC; once PR-5 establishes the integration works, higher-leverage substrate surfaces (Cypher playground, similarity search, attention-as-table-lookup) land as additive features. |
| **P2** | `unified-bridge-consumer-migration-v1.md` Tier D D-UB-11 (cross-consumer parity test) | ~120 LOC + 4 tests | Regression gate; lands after all three consumers' Phase 1-3 closure to prevent per-consumer drift of the dictionary layer. |
| **P3** | Odoo work-steal extraction (per `b9531cf3-odoo_work_steal_distillation.md`) | ~1 weekend per Priority-1 module | Parallel stream; supplements the existing in-tree hydrators with broader ERP coverage. Most-mature lance-graph-ontology hydrators already cover the L1-L4 layered ontology per the bO-* series. Odoo extraction is a force multiplier for the L4 jurisdictional overlay (especially `l10n_de_skr03` / `l10n_de_skr04` refresh against current DATEV-maintained data) and a path to a Python adapter for the "two-version bridge" pattern. |


## 4 — The integrated PR ladder (P0 + P1 sequenced)

Drawing from `lance-graph-in-woa-rs-v1.md` §11.3 + §12.4 (codegen-bucket refinements) + the third attached distillation doc's Phase 9 framing. Each rung produces a screenshot or external validation, not just a passing test.

| # | Source plan | Lands in woa-rs as | Visible in browser as |
|---|---|---|---|
| 1 | (greenfield) | Phase 0 vendor symlinks (`vendor/lance-graph`, `vendor/ndarray`) + workspace exclude block | (build green; `cargo metadata` passes; no UI change) |
| 2 | mirror MedCare `MedcareRegistry` shape | `crates/woa-bridge/` + `crates/woa-ontology/` skeleton crates + `WoaRegistry::hydrate(...)` helper (D-WLG-3, D-WLG-4) | (build green; cargo test passes) |
| 3 | mirror smb `smb_unified_bridge` | `woa_unified_bridge(registry, actor_role, tenant)` constructor (D-UB-4 ≡ D-WLG-equiv) | `/api/v1/health` reports "ontology hydrated: WorkOrder" |
| 4 | in-tree OGIT TTL + lance-graph-ontology hydrators (per L1-L4 distillation Phase 1-7) | Boot-time hydration menu: `hydrate_dolce` (L1) + `hydrate_owltime` + `hydrate_provo` (L2; GoBD audit trail) + `hydrate_qudt` (L2; monetary unit + Stundenzahl) + `hydrate_schemaorg` (L3; commercial-web) + `hydrate_fibo_fnd` + `hydrate_fibo_be` (L3; financial primitives) + `hydrate_skr03` + `hydrate_skr04` (L4; German chart of accounts) (D-WLG-8 `lance-cache` feature) | `/api/__ontology` admin route lists hydrated G-slots + per-family entity counts |
| 5 | **HARVEST 1 (smb-office-rs):** `hydrate_zugferd` + `hydrate_zugferd_rules` + `SchematronHydrator` + `XsdHydrator` | woa-rs ZUGFeRD/Factur-X invoice generator: `POST /vorgaenge/<wid>/rechnung/xrechnung` → returns conformant XML + downloadable Factur-X PDF | **POC MILESTONE — Stefan clicks "X-Rechnung erstellen" on a workorder, downloads a valid EN16931-conformant invoice.** Stefan's accountant validates against DATEV / official `https://xeinkauf.de/xrechnung/` tools. |
| 6 | **HARVEST 2 (MedCare-rs):** `MedcareMysqlReconciler` shell + `lance_graph_callcenter::transcode::parallelbetrieb::{Reconciler, DriftEvent, DriftField, DriftKind}` trait | `WoaMysqlReconciler<CustomerFetcher>` mirroring 1:1 with sea-orm fetchers (D-WLG-15) + production wiring (D-WLG-16) + admin route `GET /api/__parity` ring-buffer endpoint (D-WLG-17, mirrors `medcare-server::routes::parity::ingest_csharp` + dashboard read) | Admin opens `/admin/parity`, sees green/red drift status per Customer row across MySQL ↔ Lance projection. |
| 7 | smb-office-rs + MedCare-rs combined (codegen pivot per RFC v02-006 + §12.4) | Per-bucket codegen template emits `state.unified_bridge.authorize(owl_id, tenant, op)?` as first line of every generated handler in `list_for_tenant` (80 routes) + `detail_for_tenant` (43 routes) + `csrf_form_post_engine_call` (194 routes) = **317 routes upgraded by one codegen edit** | Cross-tenant URL-guessing returns 404 (not 403) across the entire app. Admin sees the per-tenant scope active in the `/admin/parity` panel. |
| 8 | (opt-in, P2) `lance-graph-planner` | `POST /api/__graph` accepts Cypher / SPARQL / GQL via the planner's 16-strategy dispatcher | `/admin/graph` becomes a query playground: `MATCH (c:Customer)-[:HAS_WORKORDER]->(wo:WorkOrder) WHERE wo.status = 'offen' RETURN c.firma, wo.betreff` returns results from the Lance projection. |
| 9 | (opt-in, P2) CAM-PQ via `lance-graph-contract::cam::CamCodecContract` | `EntityStore::similar_to` over Lance + CAM-PQ codec | `/kunden/<kdnr>/similar` returns top-10 similar Customers by address + industry + recent-Vorgang-history. Sales-pipeline feature. |

Rung 1-7 = P0 + P1 (the POC + the immediate post-POC iteration). Rungs 8-9 = P2 (opt-in once the POC is stable). The 9-rung sequence completes the visible-reward arc — every rung produces a screenshot or a validation-tool result.

## 5 — L1-L4 layered ontology dependency map

Per `f6b68582-erp_foundry_hhtl_ontology_distillation.md` §1-4. Each hydrator depends on its parent G-slot's existence (via `inherits_from: Some(OGIT::<PARENT>_V1.0)`), so the boot-time hydration menu must respect the layering:

```
L1 — Upper ontology (foundational)
    hydrate_dolce  (inherits_from: None — root)
        │
        ▼
L2 — Cross-domain alignment (every L3+ depends on these)
    hydrate_owltime   ─┐
    hydrate_provo     ─┼─ all inherits_from: Some(OGIT::DOLCE_V1.0)
    hydrate_qudt      ─┤
    hydrate_skos      ─┘
        │
        ▼
L3 — Industry business ontologies
    hydrate_schemaorg                    (e-commerce / commercial-web)
    hydrate_fibo_fnd                     (FIBO Foundations)
    hydrate_fibo_be (inherits FIBO_FND)  (FIBO Business Entities)
    [TBD bO-* future]:
        XBRL GL → OWL mapper             (journal-entry interchange)
        IFRS Taxonomy → OWL mapper       (financial reporting)
        UBL 2.4 → OWL mapper             (canonical e-invoice format — parent of XRechnung + ZUGFeRD)
        ISO 20022 → OWL mapper           (financial messaging)
        │
        ▼
L4 — Jurisdictional overlay (German for woa-rs)
    hydrate_skr03 + hydrate_skr04 (+ hydrate_skr03_bau for construction)   (German chart of accounts)
    hydrate_zugferd + hydrate_zugferd_rules                                 (German B2G e-invoice — XRechnung/Factur-X)
    SchematronHydrator + XsdHydrator                                        (used by hydrate_zugferd_rules)
    [TBD bO-* future]:
        HGB / UStG term extraction                                           (~180 German legal terms)
        GoBD SHACL constraints                                               (German digital-bookkeeping audit-trail rules)
        Datev DTVF parser                                                    (Layer-5 operational format)
```

**Coverage status post-PR-407 + PR-408:** L1 + L2 + L3 (DOLCE/OWL-Time/PROV-O/QUDT/SKOS/schema.org/FIBO-FND/FIBO-BE) all shipped. L4 partial: SKR03/04 shipped, ZUGFeRD + ZUGFeRD rules shipped, Schematron + XSD hydrators shipped. Still gaps: UBL, ISO 20022, XBRL GL → OWL mappers (L3); HGB / UStG / GoBD / Datev DTVF (L4). The POC (PR-5 XRechnung) does NOT depend on the L3 XBRL/UBL/ISO 20022 mappers — `hydrate_zugferd` jumps directly from L4 to the EN16931 invoice schema via `XsdHydrator`. The unmapped L3 standards become P2 / P3 fill-in work after the POC ships.

## 6 — Cross-plan deliverable index

D-id origin lookup (no new IDs introduced by this plan; this is a re-indexing):

| D-id | Origin plan | Description | Priority per §3 |
|---|---|---|---|
| D-UB-1 | unified-bridge | Consumer-pattern doc + signature shape | P1 |
| D-UB-2 | unified-bridge | `SmbBridge` skeleton upstream | P1 (blocks D-LGSMB-3) |
| D-UB-3 | unified-bridge | `lance_cache::ontology_cache_schema()` + `LanceCacheBootStrategy` | P1 |
| D-UB-4 | unified-bridge ≡ D-WLG-3 effect | woa-rs `woa_unified_bridge` constructor | **P0** (rung 3) |
| D-UB-5 | unified-bridge ≡ D-LGSMB-3 effect | smb-bridge type-param swap | P1 |
| D-UB-6 | unified-bridge ≡ D-LGMC-4 effect | medcare-bridge `medcare_unified_bridge` constructor | P1 |
| D-UB-7 | unified-bridge ≡ D-LGMC-1 effect | Fix `ontology_dto.rs:85` lance-phase2 build | P1 (blocks all medcare work) |
| D-UB-8 | unified-bridge ≡ D-LGMC-2 effect | medcare RLS for Treatment/Visit/VitalSign (fail-OPEN close) | P1 (safety-critical) |
| D-UB-9 | unified-bridge ≡ D-LGMC-5 effect | medcare `MulThresholdProfile::MEDICAL` | P1 |
| D-UB-10 | unified-bridge ≡ D-LGMC-6 effect | medcare `ontology_context_id` RLS axis | P1 |
| D-UB-11 | unified-bridge | Cross-consumer parity test | P2 |
| D-UB-12, 13, 14 | unified-bridge §4.6 | Read-only registry surface lock + proposal_sha256 idempotency test + versioned G-slot migration smoke | P1 |
| D-WLG-1..2 | woa-rs | Phase 0 vendor symlinks + workspace deps | **P0** (rungs 1) |
| D-WLG-3..4 | woa-rs | Phase 1 woa-bridge + woa-ontology crates | **P0** (rung 2) |
| D-WLG-5..7 | woa-rs | Phase 2 route-handler integration via codegen (Mandant↔TenantId mapping + actor_role mapping) | **P0** (rungs 3+7) |
| D-WLG-8..10 | woa-rs | Phase 3 lance-cache + Lance projection + RLS | **P0** (rung 4) + P1 (rung 7) |
| D-WLG-11..14 | woa-rs | Phase 4-5 Cypher + CAM-PQ | P2 (rungs 8-9) |
| D-WLG-15..17 | woa-rs §10.5 | WoaMysqlReconciler + production wiring + drift dashboard | P1 (rung 6) |
| D-LGMC-1..11 | medcare-rs | All Phase 1-5 (build fix + RLS + constructor + MUL + parity endpoints) | P1 |
| D-LGMC-15..21 | medcare-rs §8.4 | Round-2 reconciler expansion + production wiring + persistent sink | P1 |
| D-LGMC-22..26 | medcare-rs §8.5 | MongoDB alt cold path propagation from smb | P2 |
| D-LGSMB-1..6 | smb-office-rs | Phase A-C: SmbBridge upstream + role groups + tenant-type consolidation | P1 |
| D-LGSMB-7..9 | smb-office-rs | Phase D-E Cypher + CAM-PQ | P2 |


## 7 — Parallel substrate dependencies

The POC P0 + P1 work does NOT block on (nor is blocked by) the workspace's active substrate plans, but coordination points matter:

| Active substrate plan | POC interaction |
|---|---|
| `cognitive-substrate-convergence-v1` (sprint-12 Wave G, PR #390 + sprint-13 prep) | Substrate-only; provides CausalEdge64 v2 + QualiaI4 + WitnessCorpus. The POC consumes the SHIPPED parts (DOLCE/PROV-O/QUDT/SKR hydrators that landed via PR #383-389 + PR #407-408). No blocker. |
| `super-domain-rbac-tenancy-v1` (Tier A merged PR #363; D-SDR-3/4/5 unpushed) | The 4-stage `UnifiedBridge::authorize()` the POC consumes is in this plan. Tier A merged unblocks the POC; the 3 unpushed follow-on commits are needed for the full role-group + redaction-mask surface (PR-7 RLS unification). **Coordination point.** |
| `callcenter-membrane-v1` (DM-2/4/6 in PR; `claude/supabase-subscriber-wire-up`) | Provides the realtime Phoenix push + LanceVersionWatcher. POC doesn't strictly need it but PR-6 parity dashboard benefits from live update vs polling. P2 dependency. |
| `lance-graph-ontology-v5` (15 deliverables post-merge follow-ons) | Provides the post-PR-355 cleanup. The POC uses the already-shipped `OntologyRegistry` + `NamespaceBridge` + `MappingProposal`; the V5 follow-ons are mostly internal hygiene. No blocker. |
| `2026-05-06-splat-osint-ingestion-v1` (PR 1+2 in flight) | Orthogonal — splat ingestion is the OSINT super-domain story, not the WorkOrderBilling super-domain. No interaction with the POC. |
| `causaledge64-mailbox-rename-soa-v1` (specs SHIPPED PR #372; impl QUEUED) | Substrate-only; CausalEdge64 v2 layout reclaim is pre-existing dependency. POC consumes via the shipped `causal-edge` crate. No blocker. |

The single non-trivial coordination is **D-SDR-3/4/5 (super-domain Tier A follow-on commits, currently unpushed)** — those ship the full `RoleGroup` + `FieldRedactionMask` + `SuperDomainEntry` table that PR-7 RLS unification depends on. If those don't land before PR-7, the codegen edit at PR-7 substitutes placeholder predicates that pass tests but don't enforce the full per-slot redaction mask. Acceptable for POC; needs follow-up.

## 8 — Risk register

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| RFC v02-006 route-codegen pipeline isn't ship-quality | MED | HIGH (PR-7 RLS unification depends on bucket-template edits) | Phase 1 = build the codegen first if RFC v02-006 status is still DRAFT at session start. The classifier is 100%-coverage per the RFC; the emitter side is the gap. |
| Lance-cache feature ships untested at scale | LOW | MED | D-WLG-8 lands with idempotency + restart-recovery + concurrent-writer ordering tests per §3 Phase 3 spec. Acceptable risk for POC; production hardening is P2 work. |
| ZUGFeRD/Factur-X validation tools reject the generated invoice | LOW | HIGH (POC visible-reward fails) | PR-5 acceptance test: bytewise diff against a Mustang-generated reference invoice for the same workorder. The lance-graph-ontology `hydrate_zugferd_rules` Schematron checks are pre-flight; external validation is the ground truth. |
| `OntologyRegistry` mutation surface is reachable from consumer crates (violating §4.6 read-only spine) | MED | HIGH (cache invalidation cascades; spine drifts under ad-hoc edits per §4.6 failure mode) | D-UB-12 locks the read-only invariant before PR-5 ships. ~30 LOC + 2 tests asserting consumer crates can't call mutate methods. |
| Per-OGIT CAM storage (§4.7) violated by global codebook table | LOW | HIGH (per-family bar codes shift; consumer caches go stale) | §4.7 codified as the default for any new CAM-shaped artifact. Code review surfaces violations; D-UB-14 versioned G-slot migration smoke test catches cross-G-slot contamination. |
| Stefan's MySQL data exposes a sea-orm migration drift | MED | MED | DualSink-Pivot 2026-05-15 writer-parity discipline (Python + Rust both write MySQL; reconciler witnesses). PR-6 WoaMysqlReconciler is the regression sensor for this. |
| ndarray::simd CPU-feature mismatch on Stefan's deployment hardware | LOW | LOW | ndarray's `simd_caps()` runtime detect + scalar fallback handles any CPU. Per `lance-graph/CLAUDE.md` §19.7. |
| Build cache exhaustion on CI (GHA "no space left on device" we hit earlier this session) | MED | LOW (the failed build, not a feature blocker) | CI workflow adds `df-cleanup` step. Out-of-scope for the POC plan but ops should track. |
| codex P1 review finds a v1-accessor-under-v2-feature mistake per `I-LEGACY-API-FEATURE-GATED` | MED | LOW (resolvable before merge) | Pre-merge codex review is the canonical gate. PR-5..7 all expected to ship through it. |

## 9 — Success criteria

The POC is considered successful when **all four observable surfaces are green simultaneously** for at least one calendar day after PR-5 ships:

1. **Stefan's browser** — `/vorgaenge/<wid>/rechnung/xrechnung` returns 200 + a Factur-X PDF for at least 5 distinct workorders across at least 2 distinct tenants.
2. **External validation** — the generated XRechnung XML passes the official `https://xeinkauf.de/xrechnung/` validation tool with zero errors and zero warnings.
3. **`/admin/parity` dashboard** — PR-6 ships green status (Match) for ≥ 95% of sampled Customer rows over a 24-hour window; the < 5% drift gets investigated and the count drops monotonically.
4. **`.claude/board/AGENT_LOG.md` substrate ledger** — every substrate primitive exercised by PR-5 (per the §2 inventory) has a corresponding "first real-load test passed" entry written by the meta-agent after the smoke test concludes.

Failure modes that DON'T count as POC failures (acceptable in a v1 POC; queued for P2):
- A specific Cypher query in the playground returns the wrong row count (Phase 4 opt-in, not POC scope).
- CAM-PQ similarity returns a customer-affinity ranking Stefan disputes (Phase 5 opt-in, not POC scope).
- One of the 660 routes outside the rung-7 codegen-bucket edit doesn't authorize correctly (pre-existing; PR-7's bucket edit is incremental).

## 10 — Open questions

1. **RFC v02-006 codegen pipeline readiness** — does the per-bucket emitter (`handler_kinds/*.rs`) exist as code today, or is it still RFC-stage? Phase 1 of this plan assumes the codegen is operable; if it's not, the first PR builds it. Probably worth a focused subagent read of `crates/codegen/` (if it exists) before scheduling PR-1.
2. **Stefan's deployment topology** — does Stefan run woa-rs on Railway (per `Cargo.toml` `repository = ...woa-rs`) or on his localhost? Affects the PR-5 visible-reward smoke test plan: if Railway, the demo is a URL Stefan opens; if localhost, the demo is a `cargo run -p woa-rs` we walk through with him.
3. **D-SDR-3/4/5 follow-on PR timing** — those commits exist locally on the lance-graph workspace per the substrate plan's status note but are unpushed. PR-7 RLS unification depends on them. Push when?
4. **Odoo work-steal scope for the POC window** — the third attached doc (Odoo distillation) lays out a parallel extraction stream. SKR03/04 are already shipped via `hydrate_skr03` / `hydrate_skr04` so the headline Phase O4 win is already realised. Is the Odoo `account.move` ↔ `fibo:Transaction` mapping (Phase O3) worth pulling forward to reinforce PR-5's `fibo:Transaction` projection? Probably P3 — defer.
5. **Foundry-style typed-object surface** — the third doc's §"Foundry-side semantic surface" maps Object/Link/Action/Function types to OWL/RDF substrate. PR-5 exercises Object Type (`fibo:Transaction`) + Link Types (the SPO triples wo→customer→country); PR-8 (Cypher playground) is closer to Function Type territory. Is exposing the typed-object surface as a first-class REST API in scope for the POC (e.g., `GET /api/object/transaction/<id>` returning the typed-object JSON view), or is it post-POC iteration? Lean toward post-POC.
6. **HGB / UStG / GoBD term extraction** — the third doc names these as Phase 6-7 L4 work. The POC doesn't need them at PR-5 (XRechnung's compliance is via Schematron + EN16931, not HGB term resolution). Push to P2 / P3.

## 11 — Status

- **Architecture:** Working. Substrate + 4 consumer plans + 3 distillation docs all converge on the POC framing.
- **POC milestone (PR-5):** Not started. Critical path: Phase 0 → Phase 1 → Phase 2 (rung 3 `woa_unified_bridge`) → Phase 3 hydration menu (rung 4) → Phase 5 rung (XRechnung).
- **P0 estimated effort:** ~7-8 days net per `lance-graph-in-woa-rs-v1.md` §12.6.
- **P1 estimated effort:** ~15-20 days net across woa-rs PR-6/7 + medcare Phase 1-3 + smb Phase A-B.
- **P2 estimated effort:** ~10-18 days across all three plans' opt-in surfaces.

**Confidence:** HIGH on the POC framing; HIGH on the substrate readiness (every primitive the POC depends on has shipped); MED on the codegen pipeline readiness (RFC v02-006 is DRAFT; emitter side is the gap); MED on coordination with the unpushed D-SDR-3/4/5 follow-on PR.

## 12 — References

### Source plans (in this workspace)
- `unified-bridge-consumer-migration-v1.md` — substrate spec + DTO mapper + CAM bar codes + read-only spine + per-OGIT storage
- `lance-graph-in-woa-rs-v1.md` — woa-rs consumer plan (with §9-§13 session refinements that gave this POC plan its framing)
- `lance-graph-in-smb-office-rs-v1.md` — smb-office-rs as template source
- `lance-graph-in-medcare-rs-v1.md` — MedCare-rs as substrate-complete target + MongoDB alt cold path
- `super-domain-rbac-tenancy-v1.md` — RBAC + tenancy substrate

### Attached distillation docs (user-supplied 2026-05-21)
- `d7c12d03-prbO1dolcehydrator_1.md` — PR-bO-1 DOLCE+DUL hydrator spec (now SHIPPED via PR #407)
- `b9531cf3-odoo_work_steal_distillation.md` — Odoo as parallel extraction source for ERP semantics + l10n_de SKR03/04 chart refresh
- `f6b68582-erp_foundry_hhtl_ontology_distillation.md` — Master L1-L4 layered ontology distillation; this POC plan implements its Phase 9 ("First Foundry-style projection: `fibo:Transaction`") as the woa-rs PR-5 milestone

### External validation targets
- `https://xeinkauf.de/xrechnung/` — official German XRechnung validator (POC external-validation surface)
- `https://www.ferd-net.de/zugferd/` — ZUGFeRD/Factur-X reference
- `https://docs.oasis-open.org/ubl/UBL-2.4.html` — UBL 2.4 (parent format for ZUGFeRD/XRechnung)
- Mustang library (`https://www.mustangproject.org/`) — reference implementation for bytewise validation of PR-5's generated invoices

## 13 — One-line summary

> The first POC slice is woa-rs PR-5 (XRechnung visible reward) — the moment 6 months of lance-graph substrate work produces its first customer-deliverable artefact. P0 effort: ~7-8 days. P1 closes immediate post-POC iteration (parity dashboard + RLS unification via codegen-bucket pivot + MedCare/SMB cross-consumer harvest). P2 lights up the opt-in Cypher / similarity / MongoDB-alt-cold-path surfaces. The L1-L4 layered ontology dependency map (per the third attached distillation doc) is post-PR-407 / 408 already covered through L4 for the German-jurisdictional path the POC exercises; gaps (UBL, ISO 20022, XBRL GL, HGB/UStG/GoBD term extraction, Odoo work-steal) are P2 / P3 fill-in work AFTER the POC ships.

---

## 14 — Odoo OWL glue: parallel substrate stream feeding the cognitive models (2026-05-21, same session)

User strategic reframe: **the Odoo-harvested OWL glue isn't a P3 nice-to-have — it's immediately important because the cognitive models need to be built AROUND it. In the end it's the most rewarding outcome once wired into lance-graph.** §3 priority ranking put Odoo at P3 (parallel stream); §14 elevates it to **parallel substrate stream feeding the cognitive models** — running concurrent with the POC, not after.

### 14.1 The structural claim

The cognitive substrate (thinking-engine, NARS inference, AriGraph triplet store, BindSpace SoA, CausalEdge64, MUL gate, 16-strategy planner) is only as semantically grounded as the OWL ontology it reasons over. Bar-code substrate without semantic anchor is just identity fingerprints addressing nothing meaningful. The L1-L4 layered hydrators shipped post-PR-407 + PR-408 give us the **formal ontology spine** (DOLCE upper + OWL-Time + PROV-O + QUDT + SKOS + FIBO-FND + FIBO-BE + schema.org + SKR03/04 + ZUGFeRD); what they DON'T give us is the **operational ERP vocabulary** that the cognitive models need to reason about Partners, Accounts, Invoices, Stock Moves, Manufacturing BOMs, HR Contracts, Payslips, Projects, Tasks, Mail Threads.

Odoo has 20+ years of that operational vocabulary formalized as Python models + XML data files. Re-modelling it from FIBO/UBL/XBRL-GL alone is years of work; reading Odoo's source and emitting aligned OWL/SHACL is weeks (per `b9531cf3-odoo_work_steal_distillation.md` §"Premise"). **The Odoo work-steal is the cheapest path to a comprehensive OWL substrate for the cognitive models.**

### 14.2 What "OWL glue" means concretely

Per the third paragraph of the Odoo distillation doc and its §"Naming convention" + §"Concrete example" — the "glue" is the **alignment TTL** that ties Odoo URIs to upstream ontology classes:

```turtle
@prefix odoo: <https://ada.world/onto/odoo#> .
@prefix fibo: <https://spec.edmcouncil.org/fibo/ontology/...> .
@prefix vcard: <http://www.w3.org/2006/vcard/ns#> .

odoo:res.partner  owl:equivalentClass  vcard:Kind .
odoo:res.partner.vat  owl:equivalentProperty  fibo:hasTaxIdentifier .
odoo:res.partner.Company  owl:equivalentClass  fibo:LegalEntity .

odoo:account.move  owl:equivalentClass  gl-cor:entryHeader, fibo:Transaction .
odoo:account.account  owl:equivalentClass  skr:Konto, fibo:Account .
odoo:account.tax  owl:equivalentClass  de:Steuerschlüssel .

odoo:product.template  owl:equivalentClass  schema:Product .
odoo:uom.uom  owl:equivalentClass  qudt:Unit .
```

The glue is what makes a row of Odoo data reasonable-about as a `fibo:LegalEntity` + `vcard:Kind` simultaneously. The cognitive shader's 16-bit DOLCE slot (high-byte upper category + low-byte DnS role) classifies via the upper ontology; the OWL glue is what lets the slot classifier *know* that a `res.partner` row IS a `fibo:LegalEntity` (Endurant + Agent) and not, say, a `schema:Event` (Perdurant + Action).

### 14.3 Why building cognitive models around it matters

The cognitive shader doesn't reason about strings ("res.partner"); it reasons about typed semantic objects. **The OWL glue is what types them.** Without it:

| Cognitive layer | What it does WITHOUT OWL glue | What it does WITH OWL glue |
|---|---|---|
| **AriGraph triplet store** | Stores triples like `<row_42, has_field_named, "name">` — opaque strings | Stores triples like `<row_42:fibo:LegalEntity, foaf:name, "Stefan IT GmbH">` — typed, reasoner-friendly |
| **NARS inference** | Operates on free-floating term identifiers; no semantic prior | Operates on FIBO-typed entities; can dispatch by `fibo:LegalEntity` vs `fibo:Counterparty` vs `vcard:Individual` |
| **MUL gate (Dunning-Kruger + trust)** | Calibrates uncertainty against undifferentiated "row types" | Calibrates differently per super-domain (Healthcare → conservative `MulThresholdProfile::MEDICAL`; WorkOrderBilling → permissive default) — the FIBO-shaped row IS the dispatch key |
| **16-strategy planner** | Selects strategy by query-shape heuristics | Selects strategy by query-over-FIBO-shape: a `MATCH (c:fibo:LegalEntity)` invokes different cost model than `MATCH (e:fibo:Transaction)`; the cognitive policy reads OWL types |
| **Cognitive shader 16-bit DOLCE slot classifier** | Returns slot 0xFFFF (undefined) for every row | Returns concrete (upper category, DnS role) for every row by walking `rdfs:subClassOf*` to the DOLCE root |
| **CausalEdge64 v2 (per `cognitive-substrate-convergence-v1.md`)** | Edges carry causal weights but no semantic identity | Edges carry causal weights AND a 16-bit DOLCE slot of source + target so SpoWitness chains can reason FIBO-typed |

This is what "build cognitive models around it" means structurally. The cognitive models are *parameterised by* the OWL ontology; they read OWL types at decision points. The Odoo glue provides the operational vocabulary those decisions key against.

### 14.4 The wiring path — Odoo extraction → lance-graph cognitive substrate

The path from Odoo's Python source to lance-graph's cognitive runtime:

```
Odoo source tree           Odoo extractor          Per-module TTL          Alignment TTL          hydrate_odoo_*           OntologyRegistry         Cognitive substrate
(Python + XML + CSV)       (Rust, crates/         (one per Odoo            (hand-curated +        per-module functions    (CAM-addressable        (thinking-engine,
                            odoo-extract/)         module, e.g.             LLM-draft-then-        following the           bar codes for          NARS, MUL, planner,
                            (~1500 LOC)            odoo/base.ttl,           reviewed; ~100s of     established bO-*        every Odoo class +     CausalEdge64,
                                                   odoo/account.ttl,        axioms each)           pattern (~50 LOC        every alignment        AriGraph, shader)
                                                   odoo/l10n_de_skr04.ttl)                         per hydrator)           axiom)                  consume OWL types
                                                                                                                                                    natively at every
                                                                                                                                                    decision point
        │                          │                       │                       │                       │                       │                       │
        ▼                          ▼                       ▼                       ▼                       ▼                       ▼                       ▼
    rustpython-parser         AST walker per          One TTL per             owl:equivalentClass     OwlHydrator             per-family codebook    Reasoner-friendly
    + quick-xml + csv         models/*.py +           module + one CSV/       owl:equivalentProperty  with inherits_from:     storage (per-OGIT      Object Type + Link
    parse the source          XML walker per          XML-derived TTL         + per-domain            Some(OGIT::DOLCE_       per §4.7) + Lance      Type + Action Type
    tree without              data/*.{xml,csv}        per data file           alignment files         V1.0); inherits-        cache persistence      + Function Type
    running Odoo                                                              (odoo→fibo,             from chain                                     surface
                                                                              odoo→ubl, odoo→skr)     propagates DOLCE
                                                                                                      categories
```

### 14.5 Deliverables — Odoo OWL glue as parallel substrate stream

Re-ranking the Odoo work-steal from `b9531cf3-odoo_work_steal_distillation.md` §"Phased roadmap aligned with the L1-L4 distillation" against the POC priorities:

| D-id (NEW) | From Odoo distillation Phase | Priority | Description | Effort |
|---|---|---|---|---|
| **D-ODOO-1** | O0 (bootstrap extractor) | **P1** (parallel to POC P0) | `crates/odoo-extract/` skeleton: `manifest.rs` (parse __manifest__.py) + `python_ast.rs` (rustpython-parser AST walker over models/*.py) + `emit_owl.rs` (sophia TTL emitter). Round-1 = `base` module only. | ~1 weekend (1500 LOC) |
| **D-ODOO-2** | O1 (extract `base`) | **P1** (parallel to POC P0) | Run D-ODOO-1 against `base` module → `data/ontologies/odoo/base.ttl`. ~80 fields across `res.partner` / `res.users` / `res.company` / `res.country` / `res.currency` / `res.bank` / `res.lang`. | ~1 evening |
| **D-ODOO-3** | O2 (hand-curate alignment for `base`) | **P1** (parallel to POC P0) | `data/ontologies/odoo/alignment/odoo-to-fibo.ttl` + `odoo-to-vcard.ttl` + `odoo-to-foaf.ttl`. ~100 alignment axioms. LLM-draft-then-reviewed acceptable. | ~1 evening |
| **D-ODOO-4** | (new) | **P1** (parallel to POC P0) | `lance-graph-ontology::hydrators::odoo::hydrate_odoo_base` following the bO-* pattern; `inherits_from: Some(OGIT::DOLCE_V1.0)`; registers edge whitelist; lands at `OGIT::ODOO_BASE_V1`. ~50 LOC + 4 tests. | ~1 day |
| **D-ODOO-5** | O3 (extract `account`) | **P1** (parallel to POC P1) | `odoo/account.ttl` + `alignment/odoo-to-fibo-gl.ttl`. `account.move` ↔ `gl-cor:entryHeader` + `fibo:Transaction` (the dual-nature mapping per §"Important" in the doc). `account.move.line` ↔ `gl-cor:entryDetail` + `fibo:JournalEntryLine`. | ~2 evenings + ~1 day for hydrator |
| **D-ODOO-6** | O4 (extract `l10n_de_skr03` + `l10n_de_skr04`) | **P2** (substrate already covered by `hydrate_skr03` / `hydrate_skr04` from in-tree DATEV CSV) | Refresh path: pull Odoo's `account.account.template.csv` to confirm the in-tree SKR03/04 hasn't drifted. Diff against `data/skr/SKR0[34].csv`. If drift detected, regenerate. | ~1 evening (mostly a diff job) |
| **D-ODOO-7** | (new — for the POC) | **P1** (gated on POC PR-5) | `data/ontologies/odoo/alignment/odoo-to-zugferd.ttl` — align `account.move` (move_type=out_invoice) to ZUGFeRD/Factur-X invoice shape. The PR-5 generator can then attribute the generated XRechnung to the Odoo-typed source row. | ~1 day |
| **D-ODOO-8** | O9 (extract product / sale / purchase / stock) | **P2** (post-POC; broader ERP coverage) | Five more per-module TTLs + alignment files. ~1 weekend per module batch. | ~1 weekend |
| **D-ODOO-9** | O7 (Python adapter for live ingest) | **P2** (post-POC; "two-version bridge" pattern) | `odoo-ada-adapter` package: `as_rdf(record) → rdflib.Graph` for export. Allows live Odoo deployments to feed RDF directly into the lance-graph cognitive substrate. | ~3 evenings |
| **D-ODOO-10** | O8 (end-to-end demo) | **P2** (post-POC; the rewarding end-state) | First end-to-end demo: live Odoo instance → adapter → shared ontology → Rust cognitive cascade. Cognitive shader reasons about a real Odoo `res.partner` row as `fibo:LegalEntity + vcard:Kind`. | ~1 weekend |

### 14.6 Revised §3 priority ranking — Odoo elevated to P1

Updated priority table (revising the §3 row that had Odoo at P3):

| Priority | Plan / phase | LOC + days | Reasoning |
|---|---|---|---|
| **P0** | woa-rs Phase 0-3 + PR-5 (XRechnung) | ~7-8 days | POC milestone. First customer-visible artefact. |
| **P1** | woa-rs PR-6/7 + MedCare Phase 1-3 + SMB Phase A-B | ~15-20 days | Immediate post-POC iteration + cross-consumer harvest. |
| **P1** (RAISED FROM P3) | **Odoo OWL glue D-ODOO-1 through D-ODOO-5 + D-ODOO-7** (~5 deliverables + alignment with PR-5) | **~1 weekend + ~3 days per deliverable = ~10-12 days net** | **Parallel substrate stream feeding the cognitive models. NOT blocking on POC completion — runs concurrently.** D-ODOO-7 lands gated on PR-5 to attribute generated XRechnung to Odoo-typed source rows. |
| **P2** | All three consumer plans' Phase 4-5 (Cypher / SPARQL / CAM-PQ opt-in) | ~10-18 days | Post-POC opt-in features. |
| **P2** | unified-bridge D-UB-11 cross-consumer parity test | ~120 LOC + 4 tests | Regression gate. |
| **P2** | Odoo D-ODOO-6, D-ODOO-8, D-ODOO-9, D-ODOO-10 | ~3 weekends | Broader Odoo coverage + Python adapter + end-to-end demo. |

### 14.7 Why this is the most rewarding end-state

Once D-ODOO-10 lands, the lance-graph cognitive substrate reasons about ERP entities natively. Concretely:

- **A Cypher query** like `MATCH (c:fibo:LegalEntity)-[:fibo:hasAccount]->(a:fibo:Account) WHERE c.country = 'DE' AND a.skr_konto STARTS WITH '8'` runs across the Lance projection of Odoo-extracted ERP data + DATEV SKR04 chart + FIBO foundations. Three ontology stacks, one query.
- **The NARS inference engine** dispatches different evidence-update rules per OWL type: a contradiction between two `fibo:Counterparty` rows triggers identity-resolution; a contradiction between a `fibo:Transaction` and its `fibo:Account` triggers double-entry validation.
- **The MUL gate** picks `MulThresholdProfile::FINANCIAL` for FIBO-typed reasoning (per the upcoming `super-domain-rbac-tenancy-v1.md` extension), `::MEDICAL` for Healthcare, `::DEFAULT` elsewhere — keyed on the OWL super-domain.
- **The cognitive shader's 16-bit DOLCE slot classifier** has every Odoo entity pre-classified at hydration time (per §4.4's O(1) inheritance from family buckets); the cascade activates with `slot[high_byte=Endurant.Agent, low_byte=DnS.LegalActor]` for every `res.partner` row without runtime classification cost.
- **The Foundry-style typed-object surface** (per `f6b68582-erp_foundry_hhtl_ontology_distillation.md` §"Foundry-style semantic surface") becomes navigable across Odoo's full operational vocabulary: every `res.partner` is an Object Type with typed Link Types (`fibo:hasAccount`, `fibo:hasCounterparty`), Action Types (`BookJournalEntry`, `ReconcileBankStatement`), Function Types (`computeVATPosition`, `runningBalance`).

The reward isn't "wired correctly to one app" (that's PR-5). The reward is **the cognitive substrate becomes a Palantir-Foundry-class semantic-reasoning system grounded in real ERP semantics**. PR-5 is the proof the wiring works on one route; D-ODOO-10 is the proof the wiring works on the entire ERP semantic surface.

### 14.8 What this DOES NOT change

- woa-rs PR-5 stays the POC milestone (§2). Odoo work-steal does not displace the customer-visible reward arc.
- The L1-L4 layered ontology dependency map (§5) stays. Odoo extraction LANDS on top of L1-L4 via `inherits_from: Some(OGIT::DOLCE_V1.0)`; it doesn't replace any L1-L4 hydrator.
- The §4.7 per-OGIT-G-slot storage rule applies: every Odoo per-module TTL lands at its own `OGIT::ODOO_<MODULE>_V1` slot. `odoo:base` ≠ `odoo:account` ≠ `odoo:l10n_de_skr04` at the CAM substrate.
- The §4.6 read-only spine governance applies: Odoo extraction emits `MappingProposal` streams via `OntologyRegistry::append_proposals` (per `b9531cf3-odoo_work_steal_distillation.md` §"Extraction methodology" Path A static path); no consumer crate mutates the spine.

### 14.9 One-line summary

> The Odoo work-steal is the parallel substrate stream feeding the cognitive models — elevated from P3 (parallel after POC) to **P1 (parallel concurrent with POC)** because the cognitive shader, NARS engine, MUL gate, 16-strategy planner, and CausalEdge64 substrate all reason BETTER per OWL type, and Odoo is the cheapest path to a comprehensive OWL substrate covering 20+ years of ERP semantics. D-ODOO-1 through D-ODOO-5 + D-ODOO-7 ship in parallel with woa-rs PR-1..PR-5. D-ODOO-10 (live Odoo → adapter → cognitive cascade end-to-end) is the rewarding end-state — the proof the substrate reasons about the entire ERP semantic surface, not just one route.

