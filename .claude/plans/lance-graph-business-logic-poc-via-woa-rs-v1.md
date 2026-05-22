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


---

## 15 — Odoo ↔ Palantir Foundry convergence + lessons for OGIT/OWL/DOLCE + WoA work-steal map (2026-05-21, same session)

User request: analyse how Odoo OWL and Palantir Foundry ontology converge; what we can learn for the long run with OGIT/OWL/DOLCE; what WoA could wire / repurpose / work-steal / reinvent from Odoo and smb-office-rs.

### 15.1 Where Odoo OWL and Palantir Foundry converge

The two systems are structurally homologous at the **typed-object semantic surface** despite radically different runtime models. The convergence is the proof OGIT can subsume both: a single OWL substrate addresses the same conceptual entities they both manipulate, with their respective surfaces becoming projections over it.

| Convergence axis | Odoo (Python ERP) | Palantir Foundry | Shared substrate (OGIT/OWL/DOLCE) |
|---|---|---|---|
| **Entity typing** | Each `models.Model` subclass with `_name = 'res.partner'` defines a class | Each "Object Type" with declared properties + primary key | `owl:Class` (declared at hydration time) + `OwlIdentity` bar code (CAM addressing) |
| **Field typing** | `fields.Char` / `fields.Many2one` / `fields.Selection` etc. (datatype + relation discriminator) | DatatypeProperty + ObjectProperty distinction; primary-key declaration | `owl:DatatypeProperty` (with `rdfs:range` to xsd:*) vs `owl:ObjectProperty` (with `rdfs:domain`/`rdfs:range` to OWL classes) |
| **Inheritance** | `_inherit = 'res.partner'` (dynamic Python class composition) | Foundry Object Type inheritance + interface-like extension | `rdfs:subClassOf` chain + `inherits_from` G-slot DAG (per §4.4 O(1) inheritance) |
| **Relationships** | `Many2one` / `One2many` / `Many2many` field declarations | Link Type with declared cardinality + inverse | `owl:ObjectProperty` + `owl:inverseOf` + (optional) `owl:FunctionalProperty` cardinality marker |
| **Constraints** | `@api.constrains('field')` (Python validation) + `_sql_constraints` (DB) | SHACL pre/post on Action Types | SHACL `sh:NodeShape` + `sh:property` declarations alongside the OWL T-Box |
| **Computed/derived properties** | `fields.X(compute='_compute_foo')` + `@api.depends(...)` dependency graph | Function Types (named SPARQL bound to function URI) | SHACL `sh:rule` + SPIN/SHACL Functions; or named SPARQL projections published as `owl:DatatypeProperty` |
| **Provenance / audit** | `mail.thread` mixin tracks who-changed-what-when on every record | Foundry Audit Trail attached to every Object/Action | PROV-O (already shipped via `hydrate_provo`); per-row `prov:wasAttributedTo` + `prov:wasGeneratedBy` + `prov:wasDerivedFrom` |
| **Localization / i18n** | `_description` + `string="..."` on fields + `ir.translation` table | Foundry localized labels per Object Type | `rdfs:label@<lang-tag>` + `rdfs:comment@<lang-tag>` (German + English baseline via existing hydrators) |
| **Selection / enum types** | `fields.Selection([('person', 'Individual'), ...])` | Enum types attached to Foundry properties | SKOS Concept Scheme (`hydrate_skos` already ships); one Concept per selection option |
| **Action / workflow** | Odoo workflow engine + button handlers | Action Types with SHACL pre/post + SPARQL UPDATE templates | SHACL `sh:rule` + `dul:Plan` (DOLCE+DnS DnS Plan classification) — already preserves Action semantics in the L1 substrate |
| **Source-of-truth identity** | `res.partner.id` (PostgreSQL primary key) | Foundry Object Locator (system-generated stable URI) | `OwlIdentity` u16 bar code (per §4.5 CAM substrate) + OGIT URI alias for human-readable reference |
| **Reference data (A-Box)** | `data/*.xml` + `data/*.csv` loaded at module install | Foundry Reference Data Sets | `MappingProposal` stream into `OntologyRegistry`; Lance dataset persistence under `lance-cache` feature |
| **Per-tenant scope** | Multi-company sharing semantics + `res.company` discriminator | Foundry project scoping + access control | `NamespaceBridge.g_lock()` + `UnifiedBridge.authorize()` 4-stage flow (per super-domain-rbac-tenancy-v1.md §3.9) |

**The convergence is the proof.** When two systems with proprietary divergent runtimes (Odoo Python+PostgreSQL vs Foundry proprietary Spark-shaped backend) both reduce to the same OWL/SHACL/SKOS+PROV-O substrate, that substrate IS the universal semantic layer. OGIT is positioned to host both as projections — Odoo's as an extracted ontology + Python adapter (per the work-steal distillation); Foundry's as the typed-object projection surface per `f6b68582-erp_foundry_hhtl_ontology_distillation.md` §"Foundry-style semantic surface".

### 15.2 Where they diverge — and what each divergence teaches

| Divergence axis | Odoo posture | Foundry posture | What OGIT/OWL/DOLCE inherits |
|---|---|---|---|
| **Ontology authoring** | **Implicit** — emerges from Python source; extraction is post-hoc reverse engineering | **Explicit** — Ontology Manager UI; authored upfront by data engineers | **Hybrid.** L1-L4 hydrators (DOLCE/PROV-O/QUDT/FIBO/etc.) are upfront-authored standards; `hydrate_odoo_*` extracts post-hoc. Both feed the same `OntologyRegistry`. The OGIT spine accommodates both authoring styles because the producer-side (per §4.6) is single — `MappingProposal` regardless of source. |
| **Coupling to runtime** | **Tight** — Odoo classes inherit Odoo's full ORM stack; can't extract Partner without inheriting `mail.thread` etc. | **Loose** — Object Types are runtime-independent; same Object Type can power a Foundry workflow + a REST API + a Spark job | OGIT inherits the **loose-coupling** discipline. `lance-graph-ontology` ships the spine; consumer crates (woa-bridge, medcare-bridge, smb-bridge) bind to it without contaminating it. The §4.6 read-only spine + §4.7 per-OGIT storage are the structural enforcement of this. |
| **Granularity of typing** | **Operational** — every field is in the data model; ~50k modules cover near-everything | **Curated** — Object Types are designed for analytic use; not auto-generated from raw rows | OGIT supports **both**. Odoo work-steal gives operational coverage; manual L3 hydrators (FIBO-FND/BE, schema.org subsets) give curated coverage. The cognitive substrate doesn't care; it reasons over both via the same CAM bar-code address space. |
| **Versioning** | **Module-version + Odoo-release-version** (annual major bumps; module deps pin per-release) | **Foundry semantic versioning** per Object Type | OGIT inherits the **versioned-G-slot** pattern (per §4.6's `OGIT::*_V1.0` (slot, version) tuple). Migration is consumer-driven opt-in; V1 and V2 coexist physically separate per the per-OGIT-storage rule. |
| **Schema evolution** | **Migration scripts per Odoo upgrade** (often hand-authored) | **Foundry ontology evolution** with backward-compat layers | OGIT inherits the **hydrate-once-per-version, version-coexist** model. New hydrator version (`hydrate_dolce_v2`) lands alongside `hydrate_dolce_v1`; consumers migrate when they pin to V2. |
| **Licensing posture** | **LGPL/OEEL split** — community is extractable; enterprise modules are not | **Proprietary** — no extraction path; only API consumption | OGIT must respect both. Odoo extraction stays LGPL-only per the work-steal doc; Foundry consumption is API-only. The OGIT spine is the public-good substrate both feed into. |
| **Naming conventions** | **Python identifiers** (`res.partner`, `account.move.line`) — Odoo-internal | **Domain-driven names** (`Counterparty`, `Position`, `JournalEntryLine`) — business-facing | OGIT names use **canonical OGIT URIs** (`ogit.Network:IPAddress`, `ogit.WorkOrder:Customer`, `ogit.Finance:Transaction`). Odoo names alias via `owl:equivalentClass`; Foundry-style names project via SPARQL. Same substrate, multiple aliases. |
| **Reasoning depth** | **Limited** — Odoo's reasoner is the Python eval of computed fields | **Moderate** — Foundry's reasoner is the typed-object surface + ad-hoc Spark queries | **Deep** — OGIT/lance-graph adds the cognitive substrate (NARS, MUL, DOLCE 16-bit slot classifier, 16-strategy planner). The convergence's structural gap (neither Odoo nor Foundry has a full cognitive layer) IS lance-graph's differentiation. |

**Headline lesson from the divergences:** Odoo gives us operational coverage at the cost of runtime coupling; Foundry gives us loose coupling at the cost of curated authoring. OGIT/lance-graph **decouples authoring from coverage** by accepting both upfront-authored standards (L1-L4 hydrators) and post-hoc extracted vocabularies (Odoo work-steal) into one CAM-addressable substrate, then adds the cognitive layer neither competitor has.

### 15.3 Long-run lessons for OGIT/OWL/DOLCE

Six structural lessons the convergence teaches, ordered by load-bearing-ness:

1. **The typed-object surface is a projection, not a separate ontology.** Foundry's Object/Link/Action/Function types map 1:1 to OWL primitives per the third distillation doc §"Foundry-side semantic surface". OGIT must NOT introduce a second ontology layer for the "user-facing typed object" view — it's a SPARQL/SHACL projection over the OWL substrate. Concrete consequence: the future `crates/foundry-projection/` (if ever built) is a thin SPARQL-template emitter, not a competing class hierarchy.
2. **Operational vocabulary needs an extraction source; pure formal-ontology authoring is insufficient.** Re-modelling 20 years of ERP from FIBO+UBL+XBRL-GL alone is years of work that Odoo already did. The hydrator pattern (per §4.3) generalizes to "any source with formal-enough structure": Python ORMs (Odoo), XSD schemas (UBL/ZUGFeRD), CSV reference data (DATEV SKR), legal text (HGB term extraction). Build the extractors; don't try to author the whole vocabulary by hand.
3. **DOLCE-as-root is the structural decision that makes everything else compose.** Every L2/L3/L4 hydrator declares `inherits_from: Some(OGIT::DOLCE_V1.0)` (per §4.3 layered ontology table). The 16-bit cognitive-shader DOLCE slot is *defined* relative to DOLCE's upper categories (Endurant/Perdurant/Quality/Abstract). Without DOLCE, downstream hydrators have no shared root — they classify against incompatible upper ontologies (BFO vs DOLCE vs SUMO) and the cognitive reasoner can't dispatch uniformly. DOLCE was the right L1 pick; keep it.
4. **The alignment TTL is the "glue", and its authoring workflow matters.** Per `b9531cf3-odoo_work_steal_distillation.md` §"Naming convention", alignment files (`owl:equivalentClass`, `owl:equivalentProperty`, `owl:sameAs`) are what tie extracted vocabularies to upstream standards. Authoring is mostly mechanical for the obvious cases (`res.partner.name owl:equivalentProperty foaf:name`) but human-judgement-required for the subtle cases (`account.move` IS BOTH `gl-cor:entryHeader` AND `fibo:Transaction` AND `ubl:Invoice` depending on `move_type` discriminator). The workflow has to support both: LLM-draft-then-reviewed for the obvious, human-authored for the subtle. Per the work-steal doc §"Open questions" — this is an open question worth answering before D-ODOO-3.
5. **Per-OGIT storage (§4.7) + read-only spine (§4.6) + CAM bar codes (§4.5) compose into a self-reinforcing pattern.** Each per-source TTL lands at its own `OGIT::<SOURCE>_V1` G-slot (Odoo `base` ≠ Odoo `account` ≠ FIBO-FND ≠ DOLCE). Per-G-slot CAMs are independently versionable + invalidatable + persistent. The bar-code addressing scheme + the inherits-from chain + the dense per-family arrays + the controlled producer-side appender = a substrate that scales to "everything anyone has ever ontologized" without thundering cache invalidation. This is the structural property that justifies building cognitive models around the substrate — the substrate doesn't drift under load.
6. **PROV-O is non-optional for any regulated domain.** GoBD (German digital bookkeeping), HIPAA (US healthcare), SOX (US financial reporting), GDPR (EU privacy), MiFID II (EU financial markets) — every regulated regime requires an unbroken provenance chain from output to source. Per the third distillation doc §"PROV-O" entry: "every entity in the ontology needs `prov:wasGeneratedBy`, `prov:wasDerivedFrom`, `prov:wasAttributedTo` to satisfy the German bookkeeping audit trail requirement." Both Odoo (via mail.thread mixin) and Foundry (via audit trail) ship this; OGIT already has `hydrate_provo` for it. Lesson: don't carry data through any layer that strips PROV-O — including the cognitive layer (CausalEdge64 v2's W-slot is the PROV-O carrier per `cognitive-substrate-convergence-v1.md`).

**The meta-lesson:** the convergence of Odoo + Foundry around an OWL substrate isn't accidental. Both arrived at the same shape by independent paths because the shape is structurally inevitable for any system that needs to **reason over typed business entities with regulatory auditability**. OGIT/lance-graph is positioned to be the open canonical implementation of that shape — and the cognitive substrate is the differentiator neither competitor has.

### 15.4 WoA work-steal map from Odoo

Direct correspondences between Odoo modules and WoA's existing sea-orm entities, with the wire/repurpose/work-steal/reinvent classification:

| WoA entity | Odoo source | Action | Notes |
|---|---|---|---|
| Customer (`Kunde`) | `res.partner` + `res.partner.bank` | **WORK-STEAL** | ~80 Odoo fields cover identity / address / contact / banking / tax IDs / company-vs-individual discriminator. WoA's Customer has ~25 fields; absorb the missing ~55 (or at least the alignment axioms) via `hydrate_odoo_base`. **D-ODOO-2 deliverable.** |
| Tenant (`Mandant`) | `res.company` | **WORK-STEAL** | Odoo's company master with chart-of-accounts binding maps directly to WoA's Mandant. Especially the `currency_id` / `country_id` / `chart_template_id` cross-links. |
| User (`Benutzer`) | `res.users` + `res.partner` (linked) | **WORK-STEAL** | Odoo's two-table user model (Partner-base + User-overlay) is cleaner than WoA's flat User table; the alignment via PROV-O `prov:Agent + foaf:Person` lets the cognitive shader reason about user-authored actions uniformly. |
| WorkOrder (`Vorgang`) | `account.move` (dual nature: journal entry + invoice) + `sale.order` | **WORK-STEAL** (the dual-projection pattern) | Odoo's `move_type` discriminator (`out_invoice`, `in_invoice`, `out_refund`, `in_refund`, `entry`) maps to WoA's Vorgang `kind` (Angebot / Auftragsbestätigung / Lieferschein / Rechnung / Gutschrift). Emit BOTH `fibo:Transaction` AND `ubl:Order|Invoice` per record — the cognitive layer can reason about either projection. |
| Position (`Position`) | `account.move.line` + `sale.order.line` | **WORK-STEAL** | Odoo's move-line model with per-line tax_ids + analytic accounting + product reference is the right shape for WoA Position. The `tax_ids` cross-link via `account.tax` connects to DATEV Steuerschlüssel naturally. |
| Mahnung (`Mahnung`) | (no direct Odoo equivalent in standard modules; OCA `account_due_list` or hand-roll) | **REPURPOSE** | Mahnwesen (German dunning) is German-specific; Odoo's community modules cover it via OCA. Repurpose the `account.payment.term` structure + escalation rules; reinvent the 3-stage Mahnstufe escalation in woa-bridge. |
| Stundenzettel-Eintrag | `hr.timesheet` (`account.analytic.line` in modern Odoo) | **WORK-STEAL** | Odoo's analytic-line + 15-minute rounding patterns match WoA's Stundenzettel Takt-15. Wire via `hydrate_odoo_hr` (P2 / D-ODOO-8). |
| Einsatz | `project.task` + `helpdesk.ticket` (community) + `mrp.workorder` | **REPURPOSE** | Multiple Odoo modules cover the live on-site engagement concept; none are perfect. Cherry-pick the time-tracking + photo-capture + signed-checkout fields from the OCA `project_task_signature` community module. |
| Logbook-Eintrag | `mail.thread` mixin records | **WORK-STEAL** | Odoo's mail.thread is exactly the audit-trail shape WoA's Logbook needs. PROV-O alignment via the work-steal doc §"Cross-cutting Odoo concerns to extract" → `mail.thread` row. |
| Dokument | `ir.attachment` | **WORK-STEAL** | Odoo's document-binary linkage model is well-tested; WoA's Dokument can absorb the same shape including the per-record binary refs + MIME-type discriminators. |
| Setting / `tenant_settings` | `ir.config_parameter` + `res.config.settings` | **REINVENT** | Per RFC-001 (`tenant_settings` typed struct), WoA explicitly diverged from Python's KeyValue Setting. Don't re-absorb Odoo's key-value pattern; keep the typed-struct discipline. |
| SKR03/SKR04 chart of accounts | `l10n_de_skr03` + `l10n_de_skr04` modules | **ALREADY WIRED** | `hydrate_skr03` + `hydrate_skr04` ship in lance-graph-ontology post-PR-407. **D-ODOO-6 is a P2 diff/refresh job**, not new work. |
| DATEV Steuerschlüssel | `l10n_de` tax templates (`account.tax.template.xml`) | **ALREADY WIRED PARTIALLY** | The chart is in tree; the Steuerschlüssel-to-USt-line mapping (`l10n_de_tax_statement`) is a Phase O5 follow-up worth pulling. **P2.** |
| ZUGFeRD/XRechnung invoice generator | (Odoo's e-invoicing module + l10n_de_zugferd community) | **ALREADY WIRED** | `hydrate_zugferd` + `hydrate_zugferd_rules` ship in lance-graph-ontology post-PR-407. The POC PR-5 milestone consumes these. **D-ODOO-7 adds the alignment axioms tying `account.move` → ZUGFeRD invoice shape.** |
| `mail.thread` audit trail | Same | **WORK-STEAL** | Already noted above for Logbook; reused here because it's also the substrate WoA needs for every entity's PROV-O annotation. One Odoo concept, many WoA consumers. |
| Computed-field dependency graph | `@api.depends(...)` decorators | **REINVENT** as SHACL | WoA's invoice totals + Mahnwesen escalation triggers + tax-base calculations all follow Odoo's compute-on-depends shape. Reinvent as SHACL `sh:rule` + SPIN; emits the same shape but lance-graph-native rather than Python-eval. |

**Headline win:** D-ODOO-2 + D-ODOO-3 + D-ODOO-4 (extract Odoo `base` module + alignment + `hydrate_odoo_base`) gives WoA's Customer / Tenant / User entities the 20-year-validated Odoo Partner field set + alignment axioms to FIBO/vcard/foaf — directly addressing the §10.5 D-WLG-15 `CanonicalCustomerRow` deliverable. The WoA reconciler diffs a richer canonical row than it would by re-deriving fields from scratch.


### 15.5 WoA work-steal map from smb-office-rs

smb-office-rs and WoA are sister consumers — both ports of inherited legacy systems, both subject to the unified-bridge consumer pattern, both end up reading the same lance-graph-ontology spine. But smb sources from C# WinForms + MongoDB; WoA sources from Python Flask + MySQL. Direct correspondences:

| WoA artifact | smb-office-rs source | Action | Notes |
|---|---|---|---|
| `crates/woa-bridge/` crate skeleton | `crates/smb-bridge/` | **WIRE** (mirror structure 1:1) | Per `lance-graph-in-woa-rs-v1.md` §10 — the smb-bridge crate layout (mod batch / error / number_sequence / orchestration / rls / settings / wal) is the template. Wire same shape into woa-bridge. |
| `crates/woa-ontology/` crate skeleton | `crates/smb-ontology/` | **WIRE** (mirror structure 1:1) | Same; smb's customer.rs / mahnung.rs / markings.rs pattern maps to woa-ontology/{customer,workorder,position,setting,user,document}.rs. |
| `woa_unified_bridge(...)` constructor | `smb_unified_bridge(...)` in `unified_bridge_wiring.rs` (~90 LOC) | **WIRE** (parameterise over WoaBridge instead of OgitBridge) | Per `lance-graph-in-smb-office-rs-v1.md` §7 — smb is the canonical template source for this constructor across all consumer plans. **D-UB-4 / D-WLG-3 implementation = 1:1 mirror with type-param swap to WoaBridge.** |
| `WoaMysqlConnector` impl of `EntityStore + EntityWriter` | `smb-bridge::mongo::MongoConnector` (313 LOC, gated `[features] mongo`) | **REINVENT** for sea-orm/MySQL | The trait surface (`lance-graph-contract::repository::{EntityStore, EntityWriter}`) transfers; the implementation differs (sea-orm Entity queries vs MongoDB BSON cursor iter). ~300 LOC + ~6 tests mirroring smb-bridge's shape. |
| `WoaLanceConnector` impl of same | `smb-bridge::lance::LanceConnector` | **WIRE** (essentially identical) | The Lance-side EntityStore implementation is storage-agnostic; reuse the smb-bridge structure for the Lance projection of woa MySQL tables. |
| `WoaMysqlReconciler` | `SmbMongoReconciler` (395 LOC, mirrors `MedcareMysqlReconciler` per `lance-graph-in-medcare-rs-v1.md` §8) | **WORK-STEAL** the shell + reinvent the fetchers | The reconciler shell (route parser + diff machinery + DriftEvent emission via the `lance_graph_callcenter::transcode::parallelbetrieb::Reconciler` trait) transfers verbatim. Only `CustomerFetcher` impl differs — sea-orm `Entity::find_by_id` vs MongoDB `MongoConnector::find_by_id`. Per §10.5 D-WLG-15. |
| `sea-orm-schema-warden` agent | `mongo-schema-warden` agent (smb's `.claude/agents/`) | **REINVENT** for sea-orm | smb's agent enforces BSON field name parity to C# `db_*.cs`. WoA's analogue enforces sea-orm Entity field name parity to Python `models.py` (per Iron Rule №2 — read Python source before writing Rust). ~200-line agent card mirroring smb's pattern. |
| `transcode-auditor` agent | smb-office-rs's `transcode-auditor` | **WIRE** (reuse verbatim with WoA-specific reference path) | smb's transcode-auditor fires when porting `db_*.cs` files; WoA's analogue fires when porting `woa/<group>.py` route blueprints. Same agent shape, different reference root. |
| `unified_bridge_wiring.rs` doc-comment design map | smb-bridge `unified_bridge_wiring.rs` lines 9-14 + 16-25 + 27-33 | **WIRE** (carry over verbatim with substitutions) | smb's forward-looking comments about the SmbBridge type-parameter swap + auth::TenantId consolidation + post-D-SDR-2/3 shrink are templated thinking. WoA's `woa_unified_bridge` ships with analogous comments substituting `WoaBridge` for `SmbBridge` and `Mandant.id` for `praxis_id`/`kdnr`. |
| Iron Rule "lance-graph is additive-only" | smb-office-rs CLAUDE.md Iron Rule 3 | **WIRE as policy** | WoA inherits this implicitly via being a consumer of `lance-graph-ontology` (the same upstream). Make it explicit in woa-rs CLAUDE.md as a stack-decision row. |
| Per-customer binary pattern | smb-office-rs's `customer-<name>-bin` crate pattern (per smb CLAUDE.md "Single binary per customer") | **WIRE LATER** (P3, not POC scope) | Currently WoA is Stefan-single-tenant; multi-tenant deployments would absorb smb's per-customer cargo-feature subsetting model. Defer until WoA actually has multi-tenant deployment demand. |
| FFI to WinForms | smb-bridge's JSON-over-C-ABI FFI | **DO NOT TAKE** | WoA has a web UI (axum + askama), not a desktop UI. FFI is unnecessary and would add complexity. |
| MongoDB connector (smb-mongo + smb-bridge::mongo) | Same | **DO NOT TAKE** | WoA's cold path is MySQL via sea-orm per the DualSink-Pivot. MongoDB is irrelevant. Reinvent the EntityStore/EntityWriter for sea-orm instead (above row). |
| BSON-schema-specific Customer canonical row | smb's `CanonicalCustomerRow` | **REINVENT** with WoA field set | smb's row has German BSON field names (kdnr, firma, vorname, lastname, plz, ort, strasse); WoA's row uses German MySQL field names (kdnr, firma, vorname, nachname, plz, ort, strasse) — similar shape, different storage. Reinvent. |

**Headline wins from smb work-steal:**
1. **Constructor mirror.** D-WLG-3 (`woa_unified_bridge` constructor) is a 1:1 mirror of `smb_unified_bridge` with `WoaBridge` substituted for `OgitBridge`. ~50 LOC + 2 tests. This is the cheapest possible deliverable in the entire POC P0 path.
2. **Reconciler shell harvest.** The cross-source comparison shell (`SmbMongoReconciler` ← copies `MedcareMysqlReconciler` ← cited as the canonical sister) transfers cleanly to `WoaMysqlReconciler`. Only the `CustomerFetcher` impl is per-consumer.
3. **Agent ensemble templates.** smb's `mongo-schema-warden` + `transcode-auditor` + `truth-architect` + `integration-lead` agent cards transfer with reference-path substitutions. WoA gets a working agent ensemble without designing one from scratch.

### 15.6 The unified roadmap implication

The §15.4 (Odoo) + §15.5 (smb-office-rs) work-steal maps + the §14 Odoo OWL glue elevation collapse into one observation:

**WoA's POC path is mostly composition, not invention.** Specifically:

- The substrate (lance-graph-ontology spine, hydrators, unified-bridge, parallelbetrieb reconciler) is shipped or in-flight from sister plans (lance-graph + medcare-rs + smb-office-rs).
- The Odoo work-steal supplies the operational ERP vocabulary (~80-field Customer + dual-nature Vorgang/Transaction + chart of accounts + tax structures + audit trail) that WoA's entities map onto.
- The smb-office-rs work-steal supplies the consumer-crate scaffolding (woa-bridge / woa-ontology shape + unified-bridge constructor + reconciler shell + agent ensemble).
- The codegen pipeline (RFC v02-006) propagates each integration step to the 660 routes via per-bucket template edits — one PR upgrades hundreds of route handlers.
- The Foundry-style typed-object surface is a SPARQL projection over the OWL substrate; not a separate ontology layer to build.

**What WoA actually has to invent:**
1. The sea-orm fetchers for `WoaMysqlReconciler` (per §15.5 D-WLG-16; ~50 LOC of `Entity::find_by_id` per entity).
2. The `Mandant.id` (i32) → `lance_graph_callcenter::TenantId` (u32) mapping (per §3 D-WLG-6; ~40 LOC + 2 tests).
3. The permission-to-actor_role mapping (per §3 D-WLG-7; ~50 LOC + 4 tests covering each WoA permission).
4. The WoA-specific route handlers that aren't bucket-generic (the ~9 routes in the `signed_link_action` + `sa_admin_view` security-critical buckets per RFC v02-006).
5. The askama templates for any new WoA-specific UI views (e.g., `/admin/parity` for PR-6; the codegen handles the route-handler side but the template body is per-bucket-template).

**That's it.** Everything else is harvest. The 6-month lance-graph substrate work + the smb/medcare consumer-plan work + the Odoo work-steal + the RFC v02-006 codegen pipeline collectively cover ~90% of what WoA's POC needs. WoA's actual invention surface is ~5 deliverables totalling ~250 LOC + ~10 tests.

This is what makes the POC achievable in ~7-8 days net per §12.6 — the substrate has been pre-built to the point where the customer-visible integration is mostly wiring.

### 15.7 One-line summary

> Odoo and Palantir Foundry converge on a typed-object semantic surface that reduces to OWL/SHACL/SKOS+PROV-O; OGIT/OWL/DOLCE is positioned to host both as projections (Odoo via extraction + alignment; Foundry via SPARQL/SHACL projection). The long-run lessons are six: typed-object surface IS a projection not a separate ontology; operational vocabulary needs an extraction source; DOLCE-as-root is the load-bearing structural decision; alignment TTL authoring workflow matters; per-OGIT-storage + read-only-spine + CAM-bar-codes compose into a self-reinforcing pattern; PROV-O is non-optional for any regulated domain. WoA work-steals ~80% of its Customer + Tenant + User + Vorgang + Position + Logbook + Dokument entity shapes from Odoo's `base` + `account` + `mail.thread` modules; ~80% of its consumer-crate scaffolding (woa-bridge, woa-ontology, unified-bridge constructor, reconciler shell, agent ensemble) from smb-office-rs's templates. The actual invention surface is ~5 deliverables, ~250 LOC. The 6 months of lance-graph substrate + sister-consumer-plan harvest + Odoo OWL glue + RFC v02-006 codegen collectively cover ~90% of what WoA's POC PR-5 needs.

