# Plan v1 — OGAR sink-in + consumer-bridge removal (medcare-bridge → UnifiedBridge → delete)

> **Status:** PROPOSED (doc-only plan). Most of the mechanical migration is
> **already shipped** (see §1); this plan codifies the *sink-in layering* and
> sequences the **final deletion** of the deprecated per-consumer bridges +
> the medcare domain's split into reusable-patterns vs ontology-enrichment.
> **Extends, does not supersede:** `docs/CONSUMER-BRIDGE-DEPRECATION.md`
> (OGAR#95) + `.claude/plans/ogar-vocab-contract-codebook-migration-v1.md`
> (D-OVC-*). **Gated on operator sign-off** for any classid realign + the
> deletion timing (consumer repos must migrate first).
> **Source of truth:** `AdaWorldAPI/OGAR` `docs/{APP-CLASS-CODEBOOK-LAYOUT,
> CONSUMER-MIGRATION-HOWTO, CLASSID-RBAC-KEYSTONE-SPEC}.md`.

---

## 0. The ask (operator)

1. Decide **what sinks into OGAR + lance-graph-ogar** vs **stays in
   lance-graph-contract + lance-graph-rbac**.
2. Migrate **medcare-bridge → UnifiedBridge**, then **remove medcare-bridge
   from lance-graph** — recasting it as **reusable patterns ↔ domain-specific
   Ontology Schema enrichment** (the medcare domain is no longer a *bridge*; it
   is a shared classid surface + a domain ontology that consumers pull + enrich).

---

## 1. Current state (grounded — do NOT re-litigate)

- **All 6 per-consumer bridges are already `#[deprecated]` one-line aliases**
  over a single generic harness: `pub type MedcareBridge =
  UnifiedBridge<HealthcarePort>` (`crates/lance-graph-ogar/src/bridges/medcare_bridge.rs`).
  Same for odoo/redmine/smb/openproject/woa. **No bespoke bridge struct remains.**
- **`UnifiedBridge<P: PortSpec>`** (`bridges/unified.rs`) is the *internal*
  mechanism — NOT deprecated. It is classid-driven: `entity(name)` does
  `P::class_id(name)` first, falls back to the `OntologyRegistry` with a
  `g_lock` namespace check. Everything per-port (`NAMESPACE`, `BRIDGE_ID`,
  `class_id()`, `aliases()`) comes from the OGAR `ogar_vocab::ports::PortSpec`.
- **Two migration layers:** (a) struct → `UnifiedBridge<P>` alias = **DONE**;
  (b) alias → direct `Port::class_id()` pull in consumers = **IN PROGRESS**
  (deprecation warnings live; removal pending per OGAR#95 "deletion lands later").
- **`bridge_scope_lock.rs`** locks the invariant any replacement MUST keep: a
  namespace-locked bridge refuses cross-namespace resolution
  (`BridgeError::{CrossNamespaceLeak, NotInScope}`).
- **HealthcarePort** lives in OGAR (`ogar_vocab::ports`); Healthcare codebook =
  the `0x09XX` block (Patient `0x0901`); **FMA-V3 `0x1000_0A01` / CPIC-V3
  `0x1000_0E00`** minted in contract (#618).

**Conclusion:** "migrate medcare-bridge to UnifiedBridge" is **already true**.
The real remaining work is **(b) finish the consumer pull-migration, then
DELETE the deprecated aliases**, and **codify the sink-in** so the medcare
domain lands as patterns + ontology, not a bridge.

---

## 1.5 Grounding against the actual OGAR PR arc (#95–#142, read 2026-06-26)

The OGAR side is **substantially shipped** — this plan sequences the *last mile*,
not a green field. Landmarks (all merged unless noted):

- **#95** — `APP-CLASS-CODEBOOK-LAYOUT.md` + `APP-CODEBOOK-MIGRATION-PLAN.md` +
  `CONSUMER-MIGRATION-HOWTO.md` + `CLASSID-RBAC-KEYSTONE-SPEC.md`. **The §2 table
  below is #95's, verbatim:** `classid = [hi u16 APP/render][lo u16 concept]`; lo
  = WHAT (RBAC + ontology + cross-app identity, shared), hi = WHOSE render
  (ClassView/Askama, per-app). **Medcare worked example: `patient = 0x0005_0901`**
  (app prefix `0x0005`, concept `0x0901`). The consumer migration is already
  **wave-ordered** there: W0 OpenProject(`0x0001`)+Redmine(`0x0007`) → W1 WoA →
  W2 SMB → W3 Odoo (delete the `od-ontology` fork) → W4 q2. **D-SINK-2 below
  rides that wave plan; it does not invent its own.** Status `[H]`, minting gated
  on the 5+3 codebook pass; `PROBE-KV-RENDER-NOSERDE` promotes `[H]→[G]`.
- **#96/#97/#98** — `billable_work_entry` 5-way convergence pinned; typed
  `PortSpec::APP_PREFIX` + `render_classid` composition; `canonical_concept_name`
  reverse map. (The "reusable patterns" surface = these.)
- **#105/#106/#109** — **OGIT 1:1 import** + `ogar-from-schema` producer; "OGIT
  was already the symbol table." (The ontology-enrichment sink — D-SINK-5 — is
  largely done at the *import* layer.)
- **#110/#111/#120/#126/#127** — codebook mints: AuthStore `0x0B`, PRODUCT/
  ACCOUNTING_ACCOUNT/PRICELIST/UOM (`0x02XX`), Automation `0x0C`, HR `0x0DXX`.
- **#114–#125 — the HIRO DO arm is SHIPPED.** #114 the **lossless-DO rule**
  (`ActionDef` *points to* the body, never DDL-inline — `I-VSA-IDENTITIES` for
  DO; shape `[G]`, execution CONJECTURE pending `PROBE-OGAR-DO-ARM-LIFT`); #119
  HIRO→`ActionDef` producer; #121 arago ActionHandler contract; #125
  **class-late-bound dispatch (ResolvingDaemon)** = the classaction pointer.
  **DO arm + RBAC meet at `ActionHandler` (`auth_store 0x0B01`).**
- **#116/#117** — **FMA skeletal spine already minted** (bones as the clamped
  convergence anchor, 3D-octree CRS, projection contract). *MedCare's FMA is not
  just a classid to consume — OGAR already has the anatomy spine.*
- **#128** — value-slab facet closure + **classid-driven envelope parser**
  (V2/V3, the gen marker). #130 ADR: **V3 as spine + polyglot transpiler**.
- **#131/#132/#138/#141** — `ogar-from-ruff` (Python/Odoo lift consuming
  `ruff_python_spo`) + per-class transpile substrate + **`emit_csharp`/
  `emit_python` codegen**. *This is the downstream consumer of ruff #29's C#
  harvest.*
- **#137 — the spine is the COMPILED `ClassView`, not SurrealQL.** One compiled
  reader recombines all facet carvings (`6×(1:2)`/`4×(1:2:3)`/`3×(1:2:3:4)`),
  compiled into the binary (not parsed, not a runtime interpreter); SurrealQL is
  the storage membrane only. **§2's "dynamic ClassView" = a *compiled* reader;
  Askama is the membrane emit.** (#135 closed: the same content relocated.)

**Net:** the doctrine (#95), the reusable-pattern surface (codebook + PortSpec),
the OGIT import, the DO arm, FMA, and the transpiler are all merged. What this
plan still owns: the **lance-graph-side alias deletion** (D-SINK-3) gated on the
**#95 wave consumer-migration** (D-SINK-2), the **WoaBridge reconciliation**
(D-SINK-4), and the **codebook-mint completion** (D-SINK-5, the cross-repo arc).

## 2. The sink-in layering — what lives WHERE (the decision)

The iron principle (from the deprecation doctrine): **the agnostic spine
(`lance-graph-*`) does NOT own consumer ontology.** OGAR provides classid +
schema + grant; the consumer enriches locally.

| Concern | Sinks into | Why |
|---|---|---|
| **PortSpec** (`HealthcarePort`, public-name→classid alias table, `NAMESPACE`, `BRIDGE_ID`) | **OGAR** (`ogar-vocab::ports`) | The per-consumer alias table is consumer ontology; OGAR is its home. lance-graph only re-exports it. |
| **Codebook** (`CODEBOOK` `0xDDCC`, `ConceptDomain`, `canonical_concept_id`, `LabelDTO`) | **OGAR Core** (authoritative) → **lance-graph-contract** (wire-compatible mirror, zero-dep) | Per D-OVC-1..4: contract hosts the zero-dep mirror; OGAR `ogar-vocab` is source-of-truth; the two agree on the `u16` LE wire (the parity guard). |
| **ClassView / ActionDef / Class** (THINK + DO arm IR) | **OGAR** (`ogar-class-view`, `ogar-vocab::Class`, `OgarActionProvider`) | The AR-shaped class is OGAR's unit; lance-graph-ogar re-exports + activates it. |
| **Domain ontology schema** (Healthcare/FMA/CPIC anatomy+pharma TTL) | **OGIT** (`AdaWorldAPI/OGIT`, imported into OGAR `vocab/imports/ogit`) → hydrated via **lance-graph-ontology** `OntologyRegistry` | TTL is the only ontology exchange format; OGIT is canonical; Lance is the runtime dictionary cache. |
| **UnifiedBridge harness + the 6 deprecated aliases + parity guard** | **lance-graph-ogar** (stays — it's the OGAR *activation* crate) | The harness is internal mechanism; the aliases are removed (§3); `parity::domains_agree`/`COUNT_FUSE` stays. |
| **`NamespaceBridge` trait + `OntologyRegistry` + namespace/error types** | **lance-graph-ontology** (OGIT spine) | The registry that bridges resolve against; tenant views are thin scoped reads over it. |
| **RBAC grants** (role→PrefetchDepth, predicate writes, action triggers) | **lance-graph-rbac** (depends only on contract) | Grants key on the **classid lo-u16 (shared concept)** — the render skin (hi-u16) is per-app and never gates RBAC. The CLASSID-RBAC-KEYSTONE-SPEC is the contract. |

**Two one-liners that settle the boundary:**
- **OGAR owns the *meaning* (PortSpec/codebook/ClassView/ActionDef/ontology);
  lance-graph-contract owns the *wire* (the zero-dep `u16`-LE mirror).** They
  meet at the parity guard, never by dependency.
- **lance-graph-ogar is OGAR's *activation* surface inside lance-graph; it
  re-exports OGAR and hosts only mechanism (UnifiedBridge) + parity — never a
  per-consumer bridge.**

---

## 3. The medcare-bridge → reusable-patterns ↔ ontology-enrichment recast

medcare-bridge **dissolves into three already-existing homes** — nothing new is
built; the bridge file is deleted and its responsibilities are already elsewhere:

| What medcare-bridge "was" | Becomes | Home |
|---|---|---|
| The classid alias table (`Patient→0x0901`, …) — the **reusable pattern** | `HealthcarePort::class_id(name)` | OGAR `ogar-vocab::ports` |
| The shared ERP concepts (billing, scheduling, docs) | the **same** convergent codebook ids every consumer resolves (`BILLABLE_WORK_ENTRY 0x0103`, …) — the cross-fork reuse | OGAR codebook (mirrored in contract) |
| The **domain-specific** anatomy/pharma meaning (Healthcare `0x09XX`, FMA-V3, CPIC-V3) — the **Ontology Schema enrichment** | OGIT TTL + the V3 classids, hydrated via `OntologyRegistry` | OGIT / OGAR; lance-graph-ontology cache |
| The `MedcareBridge` *type* + `NAMESPACE` const + co-located tests | **deleted** (consumer pulls `HealthcarePort::class_id`; integration coverage already in `bridge_scope_lock.rs` / `bridge_codebook_convergence.rs`) | — (removed from lance-graph) |

**The recast in one sentence:** the medcare domain is **reusable patterns**
(the shared classid/codebook surface in OGAR) **plus domain-specific Ontology
Schema enrichment** (the Healthcare/FMA/CPIC ontology in OGIT) — pulled and
enriched by `medcare-rs`, never a bridge object in the agnostic spine.

---

## 4. Deliverables (ordered; each gated on the prior)

- **D-SINK-1 — Ratify the sink-in layering (§2).** *Decision, not code.*
  Operator confirms the WHERE table (esp. the OGAR↔contract dependency
  direction and the RBAC-keys-on-lo-u16 keystone). Gate for everything below.
- **D-SINK-2 — Migrate consumers off the `*Bridge` aliases — ride OGAR #95's
  wave plan, don't reinvent it.** `APP-CODEBOOK-MIGRATION-PLAN.md` already orders
  it: **W0** OpenProject(`0x0001`)+Redmine(`0x0007`) → **W1** WoA → **W2** SMB →
  **W3** Odoo (delete the `od-ontology` fork) → **W4** q2. In each consumer:
  replace `FooBridge::new(registry)?.entity(name)?` with `FooPort::class_id(name)`
  and stamp the app prefix in the hi-u16 for the render classid (Medcare:
  `patient = 0x0005_0901`). `medcare-rs`/`woa-rs`/`smb-office-rs` work, not
  lance-graph. **Cross-repo approval gate** (Iron Rule 5).
- **D-SINK-3 — Delete the 6 deprecated aliases from `lance-graph-ogar::bridges`**
  once D-SINK-2 lands for every consumer. Keep `UnifiedBridge<P>` (mechanism)
  and the `Port` re-exports. Remove the per-module `NAMESPACE` consts +
  `OPENPROJECT_CODEBOOK`/`REDMINE_CODEBOOK` compat shims. **Preserve the
  `bridge_scope_lock` invariant** — port the scope-lock tests onto the bare
  `UnifiedBridge<P>` / `Port::class_id` path so cross-namespace refusal is still
  asserted (field-isolation discipline per `I-LEGACY-API-FEATURE-GATED`).
- **D-SINK-4 — Reconcile the two `WoaBridge` identities.** OGAR
  `lance_graph_ogar::bridges::WoaBridge` vs legacy OGIT
  `lance_graph_ontology::bridges::WoaBridge`. Decide: collapse onto the OGAR
  PortSpec path or keep the OGIT-legacy one explicitly scoped + documented.
- **D-SINK-5 — Sink the medcare ontology enrichment.** Ensure Healthcare/FMA/CPIC
  schema is in OGIT (`vocab/imports/ogit`) and mintable to the codebook — this
  is the **cross-repo codebook-mint arc** (`ISS-OGAR-GENETICS-MIRROR-PENDING`,
  `E-CODEBOOK-MINT-IS-A-CROSS-REPO-ARC`): OGAR `ogar-vocab` + contract `CODEBOOK`
  mirror + lance-graph-ogar `parity::domains_agree` move **together**. Not a
  local wire.

---

## 5. Gates / risks

- **No deletion before consumers migrate** — OGAR#95 is explicit ("nothing
  removed; deletion lands later"). D-SINK-3 is blocked on D-SINK-2 across
  woa-rs / medcare-rs / smb-office-rs (and any other PortSpec consumer).
- **Scope-lock regression** — deleting the aliases must not drop the
  `CrossNamespaceLeak`/`NotInScope` guarantee; port `bridge_scope_lock.rs` onto
  the survivor path first (test-before-delete).
- **Codebook parity** — any classid touch trips `COUNT_FUSE` /
  `assert_codebook_parity`; the cross-repo mint (D-SINK-5) must keep OGAR and
  the contract mirror in lockstep (the drift guard).
- **classid realign open item** — the merged-OSINT/FMA realign in
  `ogar-vocab-contract-codebook-migration-v1` (D-OVC) + the Canon:Custom
  hi/lo half-order are **operator-gated** and intersect this plan; do not
  pre-empt them.
- **BBB barrier (medcare-rs)** — the consumer pulls `ogar-vocab::ports` (a
  classid function), never a brain crate; `medcare-bridge` (the *consumer* crate
  in MedCare-rs) stays contract-tier.

---

## 6. References (prior art this extends — read before acting)

- `docs/CONSUMER-BRIDGE-DEPRECATION.md` — the doctrine + the before/after pull
  pattern (the spine for this plan).
- `.claude/knowledge/ogar-consumer-preflight.md` — the consumer spellbook
  (pull `*Port::class_id` / `canonical_concept_id`; never construct a `*Bridge`).
- `.claude/plans/ogar-vocab-contract-codebook-migration-v1.md` — D-OVC-1..4
  (codebook→contract seam + the classid realign).
- OGAR `docs/{APP-CLASS-CODEBOOK-LAYOUT, CONSUMER-MIGRATION-HOWTO,
  CLASSID-RBAC-KEYSTONE-SPEC}.md` + tracking **OGAR#95**.
- `crates/lance-graph-ogar/src/bridges/{mod,unified,medcare_bridge}.rs` +
  `tests/bridge_scope_lock.rs` (the code being changed).
- Companion: `.claude/handovers/2026-06-26-ast-address-to-next-session-capstone.md`
  (the OGAR-as-importable-ERP-stdlib "why").

---

## 7. One-paragraph summary

The per-consumer bridges are already collapsed onto one generic
`UnifiedBridge<P: PortSpec>` harness, and every `*Bridge` (incl. `MedcareBridge`)
is a `#[deprecated]` alias awaiting deletion once consumers pull
`Port::class_id()` directly (OGAR#95). This plan **codifies the sink-in** —
OGAR owns the meaning (PortSpec / codebook / ClassView / ActionDef / ontology),
`lance-graph-contract` owns the zero-dep wire mirror, `lance-graph-ontology`
owns the OGIT registry, `lance-graph-rbac` owns grants keyed on the shared
lo-u16 concept, and `lance-graph-ogar` keeps only the harness + parity — and
**recasts the medcare domain** from a bridge into **reusable patterns** (the
shared classid/codebook surface) **plus domain-specific Ontology Schema
enrichment** (Healthcare/FMA/CPIC in OGIT). The deletion (D-SINK-3) is gated on
the cross-repo consumer migration (D-SINK-2) and the scope-lock test port; the
ontology enrichment (D-SINK-5) is the cross-repo codebook-mint arc, not a local
wire.
