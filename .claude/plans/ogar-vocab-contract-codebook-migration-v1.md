# Migration — OGAR `ogar-vocab` codebook ⇄ `lance-graph-contract` classid (v1)

> **Status:** SHIPPING (2026-06-20). Operator signed off §5; D-OVC-1/2/4 landed
> on the jirak branch + D-OVC-3 realign landed (canon-doc cross-ref pending).
> Originally surfaced a **canon conflict** between merged `lance-graph-contract`
> classids and OGAR's `ogar-vocab` codebook.
> **The triangle:** ontology (OGAR `ogar-vocab`) → contract (`NodeGuid`/`ClassId`)
> → q2 (Quadro-2 cockpit consuming `GraphSnapshot`).

---

## 1 — The seam, grounded (file:line)

OGAR's `crates/ogar-vocab/src/lib.rs` already defines the canonical class
identity layer, and **its own doc-comment says where it belongs**:

- **`CODEBOOK`** (`lib.rs:1073`) — curated `(canonical_concept, u16)` table, ids
  assigned (never hashed). Domain-encoded `0xDDCC` (high byte = domain).
- **`ConceptDomain`** (`lib.rs:1141`) + **`canonical_concept_domain(id)→ConceptDomain`**
  (`lib.rs:1163`, routes on `id >> 8`, O(1) no-lookup).
- **`source_domain_concept("project"|"erp"|"german-erp")→ConceptDomain`**
  (`lib.rs:1186`) — the seam from `Class::source_domain` (the coarse curator tag,
  `lib.rs:193`) to the typed domain.
- **`canonical_concept_id(concept)→Option<u16>`** (`lib.rs:1214`) +
  `Class::canonical_id()/canonical_id_le()` (`lib.rs:1026/1034`).
- **`LabelDTO { label, id: u16, canonical }`** (`lib.rs:1476`) + `from_alias()` —
  consumer alias → shared codebook id. **`lib.rs:1208`:** *"The contract type
  (`LabelDTO`) lives in `ogar-vocab` today; **long-term it belongs in
  `lance-graph-contract`** alongside `ClassId` and the `NodeGuid` LE layout. Wire
  is the source of truth: any encoder/decoder agreeing on `u16` LE is compatible
  regardless of which crate exports the DTO."* And `lib.rs:1204-1206`: *"codebook
  ids and the `NodeGuid.classid` u16 low half are wire-compatible."*

So **the OGAR codebook id IS the contract classid (low u16)** — one wire value,
two crates. Contract has none of it yet (grep: no `ogar-vocab` reference in
`lance-graph-contract`).

## 2 — The conflict (the migration gap)

OGAR's `0xDDCC` domain layout vs the classids contract minted this session
(#557/#560, merged):

| Domain | OGAR `ConceptDomain` block | contract today | Aligned? |
|---|---|---|---|
| project-mgmt (OP↔Redmine) | `0x01XX` | — (un-minted) | n/a — **mint** |
| commerce/ERP (OSB↔Odoo) | `0x02XX` | — (un-minted) | n/a — **mint** |
| OSINT | **`0x07XX`** | `CLASSID_OSINT = 0x0007` | ❌ `0x0007 >> 8 = 0x00` = **Reserved**, not OSINT |
| OCR | `0x08XX` | `CLASSID_FMA = 0x0008` | ❌ `0x0008` is in OGAR's **OCR** block |
| Health (clinical) | `0x09XX` | (FMA anatomy ≈ Health) | ❌ FMA/anatomy is medical → belongs `0x09XX`, not `0x0008` |

Root cause: `CLASSID_OSINT=0x0007` was minted from the early guess "OSINT is
0x0007" before `ogar-vocab`'s domain-encoded layout was consulted. Under OGAR,
the OSINT *domain* is the high byte `0x07`, so an OSINT class is `0x07CC`
(e.g. `0x0700`), and `0x0007` is a Reserved-domain slot. `0x0008` collides with
the OCR domain; FMA (Foundational Model of Anatomy) is clinical → Health
`0x09XX` (or a dedicated anatomy domain) — never `0x0008`.

## 3 — Target state (single source of truth)

Per OGAR's own note, **`lance-graph-contract` is the long-term home** for the
class-identity codebook. Reconcile onto OGAR's `0xDDCC` scheme:

1. **Codebook + domain types live in contract.** Move (or mirror, wire-compat)
   `ConceptDomain`, `canonical_concept_domain`, `source_domain_concept`,
   `canonical_concept_id`, the `CODEBOOK`, and `LabelDTO` into
   `lance-graph-contract` (next to `ClassId`/`NodeGuid`). `ogar-vocab`
   re-exports them (OGAR→contract dep) **OR** both keep a copy and the **wire
   (`u16` LE) is the contract** (no new dep). *Decision needed — see §5.*
2. **classids follow `0xDDCC`.** `NodeGuid.classid` low u16 == the codebook id.
   - project-mgmt: `0x01XX` (mint `CLASSID_PROJECT = 0x0100` block).
   - commerce/ERP: `0x02XX` (mint `CLASSID_ERP/COMMERCE = 0x0200` block).
   - OSINT: realign `CLASSID_OSINT` → `0x0700` (Gotham domain).
   - anatomy/FMA: realign `CLASSID_FMA` → Health `0x09XX` (or a new anatomy
     domain block, reserved appended — never `0x0008`).
3. **`canonical_concept_domain` becomes the `ReadMode`/domain router** — the
   `classid → ReadMode` registry keys off `id >> 8` (the domain), so OSINT/FMA/
   project/ERP all resolve by the same O(1) high-byte rule.
4. **The per-family codebook (D-GV2-2) is the FINER scope of the SAME idea.**
   OGAR `CODEBOOK` = the *concept/classid* codebook (domain `0xDDCC`); the
   `guid-v2-tail` `FamilyCodebookRegistry` (`contract::codebook`) = the
   *within-family* label vocab. They compose: classid (domain) selects the
   coarse codebook; family selects the sub-codebook. Longest-prefix-wins, one
   rule (OGAR `CLAUDE.md` "Codebook scoping = the class routing prefix").

## 4 — Deliverables (gated on §5 decisions)

> **Update 2026-06-20:** operator signed off §5 (realign 0xDDCC / wire-compat /
> FMA = Health `0x0901`). D-OVC-1/2 SHIPPED, D-OVC-4 SHIPPED (function + tests;
> q2 display-label leg is q2-side); D-OVC-3 (cutover audit) downgraded — the
> classid realign is a const *value* change, not a bit-layout reclaim, so it is
> layout-preserving (no `ENVELOPE_LAYOUT_VERSION` bump); what remains is updating
> the `lance-graph/CLAUDE.md` canon block + OGAR `CODEBOOK` cross-doc.

- **D-OVC-1** ✅ **SHIPPED.** NEW `contract::ogar_codebook` — wire-compat mirror
  (zero-dep, no OGAR↔contract dependency): `ConceptDomain`, `canonical_concept_domain`,
  `classid_concept_domain` (the classid→domain route, D-OVC-4), `source_domain_concept`,
  `CODEBOOK` (project `0x01XX` + commerce `0x02XX`), `canonical_concept_id`,
  `LabelDTO { label, id, canonical }` + `from_canonical` + `id_le`. (Named
  `from_canonical`, not `from_alias`: the contract carries the codebook-id layer,
  NOT OGAR's curator-alias normalizer — that stays in `ogar-vocab`.) Drift-guard
  test pins the shared `0xDDCC` ids; 6 tests.
- **D-OVC-2** ✅ **SHIPPED.** Minted `CLASSID_PROJECT` (`0x0100`) + `CLASSID_ERP`
  (`0x0200`) in `canonical_node.rs` + `ReadMode::{PROJECT, ERP}` (both Cognitive /
  CoarseOnly), registered in `BUILTIN_READ_MODES`. Added `soa_graph::{PROJECT, ERP}`
  `DomainSpec`s (siblings of `OSINT_GOTHAM`/`FMA_ANATOMY`), re-exported from lib.rs.
- **D-OVC-3** ◐ **PARTIAL.** Canon realign LANDED: `CLASSID_OSINT 0x0007 → 0x0700`,
  `CLASSID_FMA 0x0008 → 0x0901` (anatomy concept in Health). Tests updated to assert
  the new values + `>>8` domain bytes. **Layout-preserving** (const value, not a bit
  reclaim) → no field-isolation matrix / version-gate needed. **Remaining:** update
  the `lance-graph/CLAUDE.md` canon block note + OGAR `CODEBOOK` cross-ref doc.
- **D-OVC-4** ✅ **SHIPPED (contract leg).** `classid_concept_domain(classid)` routes
  on the low-u16 `0xDDCC` high byte; tests assert all five classids resolve to their
  `ConceptDomain`. q2's `LabelDTO`/`canonical` display-label consumption is the q2-side
  leg (this crate exports the type + ids).

## 5 — Decisions needed (operator) — ✅ RESOLVED 2026-06-20

1. **Canon realign OSINT/FMA?** → **YES, realign to `0xDDCC`** (OSINT `0x0700`,
   FMA `0x0901`).
2. **Dependency direction?** → **(b) wire-compat now** — both define, the `u16` LE
   wire is the only contract, drift-guard test prevents divergence. No new dep.
3. **FMA/anatomy domain?** → **Health `0x09XX`** — FMA = anatomy concept `0x0901`,
   `0x0900` = Health root. (`CC = 0x00` = domain root, reserved everywhere.)

1. **Canon realign OSINT/FMA?** `CLASSID_OSINT 0x0007 → 0x0700`, `CLASSID_FMA
   0x0008 → 0x09XX`. This rewrites merged canon (#557/#560) + the `lance-graph/
   CLAUDE.md` canon block. Recommended (otherwise contract and OGAR disagree on
   what `0x07`/`0x08` mean), but it's your canon to change. Alternative: keep
   `0x0007/0x0008` and re-document OGAR's domain layout to match (worse — breaks
   the clean `id>>8` domain route).
2. **Dependency direction for the shared types:** (a) move to contract,
   `ogar-vocab` `pub use`s from it (OGAR gains a `lance-graph-contract` dep);
   or (b) both define, wire (`u16` LE) is the only contract, a parity test
   guards drift (no new dep). OGAR's note leans (a) ("belongs in contract");
   (b) is lighter and keeps OGAR dep-free. Recommend (b) now, (a) at a
   deliberate consolidation.
3. **FMA/anatomy domain:** fold into Health `0x09XX`, or mint a dedicated
   anatomy domain block (append-only reserved high byte)?

## Cross-refs

OGAR `crates/ogar-vocab/src/lib.rs` (`CODEBOOK`/`ConceptDomain`/`LabelDTO`/
`source_domain`), OGAR `CLAUDE.md` "Tier interpretation 256×256 CENTROID TILE" +
"Codebook scoping = the class routing prefix"; `contract::canonical_node`
(`CLASSID_OSINT`/`CLASSID_FMA`/`BUILTIN_READ_MODES`), `contract::codebook`
(D-GV2-2 per-family), `contract::soa_graph` (`OSINT_GOTHAM`/`FMA_ANATOMY`),
`contract::aiwar`; `guid-v2-tail-per-family-codebook-v1.md`;
`E-UNIFORM-MORTON-TILE-PYRAMID`.
