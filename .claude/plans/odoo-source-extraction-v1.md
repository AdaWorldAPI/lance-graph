# odoo-source-extraction-v1 — TIER-1 Odoo source extraction → `OdooConfidence::Extracted` backing for `D-ODOO-BP-1b` (sub-plan of `D-ODOO-BP-1f`)

> **Status:** SHIPPED (Stage 1 complete 2026-05-28; EXT-1..6 landed; 48/53 entities backed; 5 TIER-2 exemptions documented in `extracted/COVERAGE.md`). **Unfolds `D-ODOO-BP-1f`** from
> `odoo-business-logic-blueprint-v1.md` into a tractable Stage 1 over a
> TIER-1 subset (12 addons of the 622 in `/home/user/odoo/addons/`).
> Validates and backs the L-doc-curated `OdooEntity` consts that
> `D-ODOO-BP-1b` Wave 1-3 just shipped (`9507b36`..`2aca3e3`).
>
> **Confidence:** HIGH on the substrate decision (`Vsa16kF32` + L-doc
> coverage are stable; the extractor is purely additive — it cannot
> regress the curated set, only annotate it). HIGH on the tooling
> substitution (Python `ast` is stdlib, deterministic, and handles all
> ORM class/field shapes we observed). MED on the per-addon yield
> (estimate from the inventory: ~5–8K typed const declarations across
> TIER-1, ~2–3K condensed Rust LOC — actual count depends on how
> aggressively `account` and `account_payment` model graphs are folded
> into shared sub-modules). LOW on the regulation-IRI density per
> entity (anchor count varies wildly: UStVA report has 17 Kennzahlen
> anchors, but most `account.*` entities map to one HGB/UStG ref or
> zero).
>
> **Predecessors:** `D-ODOO-BP-1a` (typed surface — `OdooEntity` +
> sub-types, shipped pre-`9507b36`); `D-ODOO-BP-1b` Wave 1–3 (L1–L15
> L-doc projection — shipped through `2aca3e3`); the Odoo source
> inventory agent report (`2026-05-28`, see this session's transcript).
> Driver epiphanies: `E-CODEBOOK-INHERITS-FROM-OGIT` (regulation rules
> are codebook entries inherited from OGIT — extends `OdooProvenance`
> with a `regulation_iri` slot for UStG/HGB/GoBD/AO refs);
> `E-SAVANT-COMPOSITION-1` (savants compose over normalized typed
> DTOs — extracted-confidence entities widen the substrate v2 reasoners
> can read).
>
> **Anchored iron rules:** `I-VSA-IDENTITIES` (extraction emits
> identity-bearing const data — `model_name`, `account_code`,
> `tax_kennzahl` — not bundled content; codebook entries inherit from
> OGIT per `E-CODEBOOK-INHERITS-FROM-OGIT`). "Consult before guess"
> (the inventory IS the curated surface for what to extract first;
> TIER-2 addons are explicitly deferred, not silently skipped).
> "Audit-by-construction" (every extracted entity carries its
> `(path, line_range)` provenance + a regulation IRI when a German
> rule is the anchor — no untraceable claims).

## The diagnosis

`D-ODOO-BP-1b` Wave 1–3 just landed 15 typed lane modules
(`odoo_blueprint::l{1..15}`) with `OdooConfidence::Curated` provenance
tied to L-doc line ranges. That gives the savant-relevant filter — the
slice the v2 reasoners read.

**Missing**: the exhaustive backing. The inventory found:

- **622 addons**; 2 346 `Model` + 400 `TransientModel` + 395
  `AbstractModel` = **3 141 ORM class definitions** across **989 K
  Python LOC** (7 888 files).
- `l10n_de` is **one community addon** bundling both SKR03 (1 274
  accounts) and SKR04 (1 192 accounts); the 242-row tax tables; and
  the full UStVA `account.report` (Kennzahlen Kz.81..95).
- Eight of nine German concept anchors located in source; the ninth
  (ELSTER) is Enterprise-only and explicitly absent.
- **`tree-sitter` not installed**; the doc comment in
  `odoo_blueprint/mod.rs:321` naming it as the extraction substrate is
  stale. Python's stdlib `ast` module is the correct path — it handles
  every ORM shape we'll meet (decorated methods, `fields.X(...)` with
  keyword args, `_inherit` lists, `_sql_constraints` tuples).

The L-doc curation is the *what matters*; the source is the *what is*.
Stage 1 closes the gap on TIER-1 — the addons the German MedCare /
SMB-Office accounting blueprint actually needs — and leaves
TIER-2 (the 218 non-DE l10n shards, 25 payment providers, POS, HR,
website) for later stages once the extractor + provenance shape are
proven.

## Scope (decisions ratified 2026-05-28)

- **TIER-1 addon set (12)**: `account`, `account_payment`, `l10n_de`,
  `product`, `stock`, `uom`, `base`, `analytic`, `purchase`, `sale`,
  `account_peppol`, `account_edi_ubl_cii`. Rationale: these cover the
  full German accounting flow (Kontenrahmen → Steuersätze → Vorsteuer /
  USt → UStVA → GoBD lock + e-invoice EN 16931) plus the ORM roots
  (`base`, `res.partner`, `res.company`) the others depend on. Stock
  + product + analytic + sale + purchase are the value-flow chain into
  the GL.
- **Tooling**: Python stdlib `ast` module (no install, deterministic).
  Tree-sitter referenced anywhere in the codebase (`mod.rs:321` doc
  comment) is corrected to reflect the substitution.
- **Extractor crate site**: `tools/odoo-blueprint-extractor/` (Python
  package, sits next to `tools/dto-class-check/`). Runs offline against
  `/home/user/odoo/addons/`, emits Rust source into
  `crates/lance-graph-ontology/src/odoo_blueprint/extracted/`.
- **Output module layout**: per-addon Rust modules under
  `odoo_blueprint::extracted::{account, l10n_de, …}`. Each addon yields
  one file with all its `Model` / `TransientModel` / `AbstractModel`
  entities as `OdooConfidence::Extracted` consts. The lane modules
  (`l{1..15}`) remain canonical for `Curated`; `extracted::*` is
  additive — they MAY share entity names but never share const
  identifiers (e.g. `EXT_ACCOUNT_FISCAL_POSITION` vs lane-module
  `FISCAL_POSITION_L9`).
- **Coverage target Stage 1**: every `Model` class in TIER-1 addons
  yields one `OdooEntity` const (field/method projection per
  D-ODOO-EXT-2). `TransientModel` + `AbstractModel` are extracted into
  the same files but marked via an extra enum variant
  (`OdooEntityKind::Transient` / `Abstract` / `Model`) added in
  D-ODOO-EXT-3 — they participate in inheritance resolution but skip
  the persistence-row contracts.
- **Stage 2 deferral (not this plan)**: TIER-2 addons (POS, HR,
  website, fleet, maintenance, non-DE l10n, payment providers) wait
  until D-ODOO-EXT-1..6 are shipped and the extractor's per-construct
  fallback rate stays under 5 %.

## The extractor (`tools/odoo-blueprint-extractor/`)

Python 3 package. Single entrypoint `python -m odoo_blueprint_extractor
--addons /home/user/odoo/addons --tier 1 --out
crates/lance-graph-ontology/src/odoo_blueprint/extracted/`. Internal
shape:

```
parsers/
  classes.py       — visits ClassDef; classifies as Model/Transient/Abstract by base
  fields.py        — visits Assign rhs where Call.func.attr == 'Char'/'Many2one'/…;
                     extracts kw args (comodel_name, required, compute, default, …)
  methods.py       — visits FunctionDef; classifies by name prefix (_compute_, _check_,
                     action_, _onchange_) + decorator scan
  decorators.py    — extracts @api.depends/@api.constrains/@api.onchange arg tuples
  state_machine.py — detects fields.Selection state='state' assignments + transition methods
  constraints.py   — extracts _sql_constraints tuples + @api.constrains methods
  regulation.py    — pattern-matches docstrings/comments/field-help against UStG/HGB/AO
                     anchor table → emits regulation_iri set
emitters/
  rust.py          — formats one OdooEntity const per class (literal Rust output)
  module.py        — per-addon mod.rs aggregator
audit/
  fallback_log.py  — every construct the parser couldn't classify gets logged;
                     report at end with `--audit` flag
```

Parser is purely structural — no semantic interpretation. When a field
construct is unrecognized (rare; e.g. custom `fields.SomethingExotic`)
the emitter writes `OdooFieldKind::Other` + logs to
`audit/fallback.json`. Stage 1 success bar: the audit log shows < 5 %
`::Other` rate per addon.

## Provenance enhancement — `regulation_iri`

Per `E-CODEBOOK-INHERITS-FROM-OGIT`, regulation rules are codebook
entries. `OdooProvenance` gains one slot:

```rust
pub struct OdooProvenance {
    pub l_doc: &'static str,
    pub l_doc_lines: (u32, u32),
    pub odoo_source: &'static [OdooSourceRef],
    pub confidence: OdooConfidence,
    /// German tax/accounting law anchors that bind this entity's
    /// semantics — UStG §15 / HGB §238 / GoBD / AO refs as IRIs into
    /// the OGIT-inherited regulation codebook.
    pub regulation_iri: &'static [&'static str],
}
```

D-ODOO-EXT-3 adds this slot + back-fills `&[]` to every existing
`OdooEntity` const in `l{1..15}` (Wave 1-3 deliverables). The
extractor populates it from the `parsers/regulation.py` table:
strings like `"ogit:regulation/de/ustg/15"` (Vorsteuerabzug),
`"ogit:regulation/de/hgb/238"` (Buchführungspflicht),
`"ogit:regulation/de/gobd"` (audit trail),
`"ogit:regulation/de/ao/146a"` (Verfahrensdokumentation),
`"ogit:regulation/eu/en16931"` (e-invoice standard).

The IRI list is the canonical OGIT codebook handle — content lives in
the OGIT registry, not duplicated here. This is exactly the
`I-VSA-IDENTITIES` pattern: identity in const data, content in tables.

## Stage-1 deliverables

| D-id | Description | Site | LOC | Conf | Status |
|---|---|---|---:|:--:|:--:|
| **D-ODOO-EXT-1** | Python `ast` extractor scaffold (`tools/odoo-blueprint-extractor/`) — parsers + emitters + audit logger; smoke test on one addon (`uom`, 1 model) | `tools/odoo-blueprint-extractor/` | 800 | HIGH | Queued |
| **D-ODOO-EXT-2** | TIER-1 extraction pass: 12 addons → `odoo_blueprint::extracted::*` module tree; `<5%` `OdooFieldKind::Other` rate; `<2%` `OdooMethodKind::Helper` fallback for non-conventional method names | `crates/lance-graph-ontology/src/odoo_blueprint/extracted/` | 5000–8000 | HIGH | Queued |
| **D-ODOO-EXT-3** | `OdooProvenance.regulation_iri` slot + `OdooEntityKind::{Model,Transient,Abstract}` variant; back-fill `&[]` / `Model` to every existing const in `l{1..15}` (Wave 1-3 set); update doc comment `mod.rs:321` (tree-sitter → Python `ast`) | `crates/lance-graph-ontology/src/odoo_blueprint/mod.rs` + lane modules | 200 | HIGH | Queued |
| **D-ODOO-EXT-4** | `l10n_de` specifics: emit `OdooAccountChart` consts for SKR03 + SKR04 (CSV → const array of `(code, name, account_type, tag_ids)`); emit `OdooTaxKennzahl` consts for the 17 UStVA boxes (Kz.81..95); GoBD audit-trail flag wired through `res_company` extension | `crates/lance-graph-ontology/src/odoo_blueprint/extracted/l10n_de.rs` | 1500 | HIGH | Queued |
| **D-ODOO-EXT-5** | Curated-vs-extracted reconciliation pass: for each `(model_name)` present in both `l{1..15}` and `extracted::*`, emit a `pairing.rs` cross-reference table — `(curated_ref, extracted_ref)` tuples for every Wave 1-3 entity; flag mismatches (curated field count vs extracted field count, missing methods) into `audit/pairings.json` for human review; default preserve curated as canonical (per BP-1 plan §"merge ordering") | `crates/lance-graph-ontology/src/odoo_blueprint/extracted/pairing.rs` | 300 | MED | Queued |
| **D-ODOO-EXT-6** | Honest-coverage report: per lane (L1..L15), how many curated entities now have extracted backing, how many extracted entities have no curated equivalent (= TIER-1 surplus surface); land as `crates/lance-graph-ontology/src/odoo_blueprint/extracted/COVERAGE.md` + a `coverage_test()` that fails if any L1..L15 lane drops below 80 % extracted-backing | `extracted/COVERAGE.md` + lane test | 200 | MED | Queued |

Total Stage 1 LOC: ~8 000–10 000 (mostly D-ODOO-EXT-2's generated
const trees; the extractor itself is ~800 Python LOC).

## Execution ordering

1. **D-ODOO-EXT-3 first** — provenance shape must stabilize before
   any extracted const lands. Mechanical edit across the 15 lane
   modules; smoke-tested by `cargo test -p lance-graph-ontology`.
2. **D-ODOO-EXT-1 in parallel with EXT-3** — the extractor scaffold
   has no dep on the provenance shape (it produces Rust source text;
   the const struct it writes against is fixed by EXT-3). One agent
   on each, then merge.
3. **D-ODOO-EXT-2 in 3 waves** by addon-group to keep diffs reviewable:
   - **Wave A**: `base`, `uom`, `product`, `analytic` (foundation)
   - **Wave B**: `account`, `account_payment`, `purchase`, `sale`, `stock` (value-flow chain)
   - **Wave C**: `l10n_de`, `account_peppol`, `account_edi_ubl_cii` (DE-specific + e-invoice)
   Each wave: extractor run → `cargo check` → `cargo test
   -p lance-graph-ontology` → commit.
4. **D-ODOO-EXT-4 right after Wave C** — `l10n_de` data is the
   heaviest extraction (CSV + XML + Python), worth its own deliverable.
5. **D-ODOO-EXT-5 after EXT-2 lands** — pairing table needs both
   curated + extracted sides present.
6. **D-ODOO-EXT-6 closes Stage 1** — coverage report + lane-test gate.

## Risks / open questions

- **Naming collisions**: `OdooEntity` const idents in
  `extracted::account` will collide with anything human-named in
  `l3.rs` if we're not careful. Mitigation: extracted consts are
  prefixed `EXT_` (e.g. `EXT_ACCOUNT_MOVE`) so they never share a
  symbol with a curated lane const. Cross-reference lives in
  D-ODOO-EXT-5's `pairing.rs`.
- **Inheritance resolution**: a class with `_inherit = 'account.move'`
  in `l10n_de` is an extension, not a redefinition. The extractor's
  `classes.py` MUST detect `_inherit` (str or list) and emit the entity
  as an *extension fragment* — fields/methods only, no full DTO — that
  references the base entity. Curated lane modules already do this
  (e.g. `ACCOUNT_ACCOUNT_TAG_L15` extends `l3.rs`'s base). Open
  question: do we want a typed `OdooEntityExtension` variant, or just
  reuse `OdooEntity` with `fields` being a delta? Decision deferred to
  EXT-3 design.
- **Regulation-IRI source**: which list of UStG/HGB/AO/EN sections do
  we treat as canonical? Conjecture: the OGIT regulation registry
  (already a workspace artifact, per `E-CODEBOOK-INHERITS-FROM-OGIT`).
  Open: confirm the IRI scheme with OGIT before EXT-3 ships.
- **`account.report` projection**: UStVA is a 1 106-line XML with 17+
  `account.report.line` records as Kennzahlen Kz.81..95. Whether these
  land as `OdooEntity` consts (semantically: they're report-line
  records, not models) or as a separate `OdooTaxKennzahl` typed
  surface (EXT-4) — pre-decided to use the separate surface, since
  Kennzahlen are first-class regulatory anchors and conflating them
  with ORM rows loses precision.
- **Stage 2 trigger**: TIER-2 (POS, HR, website, payment providers,
  non-DE l10n) opens only after EXT-6 reports < 5 % fallback rate on
  Stage 1. If higher, the extractor needs hardening before scaling
  out.
- **The 989 K LOC ceiling**: full extraction across all 622 addons is
  not a goal of any current plan. The TIER-1 + TIER-2 union is
  ~25–30 addons, which projects to roughly 15–25 K Rust const-LOC.
  Beyond that, deferred indefinitely — the L-doc curated set is the
  primary substrate; extraction is the audit backing for the slice the
  reasoners actually read.

## Board updates that land with this plan

Per CLAUDE.md board-hygiene rule, the commit that creates this file
also:

- PREPENDs an entry to `.claude/board/INTEGRATION_PLANS.md` pointing at
  this file.
- (Future PR) PREPENDs `D-ODOO-EXT-{1..6}` rows to
  `.claude/board/STATUS_BOARD.md` once the first wave begins.

Epiphanies cross-referenced:

- `E-CODEBOOK-INHERITS-FROM-OGIT` (regulation rules live in the OGIT
  codebook — drives EXT-3's `regulation_iri` slot).
- `E-SAVANT-COMPOSITION-1` (typed DTOs are the substrate v2 reasoners
  compose over — extracted entities widen that substrate without
  changing its shape).
