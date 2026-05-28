# ruff-py-dto-odoo-retarget-v1

**Status:** PROPOSAL. Concrete unification target for `E-UNIFY-CROSS-DOMAIN-CONCERNS-1` and `E-RUFF-PYTHON-DTO-CHECK-IS-THE-PORTING-ENGINE-1`. Retargets `AdaWorldAPI/ruff/crates/ruff_python_dto_check` (a Flask→Rust/axum/SeaORM porting engine) at the Odoo extraction problem, replacing the stdlib-`ast` extractors shipped in `odoo-source-extraction-v1` Stage 1.

**Confidence:** HIGH on the vendoring path (the crate is in-tree at `AdaWorldAPI/ruff`, MIT-licensed, depends only on ruff's own parser + AST). HIGH on the design fit (CODEGEN-DESIGN.md explicitly names Odoo as the next target after Flask). MED on the `class_with_base` matcher PR effort (the README scopes it as future work; depends on whether the ruff team accepts an upstream PR or we ship a forked variant). MED on the per-addon target-spec authoring (12 TIER-1 addons × ~15 minutes per profile/spec = ~3-4h hand-authoring).

**Plan file:** `.claude/plans/ruff-py-dto-odoo-retarget-v1.md` (this file).

**Predecessors:**
- `odoo-source-extraction-v1` Stage 1 SHIPPED (EXT-1..6, stdlib-`ast` baseline)
- `odoo-business-logic-blueprint-v1` Wave 1-3 SHIPPED (curated `OdooEntity` consts — the canonical surface; EXT_ consts are additive)
- `.claude/odoo/{BRIEFING,BRIEFING-GAP}.md` + L1-L15 lane briefings (the **dual-axis classification spec** + 15 worked examples — the harvest's golden source of truth)
- `E-UNIFY-CROSS-DOMAIN-CONCERNS-1`, `E-RUFF-PYTHON-DTO-CHECK-IS-THE-PORTING-ENGINE-1` (driver epiphanies, same commit)
- AdaWorldAPI/ruff (the engine)

## The dual-axis the L-docs encode (the critical bit I missed in v0 of this plan)

`.claude/odoo/BRIEFING.md` defines a **dual-axis classification** that EVERY extracted rule must carry:

- **AXIS-A — DETERMINISTIC → Rust port.** Closed-form rule (balance check, sequence format, `tax = base × rate`, `residual = debit − credit`). Output: rich-AST spec → port verbatim to Rust.
- **AXIS-B — HEURISTIC / INFERENTIAL → delegate to lance-graph thinking.** Evidence-weighted multi-factor logic (reconciliation matching, fiscal-position resolution, next-best-action, anomaly detection, dunning judgement). Output: **delegation tuple** = `(ReasoningKind, InferenceType, SemiringChoice, ThinkingStyle)` with `ThinkingStyle` **inherited from the OGIT family** via `resolve_odoo_to_family()`.
- **HYBRID** — deterministic guard wrapping a heuristic core (e.g. "balance MUST be zero" [AXIS-A] gating "suggest which lines to adjust" [AXIS-B]). Both halves tagged.

Each AXIS-B rule produces a **Savant seed** of shape: `SAVANT: name=<x> family=<0x..|None> reasoning=<Kind> inference=<Type> semiring=<Choice> style=<cluster> — <why-delegated>`.

**This maps DIRECTLY to ruff-py-dto's HandlerKind algebra:** the 12 Flask HandlerKinds emerge from `(output × inputs)`; the Odoo HandlerKinds emerge from `(odoo class × rule type × axis)`. The Odoo retarget must produce BOTH:
- **AXIS-A kinds** (e.g. `field_compute_balanced`, `field_constrain_sequence`, `report_render_susa`) → emit `EXT_*` `OdooEntity` const (the original plan).
- **AXIS-B kinds** (e.g. `fiscal_position_resolve`, `reconcile_match`, `anomaly_detect`, `nba_propose`) → emit `SAVANT_*` `SavantSeed` const with the delegation tuple — bound to the savant roster in `lance-graph-callcenter::savants` + `lance-graph-contract::callcenter::ogit_uris`.
- **HYBRID kinds** → emit both consts, with the AXIS-A const carrying a `delegates_to: Option<&'static SavantSeed>` cross-ref pointer.

**Anchored iron rules:**
- `I-VSA-IDENTITIES` (identity in const data; EXT consts emit into typed Rust)
- `E-CODEBOOK-IS-THE-COMPILATION-TARGET-1` (Odoo TargetSpec emits to the codebook shape)
- "audit-by-construction" (the 5 calibration lints become the gate; the 6th — `extractor-gap` — points back at the SOURCE config to extend)
- "consult before guess" (the lesson from this epiphany itself: read DESIGN doc + target example before writing a duplicate)

## Scope

**The retarget delivers:**

1. **A new Rust crate** `tools/odoo-blueprint-extractor-rs/` that depends on `ruff_python_dto_check` and ships:
   - An Odoo `ExtractionProfile` (config JSON or built-in default)
   - An Odoo `TargetSpec` (TOML)
   - A CLI binary `odoo-extract` that runs the harvest + codegen + calibration pipeline against `/home/user/odoo/addons/<addon>/`
   - A `build.rs` integration into `lance-graph-ontology` that emits `EXT_*` Rust consts into `crates/lance-graph-ontology/src/odoo_blueprint/extracted/<addon>.rs`

2. **An upstream PR to `AdaWorldAPI/ruff`** adding `MatchKind::ClassWithBase` to `crates/ruff_python_dto_check/src/config.rs` + `src/matcher/class_with_base.rs` + golden test against a minimal Odoo addon fixture. README's "out of scope" list explicitly names this as planned.

3. **Migration of the Python extractor scripts** in `tools/odoo-blueprint-extractor/data_extractors/*.py` to thin compat shims (or deletion, once the Rust pipeline reaches parity). The Python extractors keep working until the Rust pipeline lands; they don't get deleted preemptively.

**Out of scope for v1:**
- CSV-to-Rust (SKR03/04 chart) — `odoo-skr-extract.py` stays in `woa-rs/.claude/tools/` until a CSV matcher family lands in ruff-py-dto.
- XML extraction (Odoo `<button>` views, QWeb templates) — stays Python until an XML matcher family lands.
- Python-side DTO emission from the OGIT codebook — separate concern; gets its own plan if/when needed.
- TIER-2 addons (POS, HR, website, fleet, maintenance, non-DE l10n, payment providers) — v1 ships TIER-1 (12 addons). TIER-2 lands as v2 once the v1 pipeline is stable.

## Deliverables

| D-id | Description | Site | LOC | Conf | Wave |
|---|---|---|---:|:--:|:--:|
| **D-RPYDTO-1** | Workspace dep wiring: `[workspace.dependencies] ruff_python_dto_check = { git = "https://github.com/AdaWorldAPI/ruff", rev = "<pin>" }` + new crate scaffold `tools/odoo-blueprint-extractor-rs/{Cargo.toml,src/lib.rs,src/bin/odoo-extract.rs}` | workspace `Cargo.toml`, new crate dir | 200 | HIGH | α |
| **D-RPYDTO-2a** | **Upstream PR #1 to AdaWorldAPI/ruff** — `ruff_python_dto_check` extension: (1) add `MatchKind::ClassWithBase` to `config.rs` + `matcher/class_with_base.rs` impl; (2) **add first-class `Axis` enum** (`Deterministic` / `Heuristic` / `Hybrid`) on `RouteContract` so EVERY harvested fact carries its axis classification; (3) **add `DelegationTuple` struct** (`reasoning_kind: String`, `inference_type: String`, `semiring: String`, `thinking_style: String`, `ogit_family: Option<String>`, `why_delegated: String`) on the contract for AXIS-B / HYBRID facts; (4) extend the `ExtractionProfile` config schema with an `axis_rules` block (selectors → axis tag); (5) golden tests against minimal `class X(models.Model)` fixture + dual-axis fixture (one AXIS-A field-compute + one AXIS-B fiscal-position-resolve). **The axis is FIRST-CLASS output, not a downstream interpretation** | upstream `crates/ruff_python_dto_check/src/{config,contract,matcher/*}` | 700 | MED | α |
| **D-RPYDTO-2b** | **Upstream PR #2 to AdaWorldAPI/ruff** — `ruff_python_codegen` extension: emit the `Axis` + `DelegationTuple` as **structured doc-comment annotations** on generated code so a downstream Rust build can read the axis back from the generated source. Shape: `#[doc = "axis: heuristic"] #[doc = "delegation: {reasoning_kind=...,inference_type=...,...}"]` on each emitted `EXT_*` / `SAVANT_*` const, OR an explicit `AxisAnnotation` struct in the codegen output. Plus extend `Generator` with an `annotate_axis(axis: &Axis, tuple: Option<&DelegationTuple>)` hook so the emission path threads the classification through. **`ruff_python_codegen` becomes axis-aware emission, not just style-preserving round-trip.** | upstream `crates/ruff_python_codegen/src/generator.rs` + `src/lib.rs` (new `AxisAnnotation` API) | 400 | MED | α |
| **D-RPYDTO-3** | Odoo `ExtractionProfile` (JSON config) — `class_with_base` matching `models.Model` / `models.TransientModel` / `models.AbstractModel`; emit `_name`, `_inherit`, `_inherits`, `_description`, `_order`, `_rec_name`; per-class field-assignment harvest for `fields.Char/Boolean/Integer/Float/Many2one/One2many/Many2many/Selection/Date/Datetime/Binary/Html/Text/Reference/Json/Monetary/Serialized` with their kwargs (`string=`, `required=`, `default=`, `compute=`, `inverse=`, `search=`, `related=`, `domain=`, `track_visibility=`); per-method decorator harvest for `@api.depends`, `@api.constrains`, `@api.onchange`, `@api.model`, `@api.model_create_multi`, `@api.returns` | `tools/odoo-blueprint-extractor-rs/configs/odoo.extraction.json` | 250 (JSON) | HIGH | β |
| **D-RPYDTO-4** | Odoo `TargetSpec` (TOML) — `id = "odoo-blueprint-rs"`, `models_root = "crate::odoo_blueprint::extracted"`, model mappings per addon (`account.move` → `account::AccountMove`, etc.), `emit_kinds` whitelist for the Odoo-derived HandlerKinds | `tools/odoo-blueprint-extractor-rs/targets/odoo-blueprint-rs.target.toml` | 300 (TOML) | HIGH | β |
| **D-RPYDTO-5** | **Cross-framework unified `HandlerKind` registry + dual-axis classifier.** Per `E-UNIFY-CROSS-DOMAIN-CONCERNS-1`: Flask `list_for_tenant`, Odoo `model_search_with_scope`, Django `queryset_list_view`, and FastAPI `dependency_scoped_list` are NOT separate per-framework taxonomies — they UNIFY into a single emergent `HandlerKind` set (`scoped_list`, `scoped_detail`, `form_post_mutate`, `compute_with_validation`, `inferential_resolve`, ...) derived from one `(output × inputs × axis)` algebra. ExtractionProfile maps framework-specific syntax (Flask `@bp.route`, Odoo `@api.depends`, Django `@require_GET`) → unified kind; TargetSpec maps unified kind → target-language code. Each kind is tagged AXIS-A / AXIS-B / HYBRID per `.claude/odoo/BRIEFING.md`. **AXIS-A seed kinds** (deterministic, cross-framework): `scoped_list`, `scoped_detail`, `form_post_mutate`, `soft_delete`, `toggle_bool_field`, `compute_with_validation`, `state_machine_transition`, `report_render`, `sequence_format`, `cron_job`. **AXIS-B seed kinds** (heuristic → Savant): `inferential_resolve` (covers `fiscal_position_resolve`, partner-categorisation, route-disambiguation), `evidence_match` (covers `reconcile_match`, mail-intent-classify, attachment-pair), `anomaly_detect`, `nba_propose`, `dunning_escalate`, `reservation_choose`. **HYBRID kinds**: `mutate_with_resolve` (deterministic guard + inferential core), `compute_with_inference`. Empirical distribution emerges from harvest across BOTH Flask (woa-rs/.claude/v0.2/) AND Odoo (TIER-1 corpus); the seed set is validated against the union, not per-framework. **Critical: the kind enum is shared across all Python frameworks; adding a new framework adds ExtractionProfile rules + maybe one new emergent kind, NOT a parallel kind taxonomy.** | upstream `ruff_python_dto_check::contract::HandlerKind` (extended) | 800 | MED | γ |
| **D-RPYDTO-6** | **Dual-axis Odoo target emitter** — recipes split by axis: **AXIS-A recipes** emit `OdooEntity { model_name, table_name, fields, computes, constrains, provenance, delegates_to: Option<&'static SavantSeed> }`; **AXIS-B recipes** emit `SavantSeed { name, ogit_family: Option<u8>, reasoning_kind: ReasoningKind, inference_type: InferenceType, semiring: SemiringChoice, thinking_style: ThinkingStyle, source_class: &'static str, source_method: &'static str, provenance: OdooProvenance, why_delegated: &'static str }`; **HYBRID recipes** emit both with cross-ref. The `ThinkingStyle` field is **resolved via `resolve_odoo_to_family()`** at emit time — NOT extracted from source. Lands one recipe per kind in data-shaped `format!` style | `tools/odoo-blueprint-extractor-rs/src/emit/{axis_a,axis_b,hybrid}/*.rs` | 2200 | MED | γ |
| **D-RPYDTO-7** | `build.rs` integration in `lance-graph-ontology` — invokes `odoo-extract harvest --config <profile> --target <spec> --root /home/user/odoo/addons/<addon> --out <addon>.rs` per TIER-1 addon at build time; emits `crates/lance-graph-ontology/src/odoo_blueprint/extracted/<addon>.rs` with `EXT_*` consts | `crates/lance-graph-ontology/build.rs` (new) + `tools/odoo-blueprint-extractor-rs/src/codegen/rust_ontology.rs` | 400 | MED | δ |
| **D-RPYDTO-8** | Calibration lints wired to the Odoo target — adapt the 5 calibration lints (`unmapped-model`, `dropped-fact`, `template-context-mismatch` → `template-context-mismatch` becomes `view-arch-mismatch` for Odoo XML views, `form-field-gap`, `output-kind-mismatch`) plus `extractor-gap`. Lint failures fail `cargo build` of `lance-graph-ontology` | `tools/odoo-blueprint-extractor-rs/src/calibrate.rs` | 300 | MED | δ |
| **D-RPYDTO-9** | Migration of Python extractor scripts — delete `tools/odoo-blueprint-extractor/odoo_blueprint_extractor/data_extractors/*.py` once D-RPYDTO-7 lands per-lane EXT consts at parity with the Python output; keep the package only for the orchestration shim if needed | `tools/odoo-blueprint-extractor/` | -800 (deletion) | HIGH | ε |
| **D-RPYDTO-10** | Documentation update — `tools/odoo-blueprint-extractor/README.md` redirects to `tools/odoo-blueprint-extractor-rs/README.md`; `.claude/plans/odoo-source-extraction-v1.md` gets a "Stage 2 SUPERSEDED by ruff-py-dto-odoo-retarget-v1" status update | docs | 50 | HIGH | ε |

## Wave ordering

- **α (parallel, all three):** D-RPYDTO-1 (crate scaffold), D-RPYDTO-2a (upstream `ruff_python_dto_check` extension — `ClassWithBase` + first-class `Axis` + `DelegationTuple`), D-RPYDTO-2b (upstream `ruff_python_codegen` extension — axis-aware emission). All three are independent; D-RPYDTO-2a + 2b ship as TWO separate PRs to AdaWorldAPI/ruff so they can be reviewed independently. Both can ship via vendor fork on a feature branch if upstream is slow.
- **β (parallel after α):** D-RPYDTO-3 (extraction profile — now CONFIGURES the upstream `axis_rules` block rather than carrying classification logic itself) and D-RPYDTO-4 (target spec — now CONFIGURES the upstream `AxisAnnotation` emission rather than wrapping the generator). Both depend on α's upstream changes landing.
- **γ (sequential):** D-RPYDTO-5 (HandlerKind classifier) before D-RPYDTO-6 (emitter recipes) — the recipes dispatch on kind.
- **δ (parallel after γ):** D-RPYDTO-7 (build.rs integration) and D-RPYDTO-8 (calibration lints) run in parallel.
- **ε (sequential, last):** D-RPYDTO-9 (delete Python extractors) and D-RPYDTO-10 (docs) — only after δ proves parity.

## Acceptance criteria

1. **Parity gate:** `lance-graph-ontology` build emits ≥ 95% of the Rust consts the Stage 1 stdlib-`ast` pipeline emits (the 5% headroom is for genuinely-different shapes that the new pipeline catches and the old missed, OR vice-versa — diffs reviewed in PR).
2. **Calibration green:** `cargo build -p lance-graph-ontology` passes with zero `unmapped-model` / `dropped-fact` / `extractor-gap` lints across all 12 TIER-1 addons.
3. **Round-trip test:** for each TIER-1 addon, `cargo test -p odoo-blueprint-extractor-rs --test golden_<addon>` passes (golden snapshot of harvest output against committed expected JSON).
4. **Upstream PR landed OR vendored:** D-RPYDTO-2's `ClassWithBase` either merged upstream (preferred) or vendored with a clear pin + a TODO to upstream.

## Delegation strategy

- **D-RPYDTO-1** (scaffold): handcraft. ~30 min.
- **D-RPYDTO-2a** (upstream `ruff_python_dto_check` extension: `ClassWithBase` + first-class `Axis` + `DelegationTuple`): handcraft — touching a foreign repo, contract-shape decision, needs care. ~4-5h including golden tests + the dual-axis fixture.
- **D-RPYDTO-2b** (upstream `ruff_python_codegen` extension: axis-aware emission): handcraft, ~2-3h. Smaller surface; the `AxisAnnotation` API + doc-comment emission is mechanical once 2a's contract types are pinned.
- **D-RPYDTO-3, D-RPYDTO-4** (extraction profile + target spec, JSON/TOML authoring — now configuring the upstream first-class axis support): **delegate to Sonnet** with the Flask config + axum target as template, plus the BRIEFING.md dual-axis spec. ~1h Sonnet, ~30 min main-thread review.
- **D-RPYDTO-5** (HandlerKind classifier): handcraft the priority classifier shape; **delegate the kind list discovery to Sonnet** via "run preflight + AST-hash index across TIER-1 corpus, propose kind distribution". ~1h.
- **D-RPYDTO-6** (12 emitter recipes): **delegate to Sonnet** one recipe per kind in parallel; main thread does the first one as template. ~2h Sonnet, ~30 min review.
- **D-RPYDTO-7, D-RPYDTO-8** (build integration + lints): handcraft — load-bearing for build correctness.
- **D-RPYDTO-9, D-RPYDTO-10** (delete + docs): handcraft, fast.

## Risks

- **Upstream PR latency.** D-RPYDTO-2 (`ClassWithBase`) is the critical-path enabler. If `AdaWorldAPI/ruff` upstream is slow, ship the vendor fork on a feature branch and PR-back in parallel. Don't block β on upstream review.
- **HandlerKind seed list miscalibration.** The 12 plausible kinds in D-RPYDTO-5 are a guess; the real distribution emerges from corpus harvest. Plan: run `preflight` against `/home/user/odoo/addons/account/` first, inspect the kind histogram, then commit the kind enum.
- **EXT vs curated conflict.** When extracted shapes disagree with curated, **curated stays canonical** per `odoo-source-extraction-v1` EXT-5 doctrine. Pairing diffs surface in `audit/pairings.json` for human review.
- **Build-time cost.** Running the harvest at every `cargo build` is expensive (~622 addons × ~50ms each = ~30s rebuild floor). Mitigation: cache the NDJSON bundles in `target/extracted/<addon>/` keyed by source mtime; rebuild only the addons whose Python source changed.

## Invariants

- **One engine, N configs.** `ruff_python_dto_check` is the SINGLE Python-extraction engine. The Odoo extraction is a CONFIG + TARGET-SPEC pair, not a new crate of hand-rolled walkers.
- **Curated stays canonical.** EXT_* consts NEVER override curated `OdooEntity` consts; they augment with `OdooConfidence::Extracted` provenance.
- **No project-specific Rust in `ruff_python_dto_check`.** All Odoo-specific logic lives in `tools/odoo-blueprint-extractor-rs/` (the new crate) or in the extraction profile + target spec. The crate stays clean for OpenProject + Django + future Python frameworks.
- **The 5 calibration lints fail the build.** They are not advisory — `unmapped-model` etc. blocks `cargo build` until the SOURCE config is extended.

## Cross-ref

- `E-UNIFY-CROSS-DOMAIN-CONCERNS-1` (the doctrine)
- `E-RUFF-PYTHON-DTO-CHECK-IS-THE-PORTING-ENGINE-1` (the engine description)
- `E-CAPABILITY-FROM-PREDICATE-CONJUNCTION-1` (the 12 HandlerKinds are this principle in the wild)
- `odoo-source-extraction-v1` (Stage 1 SHIPPED; this plan SUPERSEDES Stage 2)
- `odoo-business-logic-blueprint-v1` (curated `OdooEntity` consts that EXT_* augments)
- `AdaWorldAPI/ruff/crates/ruff_python_dto_check/CODEGEN-DESIGN.md` (the engine's design — names Odoo as next target)
- `AdaWorldAPI/ruff/crates/ruff_python_dto_check/examples/rust-axum-seaorm.target.toml` (template for the Odoo TargetSpec)
- `unified-spo-nars-codegen-v1` (Stage 2 trunk — the typed `OdooEntity` shape this emits to is one input to D-USN-1)
- `lance-graph-elixir-frontend-v1` (different layer — Elixir syntax frontend; orthogonal to Python extraction)
