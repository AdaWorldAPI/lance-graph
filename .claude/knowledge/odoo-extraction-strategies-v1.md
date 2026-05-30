# odoo-extraction-strategies-v1 — The three proposer legs into the Odoo SPO substrate

> **READ BY:** anyone proposing a new extractor / discoverer / proposer for Odoo entities; the council `prior-art-savant`; `style_recipe.rs` rule authors deciding which `OdooConfidence` levels their rules can read from.
> **Status:** FINDING (the three legs exist or are spec'd; this doc names the doctrine).
> **Authored:** 2026-05-29.
> **Companion docs:** `odoo-blueprint-inventory-v1.md` (the 66-entity corpus), `odoo-extraction-tools-v1.md` (what backs each leg).

---

## 0. The doctrine in one paragraph

Odoo business logic enters the SPO substrate through **exactly three proposer legs**: (1) **Curated** — humans project L-doc prose into typed `OdooEntity` consts; (2) **Extracted** — Python AST walks `/home/user/odoo/addons/*` source and emits candidate consts; (3) **ArmDiscovered** — streaming pair-stats (and optionally Aerial+ neural-symbolic) mine runtime tabular data for co-correlations expressible as truth-carrying SPO triples. Each leg has its own throughput regime, confidence basin, and council-gate posture. The three legs converge on the **same typed `OdooEntity` SoA + same `SpoStore` substrate + same `TruthValue (f, c)` calibration** — they DIFFER in what they can see, not in what they produce. Downstream consumers (`derive_style_recipe`, `op_emitter`, the cognitive-shader-driver) read from the union; the council ratification gate is what lets candidates from a noisier leg become equally first-class in the codegen path.

---

## 1. Leg 1 — Curated (D-ODOO-BP-1b)

### What it sees

Human-authored L-doc prose in `.claude/odoo/L*.md` — 15 L-docs covering Odoo's 15 business lanes (posting, reconciliation, tax, SKR chart, payment, sales/purchase, stock, product, partner-fiscalpos, analytic, lock-date, multi-company-currency, stock-valuation, HR, tax-repartition). Each L-doc carries line-ranged prose describing what an Odoo entity IS in business terms — its fields' meanings, its methods' intentions, its state machine's semantics, its regulatory anchors (UStG / HGB / GoBD / AO / EN 16931).

### What it emits

66 `pub const NAME: OdooEntity` declarations across `crates/lance-graph-ontology/src/odoo_blueprint/l{1..15}.rs`, totaling 11,563 LOC. Each const carries `confidence: OdooConfidence::Curated`, line-ranged `OdooProvenance::l_doc_lines` citation back to the L-doc, and (where applicable) `regulation_iri: &[...]` pointing into the OGIT-inherited regulation codebook.

### Method

Mechanical L-doc-prose → typed-const projection. One Sonnet agent per lane (5 agents in parallel for L1-L5 = Wave 1; 5 agents for L11-L15 = Wave 3; the Wave-2 L6-L10 band ran ~3 agents with L9 carved out for the canonical FiscalPositionResolver projection). Each agent's deliverable is *one lane's* `OdooEntity` consts.

**Per-agent algorithm:**
1. Read the lane's L-doc (one of `.claude/odoo/L{1..15}-*.md`).
2. Read `odoo_blueprint/mod.rs` to bind the `OdooEntity` typed surface (fields, methods, decorators, state machine, constraints, provenance).
3. For each Odoo entity the L-doc names in prose:
   - Map its prose fields to `OdooField { name, kind: OdooFieldKind::*, target, required, computed, depends, semantic_role: OdooSemanticRole::* }` — semantic_role is the human's read of the field's business meaning (Identity / Reference / Quantity / Date / Policy / Status / Money / Document / Address / Tax / Audit / Other).
   - Map its prose methods to `OdooMethod { name, kind: OdooMethodKind::*, return_kind: OdooReturnKind::*, triggers: &[...] }`.
   - Map its `@api.*` decorators to `OdooDecorator { kind, targets }`.
   - Map its state machine (if present) to `OdooStateMachine { state_field, states, transitions }`.
   - Map its `_sql_constraints` and `@api.constrains` clauses to `OdooConstraint`.
   - Stamp `OdooProvenance { l_doc, l_doc_lines: (start, end), odoo_source: &[OdooSourceRef { path, line_range }], confidence: Curated, regulation_iri: &[...] }`.
4. Write the lane's `tests` module with a few smoke tests (typically: assert entities const exists, assert a specific field's kind, assert a specific method's return kind).

### Throughput / confidence

- **Throughput:** 3-6 entities per agent-session (~1 hour wall-clock at human-supervised Sonnet pace). 5 agents in parallel = ~30 entities per Wave.
- **Confidence:** HIGH. The L-doc is the human-filtered semantic surface; projection is mechanical; reviewer trims (Wave-2 → Wave-3 transition added these) catch the few edge cases.
- **Failure modes:** L-doc prose may itself be wrong or out of date relative to live Odoo source — `pairing.rs::CURATED_EXTRACTED_PAIRS` is the audit table that surfaces these conflicts. **Canonical rule: curated wins on conflict, but the conflict is logged for human ratification.**

### Council posture

`OdooConfidence::Curated` is the **highest-trust tier**; council ratification is NOT required for Curated entries. Per PR #433's council-bypass exemption, human-authored rulings ship without a council pass. (The §11 rulings of PR #434 were author-stated under the same exemption; the same applies here.)

---

## 2. Leg 2 — Extracted (D-ODOO-EXT-2)

### What it sees

Python source in `/home/user/odoo/addons/{account, account_payment, l10n_de, product, stock, uom, base, analytic, purchase, sale, account_peppol, account_edi_ubl_cii}/` — the 12 TIER-1 addons covering German accounting flow + ORM roots + value chain + e-invoice. TIER-2 addons (POS, HR, website, payment providers, non-DE l10n, asset depreciation, MRP, project) are explicitly deferred.

### What it emits

11 `.rs` files in `crates/lance-graph-ontology/src/odoo_blueprint/extracted/{account, account_edi_ubl_cii, account_payment, account_peppol, analytic, base, l10n_de, l10n_de_chart, l10n_de_kennzahlen, pairing, product, purchase, sale, stock, uom}.rs`, totaling **99,209 LOC** — the bulk of which is `l10n_de_chart.rs` (24,712 LOC of SKR03/SKR04 chart-of-accounts data). Each emitted const carries `confidence: OdooConfidence::Extracted` and a `EXT_` symbol-prefix to avoid collision with the curated lane consts.

### Method

Stdlib-only Python 3 (`ast` module) — `tools/odoo-blueprint-extractor/`. **No tree-sitter, no Odoo runtime, no network.** Pure static analysis on the on-disk Python source. Per-addon walker:

1. `os.walk` the addon's `models/` directory.
2. For each `.py` file, parse the AST via `ast.parse(source)`.
3. For each `class X(models.Model | models.TransientModel | models.AbstractModel)`:
   - Bind the `_name`/`_inherit` attribute to the `model_name`.
   - Bind the base class to `OdooEntityKind::{Model, Transient, Abstract}`.
   - Walk class body for field assignments (`name = fields.Char(...)`) → `OdooField` entries.
   - Walk class body for method definitions; classify by name prefix (`_compute_`, `_inverse_`, `_check_`, `_onchange_`, `action_`, `_cron_`) + decorators → `OdooMethod` entries.
   - Walk class body for decorators on methods (`@api.depends`, `@api.constrains`, `@api.onchange`, `@api.model`, `@api.model_create_multi`) → `OdooDecorator` entries.
   - Walk class body for state-field selections + transition methods → `OdooStateMachine` (best-effort; many transitions are implicit in `write({'state': ...})` calls and may need hand-trim).
   - Bind `_sql_constraints` and `@api.constrains` methods to `OdooConstraint` entries.
4. Emit Rust source via the `emitters/` module: one `pub const EXT_<NAME>: OdooEntity = OdooEntity {...}` per class.
5. Log any field-kind fallbacks (`fields.Image`, `fields.Properties`, etc. that aren't in `OdooFieldKind`) to `--audit /tmp/fallback.json` for human review.

The CLI is `python -m odoo_blueprint_extractor --addons /home/user/odoo/addons --addon <name> --out <path> [--audit <path>]`.

### Throughput / confidence

- **Throughput:** ~10s entities per addon-run (≤1 sec per file; full TIER-1 sweep is ~minutes).
- **Confidence:** MED. AST sees what's *literally written*; misses runtime behavior (dynamic `_inherit`, registry-injected methods, computed-field side-channels). Stage-2 dark-atom gap is here: `return_kind` and `semantic_role` are conservatively inferred and often default to `Unit` / `Other` — see `odoo-blueprint-inventory-v1.md` §6.
- **Failure modes:** field-kind fallback rate is the canonical metric (gated `< 5%` per EXT-6). Methods with raise-but-no-classifier-match get `OdooMethodKind::Other`. The `OdooConfidence::Extracted` stamp tells consumers to expect this noise.

### Council posture

`OdooConfidence::Extracted` is **second tier**; ratification is NOT required for routine extracted entries, but conflicts with the curated set surface in `pairing.rs::CURATED_EXTRACTED_PAIRS` and are flagged for human ratification (NOT automatic council convening — `pairing.rs` is a manual review surface). Coverage gate (EXT-6, `extracted::coverage`) enforces 80% per-lane backing; below that, the lane fails CI.

### Coverage today (per EXT-6 report 2026-05-28)

- 48/53 curated entities have TIER-1 extracted backing (90.6% workspace-wide).
- L1-L10 all at 100% coverage; L11 at 90.6% with 5 explicit TIER-2 deferrals.
- L12-L15 (Wave-3 curated additions) are post-EXT-6 and need a fresh extractor pass to bring into the pairing table.
- 5 TIER-2 deferrals: 4 `hr.*` (HR-base) + 1 `stock.valuation.layer` (deferred to TIER-2 stock-account follow-up).

---

## 3. Leg 3 — ArmDiscovered (this branch, `streaming-arm-nars-discovery-v1.md`)

### What it sees

Streaming runtime tabular data — parquet exports of `account.move`, `account.move.line`, `stock.move`, `sale.order`, etc. — at 20K–200K rows per window. **NOT static source; NOT human prose.** The actual transaction history of the running Odoo instance.

### What it emits

Candidate SPO triples `(subject, predicate, object, frequency, confidence)` produced by:

1. **Stage A — Streaming pair-stats (deterministic trunk).** Per-window sufficient statistics over `(item_i, item_j)` pairs up to antecedent bound `a=2`. Memory bound `O(k²) ≈ 160 KB` at k=200 features. SIMD-amenable; ~2 s per 200K-row window on one core. Outputs `CandidateRule { antecedent, consequent, support, confidence, n }`.
2. **Stage A2 — Optional Aerial+ IPC fan-in.** For high-dimensional sparse feeds (k > 500), a separate Python process running the Aerial+ autoencoder (per Karabulut et al. 2025 §3.3, Algorithm 1) emits the same `CandidateRule` shape via NDJSON-over-Unix-socket. Behind the `arm-aerial` feature flag.
3. **Stage B — Translator.** Pure function `arm_to_nars(rule) -> TruthValue` maps ARM `confidence → NARS frequency f`, `(support × n) / (support × n + k) → NARS confidence c`. The output is a `CandidateTriple { s, p, o, truth, origin: ProvenanceTier::ArmDiscovered }`.
4. **Stage C — Hypothesis test.** Round-trip against `SpoStore`: revise the prior on match (NARS `revise(t_prior, candidate)`), commit a `Contradiction` edge on inversion match (per The Click), queue for ratification on novel.
5. **Stage D — Ratification.** The 5-savant epiphany-brainstorm-council vets novel candidates before they earn `ProvenanceTier::Ratified` and become consumable by `op_emitter`.

### Method

See `.claude/plans/streaming-arm-nars-discovery-v1.md` for the full 18-section spec. The crate `lance-graph-arm-discovery` is **queued, not yet implemented** (D-ARM-1 through D-ARM-12).

### Throughput / confidence

- **Throughput:** 20K-200K rows per window, continuous; theoretical 100K rows/window/sec sustainable on a 16-core machine for k=200 features (target per D-ARM-12 bench).
- **Confidence:** LOW until Stage D ratification (`ProvenanceTier::ArmDiscovered`); HIGH after (`ProvenanceTier::Ratified`). The Jirak-bounded threshold (`I-NOISE-FLOOR-JIRAK`) is the noise-floor gate at Stage A; without it, low-quality candidates would calcify into the substrate via NARS revision. With it, false-positive rate aligns with substrate's noise floor.
- **Failure modes:** weak-dependence violations (Stage A leaks below Jirak floor → calcification), council bottleneck (Stage D throughput is human-bounded), contradiction-commit semantic drift (Stage C requires `spo::truth::Contradiction` primitive to exist — verify at D-ARM-5 entry).

### Council posture

**MANDATORY ratification before codegen.** This is the only leg where Stage D ratification is non-bypassable. `op_emitter::bucket_corpus` filters candidates by `confidence ≥ Ratified` (D-ARM-10); ArmDiscovered candidates that haven't passed the council never reach the deterministic codegen path. **The council ratification gate IS the determinism firewall** — it's the only non-deterministic-to-deterministic transition in the pipeline.

---

## 4. Convergence — how the three legs feed the same substrate

```
       ┌── Leg 1 (Curated, BP-1b) ────► OdooEntity consts (66) ─────────┐
       │   l{1..15}.rs, 11,563 LOC, Sonnet-agent projection             │
       │   Stamps: OdooConfidence::Curated, regulation_iri              │
       │                                                                ▼
       │                                                          ┌───────────────────┐
       ├── Leg 2 (Extracted, EXT-2) ──► EXT_<NAME> consts (≈48) ──┤  Typed OdooEntity │
       │   extracted/<addon>.rs, 99,209 LOC, Python AST walker    │  SoA              │
       │   Stamps: OdooConfidence::Extracted, audit fallback log  │  (the common      │
       │                                                          │  union)           │
       │                                                          │                   │
       └── Leg 3 (ArmDiscovered, ARM-disc) ──► CandidateTriple ──►│                   │
           parquet/stream, pair-stats + Aerial+ fan-in            │                   │
           Stamps: ProvenanceTier::ArmDiscovered                  └─────────┬─────────┘
           Must pass council (Stage D) → Ratified                           │
                                                                            ▼
                                                                    ┌───────────────┐
                                                                    │ SpoStore +    │
                                                                    │ EdgeColumn    │
                                                                    │ + mailbox SoA │
                                                                    └───────┬───────┘
                                                                            │
                                                                            ▼
                                                                  derive_style_recipe
                                                                    + op_emitter
                                                                    + codegen
```

The three legs are **not redundant** — they see different things:

- **Curated** sees what *humans documented as semantically important*. High signal, low coverage.
- **Extracted** sees what *the source code literally states*. Medium signal, high coverage (TIER-1).
- **ArmDiscovered** sees what *runtime data confirms or contradicts*. Variable signal, runtime-dependent coverage.

A complete picture requires all three. The `pairing.rs::CURATED_EXTRACTED_PAIRS` table is the audit surface between Legs 1 and 2; the council ratification queue is the audit surface between Leg 3 and Legs 1+2.

---

## 5. Provenance ordering (for downstream consumers)

The `OdooConfidence` (a.k.a. `ProvenanceTier`) enum carries a partial order that downstream consumers respect:

```text
Curated > Extracted > Ratified > ArmDiscovered > Conjecture
```

- `Curated` and `Extracted` exist on disk today.
- `Ratified` is proposed by D-ARM-1 (new variant); applies to ArmDiscovered candidates that passed council.
- `ArmDiscovered` is proposed by D-ARM-1; applies to candidates emitted by Stage B but not yet ratified.
- `Conjecture` exists on disk today; applies to entries that are neither curated nor source-extracted nor data-discovered — placeholders awaiting one of the three.

`op_emitter::bucket_corpus` reads `confidence ≥ Ratified` (cuts at the council gate); `derive_style_recipe` reads any confidence (the recipe is just an interpretation projection; truth flows through `OdooStyleRecipe::regulation_iris` only for the Curated tier today).

---

## 6. What this doctrine forbids

- **Auto-promotion of ArmDiscovered to Ratified.** The council gate is non-negotiable. Per The Click: novel candidates are the Epiphany branch, not the Commit branch.
- **Silent override of Curated by Extracted.** Conflicts surface through `pairing.rs::CURATED_EXTRACTED_PAIRS` for human ratification, never auto-resolved.
- **A fourth proposer leg without a ProvenanceTier slot.** New proposers (LLM-based, dual-IPC variants) get a new ProvenanceTier variant + a `discovery_origin` byte slot (proposed in PR #435 §7) — they do NOT silently inject into an existing tier.
- **Bypassing the SpoStore.** All three legs write through `SpoBuilder::build_edge` only. No parallel triple store, no shadow registry. Per `E-SOA-IS-THE-ONLY`.

---

## 7. Cross-refs

- `odoo-blueprint-inventory-v1.md` — the 66-entity corpus (Leg 1 output).
- `odoo-extraction-tools-v1.md` — tool stacks behind each leg.
- `.claude/plans/odoo-business-logic-blueprint-v1.md` — Leg 1's plan.
- `.claude/plans/odoo-source-extraction-v1.md` — Leg 2's plan.
- `.claude/plans/streaming-arm-nars-discovery-v1.md` — Leg 3's plan (this PR #435).
- `style_recipe.rs` (PR #433), `op_emitter.rs` (PR #435) — downstream codegen consumers.
- CLAUDE.md `E-SOA-IS-THE-ONLY`, `I-NOISE-FLOOR-JIRAK`, "The Click" — doctrinal anchors.

End of strategies.
