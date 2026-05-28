# Odoo retarget extraction — methodology, scripts, artifacts

> **Read this once, run the scripts, get the data.** Next session
> shouldn't re-derive any of this; just pick the right lens for the
> next family/domain and run.
>
> **Single source of truth** for the session 2026-05-28 extraction work
> on branch `claude/stage2-plans-spo-nars-elixir`. Companion to
> `.claude/board/AGENT_LOG.md` (which records commit-level outcomes)
> and `.claude/odoo/taxable_item-future-shape.rs` (which shows the
> compiled end state).

---

## 0. End-state map

```
Odoo source                                          → 3555 methods, 388 families
   │   /home/user/odoo/addons/                            harvest-full/bundles/*.ndjson
   ▼
┌─ Extraction ────────────────────────────────────────────────────────┐
│  ruff-py-dto (patched) → NDJSON bundles                              │
│  extract_delegation.py → DelegationTuple per method                  │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─ Analysis (pick the lens for the question) ─────────────────────────┐
│                                                                      │
│  cluster_soc.py     → 310 synergistic + 2655 singletons              │
│  atom_decompose.py  → 1420 technical atoms, 13 cover 50%             │
│  tax_grammar.py     → 247 tax methods → 35 grammar-coded clusters    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─ Schema mapping (OGIT meta-DTO) ────────────────────────────────────┐
│  OGIT-META-DTO-ALIGNMENT.md      → meta-schema verified              │
│  ogit-extensions/                → LineItem extension proposal       │
└──────────────────────────────────────────────────────────────────────┘
   │
   ▼
┌─ Compiled end state ────────────────────────────────────────────────┐
│  taxable_item-future-shape.rs   → Elixir → Rust → recipe → shader   │
│  (no new types; every primitive already in the workspace)            │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 1. Scripts inventory

All scripts live in `.claude/odoo/` (committed) and read/write under
`/tmp/odoo-extract/` and `/tmp/work/` (ephemeral, regenerable).

### 1.1 `ruff-py-dto` (patched binary)

| | |
| --- | --- |
| **Source** | `.claude/odoo/ruff-patches/{lib.rs,matcher_function_with_decorator.rs,preflight_scanner.rs}.patched` |
| **Build** | `cargo build -p ruff_python_dto_check --release --bin ruff-py-dto` against a `ruff` source tree with those three files swapped in |
| **Patch** | Adds class-body recursion to all three top-level loops (was top-level-only; missed every Odoo `@api.*`-decorated method on `models.Model` subclasses) |
| **Input** | Python source tree (`--root /path/to/odoo/addons`), JSON config (`--config odoo.config.json`) |
| **Output** | `<out>/bundles/<family>.ndjson` (one bundle per matched method), `<out>/manifest.json` (totals) |
| **Run** | `./ruff-py-dto harvest --config .claude/odoo/odoo.config.json --root /home/user/odoo/addons --out /tmp/odoo-extract/harvest-full` |
| **Wall-clock** | ~30 s for full odoo/addons (622 addons) on one core |

### 1.2 `extract_delegation.py`

| | |
| --- | --- |
| **Path** | `.claude/odoo/extract_delegation.py` |
| **What** | For each NDJSON bundle, locate the function in the original `.py` via `(file, line_start, function_name)`, walk its AST, emit `(invokes, reads_field, writes_field, traverses_relation, reads_env, raises)` DelegationTuple |
| **Why this beats `body_source` wrapping** | `body_source` strips leading whitespace from line 1; re-wrapping breaks on control-flow first statements. Reading the original file = 100% parse success |
| **Input** | `<bundles_dir>` + `<source_root>` + `<out_dir>` |
| **Output** | `<out_dir>/delegation.ttl` + `<out_dir>/delegation-stats.json` |
| **Run** | `python3 .claude/odoo/extract_delegation.py /tmp/odoo-extract/harvest-full/bundles /home/user/odoo/addons /tmp/work/delegation-full` |
| **Stats (full tree)** | 3555 analyzed, 831 invoke edges, 2162 read edges, 519 write edges, 456 raise edges, 697 env touches, 0 parse errors |

### 1.3 `cluster_soc.py`

| | |
| --- | --- |
| **Path** | `.claude/odoo/cluster_soc.py` |
| **What** | Cluster bundles by `(name_root, primary_decorator, delegation_signature)`. Two methods are the same concern iff this tuple matches |
| **Why** | Synergy = appears in ≥2 families. Synergistic concerns register as one StyleRecipe applied N families × N times |
| **Input** | `BUNDLES_DIR` constant (hard-coded), reads from `/tmp/odoo-extract/harvest-full/bundles` |
| **Output** | `/tmp/work/high-signal-concerns.md` (copied to `.claude/odoo/high-signal-concerns.md`) |
| **Run** | `python3 .claude/odoo/cluster_soc.py` |
| **Result** | 3555 → 2965 unique clusters; 310 synergistic (n≥2) + 2655 singletons; 271 of the 310 are `@api.depends`, 17 are `@api.constrains`, 16 are `@api.onchange` |

### 1.4 `atom_decompose.py`

| | |
| --- | --- |
| **Path** | `.claude/odoo/atom_decompose.py` |
| **What** | Decompose each method into atomic SoC primitives (one per element of the delegation tuple): `read:F`, `write:F`, `invoke:M`, `raises:E`, `env`, `traverse:R` |
| **Why** | Singletons at the SoC-tuple level (cluster_soc.py) may share ATOMS with many others. Atom-set dedup is much stronger |
| **Input** | Reads from `/tmp/odoo-extract/harvest-full/bundles` |
| **Output** | `/tmp/work/atom-catalogue.md` (copied to `.claude/odoo/atom-catalogue.md`) |
| **Run** | `python3 .claude/odoo/atom_decompose.py` |
| **Result** | 4653 atom emissions, avg 1.31/method; 1420 unique atoms; **13 atoms cover 50%**; 89 atom-sets cover 2943 methods + 612 truly unique. **Singleton collapse: 77% vs SoC-tuple level** |
| **Caveat** | Atoms are TECHNICAL (`read:filtered`, `env`) — useful for register-size sizing, not for recipe-coding. Use `tax_grammar.py` style coding for recipes |

### 1.5 `tax_grammar.py`

| | |
| --- | --- |
| **Path** | `.claude/odoo/tax_grammar.py` |
| **What** | Filter to tax-related methods (family/fn/decorator contains `tax\|vat\|umsatzsteuer\|ustva\|withholding`), code each by `(T, tek, men, reg)`: transitivity, TEKAMOLO slot, mengenmaß, regulatory anchor |
| **Why** | This is the **right lens** per `E-BUSINESS-LOGIC-IS-GRAMMAR-1`. Technical atoms are noise; grammar+regulation is signal |
| **Input** | Reads from `/tmp/odoo-extract/harvest-full/bundles` |
| **Output** | `/tmp/work/tax-grammar-coded.md` (copied to `.claude/odoo/tax-grammar-coded.md`) |
| **Run** | `python3 .claude/odoo/tax_grammar.py` |
| **Result** | 247 tax methods → 35 grammar-coded clusters (87% compression vs technical atoms); 22 of 35 have explicit `reg=UStG` |

### 1.6 `bundles_to_ttl.py`

| | |
| --- | --- |
| **Path** | `.claude/odoo/bundles_to_ttl.py` |
| **What** | Emit one OGIT-conformant TTL file per family. Carries the 7-tuple grammar slots (axis classification, transitivity, causal markers) + full provenance + body source as `rdfs:comment` |
| **OWL profile** | OWL 2 EL with explicit `owl:Ontology` + `owl:versionInfo "OWL 2 EL profile"` annotation. Validated by rdflib. 0 undeclared predicates, 0 blank-node subjects, 0 `owl:Restriction` (per Keet 2025-11-17 audit checklist) |
| **Namespace caveat** | Currently uses made-up `<https://ogit.adaworldapi.com/>`. The canonical OGIT namespace is `<http://www.purl.org/ogit/>`. See `OGIT-META-DTO-ALIGNMENT.md` for the migration |
| **Run** | `python3 .claude/odoo/bundles_to_ttl.py /tmp/odoo-extract/harvest-full/bundles /tmp/work/ttl-full` |
| **Result** | 3555 method TTL resources across 388 family `.ttl` files (~4.4 MB) |

---

## 2. Data artifacts

### 2.1 Ephemeral (regenerable from scripts above)

```
/tmp/odoo-extract/
   harvest-full/
      bundles/<family>.ndjson         # 388 files, 3555 methods total
      manifest.json                    # totals
   preflight-account2/                  # account-only preflight report
   harvest-account2/                    # account-only harvest (363 bundles)

/tmp/work/
   delegation-full/delegation.ttl       # 4019 edges
   ttl-full/<family>.ttl                # 388 files, OWL 2 EL conformant
   high-signal-concerns.md              # 310 + 2655
   atom-catalogue.md                    # 13 atoms = 50%
   tax-grammar-coded.md                 # 35 clusters from 247 methods
```

### 2.2 Committed (in `.claude/odoo/`)

```
.claude/odoo/
   # Methodology + analysis (this session)
   EXTRACTION-METHODOLOGY.md            # ← you are here
   high-signal-concerns.md              # SoC-dedup table
   atom-catalogue.md                    # atom decomposition
   tax-grammar-coded.md                 # grammar-coded tax table
   taxable_item-future-shape.rs         # the compiled end-state demo
   OGIT-META-DTO-ALIGNMENT.md           # OGIT meta-schema audit
   QFGEN-FRAMEWORK-NOTES.md             # Keet/Raboanary psychometry bridge

   # Scripts (run them; they produce the artifacts above)
   bundles_to_ttl.py                    # NDJSON → OWL 2 EL TTL
   extract_delegation.py                # delegation tuples
   cluster_soc.py                       # SoC synergy clusters
   atom_decompose.py                    # technical atom catalogue
   tax_grammar.py                       # grammar-coded tax analysis

   # Config + ruff patches
   odoo.config.json                     # ruff-py-dto config for Odoo
   ruff-patches/                        # class-body-recursion patches
      lib.rs.patched
      matcher_function_with_decorator.rs.patched
      preflight_scanner.rs.patched
      README.md                         # upstream PR scope

   # Samples (committed, regenerable)
   sample-bundles.account_move.ndjson   # 93 account_move bundles
   sample-output.account_move.ttl       # full TTL with delegation edges
   sample-preflight.account.json
   sample-manifest.full-odoo.json

   # OGIT extensions (proposals for AdaWorldAPI/OGIT PR split)
   ogit-extensions/
      account_move_line-extension-proposal.md
      ruff-patches/                     # mirror of ruff patches

   # Curated L-doc corpus (predates this session; not modified here)
   L1-K3-POST.md … L15-TAX-REPARTITION.md
   BRIEFING.md, SAVANTS.md, savants/
```

---

## 3. Choose the right lens

```
question                              lens                            script
─────────────────────────────────     ──────────────────────────      ─────────────────
"is the matcher catching anything?"  preflight decorator histogram   ruff-py-dto preflight
"what methods exist?"                 bundles                         ruff-py-dto harvest
"who calls who? who reads what?"      delegation tuples               extract_delegation.py
"what concerns repeat?"               SoC-synergy dedup               cluster_soc.py
"what register-size atoms?"           technical atoms                 atom_decompose.py
"what cognitive primitives?"          grammar-coded (T/tek/men/reg)   tax_grammar.py style
"OGIT schema mapping?"                meta-DTO entity extension       ogit-extensions/*.md
"compiled end state?"                 future-shape demo               taxable_item-future-shape.rs
"OWL 2 conformance?"                  rdflib audit                    bundles_to_ttl.py + rdflib
"psychometric calibration?"           4-metric pipeline (planned)     QFGEN-FRAMEWORK-NOTES.md
```

**Default for new domain extraction:** start at *grammar-coded*. The
technical atom catalogue exists for completeness but is the wrong unit
for recipe registration. Each Odoo domain (tax, audit, payment,
manufacturing, …) gets its own `<domain>_grammar.py` script using the
`tax_grammar.py` shape — same axes (T/tek/men/reg), domain-specific
filter regex, domain-specific regulatory anchor markers.

---

## 4. The four axes (E-BUSINESS-LOGIC-IS-GRAMMAR-1)

When coding any business-logic method, the question is always:

```
T    Transitivity        T (returns/mutates) | I (raises without return)
tek  TEKAMOLO slot       TE temporal | KA causal | MO modal | LO locative | QU quantities
men  Mengenmass          money | percent | rate | count | date | categorical | none
reg  Regulatory anchor   UStG | HGB | EStG | AO | GoBD | SKR04 | SKR03 | DATEV | ELSTER | Peppol | (per-domain extension)
```

Domain → regulatory anchor pairs ("audit : GoBD :: X-ray : ICD :: tax : UStG"):

```
accounting validation  : HGB §239 (Festschreibung)
accounting period      : HGB §240 + AO §147
audit trail            : GoBD (Unveränderbarkeit)
tax compute            : UStG §12 (Steuersätze), §15 (Vorsteuer)
tax export             : DATEV / ELSTER
e-invoice              : Peppol / UBL
medical procedure      : ICD-10-GM
medical drug           : ATC
HR contract            : EStG / SozGB
```

For a new regulatory domain, extend `REGULATORY_MARKERS` in
`tax_grammar.py` (or copy the script as `<domain>_grammar.py`).

---

## 5. The future shape (recipe.rs ladder + 400ms shader)

See `.claude/odoo/taxable_item-future-shape.rs` for the worked
example. Pattern:

```
Elixir source (≤5 lines, written by domain expert)
   │
   │ tree-sitter-elixir + HM type inference per
   │ .claude/plans/lance-graph-elixir-frontend-v1.md
   ▼
Rust codegen — one fn, one cognition::advance() call
   │
   │ resolves to a StyleRecipe (lance-graph-contract::recipe)
   │ with weights over the D-ATOM-1 atom catalogue
   ▼
register_recipe() at app load (ctor, pre-actional, NOT in 400ms)
   │
   │ JIT compile via Cranelift → KernelHandle
   ▼
SurrealQL kanban pulls a goal card → pre-actional cache build
   │
   │ X cache binds (recipe, kernel, matrix, semiring)
   ▼
Ractor mailbox spawns (one mailbox = one goal set)
   │
   │ ACTIONAL 400ms sweep (Heckhausen-Rubicon phase 3)
   │   while deadline && F > floor && !goal_done:
   │     Δentropy = advance(ctx, Op::X)    -- 20ns..200μs per op
   │     emit Δentropy
   │
   │ Branch-free SIMD via bgz17::PaletteCompose semiring SpMV
   ▼
Mailbox closes → post-actional eval (mailbox state + kanban criteria)
   │
   │ card → Done | Backlog (split) | Selected (retry, revised tactic)
   ▼
LanceDB write-back; episodic memory update
```

**The shader doesn't orchestrate; it emits entropy resolution.**
Switches (basin selection, recipe pick) are pre-actional.
The shader sees only pre-resolved kernels.

---

## 6. Open items for next session (pick one)

In rough priority order:

1. **Per-domain grammar coding scripts.**
   `tax_grammar.py` is the template. Copy → adjust regex + regulatory
   markers → run. Domains queued: audit (GoBD), payment, journal,
   manufacturing, HR. Each emits a `<domain>-grammar-coded.md`.

2. **Depends-graph chain extraction.** Tax-chain density was 1 edge
   in the call-graph for 247 methods. Real chains live in the
   `@api.depends` field-cascade DAG, not call-graph. New script
   `depends_chain.py`: walk `@api.depends(field1, field2, ...)` args
   across `_compute_*` methods, build the trigger-DAG, find chains.

3. **OGIT meta-DTO PR split.** `ogit-extensions/account_move_line-extension-proposal.md`
   is a single doc with the extended `LineItem.ttl` + 20 new
   attributes + 5 new verbs. Split into ~26 per-file TTLs and push to
   `AdaWorldAPI/OGIT` as a PR (PAT verified earlier this session).

4. **D-ATOM-1 catalogue formalization.** The top ~200 atoms from
   `atom-catalogue.md` (the high-frequency universal Odoo idioms)
   should populate the `contract::atoms::Atom` enum. Currently
   BLOCKED in `recipe.rs` per its own comment block.

5. **Elixir-frontend prototype.** Pick one rule (e.g., the
   `taxable_item` one in the future-shape demo), write the actual
   Elixir source file in `rules/de/ustg/taxable_item.exs`, run
   tree-sitter-elixir, verify HM inference produces the right typed
   Rust calls. Predecessor gates are noted in
   `.claude/plans/lance-graph-elixir-frontend-v1.md`.

6. **Cross-source psychometric calibration.** Run Cronbach α + ICC +
   Spearman ρ + Pearson r across D1 (code-extracted concerns) ↔ D2
   (OGIT axioms) ↔ D3 (L-doc curated knowledge) per
   `QFGEN-FRAMEWORK-NOTES.md` reframing. Bootstrap pass = 9-factor
   null hypothesis (2 axes × 7 tuple slots).

---

## 7. Continuation checklist for a fresh session

```bash
# 1. Re-extract bundles if cache is gone (~30s)
./ruff-py-dto harvest \
    --config .claude/odoo/odoo.config.json \
    --root /home/user/odoo/addons \
    --out /tmp/odoo-extract/harvest-full

# 2. Re-build delegation tuples
python3 .claude/odoo/extract_delegation.py \
    /tmp/odoo-extract/harvest-full/bundles \
    /home/user/odoo/addons \
    /tmp/work/delegation-full

# 3. Pick the analysis lens
python3 .claude/odoo/cluster_soc.py    # high-level SoC synergy
python3 .claude/odoo/atom_decompose.py # technical atoms
python3 .claude/odoo/tax_grammar.py    # grammar-coded (the right unit)

# 4. Look at the future shape
less .claude/odoo/taxable_item-future-shape.rs

# 5. Read OGIT mapping
less .claude/odoo/OGIT-META-DTO-ALIGNMENT.md
less .claude/odoo/ogit-extensions/account_move_line-extension-proposal.md

# 6. Pick next domain or item from §6 above and go.
```

**Do not re-derive the methodology.** Pick a lens, pick a domain,
run the script. The compiled end state is `taxable_item-future-shape.rs`
— every other domain compiles into the same shape with different
recipe weights and a different regulatory anchor.
