# odoo-extraction-tools-v1 — Tool stacks behind the three proposer legs

> **READ BY:** anyone running, extending, or porting an extractor; ARM-discovery implementors; session-survival readers who need to know where the tools live.
> **Status:** FINDING (tools 1 + 2 exist on disk; tool 3 is partially scaffolded in a sibling repo).
> **Authored:** 2026-05-29.
> **Companion docs:** `odoo-blueprint-inventory-v1.md` (the 66-entity corpus), `odoo-extraction-strategies-v1.md` (the three legs).

---

## 1. Tool stack for Leg 1 (Curated) — humans + Sonnet agents

### Inputs

- `.claude/odoo/L{1..15}-*.md` — the 15 lane L-docs. Human-authored prose.
- `crates/lance-graph-ontology/src/odoo_blueprint/mod.rs` — the typed surface (`OdooEntity`, `OdooField`, `OdooMethod`, …).
- Existing lane files (read for stylistic conformance).

### Tool

**5-Sonnet-agent fan-out per Wave.** No code tool; the "tool" is the prompt + the Sonnet model. Agent cards in `.claude/agents/odoo-l-doc-projector.md` would formalize this if the pattern is reused (today it's done ad-hoc per Wave-spawn).

### Outputs

- `crates/lance-graph-ontology/src/odoo_blueprint/l{1..15}.rs` — 66 `pub const NAME: OdooEntity = OdooEntity {...}` declarations.
- 130 lane tests (per-lane smoke tests asserting field-kind correctness, method-name presence, etc.).

### Run procedure

1. User invokes the orchestrator with "spawn N agents for Wave-K, lanes Lx-Ly."
2. Main thread spawns N parallel `Agent(subagent_type=general-purpose, model=sonnet)` calls in one turn.
3. Each agent reads its L-doc + the typed surface + (optionally) an exemplar lane that already exists.
4. Each agent emits one file `l<N>.rs` with the const + the tests module.
5. Main thread (Opus) reviews; trims agent verbosity; commits as one Wave-K commit.

### Today's status

Complete. 66 entities across 15 lanes; 11,563 LOC; 130 lane tests. Last touched by `c04adf10` (EXT-3 back-fill for `OdooEntityKind` + `regulation_iri`).

---

## 2. Tool stack for Leg 2 (Extracted) — `tools/odoo-blueprint-extractor/`

### Location

`/home/user/lance-graph/tools/odoo-blueprint-extractor/` — a Python 3 package, stdlib-only.

### Layout

```text
tools/odoo-blueprint-extractor/
├── README.md
├── pyproject.toml
├── odoo_blueprint_extractor/
│   ├── __init__.py
│   ├── __main__.py            # python -m entry point
│   ├── cli.py                 # 258 LOC — argv parsing + orchestration
│   ├── pairing.py             # 382 LOC — curated/extracted pairing audit
│   ├── parsers/               # 950 LOC total — AST walkers
│   │   ├── classes.py         # 209 LOC — class body → OdooEntity
│   │   ├── fields.py          # 161 LOC — fields.X(...) → OdooField
│   │   ├── methods.py         # 147 LOC — def _compute_… → OdooMethod
│   │   ├── state_machine.py   # 157 LOC — state field + transitions
│   │   ├── constraints.py     # 138 LOC — _sql_constraints + @api.constrains
│   │   ├── decorators.py      # 73 LOC  — @api.depends/constrains/onchange
│   │   └── regulation.py      # 64 LOC  — module __manifest__.py → reg IRIs
│   ├── data_extractors/        # CSV/XML for non-Python data
│   │   ├── csv_chart.py       # SKR03/SKR04 CSV → 24,712 LOC of chart consts
│   │   ├── xml_kennzahl.py    # UStVA Kz boxes from XML
│   │   └── gobd_company.py    # GoBD audit-trail wiring from XML
│   ├── emitters/              # Rust source emitters
│   │   ├── module.py          # per-addon module shell
│   │   └── rust.py            # OdooEntity → Rust source text
│   └── audit/
│       └── fallback_log.py    # field-kind/method-kind fallback log → JSON
└── tests/
    └── test_smoke_uom.py      # one-addon smoke test (uom)
```

### Inputs

- `/home/user/odoo/addons/<addon>/` — Odoo source tree. The 12 TIER-1 addons are: `account`, `account_payment`, `l10n_de`, `product`, `stock`, `uom`, `base`, `analytic`, `purchase`, `sale`, `account_peppol`, `account_edi_ubl_cii`.

### Run procedure

```bash
# One addon, stdout (development / inspection):
python -m odoo_blueprint_extractor \
    --addons /home/user/odoo/addons \
    --addon uom \
    --out -

# All TIER-1 addons → extracted/ tree (production):
for addon in account account_payment l10n_de product stock uom base analytic purchase sale account_peppol account_edi_ubl_cii; do
  python -m odoo_blueprint_extractor \
      --addons /home/user/odoo/addons \
      --addon "$addon" \
      --out crates/lance-graph-ontology/src/odoo_blueprint/extracted \
      --audit /tmp/fallback.json
done
```

### Outputs

- `crates/lance-graph-ontology/src/odoo_blueprint/extracted/<addon>.rs` — 11 files, 99,209 LOC total.
- `--audit /tmp/fallback.json` — JSON log of field/method kinds that hit the `Other` fallback.

### Today's status

- Shipped (commits `2026-05-28` — see `extracted/COVERAGE.md`).
- 48/53 curated entities have TIER-1 backing (90.6% workspace-wide).
- 5 TIER-2 deferrals: 4 `hr.*` (HR-base lane) + 1 `stock.valuation.layer` (deferred to TIER-2 stock-account).
- Gate test in `extracted::coverage` enforces 80% per-lane backing.
- L12-L15 (Wave-3 curated additions) need a fresh extractor pass to bring into `CURATED_EXTRACTED_PAIRS` (post-EXT-6 gap).

### Known gaps (Stage-2 enrichment targets)

These would light the 6 dark D-Atoms (Money / Quantity / ApplyRate / EmitAmount / Event / FiscalCtx) currently not firing:

| Gap | Where to fix |
|---|---|
| `return_kind` mostly defaults to `Unit` | `parsers/methods.py` — needs `ast.Return` walker to peek at return-expression type |
| `semantic_role` mostly defaults to `Other` | `parsers/fields.py` — needs lexical pattern matcher on field name + comment |
| `computed: Option<&'static str>` not populated | `parsers/fields.py` — already partial; needs depends-graph cross-link |
| `state_machine` rarely captured | `parsers/state_machine.py` — needs richer detection of `write({'state': ...})` patterns inside action methods |

---

## 3. Tool stack for Leg 3 (ArmDiscovered) — `lance-graph-arm-discovery` (planned)

### Location (when implemented)

`/home/user/lance-graph/crates/lance-graph-arm-discovery/` — new crate. Specified in `.claude/plans/streaming-arm-nars-discovery-v1.md` (D-ARM-1 through D-ARM-12, ~2,400 LOC).

### Status

**Specified; not implemented.** 12 deliverables D-ARM-1..D-ARM-12 in STATUS_BOARD, all **Queued**. PR #435 (this branch) ships the plan + handover, not the crate.

### Planned components

| Component | LOC | Role |
|---|---:|---|
| `proposer::pair_stats` | 400 | Streaming pair-stats over Arrow RecordBatch (deterministic trunk) |
| `proposer::aerial_ipc` | 200 | NDJSON-over-Unix-socket client for Aerial+ subprocess (`arm-aerial` feature) |
| `translator` | 200 | `arm_to_nars(rule) -> TruthValue` + `CandidateTriple` + projector |
| `hypothesis` | 350 | SpoStore round-trip + NARS revision + contradiction commit |
| `queue` | 200 | `RatificationQueue` bounded ring buffer |
| `jirak` | 150 | Jirak-2016 weak-dependence threshold helpers |
| `feed` | 250 | `Feed` + `FeedProjector` config + Odoo `account.move` example |
| Tests + bench | 400 | End-to-end pipeline (D-ARM-12) |

### Supporting external tool — Aerial+ reference

- **Upstream:** https://github.com/DiTEC-project/aerial-rule-mining (paper authors' Python reference).
- **Workspace fork:** https://github.com/AdaWorldAPI/aerial-rule-mining (shared as the workspace's fork target).
- **Access:** **Both are outside the MCP allowlist as of 2026-05-29.** Implementation will be from paper Algorithm 1 (paper text in session record); IPC contract reads NDJSON, not Python API, so upstream-source access is not strictly required.
- **OQ-ARM-11 pending:** batch the "external-repo MCP allowlist class" question (Aerial+ + surreal_container's allowlist axis share this OQ class per cross-session review).

### Run procedure (when implemented)

```rust
use lance_graph_arm_discovery::{
    Feed, PairStatsProposer, HypothesisTest, RatificationQueue,
};

let feed = Feed::from_parquet("/data/odoo/account_move_2026.parquet")
    .window_size(100_000)
    .projector(OdooMovementProjector::new());

let mut proposer = PairStatsProposer::new(&feed);
let mut hypo = HypothesisTest::new(&spo_store);
let mut queue = RatificationQueue::new(1024);

while let Some(window) = feed.next_window().await {
    for rule in proposer.next_batch(window) {
        let candidate = arm_to_nars(&rule);
        match hypo.test(&candidate) {
            HypothesisOutcome::Revised(emission) => mailbox.emit(emission),
            HypothesisOutcome::Contradiction(pair) => spo_store.commit_contradiction(pair),
            HypothesisOutcome::Novel => queue.push(candidate),
        }
    }
}
// queue.drain() → council for ratification (manual session trigger)
```

---

## 4. Cross-cutting tool — `ruff_spo_triplet` (cross-language SPO triple core)

### Location

- **Workspace path during prior session:** `/tmp/ruff-work/crates/ruff_spo_triplet/` — was scaffolded into a workspace fork of `AdaWorldAPI/ruff` during the SPO-triplet-extraction session.
- **Upstream:** `AdaWorldAPI/ruff` — sibling repo (within MCP allowlist for read; verify before write).
- **Docs:** `crates/ruff_spo_triplet/SPO_TRIPLET_EXTRACTION.md` — 252-LOC methodology + Rails→IR mapping guide.

### Purpose

**Language-agnostic SPO triple core.** Closed 7-predicate vocabulary; `ModelGraph` IR with `expand()` semantics; `to/from_ndjson()` round-trip. Lets a new source-language frontend (Python, Ruby/Rails, etc.) emit byte-identical triple shapes consumable by `lance_graph::graph::spo`.

### Components

- `ruff_spo_triplet` — the core crate (language-agnostic).
- `ruff_ruby_spo` — scaffolded Ruby/Rails frontend (`todo!()` stubs for parser; methodology doc explains how to fill).

### Today's status

- Scaffolded; partially shipped to `AdaWorldAPI/ruff` (commit history in that repo).
- The Odoo (Python) extraction in `tools/odoo-blueprint-extractor/` was the **prior pattern** that motivated the cross-language abstraction; the ruff-side crate is the formalized version.
- ARM-discovery's Stage B translator may emit through `ruff_spo_triplet`'s NDJSON shape for cross-language compatibility (not yet specified).

---

## 5. Tool-to-leg matrix

| Tool | Leg 1 (Curated) | Leg 2 (Extracted) | Leg 3 (ArmDiscovered) |
|---|---|---|---|
| Sonnet agents in fan-out | **trunk** | review only | n/a |
| `tools/odoo-blueprint-extractor/` Python | n/a | **trunk** | n/a |
| `lance-graph-arm-discovery` Rust (planned) | n/a | n/a | **trunk** |
| Aerial+ Python subprocess (planned) | n/a | n/a | optional fan-in |
| `ruff_spo_triplet` (Python+Rust) | n/a | could replace `emitters/rust.py` | could supply IPC NDJSON shape |
| `pairing.rs::CURATED_EXTRACTED_PAIRS` | audit surface | audit surface | (separate pairing surface for ratified vs ArmDiscovered TBD) |
| Council (`epiphany-brainstorm-council` PR #433) | bypassed | bypassed | **mandatory gate** |

---

## 6. Where to find each tool if the session dies

| Tool | Path | Recoverable from |
|---|---|---|
| Lane projector (Sonnet) | `(no file — it's the agent prompt pattern)` | `.claude/agents/BOOT.md` + `.claude/odoo/L*.md` |
| Extractor Python package | `tools/odoo-blueprint-extractor/` | In main as of `2026-05-28`; will be on disk if repo is cloned |
| Extractor invocation history | Wave commits `f5702675`, `d30186e5`, `333a1ff2`, `c04adf10` | `git log --all -- 'crates/lance-graph-ontology/src/odoo_blueprint/extracted/**'` |
| ARM-discovery crate (when impl'd) | `crates/lance-graph-arm-discovery/` (planned) | Spec at `.claude/plans/streaming-arm-nars-discovery-v1.md` |
| Aerial+ reference | `DiTEC-project/aerial-rule-mining` (allowlist required) OR paper Algorithm 1 reproduction | Paper text preserved in session record |
| ruff_spo_triplet | `AdaWorldAPI/ruff` repo (within MCP allowlist) | `git log --all` on that repo |
| L-docs | `.claude/odoo/L{1..15}-*.md` | In main; gitignored if any (verify) |

---

## 7. Cross-refs

- `odoo-blueprint-inventory-v1.md` — what the tools produce (Leg 1 output).
- `odoo-extraction-strategies-v1.md` — why each leg exists.
- `.claude/plans/odoo-business-logic-blueprint-v1.md` — Leg 1's plan.
- `.claude/plans/odoo-source-extraction-v1.md` — Leg 2's plan + EXT-6 coverage report.
- `.claude/plans/streaming-arm-nars-discovery-v1.md` — Leg 3's plan.
- `crates/lance-graph-ontology/src/odoo_blueprint/extracted/COVERAGE.md` — Leg 2's current 90.6% coverage.
- `crates/lance-graph-ontology/src/odoo_blueprint/extracted/pairing.rs` — Leg 1 ↔ Leg 2 audit surface.
- Paper: Karabulut, Groth, Degeler — *Neurosymbolic Association Rule Mining* (arxiv 2504.19354v1) — Leg 3's algorithmic anchor.

End of tools doc.
