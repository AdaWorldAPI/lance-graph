# Aerial+ ARM-discovery ↔ ruff DTO / SPO / codegen — synergy map

> **READ BY:** integration-lead, truth-architect, dto-soa-savant, prior-art-savant,
> any agent touching `lance-graph-arm-discovery`, `ruff_spo_triplet`,
> `ruff_python_dto_check`, `ruff_python_codegen`, or `op_emitter.rs`.
>
> **Status:** FINDING (type-level, grounded in source read 2026-05-30) +
> CONJECTURE (the three proposed wiring deliverables, council-gated).
>
> **Anchors:** `streaming-arm-nars-discovery-v1.md` (the plan this implements
> the Aerial leg of), `E-DISCOVERY-CODEGEN-BRACKET-1` (candidate epiphany),
> `E-SOA-IS-THE-ONLY`, `I-NOISE-FLOOR-JIRAK`. Papers: Karabulut 2025
> (Aerial+, 2504.19354v1), Abreu 2025 (ontology M2M, 2511.13661v1).

---

## 0. One sentence

The Aerial+ Rust transcode (`crates/lance-graph-arm-discovery`) is the
**runtime-data proposer** of a three-frontend / one-substrate / two-codegen
bracket whose substrate and codegen legs **already exist in the `ruff` fork** —
so Aerial's job is to emit into contracts the ruff crates already define, not
to invent new ones.

```text
   PROPOSER FRONTENDS                    SUBSTRATE                 CODEGEN LEGS
 ┌───────────────────────┐                                     ┌────────────────────┐
 │ ruff_python_dto_check  │  static AST  ┐                      │ op_emitter.rs       │ Rust
 │  (Python source → IR)  │              │   ruff_spo_triplet    │  (ratified SoA →    │ dispatch
 ├───────────────────────┤              ├──►  Triple{s,p,o,f,c} ─┤   RECIPE_* + Ops)   │
 │ ruff_ruby_spo (scaffold)│  static AST  │   = NARS truth        ├────────────────────┤
 ├───────────────────────┤              │   = SPO ndjson        │ ruff_python_codegen │ Python
 │ lance-graph-arm-       │  RUNTIME     ┘   loader contract     │  (AST → source,     │ source
 │  discovery (Aerial+)   │  DATA  ◄── THIS WORK                 │   round_trip)       │
 └───────────────────────┘                                     └────────────────────┘
                                   ▲ determinism firewall = ratification gate ▲
                          (Aerial is nondeterministic → stays UPSTREAM of the gate)
```

This is the literal realisation of the two-paper bracket: Aerial+ supplies the
upstream discovery leg, the Abreu M2M paper validates the downstream codegen
leg, and `ruff_spo_triplet`'s `Triple{s,p,o,f,c}` is the SPO+NARS invariant
middle both converge on.

---

## 1. Synergy A — `ruff_spo_triplet` IS the truth/triple contract Aerial emits into

**FINDING.** `ruff_spo_triplet::triple::Triple` is `{ s, p, o, f: f32, c: f32 }`
and its doc says it *"mirrors `lance_graph::graph::spo::odoo_ontology::OntologyTriple`
field-for-field so the ndjson this crate writes loads into that store with no
transform."* `ruff_spo_triplet::ndjson::to_ndjson` emits one
`{"s","p","o","f","c"}` object per line — *"exactly what
`lance_graph::graph::spo::odoo_ontology::parse_triples` reads."*

Our transcode targets this contract directly:

| Aerial (this crate)                       | ruff_spo_triplet            | lance_graph SPO            |
| ---                                       | ---                         | ---                        |
| `translator::CandidateTriple{s,p,o,f,c}`  | `triple::Triple{s,p,o,f,c}` | `odoo_ontology::OntologyTriple` |
| `ndjson::to_ndjson` → `{"s","p","o","f","c"}\n` | `ndjson::to_ndjson` (same shape) | `parse_triples` (the loader) |
| `arm_to_nars` → `(f, c)`                  | `Provenance::truth()` → `(f, c)` | `TruthValue::new(f, c)` |

The truth scales line up exactly — `NarsTruth::expectation()` reimplements
`TruthValue::expectation` (`c·(f−0.5)+0.5`) so an Aerial-mined rule and a
ruff-extracted structural triple are gated by the **same** `TruthGate`. Test
`aerial::tests::mined_rules_serialise_to_spo_ndjson` proves a mined rule round-
trips into that line shape end-to-end.

**FINDING — the gap (the actionable bit).** `ruff_spo_triplet::Predicate` is a
**closed vocabulary**: `{rdf:type, has_function, emitted_by, depends_on,
reads_field, raises, traverses_relation}`, and `ndjson::from_ndjson`
**hard-rejects** any predicate outside it. None of these is an *association /
implication* relation. So a mined `X → Y` rule cannot today flow through the
ruff loader: its natural predicate (`implies` / `co_occurs_with`) is not in the
set. This is by design — the doc says adding a predicate is *"a deliberate
ontology change."* The Aerial `DebugProjector` emits `"implies"` precisely to
surface this seam.

→ **Deliverable D-ARM-SYN-1 (CONJECTURE, council-gated):** add `Implies`
(and/or `CoOccursWith`) to `ruff_spo_triplet::Predicate` with a new
`Provenance` tier (see §4) so ARM-discovered rules load through the *same*
`parse_triples` path as the static extractor. Touches: one enum variant + one
`as_str`/`from_str` arm + one `default_provenance` arm + the ndjson closed-
vocab test. Low LOC, deliberate ontology decision → must pass the council
(`dto-soa-savant` + `prior-art-savant`).

---

## 2. Synergy B — `ruff_python_dto_check` and Aerial are the two ends of the proposer spectrum

`ruff_python_dto_check` harvests DTO/route/handler/model facts from **static
Python source** via `ruff_python_parser` + `ruff_python_ast` into a JSON
`Bundle`, and (with `ruff_ruby_spo`) fills the shared `ruff_spo_triplet::ModelGraph`
IR. It is the Rust-native successor to the plan's `tools/odoo-blueprint-extractor`
AST walk — the **`Extracted` / "Authoritative"** proposer leg.

**FINDING — complementary, not redundant.** The dto_check extractor is *bounded
by what the source literally states* (it reads `@api.depends`, compute bodies,
`raise`s — see `ir.rs` cheat-sheet). It structurally **cannot** surface a
co-correlation that only exists in runtime rows — e.g. "invoices route through
Fiscal Position A when partner-country = B AND product-category = C", implicit
in years of `account.move` but absent from `account_fiscal_position.py`. That
is exactly the gap Aerial fills: the **`ArmDiscovered` runtime-data leg**. The
two are the same shape (`ModelGraph` → `Triple`), different evidence source:

| Leg                       | Source            | Crate                       | Provenance tier      | Truth                |
| ---                       | ---               | ---                         | ---                  | ---                  |
| Curated                   | L-docs prose      | hand                        | `Curated`            | high f, high c       |
| Extracted (static AST)    | Python/Ruby source| `ruff_python_dto_check`     | `Structural`/`Authoritative`/`Inferred` | tiered fixed |
| **ArmDiscovered (runtime)** | **parquet rows** | **`lance-graph-arm-discovery`** | **`ArmDiscovered`** | **data-derived `(f,c)`** |

→ **Deliverable D-ARM-SYN-2 (CONJECTURE):** a `CandidateRule → ModelGraph`
adapter (or a parallel `arm:`-namespaced `ModelGraph`) so Aerial output joins
the dto_check output in one graph before `expand()`. The IR is "intentionally
dumb owned data" (per `ir.rs`), so the adapter is pure mapping — the antecedent
items become a synthetic subject IRI, the consequent an object IRI, the rule a
`Implies` edge. Gated on D-ARM-SYN-1.

---

## 3. Synergy C — `op_emitter.rs` and `ruff_python_codegen` are the two codegen legs of one bracket

`op_emitter.rs` (lance-graph-ontology) turns ratified typed-SoA recipes into
**deterministic Rust** (`RECIPE_*` consts + per-kind `Op` structs + static
slices, `include!`-ed at build time). `ruff_python_codegen::round_trip`
turns a Python AST back into **deterministic Python source** (`Generator` +
`Stylist`). Same thesis, two target languages:

**FINDING.** Both are *deterministic, externalised-interpretation* codegen —
the exact doctrine the Abreu M2M paper independently validates ("externalize
mapping knowledge into ontologies + rules, not code"). Both consume an already-
ratified symbolic artifact; neither does discovery. They sit **downstream of
the determinism firewall.**

**FINDING — the firewall is load-bearing for Aerial specifically.** Aerial+'s
autoencoder is nondeterministic in general (random init, denoising mask, float
reduction order). The transcode makes it *seedable* (`aerial::Rng`,
`reproducible_from_seed` test) so it is auditable, but it is still a **fan-in
proposer, never the trunk** (`streaming-arm-nars-discovery-v1.md` §2.1: pair-
stats is the deterministic default trunk). Therefore:

- Aerial output is `CandidateRule` (a *proposal*), never a committed triple.
- Promotion to a triple `op_emitter` will compile is the **council's** job
  (Stage D), via the `op_emitter::bucket_corpus` ratification filter
  (D-ARM-10: `confidence ≥ Ratified`). An ArmDiscovered candidate that has not
  passed the council never reaches either codegen leg.

→ **Deliverable D-ARM-SYN-3 (CONJECTURE):** when D-ARM-SYN-1's `ArmDiscovered`
provenance tier lands, calibrate its `(f, c)` so that **un-ratified** ARM truth
sits *below* `op_emitter`'s ratification gate and *below* the static
`Inferred` tier `(0.85, 0.75)` — i.e. ARM rules are visible to the council but
filtered out of codegen until ratified. This makes the firewall a truth-gate,
not a separate code path.

---

## 4. The truth-tier alignment (the number that has to be right)

`ruff_spo_triplet::Provenance::truth()` fixes three tiers:
`Structural (1.0, 1.0)`, `Authoritative (0.95, 0.90)`, `Inferred (0.85, 0.75)`.
Aerial's truth is **not** a fixed tier — it is computed per rule by
`arm_to_nars`:

- `f = ARM confidence = P(Y|X)` (clamped `[0,1]`)
- `c = m / (m + k)`, `m = support × n` (evidential mass), `k` = NAL-9 personality (default 1.0)

So a well-evidenced, high-confidence ARM rule can *reach* Authoritative-tier
expectation, while a thin-evidence rule lands far below Inferred. That is the
desired behaviour: ARM truth is **continuous and evidence-driven**, and the
`I-NOISE-FLOOR-JIRAK` floor (D-ARM-7, separate gate) keeps the thin tail from
ever being emitted. The proposed `ArmDiscovered` tier is therefore best modelled
not as a fixed `(f,c)` pair but as "the `arm_to_nars` output, Jirak-floored,
council-gated for codegen."

---

## 5. Determinism boundary — who sits where

| Component                          | Det? | Side of firewall | Crate |
| ---                                | ---  | ---              | ---   |
| pair-stats proposer (D-ARM-3)      | yes  | upstream (trunk) | lance-graph-arm-discovery (future) |
| **Aerial+ proposer (this work)**   | seeded| **upstream (fan-in)** | **lance-graph-arm-discovery** |
| `ruff_python_dto_check` extractor  | yes  | upstream         | ruff |
| translator `arm_to_nars`           | yes  | upstream         | lance-graph-arm-discovery |
| ratification council (Stage D)     | n/d→d| **THE firewall** | session A2A |
| `op_emitter` Rust codegen          | yes  | downstream       | lance-graph-ontology |
| `ruff_python_codegen` round_trip   | yes  | downstream       | ruff |

The single rule: **nothing nondeterministic crosses the council.** Aerial is
the only nondeterministic node in the whole bracket, which is exactly why the
transcode keeps it standalone, seeded, and behind the `aerial` feature.

---

## 6. Cross-refs

- Code: `crates/lance-graph-arm-discovery/` (this work) — `aerial::{autoencoder, extract}`, `translator`, `ndjson`, `rule`, `encode`.
- `ruff/crates/ruff_spo_triplet/{triple.rs, ir.rs, ndjson.rs}` — the contract.
- `ruff/crates/ruff_python_dto_check/` — the static-AST sibling proposer.
- `ruff/crates/ruff_python_codegen/lib.rs` — the Python codegen sibling.
- `crates/lance-graph-ontology/src/odoo_blueprint/op_emitter.rs` — the Rust codegen leg + the ratification-filter touchpoint (D-ARM-10).
- `crates/lance-graph/src/graph/spo/truth.rs` — `TruthValue` (the truth the (f,c) feeds).
- Plan: `.claude/plans/streaming-arm-nars-discovery-v1.md` (D-ARM-9 Aerial leg; this is its transcode).
- Papers: Karabulut 2025 §2/§3.3 (truth + Algorithm 1); Abreu 2025 §4 (externalise-interpretation).
