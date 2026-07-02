# The DO-arm triage — three buckets, one classaction pointer

> **READ BY:** integration-lead, core-first-architect, adapter-shaper,
> family-codec-smith, baton-handoff-auditor, truth-architect
> **Status:** DESIGN (operator-directed triage; the THINK arm is measured —
> see `ast-as-partof-isa-address.md` brick-3 — the DO arm is the residue this
> doc partitions).
> **Cross-ref:** `ast-as-partof-isa-address.md` (the THINK arm — the
> `(part_of:is_a)` structural skeleton), OGAR `OGAR-AST-CONTRACT.md`
> (`ActionDef`/`ActionInvocation`/`KausalSpec`), OGAR
> `SURREAL-AST-AS-ADAPTER.md` (behaviour flows producer → OGAR `Class`+
> `ActionDef` → adapter, **never** producer → DDL), OGAR
> `core-first-transcode-doctrine.md` (thin classid-keyed adapters assume the
> Core), `ogar-consumer-preflight.md`.

---

## Why there is a DO arm at all

The `(part_of:is_a)` address (`ast-as-partof-isa-address.md`) holds the **THINK
arm** — the class/member/type graph, the ClassView method-resolution manifest,
the harvest's *structural* predicates (`has_function`, `inherits_from`,
`virtually_overrides`). That skeleton is now measured: `mint_factored` mints it
losslessly, truncation is disallowed, god-classes reroute by SoC.

But an ERP is not just structure — it *does* things: posts an invoice, reconciles
a bank line, escalates a dunning level, signs a checkout. That behaviour is the
**DO arm**. It does **not** live in `(part_of:is_a)` (a method body is not a
`part_of` or `is_a` relation) and it does **not** live in DDL (OGAR
`SURREAL-AST-AS-ADAPTER.md` §0 rejects the "`DEFINE EVENT … WHEN … THEN …`
carrying lifecycle" negative-beauty hijack). It lives in `ActionDef` /
`ActionInvocation` / `KausalSpec`, **keyed by** the GUID (`CausalEdge64` on the
value/edge side), never encoded in the address.

**The classaction pointer.** Because ontologies are lossless (OWL/OGIT class
round-trips proved bijective), the DO arm becomes an *object-oriented reusable
classaction pointer* in the consumer: the `ActionDef` **points to** a body; it
never inlines it. The transcoded almato/HIRO/arago `ActionHandler` (OGAR #125
`ResolvingDaemon` = class-late-bound dispatch) is exactly this — the dispatch is
resolved against the class at call time, so one `ActionDef` shape serves every
consumer that imports the class primitive. The DO arm is conditionalized *at the
cost of a ClassView classaction*, not at the cost of re-transcoding the body.

---

## The triage — three buckets

Not all DO is equal. After the THINK arm is minted, the *behavioural residue*
partitions into three buckets by **how ontologically-shaped it is**. This refines
OGAR's earlier coarse 85/15 (structure/residue) split: the 15% residue is itself
not uniform — most of it is anticipated-standard, a sliver is genuinely random.

### Bucket 1 — fuzzy-shaped (order-varying): canonicalize first

**Signature:** "something that emits X but changes the order every time
differently." The *content* is stable; the *sequence/shape* wobbles run to run
(iteration order over a hash map, non-deterministic field emission, incidental
ordering that carries no semantics).

**Treatment:** **canonicalize before you address.** Sort/normalize into a
deterministic form, *then* it collapses into a stable landing zone (usually
bucket 2). Do **not** mint an address for the wobble — that manufactures
false distinctions. This is the DO-arm analogue of the THINK-arm rule "encode
identity positions, never incidental order" (`I-VSA-IDENTITIES`).

### Bucket 2 — anticipated standard DO: an ontologically-shaped landing zone

**Signature:** the DO that *every domain has* and OGIT already anticipates —
auth/session/login, CRUD lifecycle, audit-emit, validation guards, the typical
"emit the standard auth + session" flow. OGIT's `NTO/Auth` (and the regulatory
`NTO/{Audit, Compliance, Legal}` set) already carry these as patterns.

**Treatment:** **do it ONCE as a DTO adapter, then reuse via a codebook of
tools.** Mint the standard shape a single time as an ontologically-named landing
zone (label the fields so the AST *lands* looking recognizable), and give the
consumer a **codebook-shaped swiss-knife** of operations over it:

| tool | what it does to the landed DTO |
| --- | --- |
| `open` | decode the standard DTO into its named slots |
| `filter` | select the subset a route/handler needs |
| `reorder` | re-sequence into the consumer's expected order |
| `apply_mask` | project/redact (RBAC lo-u16, PII leaf-rename, field-set) |

This is the big win: the anticipated-standard bucket is the *majority* of the DO
residue, and it collapses to **import-the-pattern + apply-the-swiss-knife**, not
re-transcode. `account.move.post`, `res.partner`-lifecycle, OpenProject
`Issue`-workflow, MedCare visit-open/close — all land in the *same* codebook of
tools over ontologically-named DTOs. Do-once, amortized across every consumer,
exactly like the THINK-arm primitive-mint.

### Bucket 3 — truly random: manual rewrite, invent new interfaces

**Signature:** behaviour so bespoke it has no ontological landing zone — genuinely
novel control flow, a one-off algorithm, a customer-specific rule that no pattern
anticipates.

**Treatment:** **hand-port**, and where the hand-port recurs, *partially invent a
new standard interface* so the next occurrence graduates to bucket 2. This is the
Frankenstein-flattening guard from `core-first-transcode-doctrine.md`: never
force an intrusive/stateful method into the adapter mold — route it to a
raw-pointer hand-port, and if the Core is missing a capability, *extend the Core
deliberately* (a new value tenant / ClassView capability, filed + reviewed), never
hack the adapter. Bucket 3 is where the Core *grows*; buckets 1–2 are where it is
*reused*.

---

## The ordering rule

```text
harvest DO residue
   → is the shape order-varying?           yes → BUCKET 1: canonicalize, re-triage
   → does a standard pattern anticipate it? yes → BUCKET 2: import DTO + swiss-knife
   → otherwise                                  → BUCKET 3: hand-port; graduate to 2 if it recurs
```

Buckets are visited in order: canonicalize *first* (else fuzz masquerades as
novelty), match the standard *second* (the common case), hand-port *last* (the
expensive case, kept small). The cost gradient is steep and deliberate — bucket 2
is meant to swallow the bulk so bucket 3 stays a sliver.

---

## The harvest gap — the DO extractor is Python-only

The THINK-arm harvesters cover multiple frontends (`ruff_cpp_spo`,
`ruff_csharp_spo`, `ruff_ruby_spo`, `ruff_python_spo`). The **DO-arm** extractor —
`ruff_python_dto_check` (DTO/route/handler harvest → matcher/contract dedup →
codegen, modules bundle/calibrate/codegen/config/contract/emit/extractors/matcher/
observations/preflight) — is currently **Python-only** (built on
`ruff_python_parser`).

**Consequence for the C#-MedCare / C++-Tesseract / Ruby-Redmine paths:** the DTO
triage above can be *reasoned* about, but there is **no automated DO harvester**
that emits DTO/route/handler bundles for those frontends yet. The `ruff_*_spo`
crates emit the *structural* predicates (the THINK arm); they do not emit the
DTO/handler observations that `ruff_python_dto_check` produces for Python.

**Open brick:** a language-agnostic (or per-frontend C#/C++) DO extractor that
emits the same DTO/route/handler bundle shape `ruff_python_dto_check` emits, so
the 3-bucket triage can run automatically on the non-Python corpora. Until then,
the DO triage is a manual classification over the THINK-arm harvest, not a
pipeline stage. This is the DO-arm counterpart to the THINK-arm's brick-3 (which
*did* run, cross-frontend, via `ruff_csharp_spo`).

---

## Boundary summary

| arm | what | where it lives | harvest status |
| --- | --- | --- | --- |
| THINK | class/member/type graph | `(part_of:is_a)` `FacetCascade` address | measured (brick-3, multi-frontend) |
| DO bucket 1 | order-varying behaviour | canonicalized → bucket 2 | manual (needs canonicalizer) |
| DO bucket 2 | anticipated-standard DO | imported DTO + codebook swiss-knife, keyed by GUID | Python-only extractor |
| DO bucket 3 | truly-novel behaviour | hand-port; grows the Core | manual |

The skeleton is in the address; the muscle is in `ActionDef`/`KausalSpec`, keyed
by the GUID. The 3-bucket triage is *how the muscle is partitioned* so that the
majority (bucket 2) collapses to import-and-apply, and only the sliver (bucket 3)
pays the hand-port cost.
