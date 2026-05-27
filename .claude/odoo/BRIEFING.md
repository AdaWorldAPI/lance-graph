# Odoo Richness Harvest — Shared Lane Briefing

> Read this fully before starting. You are ONE read-only analysis lane in a
> parallel fan-out. Output is a markdown spec draft only. **No cargo, no
> `src/` edits, no git.** Write exactly one file to
> `/home/user/woa-rs/.claude/board/odoo-richness/drafts/<your-lane>.md`.

## Mission (two stacked user directives)

1. **"Wherever odoo is richer, take the old shape and export so rich AST
   details that we can reproduce the rich business logic in Rust (woa-rs),
   and possibly later in Python (WoA — not yet)."**
   → Where odoo's logic is *richer* than woa-rs's current ERP state, capture
   the FULL behaviour — control flow, branches, field computes, ordering,
   rounding, edge cases, constraints — faithfully enough that an Opus porter
   can reproduce it in Rust **without re-reading the odoo source**. Keep the
   ontology shape (below) attached to every concept.

2. **"Check also for NARS patterns that can be delegated to thinking in
   lance-graph and OGIT-inherited thinking styles for business heuristics."**
   → Classify every business rule on a DUAL axis (below). Deterministic logic
   gets the rich-AST treatment for a Rust port. Heuristic/inferential logic
   gets flagged for *delegation* to lance-graph's thinking surface instead of
   being hard-coded as Rust if/else.

## The dual-axis classification (do this for EVERY rule you extract)

For each method / computed field / constraint, tag it:

- **AXIS-A — DETERMINISTIC → Rust port.** A closed-form rule: a balance
  check, a sequence format, a tax = base × rate, a residual = debit − credit.
  Output: rich-AST spec (see template) so it can be ported verbatim.

- **AXIS-B — HEURISTIC / INFERENTIAL → delegate to lance-graph thinking.**
  Evidence-weighted, multi-factor, ambiguous, or "best guess" logic:
  reconciliation *matching* (which open item pairs with which payment),
  fiscal-position *resolution* (which tax mapping applies to this partner),
  next-best-action, anomaly detection, stock reservation choice, dunning
  escalation judgement. These should NOT be reproduced as brittle Rust
  branches — they are delegated.

  When you tag AXIS-B, fill the **delegation tuple** using the
  `lance-graph-contract` surface (these enums already exist and are
  customer-binary-safe / BBB-allowed):

  - `ReasoningKind` ∈ { CustomerCategory, PostingAnomaly, NextBestAction,
    InvoiceCompleteness, MailIntent, Other(u32) }  — pick the closest; if none
    fits, propose an `Other` label.
  - `InferenceType` ∈ { Deduction (exact lookup), Induction (pattern/"things
    like X"), Abduction (root-cause/"why"), Revision (belief update on change),
    Synthesis (cross-domain join) }.
  - `SemiringChoice` ∈ { Boolean, HammingMin, NarsTruth, XorBundle, CamPqAdc }
    — how evidence combines (NarsTruth = evidence fusion is the common case).
  - `ThinkingStyle` cluster ∈ { Analytical, Creative, Empathic, Direct,
    Exploratory, Meta }. **This is INHERITED from the OGIT family**, not
    chosen freely: the odoo class resolves to an OGIT `FamilyEntry` via
    `resolve_odoo_to_family()` (the cache we just built — see Ontology shape),
    and the family carries the default style. State which family you'd expect
    and therefore which cluster; if the family is unmapped (returns `None`),
    say so and propose a cluster with a one-line rationale.

  A rule can be **hybrid**: a deterministic guard wrapping a heuristic core
  (e.g. "balance MUST be zero" [AXIS-A] gating "suggest which lines to adjust"
  [AXIS-B]). Tag both halves.

## Ontology shape (keep it attached — directive 1's "old shape")

We already built the OGIT→OWL→odoo cache:
`lance-graph/crates/.../lance-graph-callcenter/src/odoo_alignment.rs` and a
mirror in `woa-rs/crates/skr_data/src/odoo_alignment.rs`. The chain is:

```
odoo class  ──owl:equivalentClass──►  OWL pivot (fibo/ubl/vcard/schema)  ──►  OGIT family (8-bit) + slot (16-bit)  ──►  FamilyEntry (carries thinking style)
            resolve_odoo()                                                   OgitFamilyTable.lookup()  =  O(1)
            resolve_odoo_to_family(class, &table)  chains both legs end-to-end, O(1)
```

Seed rows already mapped: `res.partner.Company`→fibo:LegalEntity,
`account.move`→fibo:Transaction, `account.move.line`→fibo:JournalEntryLine,
`account.account`→fibo:Account, `product.*`→schema:Product, SKR concepts→
fibo:Account. Families in use: 0x61 BillingCore, 0x62 SMBAccounting,
0x80 SmbFoundryCustomer, 0x81 SmbFoundryInvoice. **Option B**: no new CAM
family, no new slot — odoo classes INHERIT an existing OGIT slot via the OWL
pivot. Classes with no existing family (`stock.move`, `sale.order`,
`hr.*`, `account.reconcile.model`, …) currently resolve to `None` — if your
lane touches one, FLAG it as "ontology-unmapped, needs a Layer-2 alignment
axiom" rather than inventing a family.

For each odoo concept your lane covers, record: `odoo:<class>` →
`owl:equivalentClass <pivot>` → expected OGIT family (or `None`). The
DOLCE marker (Endurant/Perdurant/Quality/Abstract) comes from `dolce_odoo()`
suffix rules; note it where non-obvious.

## Reading discipline (Iron Rule 4 / woa-rs Op-rule №3)

- Read the odoo Python **fully** with the `Read` tool — whole file or
  offset/limit chunks that cover the entire method. `grep`/`sed`/`head` are
  LOCATORS only, never comprehension. A snippet read produces a paraphrase
  spec that the porter then has to redo.
- Quote the odoo `file:line-range` for every rule you spec. The porter will
  spot-check against the source.
- Odoo's source is **canonical** for these semantics (we are stealing them).
  Where odoo is buggy or odd, note it; do not silently "improve".

## ERP gap context (what woa-rs is missing — the K-steps)

| K-step | Subsystem | woa-rs state |
|---|---|---|
| K3 | Double-entry posting + reconciliation | engine partial / view 501 |
| K7 | USt-Voranmeldung / tax compute / ELSTER | missing |
| K8 | German reports (BWA/SuSa/EÜR/GuV/Bilanz) | missing — engine built FRESH (odoo `account_reports` is Enterprise; only l10n_de **data/line-mappings** are stealable) |
| K9 | DATEV export | partial |
| K11 | Festschreibung (GoBD period lock) | missing |
| K12 | Anlagen (asset depreciation) | missing — odoo `account_asset` is **Enterprise**, NOT in community source |
| K13 | Lohn / payroll | missing — odoo `hr_payroll` is **Enterprise**, NOT in community (only `hr` base is present) |
| K15 | Mehrfirma (multi-company) | missing |

**Enterprise boundary — flag, do not hallucinate.** account_asset,
account_reports, hr_payroll, account_consolidation are NOT in the community
clone. If your lane's subsystem is Enterprise-only, say so explicitly and
spec only what IS present (base models, data, report STRUCTURE) — the engine
gets built fresh on the woa-rs side.

## Output template (one file, `drafts/<lane>.md`)

```markdown
# Lane <ID> — <subsystem>

## Sources read (file : line-range : depth)
- odoo/addons/.../x.py : L<a>-<b> : full
- ...

## Ontology rows
| odoo class | owl pivot | OGIT family (or None) | DOLCE |
|---|---|---|---|

## Rules extracted
### R<n> — <name>   [AXIS-A | AXIS-B | HYBRID]
- **odoo source**: file:Lx-Ly
- **What it does** (rich): <control flow, branches, ordering, rounding, edge
  cases — enough to reproduce>
- **woa-rs target**: <which K-step / which model/route this lands in>
- (AXIS-A) **Rust sketch**: <signature + key branches; Decimal/money rules; no
  hand-waving on rounding or sign>
- (AXIS-B) **Delegation tuple**: ReasoningKind=… InferenceType=… Semiring=…
  ThinkingStyle-cluster=… (inherited from OGIT family <fam>) — rationale 1 line
- **Parity notes / gotchas**: <German-tax specifics, GoBD, multi-currency, etc.>

## Enterprise gaps flagged
- <module> : <what's missing> : <what we spec from data/structure instead>

## Open questions for the Opus porter
- ...
```

## Sentinel + depth-proof (required — the agent prompts reference these)
- **First line of your draft MUST be:** `RICHNESS-LANE-OK`
- **Last section MUST be a depth-proof footer**, one line per file:
  `Read: <file> lines=<n> depth=full`

## Hard rules
- NO `cargo` (no build/check/clippy/test). NO `src/` edits. NO git ops.
- Markdown output only, to your one drafts file.
- If two lanes would overlap on a file, still read it fully for your angle;
  the Opus review pass dedups.
