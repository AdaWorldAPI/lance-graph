# odoo-classes-bitmask-render-v1 — 8-savant council verdict + auto-resolve record

> **Recorded:** 2026-05-31. Council convened on the plan at HEAD `0d2032bc` (the as-authored plan), 8 Opus reviewers spawned in parallel from one main-thread turn.
> **Status:** VERDICT (recorded for spec-owner ratification of the strategic call).
> **Purpose:** so the next session has the council's reasoning verbatim, not an executive summary. If/when the spec owner picks retire vs amend vs restructure, the verdicts here are the input.

---

## Vote distribution

| Reviewer | Lens | Verdict |
|---|---|---|
| **R1** | architectural-fit | Ship-with-3-amendments (add C8 reserve-room for discovery_origin; tighten D-CLS-3 spec-owner gate; D-CLS-9 snapshots advisory) |
| **R2** | prior-art / drift | REJECT-AS-WRITTEN — collapse D-CLS-7 into a generalization of #439's `ThoughtMask`; halt D-CLS-1 pending OD-DOLCE-VARIANT-SET; drop D-CLS-5 to u8 |
| **R3** | integration-coordination | Amend — add OD-DOLCE-VARIANT-SET (4 sites have 6/4/4/5 variants with NAME conflicts); D-CLS-5/6 must coordinate with PR #437's `MailboxSoaView::class_id()`; §6 omits `soa_view.rs` |
| **R4** | doctrine-strict | Violates 3 spec lines (classes.md:41-44 "rename" invites hand-assignment; classes.md:50 C2 is docstring promise; classes.md:73 F4 — positions persist on Odoo data before chess) |
| **R5** | test-strategy | Tests verify mechanics, not C2 invariant. Need: template grep-gate on `{% else %}`/`{% elif %}`; const-assert per-class width; Aerial+ determinism gate + noise falsifier |
| **B1** | ego/bias hunter | RETIRE — 3 of 4 OD default-leans verified-fabricated (CLASSID-WIDTH u16 is proposer-id line, CLASSID-VS-ENTITYKIND invented, DOLCE-CANONICAL author-pick). Third hallucination strike of the week. |
| **B2** | scope-creep hunter | REJECT — keep ~340 LOC (D-CLS-5 + scope-reduced 6 + contract-side 7 + NEW parent-walker); cut ~3,900 LOC (D-CLS-1 adjacent; D-CLS-2 prep; D-CLS-3 = discovered-taxonomy subsystem; D-CLS-7 ontology side; D-CLS-8 = shape-compiler-to-grid in costume; D-CLS-9 depends on 8) |
| **B3** | "stop doing things" advocate | RETIRE BRANCH — plan ungated on 4 ODs; PR #439 already shipped the bitmask pattern; classes.md:55 says "known" not "urgent"; this is the 5th plan this week without execution; spawn D-CHESS-BRINGUP-1 instead |

**Summary:** 4-of-8 push toward retire (B1, B2, B3, R2). 3-of-8 amend (R1, R4, R5). 1-of-8 restructure (R3). No 3/4 majority on a single execution path; strategic call escalates to spec owner.

---

## Verified factual corrections (auto-executed, no spec-owner gate)

The following are CORRECTIONS to plan-as-written, not opinions. They are applied to the plan regardless of whether it ships:

### F1 — `§2` OD default-leans honesty pass (B1's catch)

Three of four §2 OD "default leans" misrepresent the spec. The plan WAS amended to mark them honestly:

| OD | What plan claimed | What spec actually says |
|---|---|---|
| OD-CLASSID-WIDTH | "u16 per spec lean" | Spec line 64 says "u16" about **proposer-id** (N2). Spec is **silent** on class_id width. The lean is the author's. |
| OD-CLASSID-VS-ENTITYKIND | "coexist per orthogonal-axes argument" | `OdooEntityKind` appears **nowhere** in classes.md. The question AND the answer are author-invented. |
| OD-DOLCE-CANONICAL | "lance-graph-contract is canonical per DOLCE-Lite-Plus naming" | Spec **names no crate**. The pick is the author's taste. |
| OD-TEMPLATE-ENGINE | "askama per F3 lean" | classes.md:72 ("templates likely compile per-class (askama)") — **real spec lean**, weakly-worded ("likely") |

### F2 — Add OD-DOLCE-VARIANT-SET as 5th pre-condition (R2 + R3's catch)

The 4 sites do not just rename `Abstract`; they have DIFFERENT VARIANT COUNTS:
- `contract::cognition::entity` — 6 variants
- `ontology::hydrators::dolce_odoo` — 4 variants with `AbstractEntity`
- `arm-discovery::aerial::ontology` — 4 variants with `Abstract`
- `callcenter::super_domain::DolceMarker` — 5 variants with `Unknown`

D-CLS-1 as written fails to compile (re-export across name-incompatible enums). OD-DOLCE-VARIANT-SET added — spec owner must declare the canonical variant set, not just the canonical crate.

### F3 — PR #437 prior art (R3's catch)

PR #437 (merged) shipped `MailboxSoaView::class_id() -> &[u16]` in `crates/lance-graph-contract/src/soa_view.rs:46-58`, aliasing `entity_type`. The plan's D-CLS-5 plans a parallel `ClassId(u16)` newtype in `entity.rs` without acknowledging this. D-CLS-5/6 amended to:
- Treat the new `ClassId(u16)` newtype as a `#[repr(transparent)]` wrapper over the existing `u16` entity_type slot
- Update `MailboxSoaView::class_id()` to return `&[ClassId]` via transmute-safe `#[repr(transparent)]`
- §6 ownership matrix amended to include `soa_view.rs` under D-CLS-5

### F4 — Add D-CLS-PARENT (B2's catch)

classes.md:57 — "discriminator + parent-pointer + **parent-walking resolution against the existing cache**." Plan ships the discriminator (D-CLS-5/6) and stops. **One third of the spec's bounded-weekend triple is missing.** Added new D-CLS-10:
- `parent: Option<ClassId>` on the per-class registry
- Parent-walker function `resolve_class(class_id, &OntologyRegistry) -> ResolvedClass` that walks up to root, unions inherited fields-as-deltas (classes.md:48)
- D-CLS-7's `FieldPositionTable` amended: per-class positions are now DELTAS-over-parent, not flat unions. Classes.md:48 verbatim: *"the instance's delta from its class, as pure presence bits."*

### F5 — D-CLS-8/9 council-flag (B2 + R4's catch)

D-CLS-8 (per-class askama dispatch) IS the shape-compiler-to-grid the spec defers; D-CLS-9 snapshots ARE the N4 freeze in test-fixture costume. Both rows amended to `**Status:** Council-flagged-as-deferrable` — spec owner decides whether they ship in this plan or push to a follow-up. They are NOT auto-promoted to In-Progress when OD gates close; they require a separate ratification.

### F6 — R5's test-strategy strengthenings

Three additions to relevant D-CLS rows:
- **D-CLS-8.tests**: add template grep-gate rejecting `{% else %}` / `{% elif %}` / `{% if not mask` — structural C2 enforcement at compile-time (or `cargo build` failure via build.rs script), not runtime promise.
- **D-CLS-7.tests**: per-class width audit converted from `#[test]` to `const _: () = assert!(positions.len() <= 64);` — fails `cargo build`, not just `cargo test`.
- **D-CLS-3.tests**: add Aerial+ determinism gate (`cargo run --example odoo_66_class_discovery --seed=<X>` produces byte-identical markdown across runs, per #436's pattern) + noise falsifier (clusters > 30 OR < 4 OR singleton > 50% of input → abort wave with explicit error).

### F7 — R1's discovery_origin reserve-room note (C8)

Added C8 to plan §1: "No class_id field, ClassId table, or per-class registry may occupy a byte position reserved for `discovery_origin` widening (per core.md:55-62 N2). The classes-spec hook lands without burning byte real estate the proposer-id widening still needs."

---

## STRATEGIC DISSENT — for spec owner

The fact-fixes above are not contested. The strategic ship-vs-retire decision IS:

- **4-of-8 reviewers (B1, B2, B3, R2)** recommend RETIRE or MAJOR-REWRITE. Common thread: this is the 5th plan this week, ungated on 4 ODs, smaller pattern (#439's ThoughtMask) already shipped, chess (the N4 falsifier) still queued, plan-vs-impl ratio is wrong.
- **3-of-8 reviewers (R1, R4, R5)** recommend SHIP-WITH-AMENDMENTS. Common thread: the underlying need is real (the union-disease is documented debt); amendments above bring the plan in line.
- **1-of-8 reviewer (R3)** recommends RESTRUCTURE around PR #437's existing `MailboxSoaView::class_id()`.

**The spec owner should decide:**
- Ship the amended plan (now corrected via F1-F7) once the 5 ODs ratify?
- Retire the plan in favour of D-CHESS-BRINGUP-1 (the N4 falsifier per classes.md:79-85)?
- Cut to B2's ~340-LOC subset (D-CLS-5 + scope-reduced 6 + contract-side 7 + new D-CLS-10 parent-walker) — the genuine bounded-weekend?

**The plan's own §2 says:** "No agent should start D-CLS-1+ until all four (now five) ODs are answered." So none of the strategic options requires action TODAY. The plan + the council record sit waiting for spec-owner OD ratification regardless.

End of verdict.
