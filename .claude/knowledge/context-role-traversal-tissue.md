# context:role Traversal Tissue — one register, two axes, every domain

> **READ BY:** integration-lead, truth-architect, v3-envelope-auditor,
> trajectory-cartographer, convergence-architect, dto-soa-savant — and any
> session about to write a *walk* (menu path, document tree descent, time
> window, graph hop, basin scan) in ANY consumer domain. Read BEFORE
> designing a traversal; the walk you need almost certainly already has its
> lens and its register reading.
>
> **Status:** operator-ruled framing (2026-07-17, this doc is the capture);
> vertical axis operator-attested (q2 FMA helix maps) + probe-gated in-repo
> (P-HIER-LEIDEN-HHTL); horizontal axis EVENTUAL (probe named, not run).
> Labels below per the insight cycle — nothing here is silently promoted.

---

## The ruling (operator, 2026-07-17)

> *"part_of:is_a is logically consistent with **context:role** — whether
> it's HHTL family identity (**vertical**, tested in q2 FMA helix maps) or
> eventually even **horizontal** (6 context / episodic-witness basin:role /
> identity). Together with classid and appid and ClassView WideFieldMask we
> can reuse it for everything — screen and document representation, time
> series, AriGraph synergies — as connecting tissue traversal."*

This is not an analogy: it is the le-contract's own polymorphism ruling
read as a *traversal* doctrine. `.claude/v3/soa_layout/le-contract.md` §3
already rules **"the (8:8) pair is polymorphic — the classview selects the
reading"** — `part_of:is_a` (L1), `memberof:members` (L2),
`mereology:taxonomy` (L3), `palette256²` (L4), `area:location` **stacked**,
`basin:relationtype`. Every one of those is a **context:role** pair. The new
capture is the *orientation* claim:

- **VERTICAL** = the *stacked* reading. The six pairs are a **6-level radix
  path**: each pair refines within the one above ("progressive exactness",
  le-contract). Descending = adding exactness; ascending = widening. This IS
  HHTL family identity (HEEL→HIP→TWIG; `contract::hhtl::NiblePath` is the
  shipped descent primitive). Evidence: q2 FMA helix maps
  (operator-attested; contract read-modes `ReadMode::{FMA, FMA_V3}`).
- **HORIZONTAL** = the *6-slot-parallel* reading. The six pairs are **six
  contexts side by side**, each `basin:role` / `context:identity` — the
  episodic-witness basins spread across the plane (the le-contract "second
  GUID relationships" + "static-basin" variants). EVENTUAL — sanctioned,
  not yet exercised.

Same 12 bytes. The ClassView is the focus lens that picks the axis and the
reading; code never inspects payload bytes to guess (le-contract §3 slot
purity).

## The lens stack (all shipped — reuse, never reinvent)

| Layer | What it selects | Where (file:line-ish) |
|---|---|---|
| `classid` hi-u16 | the CONCEPT (canon-high) | `canonical_node.rs` key |
| `classid` lo-u16 = **appid** | the APP render prefix (who renders) | `ogar_codebook::{AppPrefix, render_classid}` |
| `classid → ClassView` | the READING of the 12-byte register | `canonical_node.rs::classid_read_mode`, le-contract §3 |
| `FieldMask(u64)` / `WideFieldMask` | which fields are present ⇒ attended ("attend to what's present" — structural, never per-row semantics) | `class_view.rs:70,221`; `view_angle.rs` doctrine |
| `ViewAngle` (4-bit) | which inherited view-schema attends (angles compete via `head2head`) | `view_angle.rs` |
| `RungLevel`/`RungElevator` | how DEEP/WIDE the walk goes this cycle | `cognitive_shader.rs:157,272`; `doc_graph.rs::retrieve` |
| `TemporalPov`/`VersionRange` + rung | WHEN — the version window the walk reads | `temporal_pov.rs` |

**One traversal shape, informally:**
`walk = (classid → ClassView reading) × (mask: which slots/fields live) ×
(axis: vertical descend / horizontal hop) × (rung: depth-or-width) ×
(pov: version window)`. A vertical step = *refine role within context*
(pair k → pair k+1). A horizontal step = *switch context, same role plane*
(slot i → slot j).

## Per-domain instantiation (the synergy map)

| Domain | VERTICAL (stacked exactness) | HORIZONTAL (6-context) | Shipped anchor |
|---|---|---|---|
| **Semantic / cognitive** | HHTL family identity; `NiblePath` descent; the rung ascent (0–1 rank → 2 hop → 3 community) | 6 CAM-PQ subspaces (L4); `basin:relationtype` rails | L1/L4 readings; `DocGraphQuery::retrieve`; S1 probe |
| **Screen (Klickwege / a2ui)** | `menu_address(class)` — "the radix-trie path lowered from the `is_a` rail" (`class_view.rs:2004`) | `screens_reachable_from(root, edges) → WideFieldMask` | class_view.rs — BOTH already live |
| **Document (ogar-doc / doc.v1)** | doc→section→block→line→word→glyph — a 6-level radix path in the SAME register shape | 6 regions / witness basins per page; DocumentID→KV handle (D-GR-6) | doc-W4 spec; tesseract doc.v1 regions |
| **Time series** | calendar radix as stacked pairs (yr:mo \| day:hr \| min:s …) — coarse:fine at every pair | 6 parallel lanes (`MultiLaneColumn`, D-DNV-1); per-lane basin | `temporal_pov` (episodic = Lance versions) |
| **AriGraph** | `Communities.levels` — the Louvain hierarchy IS a vertical stack; basin descent | 6 episodic-witness basins; PPR restart set per slot | `communities()`/`ppr` (#714/#716); S1 (#720) |

Read the table columns downward and the reuse is literal: the screen menu
path, the document tree, the calendar drill-down, the HHTL family descent,
and the community hierarchy are **the same vertical walk** over the same
register shape; reachability masks, page regions, parallel lanes, subspaces,
and witness basins are **the same horizontal frame**.

## The clean way to reuse it (the rules)

1. **Unify at the REGISTER, never at the API.** The tissue is the one
   `6×(8:8)` register + the lens stack. Carrier methods stay native
   (`menu_address` on the class index, `retrieve` on `DocGraphQuery`,
   `at(v, rung)` on the temporal reference, `communities()` on
   `TripletGraph`). Do NOT introduce a unifying `Traversal` trait — that is
   the Frankenstein flattening (unlike functions blurred into one codec too
   early) and it violates the carrier-method litmus. Convergence of *code*
   (if ever) is probe-gated, not assumed.
2. **Before writing ANY walk, check the lens stack first.** New menu logic →
   `menu_address` exists. New reachability → `screens_reachable_from`
   exists. New descent → `NiblePath` exists. New window → `TemporalPov`
   exists. New depth policy → `RungElevator` exists. A new walk that
   re-implements one of these is a rediscovery tax AND a drift risk.
3. **The ClassView selects; bytes never self-describe.** Axis and reading
   come from `classid → ClassView` (+ `ViewAngle` when angles compete).
   Never branch on payload bytes to guess the orientation (§3 slot purity).
4. **`u8:u8` stays two bytes.** Vertical stacking is *reading* six pairs as
   a path — never widening them into u16 radix digits (canon: never widen).
5. **appid renders, concept means.** classid hi = what the walk is *about*;
   classid lo (`AppPrefix`) = who is *rendering* the walk (screen vs
   document vs lane view of the same concept). Two walks over the same
   concept with different appids are the same traversal differently
   rendered — that is the whole point of the reuse.
6. **Ownership unchanged.** Traversal is read-side. Any persisted result of
   a walk lands per V3 write-on-behalf (`mailbox_owner()`), never as-self.

## Probes (falsifiers — one invariant, two orientations)

The S1 identity (*community ≡ basin ≡ is_a category*, `E-GRAPHRAG-DGR3B-1`,
probe shipped #720) is ONE invariant on this register; the orientations give
it two falsifiers, both ALREADY NAMED in the graphrag plan §6:

- **Vertical:** `P-HIER-LEIDEN-HHTL` — hierarchical Leiden super-communities
  (`Communities.levels`) vs the coarse HHTL tiers / taxonomy parents.
  Agreement ⇒ the community hierarchy and the stacked-pair family descent
  are the same vertical walk. [CONJECTURE — probe defined, NOT RUN; q2 FMA
  helix evidence is operator-attested, out-of-repo.]
- **Horizontal:** `P-COMMUNITY-BASIN-AGREE` **on the `basin:role` reading**
  — the shipped S1 harness (`examples/p_community_basin_agree.rs`) re-run
  with basins taken from the 6-slot horizontal frame instead of the graph
  `is_a` edges. [EVENTUAL — mechanism shipped, horizontal feed not built.]

Both consume `jc::reliability` (scientific crate — consumed, never extended;
`E-EMPIRICAL-VS-SCIENTIFIC-JC-1`).

## Where this lands in current + future plans

- **graphrag v1.2 (ACTIVE):** D-GR-2's rung walk = the vertical axis; the
  §4a DocumentID-KV seam = the document instantiation; §3b.1 sharpened to
  cite this doc.
- **doc-W4 (council-gated):** document reconstruction reads = the vertical
  radix path; "documents in this community" = horizontal. The `document
  0x080B` / `typed_field 0x080A` walks should be specified AS lens-stack
  walks, not bespoke.
- **temporal-markov (ACTIVE):** the sorted-stream window IS the temporal
  vertical (coarse→fine version ranges); D-MTS probes unaffected, this doc
  only names the correspondence.
- **triangle-tenants / D-TRI (P4 pipeline):** `triangle_for(family)` reads
  ride the same ClassView selection; the P4 reader inherits rule 3.
- **a2ui-rs / consumer repos:** screen representation consumes
  `menu_address` + `WideFieldMask` through the contract — consumers never
  re-derive the walk (ogar-consumer-preflight applies).

*Capture of the 2026-07-17 operator ruling; companion to
`E-GRAPHRAG-DGR3B-1` (S1 identity), `E-EMPIRICAL-VS-SCIENTIFIC-JC-1`
(probe/science boundary), and le-contract §3 (the polymorphism canon).*
