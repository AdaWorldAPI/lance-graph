# copy-tierA — verdict on the 11 "TIER A borrow-carrying `derive(Clone, Copy)`" candidates

**Agent:** zero-copy warden (Tier A verification)
**Branch:** `claude/x265-x266-plans-review-h9osnl`
**Input:** `.claude/board/exec-runs/copy-derive-blast-radius.txt` § TIER A (11 sites)
**Mode:** EDIT ONLY. No `cargo build`/`check`/`test`/`clippy` run; no worktree created.
**Read first:** `AGENT_LOG.md`, `.claude/knowledge/zero-copy-lens-law.md`,
`.claude/rules/data-flow.md` §2, the `WitnessLens` comment at
`crates/lance-graph-contract/src/witness_fabric.rs:108-122` (fixed in `b3515ba`).

---

## Headline — the Tier A list is 8/11 FALSE POSITIVE. Overturned, not confirmed.

You were right to expect false positives, and the mechanism is exactly the one
you named: **the heuristic's second disjunct ("a field spelled `&`") fires on
`&'static`.** All eight non-violations were flagged on a `&'static str`,
`Option<&'static str>`, or `&'static [T]` field. **Not one of the eight carries
a lifetime parameter** — the first disjunct (the discriminating one) fires on
exactly the three types I confirmed, and on nothing else.

`&'static` is not a mailbox borrow. It points into baked program data (rodata /
a `const` table) that outlives every mailbox, so there is no compartment for it
to escape *from*. Stripping `Copy` from those eight buys zero containment and
costs real friction (they are const-table rows and by-value `self` methods).
Worse, per `.claude/rules/data-flow.md` §2 those eight ARE the "reasoning =
owned `Copy` microcopy" category the rule *requires* `Copy` on — removing it
breaks the law in the other direction.

**A sharper heuristic for the next pass:** flag on a declared lifetime parameter
ONLY (`struct X<'a>` / `enum X<'a>`). On this corpus that rule is exact:
3 hits, 3 true positives, 0 false positives, 0 misses.

I also spot-checked the Tier-B names most likely to hide a borrow behind a
misleading identifier — `EntityRef`, `EdgeRef`, `SchemaPtr`, `MappingHandle`,
`ColumnWindow`. All are packed-integer value types (`u32`/`u16` fields, no
pointers). **The heuristic did not under-count**; it only over-counted.

---

## Per-site table

| # | path:line | type | VERDICT | one-line reason |
|---|---|---|---|---|
| 1 | `crates/lance-graph-contract/src/canonical_node.rs:1492` | `NodeRowPacket<'a>` | **VIOLATION — FIXED** | holds `rows: &'a [NodeRow]` — byte-identical field to the already-fixed `WitnessLens<'a>`; `as_le_bytes` re-exports the whole 512 B-strided slab |
| 2 | `crates/holograph/src/bitpack.rs:552` | `VectorSlice<'a>` | **VIOLATION — FIXED** | holds `words: &'a [u64]` into an Arrow/mmap buffer owned elsewhere; `as_words(&self) -> &'a [u64]` re-exports at full `'a` |
| 3 | `crates/deepnsm/examples/homograph_collapse.rs:63` | `Collapse<'a>` | **VIOLATION — FIXED** | `Unique(&'a str, u32)` borrows the caller's `&[Sense]` table |
| 4 | `crates/lance-graph-callcenter/src/family_table.rs:132` | `FamilyEntry` | NOT-A-VIOLATION | no lifetime param; `&'static str` + `&'static [u8]` into the TTL-baked hydration table |
| 5 | `crates/lance-graph-callcenter/src/odoo_alignment.rs:65` | `OwlPivot` | NOT-A-VIOLATION | `pivot_uri: &'static str` + `u16` + two 1-byte enums; a §2 microcopy, `identity(self)` is `const fn` by value |
| 6 | `crates/lance-graph-callcenter/src/super_domain.rs:125` | `MetaAnchors` | NOT-A-VIOLATION | `Option<&'static str>` ×2 + marker + `Option<u64>`; a `const EMPTY` row |
| 7 | `crates/lance-graph-callcenter/src/super_domain.rs:181` | `SuperDomainEntry` | NOT-A-VIOLATION | `&'static [OgitFamily]` + `&'static str`; explicitly documented "lives in static memory, ~30 B × 8 entries" |
| 8 | `crates/lance-graph-contract/examples/foveated_awareness.rs:159` | `Card` | NOT-A-VIOLATION | `&'static str` name + 2 enums + `f32` + `u8`; its own doc already says "rides as owned microcopy" — a §2 citation, correct |
| 9 | `crates/lance-graph-contract/src/cognitive_shader.rs:143` | `StyleSelector` | NOT-A-VIOLATION | enum, `Named(&'static str)` variant only; a dispatch tag |
| 10 | `crates/lance-graph-contract/src/mul.rs:233` | `MulThresholdProfile` | NOT-A-VIOLATION (one phrasing note, below) | 3×`f32` + `label: &'static str`; three `const` profiles |
| 11 | `crates/lance-graph/examples/causal_knowledge_transfer.rs:33` | `Trajectory` | NOT-A-VIOLATION | three `&'static str` relation names; a relation *signature*, not a substrate read |

---

## What I changed (3 files, derive removed + comment added, no logic touched)

Each removal mirrors the `WitnessLens` wording you fixed in `b3515ba` (the
"a `Copy` borrow is a borrow that duplicates itself silently" paragraph), with
one site-specific sentence naming *which* compartment owns the bytes.

1. **`canonical_node.rs:1492` `NodeRowPacket<'a>`** — the one that matters most.
   Same `&'a [NodeRow]` field as `WitnessLens`, and a **wider** exposure: the
   lens hands out one 12-byte register per call, whereas
   `SoaEnvelope::as_le_bytes` hands out the entire contiguous 512 B-strided
   backing slab. A `Copy` packet is a silently duplicable second holder of one
   mailbox's whole store. Comment says exactly that.
2. **`bitpack.rs:552` `VectorSlice<'a>`** — the archetypal lens, and the file's
   own doc-comment already argues the zero-copy case at length ("copies 0 bytes
   for the 999,000 that fail the cascade"). The `Copy` derive was the hole in
   that argument: `as_words(&self) -> &'a [u64]` re-exports the borrow at the
   full `'a`, so a duplicated slice outlives every scope visible at the call
   site.
3. **`homograph_collapse.rs:63` `Collapse<'a>`** — smallest, and I did NOT
   downgrade it for that. Size is not a mitigation under this law.

## What broke at the call sites: **nothing, on inspection.**

I traced every construction and consumption of the three types. This is a
static read, not a compile — the central build is yours.

- **`NodeRowPacket`** — 12 sites (`symbiont/src/bridge.rs:129`,
  `contract/src/ocr.rs:211`, 10 in-file tests, re-export in `lib.rs:190`).
  Every one is `let pkt = NodeRowPacket::new(&rows, c);` followed by `&self`
  methods (`n_rows`, `cycle`, `as_le_bytes`, `row_le`, `verify_layout`) or the
  associated-const path `<NodeRowPacket<'_> as SoaEnvelope>::LAYOUT_VERSION`.
  No by-value pass, no field storage, no `.clone()`. **`SoaEnvelope` declares no
  `Copy`/`Clone` supertrait bound** (`soa_envelope.rs:170`), so the impl is
  unaffected.
- **`VectorSlice`** — `storage.rs` (`get_slice`, `cascaded_knn`,
  `from_bytes_or_copy` at :797), `navigator.rs` (`ZeroCopyCursor::next`),
  `hamming.rs`, plus 4 tests. Every consumer takes `&dyn VectorRef` or
  `&slice`. The one move is `return Some((id, slice, stacked.total))` in
  `ZeroCopyCursor::next` — a genuine move *after* both borrows have ended, which
  is fine without `Copy`. **`VectorRef` declares no `Copy`/`Clone` bound**, and
  there are no by-value operator impls for `VectorSlice` (the `BitXor`/`BitAnd`/
  `BitOr`/`Not` impls at `bitpack.rs:696-730` are all on `BitpackedVector`).
  Note `ZeroCopyCursor.query` is `&'a BitpackedVector`, not a `VectorSlice` — it
  was the one field that would have forced a cascade, and it does not.
- **`Collapse`** — 6 sites, all in-file. The only by-value use is
  `if let (Collapse::Unique(_, pr), Collapse::Unique(_, sr)) = (p, so)` at :157,
  which moves `p`/`so` into a tuple and never touches them again. Legal.

**No cascade found, so none refused.** If the central build disagrees, the most
likely single point is a `.clone()` I did not spot on `VectorSlice` inside a
`datafusion-storage`-gated block — `holograph/src/navigator.rs` and
`storage.rs` are partly behind that feature, and my read covered the gated code
textually but a feature-off build never type-checks it either way.

---

## One phrasing note (NOT an edit, but you should see it)

`crates/lance-graph-contract/src/mul.rs:229-230`, on `MulThresholdProfile`:

> *"The struct is `Copy` so it can sit on the BindSpace per-row carrier
> **without indirection**."*

That is one clause away from the trapped formulation the warden card lists
verbatim ("12 B inline beats a 16 B pointer plus indirection"), and
`zero-copy-lens-law.md` § "Projection is not chasing" rules the indirection term
**zero in this substrate**. I did not edit it, for two reasons I want on the
record rather than assumed:

1. The type is a §2 microcopy (3 `f32` + a `&'static str` label) and `Copy` on
   it is required, not merely allowed — so the *conclusion* is right even though
   the *argument* is one of the trapped ones.
2. The sentence describes a hypothetical ("so it **can** sit on"). Today
   `for_context(id)` is a `const fn` returning a value; nothing stores a profile
   in a per-row lane.

**But if that hypothetical is ever built, it is a violation**, and the doc
already pre-argues for it: the profile is a pure `const fn` of
`ontology_context_id`, so a per-row profile lane would be a projection of a lane
that already exists — `output_rung == max(input_rungs)`, no elevation. Worth a
line in the law's cost-argument catalogue as a live in-tree instance; your call
whether that is this task's business.

## Honest limits of this run

- Verdicts are from reading the type definitions and every call site, not from
  a compiler. "Nothing broke" means "nothing in the call graph I read requires
  `Copy`", not "it compiles".
- I judged the 11 named sites only. Tier B (358 sites) is unaudited beyond the
  five spot-checks named above.
- `crates/holograph` is not in the `[workspace] members` list in `CLAUDE.md`;
  if it is built standalone or excluded, gate it explicitly rather than assuming
  the workspace run covers it.
