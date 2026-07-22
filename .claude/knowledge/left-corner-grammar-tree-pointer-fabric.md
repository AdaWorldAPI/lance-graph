# Left-corner parsing × the grammar tree as pointer fabric — four-paper double-check

> **READ BY:** any agent touching `deepnsm-v2` (`fsm.rs` / `wave.rs` / the 24×i4
> reference pointers), the grammar-resolver / temporal-stream seam
> (`E-MARKOV-TEMPORAL-STREAM-1`, `E-HORIZON-NOT-BOUND-1`), or any future
> rule-inventory / SPO-emission design. Status: FINDING (paper-grounded,
> 2026-07-22, four parallel Opus reviews of the primary sources).

**The operator's focal question:** *"the grammar tree associated"* — can the
grammar TREE associated with a sentence be represented/recovered in the
deepnsm-v2 substrate (linear SPO stream + 24×4bit signed reference pointers
±8 + global-graph escalation), instead of an explicit tree datastructure?

**Answer: YES — and the literature's own reference implementation already does
it that way.** With one load-bearing distinction (depth ≠ span) and one
citation-provenance correction, below.

## The four papers

| paper | what it is | headline for us |
|---|---|---|
| **Manning & Carpenter 1997** (IWPT-97, PLCG) | probabilistic left-corner grammars, WSJ | **the crown jewel** — tree-as-pointer-fabric IS its implementation; Table 7: max LC stack depth over ALL of WSJ = **8**, ~99.4% ≤ 5, ≥90% ≤ 3 |
| **Roark & Johnson 2000** (arXiv cs/0008017) | probabilistic top-down/LC, fully-connected left context, beam | non-local (ancestor/global) context improves accuracy AND efficiency (p.5-6) — the escalate-to-graph channel is load-bearing, not a fallback |
| **Moore 2000** (IWPT-2000) | optimized LC chart parsing, large CFGs | cheap-check-before-expensive-check (+67%); BUPM > left-factoring; minimal 2-field back-pointer suffices for full tree recovery (§7); rules-as-stored-inventory + precompiled LUT strongly validated |
| **Liu 2025** (JLM 13(2), LC parsing of Minimalist Grammars) | psycholinguistic memory modeling | arc-eager LC = human profile ⟨O(1) left, O(n) center, O(1) right⟩ (Table 2); center-embedding caps at 3 in natural text (Karlsson 2007); FSM's real blind spot = MOVEMENT (object relatives, fronting, wh) |

## The verdict ladder

1. **Tree = pointer fabric over the stream: VALIDATED, strongly.**
   Manning & Carpenter p.153: *"partial parses are maintained as pointers to
   positions in trie data structures that represent the list of parser moves …
   At the end of parsing, the lists of parser moves can be easily turned into
   parse trees."* Plus the bijection (p.150): each tree has a UNIQUE LC
   derivation. The explicit tree object is an output convenience; the substrate
   carrier is stream + pointers. Moore §7 independently shows a **2-field
   back-pointer** (mother category + start position) suffices to reconstruct
   every parse — precedent that minimal pointers carry the tree.

2. **The "8" is empirical, and it bounds DEPTH, not SPAN.**
   Manning & Carpenter Table 7 (p.155), every configuration in binarized WSJ:
   depths 1..8 with counts 71,681 / 108,544 / 39,105 / 17,000 / 3,745 / 1,291 /
   160 / **23**. Max = 8; 99.4% ≤ 5. The deepnsm-v2 **24-pointer register is
   ~3× the observed open-constituent ceiling.** BUT: the bound is on the number
   of simultaneously-open constituents (candidate attachment sites), NOT on the
   token distance a single attachment link crosses — one stack slot can span an
   arbitrarily long relative clause. **Consequence: the global-graph escalation
   must fire on ±8 OFFSET overflow (long-distance attachment), not only on
   depth overflow.** The papers' data predicts that tail is rare.

3. **Citation provenance (correction — two reviewers independently).**
   The "LC needs bounded stack for left/right-branching, grows only on
   center-embedding" theorem is **Abney & Johnson 1991 / Resnik 1992 /
   Stabler 1994** (imported by Manning & Carpenter §5 and by Liu 2025) — it is
   NOT in Roark & Johnson 2000 and NOT in Moore 2000. Do not cite those two for
   it. Liu 2025 Table 2 is the modern MG-side confirmation of the same profile;
   Karlsson 2007 (via Liu p.320) gives the natural-text center-embedding cap of
   3 layers.

4. **±8 as "left-corner bounded memory": the honest framing.**
   The ±8 window is a **recency prior over the stream, rescued by escalation**
   — sound because `wave.rs` escalates (`WaveGrounding::Escalate`, "a distant
   cause is still a cause"), and empirically well-sized because the
   open-constituent working set never exceeded 8 in WSJ. It is NOT the
   structural LC stack bound itself (different quantity). The code was already
   right; the justification text needed this correction.

5. **Escalation is the load-bearing channel, not a wart.**
   Roark & Johnson §3.3: parent/ancestor annotation (non-local context)
   improves accuracy AND cuts states-considered — *"the non-local information
   not only improves the final product of the parse, but it guides the parser
   more quickly."* The global SPO/AriGraph graph is deepnsm's ancestor-
   annotation channel; expect escalation to make resolution cheaper, not just
   more correct (testable prediction for probes).

6. **The flat 6-state FSM: right scope, one named blind spot.**
   Moore's Table 1 (PT grammar: **7.2×10²⁷ parses/sentence average**) is the
   control experiment justifying NOT building a full CFG parser. Manning &
   Carpenter run at POS level with deliberately flat treebank structure —
   flatness is a feature. The FSM's real failure mode (Liu 2025) is **movement**:
   object relatives / topicalization / wh-fronting invert canonical S/O order,
   so first-noun=subject mis-assigns the triple. Deterministic tie-breaks
   ("last verb wins", "re-anchor subject") are recency heuristics, not
   attachment decisions — documented as such in `fsm.rs`.

## Adopted invariants + logged forks

- **INVARIANT (Moore, +67%): cheap O(1) local check gates the expensive global
  check.** Wherever the resolver does two-stage filtering (±8-local vs
  graph-global), run the local check first; escalate only on its failure.
- **INVARIANT (Moore §7): keep reference pointers minimal** (identity +
  position); the rest is reconstructible. Argues against widening the 4-bit
  pointers.
- **FORK (logged, not built): modifier attachment** via RB0-style
  underspecified continuation (Roark & Johnson p.3) — would turn the FSM into a
  minimal pushdown; a real scope increase for "the big old dog" → structured NP.
- **FORK (logged, not built): promote Relativizer/Complementizer out of
  `Pos::Other`** (Liu 2025) — the cheapest mitigation for the movement blind
  spot; clause-boundary token that should NOT positionally reset S/O.
- **WARNING (Moore Table 3): do not naively left-factor a future SPO rule set**
  — left factoring injected empty categories and degraded LC parsing,
  sometimes catastrophically; bottom-up prefix merging (BUPM) is the right
  transform for a bottom-up streaming recognizer.
- **KEEP: no PCFG/beam/FOM machinery** in the FSM — determinism is a feature
  at this scope (no recursion → trivially cannot non-terminate).

## The scarcity inversion (operator observation, 2026-07-22 — read before importing ANYTHING else from this literature)

**In the whole left-corner parsing history there was never a substrate that
could hold all meanings of a book in parallel.** Every design decision in the
four papers is downstream of that scarcity:

- **Beam / k-best** (Roark & Johnson, Manning & Carpenter) — competing analyses
  are *discarded* because they cannot be held. A scarcity artifact, not a
  principle.
- **Packed chart** (Moore) — the ceiling the field reached: all parses of ONE
  sentence, polynomial space, **syntax-only**, discarded at the sentence
  boundary. The 7.2×10²⁷ ambiguity is structural; *meaning was never in the
  parser at all* — no semantic substrate existed.
- **Perfect oracle** (Liu) — ambiguity idealized away entirely.
- **Per-sentence reset** (all) — cross-sentence meaning (coreference,
  discourse) out of scope for the entire tradition.

**What the 64k SoA changes:** a whole book (≤64k verses/SPO = one 256×256
tile) resident at once; every token a 96-bit `6×cosine²` DISTRIBUTION — its
meaning-*spread*, not a beam-chosen point — all co-addressable. Consequences:

1. **Do NOT import beams / k-best / prune-at-parse.** Destructive choice at
   parse time was forced by RAM, not by language. Here ambiguity persists as
   the distribution and is resolved by a **per-reader READ**
   (`QueryReference::at(v, rung)` — late binding, non-destructive, replayable).
2. **R&J's ancestor-annotation gain generalizes to triviality.** They smuggled
   non-local context into category labels because the parser could not see
   back. In this substrate the whole book IS the resident annotation, O(1)
   addressable — the mechanism that made their parser both better AND faster
   is the substrate's default posture.
3. **No sentence-boundary reset** — coreference/discourse become in-scope for
   the SAME machinery (the pointer fabric + escalation), not a separate system.
4. **What survives unchanged:** depth ≤ 8, the pointer fabric, table-driven
   mechanics, cheap-check-first — these are properties of *language*, not of
   1997 RAM. Import the linguistics; leave the scarcity workarounds behind.

**Honest bounds:** "all meanings in parallel" = co-resident distributions +
stored edges + pointer fabric, at BOOK scale (64k cells × 6 rails) — not
unbounded superposition in one cell. `I-VSA-IDENTITIES` still applies: the
distributions are trained-codebook cells (meaning-spread), never superposed
content registers.

## Cross-refs

`E-CAM96-DISTRIBUTION-MEASURED-1` (the meaning-substrate measurements this
grammar layer sits on), `E-HORIZON-NOT-BOUND-1` (horizon = reference, not
bound — now paper-grounded), `E-MARKOV-TEMPORAL-STREAM-1`,
`E-NO-BUNDLE-STANDING-WAVE-1`, `E-LC-SCARCITY-INVERSION-1`,
`crates/deepnsm-v2/src/{fsm,wave}.rs`, plan `deepnsm-v3-convergence-v1`.
