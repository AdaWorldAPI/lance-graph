# copy-deepnsm — `derive(Clone, Copy)` verdicts for deepnsm / deepnsm-v2 / reader-lm

**Agent:** copy-derive auditor (deepnsm family)
**Branch:** `claude/x265-x266-plans-review-h9osnl`
**Mode:** EDIT ONLY. No `cargo build`/`check`/`test`/`clippy`; no worktree.
**Read first:** `AGENT_LOG.md`, `.claude/knowledge/zero-copy-lens-law.md`,
`.claude/rules/data-flow.md` §2 (in `/home/user/ndarray/`),
`.claude/rules/borrow-strategy.md` (in `/home/user/q2/`) — neither rules file
exists under `lance-graph/.claude/rules/`; both were loaded from the sibling
repos where they live.

---

## Headline — 63 sites, 1 violation, 0 edits by me (a sibling session got there first)

The census listed **26** sites for these three crates. My own exhaustive grep
found **63**. The census missed `deepnsm/src/comprehension.rs`,
`markov_bundle.rs`, `quantum_mode.rs`, and **all of `deepnsm-v2` except
`shape.rs:311`** (20 sites) — it matched only `#[derive(Clone, Copy…)]`, and
deepnsm-v2's house style is `#[derive(Debug, Clone, Copy…)]`.

Of the 63, exactly **one** carries a borrow: `Collapse<'a>`. It was already
fixed by the concurrent `copy-tierA` session (see § Overlap). The other 62 are
LEGITIMATE or ELEVATED. **No derive should be removed from any of them** — and
for many, removing it would break `data-flow.md` §2 in the other direction.

The most valuable finding is NOT a derive at all: **`WitnessStream::window_at` /
`window_range` (`deepnsm-v2/src/wave.rs:133,146`) return
`Vec<(usize, CausalWitnessFacet)>`** — byte-for-byte the gathered-window shape
the zero-copy law names as its canonical measured instance. See § The real one.

---

## Method — provenance, not size

The operator's question per site: *is this a VALUE the reasoning layer passes by
value, or a RECORD of substrate bytes owned elsewhere?* Size does not decide it.
Three mechanical tests, in order:

1. **Declared lifetime parameter** (`struct X<'a>` / `enum X<'a>`). A scripted
   scan over all 66 `.rs` files extracting each `Copy` type's full body and
   testing for `&'`, `: &`, `&[`, `&str`, `&mut` **and** a lifetime on the decl
   returned **exactly one hit** — `Collapse<'a>`. Zero `&'static` false
   positives in these three crates (unlike the workspace-wide Tier A list).
2. **Byte-slab constructor.** Grepped every
   `from_bytes` / `from_le_bytes` / `from_slice` / `ref_from` in scope. Three
   hits, none on a `Copy` type in the census (`similarity.rs` only).
   Specifically: **`Fingerprint16K` has no byte-slab constructor at all** — every
   path (`ZERO`, `from_centroid(u16)`, `from_centroid_semantic`, `xor`, `bundle`)
   is generative, and `as_bytes` is the *outgoing* lens.
3. **Gather-out-of-a-live-store.** Grepped `.copied()` / `.cloned()` /
   `.to_vec()` / deref-in-map across the three crates. This is what surfaced
   `wave.rs`.

---

## The one VIOLATION

| site | type | verdict | reason |
|---|---|---|---|
| `crates/deepnsm/examples/homograph_collapse.rs:63` | `Collapse<'a>` | **VIOLATION** | `Unique(&'a str, u32)` holds a borrow into `Sense::lemma`, a `String` owned by the caller's `HashMap<String, Vec<Sense>>`. A `Copy` borrow duplicates silently — it can be stored beside the original and leave the compartment owning the sense table with no move and nothing in a diff to point at. |

**Cascade: none.** `Collapse` is only ever matched or moved — `match
collapse(…)` (l.120), the tuple moves `(p, so)` (l.157) and
`(collapse(…), collapse(…))` (l.213). No `.clone()` anywhere. Both `Clone` and
`Copy` are removable together with zero call-site change.

**I did not make this edit** — see § Overlap. The landed edit is correct and is
what I would have written.

## The watch-list — all LEGITIMATE, with the provenance that settles each

The operator flagged six as "LARGE `Copy` structs". **Four of the six are not
large**, and none is a record of bytes owned elsewhere:

| site | type | size | verdict | reason |
|---|---|---|---|---|
| `deepnsm/src/fingerprint16k.rs:19` | `Fingerprint16K` | **2048 B** | LEGITIMATE | The only genuinely large one. **Generative provenance**: a pure function of a `u16` centroid (golden-ratio hash), or an XOR/bundle of other fingerprints. No slab it could be a second reading of. This is `data-flow.md` §2's named `Fingerprint` microcopy at 16 K width. *Separate finding:* **zero consumers outside its own module** (see § Adjacent). |
| `deepnsm/src/signed_crystal.rs:222` | `Crystal4096` | **2 B** | LEGITIMATE | `#[repr(transparent)] (u16)`. Three packed 4-bit axes, computed from three `i8` offsets. A coordinate the reasoning layer passes by value. |
| `deepnsm/src/signed_crystal.rs:320` | `SignedSentenceCrystal` | **16 B** | LEGITIMATE | `P64` (u64) + `Crystal4096` (u16). The two fields are **independent** — `coord` is built from window offsets, not folded out of `p64` — so it is not a projection of its own other field. Contrast `Sentence64` below. |
| `deepnsm/src/sentence_transformer64.rs:108` | `P64` | **8 B** | LEGITIMATE | `#[repr(transparent)] (u64)`. 8 lanes computed from vocabulary ranks + grammar tags + NSM masks. |
| `deepnsm/src/sentence_transformer64.rs:260` | `Cam4096` | **2 B** | LEGITIMATE *as a type* | `#[repr(transparent)] (u16)`. But see § Stored projections — it is *stored beside* the `P64` it folds from. |
| `deepnsm/src/sentence_transformer64.rs:341` | `Perturbation4x4` | **16 B** | LEGITIMATE | `[u8; 16]` of signed nibble pairs, built by `local_tile` from two step sizes. A generated tile, owned by nothing else. |
| `deepnsm/src/sentence_transformer64.rs:560` | `Sentence64` | **~14 B** | LEGITIMATE *as a type* | See § Stored projections for the `cam` field. |
| `deepnsm/src/cam64.rs:37` | `Cam64` | **8 B** | LEGITIMATE | `(u64)`, 8 lanes. The module doc is explicit that it is a locality **key**, not the truth — an identity, per `I-VSA-IDENTITIES`. |
| `deepnsm/src/window.rs:87` | `WindowEntry` | **~24 B** | LEGITIMATE | Holds **vocabulary ranks** (`[u16; 4]`), i.e. identity pointers into the vocabulary, never content — exactly the `I-VSA-IDENTITIES` shape. Its owner `SentenceWindow` is a `[WindowEntry; 11]` ring buffer that owns them outright and is itself **not** `Clone`/`Copy`. Pulling one entry out of a ring you own is a microcopy, not an escape. |

## LEGITIMATE — small owned value microcopies (`data-flow.md` §2 REQUIRES `Copy`)

PoS tags, parser states, roles, flags, small SPO records. Removing `Copy` here
breaks the rule in the other direction. 39 sites:

- **deepnsm/src** — `pos.rs:7 PoS` · `parser.rs:19 State`, `:34 ModRelation` ·
  `morphology.rs:36 MorphFlags(u16)` · `spo.rs:23 SpoTriple(u64 packed)` ·
  `episodic_spo.rs:39 DependencyRole`, `:54 ClauseRole`, `:67 DiscourseRole` ·
  `reader_state.rs:50 LeftCornerTrigger` · `crystal_neighborhood.rs:50
  NeighborhoodMetric` · `markov_bundle.rs:18 Kernel`, `:37 GrammaticalRole` ·
  `quantum_mode.rs:14 PhaseTag(u128)`, `:49 HolographicMode` ·
  `signed_crystal.rs:88 HorizonPolarity`, `:152 SignedOffset4(u8)` ·
  `window.rs:62 ExpectedReason`, `:79 ExpectedSlot` ·
  `sentence_transformer64.rs:401 SplatNeighbour`, `:521 EpisodicSpoHint`
- **deepnsm/examples** — `causal_edge_v3_facet.rs:52 Verb`, `:69 CausalEdgeV3` ·
  `gridlake_coca_wire.rs:17 Cell` · `gridlake_spo_ngrams.rs:21 Cell` ·
  `homograph_collapse.rs:37 Role` · `spo_anaphora_nibble.rs:49 Noun`,
  `:56 Pron` · `spo_markov_kg.rs:59 Truth`, `:79 Role`
- **deepnsm-v2/src** — `belief.rs:32 Stamp(u64)`, `:55 Copula`, `:79 CStmt` ·
  `fsm.rs:39 Pos`, `:65 Tagged`, `:82 State`, `:95 Rel` · `spo.rs:12 Spo` ·
  `shape.rs:49 ShapeClass`, `:65 Representation`, `:311 Color` (a `#[cfg(test)]`
  DFS colour)
- **reader-lm/src** — `classifier.rs:5 HtmlStructure`

Two worth naming individually:

- `deepnsm/examples/causal_edge_v3_facet.rs:69 CausalEdgeV3
  { classid: u32, payload: [u8; 12] }` — the canonical V3 4+12 facet. **Not** a
  lens violation: the probe *constructs* it via `new()` + bit setters and is the
  origin of its own bytes; there is no substrate slab underneath it to be a
  second reading of. (In the real substrate the read path is
  `from_register_ref(&[u8;12]) -> &Self`, a cast — that contract is unaffected.)
- `gridlake_*.rs Cell` — accumulators in a `Vec<Cell>` the example owns, and
  already accessed through `&grid[c]` / `&mut grid[rank]`. The `Copy` is
  **unexercised** (only `Clone` is needed, for `vec![Cell::default(); GRID]`).
  Latent, not live.

## ELEVATED — strictly higher rung than every input

Facts *about* a set of beliefs/edges, not members of it. Per the 2026-07-29
refinement of the rung test these are reproducible only by a computation across
multiple reads yielding a value of a **different kind** — the
`Locus::Quorum` / `Contradiction` shape. Storing and passing them by value is
correct. 12 sites, all `deepnsm-v2/src`:

| site | type | rung it lifts to |
|---|---|---|
| `shape.rs:103 ShapeReport` | graph-shape class + recommended representation over an edge set | rung 3 — a tactic recipe selection (`Representation`), not an observation |
| `shape.rs:188 MeasuredShape` | + measured coverage / amortization / residue | rung 3, with the measurement attached |
| `reason.rs:58 GateReport` | resolvability %, acyclicity, termination over a derivation | rung 3 — a property of the inference run |
| `evidence.rs:338 ForwardGateReport` | real vs null vs baseline ρ | rung 3 — the falsifier's verdict, exists in no belief |
| `evidence.rs:39 EvidenceBasin` | per-subject confidence/contradiction/rung aggregates | rung 2→3 — a fact about the belief population |
| `basin.rs:41 BasinCode` | basin width + members + contradiction | rung 2→3, same shape |
| `basin.rs:152 HeldOutGate` | held-out ρ vs floor | rung 3 — a gate verdict |
| `introspect.rs:24 ProvenanceReport` | derived vs composed counts | rung 3 — a fact about derivation history |
| `introspect.rs:79 ConfidenceAnswer` | c1/c2/δ across two version reads | rung 3 — **a cross-term**: exists in neither read alone |
| `belief.rs:110 ReviseOutcome` | Admitted / Revised{synthesis_c, depth} / Chosen | rung 3 — the NARS revision verdict; `synthesis_c` is the Horizontverschmelzung cross-term |
| `episodic_spo.rs:215 BasinClassification` | Reinforcement / Novelty / Wisdom / Contradiction / Epiphany | rung 3 — a contradiction is strictly higher than the observations it reconciles |
| `episodic_spo.rs:90 EpisodicSpoFrame` | the auditable witness row | **origin**, not a projection — produced by `ReadingState::step()`; nothing else holds these bytes first |

`EpisodicSpoFrame` deserves the explicit note: at ~90 B it is the largest
non-fingerprint `Copy` here and it *is* stacked in `Vec<EpisodicSpoFrame>` for a
SIMD sweep, which superficially reads like a gathered window. It is not — the
frame is the **producer** of its own bytes, not a reading of a slab that owns
them. The `Vec` is the substrate, not a copy of one.

---

## ★ The real one — a gathered window, not a derive (`wave.rs`, REPORTED not fixed)

```rust
// crates/deepnsm-v2/src/wave.rs:133 and :146
pub fn window_at(&self, ref_version: u64) -> Vec<(usize, CausalWitnessFacet)> {
    self.events.iter().enumerate()
        .filter(|(_, (v, _))| pov.admits(*v))
        .map(|(pos, (_, r))| (pos, *r))      // ← the copy
        .collect()
}
```

This is **the canonical measured instance of the zero-copy law, re-created**:
the doc's §"canonical measured instance" names `window: &[(usize,
CausalWitnessFacet)]` in `witness_fabric` as the ~768 KB-per-resolve violation
that motivated the whole law. `self.events` stays alive across the call, so
every returned facet is a second stored reading of bytes that already have one.
`ground_at` (l.173) and l.192 feed the result straight into
`standing_wave_grounded(idx, &window, …)` — the exact function the ZC-2
migration was about.

**Why I stopped rather than fixed it** (per the brief's cascade rule):

- The `_lens` twins already exist (`standing_wave_grounded_lens`,
  `resolve_chain_lens`, taking `&WitnessLens<'_>` + `visible: impl Fn(usize) ->
  bool`), so this *looks* like a one-line swap. **It is not.**
- `WitnessLens<'a>` borrows `&'a [NodeRow]` and casts at
  `NODE_ROW_STRIDE`-strided offsets. `WitnessStream::events` is
  `Vec<(u64, CausalWitnessFacet)>` — a tuple vec, **not** a strided row slab.
  `WitnessLens::at(pos)` cannot cast into it.
- This is precisely the doc's §"Not every gathered slice is a window" warning:
  *name the source before writing the twin.* The repair is either (a) reshape
  `WitnessStream::events` to a row slab so the existing lens applies, or
  (b) write a stream-shaped lens over `&[(u64, CausalWitnessFacet)]` with the
  predicate filter. Both change `WitnessStream`'s public API and touch
  `ground_at`, l.192, `examples/bible_wave.rs:267`, and 6 in-file tests.

**The correct shape already exists one file over**, which is the strongest
evidence this was a miss rather than a decision: `TemporalStream::window_at`
(`deepnsm-v2/src/lib.rs:201`) returns `impl Iterator<Item = &Spo> + '_` —
borrowed, filtered by predicate, zero copies. Two `window_at` methods in one
crate, one right and one wrong.

## Stored projections — real, but the `Copy` derive is not the lever

Two fields are **recompute-equal by construction**, i.e. a cache with a
correctness liability rather than a memory:

- `Sentence64.cam` (`sentence_transformer64.rs:563`) — the *only* constructor is
  `Sentence64::new`, which sets `cam: Cam4096::from_p64(p64)`. So `cam` is
  always exactly `Cam4096::from_p64(self.p64)`; a pure 3-nibble fold of a field
  the same struct already holds. Same kind (an address), same rung (a coarsening
  of the meaning field, not an elevation) → **projection stored**.
- `SplatNeighbour.cam` (`:406`) — identical shape, both construction sites.
- `EpisodicSpoHint` (`:521`) — re-stores `subject`/`predicate`/`object`/`role`
  read out of an `EpisodicSpoFrame` that outlives the read in
  `project_frames(frames: &[EpisodicSpoFrame]) -> Vec<Sentence64>`.

**Refused, deliberately.** Removing `derive(Copy)` from these does nothing about
it — the duplication is in the struct's *shape*, not its copy-ability. The
repair is `fn cam(&self) -> Cam4096 { Cam4096::from_p64(self.p64) }` (delete the
field) and a borrow for the hint; that is a struct-layout change rippling
through `project`, `project_from_frame`, `project_frames`, `splat_p64`,
`same_basin_as` and the module's tests. Reported for a scoped follow-up.

## Adjacent — `Fingerprint16K` has no consumers

`grep -rn 'Fingerprint16K'` over all of `crates/` returns **zero hits outside
`src/fingerprint16k.rs`** (it is `pub mod`-exported at `lib.rs:116`). Not the
doc's `#[cfg(test)]`-only "shadow of storage" — the constructors are real
production functions — but the same smell one step milder: a 2 KB `Copy` type
with a full API, 19 functions, its own test module, and nothing reaching for it.
Worth a decision (wire it or delete it) before it grows.

## Overlap with the concurrent `copy-tierA` session — no duplicate edit made

While I was reading, `crates/deepnsm/examples/homograph_collapse.rs` changed
under me (mtime `14:53:49`); `git status` showed it modified alongside an
untracked `.claude/board/agent-tags/copy-tierA.md`. A sibling session is running
the **Tier A sweep across all crates**; its scope and mine intersect at exactly
this one site, and it landed the removal with a why-comment first.

Per the `AGENT_LOG` 2026-07-29 lesson (`--force-with-lease` refusing a duplicate
fix; resolved by discarding the duplicate and taking the sibling's as canonical)
I **left their edit untouched and wrote nothing of my own**. I verified it is
correct: both `Clone` and `Copy` are gone, no `.clone()` on `Collapse` exists,
and every use site is a match or a move — it compiles with no call-site change.

Their independent conclusion also **confirms my method**: they report the
"declared lifetime parameter ONLY" heuristic is exact workspace-wide
(3 hits / 3 true positives / 0 false positives), and the `&'static` field is the
false-positive generator. My scripted scan over these three crates found exactly
1 lifetime-param site and 0 `&'static` fields — consistent, from a different
direction.

## Edits made by me

**None.** The single violation in scope was already fixed by the sibling
session. The other 62 sites are LEGITIMATE or ELEVATED and must keep `Copy`.
