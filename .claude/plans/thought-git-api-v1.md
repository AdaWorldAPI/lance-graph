# Thought git API (D-TGA) — v1

**Status:** PROPOSAL, 2026-10-17. Test-first; nothing built.
**Asked for:** "What would be genuinely valuable would be a git like API."
**Model (decided 2026-10-17, `.claude/knowledge/thinking-tables-wiring.md`
row 11):** a thought table is compute and storage. Thoughts `1..=63` branch off
main (thought 0, the givens of the current sealed version); a finished thought
is squash-committed into the next sealed version, freeing its slot; a
selection of thoughts can be squashed into a kept intermediate thought; the
rest of git applies.

## Overview

One small surface over pieces that exist, so a caller reasons about thoughts
the way they reason about branches, and every verb lowers to mask algebra and
Lance versions. No new algebra, no interpreter, no strings as addresses:
a thought is its number (`0..=63`, 64 in all), a version is a Lance version, a cell is a
coordinate; names are labels resolved once through the label codebook.

## What exists and is reused

| git verb | existing piece | file |
|---|---|---|
| reserve the address space once, share it | `AlphaAllocation` borrowed by many overlays (`over_shared`) | `crates/lance-graph-contract/src/alpha.rs` |
| one writer per branch, no lock | `AlphaTunnel` lanes, each with its own `&mut` | `crates/lance-graph-contract/src/alpha_tunnel.rs` |
| deterministic merge order | `AlphaTunnel::merge`, ordered by `(rung, seq)`, never by completion | same |
| diff between two branches | `AlphaOverlay::only_in` | `alpha.rs` |
| delete a branch | `AlphaOverlay::discard` (consumes self) | `alpha.rs` |
| mask set algebra | `AlphaMask::{and, or, and_not, xor, not}` | `alpha.rs` |
| givens, contradiction records | D-HL givens layer | `ce64-hypothesis-layers-v1.md` section "Givens layer: hydration is thought 0" |
| written-by word, coverage orientations | D-HL arm M, R/C | same plan, section "Coverage orientation: per-thought masks or per-row words" |
| source bitset (blame) | D-HL-3 evidential base | same plan, under D-HL-3 |
| review before merge | revision gate G1–G7 | `.claude/board/entries/2026-10-14-revision-layer-invariants.md` |

**The delta is the attention saccade (DECISION 2026-10-17).** A thought's
delta is not a second kind of thing beside the alpha overlay: it *is* the
overlay, the record of where attention went, the way a PR is the attention
of a code update. BASIS: "The delta IS the attention saccade. The same way a
PR is the attention of a code update." So the thought type is the alpha
overlay, carried to 64 thoughts. This is consistent with the overlay's own
doc: what it forbids is *reference* data (a value a bake measured,
calibration, coverage) living in the overlay, and it says a changed reference
value is a new reference version, never an overlay write. That is exactly
`land`: a squash writes the next sealed version, never into the overlay or
the current givens.

(Superseded first reading, same day: this paragraph previously kept a
separate thought type because "alpha holds attention, thoughts hold values".
The distinction the alpha doc draws is attention vs reference, not attention
vs value.)

What changes in the overlay to carry it:

- **Axis.** `AlphaTunnel` has 10 lanes, one per rung; D-HL places up to 64
  thoughts beside the 10 rung lanes. The thought axis is a second index over
  the same shared allocation, not a re-use of the rung index.
- **Lookup.** `AlphaOverlay` finds a claimed row through a `HashMap`; under
  the rule against key reads, positions become the rank in the overlay's own
  presence plane (see "No hash lookups" below).
- **Values (Codex review on #1599).** Today a claimed overlay row carries
  only its `AlphaStamp` in value slot 0 and zeroes the rest of the value
  slab and the edge block (`alpha.rs`, `claim`). That records where
  attention went, not what the thought wrote there, so `diff`, the rebase
  value comparison, `blame` and `land` have nothing to read. Carrying the
  overlay to thoughts therefore needs the written cells (lane, value,
  source bits) at the same coordinate as the claim. Two layouts keep that
  rule: (a) the claimed row's own value slab, after the stamp; (b) a sparse
  value plane per thought indexed by the same presence rank. The choice is
  D-TGA-1's first decision; either one changes a stored layout and needs
  the field-isolation tests (I-LEGACY-API-FEATURE-GATED). The values a
  thought writes are its own derivations, not reference data, so neither
  layout conflicts with the alpha doc's rule.

## The surface (methods on carriers)

```text
Main            (sealed version v, givens table thought 0)
  .branch()                  -> Result<Thought, BranchesFull>   // takes a free slot in 1..=63
  .version()                 -> LanceVersion
  .checkout(v)               -> Main                            // read-only, Lance version read
  .log()                     -> versions with the squash that made each
  .blame(addr, lane)         -> SourceBits                      // D-HL-3

Thought         (slot 1..=63, merge_base = the version it branched from)
  .write(addr, lane, cell, source)                              // W; givens refuse, contradiction recorded
  .read(aperture)            -> mask                            // R, recorded (new plane)
  .diff()                    -> mask                            // W
  .status(&main)             -> {behind, clean, agreed, conflicts, stale}
  .rebase(&main)             -> RebaseReport                    // four masks, no data moved
  .squash_into(self, others) -> Thought                         // partial squash, kept intermediate
  .discard(self)                                                // frees the slot

MergeQueue      (FIFO of thoughts ready to land)
  .enqueue(Thought)
  .land(&mut main)           -> Result<LanceVersion, Refused>   // up-to-date rule, gate, squash
```

`squash_into`, `discard` and `enqueue` take the thought by value, so a
squashed or discarded thought cannot be written again: slot freeing is a
compile-time property, as `discard` already is for overlays.

## How the verbs are executed (the optimisation)

Every verb is one of four existing executions; which one is chosen by the
shape and density of its operands, never per verb by hand.

| execution | what it is | verbs that use it |
|---|---|---|
| **range fold** (Schnittpunkt, `fold_ranges`) | intersect apertures symbolically before anything runs; most reads are apertures, so `R ∧ M` is often a range intersection with no plane touched | `read`, `status`, `rebase` (stale check), `checkout` bounds |
| **presence-bit masks** | word-wise AND / AND-NOT / popcount over the 65,536-bit planes, dense and branch-free; measured 0.9–1.2 ns per 64-address word (H14c) | `diff`, `status`, `rebase` when W, R, M are dense |
| **leapfrog seek** (`OrderedLane::seek_from`, galloping) | intersect sorted sparse address lists in time proportional to their alternation, not to the plane size | `rebase` and `status` when W, R, M are sparse; `blame` lookups |
| **ordered merge / fold** | k-way merge of sorted sparse deltas by address in a fixed order, as `AlphaTunnel::merge` does by `(rung, seq)` | `squash_into`, `land`, `MergeQueue` (FIFO), the stale-read fixpoint re-run (semi-naive fold) |

Above the four executions sits the choice of IR, by whether the verb
recurses:

| IR | used for | why |
|---|---|---|
| **relalg** (`Rel` + `normalize`, D-RA-1; proven rewrites) | the non-recursive verbs: `diff`, `status`, the four rebase masks, `land`'s up-to-date check | each is a fixed set expression over W, R, M and the givens; `normalize` simplifies it (absorption, range merging) before it lowers to one of the executions above |
| **Datalog** (semi-naive fixpoint; recursion in the external d′ driver, D-RA-4) | the stale-read re-run, any rule a thought derives with | recursion needs μ, which relalg does not have; Δ = new AND NOT old is the same step as row 7 of the wiring doc |
| **stratification** | `MergeQueue` order, squashes that read another thought's result negated | a negated read of a thought that has not reached its fixpoint is unsafe. FIFO order alone does not give strata (Codex review on #1599): a caller can enqueue B, which reads A negated, before A. So each thought records which thoughts it reads negated; `land` refuses B while A is unlanded, the queue orders by that dependency and keeps FIFO only among independent thoughts, and a cycle through negation is refused at `enqueue` (wiring doc row 8, N1) |

The relalg path waits on D-RA-2 (the absorption defect and where the set IR
lives, synergy map D1); the Datalog path on D-RA-4's step being rewritten
without the gather-bannered preimage.

Two consequences:

- **Dense vs sparse is the open measurement.** A thought usually writes few
  cells of a 64k table, so its W is sparse while M after a big squash may be
  dense. The crossover between mask AND and leapfrog seek is exactly the
  OPEN rows P3/P11 of the PG-Ring catalogue (#1600); the rebase verb is that
  probe's first real consumer.
- **No hash lookups.** Positions inside a thought's sparse storage are the
  rank of the address in the thought's own presence plane (popcount of the
  bits before it). `AlphaOverlay` keeps a `HashMap` from address to slot
  today; carrying it to thoughts replaces that with the rank.

## A sealed version is a PR entry (DECISION 2026-10-17)

BASIS: "If we use sealed version as PR entries it would make versions
cheaper and cleaner reference." Each `land` produces exactly one sealed
version, and that version *is* the record of the landed thought: what it
changed (its delta), which sources it carried (source bitsets), what the
revision gate decided, which conflicts were recorded, which version it was
rebased onto. `Main::log()` is then the PR history.

Why it is cheaper: a version holds only the squashed delta, so it costs the
cells the thought changed, not a rewrite of the table. The split-tunnel
pattern already asks for one batch = one Lance version from one writer
(`split-tunnel-sparse-append-resident-fold.md`, rule 2); a landed thought is
that batch.

Why it is a cleaner reference: a version number is never reused, while a
thought number is (slots are freed by squash). So:

- **blame** returns `(landing version, source bitset)`: which PR brought a
  fact in, and from what evidence;
- **revert** is a new version that undoes one PR entry;
- **checkout** reads main as of a PR entry;
- **merge-base** is a PR entry, so rebase compares against a named, fixed
  reference;
- a citation, a contradiction record or a NAL stamp can point at the
  version that landed a fact, a stable address, instead of at a slot.

Open: where the PR record lives (alongside the version as its metadata, or
as rows of a small log table keyed by version); what Lance 13 lets a commit
carry has to be checked before choosing.

## Rules

1. **Up to date before landing.** `land` refuses a thought whose
   `merge_base` is not main's current version; the caller rebases first.
2. **Rebase moves no data.** Four masks over `W`, `R` and `M` (main's change
   since `merge_base`): clean → fast-forward; same value → drop the cell;
   different value → contradiction record; `R ∧ M ∧ ¬W` → re-run the
   thought's fixpoint from that delta.
   A semi-naive re-run only adds facts, so a deletion or changed value in
   `M` would leave conclusions derived from the old value behind (Codex
   review on #1599). Before the re-run, every derived cell whose derivation
   read a cell in `R ∧ M` is retracted, then re-derived if it still follows
   (delete-and-rederive, DRed: Gupta, Mumick, Subrahmanian 1993). This
   needs, per derived cell, the cells it was derived from: the per-fact
   source bitset (D-HL-3) or a read-provenance plane, decided in D-TGA-3.
3. **Landing writes the next version.** A given is never overwritten and the
   givens table is read-only per sealed version; `land` produces version
   v+1 and the givens are rebuilt at its seal.
4. **Blame survives the squash.** Every squashed cell keeps its source
   bitset and the version that landed it; the thought number does not
   survive (slots are reused), the version number does.
5. **Strata survive.** A partial squash per stratum keeps the stratum axis;
   squashing strata straight into main is refused unless the caller declares
   the pooling (Simpson map).

## Work items (test-first; each lands with its falsifier first)

- [ ] **D-TGA-0** Pre-registration: fixtures (clean, agreement, conflict,
  stale read, partial squash, merge queue of three), the oracle (a thought
  re-run from scratch on the new version), and the disables, written before
  any code.
- [ ] **D-TGA-1** The alpha overlay carried to 64 thoughts (a thought axis
  beside the rung lanes, presence-rank lookup instead of `HashMap`); `branch` /
  `discard` / `write` / `diff`; limit of 63 branches (64 thoughts with
  main); givens refusal with a contradiction record. Falsifiers at both
  edges: with 62 branches open the 63rd `branch()` succeeds and gets the
  last free slot; with 63 open the 64th is refused; slot 0 is never handed
  out; a write to a given's lane with a new value records a contradiction
  and changes nothing.
- [ ] **D-TGA-2** Read plane `R` and `status`. Falsifier: a read outside
  `R` is never reported stale.
- [ ] **D-TGA-3** `rebase`. Falsifier: rebase-then-squash equals a
  from-scratch re-run on v+1, cell for cell, records included, on fixtures
  that include a deletion in main (the thought derived B from A, main
  deletes A: B must be gone after rebase). Disables: drop the stale-read
  re-run; skip the conflict check; skip the retraction step.
- [ ] **D-TGA-4** `land` through the revision gate into the next Lance
  version (D-HL-8). Falsifier: landing a thought that is behind is refused.
- [ ] **D-TGA-5** `squash_into` and `blame`. Falsifier: after a partial
  squash and slot reuse, `blame` still returns the original sources.
- [ ] **D-TGA-6** `MergeQueue`. Falsifiers: three independent thoughts
  landed through the queue equal the same three landed by hand in FIFO
  order, rebasing each; B enqueued before A, with B reading A negated,
  lands after A; a cycle through negation is refused at `enqueue`.
  Disable: order by FIFO only.

## Open

- Whether `R` is per cell or per aperture.
- Where the thought type lives: contract (types only) with execution in
  mask-risc / quack, as every other carrier.
- Timings: v4 only, ns per cell, tier printed by the program.
- The dense/sparse crossover that picks presence masks or leapfrog seek per
  operand (catalogue P3/P11); measured, not assumed.
