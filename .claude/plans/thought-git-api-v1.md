# Thought git API (D-TGA) — v1

**Status:** PROPOSAL, 2026-10-17. Test-first; nothing built.
**Asked for:** "What would be genuinely valuable would be a git like API."
**Model (decided 2026-10-17, `.claude/knowledge/thinking-tables-wiring.md`
row 11):** a thought table is compute and storage. Thoughts 1..63 branch off
main (thought 0, the givens of the current sealed version); a finished thought
is squash-committed into the next sealed version, freeing its slot; a
selection of thoughts can be squashed into a kept intermediate thought; the
rest of git applies.

## Overview

One small surface over pieces that exist, so a caller reasons about thoughts
the way they reason about branches, and every verb lowers to mask algebra and
Lance versions. No new algebra, no interpreter, no strings as addresses:
a thought is its number (`0..64`), a version is a Lance version, a cell is a
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
| givens, contradiction records | D-HL givens layer | `.claude/plans/ce64-hypothesis-layers-v1.md:696-725` |
| written-by word, coverage orientations | D-HL arm M, R/C | same, `:596-660` |
| source bitset (blame) | D-HL-3 evidential base | same, `:73-90` |
| review before merge | revision gate G1–G7 | `.claude/board/entries/2026-10-14-revision-layer-invariants.md` |

**Caveat, load-bearing:** the alpha overlay records *attention* and its module
doc forbids it from holding a measured value ("a changed reference value is a
new reference version from a new bake, never an overlay write"). Thought
tables hold *values*. So D-TGA reuses the alpha mechanics (shared
allocation, one writer, ordered merge, consume-to-discard) but does not reuse
`AlphaOverlay` or `AlphaMask` as the thought type; a thought gets its own
type over the same word layout, as the O2 note in the wiring doc already
requires.

## The surface (methods on carriers)

```text
Main            (sealed version v, givens table thought 0)
  .branch()                  -> Result<Thought, BranchesFull>   // takes a free slot 1..63
  .version()                 -> LanceVersion
  .checkout(v)               -> Main                            // read-only, Lance version read
  .log()                     -> versions with the squash that made each
  .blame(addr, lane)         -> SourceBits                      // D-HL-3

Thought         (slot 1..63, merge_base = the version it branched from)
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

Two consequences:

- **Dense vs sparse is the open measurement.** A thought usually writes few
  cells of a 64k table, so its W is sparse while M after a big squash may be
  dense. The crossover between mask AND and leapfrog seek is exactly the
  OPEN rows P3/P11 of the PG-Ring catalogue (#1600); the rebase verb is that
  probe's first real consumer.
- **No hash lookups.** Positions inside a thought's sparse storage are the
  rank of the address in the thought's own presence plane (popcount of the
  bits before it). `AlphaOverlay` keeps a `HashMap` from address to slot;
  the thought type does not copy that.

## Rules

1. **Up to date before landing.** `land` refuses a thought whose
   `merge_base` is not main's current version; the caller rebases first.
2. **Rebase moves no data.** Four masks over `W`, `R` and `M` (main's change
   since `merge_base`): clean → fast-forward; same value → drop the cell;
   different value → contradiction record; `R ∧ M ∧ ¬W` → re-run the
   thought's fixpoint from that delta.
3. **Landing writes the next version.** A given is never overwritten and the
   givens table is read-only per sealed version; `land` produces version
   v+1 and the givens are rebuilt at its seal.
4. **Blame survives the squash.** Every squashed cell keeps its source
   bitset; the thought number does not survive (slots are reused).
5. **Strata survive.** A partial squash per stratum keeps the stratum axis;
   squashing strata straight into main is refused unless the caller declares
   the pooling (Simpson map).

## Work items (test-first; each lands with its falsifier first)

- [ ] **D-TGA-0** Pre-registration: fixtures (clean, agreement, conflict,
  stale read, partial squash, merge queue of three), the oracle (a thought
  re-run from scratch on the new version), and the disables, written before
  any code.
- [ ] **D-TGA-1** Thought type over the D-HL word layout; `branch` /
  `discard` / `write` / `diff`; 64-slot limit; givens refusal with a
  contradiction record. Falsifier: the 65th branch is refused; a write to a
  given's lane with a new value records a contradiction and changes nothing.
- [ ] **D-TGA-2** Read plane `R` and `status`. Falsifier: a read outside
  `R` is never reported stale.
- [ ] **D-TGA-3** `rebase`. Falsifier: rebase-then-squash equals a
  from-scratch re-run on v+1, cell for cell, records included. Disables: drop
  the stale-read re-run; skip the conflict check.
- [ ] **D-TGA-4** `land` through the revision gate into the next Lance
  version (D-HL-8). Falsifier: landing a thought that is behind is refused.
- [ ] **D-TGA-5** `squash_into` and `blame`. Falsifier: after a partial
  squash and slot reuse, `blame` still returns the original sources.
- [ ] **D-TGA-6** `MergeQueue`. Falsifier: three thoughts landed through the
  queue equal the same three landed by hand in FIFO order, rebasing each.

## Open

- Whether `R` is per cell or per aperture.
- Where the thought type lives: contract (types only) with execution in
  mask-risc / quack, as every other carrier.
- Timings: v4 only, ns per cell, tier printed by the program.
- The dense/sparse crossover that picks presence masks or leapfrog seek per
  operand (catalogue P3/P11); measured, not assumed.
