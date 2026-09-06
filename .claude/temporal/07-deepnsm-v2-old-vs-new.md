# 07 — deepnsm v1 vs v2, audited brutally

Requested as a brutal old-vs-new. It corrects two claims this session made
earlier on the strength of doc comments rather than measurement.

## ⊘ TWO CORRECTIONS TO THIS SESSION'S OWN EARLIER REPORTING

**1. The "63.3% of same-subject links reach beyond ±5" figure is CLAIMED,
UNVERIFIED.** `bible_wave.rs:237` `println!`s it; there is **no assertion**, and
the example cannot run in-tree (it needs an uncommitted `pg10.txt` plus release
assets). I cited it as the measurement that retired the fixed ±5 window. It is
a printed number, not a gate.

**2. deepnsm-v2 does NOT enforce the no-hindsight property.** Hindsight
blocking IS a `TemporalPov::at` property (range `[0, ref+1)`,
`temporal_pov.rs:177-182`), but **`window_range` bypasses that automatic bound**: it is constrained ONLY by
the caller's explicit `VersionRange`, never by the reader's own position. So the
path is range-constrained rather than unconstrained — and `bible_wave.rs:249`
passes `VersionRange::new(0, verses.len())`, a range that can admit versions
after an earlier reference version. The no-hindsight property is therefore not
ENFORCED for this consumer; it is enforced only by the other one. So the structural
no-hindsight gate is a **stockfish discipline**, not a shared property of both
consumers. My earlier framing implied both enforced it. Only one does.

Related: **rung is INERT in `TemporalPov`** — `temporal_pov.rs:194`, `admits`
ignores it. The version-range half is real; the rung half is decoration at that
layer.

## Where v2 genuinely wins — two axes, both thin

| axis | evidence | caveat |
|---|---|---|
| CAM-96 meaning space | ρ 0.828 / 0.774 | cited to `probes/README.md`; `data/` is EMPTY, so nothing reproduces in-tree |
| vocabulary capacity 4096 → 65536 | `vocab.rs:26-59` | it is a CONSTANT. No coverage measurement accompanies it |

## What `TemporalStream` actually is

`Vec<(u64, Spo)>` plus two filter closures (`lib.rs:196-259`). And a "version"
in this crate is **the verse index** (`bible_wave.rs:141`) — **never a Lance
version.** That matters for the whole temporal thread: deepnsm-v2 borrows the
version-range TYPE without the version-chain SUBSTRATE.

The ±5 regression test is weak: `lib.rs:305`
`assert_eq!(window_at(4).count(), 5)` would still pass under a hard ±5 ring.
The genuinely non-vacuous test is `lib.rs:326` — an independent recount sweep
asserting empty / partial / full coverage.

## The strawman

v1's ±5 was `default_window()` (`context.rs:34,44`), **not a constraint**. The
v2 framing presents it as a hard ceiling that had to be escaped. The real v1
defect is different and under-stated: the window **overwrites** (it is a ring).
So the doc over-states the width problem and under-states the actual one.

## Delta: a THIRD instance of materialize-both-and-compare

`introspect.rs:115-133` runs **two full prefix scans and subtracts**, never
using its own `window_range`. It borrows rather than allocating, but the shape
is the same `O(stream) × 2` as surrealdb's `Timeline` and `VersionedGraph::diff`.

That makes **four** known instances of the pattern in this workspace:

| layer | mechanism |
|---|---|
| surrealdb `Timeline` (lance 7) | `view_at(v).scan()` per version, compare |
| `VersionedGraph::diff` | `read_all_batches` both versions, HashSet-diff |
| alpha overlay | a 512-byte row per touched address |
| **deepnsm-v2 `introspect`** | two full prefix scans, subtract |

## No masks in v2 — and v1 HAD them

**v2 has no mask, no bitset, nothing `&[u64]`-shaped.**

v1 did: `encoder.rs:101` `as_words() -> &[u64; VSA_WORDS]`, plus `popcount`,
`hamming`, and XOR `bind`. **Deleted in v2.**

So the mask-synergy question ("does masking have synergies for cached ternlog
stacked temporal awareness") gets a blunt answer on this crate: v2 removed the
one word-slice surface that would have plugged straight into
`mask_ternlog_assign`.

## What v2 LOST — the list that is missing from its docs

- the tokenizer (`bible_wave` reads **v1's** `word_frequency/` by relative path)
- VSA `bind` / `bundle` / `unbind`
- five PoS categories including **Negation** — so *"did not bite"* encodes
  identically to *"bite"*
- pronouns and `SentenceWindow` coreference
- parse-coverage and ticket escalation
- modifiers
- `process_sentence(&str)`
- `named_entities`
- **`nsm_primes`** — the thing the crate is named after

## Dormancy, split

| surface | consumers |
|---|---|
| vocab / fsm / codebook / `Nsm` | wired into `tesseract-paperless::consistency` — **production** |
| `TemporalStream`, `WitnessStream` | **ZERO external consumers**; `wave.rs:80-83` admits it in its own docs |

And **v1 is still live** in `tesseract-ogar`. v2 replaced nothing downstream;
both are in use, for different things.

## One internal inconsistency worth fixing

`wave.rs:133-153` `.collect()`s a full `Vec` per window and rebuilds it on every
`ground_at` / `resolve_at`. The zero-copy discipline that `lib.rs:224-231`
argues for at length is not applied there.

## Consequence for the temporal work

deepnsm-v2 is a weaker witness for `TemporalPov` than it looked. It uses the
type over a verse index rather than a version chain, bypasses the admission
gate on its main path, and its headline number is unasserted. **stockfish-rs
remains the one place where the structural no-hindsight property is actually
enforced and asserted.** Treat `TemporalPov`'s "two independent real corpora"
credential as one-and-a-half.
