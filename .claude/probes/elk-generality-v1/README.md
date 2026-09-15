# PROBE-ELK-GENERALITY-1 — is the meet/horizon behaviour the LENS's, or the taxonomy's shape?

**Partly re-runnable, and the split is deliberate.** The MQ arm is synthetic and
seeded, so it runs anywhere forever. The MONDO arm reads `obo-core.soa`, a
31 MB binary bake that is **not committed to this repo** (size, not policy —
the path is configurable); that arm skips with a notice when the file is absent.

```sh
python3 .claude/probes/elk-generality-v1/generality.py
# optional: OBO_CORE_SOA=/path/to/obo-core.soa python3 …
```

## Why a control was not optional

Every number this session reported for the `ogar-elk` lens came from **one
graph** — the MONDO `is_a` spine. A taxonomy is precisely the graph shape most
likely to produce those numbers trivially, so "the lens behaves like X" was never
separable from "a taxonomy behaves like X" without a second graph.

**MQ** is a Mississippi-Queen river course: positions flowing downstream, the
channel braiding and rejoining, bounded legal moves (advance 1–3, shift one
lane). It is a DAG like a taxonomy — but its "ancestors" are *upstream positions
you could have come from*, not generalisations, and the meet of two positions is
the **confluence**, not a common genus. Same algebra, different semantics,
different degree profile. Both graphs run through the **same arm functions**;
that shared code is what makes the comparison mean anything.

## Pre-registered reading (written before the MONDO arm ran in this file)

> If ASCENDING agrees across both, the ranking result is the **lens's**.
> If it diverges, it was the **taxonomy's shape** all along.

`A4` and `A5` are the two confounds that could fake a divergence. A gap that
survives both is structural.

## Result

| arm | MQ river | MONDO `is_a` | Δ |
|---|---|---|---|
| A0 mean \|ancestors\| | 69.4 / 151 nodes | 14.1 / 60,467 nodes | ~2000× in relative coverage |
| A1 mean \|intersection\| | 46.94 (0 empty) | 4.37 (5 empty) | — |
| A3 depth **descending** | 5.1 % | 6.7 % | 1.6 pp |
| A3 depth **ascending** | **36.0 %** | **84.6 %** | **48.6 pp** |
| A4 non-vacuous (\|I\|>1) | 36.0 % | 84.6 % | 48.6 pp |
| A5 braided pairs only | 36.0 % | 83.5 % | 47.5 pp |

**1. The depth-rank ≈ `most_specific` equivalence is NOT a property of the lens.**
Ascending-depth argmin reproduces `LensClosure::most_specific` on 84.6 % of MONDO
pairs and only 36.0 % of MQ pairs. The claim is correct **scoped to a taxonomy**
and must never be stated as lens behaviour.

**2. A4 — the vacuity worry was unfounded, and now measured rather than assumed.**
At `|I| == 1` the argmin *is* the only `most_specific` member, so agreement there
would be arithmetic. **Neither graph has a single such pair** (0 dropped, both
sides), so the headline is not carried by arithmetic. The stratification also
kills the obvious confound in the opposite direction: MONDO's agreement **rises**
with intersection size (75.9 → 90.5 → 100 %) while MQ's **falls** (100 → 42.9 →
31.8 %). If "small intersections make it easy" were the explanation, both graphs
would trend the same way. They trend opposite. MONDO's small-`|I|` majority is in
fact its *worst* bucket, so the headline **understates** the taxonomy effect.

**3. A5 — the proposed mechanism is FALSIFIED, and that is the result.**
⊘ **HISTORICAL as published; two later results narrow it.** (a) The density sweep
(`.claude/probes/density-sweep-v1/`) falsified DENSITY as the cause **within MQ** and named
reachable path-length SPAN as the current candidate — so the binary braided-only verdict below
is underpowered rather than simply right. (b) A review on `43dbfde` found `supers_minmax`
propagated only a node's FIRST discovery, understating spread by up to 21× (MQ mean 0.73 → 15.55,
MONDO 0.44 → 1.35). The A5 flat/braided SPLIT barely moved (Δ 47.5 → 47.8 pp) because it reads
the sign of spread, not its size — but any statement about spread MAGNITUDES published before
that fix is void. **Cross-family transfer remains unmeasured.**
`supers_of` keeps the MINIMUM depth, so the natural hypothesis is that depth stops
tracking specificity once a DAG **braids** (several path lengths to the same
ancestor, letting a general ancestor score shallow via a shortcut). Braiding does
have a real directional effect *inside* MONDO — flat pairs 13/13 = 100 %, braided
152/182 = 83.5 % — but it cannot carry a 48.6 pp gap: comparing braided pairs
only, the gap is **47.5 pp**, essentially unchanged. Controlling for the mechanism
removes almost none of the divergence.

**4. The horizon behaves differently too, in the direction that matters for a
cheap gate.** MONDO saturates by `w = 14` (CUT 0, recall 100 %) and is already at
84.5 % recall by `w = 7`. MQ is still cutting 21/197 pairs at `w = 14` and reaches
only 24.2 % recall at `w = 7`. A window tuned on a taxonomy is not transferable.

## What is NOT established

The **mechanism is unnamed.** The remaining unisolated variable is ancestry
density — MQ's typical node has 69.4 ancestors out of 151 nodes (46 % of the
graph) against MONDO's 14.1 of 60,467 (0.02 %). MQ was built as a *different*
graph, not as a density sweep, so it demonstrates non-universality without
identifying the responsible property. A density-swept family of MQ variants is the
measurement that would name it; it has not been run.

MQ is also one synthetic graph with one seed. It is a sufficient falsifier for a
universal claim (one counterexample suffices) and is **not** evidence about any
other real ontology.
