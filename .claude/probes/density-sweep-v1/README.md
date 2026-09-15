# PROBE-DENSITY-SWEEP-1 — the density hypothesis is FALSIFIED; the variable is path-length SPREAD

**Re-runnable.** The MQ arms are synthetic and seeded and run anywhere. The MONDO arm reads a
31 MB binary bake not committed to this repo (size, not policy — path via `OBO_CORE_SOA`); it
skips with a notice when absent.

```sh
python3 .claude/probes/density-sweep-v1/sweep.py
```

## What it was built to answer

`ISS-ELK-DENSITY-UNISOLATED` filed **ancestry density** as the named-but-unisolated variable
behind the 48.6 pp MQ↔MONDO divergence in depth-rank-vs-`most_specific`. One axis, three
questions, because they share an independent variable:

- **Q1** — does agreement track density monotonically? If so, density is the property and the
  claim becomes a checkable precondition: *"depth-rank works while ancestry stays sparse."*
- **Q2** — the torch's **anti-vacuity** arm. If bounded-k recall is high *everywhere*, the result
  is about these graphs; it must degrade somewhere or it says nothing.
- **Q3** — the **amortization** band. Masking pays for longer reach only if bounded recall holds
  **as distance grows** — hence hops 4 / 8 / 12, not 12 alone.

**Deliberately NOT measured: "top-k successor mass."** That is the D-HXP-1 SIGNAL half, already
**STRUCK as unanswerable by that instrument** — `uniform_expected = min(6,d)/d` is biased by
small-sample concentration, and on an unweighted graph every successor carries equal mass, so
the statistic is arithmetic. Bounded-k **recall** has no such bias: truncating a wide frontier
either costs reachability or it does not.

## Q1 — density is FALSIFIED, conclusively

The first sweep varied width, advances and shifts together — and its own ordering already
refuses a density story (4× density change, same answer; 2× density change, same answer). But
that sweep is **confounded**: width covaried with advances (2,3 / 4,5 / 6,6), so reading any
mechanism off it would repeat the exact error the density hypothesis was dying of.

The **controlled arm** pins width and shifts and moves advances alone:

| advances | \|adv\| | density | ASC |
|---|---|---|---|
| `(1,)` | 1 | 45.05 % | **100.0 %** |
| `(2,)` | 1 | **21.90 %** | **100.0 %** |
| `(1,2)` | 2 | 45.05 % | 55.3 % |
| `(1,3)` | 2 | 45.05 % | **37.1 %** |
| `(1,2,3)` | 3 | 45.05 % | 36.0 % |
| `(1,2,3,4)` | 4 | 45.05 % | **26.9 %** |

- **Density pinned at exactly 45.05 %, agreement spans 26.9 % → 100.0 %.** A 73-point swing at
  identical density. Density cannot be the cause.
- **`(1,)` vs `(2,)`** — multiplicity 1 in both, density differs ~2×, agreement **identical**.
  Density is inert at fixed multiplicity.
- **`(1,2)` vs `(1,3)`** — same count, same density, **18 points apart**. So it is the **SPAN**
  of path lengths, not their number.

## This RE-OPENS A5, which I closed as falsified

`E-DEPTH-RANK-REPRODUCES-MOST-SPECIFIC-BUT-ONLY-ON-A-TAXONOMY-1` reports A5 as *"the braid
mechanism I proposed is FALSIFIED as the explanation"* — braided-only pairs still left 47.5 pp.
That arm split pairs on a **binary** indicator (`spread == 0` vs `> 0`) **within** each graph.
This design varies braiding as a **dose** and gets a clean monotone.

**The binary indicator was underpowered, not wrong.** It lumps spread 0.1 with spread 3, and the
two graphs' means (MONDO 0.44, MQ 0.73) are close on that scale while being structurally
different regimes — every MQ node reachable at 3 path lengths vs MONDO's incidental braiding. A
coarse indicator with almost no variance in the treatment cannot see a dose-response.

## What is NOT established

**MONDO does not sit on this axis.** It braids (mean spread 0.44, 182/195 pairs) and scores
**84.6 %** — *higher* than MQ `(1,2)` at 55.3 %. The dose-response is **within the MQ family**;
cross-family transfer is **unmeasured**, and the sweep gives no licence to predict a real
ontology's agreement from its spread.

**Q2 anti-vacuity is NOT satisfied.** `k=50` returns **100 % on every MQ config and 99.4 % on
MONDO**. A bound that never costs anything on any graph tested is the fires-on-everything shape.
`k=6` discriminates only mildly (89–100 %). **No graph has been found where bounded-k genuinely
fails**, so the recall result is about *these* graphs until it degrades somewhere. Finding that
graph is the outstanding work, not a footnote.

**A number needing re-derivation before it is cited again.** This run reports MONDO peak frontier
@12 = **142** over 60 sampled seeds; an earlier run reported the `is_a` frontier profile peaking
at **6,297** at hop 6. Different seed samples. Both cannot stand as "the" peak.

## Q3 — amortization: SUPPORTED

Bounded-k recall is **flat across distance** in every config — it does not decay as reach grows:

| graph | k | h4 | h8 | h12 |
|---|---|---|---|---|
| MONDO `is_a` | 6 | 92.1 % | 89.5 % | **89.1 %** |
| MONDO `is_a` | 50 | 99.5 % | 99.4 % | **99.4 %** |
| MQ `(1,2,3,4)` | 6 | 94.0 % | 95.2 % | **95.0 %** |

Read with Q2's caveat: flat-and-high is consistent with amortization **and** with a bound that is
simply never binding. The two are separated only by finding a graph where truncation costs.
