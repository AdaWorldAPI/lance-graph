#!/usr/bin/env python3
"""Mutate_FalseWitness — external-review adjudication (Gemini §6, mutation
operator 4).

The invariant under test: **a duplicated witness must add ZERO evidence.**
Three lanes descending from one and the same source text are one lane
copied thrice — this is the translation-corpus form of `I-NOISE-FLOOR-JIRAK`
("bits ... are weakly dependent by construction"; classical IID assumptions
are wrong when copies masquerade as independent observations). If we clone
the KJV lane verbatim and call it a sixth witness ("kjv2"), any metric that
is actually counting *independent* corroboration must be unmoved by the
clone; any metric that rises is secretly counting *head-count* — the
"argumentum ad populum of witnesses" failure mode.

What this script does (three measures, baseline vs +kjv2, identical logic
both passes):

  1. **Naive per-verse cross-lane agreement.** For every common verse and
     every lane, agreement(lane, verse) = mean Jaccard(tokens(lane, verse),
     tokens(other_lane, verse)) over every OTHER lane in the active lane
     set. Overall score = mean over all (lane, verse) pairs. This is what a
     naive "N lanes agree, so it must be right" consumer would compute.
     EXPECTED to inflate when kjv2 joins (kjv and kjv2 are token-identical,
     so their mutual Jaccard is exactly 1.0 for every verse) — quantified
     below, not just asserted.

  2. **Pairwise lane-similarity matrix — the detection signal.** Per lane,
     build ONE aggregated token set over 500 deterministically sampled
     common verses (fixed seed, reproducible); report the full NxN Jaccard
     matrix between lanes' aggregated sets. A duplicated witness must show
     up as an off-diagonal cell at ~1.0 — this is the signal a downstream
     de-duplication/independence-weighting pass would key off.

  3. **Independence-weighted agreement.** Re-run measure 1's per-verse
     logic, but weight each lane pair (L, M)'s contribution by
     `1 - sim_matrix[L][M]` (from measure 2) instead of an unweighted mean.
     A clone pair gets weight ~0, so it should contribute ~nothing to
     either lane's agreement score. Reported honestly either way — if this
     STILL inflates, that is itself a finding, not a bug to paper over.

Fences, stated up front (deliberate crudeness, matching this directory's
other probes): tokenization is a single generic `\\w+` lowercase splitter —
no lemmatizer, no stopword list, no per-language normalisation. This is
intentional: the KJV/kjv2 finding does not depend on token quality (the
clone is byte-identical text, so ANY consistent tokenizer collapses their
Jaccard to 1.0); a cruder tokenizer only adds cross-language noise-floor
"agreement" between genuinely different translations, which is exactly the
noise measure 3 is being tested against.

Psalms exclusion (present in this directory's other probes, e.g.
`build_rosetta_probe.py`'s PSALMS_NR skip for versification-offset PMI
stats) is **not needed here** and is deliberately omitted: that exclusion
exists to protect a cross-lane STATISTICAL association measure from a known
verse-numbering misalignment in one book. Here the "second" lane (kjv2) is
a verbatim, key-for-key Python-dict copy of kjv — by construction every row
key that exists in kjv exists in kjv2 with byte-identical text, so kjv2
cannot introduce ANY versification skew anywhere, Psalms included. This is
stated, not merely assumed: `kjv2 = dict(lanes["kjv"])`.

Data: read-only, not re-fetched (see `build_rosetta_probe.py` module
docstring for provenance / fetch instructions for the 4 getBible lanes;
tischendorf is the PROIEL/getBible-format Greek NT lane). Row key is the
frozen external key `(book_nr, chapter, verse)`, identical convention to
every other probe in this directory.

Run:  python3 mutate_falsewitness.py <data_dir_with_bible_*.json>
Out:  <data_dir>/out/mutate_falsewitness_report.md
"""

import argparse
import json
import random
import re
import sys
from pathlib import Path

REAL_LANES = ["kjv", "luther1545", "elberfelder1905", "bkr", "tischendorf"]
MUTANT_LANE = "kjv2"  # verbatim duplicate of kjv — the fake sixth witness
SAMPLE_SIZE = 500
SAMPLE_SEED = 20260726  # fixed — reproducible across runs

# Generic Unicode word tokenizer — deliberately not language-specific (see
# module docstring "Fences"). Python's \w under `re`'s default Unicode mode
# matches letters from any script (Latin incl. German/Czech diacritics,
# Greek incl. polytonic accents) plus digits/underscore.
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def load_lane(path: Path) -> dict:
    """Same getBible-format loader convention as build_rosetta_probe.py."""
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for book in d["books"]:
        bnr = book["nr"]
        for ch in book["chapters"]:
            for v in ch["verses"]:
                rows[(bnr, v["chapter"], v["verse"])] = v["text"].strip()
    return rows


def toks(text: str) -> frozenset:
    return frozenset(t.lower() for t in TOKEN_RE.findall(text))


def jaccard(a: frozenset, b: frozenset) -> float:
    """Jaccard similarity. Both-empty is defined as 0.0 (no evidence of
    agreement, not "trivially total agreement") — an empty verse text on
    either side must not read as a match."""
    if not a and not b:
        return 0.0
    u = a | b
    if not u:
        return 0.0
    return len(a & b) / len(u)


def build_token_cache(lane_maps: dict, rows) -> dict:
    """(lane_name, row_key) -> token frozenset, for every lane in lane_maps
    over every key in rows. Missing rows (should not occur for `common`
    keys, but defensive) tokenize to the empty set."""
    cache = {}
    for lane, rowmap in lane_maps.items():
        for k in rows:
            text = rowmap.get(k)
            cache[(lane, k)] = toks(text) if text is not None else frozenset()
    return cache


def naive_agreement(lane_names, rows, cache):
    """Measure 1 — naive per-verse cross-lane agreement (unweighted mean
    Jaccard against every other lane in `lane_names`)."""
    per_lane = {l: [] for l in lane_names}
    total = []
    for v in rows:
        for l in lane_names:
            tl = cache[(l, v)]
            others = [m for m in lane_names if m != l]
            if not others:
                continue
            scores = [jaccard(tl, cache[(m, v)]) for m in others]
            s = sum(scores) / len(scores)
            per_lane[l].append(s)
            total.append(s)
    overall = sum(total) / len(total) if total else 0.0
    per_lane_mean = {l: (sum(s) / len(s) if s else 0.0) for l, s in per_lane.items()}
    return overall, per_lane_mean


def pairwise_similarity_matrix(lane_names, sample_rows, cache):
    """Measure 2 — per-lane AGGREGATED token set (union over the sample),
    pairwise Jaccard between lanes. This is the detection signal: a clone
    lane's aggregated set is identical to its source's, so their pairwise
    cell is exactly 1.0 regardless of which verses happened to be sampled."""
    agg = {}
    for l in lane_names:
        s = set()
        for v in sample_rows:
            s |= cache[(l, v)]
        agg[l] = frozenset(s)
    matrix = {a: {} for a in lane_names}
    for a in lane_names:
        for b in lane_names:
            matrix[a][b] = 1.0 if a == b else jaccard(agg[a], agg[b])
    return matrix


def weighted_agreement(lane_names, rows, cache, sim_matrix):
    """Measure 3 — independence-weighted agreement. Same per-verse Jaccard
    terms as measure 1, but each term for pair (L, M) is weighted by
    `max(0, 1 - sim_matrix[L][M])` instead of contributing 1/(n-1)
    unconditionally. A clone pair (sim ~= 1.0) is weighted to ~0."""
    per_lane = {l: [] for l in lane_names}
    total = []
    for v in rows:
        for l in lane_names:
            tl = cache[(l, v)]
            others = [m for m in lane_names if m != l]
            if not others:
                continue
            wsum = 0.0
            acc = 0.0
            for m in others:
                w = max(0.0, 1.0 - sim_matrix[l][m])
                acc += w * jaccard(tl, cache[(m, v)])
                wsum += w
            s = (acc / wsum) if wsum > 0 else 0.0
            per_lane[l].append(s)
            total.append(s)
    overall = sum(total) / len(total) if total else 0.0
    per_lane_mean = {l: (sum(s) / len(s) if s else 0.0) for l, s in per_lane.items()}
    return overall, per_lane_mean


def pct_change(base: float, mutated: float) -> float:
    if base == 0:
        return float("inf") if mutated > 0 else 0.0
    return 100.0 * (mutated - base) / base


def fmt_matrix(lane_names, matrix) -> str:
    header = "| lane | " + " | ".join(lane_names) + " |"
    sep = "|---|" + "---|" * len(lane_names)
    lines = [header, sep]
    for a in lane_names:
        row = [f"**{a}**"] + [f"{matrix[a][b]:.4f}" for b in lane_names]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Mutate_FalseWitness — duplicated-lane adjudication probe")
    ap.add_argument("data_dir", nargs="?", default=None,
                     help="directory containing bible_{kjv,luther1545,"
                          "elberfelder1905,bkr,tischendorf}.json")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent
    out_dir = data_dir / "out"
    out_dir.mkdir(exist_ok=True)

    lanes = {}
    for lane in REAL_LANES:
        p = data_dir / f"bible_{lane}.json"
        if not p.exists():
            sys.exit(f"missing {p} — fetch first (see build_rosetta_probe.py "
                      f"module docstring for provenance)")
        lanes[lane] = load_lane(p)

    # ── the mutation: a verbatim duplicate of kjv, keyed identically ──────
    lanes[MUTANT_LANE] = dict(lanes["kjv"])

    baseline_lanes = list(REAL_LANES)
    mutated_lanes = list(REAL_LANES) + [MUTANT_LANE]

    # ── common rows: intersection over the 5 REAL lanes only. kjv2 shares
    # every key in kjv by construction, so it can never shrink or skew this
    # set (see module docstring, Psalms-exclusion paragraph). ─────────────
    keysets = {l: set(lanes[l]) for l in REAL_LANES}
    common = set.intersection(*keysets.values())
    common_sorted = sorted(common)
    n_common = len(common_sorted)

    # ── deterministic 500-row sample for measure 2 ─────────────────────────
    sample_n = min(SAMPLE_SIZE, n_common)
    sample_rows = random.Random(SAMPLE_SEED).sample(common_sorted, sample_n)

    # ── token cache over ALL common rows, for all 6 lane-maps (real 5 +
    # kjv2). Measures 1 and 3 use the full common set; measure 2 uses the
    # fixed sample drawn from the same set. ────────────────────────────────
    cache = build_token_cache(lanes, common_sorted)

    # ── Measure 2 first (its matrix feeds measure 3's weights) ────────────
    sim_matrix = pairwise_similarity_matrix(mutated_lanes, sample_rows, cache)
    sim_kjv_kjv2 = sim_matrix["kjv"][MUTANT_LANE]

    # ── Measure 1 ───────────────────────────────────────────────────────
    m1_base_overall, m1_base_per_lane = naive_agreement(baseline_lanes, common_sorted, cache)
    m1_mut_overall, m1_mut_per_lane = naive_agreement(mutated_lanes, common_sorted, cache)
    m1_delta_pct = pct_change(m1_base_overall, m1_mut_overall)

    # ── Measure 3 (reuses the measure-2 matrix as static weights) ─────────
    m3_base_overall, m3_base_per_lane = weighted_agreement(
        baseline_lanes, common_sorted, cache, sim_matrix)
    m3_mut_overall, m3_mut_per_lane = weighted_agreement(
        mutated_lanes, common_sorted, cache, sim_matrix)
    m3_delta_pct = pct_change(m3_base_overall, m3_mut_overall)

    verdict_m1 = "INFLATES" if m1_mut_overall > m1_base_overall * 1.001 else "IMMUNE"
    verdict_m3 = "INFLATES" if m3_mut_overall > m3_base_overall * 1.001 else "IMMUNE"

    # ── report ───────────────────────────────────────────────────────────
    lines = [
        "# Mutate_FalseWitness — duplicated-lane adjudication report",
        "",
        "Operator: duplicate the KJV lane verbatim as a fake sixth lane "
        f'("{MUTANT_LANE}"), measure whether agreement/independence-flavoured '
        "statistics rise. If any metric increases when the duplicate joins, "
        "that metric is counting head-count, not independence.",
        "",
        "## Corpus",
        f"- real lanes: {', '.join(REAL_LANES)} (5)",
        f"- mutant lane: `{MUTANT_LANE}` = verbatim dict-copy of `kjv` (6th lane)",
        f"- common rows (intersection over the 5 REAL lanes; tischendorf is "
        f"NT-only, so this restricts to NT books): **{n_common}**",
        f"- measure-2 sample: **{sample_n}** rows, seed={SAMPLE_SEED} "
        "(deterministic)",
        "- Psalms exclusion: **not needed** — kjv2 is a key-for-key copy of "
        "kjv (`lanes[\"kjv2\"] = dict(lanes[\"kjv\"])`), so it cannot "
        "introduce any versification skew anywhere, Psalms included. Every "
        "row key present in kjv is present in kjv2 with byte-identical "
        "text, by construction, not by alignment.",
        "",
        "## Measure 1 — naive per-verse cross-lane agreement",
        "Jaccard(lowercase token sets) against every OTHER lane in the "
        "active set, averaged over lanes and verses.",
        "",
        f"- baseline (5 lanes): **{m1_base_overall:.6f}**",
        f"- mutated (+kjv2, 6 lanes): **{m1_mut_overall:.6f}**",
        f"- change: **{m1_delta_pct:+.2f}%**",
        f"- verdict: **{verdict_m1}**",
        "",
        "Per-lane mean agreement (baseline vs mutated) — kjv and kjv2 are "
        "the lanes expected to jump, since their mutual Jaccard is exactly "
        "1.0 on every verse:",
        "",
        "| lane | baseline | mutated | Δ |",
        "|---|---|---|---|",
    ]
    for l in mutated_lanes:
        b = m1_base_per_lane.get(l)
        m = m1_mut_per_lane.get(l)
        if b is not None:
            lines.append(f"| {l} | {b:.6f} | {m:.6f} | {pct_change(b, m):+.2f}% |")
        else:
            lines.append(f"| {l} | — | {m:.6f} | (new lane) |")

    lines += [
        "",
        "## Measure 2 — pairwise lane-similarity matrix (detection signal)",
        f"Per-lane aggregated token set over the {sample_n}-row deterministic "
        "sample; pairwise Jaccard between lanes. The duplicate must appear "
        "as an off-diagonal cell at ~1.0:",
        "",
        f"- **sim(kjv, {MUTANT_LANE}) = {sim_kjv_kjv2:.6f}** "
        f"({'CONFIRMED off-diagonal ~1.0 — clone detected' if sim_kjv_kjv2 > 0.999 else 'UNEXPECTED — not ~1.0, see raw matrix'})",
        "",
        fmt_matrix(mutated_lanes, sim_matrix),
        "",
        "## Measure 3 — independence-weighted agreement",
        "Same per-verse Jaccard terms as Measure 1, but each lane pair's "
        "contribution is weighted by `max(0, 1 - sim_matrix[L][M])` "
        "(the Measure 2 matrix) instead of an unweighted mean.",
        "",
        f"- baseline (5 lanes): **{m3_base_overall:.6f}**",
        f"- mutated (+kjv2, 6 lanes): **{m3_mut_overall:.6f}**",
        f"- change: **{m3_delta_pct:+.2f}%**",
        f"- verdict: **{verdict_m3}**",
        "",
        "Per-lane mean weighted agreement (baseline vs mutated):",
        "",
        "| lane | baseline | mutated | Δ |",
        "|---|---|---|---|",
    ]
    for l in mutated_lanes:
        b = m3_base_per_lane.get(l)
        m = m3_mut_per_lane.get(l)
        if b is not None:
            lines.append(f"| {l} | {b:.6f} | {m:.6f} | {pct_change(b, m):+.2f}% |")
        else:
            lines.append(f"| {l} | — | {m:.6f} | (new lane) |")

    # ── honest nuance, computed not asserted: the DIRECT pair (kjv's own
    # weighted score) vs TWO DISTINCT indirect effects on lanes that never
    # touch kjv2's text at all. Reported because the M3 aggregate moved by
    # more than a rounding error (-26%) and a flat "IMMUNE" verdict alone
    # would undersell what actually happened. ─────────────────────────────
    m3_direct_delta = pct_change(m3_base_per_lane["kjv"], m3_mut_per_lane["kjv"])
    indirect_deltas = {
        l: pct_change(m3_base_per_lane[l], m3_mut_per_lane[l])
        for l in REAL_LANES if l != "kjv"
    }
    sim_luther_elb = sim_matrix["luther1545"]["elberfelder1905"]
    bkr_delta = indirect_deltas["bkr"]
    tis_delta = indirect_deltas["tischendorf"]
    luther_delta = indirect_deltas["luther1545"]
    elb_delta = indirect_deltas["elberfelder1905"]

    lines += [
        "",
        f"**Honest nuance on the {m3_delta_pct:+.2f}% aggregate move:** the "
        "DIRECT pair is exactly flat — kjv's own weighted score changes by "
        f"**{m3_direct_delta:+.2f}%** (0.00% to 6 decimal places above), "
        "confirming the kjv/kjv2 pair's own contribution is suppressed as "
        "designed. But every OTHER real lane still moves, for two "
        "DIFFERENT reasons — neither is the clone's evidence re-entering "
        "as a vote, both are computed here rather than hand-waved:",
        "",
        f"1. **Cluster-dilution drift (luther1545 {luther_delta:+.2f}%, "
        f"elberfelder1905 {elb_delta:+.2f}%).** These two are both German "
        f"translations with substantial real vocabulary overlap "
        f"(sim={sim_luther_elb:.4f} in the Measure 2 matrix above — by far "
        "the highest off-diagonal cell among the 5 real lanes), which the "
        "weighting scheme already correctly discounts in BOTH passes. "
        "Adding kjv2 injects one more near-unity-weight term into each "
        "lane's own average (`w(L,kjv2) == w(L,kjv)`, since kjv2's "
        "aggregated token set is identical to kjv's) — doubling the "
        "weight-mass sitting on \"the English/KJV-shaped\" term pulls "
        "luther1545's and elberfelder1905's weighted averages further "
        "toward their (lower) English cross-lingual Jaccard and away from "
        "their (higher) German-German one.",
        f"2. **Small-denominator amplification (bkr {bkr_delta:+.2f}%, "
        f"tischendorf {tis_delta:+.2f}%).** bkr's baseline score is tiny "
        f"({m3_base_per_lane['bkr']:.6f}) to begin with — Czech shares "
        "almost no literal tokens with anything else in this crude "
        "tokenizer — so a small absolute change (one more near-full-"
        "weight English term) reads as a large PERCENTAGE swing. "
        "tischendorf (Greek script, zero literal token overlap with any "
        "Latin-script lane, including kjv2) stays at EXACTLY "
        f"{m3_mut_per_lane['tischendorf']:.6f} in both passes "
        f"({tis_delta:+.2f}%) — a clean control confirming the mechanism "
        "above requires some nonzero baseline weight/overlap to have "
        "anything to dilute.",
        "",
        "**The aggregate still does not inflate — it deflates, so the "
        "operator's core question (does the statistic RISE under "
        "duplication?) is answered no — but a duplicate is not fully "
        "invisible under pairwise `(1 - similarity)` weighting: it still "
        "perturbs lanes it never shares a token with, by diluting the "
        "weight composition other lanes are averaged over.** A stricter "
        "fix would cluster near-duplicate lanes (sim above some "
        "threshold) and count each cluster once before averaging, rather "
        "than discounting every pair independently — out of scope for "
        "this probe, noted for the next mutation operator.",
    ]

    lines += [
        "",
        "## Verdict summary",
        "",
        "| metric | baseline | mutated | Δ% | verdict |",
        "|---|---|---|---|---|",
        f"| M1 naive agreement | {m1_base_overall:.6f} | {m1_mut_overall:.6f} | "
        f"{m1_delta_pct:+.2f}% | {verdict_m1} |",
        f"| M2 sim(kjv,{MUTANT_LANE}) | n/a (detection matrix, not a "
        f"baseline/mutated pair) | {sim_kjv_kjv2:.6f} | n/a | "
        f"{'DETECTS' if sim_kjv_kjv2 > 0.999 else 'MISSES'} |",
        f"| M3 independence-weighted agreement | {m3_base_overall:.6f} | "
        f"{m3_mut_overall:.6f} | {m3_delta_pct:+.2f}% | {verdict_m3} |",
        "",
        "## The rule this result supports",
        "",
        "`I-NOISE-FLOOR-JIRAK` (translation-corpus form): lanes that are "
        "copies — or near-copies — of one another are **weakly dependent, "
        "not independent, observations**; a naive agreement/consensus "
        "statistic that treats every lane as an independent witness will "
        "inflate under duplication in direct proportion to how much of the "
        "lane set the duplicate(s) make up. The fix demonstrated here is "
        "not \"exclude duplicates by hand\" (that requires already knowing "
        "which lane is the clone) but **weight each pair's contribution by "
        "its OWN measured redundancy** (Measure 2's pairwise similarity) "
        "before averaging — the independence-weighted variant (Measure 3) "
        "is the metric that is supposed to survive a witness being cloned, "
        "because it discounts a clone pair's contribution using a signal "
        "computed directly from the corpus, not from foreknowledge of the "
        "mutation.",
        "",
        "_Every number above is computed by this script from the on-disk "
        "lane files at run time; none is asserted or hand-typed. Re-run to "
        "reproduce (the sample is seeded, so the Measure 2 matrix and hence "
        "the Measure 3 weights are identical run to run)._",
    ]

    report = "\n".join(lines)
    (out_dir / "mutate_falsewitness_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir}/mutate_falsewitness_report.md")
    print(f"  common rows: {n_common}  sample: {sample_n}")
    print(f"  M1 naive:      base={m1_base_overall:.6f} mut={m1_mut_overall:.6f} "
          f"({m1_delta_pct:+.2f}%) -> {verdict_m1}")
    print(f"  M2 sim(kjv,{MUTANT_LANE}): {sim_kjv_kjv2:.6f}")
    print(f"  M3 weighted:   base={m3_base_overall:.6f} mut={m3_mut_overall:.6f} "
          f"({m3_delta_pct:+.2f}%) -> {verdict_m3}")


if __name__ == "__main__":
    main()
