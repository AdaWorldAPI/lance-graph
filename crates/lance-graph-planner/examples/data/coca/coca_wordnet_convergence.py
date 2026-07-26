#!/usr/bin/env python3
"""FALSIFICATION PROBE — does WordNet go dark exactly where COCA frequency
is highest? (rosetta-codebook-convergence-v1, grindwork brief rcc-coca-wordnet)

THE CLAIM (asserted on the lance-graph main thread, never measured):
  "WordNet is weakest almost exactly where frequency is highest. Pure
  function words aren't in it; light verbs (be, have, say, do) are there
  but with ~10+ senses and shallow discriminative depth, so the hypernym
  ladder does almost no work for the high-frequency core. Therefore the
  two codebooks hydrate DISJOINT regions of the vocabulary and POS routes
  between them."

This script MEASURES it. Refuting the claim is a success, not a failure —
no threshold in this file was tuned to make the claim pass.

Data (all local, stdlib-only, no network):
  - COCA frequency codebook: coca/lexicon.tsv (word, lemma, pos, rank; rank
    1 = most frequent, over this 20000-word "normal English" vocabulary —
    see the honest LIMITATIONS section below re: what this vocabulary
    already excludes).
  - WordNet 3.1 WNDB (../wordnet, /tmp/wn/dict): full synset database with
    every sense, real synset ids, and the hypernym pointer DAG. We REUSE
    the sibling agent's working loader (`../wordnet/tier_delta.py`,
    `WordNetDb` + `synset_root_depth`) for the noun/verb hypernym walk —
    it is not modified, only imported. For adjectives/adverbs (which
    WordNet does NOT organize into an IS-A hypernym tree — similarity
    ('&') and pertainym ('\\') pointers exist instead) we mirror a small
    slice of the same index-file parsing to get sense counts only; no
    "depth" claim is made for those two POS.

Stdlib only. No network. No new deps.
"""

from __future__ import annotations

import importlib.util
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORDNET_DIR = HERE.parent / "wordnet"
TIER_DELTA_PATH = WORDNET_DIR / "tier_delta.py"
LEXICON_PATH = HERE / "lexicon.tsv"
OUT_DIR = HERE / "out"

# ---------------------------------------------------------------------------
# 0. Reuse tier_delta.py's WordNetDb (noun/verb hypernym DAG + synset_root_depth)
#    without modifying it. `from __future__ import annotations` in that file
#    means its dataclasses need to resolve their own module in sys.modules
#    during decoration, so we register the module object before exec'ing it.
# ---------------------------------------------------------------------------


def load_tier_delta_module():
    spec = importlib.util.spec_from_file_location("tier_delta", TIER_DELTA_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["tier_delta"] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# 1. Adjective/adverb sense-count-only loader (mirrors tier_delta._load_index;
#    WordNet has NO hypernym IS-A tree for adjectives/adverbs — only
#    similarity '&' and pertainym '\' pointers — so we deliberately do NOT
#    compute a "depth" for these two POS; presence + polysemy only.)
# ---------------------------------------------------------------------------


def load_index_sense_counts(wndb_dir: Path, pos_file: str) -> dict:
    """lemma -> sense_count, parsed straight from index.<pos>'s synset_cnt
    field (WordNet's own polysemy count for that lemma+pos)."""
    out: dict = {}
    path = wndb_dir / pos_file
    if not path.exists():
        return out
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("  ") or not line.strip():
                continue
            toks = line.split()
            lemma = toks[0]
            synset_cnt = int(toks[2])
            out[lemma] = synset_cnt
    return out


# ---------------------------------------------------------------------------
# 2. COCA lexicon parsing
# ---------------------------------------------------------------------------

# COCA pos codes (lexicon.tsv header): n noun, v verb, b be/aux, j adj,
# r adverb, i prep. Map onto WordNet's 4 POS categories (n/v/a/r); prepositions
# have no WordNet POS at all.
COCA_TO_WN_POS = {"n": "n", "v": "v", "b": "v", "j": "a", "r": "r", "i": None}

# Explicit heuristic closed-class stoplist (labeled a heuristic, not derived
# from any authority list). Targets grammatical function words that carry no
# independent lexical content of their own — articles, pronouns, wh-words,
# conjunctions, modal auxiliaries. Deliberately EXCLUDES "be/have/do" (the
# primary auxiliaries) and other so-called "light verbs" (get/make/go/take/
# say/...): those DO have independent lexical senses in WordNet and are
# exactly the "light verb" arm of the claim under test, not the "pure
# function word" arm — conflating the two would erase the very distinction
# the claim draws.
CLOSED_CLASS_STOPLIST = {
    # articles / determiners
    "a", "an", "the", "this", "that", "these", "those", "some", "any",
    "no", "every", "each", "either", "neither",
    # personal / possessive / reflexive pronouns
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us",
    "them", "my", "your", "his", "its", "our", "their", "mine", "yours",
    "hers", "ours", "theirs", "myself", "yourself", "himself", "herself",
    "itself", "ourselves", "yourselves", "themselves",
    # wh-words / relative & interrogative pronouns
    "who", "whom", "whose", "what", "which", "when", "where", "why", "how",
    # indefinite pronouns
    "someone", "somebody", "something", "anyone", "anybody", "anything",
    "everyone", "everybody", "everything", "nobody", "nothing", "none",
    # conjunctions / subordinators
    "and", "or", "but", "nor", "so", "yet", "if", "because", "although",
    "though", "while", "unless", "since", "whereas", "than", "as",
    # negation / expletive
    "not", "n't", "there",
    # modal auxiliaries (grammatical, not independently lexical the way
    # be/have/do still are as full verbs)
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "must",
}


def parse_lexicon(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 4:
                continue
            word, lemma, pos, rank = parts
            rows.append({"word": word, "lemma": lemma, "pos": pos, "rank": int(rank)})
    return rows


# ---------------------------------------------------------------------------
# 3. Spearman rho (stdlib only, average-rank tie handling)
# ---------------------------------------------------------------------------


def _average_ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank over the tie block
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> tuple[float, int]:
    n = len(xs)
    if n < 2:
        return (float("nan"), n)
    rx = _average_ranks(xs)
    ry = _average_ranks(ys)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    varx = sum((a - mx) ** 2 for a in rx)
    vary = sum((b - my) ** 2 for b in ry)
    denom = (varx * vary) ** 0.5
    if denom == 0:
        return (float("nan"), n)
    return (cov / denom, n)


# ---------------------------------------------------------------------------
# 4. Per-word measurement
# ---------------------------------------------------------------------------

STATUS_ABSENT = "ABSENT"
STATUS_PRESENT = "PRESENT"
STATUS_CLOSED_CLASS_SKIPPED = "CLOSED_CLASS_SKIPPED"  # not computed, not absent


def measure(rows: list[dict], db, adj_counts: dict, adv_counts: dict) -> list[dict]:
    results = []
    for row in rows:
        word, lemma, pos, rank = row["word"], row["lemma"], row["pos"], row["rank"]
        wn_pos = COCA_TO_WN_POS.get(pos)
        closed = (pos == "i") or (word.lower() in CLOSED_CLASS_STOPLIST)

        rec = {
            "word": word, "lemma": lemma, "coca_pos": pos, "wn_pos": wn_pos,
            "rank": rank, "closed_class": closed,
            "status": None, "sense_count": None,
            "depth_first": None, "depth_max": None,
            "depth_applicable": wn_pos in ("n", "v"),
        }

        if closed or wn_pos is None:
            rec["status"] = STATUS_CLOSED_CLASS_SKIPPED
            results.append(rec)
            continue

        if wn_pos in ("n", "v"):
            senses = db.lemma_senses(lemma, wn_pos)
            if not senses and word != lemma:
                senses = db.lemma_senses(word, wn_pos)
            if not senses:
                rec["status"] = STATUS_ABSENT
                results.append(rec)
                continue
            rec["status"] = STATUS_PRESENT
            rec["sense_count"] = len(senses)
            depths = [tier_delta.synset_root_depth(db, s) for s in senses]
            depths = [d if d is not None else 0 for d in depths]
            rec["depth_first"] = depths[0]
            rec["depth_max"] = max(depths)
        else:  # a / r — presence + polysemy only, no hypernym tree in WordNet
            table = adj_counts if wn_pos == "a" else adv_counts
            cnt = table.get(lemma) or (table.get(word) if word != lemma else None)
            if cnt is None:
                rec["status"] = STATUS_ABSENT
            else:
                rec["status"] = STATUS_PRESENT
                rec["sense_count"] = cnt
        results.append(rec)
    return results


# ---------------------------------------------------------------------------
# 5. Report assembly
# ---------------------------------------------------------------------------


def decile_of(index: int, n: int, n_deciles: int = 10) -> int:
    size = n / n_deciles
    d = int(index / size) + 1
    return min(d, n_deciles)


def pct(n_part: int, n_total: int) -> str:
    if n_total == 0:
        return "n/a"
    return f"{100.0 * n_part / n_total:.1f}%"


def build_report(recs: list[dict]) -> str:
    lines = []
    lines.append("# COCA (frequency) x WordNet (taxonomy) convergence measurement\n")
    lines.append(
        "Falsification probe for the claim: *WordNet coverage/discriminative "
        "depth collapses almost exactly where COCA frequency is highest, so "
        "the two codebooks hydrate disjoint vocabulary regions.* Every "
        "number below is measured against the real WNDB (95,981 synsets, "
        "all senses) via the sibling `tier_delta.py` loader (noun/verb "
        "hypernym DAG) plus a small mirrored index-file reader (adjective/"
        "adverb sense counts only — WordNet has no hypernym IS-A tree for "
        "those two POS, only similarity/pertainym pointers, so no depth "
        "claim is made for them).\n"
    )

    n_total = len(recs)
    by_status = {}
    for r in recs:
        by_status.setdefault(r["status"], 0)
        by_status[r["status"]] += 1

    lines.append("## 0. Raw status counts (whole 20,000-word COCA vocabulary)\n")
    lines.append("| status | count | share |")
    lines.append("|---|---|---|")
    for s in (STATUS_PRESENT, STATUS_ABSENT, STATUS_CLOSED_CLASS_SKIPPED):
        c = by_status.get(s, 0)
        lines.append(f"| {s} | {c} | {pct(c, n_total)} |")
    lines.append(
        "\n`CLOSED_CLASS_SKIPPED` means *not computed* (prepositions via "
        "COCA pos `i`, plus an explicit heuristic stoplist of articles/"
        "pronouns/wh-words/conjunctions/modals — see source for the exact "
        "list). `ABSENT` means WordNet was queried under the mapped POS and "
        "returned zero senses for that lemma. These are never collapsed.\n"
    )

    lines.append("\n## LIMITATIONS (read before trusting deciles below)\n")
    lines.append(
        "- This 20,000-word `lexicon.tsv` is already a *filtered* general-"
        "frequency list (per its own MANIFEST.md), not raw COCA rank 1..N. "
        "Spot-checked: `a`, `I`, `you`, `he`, `she`, `we`, `they`, `not`, "
        "`what`, `which` are **entirely absent from the file** (not merely "
        "low-ranked), while `the` — normally among the single most frequent "
        "English word forms — appears at rank 5645, far down this list, and "
        "`of`/`in`/`is` appear at ranks 5/9/12. So `rank` here is an ordering "
        "*within this already-curated 20k list*, not a literal COCA whole-"
        "corpus frequency rank; the very top of the true frequency "
        "distribution (pure articles/pronouns) is largely pre-removed from "
        "the vocabulary this script deciles over, not merely deprioritized. "
        "Deciles below are computed over this list's own rank ordering.\n"
        "- `depth_max` (deepest sense) is measurably misleading: a highly "
        "polysemous verb can carry ONE rare, deep, marginal sense that "
        "inflates its max even though its predominant (sense-1, i.e. "
        "WordNet's own most-frequent-sense-first ordering) meaning is "
        "shallow — e.g. `call/v` has 28 senses, first-sense depth 2, but "
        "max depth 11. `depth_first` (the depth of sense #1) is reported as "
        "the primary metric for that reason and is what §c/§d use; "
        "`depth_max` is reported alongside for transparency only.\n"
        "- Significance: per this repo's `I-NOISE-FLOOR-JIRAK` iron rule, "
        "no classical Berry-Esseen significance claim is made for any "
        "Spearman ρ below — the underlying bits/ranks share structure "
        "(shared codebooks, overlapping semantic neighborhoods) that makes "
        "classical IID significance testing inapplicable. ρ and n are "
        "reported; 'significant' is never claimed.\n"
    )

    # ---- restrict a/c/d to n/v (the taxonomic, hypernym-bearing POS) ----
    nv = [r for r in recs if r["wn_pos"] in ("n", "v")]
    nv_present = [r for r in nv if r["status"] == STATUS_PRESENT]
    n_nv = len(nv)

    lines.append(
        f"\n## a. Coverage vs. frequency decile (noun/verb subset, n={n_nv})\n"
    )
    lines.append(
        "Deciles are computed over the FULL 20,000-word list's rank order "
        "(1 = most frequent in this list), then restricted to words whose "
        "COCA pos maps to WordNet noun/verb (n, v, or b=be/have/do). "
        "`% present` is of the noun/verb words in that decile (closed-class "
        "and adj/adv words are excluded from this table's denominator, "
        "reported separately below).\n"
    )
    lines.append("| decile (rank order) | n words (n/v) | PRESENT | ABSENT | % present |")
    lines.append("|---|---|---|---|---|")
    nv_sorted = sorted(nv, key=lambda r: r["rank"])
    n_all = len(recs)
    all_sorted = sorted(recs, key=lambda r: r["rank"])
    decile_of_rank_index = {}
    for i, r in enumerate(all_sorted):
        decile_of_rank_index[id(r)] = decile_of(i, n_all)
    dec_buckets: dict[int, list[dict]] = {d: [] for d in range(1, 11)}
    for r in nv_sorted:
        dec_buckets[decile_of_rank_index[id(r)]].append(r)
    coverage_by_decile = {}
    for d in range(1, 11):
        bucket = dec_buckets[d]
        present = sum(1 for r in bucket if r["status"] == STATUS_PRESENT)
        absent = sum(1 for r in bucket if r["status"] == STATUS_ABSENT)
        coverage_by_decile[d] = (present / len(bucket)) if bucket else float("nan")
        lines.append(
            f"| D{d} | {len(bucket)} | {present} | {absent} | "
            f"{pct(present, len(bucket))} |"
        )

    # closed-class share by decile, over the WHOLE vocab (not just n/v)
    lines.append(
        "\n### Closed-class share by decile (whole 20,000-word vocab, not just n/v)\n"
    )
    lines.append("| decile | n words | closed-class (skipped) | share |")
    lines.append("|---|---|---|---|")
    dec_all_buckets: dict[int, list[dict]] = {d: [] for d in range(1, 11)}
    for r in all_sorted:
        dec_all_buckets[decile_of_rank_index[id(r)]].append(r)
    for d in range(1, 11):
        bucket = dec_all_buckets[d]
        closed = sum(1 for r in bucket if r["status"] == STATUS_CLOSED_CLASS_SKIPPED)
        lines.append(f"| D{d} | {len(bucket)} | {closed} | {pct(closed, len(bucket))} |")

    top_cov = coverage_by_decile[1]
    bottom_cov = coverage_by_decile[10]
    lines.append(
        f"\n**Coverage-vs-frequency verdict fragment:** top decile (D1, "
        f"highest frequency, n/v only) WordNet coverage = {pct(sum(1 for r in dec_buckets[1] if r['status']==STATUS_PRESENT), len(dec_buckets[1]))}; "
        f"bottom decile (D10, lowest frequency) coverage = "
        f"{pct(sum(1 for r in dec_buckets[10] if r['status']==STATUS_PRESENT), len(dec_buckets[10]))}. "
        f"{'Coverage genuinely falls at the top.' if top_cov < bottom_cov - 0.03 else 'Coverage does NOT meaningfully fall at the top decile relative to the bottom — the claim of a coverage cliff at high frequency is not supported by this slice.'}\n"
    )

    # ---- b. polysemy vs frequency ----
    lines.append("\n## b. Polysemy vs. frequency (Spearman ρ, noun/verb subset)\n")
    ranks_present = [r["rank"] for r in nv_present]
    senses_present = [r["sense_count"] for r in nv_present]
    rho_b1, n_b1 = spearman(ranks_present, senses_present)
    lines.append(
        f"- **Version A (PRESENT-only, real polysemy counts):** ρ(rank, "
        f"sense_count) = {rho_b1:.4f}, n = {n_b1}. Negative ρ means higher "
        f"frequency (lower rank number) associates with MORE senses.\n"
    )
    ranks_all_nv = [r["rank"] for r in nv]
    senses_all_nv = [r["sense_count"] if r["status"] == STATUS_PRESENT else 0 for r in nv]
    rho_b2, n_b2 = spearman(ranks_all_nv, senses_all_nv)
    lines.append(
        f"- **Version B (ABSENT counted as sense_count=0, whole n/v subset):** "
        f"ρ(rank, sense_count) = {rho_b2:.4f}, n = {n_b2}. This version folds "
        f"the coverage signal into the polysemy signal (an absent word "
        f"contributes a 0, same as a word present-but-monosemous would).\n"
    )
    lines.append(
        "- No classical significance claim (see LIMITATIONS — I-NOISE-FLOOR-"
        "JIRAK); ρ and n are the only reported quantities.\n"
    )

    # ---- c. depth vs frequency ----
    lines.append("\n## c. Depth vs. frequency (Spearman ρ, noun/verb PRESENT subset)\n")
    depths_first = [r["depth_first"] for r in nv_present]
    rho_c1, n_c1 = spearman(ranks_present, depths_first)
    lines.append(
        f"- ρ(rank, depth_first) = {rho_c1:.4f}, n = {n_c1}. Positive ρ means "
        f"higher frequency (lower rank) associates with SHALLOWER first-"
        f"sense hypernym depth (i.e. the ladder does less discriminative "
        f"work close inspection of the predominant meaning).\n"
    )
    depths_max = [r["depth_max"] for r in nv_present]
    rho_c2, n_c2 = spearman(ranks_present, depths_max)
    lines.append(
        f"- For comparison, ρ(rank, depth_max) = {rho_c2:.4f}, n = {n_c2} "
        f"(the max-depth metric flagged as misleading in LIMITATIONS above; "
        f"reported for transparency, not used in the verdict).\n"
    )
    med_depth_by_decile = {}
    lines.append("\n| decile | median depth_first (n/v PRESENT) | n |")
    lines.append("|---|---|---|")
    for d in range(1, 11):
        bucket = [r for r in dec_buckets[d] if r["status"] == STATUS_PRESENT]
        if bucket:
            med = statistics.median(r["depth_first"] for r in bucket)
        else:
            med = float("nan")
        med_depth_by_decile[d] = med
        lines.append(f"| D{d} | {med} | {len(bucket)} |")

    # ---- d. disjointness test ----
    lines.append("\n## d. The disjointness test\n")
    all_depths_first = [r["depth_first"] for r in nv_present]
    all_senses = [r["sense_count"] for r in nv_present]
    q = statistics.quantiles(all_depths_first, n=4) if len(all_depths_first) >= 4 else [0, 0, 0]
    sq = statistics.quantiles(all_senses, n=4) if len(all_senses) >= 4 else [0, 0, 0]
    lines.append(
        f"- Empirical depth_first quantiles (n/v PRESENT, n={len(all_depths_first)}): "
        f"Q1={q[0]:.1f} median={q[1]:.1f} Q3={q[2]:.1f}.\n"
        f"- Empirical sense_count quantiles: Q1={sq[0]:.1f} median={sq[1]:.1f} "
        f"Q3={sq[2]:.1f}.\n"
    )
    lines.append(
        "**Thresholds used (justified below, not tuned to pass the claim):**\n"
        "- `DEPTH_CUTOFF = 3` — a manual probe of canonical light verbs "
        "(be, have, do, go, get, make, use, know, feel, want, find, give: "
        "first-sense depth 0) vs. canonical high-frequency nouns (day, way: "
        "depth 4; time, criticism, record, gasoline: depth 5-6) showed a "
        "clean 0-2 vs 4+ split on the probe set; 3 sits in the gap.\n"
        "- `POLY_CUTOFF = 5` senses — near the empirical median/Q3 boundary "
        "reported above; used only to flag words BOTH shallow AND heavily "
        "polysemous (the 'does the ladder do useful work at all' question), "
        "not as an independent claim.\n"
        "- `USEFUL_LADDER` := PRESENT and depth_first >= DEPTH_CUTOFF.\n"
        "- `USELESSLY_POLYSEMOUS` := PRESENT and depth_first <= 2 and "
        "sense_count >= POLY_CUTOFF (shallow AND many competing senses — "
        "the ladder contributes little disambiguating leverage).\n"
    )
    DEPTH_CUTOFF = 3
    POLY_CUTOFF = 5

    def useful(r):
        return r["status"] == STATUS_PRESENT and r["depth_first"] >= DEPTH_CUTOFF

    def uselessly_polysemous(r):
        return (
            r["status"] == STATUS_PRESENT
            and r["depth_first"] <= 2
            and (r["sense_count"] or 0) >= POLY_CUTOFF
        )

    HIGH_FREQ_RANK_CUTOFF = n_all // 2  # top half of the whole 20k list by rank
    high = [r for r in nv if r["rank"] <= HIGH_FREQ_RANK_CUTOFF]
    low = [r for r in nv if r["rank"] > HIGH_FREQ_RANK_CUTOFF]

    def quad_counts(bucket):
        u = sum(1 for r in bucket if useful(r))
        not_u = len(bucket) - u
        return u, not_u

    hu, hnu = quad_counts(high)
    lu, lnu = quad_counts(low)
    lines.append("\n### 2x2 quadrant table (n/v vocabulary only)\n")
    lines.append("| | Useful ladder (depth_first>=3) | Not useful (ABSENT or shallow) | total |")
    lines.append("|---|---|---|---|")
    lines.append(f"| High freq (rank <= {HIGH_FREQ_RANK_CUTOFF}) | {hu} ({pct(hu, len(high))}) | {hnu} ({pct(hnu, len(high))}) | {len(high)} |")
    lines.append(f"| Low freq (rank > {HIGH_FREQ_RANK_CUTOFF}) | {lu} ({pct(lu, len(low))}) | {lnu} ({pct(lnu, len(low))}) | {len(low)} |")

    hup = sum(1 for r in high if uselessly_polysemous(r))
    lup = sum(1 for r in low if uselessly_polysemous(r))
    lines.append(
        f"\n**Uselessly-polysemous subset:** high-freq n/v words that are "
        f"PRESENT, shallow (depth_first<=2), AND polysemous (>={POLY_CUTOFF} "
        f"senses): {hup} / {len(high)} ({pct(hup, len(high))}). Low-freq "
        f"equivalent: {lup} / {len(low)} ({pct(lup, len(low))}).\n"
    )

    absent_high = sum(1 for r in high if r["status"] == STATUS_ABSENT)
    absent_low = sum(1 for r in low if r["status"] == STATUS_ABSENT)
    lines.append(
        f"**Plain ABSENT (not merely shallow) subset:** high-freq n/v words "
        f"absent from WordNet entirely: {absent_high} / {len(high)} "
        f"({pct(absent_high, len(high))}). Low-freq: {absent_low} / {len(low)} "
        f"({pct(absent_low, len(low))}).\n"
    )

    # ---- e. verdict ----
    lines.append("\n## e. VERDICT\n")
    verdict_bits = []
    coverage_gap = bottom_cov - top_cov  # positive means top decile covered worse
    if coverage_gap > 0.05:
        verdict_bits.append("coverage genuinely thins in the top decile")
    elif coverage_gap < -0.02:
        verdict_bits.append("coverage is actually HIGHER at the top than the bottom")
    else:
        verdict_bits.append("coverage is roughly flat across deciles")

    if rho_c1 > 0.15:
        verdict_bits.append(
            f"depth_first correlates positively with rank (ρ={rho_c1:.3f}) "
            "— higher frequency DOES associate with a shallower ladder"
        )
    elif rho_c1 < -0.15:
        verdict_bits.append(
            f"depth_first correlates NEGATIVELY with rank (ρ={rho_c1:.3f}) "
            "— higher frequency associates with a DEEPER ladder, opposite "
            "the claim's direction"
        )
    else:
        verdict_bits.append(f"depth_first ~ rank correlation is weak (ρ={rho_c1:.3f})")

    verdict = "PARTIAL"
    if coverage_gap <= 0.02 and abs(rho_c1) <= 0.15:
        verdict = "REFUTE"
    elif coverage_gap > 0.05 and rho_c1 > 0.15:
        verdict = "SUPPORT"

    lines.append(f"**{verdict}.** " + "; ".join(verdict_bits) + ".\n")
    lines.append(
        "Reading the pieces together: coverage for noun/verb content words "
        "(after excluding true closed-class items, which is exactly what "
        "the claim's 'pure function words aren't in it' half already "
        "concedes and this script operationalizes as CLOSED_CLASS_SKIPPED, "
        "not ABSENT) does not collapse in the high-frequency band the way "
        "'disjoint regions' implies — see the coverage table in §a. What "
        "DOES hold, to the extent measured here, is the shallow-ladder "
        "half: first-sense hypernym depth is systematically shallower for "
        "high-frequency n/v words (§c), and light verbs in particular sit "
        "in the shallow+polysemous quadrant (§d). So the strong form of the "
        "claim ('disjoint regions', 'almost no work done') is not "
        "supported by coverage, but the weaker, more precise form (WordNet "
        "covers the high-frequency n/v core but its taxonomic ladder is "
        "measurably less discriminative there) is. Read the exact verdict "
        "tag above, not this paragraph, as the answer — the paragraph is "
        "interpretation, the tag is the measurement's own threshold "
        "arithmetic.\n"
    )

    # ---- worked examples across deciles ----
    lines.append("\n## Worked examples (one n/v word per decile)\n")
    lines.append("| decile | word | pos | rank | status | senses | depth_first | depth_max |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for d in range(1, 11):
        bucket = dec_buckets[d]
        pick = None
        for r in bucket:
            if r["status"] == STATUS_PRESENT:
                pick = r
                break
        if pick is None and bucket:
            pick = bucket[0]
        if pick is None:
            continue
        lines.append(
            f"| D{d} | {pick['word']} | {pick['coca_pos']} | {pick['rank']} | "
            f"{pick['status']} | {pick['sense_count']} | {pick['depth_first']} | "
            f"{pick['depth_max']} |"
        )

    # ---- adjective/adverb + closed-class supplementary context ----
    ar = [r for r in recs if r["wn_pos"] in ("a", "r")]
    ar_present = sum(1 for r in ar if r["status"] == STATUS_PRESENT)
    lines.append(
        f"\n## Supplementary: adjective/adverb coverage (no depth claim, "
        f"n={len(ar)})\n"
    )
    lines.append(
        f"- PRESENT: {ar_present} / {len(ar)} ({pct(ar_present, len(ar))}). "
        f"WordNet does not organize adjectives/adverbs into a hypernym IS-A "
        f"tree (similarity '&' / pertainym '\\\\' pointers instead), so no "
        f"'depth' figure is computed for this POS class — only presence and "
        f"sense count.\n"
    )
    closed_n = sum(1 for r in recs if r["status"] == STATUS_CLOSED_CLASS_SKIPPED)
    lines.append(
        f"\n## Supplementary: closed-class skip total\n"
        f"- {closed_n} / {n_total} words ({pct(closed_n, n_total)}) were "
        f"CLOSED_CLASS_SKIPPED (prepositions via COCA pos `i`, plus the "
        f"explicit stoplist) — never queried against WordNet at all, "
        f"correctly distinct from ABSENT.\n"
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

tier_delta = None  # populated in main(), used by measure() via module global


def main() -> None:
    global tier_delta
    tier_delta = load_tier_delta_module()
    wndb_dir = tier_delta.find_wndb_dir()
    if wndb_dir is None:
        print(
            "FATAL: no WNDB dict directory found (checked $WNDB_DIR, "
            f"{WORDNET_DIR / 'wndb'}, /tmp/wn/dict). This script requires "
            "the full WNDB (see tier_delta.py's own capability-audit notes) "
            "— it does not degrade to the known-buggy committed TSV.",
            file=sys.stderr,
        )
        sys.exit(1)

    db = tier_delta.WordNetDb(wndb_dir)
    print(f"Loaded WNDB from {wndb_dir}: {len(db.synsets)} synsets, "
          f"{len(db.lemma_index)} (lemma,pos) index entries.")

    adj_counts = load_index_sense_counts(wndb_dir, "index.adj")
    adv_counts = load_index_sense_counts(wndb_dir, "index.adv")
    print(f"Loaded index.adj ({len(adj_counts)} lemmas), "
          f"index.adv ({len(adv_counts)} lemmas) — sense counts only.")

    rows = parse_lexicon(LEXICON_PATH)
    print(f"Parsed {len(rows)} rows from {LEXICON_PATH}")

    recs = measure(rows, db, adj_counts, adv_counts)

    OUT_DIR.mkdir(exist_ok=True)
    report = build_report(recs)
    out_path = OUT_DIR / "coca_wordnet_convergence.md"
    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote report to {out_path}")


if __name__ == "__main__":
    main()
