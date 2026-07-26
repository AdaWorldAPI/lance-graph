#!/usr/bin/env python3
"""Rank-matched dispersion detector for closed-class tokens (no POS tagger).

Grindwork task #20. Fixes the defect recorded in `.claude/board/EPIPHANIES.md`
`E-LANE-CODEBOOKS-MORPHOLOGY-ORDERING-1`: `build_lane_codebooks.py`'s
`closed_class_guess` column is `rank<=150 AND dispersion>=0.60`, and the
dispersion conjunct is nearly always true in that rank range (it rejects
about ONE token per lane out of 150) — so the flag is operationally just
"rank<=150" and cannot do its intended job of routing qualia hydration
(open class -> WordNet ladder; closed class -> construction statistics) for
languages with no POS tagger (Czech `bkr`; there is no Greek lane in this
data set, see `codebook_summary.md`'s lane-roster correction).

Method
------
The core, independent signal is a RANK-MATCHED dispersion z-score. Raw
Juilland's D dispersion correlates strongly with rank on its own (frequent
tokens get more chances to spread across books, so their dispersion is
mechanically higher) — a flat `dispersion>=0.6` cutoff is really measuring
"is this token frequent", which is what `rank<=150` already says. To ask
the independent question "is this token *unusually evenly spread for a
token at this frequency*", each token's dispersion is compared against the
mean/std of dispersion for OTHER tokens in the same log-scaled rank bucket:

    z_disp(tok) = (dispersion(tok) - bin_mean) / max(bin_std, MIN_STD)

A token with a strongly positive z_disp is behaving like a function word
even relative to its frequency peers — this is the actual, non-circular
detector. `rank<=150` is retained ONLY as the pre-existing baseline for
comparison, not as part of the new detector.

Two supplementary signals are computed and reported (their effect on the
final F1 is measured, not assumed — see the German validation section of
the emitted report):

  - `rep_ratio = freq / verse_df` (>= 1): how often a token repeats within
    the SAME verse. Short closed-class words (conjunctions, articles,
    pronouns) recur within a single sentence far more than open-class
    content words; also z-scored per rank bin (`z_rep`) so it isn't just
    re-measuring frequency.
  - token length: closed-class words are short in English, German, AND
    Czech (a genuine cross-lingual regularity), but length is deliberately
    given the SMALLEST weight in the combined score — it is the one
    signal that would "transfer" to any language even if it were doing all
    the classifying, which is exactly the failure mode the brief warns
    against (a shortcut that looks reasonable but is not testing the
    hypothesis).

Combined score (bin-relative, all three z-scored the same way):

    score(tok) = z_disp(tok) + REP_ALPHA * z_rep(tok) - LEN_ALPHA * z_len(tok)
    predicted_closed = (score >= Z_THRESH) and (freq >= MIN_FREQ)

`MIN_FREQ` exists because Juilland's D on a handful of occurrences is
noisy (a hapax has D defined on n=1 book-frequency and is meaningless);
excluding low-support tokens is a support filter, not a rank filter — it
does not privilege frequent tokens beyond what's needed for a stable
dispersion estimate.

Validation
----------
`de/lexicon.tsv` (UD German-GSD + German-HDT derived, one row per unique
surface form: word, lemma, POS, rank; POS is a single-letter scheme:
`n v j r i d p m c t x`) is REAL ground truth with no ambiguity (one POS
per word form in that file, verified: 95,855 unique words, 0 collisions).
POS -> class mapping used here (documented, not silently assumed):

    closed = {i, d, p, c, t}   adposition, determiner, pronoun, conjunction, particle
    open   = {n, v, j, r}      noun, verb, adjective, adverb
    excluded = {m, x}          numeral, other -- genuinely ambiguous class status
                               (NUM in particular is treated as closed by
                               some POS schemes and open by others; excluded
                               from scoring rather than silently assigned)

Both German lanes in this data set (`luther1545`, `elberfelder1905`) are
scored against this lexicon by direct lowercase surface-form match (both
sides are already lowercase — verified: no uppercase tokens survive this
tokenizer's normalisation). Coverage (the fraction of codebook tokens that
matched a lexicon entry) is reported explicitly; unmatched tokens are
excluded from precision/recall, not counted as either class.

The DETECTOR CONFIG (Z_THRESH, MIN_FREQ, REP_ALPHA, LEN_ALPHA) is chosen by
grid search maximising closed-class F1 on the German validation set. This
is doing the small-grid-search-on-the-validation-set thing honestly, not
holding out a separate test split — with a single ~46-parameter grid and
two language lanes of a few thousand matched tokens each, that is a
reasonable trade for a grindwork task; it is disclosed in the report's
limitations section rather than hidden.

The Czech lane (`bkr`) has NO ground truth in this repo. It is scored with
the SAME config chosen on German (no separate Czech-specific tuning) and
explicitly marked UNVALIDATED in the report, with the top-30 flagged
tokens listed for human eyeballing.

No network, no third-party packages -- stdlib only (`csv`, `math`,
`statistics`, `collections`, `pathlib`).

Data (gitignored, already generated by sibling scripts -- not fetched here):
  <scratch>/out/codebook_{kjv,luther1545,elberfelder1905,bkr}.tsv
  <repo>/crates/lance-graph-planner/examples/data/de/lexicon.tsv

Run:
  python3 closed_class.py <scratch_out_dir> <repo_root> [--min-freq N] [--z-thresh Z]
Out:
  <scratch_out_dir>/closed_class_report.md
"""

from __future__ import annotations

import argparse
import statistics
from collections import defaultdict
from pathlib import Path

# ---------------------------------------------------------------------------
# Ground-truth POS -> class mapping (documented, see module docstring).
# ---------------------------------------------------------------------------
CLOSED_POS = {"i", "d", "p", "c", "t"}
OPEN_POS = {"n", "v", "j", "r"}
# "m" (numeral) and "x" (other/unclear) are deliberately excluded from
# scoring -- neither set claims them, see docstring.

# Rank bins: log-scaled edges shared across all lanes. A token's rank falls
# into exactly one half-open bin [lo, hi). The last bin is open-ended so it
# covers every lane's long tail regardless of vocabulary size (kjv maxes out
# near rank 12.4k, bkr near rank 40k).
RANK_BIN_EDGES = [1, 50, 150, 400, 1000, 2500, 6000, 15000, 10**9]

MIN_STD = 0.03  # floor on a bin's std so a near-degenerate bin doesn't blow up z


def rank_bin_index(rank: int) -> int:
    for i in range(len(RANK_BIN_EDGES) - 1):
        if RANK_BIN_EDGES[i] <= rank < RANK_BIN_EDGES[i + 1]:
            return i
    return len(RANK_BIN_EDGES) - 2


def read_codebook(path: Path) -> list[dict]:
    """Parse a `codebook_<lane>.tsv`, skipping the `#`-prefixed doc header."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        header_seen = False
        for line in f:
            if line.startswith("#"):
                continue
            if not header_seen:
                header_seen = True  # this is the real (non-#) header row
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 7:
                continue
            token, freq, verse_df, rank, dispersion, is_hapax, baseline_guess = parts
            rows.append(
                {
                    "token": token,
                    "freq": int(freq),
                    "verse_df": int(verse_df),
                    "rank": int(rank),
                    "dispersion": float(dispersion),
                    "is_hapax": is_hapax == "1",
                    "baseline_guess": baseline_guess == "1",
                }
            )
    return rows


def load_german_lexicon(path: Path) -> dict[str, str]:
    """word (lowercase) -> single-letter POS. One row per word, no dupes."""
    lex: dict[str, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            word, _lemma, pos = parts[0], parts[1], parts[2]
            lex[word.lower()] = pos
    return lex


def pos_to_class(pos: str) -> str | None:
    if pos in CLOSED_POS:
        return "closed"
    if pos in OPEN_POS:
        return "open"
    return None  # excluded (m, x)


# ---------------------------------------------------------------------------
# Feature computation: bin-relative z-scores.
# ---------------------------------------------------------------------------
def compute_bin_stats(rows: list[dict], value_key: str) -> dict[int, tuple[float, float]]:
    buckets: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        buckets[rank_bin_index(r["rank"])].append(r[value_key])
    stats: dict[int, tuple[float, float]] = {}
    for b, vals in buckets.items():
        mean = statistics.fmean(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        stats[b] = (mean, max(std, MIN_STD))
    return stats


def annotate_features(rows: list[dict]) -> None:
    """Mutates rows in place: adds rep_ratio, length, and per-bin z-scores."""
    for r in rows:
        r["rep_ratio"] = r["freq"] / r["verse_df"] if r["verse_df"] else 1.0
        r["length"] = len(r["token"])

    disp_stats = compute_bin_stats(rows, "dispersion")
    rep_stats = compute_bin_stats(rows, "rep_ratio")
    len_stats = compute_bin_stats(rows, "length")

    for r in rows:
        b = rank_bin_index(r["rank"])
        d_mean, d_std = disp_stats[b]
        rep_mean, rep_std = rep_stats[b]
        len_mean, len_std = len_stats[b]
        r["z_disp"] = (r["dispersion"] - d_mean) / d_std
        r["z_rep"] = (r["rep_ratio"] - rep_mean) / rep_std
        r["z_len"] = (r["length"] - len_mean) / len_std


def score_row(r: dict, rep_alpha: float, len_alpha: float) -> float:
    return r["z_disp"] + rep_alpha * r["z_rep"] - len_alpha * r["z_len"]


def detect(rows: list[dict], z_thresh: float, min_freq: int, rep_alpha: float, len_alpha: float) -> list[bool]:
    out = []
    for r in rows:
        s = score_row(r, rep_alpha, len_alpha)
        out.append(s >= z_thresh and r["freq"] >= min_freq)
    return out


# ---------------------------------------------------------------------------
# Evaluation against ground truth.
# ---------------------------------------------------------------------------
def evaluate(
    rows: list[dict], lexicon: dict[str, str], predicted: list[bool]
) -> dict:
    """Precision/recall/F1 for the "closed" label, restricted to tokens with
    an unambiguous ground-truth class (excludes unmatched + m/x POS)."""
    tp = fp = fn = tn = 0
    matched = 0
    total = len(rows)
    for r, pred in zip(rows, predicted):
        pos = lexicon.get(r["token"])
        if pos is None:
            continue
        cls = pos_to_class(pos)
        if cls is None:
            continue
        matched += 1
        truth_closed = cls == "closed"
        if pred and truth_closed:
            tp += 1
        elif pred and not truth_closed:
            fp += 1
        elif not pred and truth_closed:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "matched": matched,
        "total": total,
        "coverage": matched / total if total else 0.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def grid_search(
    rows: list[dict], lexicon: dict[str, str]
) -> tuple[dict, dict]:
    """Returns (best_config, best_eval) maximising F1 over the grid."""
    z_thresh_grid = [-0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    min_freq_grid = [1, 5, 10, 20, 30, 50]
    rep_alpha_grid = [0.0, 0.25, 0.5, 1.0]
    len_alpha_grid = [0.0, 0.1, 0.25]

    best_cfg = None
    best_eval = None
    for z in z_thresh_grid:
        for mf in min_freq_grid:
            for ra in rep_alpha_grid:
                for la in len_alpha_grid:
                    pred = detect(rows, z, mf, ra, la)
                    ev = evaluate(rows, lexicon, pred)
                    if best_eval is None or ev["f1"] > best_eval["f1"]:
                        best_eval = ev
                        best_cfg = {
                            "z_thresh": z,
                            "min_freq": mf,
                            "rep_alpha": ra,
                            "len_alpha": la,
                        }
    return best_cfg, best_eval


def baseline_eval(rows: list[dict], lexicon: dict[str, str]) -> dict:
    predicted = [r["baseline_guess"] for r in rows]
    return evaluate(rows, lexicon, predicted)


def apply_config(rows: list[dict], cfg: dict) -> list[bool]:
    return detect(rows, cfg["z_thresh"], cfg["min_freq"], cfg["rep_alpha"], cfg["len_alpha"])


# ---------------------------------------------------------------------------
# Report.
# ---------------------------------------------------------------------------
def fmt_pct(x: float) -> str:
    return f"{100 * x:.2f}%"


def build_report(
    german_lanes: list[str],
    german_rows_by_lane: dict[str, list[dict]],
    lexicon_path: Path,
    lexicon_size: int,
    best_cfg: dict,
    combined_new_eval: dict,
    combined_baseline_eval: dict,
    per_lane_new_eval: dict[str, dict],
    per_lane_baseline_eval: dict[str, dict],
    bkr_rows: list[dict],
    bkr_flagged: list[dict],
) -> str:
    lines: list[str] = []
    lines.append("# Closed-class detector — rank-matched dispersion z-score")
    lines.append("")
    lines.append(
        "Task #20 grindwork. Replaces the operationally-inert "
        "`closed_class_guess` column (`rank<=150 AND dispersion>=0.60`, "
        "flags 148-150/150 tokens per lane — see `EPIPHANIES.md` "
        "`E-LANE-CODEBOOKS-MORPHOLOGY-ORDERING-1`) with a detector that "
        "measures dispersion RELATIVE to a rank-matched baseline, so it is "
        "not just re-measuring rank."
    )
    lines.append("")

    lines.append("## Method")
    lines.append("")
    lines.append(
        "For each token, bin it by rank (log-scaled bins: "
        + ", ".join(f"[{a},{b})" for a, b in zip(RANK_BIN_EDGES, RANK_BIN_EDGES[1:-1] + ['inf']))
        + "). Within its bin, z-score three signals against the OTHER tokens "
        "in that bin:"
    )
    lines.append("")
    lines.append("- `z_disp` — Juilland's D dispersion (the primary, independent signal)")
    lines.append(
        "- `z_rep` — repetition-within-verse ratio (`freq / verse_df`), "
        "small weight, function words repeat inside one sentence more than "
        "content words"
    )
    lines.append(
        "- `z_len` — token length, SMALLEST weight deliberately (closed-class "
        "words are short in English/German/Czech, but length alone is a "
        "shortcut that would not prove anything about the dispersion "
        "hypothesis, so it is capped low)"
    )
    lines.append("")
    lines.append("Combined score: `score = z_disp + REP_ALPHA*z_rep - LEN_ALPHA*z_len`.")
    lines.append("")
    lines.append("`predicted_closed = (score >= Z_THRESH) and (freq >= MIN_FREQ)`.")
    lines.append("")
    lines.append(f"`MIN_STD` floor on bin std: `{MIN_STD}` (prevents z-blowup in low-variance bins).")
    lines.append("")

    lines.append("## Thresholds in force (selected by grid search on German)")
    lines.append("")
    lines.append("Grid: `Z_THRESH in [-0.5..1.5, 9 values]`, `MIN_FREQ in [1,5,10,20,30,50]`, "
                  "`REP_ALPHA in [0,0.25,0.5,1.0]`, `LEN_ALPHA in [0,0.1,0.25]` "
                  "(9*6*4*3 = 648 combos), maximising closed-class F1 "
                  "on the combined German (luther1545 + elberfelder1905) validation set.")
    lines.append("")
    lines.append(f"- `Z_THRESH = {best_cfg['z_thresh']}`")
    lines.append(f"- `MIN_FREQ = {best_cfg['min_freq']}`")
    lines.append(f"- `REP_ALPHA = {best_cfg['rep_alpha']}`")
    lines.append(f"- `LEN_ALPHA = {best_cfg['len_alpha']}`")
    lines.append("")
    lines.append(
        "**Honest caveat on tuning:** this grid search maximises F1 ON the "
        "German validation set itself (no held-out split) — a small, "
        "declared grid, not a hidden hyperparameter search. Treat the "
        "German F1 below as an upper bound on out-of-sample performance, "
        "not an unbiased estimate."
    )
    lines.append("")

    lines.append("## German ground-truth validation")
    lines.append("")
    lines.append(f"Ground truth: `{lexicon_path}` ({lexicon_size} unique German word forms, "
                  "one POS letter per word, 0 ambiguous duplicates verified). "
                  "POS -> class mapping (documented in the module docstring):")
    lines.append("")
    lines.append("- closed = `{i, d, p, c, t}` (adposition, determiner, pronoun, conjunction, particle)")
    lines.append("- open = `{n, v, j, r}` (noun, verb, adjective, adverb)")
    lines.append("- excluded from scoring = `{m, x}` (numeral, other — genuinely ambiguous class)")
    lines.append("")
    lines.append(
        "Both German lanes (`luther1545`, `elberfelder1905`) matched to the "
        "lexicon by direct lowercase surface-form match (both sides already "
        "lowercase, verified no uppercase survives this tokenizer)."
    )
    lines.append("")

    lines.append("### Combined German (both lanes) — detector vs baseline")
    lines.append("")
    lines.append("| metric | new detector (rank-matched z) | old baseline (`rank<=150`) |")
    lines.append("|---|---:|---:|")
    lines.append(f"| coverage (matched/scored tokens) | {fmt_pct(combined_new_eval['coverage'])} ({combined_new_eval['matched']}/{combined_new_eval['total']}) | {fmt_pct(combined_baseline_eval['coverage'])} ({combined_baseline_eval['matched']}/{combined_baseline_eval['total']}) |")
    lines.append(f"| TP | {combined_new_eval['tp']} | {combined_baseline_eval['tp']} |")
    lines.append(f"| FP | {combined_new_eval['fp']} | {combined_baseline_eval['fp']} |")
    lines.append(f"| FN | {combined_new_eval['fn']} | {combined_baseline_eval['fn']} |")
    lines.append(f"| TN | {combined_new_eval['tn']} | {combined_baseline_eval['tn']} |")
    lines.append(f"| **Precision** | **{combined_new_eval['precision']:.4f}** | {combined_baseline_eval['precision']:.4f} |")
    lines.append(f"| **Recall** | **{combined_new_eval['recall']:.4f}** | {combined_baseline_eval['recall']:.4f} |")
    lines.append(f"| **F1** | **{combined_new_eval['f1']:.4f}** | {combined_baseline_eval['f1']:.4f} |")
    lines.append("")

    delta_f1 = combined_new_eval["f1"] - combined_baseline_eval["f1"]
    if delta_f1 > 0.001:
        verdict = f"**The new detector beats the baseline by {delta_f1:+.4f} F1.**"
    elif delta_f1 < -0.001:
        verdict = (
            f"**The new detector does NOT beat the baseline (delta {delta_f1:+.4f} F1) "
            "— reporting this honestly per the task brief.** The baseline's "
            "TN-heavy composition (it almost never flags anything outside "
            "rank<=150, so recall is capped near ~150/N-closed but precision "
            "can still be high on the tokens it does flag) is a real, if "
            "brittle, strategy; the rank-matched detector's independence "
            "from the raw rank cutoff trades some baseline precision for "
            "broader recall (it also flags closed-class tokens outside the "
            "top-150), and on this validation set that trade did not net "
            "positive."
        )
    else:
        verdict = "**No meaningful F1 difference on this validation set.**"
    lines.append(verdict)
    lines.append("")

    lines.append("### Per-lane breakdown")
    lines.append("")
    lines.append("| lane | new P | new R | new F1 | baseline P | baseline R | baseline F1 |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for lane in german_lanes:
        ne = per_lane_new_eval[lane]
        be = per_lane_baseline_eval[lane]
        lines.append(
            f"| {lane} | {ne['precision']:.4f} | {ne['recall']:.4f} | {ne['f1']:.4f} "
            f"| {be['precision']:.4f} | {be['recall']:.4f} | {be['f1']:.4f} |"
        )
    lines.append("")

    lines.append("## Czech (bkr) application — UNVALIDATED")
    lines.append("")
    lines.append(
        "**No ground truth exists for Czech in this repo.** The tuned config "
        "above (chosen on German only, no Czech-specific tuning) is applied "
        "as-is. This arm is exploratory, not a validated result."
    )
    lines.append("")
    lines.append(f"- Total bkr tokens scored: {len(bkr_rows)}")
    lines.append(f"- Flagged closed-class: {len(bkr_flagged)} ({fmt_pct(len(bkr_flagged)/len(bkr_rows) if bkr_rows else 0.0)})")
    lines.append(f"- Old baseline (`rank<=150`) flagged: {sum(1 for r in bkr_rows if r['baseline_guess'])}")
    lines.append("")
    lines.append("Top-30 flagged tokens (by score, descending) for human eyeballing:")
    lines.append("")
    lines.append("| rank | token | freq | dispersion | z_disp | score |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for r in bkr_flagged[:30]:
        lines.append(
            f"| {r['rank']} | {r['token']} | {r['freq']} | {r['dispersion']:.4f} "
            f"| {r['z_disp']:.3f} | {r['_score']:.3f} |"
        )
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- **No held-out split for German.** The reported German F1 is the "
        "best F1 found by grid search ON that same set; treat it as an "
        "optimistic estimate, not a clean generalisation number."
    )
    lines.append(
        "- **Czech has zero ground truth.** The bkr application is "
        "plausibility-only; nothing in this report proves the Czech flags "
        "are correct."
    )
    lines.append(
        "- **Surface-form matching, not lemmatisation.** German is "
        "morphologically inflected; a lexicon entry for `der` does not "
        "automatically cover `dessen`/`deren`/etc. — those either have their "
        "own lexicon rows (if UD saw them) or fall into the unmatched/"
        "excluded bucket, lowering coverage rather than corrupting precision."
    )
    lines.append(
        "- **`m` (numeral) and `x` (other) POS classes are excluded from "
        "scoring entirely**, not silently folded into either class — this "
        "is a real, disclosed reduction in the number of tokens the "
        "precision/recall numbers are computed over (see `coverage` in the "
        "table above, which already reflects this)."
    )
    lines.append(
        "- **The dispersion formula itself (Juilland's D) is inherited "
        "unchanged from `build_lane_codebooks.py`** — this task only "
        "changes how dispersion is INTERPRETED (rank-matched z-score vs "
        "flat 0.60 cutoff), not how it is computed."
    )
    lines.append(
        "- **Rank-bin edges are hand-picked, not learned.** They were "
        "chosen to give roughly log-uniform coverage across each lane's "
        "vocabulary; a finer or coarser binning was not swept."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scratch_out_dir", type=Path, help="dir containing codebook_*.tsv (the 'out/' scratch dir)")
    ap.add_argument("repo_root", type=Path, help="lance-graph repo root (for crates/.../data/de/lexicon.tsv)")
    args = ap.parse_args()

    out_dir: Path = args.scratch_out_dir
    lexicon_path = args.repo_root / "crates/lance-graph-planner/examples/data/de/lexicon.tsv"

    german_lanes = ["luther1545", "elberfelder1905"]
    lane_paths = {lane: out_dir / f"codebook_{lane}.tsv" for lane in german_lanes}
    for lane, p in lane_paths.items():
        if not p.exists():
            raise SystemExit(f"missing {p}")
    if not lexicon_path.exists():
        raise SystemExit(f"missing {lexicon_path}")

    lexicon = load_german_lexicon(lexicon_path)

    german_rows_by_lane: dict[str, list[dict]] = {}
    combined_rows: list[dict] = []
    for lane in german_lanes:
        rows = read_codebook(lane_paths[lane])
        annotate_features(rows)
        german_rows_by_lane[lane] = rows
        combined_rows.extend(rows)

    best_cfg, _ = grid_search(combined_rows, lexicon)

    combined_predicted = apply_config(combined_rows, best_cfg)
    combined_new_eval = evaluate(combined_rows, lexicon, combined_predicted)
    combined_baseline_eval = baseline_eval(combined_rows, lexicon)

    per_lane_new_eval = {}
    per_lane_baseline_eval = {}
    for lane in german_lanes:
        rows = german_rows_by_lane[lane]
        pred = apply_config(rows, best_cfg)
        per_lane_new_eval[lane] = evaluate(rows, lexicon, pred)
        per_lane_baseline_eval[lane] = baseline_eval(rows, lexicon)

    # Czech application (unvalidated).
    bkr_path = out_dir / "codebook_bkr.tsv"
    if not bkr_path.exists():
        raise SystemExit(f"missing {bkr_path}")
    bkr_rows = read_codebook(bkr_path)
    annotate_features(bkr_rows)
    bkr_predicted = apply_config(bkr_rows, best_cfg)
    for r, pred in zip(bkr_rows, bkr_predicted):
        r["_flagged"] = pred
        r["_score"] = score_row(r, best_cfg["rep_alpha"], best_cfg["len_alpha"])
    bkr_flagged = sorted(
        (r for r in bkr_rows if r["_flagged"]), key=lambda r: r["_score"], reverse=True
    )

    report = build_report(
        german_lanes=german_lanes,
        german_rows_by_lane=german_rows_by_lane,
        lexicon_path=lexicon_path,
        lexicon_size=len(lexicon),
        best_cfg=best_cfg,
        combined_new_eval=combined_new_eval,
        combined_baseline_eval=combined_baseline_eval,
        per_lane_new_eval=per_lane_new_eval,
        per_lane_baseline_eval=per_lane_baseline_eval,
        bkr_rows=bkr_rows,
        bkr_flagged=bkr_flagged,
    )

    report_path = out_dir / "closed_class_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"wrote {report_path}")
    print(f"German combined: new F1={combined_new_eval['f1']:.4f} vs baseline F1={combined_baseline_eval['f1']:.4f}")
    print(f"config: {best_cfg}")
    print(f"bkr flagged: {len(bkr_flagged)}/{len(bkr_rows)}")


if __name__ == "__main__":
    main()
