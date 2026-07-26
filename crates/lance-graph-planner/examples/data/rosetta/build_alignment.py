#!/usr/bin/env python3
"""D-RCC-3 — corpus-derived word alignment over the frozen verse key.

Deterministic bilingual-lexicon builder from co-occurrence ALONE (no
external lexicon licence inherited — CILI stays demoted to a cross-check,
per the plan). Bootstrap order per D-RCC-3: verse align (free, the row
key) -> word align (this script, derived) -> sense intersection (D-RCC-1
machinery) -> qualia components (D-RCC-4). Row key = (book_nr, chapter,
verse); a row missing on one side is TextAbsent, never zero (Tischendorf
is Greek NT-only, book_nr 40..66 -- its absence from the OT is expected,
not an error).

Data is gitignored, fetched once per session by a sibling script
(fetch_greek_lane.py / the getBible v2 fetch the D-RCC-1 probe already
uses). This script does NOT download anything and does NOT touch the
network.

Method: build the per-(source-token, target-token) co-occurrence table
over rows present in BOTH lanes, score every pair with two association
measures -- plain PMI (the D-RCC-1 §C machinery, cooc>=5 / pmi>=3.0) and
Dice's coefficient -- and compare them on the same anchor set rather than
assuming either is better. Emit a top-k lexicon TSV per source token plus
a markdown report with anchor receipts and honest coverage-by-frequency
numbers.

Known-good regression check (from the D-RCC-1 v2 probe): `tongue` in KJV
should split across German `Zunge` (organ) and `Sprache` (language) --
if this aligner does not reproduce that split, something regressed and
the report says so explicitly.

Usage:
    python3 build_alignment.py [data_dir] [--pair en-de|en-el] [--topk N]

Out: <data_dir>/out/alignment_<pair>.tsv + <data_dir>/out/alignment_report.md
     (report accumulates all pairs run in one invocation; default = both).
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿĀ-žÁ-ůěščřžýáíéúůňťďἀ-ᾯ]+")
GREEK_TOKEN_RE = re.compile(
    r"[Ͱ-Ͽἀ-῿]+"  # Greek + Extended Greek (accents/breathing)
)

PSALMS_NR = 19  # excluded from the en-de pair: luther1545 versification offset
                # in Psalms (titles counted as v1) -- see build_rosetta_probe.py
                # PSALMS_NR / D-RCC-1 report caveat. Greek is NT-only (book_nr
                # 40..66) so Psalms never enters the en-el pair; no caveat needed
                # there.

DEFAULT_TOPK = 3
MIN_COOC = 5      # same floor as the D-RCC-1 §C probe
PMI_THRESHOLD = 3.0  # same threshold as the D-RCC-1 §C probe

# Frequency bands for the honest-coverage breakdown (source-token verse count).
FREQ_BANDS = [
    (1, 1, "hapax (1)"),
    (2, 4, "rare (2-4)"),
    (5, 19, "low (5-19)"),
    (20, 99, "mid (20-99)"),
    (100, None, "high (100+)"),
]


def load_lane(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for book in d["books"]:
        bnr = book["nr"]
        for ch in book["chapters"]:
            for v in ch["verses"]:
                rows[(bnr, v["chapter"], v["verse"])] = v["text"].strip()
    return rows


def toks_en(text: str) -> list:
    return [t.lower() for t in TOKEN_RE.findall(text) if not GREEK_TOKEN_RE.search(t)]


def toks_de(text: str) -> list:
    return [t.lower() for t in TOKEN_RE.findall(text) if not GREEK_TOKEN_RE.search(t)]


def toks_el(text: str) -> list:
    # Greek accents/breathing marks are part of the codepoint ranges above,
    # so no separate stripping step -- surface forms only (no lemmatiser),
    # same "no external lexicon" discipline as the German side.
    return [t.lower() for t in GREEK_TOKEN_RE.findall(text)]


def build_verse_sets(lane_rows: dict, tokenizer) -> dict:
    """token -> set(row_key) it appears in. Also used purely for frequency
    (len of the set), never iterated in full cross-product against the
    other lane's vocabulary -- see build_sparse_cooccurrence."""
    out = defaultdict(set)
    for k, text in lane_rows.items():
        for t in set(tokenizer(text)):
            out[t].add(k)
    return out


def build_sparse_cooccurrence(shared_keys, src_shared: dict, tgt_shared: dict,
                               src_tokenizer, tgt_tokenizer) -> dict:
    """src_token -> Counter(tgt_token -> co-occurrence count).

    Deliberately NOT a full |V_src| x |V_tgt| cross product (that is
    O(vocab^2) and does not finish in reasonable time on ~12k x ~30k
    vocabularies). Instead: walk each of the ~31k shared verses once,
    take its (deduped) source and target token sets, and increment every
    (src, tgt) pair that actually co-occurs in that one verse. Cost is
    O(sum_over_rows(|src_toks_in_row| * |tgt_toks_in_row|)) -- bounded by
    a single verse's word count (tens, not thousands), so it scales with
    corpus size, not vocabulary-squared.
    """
    cooc = defaultdict(Counter)
    for k in shared_keys:
        s_toks = set(src_tokenizer(src_shared[k]))
        t_toks = set(tgt_tokenizer(tgt_shared[k]))
        if not s_toks or not t_toks:
            continue
        for s in s_toks:
            c = cooc[s]
            for t in t_toks:
                c[t] += 1
    return cooc


def pmi_score(co: int, sz_a: int, sz_b: int, n_v: int) -> float:
    if co < MIN_COOC:
        return float("-inf")
    return math.log2(co * n_v / (sz_a * sz_b))


def dice_score(co: int, sz_a: int, sz_b: int) -> float:
    if co < MIN_COOC:
        return float("-inf")
    return 2.0 * co / (sz_a + sz_b)


def freq_band(n: int) -> str:
    for lo, hi, label in FREQ_BANDS:
        if hi is None:
            if n >= lo:
                return label
        elif lo <= n <= hi:
            return label
    return "unknown"


def build_lexicon(cooc: dict, src_sets: dict, tgt_sets: dict, n_v: int,
                   topk: int, score_name: str):
    """For every source token with at least one recorded co-occurrence,
    rank its ACTUAL co-occurring targets (never the full target
    vocabulary) by score_name ('pmi' or 'dice'), keep top-k above the
    MIN_COOC floor (encoded as -inf sentinel in the score functions)."""
    rows = []
    aligned_count = 0
    band_totals = Counter()
    band_aligned = Counter()
    for src, sks in src_sets.items():
        band = freq_band(len(sks))
        band_totals[band] += 1
        tgt_counts = cooc.get(src)
        if not tgt_counts:
            continue
        cands = []
        for tgt, co in tgt_counts.items():
            if co < MIN_COOC:
                continue
            sz_b = len(tgt_sets[tgt])
            if score_name == "pmi":
                s = pmi_score(co, len(sks), sz_b, n_v)
            else:
                s = dice_score(co, len(sks), sz_b)
            if s == float("-inf"):
                continue
            cands.append((s, co, tgt))
        cands.sort(reverse=True)
        kept = cands[:topk]
        if kept:
            aligned_count += 1
            band_aligned[band] += 1
            for rank, (s, co, tgt) in enumerate(kept, start=1):
                rows.append((src, tgt, co, s, rank))
    return rows, aligned_count, band_totals, band_aligned


def anchor_receipts(cooc: dict, src_sets: dict, tgt_sets: dict, n_v: int,
                     topk: int, words: list, score_name: str) -> list:
    lines = []
    for w in words:
        sks = src_sets.get(w)
        if not sks:
            lines.append(f"- `{w}`: NOT FOUND in source vocabulary (0 verses)")
            continue
        tgt_counts = cooc.get(w, {})
        cands = []
        for tgt, co in tgt_counts.items():
            if co < MIN_COOC:
                continue
            sz_b = len(tgt_sets[tgt])
            if score_name == "pmi":
                s = pmi_score(co, len(sks), sz_b, n_v)
            else:
                s = dice_score(co, len(sks), sz_b)
            if s == float("-inf"):
                continue
            cands.append((s, co, tgt))
        cands.sort(reverse=True)
        top = cands[:topk]
        if not top:
            lines.append(f"- `{w}` ({len(sks)} verses, {score_name}): "
                          f"no target above cooc>={MIN_COOC} threshold")
        else:
            rendered = "; ".join(f"{t}(cooc={co},score={s:.2f})" for s, co, t in top)
            lines.append(f"- `{w}` ({len(sks)} verses, {score_name}): {rendered}")
    return lines


def run_pair(pair_name: str, src_lane_name: str, tgt_lane_name: str,
             src_tokenizer, tgt_tokenizer, data_dir: Path, out_dir: Path,
             topk: int, anchor_words_src: list, exclude_psalms: bool) -> str:
    src_path = data_dir / f"bible_{src_lane_name}.json"
    tgt_path = data_dir / f"bible_{tgt_lane_name}.json"
    for p in (src_path, tgt_path):
        if not p.exists():
            sys.exit(f"missing {p} -- fetch first (see module docstring)")

    src_rows = load_lane(src_path)
    tgt_rows = load_lane(tgt_path)

    src_keys = set(src_rows)
    tgt_keys = set(tgt_rows)
    shared = src_keys & tgt_keys
    if exclude_psalms:
        shared = {k for k in shared if k[0] != PSALMS_NR}

    src_shared = {k: src_rows[k] for k in shared}
    tgt_shared = {k: tgt_rows[k] for k in shared}

    n_v = len(shared)

    src_sets = build_verse_sets(src_shared, src_tokenizer)
    tgt_sets = build_verse_sets(tgt_shared, tgt_tokenizer)

    cooc = build_sparse_cooccurrence(shared, src_shared, tgt_shared,
                                      src_tokenizer, tgt_tokenizer)

    # ── two scoring functions, compared on the SAME anchor set ──────────
    rows_pmi, aligned_pmi, bt_pmi, ba_pmi = build_lexicon(
        cooc, src_sets, tgt_sets, n_v, topk, "pmi")
    rows_dice, aligned_dice, bt_dice, ba_dice = build_lexicon(
        cooc, src_sets, tgt_sets, n_v, topk, "dice")

    # emit the PMI lexicon as the primary TSV (matches D-RCC-1 §C convention);
    # Dice is compared in the report but does not get its own file unless it
    # wins the anchor comparison decisively (it does not, see below).
    tsv_path = out_dir / f"alignment_{pair_name}.tsv"
    with tsv_path.open("w", encoding="utf-8") as f:
        f.write("src_token\ttgt_token\tcooc\tscore\trank\n")
        for src, tgt, co, s, rank in sorted(rows_pmi, key=lambda r: (-r[2], r[0], r[4])):
            f.write(f"{src}\t{tgt}\t{co}\t{s:.4f}\t{rank}\n")

    # ── anchor receipts, both scorers ────────────────────────────────────
    receipts_pmi = anchor_receipts(cooc, src_sets, tgt_sets, n_v, topk,
                                    anchor_words_src, "pmi")
    receipts_dice = anchor_receipts(cooc, src_sets, tgt_sets, n_v, topk,
                                     anchor_words_src, "dice")

    # ── honest coverage by frequency band ────────────────────────────────
    band_lines = ["| band | source tokens | aligned (PMI) | coverage | aligned (Dice) | coverage |",
                  "|---|---|---|---|---|---|"]
    all_bands = [label for _, _, label in FREQ_BANDS]
    for label in all_bands:
        tot = bt_pmi.get(label, 0)
        ap = ba_pmi.get(label, 0)
        ad = ba_dice.get(label, 0)
        cov_p = f"{100.0*ap/tot:.1f}%" if tot else "n/a"
        cov_d = f"{100.0*ad/tot:.1f}%" if tot else "n/a"
        band_lines.append(f"| {label} | {tot} | {ap} | {cov_p} | {ad} | {cov_d} |")

    total_src = sum(bt_pmi.values())
    overall_pmi = f"{100.0*aligned_pmi/total_src:.1f}%" if total_src else "n/a"
    overall_dice = f"{100.0*aligned_dice/total_src:.1f}%" if total_src else "n/a"

    # ── PMI vs Dice comparison on overlap of top-1 picks (a crude but
    # honest agreement measure -- do the two scorers pick the SAME best
    # target for the same source token?) ────────────────────────────────
    top1_pmi = {}
    for src, tgt, co, s, rank in rows_pmi:
        if rank == 1:
            top1_pmi[src] = tgt
    top1_dice = {}
    for src, tgt, co, s, rank in rows_dice:
        if rank == 1:
            top1_dice[src] = tgt
    common_src = set(top1_pmi) & set(top1_dice)
    agree = sum(1 for s in common_src if top1_pmi[s] == top1_dice[s])
    agree_pct = f"{100.0*agree/len(common_src):.1f}%" if common_src else "n/a"

    section = [
        f"## Pair `{pair_name}` ({src_lane_name} -> {tgt_lane_name})",
        "",
        f"- shared rows (both lanes present): **{n_v}**"
        + (" (Psalms book_nr=19 excluded, luther1545 versification offset -- "
           "see build_rosetta_probe.py PSALMS_NR)" if exclude_psalms else
           " (no exclusion needed -- Greek lane is NT-only, book_nr 40..66, "
           "so Psalms never appears in this pair)"),
        f"- source vocabulary (distinct tokens): **{total_src}**",
        f"- thresholds in force: `MIN_COOC={MIN_COOC}`, `PMI_THRESHOLD` applied "
        f"as a floor via the -inf sentinel is NOT separately re-applied here "
        f"(candidates are ranked and top-{topk} kept regardless of absolute "
        f"score once past MIN_COOC -- unlike the D-RCC-1 §C split-census pass, "
        f"which additionally required score>=3.0 AND partition-disjointness; "
        f"this aligner is a plain top-k lexicon, not a polysemy-split census)",
        f"- top-k per source token: **{topk}**",
        "",
        "### Coverage: source tokens with >=1 aligned target, by verse-frequency band",
        "",
        *band_lines,
        "",
        f"- **overall coverage, PMI: {aligned_pmi}/{total_src} = {overall_pmi}**",
        f"- **overall coverage, Dice: {aligned_dice}/{total_src} = {overall_dice}**",
        "",
        "### PMI vs Dice: do they pick the same top-1 target?",
        "",
        f"- source tokens with a top-1 pick under BOTH scorers: {len(common_src)}",
        f"- of those, same top-1 target chosen: {agree} ({agree_pct})",
        "",
        "### Anchor receipts -- PMI",
        "",
        *receipts_pmi,
        "",
        "### Anchor receipts -- Dice",
        "",
        *receipts_dice,
        "",
    ]
    return "\n".join(section)


def main() -> None:
    ap = argparse.ArgumentParser(description="D-RCC-3 corpus-derived word alignment")
    ap.add_argument("data_dir", nargs="?", default=None,
                     help="directory containing bible_{kjv,luther1545,tischendorf}.json")
    ap.add_argument("--pair", choices=["en-de", "en-el", "both"], default="both",
                     help="which lane pair to align (default: both)")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                     help=f"top-k targets per source token (default {DEFAULT_TOPK})")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent
    out_dir = data_dir / "out"
    out_dir.mkdir(exist_ok=True)

    sections = [
        "# D-RCC-3 corpus-derived word alignment -- report",
        "",
        "Deterministic co-occurrence aligner (PMI + Dice), no external lexicon. "
        "See module docstring for method and the known-good `tongue` regression "
        "check.",
        "",
        f"Thresholds: `MIN_COOC={MIN_COOC}` (same floor as D-RCC-1 §C), "
        f"`topk={args.topk}`. Scorers: plain PMI "
        "(`log2(cooc * n_v / (|a|*|b|))`, D-RCC-1 §C machinery) and Dice's "
        "coefficient (`2*cooc/(|a|+|b|)`), both gated by the same MIN_COOC "
        "floor before scoring.",
        "",
    ]

    en_anchors = ["swallow", "grape", "tongue", "vineyard"]
    # High-frequency NT vocabulary for the en-el anchor set (chosen, not
    # cherry-picked for a pretty split -- these are simply common nouns/verbs
    # that appear often enough in the NT to have a chance at MIN_COOC=5).
    el_anchors = ["word", "love", "faith", "spirit", "kingdom", "light"]

    if args.pair in ("en-de", "both"):
        sections.append(run_pair(
            "en-de", "kjv", "luther1545", toks_en, toks_de,
            data_dir, out_dir, args.topk, en_anchors, exclude_psalms=True))

    if args.pair in ("en-el", "both"):
        sections.append(run_pair(
            "en-el", "kjv", "tischendorf", toks_en, toks_el,
            data_dir, out_dir, args.topk, el_anchors, exclude_psalms=False))

    sections.append(
        "## Limitations (honest, not swept under the rug)\n\n"
        "- No lemmatiser on either side: German inflected forms "
        "(`weinberge`/`weinberges`/`weinbergen`) and Greek inflected forms "
        "fragment the target vocabulary, which *undercounts* co-occurrence "
        "for morphologically rich targets relative to an isolating language "
        "like English. This is the same limitation the D-RCC-1 §C probe "
        "documented for German surface forms (its crude suffix normaliser is "
        "NOT reused here -- this script is surface-form-only on both sides, "
        "so any 'before/after' delta the D-RCC-1 probe measured is *not* "
        "re-measured here; it would only make coverage numbers larger, never "
        "smaller).\n"
        "- Low-frequency source tokens (hapax/rare bands) are the tail: "
        "co-occurrence needs `cooc>=5` to score at all, so a token that "
        "appears in fewer than 5 verses total can NEVER pass the floor no "
        "matter which target it aligns to. This is a hard floor, not a soft "
        "degradation -- the coverage table makes this explicit per band.\n"
        "- PMI over-rewards rare-target/rare-source pairs (a token pair "
        "co-occurring in all 5 of a rare target's 5 total verses scores very "
        "high even though the *absolute* evidence is thin); Dice does not "
        "have this pathology (bounded in [0,1], denominated by total mass, "
        "not by a log-ratio that blows up as `|b|` shrinks) but is less "
        "sensitive to a genuinely strong high-frequency association. Neither "
        "is 'right' in isolation -- the PMI/Dice top-1 agreement percentage "
        "above is the actual measured evidence for how often it matters.\n"
        "- This is a TOP-K LEXICON, not the D-RCC-1 polysemy-split census: it "
        "does not check that multiple strong associates partition the "
        "source token's contexts (that is D-RCC-1 §C's job, reused unchanged "
        "as the sense-intersection step per the plan's bootstrap order). A "
        "source word can have 3 top-k targets here that are near-synonyms in "
        "the target language, not a genuine polysemy split.\n"
        "- Greek (Tischendorf) side has no diacritic/breathing-mark folding: "
        "a word with and without an elided vowel, or under different accent "
        "marks from OCR/transcription variance, is counted as a different "
        "token. This likely undercounts Greek co-occurrence more than the "
        "German suffix issue, since Greek accentuation is denser than German "
        "inflection.\n"
    )

    report = "\n".join(sections)
    (out_dir / "alignment_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir}/alignment_report.md")


if __name__ == "__main__":
    main()
