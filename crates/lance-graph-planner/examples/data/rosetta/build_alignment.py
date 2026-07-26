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

Lemma-key mode (`--lemma-key`, OFF by default -- task #34, D-D-RCC-3
follow-up). `grape` and `grapes` are different surface tokens with no
lemmatiser: the singular's 8-verse co-occurrence surfaces stopword noise
while the real signal (`grapes -> trauben, pmi 9.54`) sits at a different
key. This is the same limit that moved the split census 48.9% -> 43.0%
(`E-RCC-1-V2-SPLIT-SURVIVES-NORMALISATION-1`). `--lemma-key` folds surface
tokens to a crude approximate stem BEFORE building verse-sets/co-occurrence,
merging counts across inflected forms:
  - German target side reuses the `build_rosetta_probe.py` `DE_SUFFIXES` /
    `DE_MIN_STEM_LEN` approach (copied here with attribution, NOT imported
    -- that module is a separate deliverable and may move independently).
  - English source side gets an equivalent crude suffix table
    (`EN_SUFFIXES`/`normalize_en`): plural/verb endings (`-s`, `-es`,
    `-ies` -> `-y`, `-ed`, `-ing`) PLUS archaic KJV 2nd/3rd-person verb
    endings (`-eth`, `-est`, e.g. `giveth`/`believest`), all min-stem-length
    guarded.
  - Greek (Tischendorf) target side has NO normaliser (none was asked for
    and none is safely craftable without touching diacritics/breathing
    marks, which is out of scope here) -- the en-el pair's lemma pass only
    folds the English side.
  - **This is NOT a lemmatiser** on either side: no dictionary, no ablaut/
    umlaut correction, no compound splitting, no irregular-verb table
    (`hath`, `saith` are NOT folded to `have`/`say` -- they simply don't
    match any suffix and pass through unchanged, a stated gap, not a bug).
  - Default OFF: normal invocations (no flag) produce byte-identical
    output to before this mode existed -- the primary `alignment_<pair>.tsv`
    is ALWAYS built from the raw (un-normalised) pass, flag or no flag, so
    a downstream consumer reading that file never sees a behavioural
    change. When `--lemma-key` is passed, an ADDITIONAL
    `alignment_<pair>_lemmakey.tsv` is written (new file, existing file
    untouched) and the report gains a before/after comparison section.

Usage:
    python3 build_alignment.py [data_dir] [--pair en-de|en-el] [--topk N] [--lemma-key]

Out: <data_dir>/out/alignment_<pair>.tsv (always, raw pass)
     + <data_dir>/out/alignment_<pair>_lemmakey.tsv (only with --lemma-key)
     + <data_dir>/out/alignment_report.md
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

# ── lemma-key normalisers (task #34, --lemma-key, OFF by default) ──────────
#
# German side: the SAME crude longest-suffix-strip approach as
# build_rosetta_probe.py's DE_SUFFIXES/normalize_de -- copied here with
# attribution rather than imported (that module is a separate deliverable
# and may move/change shape independently of this one). Explicitly NOT a
# lemmatiser: no dictionary, no ablaut/umlaut correction, no compound
# splitting.
DE_SUFFIXES = tuple(sorted({
    "ungen", "heiten", "keiten", "schaften",
    "chen", "lein",
    "ung", "heit", "keit", "schaft",
    "isch", "lich", "bar", "sam",
    "esse", "eren", "ern",
    "end", "ende", "enden", "endes", "ender", "est", "et",
    "en", "em", "es", "er", "e", "n", "s", "t",
}, key=len, reverse=True))
DE_MIN_STEM_LEN = 4  # guard: never strip a suffix if the remainder is shorter


def normalize_de(tok: str) -> str:
    """Crude longest-suffix strip with a minimum-stem-length guard.

    Copied from build_rosetta_probe.py's normalize_de (same table, same
    guard) -- see that module's docstring for the fuller caveat. Approximation
    only: no dictionary lookups, no ablaut/umlaut correction, no compound
    decomposition.
    """
    for suf in DE_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= DE_MIN_STEM_LEN:
            return tok[: -len(suf)]
    return tok


# English side: an equivalent crude table for KJV English -- ordinary
# plural/verb-form endings plus archaic 2nd/3rd-person singular verb
# endings that are common in KJV prose (giveth, believest). Order matters:
# -ies/-eth/-est are checked before the shorter -es/-s/-ed so a word is not
# stripped by the wrong (shorter) suffix first.
EN_MIN_STEM_LEN = 4  # same guard discipline as the German side


def normalize_en(tok: str) -> str:
    """Crude English surface-form fold: plurals, -ed/-ing, archaic KJV verb
    endings (-eth, -est). NOT a lemmatiser -- no irregular-verb table, so
    `hath`/`saith`/`doth` (irregular, not simple suffixation) pass through
    UNCHANGED rather than folding to `have`/`say`/`do`. This is a stated
    gap, not a bug: a true lemmatiser is out of scope for this script's
    "no external lexicon" discipline (see module docstring).
    """
    t = tok
    if t.endswith("ies") and len(t) - 3 + 1 >= EN_MIN_STEM_LEN:
        return t[:-3] + "y"
    for suf in ("eth", "est", "ing"):
        if t.endswith(suf) and len(t) - len(suf) >= EN_MIN_STEM_LEN:
            return t[: -len(suf)]
    if t.endswith("ed") and len(t) - 2 >= EN_MIN_STEM_LEN:
        return t[:-2]
    if t.endswith("es") and len(t) - 2 >= EN_MIN_STEM_LEN:
        # "-es" is added after a sibilant-ending stem (box->boxes,
        # dish->dishes, church->churches); anything else spelled "-es" is
        # really a silent-e stem + plain "-s" (grape->grapes), so strip
        # only the final "s" and keep the "e" -- this is the difference
        # between "grap" (wrong, the bug this comment replaces) and
        # "grape" (right, what makes grape/grapes actually share a key).
        # Crude and orthography-shaped, not a real morphological analyser.
        stem_no_es = t[:-2]
        if stem_no_es and (stem_no_es[-1] in "sxz" or stem_no_es.endswith(("ch", "sh"))):
            return stem_no_es
        if len(t) - 1 >= EN_MIN_STEM_LEN:
            return t[:-1]
        return stem_no_es
    if t.endswith("s") and not t.endswith("ss") and len(t) - 1 >= EN_MIN_STEM_LEN:
        return t[:-1]
    return t


def wrap_tokenizer(tokenizer, normalizer):
    """Compose a base tokenizer with an optional per-token normaliser.
    normalizer=None returns the base tokenizer unchanged (the raw pass)."""
    if normalizer is None:
        return tokenizer

    def wrapped(text: str) -> list:
        return [normalizer(t) for t in tokenizer(text)]

    return wrapped


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


def anchor_candidates(cooc: dict, src_sets: dict, tgt_sets: dict, n_v: int,
                       topk: int, word: str, score_name: str):
    """Top-k (score, cooc, target) tuples for ONE source word, or None if
    the word is absent from the source vocabulary. Factored out of
    anchor_receipts so callers that need the raw ranking (e.g. the
    tongue-survives lemma-key regression check) don't have to re-parse
    rendered report text."""
    sks = src_sets.get(word)
    if not sks:
        return None
    tgt_counts = cooc.get(word, {})
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
    return cands[:topk]


def anchor_receipts(cooc: dict, src_sets: dict, tgt_sets: dict, n_v: int,
                     topk: int, words: list, score_name: str) -> list:
    lines = []
    for w in words:
        sks = src_sets.get(w)
        if not sks:
            lines.append(f"- `{w}`: NOT FOUND in source vocabulary (0 verses)")
            continue
        top = anchor_candidates(cooc, src_sets, tgt_sets, n_v, topk, w, score_name)
        if not top:
            lines.append(f"- `{w}` ({len(sks)} verses, {score_name}): "
                          f"no target above cooc>={MIN_COOC} threshold")
        else:
            rendered = "; ".join(f"{t}(cooc={co},score={s:.2f})" for s, co, t in top)
            lines.append(f"- `{w}` ({len(sks)} verses, {score_name}): {rendered}")
    return lines


def run_pair(pair_name: str, src_lane_name: str, tgt_lane_name: str,
             src_tokenizer, tgt_tokenizer, data_dir: Path, out_dir: Path,
             topk: int, anchor_words_src: list, exclude_psalms: bool,
             lemma_key: bool = False) -> str:
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

    # ── honest plural-form check (surface-form tokenization has no
    # lemmatiser: "grape" and "grapes" are different tokens; measure both
    # so a low-frequency singular anchor's noisy top-k isn't mistaken for
    # the aligner failing on the LEMMA when it is really a frequency-split
    # artifact of no lemmatisation) ───────────────────────────────────────
    plural_note_lines = []
    for w in anchor_words_src:
        plural = w + "s"
        if plural in src_sets and w in src_sets and plural != w:
            n_sg, n_pl = len(src_sets[w]), len(src_sets[plural])
            if n_pl != n_sg:
                pl_receipt = anchor_receipts(cooc, src_sets, tgt_sets, n_v,
                                              topk, [plural], "pmi")[0]
                plural_note_lines.append(
                    f"- `{w}` ({n_sg} verses) vs `{plural}` ({n_pl} verses, "
                    f"pmi): {pl_receipt.split(': ', 1)[1]}")

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
        "### Plural-form check (no lemmatiser -- surface forms only)",
        "",
        *(plural_note_lines if plural_note_lines else
          ["- no anchor word had a distinct plural form present in the "
           "source vocabulary"]),
        "",
    ]

    # ── lemma-key pass (--lemma-key, OFF by default) ────────────────────
    # Everything above this point is the RAW pass, unchanged from before
    # this mode existed -- the primary TSV was already written from it.
    # This block runs a SECOND pass with normalised tokenizers and reports
    # before/after, never mutating the raw pass's numbers above.
    if lemma_key:
        tgt_normalizer = normalize_de if tgt_lane_name == "luther1545" else None
        src_tok_lemma = wrap_tokenizer(src_tokenizer, normalize_en)
        tgt_tok_lemma = wrap_tokenizer(tgt_tokenizer, tgt_normalizer)

        src_sets_l = build_verse_sets(src_shared, src_tok_lemma)
        tgt_sets_l = build_verse_sets(tgt_shared, tgt_tok_lemma)
        cooc_l = build_sparse_cooccurrence(shared, src_shared, tgt_shared,
                                            src_tok_lemma, tgt_tok_lemma)

        rows_pmi_l, aligned_pmi_l, bt_pmi_l, ba_pmi_l = build_lexicon(
            cooc_l, src_sets_l, tgt_sets_l, n_v, topk, "pmi")

        # new artifact only, never touches the primary alignment_<pair>.tsv
        lemma_tsv_path = out_dir / f"alignment_{pair_name}_lemmakey.tsv"
        with lemma_tsv_path.open("w", encoding="utf-8") as f:
            f.write("src_token\ttgt_token\tcooc\tscore\trank\n")
            for src, tgt, co, s, rank in sorted(rows_pmi_l, key=lambda r: (-r[2], r[0], r[4])):
                f.write(f"{src}\t{tgt}\t{co}\t{s:.4f}\t{rank}\n")

        total_src_l = sum(bt_pmi_l.values())
        overall_pmi_l = f"{100.0*aligned_pmi_l/total_src_l:.1f}%" if total_src_l else "n/a"

        # side-by-side band table -- RAW and LEMMA-KEY each computed against
        # their OWN post-fold vocabulary/frequencies (folding changes which
        # band a token falls in, same as the split-census before/after
        # measurement did -- this is not the same universe on both sides,
        # stated explicitly per the falsifiability rule).
        band_lines_l = [
            "| band | raw src tokens | raw aligned | raw coverage "
            "| lemma-key src tokens | lemma-key aligned | lemma-key coverage |",
            "|---|---|---|---|---|---|---|",
        ]
        for label in all_bands:
            tot_r, ap_r = bt_pmi.get(label, 0), ba_pmi.get(label, 0)
            tot_l, ap_l = bt_pmi_l.get(label, 0), ba_pmi_l.get(label, 0)
            cov_r = f"{100.0*ap_r/tot_r:.1f}%" if tot_r else "n/a"
            cov_l = f"{100.0*ap_l/tot_l:.1f}%" if tot_l else "n/a"
            band_lines_l.append(
                f"| {label} | {tot_r} | {ap_r} | {cov_r} | {tot_l} | {ap_l} | {cov_l} |")

        # ── the actual "lift" measurement: for each RAW hapax/rare/low
        # source token that was NOT aligned in the raw pass, does its
        # NORMALISED key become aligned in the lemma-key pass? This is a
        # direct token-level flip count, not two independently-banded
        # tables read side by side -- it is the honest answer to "does
        # merging counts over the cooc>=5 floor actually lift low-frequency
        # coverage, and by how much."
        aligned_src_raw = {r[0] for r in rows_pmi}
        aligned_src_lemma = {r[0] for r in rows_pmi_l}
        lift_considered = Counter()
        lift_flipped = Counter()
        low_bands = {"hapax (1)", "rare (2-4)", "low (5-19)"}
        for w, sks in src_sets.items():
            band = freq_band(len(sks))
            if band not in low_bands or w in aligned_src_raw:
                continue
            lift_considered[band] += 1
            if normalize_en(w) in aligned_src_lemma:
                lift_flipped[band] += 1
        lift_lines = ["| band | raw-unaligned tokens | now aligned via normalised key | flip rate |",
                      "|---|---|---|---|"]
        total_considered = total_flipped = 0
        for label in ("hapax (1)", "rare (2-4)", "low (5-19)"):
            c = lift_considered.get(label, 0)
            f_ = lift_flipped.get(label, 0)
            total_considered += c
            total_flipped += f_
            rate = f"{100.0*f_/c:.1f}%" if c else "n/a"
            lift_lines.append(f"| {label} | {c} | {f_} | {rate} |")
        total_rate = f"{100.0*total_flipped/total_considered:.1f}%" if total_considered else "n/a"
        lift_lines.append(f"| **all three bands** | {total_considered} | {total_flipped} | **{total_rate}** |")

        # ── anchor receipts under lemma-key, same anchor words (none of
        # the stock anchors -- swallow/grape/tongue/vineyard -- are
        # themselves suffix-stripped by normalize_en, so the literal word
        # is still the right lookup key; they only GAIN co-occurrence mass
        # from other surface forms folding into them) ──────────────────
        receipts_pmi_lemma = anchor_receipts(cooc_l, src_sets_l, tgt_sets_l,
                                              n_v, topk, anchor_words_src, "pmi")

        # ── tongue-survives regression check, programmatic (en-de only:
        # Zunge/Sprache is a German-target phenomenon). Targets are
        # compared as NORMALISED keys, since the German side is folded too
        # (normalize_de("zunge") -> "zung", normalize_de("sprache") ->
        # "sprach") -- the check is whether the ORGAN-sense and
        # LANGUAGE-sense associates both survive as distinct top-k
        # entries, not whether the exact spelling "zunge" reappears. ──
        tongue_lines = []
        if pair_name == "en-de" and "tongue" in anchor_words_src:
            top_lemma = anchor_candidates(cooc_l, src_sets_l, tgt_sets_l,
                                           n_v, topk, "tongue", "pmi") or []
            got = {t for _, _, t in top_lemma}
            zunge_key = normalize_de("zunge")
            sprache_key = normalize_de("sprache")
            survived = zunge_key in got and sprache_key in got
            rendered = ("; ".join(f"{t}(cooc={co},score={s:.2f})"
                                   for s, co, t in top_lemma)
                        if top_lemma else "(no candidates above threshold)")
            tongue_lines = [
                f"- expects normalised targets `{zunge_key}` (from `zunge`, "
                f"organ sense) AND `{sprache_key}` (from `sprache`, language "
                f"sense) both present in top-{topk}.",
                f"- lemma-key top-{topk} for `tongue`: {rendered}",
                f"- **regression check: "
                f"{'SURVIVED' if survived else 'REGRESSED — DO NOT TUNE AWAY, REPORT AS-IS'}**",
            ]
        elif pair_name == "en-de":
            tongue_lines = ["- `tongue` is not in this pair's anchor set; "
                             "check not applicable"]

        section.extend([
            "### Lemma-key pass (`--lemma-key`) — before/after",
            "",
            f"- raw source vocabulary: **{total_src}**, lemma-key source "
            f"vocabulary: **{total_src_l}** (fewer distinct keys = folding "
            f"happened; identical count would mean the normaliser never "
            f"fired on this vocabulary)",
            f"- **overall coverage, PMI raw: {aligned_pmi}/{total_src} = "
            f"{overall_pmi}** vs "
            f"**lemma-key: {aligned_pmi_l}/{total_src_l} = {overall_pmi_l}**",
            "",
            "#### Coverage by band, raw vs lemma-key (each on its own "
            "post-fold vocabulary)",
            "",
            *band_lines_l,
            "",
            "#### Low-frequency LIFT: raw-unaligned tokens whose normalised "
            "key becomes aligned",
            "",
            *lift_lines,
            "",
            "#### Anchor receipts under lemma-key (PMI)",
            "",
            *receipts_pmi_lemma,
            "",
            "#### `tongue` regression check (known-good anchor, must not "
            "break)",
            "",
            *tongue_lines,
            "",
        ])

    return "\n".join(section)


def main() -> None:
    ap = argparse.ArgumentParser(description="D-RCC-3 corpus-derived word alignment")
    ap.add_argument("data_dir", nargs="?", default=None,
                     help="directory containing bible_{kjv,luther1545,tischendorf}.json")
    ap.add_argument("--pair", choices=["en-de", "en-el", "both"], default="both",
                     help="which lane pair to align (default: both)")
    ap.add_argument("--topk", type=int, default=DEFAULT_TOPK,
                     help=f"top-k targets per source token (default {DEFAULT_TOPK})")
    ap.add_argument("--lemma-key", action="store_true", default=False,
                     help="OFF by default. Adds a second pass that folds "
                     "surface tokens to a crude approximate stem before "
                     "building co-occurrence (English suffix table + reused "
                     "German suffix table; see module docstring). Writes an "
                     "ADDITIONAL alignment_<pair>_lemmakey.tsv and a "
                     "before/after section in the report; the primary "
                     "alignment_<pair>.tsv is always the raw (un-normalised) "
                     "pass, flag or no flag, so existing consumers of that "
                     "file see no change.")
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
        f"`--lemma-key`: **{'ON' if args.lemma_key else 'OFF (default)'}**"
        + (" -- English suffix table `EN_SUFFIXES`/`normalize_en` "
           f"(min stem {EN_MIN_STEM_LEN}) + German suffix table "
           f"`DE_SUFFIXES`/`normalize_de` (min stem {DE_MIN_STEM_LEN}, copied "
           "from build_rosetta_probe.py). Greek target side has no "
           "normaliser." if args.lemma_key else " -- pass `--lemma-key` to "
           "run the additional before/after pass (see module docstring)."),
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
            data_dir, out_dir, args.topk, en_anchors, exclude_psalms=True,
            lemma_key=args.lemma_key))

    if args.pair in ("en-el", "both"):
        sections.append(run_pair(
            "en-el", "kjv", "tischendorf", toks_en, toks_el,
            data_dir, out_dir, args.topk, el_anchors, exclude_psalms=False,
            lemma_key=args.lemma_key))

    sections.append(
        "## Limitations (honest, not swept under the rug)\n\n"
        "- No lemmatiser on either side BY DEFAULT: German and English "
        "inflected forms (`weinberge`/`weinberges`/`weinbergen`, "
        "`grape`/`grapes`) and Greek inflected forms fragment the "
        "vocabulary, which *undercounts* co-occurrence for morphologically "
        "richer forms and can hide real signal behind a low-frequency "
        "surface split (the `grape`/`grapes` case in `E-D-RCC-3-ALIGNER-"
        "SHIPPED-DICE-NOT-BETTER-1`). This is the same limitation the "
        "D-RCC-1 §C probe documented for German surface forms "
        "(48.9% -> 43.0%, `E-RCC-1-V2-SPLIT-SURVIVES-NORMALISATION-1`). "
        "**Task #34 added `--lemma-key`** (OFF by default, so this script's "
        "default behaviour and primary TSV output are unchanged) reusing "
        "the German suffix table and adding an equivalent English one "
        "(including archaic KJV `-eth`/`-est` verb endings); see the module "
        "docstring and the report's per-pair \"Lemma-key pass\" section for "
        "the measured before/after. Neither normaliser is a lemmatiser: no "
        "dictionary, no irregular forms (`hath`/`saith` do not fold), no "
        "ablaut/umlaut correction, no compound splitting.\n"
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
        "- **Lemma-key is a real net positive on coverage (task #34: "
        "+3.8pp overall en-de, low-band flip rate ~21.6%, `grape` finds "
        "`traub`/`herling` instead of stopwords) but it is NOT uniformly "
        "beneficial, and the `tongue` anchor is the measured "
        "counter-example, reported as-is rather than tuned away.** The "
        "reused `DE_SUFFIXES` table's `chen` entry (meant for diminutives, "
        "e.g. `Mädchen`) spuriously matches the end of `sprachen` "
        "(`spra`+`chen`), folding it to `spra`, while the singular "
        "`sprache` folds to `sprach` (only the `e` suffix applies there) "
        "-- two forms of the SAME word land on two DIFFERENT normalised "
        "keys, so their evidence does not merge and `sprach`/`spra` "
        "individually rank below newly-boosted associates (`lipp`, "
        "`schweig`, `falsch`) that gained mass from unrelated folding "
        "elsewhere. This is a suffix-table collision, not a normalisation "
        "bug in this script's own logic -- it is the risk this task's "
        "brief warned about (a normaliser can both fix one fragmentation "
        "and introduce another), and the honest resolution is reporting "
        "the regression, not hand-tuning the DE_SUFFIXES table until the "
        "anchor looks right again.\n"
    )

    report = "\n".join(sections)
    (out_dir / "alignment_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir}/alignment_report.md")


if __name__ == "__main__":
    main()
