#!/usr/bin/env python3
"""D-RCC — per-lane VERSE-ATTESTED frequency + dispersion codebooks.

Companion to `build_rosetta_probe.py` (read-only reference for loader/style
conventions; not modified by this script). That probe measures cross-lane
overlap/alignment; THIS script builds one standalone codebook PER LANE —
English (kjv) already has a COCA frequency codebook and German (luther1545 /
via the UD-derived `de/out/lexicon.tsv`) already has one; Czech (bkr) and
(if present) Greek have none. Rather than hunt for a Czech/Greek treebank,
this builds the codebook the corpus itself licenses: how each surface form
behaves across the 66-ish books of its OWN lane, using nothing but the verse
text.

Data note (checked against the actual scratchpad fetch): the four lanes
shipped for this arc are `kjv` (English), `luther1545` (German),
`elberfelder1905` (German), `bkr` (Czech, Bible kralická) — i.e. TWO German
lanes, one English, one Czech. No Greek Bible JSON is present in this data
set (`--anchor`-style Greek New Testament material exists elsewhere in the
scratchpad as `proiel-greek-nt.xml`, but that is a different, non-verse-JSON
corpus and out of scope for this script). The summary report says this
explicitly rather than silently processing 4 lanes and letting a reader
assume one of them is Greek.

What each `out/codebook_<lane>.tsv` row means (documented here + in the file
header so the TSV is self-describing without this script):

  token                 — lowercased surface form (NOT a lemma — see caveats)
  freq                  — raw token count in this lane (whole Bible text)
  verse_df              — verse "document frequency": number of DISTINCT
                           verses (by (book, chapter, verse) key) in which
                           the token appears at least once
  rank                  — frequency rank within this lane, 1 = most frequent,
                           ties broken alphabetically for determinism
  dispersion            — Juilland's D across this lane's own books (0..1;
                           0 = concentrated in one/few books, 1 = evenly
                           spread across every book) — see formula below
  is_hapax              — 1 if freq == 1, else 0
  closed_class_guess    — 1/0 heuristic from rank+dispersion ONLY (see
                           CLOSED_CLASS_RANK_CUTOFF / _DISPERSION_MIN below).
                           NOT a POS tag. No POS tagger exists for cs/el in
                           this environment, so this is a crude proxy: high-
                           frequency AND well-dispersed tokens tend to be
                           function words, but proper nouns that recur in a
                           genealogy chapter, or a book-specific refrain, can
                           still slip through. Labelled a GUESS on purpose.

Juilland's D (dispersion), computed identically for all 4 lanes:
  For token t across this lane's B books, let rel[b] = freq(t, book b) /
  total_tokens(book b) (relative frequency, so book-length differences don't
  dominate). mean = avg(rel), population stdev = std(rel).
  D = 1 - (std/mean) / sqrt(B - 1), clamped to [0, 1] (mean == 0 -> D = 0.0).
  D near 1 means the token is used at roughly the same relative rate in
  every book (spread evenly); D near 0 means it is concentrated in very few
  books (e.g. a name that only occurs in one genealogy). This is the classic
  corpus-linguistics dispersion measure (Juilland & Chang-Rodriguez 1964);
  chosen over raw entropy because it is already normalized to [0, 1] and
  explicitly designed to punish per-part frequency spikes -- exactly the
  "proper noun frequent in one book only" case this codebook needs to catch.

Identical-logic rule (Iron Rule): tokenisation, dispersion formula, hapax
flag, and closed-class heuristic thresholds are THE SAME across all 4 lanes.
The only per-lane variable is the input text itself. Any place this script
deviates by language is called out explicitly in the summary report -- there
are none by design.

Tokenisation is crude: `[^\\W\\d_]+` (Unicode-aware "letters only" runs),
lowercased via Python's `str.lower()`. This is NOT a lemmatizer for any of
the 4 lanes -- surface-form frequencies only. Concretely:
  - German (luther1545, elberfelder1905): compounding is NOT split
    (`Weinberg`, `Weinberge`, `Weinberges`, `Weinbergen` are 4 distinct
    tokens); strong-verb ablaut and case endings are not folded.
  - Czech (bkr): heavy case/gender inflection is not folded (nominative/
    genitive/accusative forms of the same lemma are distinct tokens);
    diacritics are preserved as part of the token identity (`hltal` and
    "hltal" with any diacritic variant are different tokens).
  - English (kjv): -s/-ed/-ing surface inflections are not folded either
    (word/words/worded would be 3 tokens) -- English is simply less
    inflected, so this matters less, but the SAME crude rule is applied.
  - Greek: N/A, no Greek lane present in this data set (see note above).
This under-counts true lexical types and over-counts hapax legomena for the
more inflected languages (German, and especially Czech) relative to English
-- called out quantitatively in the summary's type/token/TTR table.

Absent =/= zero: a codebook is built entirely from ITS OWN lane's verses.
Cross-lane presence/absence (e.g. TextAbsent rows in build_rosetta_probe's
census) is not modelled here at all -- this script never compares lanes to
each other, it only describes each lane on its own terms.

No network, no third-party deps: stdlib only (`json`, `re`, `math`,
`collections`, `pathlib`).

Run:  python3 build_lane_codebooks.py <data_dir_with_bible_*.json>
Out:  <data_dir>/out/codebook_<lane>.tsv (one per lane)
      <data_dir>/out/codebook_summary.md
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LANES = ["kjv", "luther1545", "elberfelder1905", "bkr"]
LANE_LANGUAGE = {
    "kjv": "English",
    "luther1545": "German",
    "elberfelder1905": "German",
    "bkr": "Czech",
}

# Unicode-aware "letters only" run: excludes digits/underscore, includes any
# script's alphabetic characters (German umlauts/ß, Czech diacritics, etc.)
# without a hand-maintained per-language character-class list. Applied
# IDENTICALLY to all 4 lanes -- the one deliberate uniformity choice this
# script insists on (see module docstring "Identical-logic rule").
TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Closed-class heuristic thresholds (rank + dispersion ONLY, no POS tagger
# available for cs/el). Same constants for every lane.
CLOSED_CLASS_RANK_CUTOFF = 150
CLOSED_CLASS_DISPERSION_MIN = 0.60


def load_lane(path: Path) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    rows = {}
    for book in d["books"]:
        bnr = book["nr"]
        for ch in book["chapters"]:
            for v in ch["verses"]:
                rows[(bnr, v["chapter"], v["verse"])] = v["text"].strip()
    return rows


def toks(text: str) -> list:
    return [t.lower() for t in TOKEN_RE.findall(text)]


def juilland_d(counts_per_book: list, totals_per_book: list) -> float:
    """Juilland's D dispersion, 0..1. See module docstring for the formula."""
    b = len(counts_per_book)
    if b <= 1:
        return 1.0
    rel = [
        (counts_per_book[i] / totals_per_book[i]) if totals_per_book[i] > 0 else 0.0
        for i in range(b)
    ]
    mean = sum(rel) / b
    if mean == 0.0:
        return 0.0
    var = sum((r - mean) ** 2 for r in rel) / b
    std = var**0.5
    cv = std / mean
    d = 1.0 - cv / math.sqrt(b - 1)
    return max(0.0, min(1.0, d))


def build_codebook(rows: dict) -> dict:
    """Returns dict with per-token rows + corpus-level stats for one lane."""
    book_ids = sorted({k[0] for k in rows})
    book_index = {b: i for i, b in enumerate(book_ids)}
    n_books = len(book_ids)

    totals_per_book = [0] * n_books
    freq = Counter()
    verse_df = Counter()
    per_book_freq = defaultdict(lambda: [0] * n_books)
    n_tokens_total = 0

    for key, text in rows.items():
        bi = book_index[key[0]]
        tk = toks(text)
        totals_per_book[bi] += len(tk)
        n_tokens_total += len(tk)
        for t in set(tk):
            verse_df[t] += 1
        for t in tk:
            freq[t] += 1
            per_book_freq[t][bi] += 1

    dispersion = {
        t: juilland_d(per_book_freq[t], totals_per_book) for t in freq
    }

    # rank: freq desc, ties broken alphabetically for determinism
    ranked = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    rank_of = {t: i + 1 for i, (t, _) in enumerate(ranked)}

    codebook_rows = []
    for t, f in ranked:
        r = rank_of[t]
        d = dispersion[t]
        is_hapax = 1 if f == 1 else 0
        closed_class_guess = 1 if (
            r <= CLOSED_CLASS_RANK_CUTOFF and d >= CLOSED_CLASS_DISPERSION_MIN
        ) else 0
        codebook_rows.append(
            (t, f, verse_df[t], r, d, is_hapax, closed_class_guess)
        )

    n_types = len(freq)
    n_hapax = sum(1 for _, f in freq.items() if f == 1)

    return {
        "rows": codebook_rows,
        "n_books": n_books,
        "n_types": n_types,
        "n_tokens": n_tokens_total,
        "n_hapax": n_hapax,
        "n_verses": len(rows),
    }


def write_codebook_tsv(out_path: Path, lane: str, cb: dict) -> None:
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"# {lane} verse-attested frequency + dispersion codebook\n")
        f.write(
            "# token\tfreq\tverse_df\trank\tdispersion\tis_hapax\t"
            "closed_class_guess\n"
        )
        f.write(
            "# dispersion = Juilland's D over this lane's own books, 0..1 "
            "(see build_lane_codebooks.py docstring for the formula).\n"
        )
        f.write(
            "# closed_class_guess is a rank+dispersion HEURISTIC (no POS "
            f"tagger): 1 iff rank<={CLOSED_CLASS_RANK_CUTOFF} and "
            f"dispersion>={CLOSED_CLASS_DISPERSION_MIN}. Not a POS label.\n"
        )
        f.write(
            "token\tfreq\tverse_df\trank\tdispersion\tis_hapax\t"
            "closed_class_guess\n"
        )
        for t, freq, vdf, r, d, hap, cc in cb["rows"]:
            f.write(f"{t}\t{freq}\t{vdf}\t{r}\t{d:.4f}\t{hap}\t{cc}\n")


def format_top(rows, n=20):
    return [(t, f) for t, f, *_ in rows[:n]]


def format_top_dispersion(rows, min_freq=10, n=20):
    filtered = [r for r in rows if r[1] >= min_freq]
    filtered.sort(key=lambda r: (-r[4], -r[1]))
    return [(t, d) for t, f, vdf, rk, d, hap, cc in filtered[:n]]


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(
            "usage: python3 build_lane_codebooks.py "
            "<data_dir_with_bible_*.json>"
        )
    data_dir = Path(sys.argv[1])
    out_dir = data_dir / "out"
    out_dir.mkdir(exist_ok=True)

    lane_cbs = {}
    for lane in LANES:
        p = data_dir / f"bible_{lane}.json"
        if not p.exists():
            sys.exit(f"missing {p} — fetch first (see module docstring)")
        rows = load_lane(p)
        cb = build_codebook(rows)
        lane_cbs[lane] = cb
        out_path = out_dir / f"codebook_{lane}.tsv"
        write_codebook_tsv(out_path, lane, cb)
        print(
            f"wrote {out_path} "
            f"({cb['n_types']} types, {cb['n_tokens']} tokens, "
            f"{cb['n_hapax']} hapax, {cb['n_books']} books, "
            f"{cb['n_verses']} verses)"
        )

    # ── summary ────────────────────────────────────────────────────────────
    lines = [
        "# Per-lane verse-attested codebook summary",
        "",
        "Built by `build_lane_codebooks.py`. Four PD Bible lanes, one "
        "frozen verse key `(book_nr, chapter, verse)`. Each codebook is "
        "built entirely from its OWN lane's text — no cross-lane "
        "comparison happens in this script (that is `build_rosetta_probe.py`'s "
        "job).",
        "",
        "**Lane roster note (correcting the aspirational \"Czech and Greek\" "
        "framing this arc started from):** the 4 lanes actually shipped are "
        "`kjv` (English), `luther1545` (German), `elberfelder1905` "
        "(German), `bkr` (Czech). That is **two German lanes, one English, "
        "one Czech** — there is **no Greek Bible JSON** in this data set. "
        "(A Greek New Testament XML exists elsewhere in the scratchpad — "
        "`proiel-greek-nt.xml` — but it is a different corpus format, not "
        "verse-JSON, and out of scope here.) English already had a COCA "
        "codebook and German already had a UD-derived codebook "
        "(`de/out/lexicon.tsv`); this script's actual NEW contribution is "
        "the Czech codebook (`bkr`) plus a second, independently-built "
        "codebook for each of the two German lanes and for English, all in "
        "one directly-comparable shape.",
        "",
        "## Type/token counts",
        "",
        "| lane | language | verses | tokens | types | TTR | hapax | "
        "hapax rate |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for lane in LANES:
        cb = lane_cbs[lane]
        ttr = cb["n_types"] / cb["n_tokens"] if cb["n_tokens"] else 0.0
        hapax_rate = cb["n_hapax"] / cb["n_types"] if cb["n_types"] else 0.0
        lines.append(
            f"| {lane} | {LANE_LANGUAGE[lane]} | {cb['n_verses']} | "
            f"{cb['n_tokens']} | {cb['n_types']} | {ttr:.4f} | "
            f"{cb['n_hapax']} | {hapax_rate:.4f} |"
        )

    lines += [
        "",
        "_TTR (type-token ratio) rises with morphological inflection under "
        "this crude, non-lemmatizing tokenizer: German compounding and "
        "Czech case/gender inflection both mint new surface-form types "
        "that a lemmatizer would collapse. A higher TTR here is a "
        "tokenizer-limitation artifact for German/Czech, not evidence the "
        "text itself is lexically richer than the English lane — see the "
        "caveats section below._",
        "",
        "## Top-20 by raw frequency vs top-20 by dispersion (per lane)",
        "",
        "Frequency and dispersion answer different questions: frequency "
        "asks \"how often\", dispersion (Juilland's D) asks \"how evenly "
        "spread across the 66-ish books\". A word can be very frequent but "
        "clumped (a name repeated many times in one genealogy chapter) or "
        "moderately frequent but perfectly even (a core function word). "
        "The two lists below are restricted to tokens with freq>=10 for "
        "the dispersion side (so hapax-adjacent noise doesn't dominate); "
        "the frequency side has no such floor. Where the two lists mostly "
        "agree (function words dominate both), that itself is the "
        "informative case; where they diverge, the divergence is the "
        "point of building this column at all.",
        "",
    ]
    for lane in LANES:
        cb = lane_cbs[lane]
        top_freq = format_top(cb["rows"], 20)
        top_disp = format_top_dispersion(cb["rows"], min_freq=10, n=20)
        lines.append(f"### {lane} ({LANE_LANGUAGE[lane]})")
        lines.append("")
        lines.append("| rank | top-freq token | freq | | top-dispersion token | D |")
        lines.append("|---:|---|---:|---|---|---:|")
        for i in range(max(len(top_freq), len(top_disp))):
            f_tok, f_val = top_freq[i] if i < len(top_freq) else ("", "")
            d_tok, d_val = top_disp[i] if i < len(top_disp) else ("", "")
            d_str = f"{d_val:.4f}" if d_val != "" else ""
            lines.append(f"| {i + 1} | {f_tok} | {f_val} | | {d_tok} | {d_str} |")
        lines.append("")

    lines += [
        "## Caveats (read before using these codebooks for anything else)",
        "",
        "- **Surface forms, not lemmas.** No lemmatizer was available for "
        "any of the 4 lanes (there is a UD-derived German lemma table "
        "elsewhere in this repo, `de/out/lexicon.tsv`, but this script "
        "deliberately does NOT consult it, to keep all 4 lanes on "
        "identical logic per the Iron Rule). `freq`/`verse_df`/`dispersion` "
        "are all surface-form statistics.",
        "- **Tokenizer is one Unicode letter-run regex "
        "(`[^\\W\\d_]+`) for every lane.** No per-language special-casing. "
        "This means: German compounds are not split (inflates German type "
        "count and hapax rate relative to a lemmatized count); Czech "
        "case/gender/number inflection is not folded (same effect, more "
        "severe — Czech is more synthetic than German); English -s/-ed/-ing "
        "inflections are likewise not folded, though English's lower "
        "inflectional morphology makes this less distorting in practice. "
        "See the TTR table above for the quantitative shape of this effect.",
        "- **`closed_class_guess` is NOT a part-of-speech tag.** It is "
        f"purely `rank <= {CLOSED_CLASS_RANK_CUTOFF} and dispersion >= "
        f"{CLOSED_CLASS_DISPERSION_MIN}`, chosen because function words "
        "tend to be both frequent and evenly spread. It will mislabel any "
        "high-frequency, evenly-spread CONTENT word (e.g. a very common "
        "theological term repeated throughout, like \"God\"/\"Lord\"/"
        "\"Herr\"/\"Pán\") as closed-class, and will miss a genuinely "
        "closed-class word that happens to be rarer or unevenly used in a "
        "particular translation's register. Do not treat this column as "
        "ground truth.",
        "- **Dispersion (Juilland's D) is computed per-lane over that "
        "lane's own book segmentation**, not a shared/aligned book axis "
        "across lanes — a lane's own `book_nr` values are used directly, "
        "so book counts can differ slightly between lanes if a lane's "
        "source JSON segments books differently (e.g. combined vs split "
        "books). This does not affect the meaning of D within a single "
        "lane's own codebook, only cross-lane numeric comparison of D "
        "values (not attempted by this script).",
        "- **Absent =/= zero.** Nothing here compares lanes; a token "
        "missing from one lane's codebook says nothing about another "
        "lane's codebook. Cross-lane presence/absence is `build_rosetta_"
        "probe.py`'s job (its `TextAbsent` census), not this script's.",
        "- **No network, no external dependencies** — stdlib only, so "
        "these numbers are fully reproducible from the same input JSON "
        "files with nothing but a Python 3 interpreter.",
    ]

    report = "\n".join(lines)
    (out_dir / "codebook_summary.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir}/codebook_summary.md")


if __name__ == "__main__":
    main()
