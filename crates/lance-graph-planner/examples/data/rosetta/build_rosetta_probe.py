#!/usr/bin/env python3
"""D-RCC-1 — lanes-to-singleton CALIBRATOR (rosetta-codebook-convergence-v1).

Research probe over public-domain, verse-keyed Bible lanes (getBible v2 JSON:
kjv, luther1545, elberfelder1905, bkr). The verse address (book_nr, chapter,
verse) is the frozen external key; each translation is a LANE on that row.

What it measures (calibrates — blocks nothing, per the operator correction):
  A. Row/overlap census — the SoA feasibility numbers + TextAbsent census.
  B. Anchor receipts — `swallow` (Schwalbe vs verschlingen; vlaštovka vs
     požírat/sehltiti) and `grape` per verse, with lane text quoted.
  C. Crude extensional split census — PMI co-occurrence alignment en→de:
     which English content words have >=2 strong German associates that
     PARTITION their verse contexts (the German lane splits the English
     polysemy)? Computed BOTH on raw German surface forms and on a crude
     suffix-normalised German stem (see § v2 changes) — before/after.

v2 changes (this pass, evidence-driven — see exec-run notes):
  - Anchor B `swallow` verb regex hardened against v1's regex-coverage gap
    (22/50 KJV verses fell through unresolved in v1, 44%). Root cause,
    read off the actual unresolved lane texts: (a) German strong-verb
    ablaut participle "verschlungen" was missing (only "verschling" /
    "verschlang" were present); (b) the v1 Czech alternation "pozř" was a
    diacritic typo — the real aorist/perfect stem is "požř" (ž, not z);
    (c) the whole "hltat/pohltit/sehltit" Czech verb family alternates
    consonant t/c/ť under Czech palatalization (pohltiti -> pohlceni;
    sehltiti -> sehlcen / sehlťme) and only one variant was present.
    Fixed by adding the missing stems, evidenced against the actual
    corpus (see exec-run notes for the false-positive check on broader
    substrings like bare "hlt"/"hlc"/"hlť", which were rejected as
    over-broad — e.g. bare "hlc" collides with "poběhlce" (fugitive),
    unrelated to swallowing).
  - Section C now also runs a CRUDE, DOCUMENTED German suffix
    normaliser (longest-suffix-strip, minimum-stem-length guard) purely
    for the association-counting pass, to fold surface inflections
    (weinberge/weinberges/weinbergen) into one approximate stem before
    counting co-occurrence. This is explicitly NOT a lemmatizer (no
    dictionary, no irregular forms, no compounding awareness) — it is a
    hand-written suffix table, and the report states before/after
    numbers so the effect is visible rather than assumed.
  - New `--anchor WORD` CLI flag: dumps every KJV verse containing WORD
    plus its lane texts (no classification) so a future anchor can be
    scoped from real text before anyone writes a regex for it. Absent,
    behaviour is identical to running with only the built-in anchors.

Deliberate crudeness (still true in v2): no lemmatizer, no stopword lists
beyond frequency bounds, Psalms excluded from stats (versification offset —
the known Masoretic/LXX blocker; visible in the anchor receipts instead).
This is the calibrator, not the aligner (D-RCC-3). The suffix normaliser
above is a stated approximation, not a substitute for real lemmatisation —
see the caveat paragraph at the end of the generated report for the exact
thresholds in force.

Data: gitignored. Fetch (PD texts):
  curl -sL https://api.getbible.net/v2/{kjv,luther1545,elberfelder1905,bkr}.json
Run:  python3 build_rosetta_probe.py <data_dir_with_bible_*.json> [--anchor WORD]
Out:  <data_dir>/out/rosetta_probe_report.md + en_de_splits.tsv
"""

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LANES = ["kjv", "luther1545", "elberfelder1905", "bkr"]
PSALMS_NR = 19  # excluded from PMI stats: versification offset (titles as v1)

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿĀ-žÁ-ůěščřžýáíéúůňťď]+")

# ── v2: crude German suffix normaliser (association counting ONLY) ────────
# Longest-suffix-strip, evidence-picked from common German case/plural/weak-
# verb endings (NOT a lemmatizer: no dictionary, no irregular/strong-verb
# forms, no compound splitting, no umlaut-undo — e.g. it will not fold
# "gut"/"gute"/"guten" onto one stem; it folds "guten"->"gute" only).
# Ordered longest-first so e.g. "-ungen" is tried before "-en"/"-n".
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

    Approximation only — documented in the module docstring and the report
    caveat paragraph. Not lemmatisation: no dictionary lookups, no ablaut/
    umlaut correction, no compound decomposition.
    """
    for suf in DE_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= DE_MIN_STEM_LEN:
            return tok[: -len(suf)]
    return tok


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


def compute_split_census(cand, de_items, n_v):
    """PMI co-occurrence split census: en candidate -> partitioning de associates.

    Shared by the before/after (raw vs suffix-normalised) passes in §C so the
    two runs are guaranteed to use identical thresholds/logic.
    """

    def pmi(a: set, b: set) -> float:
        co = len(a & b)
        if co < 5:
            return -9.0
        return math.log2(co * n_v / (len(a) * len(b)))

    split_rows, split_hist = [], Counter()
    for w, wks in cand:
        assoc = []
        for g, gks in de_items:
            if len(wks & gks) >= 5:
                s = pmi(wks, gks)
                if s >= 3.0:
                    assoc.append((s, g, wks & gks))
        assoc.sort(reverse=True)
        # strong associates that PARTITION w's contexts (low mutual overlap)
        kept = []
        for s, g, cov in assoc:
            if all(len(cov & c2) <= 0.3 * min(len(cov), len(c2))
                   for _, _, c2 in kept):
                kept.append((s, g, cov))
        split_hist[min(len(kept), 5)] += 1
        if len(kept) >= 2:
            split_rows.append(
                (w, len(wks),
                 "; ".join(f"{g}({len(cov)},pmi={s:.1f})"
                           for s, g, cov in kept[:4])))
    split_rows.sort(key=lambda r: -r[1])
    return split_rows, split_hist


def main() -> None:
    ap = argparse.ArgumentParser(
        description="D-RCC-1 lanes-to-singleton probe (v2)")
    ap.add_argument("data_dir", nargs="?", default=None,
                     help="directory containing bible_{kjv,luther1545,"
                          "elberfelder1905,bkr}.json")
    ap.add_argument("--anchor", default=None, metavar="WORD",
                     help="debug aid: dump every KJV verse containing WORD "
                          "plus its raw lane texts (no classification), so a "
                          "future anchor regex can be scoped from real text "
                          "without editing this script. Additive only — "
                          "omitting this flag reproduces the default report.")
    args = ap.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else Path(__file__).parent
    out_dir = data_dir / "out"
    out_dir.mkdir(exist_ok=True)

    lanes = {}
    for lane in LANES:
        p = data_dir / f"bible_{lane}.json"
        if not p.exists():
            sys.exit(f"missing {p} — fetch first (see module docstring)")
        lanes[lane] = load_lane(p)

    # ── A. Row census ────────────────────────────────────────────────────
    keysets = {l: set(r) for l, r in lanes.items()}
    all_keys = set().union(*keysets.values())
    common = set.intersection(*keysets.values())
    census = [f"| {l} | {len(keysets[l])} | {len(all_keys - keysets[l])} absent |"
              for l in LANES]

    # ── B. Anchor receipts ───────────────────────────────────────────────
    anchors = {
        "swallow": {
            "en": re.compile(r"\bswallow(s|ed|eth|ing)?\b", re.I),
            "bird": re.compile(r"schwalbe|vlaštovic|vlaštovk", re.I),
            # v2: added verschlung (ablaut participle of verschlingen),
            # pohlc/sehlt/sehlc/sehlť/nahlt (the pohltit/sehltit Czech
            # consonant-alternating family), and fixed the pozř->požř
            # diacritic typo. See module docstring "v2 changes" for the
            # per-stem evidence (verse + lane text) that motivated each.
            "verb": re.compile(
                r"verschling|verschlung|verschluck|verschlang|schluck|"
                r"pohlt|pohlc|sehlt|sehlc|sehlť|nahlt|"
                r"požír|sežr|požř",
                re.I),
        },
        "grape": {
            "en": re.compile(r"\bgrapes?\b", re.I),
            "bird": re.compile(r"traube|beere|hrozn|hrozen", re.I),
            "verb": re.compile(r"$^"),
        },
    }
    receipts = []
    for name, spec in anchors.items():
        hits = [k for k, t in lanes["kjv"].items() if spec["en"].search(t)]
        n_bird = n_verb = n_neither = 0
        lines = [f"### `{name}` — {len(hits)} KJV verses"]
        for idx, k in enumerate(sorted(hits)):
            row = [f"- **{k[0]}.{k[1]}:{k[2]}** en: “{lanes['kjv'][k]}”"]
            cls = "neither"
            for l in LANES[1:]:
                t = lanes[l].get(k, "(TextAbsent)")
                mark = ("🐦" if spec["bird"].search(t)
                        else "🫗" if spec["verb"].search(t) else "·")
                if mark == "🐦":
                    cls = "bird"
                elif mark == "🫗" and cls != "bird":
                    cls = "verb"
                row.append(f"    - {l} {mark} “{t}”")
            if cls == "bird":
                n_bird += 1
            elif cls == "verb":
                n_verb += 1
            else:
                n_neither += 1
            # Cap the per-anchor verse dump at 400 rows. The original
            # condition (`len(receipts) < 400`) checked the OUTER list --
            # which holds one entry per ANCHOR NAME (appended once, after
            # this loop finishes) -- so it was 0 or 1 for the whole run
            # and the cap never fired; every matching verse was dumped
            # for every anchor. `idx` is the per-anchor verse counter,
            # which is what the cap was actually meant to bound.
            if name == "swallow" or idx < 400:
                lines.append("\n".join(row))
        lines.insert(1, f"lane-resolved: bird={n_bird} verb={n_verb} "
                        f"unresolved-by-regex={n_neither}")
        receipts.append("\n".join(lines))

    # ── B'. Optional --anchor inspection (debug aid, additive only) ──────
    anchor_inspect_section = ""
    if args.anchor:
        word = args.anchor
        word_re = re.compile(r"\b" + re.escape(word) + r"\w*", re.I)
        hits = [k for k, t in lanes["kjv"].items() if word_re.search(t)]
        lines = [f"## Anchor Inspection (debug, --anchor {word!r})",
                 f"- {len(hits)} KJV verses match `\\b{word}\\w*` "
                 f"(unclassified — raw lane text dump for scoping a future "
                 f"anchor regex)"]
        for k in sorted(hits):
            lines.append(f"- **{k[0]}.{k[1]}:{k[2]}** en: “{lanes['kjv'][k]}”")
            for l in LANES[1:]:
                t = lanes[l].get(k, "(TextAbsent)")
                lines.append(f"    - {l} “{t}”")
        anchor_inspect_section = "\n".join(lines)
        print(f"--anchor {word!r}: {len(hits)} KJV verses (see report)")

    # ── C. Extensional split census (en → de, PMI) ───────────────────────
    stat_keys = [k for k in common if k[0] != PSALMS_NR]
    en_vf = defaultdict(set)   # english token -> verse keys
    de_vf_raw = defaultdict(set)   # german surface token -> verse keys
    de_vf_norm = defaultdict(set)  # v2: german normalized stem -> verse keys
    de_surface_terms = set()
    for k in stat_keys:
        for t in set(toks(lanes["kjv"][k])):
            en_vf[t].add(k)
        de_toks = set(toks(lanes["luther1545"][k]))
        for t in de_toks:
            de_surface_terms.add(t)
            de_vf_raw[t].add(k)
            de_vf_norm[normalize_de(t)].add(k)
    n_v = len(stat_keys)

    cand = [(w, ks) for w, ks in en_vf.items()
            if 10 <= len(ks) <= 500 and len(w) >= 4]
    de_items_raw = [(w, ks) for w, ks in de_vf_raw.items()
                     if 5 <= len(ks) <= 800 and len(w) >= 4]
    de_items_norm = [(w, ks) for w, ks in de_vf_norm.items()
                      if 5 <= len(ks) <= 800 and len(w) >= 4]

    split_rows_raw, split_hist_raw = compute_split_census(cand, de_items_raw, n_v)
    split_rows_norm, split_hist_norm = compute_split_census(cand, de_items_norm, n_v)

    n_cand = len(cand)
    n_split_before = sum(v for k, v in split_hist_raw.items() if k >= 2)
    n_split_after = sum(v for k, v in split_hist_norm.items() if k >= 2)
    pct_before = 100 * n_split_before / max(n_cand, 1)
    pct_after = 100 * n_split_after / max(n_cand, 1)

    # normaliser merge stats (over the full de surface vocabulary, not just
    # the frequency-bounded candidate pool, since the folding effect is a
    # property of the normaliser itself)
    norm_stems = {normalize_de(t) for t in de_surface_terms}
    n_merged = len(de_surface_terms) - len(norm_stems)
    # how many stems actually absorbed >=2 distinct surface forms
    stem_groups = defaultdict(set)
    for t in de_surface_terms:
        stem_groups[normalize_de(t)].add(t)
    n_folding_stems = sum(1 for ws in stem_groups.values() if len(ws) > 1)

    # the "after" pass (suffix-normalised) is the improved analysis; its
    # split_rows become the canonical tsv output.
    split_rows = split_rows_norm
    with (out_dir / "en_de_splits.tsv").open("w", encoding="utf-8") as f:
        f.write("en_word\tverses\tpartitioning_de_associates_normalized\n")
        for w, n, a in split_rows:
            f.write(f"{w}\t{n}\t{a}\n")

    # ── report ───────────────────────────────────────────────────────────
    report_sections = [
        "# D-RCC-1 lanes-to-singleton probe — report (calibrator, v2)",
        "",
        "## A. Row census (frozen verse address as key)",
        f"- union rows: **{len(all_keys)}**, common to all 4 lanes: "
        f"**{len(common)}**",
        "| lane | rows | vs union |", "|---|---|---|", *census,
        "",
        "## B. Anchor receipts", *receipts,
    ]
    if anchor_inspect_section:
        report_sections += ["", anchor_inspect_section]
    report_sections += [
        "",
        "## C. Extensional split census (en→luther1545, PMI, Psalms excluded)",
        f"- candidate English words (freq 10..500, len>=4): **{n_cand}**",
        "- **before** suffix normalisation (raw German surface forms): "
        f"**{n_split_before}** ({pct_before:.1f}%) with >=2 partitioning "
        "German associates",
        "- **after** suffix normalisation (crude German stem, see below): "
        f"**{n_split_after}** ({pct_after:.1f}%) with >=2 partitioning "
        "German associates",
        f"- normaliser fold: {len(de_surface_terms)} distinct German surface "
        f"tokens -> {len(norm_stems)} distinct stems "
        f"({n_merged} tokens folded away; {n_folding_stems} stems each "
        "absorbed >=2 surface forms)",
        "- partition-count histogram, AFTER normalisation (capped at 5): "
        + ", ".join(f"{k}:{v}" for k, v in sorted(split_hist_norm.items())),
        "- partition-count histogram, BEFORE normalisation (capped at 5): "
        + ", ".join(f"{k}:{v}" for k, v in sorted(split_hist_raw.items())),
        f"- full list (post-normalisation, canonical output): "
        f"`en_de_splits.tsv` ({len(split_rows)} rows)",
        "",
        "_Crudeness caveats: the German side of §C now runs through a "
        "hand-written suffix-strip table (DE_SUFFIXES, longest-match-first, "
        f"minimum stem length {DE_MIN_STEM_LEN}) that folds simple case/"
        "plural/weak-verb endings (e.g. weinberge/weinberges/weinbergen -> "
        "weinberg) — this is a stated APPROXIMATION, not a lemmatizer: no "
        "dictionary, no strong-verb ablaut correction, no compound "
        "splitting, no umlaut normalisation, and it under-folds short words "
        "(gut/gute/guten collapse to only two stems, not one) by design of "
        "the minimum-stem-length guard. English candidates are NOT "
        "normalised (still surface forms) — only the German association "
        "side is. PMI threshold 3.0, cooc>=5, overlap<=0.3 hand-set, "
        "unchanged from v1; Psalms excluded (versification offset). This "
        "calibrates lane count + routing; it does not adjudicate senses "
        "(D-RCC-3/4). The swallow-anchor verb regex in §B was extended "
        "this pass from real unresolved-verse evidence (see module "
        "docstring); a residual of genuinely divergent translations (the "
        "German/Czech lanes choose an unrelated verb entirely, e.g. "
        "\"in vain\" for \"swallowed up\") is expected and reported, not "
        "forced to zero._",
    ]
    report = "\n".join(report_sections)
    (out_dir / "rosetta_probe_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir}/rosetta_probe_report.md "
          f"({len(split_rows)} split rows; before={n_split_before} "
          f"({pct_before:.1f}%) after={n_split_after} ({pct_after:.1f}%))")


if __name__ == "__main__":
    main()
