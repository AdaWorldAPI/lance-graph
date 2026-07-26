#!/usr/bin/env python3
"""D-RCC-1 — lanes-to-singleton CALIBRATOR (rosetta-codebook-convergence-v1).

Research probe over public-domain, verse-keyed Bible lanes (getBible v2 JSON:
kjv, luther1545, elberfelder1905, bkr). The verse address (book_nr, chapter,
verse) is the frozen external key; each translation is a LANE on that row.

What it measures (calibrates — blocks nothing, per the operator correction):
  A. Row/overlap census — the SoA feasibility numbers + TextAbsent census.
  B. Anchor receipts — `swallow` (Schwalbe vs verschlingen; vlaštovka vs
     požírat) and `grape` per verse, with lane text quoted.
  C. Crude extensional split census — PMI co-occurrence alignment en→de:
     which English content words have >=2 strong German associates that
     PARTITION their verse contexts (the German lane splits the English
     polysemy)? Distribution + receipts.

Deliberate crudeness (v1): no lemmatizer, no stopword lists beyond frequency
bounds, Psalms excluded from stats (versification offset — the known
Masoretic/LXX blocker; visible in the anchor receipts instead). This is the
calibrator, not the aligner (D-RCC-3).

Data: gitignored. Fetch (PD texts):
  curl -sL https://api.getbible.net/v2/{kjv,luther1545,elberfelder1905,bkr}.json
Run:  python3 build_rosetta_probe.py <data_dir_with_bible_*.json>
Out:  <data_dir>/out/rosetta_probe_report.md + en_de_splits.tsv
"""

import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

LANES = ["kjv", "luther1545", "elberfelder1905", "bkr"]
PSALMS_NR = 19  # excluded from PMI stats: versification offset (titles as v1)

TOKEN_RE = re.compile(r"[A-Za-zÀ-ÿĀ-žÁ-ůěščřžýáíéúůňťď]+")


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


def main() -> None:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent
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
            "verb": re.compile(
                r"verschling|verschluck|verschlang|schluck|pohlt|požír|sežr|pozř",
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
        for k in sorted(hits):
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
            if name == "swallow" or len(receipts) < 400:
                lines.append("\n".join(row))
        lines.insert(1, f"lane-resolved: bird={n_bird} verb={n_verb} "
                        f"unresolved-by-regex={n_neither}")
        receipts.append("\n".join(lines))

    # ── C. Extensional split census (en → de, PMI) ───────────────────────
    stat_keys = [k for k in common if k[0] != PSALMS_NR]
    en_vf = defaultdict(set)   # english token -> verse keys
    de_vf = defaultdict(set)
    for k in stat_keys:
        for t in set(toks(lanes["kjv"][k])):
            en_vf[t].add(k)
        for t in set(toks(lanes["luther1545"][k])):
            de_vf[t].add(k)
    n_v = len(stat_keys)

    def pmi(a: set, b: set) -> float:
        co = len(a & b)
        if co < 5:
            return -9.0
        return math.log2(co * n_v / (len(a) * len(b)))

    split_rows, split_hist = [], Counter()
    cand = [(w, ks) for w, ks in en_vf.items()
            if 10 <= len(ks) <= 500 and len(w) >= 4]
    de_items = [(w, ks) for w, ks in de_vf.items()
                if 5 <= len(ks) <= 800 and len(w) >= 4]
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
    with (out_dir / "en_de_splits.tsv").open("w", encoding="utf-8") as f:
        f.write("en_word\tverses\tpartitioning_de_associates\n")
        for w, n, a in split_rows:
            f.write(f"{w}\t{n}\t{a}\n")

    # ── report ───────────────────────────────────────────────────────────
    n_cand = len(cand)
    n_split = sum(v for k, v in split_hist.items() if k >= 2)
    report = "\n".join([
        "# D-RCC-1 lanes-to-singleton probe — report (calibrator, v1)",
        "",
        "## A. Row census (frozen verse address as key)",
        f"- union rows: **{len(all_keys)}**, common to all 4 lanes: "
        f"**{len(common)}**",
        "| lane | rows | vs union |", "|---|---|---|", *census,
        "",
        "## B. Anchor receipts", *receipts,
        "",
        "## C. Extensional split census (en→luther1545, PMI, Psalms excluded)",
        f"- candidate English words (freq 10..500, len>=4): **{n_cand}**",
        f"- with >=2 partitioning German associates (the lane SPLITS the "
        f"English word): **{n_split}** ({100 * n_split / max(n_cand, 1):.1f}%)",
        "- partition-count histogram (capped at 5): "
        + ", ".join(f"{k}:{v}" for k, v in sorted(split_hist.items())),
        f"- full list: `en_de_splits.tsv` ({len(split_rows)} rows)",
        "",
        "_Crudeness caveats: no lemmatizer (inflection splits are surface, "
        "not sense); PMI threshold 3.0, cooc>=5, overlap<=0.3 hand-set; "
        "Psalms excluded (versification offset). This calibrates lane "
        "count + routing; it does not adjudicate senses (D-RCC-3/4)._",
    ])
    (out_dir / "rosetta_probe_report.md").write_text(report, encoding="utf-8")
    print(f"wrote {out_dir}/rosetta_probe_report.md "
          f"({len(split_rows)} split rows)")


if __name__ == "__main__":
    main()
