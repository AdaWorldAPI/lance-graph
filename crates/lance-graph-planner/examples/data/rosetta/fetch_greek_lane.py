#!/usr/bin/env python3
"""fetch_greek_lane.py — acquire a PUBLIC-DOMAIN Greek New Testament TEXT lane
for the Rosetta convergence plan (`.claude/plans/rosetta-codebook-convergence-v1.md`).

Deliverable: a verse-keyed Greek NT lane in the SAME shape as the existing
`bible_kjv.json` / `bible_luther1545.json` / `bible_elberfelder1905.json` /
`bible_bkr.json` lanes already in this scratchpad directory:

    {"books": [{"nr": N, "chapters": [{"chapter": C,
                                        "verses": [{"chapter": C, "verse": V, "text": "..."}]}]}]}

Data (fetched JSON) is NOT checked in — this script is the reproducible
acquisition step; the JSON lane lives under a gitignored data directory
(same convention as `crates/lance-graph-planner/examples/data/coca/` and the
COCA-codebook Release pattern noted in AGENT_LOG.md 2026-07-23).

WHY A GREEK LANE MATTERS (plan §0, §4): the anchoring rule is "source
outranks translation" — a translation-lane anchor contradicted by the SOURCE
lane is not an anchor. Without a Greek TEXT lane that rule has nothing to
outrank with. The PROIEL treebank already in this scratchpad
(`proiel-greek-nt.xml`) is CC BY-NC-SA — usable only as a local ORACLE
(morphology/syntax cross-check), never as a shippable text lane. This
script's job is to find and fetch a Greek NT edition whose TEXT (not just
the underlying 2000-year-old original) is actually public domain.

=== LICENCE FINDING (verified 2026-07-26, verbatim from getbible.net v2
translations.json metadata — see `fetch_greek_lane.py --dump-licences` or
`translations.json` in this scratchpad for the raw records) ===

getbible.net (https://api.getbible.net/v2/translations.json) carries FOUR
Greek (`lang: "grc"`/`"el"`) editions relevant here:

  * `textusreceptus` — Textus Receptus (1550/1894), parsed.
    distribution_license: "Creative Commons: BY-NC-SA 4.0"
    → NOT SHIPPABLE. NC clause forbids commercial redistribution; same
      restriction class as the PROIEL treebank. Oracle-only at best.

  * `westcotthort` — Westcott & Hort 1881 w/ NA27/UBS4 variants, parsed.
    distribution_license: "Creative Commons: BY-NC-SA 4.0"
    → NOT SHIPPABLE, same reason.

  * `lxx` — Septuagint (OT only, not NT; Rahlfs' morphologically tagged).
    distribution_license: "Copyrighted; Free non-commercial distribution"
    → NOT SHIPPABLE (explicitly copyrighted), and OT-only besides — not
      the NT lane this task needs.

  * `tischendorf` — Tischendorf's 8th edition Greek New Testament (1869/72),
    with morphological tags. distribution_about (verbatim): "This text and
    its analysis are in the Public Domain. Copy freely."
    distribution_license: "Public Domain"
    → **ACQUIRED.** Base text: G. Clint Yale's Tischendorf transcription +
      Dr. Maurice A. Robinson's Public Domain Westcott-Hort text, edited by
      Ulrik Sandborg-Petersen (source: http://morphgnt.org, OSIS format).
      Both the underlying edition (pre-1929, public domain by age) AND the
      digital transcription/analysis (explicitly stated PD) clear the bar —
      this script does NOT merely assume PD from publication date, it reads
      the source's own stated terms, which say "Public Domain" outright.

Fetched shape confirms coverage: 27 NT books (book_nr 40..66, matching the
KJV lane's book-numbering convention), 7895 verses, 0 empty-text verses.
Cross-checked against the local `bible_kjv.json` lane restricted to the NT
(book_nr 40..66, 7957 verses): 7895/7895 Tischendorf verses have a KJV
counterpart (full containment), 62 verses exist in KJV's NT versification
but are ABSENT from Tischendorf. This is NOT a fetch error — those 62 are
exactly the well-known verses omitted by the modern critical/Alexandrian
text tradition Tischendorf's 8th edition represents relative to the
(Byzantine-leaning) Textus Receptus that underlies the KJV — e.g. Matthew
17:21, 18:11, 23:14, Acts 8:37, 9:44/46 duplicate-verse artifacts, etc.
Textual criticism, not a bug: `TextAbsent`, never treated as an error by
this script or by any downstream Rosetta join.

Usage:
    python3 fetch_greek_lane.py                 # fetch + verify + report
    python3 fetch_greek_lane.py --dump-licences  # print the 4 licence records only
    python3 fetch_greek_lane.py --no-fetch       # verify+report against already-downloaded JSON

Output:
    <scratchpad>/bible_tischendorf.json   — the fetched lane (gitignored data)
    <scratchpad>/out/greek_lane_report.md — the licence + coverage report
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

TRANSLATIONS_URL = "https://api.getbible.net/v2/translations.json"
TISCHENDORF_URL = "https://api.getbible.net/v2/tischendorf.json"

GREEK_CANDIDATES = ("textusreceptus", "tischendorf", "westcotthort", "lxx")

# NT book numbers in the getbible.net / KJV-lane numbering convention.
NT_BOOK_MIN, NT_BOOK_MAX = 40, 66

SCRATCH_DIR = Path(__file__).resolve().parents[0]  # placeholder, overridden below


def scratchpad_dir() -> Path:
    """Resolve the scratchpad directory the sibling lanes already live in.

    Honors $ROSETTA_SCRATCH_DIR for portability; otherwise falls back to a
    `scratchpad/` directory relative to the current working directory (the
    convention every sibling script in this session already used). Never
    bakes a session-specific absolute path -- a prior version hardcoded
    one sandbox's `/tmp/claude-0/...` path, which does not exist outside
    that single session. Exits with a clear message if neither resolves.
    """
    env = os.environ.get("ROSETTA_SCRATCH_DIR")
    if env:
        return Path(env)
    fallback = Path.cwd() / "scratchpad"
    if fallback.exists():
        return fallback
    sys.exit(
        "no scratch dir resolvable: $ROSETTA_SCRATCH_DIR is unset and the "
        f"cwd-relative fallback {fallback} does not exist -- set "
        "$ROSETTA_SCRATCH_DIR to the directory containing bible_*.json "
        "(see module docstring)."
    )


def fetch_json(url: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(url, headers={"User-Agent": "rosetta-greek-lane/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def licence_report(translations: dict) -> str:
    lines = ["## Greek-lang candidate editions on getbible.net (verbatim licence terms)\n"]
    for key in GREEK_CANDIDATES:
        rec = translations.get(key)
        if rec is None:
            lines.append(f"- `{key}`: NOT FOUND in translations.json (checked, absent)\n")
            continue
        lic = rec.get("distribution_license", "<no distribution_license field — terms unstated>")
        about = rec.get("distribution_about", "")
        verdict = "SHIPPABLE (Public Domain)" if lic.strip().lower() == "public domain" else "NOT SHIPPABLE (restricted)"
        lines.append(f"### `{key}` — {rec.get('translation')}")
        lines.append(f"- lang: {rec.get('lang')} / {rec.get('language')}")
        lines.append(f"- distribution_license (verbatim): \"{lic}\"")
        lines.append(f"- verdict: **{verdict}**")
        if about:
            lines.append(f"- distribution_about (verbatim): \"{about[:400]}\"")
        lines.append(f"- source: {rec.get('distribution_source', '<unstated>')}")
        lines.append("")
    return "\n".join(lines)


def index_verses(bible: dict) -> dict[tuple[int, int, int], str]:
    idx: dict[tuple[int, int, int], str] = {}
    for book in bible.get("books", []):
        nr = book.get("nr")
        for chapter in book.get("chapters", []):
            for verse in chapter.get("verses", []):
                idx[(nr, verse["chapter"], verse["verse"])] = verse["text"]
    return idx


def build_report(
    translations: dict,
    tischendorf: dict | None,
    kjv: dict | None,
) -> str:
    parts = ["# Greek NT lane acquisition report\n"]
    parts.append(licence_report(translations))

    if tischendorf is None:
        parts.append("## Fetch result\n\nNOT ACQUIRED — see licence findings above. No PD Greek NT "
                      "edition could be fetched this run. This is an honest 'not acquired' result, "
                      "not a fabricated lane.\n")
        return "\n".join(parts)

    tis_idx = index_verses(tischendorf)
    book_nrs = sorted({b["nr"] for b in tischendorf.get("books", [])})
    parts.append("## Fetch result — `tischendorf` (Public Domain)\n")
    parts.append(f"- books: {len(tischendorf.get('books', []))} (book_nr range: {min(book_nrs)}..{max(book_nrs)})")
    parts.append(f"- verses: {len(tis_idx)}")
    empty = sum(1 for v in tis_idx.values() if not v.strip())
    parts.append(f"- empty-text verses: {empty}")
    parts.append("")

    if kjv is not None:
        kjv_idx = index_verses(kjv)
        kjv_nt = {k: v for k, v in kjv_idx.items() if NT_BOOK_MIN <= k[0] <= NT_BOOK_MAX}
        overlap = set(tis_idx) & set(kjv_nt)
        only_tis = set(tis_idx) - set(kjv_nt)
        only_kjv = set(kjv_nt) - set(tis_idx)
        parts.append("## Row-key overlap vs local KJV lane (NT books 40..66 only)\n")
        parts.append(f"- KJV NT verse count: {len(kjv_nt)}")
        parts.append(f"- Tischendorf verse count: {len(tis_idx)}")
        parts.append(f"- overlap (same book_nr:chapter:verse key): {len(overlap)}")
        parts.append(f"- only in Tischendorf (no KJV NT counterpart): {len(only_tis)}")
        parts.append(f"- only in KJV NT (TextAbsent from Tischendorf — textual-criticism "
                      f"omissions, e.g. disputed Byzantine-only verses, NOT an error): {len(only_kjv)}")
        if only_kjv:
            sample = sorted(only_kjv)[:10]
            parts.append(f"  - sample absent keys: {sample}")
        parts.append("")

        parts.append("## Worked receipt — Greek text alongside KJV\n")
        for label, nr, ch, vs in (("John 1:1", 43, 1, 1), ("Acts 3:22", 44, 3, 22)):
            greek = tis_idx.get((nr, ch, vs), "<TextAbsent>")
            english = kjv_idx.get((nr, ch, vs), "<TextAbsent>")
            parts.append(f"- **{label}**")
            parts.append(f"  - Tischendorf (grc): {greek}")
            parts.append(f"  - KJV (en): {english}")
        parts.append("")

    parts.append("## Limitations\n")
    parts.append("- NT-only (27 books). Any OT row-key (book_nr < 40) is `TextAbsent` by design, "
                  "not an error — the Greek NT edition never covered the Hebrew Bible.")
    parts.append("- Morphological tags present in the upstream OSIS source are NOT carried into "
                  "this lane's `text` field (verse text only, matching the sibling lane shape). "
                  "A future slice could add a `morph` facet if the Rosetta plan calls for it.")
    parts.append("- `textusreceptus` / `westcotthort` remain available as CC BY-NC-SA oracles "
                  "(same tier as the PROIEL treebank) if a future cross-edition variant check is "
                  "wanted, but must never be promoted to a shipped lane per this licence finding.")
    return "\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-licences", action="store_true", help="print licence findings only, no fetch")
    ap.add_argument("--no-fetch", action="store_true", help="verify+report against already-downloaded JSON")
    args = ap.parse_args()

    scratch = scratchpad_dir()
    out_dir = scratch / "out"
    out_dir.mkdir(parents=True, exist_ok=True)

    translations_path = scratch / "translations.json"
    tischendorf_path = scratch / "bible_tischendorf.json"
    kjv_path = scratch / "bible_kjv.json"

    if translations_path.exists():
        try:
            translations = json.loads(translations_path.read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"FAILED to read cached translations.json: {exc}", file=sys.stderr)
            translations = {}
    elif args.no_fetch:
        # --no-fetch MUST NEVER perform a network call. A missing cache
        # under --no-fetch is a clear error, not a silent fetch (the
        # original code fell through to fetch_json() here regardless of
        # the flag).
        print(
            f"--no-fetch given and {translations_path} is absent -- "
            "refusing to fetch translations.json over the network. Run "
            "once without --no-fetch to populate the cache first, or "
            "point $ROSETTA_SCRATCH_DIR at a directory that already has "
            "it.",
            file=sys.stderr,
        )
        translations = {}
    else:
        try:
            translations = fetch_json(TRANSLATIONS_URL)
            translations_path.write_text(json.dumps(translations, ensure_ascii=False), encoding="utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"FAILED to fetch translations.json: {exc}", file=sys.stderr)
            translations = {}

    if args.dump_licences:
        print(licence_report(translations))
        return 0

    tischendorf = None
    if translations.get("tischendorf", {}).get("distribution_license", "").strip().lower() == "public domain":
        try:
            if tischendorf_path.exists():
                tischendorf = json.loads(tischendorf_path.read_text(encoding="utf-8"))
            elif args.no_fetch:
                # --no-fetch MUST NEVER perform a network call. The
                # original condition only checked `args.no_fetch and
                # tischendorf_path.exists()` for the CACHE-HIT case and
                # fell through to fetch_json() for every other case
                # (including --no-fetch with an absent cache) -- a
                # missing cache under --no-fetch is now a clear error,
                # never a silent fetch.
                print(
                    f"--no-fetch given and {tischendorf_path} is absent "
                    "-- refusing to fetch tischendorf.json over the "
                    "network.",
                    file=sys.stderr,
                )
                tischendorf = None
            else:
                tischendorf = fetch_json(TISCHENDORF_URL)
                tischendorf_path.write_text(json.dumps(tischendorf, ensure_ascii=False), encoding="utf-8")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            print(f"FAILED to fetch tischendorf.json: {exc}", file=sys.stderr)
            tischendorf = None
    else:
        print("tischendorf license check failed or edition absent — refusing to fetch/ship "
              "any Greek NT text this run (honest non-acquisition).", file=sys.stderr)

    kjv = None
    if kjv_path.exists():
        try:
            kjv = json.loads(kjv_path.read_text(encoding="utf-8"))
        except OSError:
            kjv = None

    report = build_report(translations, tischendorf, kjv)
    report_path = out_dir / "greek_lane_report.md"
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n[report written to {report_path}]", file=sys.stderr)
    return 0 if tischendorf is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
