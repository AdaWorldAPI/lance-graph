#!/usr/bin/env python3
"""
DATEV SKR -> CSV (v5): character-column slicing of `pdftotext -layout` output.

Both PDFs use stable column layouts. We slice each text line into N character
ranges (one per logical column), then process each column as an independent
account stream.

Each stream walks rows top-to-bottom:
  - A row whose first non-whitespace token is an account-number pattern
    starts a new account.
  - A row whose first non-whitespace content is at the SAME indent (i.e.
    inside the column range) but does NOT start with a number is a
    continuation of the current account's name.
  - A blank row in this column = section break, flush current account.
"""

import csv
import re
import subprocess
import sys
from pathlib import Path

NUM_4 = re.compile(r"^(?:F\s+|S\s+|R\s+|AV\s+|AM\s+|U\s+F\s+|U\s+S\s+|U\s+AM\s+|"
                   r"S/AV\s+|S/AM\s+|G\s+K\s+)?(\d{4})(?:\s+|$)")
NUM_6 = re.compile(r"^(?:F\s+|S\s+|R\s+|AV\s+|AM\s+|U\s+F\s+|U\s+S\s+|U\s+AM\s+|"
                   r"S/AV\s+|S/AM\s+|G\s+K\s+)?(\d{4})\s+(\d{2})(?:\s+|$)")
RESERVED = re.compile(r"^R\s+(\d{4})\s*$")
RANGE_TAIL = re.compile(r"^-\d+\s*$")

# SKR 04 (Abschlussgliederungsprinzip — balance-sheet-structure):
FAMILY_SKR04 = {
    "0": "Anlagevermögen", "1": "Umlaufvermögen", "2": "Eigenkapital",
    "3": "Fremdkapital", "4": "Erträge", "5": "Materialaufwand",
    "6": "Personalaufwand/sonstige Aufwendungen",
    "7": "Finanzergebnis/Steuern", "8": "frei",
    "9": "Statistische Konten",
}

# SKR 03 (Prozessgliederungsprinzip — process-oriented):
FAMILY_SKR03 = {
    "0": "Anlage- und Kapitalkonten",
    "1": "Finanz- und Privatkonten",
    "2": "Abgrenzungskonten",
    "3": "Wareneingang und Bestandskonten",
    "4": "Betriebliche Aufwendungen",
    "5": "frei",
    "6": "frei",
    "7": "Bestände an Erzeugnissen",
    "8": "Erlöse",
    "9": "Vortrags-, Statistische Konten",
}


def family_skr04(num: str) -> str:
    return FAMILY_SKR04.get(num[0], "Unknown")


def family_skr03(num: str) -> str:
    return FAMILY_SKR03.get(num[0], "Unknown")


def extract_layout_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-layout", str(pdf_path), "-"],
        capture_output=True, text=True, check=True,
    )
    return result.stdout


def slice_columns(text: str, ranges: list[tuple[int, int]]) -> list[list[str]]:
    """For each (start, end) char range, return the list of substrings
    (one per text line). Lines shorter than `start` produce empty strings."""
    lines = text.splitlines()
    out = []
    for start, end in ranges:
        col_lines = []
        for line in lines:
            if len(line) <= start:
                col_lines.append("")
            else:
                col_lines.append(line[start:end].rstrip())
        out.append(col_lines)
    return out


# Section-header tokens that, when they appear ALONE on a line in a column,
# signal a category break and force a flush of the current account. Kept
# conservative — only words that are NEVER part of an account name.
HARD_SECTION_HEADERS = {
    "Anlagevermögen", "Anlagevermögenskonten", "Umlaufvermögen",
    "Umlaufvermögenskonten", "Eigenkapital", "Eigenkapitalkonten",
    "Fremdkapital", "Fremdkapitalkonten", "Sachanlagen", "Finanzanlagen",
    "Vorräte", "Materialaufwand",
}


def walk_skr04_column(col_lines: list[str], accounts: dict[str, str]) -> None:
    cur_num = None
    cur_parts = []
    for raw in col_lines:
        s = raw.strip()
        if not s:
            # Blank slice = this column has no content on this row (could be a
            # right-page-only line). Just skip — do NOT flush. Flush is driven
            # purely by new-account-number-detection and hard section headers.
            continue
        if RESERVED.match(s) or RANGE_TAIL.match(s):
            if cur_num and cur_parts:
                _stash(accounts, cur_num, cur_parts)
            cur_num = None; cur_parts = []
            continue
        # Hard section header alone on a line -> flush.
        if s in HARD_SECTION_HEADERS:
            if cur_num and cur_parts:
                _stash(accounts, cur_num, cur_parts)
            cur_num = None; cur_parts = []
            continue
        m = NUM_4.match(s)
        if m:
            if cur_num and cur_parts:
                _stash(accounts, cur_num, cur_parts)
            cur_num = m.group(1)
            cur_parts = [s[m.end():].strip()]
        else:
            if cur_num is not None:
                cur_parts.append(s)
    if cur_num and cur_parts:
        _stash(accounts, cur_num, cur_parts)


def walk_skr03_column(col_lines: list[str], accounts: dict[str, str]) -> None:
    cur_num = None
    cur_parts = []
    for raw in col_lines:
        s = raw.strip()
        if not s:
            continue
        if RESERVED.match(s) or RANGE_TAIL.match(s):
            if cur_num and cur_parts:
                _stash(accounts, cur_num, cur_parts)
            cur_num = None; cur_parts = []
            continue
        if s in HARD_SECTION_HEADERS:
            if cur_num and cur_parts:
                _stash(accounts, cur_num, cur_parts)
            cur_num = None; cur_parts = []
            continue
        m = NUM_6.match(s)
        if m:
            if cur_num and cur_parts:
                _stash(accounts, cur_num, cur_parts)
            cur_num = m.group(1) + m.group(2)
            cur_parts = [s[m.end():].strip()]
        else:
            if cur_num is not None:
                cur_parts.append(s)
    if cur_num and cur_parts:
        _stash(accounts, cur_num, cur_parts)


def _stash(accounts: dict[str, str], num: str, parts: list[str]) -> None:
    name = " ".join(p.strip() for p in parts if p.strip()).strip()
    name = re.sub(r"\s+", " ", name)
    # Strip trailing markers.
    while True:
        m = re.match(r"^(.*\S)\s+(HB|SB|EÜR)$", name)
        if not m:
            break
        name = m.group(1).strip()
    # Strip footer fragments.
    name = re.sub(r"\s+Art\.-Nr\.\s+\d+.*$", "", name).strip()
    name = re.sub(r"\s+Seite\s+\d+\s*$", "", name).strip()
    # Repair hyphenated line breaks like "Konzessio- nen" -> "Konzessionen".
    # Don't merge if the next word is a separable German function word
    # ("oder", "und", "der", "die", "das") — those are real word boundaries
    # in DATEV's compound notation like "Geschäfts- oder Firmenwert".
    def _maybe_merge(match):
        next_word = match.group(1)
        if next_word in {"oder", "und", "der", "die", "das", "im", "in",
                         "an", "auf", "für", "mit", "zur", "zum"}:
            return "- " + next_word  # keep hyphen
        return next_word  # merge

    name = re.sub(r"-\s+([a-zäöüß]{1,6})(?=\s|$)", _maybe_merge, name)
    if not name:
        return
    # Reject standalone section headers.
    if name in {"Anlagevermögen", "Umlaufvermögen", "Eigenkapital",
                "Fremdkapital", "Sachanlagen", "Finanzanlagen",
                "Vorräte", "Rückstellungen", "Verbindlichkeiten"}:
        return
    if num not in accounts:
        accounts[num] = name


def write_csv(rows, path: Path, source: str):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["account_number", "account_name", "family", "source"])
        for num, name, fam in rows:
            w.writerow([num, name, fam, source])


def main():
    skr04_pdf = Path(
        "/root/.claude/uploads/8b4bdcc8-f92c-424f-911e-928ee9191fb8/edb5ec8e-st26301494027_de.pdf"
    )
    skr03_pdf = Path(
        "/root/.claude/uploads/8b4bdcc8-f92c-424f-911e-928ee9191fb8/4ad3968e-19606_HGB_SKR_03_Bau_und_Handwerk_2026.pdf"
    )
    out_dir = Path("/home/user/lance-graph/data/ontologies/skr-datev")
    out_dir.mkdir(parents=True, exist_ok=True)

    # SKR 04: 4 logical columns. The account-num+name columns sit at char
    # ranges [37, 115) (left page) and [115, 200) (right page). The right
    # boundary of col 1 is conservatively past Posten-col-2 because the
    # only thing in chars 80-115 on a row that ALSO has account content in
    # col 1 is more col-1 name continuation.
    skr04_text = extract_layout_text(skr04_pdf)
    skr04_cols = slice_columns(skr04_text, [(37, 100), (110, 200)])
    accounts04 = {}
    for col in skr04_cols:
        walk_skr04_column(col, accounts04)
    skr04 = sorted(
        [(n, name, family_skr04(n)) for n, name in accounts04.items()],
        key=lambda r: r[0],
    )
    write_csv(skr04, out_dir / "skr04.csv",
              "DATEV SKR 04 (Art.-Nr. 11175, 2023)")
    print(f"SKR 04: {len(skr04)}")

    # SKR 03 Bau has 4 logical columns; account name+number columns are at
    # char ranges [44, 90) and [130, 200).
    skr03_text = extract_layout_text(skr03_pdf)
    skr03_cols = slice_columns(skr03_text, [(44, 90), (130, 200)])
    accounts03 = {}
    for col in skr03_cols:
        walk_skr03_column(col, accounts03)
    bau = sorted(
        [(n, name, family_skr03(n[:4])) for n, name in accounts03.items()],
        key=lambda r: r[0],
    )
    write_csv(bau, out_dir / "skr03-bau.csv",
              "DATEV SKR 03 Bau und Handwerk (Art.-Nr. 19606, 2026)")
    print(f"SKR 03 Bau: {len(bau)}")

    # Canonical SKR 03 = Bau (NN=00) entries.
    canonical = [(n[:4], name, fam) for n, name, fam in bau if n[4:] == "00"]
    write_csv(canonical, out_dir / "skr03.csv",
              "Canonical DATEV SKR 03 (NN=00 subset of Bau variant)")
    print(f"SKR 03 canonical: {len(canonical)}")

    # Quality samples.
    print("\n--- SKR 04 samples ---")
    for r in skr04[:8]: print(f"  {r}")
    print("\n--- Mid-range SKR04 (5000-5050) ---")
    for r in skr04:
        if "5000" <= r[0] <= "5050":
            print(f"  {r}")
    print("\n--- SKR 03 canonical samples ---")
    for r in canonical[:8]: print(f"  {r}")
    print("\n--- SKR 03 Bau extensions (NN != 00) ---")
    extensions = [r for r in bau if r[0][4:] != "00"]
    print(f"  total: {len(extensions)} trade-specific extensions")
    for r in extensions[:5]: print(f"  {r}")

    # Quality stats.
    susp_4 = [r for r in skr04 if len(r[1]) > 100]
    susp_3 = [r for r in canonical if len(r[1]) > 100]
    print(f"\nNames >100 chars (review candidates): SKR04={len(susp_4)}, SKR03={len(susp_3)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
