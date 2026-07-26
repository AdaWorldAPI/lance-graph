#!/usr/bin/env python3
"""D-RCC-5 rail rebuild — a polysemy-complete, sense-correct WordNet 3.1
is-a rail, replacing the committed `wordnet31_isa.tsv`
(rosetta-codebook-convergence-v1).

WHY THIS EXISTS (see `.claude/board/EPIPHANIES.md`
`E-WORDNET-RAIL-KEEPFIRST-IS-ALSO-WRONG-SENSE-1`, verified on the main
thread against WNDB ground truth): the committed rail has TWO stacked
defects, not one.

  1. It is keep-first: exactly one row per (word, pos), so polysemy is
     unreachable (verified empirically in `tier_delta.py`'s capability
     audit — max rows sharing a (word,pos) key is 1).
  2. Its "first-sense hypernym per lemma" claim is FALSE. Audited 2/2:
     `grape` -> `shot` is the hypernym of SENSE 3 (03458491 "grapeshot"),
     not sense 1 (07774656, the fruit, true hypernym "edible_fruit").
     `swallow` -> `consumption` is the hypernym of SENSE 2 (00841439,
     "the act of swallowing"), not sense 1 (07594841, "a small amount
     of liquid food (sup)", true hypernym "taste"). The bird sense
     (01597013, swallow/n sense 3) is unreachable from the old rail at
     any depth.

So fixing polysemy alone would still inherit a wrong default sense, and
fixing sense-selection alone would still leave the polysemy hole. This
generator fixes both by emitting EVERY (word, pos, sense) hypernym edge,
with sense numbers taken directly from WordNet's own sense-ranked order
(the `index.<pos>` file's synset-offset list — see `_load_index` below;
WordNet's own docs state this list is frequency-of-use ranked, so
"sense 1" here means exactly what a human WordNet lookup means by
"sense 1", not an artifact of file order).

Scope: nouns + verbs only (`n`, `v`), matching the committed rail's own
scope (its `pos` column is exclusively n/v — verified empirically).
Adjectives/adverbs don't carry the same `@` hypernym taxonomy in WordNet
(they use `&` similar-to instead) and are out of scope here, same as v1.

Data: gitignored (same convention as `wordnet31_isa.tsv` and the
`rosetta/` probes in this directory) — this generator is committed, the
WNDB dict + emitted TSVs are not.

Requires the WordNet 3.1 WNDB "dict" database (index.noun/data.noun +
index.verb/data.verb, classic Princeton lexicographer-file format) at
a directory named by $WNDB_DIR, or (session-local convenience, NOT
relied upon for reproducibility) /tmp/wn/dict if present. See
`fetch_wordnet.sh` in this directory for acquisition. Stdlib only, no
network access from this script itself.

Usage:
  python3 build_wordnet_rail.py                 # auto-detect WNDB_DIR / /tmp/wn/dict
  WNDB_DIR=/path/to/dict python3 build_wordnet_rail.py
  python3 build_wordnet_rail.py --verify         # re-check the two named
                                                  # anchors (grape, swallow)
                                                  # and print PASS/FAIL,
                                                  # nothing else written
  python3 build_wordnet_rail.py --sample N       # cap the diff audit to
                                                  # the first N (word,pos)
                                                  # keys of the v1 TSV,
                                                  # for a quick smoke run;
                                                  # omitted/0 = full audit
                                                  # (all 129,063 keys)

Out (in ./out/, relative to this file):
  wordnet31_isa_v2.tsv    — the corrected, all-senses rail
  wordnet_rail_diff.md    — the v1-vs-v2 audit (the headline error rate)
"""

from __future__ import annotations

import os
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
V1_TSV_PATH = HERE / "wordnet31_isa.tsv"
OUT_DIR = HERE / "out"
V2_TSV_PATH = OUT_DIR / "wordnet31_isa_v2.tsv"
DIFF_MD_PATH = OUT_DIR / "wordnet_rail_diff.md"

POS_LIST = ["n", "v"]
POS_DATA_FILES = {"n": "data.noun", "v": "data.verb"}
POS_INDEX_FILES = {"n": "index.noun", "v": "index.verb"}

HYPERNYM_ISA = "@"
HYPERNYM_INST = "@i"
HYPERNYM_SYMS = {HYPERNYM_ISA: "isa", HYPERNYM_INST: "inst"}

# The two named receipts from the epiphany, re-checked by --verify.
VERIFY_ANCHORS = {
    "grape": {
        "pos": "n",
        "v1_hypernym": "shot",
        "expected_true_sense1_offset": "07774656",
        "expected_true_sense1_hypernym": "edible_fruit",
    },
    "swallow": {
        "pos": "n",
        "v1_hypernym": "consumption",
        "expected_true_sense1_offset": "07594841",
        "expected_true_sense1_hypernym": "taste",
    },
}


def find_wndb_dir() -> Path | None:
    candidates = []
    env = os.environ.get("WNDB_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(HERE / "wndb")  # if ever vendored locally
    candidates.append(Path("/tmp/wn/dict"))  # session-local convenience only
    for c in candidates:
        if c.is_dir() and (c / "data.noun").exists() and (c / "index.noun").exists():
            return c
    return None


@dataclass
class Synset:
    offset: str
    pos: str
    words: list
    # list of (kind_sym, target_offset, target_pos) for @ / @i pointers only,
    # in the ORDER they appear in the data file (a synset may carry more
    # than one hypernym pointer — rare but real; we keep all of them).
    hypernyms: list
    gloss: str


class WordNetDb:
    """Loads WNDB data.<pos>/index.<pos> for pos in POS_LIST."""

    def __init__(self, wndb_dir: Path):
        self.wndb_dir = wndb_dir
        self.synsets: dict = {}  # (offset,pos) -> Synset
        self.lemma_senses: dict = {}  # (lemma,pos) -> [ (offset,pos), ... ] sense order
        for pos in POS_LIST:
            self._load_data(wndb_dir / POS_DATA_FILES[pos], pos)
        for pos in POS_LIST:
            self._load_index(wndb_dir / POS_INDEX_FILES[pos], pos)

    def _load_data(self, path: Path, pos: str) -> None:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("  ") or not line.strip():
                    continue  # license header padding lines
                if " | " in line:
                    body, gloss = line.split(" | ", 1)
                else:
                    body, gloss = line, ""
                toks = body.split()
                if len(toks) < 4:
                    continue
                offset = toks[0]
                # toks[1] = lex_filenum, toks[2] = ss_type — not needed here
                w_cnt = int(toks[3], 16)
                idx = 4
                words = []
                for _ in range(w_cnt):
                    words.append(toks[idx])
                    idx += 2  # word, lex_id
                p_cnt = int(toks[idx])
                idx += 1
                hypernyms = []
                for _ in range(p_cnt):
                    sym = toks[idx]
                    target_offset = toks[idx + 1]
                    target_pos = toks[idx + 2]
                    # idx+3 is the source/target word-number hex field
                    # (0000 = whole-synset pointer); not needed for a
                    # synset-level representative-word rail.
                    idx += 4
                    if sym in HYPERNYM_SYMS:
                        hypernyms.append((sym, target_offset, target_pos))
                self.synsets[(offset, pos)] = Synset(
                    offset=offset, pos=pos, words=words,
                    hypernyms=hypernyms, gloss=gloss.strip(),
                )

    def _load_index(self, path: Path, pos: str) -> None:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("  ") or not line.strip():
                    continue
                toks = line.split()
                lemma = toks[0]
                synset_cnt = int(toks[2])
                p_cnt = int(toks[3])
                idx = 4 + p_cnt  # skip ptr_symbols
                idx += 2  # sense_cnt, tagsense_cnt
                # The offsets list here IS the sense-ranked order: WordNet's
                # own db format docs (wndb(5wn)) state index.<pos> lists a
                # lemma's synsets "in the order corresponding to the sense
                # numbers" — i.e. index position == WordNet sense number.
                # We preserve that order verbatim as sense_num (1-based).
                offsets = toks[idx: idx + synset_cnt]
                self.lemma_senses[(lemma, pos)] = [(o, pos) for o in offsets]

    def hypernym_word(self, target_offset: str, target_pos: str) -> str:
        syn = self.synsets.get((target_offset, target_pos))
        if syn is None or not syn.words:
            return ""
        return syn.words[0]


@dataclass
class RailRow:
    word: str
    pos: str
    sense_num: int
    synset_offset: str
    kind: str  # isa | inst | root
    hypernym_word: str
    hypernym_offset: str


def build_rail(db: WordNetDb) -> list:
    rows: list = []
    for pos in POS_LIST:
        # Iterate lemmas in the order the index file gave them (stable,
        # reproducible — Python dict preserves insertion order).
        for (lemma, lpos), senses in db.lemma_senses.items():
            if lpos != pos:
                continue
            for sense_num, synset_id in enumerate(senses, start=1):
                syn = db.synsets.get(synset_id)
                if syn is None:
                    continue
                if not syn.hypernyms:
                    rows.append(RailRow(
                        word=lemma, pos=pos, sense_num=sense_num,
                        synset_offset=syn.offset, kind="root",
                        hypernym_word="", hypernym_offset="",
                    ))
                    continue
                for sym, target_offset, target_pos in syn.hypernyms:
                    rows.append(RailRow(
                        word=lemma, pos=pos, sense_num=sense_num,
                        synset_offset=syn.offset,
                        kind=HYPERNYM_SYMS[sym],
                        hypernym_word=db.hypernym_word(target_offset, target_pos),
                        hypernym_offset=target_offset,
                    ))
    return rows


def write_rail_tsv(rows: list, path: Path) -> None:
    with path.open("w", encoding="utf-8") as fh:
        fh.write(
            "# Open/Princeton WordNet 3.1 -- ALL senses, ALL hypernym edges "
            "(v2; supersedes wordnet31_isa.tsv's keep-first + wrong-sense "
            "extraction, see E-WORDNET-RAIL-KEEPFIRST-IS-ALSO-WRONG-SENSE-1).\n"
        )
        fh.write(
            "# columns: word\\tpos\\tsense_num\\tsynset_offset\\tkind\\t"
            "hypernym_word\\thypernym_offset\n"
        )
        fh.write(
            "# sense_num: 1-based, in WordNet's own sense-ranked order "
            "(index.<pos> synset-offset list order -- wndb(5wn): this list "
            "is in sense-number order for the lemma).\n"
        )
        fh.write(
            "# kind: isa (hypernym @) | inst (instance_hypernym @i) | root "
            "(this sense has no hypernym pointer at all -- a unique "
            "beginner / top of its hierarchy; hypernym_word/offset empty).\n"
        )
        fh.write(
            "# A synset may carry more than one hypernym pointer (rare "
            "multiple inheritance) -- each becomes its own row, so a "
            "(word,pos,sense_num) key is NOT guaranteed unique here.\n"
        )
        for r in rows:
            fh.write(
                f"{r.word}\t{r.pos}\t{r.sense_num}\t{r.synset_offset}\t"
                f"{r.kind}\t{r.hypernym_word}\t{r.hypernym_offset}\n"
            )


def load_v1_tsv(path: Path) -> list:
    """Returns list of (word, pos, kind, hypernym_word) rows, comments skipped."""
    out = []
    if not path.exists():
        return out
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            out.append((parts[0], parts[1], parts[2], parts[3]))
    return out


def true_sense1_hypernym(db: WordNetDb, word: str, pos: str):
    """Returns (status, sense1_offset, first_hypernym_word_or_None).

    status in {"ABSENT", "ROOT", "OK"}. ROOT = sense 1 exists but has no
    hypernym pointer at all (distinct from ABSENT -- lemma simply not in
    WNDB for this pos).
    """
    senses = db.lemma_senses.get((word, pos))
    if not senses:
        return "ABSENT", None, None
    sense1_offset, sense1_pos = senses[0]
    syn = db.synsets.get((sense1_offset, sense1_pos))
    if syn is None or not syn.hypernyms:
        return "ROOT", sense1_offset, None
    sym, target_offset, target_pos = syn.hypernyms[0]
    return "OK", sense1_offset, db.hypernym_word(target_offset, target_pos)


def which_sense_has_hypernym(db: WordNetDb, word: str, pos: str, hyp_word: str):
    """Scan every sense of (word,pos) and return the 1-based sense_num of
    the FIRST sense whose FIRST hypernym pointer resolves to hyp_word, or
    None if no sense matches. Used to show which sense the old (buggy)
    rail's hypernym string actually belongs to.
    """
    senses = db.lemma_senses.get((word, pos), [])
    for i, synset_id in enumerate(senses, start=1):
        syn = db.synsets.get(synset_id)
        if syn is None or not syn.hypernyms:
            continue
        sym, target_offset, target_pos = syn.hypernyms[0]
        if db.hypernym_word(target_offset, target_pos) == hyp_word:
            return i
    return None


def run_verify(db: WordNetDb) -> int:
    """Re-checks the two named anchors. Returns process exit code
    (0 = both PASS, 1 = at least one FAIL)."""
    all_pass = True
    print("=== --verify: re-checking named anchors ===")
    for word, expect in VERIFY_ANCHORS.items():
        pos = expect["pos"]
        status, sense1_offset, true_hyp = true_sense1_hypernym(db, word, pos)
        checks = []

        c1 = status == "OK" and sense1_offset == expect["expected_true_sense1_offset"]
        checks.append((
            f"sense-1 offset == {expect['expected_true_sense1_offset']!r}", c1,
            f"got status={status} offset={sense1_offset!r}",
        ))

        c2 = true_hyp == expect["expected_true_sense1_hypernym"]
        checks.append((
            f"sense-1 true hypernym == {expect['expected_true_sense1_hypernym']!r}",
            c2, f"got {true_hyp!r}",
        ))

        c3 = true_hyp != expect["v1_hypernym"]
        checks.append((
            f"sense-1 true hypernym != v1's recorded {expect['v1_hypernym']!r} "
            "(i.e. v1 is confirmed wrong)",
            c3, f"true={true_hyp!r} v1={expect['v1_hypernym']!r}",
        ))

        wrong_sense = which_sense_has_hypernym(db, word, pos, expect["v1_hypernym"])
        c4 = wrong_sense is not None and wrong_sense > 1
        checks.append((
            f"v1's hypernym {expect['v1_hypernym']!r} belongs to some sense > 1",
            c4, f"found at sense_num={wrong_sense}",
        ))

        word_pass = all(ok for _, ok, _ in checks)
        all_pass = all_pass and word_pass
        print(f"\n-- {word}/{pos} --  overall: {'PASS' if word_pass else 'FAIL'}")
        for desc, ok, detail in checks:
            print(f"  [{'PASS' if ok else 'FAIL'}] {desc} ({detail})")

    print(f"\n=== --verify result: {'PASS' if all_pass else 'FAIL'} ===")
    return 0 if all_pass else 1


def run_diff(db: WordNetDb, sample: int) -> None:
    v1_rows = load_v1_tsv(V1_TSV_PATH)
    if sample and sample > 0:
        v1_rows = v1_rows[:sample]
        sample_note = (
            f"**SAMPLE run**: first {sample} of {sum(1 for _ in load_v1_tsv(V1_TSV_PATH))} "
            "v1 rows only (deterministic prefix of the file, not random). "
            "Use `--sample 0` (or omit `--sample`) for the full audit."
        )
    else:
        sample_note = (
            f"**FULL audit**: all {len(v1_rows)} rows of the committed "
            "v1 TSV, no sampling."
        )

    total = 0
    absent = 0
    root_in_v1 = 0  # v1 recorded a hypernym but true sense-1 has none
    correct = 0
    wrong = 0
    case_only_diff = 0  # wrong by exact string, but identical case-folded
    wrong_examples = []
    by_pos_wrong = Counter()
    by_pos_total = Counter()

    for word, pos, kind, v1_hyp in v1_rows:
        total += 1
        by_pos_total[pos] += 1
        status, sense1_offset, true_hyp = true_sense1_hypernym(db, word, pos)
        if status == "ABSENT":
            absent += 1
            continue
        if status == "ROOT":
            root_in_v1 += 1
            wrong += 1
            by_pos_wrong[pos] += 1
            if len(wrong_examples) < 10:
                wrong_examples.append((
                    word, pos, v1_hyp, "ROOT (sense 1 has no hypernym at all)",
                    None,
                ))
            continue
        if true_hyp == v1_hyp:
            correct += 1
        else:
            wrong += 1
            by_pos_wrong[pos] += 1
            if true_hyp.lower() == v1_hyp.lower():
                case_only_diff += 1
            if len(wrong_examples) < 10:
                which = which_sense_has_hypernym(db, word, pos, v1_hyp)
                wrong_examples.append((word, pos, v1_hyp, true_hyp, which))

    comparable = total - absent
    error_rate = (wrong / comparable * 100.0) if comparable else 0.0
    real_wrong = wrong - case_only_diff
    real_error_rate = (real_wrong / comparable * 100.0) if comparable else 0.0

    lines = []
    lines.append("# WordNet rail v1-vs-v2 audit\n")
    lines.append(
        "Generated by `build_wordnet_rail.py`. Companion to "
        "`E-WORDNET-RAIL-KEEPFIRST-IS-ALSO-WRONG-SENSE-1` in "
        "`.claude/board/EPIPHANIES.md` -- converts the '2/2 audited anchors "
        "wrong' finding into a MEASURED error rate over the full committed "
        "rail.\n"
    )
    lines.append(f"\n{sample_note}\n")
    lines.append("\n## Headline\n")
    lines.append(f"- v1 rows examined: **{total}**")
    lines.append(f"- absent from WNDB (lemma+pos not indexed): **{absent}** (excluded from the rate below -- absent != wrong)")
    lines.append(f"- comparable rows (present in WNDB): **{comparable}**")
    lines.append(f"- v1 hypernym matches true sense-1 hypernym: **{correct}**")
    lines.append(f"- v1 hypernym is WRONG (mismatched sense, or sense-1 is actually root): **{wrong}**")
    lines.append(f"  - of which sense-1 is actually ROOT (no hypernym at all, so ANY v1 hypernym is fabricated): **{root_in_v1}**")
    lines.append(f"  - of which the mismatch is CASE-FOLDING ONLY (e.g. `v-day` vs `V-day` -- same lemma, capitalization artifact, not a sense-selection bug): **{case_only_diff}**")
    lines.append(f"\n**v1 error rate (exact-string match): {error_rate:.2f}% of comparable rows ({wrong}/{comparable}).**")
    lines.append(f"\n**v1 error rate (case-insensitive, i.e. real sense-selection errors only): {real_error_rate:.2f}% of comparable rows ({real_wrong}/{comparable}).**\n")
    lines.append(
        "\nBoth numbers are reported because case-folding artifacts (proper "
        "nouns like `V-day`) are a real but DIFFERENT defect from sense "
        "misattribution -- collapsing them into one number would either "
        "overstate the sense-selection bug or hide the casing issue. The "
        "case-insensitive figure is the more honest headline for \"is the "
        "extractor picking the wrong SENSE\"; the exact-string figure is "
        "the more honest headline for \"does this rail need post-processing "
        "before exact-match lookups against it are safe.\"\n"
    )
    lines.append("\n### By POS\n")
    lines.append("| pos | total | wrong | error rate |")
    lines.append("|---|---|---|---|")
    for pos in POS_LIST:
        t = by_pos_total.get(pos, 0)
        w = by_pos_wrong.get(pos, 0)
        rate = (w / t * 100.0) if t else 0.0
        lines.append(f"| {pos} | {t} | {w} | {rate:.2f}% |")

    lines.append("\n## Named receipts\n")
    for word, expect in VERIFY_ANCHORS.items():
        pos = expect["pos"]
        status, sense1_offset, true_hyp = true_sense1_hypernym(db, word, pos)
        which = which_sense_has_hypernym(db, word, pos, expect["v1_hypernym"])
        lines.append(
            f"- `{word}/{pos}`: v1 recorded hypernym `{expect['v1_hypernym']}` "
            f"(that string is actually the hypernym of sense {which}); true "
            f"sense-1 offset is `{sense1_offset}` with true hypernym "
            f"`{true_hyp}`."
        )

    lines.append("\n## 10 worked examples (v1 wrong, first 10 found)\n")
    lines.append("| word | pos | v1 hypernym | true sense-1 hypernym | v1's string belongs to sense # |")
    lines.append("|---|---|---|---|---|")
    for word, pos, v1_hyp, true_hyp, which in wrong_examples:
        lines.append(f"| {word} | {pos} | {v1_hyp} | {true_hyp} | {which if which is not None else '-'} |")

    lines.append(
        "\n## Notes\n"
        "- \"Absent\" (lemma+pos not in WNDB), \"root\" (sense 1 has no "
        "hypernym), and \"measured mismatch\" are kept distinct throughout "
        "-- absence is not the same as wrongness, and a v1 hypernym string "
        "attached to a rootless sense-1 is not merely mis-ranked, it is "
        "fabricated (there is no true hypernym to compare against).\n"
        "- \"v1's string belongs to sense #\" is found by scanning every "
        "sense of the lemma for a FIRST-hypernym match on the exact string "
        "v1 recorded; `-` means no sense of this lemma has that hypernym at "
        "all (v1's value doesn't correspond to ANY sense of this word in "
        "current WNDB -- a stronger bug than mere sense-misattribution).\n"
    )

    DIFF_MD_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"v1 error rate: {error_rate:.2f}% ({wrong}/{comparable} comparable rows wrong)")
    print(f"Wrote diff report to {DIFF_MD_PATH}")


def main() -> int:
    verify_mode = "--verify" in sys.argv
    sample = 0
    if "--sample" in sys.argv:
        i = sys.argv.index("--sample")
        if i + 1 < len(sys.argv):
            sample = int(sys.argv[i + 1])

    wndb_dir = find_wndb_dir()
    if wndb_dir is None:
        print(
            "ERROR: no WNDB dict directory found (checked $WNDB_DIR, "
            f"{HERE / 'wndb'}, /tmp/wn/dict). Run fetch_wordnet.sh first, "
            "or set WNDB_DIR.",
            file=sys.stderr,
        )
        return 2

    print(f"Loading WNDB from {wndb_dir} ...")
    db = WordNetDb(wndb_dir)
    print(f"Loaded {len(db.synsets)} synsets, {len(db.lemma_senses)} (lemma,pos) keys.")

    if verify_mode:
        return run_verify(db)

    OUT_DIR.mkdir(exist_ok=True)
    print("Building all-senses rail ...")
    rows = build_rail(db)
    print(f"Built {len(rows)} rail rows (all senses, all hypernym edges).")
    write_rail_tsv(rows, V2_TSV_PATH)
    print(f"Wrote {V2_TSV_PATH}")

    print("Running v1-vs-v2 diff audit ...")
    run_diff(db, sample)
    return 0


if __name__ == "__main__":
    sys.exit(main())
