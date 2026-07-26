#!/usr/bin/env python3
"""D-RCC-5 taxonomic arm — WordNet hypernym-tier-delta as CHAODA anomaly
magnitude (rosetta-codebook-convergence-v1).

The plan's thesis (see `.claude/plans/rosetta-codebook-convergence-v1.md`
§2 D-RCC-4/D-RCC-5, and the two operator-correction blocks above D-RCC-5):
a translation error / doctrinal substitution shows up as a hypernym-tier
DELTA. Sibling synsets (translational freedom) meet close to the disputed
terms; an inherited doctrinal substitution (canonical example: German
`Erbsünde` "original sin" standing in for Greek/source `Tod`/`Thanatos`
"death") only meets its true counterpart near the taxonomy ROOT. The tier
delta is deterministic and auditable — never a learned weight.

THIS SCRIPT'S FIRST JOB IS AN HONEST CAPABILITY AUDIT, not a pretty table.
See the "CAPABILITY AUDIT" section emitted at the top of the report and
printed first to stdout. Read it before trusting the scored pairs below it.

Data (gitignored, see .gitignore rule for this directory):
  - `wordnet31_isa.tsv` in this directory (committed generator: this file;
    the TSV itself is NOT committed) — Open/Princeton WordNet 3.1, but
    the file's own header says "First-sense hypernym per lemma": ONE
    hypernym edge per (word, pos), no synset id, no polysemy. Verified
    empirically below: max count of (word, pos) pairs in the file is 1.
  - WordNet 3.1 WNDB "dict" database (index.noun/index.verb +
    data.noun/data.verb, the classic Princeton lexicographer-file format)
    at a directory named by $WNDB_DIR, or (session-local convenience,
    NOT relied upon for reproducibility) /tmp/wn/dict if present. This
    format carries ALL senses per lemma, real synset ids, and the full
    hypernym pointer graph (multiple-inheritance DAG, not a tree) — it is
    what the tier-delta measure actually needs. If absent, the script
    degrades to the TSV-only path and says so, loudly, in the report.

No network access. No package installs. Stdlib only.

Usage:
  python3 tier_delta.py                      # auto-detect WNDB_DIR / /tmp/wn/dict
  WNDB_DIR=/path/to/dict python3 tier_delta.py
Out:
  out/tier_delta_report.md in this directory.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
TSV_PATH = HERE / "wordnet31_isa.tsv"
OUT_DIR = HERE / "out"

POS_FILES = {"n": "data.noun", "v": "data.verb"}
INDEX_FILES = {"n": "index.noun", "v": "index.verb"}
HYPERNYM_SYMS = {"@", "@i"}  # hypernym, instance-hypernym


# ------------------------------------------------------------------
# §1 — Capability audit over the committed TSV (runs unconditionally)
# ------------------------------------------------------------------


@dataclass
class TsvAudit:
    total_rows: int = 0
    max_rows_per_word_pos: int = 0
    duplicate_word_pos_examples: list = field(default_factory=list)
    swallow_rows: list = field(default_factory=list)
    grape_rows: list = field(default_factory=list)
    verdict_lines: list = field(default_factory=list)


def audit_tsv(path: Path) -> TsvAudit:
    audit = TsvAudit()
    if not path.exists():
        audit.verdict_lines.append(f"TSV NOT FOUND at {path} — cannot audit.")
        return audit
    counts: Counter = Counter()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 4:
                continue
            word, pos, kind, typ = parts[0], parts[1], parts[2], parts[3]
            audit.total_rows += 1
            counts[(word, pos)] += 1
            if word == "swallow":
                audit.swallow_rows.append((word, pos, kind, typ))
            if word == "grape":
                audit.grape_rows.append((word, pos, kind, typ))
    audit.max_rows_per_word_pos = max(counts.values()) if counts else 0
    dupes = [k for k, v in counts.items() if v > 1]
    audit.duplicate_word_pos_examples = dupes[:10]

    audit.verdict_lines.append(
        f"TSV rows: {audit.total_rows}; distinct (word,pos) keys: {len(counts)}."
    )
    audit.verdict_lines.append(
        f"Max rows sharing one (word,pos) key: {audit.max_rows_per_word_pos} "
        f"(1 == every lemma+POS collapsed to a single hypernym edge)."
    )
    if audit.max_rows_per_word_pos <= 1:
        audit.verdict_lines.append(
            "CONFIRMED: the file's own header claim ('First-sense hypernym "
            "per lemma') is empirically true on this data — it is STRICTLY "
            "one row per (word, pos), i.e. keep-first. It CANNOT distinguish "
            "swallow(bird) from swallow(gulp/ingest) — both collapse onto "
            "whatever single hypernym the extractor happened to keep for "
            "'swallow'/n and 'swallow'/v respectively."
        )
    else:
        audit.verdict_lines.append(
            "UNEXPECTED: found (word,pos) keys with >1 row — the file is "
            "NOT strictly keep-first after all; re-examine before trusting "
            "the 'first-sense-only' framing below."
        )
    if audit.swallow_rows:
        audit.verdict_lines.append(f"swallow rows in TSV: {audit.swallow_rows}")
    if audit.grape_rows:
        audit.verdict_lines.append(f"grape rows in TSV: {audit.grape_rows}")
    return audit


# ------------------------------------------------------------------
# §2 — WNDB (full synset) loader, optional richer path
# ------------------------------------------------------------------


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
    lex_filenum: str
    words: list
    hypernyms: list  # list of (offset, pos)
    gloss: str

    @property
    def id(self):
        return (self.offset, self.pos)


class WordNetDb:
    """Loads WNDB data.<pos>/index.<pos> for pos in {n, v}."""

    def __init__(self, wndb_dir: Path):
        self.wndb_dir = wndb_dir
        self.synsets: dict = {}  # (offset,pos) -> Synset
        self.lemma_index: dict = {}  # (lemma,pos) -> [ (offset,pos), ... ] sense order
        for pos, fname in POS_FILES.items():
            self._load_data(wndb_dir / fname, pos)
        for pos, fname in INDEX_FILES.items():
            self._load_index(wndb_dir / fname, pos)

    def _load_data(self, path: Path, pos: str) -> None:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("  "):  # license header padding lines
                    continue
                if not line.strip():
                    continue
                # split off gloss
                if " | " in line:
                    body, gloss = line.split(" | ", 1)
                else:
                    body, gloss = line, ""
                toks = body.split()
                if len(toks) < 4:
                    continue
                offset = toks[0]
                lex_filenum = toks[1]
                ss_type = toks[2]
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
                    # idx+3 is the source/target hex word field, skip
                    idx += 4
                    if sym in HYPERNYM_SYMS:
                        hypernyms.append((target_offset, target_pos))
                syn = Synset(
                    offset=offset,
                    pos=pos,
                    lex_filenum=lex_filenum,
                    words=words,
                    hypernyms=hypernyms,
                    gloss=gloss.strip(),
                )
                self.synsets[(offset, pos)] = syn

    def _load_index(self, path: Path, pos: str) -> None:
        with path.open(encoding="utf-8", errors="replace") as fh:
            for line in fh:
                if line.startswith("  ") or not line.strip():
                    continue
                toks = line.split()
                lemma = toks[0]
                # toks[1] == pos
                synset_cnt = int(toks[2])
                p_cnt = int(toks[3])
                idx = 4 + p_cnt  # skip ptr_symbols
                idx += 2  # sense_cnt, tagsense_cnt
                offsets = toks[idx : idx + synset_cnt]
                self.lemma_index[(lemma, pos)] = [(o, pos) for o in offsets]

    # -- ancestry / tier-delta -----------------------------------------

    def ancestors_with_depth(self, synset_id) -> dict:
        """BFS over hypernym edges; returns {ancestor_id: shortest_depth}.

        Includes the synset itself at depth 0. Multiple inheritance (a
        synset with >1 hypernym) is a DAG, not a tree — BFS gives the
        SHORTEST edge-count path to every reachable ancestor, which is
        the standard edge-counting convention (Rada et al. 1989).
        """
        depths = {synset_id: 0}
        q = deque([synset_id])
        while q:
            cur = q.popleft()
            syn = self.synsets.get(cur)
            if syn is None:
                continue
            for hyp in syn.hypernyms:
                if hyp not in depths:
                    depths[hyp] = depths[cur] + 1
                    q.append(hyp)
        return depths

    def lemma_senses(self, lemma: str, pos: str):
        return self.lemma_index.get((lemma, pos), [])

    def gloss_of(self, synset_id) -> str:
        syn = self.synsets.get(synset_id)
        return syn.gloss if syn else ""

    def words_of(self, synset_id) -> list:
        syn = self.synsets.get(synset_id)
        return syn.words if syn else []


# Outcome tags for tier_delta — "absent" and "no common ancestor" are
# DISTINCT from a measured 0, per the iron rule (absence != zero).
ABSENT = "ABSENT"
NO_COMMON_ANCESTOR = "NO_COMMON_ANCESTOR"
MEASURED = "MEASURED"


@dataclass
class TierDeltaResult:
    status: str
    delta: int | None = None
    lca: tuple | None = None
    lca_depth_from_root: int | None = None
    lca_gloss: str = ""
    note: str = ""


def synset_root_depth(db: WordNetDb, synset_id) -> int | None:
    """Depth from `synset_id` up to a synset with zero hypernyms (a true
    root / unique beginner). Nouns have one true root (entity); verbs
    have ~15 unique beginners, so 'root depth' is depth to WHICHEVER
    top synset the chain reaches, not a single universal top for verbs.
    """
    depths = db.ancestors_with_depth(synset_id)
    # the/a root is any ancestor with zero outgoing hypernyms and max depth
    best = None
    for anc, d in depths.items():
        syn = db.synsets.get(anc)
        if syn is not None and not syn.hypernyms:
            if best is None or d > best:
                best = d
    return best


def tier_delta_between_synsets(db: WordNetDb, a_id, b_id) -> TierDeltaResult:
    if a_id == b_id:
        return TierDeltaResult(
            status=MEASURED, delta=0, lca=a_id, lca_depth_from_root=0,
            lca_gloss=db.gloss_of(a_id), note="identical synset",
        )
    depths_a = db.ancestors_with_depth(a_id)
    depths_b = db.ancestors_with_depth(b_id)
    common = set(depths_a) & set(depths_b)
    if not common:
        return TierDeltaResult(status=NO_COMMON_ANCESTOR)
    best_delta = None
    best_lca = None
    for c in common:
        d = depths_a[c] + depths_b[c]
        if best_delta is None or d < best_delta:
            best_delta = d
            best_lca = c
    root_depth = synset_root_depth(db, best_lca)
    return TierDeltaResult(
        status=MEASURED,
        delta=best_delta,
        lca=best_lca,
        lca_depth_from_root=root_depth,
        lca_gloss=db.gloss_of(best_lca),
    )


def tier_delta_between_lemmas(
    db: WordNetDb, word_a: str, pos_a: str, word_b: str, pos_b: str
) -> TierDeltaResult:
    """Best-case (minimum) tier delta across ALL sense pairs of the two
    lemmas — i.e. "is there SOME reading under which these are close".
    Also returns which sense pair achieved it (the disambiguation the
    naive first-sense TSV cannot perform).
    """
    senses_a = db.lemma_senses(word_a, pos_a)
    senses_b = db.lemma_senses(word_b, pos_b)
    if not senses_a or not senses_b:
        missing = []
        if not senses_a:
            missing.append(f"{word_a}/{pos_a}")
        if not senses_b:
            missing.append(f"{word_b}/{pos_b}")
        return TierDeltaResult(status=ABSENT, note=f"absent from WNDB: {missing}")

    best: TierDeltaResult | None = None
    best_pair = None
    for sa in senses_a:
        for sb in senses_b:
            r = tier_delta_between_synsets(db, sa, sb)
            if r.status != MEASURED:
                continue
            if best is None or r.delta < best.delta:
                best = r
                best_pair = (sa, sb)
    if best is None:
        return TierDeltaResult(status=NO_COMMON_ANCESTOR)
    best.note = f"best sense pair: {best_pair}"
    return best


# ------------------------------------------------------------------
# §3 — TSV-only fallback tier delta (uses hypernym LEMMA STRINGS, not
# synset ids, because the TSV carries no synset id — see the TSV header
# format `word\tpos\tkind\ttype` where `type` is a bare lemma string
# naming the hypernym CONCEPT, not a synset offset). This path is
# strictly weaker: it builds a lemma-string hypernym graph (one edge
# per (word,pos), first-sense only) and can only ever find ONE reading
# per word, so it can never resolve the swallow-bird/swallow-gulp
# ambiguity — it is included so the script still produces SOMETHING
# useful when WNDB is unavailable, and so the report can show the
# degraded numbers side-by-side with the WNDB numbers when both exist.
# ------------------------------------------------------------------


class TsvHypernymGraph:
    def __init__(self, path: Path):
        self.hypernym_of: dict = {}  # (word,pos) -> hypernym_lemma (string)
        # hypernym lemma strings are bare words; to walk further UP we
        # need a hypernym-of-hypernym edge, but the TSV only records
        # pos for the SOURCE word, not for the hypernym target — so we
        # try both 'n' and 'v' for the next hop and prefer whichever
        # exists. This is a best-effort widening, clearly a degraded
        # substitute for real synset ids.
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 4:
                    continue
                word, pos, _kind, typ = parts[0], parts[1], parts[2], parts[3]
                self.hypernym_of[(word, pos)] = typ

    def ancestors_with_depth(self, word: str, pos: str) -> dict:
        depths = {(word, pos): 0}
        frontier = [(word, pos)]
        seen_words = {word}
        d = 0
        while frontier:
            d += 1
            nxt = []
            for w, p in frontier:
                hyp = self.hypernym_of.get((w, p))
                if hyp is None or hyp in seen_words:
                    continue
                seen_words.add(hyp)
                # try both POS for the next hop (TSV loses target POS)
                placed = False
                for hp in ("n", "v"):
                    key = (hyp, hp)
                    if key not in depths:
                        depths[key] = d
                        nxt.append(key)
                        placed = True
                if not placed:
                    depths.setdefault((hyp, "?"), d)
            frontier = nxt
            if d > 30:  # safety valve against any cycle
                break
        return depths

    def tier_delta(self, word_a, pos_a, word_b, pos_b) -> TierDeltaResult:
        if (word_a, pos_a) not in self.hypernym_of and word_a not in (
            w for (w, _p) in self.hypernym_of
        ):
            return TierDeltaResult(status=ABSENT, note=f"{word_a}/{pos_a} absent")
        depths_a = self.ancestors_with_depth(word_a, pos_a)
        depths_b = self.ancestors_with_depth(word_b, pos_b)
        # match ignoring the '?'-pos placeholder when comparing keys
        norm_a = {w: d for (w, p), d in depths_a.items()}
        norm_b = {w: d for (w, p), d in depths_b.items()}
        common = set(norm_a) & set(norm_b)
        if not common:
            return TierDeltaResult(status=NO_COMMON_ANCESTOR)
        best_delta = min(norm_a[c] + norm_b[c] for c in common)
        best_lca = min((c for c in common if norm_a[c] + norm_b[c] == best_delta))
        return TierDeltaResult(status=MEASURED, delta=best_delta, lca=(best_lca, "?"))


# ------------------------------------------------------------------
# §4 — Report assembly
# ------------------------------------------------------------------

ANCHOR_PAIRS = [
    # (word_a, pos_a, word_b, pos_b, note)
    ("sin", "n", "death", "n", "ANCHOR: Erbsünde/Tod proxy — doctrinal "
     "substitution should show a LARGE tier delta (they meet only near "
     "the taxonomy root, if at all)."),
]

POLYSEMY_PROBES = [
    ("swallow", "n", "swallow", "v", "polysemy probe: swallow(n, bird "
     "sense available in WNDB) vs swallow(v, ingest) — does the "
     "MINIMUM cross-POS delta land on the bird sense or the "
     "ingest/consumption sense? (cross-POS so at least one of each "
     "lemma's senses is compared; WordNet nouns and verbs are SEPARATE "
     "hierarchies with no direct hypernym edges between them, so a "
     "same-POS probe is more informative — see the dedicated "
     "swallow(n)-senses probe below.)"),
]

KEEP_FIRST_BUG_PAIRS = [
    ("grape", "n", "shot", "n", "known keep-first bug pair: the "
     "committed TSV maps grape(n) -> hypernym 'shot' via its (buggy) "
     "first-sense pick, when grape(n)'s highest-frequency WNDB sense is "
     "actually the FRUIT (hypernym 'edible fruit'), not 'grapeshot'."),
]

CONTROL_SMALL = [
    ("dog", "n", "wolf", "n", "control, expect SMALL delta (siblings under Canis)"),
    ("boat", "n", "ship", "n", "control, expect SMALL delta (near-synonyms)"),
    ("house", "n", "dwelling", "n", "control, expect SMALL delta (near-synonyms)"),
]

CONTROL_LARGE = [
    ("death", "n", "vineyard", "n", "control, expect LARGE delta (unrelated domains)"),
    ("stone", "n", "mercy", "n", "control, expect LARGE delta (unrelated domains)"),
]


def fmt_synset(db: WordNetDb, synset_id) -> str:
    if synset_id is None:
        return "-"
    words = db.words_of(synset_id)
    gloss = db.gloss_of(synset_id)
    return f"{synset_id[1]}#{synset_id[0]} {{{', '.join(words)}}} — {gloss[:70]}"


def run_wndb_scored_table(db: WordNetDb, lines: list) -> None:
    groups = [
        ("Anchor pairs (translation-error proxy)", ANCHOR_PAIRS),
        ("Polysemy probes", POLYSEMY_PROBES),
        ("Known keep-first bug pairs", KEEP_FIRST_BUG_PAIRS),
        ("Control pairs — expect SMALL delta", CONTROL_SMALL),
        ("Control pairs — expect LARGE delta", CONTROL_LARGE),
    ]
    small_deltas = []
    large_deltas = []
    for title, pairs in groups:
        lines.append(f"\n### {title}\n")
        lines.append("| a | b | status | delta | LCA | LCA depth-from-root | note |")
        lines.append("|---|---|---|---|---|---|---|")
        for wa, pa, wb, pb, note in pairs:
            r = tier_delta_between_lemmas(db, wa, pa, wb, pb)
            lca_str = fmt_synset(db, r.lca) if r.lca else "-"
            lines.append(
                f"| {wa}/{pa} | {wb}/{pb} | {r.status} | "
                f"{r.delta if r.delta is not None else '-'} | {lca_str} | "
                f"{r.lca_depth_from_root if r.lca_depth_from_root is not None else '-'} "
                f"| {note} {r.note} |"
            )
            print(f"[{title}] {wa}/{pa} vs {wb}/{pb}: {r.status} delta={r.delta} "
                  f"lca={lca_str}")
            if r.status == MEASURED:
                if title.startswith("Control pairs — expect SMALL"):
                    small_deltas.append(r.delta)
                elif title.startswith("Control pairs — expect LARGE"):
                    large_deltas.append(r.delta)

    lines.append("\n### swallow(n) — full sense inventory (the polysemy probe, spelled out)\n")
    senses = db.lemma_senses("swallow", "n")
    lines.append("| sense # | synset | hypernym (@) |")
    lines.append("|---|---|---|")
    for i, s in enumerate(senses, 1):
        syn = db.synsets.get(s)
        hyp = fmt_synset(db, syn.hypernyms[0]) if syn and syn.hypernyms else "-"
        lines.append(f"| {i} | {fmt_synset(db, s)} | {hyp} |")
        print(f"swallow(n) sense {i}: {fmt_synset(db, s)} -> hypernym {hyp}")

    lines.append("\n### grape(n) — full sense inventory (the keep-first-bug, spelled out)\n")
    senses = db.lemma_senses("grape", "n")
    lines.append("| sense # | synset | hypernym (@) |")
    lines.append("|---|---|---|")
    for i, s in enumerate(senses, 1):
        syn = db.synsets.get(s)
        hyp = fmt_synset(db, syn.hypernyms[0]) if syn and syn.hypernyms else "-"
        lines.append(f"| {i} | {fmt_synset(db, s)} | {hyp} |")
        print(f"grape(n) sense {i}: {fmt_synset(db, s)} -> hypernym {hyp}")

    lines.append("\n### Separation verdict\n")
    if small_deltas and large_deltas:
        margin_ok = max(small_deltas) < min(large_deltas)
        lines.append(
            f"- SMALL-control deltas measured: {small_deltas} "
            f"(max={max(small_deltas)})"
        )
        lines.append(
            f"- LARGE-control deltas measured: {large_deltas} "
            f"(min={min(large_deltas)})"
        )
        if margin_ok:
            lines.append(
                "- **VERDICT: clean separation** — every SMALL-control delta "
                "is strictly below every LARGE-control delta. The measure "
                "behaves as the plan's thesis requires on this small probe set."
            )
            print("VERDICT: clean separation between small/large controls.")
        else:
            lines.append(
                "- **VERDICT: NO clean separation** — at least one SMALL-control "
                "delta is >= a LARGE-control delta on this probe set. The "
                "measure as implemented does NOT cleanly separate them here; "
                "do not claim it does."
            )
            print("VERDICT: NO clean separation — see report.")
    else:
        lines.append(
            "- **VERDICT: inconclusive** — one or both control groups produced "
            "no MEASURED deltas (see status column above); cannot assess "
            "separation from this probe set."
        )
        print("VERDICT: inconclusive (missing measured deltas in a control group).")


def run_tsv_only_table(graph: TsvHypernymGraph, lines: list) -> None:
    lines.append(
        "\n**TSV-ONLY PATH (WNDB unavailable) — illustrative only.** Every "
        "row below uses the single first-sense hypernym LEMMA STRING "
        "recorded in the TSV; it cannot disambiguate senses and the "
        "'graph' is a chain of bare hypernym words, not a real synset "
        "DAG, so treat the numbers as a rough approximation, not a "
        "validated measure.\n"
    )
    groups = [
        ("Anchor pairs", ANCHOR_PAIRS),
        ("Polysemy probes (WILL be uninformative — see capability audit)", POLYSEMY_PROBES),
        ("Known keep-first bug pairs", KEEP_FIRST_BUG_PAIRS),
        ("Control — expect SMALL delta", CONTROL_SMALL),
        ("Control — expect LARGE delta", CONTROL_LARGE),
    ]
    for title, pairs in groups:
        lines.append(f"\n### {title}\n")
        lines.append("| a | b | status | delta | lca (lemma) |")
        lines.append("|---|---|---|---|---|")
        for wa, pa, wb, pb, note in pairs:
            r = graph.tier_delta(wa, pa, wb, pb)
            lca_str = r.lca[0] if r.lca else "-"
            lines.append(
                f"| {wa}/{pa} | {wb}/{pb} | {r.status} | "
                f"{r.delta if r.delta is not None else '-'} | {lca_str} |"
            )
            print(f"[TSV-only][{title}] {wa}/{pa} vs {wb}/{pb}: {r.status} "
                  f"delta={r.delta} lca={lca_str}")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    report_lines = []
    report_lines.append("# D-RCC-5 tier-delta probe report\n")
    report_lines.append(
        "Generated by `tier_delta.py`. See the plan's D-RCC-5 entry for the "
        "thesis this probe is testing.\n"
    )

    # --- capability audit (always runs first, always reported first) ---
    report_lines.append("## Capability audit (READ THIS FIRST)\n")
    audit = audit_tsv(TSV_PATH)
    for line in audit.verdict_lines:
        print("[AUDIT]", line)
        report_lines.append(f"- {line}")

    wndb_dir = find_wndb_dir()
    if wndb_dir:
        report_lines.append(
            f"\n- **Richer local data FOUND**: WNDB dict directory at "
            f"`{wndb_dir}` (index.noun/data.noun/index.verb/data.verb — the "
            f"classic Princeton lexicographer-file format). This carries ALL "
            f"senses per lemma with real synset ids and the full hypernym "
            f"DAG (multiple inheritance). **This is the path this run uses "
            f"for the scored table below** — it is NOT the same data as the "
            f"committed TSV generator; if this path is under `/tmp`, "
            f"note this is a SESSION-LOCAL convenience path (ephemeral /tmp), "
            f"not a reproducible pinned dependency — future runs without "
            f"that directory will fall back to the TSV-only path below."
        )
        print(f"[AUDIT] WNDB found at {wndb_dir} — using it for the scored table.")
    else:
        report_lines.append(
            "\n- **No richer local data found** (checked $WNDB_DIR, "
            f"`{HERE / 'wndb'}`, `/tmp/wn/dict`). Falling back to the "
            "TSV-only path, which is honestly first-sense-only and CANNOT "
            "resolve polysemy. The headline conclusion of this run is: "
            "**the taxonomic arm needs full synset data to do what D-RCC-5 "
            "actually asks (separating swallow-bird from swallow-gulp, "
            "picking the correct grape sense, etc.) — the committed TSV "
            "alone is illustrative only.**"
        )
        print("[AUDIT] No WNDB found — TSV-only fallback path.")

    report_lines.append("\n## Measure definition\n")
    report_lines.append(
        "`tier_delta(a, b)` = min over common ancestors c of "
        "`depth(a→c) + depth(b→c)`, where `depth` is the shortest-path "
        "edge count along hypernym (`@`/`@i`) pointers (BFS over what is "
        "in general a DAG, since WordNet synsets may have more than one "
        "hypernym — multiple inheritance). This is the standard "
        "edge-counting taxonomic distance (Rada et al. 1989 path-length "
        "family); `c` achieving the minimum is reported as the LCA "
        "(lowest common ancestor by this metric), along with its own "
        "depth from a true root/unique-beginner synset (so a delta near "
        "the root reads as 'these only agree at the most generic level', "
        "i.e. weak/doctrinal agreement) vs a shallow LCA (sibling-like, "
        "translational-freedom agreement). Three DISTINCT outcomes are "
        "reported, never conflated: `ABSENT` (lemma+pos not in the "
        "vocabulary), `NO_COMMON_ANCESTOR` (both resolved, no shared "
        "ancestor found — should not happen for two nouns/two verbs "
        "given a single connected root region, but IS expected/normal "
        "when comparing across pos with no shared hierarchy), and "
        "`MEASURED` (an actual delta)."
    )

    if wndb_dir:
        db = WordNetDb(wndb_dir)
        report_lines.append(
            f"\nLoaded WNDB: {len(db.synsets)} synsets, "
            f"{len(db.lemma_index)} (lemma,pos) index entries."
        )
        print(f"Loaded WNDB: {len(db.synsets)} synsets, {len(db.lemma_index)} lemma keys.")
        report_lines.append("\n## Scored table (WNDB path — full senses, real synsets)\n")
        run_wndb_scored_table(db, report_lines)
    else:
        graph = TsvHypernymGraph(TSV_PATH)
        report_lines.append("\n## Scored table (TSV-only fallback path)\n")
        run_tsv_only_table(graph, report_lines)

    out_path = OUT_DIR / "tier_delta_report.md"
    out_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    print(f"\nWrote report to {out_path}")


if __name__ == "__main__":
    sys.exit(main())
