#!/usr/bin/env python3
"""Emit the WordNet noun is-a TREE (offset -> primary hypernym) for the 4^4
spatial-activation probe (`bgz17/examples/probe_wordnet_44_activation.rs`).

DIFFERENT SHAPE from its sibling `build_wordnet_rail.py`, which emits a
(word, pos, sense) -> hypernym-lemma RAIL for lexical lookup. This one emits
the raw SYNSET TREE — the structure itself — because the probe's subject is
the taxonomy's branching, not word senses:

    child_offset <TAB> parent_offset <TAB> depth <TAB> lemma

`parent_offset` is the FIRST `@` (hypernym) or `@i` (instance-hypernym)
pointer in the synset's pointer list, which is WordNet's own primary-parent
order; `0` marks a root. Taking only the first parent turns WordNet's noun
DAG into a TREE — a deliberate, declared simplification (multiple inheritance
is real in WordNet; a fixed-arity address cannot carry it). The probe measures
distance ON THIS TREE, so both sides of every comparison see the same
simplification and it cannot flatter the result.

Data policy: WNDB is gitignored (same convention as every other generator in
this directory) — this script is committed, the dict and the emitted TSV are
not. Fetch:

    curl -sSL -o /tmp/wn31.tar.gz https://wordnetcode.princeton.edu/wn3.1.dict.tar.gz
    mkdir -p /tmp/wn31 && tar xzf /tmp/wn31.tar.gz -C /tmp/wn31
    python3 build_isa_tree.py /tmp/wn31/dict /tmp/wordnet_isa_tree.tsv
"""

import sys
from pathlib import Path


def parse_data_noun(path: Path):
    """offset -> (first_lemma, first_hypernym_offset_or_None)."""
    nodes = {}
    with path.open("r", encoding="latin-1") as fh:
        for line in fh:
            if line.startswith("  "):  # licence header
                continue
            data = line.split("|", 1)[0].split()
            if len(data) < 6:
                continue
            offset = int(data[0])
            w_cnt = int(data[3], 16)
            lemma = data[4]
            # words: w_cnt pairs of (word, lex_id) starting at index 4
            idx = 4 + 2 * w_cnt
            p_cnt = int(data[idx])
            idx += 1
            parent = None
            for _ in range(p_cnt):
                sym, target, pos = data[idx], data[idx + 1], data[idx + 2]
                idx += 4  # sym, offset, pos, source/target
                if parent is None and pos == "n" and sym in ("@", "@i"):
                    parent = int(target)
            nodes[offset] = (lemma, parent)
    return nodes


def depth_of(offset, nodes, memo):
    """Root distance; cycle-safe (WordNet has none, but a bad parse could)."""
    seen = []
    cur = offset
    while cur is not None and cur not in memo:
        if cur in seen:  # cycle -> treat as root
            memo[cur] = 0
            break
        seen.append(cur)
        entry = nodes.get(cur)
        if entry is None or entry[1] is None:
            memo[cur] = 0
            break
        cur = entry[1]
    for off in reversed(seen):
        if off in memo:
            continue
        parent = nodes[off][1]
        memo[off] = 0 if parent is None else memo.get(parent, 0) + 1
    return memo.get(offset, 0)


def main():
    dict_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/wn31/dict")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "/tmp/wordnet_isa_tree.tsv")

    nodes = parse_data_noun(dict_dir / "data.noun")
    memo = {}
    rows = 0
    roots = 0
    with out.open("w", encoding="utf-8") as fh:
        fh.write("# child_offset\tparent_offset\tdepth\tlemma\n")
        for offset in sorted(nodes):
            lemma, parent = nodes[offset]
            d = depth_of(offset, nodes, memo)
            if parent is None:
                roots += 1
            fh.write(f"{offset}\t{parent or 0}\t{d}\t{lemma}\n")
            rows += 1

    print(f"synsets={rows} roots={roots} max_depth={max(memo.values())} -> {out}")


if __name__ == "__main__":
    main()
