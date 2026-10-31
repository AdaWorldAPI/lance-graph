#!/usr/bin/env python3
"""Compare two `DUMP=1` runs of sag-ra-probe (D-SAG-6-2 lab).

    python3 -I ident_diff.py <dump_rev_a.txt> <dump_rev_b.txt>

Keys every function, method and type by rust-analyzer's moniker (its
crate-qualified path) and classifies each pair across the two revisions:

  SAME            same moniker, same body digest
  BODY_CHANGED    same moniker, different body digest
  POSSIBLE_MOVE   moniker gone in B, a new moniker of the same kind in B has
                  the same body digest. A candidate only: equal text is not
                  the same symbol (see DIGEST_SHARED)
  REMOVED/ADDED   no counterpart

and, within one revision, DIGEST_SHARED for distinct monikers with the same
body digest, which is why a digest must never be used as an identity.
"""
import collections
import sys

KINDS = {"Function", "Method", "TraitMethod", "Struct", "Trait", "Enum"}


def load(path):
    rows = {}
    for line in open(path):
        if not line.startswith("sym\t"):
            continue
        cols = line.rstrip("\n").split("\t")
        f = dict(c.split("=", 1) for c in cols[2:] if "=" in c)
        kind, moniker = f.get("kind"), f.get("moniker", "none")
        if kind not in KINDS or moniker in ("none", "local"):
            continue
        if moniker.split("::")[0] in ("core", "alloc", "std"):
            continue
        rows[moniker] = (kind, f.get("body", "none"))
    return rows


def shared(rev, rows):
    by = collections.defaultdict(list)
    for m, (k, b) in rows.items():
        if b != "none":
            by[(k, b)].append(m)
    for (k, _), ms in by.items():
        if len(ms) > 1:
            print(f"{rev}\tDIGEST_SHARED\t{k}\t{' | '.join(sorted(ms))}")


def main(a_path, b_path):
    a, b = load(a_path), load(b_path)
    shared("A", a)
    shared("B", b)
    gone = {m: v for m, v in a.items() if m not in b}
    new = {m: v for m, v in b.items() if m not in a}
    for m in sorted(a.keys() & b.keys()):
        cls = "SAME" if a[m][1] == b[m][1] else "BODY_CHANGED"
        print(f"{cls}\t{a[m][0]}\t{m}")
    for m, (k, body) in sorted(gone.items()):
        hits = [n for n, (k2, b2) in new.items() if k2 == k and b2 == body and body != "none"]
        if hits:
            print(f"POSSIBLE_MOVE\t{k}\t{m} -> {' | '.join(sorted(hits))}")
        else:
            print(f"REMOVED\t{k}\t{m}")
    moved_to = set()
    for m, (k, body) in gone.items():
        moved_to |= {n for n, (k2, b2) in new.items() if k2 == k and b2 == body and body != "none"}
    for m in sorted(new.keys() - moved_to):
        print(f"ADDED\t{new[m][0]}\t{m}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
