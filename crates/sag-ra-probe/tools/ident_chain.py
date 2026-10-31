#!/usr/bin/env python3
"""Symbol identity that survives a move, only when a move is witnessed
(D-SAG-6-2 lab).

    python3 -I ident_chain.py [--declared rebind.tsv] A.txt B.txt [C.txt ...]

Each input is a `DUMP=1 RESOLVE=...` run of sag-ra-probe on one revision, in
revision order. RESOLVE must name every path that vanished since the previous
revision; `--needs A.txt B.txt` prints that list.

An identity is minted the first time an item is seen and continues to a later
revision only by one of:

  SAME          same moniker, same body digest
  BODY_CHANGED  same moniker, different body (an ordinary edit)
  MOVED         the old moniker is gone, a later revision RESOLVES the old path
                to a new item (a re-export: the compiler's own resolution), and
                the body digest is equal
  DECLARED      a row in --declared (old, new, who, why). A human decision,
                recorded as such, never inferred

Everything else ends the old identity:

  REBOUND        the old path resolves to an item with a different body. The
                 path continues, the item does not. Continuation needs DECLARED
  POSSIBLE_MOVE  one new item with the same kind and body, but no witness
  AMBIGUOUS      several such candidates, or the body is shared in the source
  CONFLICT       the old path resolves to an item that already has its own
                 identity
  ENDED          gone, no candidate
  UNRESOLVED     gone, and the run carries no resolve line for it: UNKNOWN,
                 not evidence of absence

A body digest is never an identity on its own: equal text occurs in distinct
items (DIGEST_SHARED). Output: one `ident` row per item and revision, then
one `edge` row per transition.
"""
import sys


def load(path):
    items, resolve = {}, {}
    with open(path) as fh:
        lines = fh.readlines()
    for line in lines:
        cols = line.rstrip("\n").split("\t")
        if cols[0] == "resolve":
            resolve[cols[1]] = cols[2]
            continue
        if cols[0] != "sym":
            continue
        f = dict(c.split("=", 1) for c in cols[2:] if "=" in c)
        m = f.get("moniker", "none")
        if m in ("none", "local") or f.get("at", "none") in ("none",) or f["at"].startswith("external"):
            continue
        items[m] = (f["kind"], f.get("body", "none"))
    return items, resolve


def needs(prev, cur):
    return ",".join(sorted(set(prev) - set(cur)))


def main(argv):
    declared = {}
    if argv and argv[0] == "--needs":
        print(needs(load(argv[1])[0], load(argv[2])[0]))
        return
    if argv and argv[0] == "--declared":
        with open(argv[1]) as fh:
            rebinds = fh.readlines()
        for line in rebinds:
            if line.strip() and not line.startswith("#"):
                old, new, who, why = line.rstrip("\n").split("\t")
                declared[old] = (new, who, why)
        argv = argv[2:]
    revs = [load(p) for p in argv]
    ident, next_id = {}, 0

    def mint(m):
        nonlocal next_id
        next_id += 1
        return f"sid-{next_id}"

    for m in sorted(revs[0][0]):
        ident[m] = mint(m)
    rows = [("ident", "r0", m, ident[m]) for m in sorted(ident)]
    edges = []
    for r in range(1, len(revs)):
        (prev, _), (cur, res) = revs[r - 1], revs[r]
        new = {m for m in cur if m not in prev}
        digests_prev = [d for _, d in prev.values()]
        nxt = {}
        claimed = set()
        for m in sorted(prev):
            kind, dig = prev[m]
            if m in cur:
                status = "SAME" if cur[m][1] == dig else "BODY_CHANGED"
                nxt[m] = ident[m]
                edges.append((f"r{r}", m, m, status, "path"))
                continue
            target = res.get(m)
            status, to, witness = None, "-", "-"
            if m in declared and declared[m][0] in new:
                to, who, why = declared[m]
                status, witness = "DECLARED", f"{who}: {why}"
            elif target is None:
                status = "UNRESOLVED"
            elif target not in ("unresolved", "ambiguous"):
                to, witness = target, "resolve"
                if target not in new:
                    status = "CONFLICT"
                elif cur[target][0] != kind or cur[target][1] != dig:
                    status = "REBOUND"
                else:
                    status = "MOVED"
            if status is None:
                cands = [n for n in new if cur[n][0] == kind and cur[n][1] == dig]
                shared = digests_prev.count(dig) > 1
                if len(cands) == 1 and not shared:
                    status, to = "POSSIBLE_MOVE", cands[0]
                elif cands:
                    status, to = "AMBIGUOUS", ",".join(sorted(cands))
                else:
                    status = "ENDED"
            if status in ("MOVED", "DECLARED") and to not in claimed:
                nxt[to] = ident[m]
                claimed.add(to)
            edges.append((f"r{r}", m, to, status, witness))
        for m in sorted(cur):
            if m not in nxt:
                nxt[m] = mint(m)
        ident = nxt
        rows += [("ident", f"r{r}", m, ident[m]) for m in sorted(ident)]
    for row in rows:
        print("\t".join(row))
    for e in edges:
        print("edge\t" + "\t".join(e))


if __name__ == "__main__":
    main(sys.argv[1:])
