#!/usr/bin/env python3
"""Carrier sufficiency, measured by exhaustive enumeration.

Question: after k hops of a forward chain, which CARRIER (the state handed to
the next step) still determines a later OBSERVATION, given that the relation
(edge table) stays resident?

Method: every directed multigraph on 3 nodes with up to 4 edges (parallel
edges and self-loops included; edge id = position), every FROM population.
A carrier C is SUFFICIENT for an observation O iff no two FROM populations on
the same graph give the same C but a different O. Every "n" is backed by a
printed witness `(edges, FROM_a, FROM_b, O_a, O_b)`.

Walk = edges may repeat within a match (DataFusion, Ladybug default, SQL
joins). Trail = no edge twice within a match (openCypher relationship
uniqueness).

Carriers after hop k (Markov: computed from the matches up to hop k only):
  S  node support of the last node          (a Boolean node mask)
  K  per-node match count of the last node  (plus-times counts)
  E  edge population of hop k               (a Boolean edge mask)
  W  per-edge walk count of hop k
  TW per-edge trail count of hop k
  B  all matches so far (bindings) -- the reference, always sufficient

Usage: python3 .claude/tools/carrier_sufficiency.py   (stdlib only, ~1 min)
Referenced by .claude/plans/cypher-mask-multiplicity-contract-v1.md §3.3.
"""
import collections
import itertools

N = 3
NODES = range(N)
PAIRS = [(u, v) for u in NODES for v in NODES]
MAX_EDGES = 4


def graphs():
    for m in range(MAX_EDGES + 1):
        for combo in itertools.combinations_with_replacement(PAIRS, m):
            yield list(combo)


def populations():
    for k in range(N + 1):
        for s in itertools.combinations(NODES, k):
            yield frozenset(s)


def matches(g, start, pattern):
    """All matches as (edge-id tuple, node tuple). 'f' follows src->dst,
    'b' follows dst->src (a direction change)."""
    out = []

    def rec(nodes, edges):
        if len(edges) == len(pattern):
            out.append((tuple(edges), tuple(nodes)))
            return
        here, d = nodes[-1], pattern[len(edges)]
        for i, (u, v) in enumerate(g):
            if d == "f" and u == here:
                rec(nodes + [v], edges + [i])
            elif d == "b" and v == here:
                rec(nodes + [u], edges + [i])

    for a in start:
        rec([a], [])
    return out


def trails(ms):
    return [m for m in ms if len(set(m[0])) == len(m[0])]


def observations(ms, hops):
    o = {"Exists": bool(ms), "Count": len(ms)}
    for i in range(hops + 1):
        name = "n%d" % i
        o["Support(%s)" % name] = frozenset(m[1][i] for m in ms)
        o["CountBy(%s)" % name] = tuple(sorted(collections.Counter(m[1][i] for m in ms).items()))
    return o


def carriers(g, start, k):
    walks = matches(g, start, "f" * k)
    tr = trails(walks)
    last = [m[0][-1] for m in walks]
    return {
        "S": frozenset(m[1][-1] for m in walks),
        "K": tuple(sum(1 for m in walks if m[1][-1] == x) for x in NODES),
        "E": frozenset(last),
        "W": tuple(last.count(i) for i in range(len(g))),
        "TW": tuple(sum(1 for m in tr if m[0][-1] == i) for i in range(len(g))),
        "B": tuple(sorted(walks)),
    }


def study(k, patterns):
    ok, witness = {}, {}
    for g in graphs():
        rows = []
        for start in populations():
            obs = {}
            for pname, pat in patterns.items():
                ms = matches(g, start, pat)
                for sem, sel in (("walk", ms), ("trail", trails(ms))):
                    for oname, val in observations(sel, len(pat)).items():
                        obs["%s|%s|%s" % (pname, sem, oname)] = val
            rows.append((carriers(g, start, k), obs, sorted(start)))
        for cname in rows[0][0]:
            groups = collections.defaultdict(list)
            for c, obs, start in rows:
                groups[c[cname]].append((obs, start))
            for grp in groups.values():
                o0, s0 = grp[0]
                for key in o0:
                    ok.setdefault((cname, key), True)
                    for o1, s1 in grp[1:]:
                        if o1[key] != o0[key] and ok[(cname, key)]:
                            ok[(cname, key)] = False
                            witness[(cname, key)] = (g, s0, s1, o0[key], o1[key])
    return ok, witness


def report(title, k, patterns):
    ok, witness = study(k, patterns)
    cols = ["S", "K", "E", "W", "TW", "B"]
    print("== %s (carrier after hop %d) ==" % (title, k))
    print("%-30s %s" % ("observation", "  ".join("%-2s" % c for c in cols)))
    for key in sorted({key for (_, key) in ok}):
        print("%-30s %s" % (key, "  ".join("%-2s" % ("Y" if ok[(c, key)] else "n") for c in cols)))
    print("-- witnesses (edges, FROM_a, FROM_b, obs_a, obs_b) --")
    for (c, key), w in sorted(witness.items()):
        print("%-3s %-30s %s" % (c, key, w))
    print()


if __name__ == "__main__":
    report("one hop, then questions about 1 and 2 hops", 1,
           {"1hop": "f", "2hop": "ff", "vee": "fb"})
    report("two hops, then questions about 2 and 3 hops", 2,
           {"2hop": "ff", "3hop": "fff"})
