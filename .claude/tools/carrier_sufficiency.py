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

Usage: python3 .claude/tools/carrier_sufficiency.py         (stdlib only, ~1 min)
       python3 .claude/tools/carrier_sufficiency.py --min   (the Quack table below)
Referenced by .claude/plans/cypher-mask-multiplicity-contract-v1.md §3.3 (that
plan lives on the closed #1305 branch `ccr-2fcc2bd3-8o7m2l` @ 67abd29; this
tool was revived from there unchanged, plus `--min`).

--min: for every WALK observation, the CHEAPEST sufficient carrier and the
existing Quack carrier that realises it, over a tabular edge table (one row per
edge, fks into the node table). A lowering that picks a carrier NOT in the
sufficient set for its observation is wrong on some graph; the printed witness
is the counterexample. SUFFICIENT means the carrier (with the resident edge
table) DETERMINES the answer; it does not mean Quack has an op that finishes
the computation from it. E.g. "2hop Count" from E after hop 1 is
sum_{e in E} out(dst e): determined, but it needs the foreign-value sum
(an operation gap), not a new carrier. Cost order (cheapest first) and the Quack realisation:
  S  node mask of the last node     Agg::Rows / Terminal::Keep; after a hop,
                                    the target support is Agg::ScatterOrU32
                                    (terminal only) or CountDistinctOrderedU32
  E  edge population of hop k       a Filter over the edge table (re-anchor)
  K  per-node walk count            Agg::GroupReduce{Local(dst), Count}
  W  per-edge walk count            no carrier: at hop 1 every edge row is 1,
                                    so E already is W; past hop 1 it is a
                                    weighted edge lane (foreign-value sum, gap)
  TW per-edge trail count           none (Quack is WALK, like DataFusion)
  B  bindings                       none (row enumeration)
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


QUACK = {
    "S": "node mask (Rows/Keep; ScatterOrU32 or CountDistinctOrderedU32 after a hop)",
    "E": "edge-table filter (re-anchor on the edge population)",
    "K": "GroupReduce{Local(dst), Count} per-node counts",
    "W": "weighted edge lane (gap: foreign-value sum)",
    "TW": "none (trail count)",
    "B": "none (bindings)",
}
COST = ["S", "E", "K", "W", "TW", "B"]


def report_min(title, k, patterns):
    ok, _ = study(k, patterns)
    print("== %s (carrier after hop %d), WALK only ==" % (title, k))
    print("%-24s %-4s %s" % ("observation", "min", "Quack carrier"))
    for key in sorted({key for (_, key) in ok if "|walk|" in key}):
        best = next(c for c in COST if ok[(c, key)])
        print("%-24s %-4s %s" % (key.replace("|walk|", " "), best, QUACK[best]))
    print()


if __name__ == "__main__" and "--min" in __import__("sys").argv:
    report_min("one hop, then questions about 1 and 2 hops", 1,
               {"1hop": "f", "2hop": "ff", "vee": "fb"})
    report_min("two hops, then questions about 2 and 3 hops", 2,
               {"2hop": "ff", "3hop": "fff"})
elif __name__ == "__main__":
    report("one hop, then questions about 1 and 2 hops", 1,
           {"1hop": "f", "2hop": "ff", "vee": "fb"})
    report("two hops, then questions about 2 and 3 hops", 2,
           {"2hop": "ff", "3hop": "fff"})
