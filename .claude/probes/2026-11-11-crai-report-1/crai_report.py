#!/usr/bin/env python3
"""D-CRAI-REPORT-1: render compiler evidence as a CodeRabbit-shaped review, read-only.

  render  --input REVIEW.json --md OUT.md --ndjson OUT.ndjson
  from-packet --packet PACKET.json --repo R --pr N --head SHA [--scope S] > REVIEW.json
  --self-test

The finding record keeps CodeRabbit's public `--agent` fields (`fileName`, `comment`, `severity`,
`suggestions`, `codegenInstructions`; CR-SRC-01) and adds crai-rs's five separate axes
(CR-H0-REVIEW-SEMANTICS §2): severity, evidence class, confidence, provenance, lifecycle.

The renderer computes, never accepts:
- the evidence class: the highest grade among witnesses that are current. Severity, confidence
  and repetition never raise it (RS-F01, RS-F08). TEST_OBSERVED needs a test that was run and
  shown able to fail (a disable or mutation run red, then the code green; §5, RS-F05).
- the finding status: a witness from a revision other than the review head, or from a stale
  file, makes the finding `stale` with no coverage claim (RS-F04).
- whether the review is clean: only a COMPLETE run on the head with zero open findings
  (§3, RS-F03).

It never posts, applies or uploads anything. A suggestion is printed as an inert proposal bound to
its base revision and span; reviewer and agent text is printed as quoted data (RS-F09).
"""

import argparse
import json
import sys
import tempfile
import unittest
from pathlib import Path

GRADES = ["ASSERTED", "SOURCE_STRUCTURE", "COMPILER_DERIVED", "TEST_OBSERVED"]
RUN_STATUSES = {"NOT_RUN", "SKIPPED", "PARTIAL", "COMPLETE", "FAILED"}
FINDING_STATUSES = {"open", "resolved", "outdated", "stale", "withdrawn"}
KINDS = {"defect", "impact", "coverage", "admission", "research"}
SEMANTICS = "crai-rs CR-H0-REVIEW-SEMANTICS (c43ab35)"


class InputError(Exception):
    pass


def heads_of(review):
    """Head revision per repository: the PR's own repo plus any declared in `review.heads`. A
    witness from a repository with no declared head can never be current."""
    r = review["review"] if "review" in review else review
    return {r["repo"]: r["head_sha"], **(r.get("heads") or {})}


def witness_is_current(w, heads, repo):
    want = heads.get(w.get("repo", repo))
    return want is not None and w.get("revision") == want and w.get("fresh", True)


def evidence_class(finding, heads, repo):
    """Highest grade among current witnesses. A TEST_OBSERVED witness counts only when its
    falsifier ran red under a disable/mutation and green on the code (§5)."""
    best = 0
    for w in finding.get("evidence", []):
        if not witness_is_current(w, heads, repo):
            continue
        g = w.get("grade")
        if g not in GRADES:
            raise InputError(f"{finding['id']}: unknown grade {g!r}")
        if g == "TEST_OBSERVED" and finding.get("falsifier", {}).get("status") != "RAN_RED_THEN_GREEN":
            g = "SOURCE_STRUCTURE" if w.get("statement") else "ASSERTED"
        best = max(best, GRADES.index(g))
    return GRADES[best]


def finding_status(finding, heads, repo):
    stated = finding.get("status", "open")
    if stated not in FINDING_STATUSES:
        raise InputError(f"{finding['id']}: unknown status {stated!r}")
    if stated in ("withdrawn", "resolved", "outdated"):
        return stated
    ev = finding.get("evidence", [])
    if any(not witness_is_current(w, heads, repo) for w in ev):
        return "stale"
    return stated


def normalise(review):
    """Validate the input and derive every computed field. Deterministic: no clock, no
    environment, stable ordering."""
    r = review["review"]
    head, repo, heads = r["head_sha"], r["repo"], heads_of(review)
    run = review["run"]
    if run["status"] not in RUN_STATUSES:
        raise InputError(f"unknown run status {run['status']!r}")
    out = []
    seen_claims = {}
    for f in sorted(review.get("findings", []), key=lambda f: f["id"]):
        if f.get("kind") not in KINDS:
            raise InputError(f"{f['id']}: unknown kind {f.get('kind')!r}")
        g = dict(f)
        g["evidence_class"] = evidence_class(f, heads, repo)
        g["status"] = finding_status(f, heads, repo)
        g["confidence"] = f.get("confidence") if (f.get("confidence") or {}).get("source") else {"value": "UNKNOWN", "source": None}
        g["severity"] = f.get("severity")  # verbatim reviewer label, or None; never computed
        fs = f.get("falsifier") or {}
        g["falsifier"] = {"description": fs.get("description", "NOT_SPECIFIED"), "command": fs.get("command"),
                          "status": fs.get("status", "NOT_RUN"), "disable_run": fs.get("disable_run")}
        # A repeat is the same claim, at the same place, in the same lifecycle state. A resolved
        # record never absorbs an open one elsewhere or in another state.
        loc = f.get("location") or {}
        claim = (f.get("claim") or "").strip().lower()
        key = (claim, loc.get("file"), loc.get("symbol"), g["status"]) if claim else None
        if key and key in seen_claims:
            g["duplicate_of"] = seen_claims[key]  # repetition is not a second witness (RS-F08)
        elif key:
            seen_claims[key] = f["id"]
        reach = f.get("reachability") or {}
        g["reachability"] = {"confirmed": sorted(reach.get("confirmed", [])),
                             "possible": sorted(reach.get("possible", [])),
                             "unknown": sorted(reach.get("unknown", []), key=json.dumps)}
        out.append(g)
    open_findings = [f for f in out if f["status"] == "open" and "duplicate_of" not in f]
    # A stale finding was neither confirmed nor refuted at this head, so it blocks a clean verdict.
    stale = [f for f in out if f["status"] == "stale" and "duplicate_of" not in f]
    clean = (run["status"] == "COMPLETE" and not open_findings and not stale
             and run.get("head_sha", head) == head)
    counts = f"{len(open_findings)} open finding(s)" + (f", {len(stale)} stale, not revalidated at the head" if stale else "")
    verdict = ("CLEAN: complete run on the head, zero open or stale findings" if clean
               else f"INCOMPLETE REVIEW: run {run['status']}; not a defect verdict ({counts})" if run["status"] != "COMPLETE"
               else counts)
    return {"review": r, "run": run, "findings": out, "verdict": verdict, "clean": clean}


def site(f):
    loc = f.get("location") or {}
    lines = loc.get("lines")
    return loc.get("file", "-") + (f":{lines[0]}-{lines[1]}" if lines else "")


def to_ndjson(n):
    r, rows = n["review"], []
    rows.append({"type": "review_start", "repo": r["repo"], "pr": r.get("pr"), "base_sha": r.get("base_sha"),
                 "head_sha": r["head_sha"], "heads": heads_of(n), "scope": r.get("scope"), "reviewer": r.get("reviewer"),
                 "semantics": SEMANTICS})
    for f in n["findings"]:
        sug = f.get("suggestion") or {}
        rows.append({
            "type": "finding", "id": f["id"], "kind": f["kind"],
            "fileName": (f.get("location") or {}).get("file"), "lines": (f.get("location") or {}).get("lines"),
            "symbol": (f.get("location") or {}).get("symbol"),
            "comment": f.get("claim"), "severity": f["severity"],
            "suggestions": [sug] if sug.get("text") else [],
            "codegenInstructions": None,
            "evidence_class": f["evidence_class"], "confidence": f["confidence"],
            "provenance": f.get("provenance"), "status": f["status"],
            "evidence": f.get("evidence", []), "reachability": f["reachability"],
            "coverage": f.get("coverage", {}), "falsifier": f["falsifier"],
            "refusal": sug.get("refusal"), "duplicate_of": f.get("duplicate_of"),
            "status_detail": f.get("status_detail"),
        })
    rows.append({"type": "complete", "status": n["run"]["status"], "reasons": n["run"].get("reasons", []),
                 "findings": len(n["findings"]), "clean": n["clean"], "verdict": n["verdict"]})
    return "".join(json.dumps(x, sort_keys=True, ensure_ascii=False) + "\n" for x in rows)


def q(text):
    """Untrusted text as a Markdown blockquote: never a heading, link target or instruction."""
    return "\n".join("> " + ln.replace("`", "'") for ln in str(text).splitlines()) or "> "


def to_md(n):
    r = n["review"]
    L = [f"# Review: {r['repo']}" + (f" #{r['pr']}" if r.get("pr") else ""), "",
         f"- **Head:** `{r['head_sha']}`" + (f" (base `{r['base_sha']}`)" if r.get("base_sha") else ""),
         *[f"- **Evidence head, {k}:** `{v}`" for k, v in sorted(heads_of(n).items()) if k != r["repo"]],
         f"- **Scope:** {r.get('scope', 'not stated')}",
         f"- **Reviewer:** {json.dumps(r.get('reviewer'), sort_keys=True)}",
         f"- **Run:** `{n['run']['status']}`" + (" (" + "; ".join(n["run"].get("reasons", [])) + ")" if n["run"].get("reasons") else ""),
         f"- **Verdict:** {n['verdict']}", f"- **Semantics:** {SEMANTICS}. Severity, evidence and confidence are separate axes.", ""]
    if not n["findings"]:
        L += ["No findings." + ("" if n["clean"] else " This is **not** a clean review: the run did not complete."), ""]
    for f in n["findings"]:
        L += [f"## {f['id']} · {f['kind']} · {f['status']}", "",
              f"**Location:** `{site(f)}`" + (f" · `{(f.get('location') or {}).get('symbol')}`" if (f.get('location') or {}).get('symbol') else ""), "",
              f"| severity | evidence class | confidence |", "|---|---|---|",
              f"| {f['severity']['value'] + ' (' + f['severity']['source'] + ')' if f['severity'] else 'not assessed'} | {f['evidence_class']} | {f['confidence']['value']}"
              + (f" ({f['confidence']['source']})" if f['confidence']['source'] else "") + " |", "",
              "**Claim**", "", q(f.get("claim", "")), ""]
        if f.get("duplicate_of"):
            L += [f"Same claim as {f['duplicate_of']}; not counted as a second witness.", ""]
        if f.get("evidence"):
            L += ["**Evidence**", "", "| grade | qualifier | statement | witness | revision |", "|---|---|---|---|---|"]
            for w in f["evidence"]:
                cur = "" if witness_is_current(w, heads_of(n), r["repo"]) else " (not current)"
                where = (w.get("repo") or r["repo"]).split("/")[-1]
                L.append(f"| {w['grade']} | {w.get('qualifier') or ''} | {w.get('statement', '').replace('|', '/')} | `{where}:{w.get('witness', '-')}` | `{(w.get('revision') or '-')[:12]}`{cur} |")
            L.append("")
        rc = f["reachability"]
        if any(rc.values()):
            L += ["**Reachability**", "", f"- confirmed: {', '.join(f'`{x}`' for x in rc['confirmed']) or 'none observed'}",
                  f"- possible: {', '.join(f'`{x}`' for x in rc['possible']) or 'none observed'}",
                  "- unknown: " + ("; ".join(u.get("what", str(u)) + (f" ({u['why']})" if u.get("why") else "") for u in rc["unknown"]) or "none recorded"), ""]
        cov = f.get("coverage") or {}
        if cov:
            L += ["**Coverage**", ""] + [f"- {k}: {v}" for k, v in sorted(cov.items())] + [""]
        fs = f["falsifier"]
        L += ["**Falsifier**", "", f"- {fs['description']}", f"- status: `{fs['status']}`"
              + (f", command `{fs['command']}`" if fs.get("command") else "")]
        if fs.get("disable_run"):
            L.append(f"- disable run: {fs['disable_run']}")
        L.append("")
        sug = f.get("suggestion") or {}
        if sug.get("text"):
            L += ["**Suggestion** (inert proposal; applying it needs the owner's approval and revalidation"
                  f" against `{sug.get('base_sha', '?')[:12]}`)", "", q(sug["text"]), ""]
        elif sug.get("refusal"):
            L += [f"**No suggestion:** {sug['refusal']}", ""]
        if f.get("status_detail"):
            L += [f"**Status detail:** {f['status_detail']}", ""]
    return "\n".join(L).rstrip("\n") + "\n"


# --------------------------------------------------------------------------------------------
# Adapter: a recon.py ReviewEvidencePacket into review input. Facts only; no severity.

def from_packet(p, repo, pr, head, scope):
    ident, cs, aff = p["identity"], p["changed_source"], p["affected_symbols"]
    src = ident["source_span"]
    rev = ident["commit_sha"]
    fresh = not cs["stale"] and not cs.get("dependency_evidence_stale", False)
    prov = {"tool": "lance-graph recon.py packet over a Charon/ruff_charon store", "revision": rev,
            "crate_instance": ident["crate_instance"], "features": ident["features"],
            "target": ident["target_triple"], "rustc": ident["rustc_version"]}
    loc = {"file": src["file"], "lines": src["lines"], "symbol": cs["subject"]}
    wit = [{"grade": "COMPILER_DERIVED", "statement": f"`{e['caller']}` calls `{cs['subject']}`", "witness": s,
            "tool": "charon", "revision": rev, "fresh": fresh and e.get("fresh", True)}
           for e in p.get("witnessed_edges", []) for s in e["sites"]]
    tests = [i for f in p["findings"] if f["id"] == "tests" for i in f["items"]]
    findings = [{
        "id": "F1", "kind": "impact", "location": loc, "provenance": prov,
        "claim": f"Changing `{cs['subject']}` affects its in-store callers. Callers outside the analysed crates are not observed.",
        "evidence": wit,
        "reachability": {"confirmed": aff["confirmed"], "possible": aff["possible"], "unknown": aff["unknown"]},
        "coverage": {"observed_crate": ident["crate_instance"], "unsupported_regions": len(p.get("unsupported_regions", []))},
        "falsifier": {"description": "Remove one witnessed call site and rebuild the store: the caller must leave `confirmed`.",
                      "status": "NOT_RUN"},
        "suggestion": {"refusal": "impact is a fact, not a defect; nothing to change"},
    }, {
        "id": "F2", "kind": "coverage", "location": loc, "provenance": prov,
        "claim": f"{len(tests)} test call sites match `{cs['subject'].split('::')[-1]}` by name. None is resolved to this symbol, and none ran.",
        "evidence": [{"grade": "SOURCE_STRUCTURE", "qualifier": "POSSIBLE", "statement": f"name match, receiver unresolved ({i['ra']})",
                      "witness": i["site"], "tool": "source search", "revision": rev, "fresh": fresh} for i in tests],
        "reachability": {"unknown": [{"what": "test callers", "why": "library targets only; macro regions and cfg(test) modules not indexed"}]},
        "falsifier": {"description": "Extract the test targets and re-run impact: name-matched sites either become confirmed callers or drop out.",
                      "status": "NOT_RUN"},
        "suggestion": {"refusal": "coverage gap; the next step is extraction, not a code change"},
    }]
    return {"review": {"repo": repo, "pr": pr, "head_sha": head, "base_sha": cs.get("base"), "scope": scope,
                       "reviewer": {"tool": "crai_report.py", "input": "ReviewEvidencePacket"}},
            "run": {"status": "PARTIAL", "reasons": ["library targets only", "consumer crates not in store", "no test executed"]},
            "findings": findings}


# --------------------------------------------------------------------------------------------
# Negative controls (D-CRAI-REPORT-1 Gate 3; crai-rs RS-F01..F10 where they apply).

HEAD = "a" * 40
OLD = "b" * 40


def base(findings, status="COMPLETE"):
    return {"review": {"repo": "x/y", "pr": 1, "head_sha": HEAD, "scope": "PR"}, "run": {"status": status}, "findings": findings}


def fnd(fid="F1", **kw):
    f = {"id": fid, "kind": "defect", "claim": "c", "location": {"file": "a.rs", "lines": [1, 2]}}
    f.update(kw)
    return f


def w(grade, rev=HEAD, fresh=True, **kw):
    return {"grade": grade, "statement": "s", "witness": "a.rs:1", "revision": rev, "fresh": fresh, **kw}


class Negatives(unittest.TestCase):
    def test_missing_consumer_crates_yield_unknown_not_unaffected(self):
        p = json.loads(Path(__file__).with_name("packet-LaneBinding-check.json").read_text())
        n = normalise(from_packet(p, "x/y", 1, p["identity"]["commit_sha"], "PR"))
        unk = n["findings"][0]["reachability"]["unknown"]
        self.assertTrue(any("outside the store" in u["what"] for u in unk))
        self.assertNotIn("unaffected", to_md(n).lower())

    def test_stale_packet_cannot_produce_a_current_head_finding(self):  # RS-F04
        n = normalise(base([fnd(evidence=[w("COMPILER_DERIVED", rev=OLD)])]))
        self.assertEqual(n["findings"][0]["status"], "stale")
        self.assertEqual(n["findings"][0]["evidence_class"], "ASSERTED")
        n = normalise(base([fnd(evidence=[w("COMPILER_DERIVED", fresh=False)])]))
        self.assertEqual(n["findings"][0]["status"], "stale")

    def test_cross_repo_evidence_is_current_only_at_its_own_head(self):
        other = {"grade": "COMPILER_DERIVED", "statement": "s", "witness": "b.rs:1", "repo": "x/z", "revision": OLD}
        undeclared = normalise(base([fnd(evidence=[other])]))
        self.assertEqual(undeclared["findings"][0]["status"], "stale")  # no declared head for x/z
        rv = base([fnd(evidence=[other])])
        rv["review"]["heads"] = {"x/z": OLD}
        self.assertEqual(normalise(rv)["findings"][0]["status"], "open")
        self.assertEqual(normalise(rv)["findings"][0]["evidence_class"], "COMPILER_DERIVED")

    def test_partial_or_skipped_with_zero_findings_is_never_clean(self):  # RS-F03
        for st in ("PARTIAL", "SKIPPED", "NOT_RUN", "FAILED"):
            n = normalise(base([], status=st))
            self.assertFalse(n["clean"], st)
            self.assertIn("INCOMPLETE REVIEW", n["verdict"])
            self.assertNotIn("CLEAN:", n["verdict"])
        self.assertTrue(normalise(base([]))["clean"])  # twin: COMPLETE with zero findings is clean

    def test_a_named_test_without_an_executed_falsifier_stays_not_run(self):  # RS-F05
        n = normalise(base([fnd(evidence=[w("TEST_OBSERVED")], falsifier={"description": "t", "command": "cargo test x"})]))
        self.assertEqual(n["findings"][0]["falsifier"]["status"], "NOT_RUN")
        self.assertNotEqual(n["findings"][0]["evidence_class"], "TEST_OBSERVED")
        green_only = fnd(evidence=[w("TEST_OBSERVED")], falsifier={"description": "t", "status": "RAN_GREEN_ONLY"})
        self.assertNotEqual(normalise(base([green_only]))["findings"][0]["evidence_class"], "TEST_OBSERVED")
        ok = fnd(evidence=[w("TEST_OBSERVED")], falsifier={"description": "t", "status": "RAN_RED_THEN_GREEN"})
        self.assertEqual(normalise(base([ok]))["findings"][0]["evidence_class"], "TEST_OBSERVED")

    def test_compiler_evidence_and_reviewer_hypotheses_keep_separate_grades(self):  # RS-F01/F02
        hyp = fnd("F1", severity={"value": "critical", "source": "reviewer"}, evidence=[w("ASSERTED")])
        comp = fnd("F2", claim="d", evidence=[w("COMPILER_DERIVED")])
        a, b = normalise(base([hyp, comp]))["findings"]
        self.assertEqual((a["evidence_class"], a["severity"]["value"]), ("ASSERTED", "critical"))
        self.assertEqual((b["evidence_class"], b["severity"]), ("COMPILER_DERIVED", None))
        self.assertEqual(a["confidence"]["value"], "UNKNOWN")  # never derived from severity

    def test_an_irrelevant_benchmark_does_not_become_a_defect(self):
        # A benchmark result with no witness on changed code is research, and stays ASSERTED.
        bench = fnd(kind="research", claim="route F halves O6 on AVX-512", evidence=[w("ASSERTED")],
                    location={"file": "examples/u64x64/main.rs"})
        n = normalise(base([bench]))
        self.assertEqual((n["findings"][0]["kind"], n["findings"][0]["evidence_class"]), ("research", "ASSERTED"))
        self.assertNotIn("defect", to_ndjson(n))

    def test_repeated_input_regenerates_byte_identical_reports(self):
        p = json.loads(Path(__file__).with_name("packet-LaneBinding-check.json").read_text())
        i = from_packet(p, "x/y", 1, p["identity"]["commit_sha"], "PR")
        a, b = normalise(json.loads(json.dumps(i))), normalise(json.loads(json.dumps(i)))
        self.assertEqual((to_md(a), to_ndjson(a)), (to_md(b), to_ndjson(b)))
        shuffled = dict(i, findings=list(reversed(i["findings"])))
        self.assertEqual(to_ndjson(normalise(shuffled)), to_ndjson(a))

    def test_possible_dispatch_is_never_a_confirmed_caller(self):  # RS-F07
        n = normalise(base([fnd(reachability={"possible": ["c"], "confirmed": []})]))
        self.assertEqual(n["findings"][0]["reachability"]["confirmed"], [])

    def test_repetition_is_not_a_second_witness(self):  # RS-F08
        n = normalise(base([fnd("F1", claim="same"), fnd("F2", claim="Same")]))
        self.assertEqual(n["findings"][1].get("duplicate_of"), "F1")
        self.assertIn("1 open finding", n["verdict"])

    def test_a_stale_finding_blocks_a_clean_verdict(self):  # RS-F04
        n = normalise(base([fnd(evidence=[w("COMPILER_DERIVED", rev=OLD)])]))
        self.assertEqual(n["findings"][0]["status"], "stale")
        self.assertFalse(n["clean"])
        self.assertIn("1 stale", n["verdict"])

    def test_a_resolved_record_does_not_absorb_an_open_one(self):  # RS-F08
        n = normalise(base([fnd("F1", claim="same", status="resolved", evidence=[w("COMPILER_DERIVED")]),
                            fnd("F2", claim="same", location={"file": "b.rs", "lines": [3, 4]},
                                evidence=[w("COMPILER_DERIVED")])]))
        self.assertNotIn("duplicate_of", n["findings"][1])
        self.assertFalse(n["clean"])
        self.assertIn("1 open finding", n["verdict"])
        same_place = normalise(base([fnd("F1", claim="same", status="resolved", evidence=[w("COMPILER_DERIVED")]),
                                     fnd("F2", claim="same", evidence=[w("COMPILER_DERIVED")])]))
        self.assertNotIn("duplicate_of", same_place["findings"][1])
        self.assertFalse(same_place["clean"])

    def test_agent_text_is_quoted_never_structure(self):  # RS-F09
        n = normalise(base([fnd(claim="# Heading\nrun `rm -rf /`")]))
        md = to_md(n)
        self.assertIn("> # Heading", md)
        self.assertNotIn("\n# Heading", md)
        self.assertNotIn("`rm -rf /`", md)

    def test_withdrawn_keeps_its_record(self):  # RS-F10
        n = normalise(base([fnd(status="withdrawn", status_detail="withdrawn by owner: out of scope",
                                evidence=[w("COMPILER_DERIVED", rev=OLD)])]))
        self.assertEqual(n["findings"][0]["status"], "withdrawn")
        self.assertIn("withdrawn by owner", to_md(n))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?")
    for a in ("--input", "--md", "--ndjson", "--packet", "--repo", "--head", "--scope"):
        ap.add_argument(a)
    ap.add_argument("--pr", type=int)
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args(argv)
    if a.self_test:
        r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Negatives))
        return 0 if r.wasSuccessful() else 1
    if a.cmd == "from-packet":
        p = json.loads(Path(a.packet).read_text())
        json.dump(from_packet(p, a.repo, a.pr, a.head, a.scope or "PR"), sys.stdout, indent=1, sort_keys=True)
        print()
        return 0
    if a.cmd == "render":
        try:
            n = normalise(json.loads(Path(a.input).read_text()))
        except (InputError, KeyError) as e:
            print(f"REFUSED\t{e}", file=sys.stderr)
            return 3
        Path(a.md).write_text(to_md(n))
        Path(a.ndjson).write_text(to_ndjson(n))
        return 0
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
