#!/usr/bin/env python3
"""Tests for ident_chain.py and span_join.py on hand-written inputs (D-SAG-6-2 lab).

    python3 -I tools/test_tools.py

These test the tools' rules (ident_chain.py, span_join.py), not a frontend:
the dumps and the LLBC are written here, in the shape the frontends emit. The fixtures under fixtures/sag_ident are the
frontend evidence; this covers the cases they do not reach (CONFLICT,
UNRESOLVED, a declaration that names a stale target).
"""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ident_chain  # noqa: E402
import span_join  # noqa: E402

TMP = tempfile.TemporaryDirectory()


def dump(items, resolve=()):
    lines = [
        f"sym\t{m.rsplit('::', 1)[-1]}\tkind={k}\tmoniker={m}\tbody={b}\tat=x/src/lib.rs:1"
        for m, (k, b) in items.items()
    ]
    lines += [f"resolve\t{a}\t{b}" for a, b in resolve]
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, dir=TMP.name)
    f.write("\n".join(lines) + "\n")
    f.close()
    return f.name


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        ident_chain.main(list(argv))
    edges = {}
    ident = {}
    for line in out.getvalue().splitlines():
        c = line.split("\t")
        if c[0] == "edge":
            edges[c[2]] = (c[3], c[4])
        elif c[0] == "ident":
            ident[(c[1], c[2])] = c[3]
    return edges, ident


FN = "Function"


class IdentChain(unittest.TestCase):
    def test_witnessed_move_continues_identity(self):
        a = dump({"k::u::f": (FN, "d1")})
        b = dump({"k::o::f": (FN, "d1")}, [("k::u::f", "k::o::f")])
        edges, ident = run(a, b)
        self.assertEqual(edges["k::u::f"], ("k::o::f", "MOVED"))
        self.assertEqual(ident[("r1", "k::o::f")], ident[("r0", "k::u::f")])

    def test_rebound_path_does_not_continue_identity(self):
        a = dump({"k::u::f": (FN, "d1")})
        b = dump({"k::o::g": (FN, "d2")}, [("k::u::f", "k::o::g")])
        edges, ident = run(a, b)
        self.assertEqual(edges["k::u::f"][1], "REBOUND")
        self.assertNotEqual(ident[("r1", "k::o::g")], ident[("r0", "k::u::f")])

    def test_resolve_to_an_existing_item_is_a_conflict(self):
        # The old path now re-exports an item that already had its own
        # identity: two identities cannot merge into one.
        a = dump({"k::u::f": (FN, "d1"), "k::o::g": (FN, "d1")})
        b = dump({"k::o::g": (FN, "d1")}, [("k::u::f", "k::o::g")])
        edges, ident = run(a, b)
        self.assertEqual(edges["k::u::f"][1], "CONFLICT")
        self.assertEqual(ident[("r1", "k::o::g")], ident[("r0", "k::o::g")])

    def test_missing_resolve_line_is_unknown_not_ended(self):
        a = dump({"k::u::f": (FN, "d1")})
        b = dump({"k::o::f": (FN, "d1")})
        edges, _ = run(a, b)
        self.assertEqual(edges["k::u::f"][1], "UNRESOLVED")

    def test_unwitnessed_unique_candidate_is_only_possible(self):
        a = dump({"k::u::f": (FN, "d1")})
        b = dump({"k::o::f": (FN, "d1")}, [("k::u::f", "unresolved")])
        edges, ident = run(a, b)
        self.assertEqual(edges["k::u::f"], ("k::o::f", "POSSIBLE_MOVE"))
        self.assertNotEqual(ident[("r1", "k::o::f")], ident[("r0", "k::u::f")])

    def test_shared_body_blocks_a_candidate(self):
        a = dump({"k::u::f": (FN, "d1"), "j::u::f": (FN, "d1")})
        b = dump({"k::o::f": (FN, "d1"), "j::u::f": (FN, "d1")}, [("k::u::f", "unresolved")])
        edges, _ = run(a, b)
        self.assertEqual(edges["k::u::f"][1], "AMBIGUOUS")

    def test_declaration_continues_identity_and_is_attributed(self):
        a = dump({"k::u::f": (FN, "d1")})
        b = dump({"k::o::f": (FN, "d2")}, [("k::u::f", "unresolved")])
        rb = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, dir=TMP.name)
        rb.write("k::u::f\tk::o::f\tjan\tsplit\n")
        rb.close()
        edges, ident = run("--declared", rb.name, a, b)
        self.assertEqual(edges["k::u::f"], ("k::o::f", "DECLARED"))
        self.assertEqual(ident[("r1", "k::o::f")], ident[("r0", "k::u::f")])

    def test_declaration_naming_an_existing_item_is_ignored(self):
        # A declaration may only point at an item that is new in this
        # revision; otherwise it would merge two identities.
        a = dump({"k::u::f": (FN, "d1"), "k::o::f": (FN, "d1")})
        b = dump({"k::o::f": (FN, "d1")}, [("k::u::f", "unresolved")])
        rb = tempfile.NamedTemporaryFile("w", suffix=".tsv", delete=False, dir=TMP.name)
        rb.write("k::u::f\tk::o::f\tjan\tsplit\n")
        rb.close()
        edges, ident = run("--declared", rb.name, a, b)
        self.assertNotEqual(edges["k::u::f"][1], "DECLARED")
        self.assertEqual(ident[("r1", "k::o::f")], ident[("r0", "k::o::f")])


def ident(*segs):
    return [{"Ident": [x, 0]} for x in segs]


def fun(fid, path, beg, end, impl=None):
    meta = {"name": path, "is_local": True,
            "span": {"Untagged": {"data": {"file_id": 0, "beg": {"line": beg, "col": 0},
                                           "end": {"line": end, "col": 1}}}}}
    src = {"TraitImpl": {"impl_ref": {"id": impl}}} if impl is not None else "TopLevel"
    return {"def_id": fid, "item_meta": meta, "src": src}


def adt(i):
    return {"types": [{"Untagged": {"Adt": {"id": i}}}]}


class SpanJoin(unittest.TestCase):
    """Two methods Charon names identically, a closure, and a derived impl."""

    def setUp(self):
        ra = [
            ("test", "Method", "c::impl::A::P::test", 14),
            ("test", "Method", "c::impl::B::P::test", 19),
            ("run", "Function", "c::run", 25),
            ("A", "Struct", "c::A", 10),
        ]
        self.ra = os.path.join(TMP.name, "ra.txt")
        with open(self.ra, "w") as fh:
            for name, kind, mon, line in ra:
                fh.write(f"sym\t{name}\tkind={kind}\tmoniker={mon}\timpls=-\tat=src/lib.rs:{line}\n")
        derived = {"def_id": 0, "impl_trait": {"id": 0, "generics": adt(0)},
                   "item_meta": {"is_local": True, "attr_info": {"attributes": [{"Builtin": "AutomaticallyDerived"}]}},
                   "methods": [1]}
        llbc = {"translated": {
            "files": [{"id": 0, "name": {"Local": "src/lib.rs"}}],
            "type_decls": [{"def_id": 0, "item_meta": {"name": ident("c", "A")}}],
            "trait_decls": [{"def_id": 0, "item_meta": {"name": ident("core", "clone", "Clone")}}],
            "trait_impls": [derived],
            "fun_decls": [
                fun(0, ident("c") + [{"Impl": 0}] + ident("test"), 14, 16),
                fun(1, ident("c") + [{"Impl": 1}] + ident("test"), 19, 21),
                fun(2, ident("c", "run"), 25, 28),
                fun(3, ident("c", "run") + [{"Impl": 2}] + ident("call_mut"), 26, 26),
                fun(4, ident("c") + [{"Impl": 0}] + ident("clone"), 9, 9, impl=0),
            ]}}
        self.llbc = os.path.join(TMP.name, "x.llbc")
        with open(self.llbc, "w") as fh:
            json.dump(llbc, fh)

    def joins(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            span_join.main(self.ra, self.llbc, "c")
        return {c[1]: (c[3], c[4]) for c in (line.split("\t") for line in out.getvalue().splitlines())
                if c[0] == "join"}

    def test_same_named_methods_are_told_apart_by_position(self):
        j = self.joins()
        self.assertEqual(j["0"], ("EXACT", "c::impl::A::P::test"))
        self.assertEqual(j["1"], ("EXACT", "c::impl::B::P::test"))

    def test_closure_joins_to_its_enclosing_function(self):
        self.assertEqual(self.joins()["3"], ("ENCLOSED", "c::run"))

    def test_derived_method_joins_to_its_self_type(self):
        self.assertEqual(self.joins()["4"], ("DERIVED", "c::A"))


if __name__ == "__main__":
    unittest.main()
