#!/usr/bin/env python3
"""Tests for kythe_entries.py on hand-written occ rows.

    python3 -I tools/test_kythe_entries.py
"""
import base64
import contextlib
import io
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import kythe_entries  # noqa: E402

SRC = "fn foo() {}\nfn bar() { foo(); }\n"


def occ(m, kind, s, e, role, path="src/lib.rs"):
    return f"occ\t{m}\t{kind}\t{path}\t{s}\t{e}\t{role}"


class Entries(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(self.dir.name, "src"))
        with open(os.path.join(self.dir.name, "src/lib.rs"), "w") as f:
            f.write(SRC)

    def tearDown(self):
        self.dir.cleanup()

    def build(self, lines):
        return kythe_entries.build(lines, self.dir.name, "c")

    def facts(self, entries, name):
        return [e for e in entries if e.get("fact_name") == name]

    def test_counts_and_roundtrip(self):
        entries, skipped = self.build([
            occ("k::foo", "Function", 3, 6, "def"),
            occ("k::foo", "Function", 23, 26, "ref"),
        ])
        # file(2) + symbol(1) + 2 anchors x (3 facts + 1 edge)
        self.assertEqual(len(entries), 3 + 8)
        self.assertEqual(skipped, {})
        text = self.facts(entries, "/kythe/text")[0]
        self.assertEqual(base64.b64decode(text["fact_value"]).decode(), SRC)
        starts = sorted(base64.b64decode(e["fact_value"]) for e in self.facts(entries, "/kythe/loc/start"))
        self.assertEqual(starts, [b"23", b"3"])

    def test_edge_kinds_and_shapes(self):
        entries, _ = self.build([
            occ("k::foo", "Function", 3, 6, "def"),
            occ("k::foo", "Function", 23, 26, "ref"),
        ])
        edges = {e["source"]["signature"]: e for e in entries if "edge_kind" in e}
        self.assertEqual(edges["@3:6"]["edge_kind"], "/kythe/edge/defines/binding")
        self.assertEqual(edges["@23:26"]["edge_kind"], "/kythe/edge/ref")
        for e in edges.values():
            self.assertEqual(e["fact_name"], "/")
            self.assertEqual(e["target"]["signature"], "k::foo")
            self.assertEqual(e["target"]["language"], "rust")
            self.assertEqual(e["source"]["path"], "src/lib.rs")
        self.assertFalse(any("call" in e.get("edge_kind", "") for e in entries))

    def test_dedup(self):
        row = occ("k::foo", "Function", 3, 6, "def")
        once, _ = self.build([row])
        twice, _ = self.build([row, row])
        self.assertEqual(once, twice)

    def test_unknown_kind_is_skipped_and_counted(self):
        entries, skipped = self.build([
            occ("k::x", "Weird", 3, 6, "def"),
            occ("k::foo", "Function", 3, 6, "def"),
        ])
        self.assertEqual(skipped, {"Weird": {"k::x"}})
        self.assertFalse(any(e["source"]["signature"] == "k::x" for e in entries))
        self.assertTrue(any(e["source"]["signature"] == "k::foo" for e in entries))

    def test_non_occ_lines_ignored(self):
        entries, _ = self.build(["sym\tfoo\tkind=Function", "", "garbage", "occ\tshort"])
        self.assertEqual(entries, [])

    def test_node_kind_table(self):
        entries, _ = self.build([occ("k::T", "Trait", 3, 6, "def")])
        kinds = [base64.b64decode(e["fact_value"]) for e in self.facts(entries, "/kythe/node/kind")
                 if e["source"]["signature"] == "k::T"]
        self.assertEqual(kinds, [b"interface"])

    def test_range_rejected(self):
        for s, e in ((3, 999), (6, 3)):
            with self.assertRaises(ValueError):
                self.build([occ("k::foo", "Function", s, e, "def")])

    def test_cli_exit_2_on_bad_range(self):
        path = os.path.join(self.dir.name, "occ.txt")
        with open(path, "w") as f:
            f.write(occ("k::foo", "Function", 3, 999, "def") + "\n")
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            kythe_entries.main(["--root", self.dir.name, path])
        self.assertEqual(cm.exception.code, 2)
        self.assertIn("bad range", err.getvalue())


if __name__ == "__main__":
    unittest.main()
