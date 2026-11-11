#!/usr/bin/env python3
"""CG-BEH-1: trace  code symbol -> registered algebra word -> ordered method uses -> owning classid,
and state for every hop what kind of evidence it rests on.

The probe adds no word, classid or binding and executes no method. It reads source text and a
uses file; neither is written. Commands:

  words    --src SRC                                    the POLYFILL table, one row per position
  dispatch --src SRC                                    the FoldDialect::call match arms per opcode
  impact   --src SRC --uses USES.tsv --symbol SYM [--line N] [--witness-fnv HEX] [--drop-dispatch OP]
  --self-test                                           the CG-BEH-1 negatives

Evidence vocabulary: SOURCE_STRUCTURE (read from source text: the POLYFILL array position, a
match arm of `FoldDialect::call`) and REGISTRATION (a row of the uses file, read from a BindStore
through admit() in a test run). Source evidence ends at the POLYFILL table; every `uses` row is
registration evidence. A symbol that dispatches no opcode has no behaviour edge (NONE, not "no
impact"); a word with no user in the uses file is store-scoped, not "unaffected"; a used word
whose opcode has no dispatch arm is UNKNOWN.

A missing or invalid artifact is an error (ARTIFACT_MISSING / ARTIFACT_INVALID, exit 3), never an
empty answer. A dispatch witness fingerprint that differs from the current lib.rs gives no answer
(STALE, exit 4).
"""

import argparse
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

POLY_PATH = "crates/lance-graph-r2il/src/polyfill.rs"
BIND_PATH = "crates/lance-graph-r2il/src/binding.rs"
LIB_PATH = "crates/lance-graph-r2il/src/lib.rs"
LENS = 0xFFFFF000
LENS_TEXT = "0x%08X" % LENS
DISPATCH_SYMBOL = "lance_graph_r2il::FoldDialect::call"
MAX_WORDS = 256  # AlgebraRef.word is a u8
GRADE = "SOURCE_STRUCTURE"
KEYWORDS = {"if", "while", "match", "for", "return", "let", "else", "loop", "in", "as", "fn",
            "move", "ref", "mut", "unsafe", "where", "impl", "dyn"}
RESIDUALS = ["native lowering path (binding_native::prepare_native) not mapped",
             "uses file covers one test store only",
             "realizes is arm-level source structure, not a behavioural witness"]
USES_COLS = ["class", "method", "sig_rev", "ordinal", "lens", "word", "opcode", "r2il_name",
             "arity", "target", "args"]
NONE_ROW = "behaviour\tNONE\tno behaviour edge for this symbol (not: no impact)"


class ArtifactError(Exception):
    def __init__(self, kind, msg):
        self.kind = kind
        super().__init__(f"{kind}\t{msg}")


def fnv64(data):
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return f"{h:016x}"


def invalid(msg):
    return ArtifactError("ARTIFACT_INVALID", msg)


def read_bytes(path):
    p = Path(path)
    if not p.is_file():
        raise ArtifactError("ARTIFACT_MISSING", str(p))
    return p.read_bytes()


def scan(text, keep_strings):
    """Rust source with comments (and, unless keep_strings, string and char contents) replaced by
    spaces. Length and newlines are preserved, so offsets and line numbers still apply."""
    out = list(text)
    n, i = len(text), 0
    char_re = re.compile(r"'(?:\\[^']*|[^\\'])'")
    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            for k in range(i, j):
                out[k] = " "
            i = j
        elif text.startswith("/*", i):
            depth, j = 1, i + 2
            while j < n and depth:
                if text.startswith("/*", j):
                    depth, j = depth + 1, j + 2
                elif text.startswith("*/", j):
                    depth, j = depth - 1, j + 2
                else:
                    j += 1
            for k in range(i, j):
                if text[k] != "\n":
                    out[k] = " "
            i = j
        elif c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            if not keep_strings:
                for k in range(i + 1, min(j, n)):
                    if text[k] != "\n":
                        out[k] = " "
            i = j + 1
        elif c == "'":
            m = char_re.match(text, i)
            if m:
                if not keep_strings:
                    for k in range(i + 1, m.end() - 1):
                        out[k] = " "
                i = m.end()
            else:
                i += 1
        else:
            i += 1
    return "".join(out)


def match_close(text, i):
    """Index of the bracket closing the one at text[i]."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] in "([{":
            depth += 1
        elif text[j] in ")]}":
            depth -= 1
            if depth == 0:
                return j
    raise invalid(f"unbalanced bracket at offset {i}")


def split_top(text, a, b, sep=","):
    """(start, end) spans of text[a:b] split at `sep` outside any bracket."""
    spans, depth, s = [], 0, a
    for j in range(a, b):
        ch = text[j]
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        elif ch == sep and depth == 0:
            spans.append((s, j))
            s = j + 1
    spans.append((s, b))
    return [(x, y) for x, y in spans if text[x:y].strip()]


def line_of(text, idx):
    return text.count("\n", 0, idx) + 1


def parse_executable(src):
    raw = read_bytes(Path(src) / BIND_PATH).decode()
    t = scan(raw, False)
    m = re.search(r"pub const EXECUTABLE:\s*\[FnIndex;\s*(\d+)\]\s*=\s*\[", t)
    if not m:
        raise invalid(f"{BIND_PATH}: no `pub const EXECUTABLE: [FnIndex; N]`")
    close = match_close(t, m.end() - 1)
    names = [t[a:b].strip() for a, b in split_top(t, m.end(), close)]
    if len(names) != int(m.group(1)):
        raise invalid(f"{BIND_PATH}: EXECUTABLE declares {m.group(1)} but lists {len(names)}")
    return names


def parse_words(src):
    path = Path(src) / POLY_PATH
    raw_bytes = read_bytes(path)
    raw = raw_bytes.decode()
    nostr, withstr = scan(raw, False), scan(raw, True)
    m = re.search(r"pub const POLYFILL:\s*\[FrozenWord;\s*(\d+)\]\s*=\s*\[", nostr)
    if not m:
        raise invalid(f"{POLY_PATH}: no `pub const POLYFILL: [FrozenWord; N]`")
    declared = int(m.group(1))
    close = match_close(nostr, m.end() - 1)
    entry_re = re.compile(r"\bw\s*\(")
    entries, pos = [], m.end()
    while True:
        mm = entry_re.search(nostr, pos, close)
        if not mm:
            break
        o = mm.end() - 1
        c = match_close(nostr, o)
        args = split_top(nostr, o + 1, c)
        if len(args) < 3:
            raise invalid(f"{POLY_PATH}:{line_of(raw, mm.start())}: entry has {len(args)} arguments")
        opcode = nostr[args[0][0]:args[0][1]].strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", opcode):
            raise invalid(f"{POLY_PATH}:{line_of(raw, mm.start())}: opcode {opcode!r} is not a constant name")
        try:
            arity = int(nostr[args[1][0]:args[1][1]].strip())
        except ValueError:
            raise invalid(f"{POLY_PATH}:{line_of(raw, mm.start())}: arity is not an integer")
        aliases = re.findall(r'"((?:[^"\\]|\\.)*)"', withstr[args[2][0]:args[2][1]])
        if not aliases:
            raise invalid(f"{POLY_PATH}:{line_of(raw, mm.start())}: entry has no aliases")
        entries.append({"opcode": opcode, "arity": arity, "r2il_name": aliases[0],
                        "aliases": aliases, "line": line_of(raw, mm.start())})
        pos = c + 1
    if len(entries) != declared:
        raise invalid(f"{POLY_PATH}: array type says {declared} entries, parsed {len(entries)}")
    if len(entries) > MAX_WORDS:
        raise invalid(f"{POLY_PATH}: {len(entries)} entries exceed {MAX_WORDS} (word is a u8)")
    exe = parse_executable(src)
    if exe != [e["opcode"] for e in entries]:
        raise invalid(f"{BIND_PATH}: EXECUTABLE order differs from POLYFILL opcode order")
    digest = fnv64(raw_bytes)
    for p, e in enumerate(entries):
        e.update({"word": p, "file": POLY_PATH, "fnv": digest})
    return entries


CONST_RE = re.compile(r"(?:[A-Za-z_][A-Za-z0-9_]*::)*([A-Z][A-Z0-9_]+)")
BIND_RE = re.compile(r"[a-z_][A-Za-z0-9_]*")


def arm_opcodes(pattern):
    pattern = re.sub(r"\s+if\s+.*$", "", pattern.strip(), flags=re.S)
    ops = []
    for a, b in split_top(pattern, 0, len(pattern), "|"):
        alt = pattern[a:b].strip()
        m = CONST_RE.fullmatch(alt)
        if m:
            op = m.group(1)
        elif alt == "_" or BIND_RE.fullmatch(alt):
            op = "*"
        else:
            continue
        if op not in ops:
            ops.append(op)
    return ops


def arm_callees(body):
    found = set()
    for m in re.finditer(r"\bself\.([a-z_]\w*)\s*\(", body):
        found.add(m.group(1))
    for m in re.finditer(r"(?<![\w.:])([a-z_]\w*)\s*\(", body):
        if m.group(1) not in KEYWORDS:
            found.add(m.group(1))
    return sorted(found)


def parse_dispatch(src):
    raw_bytes = read_bytes(Path(src) / LIB_PATH)
    raw = raw_bytes.decode()
    t = scan(raw, False)
    m = re.search(r"\bimpl(?:<[^>]*>)?\s+FoldDialect\b[^{;]*\{", t)
    if not m:
        raise invalid(f"{LIB_PATH}: no inherent `impl FoldDialect`")
    f = t.find("pub fn call(", m.end())
    if f < 0:
        raise invalid(f"{LIB_PATH}: no `pub fn call(` in `impl FoldDialect`")
    sig_close = match_close(t, f + len("pub fn call"))
    body_open = t.find("{", sig_close)
    mm = re.compile(r"\bmatch\s+f\s*\{").search(t, body_open)
    if body_open < 0 or not mm:
        raise invalid(f"{LIB_PATH}: no `match f {{` in `FoldDialect::call`")
    open_i = mm.end() - 1
    close_i = match_close(t, open_i)
    rows, i = [], open_i + 1
    while True:
        while i < close_i and (t[i].isspace() or t[i] == ","):
            i += 1
        if i >= close_i:
            break
        pat_start, depth, j = i, 0, i
        while j < close_i:
            ch = t[j]
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif depth == 0 and t.startswith("=>", j):
                break
            j += 1
        else:
            raise invalid(f"{LIB_PATH}:{line_of(raw, pat_start)}: match arm without `=>`")
        pattern = t[pat_start:j]
        k = j + 2
        while t[k].isspace():
            k += 1
        if t[k] == "{":
            end = match_close(t, k)
            k = end + 1
        else:
            d = 0
            while k < close_i:
                ch = t[k]
                if ch in "([{":
                    d += 1
                elif ch in ")]}":
                    d -= 1
                elif ch == "," and d == 0:
                    break
                k += 1
            end = k - 1
            while t[end].isspace():
                end -= 1
        body = t[j + 2:end + 1]
        arm_line, arm_end = line_of(raw, pat_start), line_of(raw, end)
        callees = "|".join(arm_callees(body))
        for op in arm_opcodes(pattern):
            rows.append({"opcode": op, "symbol": DISPATCH_SYMBOL, "file": LIB_PATH,
                         "arm_line": arm_line, "arm_end": arm_end, "callees": callees,
                         "fnv": fnv64(raw_bytes), "grade": GRADE,
                         "note": "match arm; one symbol dispatches every opcode"})
        while k < close_i and t[k].isspace():
            k += 1
        if k < close_i and t[k] == ",":
            k += 1
        i = k
    return rows


def load_uses(path, nwords):
    p = Path(path)
    if not p.is_file():
        raise ArtifactError("ARTIFACT_MISSING", str(p))
    cols, rows = None, []
    for n, line in enumerate(p.read_text().splitlines(), 1):
        if line.startswith("#schema "):
            cols = line[len("#schema "):].split("\t")
        elif line and not line.startswith("#"):
            if cols is None or any(c not in cols for c in USES_COLS):
                raise invalid(f"{p}:{n}: uses file has no #schema with columns {USES_COLS}")
            r = dict(zip(cols, line.split("\t")))
            try:
                wi, ordinal = int(r["word"]), int(r["ordinal"])
                lens = int(r["lens"], 0)
            except (KeyError, ValueError):
                raise invalid(f"{p}:{n}: word, ordinal or lens is not an integer")
            if lens != LENS:
                raise invalid(f"{p}:{n}: lens {r['lens']} is not the polyfill lens {LENS_TEXT}")
            if wi >= nwords:
                raise invalid(f"{p}:{n}: word {wi} is outside the {nwords} words of lens {LENS_TEXT}")
            r["word"], r["ord_int"] = wi, ordinal
            rows.append(r)
    return rows


def tsv(*cols):
    return "\t".join(str(c) for c in cols)


def cmd_words(src):
    words = parse_words(src)
    lines = ["#cg-beh v1",
             "#schema lens\tword\topcode\tarity\tr2il_name\taliases\tfile\tline\tfile_fnv64\tgrade"]
    for w in words:
        lines.append(tsv(LENS_TEXT, w["word"], w["opcode"], w["arity"], w["r2il_name"],
                         "|".join(w["aliases"]), w["file"], w["line"], w["fnv"], GRADE))
    return "\n".join(lines) + "\n"


def cmd_dispatch(src):
    lines = ["#cg-beh v1",
             "#schema opcode\tsymbol\tfile\tarm_line\tarm_end\tcallees\tfile_fnv64\tgrade\tnote"]
    for d in parse_dispatch(src):
        lines.append(tsv(d["opcode"], d["symbol"], d["file"], d["arm_line"], d["arm_end"],
                         d["callees"], d["fnv"], d["grade"], d["note"]))
    return "\n".join(lines) + "\n"


def impact(src, uses_path, symbol, line=None, witness_fnv=None, drop_dispatch=None):
    """Pure: (exit_code, text) for one symbol. Nothing is read beyond SRC and the uses file."""
    try:
        words = parse_words(src)
        disp = parse_dispatch(src)
        uses = load_uses(uses_path, len(words))
    except ArtifactError as e:
        return 3, str(e) + "\n"
    current = disp[0]["fnv"] if disp else fnv64(read_bytes(Path(src) / LIB_PATH))
    if witness_fnv is not None and witness_fnv != current:
        return 4, "STALE\tdispatch witness is for another revision of lib.rs; no current answer\n"
    if drop_dispatch:
        disp = [d for d in disp if d["opcode"] != drop_dispatch]
    arms = [d for d in disp if d["symbol"] == symbol]
    if line is not None:
        arms = [d for d in arms if d["arm_line"] <= line <= d["arm_end"]]
    if not arms:
        return 0, NONE_ROW + "\n"
    by_op = {w["opcode"]: w for w in words}
    dispatched = {d["opcode"] for d in disp}
    out = ["#cg-beh v1", tsv("query", symbol, "-" if line is None else line)]
    seen, hit = set(), []
    for d in arms:
        out.append(tsv("realizes", d["symbol"], f"{d['arm_line']}-{d['arm_end']}", d["opcode"], GRADE,
                       "dispatch match arm (read from source; not a compiler call fact, not a behavioural test)"))
        op = d["opcode"]
        if op in seen:
            continue
        seen.add(op)
        w = by_op.get(op)
        if w is None:
            out.append(tsv("opcode", op, f"not a word of lens {LENS_TEXT}"))
            continue
        p = w["word"]
        out.append(tsv("word", LENS_TEXT, p, w["r2il_name"], GRADE, "position in the POLYFILL table"))
        us = sorted([u for u in uses if u["word"] == p],
                    key=lambda u: (u["class"], u["method"], u["ord_int"]))
        if not us:
            out.append(tsv("word", p, w["r2il_name"],
                           "no registered user in this uses file (store-scoped, not: unaffected)"))
        for u in us:
            out.append(tsv("uses", u["class"], u["method"], u["sig_rev"], u["ordinal"], p, "REGISTRATION",
                           "read from a BindStore through admit() in a test run"))
            hit.append(u)
    out.append("boundary\tsource evidence ends at the POLYFILL table; everything in 'uses' rows is DT64 registration evidence")
    methods = {(u["class"], u["method"], u["sig_rev"]) for u in hit}
    classes = {u["class"] for u in hit}
    out.append(tsv("affected_methods", len(methods)))
    out.append("affected_classes\t" + ",".join(sorted(classes)))
    out.append("coverage\tPARTIAL\t" + "; ".join(RESIDUALS))
    used_words = {u["word"] for u in uses}
    for w in words:
        if w["word"] in used_words and w["opcode"] not in dispatched:
            out.append(tsv("UNKNOWN", f"word {w['word']} {w['r2il_name']}",
                           "no dispatch witness; its users cannot be confirmed or excluded"))
    return 0, "\n".join(out) + "\n"


# ---------------------------------------------------------------- self-test

FAKE_POLY = '''pub const POLYFILL_LENS: u32 = 0;
pub const POLYFILL: [FrozenWord; 3] = [
    w(INT_ADD, 2, &["IntAdd", "+"], Some(1), None),
    w(
        INT_S_LESS,
        2,
        &["IntSLess", "<", "Cmp::LtI32"],
        Some(1),
        Some(2),
    ),
    w(POP_COUNT, 1, &["PopCount", "COUNT"], Some(1), None),
];
'''

FAKE_BIND = '''/// Executable subset.
pub const EXECUTABLE: [FnIndex; 3] = [
    INT_ADD, INT_S_LESS, POP_COUNT,
];
'''

FAKE_LIB = '''use x;

pub struct FoldDialect<'a> {
    cell: &'a u8,
}

impl Dialect for FoldDialect<'_> {
    fn call(&mut self, f: FnIndex, v: [u8; 3], stack: &mut Vec<Val>) -> Result<(), FoldError> {
        FoldDialect::call(self, f, v, stack)
    }
}

fn int_s_less(a: i64, b: i64) -> bool {
    a < b
}

impl FoldDialect<'_> {
    pub fn call(&mut self, f: FnIndex, v: [u8; 3], stack: &mut Vec<Val>) -> Result<(), FoldError> {
        match f {
            FnIndex::NUMBER => {
                stack.push(Val::Scalar(i64::from(v[0])));
                Ok(())
            }
            INT_ADD | INT_S_LESS => {
                // INT_NOT => decoy in a comment
                let b = stack.pop().ok_or(FoldError::Underflow(f))?;
                self.emit_pred(b, stack);
                helper(b);
                Ok(())
            }
            POP_COUNT => self.count(stack),
            other => Err(FoldError::Unimplemented(other)),
        }
    }
}
'''

UCOLS = "#schema " + "\t".join(USES_COLS) + "\n"


def urow(cls, method, ordinal, word, lens=LENS_TEXT, opcode="X", name="N"):
    return "\t".join([cls, method, "r1", str(ordinal), lens, str(word), opcode, name, "2", "t", "a"]) + "\n"


def mk_tree(td, poly=FAKE_POLY, bind=FAKE_BIND, lib=FAKE_LIB):
    root = Path(td) / "src"
    for rel, body in ((POLY_PATH, poly), (BIND_PATH, bind), (LIB_PATH, lib)):
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return root


def mk_uses(td, rows, name="uses.tsv"):
    p = Path(td) / name
    p.write_text("#cg-beh-uses v1\n" + UCOLS + "".join(rows))
    return p


BASE_USES = [urow("ClsA", "m1", 0, 0, opcode="INT_ADD", name="IntAdd"),
             urow("ClsB", "m2", 1, 2, opcode="POP_COUNT", name="PopCount")]


class Negatives(unittest.TestCase):
    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.src = mk_tree(self.td.name)
        self.uses = mk_uses(self.td.name, BASE_USES)

    def tearDown(self):
        self.td.cleanup()

    def run_impact(self, symbol=DISPATCH_SYMBOL, **kw):
        return impact(self.src, self.uses, symbol, **kw)

    def test_01_missing_dispatch_mapping_is_unknown_not_unaffected(self):
        """A used word with no dispatch arm must be UNKNOWN, and other words' rows must not move."""
        _, base = self.run_impact()
        code, dropped = self.run_impact(drop_dispatch="POP_COUNT")
        self.assertEqual(code, 0)
        self.assertIn("UNKNOWN\tword 2 PopCount\tno dispatch witness", dropped)
        self.assertNotIn("UNKNOWN", base)

        def stable(text):
            return [l for l in text.splitlines()
                    if l.split("\t")[0] in ("realizes", "word", "uses")
                    and "POP_COUNT" not in l and "PopCount" not in l and "\t2\t" not in l
                    and "\tm2\t" not in l]
        self.assertEqual(stable(base), stable(dropped))
        self.assertNotIn("\tm2\t", dropped)

    def test_02_stale_witness_gives_no_answer(self):
        """A witness for another lib.rs revision must exit 4 and print no realizes or uses row."""
        code, text = self.run_impact(witness_fnv="0000000000000000")
        self.assertEqual(code, 4)
        self.assertTrue(text.startswith("STALE\t"))
        self.assertNotIn("realizes", text)
        self.assertNotIn("uses\t", text)
        cur = fnv64((self.src / LIB_PATH).read_bytes())
        self.assertEqual(self.run_impact(witness_fnv=cur)[0], 0)

    def test_03_a_matching_name_is_not_a_behaviour_edge(self):
        """A function named like a word but outside the dispatch match must give NONE."""
        for sym in ("int_s_less", "lance_graph_r2il::int_s_less"):
            code, text = self.run_impact(sym)
            self.assertEqual(code, 0)
            self.assertIn("behaviour\tNONE", text)
            self.assertNotIn("realizes", text)

    def test_04_a_word_without_user_is_stated_not_omitted(self):
        """A dispatched word with no registered user must be reported as store-scoped, not unaffected."""
        _, text = self.run_impact()
        self.assertIn("no registered user in this uses file (store-scoped, not: unaffected)", text)
        self.assertIn("word\t1\tIntSLess\tno registered user", text)

    def test_05_a_non_dispatching_symbol_has_no_behaviour_edge(self):
        """A symbol that dispatches nothing must print `behaviour NONE ... (not: no impact)`."""
        code, text = self.run_impact("lance_graph_r2il::nothing")
        self.assertEqual(code, 0)
        self.assertEqual(text, NONE_ROW + "\n")

    def test_06_polyfill_order_must_match_executable(self):
        """POLYFILL opcode order differing from EXECUTABLE must be ARTIFACT_INVALID."""
        src = mk_tree(Path(self.td.name) / "b", bind=FAKE_BIND.replace("INT_ADD, INT_S_LESS,", "INT_S_LESS, INT_ADD,"))
        code, text = impact(src, self.uses, DISPATCH_SYMBOL)
        self.assertEqual(code, 3)
        self.assertIn("ARTIFACT_INVALID", text)
        with self.assertRaises(ArtifactError):
            parse_words(src)

    def test_07_a_word_outside_the_lens_is_invalid(self):
        """A uses row whose word is >= the word count must be ARTIFACT_INVALID."""
        uses = mk_uses(self.td.name, [urow("ClsA", "m1", 0, 3)], "u7.tsv")
        code, text = impact(self.src, uses, DISPATCH_SYMBOL)
        self.assertEqual(code, 3)
        self.assertIn("ARTIFACT_INVALID", text)

    def test_08_another_lens_is_invalid(self):
        """A uses row naming a lens other than 0xFFFFF000 must be ARTIFACT_INVALID."""
        uses = mk_uses(self.td.name, [urow("ClsA", "m1", 0, 0, lens="0xFFFFF100")], "u8.tsv")
        code, text = impact(self.src, uses, DISPATCH_SYMBOL)
        self.assertEqual(code, 3)
        self.assertIn("ARTIFACT_INVALID", text)

    def test_09_same_input_gives_identical_output(self):
        """Two runs on one input, in-process and across processes, must be byte-identical."""
        rows = [urow(c, "m", i, 0) for i, c in enumerate(("Z", "A", "M", "Q", "B"))]
        uses = mk_uses(self.td.name, rows, "u9.tsv")
        self.assertEqual(impact(self.src, uses, DISPATCH_SYMBOL), impact(self.src, uses, DISPATCH_SYMBOL))
        argv = [sys.executable, "-I", __file__, "impact", "--src", str(self.src), "--uses", str(uses),
                "--symbol", DISPATCH_SYMBOL]
        outs = {subprocess.run(argv, capture_output=True, check=True).stdout for _ in range(4)}
        self.assertEqual(len(outs), 1)
        self.assertIn(b"affected_classes\tA,B,M,Q,Z\n", outs.pop())

    def test_10_line_selects_one_arm(self):
        """--line must select only the arm containing it; a line in no arm must give NONE."""
        lines = FAKE_LIB.splitlines()
        pop = next(i for i, l in enumerate(lines, 1) if l.startswith("            POP_COUNT =>"))
        add = next(i for i, l in enumerate(lines, 1) if "INT_ADD | INT_S_LESS" in l) + 2
        _, text = self.run_impact(line=pop)
        self.assertIn("\tPOP_COUNT\t", text)
        self.assertNotIn("INT_ADD", text)
        _, both = self.run_impact(line=add)
        self.assertIn("\tINT_ADD\t", both)
        self.assertIn("\tINT_S_LESS\t", both)
        self.assertNotIn("POP_COUNT", both)
        code, none = self.run_impact(line=1)
        self.assertEqual((code, none), (0, NONE_ROW + "\n"))

    def test_11_uses_print_in_ordinal_order(self):
        """Uses listed in reverse file order must print by ordinal, with the ordinal column kept."""
        rows = [urow("ClsA", "m1", o, 0) for o in (2, 0, 1)]
        uses = mk_uses(self.td.name, rows, "u11.tsv")
        _, text = impact(self.src, uses, DISPATCH_SYMBOL)
        ords = [l.split("\t")[4] for l in text.splitlines() if l.startswith("uses\t")]
        self.assertEqual(ords, ["0", "1", "2"])

    def test_12_missing_uses_file_is_not_an_empty_impact(self):
        """A missing uses file must be ARTIFACT_MISSING with exit 3."""
        code, text = impact(self.src, Path(self.td.name) / "absent.tsv", DISPATCH_SYMBOL)
        self.assertEqual(code, 3)
        self.assertTrue(text.startswith("ARTIFACT_MISSING"))

    def test_13_dispatch_parse_reads_arms_not_comments_or_other_receivers(self):
        """The arm table must hold the constants of the real patterns, the catch-all, and self/bare callees only."""
        rows = parse_dispatch(self.src)
        self.assertEqual([r["opcode"] for r in rows], ["NUMBER", "INT_ADD", "INT_S_LESS", "POP_COUNT", "*"])
        by = {r["opcode"]: r for r in rows}
        self.assertEqual(by["INT_ADD"]["callees"], "emit_pred|helper")
        self.assertEqual(by["POP_COUNT"]["callees"], "count")
        self.assertEqual(by["NUMBER"]["callees"], "")
        self.assertEqual((by["INT_ADD"]["arm_line"], by["INT_ADD"]["arm_end"]), (24, 30))


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("words", "dispatch", "impact"):
        sp = sub.add_parser(name)
        sp.add_argument("--src", required=True)
        if name == "impact":
            sp.add_argument("--uses", required=True)
            sp.add_argument("--symbol", required=True)
            sp.add_argument("--line", type=int)
            sp.add_argument("--witness-fnv")
            sp.add_argument("--drop-dispatch")
    a = ap.parse_args(argv)
    if a.self_test:
        r = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Negatives))
        return 0 if r.wasSuccessful() else 1
    if a.cmd is None:
        ap.print_help()
        return 2
    try:
        if a.cmd == "words":
            sys.stdout.write(cmd_words(a.src))
        elif a.cmd == "dispatch":
            sys.stdout.write(cmd_dispatch(a.src))
        else:
            code, text = impact(a.src, a.uses, a.symbol, a.line, a.witness_fnv, a.drop_dispatch)
            sys.stdout.write(text)
            return code
    except ArtifactError as e:
        print(e)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
