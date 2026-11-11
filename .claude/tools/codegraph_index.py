#!/usr/bin/env python3
"""Generate the `.codegraph/` memory directory from the repo itself.

WHY GENERATED, AND WHY IT IS ONLY A PROJECTION. A session that has to answer
"who depends on X", "where is ClassView defined", "what imports this path"
re-derives the answer by grepping, and the answer drifts between sessions. This
projects source-backed facts into line-oriented TSV + small Markdown so
`rg -n NAME .codegraph/` returns whole facts. It is NOT an authority: the code
stays the truth, there is no database, no symbol registry, and nothing here is
parsed back. Calls are never analysed here -- they come ONLY from existing
ruff_charon store artifacts, and when those are absent the output says
ARTIFACT_MISSING instead of guessing.

    python3 .claude/tools/codegraph_index.py --write
    python3 .claude/tools/codegraph_index.py --check       # CI / freshness
    python3 .claude/tools/codegraph_index.py --self-test
    [--charon-store DIR]... [--charon-src DIR] [--view SYMBOL]... [--root DIR]

`--write` and `--check` must be given the same --charon-* options: calls.tsv is
a function of them. `.codegraph/architecture.tsv` is HAND-MAINTAINED input; this
tool validates and projects it (decisions.tsv, gaps.tsv) and never writes it
(except creating a header-only file when absent).

Determinism: same inputs give byte-identical output. Everything is sorted by
value, never by discovery order; no timestamps, no absolute paths, no HEAD sha
(a commit cannot contain its own hash -- same rule as PROCESSED_THROUGH). The
per-crate freshness is a digest of (relpath, git blob id) pairs.
"""
import argparse
import hashlib
import os
import re
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT = ".codegraph"
DEFAULT_VIEWS = ["ClassView", "WideFieldMask", "LaneBinding", "MethodBinding",
                 "BindStore", "LaneSource", "FrameStore"]
ARCH_STATUS = {"CONFIRMED", "LAB", "OPEN-PR", "PROPOSED", "GAP"}
ARCH_HEADER = "subject\trelation\tobject\tstatus\tevidence\tanchor\tnote"
MISSING = "ARTIFACT_MISSING"
CAPS = {"public": 200, "imports": 100, "calls": 100, "deps": 80, "occ": 300}

# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def clean(v):
    return str(v).replace("\t", " ").replace("\n", " ").replace("\r", " ")


def tsv(schema, rows):
    out = ["#codegraph v1", "#schema " + "\t".join(schema)]
    for r in sorted(set(tuple(clean(c) for c in row) for row in rows)):
        assert len(r) == len(schema), (schema, r)
        out.append("\t".join(r))
    return "\n".join(out) + "\n"


def blob_id(data):
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def fnv64(data):
    """FNV-1a 64 over the file bytes -- the ruff_charon files.tsv hash
    (verified: rev1 cg_app/src/lib.rs -> 0effe8b102e60ce2)."""
    h = 0xCBF29CE484222325
    for c in data:
        h = ((h ^ c) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return "%016x" % h


def lib_name(crate):
    return crate.replace("-", "_")


# --------------------------------------------------------------------------
# Rust lexer: comments/strings stripped, line numbers kept
# --------------------------------------------------------------------------

_TOK = re.compile(
    r"(?P<ws>[ \t\r\f\v]+)|(?P<nl>\n)|(?P<lc>//[^\n]*)|(?P<bc>/\*)"
    r"|(?P<ident>[^\W\d]\w*)|(?P<num>\d\w*(?:\.\d\w*)*)|(?P<str>\")|(?P<chr>')|(?P<p>.)",
    re.S)
_OPEN = {"(": ")", "[": "]", "{": "}"}
_CLOSE = {")", "]", "}"}


def _scan_str(s, j, line):
    n = len(s)
    while j < n:
        c = s[j]
        if c == "\\":
            if j + 1 < n and s[j + 1] == "\n":
                line += 1
            j += 2
            continue
        if c == "\n":
            line += 1
        if c == '"':
            return j + 1, line
        j += 1
    return n, line


def _scan_raw(s, j, hashes, line):
    close = '"' + "#" * hashes
    k = s.find(close, j)
    if k < 0:
        return len(s), line + s.count("\n", j)
    return k + len(close), line + s.count("\n", j, k)


def lex(s):
    """-> (tokens, pair). token = (kind, text, line); kind i|p|l. pair maps the
    index of each open bracket to its closer."""
    toks = []
    pair = {}
    stack = []
    i, n, line = 0, len(s), 1
    match = _TOK.match
    while i < n:
        m = match(s, i)
        g = m.lastgroup
        if g == "ws":
            i = m.end()
        elif g == "nl":
            line += 1
            i += 1
        elif g == "lc":
            i = m.end()
        elif g == "bc":
            depth, j = 1, i + 2
            while j < n and depth:
                if s.startswith("/*", j):
                    depth += 1
                    j += 2
                elif s.startswith("*/", j):
                    depth -= 1
                    j += 2
                else:
                    if s[j] == "\n":
                        line += 1
                    j += 1
            i = j
        elif g == "ident":
            w = m.group()
            e = m.end()
            nx = s[e] if e < n else ""
            if w in ("r", "br", "cr") and (nx == '"' or nx == "#"):
                k = e
                while k < n and s[k] == "#":
                    k += 1
                if k < n and s[k] == '"':
                    st = line
                    j, line = _scan_raw(s, k + 1, k - e, line)
                    toks.append(("l", s[i:j], st))
                    i = j
                    continue
                if w == "r" and nx == "#" and k == e + 1 and k < n and re.match(r"[^\W\d]", s[k]):
                    m2 = _TOK.match(s, k)
                    toks.append(("i", m2.group(), line))
                    i = m2.end()
                    continue
            if w in ("b", "c") and nx == '"':
                st = line
                j, line = _scan_str(s, e + 1, line)
                toks.append(("l", s[i:j], st))
                i = j
                continue
            if w == "b" and nx == "'":
                j = e + 1
                while j < n:
                    if s[j] == "\\":
                        j += 2
                        continue
                    if s[j] == "'":
                        break
                    j += 1
                toks.append(("l", s[i:j + 1], line))
                i = j + 1
                continue
            toks.append(("i", w, line))
            i = e
        elif g == "num":
            toks.append(("l", m.group(), line))
            i = m.end()
        elif g == "str":
            st = line
            j, line = _scan_str(s, i + 1, line)
            toks.append(("l", s[i:j], st))
            i = j
        elif g == "chr":
            nx = s[i + 1] if i + 1 < n else ""
            if nx == "\\":
                j = i + 1
                while j < n:
                    if s[j] == "\\":
                        j += 2
                        continue
                    if s[j] == "'":
                        break
                    j += 1
                toks.append(("l", s[i:j + 1], line))
                i = j + 1
            elif i + 2 < n and s[i + 2] == "'":
                toks.append(("l", s[i:i + 3], line))
                i += 3
            else:
                m2 = re.compile(r"'\w*").match(s, i)
                toks.append(("l", m2.group(), line))
                i = m2.end()
        else:
            c = m.group()
            idx = len(toks)
            toks.append(("p", c, line))
            if c in _OPEN:
                stack.append(idx)
            elif c in _CLOSE and stack:
                pair[stack.pop()] = idx
            i += 1
    return toks, pair


def join_tokens(toks):
    out, prev = [], False
    for k, t, _ in toks:
        word = k in "il"
        if word and prev:
            out.append(" ")
        out.append(t)
        prev = word
    return "".join(out)


def join_cfg(toks):
    out = []
    for _, t, _ in toks:
        out.append(t)
        if t == ",":
            out.append(" ")
    return "".join(out).strip()


def combine_cfg(parts):
    parts = list(dict.fromkeys(p for p in parts if p and p != "-"))  # dedupe, keep order
    if not parts:
        return "-"
    return parts[0] if len(parts) == 1 else "all(" + ", ".join(parts) + ")"


_QUAL = {"unsafe", "async", "const", "extern"}


def module_cfgs(files):
    """Cfg each file inherits from the `#[cfg] mod name;` that declares it, keyed by
    repo-relative path. Resolution follows the default module layout
    (`dir/name.rs` or `dir/name/mod.rs`); `#[path]` and `mod` declarations nested
    in an inline module are not followed."""
    def owner_dir(rel):
        d, base = rel.rsplit("/", 1)
        return d if base in ("lib.rs", "main.rs", "mod.rs") else d + "/" + base[:-3]
    # Recompute from scratch each pass until stable: a parent may sort after its
    # child, and a file declared from two places (lib.rs and main.rs) keeps every gate.
    inherited = {}
    for _ in range(len(files) + 1):
        nxt = {}
        for rel in sorted(files):
            mine = inherited.get(rel, "-")
            for name, cfg in files[rel].mod_decls:
                gate = combine_cfg([mine, cfg])
                if gate == "-":
                    continue
                for child in (f"{owner_dir(rel)}/{name}.rs", f"{owner_dir(rel)}/{name}/mod.rs"):
                    if child in files:
                        nxt.setdefault(child, set()).add(gate)
        nxt = {k: combine_cfg(sorted(v)) if len(v) == 1 else "any(" + ", ".join(sorted(v)) + ")"
               for k, v in nxt.items()}
        if nxt == inherited:
            break
        inherited = nxt
    return inherited


class RustFile:
    """Item-level facts of one source file. Items produced by macros and `use`
    inside function bodies are invisible by construction."""

    def __init__(self, text):
        self.T, self.pair = lex(text)
        self.symbols = []   # (line, kind, name, vis, cfg)
        self.imports = []   # (line, path, alias, kind, cfg)
        self.mod_decls = []  # (name, cfg) for `mod name;` (body in another file)
        self.macro_rules = sum(
            1 for i, t in enumerate(self.T[:-1])
            if t[0] == "i" and t[1] == "macro_rules" and self.T[i + 1][1] == "!")
        self.parse_items(0, len(self.T), "-")

    # -- helpers ---------------------------------------------------------
    def tx(self, i):
        return self.T[i][1] if i < len(self.T) else ""

    def skip_to_semi(self, k, e):
        while k < e:
            t = self.tx(k)
            if t in ("(", "["):
                k = self.pair.get(k, k) + 1
            elif t == "{":
                k = self.pair.get(k, k) + 1
            elif t == ";":
                return k + 1
            else:
                k += 1
        return e

    def skip_item(self, k, e):
        while k < e:
            t = self.tx(k)
            if t in ("(", "["):
                k = self.pair.get(k, k) + 1
            elif t == "{":
                return self.pair.get(k, k) + 1
            elif t == ";":
                return k + 1
            else:
                k += 1
        return e

    def attrs(self, i, e):
        # Outer `#[cfg]` gates the next item; inner `#![cfg]` gates the whole
        # enclosing scope (file or inline module), so it is returned apart.
        cfgs, inner, export = [], [], False
        while i < e and self.tx(i) == "#":
            j = i + 1
            is_inner = self.tx(j) == "!"
            if is_inner:
                j += 1
            if self.tx(j) == "[" and j in self.pair:
                c = self.pair[j]
                if self.tx(j + 1) == "cfg" and self.tx(j + 2) == "(" and (j + 2) in self.pair:
                    (inner if is_inner else cfgs).append(join_cfg(self.T[j + 3:self.pair[j + 2]]))
                elif self.tx(j + 1) == "macro_export":
                    export = True
                i = c + 1
            else:
                break
        return i, cfgs, inner, export

    def vis(self, i):
        if self.tx(i) != "pub":
            return "private", i
        i += 1
        if self.tx(i) == "(" and i in self.pair:
            c = self.pair[i]
            inner = join_tokens(self.T[i + 1:c])
            v = {"crate": "pub(crate)", "super": "pub(super)", "self": "private"}.get(inner, "pub(in)")
            return v, c + 1
        return "pub", i

    def emit(self, line, kind, name, vis, cfg):
        if name and name != "_" and vis in ("pub", "pub(crate)"):
            self.symbols.append((line, kind, name, vis, cfg))

    # -- module-level items ---------------------------------------------
    def parse_items(self, i, e, cfg_in):
        T = self.T
        while i < e:
            start = i
            i, cfgs, inner, export = self.attrs(i, e)
            if inner:
                cfg_in = combine_cfg([cfg_in] + inner)
            if i >= e:
                break
            cfg = combine_cfg([cfg_in] + cfgs)
            vis, i = self.vis(i)
            j = i
            foreign = False
            while j < e and T[j][0] == "i" and T[j][1] in _QUAL:
                t = T[j][1]
                nx = self.tx(j + 1)
                if t == "const" and nx not in ("fn", "unsafe", "async", "extern"):
                    break
                if t == "extern":
                    if nx == "crate":
                        j += 1
                        break
                    j += 1
                    if j < e and T[j][0] == "l":
                        j += 1
                    if self.tx(j) == "{" and j in self.pair:
                        self.parse_items(j + 1, self.pair[j], cfg)
                        i = self.pair[j] + 1
                        foreign = True
                    break
                j += 1
            if foreign:
                continue
            kw = T[j][1] if j < e and T[j][0] == "i" else ""
            line = T[j][2] if j < e else 0
            i = j
            if kw == "fn":
                self.emit(line, "fn", self.tx(j + 1), vis, cfg)
                i = self.skip_item(j + 1, e)
            elif kw in ("struct", "enum", "trait", "type", "static", "const", "union"):
                k = j + 1
                if kw == "union" and not (T[k][0] == "i" if k < len(T) else False):
                    i = j + 1
                    continue
                if kw == "static" and self.tx(k) == "mut":
                    k += 1
                if kw == "trait" and self.tx(k) == "auto":
                    k += 1
                self.emit(T[k][2] if k < len(T) else line, kw, self.tx(k), vis, cfg)
                i = self.skip_to_semi(k, e) if kw in ("type", "static", "const") else self.skip_item(k, e)
            elif kw == "mod":
                name = self.tx(j + 1)
                self.emit(line, "mod", name, vis, cfg)
                if self.tx(j + 2) == "{" and (j + 2) in self.pair:
                    self.parse_items(j + 3, self.pair[j + 2], cfg)
                    i = self.pair[j + 2] + 1
                else:
                    self.mod_decls.append((name, cfg))
                    i = self.skip_to_semi(j + 1, e)
            elif kw == "use":
                k = self.skip_to_semi(j + 1, e)
                leaves, _ = self.use_leaves(j + 1, k - 1 if self.tx(k - 1) == ";" else k, [])
                kind = "pub-use" if vis != "private" else "use"
                for path, alias, ln in leaves:
                    self.imports.append((ln, "::".join(path), alias or "-", kind, cfg))
                i = k
            elif kw == "macro_rules" and self.tx(j + 1) == "!":
                self.emit(line, "macro", self.tx(j + 2), "pub" if export else "private", cfg)
                i = self.skip_item(j + 3, e)
            elif kw == "impl":
                i = self.parse_impl(j, e, cfg)
            elif kw == "extern":
                i = self.skip_to_semi(j, e)
            else:
                # unknown: advance one token, skipping a whole bracket group
                if j < e and self.tx(j) in _OPEN and j in self.pair:
                    i = self.pair[j] + 1
                else:
                    i = max(j + 1, start + 1)

    def parse_impl(self, j, e, cfg):
        T = self.T
        k = j + 1
        hdr_end = k
        while hdr_end < e:
            t = self.tx(hdr_end)
            if t in ("(", "["):
                hdr_end = self.pair.get(hdr_end, hdr_end) + 1
            elif t in ("{", ";"):
                break
            else:
                hdr_end += 1
        if hdr_end >= e or self.tx(hdr_end) != "{" or hdr_end not in self.pair:
            return min(hdr_end + 1, e)
        s = k
        if self.tx(s) == "<":
            depth = 0
            while s < hdr_end:
                t = self.tx(s)
                if t == "<":
                    depth += 1
                elif t == ">" and self.tx(s - 1) != "-":
                    depth -= 1
                    if depth == 0:
                        s += 1
                        break
                s += 1
        hdr = []
        depth = 0
        for x in range(s, hdr_end):
            t = self.tx(x)
            if t == "<":
                depth += 1
            elif t == ">" and self.tx(x - 1) != "-":
                depth -= 1
            elif t == "where" and depth <= 0:
                break
            hdr.append(x)
        toks = [T[x] for x in hdr]
        depth, forat = 0, None
        for q, (kk, t, _) in enumerate(toks):
            if t == "<":
                depth += 1
            elif t == ">" and not (q and toks[q - 1][1] == "-"):
                depth -= 1
            elif t == "for" and depth <= 0:
                forat = q
        self_ty = join_tokens(toks[forat + 1:]) if forat is not None else join_tokens(toks)
        self.symbols.append((T[j][2], "impl", join_tokens(toks), "-", cfg))
        # members
        body_s, body_e = hdr_end + 1, self.pair[hdr_end]
        i = body_s
        while i < body_e:
            st = i
            i, cfgs, _inner, _ = self.attrs(i, body_e)
            if i >= body_e:
                break
            mcfg = combine_cfg([cfg] + cfgs)
            vis, i = self.vis(i)
            q = i
            while q < body_e and T[q][0] == "i" and T[q][1] in _QUAL:
                if T[q][1] == "const" and self.tx(q + 1) not in ("fn", "unsafe", "async", "extern"):
                    break
                q += 1
            kw = self.tx(q)
            if kw in ("fn", "const"):
                self.emit(T[q][2], kw, self_ty + "::" + self.tx(q + 1), vis, mcfg)
                i = self.skip_item(q + 1, body_e) if kw == "fn" else self.skip_to_semi(q + 1, body_e)
            elif kw == "type":
                i = self.skip_to_semi(q + 1, body_e)
            elif q < body_e and self.tx(q) in _OPEN and q in self.pair:
                i = self.pair[q] + 1
            else:
                i = max(q + 1, st + 1)
        return body_e + 1

    # -- use trees ---------------------------------------------------------
    def use_leaves(self, i, e, prefix):
        T = self.T
        out, path = [], list(prefix)
        while i < e and self.tx(i) == ":":
            i += 1
        while i < e:
            k, t, ln = T[i]
            if k == "i" and t != "as":
                path.append(t)
                i += 1
                if i + 1 < e and self.tx(i) == ":" and self.tx(i + 1) == ":":
                    i += 2
                    continue
                alias = None
                if i < e and self.tx(i) == "as":
                    alias = self.tx(i + 1)
                    i += 2
                p = path[:-1] if (path[-1] == "self" and len(path) > 1) else path
                out.append((p, alias, ln))
                return out, i
            if t == "*":
                out.append((path + ["*"], None, ln))
                return out, i + 1
            if t == "{" and i in self.pair:
                close = self.pair[i]
                q = i + 1
                while q < close:
                    sub, nq = self.use_leaves(q, close, path)
                    out += sub
                    q = nq if nq > q else q + 1
                    if q < close and self.tx(q) == ",":
                        q += 1
                return out, close + 1
            i += 1
        return out, i


# --------------------------------------------------------------------------
# workspace + manifests
# --------------------------------------------------------------------------

_DEPSEC = {"dependencies": "normal", "dev-dependencies": "dev",
           "build-dependencies": "build", "dev_dependencies": "dev",
           "build_dependencies": "build"}


def dep_lines(text):
    """(kind, target, key) -> first line of that dependency in the manifest."""
    lines, sec = {}, None
    for ln, raw in enumerate(text.splitlines(), 1):
        s = raw.strip()
        if s.startswith("["):
            m = re.match(r"^\[\[?\s*(.*?)\s*\]\]?\s*(#.*)?$", s)
            hdr = m.group(1) if m else ""
            parts = re.findall(r"\"[^\"]*\"|'[^']*'|[^.\s]+", hdr)
            sec = None
            if parts and parts[0] != "workspace":
                target = "-"
                if parts[0] == "target" and len(parts) >= 3:
                    target = parts[1].strip("\"'")
                    parts = parts[2:]
                if parts and parts[0] in _DEPSEC:
                    kind = _DEPSEC[parts[0]]
                    if len(parts) > 1:
                        lines.setdefault((kind, target, parts[1].strip("\"'")), ln)
                    else:
                        sec = (kind, target)
        elif sec and s and not s.startswith("#"):
            m = re.match(r"^(\"[^\"]+\"|[A-Za-z0-9_-]+)(?:\.[A-Za-z0-9_-]+)*\s*=", s)
            if m:
                lines.setdefault((sec[0], sec[1], m.group(1).strip("\"")), ln)
    return lines


def load_workspace(root):
    data = tomllib.loads((root / "Cargo.toml").read_text(encoding="utf-8"))
    ws = data.get("workspace", {})
    crates = {}
    for key, label in (("members", "member"), ("exclude", "excluded")):
        for pat in ws.get(key, []):
            paths = sorted(p.parent.relative_to(root).as_posix()
                           for p in root.glob(pat + "/Cargo.toml")) if any(c in pat for c in "*?[") \
                else [pat.rstrip("/")]
            for rel in paths:
                mf = root / rel / "Cargo.toml"
                if not mf.is_file():
                    continue
                try:
                    pk = tomllib.loads(mf.read_text(encoding="utf-8")).get("package", {})
                except tomllib.TOMLDecodeError:
                    continue
                name = pk.get("name")
                if not isinstance(name, str):
                    continue
                if name not in crates or label == "member":
                    crates[name] = {"name": name, "path": rel, "workspace": label}
    return ws, crates


def scan_crate(root, c):
    cdir = root / c["path"]
    mf = cdir / "Cargo.toml"
    text = mf.read_text(encoding="utf-8")
    data = tomllib.loads(text)
    files = sorted(p for p in (cdir / "src").rglob("*.rs") if p.is_file()) if (cdir / "src").is_dir() else []
    blobs = [(mf.relative_to(root).as_posix(), blob_id(mf.read_bytes()))]
    parsed = {}
    for p in files:
        raw = p.read_bytes()
        rel = p.relative_to(root).as_posix()
        blobs.append((rel, blob_id(raw)))
        parsed[rel] = (raw.decode("utf-8", "replace"))
    c["digest"] = hashlib.sha256(
        "".join(f"{a}\0{b}\n" for a, b in sorted(blobs)).encode()).hexdigest()[:16]
    c["nfiles"] = len(blobs)
    c["skipped"] = sum(1 for d in ("tests", "examples", "benches") if (cdir / d).is_dir()
                       for p in (cdir / d).rglob("*.rs"))
    c["manifest"] = (mf.relative_to(root).as_posix(), text, data)
    c["sources"] = parsed
    notes = []
    for tbl in [data.get("lib", {})] + list(data.get("bin", [])):
        pth = tbl.get("path") if isinstance(tbl, dict) else None
        if pth and not pth.startswith("src/"):
            notes.append("path-outside-src")
    c["notes"] = sorted(set(notes))


def dep_rows(crate, ws_deps, in_scope):
    mpath, text, data = crate["manifest"]
    lines = dep_lines(text)
    feats = {}
    for f, items in (data.get("features") or {}).items():
        for it in items:
            m = re.match(r"^(?:dep:)?([\w-]+)(/.*)?$", it)
            if m:
                feats.setdefault(m.group(1), set()).add(f)
    tables = []
    for sect, kind in _DEPSEC.items():
        if sect in data:
            tables.append((kind, "-", data[sect]))
        for tgt, tv in (data.get("target") or {}).items():
            if isinstance(tv, dict) and sect in tv:
                tables.append((kind, tgt, tv[sect]))
    rows = []
    for kind, tgt, tbl in tables:
        for key, spec in tbl.items():
            spec = spec if isinstance(spec, dict) else {}
            pkg = spec.get("package")
            if spec.get("workspace") and isinstance(ws_deps.get(key), dict):
                pkg = pkg or ws_deps[key].get("package")
            pkg = pkg or key
            ln = lines.get((kind, tgt, key), 1)
            rows.append((crate["name"], pkg, kind, "true" if spec.get("optional") else "false", tgt,
                         ",".join(sorted(feats.get(key, ()))) or "-", f"{mpath}:{ln}",
                         "CONFIRMED", "yes" if pkg in in_scope else "no"))
    return rows


# --------------------------------------------------------------------------
# Charon
# --------------------------------------------------------------------------


def read_schema_tsv(path):
    schema, rows = None, []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("#schema"):
            schema = raw[len("#schema"):].strip().split("\t")
        elif raw and not raw.startswith("#"):
            rows.append(dict(zip(schema, raw.split("\t"))))
    return rows


_CALLKIND = {"CALLS_RESOLVED": ("resolved", "CONFIRMED"),
             "MAY_DISPATCH": ("dispatch", "POSSIBLE"),
             "UNRESOLVED_CALL": ("unresolved", "UNKNOWN")}


def load_charon(stores, src, in_scope):
    """-> (rows, have) rows are calls.tsv tuples, have = {crate: store dir name}."""
    rows, have, seen = [], {}, {}
    by_lib = {lib_name(n): n for n in in_scope}
    cache = {}
    for st in stores:
        st = Path(st)
        for rf in sorted(st.glob("*.rows.tsv")):
            ck = rf.name[:-len(".rows.tsv")]
            crate = by_lib.get(ck)
            if crate is None:
                continue
            if crate in seen:
                raise SystemExit(f"error: crate {crate} appears in two charon stores "
                                 f"({seen[crate]}, {st}); pass one")
            seen[crate] = st
            have[crate] = st.name
            fh = {}
            fp = st / f"{ck}.files.tsv"
            if fp.is_file():
                fh = {r["path"]: r["fnv64"] for r in read_schema_tsv(fp)}
            crate_grade = "POSSIBLE"
            ncall = 0
            for r in read_schema_tsv(rf):
                if r["predicate"] == "CRATE":
                    crate_grade = "CONFIRMED" if r["object"] == "COMPLETE" else "POSSIBLE"
                if r["predicate"] not in _CALLKIND:
                    continue
                kind, ev = _CALLKIND[r["predicate"]]
                f = r["file"]
                line, ok = "-", False
                if src is not None and f != "-":
                    sp = Path(src) / f
                    if sp.is_file():
                        if sp not in cache:
                            cache[sp] = sp.read_bytes()
                        b = cache[sp]
                        ok = fh.get(f) == fnv64(b)
                        try:
                            line = str(b[:int(r["start"])].count(b"\n") + 1)
                        except ValueError:
                            line = "-"
                if not ok:
                    ev = "STALE"
                ncall += 1
                rows.append((crate, r["subject"], r["object"], kind, f, line, ev))
            if ncall == 0:
                rows.append((crate, "-", "-", "none", "-", "-", crate_grade))
    return rows, have


# --------------------------------------------------------------------------
# architecture
# --------------------------------------------------------------------------


def read_arch(root):
    p = root / OUT / "architecture.tsv"
    rows, errs = [], []
    if not p.is_file():
        return rows, errs
    for n, raw in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        if not raw.strip() or raw.startswith("#") or raw.startswith("subject\t"):
            continue
        c = raw.split("\t")
        if len(c) != 7:
            errs.append(f"architecture.tsv:{n}: expected 7 columns, got {len(c)}")
        elif c[3] not in ARCH_STATUS:
            errs.append(f"architecture.tsv:{n}: bad status {c[3]!r}")
        else:
            rows.append(tuple(c))
    return rows, errs


def validate_arch(root, rows):
    out = []
    for r in rows:
        subj, rel, obj, status, ev, anchor, note = r
        m = re.match(r"^([^\s:]+):(\d+)$", ev)
        if not m:
            v = "EXTERNAL"
        else:
            fp = root / m.group(1)
            if not fp.is_file():
                v = "FILE-MISSING"
            else:
                lines = fp.read_text(encoding="utf-8", errors="replace").splitlines()
                ln = int(m.group(2))
                v = "OK" if (anchor != "-" and 1 <= ln <= len(lines) and anchor in lines[ln - 1]) \
                    else "ANCHOR-MISSING"
        out.append(r + (v,))
    return out


# --------------------------------------------------------------------------
# generation
# --------------------------------------------------------------------------


def ev_line(path, line, ev):
    return f"{path}:{line} {ev}"


def tokset(*vals):
    return set(re.findall(r"[A-Za-z_][\w-]*", " ".join(vals)))


def san(name):
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def generate(root, stores=(), charon_src=None, views=None):
    root = Path(root)
    views = list(DEFAULT_VIEWS if views is None else views)
    ws, crates = load_workspace(root)
    ws_deps = ws.get("dependencies", {}) or {}
    for c in crates.values():
        scan_crate(root, c)
    names = sorted(crates)
    in_scope = set(names)
    parsed, macros = {}, {}
    for n in names:
        macros[n] = 0
        for rel, text in crates[n]["sources"].items():
            rf = RustFile(text)
            parsed[(n, rel)] = rf
            macros[n] += rf.macro_rules

    deps = [r for n in names for r in dep_rows(crates[n], ws_deps, in_scope)]
    rdeps = [(r[1], r[0]) + r[2:] for r in deps]
    symbols, imports = [], []
    inh = {}
    for n in names:
        inh.update(module_cfgs({rel: rf for (m, rel), rf in parsed.items() if m == n}))
    for (n, rel), rf in parsed.items():
        up = inh.get(rel, "-")
        for ln, kind, name, vis, cfg in rf.symbols:
            symbols.append((n, rel, str(ln), kind, name, vis, combine_cfg([up, cfg]), "CONFIRMED"))
        for ln, path, alias, kind, cfg in rf.imports:
            imports.append((n, rel, str(ln), path, alias, kind, combine_cfg([up, cfg]),
                            "POSSIBLE" if path.endswith("::*") or path == "*" else "CONFIRMED"))
    calls, have = load_charon(stores, charon_src, in_scope)
    for n in names:
        if n not in have:
            calls.append((n, "-", "-", MISSING, "-", "-", "UNKNOWN"))

    arch, arch_errs = read_arch(root)
    if arch_errs:
        raise SystemExit("error: " + "; ".join(arch_errs))
    decisions = validate_arch(root, arch)
    gaps = [("architecture-gap", d[0], d[1], d[2], d[4], d[6]) for d in decisions if d[3] == "GAP"]
    gaps += [("calls-missing", c[0], "calls", "-", "UNKNOWN", MISSING) for c in calls if c[3] == MISSING]

    files = {}
    files["crates.tsv"] = tsv(["crate", "path", "workspace", "digest"],
                              [(n, crates[n]["path"], crates[n]["workspace"], crates[n]["digest"]) for n in names])
    ds = ["from", "to", "kind", "optional", "target", "features_enabling", "source", "evidence", "internal"]
    files["dependencies.tsv"] = tsv(ds, deps)
    files["reverse_dependencies.tsv"] = tsv(["to", "from"] + ds[2:], rdeps)
    files["symbols.tsv"] = tsv(["crate", "file", "line", "kind", "name", "visibility", "cfg", "evidence"], symbols)
    files["imports.tsv"] = tsv(["crate", "file", "line", "path", "alias", "kind", "cfg", "evidence"], imports)
    files["calls.tsv"] = tsv(["crate", "caller", "callee", "kind", "file", "line", "evidence"], calls)
    files["decisions.tsv"] = tsv(["subject", "relation", "object", "status", "evidence", "anchor", "note",
                                  "validation"], decisions)
    files["gaps.tsv"] = tsv(["kind", "subject", "relation", "object", "evidence", "detail"], gaps)
    files["freshness.tsv"] = tsv(["crate", "digest", "generated_from"],
                                 [(n, crates[n]["digest"], f"Cargo.toml+src ({crates[n]['nfiles']} files)")
                                  for n in names])
    files["coverage.tsv"] = tsv(
        ["crate", "files_scanned", "files_skipped(tests/examples/benches)", "macro_rules_count", "note"],
        [(n, str(len(crates[n]["sources"])), str(crates[n]["skipped"]), str(macros[n]),
          ";".join(["macro-generated items invisible", "fn-local use not scanned"] + crates[n]["notes"]))
         for n in names])

    # ---- per-crate markdown ---------------------------------------------
    sym_by = {}
    for r in symbols:
        sym_by.setdefault(r[0], []).append(r)
    imp_by = {}
    for r in imports:
        imp_by.setdefault(r[0], []).append(r)
    call_by = {}
    for r in calls:
        call_by.setdefault(r[0], []).append(r)
    dep_by, rev_by = {}, {}
    for r in deps:
        dep_by.setdefault(r[0], []).append(r)
        if r[8] == "yes":
            rev_by.setdefault(r[1], []).append(r)
    libmap = {lib_name(n): n for n in names}
    idx = {n: i for i, n in enumerate(names)}

    def nid(n):
        return f"n{idx[n]}"

    for n in names:
        c = crates[n]
        L = [f"# {n}", "",
             f"crate: {n}", f"path: {c['path']}", f"workspace: {c['workspace']}", f"digest: {c['digest']}",
             "evidence legend: ../index.md#evidence-legend", "",
             "Generated; the code is the truth. See ../index.md.", "", "## Depends on", ""]
        mine = sorted(dep_by.get(n, []), key=lambda r: (r[8] != "yes", r[1], r[2], r[4]))
        for r in mine[:CAPS["deps"]]:
            extra = (" optional" if r[3] == "true" else "") + ("" if r[4] == "-" else f" target={r[4]}")
            L.append(f"- dep {r[1]} {r[2]}{extra} — {r[6]} {r[7]}")
        if len(mine) > CAPS["deps"]:
            L.append(f"- truncated ({len(mine)} total), see dependencies.tsv")
        L += ["", "## Depended on by", ""]
        for r in sorted(rev_by.get(n, []), key=lambda r: (r[0], r[2], r[4])):
            extra = (" optional" if r[3] == "true" else "") + ("" if r[4] == "-" else f" target={r[4]}")
            L.append(f"- rdep {r[0]} {r[2]}{extra} — {r[6]} {r[7]}")
        L += ["", "## Public items", ""]
        ms = sorted((r for r in sym_by.get(n, []) if r[3] != "impl"),
                    key=lambda r: (r[1], int(r[2]), r[4]))
        for r in ms[:CAPS["public"]]:
            cf = "" if r[6] == "-" else f" cfg={r[6]}"
            L.append(f"- {r[3]} {r[4]} [{r[5]}]{cf} — {r[1]}:{r[2]} {r[7]}")
        if len(ms) > CAPS["public"]:
            L.append(f"- truncated ({len(ms)} total), see symbols.tsv")
        L += ["", "## Imports from in-scope crates", ""]
        groups = {}
        for r in imp_by.get(n, []):
            seg = r[3].split("::")[0]
            t = libmap.get(seg)
            if t and t != n:
                groups.setdefault(t, []).append(r)
        shown = 0
        for t in sorted(groups):
            L.append(f"### {t}")
            for r in sorted(groups[t], key=lambda r: (r[1], int(r[2]), r[3])):
                if shown >= CAPS["imports"]:
                    break
                shown += 1
                al = "" if r[4] == "-" else f" as {r[4]}"
                cf = "" if r[6] == "-" else f" cfg={r[6]}"
                L.append(f"- {r[5]} {r[3]}{al}{cf} — {r[1]}:{r[2]} {r[7]}")
        if shown >= CAPS["imports"]:
            L.append(f"- truncated at {CAPS['imports']}, see imports.tsv")
        if not groups:
            L.append("- none")
        L += ["", "## Calls", ""]
        if n in have:
            L.append(f"CALL_GRAPH=CHARON {have[n]}")
            cs = sorted(call_by[n], key=lambda r: (r[4], r[5] if r[5] == "-" else r[5].rjust(9), r[1], r[2]))
            for r in cs[:CAPS["calls"]]:
                L.append(f"- call {r[1]} -> {r[2]} {r[3]} — {r[4]}:{r[5]} {r[6]}")
            if len(cs) > CAPS["calls"]:
                L.append(f"- truncated ({len(cs)} total), see calls.tsv")
        else:
            L.append(f"CALL_GRAPH={MISSING}")
            L.append(f"- calls {MISSING} — no ruff_charon artifact for this crate UNKNOWN")
        L += ["", "## Architecture", ""]
        symn = {r[4] for r in sym_by.get(n, []) if r[3] != "impl"}
        symn |= {s.split("::")[0] for s in symn}
        ar = []
        for d in decisions:
            ts = tokset(d[0], d[2])
            if n in ts or (ts & symn):
                ar.append(d)
        for d in sorted(ar):
            L.append(f"- arch {d[0]} {d[1]} {d[2]} [{d[3]}] validation={d[7]} — {d[4]} {d[3]}")
        if not ar:
            L.append("- none")
        L += ["", "Generated from Cargo manifests — extracted relationships only.", "",
              "```mermaid", "flowchart LR"]
        edges = set()
        for r in dep_by.get(n, []):
            if r[8] == "yes" and r[2] == "normal":
                edges.add((n, r[1]))
        for r in rev_by.get(n, []):
            if r[2] == "normal":
                edges.add((r[0], n))
        nodes = sorted({x for e in edges for x in e} | {n}, key=lambda x: idx[x])
        for x in nodes:
            L.append(f'  {nid(x)}["{x}"]')
        for a, b in sorted(edges, key=lambda e: (idx[e[0]], idx[e[1]])):
            L.append(f"  {nid(a)} --> {nid(b)}")
        L += ["```", ""]
        files[f"crates/{n}.index.md"] = "\n".join(L)

    # ---- views ----------------------------------------------------------
    src_text = {}
    for n in names:
        for rel, t in crates[n]["sources"].items():
            src_text[(n, rel)] = t
    for sym in views:
        pat = re.compile(r"\b" + re.escape(sym) + r"\b")
        L = [f"# view: {sym}", "", "Generated; the code is the truth. Same-name symbols stay separate per file.", "",
             "## Definitions", ""]
        defs = sorted((r for r in symbols if r[3] != "impl" and r[4] == sym), key=lambda r: (r[0], r[1], int(r[2])))
        grp = {}
        for r in defs:
            grp.setdefault((r[0], r[1]), []).append(r)
        for (cn, f) in sorted(grp):
            L.append(f"### {cn} {f}")
            for r in grp[(cn, f)]:
                cf = "" if r[6] == "-" else f" cfg={r[6]}"
                L.append(f"- {r[3]} {r[4]} [{r[5]}]{cf} — {f}:{r[2]} {r[7]}")
        if not grp:
            L.append("- no definition found in scanned sources UNKNOWN")
        L += ["", "## Impls mentioning it", ""]
        im = [r for r in symbols if r[3] == "impl" and pat.search(r[4])]
        for r in sorted(im, key=lambda r: (r[0], r[1], int(r[2])))[:CAPS["imports"]]:
            cf = "" if r[6] == "-" else f" cfg={r[6]}"
            L.append(f"- impl {r[4]}{cf} — {r[1]}:{r[2]} POSSIBLE")
        if not im:
            L.append("- none")
        cands = set()
        for (cn, f) in grp:
            stem = Path(f).stem
            if stem in ("mod",):
                cands.add(Path(f).parent.name)
            elif stem in ("lib", "main"):
                cands.add(lib_name(cn))
            else:
                cands.add(stem)
        L += ["", "## Imports (not resolved to a definition)", ""]
        ir = []
        for r in imports:
            segs = r[3].split("::")
            if segs[-1] == sym or (segs[-1] == "*" and len(segs) > 1 and segs[-2] in cands):
                ir.append(r)
        for r in sorted(ir, key=lambda r: (r[0], r[1], int(r[2]), r[3]))[:CAPS["occ"]]:
            cf = "" if r[6] == "-" else f" cfg={r[6]}"
            L.append(f"- {r[5]} {r[3]}{cf} — {r[1]}:{r[2]} POSSIBLE")
        if not ir:
            L.append("- none")
        occ = []
        for (cn, f), t in sorted(src_text.items()):
            if sym not in t:
                continue
            for i, line in enumerate(t.splitlines(), 1):
                if pat.search(line):
                    occ.append((f, i))
        L += ["", f"## Textual occurrences ({len(occ)} total)", ""]
        for f, i in sorted(occ)[:CAPS["occ"]]:
            L.append(f"- {f}:{i} POSSIBLE")
        if len(occ) > CAPS["occ"]:
            L.append(f"- truncated at {CAPS['occ']}")
        if not occ:
            L.append("- none")
        L += ["", "## Architecture", ""]
        ar = [d for d in decisions if sym in tokset(d[0], d[2])]
        for d in sorted(ar):
            L.append(f"- arch {d[0]} {d[1]} {d[2]} [{d[3]}] validation={d[7]} — {d[4]} {d[3]}")
        if not ar:
            L.append("- none")
        files[f"views/{san(sym)}.md"] = "\n".join(L) + "\n"

    # ---- index.md -------------------------------------------------------
    internal = sorted({(r[0], r[1]) for r in deps if r[8] == "yes" and r[2] == "normal"})
    withedge = sorted({x for e in internal for x in e}, key=lambda x: idx[x])
    I = ["# .codegraph", "",
         "A GENERATED projection of source-backed facts (Cargo manifests, Rust item declarations, `use` trees,",
         "ruff_charon call artifacts, and the hand-curated `architecture.tsv`). **The code is the truth**; this",
         "is a grep-friendly index, never an authority. Calls come only from existing Charon artifacts;",
         f"without one the crate says `{MISSING}`.", "",
         "## Freshness", "",
         "```",
         "python3 .claude/tools/codegraph_index.py --check   # exit 1 lists STALE crates and differing files",
         "python3 .claude/tools/codegraph_index.py --write   # regenerate",
         "```", "",
         "No timestamps, absolute paths or HEAD sha are recorded. A crate digest is sha256 over sorted",
         "(relpath, git blob id) of its Cargo.toml and `src/**/*.rs`. `--check` also fails on",
         "ANCHOR-MISSING / FILE-MISSING rows in decisions.tsv.", "",
         "## Evidence legend", "",
         "- `CONFIRMED` — read directly from an authoritative artifact at the cited `path:line` (a Cargo manifest",
         "  key, a Rust declaration or `use` statement as written, a Charon COMPILER-graded resolved call).",
         "- `POSSIBLE` — a relationship inferred, not proven: a glob import, a re-export chain not resolved,",
         "  a `MAY_DISPATCH` / dynamic call, a textual mention.",
         "- `UNKNOWN` — no fact source covers it (e.g. calls when no Charon artifact).",
         "- `LAB` / `OPEN-PR` / `PROPOSED` / `GAP` — only in architecture-derived rows; copied from",
         "  architecture.tsv, never inferred.",
         "- `STALE` — the cited source changed since the fact was recorded (Charon file hash mismatch).", "",
         "## Files", "",
         "- `crates.tsv`, `freshness.tsv`, `coverage.tsv` — crate inventory, digests, what the scanner could not see.",
         "- `dependencies.tsv` / `reverse_dependencies.tsv` — `features_enabling` = this crate's own Cargo features",
         "  that switch the dependency on; `internal` = target is an in-scope crate.",
         "- `symbols.tsv` — pub / pub(crate) top-level items and `pub fn`/`const` in impl blocks (`Type::method`);",
         "  `impl` rows carry visibility `-`. `imports.tsv` — one row per use-tree leaf, cfg kept.",
         "- `calls.tsv` — Charon rows only. `decisions.tsv` / `gaps.tsv` — validated architecture rows.",
         "- `crates/<crate>.index.md`, `views/<symbol>.md`.", "",
         "Not scanned: macro-generated items, function-local `use`, `tests/`, `examples/`, `benches/`.", "",
         "## Grep recipes (NEW / DERIVED 2026-11-11)", "",
         "The original ~15 navigation commands referenced in earlier discussion were not recovered; these recipes "
         "are new and derived, not historical.", "",
         "<!-- recipes -->",
         "```",
         "rg -n '^lance-graph-contract\\t' .codegraph/reverse_dependencies.tsv   # who depends on a crate",
         "rg -n '\\tClassView\\t' .codegraph/symbols.tsv                          # where a symbol is defined",
         "rg -n 'lance_graph_contract::class_view' .codegraph/imports.tsv         # who imports a path",
         "rg -n 'feature=' .codegraph/imports.tsv .codegraph/symbols.tsv         # cfg-gated imports and items",
         f"rg -n '{MISSING}' .codegraph/gaps.tsv                                # crates with no call graph",
         "```",
         "<!-- /recipes -->", "",
         "## Crates", "", "| crate | workspace | path |", "|---|---|---|"]
    for n in names:
        I.append(f"| [{n}](crates/{n}.index.md) | {crates[n]['workspace']} | {crates[n]['path']} |")
    I += ["", "## Crate dependency diagram", "",
          "Generated from Cargo manifests — extracted relationships only. Internal `normal` dependencies; "
          f"only crates with at least one internal edge are drawn ({len(withedge)} of {len(names)}).", "",
          "```mermaid", "flowchart LR"]
    for x in withedge:
        I.append(f'  {nid(x)}["{x}"]')
    for a, b in internal:
        I.append(f"  {nid(a)} --> {nid(b)}")
    I += ["  classDef shipped fill:#d8f0d8,stroke:#2e7d32,color:#111",
          "  classDef lab fill:#fff3cd,stroke:#a67c00,color:#111",
          "  classDef open fill:#d6e6fa,stroke:#1565c0,color:#111",
          "  classDef gap fill:#fadadd,stroke:#c62828,stroke-dasharray:5 3,color:#111"]
    mem = [nid(x) for x in withedge if crates[x]["workspace"] == "member"]
    exc = [nid(x) for x in withedge if crates[x]["workspace"] != "member"]
    if mem:
        I.append("  class " + ",".join(mem) + " shipped")
    if exc:
        I.append("  class " + ",".join(exc) + " lab")
    I += ["```", ""]
    files["index.md"] = "\n".join(I)
    return files, decisions


GENERATED_DIRS = ("crates", "views")


def disk_files(root):
    base = Path(root) / OUT
    out = {}
    if base.is_dir():
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base).as_posix()
            if p.is_file() and rel != "architecture.tsv":
                out[rel] = p.read_text(encoding="utf-8")
    return out


def write_all(root, files):
    base = Path(root) / OUT
    base.mkdir(exist_ok=True)
    arch = base / "architecture.tsv"
    if not arch.exists():
        arch.write_text(ARCH_HEADER + "\n", encoding="utf-8")
    old = disk_files(root)
    for rel in old:
        if rel not in files:
            (base / rel).unlink()
    for rel, text in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if old.get(rel) != text:
            p.write_bytes(text.encode("utf-8"))


def do_check(root, files, decisions):
    msgs = []
    disk = disk_files(root)
    mine = {}
    for ln in files["freshness.tsv"].splitlines():
        if ln and not ln.startswith("#"):
            a = ln.split("\t")
            mine[a[0]] = a[1]
    theirs = {}
    for ln in disk.get("freshness.tsv", "").splitlines():
        if ln and not ln.startswith("#"):
            a = ln.split("\t")
            theirs[a[0]] = a[1]
    if not disk:
        msgs.append("MISSING .codegraph/ (run --write)")
    for c in sorted(set(mine) | set(theirs)):
        if mine.get(c) != theirs.get(c):
            msgs.append(f"STALE crate {c}")
    for rel in sorted(set(files) | set(disk)):
        if files.get(rel) != disk.get(rel):
            msgs.append(f"DIFF {OUT}/{rel}")
    for d in decisions:
        if d[7] in ("ANCHOR-MISSING", "FILE-MISSING"):
            msgs.append(f"ARCH {d[7]}: {d[0]} | {d[1]} | {d[2]} @ {d[4]}")
    return msgs


# --------------------------------------------------------------------------
# self-test
# --------------------------------------------------------------------------


def mk_ws(td, crates, members=None, excluded=()):
    root = Path(td)
    members = list(crates if members is None else members)
    (root / "Cargo.toml").write_text(
        "[workspace]\nmembers = [" + ", ".join(f'"crates/{m}"' for m in members) + "]\n"
        "exclude = [" + ", ".join(f'"crates/{m}"' for m in excluded) + "]\n")
    for n, (src, extra) in crates.items():
        d = root / "crates" / n
        (d / "src").mkdir(parents=True, exist_ok=True)
        (d / "Cargo.toml").write_text(f'[package]\nname = "{n}"\nversion = "0.1.0"\n{extra}')
        (d / "src" / "lib.rs").write_text(src)
    return root


def rows_of(files, name):
    return [l.split("\t") for l in files[name].splitlines() if l and not l.startswith("#")]


class T(unittest.TestCase):
    def setUp(self):
        self._td = tempfile.TemporaryDirectory()
        self.td = self._td.name

    def tearDown(self):
        self._td.cleanup()

    def test_01_stale(self):
        root = mk_ws(self.td, {"alpha": ("pub fn a() {}\n", ""), "beta": ("pub fn b() {}\n", "")})
        f, d = generate(root)
        write_all(root, f)
        f, d = generate(root)
        self.assertEqual(do_check(root, f, d), [])
        (root / "crates/alpha/src/lib.rs").write_text("pub fn a() {}\npub fn a2() {}\n")
        f, d = generate(root)
        m = do_check(root, f, d)
        self.assertIn("STALE crate alpha", m)
        self.assertNotIn("STALE crate beta", m)

    def test_02_cfg_and_target(self):
        src = '#[cfg(feature="x")]\nuse a::b;\n#[cfg(feature="x")]\nmod m { use c::d; }\n'
        extra = '\n[target.\'cfg(unix)\'.dependencies]\nlibc = "0.2"\n'
        root = mk_ws(self.td, {"alpha": (src, extra)})
        f, _ = generate(root)
        imp = {r[3]: r[6] for r in rows_of(f, "imports.tsv")}
        self.assertEqual(imp["a::b"], 'feature="x"')
        self.assertEqual(imp["c::d"], 'feature="x"')
        dep = [r for r in rows_of(f, "dependencies.tsv") if r[1] == "libc"][0]
        self.assertEqual(dep[4], "cfg(unix)")
        self.assertTrue(dep[6].endswith("Cargo.toml:6"), dep[6])

    def _charon(self, kinds):
        st = Path(self.td) / "store"
        st.mkdir()
        sd = Path(self.td) / "srcroot" / "alpha" / "src"
        sd.mkdir(parents=True)
        body = b"pub fn a() { b(); c(); d(); }\n"
        (sd / "lib.rs").write_bytes(body)
        rows = ["#version 2", "#schema predicate\tsubject\tobject\tgrade\tfile\tstart\tend"]
        for i, k in enumerate(kinds):
            rows.append(f"{k}\talpha::a\tx::f{i}\tCOMPILER\talpha/src/lib.rs\t0\t5")
        (st / "alpha.rows.tsv").write_text("\n".join(rows) + "\n")
        (st / "alpha.files.tsv").write_text(f"#version 1\n#schema path\tfnv64\nalpha/src/lib.rs\t{fnv64(body)}\n")
        return st, Path(self.td) / "srcroot"

    def test_03_dispatch_not_confirmed(self):
        root = mk_ws(self.td, {"alpha": ("pub fn a() {}\n", "")})
        st, sr = self._charon(["CALLS_RESOLVED", "MAY_DISPATCH", "UNRESOLVED_CALL"])
        f, _ = generate(root, [st], sr)
        ev = {r[3]: r[6] for r in rows_of(f, "calls.tsv")}
        self.assertEqual(ev, {"resolved": "CONFIRMED", "dispatch": "POSSIBLE", "unresolved": "UNKNOWN"})

    def test_04_plan_not_shipped(self):
        root = mk_ws(self.td, {"alpha": ("pub fn a() {}\n", "")})
        (root / OUT).mkdir()
        (root / OUT / "architecture.tsv").write_text(
            ARCH_HEADER + "\nS\tr\tO\tPROPOSED\tEXO-1\t-\tn\nG\tr\tO\tGAP\t#12\t-\tm\n")
        f, _ = generate(root)
        dec = {r[0]: r for r in rows_of(f, "decisions.tsv")}
        self.assertEqual(dec["S"][3], "PROPOSED")
        self.assertEqual(dec["S"][7], "EXTERNAL")
        gaps = [r for r in rows_of(f, "gaps.tsv") if r[0] == "architecture-gap"]
        self.assertEqual([r[1] for r in gaps], ["G"])

    def test_05_same_name(self):
        root = mk_ws(self.td, {"alpha": ("pub struct Foo;\n", ""), "beta": ("pub struct Foo(u8);\n", "")})
        f, _ = generate(root, views=["Foo"])
        v = f["views/Foo.md"]
        self.assertEqual(sum(1 for l in v.splitlines() if l.startswith("### ")), 2)
        self.assertIn("### alpha crates/alpha/src/lib.rs", v)
        self.assertIn("### beta crates/beta/src/lib.rs", v)

    def test_06_determinism(self):
        c = {"alpha": ("pub fn a() {}\nuse beta::x;\n", ""), "beta": ("pub fn b() {}\n", "")}
        root = mk_ws(self.td, c)
        f1, _ = generate(root)
        f2, _ = generate(root)
        self.assertEqual(f1, f2)
        with tempfile.TemporaryDirectory() as td2:
            root2 = mk_ws(td2, c, members=["beta", "alpha"])
            f3, _ = generate(root2)
            self.assertEqual(f1, f3)

    def test_07_missing_artifact(self):
        root = mk_ws(self.td, {"alpha": ("pub fn a() {}\n", ""), "beta": ("pub fn b() {}\n", "")})
        st, sr = self._charon(["CALLS_RESOLVED"])
        f, _ = generate(root, [st], sr)
        beta = [r for r in rows_of(f, "calls.tsv") if r[0] == "beta"]
        self.assertEqual(beta, [["beta", "-", "-", "ARTIFACT_MISSING", "-", "-", "UNKNOWN"]])
        self.assertFalse([r for r in rows_of(f, "calls.tsv") if r[0] == "alpha" and r[3] == MISSING])
        self.assertIn("\nCALL_GRAPH=ARTIFACT_MISSING\n", f["crates/beta.index.md"])
        self.assertIn("\nCALL_GRAPH=CHARON store\n", f["crates/alpha.index.md"])

    def test_08_charon_fixture(self):
        fx = REPO / "crates/sag-ra-probe/fixtures/codegraph"
        store = REPO / "crates/lance-graph-codegraph/tests/data/rev1"
        import shutil
        for n in ("cg_app", "cg_lib"):
            shutil.copytree(fx / "rev1" / n, Path(self.td) / "crates" / n, ignore=shutil.ignore_patterns("target"))
        (Path(self.td) / "Cargo.toml").write_text(
            '[workspace]\nmembers = ["crates/cg_app", "crates/cg_lib"]\n')
        f, _ = generate(self.td, [store], fx / "rev1")
        want = ["cg_app", "cg_app::run", "cg_lib::area_of", "resolved", "cg_app/src/lib.rs", "7", "CONFIRMED"]
        self.assertIn(want, rows_of(f, "calls.tsv"))
        f2, _ = generate(self.td, [store], fx / "rev2")
        stale = [r for r in rows_of(f2, "calls.tsv") if r[0] == "cg_app" and r[1] == "cg_app::run"]
        self.assertTrue(stale)
        self.assertTrue(all(r[6] == "STALE" for r in stale), stale)
        self.assertEqual(fnv64((fx / "rev1/cg_app/src/lib.rs").read_bytes()), "0effe8b102e60ce2")

    def test_09_use_tree(self):
        rf = RustFile("use a::{b, c::{d as e, f::*}, self as g};\npub use x::y;\nuse ::z::{self};\n"
                      "use h::{self, i};\n")
        got = sorted((p, a, k) for _, p, a, k, _ in rf.imports)
        self.assertEqual(got, sorted([
            ("a::b", "-", "use"), ("a::c::d", "e", "use"), ("a::c::f::*", "-", "use"), ("a", "g", "use"),
            ("x::y", "-", "pub-use"), ("z", "-", "use"), ("h", "-", "use"), ("h::i", "-", "use")]))

    def test_10_anchor(self):
        root = mk_ws(self.td, {"alpha": ("pub fn a() {}\n", "")})
        (root / OUT).mkdir()
        (root / OUT / "architecture.tsv").write_text(
            ARCH_HEADER + "\nS\tr\tO\tCONFIRMED\tcrates/alpha/src/lib.rs:1\tpub fn a\tn\n"
            "S2\tr\tO\tCONFIRMED\tcrates/alpha/src/lib.rs:1\tNOPE\tn\n"
            "S3\tr\tO\tCONFIRMED\tcrates/alpha/src/zzz.rs:1\tx\tn\n")
        f, d = generate(root)
        v = {r[0]: r[7] for r in rows_of(f, "decisions.tsv")}
        self.assertEqual(v, {"S": "OK", "S2": "ANCHOR-MISSING", "S3": "FILE-MISSING"})

    def test_11_lexer_and_items(self):
        src = ('// pub fn no() {}\n/* pub struct No; /* nested */ */\n'
               'const S: &str = "pub fn str() {}"; const R: &str = r#"pub fn raw() {}"#;\n'
               "pub struct A<'a>(&'a u8);\npub(crate) enum E { X }\nfn private() {}\n"
               "pub trait Tr { fn m(); }\nimpl<T> Tr for Foo<T> where T: Sized { }\n"
               "impl Foo { pub fn new() -> Self { Foo } fn hid() {} pub const K: u8 = 1; }\n"
               "#[macro_export] macro_rules! mm { () => { pub fn inner() {} } }\n"
               "pub mod m { pub fn inm() {} }\npub const fn cf() {}\nlet_c: char = '}';\n")
        rf = RustFile(src)
        got = {(k, n) for _, k, n, v, c in rf.symbols if v in ("pub", "pub(crate)")}
        self.assertEqual(got, {("struct", "A"), ("enum", "E"), ("trait", "Tr"), ("fn", "Foo::new"),
                               ("const", "Foo::K"), ("macro", "mm"), ("mod", "m"), ("fn", "inm"),
                               ("fn", "cf")})

    def test_12_inner_cfg_scopes_the_whole_file_and_inline_mod(self):
        # #1999 review: `#![cfg]` must gate every later item, not only the first.
        rf = RustFile('#![cfg(feature = "x")]\nuse a::b;\npub fn one() {}\npub const C: u8 = 0;\n'
                      'pub mod m { #![cfg(test)]\n pub fn two() {} }\n')
        cfg = {n: c for _, k, n, v, c in rf.symbols}
        self.assertEqual(cfg["one"], 'feature="x"')
        self.assertEqual(cfg["C"], 'feature="x"')
        self.assertEqual(cfg["two"], 'all(feature="x", test)')
        self.assertEqual([c for *_, c in rf.imports], ['feature="x"'])
        # silence twin: a file without inner attributes stays unconditional
        rf2 = RustFile("pub fn three() {}\n")
        self.assertEqual([c for *_, c in rf2.symbols], ["-"])

    def test_13_cfg_mod_decl_gates_the_child_file(self):
        files = {"k/src/lib.rs": RustFile('#[cfg(feature = "y")]\npub mod child;\npub mod plain;\n'),
                 "k/src/child.rs": RustFile("pub fn c() {}\npub mod deep;\n"),
                 "k/src/child/deep.rs": RustFile("pub fn d() {}\n"),
                 "k/src/plain.rs": RustFile("pub fn p() {}\n")}
        inh = module_cfgs(files)
        self.assertEqual(inh.get("k/src/child.rs"), 'feature="y"')
        self.assertEqual(inh.get("k/src/child/deep.rs"), 'feature="y"')
        self.assertEqual(inh.get("k/src/plain.rs", "-"), "-")


def self_test():
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(T)
    res = unittest.TextTestRunner(verbosity=2).run(suite)
    print("self-test: " + ("ALL PASSED" if res.wasSuccessful() else "FAILURE(S)"))
    return 0 if res.wasSuccessful() else 1


# --------------------------------------------------------------------------


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--charon-store", action="append", default=[])
    ap.add_argument("--charon-src")
    ap.add_argument("--view", action="append")
    ap.add_argument("--root", default=str(REPO))
    a = ap.parse_args(argv)
    if a.self_test:
        return self_test()
    if not (a.write or a.check):
        ap.error("one of --write / --check / --self-test is required")
    root = Path(a.root)
    files, decisions = generate(root, a.charon_store, a.charon_src, a.view)
    if a.check:
        msgs = do_check(root, files, decisions)
        for m in msgs:
            print(m)
        if msgs:
            print("::error::.codegraph is stale or inconsistent. Regenerate: "
                  "python3 .claude/tools/codegraph_index.py --write")
            return 1
        print(f".codegraph is current ({len(files)} generated files)")
        return 0
    write_all(root, files)
    bad = [d for d in decisions if d[7] in ("ANCHOR-MISSING", "FILE-MISSING")]
    print(f"wrote {len(files)} files under {OUT}/ ({len(bad)} architecture rows failing validation)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
