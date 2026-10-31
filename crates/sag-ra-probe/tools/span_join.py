#!/usr/bin/env python3
"""Join Charon functions to rust-analyzer symbols by source position, not by
path spelling (D-SAG-6-2 lab).

    python3 -I span_join.py <ra_dump.txt> <file.llbc> <crate>

The two tools spell some items differently (Charon: `crate::Impl::test`,
twice, for two impls; rust-analyzer: `crate::impl::Below::Pred::test`), so a
path join leaves them UNKNOWN. Both report source positions, and positions
are the same in both. One `join` row per local Charon function:

  EXACT     one rust-analyzer function whose name sits on the line Charon's
            span starts at (both 1-based, same file), and the names agree
  ENCLOSED  no symbol of its own (a closure, or its `call_once` shim): joined
            to the innermost EXACT function whose span contains it
  DERIVED   a method of an impl Charon marks `AutomaticallyDerived`: joined
            to the rust-analyzer moniker of the impl's Self type
  AMBIGUOUS several candidates; nothing is picked
  UNJOINED  no candidate: UNKNOWN, never "no such item"

Then impls are paired one by one, keyed (owner, trait, self_ty), where owner
is a type or trait of this crate and trait is `inherent` for an inherent block:

  impl_pair   one row per key: PAIRED (counts equal), RA_ONLY (rust-analyzer
              has more) or CHARON_ONLY (Charon has more), with both counts and
              the origin of Charon's impls: `closure` (a closure's state type,
              which rust-analyzer has no ADT for), `derived`, `drop_glue`
              (compiler drop code), or `source`. Inherent blocks are counted
              by placing Charon's methods in rust-analyzer's block ranges;
              `unplaced` marks methods that fit no block. An
              RA_ONLY inherent row also carries min_items: a block with no
              items is invisible to Charon, which sees inherent blocks only
              through their methods.
  impl_scope  an impl of a type or trait of this crate that is written in
              another crate. Charon on one crate cannot see it: reported,
              not compared.
  impl_summary  totals over the rows above.

Nothing is excluded or renamed to make the counts agree.
"""
import json
import sys
from collections import Counter

# rust-analyzer kinds whose items Charon translates into a function body:
# functions and methods, constant and static initializers, and enum-variant
# constructors.
FN_KINDS = {"Function", "Method", "StaticMethod", "TraitMethod",
            "Constant", "StaticVariable", "EnumMember"}


def pname(path):
    return "::".join(
        str(c["Ident"][0]) if "Ident" in c else next(iter(c)) for c in path
    )


def last_ident(path):
    ids = [c["Ident"][0] for c in path if "Ident" in c]
    return ids[-1] if ids else None


def span(meta):
    sp = meta["span"]
    d = sp.get("Untagged", sp)["data"]
    return d["file_id"], d["beg"]["line"], d["end"]["line"]


def ra_symbols(path):
    out = []
    for line in open(path):
        cols = line.rstrip("\n").split("\t")
        if cols[0] != "sym":
            continue
        f = dict(c.split("=", 1) for c in cols[2:] if "=" in c)
        at = f.get("at", "none")
        if at in ("none",) or at.startswith("external") or ":" not in at:
            continue
        file, ln = at.rsplit(":", 1)
        out.append({
            "name": cols[1], "kind": f["kind"], "moniker": f["moniker"],
            "file": file, "line": int(ln), "impls": f.get("impls", "-"),
        })
    return out


def ra_impls(path):
    out = []
    with open(path) as fh:
        lines = fh.readlines()
    for line in lines:
        cols = line.rstrip("\n").split("\t")
        if cols[0] != "impl" or len(cols) < 6:
            continue
        loc = cols[6] if len(cols) > 6 else "none"
        out.append((cols[1], cols[2], cols[3], cols[4], int(cols[5]), loc))
    return out


def same_file(ra_file, charon_file):
    return ra_file == charon_file or ra_file.endswith("/" + charon_file)


def self_adt(impl):
    """The Self type of a trait impl, if it is an ADT: its type_decls id."""
    tys = impl["impl_trait"]["generics"]["types"]
    if not tys:
        return None
    ty = tys[0].get("Untagged", tys[0])
    return ty["Adt"]["id"] if isinstance(ty, dict) and "Adt" in ty else None


def main(ra_path, llbc_path, crate):
    ra = ra_symbols(ra_path)
    t = json.load(open(llbc_path))["translated"]
    files = {f["id"]: next(iter(f["name"].values())) for f in t["files"]}
    types = {d["def_id"]: pname(d["item_meta"]["name"]) for d in t["type_decls"] if d}
    impls = {d["def_id"]: d for d in t["trait_impls"] if d}
    traits = {d["def_id"]: pname(d["item_meta"]["name"]) for d in t["trait_decls"] if d}
    ra_monikers = {s["moniker"] for s in ra}

    funs = []
    for f in t["fun_decls"]:
        if not f or not f["item_meta"]["is_local"]:
            continue
        nm = pname(f["item_meta"]["name"])
        if not nm.startswith(crate + "::"):
            continue
        fid, beg, end = span(f["item_meta"])
        funs.append((f, nm, files[fid], beg, end))

    joined = {}
    rows = []
    for f, nm, file, beg, end in funs:
        cands = [s for s in ra if s["kind"] in FN_KINDS and same_file(s["file"], file)
                 and s["moniker"].split("::")[0] == crate]
        on_line = [s for s in cands if s["line"] == beg]
        inside = [s for s in cands if beg <= s["line"] <= end]
        pick = on_line if on_line else inside
        name = last_ident(f["item_meta"]["name"])
        pick = [s for s in pick if s["name"] == name]
        if len(pick) == 1:
            joined[f["def_id"]] = pick[0]["moniker"]
            rows.append((f["def_id"], nm, "EXACT", pick[0]["moniker"], f"{file}:{beg}"))
        elif len(pick) > 1:
            rows.append((f["def_id"], nm, "AMBIGUOUS",
                         ",".join(s["moniker"] for s in pick), f"{file}:{beg}"))
        else:
            rows.append((f["def_id"], nm, None, None, f"{file}:{beg}-{end}"))

    by_id = {f["def_id"]: (file, beg, end) for f, _, file, beg, end in funs}
    src = {f["def_id"]: f.get("src") for f, *_ in funs}
    final = []
    for fid, nm, cls, mon, where in rows:
        if cls is None:
            file, beg, end = by_id[fid]
            s = src[fid]
            impl = impls.get(s["TraitImpl"]["impl_ref"]["id"]) if isinstance(s, dict) and "TraitImpl" in s else None
            derived = impl and any(a == {"Builtin": "AutomaticallyDerived"}
                                   for a in impl["item_meta"]["attr_info"]["attributes"])
            if derived:
                adt = types.get(self_adt(impl))
                trait = traits.get(impl["impl_trait"]["id"], "?")
                if adt in ra_monikers:
                    cls, mon, where = "DERIVED", adt, f"{trait} at {where}"
            if cls is None:
                outer = [(e - b, joined[o]) for o, (f2, b, e) in by_id.items()
                         if o in joined and o != fid and f2 == file and b <= beg and end <= e]
                if outer:
                    cls, mon = "ENCLOSED", min(outer)[1]
                else:
                    cls, mon = "UNJOINED", "-"
        final.append((fid, nm, cls, mon, where))

    for row in final:
        print("join\t" + "\t".join(map(str, row)))

    # Inherent impl blocks are not trait impls; Charon shows them only as the
    # `InherentImplBlock` path segment of their methods, so a block with no
    # methods is invisible here. rust-analyzer counts inherent blocks too.
    blocks = {}
    methods = {}
    for f, _, file, beg, _ in funs:
        for seg in f["item_meta"]["name"]:
            ty = seg.get("Impl", {}).get("Ty") if isinstance(seg.get("Impl"), dict) else None
            if ty and ty.get("kind") == "InherentImplBlock":
                sb = ty["skip_binder"].get("Untagged", ty["skip_binder"])
                adt = types.get(sb["Adt"]["id"]) if "Adt" in sb else None
                if adt:
                    blocks.setdefault(adt, set()).add(json.dumps(seg, sort_keys=True))
                    methods.setdefault(adt, []).append((file, beg, json.dumps(seg, sort_keys=True)))

    def in_crate(path):
        return path is not None and path.split("::")[0] == crate

    # Where each Charon impl comes from: a closure's state type (rust-analyzer
    # has no ADT for it), a derive (`AutomaticallyDerived`), or written source.
    fun_meta = {f["def_id"]: f["item_meta"]["name"] for f in t["fun_decls"] if f}
    closure_types = {types[d["def_id"]] for d in t["type_decls"]
                     if d and isinstance(d.get("src"), dict) and "Closure" in d["src"]}
    origin = {}
    # Two inherent blocks of one type can have identical Charon path segments,
    # so segments alone undercount blocks. Each Charon inherent method is
    # placed in the rust-analyzer inherent block whose source range contains
    # it; a block counts once if any method lands in it. Methods that land in
    # no block fall back to one block per distinct segment.
    ra_blocks = {}
    for owner, trait, self_ty, impl_crate, _, loc in ra_impls(ra_path):
        if trait == "inherent" and impl_crate == crate and loc != "none":
            file, lines = loc.rsplit(":", 1)
            b, e = map(int, lines.split("-"))
            ra_blocks.setdefault(owner, set()).add((file, b, e))
    charon = Counter()
    for adt in blocks:
        if not in_crate(adt):
            continue
        placed, loose = set(), set()
        for file, beg, seg in methods[adt]:
            hit = [blk for blk in ra_blocks.get(adt, ()) if same_file(blk[0], file) and blk[1] <= beg <= blk[2]]
            if hit:
                placed.add(min(hit, key=lambda blk: blk[2] - blk[1]))
            else:
                loose.add(seg)
        charon[(adt, "inherent", adt)] += len(placed) + len(loose)
        origin.setdefault((adt, "inherent", adt), set()).add("source" if not loose else "source,unplaced")
    for impl in impls.values():
        if not impl["item_meta"]["is_local"]:
            continue
        trait = traits.get(impl["impl_trait"]["id"], "?")
        adt = types.get(self_adt(impl))
        self_ty = adt if adt else "non-adt"
        derived = any(a == {"Builtin": "AutomaticallyDerived"}
                      for a in impl["item_meta"]["attr_info"]["attributes"])
        # Drop glue: the impl's method is named with a `Builtin: DropGlue` segment.
        glue = any(
            seg.get("Builtin", [None])[0] == "DropGlue"
            for m in impl.get("methods", []) if isinstance(m, dict)
            for seg in fun_meta.get(m.get("skip_binder", {}).get("id"), []))
        kind = ("closure" if adt in closure_types else "derived" if derived
                else "drop_glue" if glue else "source")
        for owner in (adt, trait):
            if in_crate(owner):
                charon[(owner, trait, self_ty)] += 1
                origin.setdefault((owner, trait, self_ty), set()).add(kind)

    rust_a = Counter()
    items = {}
    scope = []
    for owner, trait, self_ty, impl_crate, n_items, _ in ra_impls(ra_path):
        if not in_crate(owner):
            continue
        if impl_crate != crate:
            scope.append((owner, trait, self_ty, impl_crate))
            continue
        rust_a[(owner, trait, self_ty)] += 1
        items.setdefault((owner, trait, self_ty), []).append(n_items)

    paired = ra_only = charon_only = 0
    for key in sorted(set(rust_a) | set(charon)):
        a, b = rust_a[key], charon[key]
        if a == b:
            cls = "PAIRED"
            paired += 1
        elif a > b:
            cls = "RA_ONLY"
            ra_only += a - b
        else:
            cls = "CHARON_ONLY"
            charon_only += b - a
        row = (f"impl_pair\t{key[0]}\t{key[1]}\t{key[2]}\t{cls}\tra={a}\tcharon={b}"
               f"\torigin={','.join(sorted(origin.get(key, {'-'})))}")
        if cls == "RA_ONLY" and key[1] == "inherent":
            row += f"\tmin_items={min(items[key])}"
        print(row)
    for row in sorted(scope):
        print("impl_scope\t" + "\t".join(row))
    print(f"impl_summary\tpaired={paired}\tra_only={ra_only}\t"
          f"charon_only={charon_only}\tscope={len(scope)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
