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

Then one `impls` row per type and trait of this crate that rust-analyzer counted
impls for, with Charon's count and its `trait/method-count` list beside it.
`inherent/-` entries are inherent impl blocks, seen through their methods.
MARKER_GAP: the counts agree once the compiler-generated marker impls in
MARKERS are left out. DISAGREE is kept as found, not reconciled.
"""
import json
import sys

# rust-analyzer kinds whose items Charon translates into a function body:
# functions and methods, constant and static initializers, and enum-variant
# constructors.
MARKERS = {"core::marker::StructuralPartialEq", "core::clone::TrivialClone",
           "core::marker::Destruct"}

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

    seen = {}
    # Inherent impl blocks are not trait impls; Charon shows them only as the
    # `InherentImplBlock` path segment of their methods, so a block with no
    # methods is invisible here. rust-analyzer counts inherent blocks too.
    for f, *_ in funs:
        for seg in f["item_meta"]["name"]:
            ty = seg.get("Impl", {}).get("Ty") if isinstance(seg.get("Impl"), dict) else None
            if ty and ty.get("kind") == "InherentImplBlock":
                sb = ty["skip_binder"].get("Untagged", ty["skip_binder"])
                adt = types.get(sb["Adt"]["id"]) if "Adt" in sb else None
                if adt:
                    entry = f"inherent:{json.dumps(seg, sort_keys=True)}"
                    if entry not in seen.setdefault(adt + "#blocks", set()):
                        seen[adt + "#blocks"].add(entry)
                        seen.setdefault(adt, []).append("inherent/-")
    for impl in impls.values():
        if not impl["item_meta"]["is_local"]:
            continue
        trait = traits.get(impl["impl_trait"]["id"], "?")
        for key, label in ((types.get(self_adt(impl)), trait), (trait, types.get(self_adt(impl), "?"))):
            if key:
                seen.setdefault(key, []).append(f"{label}/{len(impl['methods'])}")
    for s in ra:
        if (s["kind"] in ("Struct", "Trait") and s["impls"] not in ("-", "cancelled")
                and s["moniker"].split("::")[0] == crate):
            got = seen.get(s["moniker"], [])
            want = int(s["impls"])
            # Compiler-generated marker impls that rust-analyzer's
            # goto_implementation does not list. Named, so a gap they
            # explain stays visible as MARKER_GAP rather than AGREE.
            unmarked = [g for g in got if g.split("/")[0] not in MARKERS]
            verdict = ("AGREE" if len(got) == want
                       else "MARKER_GAP" if len(unmarked) == want else "DISAGREE")
            # `trait/methods`: a zero-method impl (a compiler marker such as
            # StructuralPartialEq) is one rust-analyzer does not report.
            print(f"impls\t{s['moniker']}\tra={s['impls']}\tcharon={len(got)}\t{verdict}\t{','.join(sorted(got))}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
