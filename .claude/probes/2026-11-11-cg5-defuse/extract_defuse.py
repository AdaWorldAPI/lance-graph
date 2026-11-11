#!/usr/bin/env python3
"""LLBC -> def-use rows for every function that holds a Div or Rem.

Usage: extract_defuse.py <crate.llbc> <out.tsv>

Rows (tab separated, one per fact), per function `fn`:
  ARG   fn  local  name
  DEF   fn  local  stmt  kind  line  loop_depth  src,src,...   (one row per defining statement)
  ELEM  fn  local  stmt  callee                                (dest of a call to an iterator `next`)
  INDEX fn  local  stmt                                        (def reads through an index projection)
  SITE  fn  stmt   op    file  line  loop_depth  lhs  rhs      (operands: local id or `const`)
`kind` is assign | call | mutcall (a local reached through a `&mut` passed to a call).
Flow-insensitive: a local with two DEF rows has two definitions.
"""
import json, sys

def name_of(f):
    out = []
    for seg in f['item_meta']['name']:
        if 'Ident' in seg:
            out.append(seg['Ident'][0])
        else:
            out.append('{impl}')
    return '::'.join(out)

def span(st):
    sp = st['span']; sp = sp.get('Untagged', sp)
    return sp['data']

def base_local(place):
    k = place['kind']
    while 'Projection' in k:
        k = k['Projection'][0]['kind']
    return k.get('Local')

def place_locals(place, acc, idx):
    """Locals a place READ depends on: its base and any index operand."""
    k = place['kind']
    while 'Projection' in k:
        inner, elem = k['Projection']
        if isinstance(elem, dict) and ('Index' in elem or 'Subslice' in elem):
            idx.append(True)
            collect(elem, acc, idx)
        k = inner['kind']
    if 'Local' in k:
        acc.add(k['Local'])

def collect(o, acc, idx):
    """Every local read anywhere inside an rvalue / operand tree."""
    if isinstance(o, dict):
        if 'kind' in o and 'ty' in o and isinstance(o['kind'], dict) and (
                'Local' in o['kind'] or 'Projection' in o['kind']):
            place_locals(o, acc, idx)
            return
        for v in o.values():
            collect(v, acc, idx)
    elif isinstance(o, list):
        for v in o:
            collect(v, acc, idx)

def operand_id(op):
    for tag in ('Copy', 'Move'):
        if tag in op:
            b = base_local(op[tag])
            return 'global' if b is None else str(b)
    return 'const'

def find_binop(o):
    if isinstance(o, dict):
        if 'BinaryOp' in o:
            return o['BinaryOp']
        for v in o.values():
            r = find_binop(v)
            if r: return r
    elif isinstance(o, list):
        for v in o:
            r = find_binop(v)
            if r: return r
    return None

def blocks_in(o):
    """Nested blocks of a statement kind payload, with whether a Loop was crossed."""
    if isinstance(o, dict):
        if 'statements' in o and isinstance(o['statements'], list):
            yield o
            return
        for v in o.values():
            yield from blocks_in(v)
    elif isinstance(o, list):
        for v in o:
            yield from blocks_in(v)

def main():
    d = json.load(open(sys.argv[1]))
    t = d['translated']
    fd = t['fun_decls']
    files = t['files']
    rows = []
    for f in fd:
        if not f or not isinstance(f.get('body'), dict) or 'Structured' not in f['body']:
            continue
        s = f['body']['Structured']
        if not any(k in json.dumps(s['body']) for k in ('"Div"', '"Rem"')):
            continue
        fn = name_of(f)
        local_rows = []
        nargs = s['locals']['arg_count']
        for l in s['locals']['locals']:
            if 1 <= l['index'] <= nargs:
                local_rows.append(('ARG', fn, l['index'], l.get('name') or '_'))
        mutref = {}          # ref local -> borrowed local
        counter = [0]
        def walk(block, depth):
            for st in block['statements']:
                k = st['kind']
                counter[0] += 1
                sid = counter[0]
                if isinstance(k, str):
                    continue
                (tag, val), = k.items()
                line = span(st)['beg']['line']
                if tag == 'Assign':
                    place, rv = val
                    dest = base_local(place)
                    acc, idx = set(), []
                    collect(rv, acc, idx)
                    if 'Projection' in place['kind']:
                        # a write through a projection keeps the old value too
                        ps, _ = set(), []
                        place_locals(place, ps, _)
                        acc |= ps
                    local_rows.append(('DEF', fn, dest, sid, 'assign', line, depth,
                                       ','.join(map(str, sorted(acc))) or '-'))
                    if idx:
                        local_rows.append(('INDEX', fn, dest, sid))
                    if isinstance(rv, dict) and 'Ref' in rv:
                        r = rv['Ref']
                        if 'Mut' in json.dumps(r.get('kind', r))[:40]:
                            b = base_local(r['place'])
                            if b is not None:
                                mutref[dest] = b
                    bo = find_binop(rv)
                    if bo:
                        op = bo[0]
                        opname = op if isinstance(op, str) else next(iter(op))
                        if opname in ('Div', 'Rem'):
                            fid = span(st)['file_id']
                            local_rows.append(('SITE', fn, sid, opname, fid, line, depth,
                                               operand_id(bo[1]), operand_id(bo[2])))
                elif tag == 'Call':
                    call = val['call'] if 'call' in val else val
                    dest = base_local(call['dest'])
                    acc, idx = set(), []
                    collect(call['args'], acc, idx)
                    callee = '?'
                    try:
                        fid = call['func']['Regular']['kind']['Fun']
                        callee = name_of(fd[fid]) if isinstance(fid, int) and fd[fid] else str(fid)
                    except (KeyError, TypeError):
                        try:
                            callee = 'trait::' + json.dumps(call['func']['Regular']['kind'])[:0] + \
                                str(call['func']['Regular']['kind']['Trait'][1])
                        except (KeyError, TypeError):
                            pass
                    local_rows.append(('DEF', fn, dest, sid, 'call', line, depth,
                                       ','.join(map(str, sorted(acc))) or '-'))
                    if callee.split('::')[-1] == 'next' and depth > 0:
                        local_rows.append(('ELEM', fn, dest, sid, callee))
                    for a in sorted(acc):
                        if a in mutref:
                            local_rows.append(('DEF', fn, mutref[a], sid, 'mutcall', line, depth,
                                               ','.join(map(str, sorted(acc))) or '-'))
                else:
                    for b in blocks_in(val):
                        walk(b, depth + (1 if tag == 'Loop' else 0))
        walk(s['body'], 0)
        rows.extend(local_rows)
    with open(sys.argv[2], 'w') as out:
        out.write('#version 1\n#schema kind\tfn\tfields\n')
        for r in rows:
            if r[0] == 'SITE':
                r = list(r)
                nm = files[r[4]]['name']
                r[4] = nm.get('Local') or nm.get('Virtual') or json.dumps(nm)
            out.write('\t'.join(map(str, r)) + '\n')
    print(len(rows), 'rows')

main()
