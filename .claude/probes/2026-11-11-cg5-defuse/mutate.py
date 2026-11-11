import sys
p='src/method.rs'; s=open(p).read()
LOOP="""        all.push(SetExpr::i32((STOCK0 + k) as u16, Cmp::Ge, d * q));
    }
    // staff · per_staff ≥ q  ⇔  staff ≥ ⌈q / per_staff⌉ for positive per_staff.
    let staff = (q + r.per_staff - 1) / r.per_staff;
"""
HEAD="    for &(k, d) in &r.demand {\n        all.push(SetExpr::Sym((KNOWN0 + k) as u32));"
DIV="    let staff = (q + r.per_staff - 1) / r.per_staff;\n"
m=sys.argv[1]
def rep(old,new):
    global s
    assert s.count(old)==1, old
    s=s.replace(old,new)
if m=='m1':   # division moved into the loop, per-element divisor
    rep(LOOP, LOOP.split('    }\n')[0]+"        staff = (q + d - 1) / d;\n    }\n"); rep(HEAD,"    let mut staff = 0;\n"+HEAD)
elif m=='m2': # division removed
    rep(DIV,"    let staff = q + r.per_staff - 1;\n")
elif m=='m3': # division moved into the loop, operands unchanged
    rep(LOOP, LOOP.split('    }\n')[0]+"        staff = (q + r.per_staff - 1) / r.per_staff;\n    }\n"); rep(HEAD,"    let mut staff = 0;\n"+HEAD)
open(p,'w').write(s)
