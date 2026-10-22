# 2026-10-22 — D-DRMC-5: associativity probe started

**Status:** probe slice 1 landed on `lab/dr-math-5-assoc-probe-20261022`
(stacked on #1704). Operator request: "start the associativity probe".

`crates/dr-math-binding-probe/src/relcomp.rs`: typed composition chain over
bound DR64 operands (domains, frame issue, focus, scope, Boolean semiring
only); law admission for `BooleanCompose` over two-valued Distinct; egglog
proof of reassociation from the admitted rule, with commutation refused;
push/compose lowering with a round-trip check; `certify_reassociation`, the
first issuer of a `RecipeCertificate`.

Measured (example `assoc_orders`, release, 4-vCPU shared Xeon @ 2.10 GHz):
14 orders of one dense-to-sparse 5-chain, same relation, 6 414 to 15 310
row-ORs, about 4.4 to 10.8 µs per evaluation. Cold egglog proof about
12.5 ms, reported, not a gate.

OPEN: specialization selection (choose an order without evaluating all of
them), certificate rebinding to a second frame (X8), any GF(2)/M4RI
reduction, R2IL/Quack wiring.
