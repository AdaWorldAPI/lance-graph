# Clifford fold — proof of concept

A plan, not an emulator. The feasible slice from the handover notes: a stabilizer circuit is a fold over bit planes. A T gate is a refusal.

Code: `crates/lance-graph-lane-fold/src/clifford_fold.rs`.

## Claim

H, S, and CNOT rewrite rows of a tableau. A measurement is a parity of one masked row. The statevector is not built. A T gate does not enter the fold.

## Non-claim

This does not emulate a general circuit. It does not model noise. It does not put the fold on an optical mesh. Heap zero is not asserted: the tableau is four rows of bits, which is the state, and it is the population this slice is allowed to keep. A `2^n` amplitude buffer is the population it is not allowed to keep.

## Done when

`cargo test -p lance-graph-lane-fold --lib clifford` passes. Bell ZZ parity is even. A T gate is `Left`. No amplitude buffer exists in the module.
