# Substrate synergies to test

Synergies between the Cartesian stencil and the plans already written. Each row is a test, not a feature. `hexagon-plasticity-v1` remains the hex plan. This note is the Moore-and-Queen comparison it does not replace.

`CausalEdge64` is not touched. A test that allocates a pair list has failed, whatever else it shows.

## S1 — Moore mask is eight shifts

A cell ordinal and one byte. Bit `i` selects neighbor `i`, computed by an arithmetic shift of the trie code. The fold reads the strength byte only where the bit is set.

Pass: eight neighbors, no allocation, skipped bits unread. Fail: a neighbor lookup that builds a pair, or a diagonal that a hex rail would not have, silently treated as a hex face.

## S2 — Queen mask is three bits, not the Moore byte

A heading is 3 bits, six faces. A turn cue is 2 bits. The Queen fold selects one face, plus an adjacent face only if the turn bit is set.

Pass: a Moore byte and a Queen heading applied to the same ordinal produce different neighbor sets, and the test prints both. Fail: the 8-bit mask used as a heading, or a Queen move landing on a square-lattice diagonal.

## S3 — Hebbian write stays in the byte

Co-firing of center and neighbor `i` saturates `strength[i]`. The neighbor relation is not allocated.

Pass: the ordinal set is unchanged after a thousand co-firings. Only the byte moved. Fail: a co-firing that mints a SPOFC triple or a new edge.

## S4 — Bayesian byte is a count

Hits and misses on bit `i` update a count. The strength used by the fold is the quantized mean. Fisher-z is used only if the averaged value is a correlation. A count does not pass through `arctanh`.

Pass: a count-only cell never calls the Fisher-z path. A correlation cell stores a palette bin, not an `f64`. Fail: a `f64` average whose byte is only a display code.

## S5 — Local header is the mask plus eight bins

Attention, if tested, is a softmax over the set bits of the Moore byte and a weighted read of those neighbors. Radius 1.

Pass: a header cannot name a cell outside the stencil. Fail: a head over a sequence, or a score computed against the palette vocabulary.

## S6 — SPOFC is not the wire

The semantic coordinate stays a SPOFC triple. The energy stays in the strength byte. A hot bit does not promote itself.

Pass: a fold returns energy and the same SPOFC it was given. Fail: a promotion from a hot neighbor into a new triple. That is the BPE bag inside the cell.

## Order

S1 before S2. S3 and S4 share S1's byte and can run together. S5 after S1. S6 last, because it is the one that says the others did not smuggle a pair.
