# Seventeen rooms ahead

Each room deletes the intermediate the previous room still built. The residual is named at the door. A room that cannot name its residual is not a room.

The literature stops at room 2. Indexed stream fusion stops at room 3. The lane fold, as landed, is room 1 with the tools for room 4.

## 1. The pair list

Delete the N×M route. Keep an 8 KB mask. This is Abadi 2007 and Gill 1993. Residual: the mask. `pair_relation_bytes` must print 0.

## 2. The mask, when the fold is a homomorphism

Delete the aperture if every consumer is sum, count, min, or max of the same gate. Sum of sums is one sum. Residual: a non-associative consumer. A million word-folds at 0.6 ns is 0.6 ms only if this room is not taken. If it is taken, the million is one plane pass.

## 3. The zipper

Delete the alignment walk when the key is the `u16` rail. Tick `i` is tick `i+d`. This is the case arXiv:2507.06456 handles and shortcut fusion cannot. Residual: a key that is ordered but is not the address. That key gets one alignment, once.

## 4. The tile

Delete the 256-word walk when the tile gate is all zeros. Zone-map pruning at `TILE_WORDS`. Residual: a tile with one live word. It stays live. The word gate inside it still applies.

## 5. The fused pass

Delete a ternlog pass when the dead-word fraction is high, and delete the gate test when it is low. The count probe already showed fusion losing, 636 ns against 605 ns, on a hot dense plane. Residual: a plan that claims both. `Plan::check` refuses it.

## 6. The popcount

Delete the selectivity scan by keeping the live count on the aperture at the moment it is built. `popcount / 65536` is then a field, not a walk. Residual: an aperture built by a gate whose count was not threaded through. That one pays the 8 KB scan once.

## 7. The K masks

Delete K group planes when K fits in a histogram the word already addresses. One pass, K counters. Residual: K past the knee. Refuse, under a different name. Do not call the hash table a fold.

## 8. The validity plane

Delete the side bitmap for NULL by a reserved `u16` the rail does not use as a row. Three-valued AND reads that sentinel. Residual: a NULL that is a value, not an address. Strings and NULL literals stay out.

## 9. The shift add

Delete the `i+d` in the hot loop by rebasing the borrowed lane pointer. The fold reads `p[i]` and the shift is in the borrow. Residual: a shift that differs per row. That is a gather, and the gather is room 3's residual, not this one.

## 10. The fixture view

Delete `fixture_view_bytes`. The differential must hand the terminal a resident plane, not a reordered copy the harness built so the fold could look pure. Residual: a foreign lane that is genuinely not resident. That build is an ingest line, checksummed, not a fold nanosecond.

## 11. The plan vector

Delete the `Vec` of ops. A plan is a composition of borrowed apertures and one terminal. The op list is the tree this room exists to remove. Residual: a terminal that needs a scratch slot the composition cannot name. That slot is caller-owned, counted, and capped.

## 12. The crate boundary

Delete the wall between quack and mask-risc for the purpose of the rewrite. Fusion fails today because the producer and the consumer are in different crates and the second `foldr` cannot see the first `build`. Residual: the execution. Quack still must not evaluate. The rewrite sees both. The executor stays one.

## 13. The closed form

Delete the remaining pass when the fold has a closed form. Count of a full rail is 65536. Count of a shift window of width `w` is `w`. Sum of a constant lane is the constant times the popcount. Residual: a lane whose values are not a function of the address. That one walks.

## 14. The checksum

Delete the output buffer of the proof. xxh3 of the rows is a fold over the same aperture. The proof line is a terminal, not a materialised vector that is then hashed. Residual: a consumer who asked to see the rows. Cap it. Print it as materialization.

## 15. The refusal allocation

Delete the error object on the refusal path. A refusal is an enum, `Copy`, no string. The plan that cannot answer must be cheaper than the plan that allocates the route. Residual: a diagnostic the human asked for. That string is off the hot path and off the refusal path.

## 16. The note

Delete this file as a source of truth. The type system is the plan. `OrderedLane` does not construct from an unsorted buffer. `Row` does not construct from a `u32` that does not fit. A note that restates the type is already stale. Residual: the measurement. Types do not time a tile. The probe stays.

## 17. The residual that cannot be deleted

A non-associative consumer. A real zip. A string. A scattered extract the caller asked to see. A cyclic n-hop. These are not rooms. They are the door at the end of the corridor. A planner that deletes them has not deforested. It has changed the question. Name them, refuse them, and do not call the refusal a fold.
