# D-NFR-6 v2 — native operations over the SoaEnvelope borrow, no guard

**Status:** MEASURED, lab only (`crates/soa-view-probe`). No production contract, ABI or ClassId/LazyLock change. Supersedes `d-nfr-6-validate-once-v1.md`.

**Rule:** mutation safety belongs to `&self` / `&mut self`; the mathematics belongs to ndarray. A view of SoA rows is the contract's existing borrow (`NodeRowPacket`), and an operation over it is the ndarray kernel it names. Nothing between them is checked at runtime.

## 1. What v1 got wrong

v1 built a `Table` that held its own `FrameStore` beside the SoA, issued frames, and checked a frame on each operation through `bind_dr64` (measured: `FrameStore::check` 25.7–28.3 ns, 96 instructions, six `lock` RMWs; `bind_dr64` 59–67 ns). That re-derived ownership the contract already has. `SoaEnvelope`'s own docs say views are `&self` borrows of the owner, a batch-writer `&mut` cannot begin while a view is live, and "nothing is checked at runtime." The frame check guarded against a mutation the borrow checker already makes impossible.

Frame provenance (who sealed these bytes) and external authorization (may this caller read them) are real questions. They are answered by whoever hands out the packet, once; they are not a reason for a per-operation check. v1's conflation of the two is withdrawn, and so is its D-NFR-6a proposal (a faster `FrameStore::check`): making an unneeded guard cheaper is not the fix.

## 2. Design (lab)

| part | what it is | where |
|---|---|---|
| owner | the `Vec<NodeRow>` the mailbox holds; writes through `&mut` (`set_value_facet`) | caller |
| view | `NodeRowPacket<'_>`: `&self` borrow, 512-byte row stride, not `Clone` | `lance-graph-contract` (unchanged) |
| facet address | `FacetSlot`, one of the 32 16-byte slots; built once, in range by construction | `soa-view-probe` |
| operations | `FacetFold` on `NodeRowPacket`: `classid_mask` → `eq_u32_strided_to_mask`; `facet_match_mask` → `ternary_match_strided16_to_mask`; `payload_byte_sum` → `masked_strided_group_sum` over 12 × `u8` (never widened); counts via `popcount_batch_u64` | ndarray |

No new ownership type, no frame, no epoch, no dispatch.

**Assumption:** all mutation goes through the owner by ordinary borrowing (`&mut`). The guarantee does not extend to interior mutability or an `unsafe` write path beside the owner; `NodeRow`'s fields are plain byte arrays, so neither exists here.

## 3. Falsifiers

| claim | how it is pinned | disable run |
|---|---|---|
| no owner write while a view is live | `compile_fail,E0502` | — (rustdoc checks the code) |
| no view outlives its owner | `compile_fail,E0597` | — |
| no second holder of a view | `compile_fail,E0599` (packet has no `clone`) | — |
| a view taken after a write sees the write | runtime test + the compiling doctest twin | — |
| packet result = direct ndarray = row loop, 8 sizes (1..4096), both classids, mask/match/sum | `packet_operations_equal_direct_ndarray_and_the_row_loop` | classid offset +1 → 3 tests red; payload width 11 → 1 red |
| fixture is not vacuous | `the_fixture_is_not_vacuous` | one class only → red |
| facet geometry (slot 0 key, value facet 0 at byte 32, 32 slots) | `facet_slots_are_the_canonical_16_byte_grid` | covered by the offset disable |

## 4. Cost (reported, not gated)

`examples/view_cost.rs`, release, AVX-512 host, load ≈ 0.45. Medians of 21 batches; three runs agree within noise. A = the ndarray kernel on the raw LE bytes; B = the same kernel through `FacetFold` on the packet. Allocations are counted around the operation loop only.

| rows | op | A µs/call | B µs/call | ns/row (A ≈ B) | allocs |
|---:|---|---:|---:|---:|---|
| 64 | classid mask + count | 0.13 | 0.12 | ~2 | 0 / 0 |
| 64 | mask + payload sum | 0.33 | 0.30 | ~5 | 0 / 0 |
| 64 | ternary match + count | 0.06 | 0.06 | ~0.9 | 0 / 0 |
| 4096 | classid mask + count | 11.1 | 10.5 | ~2.7 | 0 / 0 |
| 4096 | mask + payload sum | 27.0 | 26.8 | ~6.6 | 0 / 0 |
| 4096 | ternary match + count | 10.3 | 10.3 | ~2.5 | 0 / 0 |
| 65536 | classid mask + count | 356 | 357 | ~5.4 | 0 / 0 |
| 65536 | mask + payload sum | 903 | 861 | ~13.5 | 0 / 0 |
| 65536 | ternary match + count | 335 | 336 | ~5.1 | 0 / 0 |

The per-row cost is the strided read itself (one 64-byte line per 512-byte row; 65536 rows is 32 MB), and it is the same in both arms.

Assembly (`cargo rustc --release --example view_cost -- --emit=asm`, `probe_a_*` vs `probe_b_*`): each pair calls the same ndarray kernels and nothing else. B is 4–7 instructions longer: one `shlq $9` (the `n_rows × 512` length `as_le_bytes` computes) and register moves for the different argument shape. No `lock`, no `cpuid`, no lazy initialization in any probe.

## 5. Answers

**Q1 — what does the old 27–28 ns guard execute?** Unchanged from v1: `FrameStore::check`, dominated by creating and dropping two temporary `Weak`s. v2's answer is that it should not execute at all on this path.

**Q2 — what does borrowing guarantee?** That no write to the rows happens while any view is live, that no view outlives the rows, and that a view is not duplicated into a second holder (§3, compile errors).

**Q3 — what needs runtime validation?** Not mutation. Arguments that the type system does not bound are bounded once, by construction (`FacetSlot`), not per call. Frame provenance and authorization are decided where the packet is handed out; they are outside this probe.

**Q4 — at which boundary?** Where the owner gives out `&rows`. That is a single point, and it is already the borrow.

**Q5 — can several operations reuse one view?** Yes; any number, with no per-operation cost (§4).

**Q6 — what does it cost against direct ndarray?** Nothing measurable: equal within run-to-run noise, zero allocations in both arms, same kernel calls in the asm (§4).

## 6. Acceptance

- [x] no `Table`, no duplicate ownership, no frame guard
- [x] view = `NodeRowPacket`, 512-byte row stride, 16-byte facets
- [x] every operation is one ndarray kernel
- [x] compile-time refusals pinned with error codes
- [x] packet = direct = row loop; 3 disable runs red
- [x] cost: equal, 0 allocations, asm shows only the kernel calls
- [x] fmt, clippy `-D warnings`, tests (own CI workflow `soa-view-probe.yml`)
- [ ] independent review
- [ ] operator authorization before merge
