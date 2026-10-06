# 2026-10-06 — D-CE64-NEXTSTATE-0: the CE64 register has clean sparse next-state semantics

## VERIFIED-IN-CODE — inventory

1. **Physical representation.** `CausalEdge64(pub u64)`, `#[repr(transparent)]`;
   `MailboxSoA::edges` is `[CausalEdge64; N]`, read zero-copy as `&[u64]` by
   `MailboxSoaView::edges_raw`. A native `u64` from `pack_v2` through
   `write_row`, `tick` and the view.
2. **Layout.** `causal_edge::layout`: S 0..7, P 8..15, O 16..23, F 24..31,
   C 32..39, Pearl 40..42, direction 43..45, mantissa 46..49, plasticity
   50..52, W 53..58, truth 59..60, spare/band 61..63; a const assert covers all
   64 bits exactly once. Bit 0 = LSB.
3. **Endian contract.** None stated for the v2 edge before this change: it had
   no `to_le_bytes` (only `CausalEdgeV3`, 12 bytes, and the sibling
   `EpisodicEdges64` did), though `soa_envelope.rs` said it had. No production
   path persists a v2 edge as bytes; no code reads it in host byte order
   (`facet.rs` is the only `target_endian` guard, for facets).
4. **Masked update helpers.** `with_truth`, `with_topology`, `with_spare`,
   `with_reasoning_band`, `with_routing` and the `set_*` twins are each one
   mask and insert. No joint 5-bit writer exists; the #1370 law reads
   `spare() << 2 | truth_raw()` and writes with `with_spare` + `with_truth`.
5. **Temporal contract.** `write_row` is cycle-gated (`Accepted` only when
   `cycle == current_cycle`) and stamps `last_write_cycle[row]`; the write
   lands in place. There is no double buffer, so "k vs k+1" is implicit, not
   enforced.

## MEASURED

`ce64_nextstate_probe.rs`: 6 tests, 7 disable runs red (no clear, clearing W
too, no write, no stamp check, big-endian persist, lossy restore, big-endian
`CausalEdge64::to_le_bytes`).

- **Codegen** (x86-64, release, `--emit asm`): the shipped two-writer path
  is `bzhi rax, rdi, 59; shl rsi, 59; or rax, rsi; ret`, and the masked
  insert compiled to the identical body; LLVM folded the two functions.
- **Same-cycle authority:** a plain read in cycle k already sees the write. A
  committed-only read (refuse the row while `last_write_cycle == cycle`)
  gives `None` until `tick`, but the pre-write value is gone; cycle k keeps
  its start-of-cycle copy.

## DECISION

- **SCOPE:** no `with_epistemic_state`. The machine code is already the
  minimal sparse update, and a fifth name over bits 59..63 (already read as
  `truth`, `topology`, `spare` and `reasoning_band`) would add a lens, not
  correctness.
- **Added:** `CausalEdge64::to_le_bytes` / `from_le_bytes`, mirroring
  `EpisodicEdges64`, and the register contract (bit numbering, LE form,
  sparse update, cycle semantics) on the type.
- **REVISIT WHEN:** a production writer of bits 59..63 lands (the gap in
  `ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE`) and needs a named joint
  field, or a reader needs enforced next-cycle authority (double buffering).

## OPEN

- The evidence → EpistemicState5 certification step still has no production
  writer (`ISS-NO-EVIDENCE-WRITER-FOR-EPISTEMIC-STATE`, unchanged).
- Next-cycle authority is not enforced by the mailbox.
