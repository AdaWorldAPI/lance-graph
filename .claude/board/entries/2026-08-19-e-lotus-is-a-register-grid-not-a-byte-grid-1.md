## 2026-08-19 — E-LOTUS-IS-A-REGISTER-GRID-NOT-A-BYTE-GRID-1

**Status:** RULING `[operator]` (correction to the #968 seal map) +
archaeology verdict.

**The correction, in one line:** Morton orders ADDRESSES, not payload;
Lotus closes REGISTERS, not byte buffers. The cascade holds canonical
locus + pointer/descriptor into the SoA backing store + resolved/present
state + phase + tiny digest state — never copied 512-B rows, never
materialized 8-KiB petals, never a 32-MiB image. A petal = 16 register
positions + resolved mask + pointers + digest state. Phase + canonical
register position CONSTRUCT the ordering (no sort step). The digest seam
is the ONE unavoidable flush dereference: the same hot read feeds the
Lance serializer AND the leaf digest — the checksum is parasitic on the
byte stream that must cross the membrane anyway. Identity splits
ContentRoot (final referenced content at canonical loci — superseded
bytes deliberately NOT hashed) / ControlRoot (tiny trajectory metadata) /
DatasetVersion (publication coordinate);
BatchIdentity = H(cycle ‖ base_version ‖ ControlRoot ‖ ContentRoot).
Seam A/B in the prior map was the wrong question — cast stays
content-blind (descriptor purity), freeze freezes registers not bytes.
Payload is touched exactly TWICE ever: production + the flush
dereference. Full text verbatim in CASCADE-ACCUMULATED-SEAL-SPEC.md.

**Archaeology verdict (mandated before ratification): the substrate
EXISTS, operator-ruled, UNWIRED.** `NodeRowPacket` (canonical_node.rs:1511)
is the zero-copy SoaEnvelope over `&[NodeRow]` — deliberately NOT
Clone/Copy so a borrow cannot escape its mailbox; batch_writer.rs
Addendum-6 rules P = descriptor with flush-time `as_le_bytes`; SweepSlot
.payload's own doc says "a NodeRowPacket slice in production; bytes
here". But `as_le_bytes` has NO live caller (only the deprecated symbiont
bridge + tests) and `cast()` has zero production call sites — the
`BatchWriter<Vec<u8>>` + byte-cloning freeze path my map documented is
confirmed interim wiring. **Consequence: the seal work WIRES the declared
contract to persistence; it creates no payload Morton tree.** Map §6
carries the full seven-question table.

