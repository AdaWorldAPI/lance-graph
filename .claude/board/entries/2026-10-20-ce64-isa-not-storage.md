# 2026-10-20 — CE64 is an ISA contract: thought-aware in the hot path, arbitrary after a seal

**Status:** DECISION (operator, 2026-10-20). **Basis:** the operator's words: "Causaledge64 is an Isa contract / Not a storage contract / so when causaledge64 is having meta awareness in hot path it is thought aware / If it's lowered into lance seal, the in ram reference becomes arbitrary / The thinking styles used and eventually the reasoning replay should rather live in the thinking, or be linked with the kanban lane versions".

- Closes capstone §8 row 4 ("CE64 is not storage" vs "the result lives in 59..63"): both hold, in different places. In the hot path, inside a live thought, W names the lane and 59..63 carries the result. In a sealed Lance version W is arbitrary, because slot numbers recycle.
- Persisting CE64 does not make it a storage format: like BF16 written to disk, the bits are kept bit-exact and keep their ISA meaning; storage is the container, not the contract. Only a field whose referent exists at run time loses its meaning in a sealed copy: W, the slot number of a live thought (as a TCP port in a saved pcap still names a connection that is gone). (operator clarification, same day: "It's like TCP IP BF16 bit exact mantissa / if you save BF16 to Disk it's also ISA even if it's used in storage?")
- Extends entry 2026-10-17-ce64-one-layout D1 ("CE64 is not storage") to the seal boundary.
- Thinking styles used and the reasoning replay live in the thought (program or template id) or link to the kanban lane's Lance versions, never in sealed CE64 bits.
- Applied in `thought-git-api-v1.md` (correction item 7), `thought-git-basics-v1.md` (D-TGB-3, D-TGB-4), and the capstone (§8 row 4, field table row 53..58).

OPEN: which of the two homes (in the thought, or linked to the kanban lane version) a landed receipt uses; D-TGB-4 measures replay either way.
