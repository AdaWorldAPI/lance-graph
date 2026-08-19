# The Cascade-Accumulated Seal (operator STORNO + spec, 2026-08-19)

> **STORNO'd:** "replace FNV with a faster whole-cycle seal hash." The target
> is NOT another whole-cycle hash. The Morton-cascade image already exists in
> memory before persistence; **integrity is accumulated as that image becomes
> resolved.** This supersedes the consolidation's Tier-1 seal item's wording
> and the X-C2-3 framing of the digest as a post-hoc scheme.

## Requirements (operator, verbatim)

1. No post-finalization full-cycle payload rescan.
2. No storage reread.
3. No encryption.
4. No physical-order dependency.
5. Digest binds canonical locus + resolved/present state + content.
6. Petal digest computed while petal bytes are hot.
7. Higher cascade levels reduce child digests only.
8. Final root exists before Lance publication.
9. Lance publishes image + root in one durable DatasetVersion.
10. DatasetVersion is publication identity; root is content identity.
11. Benchmark CRC32C/BLAKE3/etc only inside this architecture.
12. FNV whole-cycle serial pass is deleted, not optimized.

## Design mapping (probe: `crates/rp-seal-t0-probe/src/cascade_seal.rs`)

- **Petal digest** = `D(slot ‖ version ‖ resolved-flag ‖ content)`, computed
  at the moment the petal's bytes land (req 5, 6). An unresolved petal at
  finalize digests as `D(slot ‖ version ‖ UNRESOLVED)` — absence is part of
  content identity, so erasure/duplication cannot be root-invisible (req 5).
- **Interior node** = `D(level ‖ index ‖ child-digests in canonical child
  order)` — digests only, payload never re-touched (req 1, 7). Fanout 4
  (the Morton 2bit×2bit cascade), parameterizable.
- **Online bubble-up:** a node reduces the moment its last child resolves,
  so after the final petal lands only that petal's root path (log₄ n
  reduces) remains — the root exists before publication with no finalize
  pass (req 8). Keying is by canonical tree position, never arrival order
  (req 4).
- **Publication:** the writer hands Lance the image + the root in the one
  existing durable commit (req 9); `DatasetVersion` = publication identity
  (T0.3's W_write), root = content identity (req 10). The two identities
  are independent by construction: a republication at a new version with
  identical content keeps the root; identical version can never carry two
  roots (the fence).
- **No encryption anywhere** (req 3); digests are integrity checksums.
- **Verification of a read** recomputes the touched petal's digest + its
  path against the published root — no storage reread of anything the read
  did not already load (req 2). Interior digests MAY be stored for O(log n)
  localization (a Merkle-proof-shaped option); the minimum durable artifact
  is image + root.
- **Digest primitive choice** (CRC32C / BLAKE3 / xxh3 / …) is made ONLY
  from the in-architecture benchmark (req 11):
  `examples/cascade_seal_bench.rs` measures petal-digest-while-hot +
  reduction + root-after-last-petal latency + single-petal incremental
  re-seal, under randomized arrival order, against the DELETED whole-cycle
  FNV as the contrast row. Isolated hash throughput numbers are
  inadmissible for this decision.
- **Req 12 (the deletion)** is the follow-on substrate PR:
  `DetachedCycleBatch::content_hash`'s serial FNV pass is removed and the
  accumulated root becomes the batch's content identity. The durable
  `(cycle, batch_hash)` reconciliation identity changes algorithm, so the
  landing carries a seal-version gate per I-LEGACY-API-FEATURE-GATED
  (stored FNV-era FrameMeta stays readable; new cycles mint the root).

## Falsifiers (each disable-verified)

- **F-SEAL-ORDER:** any arrival permutation → byte-identical root.
- **F-SEAL-NORESCAN:** payload bytes digested == payload bytes written,
  exactly once; finalize touches zero payload bytes.
- **F-SEAL-PRESENCE:** an unresolved petal yields a different root than the
  same cycle fully resolved; erasing a resolved petal's digest is
  impossible without the root changing.
- **F-SEAL-ROOT-LATENCY:** after the last petal, only that petal's path
  (log₄ n node reduces) runs before the root exists.
- The X-C2-1 injection matrix applies to the root: I3/I4/I5/I6 all move it
  (locus + version + presence binding).


---

## ARCHITECTURAL CORRECTION (operator, 2026-08-19) — LOTUS IS A REGISTER GRID, NOT A BYTE GRID

> Verbatim. Outranks any conflicting sentence above; the ⊘ marks in
> `SEAL-FINALIZATION-MAP.md` apply it.

The Morton/Lotus cascade does NOT own or materialize witness payload bytes.

Canonical target:

    SoA backing store owns the 512-B rows.

    Lotus/Morton holds:
        canonical register/locus
        pointer/descriptor to the SoA row
        resolved/present state
        phase/closure state
        tiny digest state if needed

    It does NOT hold:
        copied 512-B rows
        materialized 8-KiB petals
        a materialized 32-MiB cycle image

The 8-KiB "petal" is a logical group of 16 registers/pointers, not an
8192-byte buffer.

**PHASE / ORDER.** Do not introduce another sort/materialization step
merely to obtain canonical temporal order. The intended Lotus property:
phase + canonical register position CONSTRUCT the ordering — never
arrival bytes → sort later → reorder physically. The phase washes
descriptors/registers into deterministic temporal position. Payload bytes
do not move. Source-audit the exact existing phase/register mechanism and
prove this before creating anything new.

**ZERO-COPY CONTRACT.** batch_writer.rs already documents the target
shape: P is a descriptor; payload bytes remain in the SoA backing store;
the sink reads them through NodeRowPacket::as_le_bytes at flush. Treat
the current BatchWriter<Vec<u8>> path as implementation debt / interim
wiring, NOT architectural ownership. Do not hash payload at
BatchWriter::cast merely because it is currently available there. The
descriptor layer stays content-blind.

**DIGEST SEAM.** The correct payload digest seam is the ONE unavoidable
dereference used for persistence. For each final resolved Lotus register:
ptr → existing SoA bytes; then, in the same payload read: bytes → Lance
serializer AND bytes → leaf digest (binding canonical locus,
resolved/present state, base_version/generation, bytes). NO separate
payload traversal. NO bytes copied into Morton/Lotus. NO
post-finalization payload hash pass. NO pre-hash pass at cast time.
Higher levels reduce DIGESTS ONLY.

**LOTUS SHAPE.** A petal = 16 register positions + resolved mask +
pointers/descriptors + digest state — NOT 16 × 512 B materialized
payload. Closure = all required register states resolved. Then
4096 → 1024 → 256 → 64 → 16 → 4 → 1 is a hierarchy of register/digest
closure, not copied payload.

**IDENTITY.** Separate: ContentRoot (final durable referenced content,
bound to canonical Lotus loci) · ControlRoot (tiny persisted
trajectory/control metadata if required) · DatasetVersion (publication
coordinate returned by Lance). Possible batch identity:
H(cycle ‖ base_version ‖ ControlRoot ‖ ContentRoot). Do NOT hash
superseded payload bytes that do not survive durably merely because old
FNV did.

**MAXIM.** MORTON ORDERS ADDRESSES, NOT PAYLOAD. LOTUS CLOSES REGISTERS,
NOT BYTE BUFFERS. THE BYTES STAY IN SoA. THE POINTERS CLICK INTO THE
GRID. THE PHASE GIVES THEM TIME. THE ONE WRITE-SIDE DEREFERENCE PAYS FOR
BOTH: persistence + integrity. ZERO COPY UNTIL THE MEMBRANE.
