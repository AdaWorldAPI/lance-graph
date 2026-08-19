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
