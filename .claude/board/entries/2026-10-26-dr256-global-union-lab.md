# 2026-10-26 — D-DR256 lab: 64 chaotically mapped DR64 registers -> one global 8 KiB mask

**Status:** MEASURED (lab probe). Decision: the global mask **EXISTS** as a demanded result; using it as an input to
further Boolean algebra is a **CONTRACT BLOCKER**, except for one legal form (`G AND h`, below). Nothing in
production changed.
Probe: `crates/drf-focus-probe/examples/global_union.rs` (workspace-excluded lab crate). Run from that crate:
`cargo run --release --example global_union`.

## Source facts

- **Mapping representation.** One register = `drf_focus_probe::SealedMap { id, seal, ord: Box<[u16; 4096]> }`
  (`crates/drf-focus-probe/src/lib.rs:48-51`). `ord[slot]` is the slot's canonical address in the 65,536 universe,
  so the global universe is exactly the u16 domain. `membership::BoundMap::bind` checks seal == source
  `index_seal` and every ordinal < `n_rows` (`membership.rs:38`). `RowSet` is already a 1,024-word global bitset
  (`membership.rs:22`). The existing direction is global → local (`membership`, `membership.rs:103`); the
  union asked for here is the reverse, local → global.
- **Scatter.** `ndarray::simd::mask_scatter_or_u32` ORs `src` bits into `out` at `index[i]`, cost ∝ selected bits,
  repeats union, and does not zero its output, so 64 calls union (`simd_masking_ops.rs:1322`, test `:6700`;
  ndarray `195844bd`). `Terminal::ScatterOrU32` wraps it (`ir.rs:322-341`), but the executor zeroes `Out::Mask`
  at the start of each program (`exec.rs:1604-1607`). So the 64 registers must run as ONE program over a
  262,144-row population: register r at rows `r*4096..`, masks laid out contiguously, one u32 index lane.
- **The restriction.** `ScatterOrU32`'s survival condition (`ir.rs:329-337`): the scattered mask "is legal only
  when it IS the externally demanded result … It is never an intermediate: a follow-on program or fold that
  consumes it is the forbidden projection → population → projection shape." TERNLOG combines aligned planes
  (`MaskOp::Ternlog`, `ir.rs:250`); it never permutes addresses.
- **What stays legal.** A Boolean function h of global resident planes can be built with `Ternlog` + `Keep` and
  then pushed down: `Gather` h through each map under `L` (`MaskOp::Gather`, `ir.rs:277`, gated form), then
  scatter. This gives `scatter(L AND h∘ord) = G AND h` without reading G. `G OR h`, `NOT G`, `G XOR h` and
  `G1 AND G2` of two scattered masks are not expressible through the current mask-risc terminal; they would
  consume G. That is not "not expressible at all": `mask_scatter_or_u32` preserves its destination, so seeding
  the demanded output with h before the scatter could yield `G OR h` as one terminal result without an
  intermediate G. mask-risc zeroes `Out::Mask` first (`exec.rs:1604-1607`), so this is a hypothesis for a
  later contract review, not a capability. The demonstrated consumer needs neither: G is the demanded output,
  and its zero bits already exclude the other 61,440 addresses without a complement.

## Results (two runs, avx512f, 4 cores, load 0.9–2.9; ns per run, median of 31 batches; reported, not gated)

| fixture | selected slots | global bits | nonzero words | bind | widen | ndarray x64 (B1) | mask-risc 1 prog (B2) | oracle (A) |
|---|---|---|---|---|---|---|---|---|
| chaotic, 50 % | 130,990 | 56,927 | 1,024 | 86–108 µs | 35–37 µs | 157–168 µs | 198–227 µs | 16,753–16,813 µs |
| clustered 64×64 | 4,096 | 4,096 | 64 | 59–76 µs | 37–43 µs | 9.0–12.0 µs | 14.7–19.0 µs | 624–712 µs |
| scattered 4,096 | 4,096 | 4,096 | 1,008 | 61–73 µs | 38–45 µs | 11.1–11.2 µs | 18.1–19.4 µs | 666–700 µs |
| empty | 0 | 0 | 0 | 58–59 µs | 36–37 µs | 1.6–2.1 µs | 6.2–7.4 µs | 243–251 µs |
| full | 262,144 | 65,536 | 1,024 | 58–82 µs | 37–40 µs | 267–411 µs | 376–488 µs | 27,856–28,430 µs |

- Parity: B1 and B2 equal the oracle exactly on every fixture.
- Scatter cost is about 1.0–1.6 ns per selected slot. The oracle is a `BTreeSet` reference arm, reported in µs and
  not gated.
- Bind (64 seal and range checks over 262,144 ordinals) and widen (u16 → u32, a 1 MiB derived lane) are charged
  separately, and at small selections they cost more than the scatter.
- `G AND h`: h = `x & (y | !z)` over three global planes, 20,078 bits. Building h: 0.38–0.81 µs. Gated
  gather + scatter: 11.5 µs (empty), 26.8–41.0 µs (4,096 selected), 422–819 µs (full / 50 %). Parity is exact on
  every fixture.
- The clustered block occupies 64 of 1,024 mask words, and the scattered set 1,008. That is selection
  geometry, not a statement about Lance bytes (P1, #1789, failed).

## Controls (all asserted)

- Identity map: G equals L zero-extended.
- Chaotic maps: collisions occur (130,990 selected slots → 56,927 members).
- Clustered: exactly 4,096 members, 61,440 excluded.
- Empty and full fixtures.
- Two local coordinates of one entity give one member, with no false separation. Equal slot numbers in two
  registers naming different entities give two members, where a slot-number union would give 1: no false merge.
- Stale seal → `StaleMap`; ordinal outside a declared 61,440 universe → `OrdinalOutOfRange`; one register at
  another source version → `DomainMismatch`; another scope → `ScopeMismatch`. All refuse before any union.
- Negative controls, asserted in the probe on every run (added after codex review on #1790; the first version
  reported these from uncommitted disable runs, which the command could not reproduce):
  - index XOR: flipping bit 15 of every index (a bijection on the universe) must make B1 disagree with the
    oracle on chaotic, clustered and scattered, and must leave `empty` and `full` equal. Bit 0 would not do:
    the clustered block (x 64..128) is closed under XOR 1, and a run with bit 0 fails on that fixture.
  - gather gate: the `G AND h` program without its `under` gate must disagree with the oracle on every
    fixture whose L is not all ones, and agree on `full`.
  - Each control goes red when its perturbation is removed (`^ 0`; gate kept), so it cannot pass vacuously.
- The Boolean union collapses multiplicity (2 selected slots → 1 member), so G is never a count, a Bag, a
  provenance record or a K3 result. Inputs must already be TRUE-only.

## Gaps (named, not built)

- No u16-index scatter or gather: the u16 maps are widened to a 1 MiB u32 lane per 64 registers.
- No multi-register union witness: per-map `bind` exists, but "all 64 bound to one source and one scope" is
  checked only in the probe.
- Running through mask-risc needs the 64 local masks contiguous (one population); a scattered layout would need a
  copy or 64 ndarray calls.
- Keeping G resident for reuse would be a maintained inverse; D-HOLO H5 found that is a cost, not a default.

OPEN: further Boolean operators over G wait for a concrete consumer (the current one needs only G); whether G should ever feed a sealed-read `RowAddrMask` (separate decision; MemWAL has no such input).
