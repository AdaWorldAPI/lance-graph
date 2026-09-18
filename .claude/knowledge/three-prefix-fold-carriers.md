# Three prefix-fold carriers — name the carrier before you move the fold

> **READ BY:** any session touching a prefix / LCP / shared-depth / distance
> fold, any `trailing_zeros` / `leading_zeros` / `popcount` readout, any
> "optimize this loop into a mask" proposal, `facet.rs`, `hhtl.rs`,
> `mailbox_soa.rs`, `mailbox_scan.rs`, or `ndarray::simd`'s masking ops.
> **MANDATORY** before changing any of the three folds catalogued below.
>
> Born 2026-09-17 from `E-THREE-CARRIERS-THREE-FOLDS-1`: a fold optimization
> measured on one carrier was shipped into another, where it is 2.1× slower.
> Nothing in the shared vocabulary ("prefix fold", "LCP", "shared depth")
> flagged the crossing.

## §1 The law

**A measurement is a statement about `(operation, carrier, workload)`.** Drop
the carrier and it becomes a slogan that will be applied where it is false.

**Masking wins when the slice is GRANULAR. PEEK wins when the slice is
ADDRESSED.** (Operator, 2026-09-17.) Granular = the datum has no byte address
of its own (a bit in a plane, a nibble in a packed word) and must be extracted
arithmetically, so you may as well extract all of them at once. Addressed = the
datum sits at a constant byte offset, so the CPU can load and compare it
directly and stop at the first mismatch.

**Corollary — PEEK is not a fallback, it is a different instruction sequence.**
A masked readout over an addressed carrier performs *the same loads* as the
PEEK and then pays extra to reassemble them, and it forfeits the early exit.
It is strictly more work, never less.

## §2 The three carriers

| # | name | type / site | granularity | stride | fold today | right fold |
|---|---|---|---|---|---|---|
| 1 | **bit-planes** | `MailboxSoA` / `MailboxSoaView::identity_plane_at → &[u64]` (`cognitive-shader-driver/src/mailbox_soa.rs`), `N × WORDS_PER_FP`; consumed by `graph::mailbox_scan::DistanceMeans::Hamming` | 1 bit | 64-bit word | mask + popcount | **mask** ✓ correct |
| 2 | **nibble path** | `NiblePath { path: u64, depth: u8 }` (`lance-graph-contract/src/hhtl.rs:251`), 16 nibbles, `MAX_DEPTH = 16`; consumed by `graph/mailbox_scan.rs:263` for CAKES nearest | 4 bits | 4-bit nibble | **per-nibble walk** | **mask** ✗ opportunity |
| 3 | **facet cascade** | `FacetCascade` (`lance-graph-contract/src/facet.rs`), `repr(C, align(16))`, `classid: u32` + `tiers: [FacetTier { lo: u8, hi: u8 }; 6]` | 8 bits | 16-bit tier, byte-addressed | **PEEK chain** | **PEEK** ✓ correct (since the 2026-09-17 revert) |

### Carrier 1 — bit-planes

Sub-byte with no addresses at all: a fingerprint bit has no offset you can name
at compile time. Every operation is necessarily whole-word arithmetic —
`popcount`, `ternlog`, `xor`-then-count. There is no early exit to forfeit
because there is nothing to exit from: Hamming distance needs every word.
**Masking is not merely right here, it is the only shape.** This carrier is
also where `ndarray::simd`'s masking ops and the `ternlogq` tail descent live,
and those results are sound on their own terms.

### Carrier 2 — the nibble path

Sub-byte but packed into ONE register. A nibble has no byte address, so
extracting nibble `t` costs a shift and a mask; extracting *all* of them costs
one `xor` and one `leading_zeros`. The shipped fold does the expensive thing 16
times:

```rust
// hhtl.rs:251 — as of 2026-09-17
while d < max {
    match (self.prefix(next), other.prefix(next)) {   // shift + reconstruct + Option
        (Some(a), Some(b)) if a.path == b.path && a.depth == b.depth => d = next,
        _ => break,
    }
}
```

The masked form is `((self.path ^ other.path).leading_zeros() >> 2)` clamped to
`min(self.depth, other.depth)` — root-first, so `leading_zeros`, not
`trailing_zeros`. **This is the one place in the tree where the 2026-09-16
instinct was correct and was never applied.** It is CONJECTURE until probed:
see §4.

### Carrier 3 — the facet cascade

Byte-addressed with compile-time-constant offsets. Tier `t`'s `lo` byte is at
`4 + 2t`, its `hi` at `5 + 2t`. In C64 terms: a **PEEK**. The address is known,
so the load is one instruction and the comparison can happen against the other
facet's memory without either side being assembled into anything:

```
movzbl 0x5(%rdi),%eax     ; tier 0 hi of a
cmp    0x5(%rsi),%al      ; straight against b's memory
jne    <exit>             ; first mismatch ends it
```

`[u8; 6]` in the source (`hi_chain` / `lo_chain`) **is a lens, not a
materialization** — LLVM never builds the array. That is what the operator
meant by *"`[u8; 6]` was meant to preserve zero copy"*: it names six addressed
bytes; it does not gather them.

The masked alternative (`xor` the `u128`, `&` an axis mask, `tzcnt`, subtract
32, divide by 16) issues the same `movzbl`s, then `shl`/`or`s them back
together, materializes a 64-bit mask constant, runs `tzcnt` + `cmove`, and
applies the offset correction — with no early exit.

**The `−32`** is where the carrier mismatch shows up in the source itself: the
classid's 32 bits sit *below* the tiers in the LE `u128`, so the mask removes
the classid's *bits* but not its *offset*. A correction term whose only job is
to undo a coordinate system the operation didn't want is a carrier smell.

## §3 Measured (2026-09-17, `examples/facet_axis_lcp_probe.rs`)

Four arms, 64K pairs, min of 7 runs, both axes, ns/op. All four arms verified
against the shipped API on every workload before timing (oracle-first); the
depth knob is verified to bind 0..=6.

| workload | A chain PEEK | B pack-to-`u64` PEEK | C masked `u128` | D PEEK, shared loads |
|---|---|---|---|---|
| random | **1.72** | 4.75 | 3.64 | 5.18 |
| depth 0 | **1.66** | 4.60 | 4.68 | 4.70 |
| depth 1 | **1.91** | 5.07 | 3.55 | 4.85 |
| depth 2 | **2.27** | 5.34 | 3.45 | 4.70 |
| depth 3 | **2.50** | 5.14 | 3.56 | 4.61 |
| depth 4 | **2.98** | 4.77 | 3.56 | 5.07 |
| depth 5 | **3.50** | 4.88 | 3.56 | 4.50 |
| identical | **3.30** | 4.46 | 3.73 | 4.49 |

Re-run post-revert reproduced the ordering and magnitudes (A 1.76 / C 4.48 on
random; run-to-run spread ≈ ±0.9 ns on the masked arms, ≈ ±0.05 on A).

Reading, including the parts that went against the author's prediction:

- **A wins everywhere**, and by most at shallow depth — the early exit is the
  mechanism. A's own depth-0 → depth-5 slope (+111%) is the anti-vacuity
  evidence that the knob binds.
- **B is the slowest arm.** Packing six addressed bytes into a `u64` to then
  `tzcnt` them costs twelve shift-or pairs — more than the compare it replaces.
  The prediction that B would win was wrong. *Recorded as wrong: a PEEK carrier
  does not want its bytes assembled, even into a register it could then mask.*
- **C only catches A at depth 5 / identical**, i.e. exactly where A has no early
  exit left to use. That is the honest boundary of the masked form's value on
  this carrier, and it is a tie, not a win.

## §4 What is NOT settled

- **What (16)'s 12.5 ns measured.** No harness was committed with it. The
  carrier-2 account is the strongest available explanation, not a finding.
- **Carrier 2's rewrite.** Unmeasured. `leading_zeros` on a `u64` is one
  instruction, but the depth-clamp and the `EMPTY`/ancestor edge cases are
  where a fast fold gets subtly wrong. Probe before rewriting.
- **Production callers of carrier 3's axis distances.** `hi_distance` /
  `lo_distance` have **zero** callers in this tree (out-of-tree consumers are
  not verifiable from here). The revert is therefore near-zero-risk *and* the
  performance of either form is currently unobserved by anything shipping.

## §5a A fourth fold, on carrier 3, under a named lens: BOUND (2026-09-18, D-DIAMOND-1)

Not a fourth carrier — §5's falsifier for that is unchanged and still holds
(no fourth carrier found; no carrier consumes another's fold). This is a
fourth FOLD on carrier 3 (the facet cascade), gated by a precondition the
other three folds don't have: **the population must be ordered under a named
lens** (`SemanticLens` — storage is a content-blind ordinal, "sorted" only
means something under a projection, and one physical sequence is monotone
under exactly one lens at a time). Given that, `SealedFacetLane::bound`
locates a contiguous row range with two `partition_point`s — cost O(log N),
answer size O(1) (two integers) — instead of visiting every row.

Measured (`crates/d-diamond-1-probe`, N=1M): bound 238–265 ns flat, vs a full
sweep at 429,100–910,240 ns. **119×–707× is the `bound + touched_write`
TOTAL against that sweep** — the comparable pair, since the sweep produces a
mask and the bound alone produces two integers. The bound-alone ratio against
the same sweeps is larger (≈1,619×–3,435×) and is not the number to quote: it
compares a range against a mask. A second, independently-ordered
population over the SAME rows (a correlated tenant lane) is not sorted under
the ontology's lens and gets no bound of its own by construction
(`WitnessError`, confirmed) — but a JOINT lens built over both (a Morton
interleave, probe-only, `JointIndex`) turns their intersection into ONE bound:
69–79 ns, or **89–98 ns including `materialize_rows`** — the remap back to the
world's own ordinal, which is the output the comparator actually produces —
against 745,473–797,268 ns for two sweeps + AND: **8,135×–8,376×**, quoting
the materialized column. The win is **conditional on a prebuilt `JointIndex`**
(≈61 ms per 1M rows, a real one-time cost amortized over queries), never a
free property of the substrate.

**The corollary this fold enforces, learned the hard way mid-arc
(`E-NO-FOLD-REPORTS-AN-O-POPULATION-COST-1`):** a bound's answer is `(lo, hi)`
— two integers. Turning it into a mask sized to the WHOLE lane (rather than to
`hi` alone) or feeding its range into a sweep (rather than a narrowed
AND/popcount) silently reintroduces the O(N) cost bound exists to avoid. The
fold's cost must be a function of the ANSWER's size, never of the lane's.

Full measurement and the BOUNDED verdict:
`.claude/plans/d-diamond-1-dual-fold-substrate-v1.md` §5.

## §5 Falsifier for this page

- The four-arm probe is in-tree, runs in ~2 s, and is the falsifier for §3. If
  a future toolchain inverts the ordering, this page is wrong and says so by
  failing to reproduce.
- §2's carrier assignment is falsified by finding a *fourth* carrier, or by
  finding one of the three consuming another's fold. Both are greps that must
  come back empty: `FacetCascade` in `mailbox_soa.rs` (it does not appear);
  `trailing_zeros`/`leading_zeros` in `hhtl.rs` (it does not appear).
- The law in §1 is falsified by a measured case where a masked readout beats a
  PEEK on a byte-addressed carrier with a live early exit. None is known.
