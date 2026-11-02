# D-DRCA-3: DR #4 method routing falsifier (lab)

**Date:** 2026-10-30. **Status:** PROPOSED → implemented as a lab probe in
`crates/dr-math-binding-probe` (`src/dr4.rs`, `tests/dr4_method_routing.rs`).
**Parent:** D-DRDT (`d-drdt-dr-dt-ownership-amendment-v1.md`, #1835) §5 "Handoff:
DR #4", first falsifier. **Not:** a DT resolver, a MethodKey mint, an ABI, a
contract change, or method execution.

## Overview

#1835 §5 defines DR #4 as the route from a mapped coordinate set to a method
operation, and fixes what it must preserve and refuse. Its first falsifier:
*two methods on one cell with the same read/write sets and different effect
order must stay distinct through routing, and a dropped guard must refuse.*

DR stores neither a method's meaning nor its effects (#1835 §1). So the lab
treats every DT-owned value as **opaque**: the method key, the signature
revision, the guard witness and the effect-order reference are carried, never
interpreted. The meaning lives in the test's DT stub, which is also the
oracle that proves two routed methods really are different.

### Ownership boundaries kept

- **Method identity is DT's, and is not minted here.** D-CMH
  (`class-method-schema-hotplug-v1.md` §2.1, decision ledger item 2) leaves the
  16-byte `MethodKey` to the OGAR owner. The lab uses a test-minted
  `[u8; 16]`, named as such.
- **The DT resolver is D-CMH-1's `PreparedClass`, not this probe.** No second
  resolver, no contract type. `Resolution` here is a lab stand-in for DT's
  answer, constructed only by the test.
- **The effect-order law is #395's** (ndarray `examples/pof_antennas/order_law.rs`:
  `Effect`, `run_effects`, `commutes`, `check_reorder`), ported verbatim into
  the test as the oracle. DR does not decide commutation.

## Checklist

- [x] Read the parent contract (#1835 §4–§5), D-CMH (§2.1, §7), the #395 order law,
  and the existing lab API (`lib.rs`, `where_dr3.rs`, `hhtl16.rs`).
- [x] Tests first (`tests/dr4_method_routing.rs`), confirmed red before the module exists.
- [x] `src/dr4.rs`: `route` over one admitted `BoundDr64` and one `Resolution`.
- [x] Disable runs, each red on a named test.
- [x] README section, board entry, STATUS_BOARD row, entries index.

## Contract under test

`route(store, register, resolution, op, observed_guards) -> Result<Routed, Dr4Refusal>`

**Preserves** (one output entry per input method, in input order): method key,
signature revision, read and write field sets, guard witness, effect-order
reference, and the register's frame.

**Refuses**, in this order:

| refusal | when |
|---|---|
| `Frame(Refusal)` | the register's frame is not its store's current seal |
| `ResolvedUnderOtherFrame` | DT resolved the methods under another frame issue (another revision, a reseal or another store, even at equal numbers) |
| `ForeignClass` | the resolution or a method belongs to another class than the register |
| `DuplicateMethod` | one method key appears twice (two keys with equal sets are fine) |
| `OutOfRegister` | the operation's coordinates are outside 64 × 64 |
| `UnsupportedAlgebra` | the method declares an algebra DR does not have (a numeric fold) |
| `NotBoolean` | a presence method over a three-valued register |
| `GuardDropped` / `GuardMismatch` | a declared guard has no witness, or another one |

## Falsifiers

1. Two methods, equal read/write sets, effects `[Set(1), Add(1)]` vs
   `[Add(1), Set(1)]`: the oracle gives 2 vs 1 and `check_reorder` refuses the
   swap; routing returns two entries with their own effect references.
2. Commuting effects (`[Add(2), Add(3)]` vs reversed): the oracle gives equal
   results and the swap is legal; routing **still** returns two entries.
   Merging is not DR's decision.
3. A required guard with no witness refuses; a different witness refuses; the
   right witness is carried into the output. An unguarded method needs none.
4. Stale register frame, stale resolution, foreign store at equal numbers,
   foreign class, duplicate key, out-of-register, numeric algebra, presence over
   three-valued: each refuses with its own variant.
5. Values equal an independent scalar loop for cell, row and column, two- and
   three-valued (K3 value masked by validity).

## Not done

DT resolver (D-CMH-1); MethodKey mint; guard evaluation (DR only carries
witnesses); effect execution; DR #4 over a `LocalSelection` (D-DRCA-2) or a
`Chain` composition; production placement.
