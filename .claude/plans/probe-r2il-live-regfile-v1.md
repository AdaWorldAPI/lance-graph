# PROBE-R2IL-LIVE-REGFILE — plan v1

> **Status:** GREEN (2026-08-26) — 18/18, seven disable runs red-then-green.
> Code: r2sleigh `crates/r2conc/tests/{live_regfile,mos6502_oracle}.rs`
> (feature `probe-6502`). Design ratified below; measured
> facts are marked as such and were filled in only after the measurement
> ran. Parent doctrine: `r2il-machine-semantic-contract-v1.md` §7.8.
> Prerequisite (SHIPPED): `r2conc`, the concrete slab executor
> (r2sleigh PR #5, merged).

## 1. What this probe claims, and what it does not

§7.8 states the V4 space binding as doctrine:

> `register` = the node's own 12-byte facet register · `ram` =
> classid-prefixed SoA lanes (the GUID is the pointer) · `unique` =
> never-persisted scratch.

PROBE-R2IL-LIVE-REGFILE is the falsifier for the executable half of that
claim. It asserts exactly one thing:

> **A real 6502 routine, lifted to R2IL by Ghidra's own SLEIGH spec and
> executed through `r2conc::SlabState` with the machine state bound to
> borrowed slabs, produces byte-identical architectural state to an
> independently-written reference 6502.**

It does **not** claim: cycle accuracy, full instruction coverage,
correctness of the SLEIGH spec itself, or anything about the effectful
fragment beyond the `Store`s this routine performs. Those are out of
scope and named here so a later reader does not inherit an overclaim.

## 2. The validity problem, and the three-leg answer

The obvious failure mode of any differential test is that **both sides
share one author's misconception**, agree, and prove nothing. This probe
is designed against that failure specifically, and the design is the
substance of the plan.

### Leg 1 — side A's semantics are NOT mine

The R2IL under test is **not hand-written**. It is produced by
`r2sleigh_lift::Disassembler::from_sla(SLA_6502, PSPEC_6502, "6502")`
lifting real machine-code bytes through Ghidra's own compiled 6502
SLEIGH specification (shipped by the crates.io `sleigh-config` crate,
feature `6502`, which compiles Ghidra's vendored `6502.slaspec`).

That matters more than it first appears: the instruction semantics on
side A were authored by Ghidra's SLEIGH authors, years ago, with no
knowledge of this workspace. They are genuinely external. What this
session contributes on side A is only the *execution* of those
semantics (`r2conc`) and the *sequencing* around them.

### Leg 2 — side B was written in enforced isolation

The reference emulator (`crates/r2conc/tests/mos6502_oracle.rs`) was
written by an agent under a hard isolation rule: forbidden from reading
`r2conc`, `r2il`, `r2sym`, or anything naming R2IL/SLEIGH/Varnode/p-code,
and forbidden from copying an existing emulator. It was given the opcode
list and the struct shape, and **deliberately not given the flag
semantics, the status-bit layout, the branch offset arithmetic, or the
rotate/carry behaviour** — precisely the parts that must be independent
for the test to mean anything.

This is real isolation rather than theatrical isolation: the two sides
were produced by different processes from different sources, and neither
could see the other while being written.

### Leg 3 — arithmetic ground truth, external to both

Legs 1 and 2 could still both be wrong in the same way by coincidence.
Leg 3 removes that: the routine under test is an **8×8→16 unsigned
multiply**, so its correct answer is a fact about *arithmetic*, not about
any implementation. The test asserts the 16-bit result equals
`(a as u16) * (b as u16)` computed by Rust.

Operand choice is load-bearing: the pairs must include products **> 255**
so the high byte is non-zero. A routine (or an executor) that silently
drops the high byte passes `7 × 13 = 91` and fails `55 × 13 = 715`.

Leg 3 is reinforced inside the oracle itself by the canonical eight-case
ADC signed-overflow truth table, which pins the flag semantics against
published reference material rather than against the oracle author's own
reasoning.

**Together:** Ghidra's spec, an isolated datasheet implementation, and
arithmetic. Three sources, no shared author. Agreement between all three
is evidence; disagreement between any two is informative about which.

## 3. The routine

Hand-assembled by this session (so it carries no third-party licence),
origin `$C000`, 20 bytes. The canonical 6502 shift-and-add multiply,
which reuses the multiplier's own zero-page cell as the result's low
byte:

```
$C000  A9 00     LDA #$00      ; A = running high byte
$C002  A2 08     LDX #$08      ; 8 bits
$C004  46 F0     LSR $F0       ; shift multiplier, LSB -> C
$C006  90 03     BCC $C00B     ; loop head
$C008  18        CLC
$C009  65 F1     ADC $F1       ; add multiplicand
$C00B  6A        ROR A         ; rotate result right...
$C00C  66 F0     ROR $F0       ; ...into the low byte
$C00E  CA        DEX
$C00F  D0 F5     BNE $C006
$C011  85 F2     STA $F2       ; high byte out
$C013  EA        NOP           ; halt marker
```

Bytes: `A9 00 A2 08 46 F0 90 03 18 65 F1 6A 66 F0 CA D0 F5 85 F2 EA`

Branch offsets verified by hand: BCC at `$C006` is 2 bytes, next `$C008`,
target `$C00B` ⇒ `+3`. BNE at `$C00F` is 2 bytes, next `$C011`, target
`$C006` ⇒ `-11` = `0xF5`.

Why this routine and not something simpler: it runs ~50 instructions
across 8 loop iterations, and it exercises the parts of `r2conc` most
likely to be subtly wrong — carry propagation across `LSR`/`ADC`/`ROR`,
a conditional branch in each direction, a backward branch, a
read-modify-write to memory, and a loop counter. A straight-line
sequence would prove far less.

## 4. What the probe measures, beyond pass/fail

Two architectural facts are *measured and reported*, not assumed:

- **The SLEIGH register-space extent.** §7.8's binding says the CPU state
  lives in a V3 facet register (12 bytes). The 6502's *semantic* register
  file is 7 bytes (A, X, Y, SP, P, PC:2), which fits with room to spare —
  but SLEIGH's register-space *layout* may be sparse (Ghidra's 6502 spec
  models each status flag as its own register). If the layout extent
  exceeds 12 bytes, the honest statement is that the facet binding is a
  **projection** of the register file, not an identity with SLEIGH's
  layout. Measured value recorded in §6.
- **Which `SpaceId` 6502 memory actually lifts to.** See §5 — this
  corrected a claim already committed to two repositories.

## 5. A correction this probe forced (recorded before it can dilute)

`r2conc`'s own module doc and `r2il-machine-semantic-contract-v1.md`
§7.8 both assert that the 6502 lift mints `Custom` spaces because the
space-alias map is case-sensitive (6502 names its space `RAM`, the map
keys are lowercase).

**The claim is true of one code path and false of the other, and the two
were conflated.**

- TRUE for `LiftContext` / `ArchSpec` (`r2sleigh-lift/src/context.rs`):
  the alias `HashMap<String, SpaceId>` is seeded with lowercase keys only
  and looked up by exact match, so `"RAM"` misses and a fresh
  `SpaceId::Custom(n)` is minted. This is the *metadata* path.
- FALSE for the `Vec<R2ILOp>` stream (`r2sleigh-lift/src/disasm.rs`,
  `translate_space`): that function dispatches on libsla's
  `AddressSpaceType`, never on the space's name. The 6502 declares
  `RAM type=ram_space`, which libsla reports as `Processor`, which maps
  to `SpaceId::Ram`. Case never enters the decision.

So a program lifted for execution sees `SpaceId::Ram`, exactly as one
would want, and `r2conc`'s `Custom(n)` support is *not* required for
6502 memory (it remains correct and useful machinery, just not for this
reason). Both sites are corrected in the same arc as this probe.

The generalizable lesson, which is the reason this section exists: **a
finding about a data structure is not automatically a finding about every
path that touches it.** The alias map is real, the bug in it is real, and
the conclusion drawn about the op stream did not follow.

## 6. Measured facts

Each line names what produced it. Nothing here was predicted.

| fact | value | source |
|---|---|---|
| 6502 memory's `SpaceId` | **`Ram`** (not `Custom(n)`) | `LSR $F0` lifts to `Copy { src: [0xf0]:1 }`; `[..]` is `format_varnode`'s `Ram` rendering |
| SLEIGH register-space extent | **55 bytes** (`0x37`) | `6502.slaspec`: `A/X/Y/P` at `0x00-0x03`, `PC/SP` at `0x20-0x23`, and **each status flag its own byte register** at `0x30-0x36` |
| 6502 semantic register file | **7 bytes** (A, X, Y, P, PC:2, SP — SP read as 1 byte on hardware) | same |
| branch target varnode | space `Ram`, **size 2**, offset = the true absolute address | `BCC` at `$C006` → `CBranch { target: [0xc00b]:2 }`; no `target_width_mismatch` |
| p-code-relative branches in this routine | **none** | every `CBranch` target is `Ram`, not `Const` |
| unique-space high-water mark | `0x5580` | `ROR` lifts to `tmp:0x5500`/`tmp:0x5580` |
| `NOP` p-code op count | **0 ops** | `$C013` lifts to `P-code (0 ops)` — the sequencer must not hang on an empty instruction |
| Ghidra `ADC` carry-out | `IntCarry(A, M)` — **ignores carry-in** | `$C009` op 2 |
| Ghidra `ADC` overflow | `Copy { dst: V, src: C }` — **V := C** | `$C009` op 7 |

**Consequence for §7.8's binding, stated honestly:** the 6502's *semantic*
register file (7 bytes) fits a 12-byte V3 facet register with room to
spare, but SLEIGH's *layout* spans 55 bytes because Ghidra models each
status flag as its own byte register. So the facet binding is a
**projection** of the register file, never a byte-identity with SLEIGH's
layout. Pinned two-sided by
`the_sleigh_register_space_is_sparse_so_the_facet_binding_is_a_projection`
— which fails if SLEIGH ever packs the 6502 into 12 bytes, forcing a
re-measure rather than letting a stale note stand.

## 6a. A defect the probe found, and a prediction it corrected

Ghidra's 6502 `ADC` is wrong twice: the carry-out ignores the incoming
carry, and the signed-overflow flag is assigned the unsigned carry flag.

The plan, before the probe ran, predicted the multiply routine was immune:
it `CLC`s before every `ADC` (so carry-in is always 0 and the two carry
formulas coincide) and never *reads* `V`. **Both premises are true and the
conclusion was still wrong** — `ADC` also *writes* `V`. Measured on
`255 × 255`: both sides compute `0xFE01` (= 65025) and agree on every
other field; Ghidra leaves `V=1`, a real 6502 leaves `V=0`.

So the probe found a real spec defect in a real routine, not merely in a
synthetic probe of one. `V` is excluded from the headline comparison and
the exclusion is itself tested two-sided (it must diverge on some pairs
and agree on others), so it cannot degrade into "ignore that field". The
product is compared in full on every pair regardless.

The generalizable lesson: **"the buggy path is unreachable" is a claim
about the whole architectural state, not just about the computed result.**
A wrong value written to a field nobody reads is still a divergence.

## 7. Falsifiers — all seven verified red-then-green

Full log in the test file's module doc.

| # | disable | observed red |
|---|---|---|
| D1 | sequencer ignores `Control::Jump` | 3 red — the loop is never taken |
| D2 | drop the high byte | 3 red — the `> 255` pairs fail; `7 × 13` alone would NOT have caught it |
| D3 | assert `Custom(1)` instead of `Ram` | 1 red, naming the real space |
| D4 | resolve the register file by the wrong name (A/X swapped) | 3 red — the by-name `ArchSpec` lookup is load-bearing |
| D5 | oracle's `ADC` recomputed the way Ghidra does | 3 red **including the published ADC truth table** — the oracle's independence is what makes the divergence tests mean anything |
| D6 | remove the `CLC` from the routine | 4 red — carry-in stops being 0, immunity claim correctly fails |
| D7 | `without_v()` stops excluding `V` | 1 red — the exclusion does real work rather than hiding nothing |

## 7a. Changes the probe forced elsewhere

- **`r2conc` now refuses a `Const`-space branch target.** In p-code that
  is a branch relative to the p-code index *within one machine
  instruction*, not an address; `r2conc`'s `Control` vocabulary addresses
  machine instructions, so it refuses rather than misreading a
  displacement as an address. A not-taken conditional still falls through,
  because falling through is what it means. (This routine emits none —
  the gap was latent, and is now loud instead of silently wrong.)
- **The Custom-space claim is corrected** in `r2conc`'s docs and in §5
  above.

## 8. Deliberate scope limits

- **The Klaus Dormann functional test suite is not used**, though it is
  present on this machine under `tests/corpus/cache/` (fetched, not
  vendored, GPL-3.0). It requires the full instruction set and a complete
  memory map, far beyond `r2conc`'s concrete subset. Using it is a
  legitimate later deepening; claiming it now would be false.
- **No third-party emulator is transcribed.** Two independent 6502 cores
  exist on this machine (`rust64`, MIT; `frodo4`, GPL-2.0). Neither was
  read for opcode semantics; the oracle was written from published
  behaviour instead. The GPL one is fenced by the consuming repo's own
  rules and stays fenced.
- **The fixture is committed bytes of this session's own authorship**, so
  the probe is hermetic — no fetch, no sibling checkout, no skip-if-absent
  path. This is deliberate: a test whose fixture may be absent
  skips-and-passes, which hides failure exactly where a fresh CI would
  otherwise catch it.
