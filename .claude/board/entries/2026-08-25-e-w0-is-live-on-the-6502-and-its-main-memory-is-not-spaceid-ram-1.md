## 2026-08-25 — E-W0-IS-LIVE-ON-THE-6502-AND-ITS-MAIN-MEMORY-IS-NOT-SpaceId-Ram-1 — the arch census: 6502 mints TWO custom spaces, one of them its own RAM, because the alias map is case-sensitive

**Status:** FINDING — [MEASURED] (ArchSpec census over the real 6502 and
x86-64 SLEIGH specs, via `build_arch_spec`; instrument written, run,
deleted — no source change landed) + [CODE-READ, NOT RUN] for the
path-divergence half, graded separately below.
**Sharpens** `E-W0-THE-SPACE-ORDINAL-IS-A-RANK-RELATIVE-TO-A-TABLE-THE-CLASSID-NEVER-NAMES-1`
from **latent** to **live**. Does not change its verdict — it removes the
"there is time" half of it.

### What W0 said, and what changed under it

W0's census found **0 custom spaces in 94,536 rows across 4 x86 binaries**
and concluded: latent on x86, *"fires first on the 6502/C64 arc"*. That
arc landed the same day — `r2sleigh` PRs #2–#4 registered the 6502 family,
wired the CLI, and fetched the conformance corpus. So the prediction was
testable within hours, and was tested.

### The census — MEASURED

`build_arch_spec` over the real SLEIGH data for both architectures,
reporting the `SpaceId` each declared space maps to:

| | 6502 | x86-64 |
|---|---|---|
| `const` | Const | Const |
| `OTHER` | **CUSTOM(0)** | **CUSTOM(0)** |
| `unique` | Unique | Unique |
| RAM | **`RAM` → CUSTOM(1)** | `ram` → Ram |
| `register` | Register | Register |
| **custom count** | **2** | **1** |

x86 was never zero-custom at the ARCH level — `OTHER` was always
`Custom(0)`. W0's corpus census counted **rows**, and no row in those four
binaries ever referenced `OTHER`. Both numbers are correct; they answer
different questions, and this entry is the reason to keep them apart.

### The mechanism — a case-sensitive alias map

`context.rs:54-58` seeds the alias map with lowercase names only
(`"ram"`, `"register"`, `"unique"`, `"const"`, `"constant"`), and
`add_space_with_endianness` resolves via `space_map.get(name)` — an exact
`HashMap` hit. The 6502 SLEIGH spec names its main memory **`RAM`**,
uppercase. It misses, falls through to `SpaceId::Custom(next_custom_space++)`,
and lands as `Custom(1)`.

**The 6502's main memory is therefore not `SpaceId::Ram`.** Not a corner
case, not an exotic space — the space every load and store on that
architecture touches.

### The consequence for the address, stated conditionally

`CustomSpaceTable::from_arch` interns `{0, 1}` for the 6502, so
`ordinal_of(1) = CUSTOM_ORDINAL_BASE + 1 = 5`. **IF** a varnode reaches
`facet::project` carrying `Custom(1)`, its classid lo-u16 is **5**, while
the same concept on x86 is **0** (`SPACE_RAM`) — one kind of thing, two
readings, and nothing in the 16 bytes says which arch. That is exactly
W0's collision, now on real processor specifications rather than a
constructed pair of tables.

The "IF" is load-bearing and is the next thing to measure — see below.

### ⊘ A THIRD `Custom` source, missed by W0's entry

W0 named two (`context.rs:147`'s counter, `disasm.rs:91`'s raw index).
There is a third, and it is the worst: `disasm.rs:770-775` mints
`Custom(hash)` where `hash = name.bytes().fold(0u32, wrapping_add)` — a
**wrapping byte-sum of the space name**. Any two names with equal byte
sums collide (every anagram, trivially). W0's entry says *"two stages of
order-dependence, zero of identity"*; corrected, the chain is

```
space name → byte-sum hash (COLLIDING) → Custom(n) → sorted rank → classid lo-u16
```

— three stages, and the first loses identity before any ordering enters.

### [CODE-READ, NOT RUN] The two paths disagree about this very space

`translate_space` (`disasm.rs:757-765`) classifies by `space_type`, not by
name: a `Processor` space whose name does not contain `"register"` becomes
`SpaceId::Ram`. So for the 6502's `RAM` the ArchSpec path yields
`Custom(1)` while the lift-time path yields `Ram` — the same space, two
different `SpaceId`s depending on which code path produced the varnode.

**This half is read from source, not observed.** Whether a lifted 6502
memory access actually carries `Custom(1)` or `Ram` decides whether the
address consequence above fires at all, and it is the one measurement this
entry does not have. Naming it as unrun rather than implying it.

### What this does and does not change

- **Does not change the W0 verdict.** Fixed spaces 0–3 stay; the custom
  axis stays blocked from the `0xC4` mint as carved. The three repair
  options remain OGAR's to choose.
- **Does change the urgency.** W0 said *"the census says there is time; the
  mechanism says the time is before the mint."* The census now says the
  time is shorter than it looked: a second ArchSpec exists, in-tree, today.
- **Is not a lance-graph defect.** The case-sensitive alias map and the two
  divergent translation paths are `r2sleigh`'s, upstream of everything this
  plan owns. Recorded here because it is the falsifier for a lance-graph
  verdict, not because the fix belongs here.

**Next measurement (unrun, cheap):** lift one 6502 memory access through
the real path and read the `SpaceId` off the varnode. That settles the
conditional, and it is the difference between "the 6502 addresses wrongly"
and "the 6502 addresses correctly by accident, through the path that
disagrees with its own ArchSpec."

