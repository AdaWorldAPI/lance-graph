## 2026-08-25 — E-W0-THE-SPACE-ORDINAL-IS-A-RANK-RELATIVE-TO-A-TABLE-THE-CLASSID-NEVER-NAMES-1 — W0 run: zero custom spaces in 94,536 rows, so the defect is LATENT; but the mechanism is worse than the conjecture and blocks the mint's space axis anyway

**Status:** FINDING — [MEASURED] (census over the full 4-binary corpus)
+ [FALSIFIED IN CODE] (a test against the shipped `CustomSpaceTable` API,
written, run green, reverted — no source change landed).
Closes plan `r2il-machine-semantic-contract-v1.md` W0 and open item O3.
**Confidence:** High. The census is exhaustive over the corpus; the
collision is demonstrated, not argued.

### The census — the defect is LATENT, not live

Every `at` field of all 94,536 ore rows decoded (16-byte `VarnodeFacet`,
lo-u16 of the classid word):

| binary | rows | Ram | Register | Unique | Const | **Custom (≥4)** |
|---|---|---|---|---|---|---|
| `stress_test` | 10,003 | 3,423 | 2,461 | 3,064 | 1,055 | **0** |
| `stress_test_opt` | 7,554 | 2,826 | 2,364 | 1,485 | 879 | **0** |
| `vuln_test` | 4,409 | 1,488 | 1,231 | 1,242 | 448 | **0** |
| `build-script-build` | 72,567 | 26,886 | 19,370 | 18,011 | 8,300 | **0** |

Union of discriminants across all four: `{0,1,2,3}`. **No custom space
occurs anywhere.** 0 malformed `at` fields.

So W0's own kill condition — *"same `n`, different meaning across
binaries"* — is **unanswerable by census**: the population is empty. That
is a real result (the x86 lift never produces one here), not a null probe,
and it is why the question had to be taken to the code.

### ⊘ My §3 conjecture named the WRONG mechanism, and the real one is worse

The plan's §3 said `Custom(n)` is *"a per-binary ordinal lifted out of the
program"*. **Wrong.** The table is built by `CustomSpaceTable::from_arch`
from `ArchSpec.spaces` — per ARCHITECTURE, not per binary. Within one arch
every binary shares one table, so the ordinal IS stable across the four
binaries, exactly as a reading should be.

The actual defect is one level up. `ordinal_of` is
`CUSTOM_ORDINAL_BASE + binary_search_position(raw)` — **the lo-u16 is the
RANK of the raw id inside whichever table interned it**, and the classid
does not name that table. Demonstrated against the shipped API:

```
arch_a = from_ids([10,20,30])      arch_b = from_ids([20,30,40])
  raw_of(4)      = 10                raw_of(4)      = 20   ← same ordinal, different space
  ordinal_of(20) = 5                 ordinal_of(20) = 4    ← same space, different ordinal
⇒ project(Custom(10), arch_a)  and  project(Custom(20), arch_b)
  are BYTE-IDENTICAL VarnodeFacets.
```

And the raw `n` feeding that rank is itself order-dependent upstream —
two independent sources, both counters: `r2sleigh-lift/src/context.rs:147`
(`Custom(next_custom_space); next_custom_space += 1` — registration order
during ArchSpec construction) and `disasm.rs:91` (`Custom(idx)` — the raw
SLEIGH address-space table index, as a fallback). So the lo-u16 is a rank
over a counter: **two stages of order-dependence, zero of identity.**

### Verdict — the space axis is BLOCKED from the `0xC4` mint as carved

Not for the conjectured reason. The ratified law says *CLASSID SELECTS THE
READING*; a reading must be self-describing at the address. This one is
self-describing only **relative to an ArchSpec that is nowhere in the 16
bytes**. Two facets from two arches collide on the same classid while
denoting different spaces — the classid stops being an address.

- **Fixed spaces (0–3) are fine and stay.** They are architecture-invariant
  by construction (`facet.rs:23-27` says so), self-describing, and are the
  only ones this corpus uses.
- **The custom axis must not be minted at `0xC4` in its current carving.**
  Options, none decided here: name the arch in the address; use the raw
  SLEIGH space id rather than a rank; or move the custom space out of the
  classid into the payload/edge. That is an OGAR mint question, not a
  ruff-local one.

**Prior art credited, not claimed:** `facet.rs:18-20` already carries a
`⚠ Known tension` doc comment recording that the space discriminant is a
shape ordinal in the lo half against OGAR's rule, and defers the carving
to PR 3. This entry supplies what that note lacked — the measurement, the
mechanism, and the collision.

### Why this matters NOW rather than later

The 6502/C64 arc is precisely where the latent case fires: a second
ArchSpec means a second table, and ordinal 4 would denote one space in the
x86 corpus and another in the 6502 corpus with nothing in the address to
tell them apart. The census says there is time to fix it; the mechanism
says the time is before the mint, not after.

