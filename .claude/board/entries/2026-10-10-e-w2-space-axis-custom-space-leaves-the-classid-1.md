## 2026-10-10 — E-W2-SPACE-AXIS-CUSTOM-SPACE-LEAVES-THE-CLASSID-1 — W2's space axis is decided: fixed spaces 0–3 stay in the classid, custom spaces move to the payload or edge

**Status:** DECISION + MEASURED. Closes plan `r2il-machine-semantic-contract-v1.md`
§8.2 Q2 and Q3; W2 may resume on the space axis under this carving.
**Confidence:** High for the measurement, which is exhaustive over the real 6502
spec. The decision is a scope choice and is recorded as DECISION / SCOPE /
BASIS / REVISIT WHEN.

### Measured: the 6502's second custom space was the alias map

`r2sleigh_lift::build_arch_spec(SLA_6502, PSPEC_6502, "6502")`, every address
space by name and id (a scratch binary over `r2sleigh-lift` and
`sleigh-config` with feature `6502`):

| space | r2sleigh `master` `c20cd1d` | r2sleigh #16 |
|---|---|---|
| `RAM` | `Custom(1)` | `Ram` |
| `OTHER` | `Custom(0)` | `Custom(0)` |
| `const`, `unique`, `register` | fixed | fixed |

The lifted ops already carried `Ram` for 6502 memory (r2conc's 2026-08-26
correction; `live_regfile` sabotage row D3). Only the `ArchSpec` metadata
disagreed: `LiftContext` matched space names case-sensitively against
lowercase aliases. r2sleigh #16 matches them case-insensitively, and
`live_regfile` with `--features probe-6502` stays 18 of 18.

So the measured population of custom spaces is 0 in the W0 x86 census of
94,536 rows, and 1 on the 6502, `OTHER`.

### DECISION (2026-10-10)

Of the three options W0 named, the custom space moves **out of the classid**,
into the payload or edge, keyed by the architecture's identity and version.
The classid's space discriminant carries only the fixed spaces 0–3, which are
architecture-invariant and self-describing (`facet.rs` already says so).

- **SCOPE:** the space axis of the `0xC4` mint and of the R2IL varnode facet.
  Container concepts were never blocked.
- **BASIS:**
  - Custom spaces are rare in the measured population above. Naming the
    architecture in every address, option 1, spends classid bits on every row
    for that rare case.
  - The raw SLEIGH space id, option 2, is itself per-architecture: it comes
    from a registration counter (`r2sleigh-lift` `context.rs`,
    `next_custom_space`) or a table index (`disasm.rs`). Two architectures
    still collide on it.
  - Resolving a custom space through a versioned architecture registry
    matches the measured cost of version-keyed validation, 0.6–0.7 µs against
    802–818 µs for a content fingerprint (D-SCF-CARE-PAIR-0, plan D-RPF-3).
- **REVISIT WHEN:** a corpus shows custom spaces on a hot path, or an
  architecture needs to route by custom space at prefix level.

### Not built

The payload or edge carving itself. ruff `ruff_r2il` `facet.rs` assigns "the
real carving" to its PR 3, and its Known-tension note anticipates exactly this.
Until PR 3 lands, `facet.rs` keeps its provisional, never-persisted encoding.
