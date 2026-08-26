## 2026-08-21 — E-THE-ATTENTION-ATOM-WAS-ALREADY-SHIPPED-WHAT-WAS-MISSING-WAS-A-COMPOSITION-THAT-IS-NOT-OR-1 — D-ACR-1's basis is a reuse, and the only real gap was that every set operation in the crate is a bitset union

**Status:** FINDING (D-ACR-1, implemented; 14 new tests green, 1194 contract
tests green, clippy clean). **Confidence:** High — the atom's existence is a
read of `facet.rs`; the container's absence is a stated grep.

D-ACR-1 was written as *"the one missing primitive"*. Half of that is right,
and the half that is wrong is the more useful half.

**The atom already existed, exactly.** `contract::facet::FacetCascade` is
`facet_classid(4) | 6×(8:8) = 16 B` with `CascadeShape::G6D2` — literally the
`6 × 2 × u8` shape D-ACR-1 needed, with `index`/`group_of`/`level_of`/`shift`
shipped, `tier_bytes()` ordered coarse-first (`[t0.hi, t0.lo, …]`), and two
existing readings to copy the pattern from (`awareness_facet::SpoFacet`,
`tekamolo_facet::TekamoloFacet(pub FacetCascade)`). **Zero new bytes were
needed and none were added.** Proposing a new 12-byte type here would have been
the "type that already exists" rediscovery tax `CLAUDE.md` §Consult names.

**What was genuinely absent is narrower and sharper: a composition that is not
a bitset union.** Measured — every `union`/`intersect` in the contract crate is
a bit operation over *field positions*: `FieldMask` (u64, `MAX_FIELDS = 64`),
`WideFieldMask` (u8 positions, capped at 256 with a loud
`UniverseExceedsSocCap`), `StepMask` (u64), `rbac`. **Not one of them composes
addresses.** And a bit-OR of two addresses is not a coarser focus — it is a
*third address neither side ever visited*. That is the defect the container had
to avoid, and it is now a test (`composition_is_not_a_blind_or`: OR-ing
`0b01` and `0b10` yields `0b11`, and the union must NOT contain it).

**The composition that IS right was already in the canon, one type over.**
`NiblePath::is_ancestor_of` is coarse→fine prefix containment with an
**explicit** depth field. Reusing that rule on the 12-unit ladder gives
`covers`, and reusing its explicit-depth discipline avoids a real trap:
inferring the wildcard boundary from zero bytes would collide with the
zero-fallback ladder, where `0` is a legitimate *dormant tier*, not a
terminator. So `depth` lives OUTSIDE the 12 bytes and the wire shape stays
exactly `6 × 2 × u8`.

**This is also the answer to ">256 rows".** The `u8` per level bounds *axis
resolution* (256 centroids), never population: one shallow facet is a wildcard
over an unbounded subtree. Measured in a test — a depth-2 focus covers 65,536
addresses across only the two units varied, and a container holds 1000 distinct
entries where `FieldMask` would stop at 64 and `WideFieldMask` at 256. **The
container does not index rows at all**, which is why neither cap transfers.

**Content-blindness had to be defended twice in one deliverable.** A first
draft named the axes `Heel/Hip/Twig/Leaf/Family/Identity` — baking the cascade
reading into the low-level type, which is exactly what `FacetCascade`'s own
contract forbids (*"only the CONSUMER projects meaning onto the bytes"*). The
operator caught it while raising a second candidate reading (six **ontology
scopes**: disease/anatomy/process/substance/evidence/context). Both readings
now live only in a test, over byte-identical input, proving the projection is
free — and `FocusAxis` is `Axis0..Axis5`, a position. The general lesson is the
one this arc keeps paying for: **the moment a second plausible reading appears,
any name in the substrate is a premature commitment to the first.** Four
homonym collisions (witness / nibble / hydration / attention-mask) were the
expensive form of this; naming the axes would have been the fifth, minted by
us rather than inherited.

**Deferred, named rather than silently absent:** `RowFocusMask::difference`
keeps a partially-overlapped entry whole. Subtracting a subtree from a prefix
requires enumerating siblings — inventing addresses the focus never visited —
so splitting waits for a real consumer that needs it. `D-ACR-1`'s scope line
also asks for composability with `WideFieldMask`; per D-ACR-0's measurement
that is a **cardinality mismatch** (positions ≤ 256 vs an unbounded population)
and remains §6 **Y2**'s parked basis collision — this deliverable states its
own basis and does not cross into the other.

