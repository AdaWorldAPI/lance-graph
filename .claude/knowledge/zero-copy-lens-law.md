# Zero copy is a law without escape hatches — the lens IS the floor

> READ BY: zero-copy-warden, lens-migration-engineer, v3-envelope-auditor,
> truth-architect, and any session about to write a `from_*_bytes`, a window
> parameter, a facet reader, or any struct that holds what a lane holds.
>
> Operator rulings, 2026-07-28, in the order they landed:
> 1. *"Zero copy is a law without escape hatches."*
> 2. *"If you use a lens over existing at the cost of a cast there's no faster."*
> 3. *"And even the array is a Classview projection."*
> 4. *"The cast is so geniously efficient that you tried to unsee it."*
>
> Companion: `.claude/knowledge/le-contract-is-the-tenant.md` (the LE contract
> and the facet-serialization failure signature). This doc is its dynamic half:
> that one governs how bytes are *laid out*, this one governs how they are *read*.

## The law in three lines

- **No escape hatches.** There is no size below which a copy is acceptable.
- **The lens is the performance floor.** A cast emits no instruction, so a
  materialization is strictly worse on **both** axes — correctness and speed.
  **No cost argument can ever favour a copy.**
- **The array is already a projection.** There is no neutral byte layer
  underneath. "Materializing the registers" does not copy data; it **stores a
  second projection beside the first** — the facet-serialization anti-pattern,
  one layer down.

## Why the third line closes the loophole

Before it, a plausible defence exists: *"I'm not duplicating storage, I'm just
building a convenient array of the values."* Ruling 3 removes it. If the array
IS a ClassView projection, then the "convenient array" is a second stored
reading of bytes that already have one. You never had a raw layer to copy — so
the copy is not an optimization decision, it is an ontological duplication.

## Why this is hard to see (the mechanism, not an excuse)

**A free operation is invisible to cost-based reasoning.** When you ask "what
does this design *do*," you assemble the answer from operations that cost
something. A reborrow costs nothing, emits no instruction, has nothing to point
at — so it does not register as the answer, it registers as *the absence of
one*. "Just cast it" reads like a gap in the design rather than the design.

This explains what a failure catalogue alone cannot: **why facet-serialization
keeps happening.** Someone holding the lane writes a struct and copies into it,
because the copy *feels* like the implementation and the cast feels like
skipping it. The doctrine is not fighting a bad argument; it is fighting the
intuition that work must be visible to be real. It is also why the anti-pattern
survives review: a reviewer looking for "where's the implementation" finds the
copy and is satisfied.

**The review question is therefore not "is this zero-copy?" but:**

> **"Where did you think the work happens?"**

When the design is right the honest answer is *nowhere* — and that answer has to
stop sounding like a confession.

## The cost-argument trap

Every one of these is itself the finding, regardless of the numbers:

- "it's only N bytes" · "the copy is cheap / free / optimized away"
- "it's a `Copy` type, so by-value is fine"
- "12 B inline beats a 16 B pointer plus indirection"
- "this is a transient window, not storage"

Each compares **two materializations**. The lens is not the worse option being
weighed — it is outside the option set the author was choosing from. Naming that
is the correction; arguing the numbers is not.

## The canonical measured instance (2026-07-28)

`witness_fabric::{resolve_chain, standing_wave_grounded}` took
`window: &[(usize, CausalWitnessFacet)]` — a gathered, contiguous slice of
pairs.

Geometry: `NodeRow` is `#[repr(C, align(64))]`, `NODE_ROW_STRIDE = 512`,
`value: [u8; 480]` at row `[32..512)`; `ValueTenant::CausalWitness = 14` at
`row_offset 204`, 16 B = `classid(4) + register(12)`; so the **register lives at
value-slab `[176,188)`** and the registers across a row slice are **already an
array** — stride 512. `CausalWitnessFacet` is `#[repr(transparent)]` and
`from_register_ref(&[u8;12]) -> &Self` is a free cast.

The signature forced every caller to walk rows and pull 12 bytes out of each
512-byte stride into packed storage: **~768 KB copied per resolve over a 64k-row
sweep**, to produce what a cast already had.

The defence offered was *"the facet is a 12-byte `Copy` type, so inline values
beat a slice of references (12 B vs 16 B + indirection)."* That compared two
materializations. Overruled.

**The strongest evidence for the blindness:** the same author shipped
`from_register_ref` one task earlier under the banner *"make the cast real"* —
and reached for a `Vec` in the very next task. Possession of the mechanism did
not touch the reflex. That is why this law is carried by a mechanical warden
rather than a paragraph of doctrine.

## The lens shape

```rust
// The view: borrows the source, owns nothing.
pub struct WitnessLens<'a> { rows: &'a [NodeRow] }

impl<'a> WitnessLens<'a> {
    // The whole mechanism. No copy anywhere.
    pub fn at(&self, pos: usize) -> Option<&'a CausalWitnessFacet> { /* cast */ }
}
```

Three rules that make a lens correct rather than merely borrowed:

1. **Offsets are DERIVED, never literal.** Tie them to the tenant descriptor
   (`ValueTenant::X.value_offset()`) or pin them with a `const _` assert. An
   ungated literal offset is the same bug as a drifting reservation — see
   `le-contract-is-the-tenant.md` on "ordinal, not offset."
2. **Filter by PREDICATE, never by gather.** A visibility/version filter is
   `impl Fn(usize) -> bool`, not a `Vec` of the survivors. Filtering must cost a
   predicate call, never an allocation.
3. **Replacing a shipped resolver needs an EQUIVALENCE test** across several
   positions and budgets — lens result ≡ gathered result. Without it, a
   "refactor" is a rewrite hoping to be one.

## Consequences for new work

- **No parameter may be a gathered window.** Not `&[(usize, Facet)]`, not
  `Vec<Facet>`, not an iterator yielding owned copies out of a strided source.
  Pass the SOURCE plus the elected projection.
- **A `*_ref` twin is mandatory** wherever a `from_*_bytes` sits on a read path
  and the type is (or can be made) `#[repr(transparent)]`.
- **`project()`-style methods that return a new value are conveniences, not read
  paths.** The canonical read is the guarded single access (`elected(&mask,
  locus) -> Option<i8>`), which materializes nothing. Document the difference or
  the convenience quietly becomes the default.
- **A type whose only constructors are `#[cfg(test)]` is a shadow of storage.**
  Nothing real populates it, so it is a second projection with no producer —
  fix the producer or delete the type; do not grow it.

## Cross-refs

`le-contract-is-the-tenant.md` (layout half; two-places-to-decide,
three-the-compiler-enforces) · `.claude/v3/soa_layout/witness-nibble-lane.md`
(the A9 lane contract) · `.claude/v3/soa_layout/le-contract.md` §2 slot purity ·
`I-VSA-IDENTITIES` (identity pointers, not content) · ADR-022/023 (the Firewall:
no serialization in the hot path) · `.claude/agents/zero-copy-warden.md`
(detection) · `.claude/agents/lens-migration-engineer.md` (repair).
