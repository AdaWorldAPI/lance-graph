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

## The scale anchor — why cost is not an argument in EITHER direction

> Operator, 2026-07-29: *"The whole Bible is 32k sentences, 16 MB."*
> …and: *"Grey / white matter makes it 2× 16 MB."*

Do the arithmetic, because it settles both rulings at once:

- **32k rows × `NODE_ROW_STRIDE` 512 B = 16 MB.** The entire corpus *is* the SoA.
  It is resident. There is no I/O, no paging, no streaming — every "window" is a
  slice of something already in memory.
- **The measured violation copied ~768 KB per resolve.** That is **4.8 % of the
  whole corpus, per resolve** — ~48 resolves copy the entire Bible. The copy is
  larger, relative to the substrate, than any access it could possibly save.
  There was never an access to save: the bytes were already there.
- **A whole tenant is 32k × 16 B = 512 KB.** Storage is free at this scale.

So cost cannot be the argument in **either** direction, and that symmetry is the
point:

| | the tempting cost argument | why it is empty at 16 MB |
|---|---|---|
| **for a copy** | "gathering avoids repeated strided access" | the data is resident; the gather costs 4.8 % of the corpus to save nothing |
| **for a tenant** | "storing it avoids recomputation" | 512 KB is free; cheapness never licensed a lane |
| **against a tenant** | "another lane costs memory" | 512 KB is free; cost never *blocked* a lane either |

**What is left when cost is removed from both sides is the only real criterion:
the rung.** A projection is refused because it is the same awareness stage, not
because it is expensive. An elevation is admitted because it is a higher stage,
not because it is cheap. That is why the exception below is stated in rungs and
never in bytes or nanoseconds.

(Consistency check: 32k sentences is ~1000× the `I-VSA-IDENTITIES` Test-1 bundle
capacity of N ≤ √d/4 ≈ 32 — which is exactly why VSA is demoted to its
within-compartment niche and the `temporal.rs` sorted stream is primary. See
`E-MARKOV-TEMPORAL-STREAM-1`.)

### Grey and white — the anatomy of the exception

The second 16 MB is not more corpus. In neuroanatomy grey matter is cell bodies
(content, where computation happens) and white matter is myelinated axons
(**connectivity** — what routes between them). The operator's 2× budget maps the
two halves of this law onto that split:

| | **grey matter** ≈ 16 MB | **white matter** ≈ 16 MB |
|---|---|---|
| holds | content / observation — the corpus as read in | connection — displacements, bindings, resolved routes |
| the A9 24×i4 register | — | **here**: loci are *offsets*, never magnitudes |
| a projection over it | free, and **never stored** (same rung) | — |
| an elevation | — | **eligible to be stored** (higher rung) |

Two things fall out, and both are already-shipped rules getting their anatomical
name rather than new policy:

1. **"Cross-tenant *pointers* are legitimate; cross-tenant *values* are not"**
   (`le-contract-is-the-tenant.md`) **is the grey/white fence.** A locus holding
   *where* something was grounded is white matter. A locus holding another
   tenant's *content* at lossy `i4` is white matter impersonating grey — which is
   exactly the failure that rule was written to forbid.
2. **The elevation budget equals the observation budget.** There is as much room
   for derived structure as for the corpus itself — so scarcity is never the
   reason to refuse a derivation tenant, and abundance is never the reason to
   grant one. Only the rung decides. 32 MB total is still trivially resident.

*Honest status:* the 2×16 MB budget and the grey/white framing are the operator's;
the row-level mapping of "which existing tenant is grey vs white" is not yet
enumerated per-lane. Treat the table as the governing frame, not as a census.

## Projection is not chasing (the indirection cost that does not exist)

> Operator ruling, 2026-07-29: *"24 i4 — it's just pointer projections, NOT
> pointer chasing."*

This closes the last place a cost argument could hide. The defence overruled
above priced "a 16 B pointer **plus an indirection**" — and the indirection was
imported intuition, not a cost this substrate pays.

|  | pointer **chasing** | pointer **projection** (what the loci do) |
|---|---|---|
| how the target is obtained | loaded, then dereferenced | **computed**: `target = cur + off` |
| when the address is known | only after the load retires | as soon as the offset nibble is in a register |
| address pattern | arbitrary, data-dependent | strided into a contiguous slab (`NODE_ROW_STRIDE`) |
| hop dependency | serial — each hop waits on the last | **independent** — hops pipeline |
| prefetcher | defeated | wins |

A locus is a **displacement, not an address**. Reading it costs a mask and a
sign-extend; resolving it costs an add. Nothing is followed. So the "indirection"
term in every inline-beats-reference argument is **zero in this substrate** —
which is why no arithmetic ever rescues the copy: the copy pays a real store to
avoid a cost that was never charged.

Corollary for review: if a design justifies materialization by invoking
indirection, cache misses, or chasing over a **strided register lane**, the
finding is not that the numbers are wrong — it is that the mechanism cited is
absent.

## The one apparent exception — and why it is not one

There is exactly one thing that licenses writing a value into a tenant, and it is
**not** "I did work to produce it."

> Operator rulings, 2026-07-29, closing a hole this doc was about to open:
> 1. *"If the 24× i4 is efficient over the standing wave, calling it compute to
>    store a copy is still wrong."*
> 2. *"We're talking about nanoseconds — you need to have a higher awareness
>    stage to be stored to justify a new tenant."*

An earlier draft framed the exception as *"entropy work licenses the write."*
**That framing is void as stated**, because every computation is entropy work if
you squint — it would have re-licensed precisely the copies this law forbids.
The two rulings fence it from opposite sides:

- **Efficiency is not elevation.** A cheap read is the *mechanism*, not a
  justification to store its output. The 24×i4 register reading efficiently over
  the standing wave is exactly what **disqualifies** storing what it read: the
  lens already is the answer. Relabelling the read as "compute" does not convert
  its output into new information. **The hatch cannot be entered from below.**
- **The bar is a rung, never a clock.** At nanoseconds there is nothing to save,
  so cost is never the argument. A lane is justified only by a **strictly higher
  awareness stage** than every input it was derived from.

**The test, and it is mechanical:**

> Name the rung of every input and the rung of the output.
> `output_rung == max(input_rungs)` → **projection. Never stored**, however much
> arithmetic it took.
> `output_rung > max(input_rungs)` → **elevation. Eligible for a tenant.**

(Rungs per `.claude/v3/knowledge/persona-vs-rung-ladder.md`: 0–1 observation,
2 = the 144 verb atoms, 3 = the 34 NARS tactic recipes, 4 = StyleFamily macros.)

The falsifier that keeps it honest: **recompute the stored value from the lens.**
If it comes back equal, you stored a projection — the store was a cache with a
correctness liability, not a memory.

> **⊘ REFINED 2026-07-29, same day — the operator's Gadamer probe broke the
> letter of this test.** As stated it is type-blind: `Locus::Quorum` is a
> deterministic function of the witnesses, always recomputes equal, and is
> legitimately stored — the letter would delete shipped precedent. Likewise any
> Horizontverschmelzung: deterministic given both horizons, yet a new thing that
> lives in neither. The repair subordinates the recompute test to the rung test:
>
> - reproducible by a **CAST** (a single lane read returns the same bytes) →
>   projection. Delete.
> - reproducible only by a **computation across multiple reads, yielding a value
>   of a different KIND** (a fact about the set, not a member of it) → elevation
>   candidate, judged by the rung. Deterministic recomputability is orthogonal.
>
> Gadamer's own formulation is the rule: *understanding is not reproduction* —
> the fusion product is different understanding, never a copy of either horizon.
> In interference terms: the diagonal terms |ψᵢ|² are already in the lanes
> (storing one is a second projection); only the cross-term 2Re(ψ₁*ψ₂) exists in
> no lane and is eligible. Cost stays excluded in both directions per the
> nanoseconds ruling — the rung decides, never the recompute price.

**Worked precedent, already shipped:** `Locus::Quorum` and `Locus::Contradiction`
are stored entropy work, and `CONTENT_LOCI` excludes them ("no self-reference").
They qualify **not** because reconciling observations is expensive, but because a
contradiction is a strictly higher epistemic object than the observations it
reconciles. That is the shape every future derivation tenant must match.

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

## Not every gathered slice is a window (the axis check, added 2026-07-29)

Found while migrating the `window:` family: **a gathered parameter tells you a
copy happened, not which lens replaces it.** Two different axes wear the same
`&[…Facet]` shape, and only one of them is a row slice.

| | **position axis** (`window:`) | **version axis** (`revisions:`) |
|---|---|---|
| what the elements are | different rows, one instant | the SAME logical row at successive versions |
| where they live | contiguous, `NODE_ROW_STRIDE`-strided | Lance versions — a temporal read |
| the lens | `WitnessLens::at(pos)` — a cast | **not yet written**; source is a version-range read (`QueryReference::at(v, rung)` + deinterlace, per `E-MARKOV-TEMPORAL-STREAM-1`) |

**`WitnessLens` does not generalize to the version axis.** Applying it there
would be the anti-pattern the card names — a "lens" whose constructor takes owned
data, i.e. the copy moved rather than removed. **Name the source before writing
the twin**; if you cannot name what the borrow borrows FROM, you are not ready to
migrate. Live inventory: 8 `revisions:` functions in `witness_fabric.rs`, no
external callers, unscoped.

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
