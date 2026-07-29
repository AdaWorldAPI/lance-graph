# The LE contract IS the tenant — a named struct in a sibling Vec is a facet serialization

> READ BY: v3-envelope-auditor, v3-mailbox-warden, truth-architect,
> integration-lead, layer-boundary-warden, and any session about to add a
> "facet", a "reading", a value tenant, or a `from_*_bytes` constructor to
> the 32-tenant SoA.
>
> Born 2026-07-28 from a real finding: `CausalWitnessFacet` (A9, 24×i4 loci)
> shipped as an owned `repr(Rust)` struct in a parallel `Vec`, fronted by 16
> named accessors and a 24-entry label table, citing a canon entry
> (`le-contract §3 L9 G24N4`) **that does not exist** — §3 is L1–L8, and grep
> for `G24N4|L9` across `.claude/v3/soa_layout/` returns zero. Operator ruling
> that closed it: *"Everything humanly possible in the 32 tenant doesn't have a
> register, they must have a little endian contract … any label lookup lives
> outside the SoA … the little endian contract is the actual mandatory shape."*

## The ruling, in one line

**A tenant is its little-endian byte contract.** Not its struct, not its
accessors, not its labels. The struct is a *reading*; the labels are
*ergonomics*; only the byte law is load-bearing — and it is mandatory, not
optional, for every one of the 32 tenants and for the envelope over them.

## Why this is not style

Four things collapse without a declared LE contract, and each is a claim the
workspace already makes elsewhere:

1. **"Zero-cost" becomes folklore.** A `repr(Rust)` struct has *no* guaranteed
   layout, so "the copy is free, LLVM optimizes it" is unfalsifiable. Zero cost
   is a **contract** only when the type is a transparent view over bytes whose
   meaning is target-independent: then the view compiles to a pointer reborrow,
   provably nothing. Without the byte law, "zero-cost" quietly means
   "target-dependent semantics that happened to work on this machine."
2. **The intelligence stops living in the SoA.** With the byte law declared,
   every consumer — standing-wave resolver, Lance columnar I/O, wasm, a GPU
   shader, another language — reads identical meaning from identical bytes with
   no interpretation layer. Readers become **pure projections**; nothing
   accumulates in the reader, everything accumulates in the tenant bytes. That
   is what makes the many-ClassView Doppelspalt real rather than metaphorical:
   many slits over one substrate, interference visible in the projections while
   the bytes never move.
3. **ClassView / WideFieldMask ergonomics stop working.** Fixed offsets are
   what make an election a *field mask over known bytes* — `mask ∩ register`,
   composable, fail-closed, an unelected slot **absent** from the projection
   rather than hidden by it. A mask over `repr(Rust)` is a mask over nothing.
   (Same machinery a2ui runs one layer up: `WideFieldMask ∩ role`, unauthorized
   fields never on the wire.)
4. **Morton-cascade addressing loses its ground.** O(1) tile-local projection
   presumes the bytes are where the contract says they are.

## The two-places rule (already shipped — do not reinvent)

Adding a tenant is **exactly two places** in
`crates/lance-graph-contract/src/canonical_node.rs`:

1. the `ValueTenant` enum variant — its doc-comment carries the lane's law;
2. its `VALUE_TENANTS` descriptor row (name_id / row_offset / length).

Everything else is *derived*. `ValueTenant::value_offset()` says so in its own
doc: *"Not a new property: a derived accessor over the already-locked,
compile-asserted carve."* If adding a tenant is touching more than two places,
the design is wrong, not the rule.

**Worked precedent: `ValueTenant::Tekamolo = 13`.** Copy its idiom wholesale —
16-byte content-blind V3 4+12 facet (`classid(4) + 12`), the shape named
(`G4D3`), zero-fallback stated (*"an all-zero facet reads as unaddressed …
never a wrong circumstance"*), appended additive / reserve-don't-reclaim, the
reading type `#[repr(transparent)]` with const size asserts, the populating
extractor living in an example, and — crucially — **honest catalogue status**:
`TekamoloFacet` self-declares *"EXPERIMENTAL reading — not yet in the
operator-locked §3 catalogue."* EXPERIMENTAL is fine. A **fabricated**
citation is not.

## What a tenant's LE contract must state

Three laws, minimum. Nibble-granular example (the A9 24×i4 register):

- **Byte/nibble law** — 12 bytes; slot `k` lives in byte `k/2`; even `k` = low
  nibble, odd `k` = high nibble (the nibble-granular extension of
  little-endian: low first).
- **Value law** — each nibble is two's-complement `i4`, sign-extended on read,
  range `−8..+7`; `0` = unbound (zero-fallback); unnamed slots reserved-zero
  (RESERVE-DON'T-RECLAIM).
- **Placement law** — the reading exists **only** as a lane under the
  envelope's LE contract: `ValueTenant` entry + `VALUE_TENANTS` descriptor +
  `verify_layout` coverage + `ENVELOPE_LAYOUT_VERSION` discipline. **No lane,
  no reading.**

Code that already *does* this without *saying* it has an implementation
accident, not a contract. The shipped nibble accessors masked `& 0x0F` / `>> 4`
correctly for months while the law went unwritten.

## The label fence

`Locus`, `LOCUS_LABELS` and every name table are **ClassView-side lookup,
outside the SoA**. Permitted direction: `enum → &'static str`, for display,
indexed by discriminant. **Forbidden: `str → slot`.** A name-keyed lookup is a
serialization-shaped access path even with no serde in sight; write the fence
into the doc so a future `from_label(&str)` is illegal by construction.

24 named labels are *nice for the reader*. They are not the shape.

## Failure signature — "facet serialization"

You are looking at one when **all** of these hold:

- the type is `repr(Rust)`, so no layout is guaranteed;
- its constructors **copy** bytes into an owned struct (`from_register`) rather
  than borrowing a view;
- instances live in a **parallel owned container** (`Vec<(version, Facet)>`)
  beside the real stream, instead of on a lane;
- it is fronted by named accessors and a label table;
- it is registered in **no** catalogue, routed by **no** read-mode table, and
  carried by **no** envelope lane;
- and — the tell that closes it — it **cites** a catalogue entry that does not
  exist.

The engine on top can be entirely correct while this is true. In the A9 case
the standing-wave resolution (`standing_wave_grounded` / `resolve_chain`) was
sound and well-tested; it just ran over a private `Vec` that nothing but
fixtures ever populated. **Correct engine, wrong chassis** — which is exactly
why "the tests pass" does not detect this class.

## The audit that finds it (four greps)

1. `grep -n "repr(" <facet>.rs` — is it `transparent`, or nothing?
2. constructor shape — does anything return a *borrowed view*, or only owned
   copies?
3. `grep -rn "<Facet>::new\|<Facet> {" --include="*.rs" crates/ | grep -v test`
   — are **all** construction sites fixtures? (If yes, nothing real feeds it.)
4. the citation check — take the doc's canon reference verbatim and grep the
   canon for it. A dangling citation is the highest-signal single check in this
   list, and it costs one command.

## Consequences (non-negotiable for new work)

- **No new facet reading without its LE contract text landing first.** Contract
  file leads the commit; code conforms to it, never the reverse.
- **A layout change on a shipped type requires a field-isolation matrix** —
  write each slot, assert every other slot unchanged (`I-LEGACY-API-FEATURE-GATED`).
- **Never re-carve a content-blind register to fit one reading.** Narrowing
  slots bakes a reading into the layout and destroys the surface elections
  project over. The shipped doctrine already says it: *"the rung level occupies
  ZERO slots: escalation is carried by the elected ClassView, which
  re-interprets which loci are live, never by a stored magnitude."* Elect with
  a mask; never delete slots.
- **Masks fail closed.** A missing or narrow mask elects **nothing**, never
  everything; a `full_for`-style helper is a render convenience and never an
  election fallback (a2ui charter C1.4).
- **Cross-tenant *pointers* are legitimate; cross-tenant *values* are not.** A
  locus holding "where my kausal cause was grounded" is provenance and belongs;
  a locus holding the cause's *content* at lossy `i4` would be a re-encode of
  another tenant's axis and does not.

## Cross-refs

`.claude/v3/soa_layout/le-contract.md` (§3 catalogue, §3a grace carvings, §5
code ground truth: *exactly 3 shapes + 1 lane* — `G6D2` / `G4D3` / `G3D4` +
hi/lo_chain 2×48) · `canonical_node.rs` (`ValueTenant`, `VALUE_TENANTS`,
`value_offset`) · `tekamolo_facet.rs` (the honest-experimental idiom) ·
`awareness_facet.rs` (`SpoFacet`, the A1 6×(8:8) reading) ·
`I-LEGACY-API-FEATURE-GATED` (CLAUDE.md) · `E-V3-FACET-4-PLUS-12` ·
`compilation-vs-runtime-substrate.md` · `assembler-vs-storage-substrate.md`.
