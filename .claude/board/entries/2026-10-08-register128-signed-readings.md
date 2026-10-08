# 2026-10-08 — Register128 signed readings (32×i4, 16×i8) with a per-family law

**Status:** CURRENT-CONTRACT, TEST-PINNED. Branch
`ccr-b2e415d9-4jfvyk-register-readings`, commit `14831543`.

## What landed

The carrier is unchanged: the `Register0` / `Register1` rails (16 B each), no
classid in the payload, no layout change.

- **Carving, declared by the slab.** `SlabReading::RegisterI4x32 = 2` (32 × i4,
  dim `2k` = low nibble of byte `k`, the `atoms::I4x32` layout) and
  `SlabReading::RegisterI8x16 = 3` (16 × i8, dim `k` = byte `k`).
- **Law, declared at binding.** `RegisterLaw::{RelativeOffset, AxisPosition,
  Support}`, passed to `ResolvedReading::bind_signed_register(rails, law)` once
  per population. The returned `SignedRegisterLanes` carry concept, rails,
  carving and law.
- **Checked on every access.** `read_i4x32` / `read_i8x16` / `write_i4x32` /
  `write_i8x16` take the law the caller expects and refuse, in order, an absent
  rail, a different carving, a different law. An i4 outside `-8..=7` is refused,
  never saturated. A refused write leaves the row unchanged.
- **Separation from the unsigned reading.** `bind_register128` (4 × u32 words)
  is unchanged and refuses the signed carvings; `bind_signed_register` refuses
  `Register128`, `Facet96` and undeclared slabs.
- **EpistemicState5 has no law here.** It stays on CE64 bits 59..63.

## Inventory of the 24×i4 families on `main` (2026-10-08)

| family | on `main` | law it would bind |
|---|---|---|
| Temporal / anaphora | `CausalWitness` tenant 14, Facet96 read as G24N4 (24 i4 window pointers). EXPERIMENTAL. The only resident 24×i4 lane. | `RelativeOffset` |
| Evidence (Tarski) | Probe only: `probe_four_plane_causal_medium.rs` (G24N4 as support/falsifier). | `Support` |
| Taxonomy axes | Fixture only: `entries/2026-10-04-coresearch-evidence-stance-dependence.md` (Mammal: `terrestrial`, `placental`). | `AxisPosition` |
| Episodic basin | `EpisodicBasin` tenant 15: 32 B of references, not i4. | none (not a signed family) |
| 32×i4 carrier | `atoms::I4x32`; now also the i4 carving's layout. | — |
| 16×i8 | `ndarray` SIMD `I8x16` only. | — |

`CausalWitness` is NOT moved onto a register by this change: it is a Facet96
lane with its own classid, and re-homing it is a separate decision.

## Tests and disable runs

Hotplug: `signed_rails_are_granted_only_to_a_signed_carving`; tag test now
accepts 2 and 3. Register128: round trip of every i4 value in every dim with the
nibble layout pinned; i8 dim-k-is-byte-k; the same bytes differ under the two
carvings; law mismatch (all 3×3 pairs, both carvings) refused and writes
nothing; carving mismatch refused; out-of-range i4 in the last dim refused
before writing; writes stay inside their rail.

| disable | caught by |
|---|---|
| law check removed | `a_different_law_is_refused_and_writes_nothing` |
| carving check removed | `a_different_carving_is_refused` |
| rail check removed | `signed_writes_stay_inside_their_rail` |
| i4 range check removed | `an_out_of_range_i4_is_refused_before_writing` |
| binding accepts `Register128` as i4 | `signed_rails_are_granted_only_to_a_signed_carving` |
| binding swaps the carvings | same |
| binding skips the schema check | same |

## OPEN

- The law is declared by the binder. There is no OGAR table mapping a concept
  to its law yet, so a binder can still declare the wrong one; the contract
  only guarantees readers cannot silently read another law's values.
- Lane or reading per family is undecided until measured: 32×i4 vs 16×i8 on
  KJV anaphora and the mammal fixture. The rule "outside the local window →
  basin edge" is not changed by `RelativeOffset` under `I8x16`.
- No production writer or reader.
