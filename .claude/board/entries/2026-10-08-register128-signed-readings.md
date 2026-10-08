# 2026-10-08 — Register128 signed readings (32×i4, 16×i8) with a per-family law

**Status:** CURRENT-CONTRACT, TEST-PINNED. Branch
`ccr-b2e415d9-4jfvyk-register-readings`, commits `14831543` and `96304b88`
(law authority, after Codex P2 on #1410).

## What landed

The carrier is unchanged: the `Register0` / `Register1` rails (16 B each), no
classid in the payload, no layout change.

- **Carving and recorded law, declared by the slab.**
  `SlabReading::RegisterI4x32(law)` (32 × i4, dim `2k` = low nibble of byte
  `k`, the `atoms::I4x32` layout) and `SlabReading::RegisterI8x16(law)` (16 ×
  i8, dim `k` = byte `k`). Envelope tags 2..=7 (`to_tag` inverts `from_tag`).
  The law is the one the writer actually wrote with.
- **Concept law, declared by the authority.** `Activation::with_register_laws`
  / `register_law_for` (fail-closed: `NoRegisterLawFor`).
- **Binding checks agreement.** `ResolvedReading::bind_signed_register(&activation,
  rails)` takes no law; it refuses `RegisterLawMismatch` when the slab's law and
  the concept's differ. A writer and a reader of one population therefore hold
  the same law. The returned `SignedRegisterLanes` carry concept, rails, carving
  and law. `RegisterLaw::{RelativeOffset, AxisPosition, Support}`.
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

Hotplug: `signed_rails_need_a_signed_carving_and_an_agreeing_law`; tag test now
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
| binding accepts `Register128` as i4 | `signed_rails_need_a_signed_carving_and_an_agreeing_law` |
| binding swaps the carvings | same |
| binding skips the schema check | same |
| law mismatch check removed | `signed_rails_need_a_signed_carving_and_an_agreeing_law` |
| concept law ignored (slab law used) | same |
| `to_tag` shifts the I8x16 tags | `an_unsupported_physical_reading_fails_closed` |
| `from_tag` drops the law | same |

## OPEN

- No OGAR authority populates `with_register_laws` yet (`lance-graph-ogar`
  builds its `Activation` without laws), so in production every concept still
  refuses to bind signed registers. That is fail-closed, not a gap in the check.
- Lane or reading per family is undecided until measured: 32×i4 vs 16×i8 on
  KJV anaphora and the mammal fixture. The rule "outside the local window →
  basin edge" is not changed by `RelativeOffset` under `I8x16`.
- No production writer or reader.
