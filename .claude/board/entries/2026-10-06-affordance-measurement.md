# 2026-10-06 — D-GSO-AFF-0: recipe eligibility is a measurement over the edge

## MEASURED

`crates/cognitive-shader-driver/examples/affordance_measurement_probe.rs`:
8 tests, 7 disable runs red.

- `CausalEdge64 × RecipeLaw → EligibleRecipes: u64` is two const-table
  lookups and one AND, with 0 allocations and no scheduler, task, capability
  object or recipe list. The tables are compiled at build time from per-recipe
  `requires` / `forbids` facts and Pearl planes.
- Bits 59..63 read as one EpistemicState5 code through the shipped accessors
  (`spare() << 2 | truth_raw()`). Ten codes are declared, ordered without regard
  to strength; 22 refuse. Code 17 = Indirect × IntermediateKnown × Causes.
- F3: S/P/O bytes, F/C, bits 43..45, the mantissa, plasticity and the witness
  slot were swept over their full ranges with no bit change. Pearl moves only
  the one recipe that reads it.
- F4/F5: the compiled tables match an independent re-derivation from the rules
  on every code × projection × recipe, and every refusal names its obligation.
- Rewriting only bits 59..60 with the shipped `with_topology` turns code 17
  into 18, which refuses.

## Inventory (before the probe)

- Bits 43..45 (`direction`): no reader outside `causal-edge`. Inside it,
  `*_pathological`, `concern_level`, `CausalNetwork::detect_simpsons_paradox`,
  `compose` (inherits the weight's triad) and the V3 lift read it as the
  per-plane pathology triad.
- Witness (bits 53..58): `MailboxSoA::apply_edges` (production) drops every
  edge whose `w_slot` differs from the mailbox's. It is compared by equality
  only; a reading that varies it per evidence class would drop edges there.
  Width 6 is assumed by `MailboxSoA::new` (`< 64`), `WitnessTable<64>`, the V3
  byte 8 packing and `with_routing`.
- Bits 59..63: shipped readers split them. `truth`/`topology` read 59..60;
  `ReasoningBand`/`spare` read 61..63; `band_reading` declares and projects the
  two separately; V3 stores them in different bytes (raw copies, so the joint
  code survives the lift). External readers: `dismech_counterfactual::EdgeRole`
  copies both; `entropy_topology_probe` reads 59..60.
- S/P/O: typed by name only (`s_idx`, "palette index"); `compose` indexes
  caller-supplied 256×256 tables as `left * 256 + right`, the same orientation
  as `PairAddress`.
- #1336/#1337: `Quad8([u8; 4])` holds blind bytes; `palette_pairs` wraps them
  in `PaletteState` for `PairAddress::new`, which is `(left << 8) | right`. No
  code connects Quad8 to CE64's S/P/O bytes. Both modules call the 8×8×8×8
  product "Cartesian".
- `StepMask(u64)` exists for template step positions, a different index space
  from recipe ordinals, so the probe uses a plain `u64`. `recipe_dispatch::ladder`
  returns `Vec<RecipeStep>`.

## OPEN

- A joint 5-bit reading needs its own `band_reading` declaration, and the
  half-field writers (`with_topology`, `with_truth`, `with_reasoning_band`)
  can produce undeclared codes.
- Witness cannot be read as an evidence class while `MailboxSoA` routes on it.
- The codebook, facts and recipes are probe pins; dynamic gates, preference
  and execution are not built.
