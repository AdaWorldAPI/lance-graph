# D-V3M-0 — V3 mandatory; the reading comes from the plug; concepts are immutable addresses (2026-10-10)

**Status:** DECISION (operator, 2026-10-10). Plan: `.claude/plans/v3-mandatory-hotplug-reading-v1.md`.

> "Yes V3 mandatory, everything else in regards to capabilities hotplug.rs plug and play > ogar pattern making sure lockstep is deprecated"
>
> "Any concept is immutable. That's why Ontologies exist. Different languages are a label. The concept needs to be immutable adress"

- **V3 is mandatory:** no V1 mint, no silent V1 fallback.
- **One resolution path:** the reading comes from `HotPlug` → `OgarAuthority` → `Activation::resolve_for_context` plus the slab's `SlabDeclaration`. Lockstep tables are deprecated.
- **A concept is an immutable address.** Names in other languages ("Stundenzettel", "TimeSheet", "Zeiterfassung") are labels of it.
- **The classview fossils retire** (`0x1000` V3 marker, per-app render prefixes). Keys already stored with them stay readable.
- **Residue to remove:** `classid_read_mode` / `BUILTIN_READ_MODES` (`ISS-CLASSID-READ-MODE-IS-A-SECOND-RESOLUTION-PATH`). It selects V3 by the `0x1000` marker and falls back to V1 for an unknown classid. Callers: 4 contract modules, 4 lance-graph crates, q2 osint-bake/geo, MedCare cohorts, OGAR osm/ro/dismech/loco (plan table).
- **Waves W1–W6:**
  - W1: the contract takes the reading as a parameter.
  - W2: the canon domains' readings move into `concept_override`.
  - W3: consumers move, then `#[deprecated]`.
  - W4: stop minting at the fossil classviews.
  - W5: legacy reads go through `SlabDeclaration`.
  - W6: no V1 default.
- **Open (operator):** O1, the classview of a new canon mint once `0x1000` retires; O2, which WoA row "Stundenzettel" labels.
