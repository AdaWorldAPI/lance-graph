# concept-labels-hotplug-v1 — labels of a concept, pulled through the plug

**Status:** ACTIVE (D-LBL-0..5). DECISION by the operator, 2026-10-11. W0 (this record) lands first; W1 onward are code.

## The decisions (operator, 2026-10-11, verbatim)

> "Labels live in ogar-vocab pulled via hotplug.rs from the consumers. No need to keep labels for hundreds of consumers hot when it's separation of concerns that the plug and play pattern enables the consumers to activate the label distribution via ogar-vocab"

> "lean (a) now, with (b) when a real volume problem shows up. if you can make multilingual no breaking plan option per consumer in the board. Consumers mint the specs and Concept x language lazylock would make it cheap"

> Customer binaries activate at "build time". The German preferred label of `0x0103` is "Zeiterfassung".

What it means, in rules:

1. **Labels are ontology data in `ogar-vocab`.** OGAR is the only writer. No consumer is ever listed in the label data, so it is not a lockstep table.
2. **A consumer gets labels only for the concepts it plugs**, through `hotplug.rs`, the same way it gets readings. Nothing keeps every label hot for every consumer.
3. **Now:** an activation carries every language of each plugged concept. `HotPlug` does not change.
4. **Later (when a volume problem is measured):** a consumer selects languages through a spec it mints itself. This must not break any existing consumer (D-LBL-4).
5. **Customer binaries activate at build time.** The build emits a static activation, labels included. No OGAR link at runtime (BBB barrier).
6. **`0x0103`:** German preferred "Zeiterfassung", German alternate "Stundenzettel".

## Shape

**Rows (`ogar-vocab`).** One row per label: `(concept u16, BCP-47 tag, role, text)`, role ∈ {preferred, alternate, hidden}. Invariants, each a test:

- at most one preferred row per (concept, tag);
- the three roles are disjoint per (concept, tag);
- every concept exists in `CODEBOOK`;
- an absent language has no row. A lookup never copies another language into it.

**Index.** `static LABELS: LazyLock<HashMap<(u16, LangTag), LabelSet>>` is built once from the rows: concept × language. Filling an activation costs one lookup per plugged concept and language, independent of how many labels exist in total.

**Activation (contract, additive).** `Activation` is built only through `Activation::new(..)`. No sibling repo constructs it, and `with_register_laws(..)` already shows the additive builder pattern. Labels follow that pattern:

- `Activation::with_labels(..)` builder; `OgarAuthority::activate` fills it for `plug.classids`;
- `Activation::label(concept, &["de-AT", "de", "en"]) -> Option<&str>` returns the preferred label;
- `Activation::alt_labels(concept, lang)` returns the alternates;
- the caller passes its own language priority. The lookup walks it and returns `None` when no requested language exists, never another language.

**App-specific and private labels** stay in the consumer's own ClassView or overlay, never in `ogar-vocab`. A private repository's wording never reaches a public crate.

## D-LBL-4 — per-consumer languages, non-breaking

The consumer mints its spec beside its `HOT_PLUG`, in its own crate:

```rust
pub const LABEL_SPEC: LabelSpec = LabelSpec {
    consumer: "<crate>",          // must equal HOT_PLUG.consumer
    languages: &["de", "en"],     // BCP-47 tags
    roles: LabelRoles::PREFERRED_AND_ALTERNATE,
};
```

The authority gains a **default** trait method, so every existing authority keeps compiling:

```rust
fn activate_with_labels(&self, plug: &HotPlug, spec: &LabelSpec)
    -> Result<Activation, ActivationDrift>
{
    // default: activate(plug), then keep only spec.languages × spec.roles
}
```

- `activate(&HOT_PLUG)` keeps returning every language, so no existing caller changes behaviour.
- Filtering reads the concept × language index directly: one lookup per plugged concept and requested language.
- A tag with no rows is not an error; it yields no labels for that tag.
- `spec.consumer` must equal `plug.consumer`.
- Nothing in `ogar-vocab` lists consumers or specs. The spec lives in the consumer.

## Build-time activation (D-LBL-3)

- A customer binary's build step calls the authority and emits the activation as generated Rust constants: concepts, capabilities, readings and labels.
- The binary links only `lance-graph-contract`. Spine and test binaries may still activate at runtime.
- The generated constants also record which label set shipped.

## Waves

- [ ] **W0 — D-LBL-0: record.** This plan, `INTEGRATION_PLANS.md`, `STATUS_BOARD.md`, and the corrections below.
- [ ] **W1 — D-LBL-1: label rows + index in `ogar-vocab`** (OGAR PR). Seed `0x0103`: de preferred "Zeiterfassung", de alternate "Stundenzettel"; en preferred from the canonical name. Then the other German words the 2026-10-11 census found (`0x0202`, `0x0203`, `0x0204`, `0x0205`). Invariant tests above.
- [ ] **W2 — D-LBL-2: `Activation::with_labels` + `label()`;** `OgarAuthority::activate` fills labels for plugged concepts.
- [ ] **W3 — D-LBL-3: build-time activation** for customer binaries.
- [ ] **W4 — D-LBL-4: `LabelSpec` + default `activate_with_labels`,** when a volume problem is measured.
- [ ] **W5 — D-LBL-5: the TTL reader keeps language tags** (`ogar-from-schema` keeps only the first `rdfs:label` and drops the tag today). This feeds W1 from OGIT once OGIT carries tagged labels.

## Falsifiers

- **It fires:** a plug of `[0x0103]` gives `label(0x0103, &["de"]) == Some("Zeiterfassung")`, and the German alternates contain "Stundenzettel".
- **It stays silent:**
  - an unplugged concept's labels (`0x0202`) are absent from that activation;
  - `label(c, &["fr"])` is `None` when no French row exists, never the English one.
- **Invariants:** two preferred rows for one (concept, tag) turn the test red; a row naming a concept absent from `CODEBOOK` turns it red.
- **D-LBL-4:** a spec of `["de"]` yields no English rows; `activate()` without a spec returns exactly what it returned before the change.

## Corrections carried by this record (operator, 2026-10-11)

> "'WoA's port row' suggests that deprecated lockstep instead of hotplug.rs is used"

- **The mapping lives in the consumer's plug, not in OGAR's port tables.** WoA's `HotPlug` declares `0x0103` for its `TimeSheet` (a close match, owned by WoA). `WOA_ALIASES` and the other `PortSpec` alias tables are lockstep tables: deprecated, kept. This supersedes, in `entries/2026-10-11-coresearch-labels-address-lifetime.md`, shape piece 2 ("`PortSpec` is the mapping set") and the X14 / X17 / X18 verdicts "ADOPT in `PortSpec` / OGAR".
- **Capabilities.** The concept declares its capabilities; each consumer's plug declares which it covers. No list of consumers exists. The expected-executors check in OGAR's `resolve_hotplug` (`UnexpectedConsumer`) is a lockstep residue to deprecate. This supersedes the executor-list framing of `ogar-p18-capability-table-0x0103` in `entries/2026-10-11-synergy-ledger-classid-hotplug.md`.

## Still open (operator)

- The three colliding u16s (`v3-mandatory-hotplug-reading-v1` O3).
- A plug that asks only for a reading on a concept that has capability declarations.
- D-HPR-6b (an unknown classid errors).
