# 2026-10-30 — Co-research: Mistaber holdings as hydrated nodes, the KJV way

**Status:** EXPLORATION MAP. Nothing is adopted; the operator chooses. Follows D-MIST-NATIVE-0 (#1873).

## Question

How should Mistaber's worlds, propositions, rules and makor citations become addressed SoA nodes the way the KJV is hydrated in `deepnsm-v2`? In that path, `toc::spawn` mints the outline tree, `hydrate::hydrate` addresses each triple at its verse node, and `promote` writes basins at node keys. What does that buy?

The operator ruled "we don't use Strings": identities are addresses or nodes. The probe still interns names (`Interner(Vec<String>)`). The operator added `contract::revision` (`GadamerRevision`) mid-run.

## Premise audit (PREMISE-SPLIT)

The question names three concepts, and each has its own signature. They must not be answered as one.

| concept | what it is | where it goes |
|---|---|---|
| makor citation | a literary position, like a KJV verse | a TOC node |
| world | an authority scope; accessibility is a multi-parent DAG | an id carried as a value, plus edges |
| holding | a conclusion derived per (proposition, world) | a derived cell; its lineage points at rule and makor nodes |

A KJV triple is an observation made at a verse; a holding is a conclusion made at an authority. Filing the second under the first's mechanism would be wrong.

## Anchors

- No String as identity (operator; D-ARG-1..3, `STATUS_BOARD.md:428-435`).
- Canonical node `16|16|480`: position in the key, summaries in value lanes; V3 mint only.
- Scale is the next cascade level, never a wider field.
- Classids come from the OGAR codebook; worlds are values.
- Authority worlds are not Thought-Git branches.
- The stable-model fragment stays with Clingo.
- No licensing discussion in public artifacts.

## Crosswalk and verdicts

Sources (scouts): `toc.rs`, `hydrate.rs`, `promote.rs`, `hhtl.rs`, `causal_audit.rs`, `revision.rs` and the probe (VERIFIED-IN-CODE); the Sefaria index and shape API (fetched); CTS URN and PROV-O (SECTION-READ); arXiv 2202.10766, 2303.12773, 2608.21141, 2505.01944 and Dong et al. VLDB 2009 (SECTION-READ); Soufflé (SECTION-READ); Akoma Ntoso (ABSTRACT-ONLY).

| # | idea | evidence | verdict |
|---|---|---|---|
| X10 | **Names → label-codebook ids, unknown labels refused** | The probe's `Interner(Vec<String>)` and `WitnessEntry.source_url: Option<String>` both violate D-ARG-1. `promote::key_at` already mints via `mint_for` and refuses V1 keys (`promote.rs:99-115`). D-ARG-1..3 are Queued upstream. | **PROBE** (P10), after D-ARG-1..3 |
| X1 | **makor → TOC node; the capacity limit was misread (corrected by the operator)** | The council said: `toc.rs` gives 2 nibbles per level with a cap of 255 (`toc.rs:47-64,111-118`), Shulchan Arukh YD has 403 simanim (largest seif count 146, at siman 331, from Sefaria), and the key holds only 12 path nibbles. **The 12 is wrong for V3 keys.** A V3 key carries four tiers (heel/hip/twig/leaf): `promote::tiers_of` folds a full 16-nibble `NiblePath` into them (`promote.rs:61-70`; `MAX_DEPTH = 16`, `hhtl.rs:45`). The 3×4 reading applies only to V1 keys. **A level's 255 cap is also not fixed.** A level value can be a 9-bit codec9 lane: u8 magnitude + presence bit = 511 numeric values + the never-touched sentinel (`ndarray::hpc::codec9`, contract `lane_codec` 9-bit lanes). So siman 403 fits one level with no wider nibble path. Current codec9 is the signed DR512 form (±255). An unsigned U9 reading of the same 9 bits is the operator's proposal and has no symbol yet. Ordinal 0 must never encode: P=0 with magnitude 0 is NEVER_TOUCHED. A hilchot level stays optional, as literary structure for basins, not for capacity. The outline code is KJV-specific: it has no level-schema parameter and no generic citation→path mapper (closed search `crates/**/*.rs`). | **PROBE** (P1, revised) |
| X3 | **Worlds as value ids plus an edge DAG, never key prefixes** | `ashk_ah` has two parents (mechaber, rema), but key-prefix ancestry is a tree. The probe already holds accessibility as a 64×64 relation register. D-ACR-12's tree ascent must not be reused for world inheritance. | **ADOPT-NOW** as the design rule; no code |
| X4, X5 | **Holdings point at positions; two lineages stay two edges** | Every system surveyed makes derived facts reference source positions, never live at them. PROV separates `wasDerivedFrom` (world inheritance origin) from `hadPrimarySource` (rule → makor). The probe already computes the two lineages as separate fixpoints. What is missing: lineage values are rule ids, not node addresses, and there are no makor nodes. | **PROBE** (P3) |
| X6 | **SupportLedger counts a quoting source as distinct** | `causal_audit.rs:268-271`: "this ledger has no derived-from link". INDRA lists the derived-from link as OPEN. Interim, with no contract change: key `source` by the attestor (makor or author), never by world. The derived-from slot itself is a contract change, owned by the contract tier and needing operator sign-off. | interim **PROBE** (P4); contract slot **PARK** (owner decision) |
| X9 | **GadamerRevision as the independence policy** | Copied endorsement = inherited root, giving Echo / NoIncrease. Independent observation = a new independent root, giving IncreaseEligible. A machloket = `unresolved_tension`, which accumulates by union and is never cleared (`revision.rs:358-440,428`). Map the ENCOUNTER onto the horizon, never the world: `InterpretiveHorizon` is a reader's state (awareness, question, revision_index), and putting a world inside it is a trap. Bible lanes being inherited calque (operator, 2026-10-04) is the same pattern. | **PROBE** (P8) |
| X12 | **Mistaber goldens as copied-source fixtures for E4 / D-EBM-4 / D-EBM-9** | These gates are specified but not built (`cartesian-epistemic-self-model.md:249-257`; D-EBM scratchpad). Caught by the falsifier designer: C6 vs C6i cannot distinguish attestor-keying from world-keying, since both give 1 vs 2. Only the F1 diamond does: at (p,e), attestor-keyed gives 1 source, world-keyed gives 2. New fixtures needed: C6m (own makor, expect 2), V47-copy (expect 1), V47-indep (expect 47), E4-dup (expect 1). | **PROBE** (P4) |
| X11 | **Basin at the makor/TOC node key, `subject` as a codebook id in a value lane** | Hermeneutik fix vs the subject-keyed D-ACR-6 rail. Promote at the TOC key; `subject` goes in a value lane and never becomes a second key. A size-preserving label-shuffle null is mandatory (E-BASIN-WIDTH-IS-N-ARTIFACT-1). The Mistaber goldens are too small, so this runs on KJV first. | **PROBE** (P5) |
| X8 | **Negative lineage (why a holding does NOT hold)** | Positive-only lineage is incomplete under `not` (2608.21141). The probe records no lineage for `overridden(p,c)`. Restrict to the stratified fragment; anything else is refused and left to Clingo. | **PROBE** (P7) |
| X13 | **Stable identity separate from path** | `promote::key_at` derives the key from the path, so renumbering a seif silently re-addresses a different unit. Needs a per-edition path→identity registry. Akoma Ntoso eId/wId is only a lead (abstract-only). | **PROBE** (P9, a census first) |
| X2 | **Commentaries are separate linked works** | Shach and Taz are each both a commentary (text) and a world (authority); the Rema's gloss is not the `rema` world. `toc` has no linked-work edge. | **PARK** until P6's link-density census |
| X7 | Count source sets, not derivation coefficients | Already how the probe and `revision.rs` work (2202.10766). | **SKIP** (ALREADY-HAVE) |
| X14, X15 | Soufflé rule+height per tuple; LegalRuleML / DDL split | Confirm existing choices. | **SKIP** (ALREADY-HAVE) |
| X16 | Probabilistic copy weighting, (1−c)^outdeg | Conflicts with the deterministic substrate and Gadamer's exact zero weight for echoes. | **SKIP** (CONFLICTS-ANCHOR) |
| X17 | world as `prov:Agent`; holding as an annotation body; override as `wasInvalidatedBy` | Each conflates two of the three signatures. | **SKIP** (CONFLICTS-ANCHOR) |

## Named probes (pre-registered; none run)

- **P1, siman capacity (revised).** A 2-nibble, 255-cap level must give exactly 148 refusals for simanim 256..403 and 0 silent aliases. That is the control, showing the cap is real. A 9-bit codec9 level must give 0 refusals and 403 distinct, round-tripping keys. Every SA YD citation must resolve within `MAX_DEPTH = 16`, and its key must read back the same path via `tiers_of`. Ordinal 0 must be refused, never encoded as NEVER_TOUCHED. Disable runs: drop the presence bit (red: 148 aliases or refusals); encode ordinal 0 (red: a citation reads as never touched).
- **P3.** The goldens give identical lineage with node addresses as with strings.
- **P4, independence.** On F1, (p,e) attestor-keyed = 1; switching to world-keying must go red. C6m = 2; V47-copy = 1; V47-indep = 47; E4-dup = 1.
- **P5.** Literary-unit basin vs subject basin against a size-preserving shuffle null on KJV. Kill at ≤ 95th percentile. Fully shuffled control must fail; planted fixture must pass.
- **P6.** Commentary link density, measured before building any `comments_on` edge.
- **P7.** `why_not(r(1,2), curated)` on E-xlog-override returns the override plus the suppressed `r_R_0` support. `why_not(p, c)` on F1 returns `override(c,p)` plus base support. The other five curated propositions return nothing.
- **P8.** C6 → Echo / NoIncrease. C6i → exactly one new root. A machloket fixture leaves `unresolved_tension` uncleared.
- **P9.** Renumbering census across SA editions; then renumber one seif and require that lineage still resolves to the same node.
- **P10.** Unknown labels refused, with exact counts. Disable run: a hash fallback.

## Addendum: the HHTL16 two-tenant opt-in format (operator, 2026-10-30)

The operator named the opt-in format: tenant 1 is an `is_a` zipper chain (a distinguished-name chain or a NodeGuid), tenant 2 is `part_of`. Tenant 2 ideally matches tenant 1 vertically, but it may also carry one-to-many, location or membership. Read against the code and the plans:

- `rail_geometry.rs` already reads two rails per row: `RailAxis::{Taxonomy, Mereology}`, with `RailCarving::AxisSlab` giving 12 levels plus a 12-level continuation, one byte per level (VERIFIED-IN-CODE). `dual_rail_placement` places by the primary rail and overlays the secondary.
- A slot byte stores `1 + index`; 0 is the hole that ends the chain (`read_path`). So a byte level holds **255** children.
- The capstone plan files the 16 × 8-bit ISA and PART_OF zippers as a proposal, not ABI, and says the two must not be composed automatically (`d-qr-abi-classview-rails-capstone-v1.md` §2).

**Reading (operator, 2026-10-30): two 16-byte registers of the 512-byte SoA node row zip vertically into 16 columns, each column a `(u8:u8)` 64k address: 1 identity + 15 exact addresses.** These are two of the row's 32 sixteen-byte slots, not `ValueTenant` lanes. From `canonical_node.rs`: `NodeRow` is `key` 0..16, `edges` 16..32 (an `EdgeFacet`, a second content-blind facet by the 2026-09-17 ruling), and `value` 32..512; `GUIDS_PER_NODE = 512 / 16 = 32`. The `ValueTenant` lanes are carved inside the 480-byte value slab only. Which two slots carry the `is_a` and `part_of` registers is OPEN (asked of the operator).

```text
is_a register       1234567890123456
part_of register    1234567890123456
                    ^ column k = (is_a[k] : part_of[k]) = one 64k address
```

Column 1 is the node's identity; columns 2–16 are 15 exact addresses of other nodes. That is 32 of the row's 512 bytes, 15 in-row references, no edge table and no 255 cap. A column reading `(0, 0)` is unbound. The bytes stay two separate digits; nothing is widened into a u16. This is an addressing reading of the two registers, not a composition of the `is_a` and `part_of` relations.

| Mistaber node | column 1 (identity) | columns 2–16 (exact addresses) | overflow |
|---|---|---|---|
| makor citation `sa("yd:87:3")` | the seif | its siman, chelek and work | none |
| rule `r_…` | the rule | its world scope (one, by `scope/2`) and every makor it cites | more than 15: refused, counted |
| world | the world | every parent world (`ashk_ah`: mechaber and rema) | more than 15: refused, counted |
| holding | (proposition, world) | immediate rule, source rule, origin world, the source rule's makorot | more than 15: refused, counted |

This replaces the earlier table, which kept a rule's extra makorot and `ashk_ah`'s second parent as edges outside the rails. In-row columns carry them. X3 changes from "worlds as value ids plus an edge DAG" to "worlds as identities whose parents are in-row addresses". The DR64 fixpoint still computes holdings; it reads parents from the rows instead of a separate register.

**The vertical match is free in this corpus.** Mistaber lays out `dsl/corpus/yd_87/seif_3.hll`, so a rule's `part_of` chain (file → seif → siman) and its makor's DN chain name the same positions. Under the operator's format this is the ideal case. It is still two rails, never one composed relation.

**Capacity: settled.** With each of the 16 columns a `(u8:u8)` pair, every column addresses 64k positions. All 3,701 YD seifim, every siman and every commentary seif katan get exact addresses. The earlier one-byte-per-level reading, and the "last two bytes as the leaf" step that followed it, are superseded by the reading above. A hilchot level remains optional, for basin structure only.

X1 and P1 are amended accordingly: P1's 9-bit arm is replaced by a pair-address arm (3,701 distinct seif addresses, 0 refusals, 0 aliases; disable run: read one byte of each pair → aliases).

**P11 (new, pre-registered, not run).** On F1 and the C-mammal goldens, give every rule, world and makor node both rails and check:

- (a) `is_ancestor_of` on tenant 1 reproduces citation containment exactly (seif ⊂ siman ⊂ chelek);
- (b) tenant 2 gives each rule exactly one world, the one `scope/2` names;
- (c) `ashk_ah` carries both parents as in-row addresses, and the fixpoint reads them from the row;
- (d) holdings are unchanged (11/11 goldens).

Disable runs:

- compose the two rails, so a rule inherits its makor's chain: (a) or (b) must go red;
- drop `ashk_ah`'s second address: (c) and the goldens must go red.

## Addendum: property inheritance and the syntax tenants (operator, 2026-10-30)

The operator pointed at two more organs: the episodic-witness basins (24 × i4) for property inheritance over a mammal taxonomy, and the Markov context build (relative-pronoun anaphora pointers, 24 × i4 syntax resolution, the TEKAMOLO tenant). Checked against code:

- `contract::causal_witness::CausalWitnessFacet` reads a 12-byte register as 24 signed i4 loci, each an offset in −8..+7 inside the `temporal.rs` window, with 0 meaning unbound. It is labelled EXPERIMENTAL, not in the §3 catalogue. There are 16 named loci; 8 are reserved. They include `Antecedent` (relative pronoun to its antecedent), `BasinAnchor` (the `part_of:is_a` basin), `SupportedBy` / `Supports`, `Quorum` and `Contradiction`.
- `deepnsm-v2::wave` resolves inside ±8 and returns `Escalate` when the offset overflows.
- `contract::tekamolo_facet::TekamoloFacet` reads the same 16-byte facet in `G4D3`: four 256:256:256 lanes (Temporal, Kausal, Modal, Lokal). All-zero means unaddressed.
- `cognitive-shader-driver/examples/ontology_agreement_probe.rs` already folds property claims as three quorums over a `NiblePath` hierarchy: local (the node and its siblings), up (ancestors) and down (descendants). Its fixture holds mammal > placental (dog, cat, whale, bat) and monotreme (platypus, echidna), with properties fur, eggs, flight and milk. Whale fur and platypus eggs are its named conflicts.

**Two inheritances; keep them apart.**

| | taxonomic (`is_a` rail) | Mistaber (Kripke worlds) |
|---|---|---|
| question | does whale inherit `terrestrial` from mammal? | does observer `reader` hold `lays_eggs(platypus)` from `obs1`? |
| axis | tenant 1, the class chain | accessibility register (edges) |
| default and exception | up/down quorum; `Contradiction` / BasinConflict (whale is not terrestrial, platypus lays eggs) | inherited unless `asserts`, `endorses` or `overridden` at the child |
| organ | `ontology_agreement_probe` quorums + witness loci | the DR64 fixpoint (#1873) |

The C-mammal goldens already use both axes at once: the proposition's subject (`platypus`) is a class node, and the worlds (`obs1`, `obs2`, `reader`) are observers. Mistaber's own crosswalk refuses "taxonomy → worlds" as a category error. So the combination is a pair: (class node on the `is_a` rail, proposition) is held per world. It is never a world placed in the taxonomy.

**The syntax tenants serve encoding, not solving.** TEKAMOLO lanes and `Antecedent` loci describe the source TEXT of a seif. They map onto two of Mistaber's four encoding-decision types:

- event identification: which clause is legally salient, and what a relative pronoun refers to;
- feature selection: time, cause, manner and place conditions become rule-body features.

They do not touch the fixpoint.

**Limit found:** the TEKAMOLO heuristics are German only. `de/tekamolo.tsv` is the one lexicon in `deepnsm-v2` (closed search over its tracked files), and its own docs say German forms ground German text and nothing else. The Shulchan Arukh is Hebrew, and Sefaria's translation is English. Either lexicon must exist before the tenant can read a seif; otherwise every lane reads unaddressed.

**Still open:** which register the operator's "24 × i4 flavours" names. The board left this open on 2026-09-01: `atoms::I4x32` matches the name, and the 24-nibble facet matches the count. This note uses `CausalWitnessFacet` because it is the 24 × i4 basin and anaphora reading in code.

- **P12 (pre-registered, not run).** Put the C-mammal subjects on an `is_a` rail (mammal > placental > {whale, elephant, fox}; marsupial > {wombat, possum}; monotreme > platypus). Run the up/down quorum for `terrestrial` and `lays_eggs`. Expected: whale `terrestrial` → Contradiction; platypus `lays_eggs` → BasinConflict up, and corroborated locally once echidna is present. The Mistaber goldens must stay 11/11. Disable runs:
  - read a world as a taxonomy ancestor: a golden goes red;
  - drop the `Contradiction` locus: whale reads terrestrial.
- **P13.** Census first: how many SA YD seifim in the English translation contain a relative pronoun whose antecedent lies beyond ±8 tokens. That count tells whether `Escalate` or the window carries most of the anaphora before any lexicon is built.

## Open points for the operator

- **Undeclared worlds.** Should an undeclared world be refused? Upstream derives `holds(p,x)` for an undeclared child world (golden F6). A native refusal would be a declared divergence from the golden, not a pass.
- **Siman capacity.** Settled (operator, 2026-10-30): the last two bytes of each HHTL16 rail form a `(u8:u8)` 64k leaf. Still open: is a hilchot level wanted for basin structure, and where would its boundary data come from?
- **SupportLedger derived-from slot.** Contract change; owner and timing.
- **D-ARG-1..3.** Prerequisite for X10. Owner and order.

## Not searched

- Sefaria's actual hilchot boundary data for YD.
- Other SA sections (OC, EH, CM) and their capacities.
- Commentary seif-katan counts.
- The OASIS Akoma Ntoso spec itself.
- Any ontology beyond PROV-O, FRBR, CTS and Web Annotation.
- Mistaber's own HLL makor grammar beyond `sa("yd:N:M")`.

## Closeout

STATUS: done (map only; X1 corrected by the operator: 16 levels, codec9 9-bit levels; HHTL16 two-tenant mapping added, P11; taxonomy-vs-world inheritance split and syntax tenants, P12-P13) | OUTCOME: three-axis split (makor node / world value + DAG edges / holding derived); ten probes named; one ruled interim (attestor-keyed source) | OPEN: undeclared-world policy, hilchot data, SupportLedger derived-from owner, D-ARG-1..3 order.
