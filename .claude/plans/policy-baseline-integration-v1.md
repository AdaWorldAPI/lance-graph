# Policy baselines across GPO, Intune and Linux — v1

**Status:** PROPOSAL (2026-10-12), D-PBL-0..7. Nothing is authorized past
D-PBL-0 until the operator makes the three decisions it names.
Origin: the "Richtlinienverwaltung" panel of the identity console mockup
(2026-10-12), which shows one setting per row with a value and a status per
target (GPO, Intune, Linux). Nothing in OGAR or lance-graph reads policy
today (`grep -i 'intune|gplink|grouppolicy'` over OGAR, lance-graph and Spear
returns nothing), so the panel shows data that does not exist yet.

## Overview

The panel's premise: Intune can be shown in GPO terms, so that an
administrator who knows Group Policy sees each Intune setting under the
name they already know, and GPO, Intune and Linux line up in one row. Part
of this is real. Intune's "Administrative Templates" are ADMX-backed, and
Microsoft's Group Policy analytics maps GPO settings onto MDM CSPs, so a
documented mapping exists for some settings. For many it does not. And a
"value per target" hides three questions that must not be merged:

| question | GPO answer | Intune answer |
|---|---|---|
| **configured** — what does the policy object say? | `groupPolicyContainer` + SYSVOL files | configuration policy + its settings |
| **assigned** — who is it aimed at? | `gPLink` on OUs/domain/sites, link flags, security filtering, WMI filter | group assignments and exclusions |
| **observed** — what does a device actually have? | not observable centrally; per-machine RSoP only | per-device status reports |

This is the same separation the directory work uses (address claim ≠
provisioning conflict ≠ delivery resolution; unread ≠ explicitly empty).
The panel's "Abweichung auf 3 Geräten" is an *observed* fact and can only
exist in the Intune column. A GPO column can state *configured* and
*assigned*, never "compliant".

## Constraints carried over

- **Read-only.** Nothing writes to AD, SYSVOL, Graph or Intune. The console's
  "Bearbeiten" is a change to a simulated version, exported as a proposal.
- **Encoders hold raw records** (the `ogar-ad` / `ogar-az` pattern): no
  derived "effective" value is stored in an observation.
- **No HTTP in the encoders.** Recorded Graph JSON and LDIF/SYSVOL files are
  the inputs; fetching is a separate example or tool, like `az_ingest.rs`.
- **Data, not literals.** The GPO↔CSP mapping is read from its published
  source, never re-typed into a Rust table.
- **Concepts are minted in `ogar-vocab`** through the hot-plug path; no
  local classid.
- **Unread is not empty.** A source that was not read yields "nicht gelesen",
  never "kein Wert" and never "konform".

## Checklist

- [ ] **D-PBL-0** Decisions (operator): canonical setting key; home crate;
      Linux source. Mint the policy concepts in `ogar-vocab`.
- [ ] **D-PBL-1** Intune read: encode configuration policies, their settings
      and assignments from recorded Graph JSON into raw records.
- [ ] **D-PBL-2** GPO read: `groupPolicyContainer`, `gPLink`, `gPOptions` from
      LDIF; `Registry.pol` and `GptTmpl.inf` from SYSVOL files.
- [ ] **D-PBL-3** Setting identity: load the published GPO↔CSP mapping as data;
      unmapped settings stay visible in their source's own terms.
- [ ] **D-PBL-4** Assignment resolution over the directory simulation.
- [ ] **D-PBL-5** Observed state: Intune per-device status; GPO marked
      "not centrally observable" unless RSoP exports are supplied.
- [ ] **D-PBL-6** Console projection: the panel reads D-PBL-1..5, empty where
      no source exists.
- [ ] **D-PBL-7** Linux (PARKED until D-PBL-0 names a source).

## D-PBL-0 — decisions needed

1. **Canonical setting key.** Options: (a) the CSP path (Intune-native, covers
   settings with no GPO equivalent); (b) the ADMX `class/category/policy`
   name (GPO-native, matches the mockup's wording); (c) both, with the key
   being whichever source the setting came from and the mapping as an edge.
   Recommendation: (c). A forced single key turns every unmapped setting into
   a guess.
2. **Home.** Options: (a) extend `lance-graph-dir-sim` (policies live beside
   the directory they target, one version history); (b) a sibling
   `lance-graph-policy-sim` that reads a dir-sim `View` for assignment.
   Recommendation: (b). Policy history changes on a different cadence from
   the directory, and dir-sim's observation schema should not grow a second
   family.
3. **Linux source.** No directory holds Linux configuration. Candidates: an
   inventory export from a configuration-management tool, or an agent.
   Recommendation: park until a customer names the tool they run.

Concepts to mint (names indicative, the vocabulary owner decides):
`policy_setting`, `policy_object`, `policy_assignment`, `baseline`.

## D-PBL-1 — Intune read

Encoder in `ogar-az` (new module, same shape as `group.rs`): input is
recorded Graph JSON for configuration policies and their settings, the
ADMX-backed administrative-template policies, and the assignments of each.
Device objects are needed for device-group assignment; `ogar-az` encodes
users and groups only today, so devices are part of this deliverable.

Endpoint names and API versions (several are beta-only) are to be verified
against current Graph documentation when the encoder is written; this plan
does not pin them.

**Gate:** a recorded tenant export round-trips into records whose raw values
equal the JSON; an assignment with an exclusion group keeps the exclusion as
its own record (disable run: drop exclusions → test red).

## D-PBL-2 — GPO read

`ogar-ad` already parses LDIF. Add `groupPolicyContainer` attributes
(`gPCFileSysPath`, `versionNumber`, `flags`, extension GUID lists), and on
OUs/domain `gPLink` (with its per-link disabled/enforced flags) and
`gPOptions` (block inheritance). New pure parsers for SYSVOL: `Registry.pol`
(binary) and `GptTmpl.inf` (INI). Security filtering needs the GPC's
security descriptor; if that attribute is not exported, record "filtering
not read", never "applies to all".

**Gate:** a fixture `Registry.pol` parses to the same key/value set as an
independent reader; a link marked disabled is kept and flagged, not dropped.

## D-PBL-3 — setting identity

Load the mapping from its published form (the ADMX-backed CSP definitions
and Group Policy analytics output) as data. A mapping row records its
source and whether it is exact or partial. Settings without a mapping are
shown under their native key with "keine Entsprechung".

**Gate:** a setting present in both sources joins only through a mapping
row; removing the row splits it into two rows (disable run red).

## D-PBL-4 — assignment resolution

GPO: OU subtree (dir-sim `subtree`), link order and precedence (local, site,
domain, OU), enforced links and block inheritance, then security filtering.
Intune: transitive group members via dir-sim, minus exclusions. When a
setting is configured by both GPO and Intune on one device, report the
conflict and the device's declared precedence setting if read; do not pick
a winner without it.

**Gate:** precedence fixtures with enforced + blocked links reproduce the
documented order; a member reached only through a nested group carries the
chain.

## D-PBL-5 — observed state

Intune per-device setting status from its reports. GPO: "nicht zentral
beobachtbar" unless RSoP exports are ingested per machine. The console's
drift count is computed only from observed records.

**Gate:** a GPO-only setting can never display a drift count; a device with
no status report displays "nicht gemeldet", not "konform".

## D-PBL-6 — console projection

The panel reads D-PBL-1..5. Each cell names its source. Columns with no
reader (Linux before D-PBL-7) render empty with the reason.

## Open points

- Whether Microsoft's mapping data can be fetched and stored without a
  separate operator decision (not discussed here).
- Settings expressed only as scripts or custom OMA-URI: kept raw, never
  interpreted.
