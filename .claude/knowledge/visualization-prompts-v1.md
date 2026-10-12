# Visualization prompts — where the stack is strongest (2026-10-12)

READ BY: anyone asking an image model (GPT image, etc.) for a picture of the
current state; refresh this file before reusing it after a large merge.

Rule for every prompt below: an image model draws whatever it is told is
there. So each prompt names only what has shipped, and marks research as
research. The evidence column is for the human; do not paste it into the
prompt.

## What has shipped (evidence)

| capability | evidence |
|---|---|
| Directory simulation with version history: AD (LDIF) and Entra (Graph JSON) observed read-only, one history, undo as a new version | dir-sim; image/restore/revert #1473; Lance log D-DSP-2 #1485 |
| Group membership with the chain kept (direct vs nested vs via which group), transitive members | #1458; Graph groups + `transitiveMemberOf` observed (HubSPO wishlist 19) |
| Address claim vs delivery kept apart; a mail-enabled group grants delivery, not send-as | dir-sim delivery work; Stalwart fix (task #41) |
| LDIF export and a read-only LDAP server over a simulated version | #1469, #1453 |
| One authorization rule shared by Spear, MedCare, a2ui and OGAR: only the signed-in subject holds roles; delegated login read from the token's `act` chain | `rbac_plug::SubjectRoles` + `with_act_claim` #1491; OGAR #356; Spear #25 |
| Group → role grants in a versioned policy, unread ≠ empty | `GroupRoles::from_policy` #1486 |
| Mail server and mail hub on that directory: Stalwart resolves recipients via dir-sim; Spear scopes mailbox access by owner | Stalwart dir-sim backend; Spear IAM |

Research / proposal only (draw dashed, labelled "Forschung"): CE64
hypothesis layers and NAL polyfill (#1496, plans only), PLN/Shannon attention
(#1497), Hyperon/MeTTa projection (#1498), NARS evidence/fission (#1499),
Rubicon volition (#1500), class-method hot-plug (#1501), policy baselines
GPO/Intune (plan `policy-baseline-integration-v1`).

## Prompt 1 — one-page capability map

> A clean, modern one-page capability map in German, light background,
> flat vector style, for an IT decision-maker. Title: "LanceGraph — was heute
> läuft". Center: a column labelled "Verzeichnis-Simulation (nur lesend)"
> with four stacked tiles: "Active Directory (LDIF)", "Entra ID (Graph)",
> "Versionsverlauf mit Rückgängig", "LDAP-Server (nur lesend)". To the right,
> a tile "Eine Berechtigungsregel für alle Anwendungen" with three small app
> icons labelled "Spear (Mail-Hub)", "MedCare", "a2ui", and a small badge
> "angemeldete Person hält Rollen; Vertretung aus Token-Kette". Below it a
> tile "Mailserver Stalwart: Zustellung aus dem Verzeichnis" with a note
> "Verteiler = Zustellung, keine Berechtigung". At the bottom, a separate
> dashed area labelled "Forschung, noch nicht gebaut" with small grey tiles:
> "Hypothesen-Schichten", "Aufmerksamkeit (PLN/Shannon)",
> "Hyperon/MeTTa", "Willensbildung (Rubikon)", "Klassen-Methoden-Hotplug",
> "Richtlinien GPO/Intune". Solid colour only for built items; grey and
> dashed for research. No numbers, no percentages, no logos of real vendors
> other than plain text names.

## Prompt 2 — the strongest screen: access path with history

> A Windows-style admin application screenshot, German UI, title bar
> "LanceGraph — Verzeichnis-Simulation". Left: a directory tree with
> "contoso.local [AD]" and "contoso.onmicrosoft.com [Entra ID]", both marked
> "nur lesend". Center: a user "Anna Müller" with a table "Mitglied von"
> whose column "Art" shows "direkt", "verschachtelt über GG_Vertrieb". A
> separate, visually distinct section below it titled "Mail-Zustellung (keine
> Berechtigung)" lists "vertrieb@ — Verteiler". Right: a panel "Zugriffspfade"
> drawing Anna → GG_Vertrieb → GG_CRM_Users → "Rolle: CRM-Leser (Richtlinie
> v3)". Top of the window: a timeline bar "Versionen" with version markers,
> one labelled "Simulation: Abteilungswechsel", and a button
> "Rückgängig als neue Version". Status bar: "Simulation — es wird nichts an
> AD oder Entra geschrieben · Änderungen als Antrag exportieren". No column
> called "zuletzt genutzt", no device policies.

## Prompt 3 — architecture, one rule in many apps

> A minimalist architecture diagram, German labels, white background, thin
> lines. Left: two sources "AD (LDIF)" and "Entra ID (Graph)" with arrows
> labelled "nur lesen" into a box "Verzeichnis-Simulation (versioniert)".
> From it, an arrow labelled "Mitgliedschaft mit Kette" into a box
> "Berechtigungsregel: nur die angemeldete Person hält Rollen". A small token
> icon labelled "Anmelde-Token mit Vertretungskette (act)" feeds the same box.
> From the rule box, three arrows to "Spear", "MedCare", "a2ui", and one arrow
> to "Stalwart Mailserver" labelled "Zustellung". No arrow points back into
> AD or Entra.
