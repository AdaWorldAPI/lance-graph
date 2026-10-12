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
| Group membership, direct and transitive (flattened: the intermediate groups are not returned, so a drawn path Anna → GG_Vertrieb → GG_CRM_Users is research) | #1458; Graph groups + `transitiveMemberOf` observed (HubSPO wishlist 19) |
| Address claim vs delivery kept apart; in Stalwart a distribution list grants delivery only, and security-group membership is not send-as | dir-sim delivery work; Stalwart #13, #14 (`groups_of` = SID chain only). The dir-sim backend is a build feature, not yet selectable in Stalwart config |
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
> "Verteiler = Zustellung, keine Berechtigung" and a small grey badge
> "noch nicht per Konfiguration wählbar". At the bottom, a separate
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
> "nur lesend"; under Entra ID show only "Benutzer" and "Gruppen". Center: a
> user "Anna Müller" with fields "Objekt-GUID" (a GUID) and, separately,
> "Ort (DN)"; a field "Cloud-Identität: verknüpft über Sync-Anker". A table
> "Mitglied von" lists only security groups, with column "Art" showing
> "direkt" or "verschachtelt" (never "über" a named group). Below it, a separate box
> titled "Verteiler: Zustellung, keine Rechte" listing "vertrieb@". Right: a
> panel "Zugriffspfade" whose path Anna → GG_Vertrieb → GG_CRM_Users →
> "Rolle: CRM-Leser (Richtlinie v3)" is drawn dashed in grey with the caption
> "Pfad-Erklärung: geplant". Top: a version timeline of observed
> versions as solid dots; from one of them a dashed side branch labelled
> "Simulation: Abteilungswechsel" leaves the main line. No planned or future
> versions. A button "Rückgängig als neue Version". Status bar: "Simulation —
> es wird nichts an AD oder Entra geschrieben · Änderungen als Antrag
> exportieren". No column called "zuletzt genutzt", no device policies.

Known misreadings from the first render (2026-10-12): "keine Berechtigung"
was drawn as "you may not view this"; the distribution group was listed among
memberships; the simulation was drawn on the observed timeline next to a
"geplant" version; the DN was labelled as the object id; Entra roles and
applications appeared although nothing reads them.

## Prompt 4 — before/after of a simulated change (the strongest view)

What backs it: two versions of one history (the observed one and a simulated
one), each read as a `View`; group changes are the set difference of
`groups_transitive` on both views; mail effects come from the mail chain
(`members_transitive_through` mail-enabled); role effects from the versioned
group-role policy. What does not back it: the intermediate groups of a path,
usage dates, recommendations.

> A Windows-style admin application screenshot, German UI, title
> "LanceGraph — Verzeichnis-Simulation". Top: a timeline of observed
> versions as solid dots; from the last one a dashed branch labelled
> "Simulation: Abteilungswechsel Vertrieb → Marketing". No planned versions.
> Center: user "Anna Müller", field "Abteilung" showing "Vertrieb" struck
> through and "Marketing". A tab "Vorschau Änderungen" with a table
> "Auswirkungen der Simulation" and columns "Änderung", "Gruppe", "Art",
> "Folge". Rows: red "entfernt GG_Vertrieb (direkt, gesetzt)", red "entfernt
> GG_CRM_Users (verschachtelt, abgeleitet)", green "hinzugefügt GG_Marketing
> (direkt, gesetzt)", green "hinzugefügt GG_CMS_Editors (verschachtelt,
> abgeleitet)", grey "unverändert GG_Standort_Berlin". Each row says whether
> the simulation set it or it follows from nesting. Below, a separate box
> "Zustellung (keine Rechte)" with one row "vertrieb@ — erhält keine Mails
> mehr". Right: two columns "Vorher" and "Nachher", each listing roles only:
> "CRM-Leser (Richtlinie v3)" struck through on the left, "CMS-Redakteur
> (Richtlinie v3)" on the right; no group chains between them. Status bar:
> "Simulation — es wird nichts an AD oder Entra geschrieben · 2 entfernt ·
> 2 hinzugefügt · Änderungen als Antrag exportieren". No "zuletzt genutzt",
> no "Entfernen empfohlen", no Entra roles or applications in the tree.

## Prompt 3 — architecture, one rule in many apps

> A minimalist architecture diagram, German labels, white background, thin
> lines. Left: two sources "AD (LDIF)" and "Entra ID (Graph)" with arrows
> labelled "nur lesen" into a box "Verzeichnis-Simulation (versioniert)".
> From it, an arrow labelled "Mitgliedschaft (direkt und transitiv)" into a box
> "Berechtigungsregel: nur die angemeldete Person hält Rollen". A small token
> icon labelled "Anmelde-Token mit Vertretungskette (act)" feeds the same box.
> From the rule box, three arrows to "Spear", "MedCare", "a2ui", and one arrow
> to "Stalwart Mailserver" labelled "Zustellung". No arrow points back into
> AD or Entra.
