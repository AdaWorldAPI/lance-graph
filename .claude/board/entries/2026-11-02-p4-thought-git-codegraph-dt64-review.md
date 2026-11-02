# P4: Thought-Git × Codegraph × DT64 — Prüfung und kleinster Adapter (2026-11-02)

**Status:** MEASURED (read-only Review), keine Implementierung.
**Bezug:** D-TGA, D-TGB, EXO-148, D-CODEGRAPH-0, D-DYNBAG-4, D-DT64-AUTH, PR #1923.
**Grade:** VERIFIED / PARTIAL / MISSING / CONTRADICTED. Jeder Beleg wurde gelesen; Harvest-Treffer dienten nur als Fundort.

**Quellen:**
- `crates/thought-git-probe` (6 Quelldateien, 58 Tests);
- die Pläne `thought-git-api-v1.md`, `thought-git-basics-v1.md`, `d-dt64-4-edge-thoughtgit-hypothesis-time-v1.md` und `d-rmap-separate-tables-thought-palettes-v1.md` §7;
- STATUS_BOARD-Zeilen D-TGA-0..6, D-TGB-0..10 und die EXO-148-Zeilen;
- `ISSUES.md` `ISS-DURABLEREF-THOUGHTCOMMITID`;
- dt64 `lib.rs`/`dr.rs`, `lance-graph/src/graph/cycle_sink.rs`, `lance-graph-planner/src/persist_sink.rs`;
- die Codegraph-/xlog-Änderungen aus #1923.

## 1. Was existiert

| Baustein | Stand | Beleg |
|---|---|---|
| Slot-Pool 0 + 1..=63 mit Generation, `Stale` beim alten Handle, FIFO- oder Least-Used-Eviction mit Squash entlang der Kette | VERIFIED (Lab, D-TGB-1) | `thought-git-probe/src/slots.rs` (`Slot{index, generation}`, `SlotPool::branch/free`) |
| From-scratch-Orakel: nur `Commit` landet; ein Given wird nie überschrieben (Contradiction); uneinige Landungen speichern nichts (Divergence); absent ≠ 0 | VERIFIED (Lab, D-TGB-0) | `src/oracle.rs` `next_version` |
| Squash-Gate und geschützte Leser | VERIFIED (Lab, D-TGA-5/5D) | `src/squash.rs` |
| Receipt statt Historie; Replay ohne Slot; Fold-Zeugen; `CommitWitness` mit mehreren Thoughts pro Zyklus in einer Version (EXO-55) | VERIFIED (Lab, D-TGB-4) | `src/receipt.rs` (`Receipt`, `replay`, `VersionStore::commit_cycle`, `MetadataTable`) |
| Produktive Persistenzkante: ein Batch pro Zyklus; `Fenced` gegen `base_version`; Idempotenz über `(cycle, batch_hash)`; `HashConflict` schlägt geschlossen fehl | VERIFIED (Code) | `persist_sink.rs` `WalSink::commit_cycle`, `CommitOutcome`, `CommitError`; `cycle_sink.rs:170` `LanceCycleWriter` |
| DT64: Verhalten unveränderlich (D5); eine Revision ist eine abgeleitete Classid plus Re-Route; nur das Routing ist versioniert; Zulassung prüft `MethodRef{class, method, sig_rev, guard}` | VERIFIED (Code) | dt64 `lib.rs` Moduldoku, `register_derived`, `route`, `routed_at`, `admit` |
| Codegraph aus #1923 | VERIFIED (Code) | xlog `Cube::stage`/`commit`, codegraph `refresh_crate` |

Zum Codegraph aus #1923:
- versionierter DR-Index (`MapTable`, `map_at`, `RouteVersion`);
- Sparse-Delta-Log als Multiset mit Merge-on-Read;
- atomarer Refresh über `SymbolStage`/`PosStage`/`Cube::stage`;
- Positionen werden nie umnummeriert.

| Fehlt | Stand | Beleg |
|---|---|---|
| `DurableRef`, `ThoughtCommitId` (EXO-148) | MISSING | `ISSUES.md` `ISS-DURABLEREF-THOUGHTCOMMITID`, OPEN; im Probe ist `BaseRef{dataset, branch, version}` der Platzhalter (`receipt.rs:53-60`) |
| D-TGA-1 (Thought-Typ), -2 (Lese-Ebene R), -3 (Rebase mit DRed), -4 (land), -6 (MergeQueue) | MISSING | STATUS_BOARD Queued |
| D-TGB-2 (w_slot-Leser migrieren), -3 (Commit über `LanceCycleWriter`), -5 (resident folds), -7 (wiederverwendbare Crate `lance-graph-thought-git`) | MISSING | STATUS_BOARD Queued; `lance-graph-thought-git` gibt es nur als Plan (D-RFT, D-QSP, D-TGB-7) |
| Für dauerhaftes land/replay gilt: „No durable implementation until EXO-148 is resolved and operator authorizes“ | VERIFIED (Plan-Regel) | `d-dt64-4-…-v1.md:85` |

## 2. Identitäten: sechs Regeln, nicht mischen

| Identität | Ist | Regel | Darf nie |
|---|---|---|---|
| `RouteVersion` | Zähler `dt64::dr::VersionClock` (logisch, im Speicher; Route-Log + MapTable) | monoton, geteilt, nur Routing und Tile-Maps | als Lance-Version oder Thought-Identität gelesen werden |
| `LanceVersion` / `DatasetVersion` | Lance-Manifestversion; `contract::temporal_pov::VersionClock` bildet sie auf Zeilenbereiche ab | dauerhaft, pro Dataset | ohne Dataset und Branch verglichen werden (`d-rmap` §7.6.2) |
| `DurableRef{dataset, branch, version}` | MISSING (EXO-148) | wie oben, dataset- und branch-qualifiziert | durch eine nackte Version ersetzt werden |
| `ThoughtCommitId` | MISSING (EXO-148) | nie wiederverwendet, auch nicht nach discard/squash (`d-rmap` §7.6.1) | aus dem Slot abgeleitet werden |
| W-Slot (CE64 Bits 53..58) | Lane-Label, ein Register, keine Identität und kein Zeiger | nur zusammen mit Basisversion (für 0) oder Slot-Generation (für 1..63) lesbar | allein als Schlüssel dienen |
| Slot-Generation | `Slot.generation: u32` im Probe | lokaler ABA-Schutz des Pools, bedeutungslos nach dem Seal | über den Pool hinaus verglichen werden |

Dazu gehören zwei Kennungen, die keine Thought-Identitäten sind:
- **Quell-Git-SHA:** Herkunft der Codegraph-Eingabe (P2-PROV, „source revision“). Sie ist Inhalt, keine Ordnung.
- **`(CycleId, batch_hash)`:** die Commit-Identität von `LanceCycleWriter`.

**Verbindungen** laufen nur über explizite Datensätze, nie über Gleichsetzung:
- **Receipt:** Basis-`DurableRef`, Programm, Seeds, Slot-Generation.
- **`CommitWitness`:** Zyklus, Kohorte `thought → (Ausgabe-Fold, Outcome)`.
- **`MetaKey`:** Daten-`DurableRef`, Thought.

> **Benennungsfalle (VERIFIED):** Zwei verschiedene Typen heißen `VersionClock`.
> - `dt64::dr::VersionClock` (`dr.rs:167`) ist ein atomarer logischer Zähler.
> - `contract::temporal_pov::VersionClock` (`temporal_pov.rs:252`) bildet Lance-Version auf Zeilenzahl ab.
>
> Eine Adapterschicht muss den Pfad voll ausschreiben.

## 3. Antworten auf die zehn Fragen

1. **Ein kanonischer Codegraph, 63 Layer nur als sparse Änderungen: PARTIAL.**
   - Die Form passt: Der Codegraph ist bereits Basis plus sparse Delta-Log mit Merge-on-Read, und ein Thought ist laut Capstone eine sparse Lane.
   - Es fehlen drei Dinge:
     - **G2:** Neue Symbole erhalten Positionen `base.len()+k` (`PosStage::assign`). Zwei Branches, die beide Symbole hinzufügen, vergeben dieselben Positionen an verschiedene Namen.
     - **G7:** Ein Delta-Log pro Relation gehört heute W0; für Branch-Logs gibt es keinen Lese-Fold (Basis + W0-Log + Branch-Log).
     - **G8:** Siehe Frage 7.
2. **Ruff/Charon/MLIR/egglog prüfen Varianten ohne W0 zu ändern: PARTIAL.**
   - `Cube::stage` und der Stage-Teil von `refresh_crate` prüfen eine Variante vollständig, ohne Cube, Tabelle oder Interner zu ändern (P1-Falsifikatoren).
   - Damit ist das Prüfen gelöst, das Halten aber nicht: `StagedEdges` trägt keine Basisidentität (**G3**). Deshalb darf es nur aufrufgebunden sein und nie als Branch-Träger dienen.
   - MLIR fehlt ganz (Review vom 2026-11-02). egglog ist jetzt gepinnt, aber seine Kapsel ist nicht dauerhaft.
3. **Alle Branches nutzen dieselben F###-Methoden mit expliziten Revisionen: PARTIAL.**
   - Die Revisionsmechanik ist richtig: abgeleitete Classid plus Route; `sig_rev` in `MethodRef`.
   - **G4:** `BindStore::route` schreibt ins globale Route-Log und tickt die geteilte `RouteVersion`. Eine branch-lokale Methodenrevision wäre damit sofort global sichtbar.
   - Die Registrierung einer abgeleiteten Classid ist additiv und unschädlich: Unveränderliches Verhalten, solange nicht geroutet, ist unerreichbar. Die Route dagegen ist Zustand und gehört in den Branch.
4. **Identitäten verbinden ohne zu mischen: siehe §2.** VERIFIED für die vier existierenden, MISSING für `DurableRef`/`ThoughtCommitId`.
5. **branch/diff/squash/discard/replay im Speicher, nur land schreibt: PARTIAL.**
   - Im Probe sind branch, squash, discard und replay VERIFIED.
   - diff und rebase fehlen (D-TGA-1/3), land fehlt (D-TGA-4).
   - Für den Codegraph ist discard trivial, weil W0 erst beim Commit berührt wird.
6. **Mehrere Thoughts pro Zyklus über `LanceCycleWriter` zu einer Version, mit Receipts: PARTIAL.**
   - Der Probe beweist das Muster: `commit_cycle` mit Kohorte und `MetadataTable` mit Seal (`a_landing_needs_the_write_paths_commit_witness`).
   - Die produktive Kante hat dieselbe Form plus Fence.
   - Die Verdrahtung D-TGB-3 fehlt.
   - **G6:** Das Probe-`commit_cycle` prüft `prior` nicht gegen die Basis; das tut nur das produktive `Fenced`. Ein Adapter darf deshalb nur die produktive Kante benutzen.
7. **Verworfene, widersprüchliche oder veraltete Branches übertragen nie still Fakten: PARTIAL.**
   - Abgedeckt:
     - Orakel: Contradiction und Divergence speichern nichts;
     - Squash-Gate und geschützte Leser;
     - `Stale`-Handles;
     - Fence bei veralteter Basis;
     - `HashConflict`;
     - Receipt-Refusals.
   - Es fehlen:
     - Rule 1 „up to date before landing“ (D-TGA-4);
     - DRed-Rebase (D-TGA-3);
     - negationsbewusste MergeQueue (D-TGA-6).
   - **G8 (neu):** Die Probe-Zellen haben Wertsemantik (überschreiben, Divergenz bei Uneinigkeit). Codegraph-Kanten sind ein Bag, also Multiplizitäten mit Summensemantik. Zwei Branches, die dieselbe Quelländerung ableiten, würden beim Summieren doppelt zählen. Squash- und Merge-Regeln für Bag-Zellen sind nicht festgelegt.
8. **MCP und Mermaid zeigen standardmäßig nur W0: MISSING** (es gibt beides noch nicht). Die Regel lässt sich ohne neue Architektur festlegen:
   - Jede Projektion nimmt eine explizite Referenz; ohne Angabe ist das W0 bei seiner aktuellen `DurableRef`, nie `QueryReference::default()` (`d-rmap` §7.2).
   - Eine Branch-Sicht trägt Thought-Identität, Basis und Slot-Generation im Kopf und ist als spekulativ markiert.
9. **Vorbedingungen vor der Integration:**
   - EXO-148 (`DurableRef`, nie wiederverwendete `ThoughtCommitId`) für jedes dauerhafte land oder replay;
   - D-TGA-1 und D-TGA-4 (Rule 1);
   - D-TGA-3 oder die Ausnahme unten;
   - D-TGB-2 (w_slot-Leser);
   - D-TGB-3 (Commit über `LanceCycleWriter`);
   - D-TGB-5 (resident folds);
   - D-TGB-7: die gemeinsame Crate, deren erster Konsument der Codegraph sein sollte. Ein codegraph-eigener Branch-Mechanismus wäre ein zweiter.
   - Bereits erfüllt sind D-TGB-0/1/4 und D-TGA-5/5D (Lab).
10. **Unabhängiger Falsifikator für „W0 nach land = zulässiges Delta, Replay stellt wieder her“: MISSING.** Machbar mit vorhandenen Teilen, siehe §5.

## 4. Kleinster Adapter (Vorschlag, keine Implementierung)

Ein Codegraph-Thought ist ein Thought der künftigen gemeinsamen Thought-Git-Crate. Seine Fakten sind Quellvarianten, keine Positionen.

- **Was ein Branch hält:** seine Eingabe (die geänderten `rows`, später Binär-Lanes plus PROV), sein Receipt (Basis, Frontend und PROV als Programm, Slot-Generation) und eine Route-Overlay-Tabelle für Methodenrevisionen (Position → abgeleitete Classid). Keine Positionen, kein `StagedEdges`, kein eigener Graph.
- **Lesen im Branch:** W0 wird gefaltet; das Branch-Delta wird bei Bedarf per Stage berechnet und nie gespeichert. Bei Bedarf kommt dazu ein sparse Kanten-Log nach G7, erst wenn eine Messung es verlangt.
- **land:** Die Branch-Eingabe läuft erneut durch den Stage-Teil von `refresh_crate` gegen das aktuelle W0. Danach folgen Commit, Route-Overlay per `BindStore::route` und ein `LanceCycleWriter`-Batch für alle Landungen des Zyklus mit `CommitWitness`-Kohorte.
  - Das löst G2: Positionen vergibt nur W0, zum Zeitpunkt des Landens.
  - Es löst G3: Kein Stage überlebt einen W0-Wechsel.
  - Es erfüllt Rule 1 per Konstruktion, weil die Landung neu berechnet statt angewandt wird.
  - Für Codegraph-Fakten ist das eine echte Neuberechnung aus der Quelle, kein semi-naiver Re-Run. Die DRed-Forderung aus Rule 2 entfällt deshalb für diese Faktklasse. Diese Ausnahme muss als Entscheidung festgehalten werden, sie darf nicht stillschweigend gelten.
- **discard:** Branch-Eingabe und Overlay fallen weg; W0, `RouteVersion` und Interner bleiben unverändert (die P1-Snapshots beweisen das bereits für Stage).
- **squash:** Die Eingaben werden vereinigt, dann folgt das Squash-Gate. Bag-Zellen brauchen zuerst die Regel aus G8.
- **Was nicht dazukommt:** kein neuer Store, keine Graphkopie, keine neue Methodenautorität, kein neuer Versionszähler.

## 5. Falsifikatoren (vorgeschlagen)

- **F-LAND (unabhängig):**
  - Nach dem Landen von Branch B auf W0@v ist der Codegraph-Snapshot namensgleich mit einem frischen `Codegraph::open` der Quellmenge „W0-Eingaben mit Bs Eingaben ersetzt“. Verglichen werden alle Abfragen inklusive der Aufrufstellen-Multiplizität; das ist das Orakel aus `refresh_equals_clean_open`.
  - Ein zweiter Branch C über derselben Basis wird danach gelandet. Erwartet wird: Er wird gegen v+1 neu berechnet, und seine neuen Symbole bekommen andere Positionen als in seinem Stage gegen v (macht G2 sichtbar).
  - Disable: Cs altes Stage anwenden. Dann kollidieren die Positionen, und der Test wird rot.
- **F-REPLAY:**
  - Aus Bs Receipt allein (Basis-Platzhalter, Programm, PROV) wird die Eingabe neu erzeugt und auf eine frische Kopie von W0@v gelandet. Der Snapshot muss gleich dem aus F-LAND sein.
  - Disable: PROV aus dem Receipt entfernen. Dann muss das Replay ablehnen.
- **F-DISCARD:** Nach branch, stage und discard ist der P1-Snapshot (Interner, Positionen, Tabellenversion, Logs, `RouteVersion`) unverändert.
- **F-ROUTE:** Eine branch-lokale Methodenrevision lässt `BindStore::current_version()` bis zum Landen unverändert. Danach ergibt `routed_at(v_alt)` die alte Classid und `routed_at(v_neu)` die neue.
- **F-FENCE:** Ein Batch gegen eine veraltete Basis ergibt `Fenced`, und es wird nichts geschrieben (produktive Kante; im Probe fehlt das, G6).

## 6. Bewiesene Lücken, nach Priorität

1. **G2** (Positionskollision) und **G3** (`StagedEdges` ohne Basis): durch Lesen des Codes belegt, durch F-LAND messbar.
2. **G4** (globale Route): durch Lesen von `BindStore::route` belegt.
3. **G8** (Wert- vs. Bag-Semantik): durch Lesen von `next_version` gegen das Delta-Log belegt.
4. **G6** (Fence im Probe): durch Lesen belegt.
5. **G5** (zwei `VersionClock`): Benennungsrisiko, kein Defekt.
6. **G1** (EXO-148): bekannt, blockiert jede dauerhafte Behauptung.

## 7. Erster Schritt, wenn freigegeben

Kein Produktionscode. Ein Lab-Test im Codegraph (Fixtures wie in `mod atomic`) setzt F-LAND und F-DISCARD mit zwei Quellvarianten über derselben Basis um. Ein Disable muss G2 sichtbar machen. Der Test nutzt `refresh_crate` und Snapshots aus #1923 und keine neue Struktur.

Erst danach wird entschieden, wo die Branch-Eingabe lebt: in der D-TGB-7-Crate, nicht im Codegraph.
