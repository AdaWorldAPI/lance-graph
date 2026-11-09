# Architektur-Review: Codegraph × DR³ × DT64 × Ruff (2026-11-02)

**Status:** MEASURED (read-only Review) · **Bezug:** D-CODEGRAPH-0, D-DYNBAG-4, PR #1923
**Stand:** lance-graph `claude/clone-repositories-71a5sw` (= PR #1923, offen); ruff `origin/main` ed8fd4194f (#129–#132 gemergt).
Grade: VERIFIED / PARTIAL / MISSING / CONTRADICTED. Pfade relativ zu `crates/`, sofern nicht anders genannt.
Jeder Zeilenbeleg wurde gelesen; Grep-Treffer zählen nur als Fundort. Ein „0 Treffer“ nennt seinen geschlossenen Suchraum.

## 1 — Ein kanonischer Codegraph?

- **1.1 Normalisierte Repräsentation — PARTIAL.** Nur die Def-/Call-Ebene ist normalisiert. `Lanes` (codegraph `src/artifact.rs:187-211`) halten Prädikat + u32-Interner-IDs. Die Prädikate sind DEFINES, CALLS_RESOLVED, MAY_DISPATCH, UNRESOLVED_CALL, PUBLIC, IMPLEMENTS, SYNTHETIC, REFERENCES und CRATE (`artifact.rs:63-75`). Typen, Generics, Effekte und Commands fehlen. Die Identität ist der qualifizierte Namens-String, weder Charon-`FunDeclId` noch Kythe-Ticket noch classid (ruff_charon `src/store.rs:31-54`).
- **1.2 Verlust beim Import — VERIFIED.** `FunFacts` tragen id, Callee-Klasse, refs, deref-Writes und asserts (ruff_charon `src/lib.rs:141-154`). Persistiert werden nur predicate/subject/object/grade/file/start/end (`store.rs:31-39`). Verloren gehen: FunDeclId, Generics, Places, Effekte (nur im Debugtext `lib.rs:336-342`), Typen, Macro-Herkunft, `use`-Items und `kythe_target` (`lib.rs:352-360`).
- **1.3 Kythe und R2IL auf dieselben DR-Koordinaten — MISSING.**
  - Codegraph hängt nicht von r2il ab (codegraph `Cargo.toml:11-16`), und `src/` erwähnt keinen BindStore.
  - Ein Kythe-Knoten lässt sich verlustfrei auf eine DR-Position legen.
  - Ein R2IL-Wort ist dagegen eine `MethodBinding` (dt64 `src/lib.rs:544`), keine Codegraph-Position.
  - Beides auf eine Koordinate zu legen, würde Identität (Name) und Verhalten (Wort) zusammenwerfen. Sie bleiben getrennte Achsen, die über eine Kante verbunden werden.
- **1.4 Mehrere Stores mit gleicher Information — PARTIAL.**
  - Faktische Autorität ist der ruff_charon-TSV-Store: Codegraph liest nur `*.rows.tsv` (graph.rs:415, artifact.rs:318).
  - Codegraph-Lanes und Cube sind eine Projektion.
  - Der ruff_r2il-OfflineSink ist verwaist; kein `impl RefinedTruthSink` in `crates/`.
  - sag-ra-probe dient als Orakel.
  - Die tract-harvest Registry (`registry.rs:198`, `dr3.rs:146` `HashMap<String, Fact>`) ist eine **dritte, String-gekeyte Methodenautorität**.
  - Der dt64 BindStore ist eine vierte Autorität, die keine Codegraph-Symbole kennt.
- **1.5 Gate gegen eine zweite Source of Truth — MISSING.**
  - Codegraph fehlt in `.claude/v3/COMPONENT-MAP.md` (0 Treffer).
  - **Die Codegraph-Tests laufen in keiner CI.** `rust-test.yml` führt dt64 (:150) und xlog (:152) aus, aber nicht codegraph. Geschlossener Suchraum: alle Workflow-Dateien.

## 2 — F### und DT64 als gemeinsame Methodenautorität?

- **2.1 Auflösung über DT64 — PARTIAL.**
  - Reserviert und real:
    - `HARDCODED_BEHAVIOUR = 0xFFFF` (contract `ogar_codebook.rs:583`);
    - `is_behaviour_lens` mit der Grenze ≥ 0xF000 (`class_view.rs:1053`).
  - DT64 löst Identität und Stelligkeit auf, nie Verhalten.
    - `declare_algebra` speichert nur `arity: Vec<u8>` (dt64 `lib.rs:631-655`).
    - Doc: „decides nothing about what a word does“ (`lib.rs:68-73`).
  - Ausgeführt wird in der Linse selbst. Die F000-Einträge in `POLYFILL` (r2il `polyfill.rs:39`) verzweigen in den Opcode-`match` von `FoldDialect` (r2il `lib.rs:1129` `INT_ADD`, `:1247` `GROUP_SUM`). Das ist so gewollt, aber DT64 ist damit nur **Identitäts- und Admissions-Autorität**, keine Ausführungsautorität.
- **2.2 Nutzung der Bindings:**

  | Komponente | Grad | Beleg |
  |---|---|---|
  | R2IL F000 | VERIFIED | `binding.rs` admit :289, Permit :344 |
  | XLOG F100 | PARTIAL | Binding nur als Form-Gate; Bedeutung liefert der Aufrufer über das lokale `enum Word` (xlog `lib.rs:351-360`, „MISSING-1“) |
  | ONNX | MISSING (bewusst) | `lib.rs:11-13` |
  | egglog-r2il-probe | MISSING | ruft `FoldDialect` direkt |
  | r2il-mask-abi-probe | MISSING | — |
  | ogar-r2il / ogar-loco | MISSING | eigene `FnIndex` über `Vocabulary` |
  | xlog-relcomp-probe | PARTIAL | — |

  xlog-relcomp-probe nutzt **`dr_math_binding_probe::MethodSignature`** mit eigener `method_ordinal: u16` und eigenen `FrameStore`/`Guards`/`Refusal`. Das ist eine zweite Methodenidentität ohne dt64-Abhängigkeit.
- **2.3 Dispatch außerhalb DT64 — VERIFIED, mehrfach:**
  - `FoldDialect` `match`;
  - r2il `procs.rs` mit Opcodes, die nicht in `POLYFILL` stehen (`procs.rs:15-20` gegen `polyfill.rs:28-32`);
  - XLOG `Word`;
  - die Prädikat-Codetabelle im Codegraph (`artifact.rs:66-84`);
  - ONNX-Funktionsnamen;
  - `bind_dr64`.
- **2.4 Identität, Signatur, Körper und Implementierung getrennt — PARTIAL.** In DT64 ist das sauber getrennt: `MethodId` ≠ `name` (`lib.rs:95`, `:200-202`), `SigRev`, geordnete `effects`, `Guard`, Alias am Linsenplatz (`polyfill.rs:43-55`). Es gibt vier Kollapsstellen:
  1. `arity` ohne Einheit: Stack-Operanden bei F000, Spalten bei F100 (xlog `lens.rs:21-27`).
  2. `AlgebraRef.word: u8` ist zugleich Identität und Position. Die Ordnung wird doppelt geführt (POLYFILL und `FnIndex`) und nur per Test verbunden.
  3. Bei `ActionDef` ist `predicate: &str` selbst die Identität (contract `action.rs:79-93`).
  4. `MethodSignature.method_ordinal` (siehe 2.2).
- **2.5 Kann Ruff Adapter, Compiler-Ops und Kommandos über Bindings auflösen? — MISSING.** ruff_charon kennt nur Rust-Callees (`lib.rs:521-524`). Eine Zeile „Callee = (Linse, Wort)“ gibt es nicht. Deshalb meinen `users_of` (dt64 `lib.rs:715`) und `callers` (codegraph `query.rs:125`) verschiedene Mengen.
- **2.6 Dynamik, UNKNOWN, Guards, Effekte, Quellzeugen — PARTIAL.**
  - VERIFIED: Effekt-Reihenfolge (`lib.rs:216-221`), Guard-Witness (`:167-178`), benannte Refusals, `Graded{confirmed,possible,unknown}`.
  - Der Quellzeuge steht nicht im `MethodBinding`. Er ist nur über die Lab-`Attachment` in dt64-admission-probe verbunden.

## 3 — Lazy Hydration ohne Hot-Path-Kosten?

- **3.1 Resident nur Identität und Relationen — PARTIAL.**
  - Keine Bodies, kein Quelltext und keine IR resident: VERIFIED (graph.rs:165-178).
  - **MethodRefs: MISSING.** In `codegraph/src` gibt es 0 Treffer für `MethodRef|BindStore|RouteVersion|map_at`.
  - Versionen werden nur über Change-Digests geführt (`artifact.rs:97`, „nicht kryptografisch“).
- **3.2 Bodies und IR on demand — PARTIAL.** Die Quellen existieren: LLBC-JSON mit `files[].contents` und strukturierten Bodies. `lance-graph-hydrate` ist der **falsche Kandidat**: Es kopiert Artefakte S3→Volume→Lance, ein Item-Lookup ist das nicht (CONTRADICTED als Item-Hydration). Ein On-demand-API im Codegraph fehlt.
- **3.3 Seam ohne Re-Parse — MISSING bis auf Quelltext-Slices.** Der einzige Seam ist `(file, start, end)`. **FunDeclId wird nicht persistiert.** Wer einen Body will, muss das ganze LLBC neu laden und per Namen suchen. Bei Impl-Methoden ist das kollisionsanfällig (`lib.rs:73-90`).
- **3.4 Zugehörigkeit zu Quelle, Compiler, Frame und Version — PARTIAL.**
  - Quell-Digest (`files.tsv`, fnv64) und `ROWS_VERSION` sind VERIFIED.
  - **Charon-/LLBC-Version: MISSING.** Charon ist nur als Path-Dependency gepinnt (`Cargo.toml:17-18`).
  - Es gibt zwei Hashes (fnv64 und FastHasher), keinen gemeinsamen Content-Key.
- **3.5 Wiederverwendung — VERIFIED innerhalb des Codegraph.** Digest-Skip, Multiset-Delta, `refresh_closure` und `OnceCell` funktionieren. Aber:
  - Das Gate ist **gerissen**: Ein unveränderter Refresh kostet 1,5–1,7 ms, weil die TSV gelesen und gehasht wird.
  - `with_dependents` (`store.rs:398-411`) ist ungenutzt.

## 4 — MLIR und Rust als SoC-konforme Sichten?

- **4.1 Versionsmodelle je Compiler — PARTIAL.** Charon ist ein Fork mit gepinnter Nightly, aber seine Version steht nicht in den Zeilen. ruff_r2il liest Maschinencode; laut OGAR ist das „NOT on this path“ (ogar-r2il `lib.rs:27-33`). Die R2IL-Arity-Tabelle (82 Einträge) ist nur per Drift-Test an r2sleigh gebunden.
- **4.2 Versionierte Projektionen derselben Identität — PARTIAL.** Die Bausteine sind da: `RouteVersion`, `VersionClock`, `map_at` und `FrameRevisions` (dt64 `dr.rs:153-167`, `:349`, `:902`). Es fehlt eine Achse „Sicht / Frontend / Version“ an der Codegraph-Identität. Heute versioniert DR nur das Routing.
- **4.3 MLIR über den R2IL-Polyfill — MISSING im Code, Pfad plausibel.** Geschlossener Suchraum: alle Crate-Verzeichnisse und `Cargo.toml` in lance-graph und OGAR, Muster `mlir|dialegg|melior|llvm`: 0 Treffer. Ohne neue Architektur ginge es so: ein MLIR-Op-Name wird eine weitere `AliasLanguage` am POLYFILL-Platz (`polyfill.rs:58-65`), und ein Op ohne Wort bekommt eine eigene F###-Linse. Regionen und Kontrollfluss haben kein Gegenstück; `PROC_DEF/PROC_CALL` ist OPEN (`procs.rs:7-9`).
- **4.4 Was egglog/DialEgg brauchen — PARTIAL.**
  - Vorhanden: `RuleId{law, domain}`, eine Zulassungstabelle, ein i128-Orakel und `ReplayCapsule` mit benannten Refusals.
  - Es fehlen Effekte, Typen, Guards und Herkunft.
  - **egglog floatet auf `branch = "main"`** (egglog-r2il-probe `Cargo.toml:25`). Ein `ProofChanged` kann deshalb ein Werkzeugwechsel sein, ohne dass sich die Semantik geändert hat.
- **4.5 Fehlende Information nicht als Gleichheit lesen — PARTIAL.** Die Wächter existieren: Receipt-Erhaltung, Refusals, K3- und Guard-Refusal sowie Overflow-Refusal. Es bleiben sechs Lücken:
  1. `Arg` ist untypisiert.
  2. `arity` hat keine Einheit.
  3. Das Binding trägt keinen Quellzeugen.
  4. Frontend- und egglog-Version fehlen.
  5. egglog sieht keine Effekte.
  6. Pfade außerhalb DT64 bleiben für `users_of` unsichtbar. Ein leeres `users_of` heißt deshalb nicht, dass es keine Nutzer gibt.

## 5 — Dependency-Graph, MCP, Mermaid-Indizes?

- **5.1 Calls, References, Implementations, Dependencies — PARTIAL.** `callers`, `callees`, `references` und `impact` sind VERIFIED und geben Masken zurück (query.rs:125/157/188/239). Es fehlen eine Vorwärts-Closure und **Crate-Dependencies**: kein Prädikat, und die CRATE-Zeile ist nur ein Vollständigkeitsstatus.
- **5.2 MCP — MISSING.** `Cargo.toml` und `*.rs` enthalten kein `mcp`, `rmcp` oder `jsonrpc`. Slice D ist laut Plan bewusst als letztes geplant. Für einen Adapter ist eine Hürde zu beachten: `callers` und `callees` nehmen `&mut self`.
- **5.3 `.index.md` + Mermaid — MISSING.** Die Präzedenz existiert: `entries_index.py`, `supersession_index.py` und die CI-Regeneration mit Diff. Ein Generator muss nach Namen sortieren, weil Positionen von der Lesereihenfolge abhängen.
- **5.4–5.6 — Empfehlung, kein Code.** Ein Index pro Crate, Mermaid nur auf Crate- oder Modul-Ebene. Alles, was größen- oder zustandsabhängig ist, holt ein Agent per Query: Bodies, Masken, Closures, Diagnosen, Residuals. Die CI prüft gegen die committeten Rows, nie per Neukompilat.

## 6 — Was beweist PR #1923?

- **6.1 Implementiert und getestet — VERIFIED:**
  - dt64 `MapTable::locate`, `map_at`, `TileMap` und `VersionClock` mit F1–F6 (`tests/dr.rs:342-577`);
  - xlog `closure_wide`, `new_tiled`, Multiset-DeltaLog, merge-on-read sowie refold, refresh und rebase;
  - Codegraph `refresh_crate`, Digest-Skip und interned Lanes.

  Gemessen und **gescheitert**: der unveränderte Refresh mit 1,5–1,7 ms.
- **6.2 Alte und neue Bindings, spät eintreffende IDs — VERIFIED für dt64/xlog.** F2: ID 30 ist bei v1 `Unmapped` und bei v2 `Ok(2)`, ohne neue Generation. `extends` verlangt ein Präfix, `rebase` refusiert ein Reorder. Für den Codegraph ist das nur PARTIAL, weil dessen Tests ohne CI laufen.
- **6.3 Kann ein gescheiterter Refresh den Zustand teilweise ändern? — CONTRADICTED.** Der Code dokumentiert das sogar selbst (graph.rs:519-520):
  - **Cube-Refusal:** `pos` ist schon erweitert, `advance`, `seal` und `rebase` sind gelaufen (:598-604), und `add_edge`/`retire_edge` werden einzeln mit `?` angewandt (:607-630). Lanes, Masken und Digest folgen erst danach (:658-662). Ein Retry zählt bereits geloggte ADDs im Multiset ein zweites Mal.
  - **NeedsReopen („nothing was changed“):** `pos.truncate` setzt nur die Positionen zurück. Die Interner-Einträge aus `read_text_as`/`intern` (:538-540) und `last_owner` (:151-159) bleiben stehen. Der Test prüft die Länge des Interners nicht (refresh.rs:233-234).
- **6.4 Lokalität des Tile-Routings — PARTIAL.** `MapTable::locate` liest alle Tiles linear (dr.rs:936-954), obwohl der Codegraph die Tile als `g/256` kennt (graph.rs:71, 596). Prefetch ist nicht implementiert.
- **6.5 Noch String/TSV — VERIFIED.** Transport, Digest, Symbolidentität, Grade, `?dynamic`, `may_target`, der Pfad und das String-gekeyte Query-API.
- **6.6 Falsifikatoren an den Grenzen — PARTIAL.** dt64, xlog und r2il↔BindStore sind in der CI. Ruff-Rows↔Codegraph und Codegraph↔xlog sind **nicht** in der CI. Keine Falsifikatoren gibt es für Kythe↔DR, R2IL↔Codegraph, Symbol↔classid und die Atomizität.

## 7 — Kleinster nächster Schritt

Drei Abweichungen, nach Risiko geordnet. Keine führt einen neuen Store oder eine neue Architektur ein. Alles verwendet vorhandene Teile.

### P1 — PR #1923 korrekt machen (Defekt, keine Entscheidung nötig)
- **Änderung:**
  - `refresh_crate` erst vollständig vorbereiten: Interner-Einträge, Positionen, Edge-Delta und Cube-Prüfung laufen auf Kopien bzw. Staging. Danach wird in einem Schritt angewandt.
  - Alternativ wird der vorhandene NeedsReopen-Pfad auch für Cube-Refusals genutzt. Die Interner-Länge und `last_owner` werden dabei zurückgesetzt.
  - Dazu kommt eine CI-Zeile `cargo test --manifest-path crates/lance-graph-codegraph/Cargo.toml` in `rust-test.yml` neben :150/:152.
- **Falsifikatoren** (jeder mit Disable-Lauf, rot→grün):
  1. Ein injizierter Cube-Refusal lässt Interner-Länge, `pos`, Lanes, Masken und Digest exakt unverändert. Ein Retry ist byte-gleich zum Clean-Open, ohne doppelte ADD-Multiplizität.
  2. NeedsReopen lässt die Interner-Länge unverändert. Das erweitert den bestehenden Test `refresh.rs:233`.
  3. Die CI-Zeile fällt, wenn ein Codegraph-Test kaputt ist (einmal mit gebrochenem Test verifiziert).

### P2 — Identitätsnaht Position ↔ Charon ↔ DT64 (braucht eine Entscheidung zum Zeilenformat)
- **Änderung:**
  - Die Zeilen (bzw. der Header des Binärformats aus D-CODEGRAPH-0-BIN) bekommen drei Felder: die Charon-`FunDeclId`, die Charon/LLBC-Version und einen LLBC-Digest.
  - Ein Aufruf eines Linsenworts bekommt eine eigene Zeile mit Callee = `AlgebraRef` statt eines Namens. Das ist ein neues Prädikat in der bestehenden Codetabelle, kein neuer Store.
- **Falsifikatoren:**
  1. Ein Fixture holt den Body einer Impl-Methode über die ID, ohne Namenssuche. Zwei Impl-Methoden mit gleichem `path_name` müssen verschiedene Bodies liefern.
  2. Auf einem Fixture mit drei Aufrufern eines F000-Worts gilt `users_of(word) == callers(word)`. Ein Aufrufer, der um das Binding herumgeht, muss als Differenz sichtbar werden.
  3. Ein Wechsel der Charon-Version allein macht den Store `stale`, auch wenn die Quelle unverändert ist.

### P3 — Parallele Methodenautoritäten benennen und absperren (braucht eine Entscheidung zur Zuständigkeit)
- **Bestand:**
  - BindStore (Autorität);
  - `dr_math_binding_probe::MethodSignature`, tract `Registry`, XLOG `Word`, ogar-loco `FnIndex`;
  - der POLYFILL-Positionsabgleich, der nur per Test geführt wird.
- **Änderung, ohne neue Architektur:**
  - Ein Census-Test zählt jedes ausführbare Wort in `FoldDialect` und `procs` sowie jedes XLOG-`Word` und verlangt dafür eine BindStore-Deklaration. Ein Wort ohne Deklaration steht in einer exakten Ratchet-Liste, die in beide Richtungen fällt (neu oder veraltet).
  - egglog wird auf eine `rev` gepinnt, damit `ProofChanged` eine Bedeutung hat.
- **Falsifikatoren:**
  1. Ein neuer Opcode ohne Deklaration lässt den Census rot werden.
  2. Ein Eintrag, der nicht mehr gilt, lässt die Ratchet rot werden.
  3. Ein egglog-Wechsel ohne Pin-Bump wird vom Build abgelehnt.

### Ausdrücklich nicht jetzt
- **MCP, `.index.md`, Mermaid:** Erst nach P1/P2. Ein Index über instabile, String-gekeyte Identitäten würde nur Drift erzeugen.
- **MLIR:** Erst wenn ein konkreter Op fehlt. Dann als Alias oder F###-Linse, nie als neue Ausführungsschicht.
- **Prefetch:** Erst nach dem direkten Tile-Index (6.4), und erst gemessen.
- **D-CODEGRAPH-0-BIN** (Binärformat, Zählen über Quack) bleibt der Fix für das gerissene Refresh-Gate. Er sollte das Feldset aus P2 in den Header aufnehmen, damit das Format nicht zweimal migriert wird.

## Offene Entscheidungen
- **P2:** Trägt das Binärformat FunDeclId, Charon-Version und LLBC-Digest im Header oder je Zeile? Bekommen Linsenwort-Calls ein eigenes Prädikat?
- **P3:** Werden `MethodSignature` (dr-math-binding-probe) und die tract-`Registry` auf BindStore migriert, oder als Lab-Autoritäten ausdrücklich ausgenommen?
