# Vorschlag — Ein Substrat, viele Produkte: Die Lowering-Engine

**Status:** Vorschlag zu Richtung & Investition · **Datum:** 2026-08-26
**Erstellt von:** dem r2il- / r2conc- / ogar-loco-Arbeitsstrang

---

## Zusammenfassung

Wir schlagen vor, eine ganze Familie scheinbar verschiedener Produkte —
Reverse-Engineering-Werkzeuge, Code-Sandboxes, Emulatoren, Beschleuniger für
Alt-Laufzeiten, sogar KI-Coding-Assistenten — auf **eine Engine** zu
konsolidieren. Diese Engine tut drei Dinge: sie **senkt** beliebigen Code auf
eine gemeinsame Befehlssprache ab, **adressiert** jedes Stück über eine
kompakte Zahl und **führt** es aus, ohne dass die Daten ihren Ort je
verlassen. Jedes Produkt ist dann nur eine dünne *Richtlinien- oder
Darstellungsschicht* — ein „Handschuh" — über dieser gemeinsamen Engine.

Die tragenden Teile sind bereits gebaut und unabhängig verifiziert. Wir
schlagen vor, (a) zwei kurze Messungen abzuschließen, (b) den ersten
kundenseitigen Handschuh auszuliefern und (c) die eine Forschungswette — einen
gelernten Dekoder — zu finanzieren, die den gesamten Stack eigenständig und
portabel macht.

## Das Problem am Status quo

Der Markt für Reverse-Engineering, Binäranalyse und Sandboxing ist
zersplittert, weil jeder Anbieter denselben Kern — ein Binärformat dekodieren,
modellieren, ausführen oder analysieren — von Grund auf neu gebaut und dann
ein einziges Produkt obendrauf geschraubt hat. Der Kern ist teuer, die
Produkte sind flach. Niemand verteilt die Kernkosten über mehrere Produkte,
weil ihre Kerne mit ihren Oberflächen verwoben sind.

## Unsere These

**Ein Ding adressieren und ein Ding ausführen ist dieselbe Operation.** Das
haben wir bewiesen: zu einer Adresse in unserem Modell zu navigieren und ein
Programm auszuführen ist buchstäblich eine Algebra. Zusammen mit einer
zweiten bewiesenen Eigenschaft — die Daten müssen nie kopiert werden —
bedeutet das, dass die *Engine* tatsächlich wiederverwendbar ist und die
*Grenzkosten eines neuen Produkts eine Oberfläche und eine Richtliniendatei
sind*, keine neue Engine.

## Was bereits bewiesen ist (nicht versprochen)

- **Korrekte Ausführung, ohne Kopieren.** Ein echtes CPU-Programm aus den
  1980ern, übersetzt durch Ghidra (ein Industriestandard-Werkzeug), läuft
  durch unsere Engine und stimmt mit einem *unabhängig geschriebenen*
  Referenz-Emulator in jedem Register und jedem Flag überein — 18 von 18
  Prüfungen. Dabei hat es sogar einen zwei Jahrzehnte alten Fehler in Ghidras
  eigener Definition zutage gefördert — etwas, das nur eine wirklich korrekte
  Engine aufdecken kann.
- **Adressieren schlägt Tragen, gemessen.** Eine Milliarde Operationen
  erzeugte einen *festen* Overhead von 960 Byte — praktisch null pro Operation
  — wo der herkömmliche objekttragende Ansatz zweistellige Gigabytebeträge
  bewegen würde.
- **Navigation = Ausführung.** Bitgenau bewiesen: dieselbe Masken-Algebra
  bedient einen Datei-Browser, einen Spiele-Editor und einen Malware-Scanner.

Ebenso klar sagen wir, was **noch nicht** bewiesen ist: eine Schlagzeilen-
Durchsatzzahl (unser eigenes Experiment hat sich geweigert, eine solche
festzuschreiben — die Messung war zu verrauscht) und der „gelernte Dekoder",
der die letzte Fremdabhängigkeit entfernen würde.

## Die acht Produkte, eine Engine

1. **Bring-your-own-code-Plattform** — beliebiges Binärformat oder Dienst
   einspeisen, ein adressierbares, abfragbares Modell erhalten (Palantir-
   Foundry-artig, für Code).
2. **Reverse-Engineering- / Security-Werkbank** — ein Werkzeug der Ghidra-
   Klasse ohne JVM, CLI-nativ und skriptfähig. Die nachgelagerten
   Analysestufen existieren bereits in Rust.
3. **Zero-Trust-Code-Sandbox** — jedes Binärformat läuft *nur, wenn es auf der
   Whitelist steht*, und wird auf Malware gescannt, **bevor es ausführen
   kann**, mit autonomer Alarmierung. Das ist eine architektonische Garantie,
   keine nachträgliche Verhaltensüberwachung — die meisten Sandboxes können
   das nicht ehrlich versprechen.
4. **Retro-Spielestudio** — „Mario-Maker für den Commodore 64": ein Klassiker
   laden, seine Level und Sprites visuell bearbeiten, sofort neu ausführen.
5. **Reibungslose Alt-Java-Laufzeit** — steinzeitliches Java (EDI-Prozessoren,
   Graph-Bibliotheken) mit nahezu nativen Kosten ohne Neuschrieb ausführen,
   indem man Bahnen adressiert statt Objekte zu tragen.
6. **Wissenschaftliche / deterministische Emulation** — bytegenaue
   Wiederholung von altem numerischem Code und tausende Parametervarianten,
   parallel über denselben Code ausgeführt.
7. **Transcode mit GUI** — auf Alt-Code zeigen, eine strukturierte,
   editierbare Darstellung erhalten, die man auf eine neue Plattform
   umlenken kann.
8. **Code-Graphen für Coding-Agenten** — ein KI-Agent senkt eine Codebasis in
   einen adressierbaren Graphen ab und argumentiert darüber als *ausführbare*
   Struktur, nicht als Text: er kann „was tut das wirklich" beantworten, indem
   er ausführt, statt zu raten.

## Der Security-Keil (empfohlener erster Markt)

Produkt 3 ist der schärfste Einstieg. „Jedes Stück Code läuft nur mit
Whitelist und wird auf bekannte Schadmuster gescannt, *bevor* es handeln
kann" ist ein **Zero-Trust**-Versprechen, das die meisten Sandboxes nicht
geben können, weil sie Verhalten *nach* der Ausführung beobachten. Unsere
Engine verweigert standardmäßig und scannt den *abgesenkten* Code zuerst. Für
regulierte, air-gapped oder lieferketten-sensible Käufer ist genau diese
architektonische Garantie das Produkt.

## Worum wir bitten

1. **Richtung:** Einigung darauf, dies als *eine Engine, viele Handschuhe* zu
   behandeln, statt Einzelprodukte getrennt zu finanzieren.
2. **Zwei kurze Messungen** (Wochen): der fehlende Durchsatz-Benchmark und ein
   erschöpfender Konformitätstest unseres Executors gegen Ghidras eigene
   Referenz — jetzt, solange wir noch von ihr abhängen, damit das Ergebnis die
   Abhängigkeit überlebt.
3. **Erster Handschuh** (1–3 Monate): empfohlen die Zero-Trust-Sandbox oder
   die RE-Werkbank — je nachdem, was am meisten bereits Gebautes wiederverwendet.
4. **Die Forschungswette** (3–6 Monate): der gelernte Dekoder, der das letzte
   Stück fremdes C++ aus der Laufzeit entfernt und den Stack rein in Rust,
   portabel (WASM / Edge / air-gapped) und vollständig eigenständig macht.

## Das eine ehrliche Risiko

Die letzte Fremdkomponente — Ghidras Befehls-Dekoder — liegt noch im
Laufzeitpfad. Sie zu entfernen ist eine Forschungswette, keine Gewissheit.
Jedes obige Produkt funktioniert *heute* mit diesem Dekoder; die Arbeit am
gelernten Dekoder lässt ihn verschwinden. Wir empfehlen, die Produkte auf der
bewiesenen Engine jetzt zu finanzieren und die Dekoder-Forschung parallel zu
betreiben — die Produkte nicht an die Wette zu koppeln.

---

*Technische Details: `TECH-SPEC-AGENTS.md` (Mechanik + Falsifikator-Register)
und `TECH-SPEC-PRODUCT.md` (Reifegrad je Produkt) in diesem Ordner.*
