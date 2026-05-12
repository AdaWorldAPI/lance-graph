# L0-L4 Lane Akkumulator: Multi-Lane Lookup Prediction

> Konfidenz = Übereinstimmung zwischen Lanes.
> Nicht ein Modell das denkt — fünf Tabellen die abstimmen.

## Die 5 Lanes

```
Lane   Was                  Größe      RAM        Speed         Rolle
────   ───                  ─────      ───        ─────         ─────
L0     Codebook Index       297 KB     permanent  O(1)          "wer bin ich?"
L1     256² i16 Tabelle     128 KB     permanent  5.676 q/s     "wer ist nah?"
L2     4096² sparse         32 MB      LanceDB    2.711 t/s     "wohin führt der Pfad?"
L3     Qwopus 64L Gates     16 MB      LanceDB    277 ctx/s     "was denkt das Modell?"
L4     16Kbit VSA Bundle    512 KB     LanceDB    Hamming       "was habe ich gelernt?"
```

## Akkumulation

```
Input: Zentroid C

L0: codebook_index[token] → C = 42
    Ergebnis: "ich bin Zentroid 42"

L1: table_256[42] → Top-K = [43, 47, 41, 38, 50]
    Ergebnis: "meine Nachbarn sind 43, 47, 41"

L2: sparse_4096[42] → Branch = [43, 1029, 3847]
    Ergebnis: "mein Pfad führt zu 43 (nah) und 1029 (weit)"

L3: qwopus_gate[layer_20][42] → Peak = 43, Epiphanie bei L20
    Ergebnis: "das Modell denkt 43 ist die Antwort"

L4: fingerprint16k(42) → nearest_bundle → Reward = 0.8 bei [43]
    Ergebnis: "letztes Mal war 43 gut (Reward 0.8)"

ABSTIMMUNG:
    L0: 42 (Identität)
    L1: 43 ←
    L2: 43 ←
    L3: 43 ←  (Epiphanie!)
    L4: 43 ←  (historisch belohnt)
    
    4/4 Lanes sagen 43 → KONFIDENZ = 1.0 → EARLY EXIT
    Kein weiterer Lookup nötig.
```

## Konfidenz-Level

```
Übereinstimmung    Konfidenz    Aktion
────────────────   ──────────   ──────
4/4 Lanes einig    SICHER       Early Exit. Antwort sofort.
3/4 Lanes einig    HOCH         Eine Lane weiter konsultieren.
2/4 Lanes einig    MITTEL       Alle Lanes berechnen.
1/4 oder weniger   UNSICHER     Forward Pass on Demand.
0/4                UNBEKANNT    Spider → ReaderLM → Lernen.

Kosten pro Level:
  SICHER:    ~0.1ms (L0+L1 nur, permanent in RAM)
  HOCH:      ~1ms   (+ L2 via LanceDB mmap)
  MITTEL:    ~5ms   (+ L3 Qwopus Gate-Tabelle)
  UNSICHER:  ~500ms (Forward Pass, Jina v5 / Qwen3-VL)
  UNBEKANNT: ~2s    (Spider + ReaderLM + Lernen)
```

## LanceDB Integration

```
L0 + L1: Vec<u8> / Vec<i16> direkt in RAM (425 KB)
  → kein LanceDB nötig, reiner Array-Zugriff

L2: Lance Table mit RaBitQ
  → 4096 Zeilen × 4096 Spalten
  → RaBitQ: 1 bit/Dimension → binary Vorfilterung
  → Nur Top-K Zeilen werden in RAM geladen
  → mmap: OS cached die heißen Zeilen automatisch

L3: Lance Table mit IVF
  → 64 Partitionen (eine pro Schicht)
  → Query: "welche Schicht hat den höchsten Gate-Wert für Zentroid X?"
  → IVF findet die Partition in O(1)

L4: Lance Table mit VSA Index
  → 16Kbit Fingerprints als Binary Vektoren
  → Hamming-Distanz = LanceDB binary_distance_l2
  → Nearest Bundle in O(log n) statt O(n)
```

## Speicher-Budget

```
Permanent RAM (immer geladen):
  L0 Codebook Index:    297 KB
  L1 256² i16 Tabelle:  128 KB
  ────────────────────────────
  Total permanent:      425 KB

On-Demand (LanceDB mmap, OS cached):
  L2 4096² sparse:       32 MB (→ ~2 MB in Cache bei häufigen Pfaden)
  L3 Qwopus Gates:       16 MB (→ ~1 MB in Cache bei heißen Schichten)
  L4 VSA Bundles:       512 KB (→ komplett in Cache nach Warmup)

Peak RAM:               ~4 MB (wenn alle Lanes gleichzeitig aktiv)
Disk:                   ~50 MB

Railway Budget:
  425 KB permanent + 50 MB Disk + LanceDB overhead (~10 MB)
  = ~60 MB für das komplette 5-Lane System
  = 640 MB frei für Wikidata (22M Tripel) + AriGraph
```

## Geschwindigkeit pro Konfidenz-Level

```
Level      Lanes berechnet    Zeit       Tripel/Sek
─────      ───────────────    ────       ──────────
SICHER     L0 + L1            0.1ms      10.000
HOCH       L0 + L1 + L2      1ms        1.000
MITTEL     L0-L3              5ms        200
UNSICHER   L0-L4 + FP         500ms      2
UNBEKANNT  L0-L4 + Spider     2000ms     0.5

Erwartete Verteilung (nach Warmup):
  SICHER:    60% der Queries (→ 6.000 eff. Tripel/s)
  HOCH:      25% (→ 250)
  MITTEL:    10% (→ 20)
  UNSICHER:   4% (→ 0.08)
  UNBEKANNT:  1% (→ 0.005)
  
  Gewichteter Durchschnitt: ~5.000 Tripel/Sek
  Nach 1 Stunde Warmup: ~8.000 Tripel/Sek (mehr SICHER durch L4 Lernen)
```

## L4 Lernschleife

```
Jeder Lookup der NICHT SICHER ist → L4 lernt:

1. Query löst L2 oder L3 aus → Ergebnis wird bewertet
2. Bewertung = Übereinstimmung der Lanes die geantwortet haben
3. Bundle(Query-Zentroide, Ergebnis-Zentroide) → L4 speichern
4. Nächstes Mal: L4 erkennt das Muster → SICHER statt MITTEL
5. Über Zeit: immer mehr Queries werden SICHER
6. Asymptotisch: 90%+ SICHER → ~9.000 Tripel/Sek

L4 IST der Langzeitspeicher der den ganzen Stack beschleunigt.
Jeder Forward Pass der heute UNSICHER ist = ein L4 Eintrag der morgen SICHER macht.
```

## Verbindung zu bestehenden Komponenten

```
L0 = codebook_index.rs (thinking-engine)
L1 = f32_engine.rs mit softmax T=0.01 (thinking-engine)
L2 = SparseBranchGraph in f32_engine.rs (thinking-engine)
L3 = Qwopus Releases v0.1.2 (per-layer Gate-Tabellen)
L4 = fingerprint16k.rs (deepnsm) + l4_bridge.rs (thinking-engine)

LanceDB = lance-graph Core (das Haupt-Repo!)
RaBitQ = lance-graph/src/graph/spo/store.rs
AriGraph = lance-graph/src/graph/arigraph/triplet_graph.rs
NARS = lance-graph-planner/src/cache/nars_engine.rs
DeepNSM = deepnsm/src/ (vocabulary + parser + spo + fingerprint16k)
```
