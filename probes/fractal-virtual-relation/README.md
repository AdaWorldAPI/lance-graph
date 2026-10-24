# Fractal Graph PoC: x^n Kanten vs. mathematische Projektion

Ein ausführbarer, dependency-freier C++17-Benchmark. Er vergleicht **explizites Durchlaufen** einer fraktalen Kantenrelation mit **direktem, mathematischem Rendering**. `visualize.py` erzeugt die Abbildungen; dafür sind Python, Pillow und Matplotlib nötig. Das Rendering selbst benötigt diese Pakete nicht.

## Modell

Die boolesche Basismatrix

```text
B = [1 1]
    [1 0]
```

wird mit dem Kronecker-Produkt auf Tiefe `n` erweitert: `A_n = B ⊗ ... ⊗ B` (`n`-mal). Sie entspricht der Kantenrelation `E(u,v) ⇔ (u & v) == 0`, mit `0 ≤ u,v < 2^n`. Dies ist eine **gerichtete Adjazenzrelation mit Schleifen** (nicht ein einfacher ungerichteter Graph). Die `2^n × 2^n`-Matrix besitzt genau `3^n` gesetzte Einträge.

Für `n = 26`:

- `2^26 = 67.108.864` Knotenindizes pro Achse.
- `3^26 = 2.541.865.828.329` gesetzte Adjazenzeinträge.
- Reine Kantenliste mit zwei 32-bit-Indizes: rund **20,3 TB** ohne Overhead.
- Mathematische Darstellung: `B`, `n` und die Regel für `E`.

## Die drei Messverfahren

1. **Explicit**: iteriert tatsächlich alle `3^n` Kanten mit einer DFS über die erlaubten Bitpaare `(0,0)`, `(0,1)` und `(1,0)`. Die Kanten werden dabei **gestreamt, nicht gespeichert**, wodurch die Referenz sogar speichersparender als ein echter materialisierter Graph ist. Für jede Kante wird ein Bildpixel-Zähler inkrementiert.
2. **Predicate / Full View**: füllt Pixel unmittelbar per `(u & v)==0`. Ist `n > k` (mit `k=9`, 512×512), aggregiert ein Pixel exakt `3^(n-k)` vorhandene Kanten, sofern die beiden `k`-Bit-Präfixe miteinander kompatibel sind; sonst null. Das ist eine exakte **Anzahl-/OR-Projektion**, keine vollständige Darstellung aller einzelnen Kanten.
3. **Exact Zoom**: zeigt auf Tiefe `n` eine *echte* 512×512-Kachel der Originalmatrix, ohne Projektion. Jeder Pixel ist eine individuelle Adjazenzabfrage für konkrete `n`-Bit-Knotenindizes.

Zusätzlich gibt es `Rule Sparse`, welches nur das sichtbare Muster der Tiefe `min(n,k)` durchläuft und den exakten Kantenmultiplikator einsetzt.

**Korrektheitsprüfung:** Für alle bis `max_explicit_depth` getesteten Tiefen werden die Zählerbilder von Explicit, Predicate und Sparse **pixelweise exakt** verglichen. Für sämtliche Tiefen bis 26 wird zudem `sum(pixel_counts)==3^n` geprüft. Die exakte Zoomansicht prüft Stichproben erneut gegen die Mitgliedschaftsformel.

## Ausführen

```bash
cd fractal_graph_poc
g++ -O3 -std=c++17 -march=native -o fractal_bench main.cpp
./fractal_bench 15 5 9 > measurements.csv
python3 visualize.py
```

Oder `./run.sh 15 5 9`. Die Argumente sind `max_explicit_depth`, `repetitions`, `screen_bits`. `screen_bits = 9` bedeutet 512×512 Pixel. Für sehr hohe `max_explicit_depth` steigen die Laufzeiten exponentiell. C++17 und Python 3 sind erforderlich; Pillow/Matplotlib nur für die Visualisierung.

Der ausführbare Benchmark schreibt CSV auf stdout und vier PGM-Bilder in das aktuelle Verzeichnis. Die CSV enthält `NA` für bewusst nicht ausgeführte explizite Messungen. Zeitmessung benutzt den **Bestwert von 5 Wiederholungen**, um Scheduling-Störungen zu reduzieren. Pixelzählung ist inbegriffen, PNG-/PGM-Dateiausgabe, Darstellung am Bildschirm und GPU sind ausgeschlossen. Allokation der Bildpuffer und Checksums sind ausgeschlossen. Die nativen Benchmarks sind keine Aussage über Lance-Graph-, Rust-, SIMD- oder GPU-Performance.

## Beispielmessung

Umgebung: Intel Xeon Platinum 8573C (virtualisiert), GCC 14.2.0, `-O3 -march=native`, Messung vom 24. Oktober 2026. Es handelt sich um eine einzelne Laufserie, **keine statistisch robuste Mikrobenchmarkstudie**. Siehe `measurements.csv` für reproduzierbare Rohwerte und alle Tiefen.

| Tiefe | Kanten | Explicit [ms] | Predicate [ms] | Zoom exakt [ms] |
|---:|---:|---:|---:|---:|
| 12 | 531.441 | 1,084 | 0,067 | 0,069 |
| 15 | 14.348.907 | (siehe CSV) | ~0,07 | ~0,07 |
| 26 | 2.541.865.828.329 | nicht ausgeführt | 0,069 | 0,067 |

## Was dieser PoC NICHT beweist

- Die 2,54 Billionen Kanten werden bei Tiefe 26 **nicht vollständig ausgerendert**; ihre exakte globale Projektion oder ein einzelner exakter Ausschnitt wird erzeugt. Vollständige Enumeration hätte mindestens O(3^n) Ausgabekosten.
- Die Bitregel ist von vornherein bekannt. Der PoC **extrahiert sie nicht aus einem bestehenden Graphen**. Das wäre der nächste Forschungsschritt: Kandidatenregeln aus Beispielen synthetisieren, Ausnahmen modellieren, Äquivalenz beweisen und Cost-based Query Planning integrieren.
- Geschwindigkeit und Größe hängen von **Auflösung und Anfrageart** ab. Komplizierte Abfragen, beliebige globale Reduktionen oder hochauflösende Ausgabe können dennoch hohe Kosten verursachen.

## Möglicher Lance-Graph-Vertrag

```text
MethodId:      KRONECKER_BIT_DISJOINT
Parameters:    depth=n, axis_bits=n
Semantics:     exists(u,v) = ((u & v) == 0)
Cardinality:   3^n
Count(prefix_u, prefix_v, k):
    if ((prefix_u & prefix_v) == 0) then 3^(n-k) else 0
ClassView:     VirtualRelation(MethodId, Parameters)
Execution:     predicate | aggregated_count | exact_roi | materialize
Proof:         independent membership/cardinality equivalence tests
```

`Count` gilt für ausgerichtete, vollständige `2^(n-k)×2^(n-k)`-Blöcke und `0≤k≤n`. Bei willkürlichen ROIs ist diese einfache Kardinalitätsformel ohne weitere Zerlegung nicht verwendbar.