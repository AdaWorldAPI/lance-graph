# The Perturbation-Shape of a Cascading Grid Failure
# Die Störungs-Gestalt eines kaskadierenden Netzausfalls

*A structural-spectral screen (Weyl · Davis–Kahan · Cheeger · Kron) composed with
a DC-power-flow cascade and an AC voltage-collapse fork, validated on the open
Iberian transmission core.*

*Ein struktur-spektraler Screen (Weyl · Davis–Kahan · Cheeger · Kron), verknüpft
mit einer DC-Lastfluss-Kaskade und einem AC-Spannungskollaps-Zweig, validiert am
offenen iberischen Übertragungsnetz-Kern.*

> Reproducible companion to the `perturbation-sim` crate. All numbers below are
> produced by its examples (`validate`, `weakest_links`, `iberian`) on the
> PyPSA-Eur/OSM Iberian core. Honest by construction: every estimate is flagged.

---

## Abstract / Kurzfassung

**EN.** A line trip is a *low-rank perturbation* of the weighted graph Laplacian
`L`. We read its consequences from one operator through four named theorems:
**Weyl** (eigenvalue shift `|Δλᵢ| ≤ ‖E‖₂`), **Davis–Kahan** (Fiedler-subspace
rotation), **Cheeger** (the field↔cut exchange rate `μ₂/2 ≤ h ≤ √(2μ₂)`), and
**Kron** reduction (Schur-complement basin tiering) — plus a DC-power-flow LODF
cascade for the local collapse. On the real 261-bus Iberian core the four lenses
**coincide**: the two lines crossing the Cheeger separation seam are exactly the
two whose removal collapses algebraic connectivity the most (≈ 39 % and 20 %),
and they seed the largest cascades. The four contingency factors are
*empirically orthogonal* between the local (*infight*) and global (*Raumgewinn*)
scales (Spearman ρ ≈ ±0.05; Cronbach α = −0.83), and the severity ranking is
temporally stable (test–retest ρ = 0.90). The DC layer is a **structural
vulnerability screen**, not a reproduction of the 28 April 2025 Iberian blackout
— which was a *voltage* collapse (ENTSO-E) and requires the AC fork.

**DE.** Eine Leitungsauslösung ist eine *niedrigrangige Störung* der gewichteten
Graph-Laplace-Matrix `L`. Wir lesen ihre Folgen aus einem einzigen Operator über
vier benannte Sätze: **Weyl** (Eigenwert-Verschiebung `|Δλᵢ| ≤ ‖E‖₂`),
**Davis–Kahan** (Rotation des Fiedler-Unterraums), **Cheeger** (die
Feld↔Schnitt-Wechselrate `μ₂/2 ≤ h ≤ √(2μ₂)`) und **Kron**-Reduktion
(Schur-Komplement-Becken-Hierarchie) — ergänzt um eine DC-Lastfluss-LODF-Kaskade
für den lokalen Kollaps. Am realen iberischen 261-Knoten-Kern **fallen** die vier
Sichtweisen **zusammen**: die zwei Leitungen, die die Cheeger-Trennfuge kreuzen,
sind genau jene, deren Entfernung die algebraische Konnektivität am stärksten
einbrechen lässt (≈ 39 % und 20 %), und sie lösen die größten Kaskaden aus. Die
vier Störungs-Faktoren sind zwischen lokaler (*infight*) und globaler
(*Raumgewinn*) Skala *empirisch orthogonal* (Spearman ρ ≈ ±0,05; Cronbach
α = −0,83), und das Schwere-Ranking ist zeitlich stabil (Test-Retest ρ = 0,90).
Die DC-Ebene ist ein **struktureller Verwundbarkeits-Screen**, keine Reproduktion
des iberischen Blackouts vom 28. April 2025 — dieser war ein *Spannungs*kollaps
(ENTSO-E) und erfordert den AC-Zweig.

---

## 1. The one operator / Der eine Operator

**EN.** Everything derives from the susceptance-weighted Laplacian
`L = B·diag(b)·Bᵀ` (`bₑ = 1/xₑ`). The four methods are four readings of `L`: its
pseudo-inverse `L⁺` (effective-resistance metric + spectral embedding), its
spectrum (Weyl/Davis–Kahan), its normalized spectrum (Cheeger), and its Schur
complement (Kron). The cascade is the only non-linear layer (flow thresholds).
Because the lenses are projections of one object, their *agreements* and
*disagreements* are both informative.

**DE.** Alles leitet sich aus der suszeptanzgewichteten Laplace-Matrix
`L = B·diag(b)·Bᵀ` ab (`bₑ = 1/xₑ`). Die vier Methoden sind vier Lesarten von `L`:
ihre Pseudo-Inverse `L⁺` (Effektivwiderstands-Metrik + spektrale Einbettung), ihr
Spektrum (Weyl/Davis–Kahan), ihr normalisiertes Spektrum (Cheeger) und ihr
Schur-Komplement (Kron). Die Kaskade ist die einzige nicht-lineare Schicht
(Fluss-Schwellen). Da die Sichtweisen Projektionen eines Objekts sind, sind
sowohl ihre *Übereinstimmungen* als auch ihre *Abweichungen* aussagekräftig.

---

## 2. Method / Methode

| Lens / Sicht | Theorem | Reads / Liest | Grade |
|---|---|---|---|
| Spectral shift | **Weyl** `|λᵢ(L+E)−λᵢ(L)| ≤ ‖E‖₂` | global field eigenvalue λ₂ (Raumgewinn) | [G] |
| Subspace rotation | **Davis–Kahan** `sinθ ≤ ‖E‖₂/gap` | how the Fiedler partition turns | [G] |
| Field↔cut | **Cheeger** `μ₂/2 ≤ h ≤ √(2μ₂)` | the separation seam / exchange rate | [G] |
| Basin tiering | **Kron** (Schur complement) | basin → super-node; preserves Rₑff | [G] |
| Local collapse | DC power flow + LODF cascade | the trip footprint (infight) | [H] |
| Voltage trigger | **AC Newton–Raphson** (π-model) | voltage collapse / loading nose | [H] |

**EN.** A trip on line `k` is the rank-1 update `E = −b_k(e_a−e_b)(e_a−e_b)ᵀ`,
`‖E‖₂ = 2b_k`. The *structural* weak-link ranking uses the first-order
sensitivity `∂λ₂/∂wₑ = (v₂[a]−v₂[b])²` — one eigensolve ranks every line; the
exact λ₂-loss is recomputed for the top candidates. The cascade recomputes DC
flows on the surviving network each round and trips lines above their limit.

**DE.** Eine Auslösung von Leitung `k` ist das Rang-1-Update
`E = −b_k(e_a−e_b)(e_a−e_b)ᵀ`, `‖E‖₂ = 2b_k`. Das *strukturelle* Schwachstellen-
Ranking nutzt die Sensitivität erster Ordnung `∂λ₂/∂wₑ = (v₂[a]−v₂[b])²` — ein
Eigenlöser rangiert jede Leitung; der exakte λ₂-Verlust wird für die
Top-Kandidaten neu berechnet. Die Kaskade berechnet die DC-Flüsse im
überlebenden Netz jede Runde neu und löst Leitungen oberhalb ihrer Grenze aus.

---

## 3. Data / Daten

**EN.** Topology from the PyPSA-Eur/OSM prebuilt network (Zenodo 13358976, ODbL;
published as a release asset, never committed). The base CSV carries only
voltage/length/circuits, so reactance (`bₑ = 1/x`, `x ≈ 0.33 Ω/km·length`) and
limits are **estimated** and disclosed via `n_estimated_*`. Ground truth for
validation: the ENTSO-E expert-panel final report (electrical mechanism) and the
Eurosurveillance excess-mortality study (human footprint, 147 deaths). Missing
per-asset condition is modeled as a **uniform constant** (injects no spurious
heterogeneity) or a topology-derived density proxy (the Gegenhypothese: sparse
rural areas are older).

**DE.** Topologie aus dem PyPSA-Eur/OSM-Netz (Zenodo 13358976, ODbL;
veröffentlicht als Release-Asset, nie eingecheckt). Die Basis-CSV enthält nur
Spannung/Länge/Stromkreise, daher sind Reaktanz (`bₑ = 1/x`, `x ≈ 0,33 Ω/km·Länge`)
und Grenzen **geschätzt** und über `n_estimated_*` offengelegt. Grundwahrheit zur
Validierung: der ENTSO-E-Expertengremium-Abschlussbericht (elektrischer
Mechanismus) und die Eurosurveillance-Übersterblichkeits-Studie (menschlicher
Fußabdruck, 147 Tote). Fehlender Anlagen-Zustand wird als **gleichförmige
Konstante** modelliert (erzeugt keine Schein-Heterogenität) oder als
topologie-abgeleiteter Dichte-Proxy (die Gegenhypothese: dünn besiedelte
ländliche Gebiete sind älter).

---

## 4. Results / Ergebnisse

### 4.1 Factor battery (validate) / Faktoren-Batterie

| Statistic / Statistik | Value | Reading / Lesart |
|---|---|---|
| Cronbach α (5 factors) | **−0.83** | distinct facets, not one scale / eigenständige Facetten, keine Skala |
| ICC(2,1) | **−0.11** | factors disagree → measure different constructs |
| Spearman (spectral cluster) | **0.96–1.00** | convergent validity / konvergente Validität |
| Spearman (infight vs field) | **≈ ±0.05** | **orthogonal** — the Go duality, measured |
| Time test–retest | **0.90** | stable ranking over time / stabiles Ranking |

**EN.** Significance must use the **Jirak (2016)** weak-dependence rate
`n^(p/2−1)`, not IID Berry–Esseen: contingencies share lines and are
autocorrelated. **DE.** Signifikanz muss die **Jirak-(2016)**-Schwach-Abhängigkeits-
Rate `n^(p/2−1)` verwenden, nicht IID-Berry–Esseen: Störfälle teilen Leitungen
und sind autokorreliert.

### 4.2 Weakest links & the boundary that flaps / Schwachstellen & flatternde Grenze

**EN (the headline result).** On the 261-bus Iberian core (`base λ₂ = 3.15e-7` —
very weakly connected), the **structural** weak links by λ₂-loss are line **150
(1276–963) ≈ 39 %**, line 185 ≈ 20 %, line 294 ≈ 20 %, line **46 (1058–1446) ≈
20 %**. The **Cheeger** seam (μ₂ = 5.1e-4, φ = 1.7e-3) separates **100 | 161**
buses and **crosses exactly two lines — 46 and 150 — the top-two structural weak
links.** Operationally (10 % headroom), **23/25** top candidates cascade to ≥3
lines and most **island** the grid (seed 15 → 54 lines, seed 150 → 48). The three
lenses **coincide**: two lines hold the core together; losing either collapses
λ₂ and fragments the network.

**DE (das Kernergebnis).** Am iberischen 261-Knoten-Kern (`base λ₂ = 3,15e-7` —
sehr schwach verbunden) sind die **strukturellen** Schwachstellen nach λ₂-Verlust
Leitung **150 (1276–963) ≈ 39 %**, Leitung 185 ≈ 20 %, Leitung 294 ≈ 20 %, Leitung
**46 (1058–1446) ≈ 20 %**. Die **Cheeger**-Fuge (μ₂ = 5,1e-4, φ = 1,7e-3) trennt
**100 | 161** Knoten und **kreuzt genau zwei Leitungen — 46 und 150 — die zwei
größten strukturellen Schwachstellen.** Operativ (10 % Reserve) kaskadieren
**23/25** Top-Kandidaten auf ≥3 Leitungen und die meisten **verinseln** das Netz
(Seed 15 → 54 Leitungen, Seed 150 → 48). Die drei Sichtweisen **fallen zusammen**:
zwei Leitungen halten den Kern zusammen; der Verlust einer davon lässt λ₂
einbrechen und fragmentiert das Netz.

### 4.3 The 4 models on 4 HHTL tiers / Die 4 Modelle auf 4 HHTL-Ebenen

**EN.** Recursive spectral (Cheeger/Fiedler) bisection builds the OGAR HHTL tree
(HEEL→HIP→TWIG→LEAF); the four theorems are read at each tier:

| tier | basins | Weyl λ₂ (med) | DK gap λ₃−λ₂ | Cheeger μ₂/φ | Kron (N, out-of-family ties) |
|---|---|---|---|---|---|
| HEEL | 1 | 3.15e-7 | 1.0e-6 | 5.1e-4 / 1.7e-3 | (1, 0) |
| HIP | 2 | 2.07e-6 | 4.0e-6 | 2.7e-3 / 5.0e-3 | (2, **2**) |
| TWIG | 4 | 5.65e-6 | 2.3e-6 | 1.0e-2 / 2.0e-2 | (4, 9) |
| LEAF | 8 | 8.88e-6 | 7.3e-6 | 1.1e-2 / 2.4e-2 | (8, 20) |

The readings cohere: λ₂ rises monotonically HEEL→LEAF (**Cauchy interlacing** —
finer basins better-connected); Cheeger μ₂/φ rise as bottlenecks ease; and the
**HIP tier has exactly 2 out-of-family ties — lines 46 & 150 — the same weakest
links** the structural and operational analyses found. A second corridor between
the two HIP basins (§ reinforcement) = a **third out-of-family edge** in the
canonical `EdgeBlock` (4 such slots reserved); its λ₂ gain is bounded by
interlacing.

**DE.** Rekursive spektrale (Cheeger/Fiedler) Halbierung baut den OGAR-HHTL-Baum
(HEEL→HIP→TWIG→LEAF); die vier Sätze werden je Ebene gelesen (Tabelle oben). Die
Lesarten sind konsistent: λ₂ steigt monoton HEEL→LEAF (**Cauchy-Verschachtelung**
— feinere Becken besser verbunden); Cheeger μ₂/φ steigen, wenn Engpässe
nachlassen; und die **HIP-Ebene hat genau 2 familienfremde Verbindungen —
Leitungen 46 & 150 — dieselben Schwachstellen**, die die strukturelle und
operative Analyse fanden. Ein zweiter Korridor zwischen den beiden HIP-Becken
(§ Verstärkung) = eine **dritte familienfremde Kante** im kanonischen `EdgeBlock`
(4 solche Slots reserviert); ihr λ₂-Gewinn ist durch die Verschachtelung begrenzt.

---

## 5. Solar/wind feed-in threshold / Solar-Wind-Einspeise-Schwelle

**EN.** Practical takeaways for renewable-aware unit commitment:
1. The threshold is a **time-varying margin-to-the-nose**, not a fixed MW cap —
   the feed-in/ramp level keeping the worst N-1 contingency from cascading
   (`collapse_margin` on the AC fork + DC overload margin). Recompute per
   interval.
2. Renewables stress the grid **at the weak links**, not where generation is
   largest — throttle/curtail at the Cheeger-seam lines first (here: 46, 150).
3. Ramping gas turbines **down** removes voltage/reactive support; the binding
   constraint is **voltage**, per the 28 Apr 2025 lesson — compute the threshold
   on the **AC fork**, not the DC screen.
4. A wrong forecast = an injection perturbation → run the cascade to convert
   "MW error" into "lines at risk", with **Jirak-calibrated** bands.
5. Storage at a weak-link bus **raises** the threshold (margin restoration; a
   Pearl rung-2 `do()` intervention).

**DE.** Praktische Erkenntnisse für erneuerbaren-bewusste Kraftwerkseinsatzplanung:
1. Die Schwelle ist eine **zeitvariable Reserve-bis-zur-Nase**, keine feste
   MW-Grenze — das Einspeise-/Rampen-Niveau, das den schlimmsten N-1-Störfall am
   Kaskadieren hindert (`collapse_margin` am AC-Zweig + DC-Überlast-Reserve).
   Pro Intervall neu berechnen.
2. Erneuerbare belasten das Netz **an den Schwachstellen**, nicht dort, wo die
   Erzeugung am größten ist — zuerst an den Cheeger-Fugen-Leitungen drosseln/
   abregeln (hier: 46, 150).
3. Das **Herunterfahren** von Gasturbinen entzieht Spannungs-/Blindleistungs-
   Stützung; die bindende Größe ist die **Spannung** (Lehre vom 28.04.2025) — die
   Schwelle am **AC-Zweig** berechnen, nicht am DC-Screen.
4. Eine falsche Prognose = eine Einspeise-Störung → die Kaskade rechnen, um
   "MW-Fehler" in "gefährdete Leitungen" zu übersetzen, mit **Jirak-kalibrierten**
   Bändern.
5. Speicher an einem Schwachstellen-Knoten **hebt** die Schwelle (Reserve-
   Wiederherstellung; eine Pearl-Stufe-2-`do()`-Intervention).

---

## 6. Limitations / Grenzen

**EN.** (i) The DC cascade is the *line-overload* mechanism; the 28 Apr 2025
event was *voltage* collapse — the field tier screens the separation geometry,
the **AC fork** is required for the trigger. (ii) Reactance/limits are estimated
from open data; absolute MW need real `s_nom` + ENTSO-E/ESIOS injections.
(iii) The Walsh/Morton pyramid screen equals the graph eigenbasis exactly only on
hypercubes → it flags, the exact eigensolve certifies. (iv) The OSM ES core is
weakly connected (`λ₂ ≈ 3e-7`); results are relative-structural, not a calibrated
operational study.

**DE.** (i) Die DC-Kaskade ist der *Leitungsüberlast*-Mechanismus; das Ereignis
vom 28.04.2025 war ein *Spannungs*kollaps — die Feld-Ebene screent die
Trenn-Geometrie, der **AC-Zweig** ist für den Auslöser nötig. (ii) Reaktanz/Grenzen
sind aus offenen Daten geschätzt; absolute MW benötigen echte `s_nom` +
ENTSO-E/ESIOS-Einspeisungen. (iii) Der Walsh/Morton-Pyramiden-Screen gleicht der
Graph-Eigenbasis exakt nur auf Hyperwürfeln → er markiert, der exakte Eigenlöser
zertifiziert. (iv) Der OSM-ES-Kern ist schwach verbunden (`λ₂ ≈ 3e-7`); die
Ergebnisse sind relativ-strukturell, keine kalibrierte operative Studie.

---

## 7. Conclusion / Fazit

**EN.** One Laplacian, four theorems, one cascade: the structural weak links
(Weyl/Fiedler), the spectral separation seam (Cheeger), and the operational
cascade origins **coincide** on the real Iberian core — two lines hold it
together. The factor battery confirms the local⊥global (infight⊥Raumgewinn)
duality empirically (ρ ≈ ±0.05), with a temporally stable ranking (ρ = 0.90).
This is an honest *structural screen*; the voltage mechanism of the real blackout
is the AC fork's job.

**DE.** Eine Laplace-Matrix, vier Sätze, eine Kaskade: die strukturellen
Schwachstellen (Weyl/Fiedler), die spektrale Trennfuge (Cheeger) und die
operativen Kaskaden-Ursprünge **fallen** am realen iberischen Kern **zusammen** —
zwei Leitungen halten ihn zusammen. Die Faktoren-Batterie bestätigt die
lokal⊥global-(infight⊥Raumgewinn)-Dualität empirisch (ρ ≈ ±0,05), mit zeitlich
stabilem Ranking (ρ = 0,90). Dies ist ein ehrlicher *struktureller Screen*; der
Spannungsmechanismus des realen Blackouts ist Aufgabe des AC-Zweigs.

---

## References / Literatur

Weyl (1912) eigenvalue perturbation · Davis–Kahan (1970) sinθ theorem · Cheeger
(1970) isoperimetric inequality · Dörfler & Bullo (2013) Kron reduction · Spielman
& Srivastava (2008) effective-resistance sketch · Jirak (2016, Ann. Prob. 44(3),
arXiv:1606.01617) Berry–Esseen under weak dependence · Köstenberger & Stark (2024,
arXiv:2307.06057) · Düker & Zoubouloglou (2024, arXiv:2405.11452) · Pflug & Pichler
(2012, SIAM J. Optim. 22(1)) · Hambly & Lyons (2010, Ann. Math. 171) · Salvi et al.
(2021, arXiv:2006.14794) signature kernel · PyPSA-Eur/OSM (Zenodo 13358976, ODbL) ·
ENTSO-E (2026) Iberian-blackout final report · Eurosurveillance 30/26 (2025)
excess-mortality study. Full provenance: `DATA_SOURCES.md`; methods: `METHODS.md`.

*Numbers reproducible via the crate examples; significance via the Jirak rate.
Zahlen reproduzierbar über die Crate-Beispiele; Signifikanz über die Jirak-Rate.*
