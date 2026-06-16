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

### 4.4 Reinforcement & the two-axis insight (Braess) / Verstärkung & die Zwei-Achsen-Einsicht

**EN.** We add the **optimal third corridor** across the Cheeger seam — the
single new edge maximizing the first-order gain `∂λ₂/∂w = (v₂[a]−v₂[b])²` (the
Fiedler-extreme pair, one bus per basin; in OGAR terms a **third out-of-family
`EdgeBlock` edge**). On the ES core (buses 1199–4258):

| metric | without tie | with tie | move |
|---|---|---|---|
| algebraic connectivity λ₂ | 3.15e-7 | 7.44e-7 | **+136 %** |
| seam-trip connectivity-loss | 39.0 % | 34.4 % | −4.6 pp (better) |
| seam-trip lines cascaded | **48** | **95** | **worse** |

**The headline finding:** *one* reinforcement moves the two axes in **opposite
directions** — Raumgewinn improves (λ₂ ↑ 136 %, connectivity-loss ↓) while
infight worsens (48 → 95 trips). This is the measured **infight ⊥ Raumgewinn
orthogonality (ρ ≈ 0.05)** realized in a single intervention, and it is a
power-grid **Braess paradox**: the new low-impedance path re-routes flow into
lines whose limits were *not* re-rated, so a structurally stronger grid is
operationally *more* cascade-fragile. **The actionable rule: match the remedy to
the failure axis.** A corridor (λ₂ / Cheeger / Kron) fixes **separation /
islanding** (territorial collapse — the 28-Apr-type voltage/separation event,
when paired with reactive support); an **overload cascade** needs **limit
re-rating / redispatch**, not more connectivity — and structural reinforcement
**must be co-designed with limit upgrades** on the lines that will newly carry
flow, or it backfires. The Go-meta `Regime` classifier tells you *which* axis a
given contingency loads, hence which remedy applies. *(Caveat: the cascade
worsening is partly genuine Braess and partly the limits being calibrated to the
pre-tie flows — both reinforce the same lesson: λ₂ gain ⇏ cascade reduction.)*

**DE.** Wir fügen den **optimalen dritten Korridor** über den Cheeger-Schnitt
hinzu — die neue Kante, die den Erstordnungs-Gewinn `∂λ₂/∂w = (v₂[a]−v₂[b])²`
maximiert (das Fiedler-Extrempaar, ein Knoten je Becken; in OGAR eine **dritte
familienfremde `EdgeBlock`-Kante**). Auf dem ES-Kern (Knoten 1199–4258): λ₂
**+136 %**, Konnektivitätsverlust 39,0 %→34,4 % (besser), aber **48 → 95**
kaskadierte Leitungen (schlechter). **Kernbefund:** *eine* Verstärkung bewegt die
zwei Achsen in **entgegengesetzte Richtungen** — Raumgewinn besser (λ₂↑),
Infight schlechter — die gemessene **Infight-⊥-Raumgewinn-Orthogonalität
(ρ ≈ 0,05)** in einer einzigen Maßnahme, ein **Braess-Paradoxon** des Netzes: der
neue niederohmige Pfad lenkt Fluss in Leitungen, deren Grenzwerte *nicht*
nachgezogen wurden. **Regel: die Abhilfe der Versagensachse anpassen.** Ein
Korridor (λ₂/Cheeger/Kron) behebt **Trennung/Inselbildung**; eine
**Überlast-Kaskade** braucht **Grenzwert-Anpassung/Redispatch**, nicht mehr
Konnektivität — strukturelle Verstärkung **muss mit Grenzwert-Upgrades
ko-entworfen** werden, sonst geht sie nach hinten los. Der Go-Meta-`Regime`-
Klassifikator sagt, *welche* Achse eine Störung belastet.

### 4.5 HHTL residents & the scale-dependent coupling / skalenabhängige Kopplung

**EN.** Split HHTL as **HH (HEEL/HIP, coarse) | TL (TWIG/LEAF, fine)**. HH is
*resident* to **Raumgewinn** (basin λ₂); TL is resident to **infight** (basin
cascade fraction). Measured per leaf basin on the ES core (n=20 basins):
Pearson **+0.53**, Spearman **+0.44**, ICC(2,1) **+0.55** — *positively coupled*.
But globally, across contingencies (§ battery), infight ⊥ Raumgewinn (ρ≈0.05).
**The orthogonality is scale-dependent:** orthogonal at the global/contingency
scale, coupled inside a small basin (a well-connected basin has more lines to
cascade through). This is *why* a single fixed blend is wrong and **per-tier
weighting** is needed.

**DE.** HHTL als **HH (grob) | TL (fein)**; HH = Raumgewinn (Becken-λ₂), TL =
Infight (Becken-Kaskade). Pro Blatt-Becken (ES, n=20): Pearson +0,53, Spearman
+0,44, ICC +0,55 — *positiv gekoppelt*; global aber orthogonal (ρ≈0,05). **Die
Orthogonalität ist skalenabhängig** → deshalb Gewichtung pro Ebene.

### 4.6 Time as mediator, inertia as the clock, and the collapse number

**EN.** The cascade is **hops × time-per-hop**: `rounds` is the hop count; the
clock is set by **inertia** via the swing equation `RoCoF = f₀·ΔP/2H` (low
inertia ⇒ steep RoCoF ⇒ faster trips). **Inertia *mediates* the structural
perturbation (HH/Raumgewinn) → realized cascade (TL/infight)** — more inertia
slows the clock so fewer hops complete before protection/operators arrest it.
The **total timescale fingerprints the mechanism**: the Iberian event collapsed
in **~27 s** ⇒ electromechanical, low-inertia, voltage/frequency regime
(consistent with ENTSO-E), **not** a minutes-scale thermal cascade — *the
27-second window is itself the tell.*

**Per-tier weighting** (`timing::HHTL_WEIGHTS`): `(w_R,w_I)` = (4,1)/(3,2)/(2,3)/
(1,4) for HEEL→LEAF — coarse tiers weight Raumgewinn, fine tiers infight.

**The collapse number (proposed scaling law, CONJECTURE [H]):**

```
        Raumgewinn · spread        time · distance
  Π  =  ─────────────────────  =  ─────────────────
        infight · inertia          infight · inertia
```

The numerator `Raumgewinn · spread ≈ time · distance` (the field perturbation is
a space-time front — how far × how fast it propagates); the denominator is the
local fight damped by inertia. **High Π ⇒ fast, wide spread (blackout-prone);
inertia and infight damp it — an inverse correlation.** This unifies the arc:
Raumgewinn (the field, HH), infight (local collapse, TL), spread (Davis–Kahan /
hop distance), time (the 27 s), and inertia (the clock) in one dimensionless
group. *Honest status: a proposed dimensional law; the next probe is to fit Π
against observed cascade size / the 27 s and report Pearson/Spearman with
Jirak-honest significance before promoting [H]→[G].*

**DE.** Die Kaskade ist **Sprünge × Zeit-pro-Sprung**; die Uhr stellt die
**Trägheit** (Schwinggleichung `RoCoF = f₀·ΔP/2H`). **Trägheit *vermittelt*
Struktur (HH/Raumgewinn) → realisierte Kaskade (TL/Infight).** Die Gesamtzeit ist
der Mechanismus-Fingerabdruck: Iberien kollabierte in **~27 s** ⇒
elektromechanisch, träge-arm, Spannungs/Frequenz — **nicht** thermisch
(Minuten). Gewichtung pro Ebene (4:1→1:4). **Kollaps-Zahl (Skalengesetz,
Vermutung [H]):** `Π = (Raumgewinn·spread)/(Infight·Trägheit) =
(Zeit·Distanz)/(Infight·Trägheit)` — hohes Π ⇒ schnelle weite Ausbreitung;
Trägheit und Infight dämpfen (inverse Korrelation). Nächste Probe: Π gegen
beobachtete Kaskadengröße/27 s fitten, Jirak-signifikant, vor [H]→[G].

### 4.7 Meta-hop cascade: inertia × phase between-level propagation

**EN.** A simplification that makes the four-tier cascade tractable: treat each
HHTL tier as **one meta-hop**, and let tier `i` MODIFY tier `i+1` (L2 is the
second hop after L1). Two quantities cross every tier→tier boundary, on the
workspace's two algebras (the OGAR bipolar-phase pyramid: *sign side =
multiply/XOR, magnitude side = bundle/add*):

```
  magnitude_{i+1} = magnitude_i · gᵢ,    gᵢ = infightᵢ·(1 − Raumgewinnᵢ)   (gain)
  phase_{i+1}     = phaseᵢ · sign(Δλ₂)ᵢ                                    (±1)
  field_k         = Σ_{i≤k} phaseᵢ · magnitudeᵢ                             (bundle)
```

The realized perturbation at a tier is the **bundle (running sum) of signed
contributions**, not a chained product — so phase is the *between-level
interference* channel: aligned phases reinforce (the field grows, the cascade
reaches the leaves), alternating phases cancel (the field self-arrests in the
upper tiers). **Inertia** sets each hop's clock `dtᵢ` via the swing equation
(`per_hop_time`, `H` ramped coarse→fine: synchronous-heavy HEEL → renewable
LEAF), so the cumulative time at the penetration depth is the event wall-clock.

On the real ES core (`meta_hops` example, structural phase = sign of the
tier-to-tier λ₂ change, monotonically rising ⇒ all-`+`):

| tier | Raumgewinn λ₂ | infight | phase | H | signed_amp | field | dt | t |
|---|---|---|---|---|---|---|---|---|
| HEEL | 3.15e-7 | 0.204 | + | 6.0 | +1.000 | +1.000 | 0.68 | 0.68 |
| HIP  | 2.07e-6 | 0.358 | + | 4.5 | +0.478 | +1.478 | 0.56 | 1.24 |
| TWIG | 5.65e-6 | 0.392 | + | 3.0 | +0.344 | +1.822 | 0.44 | 1.68 |
| LEAF | 8.88e-6 | 0.032 | + | 2.0 | +0.130 | +1.951 | 0.36 | 2.04 |

The **front penetration is 3/4** (arriving `|signed_amp|` 1.0 → 0.48 → 0.34 ≥ 0.25,
then 0.13 at the leaf where the gain `g→0` absorbs it). Front reach is **gain-driven
and phase-independent** (`|±x|=x`); what the all-aligned phase buys is a *growing
interference field* (peak `|Σ|=1.95`) — alternating phases would cancel it. The
coarse→fine inertia ramp puts the fast seconds in the leaf tiers (dt 0.68 → 0.36 s),
and the cumulative ~2 s lands in the **electromechanical / low-inertia** regime — the
same mechanism class as the 27 s event (the absolute scale depends on `ΔP`, relay
band, and the true `H`-ramp; this run uses illustrative values). The lesson the model
encodes: a deep cascade needs **passing gains (weak field × strong infight) AND low
leaf-inertia** — break either (more connectivity at a tier, or more synchronous
inertia at the leaves) and the front self-arrests or slows below the protection
window; phase separately decides whether the bundled field reinforces or cancels.

*Honest status: CONJECTURE [H]. The gain law is `meta_cascade`; the phase+inertia
refinement is `meta_cascade_phase`. The structural phase (sign of Δλ₂) and the
inertia ramp are placeholders — calibrating them against an observed multi-tier
cascade (and reporting Pearson/Spearman with Jirak significance) is the [H]→[G]
probe.*

**DE.** Eine Vereinfachung, die die Vier-Ebenen-Kaskade handhabbar macht: jede
HHTL-Ebene ist **ein Meta-Sprung**, Ebene `i` modifiziert Ebene `i+1` (L2 ist der
zweite Sprung nach L1). Über jede Ebenen-Grenze laufen zwei Größen, auf den zwei
Algebren des Workspaces (Vorzeichen = Multiplikation/XOR, Betrag = Bündelung/
Summe): **Betrag** über die Durchlass-Verstärkung `gᵢ = Infightᵢ·(1−Raumgewinnᵢ)`,
**Phase** (±1) über das Vorzeichen der λ₂-Änderung. Das realisierte Feld ist die
**Bündelung (laufende Summe) vorzeichenbehafteter Beiträge** — gleichgerichtete
Phasen verstärken (tiefe Kaskade), alternierende löschen aus (Selbst-Arrest in
den oberen Ebenen). **Trägheit** stellt die Uhr `dtᵢ` (Schwinggleichung,
`H`-Rampe grob→fein). Am realen ES-Kern: **Front-Eindringtiefe 3/4** (ankommendes
`|signed_amp|` 1,0 → 0,48 → 0,34 ≥ 0,25, dann 0,13 am Blatt, wo `g→0` absorbiert).
Die Front-Reichweite ist **verstärkungs-getrieben und phasen-unabhängig** (`|±x|=x`);
die gleichgerichtete Phase erzeugt ein *wachsendes Interferenzfeld* (Spitze
`|Σ|=1,95`), alternierende Phasen würden es auslöschen. Schnelle Sekunden in den
Blatt-Ebenen (dt 0,68 → 0,36 s), kumulativ ~2 s im **elektromechanischen** Regime —
dieselbe Mechanismus-Klasse wie die 27 s. Lehre: eine tiefe Kaskade braucht
**durchlassende Verstärkungen UND niedrige Blatt-Trägheit** — bricht eines von
beiden, arretiert die Front; die Phase entscheidet separat über Feld-Verstärkung
oder -Auslöschung. Status: Vermutung [H]; Phase (Δλ₂)
und Trägheits-Rampe sind Platzhalter, Kalibrierung gegen beobachtete Kaskade ist
die [H]→[G]-Probe.

### 4.8 Validity & reliability of the mediators / Validität & Reliabilität

**EN.** Before any concept earns trust it must be shown — on real data — to be
both **valid** (it measures what it claims) and **reliable** (stable across
independent measurements). The `validate_mediators` example runs the full
psychometric battery (Pearson, Spearman, Cronbach α, ICC(2,1)) over N = 30
stride-sampled N-1 contingencies × 4 independent injection "raters" on the ES
core. Significance is read at the Jirak `n^(p/2−1)` rate (`|ρ|√n`), not IID.

| Block | What it tests | Result (ES core) | Read |
|---|---|---|---|
| **A. Criterion validity** | structural mediator → cascade size | Fiedler sensitivity ρ=**+0.77** (\|ρ\|√n=4.2); Weyl Δλ₂ ρ=−0.27; eff-resistance ρ=+0.02 | **Fiedler edge sensitivity is the valid predictor**; raw Weyl Δλ₂ is *not* (bridges fragment in one step, they don't cascade) |
| **B. Reliability** | cascade ranking across raters | ICC(2,1)=**0.71**, Cronbach α=**0.91**, test-retest ρ=**0.86** | vulnerability ranking is an **injection-independent property of the topology** — a reliable instrument |
| **C. Discriminant** | Raumgewinn vs infight | ρ=**−0.31** under stress (≈0 on the global frame) | the two axes are **distinct but trade off** when stressed (bridge = global-loss/no-cascade ↔ loaded line = cascade/no-fragmentation); not strictly orthogonal here |
| **D. Π consistency** | collapse number vs cascade | ρ=−0.29 | infight sits in Π's denominator and tracks the outcome ⇒ a negative ρ is **expected**; a clean Π test needs an infight proxy independent of the realized cascade (open [H]→[G]) |
| **E. Scale coherence** | {Weyl, Rₑ, Fiedler} as one construct | Cronbach α=0.41 | the three structural mediators are **complementary facets**, not one index — keep them separate |

The headline: **Fiedler edge sensitivity is the valid and reliable mediator**
(criterion ρ=0.77 at 4.2 noise-floor units; the ranking it induces is reproducible
at ICC 0.71 / α 0.91). The discriminant result is the honest nuance — global
(Raumgewinn) and local (infight) collapse are *separate* constructs but couple
*negatively* under load, which is itself the bridge-vs-loaded-line physics. Small
N — these are point estimates to harden with real ESIOS/ENTSO-E load.

> **Attribution note.** The Raumgewinn/infight (anti-)coupling reported in block C
> is an **emergent measurement of this crate**, not a claim of the Bardioc ZPN
> framework. The framework's own relationship between the spectral pillars is the
> **mode-instability modifier `m = Weyl × (1/Fiedler) = Δλ × (1/λ₂)`** — the Weyl
> perturbation amplified by the inverse regional connectivity (Mode 2 = Fiedler).
> That modifier is the validated/used quantity in §4.9 (the rolling floor), not an
> orthogonality assertion.

**DE.** Bevor ein Konzept Vertrauen verdient, muss es an echten Daten **valide**
(misst, was es behauptet) UND **reliabel** (stabil über unabhängige Messungen)
sein. `validate_mediators` fährt die volle psychometrische Batterie (Pearson,
Spearman, Cronbach α, ICC(2,1)) über N = 30 Kontingenzen × 4 Einspeise-„Rater"
am ES-Kern; Signifikanz an der Jirak-Rate `|ρ|√n`. Ergebnis: **(A)** Fiedler-
Kantensensitivität ist der valide Prädiktor (ρ=+0,77; \|ρ\|√n=4,2), rohes Weyl-Δλ₂
**nicht** (Brücken fragmentieren in einem Schritt). **(B)** Das
Verwundbarkeits-Ranking ist eine einspeise-unabhängige Topologie-Eigenschaft —
ICC 0,71, α 0,91, Test-Retest ρ=0,86 (reliables Instrument). **(C)** Raumgewinn
und Infight sind getrennte Konstrukte, koppeln unter Last aber **negativ**
(ρ=−0,31): Brücke = globaler Verlust/keine Kaskade ↔ belastete Leitung = Kaskade/
keine Fragmentierung — nicht strikt orthogonal. **(D)** Π trägt Infight im Nenner,
daher erwartetes negatives ρ; saubere Π-Validität braucht einen vom realisierten
Kaskaden-Lauf unabhängigen Infight-Proxy. **(E)** Die drei Struktur-Mediatoren
sind komplementäre Facetten (α=0,41), kein Einzelindex.

### 4.9 The 4-D rolling floor: L1–L4 as an HDR early-exit cascade

**EN.** The colleague's framework (Bardioc ZPN) views the grid as **eigenmodes** —
Mode 1 Global → Mode 2 Regional (the **Fiedler** vector λ₂) → finer modes — and
prescribes "monitor mode stability for **early warning**", with perturbation
theory as the *multi-scale zoom* (pillar 04) and a fluid-dynamics flow analogy
(pillar 03). We realize that literally as a **4-dimensional rolling floor**: the
L1–L4 HHTL tiers are the resolution strokes of an HDR popcount-stacking cascade
(the ndarray `Cascade` "Belichtungsmesser"), and the metered quantity per tier is
the **mode-instability modifier**

```
  m = Weyl × (1/Fiedler) = Δλ₂ / λ₂
```

— the eigenvalue perturbation a worst single trip induces, amplified by the
inverse regional connectivity (small λ₂ = near-disconnected mode = unstable).
This is the operator's modifier, and it is *not* an orthogonality claim (§4.8 C
note): it is one composite signal per mode.

On the real ES core (`rolling_floor` example; per-tier median over the basins of
that tier; floor = `mu + 2σ`, preheated coarse→fine):

| tier | basins | median λ₂ | median Δλ₂ | median m=Δλ₂/λ₂ | floor `mu+2σ` |
|---|---|---|---|---|---|
| HEEL (L1) | 1 | 3.15e-7 | 2.07e-8 | 0.066 | 0.066 |
| HIP (L2) | 2 | 2.07e-6 | 3.59e-7 | **0.173** | 0.224 |
| TWIG (L3) | 4 | 5.65e-6 | 1.03e-6 | 0.153 | 0.286 |
| LEAF (L4) | 8 | 8.88e-6 | 2.10e-6 | 0.301 | 0.640 |

The modifier rises coarse→fine — the leaf modes carry the largest *relative*
spectral shift. Running the **coarse→fine stacked early-exit**: stack L1 (0.066),
then L2 (cumulative 0.238) — which crosses the preheated HIP floor (0.224) at
**z = +2.29σ → Alarm, EXIT at L2 (early)**. The cascade fires its early warning
at the **regional/Fiedler mode** and skips the finer-mode computation — precisely
"monitor mode stability for early warning", as a confidence-gated cascade. The
preheating (coarse tiers warm the fine floors) and the rolling `mu+kσ` floor are
the Belichtungsmesser's calibration + drift-recalibration, reused unchanged.

*Honest status: CONJECTURE [H]. The σ is from a tiny weakly-dependent tier sample,
so the `mu+2σ` floor is an operating threshold, not a Gaussian-clean CI —
significance is the Jirak `n^(p/2−1)` rate. The early-exit is a real compute
saving (skip finer eigenmodes once a coarse mode alarms), to validate against an
observed staged cascade.*

#### 4.9.1 Is the modifier confirmed? (`modifier_validity` probe)

A direct criterion-validity probe (30 stride-sampled contingencies × 3 raters,
ES core): the modifier `m = Δλ₂ × (1/λ₂_local)` against two outcomes, Spearman ρ
with Jirak `|ρ|√n`:

| predictor | vs cascade size (infight) | vs connectivity-loss (Raumgewinn) |
|---|---|---|
| Weyl Δλ₂ (numerator alone) | −0.28 (√n 1.5) | **+1.00** (√n 5.5) |
| **m = Δλ₂ × (1/λ₂_local)** | −0.28 (√n 1.5) | **+0.99** (√n 5.4) |
| m = Δλ₂ × (1/λ₂_global) | −0.28 | +1.00 |

**Confirmed — with the predicted sign split, and two honest caveats:**

1. **The modifier predicts FRAGMENTATION, not cascade line-count.** Against
   connectivity-loss (the Raumgewinn axis) ρ≈+1.0 at 5.5 noise-floor units; against
   cascade size (infight) it is weakly *negative* (−0.28, below the ~2 floor). This
   is exactly the §4.8 prediction — Weyl Δλ₂ tracks the *global* collapse, bridges
   fragment without cascading many lines. So the modifier is a valid **early-warning
   signal for mode collapse / islanding**, not for how many lines trip.

2. **The +1.0 is partly definitional, and `1/Fiedler` adds no rank signal *here*.**
   Connectivity-loss `1 − λ₂'/λ₂` and Weyl `Δλ₂ = λ₂ − λ₂'` are both monotone in the
   post-trip `λ₂'` on a fixed graph, so ρ≈1.0 is closer to an internal-consistency
   check than independent external validity. And the `(1/λ₂)` amplification is *not*
   yet earning its keep: local- and global-λ₂ variants give identical ρ, and
   dividing by λ₂ slightly *lowered* it (1.00→0.99). The flat single-graph
   contingency frame collapses the `1/Fiedler` benefit — `λ₂_local` barely varies
   against the Δλ₂ signal and the outcome is tied to global λ₂. **The amplification
   should prove its worth in the per-tier / per-basin early-warning ranking** (the
   §4.9 rolling floor, where regional λ₂ genuinely varies coarse→fine) against an
   outcome that is *not* a monotone function of global λ₂ — that is the next probe.

So: the *direction* is confirmed (fragmentation axis, strongly); the `1/Fiedler`
*amplification term* is **not yet independently validated** and needs the per-basin
test where regional connectivity actually differs. CONJECTURE [H], partially
discharged.

### 4.10 Step back: the field as a resilience certificate, not a re-predictor

**EN.** The circularity above has a clean resolution: stop using the field to
re-predict *the same perturbation*, and use it for **system resilience** — a
perturbation-agnostic objective. This rests on a cluster of standard theorems
about measuring a field globally vs as compartments, anchored by a **self-inverse
eigenvalue reference**:

- **The self-inverse reference is the Laplacian pseudoinverse `L⁺` [G].** `L` and
  `L⁺` share eigenvectors with **reciprocal** eigenvalues (`λ_k ↔ 1/λ_k`, the zero
  mode fixed) and `(L⁺)⁺ = L` (an involution). So `1/λ₂` — the modifier's amplifier
  — is *literally* the top eigenvalue of `L⁺`; the natural metric in that frame is
  **effective resistance** `R_ij = (e_i−e_j)ᵀ L⁺ (e_i−e_j) = Σ_{k≥2}(v_k[i]−v_k[j])²/λ_k`.
- **Global vs compartment is governed by three results.** Any partition: **Cauchy
  interlacing** [G] bounds compartment eigenvalues between the global ones. An
  **equitable** partition: the **quotient/divisor-matrix theorem** [G] (Godsil &
  Royle) makes the compartment eigenvalues an *exact subset* of the global spectrum.
  Contraction: **Kron reduction = Schur complement** [G] (Dörfler & Bullo) preserves
  boundary effective resistance exactly. Together: measuring the field by
  compartments is self-consistent with the global field exactly on the equitable
  modes, interlacing-bounded otherwise — the spectral form of renormalization-group
  self-similarity (Bardioc pillar 04, "multi-scale zoom"; fixed points = equitable
  partitions).

**Why this dissolves the circularity:** `L⁺` is the inverse map injection→angle-field,
so its spectral invariants **integrate the response over the whole perturbation
ensemble at once** — you read resilience *once* from the field, never by replaying a
trip. The two perturbation-agnostic certificates:

```
  λ₂  = algebraic connectivity      (worst-case margin)
  Kf  = n · Σ_{k≥2} 1/λ_k = n·trace(L⁺)   (Kirchhoff index = total effective resistance)
```

On the real ES core (`resilience` example, one eigensolve, **no `simulate_outage`**):
global `λ₂ = 3.15e-7`, `Kf = 1.98e9`, 260 connected modes. Per depth-3 Cheeger
compartment, the **weakest is compartment 5** (53 buses, `λ₂ = 1.7e-6`) — the place
the next *unknown* perturbation has the least margin, found with no trip replayed.
The perturbation-agnostic reinforcement (the seam corridor maximizing first-order
`λ₂` gain) — bus 112 — bus 216 — raises `λ₂ +136%` and lowers `Kf −27%`: more
resilient to the next perturbation, read straight off the self-inverse spectrum.

**Honest scope.** `λ₂`/`Kf` are exact spectral invariants [G]; whether margin gain
reduces an *operational* cascade is the **Braess caveat** (§4.4) — raising `λ₂` can
worsen one specific flow cascade, so the new corridor's rating must be co-designed
with the margin. The margin is the perturbation-agnostic certificate; a specific
cascade is the per-perturbation check, deliberately kept separate.

**DE.** Die Zirkularität (§4.9.1) löst sich, wenn man das Feld **nicht** zur
Re-Vorhersage derselben Störung benutzt, sondern für **System-Resilienz** — ein
störungs-agnostisches Ziel. Grundlage: die **selbst-inverse Eigenwert-Referenz**
`L⁺` (gleiche Eigenvektoren, reziproke Eigenwerte `λ↔1/λ`, `(L⁺)⁺=L`) [G]; global vs
kompartimentiert geregelt durch **Cauchy-Interlacing** [G], den
**Quotienten-Satz für äquitable Partitionen** [G] (Kompartiment-Eigenwerte =
exakte Teilmenge) und **Kron-Reduktion/Schur-Komplement** [G] (exakte Erhaltung des
Rand-Widerstands). Da `L⁺` die Umkehrabbildung Einspeisung→Winkelfeld ist,
integrieren seine Invarianten die Antwort über **alle** Störungen — einmal lesen,
nie eine Störung wiederholen. Zertifikate: `λ₂` (Worst-Case-Marge) und
`Kf = n·Σ 1/λ_k` (Kirchhoff-Index). Am ES-Kern: schwächstes Kompartiment #5
(λ₂=1,7e-6); die störungs-agnostische Verstärkung (Bus 112—216) hebt λ₂ um +136%,
senkt Kf um −27%. Grenze: **Braess** (§4.4) — Margen-Gewinn kann eine konkrete
Fluss-Kaskade verschlechtern; die Korridor-Bemessung muss mit der Marge ko-designt
werden. Marge = störungs-agnostisches Zertifikat; konkrete Kaskade = Pro-Störung-Check.

### 4.11 The buffer axis: a confound found, an independent dimension added

**EN.** Two corrections close the loop. **First, a confound.** The modifier's
`1/λ₂` amplifier is the **dominant term of the Kirchhoff index** `Kf = n·Σ 1/λ_k`
(λ₂ smallest ⇒ 1/λ₂ largest), so "Weyl × (1/Fiedler)" was *confounded with the
resistive resilience certificate* — both are conductance. That is precisely why
§4.9.1's local-vs-global λ₂ gave identical ρ and the amplification added nothing:
it was effective resistance measured twice. **Second, the missing dimension.** What
the resistive axis omits is the **buffer** — the transient storage (inertia +
capacitance) that absorbs a *sudden impulse* (a Kugelstoßpendel/Newton's-cradle
strike, a line dumping power in one cycle) **elastically** until the frequency
excursion crosses the protection band, then **yields suddenly** (the **Ketchup
effect**: shear-thinning — nothing, nothing, then collapse). From the swing
equation the per-unit buffer is `Δp_max = 2·H·df_band/f₀` — set by **inertia, not
topology**, so it is **orthogonal to λ₂/Kirchhoff by construction**.

On the ES core (`buffer` example; inertia an illustrative topology-independent
scenario — real per-bus `H` needed to calibrate): the size-normalized buffer/node
is uncorrelated with the resistive axis at the Jirak rate (`Spearman(λ₂,
buffer/node) = −0.45`, `|ρ|√n = 1.3`, **below the ~2 floor** — no significant
coupling), confirming the independent dimension. The Kugelstoßpendel test then
surfaces the failure mode a resistance-only screen misses: **compartment 7 yields
first** (thinnest buffer/node) at `λ₂ = 9.6e-6` — *more connected than 5 of 8
compartments*. Resistively it ranks safe; on the buffer axis it is the collapse
seed. That is the low-inertia (renewable-rich) signature of the 28 Apr 2025 event:
**a grid can be well-meshed and still collapse to an impulse if its buffer is thin.**

Three axes, not one: **resistance/λ₂/Kirchhoff** (steady-state spread, §4.10) ⟂
**buffer/inertia** (transient impulse storage, §4.11), gated by the **Ketchup yield**
threshold (the sharp non-Newtonian collapse, §4.9 rolling floor). Resilience needs
all three; the modifier confound came from collapsing the first two into one.

**DE.** Zwei Korrekturen schließen den Kreis. **Erstens ein Confound:** der
`1/λ₂`-Verstärker ist der **dominante Term des Kirchhoff-Index** `Kf = n·Σ 1/λ_k`
(λ₂ am kleinsten ⇒ 1/λ₂ am größten) — „Weyl × (1/Fiedler)" war also mit dem
*resistiven* Resilienz-Zertifikat verwechselt (beides Leitfähigkeit); daher gab
§4.9.1 für lokal/global identisches ρ. **Zweitens die fehlende Dimension:** der
**Puffer** — die transiente Speicherung (Trägheit + Kapazität), die einen
*plötzlichen Impuls* (Kugelstoßpendel) **elastisch** auffängt, bis die
Frequenzauslenkung das Schutzband überschreitet und dann **schlagartig nachgibt**
(**Ketchup-Effekt**: nichts, nichts, Kollaps). Aus der Schwinggleichung:
`Δp_max = 2·H·df_band/f₀` — von **Trägheit, nicht Topologie** gesetzt, also
**orthogonal zu λ₂/Kirchhoff**. Am ES-Kern: Puffer/Knoten unkorreliert zur
resistiven Achse (`Spearman(λ₂, Puffer/Knoten) = −0,45`, `|ρ|√n = 1,3`, unter der
~2-Schwelle). Der Kugelstoßpendel-Test zeigt den Fehlermodus, den ein rein
resistiver Screen verfehlt: **Kompartiment 7 gibt zuerst nach** (dünnster Puffer)
bei `λ₂ = 9,6e-6` — besser vernetzt als 5 von 8. Resistiv „sicher", auf der
Puffer-Achse der Kollaps-Keim — die Niedrig-Trägheit-Signatur (erneuerbar-reich)
des 28.04.2025: **ein gut vermaschtes Netz kann an einem Impuls kollabieren, wenn
sein Puffer dünn ist.** Drei Achsen, nicht eine: **Widerstand/λ₂/Kirchhoff**
(stationär) ⟂ **Puffer/Trägheit** (transient), getort durch die **Ketchup-Schwelle**.

**DE (§4.9).** Bardioc ZPN sieht das Netz als **Eigenmoden** — Mode 1 Global → Mode 2
Regional (**Fiedler** λ₂) → feinere Moden — und fordert „Modenstabilität für
**Frühwarnung** überwachen", mit Störungstheorie als *Multiskalen-Zoom* (Säule
04) und Fluiddynamik-Analogie (Säule 03). Wir setzen das wörtlich als
**4-dimensionalen rollenden Boden** um: die L1–L4-HHTL-Ebenen sind die
Auflösungs-Stufen einer HDR-Popcount-Stapel-Kaskade (der ndarray-`Cascade`-
„Belichtungsmesser"), gemessen wird pro Ebene der **Moden-Instabilitäts-Modifier**
`m = Weyl × (1/Fiedler) = Δλ₂/λ₂`. Am realen ES-Kern steigt m grob→fein
(0,066 → 0,30); der gestapelte Frühausstieg stapelt L1 (0,066) + L2 (kumuliert
0,238), überschreitet den vorgewärmten HIP-Boden (0,224) bei **z=+2,29σ → Alarm,
AUSSTIEG bei L2 (früh)**: die Kaskade feuert die Frühwarnung an der
regionalen/Fiedler-Mode und überspringt die feinere Moden-Berechnung. Vorwärmen
(grobe Ebenen wärmen die feinen Böden) und der rollende `mu+kσ`-Boden sind die
Belichtungsmesser-Kalibrierung. Status: Vermutung [H]; σ aus kleiner
schwach-abhängiger Stichprobe ⇒ Jirak-Rate, kein sauberes Gauß-CI.

---

### 4.12 Exploration battery — falsifiable probes run to conclusion (`explore`)

**EN.** A spectra-only battery (no cascades, fast, definite pass/fail per probe;
`explore` example) was run to settle the open soundness questions and test
generalization. Findings, graded:

| # | probe | result | grade |
|---|---|---|---|
| A | self-inverse reference (Moore-Penrose) | `‖L·L⁺·L−L‖/‖L‖ = 1.3e-13`, `L⁺·L·L⁺=L⁺` 2.4e-13, reciprocal spectrum λ↔1/λ 3.3e-13, effective resistance two ways agree 1.6e-16 | **[G] verified** |
| B | Cauchy interlacing | holds on every split | **[G]** |
| B | partition equitability | ES cross-degree CV 7–9 (and 1.4–9.3 across PT/IT/GB/FR) — **never near-equitable** | **[G] (universal)** |
| C | bisection stability `(λ₃−λ₂)/λ₂` | ES 3.23, IT 2.38 (well-separated); PT 1.43, GB 1.27 (marginal); **FR 0.37 (ambiguous)** | **[H], grid-specific** |
| D | analytic closed forms | λ₂ & Kf match `K_n`/`C_n`/`P_n` to ~1e-11 | **[G] verified** |
| E | N-2 super-additivity | ES: 13/66 top-pairs super-additive, worst joint/sum **3.55×** | **[H]** |

Three load-bearing conclusions. **(1) The self-inverse reference and the spectral
engine are exact** — §4.10 and the Kirchhoff index are now verified in code, not
just asserted. **(2) The quotient-theorem exactness NEVER holds on real grids**
(equitability CV ≫ 0 on all five national grids), so compartment certificates are
always valid *per-basin* but never reproduce the global spectrum — and **bisection
trustworthiness is grid-specific and DEGRADES on large/low-λ₂ grids** (France's
top cut is ambiguous, `gap/λ₂ = 0.37`: multiple near-degenerate weak modes, so a
single Fiedler partition there is unreliable — a real limit of the compartment
method, surfaced not hidden). **(3) Unsupervised localization works**: run blind on
the Iberian peninsula (ES+PT, 312-bus AC component), the top Cheeger cut isolates a
**100-bus all-Spanish sub-region** — the weakest seam is *inside* Spain, not at the
ES–PT border. **(4) N-2 super-additivity is real**: pairs of lines whose *joint*
algebraic-connectivity loss is up to 3.55× the sum of their singles — the correlated
multi-element contingencies (the 28 Apr 2025 mode) that an N-1 screen structurally
cannot see.

**DE.** Eine reine Spektren-Batterie (`explore`, keine Kaskaden) klärt die offenen
Soundness-Fragen: **(A)** die selbst-inverse Referenz ist exakt (`L·L⁺·L=L` 1,3e-13,
reziprokes Spektrum 3,3e-13) — **[G] verifiziert**; **(B)** Cauchy-Interlacing gilt
immer **[G]**, Äquitabilität ist auf allen fünf Netzen (ES/PT/IT/GB/FR) **nie**
erfüllt (CV 1,4–9,3) ⇒ Kompartiment-Zertifikate gelten *pro Basin*, reproduzieren
nie das globale Spektrum; **(C)** Bisektions-Stabilität ist netz-spezifisch und
**verschlechtert sich** bei großen/λ₂-armen Netzen — **Frankreichs** Top-Schnitt ist
mehrdeutig (`gap/λ₂ = 0,37`); **(D)** analytische Formeln (`K_n`/`C_n`/`P_n`) auf
~1e-11 getroffen — **[G] verifiziert**; unüberwacht angewandt auf die iberische
Halbinsel isoliert der Cheeger-Schnitt eine **100-Knoten rein-spanische Teilregion**
(schwächste Naht *innerhalb* Spaniens, nicht an der ES–PT-Grenze); **(E)**
N-2-Super-Additivität ist real (13/66 Top-Paare, schlimmstes joint/sum **3,55×**) —
die korrelierten Mehrfach-Ausfälle, die ein N-1-Screen nicht sehen kann.

---

### 4.13 Application: the cross-country scorecard & the fail-first investment locator

**EN.** The three axes compose into a decision tool. The `scorecard` example reads
**topology** from the real grid (the only measured axis: λ₂, mean effective
resistance, bisection stability) and combines it with **buffer** (effective inertia
`H_eff` from the generation mix — nuclear/hydro high, wind/solar low) and **policy**
(a feed-in/curtailment/import modifier) as declared operator/literature priors, into
a dimensionless **exposure** index `meanR · policy / buffer`.

**The France paradox is the validation.** France has the *lowest* measured λ₂ of the
panel (3.1e-7 — a huge, sparse grid) yet sits near the *bottom* of the exposure
ranking, because its nuclear **buffer** (`H_eff=6 s`) is the highest. A topology-only
(λ₂/Kirchhoff) screen would wrongly red-flag France; the buffer axis explains why it
holds. Conversely **Spain tops the exposure ranking** — weak topology AND thin buffer
(wind/solar + old infra) AND permissive feed-in — the triple exposure that matched
28 Apr 2025. Norway (hydro) and Germany (pumped-storage + conservative, import-reactive
policy + meticulous forecasting) sit lowest.

**The fail-first investment locator.** For each country the **binding constraint** is
the exposure factor furthest above the panel median; it names both the fail-first
priority and the *product* that fixes it:

| binding axis | intervention (product) | panel result (marginal exposure cut) |
|---|---|---|
| **buffer** | synchronous inertia — **gas turbine / synchronous condenser / pumped storage** | **Spain −50 %**, Britain −40 %, Italy −36 %, Portugal −33 %, Germany −31 % |
| **topology** | transmission corridor (inter-basin reinforcement) | France −20 %, Norway −20 % |
| **policy** | feed-in curtailment + forecast + fast-import / storage | Poland −20 % |

This is the predictive-vulnerability case for an infrastructure sale: the model says
**where** the fail-first asset can't wait (the exposure ranking) and **how much**
resilience it buys (the marginal cut), per asset type. A synchronous-inertia product
(the Siemens gas-turbine / synchronous-condenser story) lands precisely on the
**buffer-bound** countries — and Spain's **−50 %** is the largest single lever in the
panel. Honest scope: only topology is measured; `H_eff`/policy are transparent priors,
so the percentages are *structural* not costed — feed real per-bus inertia + curtailment
data and the same machine emits a costed ROI figure per candidate site.

**DE.** Die drei Achsen ergeben ein Entscheidungswerkzeug (`scorecard`): **Topologie**
gemessen (λ₂, mittlerer Widerstand, Stabilität), **Puffer** (Trägheit `H_eff` aus dem
Erzeugungsmix) und **Politik** (Einspeise-/Curtailment-Modifikator) als deklarierte
Prioren, kombiniert zum **Exposure**-Index `meanR · policy / buffer`. **Das
Frankreich-Paradox validiert die Achsentrennung:** FR hat das *niedrigste* λ₂
(sparsam vernetzt) aber niedrige Exposure — der Nuklear-**Puffer** (`H_eff=6 s`) trägt.
**Spanien** führt die Exposure-Liste an (schwache Topologie × dünner Puffer ×
permissive Einspeisung). **Fail-first-Investitionslokator:** die bindende Achse nennt
Priorität und Produkt — **Puffer** → Synchron-Trägheit (**Gasturbine /
Synchron-Kompensator / Pumpspeicher**: Spanien **−50 %**, GB −40 %, IT −36 %, PT −33 %,
DE −31 %); **Topologie** → Übertragungskorridor (FR/NO −20 %); **Politik** →
Curtailment/Prognose/Import (PL −20 %). Die predictive-vulnerability-Story für einen
Infrastruktur-Verkauf: das Modell sagt **wo** die nicht-wartbare Erstinvestition liegt
und **wieviel** Resilienz sie kauft. Grenze: nur Topologie gemessen; `H_eff`/Politik
sind Prioren ⇒ strukturelle %, keine kostierte ROI — mit echten Trägheits-/Curtailment-
Daten liefert dieselbe Maschine eine kostierte ROI pro Standort.

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
