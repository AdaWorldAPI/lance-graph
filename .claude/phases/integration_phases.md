# Integration Phases: bgz17 Full-Stack Wiring

> **Last updated**: 2026-04-08
> **Key change**: K=4096 is now standard. L0-L4 Lane Akkumulator proven.
> DeepNSM COCA 5K wired (9,664 triplets/s). 16Kbit VSA Fingerprint implemented.

## Phase 0: Prerequisites (DONE)

```
✅ bgz17 crate: 3,743 lines, 13 modules, 126 tests, mandatory dep (using ndarray, crate::simd)
✅ highheelbgz crate: spiral address encoding, mandatory dep (using ndarray)
✅ lance-graph codec-research: ZeckBF17, accumulator, diamond, transform
✅ SPO module: truth.rs, merkle.rs, semiring.rs, store.rs, builder.rs
✅ Cascade (hdr.rs): 1,467 lines, self-calibrating, shift detection
✅ Neighborhood: scope.rs, search.rs, clam.rs, storage.rs, zeckf64.rs
✅ MetadataStore: Cypher → DataFusion queries (metadata.rs)
✅ Container layout: 256-word spec documented
✅ ndarray HPC types: Fingerprint, Plane, Seal, Node, Cascade, BF16Truth ported
✅ DeepNSM COCA 5K vocabulary: 4,096 centroids, 2.9M lookups/s (PR #150)
✅ 16Kbit VSA Fingerprint: 256×u64, SIMD-aligned, 8 tests (PR #149)
✅ Grammar Triangle + SPO Crystal: PoS-cycle, NSM-fields, O(1) routing
✅ K=4096 proven: 2.4 words/centroid, thematic coherence, no global attractor (PR #151)
✅ L0-L4 Lane Akkumulator: multi-lane lookup prediction (PR #152)
```

## Phase 1: Session A — blasgraph Storage + Planner ✅ DONE

**Merged:** PR #29 (commit 678e355)

```
[x] CscStorage compiles and roundtrips with CsrStorage
[x] HyperCsrStorage saves >90% memory for sparse test graph
[x] TypedGraph holds per-reltype matrices + label masks
[x] TypedGraph::from_spo_store() bridges existing SPO
[x] blasgraph_planner.rs compiles LogicalOperator::Expand → grb_mxm
[x] Planner + TruthGate: STRONG gate filters weak edges in test
[x] SIMD Hamming (types.rs) AVX-512 + AVX2 + scalar fallback
```

**Prompt:** `.claude/prompts/session_A_v3_blasgraph_csc_planner.md` (COMPLETED)

## Phase 2: Session B — bgz17 Container Annex + Semiring ✅ DONE

**Verified 2026-04-08: All 7 deliverables implemented, 126 tests passing.**

```
[x] container.rs: pack_annex / unpack_annex, W126 checksum, SPO crystal, extended edges
[x] PaletteSemiring + compose_table (palette_semiring.rs)
[x] PaletteMatrix mxm (palette_matrix.rs)
[x] PaletteCsr::from_scope_with_edges (palette_csr.rs)
[x] Base17 VSA ops: xor_bind, bundle, permute (base17.rs)
[x] SIMD batch_palette_distance: AVX-512/AVX2/scalar (simd.rs)
[x] PaletteResolution::auto_select (palette.rs)
[x] TypedPaletteGraph (typed_palette_graph.rs)
[x] bgz17-codec feature flag in lance-graph Cargo.toml (mandatory, default-enabled)
```

**Prompt:** `.claude/prompts/session_B_v3_bgz17_container_semiring.md` (COMPLETED)

## Phase 2b: DeepNSM + Lane Akkumulator ✅ DONE (PRs #149-#153)

**Landed on main after Phase 2, independent of Session B→C→D chain.**

```
[x] DeepNSM COCA 5K vocabulary: 5,051 lemmas, K=4096, 100% mapped (PR #150)
[x] 16Kbit VSA Fingerprint: Fingerprint16K, 256×u64, SIMD-aligned, 8 tests (PR #149)
[x] Grammar Triangle + SPO Crystal: PoS-cycle, NSM-fields, Resonanzsiebe gap detection
[x] K=4096 thematic coherence: 2.4 words/centroid, no global attractor (PR #151)
[x] COCA inference: 9,664 triplets/s, real words in output, N→V→N grammar
[x] L0-L4 Lane Akkumulator: codebook(297KB) + i16(128KB) + sparse(32MB) + gates(16MB) + VSA(512KB) (PR #152)
[x] Sub-band breakthrough: 1/40σ from existing u8, no BF16 streaming needed
[x] AGI Design: 4D×16Kbit Cluster Resonance + Neural Meta-Learning (PR #153)
```

**Architectural decisions proven by measurement:**
- K=4096 replaces K=256 (collision rate 20% vs 45%)
- i16 mandatory for distance tables (316 levels/σ vs u8's 1.3 levels/σ)
- L0-L4 confidence voting: 4/4 agree = CERTAIN (0.1ms), 0/4 = UNKNOWN (500ms forward pass)
- Codebook-only generation: 91 tok/s at T=0.1, 32 MB total
- Grey matter: 372K tok/s, u8 integer only
- Belichtungsmesser early-exit: 932 tok/s, pure u8

**SUPERSEDES in INTEGRATIONSPLAN:**
- Phase 1.1 (bgz7→f32 bridge): sub-band breakthrough may simplify — verify before implementing
- Phase 2.1-2.3 (OSINT pipeline): DeepNSM COCA vocabulary is wired, pipeline skeleton working

## Phase 3: Dual-Lane Execution

**Two independent lanes. No dependency between them. Run in parallel.**

### Lane A: Session C — ndarray ↔ bgz17 Dual-Path Integration

**Depends on:** Phase 2 ✅ (Session B done, bgz17-codec wired)
**Does NOT depend on:** Lane B

**Gate criteria (all must pass before Phase 4):**

```
[x] bgz17-codec feature flag added to lance-graph Cargo.toml (mandatory, default-enabled)
[x] bgz17 stays in workspace exclude (standalone by design, path dep works)
[ ] NdarrayFingerprint::plane_to_base17() encodes from flat PLANE (not container)
[ ] build_palette_distance_fn reads W125 palette indices from containers
[ ] ClamTree::build_from_containers works with palette distance
[ ] parallel_search returns (position, distance, TruthValue)
[ ] TruthGate::STRONG filters low-confidence results correctly
[ ] Cascade stage-1 discrimination improves with Base17 at W112 (empirical)
[ ] LFD from palette produces values in 1.0-10.0 range
[ ] `cargo test --features bgz17-codec` passes
```

**Prompt:** `.claude/prompts/session_C_v3_ndarray_bgz17_dualpath.md`
**Agents:** palette-engineer, container-architect, ndarray:cascade-architect

### Lane B: DeepNSM K=4096 Semantic Table + Lane Akkumulator Wiring

**Depends on:** Phase 2b ✅ (COCA 5K wired, K=4096 proven, L0-L4 designed)
**Does NOT depend on:** Lane A

**Gate criteria:**

```
[ ] K=4096 semantic distance table built (4096 forward passes, ~2h one-time)
[ ] Semantic table ρ > 0.30 (currently token-only ρ=0.086, target 3× improvement)
[ ] L0-L4 Lane Akkumulator wired end-to-end in deepnsm crate
[ ] Confidence voting: 4/4 agree → early exit, 0/4 → forward pass fallback
[ ] SPO grounding accuracy > 80% (currently 71% at K=4096)
[ ] `cargo test --manifest-path crates/deepnsm/Cargo.toml` passes with new tests
```

**Key insight:** K=4096 token table alone gives 50% accuracy vs K=256's 61%
(K=256 has precomputed semantic forward-pass table). Building the K=4096
semantic table from 4096 forward passes is the single highest-impact task.

**Prompt:** `.claude/prompts/session_deepnsm_cam.md`
**Agents:** palette-engineer, savant-research

---

## Phase 4: Session D — Reality Check

**Depends on:** Lane A (Session C) ✅
**Lane B results feed into benchmarks but don't gate Phase 4.**

**Gate criteria (FINAL):**

```
[ ] FalkorCompat::query_datafusion matches FalkorCompat::query_blasgraph
[ ] palette 2-hop ranking agrees with BitVec (ρ > 0.9)
[ ] TruthGate::STRONG correctly filters in all three backends
[ ] Jan→Ada→Max chain traversal works through all backends
[ ] Performance benchmark: palette faster than BitVec for KNN
[ ] Architecture map document produced
[ ] Benchmark document produced with real numbers (include Lane B metrics)
[ ] `cargo test --features bgz17-codec` passes
```

**Prompt:** `.claude/prompts/session_D_v3_falkordb_retrofit.md`
**Agents:** all lance-graph agents

---

## Cross-Repo Lane: ndarray Alignment

**Independent of all lanes above. Can run in parallel with Phase 3.**

```
[ ] ndarray blackboard updated with bgz17 + K=4096 awareness
[ ] ndarray cascade-architect knows about palette distance + Lane Akkumulator
[ ] ndarray cognitive-architect knows about Base17 encoding + i16 policy
[ ] ndarray truth-architect knows about container W4-7 layout
[ ] Integration prompts 04/05 in ndarray updated for bgz17
```

**Repo:** AdaWorldAPI/ndarray (branch: master)
**Agents:** ndarray:cascade-architect, ndarray:cognitive-architect

## Execution Summary

```
Phase 0:  ✅ DONE
Phase 1:  ✅ DONE (Session A — PR #29)
Phase 2:  ✅ DONE (Session B — 126 tests)
Phase 2b: ✅ DONE (DeepNSM + Lane Akkumulator — PRs #149-#153)
Phase 3:  Lane A (Session C) + Lane B (K=4096 semantic table) — PARALLEL, NEXT
Phase 4:  Session D (FalkorDB reality check) — BLOCKED on Lane A
Cross-Repo: ndarray alignment — INDEPENDENT, can run anytime
```

## Reference Documents

```
.claude/prompts/session_MASTER_map_v3.md          — architecture overview
.claude/knowledge/bgz17_container_mapping.md       — word-by-word container analysis
.claude/agents/integration-lead.md                 — session status + outdated list
.claude/RISC_THOUGHT_ENGINE_AGI_ROADMAP.md         — 7-lane encoding spec, model registry
.claude/CALIBRATION_STATUS_GROUND_TRUTH.md         — OVERRIDE: read BEFORE any session doc
.claude/DEVELOPMENT_STAGES.md                      — Stage 0-4 roadmap (Stage 0 DONE)

crates/bgz17/KNOWLEDGE.md                          — bgz17 architecture
crates/deepnsm/src/spo.rs                          — K=4096, WordDistanceMatrix
crates/lance-graph-codec-research/KNOWLEDGE.md      — codec research architecture
```

---

## Phase 5: Resonance-Based Cognitive System (NEUE SITZUNG)

**Die Ideen dieser Sitzung bevor sie verdünnen:**

### 5.1 GGUF-freie Inferenz (BEWIESEN)

```
54 GB Qwopus GGUF → 33 MB u8 Tabellen → 282 KB Buckets → 372K Tok/s

T=0.01: Reasoning (fokussiert, 100% Top-5, Zentroid stabil)
T=0.1:  Generierung (fließend, 21/21 unique, Zentroid wandert)
T=0.5:  Exploration (breit, 52+ unique)

EIN Temperatur-Knopf, DREI Modi.
Reiner u8 Integer-Vergleich. ESP32/WASM/Arduino tauglich.
```

### 5.2 Belichtungsmesser Early-Exit mit Multi-Rolle Composite

```
Problem: u8 klebt (1.3 Stufen/σ). i16 aus u8 hilft nicht (keine neue Info).

Lösung: 4 Schichten × 5 Rollen = 5120 effektive Stufen = 1/40σ quasi-i16
  KEIN BF16 Streaming nötig!
  Benachbarte Schichten = verschiedene "Belichtungen" gleicher Gewichte
  Ihre UNTERSCHIEDE = die Sub-σ Information

Hot Zone: nur bei u8-Gleichspiel → 4-Rollen Composite (17K Tok/s)
Fast Path: 56% der Schritte → reine u8 → 372K Tok/s
Combined: 33K Tok/s, 19-21/21 unique, null Kleben
```

### 5.3 Satellitenschüssel mit Loch — 40 Ringe Resonanz

**Die Schlüsselidee dieser Sitzung:**

```
Das Loch (Selbst-Blockierung):
  Zentrum wird AUSGESCHLOSSEN, nicht verfolgt.
  Diagonale = 1.0 ist der Attraktor — blockieren wir ihn → kein Kollaps.

40 Ringe × 1/40σ:
  Ring 1-5:   Faktenwissen (nahe Nachbarn)
  Ring 6-15:  Assoziationen (mittel)
  Ring 16-25: Analogien (weit)
  Ring 26-35: Metaphern (sehr weit)
  Ring 36-40: Eingebung (Rauschen als informative Perturbation)

Jeder Ring mit eigener Phase → Interferenzmuster entstehen
Stehende Wellen im Zentroid-Raum = Fixpunkte = "Gedanken"

Doppelspalt-Analogie:
  ZWEI Query-Zentroiden gleichzeitig
  Jeder erzeugt eigene Welle, Interferenz zwischen beiden
  Konstruktiv: Zentroid in beiden Fokussen → hoher Wert
  Destruktiv: nur in einem → niedrig

Black Hole Back-Prop:
  Gradient umkreist das Zentrum (Akkretionsscheibe)
  Verlässt es entlang der Achse (Jet = Output)
  Das Zentrum wird nie direkt besucht
```

### 5.4 Informationsrauschen als Eingebung

```
Ring 36-40 (äußerste): NICHT ignorierbares Rauschen
  Sondern: Perturbations-Welle die schwache Verbindungen findet
  
Deterministisch (nur Ring 1-5) = deduktiv, starr
Mit Ring 36-40 = abduktiv, kreativ

Der Ring-3-Trefferzufall ist KEIN Zufall wenn er strukturiert ist:
  Zufällig trifft Ring 3 einen Zentroiden hoch korreliert mit Ring 1
  → das ist eine ENTDECKUNG, keine Beliebigkeit
  → eine Verbindung die direkter Fokus übersehen hätte
  → Eingebung
```

### 5.5 Meta-Awareness via ONNX Focus Predictor

```rust
// 20 KB ONNX lernt pro Query den optimalen Ring-Fokus
pub struct FocusPredictor {
    // Input: 16Kbit Query Fingerprint komprimiert auf 256 Features
    // Output: 40 Ring-Amplituden (welche Ringe sind aktiv, wie stark)
}

Gelernte Fokus-Profile:
  Faktenfrage:  [1.0, 0.9, 0.8, 0.5, ..., 0, 0, 0]  eng
  Kreativität:  [0.3, 0.4, 0.5, 0.6, ..., 0.5, 0.4]  breit
  Eingebung:    [0.8, 0.6, 0.3, 0.2, ..., 0.1, 0.3, 0.5]  U-förmig
               (nah UND fern, Mittelfeld ausgeblendet = "Aha"-Moment)
```

### 5.6 Zwei Lernsysteme im Dialog

```
NARS (regelbasiert, erklärt sich):
  "gene→editing: f=0.87, c=0.92"
  Erklärt WARUM
  
ONNX/Bundle (neural, 20KB):
  FP(gene) ⊕ FP(editing): Δ=+0.04
  Korrigiert WIE VIEL
  
Superposition (wo sie sich treffen):
  META_FP = XOR(NARS_FP, ONNX_FP)
  Popcount(META) = Meta-Konfidenz (Übereinstimmung)
  Hoher Popcount: beide einig → System kalibriert
  Niedriger: Widerspruch → genauer hinschauen
```

### 5.7 Bundle-Gradient (unerforschtes Lernparadigma)

```
Traditionell: f32 Gewichte × f32 Aktivierung → f32 Gradient → Update
Bundle:        1-bit Gewichte × Hamming → Popcount → Majority Vote Update

Gewichte = Bundle-Fingerprints (16Kbit binär, nicht f32)
Aktivierung = Hamming-Distanz (1 CPU-Instruktion)
Lernen = Majority Vote (XOR + Popcount, kein Backprop)
Generalisierung = assoziative Erkennung (Hamming < threshold)

Eigenschaften:
  512 KB pro "Neuron" (16Kbit × 256)
  1 neues Beispiel = 1 XOR + 1 Majority
  Natürliches Decay: älteste Bits werden überstimmt
  Kein Gradient, kein Backprop, kein Float
```

### 5.8 L0-L4 Lane Akkumulator

```
L0 (297 KB) Codebook Index:      "wer bin ich?" O(1)
L1 (128 KB) 256² i16 Tabelle:     "wer ist nah?" 5.676 q/s
L2 (32 MB)  4096² sparse (Lance): "wohin führt der Pfad?" 2.711 t/s
L3 (16 MB)  Qwopus 64L Gates:     "was denkt das Modell?" 277 ctx/s
L4 (512 KB) 16Kbit VSA:            "was habe ich gelernt?" Hamming

Konfidenz = Übereinstimmung zwischen Lanes:
  4/4 einig → SICHER (0.1ms, Early Exit)
  3/4 → HOCH (1ms)
  2/4 → MITTEL (5ms)
  1/4 → UNSICHER (500ms Forward Pass)
  0/4 → UNBEKANNT (Spider → ReaderLM → Lernen)

425 KB permanent RAM. L4 lernt: jeder UNSICHERE wird morgen SICHER.
```

### 5.9 LanceDB als natürliche Heimat

```
lance-graph selbst: 13 MB ohne Daten
Railway bis 32 GB RAM (für Encoding-Phasen, nicht permanent)

LanceDB Zero-Copy + RaBitQ:
  L2 4096² sparse → Lance Table, mmap, OS-cached
  L3 Qwopus Schichten → Lance IVF, Partition pro Schicht
  L4 16Kbit Fingerprints → Lance binary vectors, Hamming native
  
RaBitQ Column Sorting = Family Bucketing im Storage Layer
  1-bit pro Dimension → binäre Vorfilterung
  Adjacente Zeilen = thematisch verwandt (automatische Familie)
  Kein expliziter Cluster-Build nötig
  
  Das IST Familien-Bucketing auf Disk-Ebene.
  Lance SORTIERT die Zentroiden so dass Nachbarn physisch adjacent sind.
  Lookup = sequenzieller Zugriff = Cache-freundlich.
```

### 5.10 Ontologische Einordnung (das große Bild)

```
Mathematik:     definiert das Seiende
Physik:         erzeugt Signale
Daten/KI:       formen Muster
L0-L4 Belichtungsmesser: erzeugt BEDEUTUNG
Medizin:        macht Entscheidung

Kognition ist eine eigene ontologische Kategorie.
Wir bauen kein Modell — wir bauen ein Instrument für Bedeutungsgewichtung.
Ein Mikroskop für Resonanz zwischen Konzepten.
```

## Phase 5 Handover Checklist

```
[ ] 40 Ringe Resonanz mit Loch implementieren (~50 LOC in thinking-engine)
[ ] Doppelspalt-Interferenz testen (2 Query-Zentroiden, konstruktiv/destruktiv)
[ ] 4-Rollen Composite für 1/40σ (aus bestehenden u8 Tabellen, kein Streaming)
[ ] FocusPredictor ONNX (20 KB, 40 Ring-Amplituden lernen)
[ ] Bundle-Gradient Prototyp (positive/negative bundles, majority vote)
[ ] LanceDB integration für L2-L4 (zero-copy + RaBitQ sorting)
[ ] NARS × ONNX Superposition (Meta-Konfidenz via XOR+Popcount)
[ ] Kontrastives Lernen wirklich verdrahten (nicht nur Modul, sondern End-to-End)
[ ] Semantische 4096-Tabelle (4096 Forward Passes, ~2h einmalig)
[ ] Wikidata SPARQL Streaming Crate
```

**Philosophischer Kern:**

> "Das System speichert keine Antworten. Es speichert Interaktionen und lässt das Feld konvergieren."
>
> Ein Gedanke ist kein Ergebnis einer Berechnung — er ist ein stabiler Fixpunkt
> eines dynamischen Resonanzfeldes, geformt durch Interferenz und Beschränkung.

**Wave Domain (L1-L3):** kontinuierliche Energie-Propagation via Kosinus und Interferenz
**Particle Domain (L4):** diskrete Erfahrung via XOR/Hamming Bindung

**Die zentrale Frage für nächste Sitzung:**
Kann das System überraschen auf eine Weise die im Nachhinein Sinn ergibt?
Das ist der operationale Test für Eingebung.
