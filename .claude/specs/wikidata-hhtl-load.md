# Wikidata Load — Maximal-Efficiency HHTL/CAM Pipeline (v0.1)

> Companion to `cognitive-risc-core.md` (v0.1) + `cognitive-risc-classes.md` (v0.2).
> This = how Wikidata lands in ~38GB instead of ~120GB by being ARCHITECTURE-driven, not compression-driven.
> Measured anchor (10 real entities): 1.43MB multilingual → 10.3KB thin+shapes = **139x** before columnar/ID-encoding.

## Principle

Don't compress Wikidata. **Don't load most of it.** Load the skeleton + basins + CAM-deduped shapes + thin rows; stream values lazy-per-basin. The reduction is structural (you store classes+masks+refs, not 115M fat JSON blobs), not a gzip trick.

## The layer model (this is the whole thing)

```
ACHSEN (deklariert, roh) — frozen identity
  Abstammungs-HHTL    subClassOf(P279)-Pfad, 16^n Nibbles      ← the ONE tree axis
  Facetten-bitmasks   geschlossene ObjectProperties             ← closed small vocab → bits
  Quartett-Werte      DatatypeProperties                        ← value tuple, position-coded
ORTHOGONAL (abgeleitet, indiziert) — re-materializable
  Reasoning-Datensatz DL-Schlüsse, einmal materialisiert, CAM-indiziert
    - transitive Hüllen (subClassOf*, partOf*)
    - inferierte Klassenzugehörigkeiten
    - Disjunktheits-Verletzungen / Konsistenz-Flags
```

Everything in the SAME SPO/Lance substrate, separated by `provenance` tier (v0.1 discovery_origin: Curated/Extracted/Derived). Raw axes = declared; reasoning store = Derived. Every edge knows if it was declared or inferred — required for GoBD/audit.

## Two-pass streaming (constant memory, never materialize)

`latest-all.json.gz` = one entity per line. Stream through gzip, parse one entity, write, forget. Constant RAM regardless of dump size; never decompress to disk.

**Pass 1 — Skeleton (structural, cheap):** per entity extract ONLY P31 (instance-of), P279 (subClassOf), en/de labels, property-id SET present. Output: the P279 DAG (= the gifted parent-pointer) + P31 classification + property registry. ~2-3M classes, ~12k properties → 1-2GB, fits RAM. Here: cut basins (HHTL levels), identify capability/facet compartments from OWL/DOLCE template.

**Pass 2 — Bucket + AST + CAM:** second stream. Per entity: class via P31 → basin via P279* reachability (precomputed pass 1) → route to HHTL bucket. Claims → AST nodes referencing the basin codebook (thin, shared). Shape = (class-set, canonical property-set) → BLAKE2b-128 → CAM. Identical shapes dedup. Entity persisted as (class_id, shape_hash, presence_bitmask, value_tuple, en, de).

## The four reduction levers (these get 120→38, not gzip)

1. **Single/few languages.** en + de as TWO separate columns, one parser (language = value not structure; both are projections of the same CAM hash). The 300+ langs are most of the 120GB — dropping them is the biggest single win.
2. **Drop references/qualifiers** (the statement provenance bloat) for the structural load.
3. **ID-encoding.** QIDs/PIDs as u32, never strings ("IDs statt Strings"). Position-coding inside a dense deck eliminates per-value prop_ids entirely.
4. **CAM shape-dedup + basin sharding.** Thousands of "human" instances share one shape; basin shards are homogeneous → compress hard.

## HHTL = the cheap bucket router (16^n)

- Fixed fan-out 16 per level → bucket path = nibble sequence → routing is bit-shift, not hash lookup. O(1) arithmetic ("super billig").
- **The mask inherits along the HHTL path as DELTAS.** A leaf deck stores only its increment over the parent path. Common wide fields live HIGH (once); specific fields live LOW (leaf). This is what prevents the sparse-union disease — decks stay dense because shared columns are inherited, not repeated.
- **ONE tree axis only (Abstammung).** Multi-parent (flying-family) is NOT a second tree branch — it's an orthogonal facet-bitmask. Bat = mammal-path + flight-bit, not two paths. Keeps 16^n a clean tree (cheap nibble addressing) AND keeps multi-parent dedup (verb "fly" stored once in the capability compartment, bit points at it).
- **Open question (the one untested assumption): P279 fan-out is wildly uneven** (some classes 2 children, some 4000). Whether it re-balances onto 16^n or forces adaptive fan-out (4^n here, 16^n there) is MEASURABLE — measure fan-out distribution on a real P279 subtree before fixing the base.

## Facets: OWL/DOLCE as the template (the brutal shortcut)

Don't guess the axes — **harvest them from OWL.** OWL declares the facet-vs-path distinction you'd otherwise measure:

| OWL construct | HHTL form |
|---|---|
| rdfs:subClassOf (transitive) | Abstammungs-path (nibbles, 16^n) |
| owl:partOf / transitive props | further path axes (inherited as delta) |
| ObjectProperty, small closed range | facet-bitmask (1 bit per range individual) |
| DatatypeProperty | Quartett slot (value-tuple position) |
| owl:oneOf / enumeration | closed vocab = exact bit-budget |
| **owl:disjointWith** | **disjoint facets = collision-free, purely additive** |
| owl:Restriction (someValuesFrom) | presence-bitmask rule (which bit must be set) |

- **disjointWith auto-solves the multi-parent conflict question.** Where OWL declares disjoint → facet bits never collide, no linearization needed. Where it doesn't (penguin: fly+swim) → exactly there you need the conflict rule. The overlap set = the non-disjoint property pairs, enumerable because declared.
- **Closed range → bitmask; open/no range → path or ref.** The template IS the decision; no empirical cardinality measurement needed.
- **DOLCE as axis skeleton (clean top facets: Object/Process/Quality/Region ≈ your Object/Organic/Properties/Shape), Wikidata properties as the fill (dirty but real leaf vocab).** DOLCE defines WHICH axes; Wikidata fills WHAT occurs in each. = the DOLCE→cross-domain→industry distillation, as HHTL axis template.

## Facet bit-budget discipline (the ISA-width trap, again)

- Closed small vocab → fixed bitmask (Habitat ~dozen → 5 bits w/ growth reserve; capabilities ~40 verbs → 6 bits). Five real facets together ≈ one u64, AND-testable in one cycle → SIMD batch-AND over the SoA facet column (the cognitive-shader-driver grid run).
- Open/unbounded vocab → NOT bitmask: `Properties` = the Quartett mask (inherits as path-delta); `ElementOf` = a ref-set (unbounded → indirection, not bits).
- **Rule:** fits permanently in 16/64 bits → bitmask. Grows unbounded → path/ref. Once facet bit-allocation is in the LE/HHTL header it's frozen: append-only, never renumber.

## Reasoning as orthogonal indexed dataset (NOT thrown away, NOT runtime)

Pre-materialize DL inferences ONCE, hash them (CAM), index them. "What follows from X" = exact-match lookup, not a reasoner run. The Derived tier.

- **CAM applied to inferences:** an inference is a derived triple `(bat, subClassOf*, vertebrate)`; materialize once, hash, index. Reasoning gets its own CAM layer.
- **Orthogonal = beside, not mixed in.** Raw axes = declared (frozen). Reasoning store = derived (separate index). Card = path + facets + values + ref-into-reasoning-layer. Declared and derived never merge — provenance preserved (GoBD).
- **Re-materializable without touching raw.** Ontology changes (new axioms/disjointness) → recompute ONLY the reasoning layer, raw axes stay. = F1's frozen-identity-under-live-resolution, for reasoning: raw frozen, reasoning re-resolvable, separately indexed.
- **Index the derived triples 3 ways:** by Subject ("all superclasses of X" / transitive closure up), by Object ("all entities implied to be Y" / down), by generating-axiom (provenance / consistency). Classic SPO store over derived triples, same Lance substrate, `provenance=Derived`.

## Open / to measure (last untested numbers)

- **Mask density per deck** (fraction of deck columns set across cards): dense deck = good Quartett = max reduction; sparse = cut basin deeper. This is THE optimization metric.
- **CAM dedup rate** (how many of N share a shape) — drives the thin-row fraction; not measurable on 10, needs a homogeneous cluster pull (~200 instances of one class).
- **P279 fan-out distribution** — does 16^n hold or force adaptive fan-out.
- **Value stream size** (the irreducible SPO-object edges that don't fold into a deck slot) — the entropy wall; position-coding shrinks it but doesn't eliminate it.

## Scaling math (115M entities, structural, en+de)
- thin rows + labels: 115M × 136B ≈ 15.6GB JSON → columnar+u32+RLE → ~5-8GB
- shape table: <0.5GB (amortizes — bounded by distinct shapes, not entities)
- statement values (separate columnar store): ~8-15GB (most compressible: sort by prop_id, dict, RLE)
- reasoning store (derived): sized by inference fan-out, separately indexed
- **Total ~15-25GB → fits 38GB with headroom** for extra languages / qualifiers / eager values.
- **Default lean:** skeleton+shapes+labels EAGER (~6GB hot), values+extra-langs LAZY-per-basin from Lance cold store. CAM hash is the key joining both. = two-clock pattern applied to the load.

---
*v0.1. The architecture is closed; this is its application to Wikidata. Numbers above the scaling section are measured/structural; scaling is projected — confirm with the mask-density + dedup-rate cluster run before trusting 15-25GB.*
