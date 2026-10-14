# Haspelmath 2003: semantic maps as grammar-ABI heuristics

**Type:** source-grounded research notes and engineering hypotheses (NOT a claim of authored ABI, universal verb table, or implemented semantic processor).  
**Paper:** Martin Haspelmath (2003), “The geometry of grammatical meaning: Semantic maps and cross-linguistic comparison,” in M. Tomasello (ed.), *The New Psychology of Language*, vol. 2, pp. 211–242, Lawrence Erlbaum.  
**Public author copy:** https://zenodo.org/records/831410 ; DOI https://doi.org/10.5281/zenodo.831410 .  
**Material examined:** 17-page scanned PDF supplied in the discussion, beginning at the printed pp. 211–212; final page includes transition to following chapter. The paper is primarily about **grammatical functions and multifunctional grammatical morphemes**, not an inventory of 144 verbs or a numerical ISA. Brief descriptions below paraphrase the source, not quotations.

## A. High-signal observations from the source

| Printed pages / figure | What the paper addresses | Useful engineering hypothesis (our inference, not the author's claim) |
|---|---|---|
| pp. 211–213, Fig. 8.1 | English preposition *on* and multifunctionality; English *to* across direction, recipient, experiencer, purpose and related functions | Use **grammatical-function nodes** and witnessed multifunctional forms rather than 1:1 lexical string IDs. |
| pp. 213–216, Fig. 8.2 | Cross-language contrast between the English *to* region and French *à*; connected semantic-functional networks | A language adapter covers a *region* of a shared function graph; overlap is possible without synonym collapse. |
| pp. 216–219, Figs. 8.3–8.5 | Constraints on constructing semantic maps and documenting connected function regions, French dative and English/French mapping | Treat graph connectedness as a **hypothesis and validation heuristic**, not proof of a universal planar coordinate system. |
| pp. 220–223, Figs. 8.6–8.8 | Semantic-map extension to indefinites, differences between English and Russian functions and their contrasts | Ablate mapping choices; keep language-specific edges, contrasts and ambiguity rather than averaging all meanings together. |
| pp. 224–227, Figs. 8.9–8.11 | Reflexive and related functions, differences across French, Russian and Turkish marking | Test whether common graph topology explains multifunctionality better than flat shared labels. |
| pp. 227–231, Figs. 8.12–8.15 | Instrumental and related semantic functions and language-dependent prepositional/case boundaries | A multi-function morpheme is an implementation covering several semantic nodes; don't mistake one surface form for one primitive action. |
| pp. 230–233 | Advantages and potential predictions of semantic maps, cross-linguistic comparison and methodological caveats | Compare held-out language coverage, exceptions and connected-region violations to a random/control graph. |
| pp. 233–237, Figs. 8.16–8.18 | Diachronic trajectories in grammatical functions, source paths and directionality | Explicitly version topology; do not overwrite immutable semantic IDs when mapping drifts. |
| pp. 237–240, Figs. 8.19–8.20 | Further uses, metalinguistic concepts such as *tree* and *number* across languages, methodological conclusion | Semantic maps are evidence-driven **comparison artifacts**. A universal grammar ABI remains a novel and falsifiable engineering extrapolation. |

Notes on page references: “p.” numbers are those printed in the chapter; they do **not** equal the PDF viewer's zero-based page index. The chapter begins with p. 211 on PDF page 1, then continues as facing-page spreads.

## B. Proposed ABI-shaped mappings (OUR design)

Source term / observation -> proposed typed interface:

- Grammatical **function** -> semantic node with immutable symbol ID, mapping-grade, witness.
- Semantic **relation** or map edge -> versioned semantic graph adjacency, with provenance and uncertainty.
- **Multifunctional morpheme** -> language-specific implementation covering a witnessed set of functions.
- **Cross-language overlap** -> partial projection between language adapters; never identity by string match.
- **Function's connected region** -> testable constraint on plausible adapter coverage.
- **Diachronic semantic extension** -> append new mapping versions, preserve old observations and old codebook seals.
- **Gap / unmatched expression** -> unresolved candidate, not an unauthorized default DO.

The ABI would have a **canonical relation graph** first, with a separate **Cartesian projection**. Only the latter is Morton quantized. The semantic map does not claim natural Euclidean distances, and connected graph paths can remain nonlocal in a two-dimensional projection.

## C. What the paper does NOT establish

The following belong to the research program and need independent evidence:

1. A complete 12×12 or 144-element inventory of universally stable predicate meanings.
2. A cyclic “color wheel” topology of predicate families.
3. Numerical 8-, 12-, or 16-bit grammar addresses and Morton refinement.
4. Qualia registers, subjective experience, valence/arousal axes or conscious action gating.
5. Moore8 pressure, Shannon entropy, foveated cognitive dynamics or HHTL identity.
6. Executable grammar ABI / NAL inference / R2IL fold equivalence.
7. Preservation of nonlocal polysemy when forcing all meanings into one 2D lattice.
8. Evidence that the resulting representation is “holographic” in an information-theoretic sense.

Do not cite Haspelmath as evidence for any of these without a separately measured derivation.

## D. Specific research questions extracted for test design

- **SM-Q01:** Are semantic map nodes universal **functions**, language-specific labeled senses, or a mixture? How are granularity disagreements represented?
- **SM-Q02:** Is a cross-language region always connected after controlling annotation and sampling bias? Document counterexamples explicitly.
- **SM-Q03:** Which edges are strongly supported versus inferred only to connect a map? Report edge provenance.
- **SM-Q04:** Does a 12×12 embedding encode semantic closeness better than a random embedding, spectral/graph embedding or a larger unconstrained graph?
- **SM-Q05:** Which semantic distances are lost when a graph is quantized into Morton-connected cells?
- **SM-Q06:** Does an additional Morton nibble discover useful semantic distinctions or simply increase memory?
- **SM-Q07:** What do the current 12 family × 12 role/TAM entries actually mean in the repo? Are these the same 144 predicate senses, or two independent projections?
- **SM-Q08:** Can multiple languages extend the shared map without rewriting old semantic IDs or confusing homographs?
- **SM-Q09:** Can uncertainty be preserved all the way to ClassView -> Quack/R2IL -> OGAR DO gate?
- **SM-Q10:** Do claimed qualia-masked DO improvements persist under stable policy, goal and evidence budgets?
- **SM-Q11:** Can partial-language evidence reconstruct a larger semantic neighborhood in a way exceeding nearest-neighbor, proving a useful *distributed* property?
- **SM-Q12:** Do the proposed mappings preserve meaning in adversarial counterfactuals and prevent ungrounded causal certification?

## E. Adversarial examples to collect from the primary source

1. English *to* vs French *à*: distinguish recipient/direction/purpose/experiencer with construction context; keep overlapping but nonidentical regions.
2. Indefinites in English and Russian: same broad topic with different cuts through specific, non-specific, free-choice, negative and polarity environments.
3. Reflexive, reciprocal and middle readings: do not mistake nearby functions for equivalence.
4. Comitative/instrument/source/cause: compare languages whose markers cover different function regions.
5. Diachronic maps: check that a predicted historically plausible extension doesn't authorize a present-tense equivalence.

For all cases preserve form, construction, language, grammatical function, annotated source passage, confidence, map version, and known exception. No invented example counts, coverage numbers or inference accuracy.

## F. Connection to existing lance-graph work

- Research RFC: docs/research/semantic-morton-qualia-masked-do-rfc.md
- CE64 layout versus working diagram: docs/architecture/ce64-semantic-upper-half.md
- #1496: CE64 and NAL hypotheses, multiple readings and preserving binary contracts.
- #1497: OpenCog/PLN vs Shannon/EIG/Moore attention controls.
- #1498: Hyperon/MeTTa/typed folding proof obligations.
- #1500: Rubikon action/volition stage boundaries.
- #1501: class-method-schema HotPlug and OGAR-only ActionDef authority.

**Research quality bar:** each implementation claim must come from exact current source code and an executable test, rather than from this bibliography or metaphors. Record implementation drift and map conflicts as unresolved, not quietly corrected.
