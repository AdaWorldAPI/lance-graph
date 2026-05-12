# Grammar Landscape (D0)

> 2026-04-28. Status: knowledge anchor for the DeepNSM-as-parser PR.
> Cross-refs: grammar-tiered-routing.md, integration-plan-grammar-crystal-arigraph.md,
> crystal-quantum-blueprints.md, cross-repo-harvest-2026-04-19.md,
> session-capstone-2026-04-18.md, endgame-holographic-agi.md.

## 1. Three Grammar Stacks

| Stack | Crate / package | Key files | LOC |
|---|---|---|---|
| Rust | `lance-graph-contract::grammar` | mod.rs, ticket.rs, context_chain.rs, role_keys.rs, thinking_styles.rs, finnish.rs, tekamolo.rs, wechsel.rs, inference.rs, free_energy.rs | (LOC TBD) |
| Rust | `deepnsm` | parser.rs, encoder.rs, codebook.rs, fingerprint16k.rs, pos.rs, spo.rs, vocabulary.rs | (LOC TBD) |
| Python / TS bridges | `agi-chat::grammar/grammar-awareness.ts` (237 LOC) | mirror of Rust thinking_styles | 237 |

## 2. The Triangle: NSM x Causality x Qualia
NSM = 65 universal primes (Wierzbicka). Causality = SPO with Pearl 2^3 mask. Qualia = 18-D emotional/evaluative signature. Each lens projects orthogonally; bundled, they hydrate a SentenceStructure into SpoWithGrammar.

## 3. TEKAMOLO Slot Schema
TE temporal, KA kausal, MO modal, LO lokal -- German pedagogical mnemonic.
Currently 3 of 6 thematic slots covered. Deferred: beneficiary, goal, source. Future-only: path, purpose, result. NOT a linguistic universal -- see Section 11.

## 4. Markov +/-5 as Context Upgrade
Pre: reasoning unit = sentence. Post: reasoning unit = trajectory carrying +/-5 sentences (Mexican-hat weighted). NARS reasons "this sentence in this flow", not "this sentence". Cross-lingual bundle: bind EN+FI parses of same entity -> Finnish case morphology disambiguates Wechsel-ambiguous English roles for free.

## 5. Case Inventories Per Language

### Finnish (15 cases -- CORRECTION applied)
| Case | Suffix | Native role |
|---|---|---|
| Nominative | -0 | Subject; Total object (plural) |
| Genitive | -n | Possessor; Total object (singular) |
| Partitive | -a/-ae | Partial / negated object |
| Accusative | -t | **PERSONAL PRONOUNS ONLY** (minut, sinut, haenet, meidaet, teidaet, heidaet) |
| Inessive | -ssa/-ssae | "in" -- TEKAMOLO Lokal |
| Elative | -sta/-stae | "from inside" -- TEKAMOLO Lokal/Source |
| Illative | -Vn/-hVn/-seen | "into" -- TEKAMOLO Lokal/Goal |
| Adessive | -lla/-llae | "at/by" -- TEKAMOLO Modal/Lokal |
| Ablative | -lta/-ltae | "from" -- TEKAMOLO Source |
| Allative | -lle | "to" -- TEKAMOLO Goal/Beneficiary |
| Essive | -na/-nae | "as" -- TEKAMOLO Modal (state) |
| Translative | -ksi | "into being" -- TEKAMOLO Modal/Purpose |
| Instructive | -in | "by means of" -- TEKAMOLO Modal |
| Abessive | -tta/-ttae | "without" -- TEKAMOLO Modal (negative) |
| Comitative | -ne- | "with" -- TEKAMOLO Modal/Companion |

NOTE: prior `grammar-tiered-routing.md` mapped Accusative `-n/-t` -> Object generally; that was a Latinate transplant. True Accusative is personal-pronoun-only; nominal total object is Nom (pl) or Gen (sg).

### Russian (6 cases)
| Case | Suffix sg masc/fem/neut | Role |
|---|---|---|
| Nominative | -0 / -a,-ya / -o,-e | Subject |
| Genitive | -a,-ya / -y,-i / -a,-ya | Possessor; negated object; partitive |
| Dative | -u,-yu / -e,-i / -u,-yu | Recipient -- TEKAMOLO Kausal |
| Accusative | =Nom (inan) / =Gen (anim) / -u,-yu / -o,-e | Direct object |
| **Instrumental** | -om,-em / -oy,-ey / -om,-em | Means/agent -- TEKAMOLO Modal |
| Prepositional | -e / -e,-i / -e | Governed by v/na/o -- TEKAMOLO Lokal/Temporal |

### German (4 cases)
Nom (subject) / Gen (possessor) / Dat (indirect object, "mit + Dat" = TEKAMOLO Modal) / Akk (direct object).

### Turkish (agglutinative)
Nom -0 / Gen -in / Dat -e / Acc -i / Loc -de / Abl -den. Suffix order: stem + plural + possessive + case + question. *evlerimizdeydiler* = ev-ler-imiz-de-y-di-ler ("they were at our houses").

### Japanese (particles)
ga (subject) / wo (object) / ni (dative/locative) / de (instrumental/locative) / he (directional) / to (companion) / kara (source) / made (terminus). Particle replaces case morphology.

NOTE: each language uses native terminology. Latinate labels can mislead -- Finnish Accusative != Russian Accusative != German Akkusativ in scope.

## 6. YAML Templates Pipeline (future, NOT in current PR)
Target: 200-500 TEKAMOLO templates per priority language as training pairs for the local 90-99% tier. Out of scope for the current PR.

## 7. Pronoun Classes
**Fixed** (axiomatic features): I/you/he/she/they/proper-names. Feature filter over +/-5 candidates IS the resolution. Cheap, permanent.
**Wechsel** (zero inherent commitment): it/that/this/which/one/singular-they. Need full meta-inference (CF axis x Markov axis x cross-lingual bundle).

Cross-linguistic commitment profile:
| Lang | Morphology | Pronoun features |
|---|---|---|
| English | weak | moderate (he/she/it on 3sg) |
| German | moderate (4) | strong (er/sie/es + case) |
| Russian | heavy (6) | strong (on/ona/ono + full case) |
| Finnish | very heavy (15) | weak (single haen gender-neutral) |
| Japanese | particles | minimal (often dropped) |
| Turkish | agglutinative | weak (single o for he/she/it) |

Finnish: easiest morphology, hardest pronoun-features. Cross-lingual EN+DE+RU+FI bundle = complementary quartet.

## 8. Markov vs Counterfactual Axes -- when each is primary
| Situation | Primary axis |
|---|---|
| Heavy morphology + clear discourse | Neither -- Deduction closes |
| Heavy morphology + weird discourse | Markov |
| Light morphology + clear discourse | Counterfactual |
| Light morphology + weird discourse | Both weak -> FailureTicket |
| Cross-lingual bundle available | Bundle collapses CF |

## 9. The 144 Verb-Role Taxonomy
12 semantic families x 12 tense/aspect/mood variants. Each cell = TEKAMOLO slot prior. 5^5 = 3125 Structured5x5 cells > 144 x ~10 x ~10, so the index space fits.

Families: BECOMES, CAUSES, SUPPORTS, CONTRADICTS, REFINES, GROUNDS, ABSTRACTS, ENABLES, PREVENTS, TRANSFORMS, MIRRORS, DISSOLVES.
Tense/aspect: present, past, future, perfect, continuous, pluperfect, future-perfect, habitual, potential, imperative, subjunctive, gerund.

## 10. Out of Scope This PR
Path 2 (holographic residue), CausalityFlow extension (modal/local/instrument), FP_WORDS=160 migration, Crystal4K persistence (H10), Int4State upper-nibble (H8), Glyph5B wide-container (H9), NER pre-pass for proper nouns, Cockpit Cypher, chess vertical, 200-500 YAML templates per language.

## 11. Caveats -- Templates, Not Universals
NSM 65 primes (Wierzbicka): cited by cogsci, contested in mainstream linguistics; doesn't survive empirical testing on polysynthetic languages.
TEKAMOLO: German pedagogical mnemonic, NOT cross-linguistic universal. Arabic *hal*, Mandarin *ba* don't fit.
Chomskyan UG: Tomasello, Evans & Levinson argue empirical weakness.
144-verb taxonomy: numerologically chosen (12 x 12 from sigma_rosetta), not empirically derived.

These are useful templates for engineering a 90-99% local parser. They are NOT theoretical claims about human language universals. The architecture works because the templates are good enough for the bounded task; we make no claim beyond that.
