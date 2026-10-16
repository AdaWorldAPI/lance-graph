# Haspelmath semantic-map → DeepNSM-v2 → CE64 → hydrate.rs
**Research/executable harness · 15 October 2026 · not a production writer**

## Thesis

The substrate already has stable word identities, a multi-reading grammatical FSM, and
verse addressing. Use Martin **Haspelmath** (not "Haspelmann") as a falsifiable
**linguistic-function continuity constraint**, not as a new grammar parser, a semantic
distance function, or an instruction set. The resulting pipeline is:

```
existing lexical source (COCA / UD / WordNet / language-aware morphology)
  -> bind_tokens: (surface, PosSet) -> Reading (u16 WordId; OOV refused)
  -> parse_readings: certain / alternative / ambiguity / overflow
  -> [separate attested grammatical-form function masks]
  -> hydrate(markers, per_verse_certain, basin): SPO + original verse path + order
  -> CausalEdge64 syntactic operand view (8:8:8 fixed register/ISA contract)
     + ONE externally pinned, immutable epoch-wide operand dictionary
     + version-qualified canonical meaning for all readers
  -> downstream authority/provenance/temporal writer: NOT IMPLEMENTED
```

This executable probe is in `crates/deepnsm-v2/src/haspelmath.rs`.

**ABI correction (15 Oct):** The original draft mistakenly introduced a
per-verse palette. That would make the same CE64 operand byte mean different
things in two verses, defeating the intention of a globally interoperable
8:8:8 register contract. The implementation now requires a single frozen,
explicit, version-identified operand dictionary for the execution epoch.
This is a test fixture only until the canonical ISA/codebook registry supplies
and verifies that dictionary across independent clients. BF16 and TCP/IP are
*interoperability analogies*, not claims that a numerical format alone grounds
semantic truth or that every NAL/Hyperon operator is equivalent.
It changes no existing grammar decision, hydration behavior, CE64 bit definition,
or Lance/OGAR writer.

## Primary source, not paraphrased into a stronger claim

Martin Haspelmath, *The Geometry of Grammatical Meaning: Semantic Maps
and Cross-Linguistic Comparison*, chapter 8 in *The New Psychology of Language,
Volume 2* (2003), user-provided scan `SemMaps.pdf`, printed pp. 211–242.

- **pp. 212–213 and Fig. 8.1**: "functions" is deliberately neutral between
  different senses and context-dependent uses. English *to* spans direction,
  recipient, experiencer, and purpose. The other dative functions in Fig. 8.1
  are not necessarily uses of English *to*.
- **p. 216, Fig. 8.3**: an edge indicates adjacency. Its drawing orientation
  and physical length have no linguistic significance. Do NOT use Euclidean,
  Morton or word-ID distance as proof of semantic proximity.
- **pp. 216–217**: candidate maps are induced from contrasts between
  languages. A multifunctional gram is **hypothesized** to occupy a contiguous
  region. The arrangement is not mechanically uniquely determined by one
  language; independent cross-linguistic observations constrain it.
- **pp. 222–223, Figs. 8.6–8.8**: indefinite-pronoun distributions are a second
  family of semantic functions, not free additional dative nodes. Different
  languages carve distinct connected subsets.
- **pp. 230–231**: linking neighboring maps into a larger map is possible, but
  the useful extent of any map depends on the question; the method does not
  require deciding whether each polyfunction is polysemy or vagueness.

**Interpretation discipline:** a disconnected, genuinely *complete* attested
function set falsifies this particular proposed map. A disconnected *partial*
sample is unresolved, not a grammar error. A connected set is compatible with
the proposed map, not proof of meaning, truth, or actual referential correctness.

The example graph is exactly the **eight Fig. 8.1 dative-function nodes**,
with the illustrated undirected connections. Its bit positions are strictly
local to this research example, not new classid, CE64 or ABI assignments.

## What is proven by code vs still missing

| Existing code, consumed | Harness addition | Still outside scope |
|---|---|---|
| `fsm::Reading / PosSet / parse_readings` | Existing lexical alternatives remain alternatives | German case, full morphological agreement, nested relatives |
| `spo::Spo` with **u16** IDs | Fixed epoch-wide dictionary (max 256 operands) for CE64's **u8** slots | Production canonical registry, full-vocabulary indirection |
| `hydrate::hydrate` | Counts/addresses verified; unaddressed triples refused | Index-only hydration replacing the TOC, full throughput claim |
| `CausalEdge64::ZERO` and S/P/O setters | Grammar-only low 24 bits, *high 40 zero* | Evidence/root persistence and NAL confidence calibration |
| `tekamolo::read_clause`, `loci`, `wave` | Explicit future join points | TEKAMOLO x Haspelmath adjudication and 24xi4 anaphora |
| H12 (#1551) semantic-map preregistration | Executable linguistic falsifier, not H12 replacement | Inference-function cross-system H12 comparisons |

**Do not conflate:**
1. English *to* vs French *à* is a comparison of multifunctionality,
   not a mathematical mapping from the lexical word "to" to its WordId neighbors.
2. CE64's three **u8 operand lanes are a versioned ALU/ISA register contract**,
   not u16 `Spo::pack` and NOT per-verse tokens. A stable, externally pinned
   operand dictionary at the ingest boundary is essential to make codes mean
   the same thing for every consumer of the same sealed epoch. Never infer
   semantic identity from coincidental byte equality between unaligned versions.
   This harness's tiny dictionary is a compatibility fixture, not a new global
   codebook authority or a solution for all 65,536 WordIds.
3. Syntax does not deserve f=1, c=1, a Pearl intervention mark, or
   `EpistemicState5::Causes`. Bare grammar output has no independent
   witness-based evidence roots. This probe leaves the CE64 header unset.
4. `form_evidence.evidence_root` is an external provenance identifier to
   track, not an independent observation count; never revise confidence twice
   just because two thought layers reference the same root.
5. `hydrate.rs` addresses the already parsed triple stream through a TOC.
   It does not derive SPO triples or parse morphology.

## Reproducible tests / falsifiers

```bash
cargo test --manifest-path crates/deepnsm-v2/Cargo.toml haspelmath
cargo test --manifest-path crates/deepnsm-v2/Cargo.toml --all-targets
cargo clippy --manifest-path crates/deepnsm-v2/Cargo.toml --all-targets -- -D warnings
cargo fmt --manifest-path crates/deepnsm-v2/Cargo.toml --check
```

| Gate | Test | Kills |
|---|---|---|
| HSM-0 | Fig. 8.1 English *to*, German *zu*: connected | Wrong adjacency topology |
| HSM-1 | Purpose+judicantis: disjoint | A vacuous "everything is adjacent" map |
| HSM-2 | Incomplete function evidence stays `Incomplete` | Mistaking missing linguistic data for contradiction |
| HSM-3 | Single verse FSM → hydrate → CE64 → u16 roundtrip (IDs >255) using ONE frozen ABI | Silent narrowing and ephemeral relabeling |
| HSM-4 | Parser alternatives preserved separately from certain hydrated givens | Silent arbitrary disambiguation |
| HSM-5 | Attempt >256 globally admitted operand codes, or an unknown word | Silent overflow, implicit remap or aliasing |
| HSM-6 | Mismatched per-verse markers refused | Cross-verse identity drift |
| HSM-7 | Missing evidence source refused | Unsupported evidence annotation |
| HSM-8 | CE64 upper bits remain zero | Fabricated NARS/Pearl/epistemic certainty |

The code tests are *authored*, not reported as locally run. CI/local Cargo
verification is required before enabling consumers or calling the gates green.

## Next rung (separate approved PRs)

**Wave A: Annotated token-to-function admission.** Read UD case/morphology,
function evidence and cross-language correspondences as externally identified
witnesses. Compare English *to*, German *zu*, French *à* on Haspelmath's
recorded examples. The grammar parser must not treat these as universal
decision rules. Study a counterexample and require a FAIL/REVISION path.

**Wave B: Grammatical Sudoku.** Cross-join the admissible function mask with
SPO/TEKAMOLO, relative-clause argument roles, and 24xi4 local antecedent
displacements as *separate typed readings* of the immutable ABI.
Preserve multiplicity, lexical unknowns, scope, ambiguity and signed local
pointer overflow. Unbound references escalate instead of being guessed.

**Wave C: Hydrated CE64 evidence.** Add a genuinely authorized writer whose
shared-ABI scope is `(classid, sealed version, register codebook/version)`,
*not* one codebook per verse. Preserve stable WordIds separately from the
agreed operand ABI, grammar receipts, original evidence roots, source timestamps,
and read authorization. The ABI must be identical for all readers of that
version, with incompatible dictionaries rejected. A full-vocabulary strategy
requires explicit, global indirection or a wider addressing contract; do not
silently redefine the 8:8:8 physical register. Require CE64 high-bit interpretations to follow the
current global header contract; no private reassignments.

**Wave D: Measurement.** Pin the KJV + Luther corpus snapshots, report
tokenization/PoS/grammar/hydration/CE64 packing separately, with cold/warm
batches, peak allocation and accuracy against held-out UD gold. A 64K
addressable corpus and fast hydration are *not* proof that grammar or
counterfactual inference was solved in the same time. Compare exact candidate
sets to the existing parser oracle before claiming a speedup.

## Kill / stop conditions

- A complete cross-linguistic attestation disconnected on a frozen map:
  **map falsified**, not an invitation to modify oracle data to fit.
- Missing canonical register operand or frozen ABI overflow: **refuse CE64 projection**.
- Any altered `parse_readings` certain/alternative accounting: **STOP**.
- Inference from grammatical contiguity to causality or truth confidence:
  **STOP**.
- Source-root duplication elevating confidence, stale temporal windows, or
  unauthorized view composition: **STOP**.
- A build lacking the relevant production writer: **do not call it wired**.
- A per-verse or per-thought register meaning remap: **STOP**, violates the
  interoperability goal (the BF16 / TCP-IP analogy).

Related plans: D-HOLO-H12 in #1551, D-ORACLE-0 in #1549,
D-HL-5 in #1564, D-MTP-0 in #1555, the DeepNSM-v2 D-LXC experiments
in #1321, and the source-of-truth register/semantic distinction.
