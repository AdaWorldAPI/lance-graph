# 2026-10-04 — deepnsm-v2: the school question test as the Pearl 2³ mask; blast radius of the deleted grammar heuristics (D-LXC-18)

**Status:** MEASURED (`UD_CTX=neigh|q|both cargo run --release --example ud_pos_eval …`,
UD r2.15 + Animal Farm silver) and VERIFIED-IN-HISTORY (`git log --diff-filter=D`,
pickaxe `-S` over the unshallowed `origin/main`, 6,042 commits).

Operator (2026-10-04): use the SPO 2³ Pearl ladder for grammar ambiguity —
the school way of resolving word classes by test questions; WordNet's value is
its HHTL DN parent/child address; TEKAMOLO lanes are 4× SPO; find the blast
radius of destroyed grammar heuristics.

## The question test as a position

`fsm::answered_questions` — for every token, the 2³ mask of SPO questions its
clause has answered before it arrives (*who/what?* S, *does what?* P,
*whom/what?* O; stepped with the FSM's own transition table; a stop starts a
new clause). The same eight masks as `nars_engine::ALL_MASKS` /
`SpoDistances::all_projections` (P3b, P6 in `lance-graph-osint`). In
`ud_pos_eval` the left context is tagged by frequency only (never gold) and the
position table can key on the mask (`UD_CTX=q`).

| context | Animal Farm adj/adv decided | German adj/adv all | Animal Farm noun/verb all |
|---|---|---|---|
| neighbours | 82.9 % on 374 | 82.9 % | **91.6 %** |
| **2³ answered questions + next** | **91.8 % on 255** (frequency 78.4 %) | **85.1 %** | 84.5 % |
| both | 82.1 % on 397 | 83.8 % | 90.5 % |

Adjective/adverb follows the question test (an adverb answers *how/when* once
*does what?* is answered; an adjective sits inside an unanswered noun phrase);
noun/verb follows adjacency (determiner→noun, subject→verb). UD English is the
exception for adj/adv (q 87.9 % vs neighbours 91.0 % with COCA). Per-pair /
per-language selection on dev is OPEN.

## Blast radius — deleted grammar heuristics and where they went

| deleted (commit) | named replacement | status of the replacement |
|---|---|---|
| deepnsm-v2 `tekamolo.rs` — German left-corner lanes (`68955ecb`) | planner `insight_coca_read` | LIVE, but **English COCA only**: the German capability was lost, not replaced |
| deepnsm-v2 `lexicon.rs` (`68955ecb`) | taggers inlined in `bible_wave`, `genre_shapes` | LIVE |
| deepnsm-v2 `loci.rs` (`68955ecb`) | deepnsm `spo_anaphora_nibble`, jc `l9_loci_real_text`, planner `probe_antecedent_binder` | LIVE (all three) |
| deepnsm-v2 `toc/hydrate/promote` (`68955ecb`) | planner path writes the TEKAMOLO tenant (`insight_reason_wired`) | LIVE, but reads the **v1** WordNet rail the release's own audit condemns (12.76 % wrong sense, 33.84 % on verbs) |
| deepnsm `ontology_vocab.rs` (`48405aa2`) | none — reverted on purpose: concept ids are identities, CAM-PQ is prohibited for ontologies | constraint stands; WordNet enters only as an exact address (`probe_wordnet_44_activation`, 4⁴ ancestry) |
| planner `data/{coca,wordnet}` generators (`b8de41a2`) | moved to branch `claude/rosetta-codebook-bakes-z30uij` | outputs in releases |
| deepnsm v1 `markov_bundle` TEKAMOLO slices (`0ae9f906`) | restored; `parser.rs` leaves `TekamoloSlots` empty "until D3" | slots exist, nothing fills them |

Pickaxe: `Frageprobe` / `wer oder was` / `QuestionProbe` / `wh_question` — 0
code commits; the school question test was never implemented before this.

## TEKAMOLO = 4 × SPO

`TekamoloFacet` is the G4D3 carving — the same bytes as the L5 SPO triplets —
so each lane can carry its own adverbial triple (Lokal = (carried, into,
hall)) with its own 2³ mask. Not built.

## OPEN

- WordNet ancestry as the lane of a PP's noun (*Wann?* → is_a* time period,
  *Wo?* → is_a* location), via the exact 4⁴ address — testable against UD
  English `obl:tmod`; not run.
- Re-measure the deleted German left-corner reader against `tekamolo_de`.
- Move `insight_reason_wired` to the v2 WordNet rail.
