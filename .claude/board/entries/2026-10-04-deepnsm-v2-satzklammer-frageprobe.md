# 2026-10-04 — deepnsm-v2: the Frageprobe through the Satzklammer; the question mask is CausalMask (D-LXC-20)

**Status:** MEASURED (`UD_RULES=1 UD_DE_INVENTORY=<train-only build> cargo run --release
--example ud_pos_eval -- de_gsd-ud-train.conllu de_gsd-ud-test.conllu`; debug-0)
and TEST-PINNED (`fsm::tests::answered_questions_follow_the_clause`).

Operator (2026-10-04): the 2³ ladder lives in CausalEdge64. Does *schnell*
with a question word mean the manner of an act (*schnell denken*) or a
property (*schnell sein*)? Satzklammer was the other heuristic used.

## The mask has one bit order

`fsm::answered_questions` now returns `causal_edge::pearl::CausalMask`
(S = `0b100`, P = `0b010`, O = `0b001`; the field CausalEdge64 packs at bits
[40:42]). The private `ANSWERED_*` constants used the reverse order and are
gone. deepnsm-v2 gains a path dependency on the zero-dependency `causal-edge`.
The tables only relabel their keys: Animal Farm adj/adv in `q` mode is still
91.8 % on 255 decided tokens.

## The Frageprobe needs the verb, and German puts the answer at the bracket

*Wie denkt er? — schnell* (adverb, Modal) and *Wie ist er? — schnell*
(predicative adjective) share the question word. What separates them is the
verb the question asks about. In German the predicative stands at the right
edge of a copula clause, and the manner adverb stands in the Mittelfeld just
before the right bracket (the participle or infinitive). Three rules, scored
in the D-LXC-19 quorum:

| German rule (test tokens) | fires | right |
|---|---|---|
| after a copula → adj (D-LXC-19) | 17 | 35.3 % |
| copula clause, right edge → adj | 26 | **88.5 %** |
| full-verb clause, before the bracket → adv | 20 | **90.0 %** |
| full-verb clause, clause-final → adv | 15 | 53.3 % (votes no) |

German adj/adv joint quorum: 81.6 → **85.1 %**. On the same tokens, frequency
is 79.3 %; position × frequency was 82.9 % on its own token set. Disable run:
with the first two rules forced silent it falls to 81.9 %. UD English (COCA)
adj/adv 92.0 → 93.0 %. In English the full-verb rules rarely fire, because
English verbs are rarely verb-only in the lexicon.

## OPEN

- No qualia-adjacency signal exists for word senses: the qualia modules
  (`contract::qualia`, `cognitive::grammar::qualia`) are felt-state
  coordinates with keyword heuristics, not a word × head compatibility table.
  The measured stand-in is the verb class the word attaches to. A real
  adjacency (does *schnell* combine with events or with entities) would need a
  head-word table mined from the treebank's `advmod` / `amod` / `xcomp`
  heads, and it is not built.
- `ontology_vocab.rs` (WordNet) was not found in lance-graph, OGAR or ndarray
  on any branch, or by org code search. Its repo is unknown.
