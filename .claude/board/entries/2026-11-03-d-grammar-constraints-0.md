# D-GRAMMAR-CONSTRAINTS-0: grammatical constraints as typed relations over UD tokens (2026-11-03)

**Status:** measured, lab only. Probe `crates/grammar-constraints-probe` (own workspace); report `.claude/probes/2026-11-03-d-grammar-constraints/REPORT.md`. No production change, no new ClassId or `F###` word.

## Measured

- **UD German-GSD annotates two different roles identically.** The recipient of *geben* (31) and the experiencer of *gefallen* (18) are both `obl:arg` + `Case=Dat`. `iobj` occurs once in train. UD's German docs say this verbatim: "verbs of giving … have one oblique dative argument". So `Case=Dat → Recipient` is false, and so is `obl:arg+Dat → Recipient`.
- **Each role rule is one mask-risc program over DR-addressed token lanes.** The head hop is read in place with `Pred::EqU32Via` (fk = HEAD).
  - Score against gold: 4/4 correct, 0 false positives, 0 false negatives, 1 abstention (UNKNOWN, not FALSE).
  - The Direction/Purpose ambiguity is kept.
  - 0 bytes allocated on the hot path; about 1.2–1.3 µs for 5 rules over 40 tokens.
  - All 8 negative controls behave as predicted; 9 source disables go red.
- **P1 census:** the constraint shape is frame(head) × slot in German, Latin (PROIEL) and English (EWT), but the slot is language-specific.
  - Latin `do`: recipient `obl:arg`+Dat (407).
  - English `give`: recipient `iobj` (197) or `obl`+*to* (23).
  - English `like`: the experiencer is `nsubj` (191).

## Open (owner)

- **First missing interface:** the DT64 WHEN (`SetExpr`) has no hop leaf, so the recipient and experiencer guards get equal witnesses. Even if the guard were statable, R2IL refuses it (`GuardNotExecutable`; 3VL is also refused).
  - Executing the hop needs `EqU32Via`, whose call sites are EXO-140-unclassified.
- **Where the frame lexicon lives.** The language-specific frame lexicon is unversioned in any contract.
- **Which role vocabulary is canonical.** CE64 orientation has `Core` + 7; deepnsm-v2's `haspelmath` list has 8 and includes Judicantis.
- **Ambiguity cannot live in one CE64 word.** The orientation field holds one role; an alternative set needs masks or Bag candidates plus an explicit selection policy.
- **CG-3's REMOVE/SELECT is ordered and non-monotone** (the last-reading guard). It is effectful DO, not a derived relation. Parked.
