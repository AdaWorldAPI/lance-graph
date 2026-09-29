# Converge: DeepNSM-v2 lexical-evidence consumer

You hold ideas about DeepNSM-v2 lexical evidence that are not in the current
proposal. Fold them into it; do not start a parallel design.

## How to read

Use the Read tool on every file you cite. `grep`, `sed`, `head` and `tail`
are not used to inspect source; a search hit only tells you where to read. A
conclusion never rests on a search result.

## Read first

1. `.claude/plans/deepnsm-v2-lexical-evidence-consumer-v1.md`: the proposal,
   including its migration section and rulings F1–F11. Its variant-B decision
   and gates G1–G7 are the baseline; argue against them only with file:line
   evidence or a measurement.
2. The migration documents it builds on:
   - `.claude/plans/deepnsm-v3-convergence-v1.md`
   - `.claude/plans/deepnsm-morton-comma-facet-v1.md`
   - `.claude/board/LATEST_STATE.md:3566` (the inbound-leg ruling) and
     `:3584-3587` (the v2 rebuild)
   - `.claude/board/TECH_DEBT.md:593-620` and `:1344`
3. The tagger history: `git show ec50f07b` and `git show 68955ecb`, and the
   deleted module `git show 68955ecb^:crates/deepnsm-v2/src/lexicon.rs`.
4. `crates/deepnsm-v2/src/lexical.rs`: what #1299 shipped.
5. `crates/deepnsm-v2/examples/bible_wave.rs`: `load_pos` (`:1020-1042`),
   `coca_pos` (`:980-988`), `archaic_pos` (`:993-1008`), and its use
   (`:138-161`).
6. `crates/deepnsm-v2/src/fsm.rs`: `Pos`, `Tagged`, `parse_to_spo`. The FSM takes
   one `Pos` per token.

## Fixed boundary (do not reopen)

- DeepNSM-v2 is the inbound leg. It preserves evidence and does not perform the
  CausalEdge64 epistemic transition. No beliefs or truth values, and no
  dependency on causal-edge, cognitive-shader-driver, arm-discovery or the
  deprecated deepnsm crate.
- Routing `WordId`s and Cam96 `codes[word_id]` do not move.
- Counts are integers; unknown is `None`, never `0`, never `f32`.
- Lemma-first (F9) holds unless the operator overrides it (D-LXC-3).
- Frequency and PoS are classid-header signals, never facet payload (F10).
- No force push. No model names or attribution in commits or PR text. No
  references to private storage in this public repo.

## Deliver

For each of your ideas, one entry:

- **Idea**: one sentence.
- **Relation to v1**: which checklist item it replaces, extends or conflicts with.
- **Evidence**: file:line you read, or a measurement with its command.
- **Cost**: what it adds to the PR (types, callers, tests).
- **Verdict**: include in this PR, next PR, or drop, with the reason.

Then write `.claude/plans/deepnsm-v2-lexical-evidence-consumer-v2.md`: v1 with
your accepted items merged in, and a short section listing what was deferred or
dropped and why. Keep the PR small: the library stays unchanged and one caller
changes. Anything that needs an FSM change or a second loader is its own PR,
unless you show it cannot be separated.

Answer these open decisions if your ideas bear on them:
- D-LXC-3: variant B (lemma table first) or A (counted first, overriding F9).
- The selection rule: highest known count, fixed tie order, no known count → `None`.

Do not implement. Commit only the v2 plan and its board entries.
