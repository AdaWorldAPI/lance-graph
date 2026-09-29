# Converge: DeepNSM-v2 lexical-evidence consumer

You hold ideas about DeepNSM-v2 lexical evidence that are not in the current
proposal. Fold them into it; do not start a parallel design.

## Read first

1. `.claude/plans/deepnsm-v2-lexical-evidence-consumer-v1.md` — the proposal.
2. `crates/deepnsm-v2/src/lexical.rs` — what #1299 shipped.
3. `crates/deepnsm-v2/examples/bible_wave.rs` `load_pos` (`:1023`) and
   `coca_pos` (`:980`) — the one consumer and its first-wins loss.
4. `crates/deepnsm-v2/src/fsm.rs` `Pos` / `Tagged` — one `Pos` per token.

## Fixed boundary (do not reopen)

- DeepNSM-v2 preserves evidence; it does not perform the CausalEdge64
  epistemic transition. No dependency on causal-edge,
  cognitive-shader-driver, arm-discovery or the deprecated deepnsm crate.
- Routing `WordId`s and Cam96 `codes[word_id]` do not move.
- Counts are integers; unknown is `None`, never `0`, never `f32`.
- No force push. No model names or attribution in commits or PR text.

## Deliver

For each of your ideas, one entry:

- **Idea** — one sentence.
- **Relation to v1** — replaces / extends / conflicts with which checklist item.
- **Evidence** — file:line you read, or a measurement with its command.
  A search hit is navigation, not evidence.
- **Cost** — what it adds to the PR (types, callers, tests).
- **Verdict** — include in this PR / next PR / drop, with the reason.

Then write `.claude/plans/deepnsm-v2-lexical-evidence-consumer-v2.md`: v1 with
your accepted items merged in and a short section listing what was deferred
or dropped and why. Keep the PR small: one read method, one caller. Anything
that needs an FSM change or a second loader is its own PR unless you show it
cannot be separated.

Answer the two open decisions in v1 if your ideas bear on them:
`lemmas_5k.csv` (drop from the consumer or add a loader) and the selection
rule (highest known count, tie-break).

Do not implement. Commit only the v2 plan and its board entries.
