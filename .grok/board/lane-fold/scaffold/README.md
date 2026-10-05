# Best-query scaffold

Code collection for the planner shape. It does not execute. It does not belong in the cognitive layers.

| file | what it is |
|---|---|
| `scaffold/best_query.rs` | The best query as types. Collapse, refusals, four tests. |
| `lane_guard.rs` | The earlier metric and address check. |
| `../../05_query_languages/associative_collapse.md` | The law the scaffold implements. |
| `../CLAUDE_LANE_FOLD_CAPSTONE.md` | What not to reopen. |

Check the scaffold from the repo root:

```text
rustc --edition 2021 --test .grok/board/lane-fold/scaffold/best_query.rs
```

A thousand identical sums lower to one terminal, with 999 collapsed and 1,000 consumers. An average lowers to a sum and a count. A fused plan that claims the skip is refused. A request for rows is refused.

## What to wire, in order

1. Call `Query::collapse` in quack before `lower` returns. Print `collapsed` and `remaining` on the metric line.
2. Keep `Repeat` and `Scale` as distinct request variants. Mixing them multiplies.
3. Feed the surviving terminals to the existing executor. Do not add a second one.
4. Add the tile-zero continue in `execute` after this lands. The scaffold does not pretend to time it.

The K knee is not in this scaffold. It is unmeasured. Do not copy the placeholder 64 out of `lane_guard.rs` into a shipped constant without a probe.
