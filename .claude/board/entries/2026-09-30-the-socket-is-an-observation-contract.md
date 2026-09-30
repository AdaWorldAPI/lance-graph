# 2026-09-30 — The execution socket is an observation contract; mask-RISC instructions sit below it

**Status:** VERIFIED-IN-CODE · OPEN — plan `.claude/plans/deepnsm-v2-cam96-pairwise-v5.md` §11R (D-CPW-15). Revises `2026-09-30-the-execution-socket-is-mask-risc-ir.md` the same day.

## Read (2026-09-30)
- **#1305, as committed** (branch `ccr-2fcc2bd3-8o7m2l`, `cypher-mask-multiplicity-contract-v1`, D-CMM-0..3):
  - a mask is support, not bag: 2-hop `count(*)` is 4 under DataFusion, 3 as a popcount;
  - a forward chain is the exact support of the terminal only: `count(DISTINCT b)` 3 vs 4;
  - DataFusion counts walks: 5 vs trail 4 on a cycle, recorded OPEN;
  - `consumer_semantics()` names five carrier kinds.
- The `Population / Relation / step` shape reported alongside #1305 is in no pushed ref; it was evaluated on its merits.
- **One directed hop is ONE mask-risc `Program` over the edge table:**
  - `Gather` through the src lane and the dst lane, plus edge predicates, gives the edge population;
  - `Any` / `Count` / `ScatterOrU32` / `GroupReduce{Lane}` / `Keep` answer Exists / Count / Support / CountBy / Edges.
  - At k = 1 the five #1305 kinds map onto these one-to-one (a binding IS an edge row).
- **The IR collapses unknown to false** (`Gather` zero fallback, `EqU32Via` no-match, `GroupKey` drop), and `ForeignPlane` is checked by length only.
- **The existing quack DuckDB fixture is a FROM–VIA–TO relation** (`doc` ← `line.doc_id` · `line` · `line.partner_id` → `partner`), and quack's contract imports are `{facet, ordered_lane}` only.

## Consequence
The stable socket is population · relation (a population with two index lanes) · step(observation) → answer or `Insufficient`. MASK · TRANSPORT · REDUCE are lowering vocabulary. Carrier sufficiency, explicit path semantics, unknown ≠ false, and rendering outside computation are frozen as law in the plan. No code.

## Open
- The count lane for k > 1 (#1305 D-CMM-4); trail identity across hops.
- `(space, epoch)` identity and the domain declaration in code (`ISS-MASK-RISC-ZERO-FALLBACK-IS-NOT-EVALUATED-FALSE`).
- Whether "Quack" names the socket.
