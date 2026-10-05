# Immaterial handover

A note, not a paper. The germ is in `BOUND_COMPUTATION.md`. Reading and prompts: `HANDOVER_READING.md`. Falsifiers: `HANDOVER_FALSIFICATION.md`. This states the pipeline rule those PRs must not violate.

Repo: https://github.com/AdaWorldAPI/lance-graph

Proofs, not the root:

- Quack bind: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-quack/src/bind.rs
- Report boundary: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-report/src/boundary.rs
- Report plan, `SourceRef { id, generation }`: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-report/src/plan.rs
- Lane-fold: https://github.com/AdaWorldAPI/lance-graph/blob/main/crates/lance-graph-lane-fold/src/query.rs

## Claim

After bind, a stage hands the next stage an address, not a population. The lane stays in the caller's scope. The terminal is the only observation.

```text
size   bound once, when the destination space is minted
i      ordinal, u16 on the 64k rail
A      resident lane, not on the pipe

handover(i) = Destination { space: A, ordinal: i }
process(d)  = read A[d.ordinal]
```

The format bind fills the address. It does not allocate the value. A loop that builds one object per `i` has already materialized.

## Three pictures, one rule

The route is a polyline. The terrain is not copied into it. A superconductor carries the coordinate with no serialization. The unobserved state is a bound destination plus a law, not an unbound name. Opening the box is a named materialization. `materialized_population_bytes` is that opening.

EXPLAIN reads the bound path stored beside the ref. It does not walk the lane. A requested row list is a sink. If those two are the same operation, every question collapses the pipeline.

## Who holds it

One contract type, borrowed. z8run schedules the ref. OGAR declares the route. The planner composes two laws. Mask-risc reads `A[i]` only when a terminal asks. Quack and Report each fill a destination. Neither owns the type. Four copies of the route object are four boxes, and the terrain got copied into each.

`SourceRef { id, generation }` is the reuse check. A ref past its generation is not handed on.

## What this note does not license

A runtime in each crate. A `BindingProjection`. A `RouteRef` before two binders mint the same destination. An executing merge trait. A pipeline object with properties and a wrapped value. The parameter is `space` plus `ordinal`. The law is metadata. The operation stays with the state.
