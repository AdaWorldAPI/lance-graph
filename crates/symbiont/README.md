# symbiont — the golden image

> **What it is:** one binary that compiles the entire Ada stack together —
> `lance-graph` + `lance =7.0.0` + `lancedb =0.30.0` + the `ndarray` fork +
> the `ractor` fork + `surrealdb` (gated to `kv-lance`) + `OGAR`.
>
> **Why it exists:** a known-good foundation to test the **kanban thinking**
> on, starting with the **perturbation simulation**. Before this, the five
> repos were built apart, not in one graph.
>
> **PROVEN (2026-06-19):** the five forks compile AND link into one 4.2 MB
> binary — `cargo build` exited 0 in 19m18s, 912 packages, zero errors. This
> is a *compile/link* milestone. It says nothing yet about *data flow*: the
> crates are linked with zero runtime edges between them (see the loose-end
> ledger in `INTEGRATION_PLAN.md`). "The substrate carries Spain's grid" is a
> GOAL, not an achievement — the `Grid → NodeRow` bridge does not exist yet.

---

## The win condition (acceptance gate)

The golden image is "done enough to build on" when **all three** hold:

1. **The 16K-node SoA substrate carries every Spanish electricity node.**
   Each node of the Spanish grid (REE / ENTSO-E topology) maps to one
   canonical 4096-bit node (`key(16) | edges(16) | value(480)`), and the
   perturbation cascade runs over the full set.
2. **The cascade is NaN-free.** No `NaN` leaks through the perturbation
   shape, the spectral step, or the scorecard — enforced by a linter guard,
   not by hope. (Foundation already laid: `cascade.rs` preserve-last-finite
   abort, `stats.rs` empty-slice guards.)
3. **`cargo clippy` and `cargo machete` are clean.** No warnings, no unused
   dependencies in the golden-image graph.

Hitting all three would, in the operator's words, be **the biggest goal** —
it would demonstrate the substrate is real (carries a real national grid),
correct (no NaN), and tight (no dead deps). **None of the three is met yet**
— all of §A–§C in `INTEGRATION_PLAN.md` is open. What IS done is the
compile/link milestone (the stack builds as one binary); the data-flow that
makes the substrate *carry* the grid is the work ahead.

---

## The stack (one graph, lockstep pins)

| Repo (fork) | Role in the image | Key pins |
|---|---|---|
| `ndarray` | The Foundation — SIMD, BLAS, Fingerprint, CAM-PQ, Walsh-Hadamard | toolchain 1.95.0 |
| `lance-graph` | The Spine — SoA substrate, codecs, query, perturbation-sim | lance =7.0.0, lancedb =0.30.0, arrow 58, datafusion 53 |
| `ractor` | The Dispatch — bounded-mailbox actors (kanban scheduler) | MSRV 1.64, builds on 1.95 |
| `surrealdb` | The Store — **`kv-lance` only** (no RocksDB/C++, no TiKV/gRPC) | lance =7.0.0, lancedb =0.30.0 |
| `OGAR` | The Ontology — SurrealQL DDL bridge, vocab | edition 2024, rust 1.95 |

**Build milestone (verified 2026-06-19):** 912 packages resolve into one
graph with no unresolvable version constraint (note: two `object_store`
majors coexist, allowed) AND compile+link into `target/debug/symbiont`
(4.2 MB, `cargo build` exit 0, 19m18s). The forks' lockstep pinning (lance 7
/ lancedb 0.30 / arrow 58 / datafusion 53) is what makes the symbiosis hold.

**datafusion stays.** It is pure Rust and is lance's internal scan/execution
engine (`lance-datafusion`) — it cannot leave the image while `lance`/`lancedb`
are present, and there is no reason for it to. The kanban loop is *compute*
(jitson formulas), not *query*; it simply doesn't call the datafusion planner.

---

## The kanban-thinking loop (what the image is a foundation for)

```
Lance Dataset::versions()        (kv-lance backend — surrealdb-core)
      │  new version committed
      ▼
LanceVersionScheduler            (ractor actor, bounded mailbox)
      │  KanbanMove { target: ExecTarget::Jit }     ← backpressure = MessagingErr::Saturated
      ▼
jitson / Cranelift formula       (compute, NOT query → no datafusion planner)
      │  SoA tenant delta
      ▼
MailboxSoaView tenant write      (zero-copy into the Lance-backed 16K-node column)
      │  commits a new Lance version  ──────► loop closes
```

The `ractor` `MessagingErr::Saturated` fix (this session) is the
**backpressure valve** between a fast Lance commit stream and a slower jitson
worker — bounded-mailbox `try_send` returns `Saturated` instead of dropping
or unbounded-buffering.

---

## Layout

```
symbiont/
├── Cargo.toml          path-deps the 5 forks + [patch] OGAR's surrealdb git → local
├── rust-toolchain.toml channel = "1.95.0"
├── src/main.rs         golden-image probe (declaring the deps compiles the graph)
├── README.md           ← this file
├── INSTALLATION.md      how to build it (commands, features, C++, disk)
└── INTEGRATION_PLAN.md  loose ends → the Spain-grid acceptance gate
```

See `INSTALLATION.md` to build and `INTEGRATION_PLAN.md` for the path from
"it compiles" to "it carries Spain's grid, NaN-free, lint-clean."
