# Fold execution laws — BIND, BUNDLE, and what a V4 must not know

> **READ BY:** `isa-anti-lasagne-warden`, `fold-carrier-scientist`,
> `cypher-lowering-warden`, and any session proposing a planner/fold/physical
> IR, an R2IL opcode, a carrier type, a JIT/CubeCL/MLIR backend, or a
> prepared-program cache.
> **Extends** `folding-doctrine.md` (map vs city) and
> `three-prefix-fold-carriers.md` (name the carrier); does not restate them.
> **Status:** WORKING-MODEL. Each law names its evidence; OPEN items are
> marked OPEN. Full analysis: `.claude/research/D-V4-FOLD-LAB.md`,
> `D-BIND-BUNDLE-0.md`.

## The four verbs (survived falsification on the live code)

PARSE decides what was said · BIND resolves it to V3 addresses and legal
carriers · V4 (`quack::Query` today) says what to compute · BUNDLE chooses how
to compute it now · EXECUTE touches the bytes. No fifth verb was forced.

## Laws

1. **Demand before lowering.** Bag/set, walk/trail, earlier-variable identity
   and fanout multiplicity are decided in the semantic plan. A backend never
   receives an unresolved multiplicity question. [TEST-PINNED,
   cypher-quack differential]
2. **Layout is a bind fact.** Sort order and run guarantees gate carrier
   LEGALITY, so they belong to BIND's clock, not BUNDLE's. [VERIFIED-IN-CODE,
   `dst_ordered`]
3. **A bundle may not change the answer.** One `Query` → `lower` /
   `lower_fused` / reordered conjuncts / mask-native gated gather / ordinal
   vector gave identical answers in 32/32 cases. [MEASURED, `bundle_probe`]
4. **Survivor gating must reach the expensive op.** `quack::lower` gates
   `Pred` only; `Gather` has no `under`, so the semijoin — the dominant cost —
   always scans everything and conjunct order is inert. A mask-native gated
   gather beats a selection vector at every density below 100 %. The fix is an
   executor gate on an existing op, not a new carrier. [MEASURED]
   Refined by D-GATED-GATHER-0: the gate must work at **bit** granularity
   inside live words (on scattered survivors, word granularity issues 47× the
   foreign loads and takes 11.6× the time at 1 % density), and
   the ungated kernel is itself 3.8× slow from a per-row branch. **The mask is
   the execution schedule.** [MEASURED, `gated_gather_probe`]
5. **No selection vector** (quack matrix R1) **survived its falsifier** on that
   workload. Re-test only with a workload where survivors cannot be gated.
6. **No semantic convenience opcode.** A new V4 op needs a proof that existing
   ops cannot express the computation faithfully and efficiently. IAM lifts
   with none. [VERIFIED-IN-CODE subset]
7. **Factorization is a demand decision.** count(A×B) = count(A)·count(B) is
   legal for count/sum/min/max/group, illegal for count-distinct across factors
   — so density or cost must never select it. [literature + reasoning]
8. **Idempotence licenses sloppy carriers.** Duplicate-tolerant frontiers and
   invalid markers are legal for any/or/min only, never for count/sum.
9. **Anti-lasagne.** A new IR must hold information that cannot live in the
   semantic plan, the binding, V4, the bundle choice or the backend program.
   None proposed so far does.
10. **Subtract from the known universe, in u64 words.** Scheduling is
    extent → skip zero words → dense full words → walk set bits → touch
    payload last. The unit is the **u64 mask word**, not a 16-self cell. The
    u16 view of the same 8 KiB is a free shift, but as a schedule it is 1.6–2.8×
    slower (geomean), and 7× slower on an empty mask (4096 tests vs 1024). 16-cells earn
    a role only in detecting full 16-runs inside a partial word, opt-in. Cost
    tracks live rows for lanes up to 16 B and live cache lines from 32 B up, never N.
    [MEASURED, `aperture16_probe`, D-APERTURE-16-0]
11. **The target side subtracts too: the next frontier IS the schedule.** For
    K propagation, writing the exact next-frontier mask in the same pass and
    clearing `K_next` by walking it beats clearing and scanning the dense array
    by up to ~19× at sparse frontiers. A coarse `[u16; 4096]` target histogram
    pre-pass never pays. [MEASURED, `aperture16_probe a6`]

## Measured state of the candidate V4 (R2IL)

R2IL's table is ISA-shaped but its fold band mirrors mask-risc 1:1, has one
test-only reader that ignores the classid, and u8 immediates. It earns a live
role the day a SECOND reader executes the same bytes. Until then
`quack::Query` is the V4 in practice.

## OPEN

- Two invalidation clocks: neither `Binding` nor `Program` carries a
  generation; staleness is undetectable today.
- Push/pull: no multi-hop executor to switch.
- JITSON removes no dynamic work today; only its cache key is worth harvesting.
- CubeCL as target: viable in model; device buffer copy must be a named
  membrane cost.
