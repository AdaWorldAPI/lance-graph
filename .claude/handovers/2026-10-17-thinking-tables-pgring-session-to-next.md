# Handover 2026-10-17 — thinking-tables wiring, PG-Ring catalogue

Written so the next session starts with the state, the open decisions and the
standing rules in the repo, not in a container that may be gone.

## 1. Where things stand

| item | state | where |
|---|---|---|
| Thinking-tables wiring (64 thought tables × set AST × presence folds × FIFO strata × metacognition) | done, in PR | lance-graph #1599, branch `ccr-9f3b8075-74u47l-thinking-tables-wiring`; `.claude/knowledge/thinking-tables-wiring.md`, entry `2026-10-17-thinking-tables-wiring.md` |
| Epistemic frontier capstone | merged | #1597 (`dd00245b`); `.claude/knowledge/epistemic-frontier-capstone.md` |
| Haiku cards: read in full, output capped | merged | #1598 |
| Board arc entries #1592, #1595, #1596, #1597, #1598 | in #1599 | `PR_ARC_INVENTORY.md`, `LATEST_STATE.md` |
| PG-Ring harvester | committed locally only, NOT pushed (ruff push needs the user's approval) | ruff branch `ccr-9f3b8075-74u47l-pgring-harvest`, commit `d2f4c2dcfb`, file `crates/ruff_cpp_spo/examples/harvest_pgring.rs` |
| PG-Ring idea catalogue | done, in PR (22 rows: 11 HAVE, 7 ADOPT, 1 REFUSED, 3 OPEN) | lance-graph #1600, branch `ccr-9f3b8075-74u47l-pgring-catalogue`; `.claude/research/2026-10-17-pgring-idea-catalogue.md` |

## 2. Decisions the user still owes

| # | Decision | Recommendation | Why | Caveats | Needs from you |
|---|---|---|---|---|---|
| 1 | "Thoughts" or "lanes" for the 64 tables | "thoughts" (a lane is a 16-byte column inside one) | the plans already count 64 thoughts × 32 lanes (`ce64-hypothesis-layers-v1.md:386-391`) | the user's model says "64 lanes"; changing either word touches docs | the word |
| 2 | Thought tables: session or storage | session by default, materialized only through the D-HL-8 commit ladder | D-HL `:474` (session addresses) vs capstone `:148` (sparse storage layer); a commit step reconciles both | a cached population then needs its own type over the same word layout, never `AlphaMask` | yes/no |
| 3 | Thought table as bracketed loop state (chained-hop option b / b-lite) | b-lite: one owner, session lifetime, commit only through the ladder | fixes the law-A1 objection at `2026-10-03-coresearch-chained-hop.md:51-53` without a new writer | needs a written amendment of law A1 | approve the amendment |
| 4 | Push the ruff PG-Ring harvester | push + draft PR | it is the only durable copy of the harvest method | PG-Ring is GPL-3.0: the example reads structure only, never vendors code | approval to push to ruff |

Older open items carry over unchanged: D-ARG-1..4; ndarray PR C (CE64 compat);
lance-graph `edge.rs` conformance; the "bits 61..63" comment in
cognitive-shader-driver's `Cargo.toml` (must name `EpistemicState5`, bits
59..63, as one field).

## 3. PG-Ring catalogue — how it was built, and how to redo it

The catalogue is written (#1600). Next steps it names: the dense-AND vs
sparse-seek probe (catalogue §3.3) and the three OPEN rows. The paper text
was read via alphaXiv (`get_paper_content`, arXiv 2608.03840, 2,145 lines
with appendices). What follows is how to rebuild the structural harvest.

Goal: a clean-room idea catalogue of PG-Ring (Gómez-Brandón, Hogan, Navarro,
ISWC 2026, arXiv 2608.03840; ring index over BWT for property graphs) as prior
art to LEARN FROM for D-ASM-11 (leapfrog triejoin). Ideas and structure only;
no PG-Ring code enters any AdaWorldAPI repo. D-ASM-11 and 11A/11B carry
VIOLATION banners (gather); a catalogue item that needs a data-indexed read is
recorded as such and mapped to presence-bit / range-intersection form
(D-HOLO H14c/H14d), never designed as a gather.

Inputs (clone if the container is fresh):

```sh
mkdir -p /home/user/adaworldapi && cd /home/user/adaworldapi
git clone https://github.com/AdaWorldAPI/pg-ring          # at ce012ba
git clone https://github.com/adriangbrandon/sdsl-lite     # at 8a1e8be
cd /home/user/ruff-pgring   # or re-create from the local commit if it survived
PGRING_SRC=/home/user/adaworldapi/pg-ring \
SDSL_INCLUDE=/home/user/adaworldapi/sdsl-lite/include \
LIBCLANG_PATH=/usr/lib/llvm-18/lib OUT_DIR=<scratch>/pgring_harvest \
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 \
  cargo run -p ruff_cpp_spo --features libclang --example harvest_pgring
```

Known harvest noise: most headers report `divsufsort.h` not found and the
`register` storage-class error from sdsl; classes and call graphs are still
emitted (63 summary files last run). If the local ruff commit is gone and
was never pushed, rewrite the example from the shape of `harvest_textord.rs` (same directory).

Method (the first attempt lost seven Haiku maps that never wrote output):
one file group per chunk, written to the catalogue file immediately with
`tee -a`, committed after each chunk. Each catalogue row: PG-Ring component
→ what it does (from the paper and the harvested signatures) → our
counterpart or gap (sealed ordered lanes, `seek_from`, `fold_ranges`,
presence-bit rank) → verdict (adopt idea / already have / refused shape).

## 4. Standing rules a fresh session must know

The full set lives in session memory, which does not survive a new
container; these are the ones that bit this session.

- Read before concluding: search finds candidates, a full Read decides.
- No foreign keys, no gather, no foreign lanes: a join is a layout that
  co-locates the pair at one coordinate; hops are range intersections,
  computed symbolically and folded into one mask before any plane AND.
- CE64 bits 59..63 are one field, `EpistemicState5`; never name a sub-range.
  S/P/O (bits 0..23) hold palette256 codes only.
- A user's mental model is not canon (e.g. "W = classid of the table" is not
  recorded; W is the thought layer, #1597).
- Do not write "operator-ruled/-pinned/-locked/-confirmed" as a status here.
- cargo always with debug 0; v4 timings only, printed tier, ns per element.
- Pushes outside ndarray need the user's approval per case; branches are
  `ccr-9f3b8075-74u47l[-slug]`; PRs as drafts, then subscribe.
- Never `git add -A`; stage by name. On an `entries/README.md` conflict take
  main's file and rerun `entries_index.py --write`.
- Decisions for the user go in a table: #, Decision, Recommendation, Why,
  Caveats, Needs from you.
