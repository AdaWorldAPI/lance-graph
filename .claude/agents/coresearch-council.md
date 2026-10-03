# Co-Research Council — exploration across the code and the outside world

**READ BY:** the orchestrating main thread before invoking `/coresearch`;
`integration-lead` and `convergence-architect` when an idea arrives from
outside the workspace.

> **Purpose.** Explore an OPEN question together: what the workspace already
> has, what the outside world already knows (arXiv, known systems such as
> DuckDB and Odoo, ontologies and established concepts), and where the two
> meet. The product is an **exploration map**, never a ratified design.
>
> **Prior art.** This formalises the one-off 2026-06 research council
> (`.claude/knowledge/research-council-semantics-papers-2026-06.md`: five
> Opus readers, papers read in full, a firewall verdict per idea), and keeps
> its verdict vocabulary. `convergence-architect` supplies the bridge lens;
> `prior-art-savant` supplies the internal-history lens.

## How it differs from the 5+3 council

| | `5plus3-council` | `coresearch-council` (this card) |
|---|---|---|
| direction | **converge**: harden one committed spec | **diverge, then map**: find and grade options |
| input | a spec with a committed resolution | a question brief with no committed answer |
| roles | 5 savants verify, 3 reviewers attack | 5 scouts gather, 3 co-architects connect and filter |
| output | ratified v3, then implementation | an exploration map: ideas, crosswalk, grades, probes |
| authority | ratifies a design | **ratifies nothing**; a chosen idea goes to the operator, then to a plan or a 5+3 |

The two compose: coresearch decides *what is worth designing*; 5+3 hardens
*the design*. Never run them in one pass.

## When to convene (and when NOT to)

Convene for an open question where outside knowledge plausibly changes the
answer:

- "has someone already solved X?" (a known system, paper or standard);
- a new subsystem before its plan exists (what to borrow, what to avoid);
- aligning with an external model (an Odoo module, an ontology, DuckDB
  semantics) where the mapping itself is the question;
- a measured anomaly with no internal explanation yet.

Do NOT convene for:

- a decision the operator has already made (it is a frozen anchor, never
  re-opened);
- a defect hunt (use the reviewers or `brutally-honest-tester`);
- a single-source lookup (read the source directly);
- anything with a committed design (that is 5+3 work).

## Phase 0 — the QUESTION BRIEF (main thread)

Written before any agent is cast. It replaces the 5+3's spec, and it holds
**no answer**:

1. **THE QUESTION**, in one or two sentences, plus why it is open now.
2. **ANCHORS**: frozen decisions the exploration must not re-litigate
   (operator rulings, iron rules, `CLAUDE.md` P0s), each cited. An idea that
   contradicts an anchor is recorded as `CONFLICTS-ANCHOR`, never adopted.
3. **INTERNAL SURFACE**: the crates, files and board ids in scope, by path.
   If unknown, run a pre-brief Explore first.
4. **EXTERNAL DOMAINS**: which outside rings to scan (see Phase 1) and the
   seed terms for each, plus the existing knowledge docs that already cover
   them (read those first; never re-harvest what a doc already holds).
5. **WHAT WOULD COUNT AS AN ANSWER**: the shape of a useful result (a
   mechanism, a mapping table, a falsifiable probe) and what is out of
   scope.
6. **BUDGET**: maximum sources per scout, and the date cut for literature.

**Premise gate.** Run `.claude/agents/premise-auditor.md` on the question
before casting the scouts. Exploring a question that files a concept under
the wrong mechanism only maps the wrong territory in more detail. Run it
again on the question together with every option in the exploration map
before it goes to the operator; when tests 1 or 2 fire, the map must offer
"the property belongs elsewhere" (`premise-auditor.md`, Step 2).

## Phase 1 — the 5 scouts (parallel; two rings)

| # | scout | ring | reads | returns |
|---|---|---|---|---|
| 1 | **code cartographer** | inside | the internal surface, first-hand (`FIRST-HAND-SOURCE-LAW.md`) | what exists, graded `VERIFIED-IN-CODE` / `CLAIMED` (doc says so, code not read) / `ABSENT` (named, closed search space only) |
| 2 | **internal prior art** (`prior-art-savant` lens) | inside | board entries, `EPIPHANIES.md`, plans, knowledge docs | earlier explorations, rulings and rejected ideas, with ids. A previously rejected idea is returned with its rejection |
| 3 | **literature scout** | outside | arXiv and papers (alphaXiv tools, web) | mechanisms, each with the paper id and a read grade |
| 4 | **systems scout** | outside | known systems: DuckDB, Odoo, PostgreSQL, Datalog engines, bitmap indexes, Lance / Arrow, and the like | how they solve the question: design, invariants, known failure modes, licence |
| 5 | **concepts and ontology scout** | outside | standards and concept frameworks: OWL, DOLCE, OGIT, schema.org, relational and lattice theory, formal concept analysis | the established vocabulary and how it maps onto ours |

Swap a scout when the question needs a different ring, and declare the swap
in the brief.

**Read grades (external).** Every external claim carries one:

- `READ-IN-FULL`: the scout read the whole source;
- `SECTION-READ`: named sections were read;
- `ABSTRACT-ONLY`: only the abstract or summary;
- `SECONDHAND`: known from another source, which is cited.

Only `READ-IN-FULL` or `SECTION-READ` may support a mechanism claim. An
`ABSTRACT-ONLY` idea is at most a lead.

**Scout output contract.** At most 10 items. Each item has:

- a name;
- one or two sentences on the mechanism;
- its source, by id or `file:line`;
- its read grade (or code grade);
- the internal surface it touches, or `none`.

No essays, no designs. Scouts are read-only and never write board files.

**Model.** Scouts 2–5 accumulate across many sources, so they run on Opus.
The code cartographer may run on Sonnet when the surface is small. Never
haiku.

## Phase 2 — the CROSSWALK (main thread, before any co-architect)

Merge the scout output into one table, one row per idea:

| idea | mechanism and evidence (1–2 sentences, from the scout) | outside source (grade) | nearest inside surface (grade) | relation |
|---|---|---|---|---|

`relation` is one of:

- `ALREADY-HAVE` (we ship it, under another name);
- `PARTIAL` (we have part; the gap is named);
- `NEW` (no counterpart inside);
- `CONFLICTS-ANCHOR` (contradicts a frozen decision; kept for the record).

Duplicates are merged. Raw scout output is banked in the scratchpad and
never forwarded, so the crosswalk must be self-contained: each row carries
the mechanism and the evidence that supports it, enough for a co-architect
to design an adoption shape or a kill probe without the raw output.

## Phase 3 — the 3 co-architects (parallel, on the crosswalk only)

| # | co-architect | lens | per idea returns |
|---|---|---|---|
| 1 | **bridge architect** (`convergence-architect` lens) | where an outside idea and an inside surface share a shape, and what the smallest adoption would be | `OPPORTUNITY` / `WORTH-EXPLORING` / `DROP`, plus the adoption shape |
| 2 | **firewall and fit critic** | the workspace's non-negotiables: no float or LLM on the hot path, zero-copy, the lance family stays upstream, AdaWorldAPI forks, contract boundaries, licence fit | `PASS` / `CONFLICT` (fits only with a quarantine seam, which is named) / `TRAP` (no seam exists) |
| 3 | **falsifier designer** | for each idea that is not dropped, the smallest probe that could kill it | one pre-registered probe: input, measurement, kill condition, cost |

Co-architects see the crosswalk and the brief, never raw scout output, and
never each other. They propose and grade; they do not decide.

## Phase 4 — the EXPLORATION MAP (main thread)

Each idea gets one final verdict:

- `ADOPT-NOW`: a firewall `PASS`, small, and offline-buildable;
- `PROBE`: worth a measurement first, and the probe is named;
- `PARK`: interesting but not now, with the reason;
- `SKIP`: a `TRAP`, a duplicate, or `CONFLICTS-ANCHOR`.

The map holds the question, the crosswalk, the verdicts, the probes and the
open points. It also states what was **not** searched, so that "nothing
found" is never read as "nothing exists".

## Phase 5 — landing (minimal residue)

- The map lands as **one** dated file in the transient tier,
  `.claude/board/entries/YYYY-MM-DD-coresearch-<topic>.md`, plus
  `python3 .claude/tools/entries_index.py --write` in the same commit.
- **Nothing is adopted by the council.** The operator chooses. A chosen
  `PROBE` becomes a `STATUS_BOARD` row; a chosen design becomes a plan, and
  a delicate one goes through `/5plus3`.
- A knowledge doc is written only when the operator asks for one, or when
  the outside ring will clearly be reused (as the 2026-06 doc was).
- `EPIPHANIES.md` only through the § Closeout admission gate, never directly
  from a council run.
- Close with the one-line closeout record, and a `MIRROR` only if it
  carries something.

## Non-negotiables for every role

- **Search is navigation, never evidence** (`FIRST-HAND-SOURCE-LAW.md`).
  The same holds outside: a search-result snippet is a lead, not a read.
- **Concepts, not code.** Outside systems are described in our own words.
  Never paste external source code; record each system's licence (for
  example Odoo LGPL-3, DuckDB MIT) next to any idea that would borrow from
  it.
- **Human authorization is provenance, not validation.** An idea is graded
  by its evidence, not by who suggested it.
- **No model identifier and no secret** in any artifact. Web access goes
  through the configured proxy and tools only.
- **One writer.** Only the main thread writes the map and the board.

## Token economy

| role | model | why |
|---|---|---|
| brief, crosswalk, map | main thread | accumulation |
| scouts 2–5 | Opus | multi-source accumulation |
| code cartographer | Sonnet or Opus | Sonnet when the surface is small |
| co-architects | Opus | synthesis across the crosswalk |

If the question can be answered by reading two sources, it is not
council-grade: read them.
