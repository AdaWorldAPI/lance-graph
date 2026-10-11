# NAL-1..9 × the 34 recipes — audit against two NARS reference engines

> READ BY: truth-architect, contradiction-cartographer, v3-template-smith,
> integration-lead, and any session touching `recipe_kernels.rs`,
> `recipe_dispatch.rs`, `rung_schedule.rs`, a NARS revision law, or a
> "meta"/"self" recipe.
> STATUS: MEASURED where a command is named; otherwise labelled per claim.
> Audit date 2026-10-11, lance-graph `main` at 3f8d13c.

## The question

Which of Pei Wang's NAL-1..9 capabilities can the existing substrate already
execute, which do the 34 recipes only resemble, and where must an executed
step become an addressable object before higher-order or self-directed
reasoning can work?

## Answer in four lines

1. **No recipe implements a NAL operation.** Every `Tactic::apply` reads and
   writes scalar `ThoughtCtx` fields; none reads a prior result by identity.
   Five recipes pass their own unit tests (status 4); none has parity
   evidence against a NAL reference. (`recipes.tsv`, VERIFIED-IN-CODE.)
2. **lance-graph's NAL-1 revision arithmetic is right in four places and
   wrong in two.** Against OpenNARS output, laws A, C, D and E reproduce all
   three points; B1 (the replay path's table) and LIN (`grammar::revise_truth`)
   reproduce none. (MEASURED, `tests/nal_revision_oracle.rs`.)
3. **MCP ("meta-cognition") is two scalars.** It cannot see any past execution,
   so it gives a calibrated and an overconfident history the same verdict. A
   receipt (executed step + confidence left + later outcome + identity) fixes
   that. (MEASURED, `tests/mcp_receipt_probe.rs`.)
4. **The receipt already exists and is thrown away.** `run_recipes` returns
   `Vec<ExecutedStep>`; `wave_dispatch::dispatch_thought` binds it to `_steps`
   and drops it, and `Tactic::apply` has no parameter that could receive it.
   (VERIFIED-IN-CODE, `wave_dispatch.rs:60-80`, `recipe_kernels.rs:373`.)

## Files

| file | what |
|---|---|
| `recipes.tsv` | 34 rows: requires/writes, actual `apply` effect with file:line, whether it reads a prior object, consumers, closest NAL capability, missing operand, status 1–5 |
| `nal_inventory.tsv` | 115 NAL capabilities (NAL-1 15 … NAL-9 22) with source, truth function where Wang gives one, oracle fixture, falsifier. 108 rows read in a primary source, 7 UNVERIFIED. `lance_graph_mapping` is left blank at row level; the level-level matrix is below |
| `nal_notes.md` | which levels change grammar, rules or control; spec-vs-engine disagreements |
| `schedule_depth.md` | why the 34-recipe plan is (almost) flat; the measured depth-2 exception |
| `oracles.md` | pinned engines, build outcomes, observed outputs, what is blocked |
| `revision_points.nal`, `RunNal.java` | the audit fixture and the 21-line driver that runs it through upstream `NALTest` |

## Status distribution of the 34 recipes

| status | meaning | count |
|---|---|---|
| 1 | not represented | 7 |
| 2 | representable, not executed | 3 |
| 3 | executable, semantic equivalence unverified | 17 |
| 4 | verified against its own declared behaviour (unit test) | 5 |
| 5 | contradicted (MCP 10, ICR 31) | 2 |

Closest resemblances, none equivalent: CR (11) detects a same-topic
contradiction but never revises; TCF (20), IDR (29) and CDT (13) pick a
median/max over anonymous scores, which resembles NAL-1 choice; MPC (27) is an
unweighted mean that resembles revision without evidence weights.

## NAL coverage by level

| NAL | in the recipe layer | elsewhere in lance-graph | evidence |
|---|---|---|---|
| 1 | resemblance only (CR, TCF/IDR/CDT, MPC) | revision laws A, C, D, E match OpenNARS; B1 and LIN do not. Deduction etc. exist in `nars::truth` | MEASURED for revision; truth functions other than revision UNVERIFIED against an engine |
| 2–4 | not represented: no statements, copulas, compound terms, product/image | not audited here | — |
| 5 | not represented: no statement can be an operand | not audited here | — |
| 6 | not represented: no variables | not audited here | — |
| 7 | TCA reads scalars only | `temporal.rs` version-range reads exist; not tested as NAL-7 inference | VERIFIED-IN-CODE (existence only) |
| 8 | none | not audited here | — |
| 9 | MCP, SSR, ASC, LSI are scalar rules | no recipe can read an executed step | MEASURED for MCP |

## What the reference engines showed

- **OpenNARS for Research** (`4d1bdb0`) implements mental operators (Believe,
  Want, Wonder, Doubt, Evaluate …) as events about its own beliefs and goals;
  9 of 215 `single_step` files were run and all passed. Its Maven build is
  BLOCKED (parent POM only existed on the retired Sonatype snapshot repo);
  a direct `javac` build works.
- **ONA** (`dc4efd0`) built and ran all seven fixtures. Its `{SELF}` fixtures
  are goal-directed behaviour (`^say` is executed), not self-inspection: ONA has
  no operation that reads its own beliefs or inference steps. Its evaluation
  script never actually checks `reasoningbench.nal` (wrong expectation syntax).
- The engines and Wang's 2009 specification **disagree** on compound-term
  confidence (spec `or(..)+and(..)`, both engines c1·c2) and ONA answered one
  reasoningbench question with c=0.816 where the fixture expects 0.729. So
  "matches NARS" must always name which reference.
- No Wang text found says NAL-9 adds no grammar or inference rules; the NARS
  introduction describes NAL-9 as NAL-8 procedural inference applied to
  internal operations.

## Architecture decision

**DECISION:** Narsese stays an external test format. OpenNARS for Research and
ONA are run outside the build, at pinned commits, as oracles. Neither becomes a
dependency, and no Narsese parser enters the hot path.
**BASIS:** the two engines disagree with each other and with the spec on
measured cases; OpenNARS's own build already broke on a retired repository;
and the one capability measured end to end (revision) needed only three
numbers from an engine, not the engine.
**SCOPE:** NAL-1 revision and the MCP/receipt question are answered. NAL-2..8
parity is not claimed and was not tested.

**PROPOSED CONTRACT (not implemented):** let a recipe read prior execution
receipts. `ExecutedStep` already exists; it lacks the confidence it left, the
later outcome, and a stable identity, and `Tactic::apply` has no way to
receive it. The probe shows the smallest useful shape, with de-duplication at
the receipt because `LearningLoop` itself counts a repeated observation twice.
**REVISIT WHEN:** someone needs a recipe to judge its own past (MCP, SSR, ASC)
or a second schedule wave from a real producer/consumer edge.

## Open

- **G2 from D-RPF-CONF-0 now has outside evidence:** the replay and
  counterfactual path emits B1, which the reference engine contradicts on all
  three points and whose confidence ignores its premises. Choosing a law is
  still a decision, not settled here.
- `grammar::revise_truth` (LIN) is documented as "Standard NARS revision" and
  feeds `GrammarStyleAwareness::revise`; its confidence can end below the
  stronger premise.
- `rung_schedule.rs:321` claims depth 1 "always"; 64 of 256 initial masks give
  depth 2 (see `schedule_depth.md`).
- Five Gate-bucket recipes read `sd` through `gate()` without declaring it, so
  the scheduler cannot see that dependency (`recipes.tsv`).
- Not run: 206 of 215 OpenNARS `single_step` files, all `multi_step` and
  `application` files, and ONA's English tests.
