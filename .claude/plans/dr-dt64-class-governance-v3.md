# D-DRDT-CLASS-GOVERNANCE-1 — plan v3: decisions of 2026-11-11 and the remaining scope

**Status:** PROPOSED, docs only. Supersedes the open namespace questions of
v2 (`dr-dt64-class-governance-v2.md`, PR 2013); v2's census files stay valid
except where §3 corrects them. Decision record: Linear EXO-254.
Coordination: EXO-253. Read at `main` `9f2137c4d`.

## 1. Decisions taken (operator, 2026-11-11, in session; EXO-254)

| # | decision |
|---|---|
| D1 | The right side is `F`, one hex digit for the technology, one byte for the command. At most 16 technologies, shared across all languages, and 256 commands each. |
| D2 | Anything exceeding that budget (a seventeenth technology, or more than 256 commands) requires a classid on the left side. |
| D3 | Nothing moves. Existing ids keep their bits and meaning. |
| D4 | A behaviour address with no left side is the globally available pattern, assumed locally ("localhost"). A `WHERE` used inside Quack is localhost. |
| D5 | Anything that needs a delegate or a remote ON × WHEN × DO must define its target through the left side (domain:appid). |
| D6 | Quack is SQL. It uses the SQL technology digit, `F4`. No separate Quack digit. |
| D7 | Gremlin and Cypher are one technology, "Gremlin/Cypher", on Gremlin's digit `F5`. (First stated as "`F7`, or Gremlin's if the scope is adjacent"; then listed as one item.) |
| D8 | Applications such as Odoo get a domain/classid on the left side. They do not take a technology digit. |
| D10 | In short: `F###` is one polyglot address space of 4,096 addresses (16 × 256), meant mainly to give algebra its addresses. |
| D9 | The right side is planned by what is desirable as a general, reusable technology (ONNX, SQL, Gremlin/Cypher and others), see §2a. |

Carried from earlier records and unchanged: the technology digits `F0`
shared relational algebra, `F1` XLOG, `F2` ONNX, `F3` TensorFlow family,
`F4` SQL, `F5` Gremlin, `F6`–`FF` free (D-DT64-7 §1d O7); DT64 as the one
method authority, with every production-exposed executable method call
bound and admitted, internal opcodes excepted (D-DT64-AUTH, 2026-11-02,
`d-dt64-method-authority-census-v1.md`). v2 did not cite D-DT64-AUTH; it
should have.

## 2. What the decisions mean against main

| point | on main today | under D1–D7 | change needed |
|---|---|---|---|
| Low byte | Numbers lenses inside a concern; a word is `AlgebraRef { lens: u32, word: u8 }` (`lance-graph-dt64/src/lib.rs:129`) | The command | None in bits: lens `..FT00` with word `w` is address `..FTww`. Production uses only `F000` and `F100`. The reading "256 lenses of 256 words per concern" (D-DT64-7 §1b, O9) is superseded by D1 and D2 |
| Test lenses with a non-zero low byte | `0xFFFF_F200`, `0xFFFF_F300`, `0x7F0B_F001`, `0x0A1E_F001` in tests | Under D1 these read as technology + command, not as separate lenses | Decide per test: keep as a lens-keyed fixture, or re-express. Tests only |
| No left side | Every DT64 lookup takes a full `u32`. Quack carries no classid at all | Correct for local use (D4) | A stated rule for how a bare right side is checked against the registry (see §4, W2) |
| `0xFFFF` on the left | "Hardcoded behaviour, registered by DT64" (`ogar_codebook.rs:583`) | Unchanged (D3). It is the left side under which the global table is registered, not a marker for "no context" | One sentence in the contract docs. No code |
| Quack words | 12 of the 17 words of `F000` have a Quack name (`polyfill.rs` `POLYFILL`). Quack's own `Cmp` has more: `NeI32`, `LeI32`, `GtI32`, `GeI32`, and there are range, gather and hop forms | Shared words stay in `F0`. What the shared table lacks goes to `F4` (D6) | A census of Quack's public vocabulary against `F0`, then the `F4` table (§4, W1) |
| More than 256 commands | D-DT64-7 cites about 1,500 operators for TensorFlow | Needs a left side (D2) | Record in D-DT64-7 as an append-only note |

## 2a. The right-side slate (D9)

Sixteen digits. What a technology is: a general vocabulary of commands that
many classes and applications reuse unchanged. What it is not: an
application, a product or a domain; those go on the left (D8).

**Assigned** (D-DT64-7 O7, with D6 and D7):

| digit | technology | state on main |
|---|---|---|
| `F0` | shared relational algebra (relalg / mask-RISC polyfill; R2IL, SQL and Quack names are aliases over it) | table of 17 commands, declared in DT64, executed |
| `F1` | XLOG (Datalog) | 3 commands declared, not executed by the binding path |
| `F2` | ONNX | one test lens; a lab registry in `tract-harvest-probe` |
| `F3` | TensorFlow family | nothing |
| `F4` | SQL, including Quack | nothing; Quack's commands beyond `F0` are unregistered (W1) |
| `F5` | Gremlin/Cypher | nothing |

**Free:** `F6`–`FF`, ten digits.

**Candidates for the free digits.** A list to choose from. No digit is
proposed for any of them, and none is assigned by this plan.

| candidate | why it is a technology and not an application | where it lives today |
|---|---|---|
| NARS inference (deduction, induction, abduction, revision) | one fixed rule set used by every reasoning consumer | `lance-graph-planner` thinking, `lance-graph-nal` |
| Semiring / GraphBLAS algebra | fixed operator families over sparse matrices | `lance-graph` `graph/blasgraph` |
| Vector and codec search (Hamming, palette distance, CAM-PQ) | fixed distance and lookup commands | ndarray, `bgz17`, planner `CamPqScanOp` |
| SPARQL / GQL | query languages already parsed beside Cypher | planner strategies; may be aliases over `F0`/`F5` and need no digit |
| MeTTa | a rewrite language with its own commands | `lance-graph-metta` (excluded crate) |
| Document and OCR operations | fixed extraction commands, but not an algebra; fails test zero | OGAR OCR domain; a domain on the left |
| Workflow / template steps | fixed step commands, but not an algebra; fails test zero | `elixir-template`, `template-runtime`; left side |

Three tests decide a candidate (the first from D10). Zero: is it an algebra, a fixed set of operations with defined operands and results? If not, it does not belong on the right side at all. First: does it only rename commands of an
assigned technology? Then it is an alias view and takes no digit (D-DT64-7
D16''). Second: is it tied to one domain? Then it belongs on the left.

**Applications on the left (D8).** Main already puts the domain in the high
half: `0x01XX` project management (OpenProject, Redmine), `0x02XX` commerce
and ERP (Odoo), `0x05XX` CRM, `0x09XX` health (MedCare), and so on
(`lance-graph-contract/src/ogar_codebook.rs`, `ConceptDomain`). An
Odoo-specific behaviour is therefore `0x02CC_FTMM`: an Odoo class on the
left, technology and command on the right. Two points follow and need
checking, not deciding here:

- Today the low half of a data classid is the application's render prefix
  (`0x0002` for Odoo). A behaviour address uses the low half for `FTMM`, so
  it cannot also carry the render prefix. D-DT64-7 already calls the render
  prefix the older reading; W2 must state how the two coexist.
- A technology that outgrows 256 commands, or a seventeenth technology,
  takes a left side as well (D2). Which left side a technology uses when it
  is not an application is open.

## 3. Corrections to v2

- **Bypass reading.** v2 counted four production paths as bypasses because
  they reach the executor without `admit` (`cypher-quack`, `xlog`
  `run_lanes`, report detail CSV, the mask-RISC graph hop). Under D4 local
  use of a global pattern needs no left side. Under D-DT64-AUTH the word
  must still be a registered one. The gap is therefore narrower and more
  specific:
  1. a production surface that uses a word with no entry in the registry
     (today: every Quack comparison except equals and less-than, plus range,
     gather and hop);
  2. a delegated or remote execution that runs without a left side and
     without admission. None was found; the census did not look for remote
     paths specifically.
- **"One selection type".** A remark made after v2 merged proposed one
  selection type with several carriers. The brief (§7) requires three
  distinct typed concepts and forbids merging them: column projection, row
  selection, relation index. v3 follows the brief (§4, W4).
- **Crosswalk X1, X2.** Decided by D1–D5. Council questions Q1–Q4 of v2 are
  closed except who guards left `0xF000..=0xFFFE`.

## 4. Remaining work, in order

Each item is read-only or docs unless marked. "Source" is what has to be
read first; none of it has been read for this plan unless §2 cites it.

| # | item (brief section) | source to read | output | gate |
|---|---|---|---|---|
| W1 | Quack vocabulary census (§1, D6) | `lance-graph-quack/src/lib.rs` public enums (`Cmp`, `Filter`, `Agg`, group and hop forms); `polyfill.rs` | A table: every Quack word, its `F0` position or "needs `F4`" | every row cites a line; counts recomputed by command |
| W2 | Resolution rule for a bare right side (§2, §3, D4) | `BindStore::declare_algebra`, `resolve_algebra`, `arity`; `binding.rs` `prepare` | A written rule: local use resolves technology + command against the global table; unknown is refused | falsifiers 2, 3, 5 as tests (test-only PR) |
| W3 | The two meanings of ON (§2) | `MethodBinding` fields (`on`, `when`, effects), `Guard`, `SetExpr`, `Population`, `Truth` | A page separating "which field positions trigger" from "in which context they hold" | reviewed against the struct definitions |
| W4 | Three typed concepts (§7) | `WideFieldMask`, `lance-graph-contract/src/selection.rs`, the lab's `Plane`, `OrderIndex`, the DR `MapTable` | Per concept: what represents it today, its witnesses, what is missing. No merged carrier | a falsifier that a relation cannot be passed where a row selection is expected |
| W5 | Snapshot root (§6) | `lance-graph-dt64-lance/src/lib.rs` and tests; PR 1993 | What a root must name (class context, technology, command, binding, schema revision, basis, source and relation versions, order witness, scope, polyfill revision), and which exist | named refusals for a failed or ambiguous seal, each found or marked missing |
| W6 | PR 1984 recovery (§8) | its three commits | A dedicated recovery PR: lab files, tests, oracle, bench note. No promotion. Notes on the 8 KiB block per occupied tile and on duplicate edges collapsing (set versus bag) | crate tests green. **Needs your go-ahead; code** |
| W7 | Cost model and data movement (§5, §9) | lance-graph PRs 1987–1994; ndarray PRs 414 and 418 (from the local checkout); `simd_stride.rs` | Per route: bind/resolve, source and G2L preparation, packing, staging, tile setup, hot path, retained bytes, reuse count, invalidation cost. Then the twelve-cell comparison in the lab | every number in ns or µs with its command; a per-query ms is a failed gate |
| W8 | Thought reuse (§10) | `thought-git-probe` slot and generation code; CE64 `w_slot` | The probe outline of v2 extended with the slot-generation check | oracle and negative control per item |
| W9 | Codegraph as proof (§11) | the artifacts of PRs 1999, 2003, 2004, 2006, 2007 | The chain entry point → context → registry → binding → admitted reference → word → executing symbol, per production surface, labelled UNKNOWN / POSSIBLE / ARTIFACT_MISSING / STALE where the graph does not show it | a planted bypass is found; an admitted path is not reported |
| W10 | Standalone probe (§12) | W1–W5 | A minimal lab crate running the falsifiers that are now decidable (1–8, 10, 21, 22, 24) | each with an independent oracle and a negative control. **Code, lab only** |
| W11 | Council | this plan and the outputs above | 5+3 review of the working model before any contract change | — |

W1–W5, W8 and W9 can run in parallel. W7 needs W6. W10 needs W1 and W2.
No public contract or codebook change lands before W11.

## 5. Open with the operator

1. The slate of §2a: which candidates get a digit, and which digits.
2. Go-ahead for the PR 1984 recovery (W6).
3. Who owns the `F4` table for Quack: `lance-graph-r2il` beside the `F0`
   table, or `lance-graph-quack` itself.

## 6. Not verified

Nothing was built or run. §2 rests on the files it cites, read in this
session, and on the v2 census. The sources listed in §4 are unread.
