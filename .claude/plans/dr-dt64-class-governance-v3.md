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
| D11 | Rendering is always application-related. The domain::appid on the left carries it, so the render prefix and `FTMM` do not conflict. |
| D12 | The right side below `F###` (`0x0000`–`0xEFFF`, 15 × 4,096 = 61,440 values per application) is not rendering's. It **can** be assigned to rendering. (Corrects an earlier wording in this plan that said it belongs to rendering.) |
| D14 | Everything is now classes: storage; methods, which are behaviour; compute / RAM (recall). |
| D13 | If the 4,096 behaviour addresses are ever used up, `E###` may be taken for behaviour as well. Not decided now; to be decided when it happens. |
| D15 | In return: DR's concern is addressing, with `WideFieldMask`. DT's concern is mapping, with `ClassView`. (Restates the D-DRDT amendment already logged; unchanged.) |
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

- Rendering is always application-related, so it is carried by the
  domain::appid on the left (D11). The render prefix in the low half of a
  data classid and `FTMM` in the low half of a behaviour address therefore
  do not compete: a behaviour address never needs a render prefix.
- To state in W2, not to decide: on main the left half of a data classid is
  domain + concept number (`0x0202` is commerce, concept 02) and the
  application id sits on the right (`0x0002` for Odoo). D11 and the
  D-DT64-7 diagram both write the left of a behaviour address as
  domain::appid. Data classids keep their bits (D3).
- A technology that outgrows 256 commands, or a seventeenth technology,
  takes a left side as well (D2). Which left side a technology uses when it
  is not an application is open.

## 2b. Everything is a class (D14)

Three kinds of thing are addressed as classes:

| kind | what it covers | addressed today by |
|---|---|---|
| storage | what is kept: the class's fields and rows | the data classid and its `ClassView` basis |
| methods (behaviour) | what is done: ON × WHEN × DO | `F###` on the right; a domain::appid on the left to delegate |
| compute / RAM (recall) | what is held in memory and recalled | not addressed as a class on main as far as this plan has read; OPEN |

The split of work over those classes is D15: DR addresses them (identities,
positions, routes, the `WideFieldMask` that selects fields and populations);
DT maps them (the `ClassView` that says what a class's positions mean, its
methods and their bindings). DR does not interpret a class; DT does not
route.

Rendering is one possible use of the right side below `F###`, not its owner
(D12). Which part of that space storage and compute/RAM classes use, and
whether they need a reserved range of their own, is not decided and not
proposed here.

## 2c. D15 at work: prepare once per frame, answer in µs (crucial)

Two concerns, kept apart (D15):

**DR — addressing / `WideFieldMask`.** Everything that says *where*:

- the `WideFieldMask`: which field positions a binding, a storage read, a
  fold and a display touch;
- the slab: 256 × 256 = 65,536 row positions as three aligned mask planes
  of 1,024 words, stacked under an outer byte when needed (up to 256 slabs,
  256³ positions); 1,000,000 rows is 16 slabs;
- the order index: a permutation of row positions, built once per frame, so
  a key range is one contiguous range of positions;
- the versioning store: every slab binding is sealed against the
  `FrameStore` (`FrameRevisions { source, relation_map, coordinates }`), an
  order index carries the revisions it was built at, and a source advance
  makes both refuse by name before any word is read. The map table writes
  one Lance version per advance through `LanceCycleWriter` (EXO-225).

  > **Note (2026-11-11, operator ruling, `dr-widefieldmask-dt-classview-v1.md`
  > §1).** What is versioned is the mapping (identities, properties, source
  > positions), not the content. Under that ruling a slab binding needs to
  > refuse only when `relation_map` or `coordinates` moved; refusing because
  > `source` alone advanced is stricter than required and is today's code,
  > not the contract, as long as the binding's interpretation also holds.
  > Everything prepared from content is different: the order index, the
  > prepared presence / scope / selection planes, a join assignment from
  > mutable keys and a stored recursion result need a matching change
  > witness or a recomputation, never reuse because coordinates are
  > unchanged. Validity follows each artifact's dependencies.

**DT — mapping / `ClassView`.** Everything that says *what*:

- which fields a class has, their names, their order, and their types;
- turning a name into a field position. That position is the only thing DT
  hands to DR; `WideFieldMask` is built from positions, never from names.

The amortization is a DR property: the mask, the planes, the index and the
frame seal are resolved once per frame; a query reads them. DT is consulted
when the selection is built, not per query.

**Measured** in PR 2010 (criterion medians, one 4-core machine, every
answer checked against a row-at-a-time count; `crates/lance-graph-benches/
benches/cypher_df_vs_quack.rs`, group `angles`; results in
`.claude/board/entries/2026-11-11-cypher-df-vs-quack-bench.md`):

| 1,000,000 rows | DataFusion | Quack, gather per query | DR256 slabs, prepared once | order index, no mask |
|---|---|---|---|---|
| filter `count(*)` | 2,170 µs | 268 µs | 188 µs | **1.76 µs** |
| one-hop `count(*)` | 26,800 µs | 3,450 µs | **4.86 µs** | — |
| hop with `WHERE a.age > 50` | 28,600 µs | refused | — | **2.17 µs** |

Once per frame (reported, not gated): nodes 3,600 µs, edges with the
foreign key read once 4,550 µs, order index (a sort) 41,300 µs. The
per-query figures include the frame check of every slab binding. The
refusal after a source advance is tested (`an_order_index_refuses_a_stale_
frame_and_a_plane_in_table_order`), not timed.

So the 1–5 µs band holds where the query reads prepared state: a key range
over an order index, or a hop over prepared planes. A whole-table fold
stays bandwidth-bound (188 µs). Against the per-query paths, preparing once
pays back after about 155 filter queries per frame (41,300 µs / 266 µs) and
after 2 hop queries (4,550 µs / 3,450 µs).

What this does not show: G2L across different source orders, a real Lance
read path, or a slab count above 16.

## 2c.1 DR Dynamic handoff is a strategy, not another executor (2026-11-12, proposal)

**Status:** HYPOTHESIS, docs only (D-DR-SELECT-K-0). Detail:
`dr-widefieldmask-dt-classview-v1.md` §1a.1. This note changes no namespace
table and ratifies nothing in §2e.

For `Lance WHERE → survivors K`, DR Dynamic's existing address/Focus mechanism is the **one** local register coordinate space. At binding time, choose **a borrowed ordered index/address view, a borrowed sparse local positions view, a K-sized FocusMask, or implicit ALL(K)** based on the follow-on operation, shape, budget and reuse. These representations are alternatives or aliases under **one bound selection**, not new stores or competing semantic planes. Query interpretation stays DT64's; existing Quack/RelAlg/R2IL own remaining algebra, mask-RISC/ndarray own lawful execution and ISA. Stage or remap once, only when a verified requirement exists. Do not reevaluate pushed storage predicates or scan N again downstream.

- **W4 addendum:** record roles, ownership and witness type of IndexVector, SelectionVector, FocusMask and `ALL(K)` without merging the three typed semantic concepts. A same-number-of-fields reorder, wrong coordinate, lost order, or reused obsolete content-dependent WHERE remains a negative test.
- **W7 addendum:** measured four-way *consumption* shootout (not four new execution paths) across K=0/64/256/1024/4096/65536, dense/sparse, order/Bag, one versus repeated folds, warm/cold, ISA and memory/IO budgets. Separate lookup, storage read, bind, optional representation conversion, prefetch, hot fold and retained bytes. Calibrate deterministic policy after evidence, not a priori.

## 2d. Two sources of class authority (existing contract, kept as an invariant)

"Everything is a class" (D14) does not mean "every class is registered
centrally in lance-graph". Main already separates two sources of class
authority, both joined by the same classid:

| | global | consumer |
|---|---|---|
| defined in | lance-graph contract (canon builtins, global algebra, `F###` words) | OGAR vocabulary (`ogar-vocab`) |
| activated by | being compiled in | the consumer's own `HotPlug` const and its call to `activate` |
| authority | the builtin tables (`BUILTIN_READ_MODES` "holds only the canon builtins") | `CapabilityAuthority`, implemented by `OgarAuthority` in `lance-graph-ogar` |
| storage reading | `classid_read_mode` for canon domains | `Activation::read_mode_for`, derived from the plug (`plug_readings`, `lance-graph-ogar/src/lib.rs:452`); fails closed, no silent V1 fallback |
| a new consumer needs | nothing in this crate | its own declaration; no edit to the contract, no substrate recompile (D-BLOCKS-HOTPLUG-1) |

Source: `crates/lance-graph-contract/src/hotplug.rs` (module doc: socket in
the contract, authority in OGAR, declaration in the consumer; OCR plugs
`0x0805`, `0x0808`, `0x0809`). The classid is `CanonHigh`
(`ogar_codebook::CLASSID_ORDER`): the high `u16` is the OGAR concept, the
low `u16` the consumer's app prefix. (The second half is an assignment of meaning, not a property of the bits; see §2e.)

Consequences for this plan:

- `F###` addresses global technology and method words. It is not a
  registration route for consumer classes, and consumer classes never need
  an entry in a global table to become valid.
- OGAR semantics does not enter DR or the agnostic executor. DR addresses
  any classid; it does not know whether the class is canon or plugged.
- Reading-only seats exist: `0x17XX` palette plugs activate without an OGAR
  capability table. A rule requiring capability registration for every
  class would break them.
- DT64 stays the authority for executable ON × WHEN × DO bindings. A
  consumer activation is not by itself an admission of a method binding.

What is not shown on main: that a class activated through OGAR reaches a
DT64 binding. `lance-graph-ogar` does not name `lance-graph-dt64` or
`BindStore`, and `lance-graph-dt64` does not name `HotPlug` (search over
those two crates only). That seam is W12 below.

## 2e. DR routes, DT64 gives meaning (operator, 2026-11-12)

Stated in session while reviewing W12, extending D12 and D15:

| | DR | DT64 |
|---|---|---|
| concern | routing (addressing) | meaning (schema, method) |
| reads `0x0901_0005` as | address 5 under class `0x0901` | whatever its binding says, e.g. the patient concept |
| interprets the bits | no | yes, through its binding |

The right half of a classid splits by range:

| right half | DT64 binds there |
|---|---|
| `0x0000`–`0xEFFF` | concepts of the class (`0x0901_0005` = concept 5 of class `0x0901`) |
| `0xF000`–`0xFFFF` | methods of the class (`0x0901_F005` = method `F005` of class `0x0901`) |

The method range is already the code's reading: `is_behaviour_lens`
(`crates/lance-graph-contract/src/class_view.rs:1053`, D-DT64-7 O9) and
`ClassView::lens_fields`.

Consequences:

- The bits carry no meaning of their own. "The consumer's app prefix" (§2d,
  and the twelve `APP_PREFIX` constants `0x0000`–`0x000C` in OGAR's ports) is
  one meaning DT64 can assign to an address in the concept range. It is not
  a property of the address, so it does not collide with a concept number
  and nothing has to move (D3).
- DR never needs to know what an address means, which is why it can route
  canon and plugged classes alike (§2d).
- Activation names the classes a consumer plugs in. It does not have to
  check the right half: resolving what a full address means is DT64's job
  (`route` → `admit` → `bind_lanes`). W12's observation that another right
  half over the same class passes activation is therefore the intended
  layering, not a gap (`ISS-ACTIVATION-IGNORES-APP-HALF`, resolved).

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
| W12 | Consumer class → DT64 binding (§2d) | `hotplug.rs`, `lance-graph-ogar` `activate` / `plug_readings`, `BindStore::register` / `bind_lanes` | A lab test: activate a consumer through the existing OGAR vocabulary, resolve its class through `ClassView`, bind and run one method through DT64; a classid that was not activated is refused; no edit to the global builtins or registers | the refusal disable-verified; no change to the contract. **Done in `crates/dt64-hotplug-probe` (2026-11-12): five tests over `OgarAuthority` + `OgarClassView`, refusal disable-verified; the activation check does not read the right half, which §2e rules is DT64's concern (`ISS-ACTIVATION-IGNORES-APP-HALF`, resolved).** |

W4 and W7 each carry an addendum in §2c.1 (2026-11-12, proposal).

W1–W5, W8 and W9 can run in parallel. W7 needs W6. W10 needs W1 and W2.
No public contract or codebook change lands before W11.

## 5. Open with the operator

1. The slate of §2a: which candidates get a digit, and which digits.
2. Go-ahead for the PR 1984 recovery (W6).
3. Who owns the `F4` table for Quack: `lance-graph-r2il` beside the `F0`
   table, or `lance-graph-quack` itself.

## 5a. Cross-check against Linear (2026-11-11)

Read in full: EXO-199, EXO-220, EXO-225 (Decisions); EXO-121, EXO-126,
EXO-153, EXO-214, EXO-227, EXO-228 (plans); EXO-206, EXO-243, EXO-245 (A2A).
No Linear decision contradicts D1–D15.

**Agree, cite from here:**

- EXO-199 (classid stays the class; a binding digest and the issuing store
  attest the binding). Same as D3 and D14: nothing is re-numbered, behaviour
  is attested per binding.
- EXO-220 (five boundaries: DR coordinate fabric, DT64 shape, source-owned
  SoA, lawful execution, IAM). D15 is boundaries 1 and 2 of it.
- EXO-121 (ClassView / WideFieldMask as field presence, append-only field
  positions; permission, relation and cross-table identity witnessed
  separately). Matches D15 and the three concepts of W4.
- EXO-153 and EXO-214 both forbid an `F###` identity "by assumption". D1, D6
  and D7 are the operator assigning them, so the rule is kept, not broken.

**Feed into work items:**

- EXO-126 (D-QSP Quack SQL parity). With D6 the D-QSP gaps (column-to-column
  comparison, COALESCE/NULLIF, nullable arithmetic, NOT IN subquery) are
  rows of the `F4` table. Add them to W1.
- EXO-227 G1 (element-wise min/argmin across lanes, decision pending). A
  word that needs an `F0` or `F4` position. Add to W1.
- EXO-225 (one Lance version per map advance via `LanceCycleWriter`). This
  is what a snapshot root can name as the map version. Add to W5.
- EXO-153 (remote async execution against the owner's planes). This is the
  execution side of D5 delegation. W2 cites it.

**Gaps v3 does not cover:**

- EXO-206 / EXO-243 / EXO-228 (D-LANESOCK-4): IAM is independent of the
  binding and checked at execution. D5 makes the left side name a delegation
  TARGET; it does not authorize the caller. v3 must say so: an address,
  local or delegated, never grants a read. EXO-243 (medcare A1, fields =
  requested ∩ visible fields, the row half) is a waiting consumer and is
  not answered by v3. Added as W12.
- EXO-245 (medcare A2, signature-only actions as method bindings). D8 puts
  such an app on the left side; the binding fields it asks about are not
  answered. Added as W13.

| # | item | source | output |
|---|---|---|---|
| W12 | Reader authorization beside addressing | EXO-206, EXO-243, D-LANESOCK-4 inventory | One statement placing IAM at the read boundary for local and delegated execution; the EXO-243 test (two roles, 12 fields, 10^3 rows) as a falsifier |
| W13 | App actions as bindings | EXO-245; `MethodBinding`; D-DT64-AUTH census | Which binding a signature-only action gets under D8, or a refusal with the reason |

## 5b. Open board items, by concern (Linear, read 2026-11-11)

**DR — addressing / `WideFieldMask` / versioning store**

| issue | ask | what the lab and the board already answer |
|---|---|---|
| EXO-244 (High, medcare A4) | first external `LaneSource` over Arrow; who owns `frames()`, when to advance the frame | The source owns its `FrameStore` and advances it when its resident dataset version changes; every slab binding and order index sealed at the old revisions then refuses (`SourceChanged`) and is re-prepared once (PR 2010, `an_order_index_refuses_a_stale_frame_and_a_plane_in_table_order`). Typed interpretation is DT (EXO-202), not DR. |
| EXO-243 (High, medcare A1) | fields = requested ∩ visible; what `Scope.id` derives from; the row half | The field half is a `WideFieldMask` intersection taken once per binding. The row half is a prepared visibility plane (the lab's scope plane). `Scope.id` is a partition witness, never authorization (EXO-206); IAM stays at execution. |
| EXO-228 (D-LANESOCK-4) | reader-scope seam inventory | same seam as EXO-243 |
| EXO-221 / EXO-225 | map-table WAL; decided: `LanceCycleWriter`, one Lance version per advance | EXO-221 still "In Progress": step 3 (rebuild PR 1993 on the chosen WAL) is not confirmed done. |
| EXO-222 / EXO-224 | whole 64×64 tile kernel (ndarray) | execution under prepared addresses; no addressing change |

**DT — mapping / `ClassView`**

| issue | ask | status |
|---|---|---|
| EXO-202 (BLOCKED) | lane interpretation: scale and codebook; metadata from OGAR to the binding | wiring proof open |
| EXO-247 (Low, medcare A5/A6) | `SchemaRev` for a manifest-defined view; `CodeTable` at 10^5 keys | unanswered |
| EXO-245 (medcare A2) | signature-only actions as method bindings | unanswered (W13) |
| EXO-246 (medcare A3) | guard lowering from an out-of-repo git consumer with one contract crate | unanswered; EXO-226 (no CI for the lab) |

## 6. Not verified

Nothing was built or run. §2 rests on the files it cites, read in this
session, and on the v2 census. The sources listed in §4 are unread.
