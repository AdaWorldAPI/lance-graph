# D-GPW-0..6 — Global pattern words: one word with a hole, its two dimensions and its absent-case policy as parameters

> **Draft / documentation only, 2026-11-19.** This file authorizes no Rust
> code, no word in any lens, no crate and no change to DT64, R2IL or Quack.
> Each step below needs its own PR with a parity test and a disable run.
>
> **Origin:** the owner, 2026-11-18 and 2026-11-19, recorded verbatim on
> EXO-329:
> - "SQL WHERE is a polyfill in F### space, never datafusion literals, not if it's our code"
> - "The above are SQL methods, not Codebook entries"
> - "If we own Zammad-rs, Why should we settle with strings instead we should go with methods"
> - "I understand LIKE 'a.%' is similar to MeTTa and spaCy?"
> - "The idea is to introduce these as global patterns and spaCy and MeTTa as 2 dimensions of the same issue"
> - "Similar to NaN handling parameters"
>
> **The reading in §1 is this session's. The owner has not confirmed it.**
> Status of every claim below is marked.

## 0. Problem

A consumer that inherits a text match has three unrelated mechanisms for one
question, and picks among them itself.

MEASURED in zammad-rs (PostgreSQL 16.15 as oracle):

- Zammad asks for "a permission and everything under it" with
  `name = ? OR name LIKE 'x.%'`: 57 of 512 SELECTs of the recorded workload.
  The tree exists only as dots in `permissions.name`.
- zammad-rs #41 replicated it: a SQL fragment reader of its own, and `LIKE`
  matched against the column's strings.
- zammad-rs #45 replaced it: the intake declares the tree as a relation of
  identities (computed with Zammad's own `Permission.with_parents`), and a
  consumer-local condition `under` answers over identities. All 57 answers
  equal PostgreSQL.
- **The two forms disagree in one case.** After the row `a` is deleted,
  `name = 'a' OR name LIKE 'a.%'` still returns `a.b.c` in PostgreSQL; the
  relation form returns nothing, because a name with no row is no member.
  zammad-rs chose the identity reading and states the difference in a test.
  A consumer should not be the place that choice is made.

The same choice appears twice more in the same work:

- A text of a known type either names a member of a declared set or it does
  not (the owner's `ValidateSet`): refuse it, or treat it as no member.
- A comparison against NULL: two-valued or Kleene. This one already has a
  home, `Logic` on a `Guard` (`lance-graph-contract/src/set_algebra.rs:195`).

## 1. Reading (WORKING-MODEL, unconfirmed)

**A pattern is one global word with a hole.** Its dimensions and its handling
of the absent case are parameters of the word, the way rounding and NaN
handling are parameters of the 8-bit codec.

VERIFIED-IN-CODE, the precedent the owner named: `ndarray/src/hpc/codec8.rs`.
`Codec8<T, S, R, N>` is one operation. `R: Rounding` and `N: NanPolicy<S>` are
independent marker parameters. `NoNaN` has no code for NaN and the checked
encoder rejects it; `NaNAtMin` reserves the code -128 to mean NaN. The
operation is not duplicated per policy, and the policy is fixed where the
codec is named, never chosen per value.

| Parameter | What it decides | spelled today as |
| --- | --- | --- |
| **within** a value (the spaCy dimension) | how a value is split into parts, so which spans a hole may cover | `Permission.with_parents` in a Ruby intake; the `%` and `_` of `LIKE` |
| **across** values (the MeTTa dimension) | which relation a hole binds through, so which members a variable may take | a declared relation and a key set in zammad-rs; `Space::query` in Hyperon |
| **the absent case** (the NaN of matching) | what a name with no member means: no code for it, or a reserved one | implicit, and different in each mechanism |

`LIKE 'a.%'` is the flat-string form: it does both dimensions at once without
naming either, and its absent-case policy is "a name needs no member".

## 2. What already exists (searched before proposing)

- VERIFIED-IN-CODE. The global lens: `lance-graph-r2il/src/polyfill.rs`,
  `POLYFILL_LENS = 0xFFFF :: 0xF000`, 17 words. Position binds; labels are
  aliases. One position has a name per language
  (`AliasLanguage::{R2il, Sql, Quack}`). No word has a hole: `=`, `<`, `AND`,
  `OR`, `NOT`, the folds. `LIKE` and `IN` have no position.
- VERIFIED-IN-CODE and MEASURED there: `.claude/plans/metta-polyfill-v1.md`
  and its umbrella `metta-datalog-assimilation-v1.md` (D-ASM). MeTTa as a
  front end over the masking-algebra polyfill, attached at
  `Space::query(&Atom) -> BindingsSet`. The crate `crates/lance-graph-metta`
  EXISTS (its own workspace root; `src/lib.rs` plus three probes; D-MTP-1 and
  D-MTP-1b measured, the `Space` adapter D-MTP-2 queued). The plan's own
  header still says "No crate exists yet"; an earlier version of this file
  repeated that without opening the crate. This plan builds on it for the
  across dimension; it does not restate it.
- VERIFIED-IN-CODE: `crates/lance-graph-vrl` (workspace member; the former
  `vrl-recipe-probe` lab, moved 2026-11-25). It lowers the read / rename / filter subset of Vector's VRL onto
  DT64 lane binding and mask-risc, checked against a naive interpreter, with
  a parity test through the frozen `F000` words. It refuses every function
  call but `del`, so `starts_with(.name, "a.")` and `split(.name, ".")`, VRL's
  spellings of this plan's pattern and of its within dimension, are refused
  there today. Owner, 2026-11-19: "If spacy is rust it might be valuable
  along with VRL". spaCy is Python with a Cython core, not Rust; VRL is Rust.
  VRL is a fourth alias language for the word, and that probe is the place
  to extend.
- CURRENT-CONTRACT, D-ASM-10: MORK runs as an oracle outside the workspace
  (`crates/mork-pathmap-probe`: "results only; no Lance storage, no
  production executor").
- The name "Ruler" is taken twice: `dr-mathematical-compiler-v1.md:41` uses
  "Ruler/Enumo" for cold rule synthesis ahead of egglog, and ndarray ships
  `CurveRuler`.
- CURRENT-CONTRACT:
  `.claude/plans/d-cgmm-lazy-codegraph-fact-demand-v1.md:125`. A spaCy-like
  `Doc/Token/Span/Matcher` pipeline is "a useful cold annotation interface
  analogy, not a second crate or text-sequence representation". This plan
  keeps that: the within dimension is a parameter of a word, never a second
  representation of text.
- CURRENT-CONTRACT, D-DT64-7 §1b: "A language that only renames the shared
  algebra gets an alias view, not a concern range." So SQL, spaCy and MeTTa
  spellings of one pattern word are alias views.
- VERIFIED-IN-CODE. `Logic::{TwoValued, Kleene}` is already an absent-case
  parameter, for truth, fixed at registration on the `Guard`.

Not searched: whether `lance-graph-nal`, `deepnsm` or
`grammar-constraints-probe` already hold a token or span type that the within
dimension should reuse. D-GPW-1 is that census.

## 3. Rules proposed

- **R1. One word, parameters, no per-language word.** `LIKE`, a spaCy token
  pattern and a MeTTa `match` over the same relation are names for one
  position. A new position is added only when no parameter choice of an
  existing one expresses the operation (the existing ISA rule).
- **R2. The absent-case policy is declared with the class field, at intake.**
  Never at the call, never by the consumer's code path. It joins `Logic` as
  the same kind of declaration ("the injecting intake defines the class
  reading").
- **R3. Text stops at the boundary.** A pattern's constants are members
  (codebook positions) by the time the word runs; splitting a value into
  parts happens at intake. A word that needs the text of a value at run time
  is refused.
- **R4. A consumer does not imitate a missing word.** Until a pattern word
  exists a consumer keeps its current path, named as "no word", and does not
  expand the pattern into other words.
- **R5. Every parameter choice has a differential oracle.** For `LIKE` that
  is PostgreSQL; for a MeTTa match the engines measured in D-MTP-0.

## 4. Deliverables

| D-id | What | Gate |
| --- | --- | --- |
| D-GPW-0 | This plan and the decisions in §5 | owner |
| D-GPW-1 | Census, counted from source with the command per number: every place lance-graph and its consumers match a pattern today (SQL `LIKE`/`IN`, the cube's `reach`/`closure`, `lance-graph-nal`, `deepnsm`, `grammar-constraints-probe`, zammad-rs `under` and its fragment reader), with the within form, the across form and the absent-case behaviour of each | every row read, not grepped; the three zammad-rs rows reproduce the numbers in §0 |
| D-GPW-2 | The absent-case policy as a declared parameter beside `Logic`: its variants and where it is stated | the zammad-rs deleted-parent case answered both ways from one declaration, each equal to its oracle (PostgreSQL for the prefix reading; the relation's own definition for the member reading); a disable run |
| D-GPW-3 | The across dimension: one pattern word over a declared relation, in the lens the owner decides (§5.1), with its SQL and MeTTa alias names | equals zammad-rs `under` on the 57 recorded lookups and equals D-MTP-0's oracle on its join program; a name that is no alias is refused |
| D-GPW-4 | The within dimension: how a value's parts are declared at intake so a hole can cover a span, reusing whatever D-GPW-1 finds | `LIKE 'a.%'`, `LIKE 'a_'` and an escaped pattern answered without text at run time, equal to PostgreSQL |
| D-GPW-5 | zammad-rs: `under` and the fragment reader's `LIKE` replaced by the word; its own paths kept only as the oracle | all 512 SELECTs still equal PostgreSQL |
| D-GPW-6 | The existing `lance-graph-metta` front end (D-MTP-2 / D-ASM-4) lowering `match` onto the same word | D-MTP's and D-ASM's own gates |

## 5. Decisions needed

1. **Which lens.** The hardcoded lens `0xFFFF :: 0xF0__` (every consumer gets
   the word; D-DT64-7 puts SQL there) or a consumer's own
   `Domain::AppId :: 0xF###`. Proposed: the hardcoded lens.
2. **The absent-case variants.** Proposed, by analogy with `NoNaN` /
   `NaNAtMin`: *no member* (an unknown name is refused or matches nothing)
   and *reserved member* (a name with no row still has a position to stand
   on, which is what a prefix match needs). Whether refusal and
   matches-nothing are two variants or one.
3. **Whether `Logic` and the absent-case policy are one parameter or two.**
   Both say what an absent thing means. Proposed: two declarations of one
   kind, kept apart until a case needs them merged.
4. **Call parameters.** A pattern's constants come with the call. A guard
   carries none (EXO-291, with zammad-rs's count: 53 conditions with
   constants for 18 shapes). D-GPW-3 cannot register a pattern once per shape
   without it.

## 5a. Where it lives (owner's options, 2026-11-19, verbatim)

> Eg Lance-graph-ruler global Or 3 different dialects Lance-graph-spaCy
> Lance-graph-mork Lance-graph-metta

Recommendation of this session, not a decision:

- **One global home for the word and its parameters.** A dialect is a set of
  names for the same positions (D-DT64-7 §1b), so three dialect crates would
  each need the word and would differ only in spelling.
- **A dialect gets a crate only where it has a front end of its own to
  hold.** MeTTa has one (a text language and Hyperon's `Space` seam) and the
  crate already exists. spaCy's and SQL's spellings are alias views in the
  global home, which also keeps the existing rule that a spaCy-like pipeline
  is not a second text representation. MORK stays an oracle (D-ASM-10)
  unless that decision is changed.
- **The name.** `lance-graph-ruler` reads well against "holding a ruler
  against a stack", and collides with Ruler/Enumo and `CurveRuler`. Either
  accept the collision knowingly or pick another name.

### 5b. A third option (owner, 2026-11-19, verbatim)

> Or just inside quack Easy and simple

**This session's recommendation now.** It needs no new crate and no new layer:

- VERIFIED-IN-CODE. Quack already has both ends. `bind.rs` is its text
  frontend, the one file where a name or a literal may appear
  (`tests/string_fence.rs` keeps text out of everything else). The executable
  vocabulary already has membership (`Filter::in_u32`) and the relation hop
  (`Filter::Semijoin`, `Filter::EqU32Via`).
- So the pattern word is a `Filter` leaf, the within dimension is resolved in
  `bind.rs` (a pattern's text becomes members there and never passes it), and
  the across dimension lowers to the hop leaves that exist. R3 of this plan
  is then Quack's existing fence, not a new rule.
- The global name stays where names live: a position in `POLYFILL` with its
  SQL, Quack and MeTTa aliases. Quack holds the reading, the lens holds the
  address, as for `=` and `AND` today.
- `lance-graph-metta` already lowers through Quack, so D-GPW-6 needs nothing
  new on that side.

Two things this option does not remove:

- `Semijoin` and `EqU32Via` are not yet admitted for a consumer's map lanes
  (EXO-140). The across dimension inherits that gate.
- The absent-case policy (§5.2) still has to be declared somewhere. Inside
  Quack it would be a parameter of the leaf, set at bind from the class
  field's declaration.

## 6. Not verified by this session

- Hyperon and PeTTa were not run here; §2 quotes metta-polyfill-v1.
- spaCy was not run or read here.
- Whether an absent-case policy can be a compile-time marker, as in
  `Codec8`, or must be a registered value like `Logic`: not examined.
