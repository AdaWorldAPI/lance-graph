# The classid is the call: ON :: DO, WHEN at registration, relations through the class view

> READ BY: any session that proposes a registry, a store or a table for
> methods, guards, relations or "behaviour"; any session that binds a consumer
> (odoo, Rails / zammad-rs, HIRO / bardioc, OGIT) to DT64; any session that
> writes a join, a foreign-key read or a "gather through" path; every reviewer
> of such a PR; `layer-boundary-warden`, `convergence-architect`,
> `prior-art-savant`, `premise-auditor`.
>
> Status: CURRENT-CONTRACT for the addressing reading (§1 to §4), each line
> carrying its own evidence grade. §5 is a WORKING-MODEL goal with nothing
> measured, beside what already exists for it. §7 lists what one session got
> wrong before it read what existed; it is here so the next session does not
> repeat it.
>
> Born 2026-11-18 from the EXO-326 / EXO-291 / EXO-328 thread and PRs #2132,
> #2133, #2134 and zammad-rs #35, #36. The decisions are the owner's, quoted
> verbatim; the readings are the writing session's and are marked as such.
> Canonical plan for the registration side:
> `.claude/plans/d-dt64-7-behaviour-addressing-registration-v1.md`. Read that
> first. This file adds no decision to it; it connects it to consumers.

## 0. Thirty-second version

- A classid has two sides. **The left side is ON** (the class a behaviour
  applies on). **The right `0xF###` is DO** (the behaviour). Nothing stores an
  ON beside the classid.
- **WHEN is what the `register` call supplies** for that classid. Its logic
  (two-valued or Kleene) is fixed there and nowhere else.
- **The `BindStore` is the registry.** It already joins the DT side (the class
  view) with the DR side (the field mask). Do not propose a second one.
- **There is no foreign key.** A relation reached through classid and ontology
  is a class view. The intake declares how a class is read; a query only uses
  what was declared.
- **The goal is a call shape**, as direct as a typed SIMD primitive, over a bag
  the cube lends. The lending half exists (`LaneBinding::borrow_all`); read §5
  before proposing any of it again.

## 1. The two sides of the address

| Side | Owner | What it is | In code |
| --- | --- | --- | --- |
| DT | the class view | what a classid's fields are and what they are called | `ClassView` (`lance-graph-contract/src/class_view.rs`); `lance-graph-ontology` implements it (`class_resolver.rs:104`, `RegistryClassView`) |
| DR | the field mask | which field positions, placed where | `WideFieldMask`; `LaneRequest.fields` (`lance-graph-dt64/src/lane.rs:150`); the route log (`BindStore.routes`) |

DECISION (owner, 2026-11-18): "DR > widefieldmask / DT > classview".
BASIS: VERIFIED-IN-CODE. `BindStore::resolve` reads a class's fields through a
`ClassView`, never copied (`lance-graph-dt64/src/lib.rs:1099`); a method's
reads and writes are `WideFieldMask`s (`lib.rs:222-248`).

Consequence for a caller: it names a class (by its route) and a field mask. It
does not name lane numbers or plane numbers. A lane source maps the field the
class view resolved (its `FieldRef`), never a position: positions shift when a
name is added (stated for OGIT by the HIRO session, where ids are row
positions of the files as read; not read by the writing session).

## 2. ON and DO are the classid; WHEN is the registration

DECISION (owner, 2026-11-18), verbatim, in the order given:

> the classid is already the answer to ON

> F###-F### for methods includes both ON and DO and when is a register call of the method

> ODOO[timesheet]-F### = on Odoo/timesheet > DO F###

> 0xFFFF :: 0xF### = on the FFFF global algebra primitive DO F###

| Classid | ON | DO |
| --- | --- | --- |
| `<Domain::AppId> :: 0xF###` | that app's class | behaviour `F###` |
| `0xFFFF :: 0xF###` | the global algebra primitive | behaviour `F###` |

BASIS: CURRENT-CONTRACT in D-DT64-7 §2 ("left (high 16): concept :: right (low
16): ClassView / WideFieldMask selection within it"), D11'' (hardcoded
behaviour is `0xFFFF :: 0xF###`), D14'' (consumers register behaviour through
`0xF###` on the right under their own `Domain::AppId`). VERIFIED-IN-CODE:
`is_behaviour_lens` (`class_view.rs:1053`), right half `0xF000..=0xFFFF`.

What follows:

- **`0xFFFF` is not an empty left side.** The algebra is the class the
  behaviour applies on.
- **A model with five callbacks is one left side with five right-side
  behaviours**, each registered with its own guard. It is not five registry
  entries with a trigger each.
- **A guard's logic is part of immutable behaviour.** Re-registering a class
  with anything different is `BehaviourImmutable`. Two-valued → Kleene is a
  derived classid plus a re-route. TEST-PINNED on the HIRO side (stated);
  VERIFIED-IN-CODE here (`lib.rs:925-934`).
- **OPEN:** `MethodBinding` carries an `on` field mask ("the field positions
  whose change triggers the method"). Under the owner's statement it repeats
  what the route already says. Removing it changes every binding digest, so it
  is a decision, not a cleanup. Nobody has ruled on it.

## 3. Registration, and what is not registered

All of this is D-DT64-7. It is repeated here because one session proposed the
opposite without having read it.

| Question | Answer | Where |
| --- | --- | --- |
| Where does registered behaviour live? | In the `BindStore`. Learned = registered | D2 |
| Who holds the registration authority? | DT64 for `0xFFFF`; a consumer under its own `Domain::AppId` | D13'', D14'', D15'' |
| Where does known-but-not-executable behaviour live (a mined rule, a harvested hypothesis)? | **Outside the store.** Explore = not yet registered. There is no lifecycle state besides versioned, replayable immutability | D2, §3 |
| How does it become executable? | By `register`. That call is the promotion | §3, constraints C1 to C8 |
| Who supplies the labels? | The consumer supplies its own `ClassView`; registration ties owner and activation to a versioned basis witness | O10 |
| Is a second authoritative registry allowed? | No. Listed under "Not introduced" | §1d, execution order |

Still OPEN, owner decisions (EXO-291):

- Whether the effect class (`ogar-vocab`'s `EffectClass`: `Validation`,
  `FieldTransform`, `LifecycleWrite`, `TransactionEffect`, `ExternalHost`, …)
  is recorded on the binding.
- Where call parameters travel. `ActionInvocation` carries none
  (`lance-graph-contract/src/action.rs:185-208`).

## 4. Relations: no foreign key, the intake declares the reading

DECISION (owner, 2026-11-18), verbatim:

> as long as something is use via classid and ontology, nothing is a foreign key, its just a classview

> spoG is always via classview

> hence there is no FK in our architecture

> the injecting intake defines the class reading

> exactly the classview already did the decision before the query

Reading (the writing session's):

- A relation between two classes is an SPO-G edge read through the class view.
  A column of keys is only how a source system happens to store the edge.
- **Whoever injects the data declares how the class is read.** A consumer of a
  read does not choose a reading afterwards. In particular, where the same
  pair can occur twice (a Rails join table with no unique index), the intake
  declares it a class of occurrences; nobody bolts a count onto a set at query
  time.
- **A query looks a relation up. It does not infer one** from the shape of an
  `ON` clause.

Evidence so far, all in a consumer (zammad-rs, PostgreSQL 16.15 as oracle):

- MEASURED: with `Filter::eq_u32_via` and `Filter::semijoin` taken off the
  default join path, 12 captured joins and 57 joins of a real workload keep
  their answers, with 0 gather-class leaves built (zammad-rs #35).
- MEASURED: with relations declared at intake and looked up by the join, 296
  tests pass and every answer pin is unchanged; an undeclared pair is refused
  by name (zammad-rs #36).
- FINDING: `ogar-vocab`'s `Association` carries a `belongs_to`'s key column and
  target, but for a `has_and_belongs_to_many` only a name: no join table and
  neither of its columns (read at OGAR `75bb38b`). A consumer must declare
  such relations through its own intake until that is carried upstream.

Not settled by this, and still EXO-140: whether `EqU32Via`, `Semijoin` and
`Gather` are admitted on a query path for a relation the ontology declares.
The statements above were not read as admitting them.

OPEN in lance-graph: the cube lends a pair plane per relation (#2131), but its
row-returning and counting terminals work on one tile (at most 256 ids across
all relations) and a cube side cannot filter on a property of the joined row
(EXO-327, unanswered).

## 5. The goal: a call shape

DECISION (owner, 2026-11-18), verbatim:

> goal is to make algebra behavior work like
> ndarray::simd::bf16::add_mul()
> riscmask::[ON]::[DO]()
> xlog::cube::{Virtual Tuple BAG as a thinking object > borrow_all}

Reading (WORKING-MODEL, nothing measured):

| Line | Reading |
| --- | --- |
| `ndarray::simd::bf16::add_mul()` | a primitive is reached by a path naming its type and its operation; no lookup object and no setup stand between caller and fold |
| `riscmask::[ON]::[DO]()` | a behaviour is called the same way, with the two sides of its classid as the path |
| `xlog::cube::{… > borrow_all}` | what a behaviour runs over is a bag of virtual tuples the cube lends whole; nothing is materialized |

**What already exists (VERIFIED-IN-CODE, 2026-11-18).** The lending half of
the goal is built; do not propose it again.

- `LaneBinding::borrow_all` (`lance-graph-dt64/src/lane.rs:539`): one check,
  then every bound lane in ascending field order; N source lookups, nothing
  allocated; any refusal is returned before a lane is borrowed.
  `borrow_all_present` (`lane.rs:565`) lends each lane's presence plane with
  it.
- **A registered method executed from its `MethodRef` over lent lanes exists**:
  `foundry-parity-lab/src/method.rs`. It admits the reference, binds exactly
  the method's `reads`, lowers the guard once per binding in `prepare`, and
  `run` only executes the kept program; the lanes come from `borrow_all`
  (`method.rs:229`).
- Other callers of `borrow_all` / `borrow_all_present`: `quack-boundary-lab`,
  `vrl-recipe-probe` (`lib.rs:420`), `xlog-coverage-probe/tests/group_by.rs`
  (one `borrow_all`, one `GroupReduce`).
- The owners that lend lanes today (`impl LaneSource`):
  `lance-graph-mailbox-lanes::MailboxLanes`, `lance-graph-report::AbiLanes`,
  and lab sources.

**What is actually missing, as far as it was read:**

- **The cube lends relations, not lanes.** `Cube` implements `RelationSource`
  (`lance-graph-xlog/src/cube.rs:1618`) and no `LaneSource`. So a virtual
  tuple bag of the cube cannot be taken with `borrow_all` yet. That is the gap
  the third line of the goal names; the mechanism on the other side of it is
  finished.
- **No production caller.** Every `borrow_all` caller above is a lab, a probe
  or a test.
- Not read: how much of r2il's call-site ceremony (resolution, route, guards,
  permit, field-to-lane map, dialect, gate) `foundry-parity-lab::method`
  already avoids. Read it before claiming the ceremony is necessary or
  removable.

Next step that would turn this into evidence: read `foundry-parity-lab/src/method.rs`
in full as the existing call shape; then decide whether the cube becomes a
`LaneSource` for its bag, and probe one hardcoded and one consumer behaviour
called as `on :: do` over `borrow_all`, gated on the same answer as the direct
mask-risc program, in nanoseconds, with nothing allocated per call.

## 6. Three-valued guards

- TEST-PINNED (#2132): `prepare_gated` admits a `Logic::Kleene` guard; the gate
  plane holds its TRUE rows; a row where it is UNKNOWN is not in the plane. The
  guard is read from the bound method, its fields are requested as a
  `WideFieldMask`, and lanes and presence planes are lent by
  `borrow_all_present`. A field whose source states no presence is
  `PresenceUnbound`, never read as all present.
- Still refused: every guarded plan on the native path; a method whose result
  is three-valued; the UNKNOWN rows as a result of the seam.
- WORKING-MODEL (owner, 2026-11-18): three-valued truth is where behaviour is
  recovered from evidence and not declared, which is the work of
  `lance-graph-arm-discovery`, ruff and a text front end. Unknown must stay
  unknown there, or "could not tell" silently becomes "does not apply".
  Nothing in those crates was run for this.

## 7. What one session got wrong, so the next does not

Each line is a mistake made on 2026-11-18 and the check that would have caught
it.

| Mistake | Check |
| --- | --- |
| Proposed a `Method` row in the ontology cache: a second registry | Search `.claude/plans/` for an existing registration plan before proposing one. D-DT64-7 forbids it in so many words |
| Then proposed a not-executable state inside the `BindStore` | Read D2: Explore lives outside the store |
| Treated ON as something a method declares, and added an "event form" for it | Read D-DT64-7 §2: the classid's left side is the answer |
| Wrote a test that handed quack lane and plane numbers | A caller names a class and a field mask. If the test names a lane, it is testing the wrong seam |
| Took a foreign-key column on the query's say-so | Ask who declared the relation. If nobody did, refuse |
| Kept a count beside a set to model duplicates | Ask what the intake declared the class to be |
| Wrote that nothing lends a bag to a method, without searching for `borrow_all` | Grep the socket's callers before stating a gap. `borrow_all` existed, with a method executed over it in `foundry-parity-lab` |
| Read "vector" as a position array | It is Vector the product and its data model, a precedent for the SoA ABI substrate with DT and DR. It is never a position lane and never VRL |

The common shape: each time, the answer already existed one level up (in the
address, in the class view, in an earlier plan) and a new mechanism was
proposed one level down.
