# D-DRDT-CLASS-GOVERNANCE-1 — Method resolution today

Status: census, docs only. Read at main 238b6d1e4 (2026-11-11).

This traces one method from registration to execution as the code stands.
Every step cites the source line read. Paths are under `crates/`.
`dt64` = `lance-graph-dt64/src/lib.rs` unless a file is named.
`binding` = `lance-graph-r2il/src/binding.rs`.
Matrix of components and statuses: [CURRENT-STATE.md](CURRENT-STATE.md).
Namespace rules: [NAMESPACE-CROSSWALK.md](NAMESPACE-CROSSWALK.md).

## Step 1. A class and its methods are registered

Entry points: `BindStore::register` (`dt64:814`) and `register_derived`
(`dt64:826`). Both end in `insert` (`dt64:843`). `register_derived` first
refuses an unbound `base` (`UnknownClass(base)`, `:834-836`) and `base ==
class` (`BehaviourImmutable`, `:837-839`).

Inputs:

| input | type | where |
|---|---|---|
| view | `&V: ClassView` | read for the field basis; never copied |
| class | `u32` (full classid) | `dt64:817` |
| schema | `ClassSchema { schema_rev: SchemaRev(u32), field_count: usize, extent: Extent }` | `dt64:254-258` |
| methods | `Vec<MethodBinding>` | `dt64:215-231` |

`MethodBinding` fields (`dt64:215-231`): `id: MethodId(u32)`,
`sig_rev: SigRev(u32)`, `name: String` (carried, "never used as identity"),
`on: WideFieldMask`, `when: Option<Guard>`, `reads: WideFieldMask`,
`effects: Vec<Effect>`, `population: Population`, `truth: Truth`,
`demand: ValueDemand`. All fields are `pub`.

- `Effect` (`dt64:146-154`): `target: u8` (field position written),
  `algebra: AlgebraRef`, `args: Vec<Arg>`. The effect list is ordered.
- `Arg` (`dt64:158-163`): `Field(u8)` or `Const(i64)`.
- `Guard` (`dt64:170-173`): `expr: SetExpr`, `logic: Logic`. Imported from
  the contract (`dt64:96`). The store keeps it and evaluates nothing
  (`dt64:11-14`).
- `ValueDemand` (`dt64:207-210`): `MapOnly` or `Values`.
- `Population` is `Distinct` or `Bag`; `Truth` is `TwoValued` or
  `ThreeValued` (imported `dt64:97`; arms read at `dt64:455-462`).

`insert` runs these checks in this order (`dt64:851-911`):

1. `validate` (`dt64:1310`): field count not over `MAX_FIELDS` (256);
   each extent axis within its cap; field count within the property
   reservation; method count within the predicate reservation; no two
   `MethodId` equal in the class; every position in ON, reads, effect
   targets and `Arg::Field` inside the field count; every `Arg::Field` read
   is in `reads` or written by an earlier effect.
2. For every effect: the algebra word's lens must be declared and the
   position inside it (`UnknownAlgebra`); the argument count must equal the
   lens's declared arity (`ArityMismatch`) (`dt64:852-870`).
3. Basis read through the view (`basis_at`, `dt64:1268`): a behaviour lens
   by full classid, any other class by its concept half. Unknown lens basis
   gives `UnknownLensBasis`.
4. Basis length equals `field_count` (`FieldBasisChanged`, `:872-878`).
5. `BasisDigest::of(fields)` (`dt64:304-320`): FNV-1a 128 over the ordered
   `predicate_iri`s.
6. Another classid read through the same basis at the same revision must
   hold the same digest (`BasisConflict`, `:881-892`).
7. One `binding_digest` per method (`dt64:411-468`): classid, basis digest,
   method id, signature revision, ON, reads, ordered effects (with each
   lens's revision and word arity), population, truth, demand. The name and
   the guard are not in it (`dt64:405-409`).
8. If the class is already bound: an equal entry is `Ok`; an unequal one is
   `BehaviourImmutable` (`:904-911`).

What `insert` does not check: who calls it, whether the classid is in the
`0xFFFF` block or any other range, whether an activation exists. I found no
such code in `lance-graph-dt64/src` (search for `0xFFFF`, `HARDCODED`,
`hotplug`, `Activation`, `withdraw`).

The store keeps per class: schema, methods, `derived_from`, basis digest, one
binding digest per method (`ClassEntry`, `dt64:273-285`).

## Step 2. A lens (word set) is declared

`BindStore::declare_algebra(lens: u32, schema_rev: SchemaRev, extent: Extent,
arity: Vec<u8>)` (`dt64:917-939`).

- A lens is "one classid whose fields are the polyfill's words"; `arity[p]`
  is the operand count of word `p` (`dt64:914-915`).
- Field count is `arity.len()` (`:926`). It goes through `validate` with no
  methods (`:929`), so it is bounded by `MAX_FIELDS = 256` (`dt64:270`,
  `:1310-1316`).
- `AlgebraRef { lens: u32, word: u8 }` (`dt64:129-132`). The word's name is
  the field label in the lens's `ClassView`, "resolved late"; "the store
  never resolves or runs the word" (`dt64:124-127`).
- It does not take a view and does not read a basis. A lens is checked later
  by revision and count only (`resolve_algebra`, `dt64:954-973`; comment
  `:961-965`: an equal-arity swap is not detected).
- Identical re-declaration is `Ok`; any other is `AlgebraImmutable` (`:931-938`).
- It does not check that `lens` is in the `0xF000..=0xFFFF` right-half range
  (`is_behaviour_lens`, `lance-graph-contract/src/class_view.rs:1053`). The
  function body (`:917-939`) has no such call.

Declared lenses in tree: `POLYFILL_LENS` `0xFFFF_F000` with 17 words
(`lance-graph-r2il/src/polyfill.rs:44`, `:116`; arities in
`lance-graph-r2il/dt64-authority-census.tsv`), `XLOG_LENS` `0xFFFF_F100`
with 3 words (`lance-graph-xlog/src/lens.rs:33`, `:47`, `:104`). A test
declares `0xFFFF_F200` as ONNX (`lance-graph-dt64/tests/bind_store.rs:1469`).
Consumer-shaped lens ids (`0x7F0B_F001`, `bind_store.rs:1498`) appear in
tests only.

Reading a lens basis: `ClassView::lens_fields(lens: u32)` (default `None`,
`class_view.rs:1084`) and `lens_fields_at(lens, schema_rev)` (`:1115`).

## Step 3. A `MethodRef` is issued

`BindStore::method_ref(class: u32, method: MethodId) -> Result<MethodRef,
Refusal>` (`dt64:1136-1157`). It can refuse `UnknownClass` or
`UnknownMethod`. It is "the only way to obtain a stamped reference"
(`:1134-1135`).

```rust
pub struct MethodRef {              // dt64:353-359
    pub class: u32,
    pub method: MethodId,
    pub sig_rev: SigRev,
    pub guard: Option<GuardWitness>,
    pub stamp: Stamp,
}

pub struct Stamp {                  // dt64:371-374
    issuer: Weak<()>,
    binding: u128,
}
```

- `issuer` is `Arc::downgrade(&self.identity)` (`dt64:1153`). `identity` is
  `Arc<()>` in the store (`dt64:779`). It is a pointer to one heap
  allocation in this process. Equality and hashing of a `Stamp` compare that
  pointer (`dt64:385-398`). A clone of the store gets `Arc::new(())`
  (`dt64:782-791`), so a clone refuses the original's references.
- `binding` is the stored binding digest for the method (`:1154`).
- `GuardWitness` is the guard's normal form under its logic (`dt64:180-198`,
  `Guard::witness`).
- No `serde` in `lance-graph-dt64` (its `Cargo.toml` `[dependencies]` lists
  only the contract). `MethodRef` derives `Debug, Clone, PartialEq, Eq,
  Hash` (`dt64:352`). A reference cannot leave the process by any code in
  this crate.
- Fields `issuer` and `binding` are private; a `compile_fail` doctest pins
  it (`dt64:367-369`). The other `MethodRef` fields are `pub`, so a caller
  can change `class`, `method`, `sig_rev`, `guard` on a copy (`admit` then
  checks them against the binding, step 4).

## Step 4. What `admit` checks

`BindStore::admit(&self, r: &MethodRef) -> Result<&MethodBinding, Refusal>`
(`dt64:1180-1212`). Order, as written:

| # | check | refusal | line |
|---|---|---|---|
| 1 | class is bound | `UnknownClass(class)` | `:1181-1184` |
| 2 | class has a method with this id | `UnknownMethod { class, method }` | `:1186-1190` |
| 3 | signature revision equals the bound one | `SignatureRevisionMismatch { class, method, bound, asked }` | `:1192-1199` |
| 4a | binding unguarded, reference has a guard | `GuardAdded` | `:1203` |
| 4b | binding guarded, reference has none | `GuardDropped` | `:1202` |
| 4c | both guarded, witnesses differ | `GuardMismatch` | `:1205` |
| 5 | stamp digest equals the STORED digest | `BindingMismatch { class, method }` | `:1207-1209` |
| 6 | stamp issuer is this store (`check_issuer`) | `ForeignStore { class }` | `:1210`, `:1161-1167` |

Eight refusal variants. Success returns `&MethodBinding` (`:1211`). The
doc states why digest comes before issuer (`:1174-1179`).

`admit` does not read the `ClassView`, the route log, the frame, or the
guard's value. `tests/bind_store.rs:481-488` pins the first as a finding.

## Step 5. What the R2IL path does after admission

Code: `binding::prepare` (`binding:321`), `binding::run` (`binding:377`),
`binding_native::prepare_native` (`:203`) and `run_native` (`:351`).

`prepare(store, view, lens: R2ilLens, r: &Resolution, frames, now)`, in
order:

1. `r.guards.check(frames, now)` (`binding:329`): frame, then focus, then
   scope (`dt64/src/dr.rs:543-553`). `Scope` is `{ id: u64 }`; the check is
   `self.scope != now.1` (`dr.rs:548`).
2. `check_route` (`:330`): the route at the captured version and now must
   both name the class (`binding:483-503`).
3. `store.resolve(view, class, schema_rev)` (`:331-333`).
4. `store.admit(r.method)` (`:334`).
5. `binding.when.is_some()` gives `GuardNotExecutable` (`:335-337`).
6. `binding.truth == ThreeValued` gives `ThreeValuedNotExecutable`
   (`:338-340`).
7. Lens gate (`:341-343`): the lens must equal `POLYFILL_LENS` **or** have a
   concept half equal to `ogar_r2il::CONCEPT_R2IL_FOLD`; otherwise
   `NotR2ilReading`. So the executor accepts two readings, not one. The
   module doc says the second is "provisional in OGAR: a lab lens"
   (`binding:30-35`). `XLOG_LENS` is neither, so it is refused here
   (`lance-graph-xlog/src/lens.rs:97-101` says the same).
8. Per effect, `opcode()` (`binding:542-604`): effect's lens must equal the
   chosen lens (`WrongLens`, `:550`); `resolve_algebra` (`:555`); word inside
   the basis (`WordOutOfBasis`). Then:
   - under `POLYFILL_LENS`, `frozen_opcode` (`:606-628`): the position binds,
     the view's label must be an alias of that position (`NotAnAlias`), the
     table arity must equal the argument count (`Arity`);
   - under the R2IL fold reading, the label must name exactly one opcode in
     `R2ILVocabulary` (`UnknownWord`), the opcode must be in `EXECUTABLE`
     (17 opcodes, `binding:100`; `NotExecutable`), OGAR's stack arity must
     equal the argument count (`Arity`), and it must push one result
     (`NoResult`).
9. Result is a `Prepared` with a `plan_id` (`binding:356`, `:511-540`) that
   hashes the method reference (including `stamp.binding()`), the lens and
   the steps.

`run(plan, store, frames, now, permit: &ExecPermit, fields, dialect)`
(`binding:377-399`):

1. `store.check_issuer(&plan.method)` (`:387`).
2. guards (`:389`), then route (`:392`).
3. `permit.methods.contains(&plan.method)` else `NotPermitted` (`:393-394`).
4. `run_steps` (`:397`).

`ExecPermit` is `{ methods: Vec<MethodRef> }` (`binding:128-130`). Its doc:
"This is a lab permission, not a production authorization: who issues it is
not decided. `BindStore::admit` succeeding is never a substitute for it"
(`binding:121-124`). Anyone can build one by cloning a `MethodRef`.

`binding_native` repeats the issuer, guard, route and permit checks
(`:355-369`) and then compiles the folds once. `prepare_native` refuses a
fold result that feeds a fold (`FoldResultFeedsFold`, `ScalarAfterFold`) and
more than 8 outputs (`MAX_NATIVE_OUTPUTS`, `binding_native.rs:61`).

Not on this path: the `WHEN` guard is refused, not run. The only code I
found that evaluates a guard over a population is in lab crates
(`foundry-parity-lab/src/method.rs:200`, via `lance-graph-relalg`
`normalize.rs:128`; both are in the root `exclude` list).

`lance-graph-xlog::bag_count_admitted` (`lance-graph-xlog/src/lib.rs:384`) is
a second execution entry for a method. It takes `&MethodBinding`, not a
`MethodRef`, and does not call `admit`, `check_issuer` or `ExecPermit`.

## Step 6. The five states, and what represents each

| # | state in the governance brief | type or function on main | file:line |
|---|---|---|---|
| 1 | reserved namespace | `HARDCODED_BEHAVIOUR: u16 = 0xFFFF` and its domain byte; a guard test that no codebook row lands in `0xFFXX`; OGAR mirror. `is_behaviour_lens(classid)` tests only that the right half is at least `0xF000`; it reserves nothing. | `lance-graph-contract/src/ogar_codebook.rs:583`, `:586`, `:925-943`; `class_view.rs:1053`; `/home/user/OGAR/crates/ogar-vocab/src/lib.rs:1879` |
| 2 | registered technology + method | **Technology: NONE.** No type for a technology id. The `F1__`-style split exists as a decision (plan `d-dt64-7-...` §1d O7) and in one test mask (`lance-graph-xlog/tests/lens.rs:11`, `classid_custom(XLOG_LENS) & 0xFF00 == 0xF100`). A committed table lists the two production lenses: `lance-graph-r2il/dt64-authority-census.tsv`, checked by `tests/integration/dt64_census.rs`. **Method:** a class method is `MethodBinding` under `MethodId(u32)` in `BindStore::register`. | `dt64:814`, `:215`, `:113` |
| 3 | contextualised behaviour lens | `BindStore::declare_algebra` plus a `ClassView` that answers `lens_fields` for the full classid; read with `resolve_algebra`. In production code the only such views are `PolyfillView` and `WithPolyfill`. A lens under a consumer's own left half exists in tests only. | `dt64:917`, `:954`; `class_view.rs:1084`; `lance-graph-r2il/src/polyfill.rs:231`, `:268`; `lance-graph-dt64/tests/bind_store.rs:1498` |
| 4 | admissible `MethodBinding` | `BindStore::admit` returns `&MethodBinding` after the eight checks in step 4. No separate type marks a binding as admitted. | `dt64:1180` |
| 5 | authorized execution | **NONE.** No type or function in the DT64, R2IL, XLOG, Quack or mask-RISC crates represents it. Where authorization lives instead is listed below. | see below |

Where authorization lives today (all read):

- `lance-graph-contract/src/rbac.rs`: `Operation::{Read, Write, Act}`
  (`:347-366`, `Act { action: &str }` at `:359`); trait `ClassRbac`
  (`:371`); `OpMask` and `ClassGrant` (`:454`, `:510`); `grants_permit`
  (`:548`).
- `lance-graph-rbac/src/authorize.rs`: `authorize` (`:66`),
  `authorize_scoped` (`:210`), `ScopedDecision` (`:170`).
- `lance-graph-contract/src/action.rs:88-90`: `ActionDef.required_role:
  Option<&'static str>`, "Checked against `ActorContext` at commit".
  `ActionDef` has no `MethodId` field (`action.rs:79-94`).
- `dt64/src/dr.rs:520`: `Scope { id: u64 }`, "What the reader is authorized
  to see. Policy, checked as its own guard; never part of frame identity"
  (`dr.rs:517-519`). Compared by equality (`dr.rs:548`). Also
  `LaneAddr.scope` and `LaneRefusal::Scope` (`lane.rs:36-41`, `:126-130`).
- `binding.rs:128` `ExecPermit`: lab only (quoted above).
- `dt64-admission-probe/src/lib.rs:163` `ExecutionPolicy::decide`: lab,
  excluded crate, "nothing depends on it" (its README).
- Link between them: none found. `lance-graph-dt64`, `-r2il`, `-xlog`,
  `-quack`, `-mask-risc` have no `rbac` in `Cargo.toml` or `src`.
  `.mermaid/06-authorization.md:8`: "Nothing yet applies it to a DT64
  binding."

## Step 7. 8-bit vs 32-bit: every declared width

Method and word identity:

| what | width | where |
|---|---|---|
| `MethodId(pub u32)` | 32 | `dt64:113` |
| `SigRev(pub u32)`, `SchemaRev(pub u32)` | 32 | `dt64:118`, `:122` |
| `AlgebraRef.lens` | 32 (a full classid) | `dt64:130` |
| `AlgebraRef.word` | 8 | `dt64:131` |
| `Effect.target`, `Arg::Field(u8)` | 8 (field position) | `dt64:148`, `:160` |
| `Arg::Const(i64)` | 64 | `dt64:162` |
| `MAX_FIELDS` (basis width) | 256 positions, `usize`; positions are `u8` | `dt64:270` |
| `BindStore::arity` returns `Option<u8>` | 8 | `dt64:943` |
| lens arity list `Vec<u8>` | 8 per word | `dt64:917-923` |
| `Resolved::action_ordinals` yields `MethodId` | 32 | `dt64:760` |
| `FnIndex(pub u8)` (OGAR opcode); `Step.opcode: FnIndex`; the opcode scan `(0..=u8::MAX).map(FnIndex)` | 8 | `/home/user/OGAR/crates/ogar-loco/src/lib.rs:380`; `binding:145`, `:526`, `:570` |
| `XlogWord.columns: u8` | 8 | `lance-graph-xlog/src/lens.rs:39` |

Class and route identity:

| what | width | where |
|---|---|---|
| classid | 32 = `[canon u16][custom u16]` | `ogar_codebook.rs:281`, `:464-491` |
| `ClassId` (ClassView key) | 16 | `class_view.rs:54` |
| behaviour lens test: right half `>= 0xF000` | 16-bit half | `class_view.rs:1054` |
| `AppPrefix::prefix() -> u16` | 16 | `ogar_codebook.rs:338` |
| route position | `u16` | `dt64:1018` |
| `RouteVersion(pub u64)` | 64 | `dt64/src/dr.rs:156` |
| `Extent` reservation cap `MAX_RESERVED = 1 << 16` | 16-bit position space | `dr.rs:151` |
| `FieldMask(pub u64)`; positions at 64 or above are dropped | 64 | `class_view.rs:70`, `:77` |
| `WideFieldMask` positions | `u8` (cap 256) | `class_view.rs:281`, `:697` |
| `ValueRow.position`, `value` | 8, 8 | `class_view.rs:1527`, `:1529` |
| `LaneAddr.lane: u16` | 16 | `lance-graph-dt64/src/lane.rs:36-41` |
| `FieldLanes.lanes: &[Option<u16>]` | 16 | `binding:113` |

Action addressing outside this repo (sibling OGAR checkout):

| what | width | where |
|---|---|---|
| `ActionRef.ordinal` | 32 | `/home/user/OGAR/crates/ogar-render-askama/src/field_view.rs:77` |
| `ActionInvoke.action_ordinal` | 32 | `/home/user/OGAR/crates/ogar-a2ui-frame/src/lib.rs:142` |

Observations (facts only): the only 8-bit identity attached to a lens word is
`AlgebraRef.word`. The only 8-bit identity attached to a class method is
none; a class method is `MethodId(u32)`. The 8-bit `FnIndex` is an OGAR
opcode identity held beside the DT64 position (census row "FnIndex opcode",
status LAB, `dt64-authority-census.tsv`). The right half of a classid is 16
bits and no code splits it further.

## Open questions found while reading

- `declare_algebra` and `register` accept any `u32`. No code ties a
  `0xFFFF_F###` id to a registrant (plan C1, C2 are tests not yet written;
  plan §8 D-DT64-7-2, -4).
- `ExecPermit` has no issuer; it is `Vec<MethodRef>` and `MethodRef: Clone`.
- `bag_count_admitted` and `binding::run` are two execution entries with
  different gates: the first takes `&MethodBinding` and checks nothing about
  admission; the second needs issuer, guards, route and a permit.
- The R2IL executor accepts lenses by concept half `CONCEPT_R2IL_FOLD`, not
  by a technology nibble. How that reading relates to the `F###` right-half
  scheme is not stated in any file I read.
- `Stamp.issuer` is a process-local pointer. No file I read says how a
  `MethodRef` would cross a process or a restore. `MethodRef` has no
  serialization in `lance-graph-dt64`.
- A `Guard` cannot be executed by any member crate (step 5). `prepare`
  refuses it, and the only evaluators found are in excluded lab crates.
- Plan `d-dt64-7-...` O1, O3, O4, O6 are listed OPEN (§7). I found no code
  that decides them.
