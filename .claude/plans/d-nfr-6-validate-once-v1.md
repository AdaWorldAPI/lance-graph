# D-NFR-6 — validate a binding once, then execute without repeated guards

> ⊘ **SUPERSEDED by `d-nfr-6-soa-borrow-v2.md` (2026-10-23).** The `Table` below kept its own `FrameStore` beside the SoA and checked a frame per access, re-deriving ownership that `SoaEnvelope`'s borrow already gives (views are `&self`, the owner's `&mut` cannot start while one is live). The crate was reworked into `crates/soa-view-probe`; D-NFR-6a is withdrawn. The measurements below remain the record of what the guard cost.

**Status:** SUPERSEDED; was MEASURED, lab only (`crates/dr-validated-view-probe`, removed). No production contract, ABI or other crate is changed. Parent: D-NFR (`d-native-repr-contract-v1.md`, #1753); the per-call guard figure comes from D-MPF-4 (#1752 §10).

**Question.** The #1752 ClassId route runs a frame/focus/scope guard on every call, and that guard measured 27–28 ns. Is that guard needed on every operation, or can ownership and borrowing guarantee the same thing at binding time?

**Answer in one line.** For data owned together with its frame and focus, and for grants owned by the caller, one validation authorizes any number of operations. After it, each operation costs what the direct `ndarray::simd` call costs. Two cases still need a runtime check:

- revocation owned by another component costs one plain load per operation;
- data the frame does not own keeps the per-use check.

## 1. Three responsibilities

| responsibility | can go stale? | where it is decided | mechanism |
|---|---|---|---|
| method resolution: which native function | no | once per process | `LazyLock<Kernels>`; a copied `fn` pointer captures nothing |
| data binding: frame, focus, scope | yes | when a view is acquired | the existing `bind_dr64`, unchanged, run once in `validate` |
| execution | no | never | `View` methods call `ndarray::simd` on borrowed data |

## 2. Source audit (read in source at `origin/main`)

`dr-math-binding-probe/src/lib.rs`:

- `FrameStore { identity: Arc<()>, current: Arc<FrameRevisions> }`. Changes go through `advance(&mut self)` only; there is no interior mutability.
- `seal()` returns `Frame { store: Weak<()>, seal: Weak<FrameRevisions>, at }`.
- `check()` compares `Weak::ptr_eq(&frame.store, &Arc::downgrade(&self.identity))`, and the same for the seal. Each comparison creates a temporary `Weak` and drops it.
- `bind_dr64` runs, in order: method / axis / semantics compares, the validity–truth match, the Bag refusal, `store.check`, then focus and scope equality against caller-supplied `now`. It returns `BoundDr64 { frame: bound_at.frame.clone(), rows: &'a [u64;64], .. }`.

`dr-math-binding-probe/src/relcomp.rs`: `Chain<'a>` holds `&'a FrameStore` and checks the frame once in `Chain::new`. Its `compile_fail` doctest shows `advance` does not compile while a chain lives. `Chain` is therefore already the validate-once pattern, but for the frame only.

`lance-graph-quack/src/bind.rs`: a bound `Query` is an owned numeric value. It carries table, column and plane indices, no lifetime and no frame. That is the D-NFR-1 gap: a bound query does not know its frame.

The seven audit questions, each answered with its evidence:

1. **Can the data change through another path while borrowed?** For `bind_dr64`, yes. Its `rows` are any `&[u64;64]` the caller passes, unrelated to the frame (test `bind_dr64_checks_the_frame_not_the_data`). The bytes themselves cannot change during a shared borrow: `[u64; 64]` and `Vec<i32>` have no interior mutability. The gap is coupling, not mutation. In the probe, a `Table` owns the data and changes it only in `advance(&mut self)`, together with the revisions.
2. **Can focus change while borrowed?** `Focus` is a `Copy` revision number supplied by the caller. Nothing owns it, and a map could change while its number stays the same. In the probe, `Table::refocus(&mut self)` issues the next revision itself, so even an identical map gets a new revision (disable run D1).
3. **Can scope change independently?** `Scope { id }` equality says nothing about authorization. In the probe, the scope comes from the grant owner (`Grants`), never from the caller (disable run D3). When revocation is owned elsewhere (`SharedGrants`: `revoke(&self)` through an atomic), the borrow cannot stop it; see §4, F5.
4. **Are source and relation-map revisions coupled to the data?** Not by `bind_dr64`. Yes by `Table::advance`.
5. **Can a cloned frame outlive validity?** Yes, and it is harmless: a clone is a pair of `Weak`s. After `advance` the seal `Arc` is replaced, so `check` refuses (`SourceChanged`, or `FrameResealed` at equal numbers). A frame from another store is refused as `ForeignFrame`.
6. **Does a seal protect data identity or only revision equality?** It protects the identity of a revision issue: `Arc` pointer identity, so re-sealing equal numbers is a new frame. It does not protect the bytes; see item 1.
7. **Can a prepared callable keep references past the borrow?** No. `Kernels` holds `fn` pointers, which capture nothing, and data reaches them only as arguments. A view cannot outlive its table (E0597) or hand out a `'static` borrow (E0597).

## 3. Design (lab)

`Table` owns `FrameStore`, `Data { rows: [u64;64], values: Vec<i32> }` and the focus map. Every change needs `&mut Table`.

`validate(&'t Table, &'t Grants, declared, requested, &Guards) -> View<'t>` runs `bind_dr64` once, taking the focus from the table and the scope from the grants.

`View<'t>` offers `count`, `sum`, `min`, `max` and `count_gt` (Greater → AND → popcount, written into caller scratch). Each calls a `KERNELS` function pointer directly.

`validate_shared(&'t Table, &'t SharedGrants, ..) -> SharedView<'t>` is the same, plus one `Acquire` load of the grant epoch per operation.

## 4. Falsifiers

**Compile time.** These are `compile_fail` doctests with pinned error codes. Each snippet was also compiled on its own and fails with exactly the pinned code:

| id | attempt | result |
|---|---|---|
| F1 | `table.advance(..)` while a view lives | E0502 |
| F2 | `table.refocus(..)` while a view lives | E0502 |
| F3 (owned grants) | `grants.revoke()` while a view lives | E0502 |
| F4 (outlive) | keep a view after its table is dropped | E0597 |
| F7 | `let r: &'static [u64;64] = view.rows()` | E0597 |

**Run time** (`tests/falsifiers.rs`, 8 tests):

- **E arm:** after the borrow ends, the old guards are refused at re-acquisition:
  - an `advance` → `SourceChanged`;
  - an identical-map `refocus` → `FocusChanged`;
  - a revocation → `ScopeRevoked`.

  Freshly taken guards validate again.
- Equal numbers re-sealed → `FrameResealed`; another table's frame → `ForeignFrame`.
- **F3, scope as grant:** guards for scope 3 under grants for scope 4 → `ScopeChanged`.
- **F5, shared revocation:** `revoke(&self)` compiles while a `SharedView` lives. The next operation returns `ScopeRevoked`, and operations before it answer. **The borrow cannot enforce this, so the check stays: one epoch load per operation.**
- **F6, semantics:** the typed refusals come through unchanged:
  - another method → `MethodChanged`;
  - other axes → `AxisChanged`;
  - other truth → `SemanticsChanged`;
  - three-valued without validity → `ThreeValuedNeedsValidity`;
  - Bag → `BagNeedsMultiplicity`.
- **Results:** all five operations equal both the direct `ndarray::simd` calls and an independent row loop, at 1, 2, 63, 64, 65, 1000, 4095 and 4096 rows and five thresholds. An anti-vacuity test pins a partial selection.

**Disable runs** (committed first; each red on exactly one test, then restored):

| disable | red test |
|---|---|
| D1 `refocus` keeps the revision | `after_the_borrow_ends_every_change_is_refused_at_acquisition` |
| D2 `SharedView` never reads the epoch | `shared_revocation_during_a_view_is_refused_per_operation` |
| D3 owned grants ignored | `after_the_borrow_ends_every_change_is_refused_at_acquisition` |
| D4 focus taken from the caller's guards instead of the table | `after_the_borrow_ends_every_change_is_refused_at_acquisition` |

**Removed as unreachable:** a post-bind "two-valued Distinct only" check. The view passes no validity plane, so `bind_dr64` already refuses every three-valued and every Bag method before it. A guard that cannot fire was deleted, not kept.

## 5. Cost attribution (reported, not gated)

Setup: `examples/guard_cost.rs`, release, AVX-512 at compile time, load 0.1–0.7. ns per call, median of 21; three runs; ranges are across the runs. Allocations per call: 0 in every arm.

**The guard, taken apart** (64 rows; independent of data size):

| piece | ns |
|---|---:|
| method + axis + semantics compares | 3.4–4.1 |
| validity–truth match | 0.9–1.0 |
| focus + scope equality | 1.2–1.3 |
| `FrameStore::check` (real) | 25.7–28.3 |
| `Frame::clone` + drop (real; `bind_dr64` clones the frame) | 23.0–26.2 |
| one `Arc::downgrade` + drop of the `Weak` (proxy) | 21.0–22.5 |
| one `Weak::ptr_eq` (proxy) | 1.0–1.2 |
| one `Weak::as_ptr == Arc::as_ptr` (proxy, no refcount) | 1.0–1.8 |
| `bind_dr64` total | 59–67 |
| `validate` total (`bind_dr64` + grant read) | 62.5–65.4 |
| `LazyLock` first initialisation (cold, single sample) | 651–1016 |
| first `validate` (cold, single sample) | 253–362 |

**Assembly** (`--emit=asm`, each function read on its own):

| function | instructions | `lock` RMWs | notes |
|---|---:|---:|---|
| `FrameStore::check` | 96 | 6 | 2 × `lock cmpxchg` (`Arc::downgrade`), 4 × `lock dec` (the temporaries' drops) |
| `bind_dr64` | 108 + `check` | 2 | 2 × `lock inc` (the `Frame` clone) |
| `validate` | 83 + `bind_dr64` | 2 | `lock dec` dropping the `BoundDr64`'s frame; `Once` check for `LazyLock` |
| `View::sum` | 20 | 0 | word count, one bounds check, tail `jmp` through the copied `fn` pointer |
| `SharedView::sum` | 38 | 0 | one plain load and compare of the epoch (`Acquire` is a plain `mov` on x86), then the same body |

So the guard's cost is the reference-count traffic. Its comparisons cost 1–4 ns, and its six atomic RMWs (eight in `bind_dr64`) are the rest.

**Per operation, 64 rows:**

| op | A direct | B prebound | C `bind_dr64` per op | C′ guard per op (#1752 route) | D view | D′ shared view |
|---|---:|---:|---:|---:|---:|---:|
| count | 2.9–3.5 | 3.8–4.5 | 61–68 | 28–31 | 3.2–3.9 | 4.4–5.5 |
| sum | 28.6–30.9 | 28.6–32.8 | 78–83 | 42–49 | 27.8–39.3 | 28.8–43.8 |
| min | 30.3–49.0 | 31.2–57.5 | 80–114 | 48–89 | 29.9–61.6 | 30.7–60.2 |
| max | 30.7–53.3 | 31.5–36.2 | 79–87 | 48–54 | 30.2–34.2 | 30.5–34.5 |
| count_gt | 33.4–36.1 | 36.6–39.4 | 94–108 | 61–74 | 35.5–48.1 | 40.7–51.3 |

The wide ranges (min, and one sum and count_gt sample) come from single noisy runs and appear in A and B as well as in D. Where D was above A in a run, the same run usually showed the same spike elsewhere.

**Five operations, 64 rows:**

| path | ns |
|---|---:|
| direct, inlined together | 107–119 |
| one `validate` + five view operations | 200–332 (≈ `validate` 65 + the five per-op view costs) |
| `bind_dr64` before each operation | 387–488 |

**4096 rows:** the kernels cost 1.7–2.0 µs (masked sum/min/max), so all arms are within run-to-run noise except:
- `count`: direct 6.6, view 5.4, `bind_dr64` per op 64;
- `count_gt`: direct 292, view 307, `bind_dr64` per op 523.

**E (refused re-validation):** 28–32 ns.

## 6. Answers

**Q1 — what does the 27–28 ns guard execute?** `FrameStore::check` (25.7–28.3 ns, 96 instructions, six `lock` RMWs) plus two `u64` equality tests for focus and scope (~1.3 ns). Inside `check`, the two `Weak::ptr_eq` comparisons cost about 1 ns each. Almost all the time goes to creating and dropping two temporary `Weak`s, which is atomic weak-count traffic. Comparing pointers does not cost it.

**Q2 — what does borrowing already guarantee?**
- **In the probe's `Table` design**, while a view lives:
  - the data, the frame store and the focus map cannot change (E0502);
  - the view cannot outlive the table (E0597);
  - no borrow escapes as `'static`;
  - the prepared `fn` pointers hold no references.
- **In the existing code**, `Chain<'a>` already holds `&'a FrameStore`, so the frame cannot advance while a chain lives. `BoundDr64`, however, holds a cloned `Frame` rather than a borrow, so the store **can** advance while a `BoundDr64` lives. That is why the #1752 route has to re-check on every call.

**Q3 — what genuinely needs runtime validation?**
1. **Acquisition, all of `bind_dr64`:** the guards were taken earlier and may be stale.
2. **Revocation owned by another component:** an atomic that a borrow cannot freeze.
3. **Data not owned with its frame:** `bind_dr64` checks the frame, not the bytes.
4. **A focus identified by a caller-chosen number:** no check, at run time or compile time, can see a map change behind an unchanged number. The fix is ownership (an owner-issued revision), not more checking.

**Q4 — at which boundary?**

| validation | boundary |
|---|---|
| method resolution | once per process (`LazyLock`) |
| frame, focus, scope, semantics | at view acquisition, and again only after a `&mut` change, which cannot happen while a view lives |
| shared revocation | per operation, one epoch load; per batch if the policy accepts revocation at batch granularity |
| data from outside the frame owner | per use, as today |

**Q5 — can several operations reuse one validated binding without weakening the protections?** Yes, for table-owned data and caller-owned grants. Stale frame, focus and scope while a view lives are compile errors, not runtime events. After the borrow they are refused at re-acquisition. Semantic mismatches are refused at acquisition with the existing typed refusals. With externally revocable grants the same holds, plus one epoch load per operation.

**Q6 — what does it cost against direct ndarray?**
- **Per operation:** D view equals A direct within run-to-run noise, at 0 allocations. In the assembly, `View::sum` is a tail call through the resolved `fn` pointer and executes no `lock` instruction.
- **Validation:** ~65 ns once per view; the first, cold one is 253–362 ns.
- **Shared view:** adds the epoch load, about 1–2 ns per operation.
- **The per-call paths it replaces cost:**
  - +25–40 ns per operation (C′, the #1752 route guard);
  - +50–75 ns per operation (C, `bind_dr64` per op).
- **Comparison points:** the 36.1 ns direct baseline and the ~0.5 ns `LazyLock` effect from #1752 are kept as comparison points, not as guarantees. Here the `fn`-pointer call measured 2.1–2.6 ns against 2.7–2.9 ns direct for a one-word count, i.e. no measurable cost.

## 7. Acceptance

- [x] A validated binding executes several operations with no routing and no repeated metadata validation.
- [x] Data, focus and scope remain valid for the whole view lifetime (compile-time for owned state; per-operation epoch for shared revocation).
- [x] Invalid reuse fails to compile, or is refused at acquisition (E arm, F4–F6).
- [x] No guard removed for speed. The one removed check was unreachable. The shared-revocation check is retained.
- [x] No correctness condition rests on an unsupported lifetime assumption. The two that a lifetime cannot carry (foreign data, external revocation) keep their runtime checks.
- [x] Zero allocation on the hot path (counted, all arms).
- [x] Direct and view results identical; also equal to an independent row loop.
- [x] Validate-once benchmarked against per-operation validation (§5).
- [x] Assembly supports the claim (§5).

## 8. Proposals (not implemented; owner coordination needed)

1. **`FrameStore::check` without refcount traffic.** Compare `Weak::as_ptr(&frame.store)` with `Arc::as_ptr(&self.identity)`, and the same for the seal, instead of creating temporaries with `Arc::downgrade`. This removes the six `lock` RMWs. The proxy measured ~1–2 ns per comparison against ~21 ns for the downgrade. The semantics are the same: `Weak::ptr_eq` compares `as_ptr`. A dangling `Weak` from a dropped store cannot alias a live `Arc` while the issuing store is alive, because the store holds its `Arc`. This needs its own review.
2. **`BoundDr64` borrows its frame** (`&'a Frame`) instead of cloning it. That removes two `lock inc` and two `lock dec`. With proposal 1, a full validation would fall from ~65 ns to roughly 10 ns (estimate, not measured).
3. **D-NFR-1:** the smallest seam (a Quack-bound query does not know its frame) takes the `View` shape here. The query is executed only through a value that borrows a frame owner, so frame validity rides on the borrow.
