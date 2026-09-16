# DuckDB HEADER harvest — the repair of the matrix's §6

The `ruff_cpp_spo` harvest behind `duckdb-to-v3-translation-matrix-v1.md` was
pointed at 22 `.cpp` translation units. Its own §6 records the result honestly:
**seven TUs harvested to 100 % `Empty`**, because DuckDB's execution is
template-dispatched and lives in HEADERS — so "no row in this matrix cites a
harvest TSV as evidence", and §6 names the fix: point it at the headers.

Done, 2026-09-14, same harvester (`ruff/examples/harvest_events`, libclang 18),
same `args.txt`. Per-header counts:

| header | methods | events |
|---|---|---|
| `common_operator_comparison_operators` | 5 | 39 |
| `common_operator_constant_operators` | 0 | 0 |
| `common_row_operations_row_matcher` | 0 | 0 |
| `common_types_selection_vector` | 26 | 310 |
| `common_types_validity_mask` | 61 | 836 |
| `common_types_vector` | 6 | 42 |
| `common_vector_operations_scalar_executor` | 10 | 286 |
| `common_vector_unified_vector_format` | 1 | 6 |
| `execution_expression_executor_state` | 2 | 11 |
| `execution_ht_entry` | 12 | 92 |

**123 methods, 1,622 events** where the `.cpp` pass yielded none for the same
concepts.

> ⊘ **HONESTY NOTE, added 2026-09-15 after review.** Two things this file
> claimed by implication and does not deliver.
>
> **The numbers are not re-derivable from this repo.** The TSV is not
> committed, there is no `args.txt` here, and no DuckDB source is in this tree
> — a repo-wide grep finds `harvest_events` and `args.txt` only inside this
> README. Treat the counts as a recorded observation of a run that happened
> against a local checkout, not as evidence a reader can verify.
>
> **§6's complaint is therefore NOT discharged.** §6 says *"no row in this
> matrix cites a harvest TSV as evidence."* That is still true: what exists
> now is a README quoting counts from a TSV that is absent. What the harvest
> genuinely changed in the consumer crate is two things — `Cmp::MatchU64` and
> the range-write note — and those stand on their own.
>
> **The headline and the inventory disagree.** The per-header table sums to
> 123 and the method inventory below lists 92 distinct names; the difference
> is overloads. Read 123 as definitions, 92 as names. Two headers yield nothing and that is information too:
`row_matcher.hpp` and `constant_operators.hpp` are pure declarations whose
bodies are templates no TU instantiates here — the .cpp pass already covered
`row_matcher.cpp`, which did produce events.

## The two findings that changed code

1. **`TemplatedValidityMask::SetRangeInvalid`** — DuckDB's own bit-plane, the
   same packed-`u64` carrier V3 uses (matrix R6), carries a RANGE write. The
   matrix files `mask_set_range` as T1 gap **G6** on the strength of V3's own
   trie-reveal measurement; this is the same operation on the other side, and
   it is why G6 is a real primitive rather than a wish. `lance-graph-quack`
   now spells the address-PREFIX predicate and says plainly that it lowers to
   a sweep, not a range write, until G6 lands.
2. **`ScalarExecutor::{Runtime,Static}SelectionSink`** — `Append`,
   `AppendInvalidRange`, `FillConstant`, `Result`. This is matrix row **C2**
   ("the sink", ELIMINATE: the mask word IS the sink), and it was entirely
   invisible to the `.cpp` pass.

## Method inventory
```
ConsecutiveChildListInfo.ConsecutiveChildListInfo
Equals.Operation
ExecuteFunctionState.GetFunctionState
ExpressionState.~ExpressionState
GreaterThan.Operation
ScalarExecutor::RuntimeSelectionSink.Append
ScalarExecutor::RuntimeSelectionSink.AppendInvalidRange
ScalarExecutor::RuntimeSelectionSink.FillConstant
ScalarExecutor::RuntimeSelectionSink.Result
ScalarExecutor::RuntimeSelectionSink.RuntimeSelectionSink
ScalarExecutor::StaticSelectionSink.Append
ScalarExecutor::StaticSelectionSink.AppendInvalidRange
ScalarExecutor::StaticSelectionSink.FillConstant
ScalarExecutor::StaticSelectionSink.Result
ScalarExecutor::StaticSelectionSink.StaticSelectionSink<HAS_TRUE_SELECTION, HAS_FALSE_SELECTION>
SelectionVector.Capacity
SelectionVector.Incremental
SelectionVector.Initialize
SelectionVector.Inverted
SelectionVector.IsSet
SelectionVector.SelectionVector
SelectionVector.data
SelectionVector.get_index
SelectionVector.get_index_unsafe
SelectionVector.operator=
SelectionVector.operator[]
SelectionVector.sel_data
SelectionVector.set_index
SelectionVector.swap
TemplatedValidityData.EntryCount
TemplatedValidityData.TemplatedValidityData<V>
TemplatedValidityMask.AllValid
TemplatedValidityMask.CanHaveNull
TemplatedValidityMask.CannotHaveNull
TemplatedValidityMask.CheckAllInvalid
TemplatedValidityMask.CheckAllValid
TemplatedValidityMask.Copy
TemplatedValidityMask.CountValid
TemplatedValidityMask.EnsureWritable
TemplatedValidityMask.EntryCount
TemplatedValidityMask.EntryWithValidBits
TemplatedValidityMask.GetAllocationSize
TemplatedValidityMask.GetData
TemplatedValidityMask.GetEntryIndex
TemplatedValidityMask.GetValidityEntry
TemplatedValidityMask.GetValidityEntryUnsafe
TemplatedValidityMask.Initialize
TemplatedValidityMask.IsMaskSet
TemplatedValidityMask.NoneValid
TemplatedValidityMask.Reset
TemplatedValidityMask.RowIsValid
TemplatedValidityMask.RowIsValidUnsafe
TemplatedValidityMask.Set
TemplatedValidityMask.SetAllInvalid
TemplatedValidityMask.SetAllValid
TemplatedValidityMask.SetInvalid
TemplatedValidityMask.SetInvalidUnsafe
TemplatedValidityMask.SetRangeInvalid
TemplatedValidityMask.SetValid
TemplatedValidityMask.SetValidUnsafe
TemplatedValidityMask.SizeInBytes
TemplatedValidityMask.TemplatedValidityMask<V>
TemplatedValidityMask.ValidityMaskSize
ValidityArray.AllValid
ValidityArray.CanHaveNull
ValidityArray.CannotHaveNull
ValidityArray.Capacity
ValidityArray.Initialize
ValidityArray.InitializeEmpty
ValidityArray.Pack
ValidityArray.RowIsValid
ValidityArray.RowIsValidUnsafe
ValidityArray.SetValid
ValidityArray.SetValidUnsafe
ValidityArray.ValidityArray
ValidityMask.ValidityMask
Vector.Buffer
Vector.BufferMutable
Vector.GetBufferRef
Vector.GetType
Vector.GetVectorType
Vector.SetBuffer
duckdb.IncrementAndWrap
ht_entry_t.ExtractSalt
ht_entry_t.GetPointer
ht_entry_t.GetPointerOrNull
ht_entry_t.GetSalt
ht_entry_t.GetSaltWithNulls
ht_entry_t.IsOccupied
ht_entry_t.SetPointer
ht_entry_t.SetSalt
ht_entry_t.ht_entry_t
```

---

## ⊘ §6 IS NOW DISCHARGED — the harvest is reproducible, 2026-09-16

The honesty note above says what was missing: *"The TSV is not committed, there
is no `args.txt` here, and no DuckDB source is in this tree."* All three are
fixed, and the fix is committed beside this file:

| file | what it is |
|---|---|
| `run.sh` | the whole harvest, one command, `DUCKDB_SRC=… ./run.sh` |
| `headers.txt` | the ten headers, by path |
| `args.txt.in` | the clang args as a TEMPLATE — `@DUCKDB_SRC@` is substituted at run time, so the include paths are not pinned to one machine |
| `ore/**` | the 40 TSVs (4 per header), 348 K, committed |
| `ore/per-header.tsv` | the counts table, regenerated |
| `ore/provenance.txt` | DuckDB commit + origin + ruff commit + clang version |

**The source is `AdaWorldAPI/duckdb`** (P0: the fork, never upstream —
`duckdb/duckdb` is correctly out of this session's scope and returns 403).
Read-clone: `GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1
https://github.com/AdaWorldAPI/duckdb /home/user/adaworldapi/duckdb`.

### The counts, re-derived rather than quoted

**Methods: 123 — an EXACT match with the table above, on every one of the ten
rows.** The recorded observation is confirmed reproducibly; it is no longer a
number quoting an absent TSV.

Events land at **1620** against the recorded 1622, differing on four rows:

| header | events now | recorded | Δ |
|---|---|---|---|
| `common_types_selection_vector` | 311 | 310 | +1 |
| `common_vector_operations_scalar_executor` | 281 | 286 | **−5** |
| `common_vector_unified_vector_format` | 7 | 6 | +1 |
| `execution_expression_executor_state` | 12 | 11 | +1 |

**The cause is not determinable, and that is itself the point.** Methods match
exactly, so no method was added or removed — the deltas are inside bodies,
which is what source drift between two DuckDB checkouts looks like. The
2026-09-14 run recorded no commit, so there is nothing to diff against. That
is precisely the gap `ore/provenance.txt` now closes: every future run carries
`duckdb_head`, and a count without its commit is an anecdote.

> **⚠ MY OWN HARNESS NEARLY PRODUCED A FALSE FINDING AGAINST THIS FILE.** The
> first `run.sh` subtracted a header row from each TSV before counting, and
> reported **115 / 1612** against the recorded 123 / 1622 — a clean, uniform
> −1 on every non-empty header, which reads exactly like "the earlier numbers
> were inflated." They were not. **Neither `methods.tsv` nor `events.tsv` has
> a header row**; line 1 of each is already data
> (`duckdb::TemplatedValidityData.EntryCount(idx_t)…`), so the raw line count
> IS the count.
>
> What caught it was the SHAPE of the disagreement: −1 on every non-empty
> header and 0 on the two empty ones is not how source drift behaves, it is
> how an off-by-one behaves. **A measurement harness is code and gets the same
> scepticism as the thing it measures** — this workspace's own rule that a
> null result is a claim about the apparatus until proven otherwise, applied
> to a counting script. The corrected rule is now stated at the counting site
> rather than assumed.

### What this does NOT yet do — the structural arm is still unrun

`harvest_events` is the BEHAVIOURAL arm (four TSVs of ordered method-body
events). `ruff_cpp_spo` also carries a STRUCTURAL arm that has never been
pointed at DuckDB: `extract_tree(root, args) -> ModelGraph` (`lib.rs:338`),
yielding `CppClass` / `CppFunction` / `CppEnum` with `has_function` /
`inherits_from` / `virtually_overrides` — the ClassView method-resolution
manifest — which then feeds `ruff_spo_triplet::reassemble` and
`ruff_cpp_codegen::{project, render}`. That is the arm that answers *what
DuckDB's operator class tree IS*, where this one answers *what one method
body does*. Running it on DuckDB is the next step, not a claim made here.
