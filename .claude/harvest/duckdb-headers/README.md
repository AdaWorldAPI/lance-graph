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
concepts. Two headers yield nothing and that is information too:
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
