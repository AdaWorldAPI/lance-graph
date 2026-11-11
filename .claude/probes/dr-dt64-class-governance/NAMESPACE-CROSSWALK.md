# D-DRDT-CLASS-GOVERNANCE-1 — Namespace crosswalk

Status: census, docs only. Read at `main` `238b6d1e4` (2026-11-11). Every row
cites the file read. Status words: SHIPPED (code on main), DECIDED (a logged
decision), PROPOSED (this brief), OPEN.

## 1. The rule in the brief

`xxxx_FX(tech)XX(method)`: high 16 bits = Domain × AppID / class context;
low 16 bits = `F`, a 4-bit technology, an 8-bit method. The brief also says:
no technology ids are assigned, and the encoding of "no left context" is not
decided.

## 2. What is on main

| fact | where | status |
|---|---|---|
| A classid is `[canon u16][custom u16]`, canon high | `lance-graph-contract/src/ogar_codebook.rs:459` (`CLASSID_ORDER = CanonHigh`), `:484` `compose_classid`, `:491` `split_classid` | SHIPPED |
| A behaviour lens is any classid whose right half is in `0xF000..=0xFFFF` | `lance-graph-contract/src/class_view.rs:1053` `is_behaviour_lens` | SHIPPED |
| A lens's field basis is read by the full `u32`; unknown lens answers `None` and the reader must refuse | `class_view.rs:1084` `ClassView::lens_fields` | SHIPPED |
| The left half of hardcoded behaviour is `0xFFFF` | `ogar_codebook.rs:583` `HARDCODED_BEHAVIOUR`; OGAR `ogar-vocab/src/lib.rs:1879` (mirror) | SHIPPED |
| The relational polyfill lens is `0xFFFF_F000` | `lance-graph-r2il/src/polyfill.rs:44` `POLYFILL_LENS` | SHIPPED |
| XLOG's lens is `0xFFFF_F100` | `lance-graph-xlog/src/lens.rs:33` `XLOG_LENS` | SHIPPED (declared; `lance-graph-r2il/src/binding.rs:341-343` executes `POLYFILL_LENS` and lenses whose concept half is the R2IL fold concept, so not this one) |
| A consumer lens under its own left half exists in tests only: `0x7F0B_F001`, `0x0A1E_F001` | `lance-graph-dt64/tests/bind_store.rs:1498`, `lance-graph-r2il/tests/integration/polyfill_views.rs:278` | SHIPPED as test fixtures |
| A word is `AlgebraRef { lens: u32, word: u8 }`; "no word gets a classid of its own" | `lance-graph-dt64/src/lib.rs:129` | SHIPPED |
| A method is `MethodId(pub u32)`, an identity within its class | `lance-graph-dt64/src/lib.rs:113` | SHIPPED |
| Concern ranges: `F0__` shared relational algebra, `F1__` XLOG, `F2__` ONNX, `F3__` TensorFlow family, `F4__` SQL-specific, `F5__` Gremlin-specific, `F6__..=FF__` reserved | `.claude/plans/d-dt64-7-behaviour-addressing-registration-v1.md` §1d, O7 | DECIDED 2026-10-31 |
| Left sides `0xF000..=0xFFFE` are unallocated | same plan §7, O1 | OPEN |

## 3. Where the brief and main agree

- The right half's top nibble `F` marks behaviour. Identical to `is_behaviour_lens`.
- The next nibble selects a technology. This is the plan's "concern" digit (O7).
- A consumer's own left half can carry a behaviour lens (`Domain::AppId :: 0xF###`). The plan calls this the lab form for XLOG (D14''); the tests use it.
- The 256-position limit is per lens and is not widened (O9).

## 4. Where they differ — these need a decision before any code

| # | brief | main / logged decision | consequence |
|---|---|---|---|
| X1 | The low byte is an 8-bit **method** | The low byte numbers **lenses** inside a concern (up to 256 per concern); the method is not in the classid. Plan §1a: "`F###` addresses a lens, not a method" | Under the brief, `0xFFFF_F000` would be technology 0, method 0. Today it is a lens holding 17+ words. One of the two readings has to be named the norm. |
| X2 | The encoding of "no left context" is not decided; `0xFFFF` must not be set as norm | `0xFFFF` is the shipped left half of hardcoded behaviour, mirrored in OGAR with a guard test, and three production constants are composed from it | The brief's caution is already overtaken by code. The decision to take is whether to ratify `0xFFFF` or migrate off it, not whether to choose one. |
| X3 | No technology ids are assigned | `F0`, `F1` are in production constants; `F2`–`F5` are assigned by O7; `0xFFFF_F200` appears as `ONNX_LENS` in a dt64 test | "Not assigned" holds only for `F6..FF`. |
| X4 | The 8-bit method must not be `MethodId(u32)` truncated or hashed | Two distinct 8-bit things exist: `AlgebraRef.word: u8` (a word of a lens) and nothing 8-bit for a class method. `MethodId` is a `u32` scoped to its class | If the low byte means "method", the candidate that already has the right width and meaning is the **word**, not `MethodId`. That contradicts X1's lens reading, so X1 and X4 are one decision. |
| X5 | Five states: reserved namespace / registered technology+method / contextualised lens / admissible `MethodBinding` / authorized execution | Code has: reserved (`HARDCODED_BEHAVIOUR`, OGAR guard), declared lens (`BindStore::declare_algebra`, `lib.rs:917`), registered class with methods (`register`, `:814`), admitted reference (`admit`, `:1180`). No type represents "authorized execution" | The fifth state has no carrier; see BYPASS-MATRIX. |

## 5. Readings that would satisfy both, for the council (none chosen here)

- **A. Keep main.** Low byte = lens index; method stays `AlgebraRef.word` (in the lens) or `MethodId` (in the class). The brief's formula is then read as `F` + concern + lens. Zero code change.
- **B. Brief literal.** Low byte = word. Then one concern holds 256 words in total, a lens is the whole concern, and TensorFlow-sized operator sets (the plan cites about 1,500) no longer fit one concern. Needs a migration for `F000`/`F100` only in meaning, not in bits, because both currently use low byte `00`.
- **C. Two-level.** `F` + concern + lens as on main, and the brief's "method" is the pair `(lens classid, word)`, 40 bits, which is what `AlgebraRef` already is.

Not decided here. Under every reading, historical ids keep their bits.
