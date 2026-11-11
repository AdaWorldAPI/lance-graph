# Review: AdaWorldAPI/lance-graph #2000

- **Head:** `90fc5a76c12714d3bd00bcc5be96125aef74d89f` (base `90fc5a76c12714d3bd00bcc5be96125aef74d89f`)
- **Scope:** change of LaneBinding::check (lance-graph-dt64 lib)
- **Reviewer:** {"input": "ReviewEvidencePacket", "tool": "crai_report.py"}
- **Run:** `PARTIAL` (library targets only; consumer crates not in store; no test executed)
- **Verdict:** INCOMPLETE REVIEW: run PARTIAL; not a defect verdict (2 open finding(s))
- **Semantics:** crai-rs CR-H0-REVIEW-SEMANTICS (c43ab35). Severity, evidence and confidence are separate axes.

## F1 · impact · open

**Location:** `crates/lance-graph-dt64/src/lane.rs:307-349` · `lance_graph_dt64::lane::{LaneBinding}::check`

| severity | evidence class | confidence |
|---|---|---|
| not assessed | COMPILER_DERIVED | UNKNOWN |

**Claim**

> Changing 'lance_graph_dt64::lane::{LaneBinding}::check' affects its in-store callers. Callers outside the analysed crates are not observed.

**Evidence**

| grade | qualifier | statement | witness | revision |
|---|---|---|---|---|
| COMPILER_DERIVED |  | `lance_graph_dt64::lane::{LaneBinding}::borrow_all` calls `lance_graph_dt64::lane::{LaneBinding}::check` | `lance-graph:crates/lance-graph-dt64/src/lane.rs:386` | `90fc5a76c127` |
| COMPILER_DERIVED |  | `lance_graph_dt64::lane::{LaneBinding}::lane` calls `lance_graph_dt64::lane::{LaneBinding}::check` | `lance-graph:crates/lance-graph-dt64/src/lane.rs:360` | `90fc5a76c127` |

**Reachability**

- confirmed: `lance_graph_dt64::lane::{LaneBinding}::borrow_all`, `lance_graph_dt64::lane::{LaneBinding}::lane`
- possible: none observed
- unknown: callers in crates outside the store (symbol is PUBLIC; only these crates' library targets were compiled); test targets, examples, cfg(test) modules (store holds library targets only)

**Coverage**

- observed_crate: lance_graph_dt64 (lib target)
- unsupported_regions: 10

**Falsifier**

- Remove one witnessed call site and rebuild the store: the caller must leave `confirmed`.
- status: `NOT_RUN`

**No suggestion:** impact is a fact, not a defect; nothing to change

## F2 · coverage · open

**Location:** `crates/lance-graph-dt64/src/lane.rs:307-349` · `lance_graph_dt64::lane::{LaneBinding}::check`

| severity | evidence class | confidence |
|---|---|---|
| not assessed | SOURCE_STRUCTURE | UNKNOWN |

**Claim**

> 48 test call sites match 'check' by name. None is resolved to this symbol, and none ran.

**Evidence**

| grade | qualifier | statement | witness | revision |
|---|---|---|---|---|
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/cognitive-polyfill-probe/tests/p0.rs:271` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/cognitive-polyfill-probe/tests/p0.rs:275` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:234` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:238` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:249` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:481` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:522` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:663` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:677` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:678` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:680` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dr-isa-stream-lab/tests/integration/g2l.rs:136` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dr-isa-stream-lab/tests/integration/g2l.rs:263` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dr-math-binding-probe/tests/binding.rs:234` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-dir-sim/tests/ordinal_chain_lab.rs:162` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-dir-sim/tests/ordinal_chain_lab.rs:722` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-dir-sim/tests/ordinal_chain_lab.rs:1012` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-dir-sim/tests/ordinal_chain_lab.rs:1328` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:58` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:60` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:72` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:82` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:86` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:89` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:90` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:93` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:776` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:777` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:808` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:814` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:824` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:827` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:828` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:227` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:277` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:314` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:369` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:373` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:403` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:407` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:424` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/thought-git-probe/tests/receipt.rs:715` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/thought-git-probe/tests/receipt.rs:715` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/xlog-relcomp-probe/tests/dto_binding.rs:233` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-report/src/lane_source/tests.rs:320` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-report/src/lane_source/tests.rs:341` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-report/src/lane_source/tests.rs:418` | `90fc5a76c127` |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-report/src/lane_source/tests.rs:623` | `90fc5a76c127` |

**Reachability**

- confirmed: none observed
- possible: none observed
- unknown: test callers (library targets only; macro regions and cfg(test) modules not indexed)

**Falsifier**

- Extract the test targets and re-run impact: name-matched sites either become confirmed callers or drop out.
- status: `NOT_RUN`

**No suggestion:** coverage gap; the next step is extraction, not a code change
