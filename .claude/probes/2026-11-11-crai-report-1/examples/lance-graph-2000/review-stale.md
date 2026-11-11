# Review: AdaWorldAPI/lance-graph #2000

- **Head:** `90fc5a76c12714d3bd00bcc5be96125aef74d89f` (base `0c9ae23ac6a20ca6cd0c50da7e7a69e0f5df34eb`)
- **Scope:** same, packet from an older head
- **Reviewer:** {"input": "ReviewEvidencePacket", "tool": "crai_report.py"}
- **Run:** `PARTIAL` (library targets only; consumer crates not in store; no test executed)
- **Verdict:** INCOMPLETE REVIEW: run PARTIAL; not a defect verdict (0 open finding(s))
- **Semantics:** crai-rs CR-H0-REVIEW-SEMANTICS (c43ab35). Severity, evidence and confidence are separate axes.

## F1 · impact · stale

**Location:** `crates/lance-graph-dt64/src/lane.rs:307-349` · `lance_graph_dt64::lane::{LaneBinding}::check`

| severity | evidence class | confidence |
|---|---|---|
| not assessed | ASSERTED | UNKNOWN |

**Claim**

> Changing 'lance_graph_dt64::lane::{LaneBinding}::check' affects its in-store callers. Callers outside the analysed crates are not observed.

**Evidence**

| grade | qualifier | statement | witness | revision |
|---|---|---|---|---|
| COMPILER_DERIVED |  | `lance_graph_dt64::lane::{LaneBinding}::borrow_all` calls `lance_graph_dt64::lane::{LaneBinding}::check` | `lance-graph:crates/lance-graph-dt64/src/lane.rs:386` | `0c9ae23ac6a2` (not current) |
| COMPILER_DERIVED |  | `lance_graph_dt64::lane::{LaneBinding}::lane` calls `lance_graph_dt64::lane::{LaneBinding}::check` | `lance-graph:crates/lance-graph-dt64/src/lane.rs:360` | `0c9ae23ac6a2` (not current) |

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

## F2 · coverage · stale

**Location:** `crates/lance-graph-dt64/src/lane.rs:307-349` · `lance_graph_dt64::lane::{LaneBinding}::check`

| severity | evidence class | confidence |
|---|---|---|
| not assessed | ASSERTED | UNKNOWN |

**Claim**

> 48 test call sites match 'check' by name. None is resolved to this symbol, and none ran.

**Evidence**

| grade | qualifier | statement | witness | revision |
|---|---|---|---|---|
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/cognitive-polyfill-probe/tests/p0.rs:271` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/cognitive-polyfill-probe/tests/p0.rs:275` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:234` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:238` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:249` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:481` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:522` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:663` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:677` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:678` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dir-rmap-probe/tests/equivalence.rs:680` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dr-isa-stream-lab/tests/integration/g2l.rs:136` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dr-isa-stream-lab/tests/integration/g2l.rs:263` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/dr-math-binding-probe/tests/binding.rs:234` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-dir-sim/tests/ordinal_chain_lab.rs:162` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-dir-sim/tests/ordinal_chain_lab.rs:722` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-dir-sim/tests/ordinal_chain_lab.rs:1012` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-dir-sim/tests/ordinal_chain_lab.rs:1328` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:58` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:60` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:72` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:82` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:86` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:89` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:90` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:93` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:776` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:777` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:808` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:814` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:824` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:827` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/dr.rs:828` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:227` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:277` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:314` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:369` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:373` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:403` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:407` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: inside a macro region) | `lance-graph:crates/lance-graph-dt64/tests/lane_socket.rs:424` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/thought-git-probe/tests/receipt.rs:715` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/thought-git-probe/tests/receipt.rs:715` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/xlog-relcomp-probe/tests/dto_binding.rs:233` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-report/src/lane_source/tests.rs:320` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-report/src/lane_source/tests.rs:341` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-report/src/lane_source/tests.rs:418` | `0c9ae23ac6a2` (not current) |
| SOURCE_STRUCTURE | POSSIBLE | name match, receiver unresolved (UNKNOWN: file not indexed) | `lance-graph:crates/lance-graph-report/src/lane_source/tests.rs:623` | `0c9ae23ac6a2` (not current) |

**Reachability**

- confirmed: none observed
- possible: none observed
- unknown: test callers (library targets only; macro regions and cfg(test) modules not indexed)

**Falsifier**

- Extract the test targets and re-run impact: name-matched sites either become confirmed callers or drop out.
- status: `NOT_RUN`

**No suggestion:** coverage gap; the next step is extraction, not a code change
