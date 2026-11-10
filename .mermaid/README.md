# Architecture diagrams

These pages draw the DR / DT64 / mask-risc stack as it is in the repository. Each diagram has a "Sources" list of `path:line` that the author opened.

Two limits apply to every page:

- Nothing here claims Foundry parity. There is no graphical low-code builder.
- A gap is drawn as a gap. If code for a link does not exist, the link is dashed.

## Status legend

| mark | meaning |
|---|---|
| solid edge, green box | in code on main |
| yellow box | lab or probe crate only (excluded from the root workspace) |
| blue box | open pull request, not on main |
| red dashed box, dashed edge | gap: not built |
| grey box | another repository |

All pages use this legend. Pages that need extra classes (grey = another repository) say so at the top.

## Reading order

1. `01-stack-overview.md`, for the layers.
2. `02-crate-dependencies.md`, for the real Cargo edges.
3. `03-dr-dt64-binding.md`, for the binding types and refusals.
4. `04-read-path.md`, for one governed read and the G2 gap.
5. `05` to `09`, for methods, authorization, the write path, scenarios and the parity status map.
6. `10-lgj-crosswalk.md`, for the Java side.

## Files

| file | what it is |
|---|---|
| `README.md` | This index. |
| `01-stack-overview.md` | The layers: contract, dt64, execution, authorization, write and scenario, lab. One overview flowchart and one table. Shows which layers depend on which and where the open gaps G2, G3 and G4 sit. |
| `02-crate-dependencies.md` | The Cargo dependency edges, read from each `Cargo.toml`. Splits workspace members from excluded crates, and shows the git edges between lance-graph and OGAR in both directions. Ends with a table that lists dev-dependencies. |
| `03-dr-dt64-binding.md` | The `lance-graph-dt64` types. Three class diagrams (DR route log and map table; frames and guards; ClassView, BindStore and lanes) and one sequence diagram of bind then borrow, with every refusal branch. |
| `04-read-path.md` | A governed read from a sealed frame to a result mask, how a `Program` gets built, and the G2 gap: an authorization decision exists but nothing applies it to a binding. |
| `05-methods.md` | The method binding, ON x WHEN x DO. DT64 stores it and executes nothing; other crates run the WHEN part and the DO part separately today. |
| `06-authorization.md` | From login to authenticated user to RBAC kernel to a decision with row scope and visible fields. The decision exists; applying it to a DT64 binding is not built. |
| `07-write-path.md` | The pieces of a write: thought outcomes, cycle commit, receipts, cube staging. They are not composed into one authorized, preconditioned and witnessed write-back (gap G4). |
| `08-scenarios.md` | What-if work: branches, land and squash in `thought-git-probe`, previews in the cube. A DT64 binding cannot read a branch yet (gap G3). |
| `09-foundry-parity.md` | A status map of evidence against the parity plan. It is not a parity claim and nothing was compared against a running Foundry. |
| `10-lgj-crosswalk.md` | The separate `lance-graph-java` repo: its layers, the one contract fence and the two membranes (Panama and Valhalla). |

## How the diagrams were checked

- Every Mermaid block was rendered with `mmdc` before the page was handed over.
- Cargo edges were read from the manifests, not from memory.
- Line numbers cite the checkout in this repository at the time of writing. They drift as files change.
