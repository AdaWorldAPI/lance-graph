Answer each question with exact references (path:line, or the symbol identity the tool printed) and end each
answer with a line `COVERAGE: <what you could not see, and why>`. Never write "none" or "no callers" unless you
can name the closed search space that proves it; otherwise write UNKNOWN and say what was not observed.

Q1  Where is the `ClassView` trait defined, and what implements it?
Q2  Which crates depend on lance-graph-dt64?
Q3  Who calls `LaneBinding::check` (lance-graph-dt64/src/lane.rs), including tests?
Q4  What is reachable from `BindStore::method_ref` and `BindStore::admit` (callers and callees)?
Q5  How does a `MethodBinding` get from DT64 into R2IL preparation? Name each step.
Q6  Which WHEN guards lower to an executable form, and where does execution refuse them?
Q7  Can a `ScopedDecision` (lance-graph-rbac) reach the DT64 lane path? Show the path or show why not.
Q8  What code would be affected if `WideFieldMask` or `MethodBinding` changed? Give counts and their scope.
Q9  For `LaneBinding::check`: list what is known (confirmed), possible (e.g. dynamic dispatch) and unknown.
Q10 Which edges drawn in .mermaid/03-dr-dt64-binding.md, 05-methods.md and 06-authorization.md cannot be
    recovered from compiler facts (call graph, definitions, references)? Classify each.
