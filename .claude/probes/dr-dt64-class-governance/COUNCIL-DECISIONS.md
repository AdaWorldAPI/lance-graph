# D-DRDT-CLASS-GOVERNANCE-1 — Questions for the 5+3 council

Status: PROPOSED. Nothing here is decided. No public contract or codebook
change may land before these are answered.

Each question names the evidence file and the options. The council is asked
to pick, or to say what measurement is missing.

| # | question | evidence | options |
|---|---|---|---|
| Q1 | What is the low byte of a behaviour address: a lens index (main today) or an 8-bit method code (brief)? | NAMESPACE-CROSSWALK §4 X1, X4, §5 | A keep main / B brief literal / C method = `(lens, word)` |
| Q2 | Is `0xFFFF` ratified as the left half of hardcoded behaviour, and is that the same thing as "no left context"? | NAMESPACE-CROSSWALK X2; `ogar_codebook.rs:583` | ratify as "DT64-registered Frozen behaviour" and define "no context" separately / ratify as both / migrate |
| Q3 | Do the O7 concern digits (`F0`–`F5`, decided 2026-10-31) stand as the technology ids? | D-DT64-7 plan §1d | stand / reopen |
| Q4 | Is left `0xF000..=0xFFFE` reserved (brief §1.4) or unallocated (plan O1)? Who guards it? | FALSIFIERS row 7 | reserve with a contract test / leave open |
| Q5 | Must every production execution of a mask-RISC program carry proof of admission? If yes, by what mechanism? | BYPASS-MATRIX: 4 direct production sites in member crates, 1 admitted | a census test that fails on an unlisted call site / a witness type the executor requires / leave Quack and report paths ungoverned by design |
| Q6 | Where does authorization join admission? | METHOD-RESOLUTION step 6; FALSIFIERS row 16 | before `admit` in the caller / a parameter of `admit` / stays independent with a test that proves the join at each entry point |
| Q7 | Is a `MethodRef` meant to survive a process? Today its stamp holds a pointer. | SNAPSHOT-VERSIONING "Not replayable today" | keep process-local and replay by re-issuing / add a content witness beside the pointer |
| Q8 | Is the `BindStore` (classes, methods, lenses) persisted, and on whose clock? | SNAPSHOT-VERSIONING | not persisted, rebuilt from compiled tables and plugs / persisted next to the DR map table |
| Q9 | Recover PR 1984 (`VirtualCsr`) to main as its own PR? | DR-DYNAMIC-COST; the files are not on main | yes, dedicated recovery PR, no promotion to production / no |
| Q10 | Which owed measurements gate a "DR Dynamic" choice rule? | DR-DYNAMIC-COST comparison table | all 12 cells / a named subset |

Suggested order: Q1 and Q2 first, because four falsifiers and every later
code cut depend on them. Q5 next, because it decides whether "enforced
centrally" is a property of the system or of one crate.
