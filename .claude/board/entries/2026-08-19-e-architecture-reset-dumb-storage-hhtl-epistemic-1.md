## 2026-08-19 — E-ARCHITECTURE-RESET-DUMB-STORAGE-HHTL-EPISTEMIC-1

**Status:** RULING `[operator]`, verbatim charter at
`docs/architecture/DUMB-STORAGE-RESET-CHARTER.md`.

**The reset, in one line:** the substrate must know only references,
hierarchical locality (HHTL trie AND explicit reference nodes — kept as
two separate doors), ClassView, WideFieldMask-as-fovea, DatasetVersion,
and temporal coordinates; it must NOT know ontology, causality, rung,
known-unknowns, awareness, or orchestration policy — those are
interpretations layered ABOVE storage. No freeze, no batch wall, no
"persistence lag is permission to think." Epistemic PARTICIPATION
(admissibility to a current-horizon quorum) is explicitly NOT execution
PERMISSION — a stale-horizon producer keeps computing; only its evidence
is marked inadmissible until it catches up.

**Immediate consequence:** the freeze/seal-centered implementation just
ratified in PR #968 (merged `66fec27`, minutes before this ruling) is
STOPPED. Task #25 (W1 descriptor purity → W2 digest seam → W3 FNV
deletion) does not launch on the #968 merge. The seal spec, its 5+3
council hardening, and its falsifier corpus are preserved as research
history by exact reference (PR_ARC entry above) — not deleted, not
implemented as production architecture.

**Deliverable order (charter §19), gated no-code-before-map (§20):**
ARC A source archaeology (lance-graph + lance-graph-java, file:line
capability matrix) → ARC B minimal dumb-storage contract → ARC C Java
mechanical integration + ontology/OSM genericity proof → ARC D
episodic/epistemic model (deferred) → ARC E orchestration meta-awareness
(deferred). 15 falsifiers pre-registered (charter §18: F-HIERARCHY-NOT-
AUTHORITY, F-TRIE-VS-NODE, F-WFM-FOVEA, F-SPARSE-INHERITANCE,
F-CONTEXT-DELTA, F-ONTOLOGY-READONLY, F-NO-FREEZE,
F-NO-BACKPRESSURE-AUTHORITY, F-EPISTEMIC-PARTICIPATION,
F-STRICT-HINDSIGHT, F-KNOWN-UNKNOWN, F-META-SECOND-ORDER, F-JAVA-PARITY,
F-DOMAIN-GENERICITY, F-NO-64K-JAVA-OBJECTS).

**Cross-refs:** `E-CROSS-VERSION-IDENTITY-MIGRATES-BLIND-SO-IT-FAILS-
CLOSED-1` and `E-LOTUS-IS-A-REGISTER-GRID-NOT-A-BYTE-GRID-1` remain valid
findings about the STOPPED design, cited as prior art if any future arc
revisits sealed-batch identity — not superseded on their own merits, only
mooted as production direction.

