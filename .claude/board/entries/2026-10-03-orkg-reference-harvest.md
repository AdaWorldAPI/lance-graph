# ORKG harvested as a scholarly-evidence reference (2026-10-03)

**Status:** ANALYSIS + VERIFIED-IN-CODE (ORKG backend `9b24878a`, frontend `beb8902d`);
no code changed. Harvest: `.claude/harvest/orkg-reference-wiring.md`. North Star gains
a REFERENCE / COMPARATOR section.

- ORKG comparisons project recorded values (columns = sources, rows = predicate
  paths, depth ≤ 10) and compute nothing over them. The facet type
  `SearchFilter(path, range, values, exact)` serves one observatory endpoint and
  compares literal labels as strings.
- Measured on live comparison R44930 (31 papers): 29 of 31 R0 estimates are stored as
  `xsd:string`, and 29 % of root cells are null.
- Our side: evidence-computing components ship on the text leg and the cohort leg.
- **OPEN:**
  - no publication source type in any provenance struct;
  - no text → ontology-address join;
  - no effect pooling (no inverse-variance Fisher-z);
  - SEPIO / ECO are registered but unbakeable (`META_STUDY_SPINE`).
