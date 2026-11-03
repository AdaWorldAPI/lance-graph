# 2026-11-03 — D-BAG-FOLD: multi-hop existence from resident presence, no stored relation

**Status:** MEASURED, lab only (`.claude/probes/2026-11-03-d-bag-fold/`, source `514948203`, rustc 1.99.0). No production change; implementation not authorized.

- **Falsified.** A key-prefix range is not edge membership. Prefix-only existence says true with R or S deleted. `Count` over a range is not DISTINCT when keys repeat. A concept-only prefix crosses classids.
- **Executed.** `∃d R(a,b(d)) ∧ S(b(d),c(d)) ∧ T(c(d),d)` and the count of distinct valid `d` evaluate in ONE Program, with no composed relation, tuple or mask handoff, in two arms:
  - **P:** an ADMISSION B prepared projection of each relation's own presence, then a row-aligned `And → Any/Count`, gather-free.
  - **L:** pointer tables `$i` plus `Filter::Semijoin` over the table's own resident planes, zero-copy, executing `MaskOp::Gather`.
  
  Both arms agree with a nested-loop oracle on 8 cases (each edge deleted, multiple candidates with one or none valid, branch-crossing decoys, another classid, version change). Disabling either Semijoin goes red.
- **Conditional.** Plane order, key uniqueness and presence version are each a silent wrong answer when unchecked (measured). No contract witness covers plane order (`ISS-WITNESSED-RANGE-DOES-NOT-ATTEST-PLANE-ORDER`) or presence version.
- **Blocked by rulings, not capability.**
  - Arm P copies presence bits, which ADMISSION B permits and the later zero-copy law does not.
  - Arm L needs the `Gather` call-site classification (EXO-140, no owner).
  - The DT64 route is refused at `r2il::prepare` (`GuardNotExecutable`), and no admitted DO word shape expresses the conjunction (`xlog::bag_count_admitted` admits only `[GroupCount, GroupCount, MulI64]`).
- **OPEN.** The P-vs-L operator ruling; a plane-naming witness; a mint guard that edges refine their source key. EXO-108 untouched.
