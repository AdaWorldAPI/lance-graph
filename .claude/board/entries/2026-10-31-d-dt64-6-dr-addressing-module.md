# D-DT64-6-DR: DR addressing as `lance_graph_dt64::dr`

**Date 2026-10-31 · Status VERIFIED-IN-CODE · Plan [`d-dt64-6-routing-bindstore-behavior-v1.md`](../../plans/d-dt64-6-routing-bindstore-behavior-v1.md) A.5 addendum 12**

DR's addressing moves from `dr-math-binding-probe` into the bind store as the module `dr`. It holds reservations, the route log, the frame witness, the guards, the typed axis and the global → local map. It names nothing from the binding side ("DR routes, DT interprets"), and a test pins that. DR's computations stay with Quack.

The thin-provisioned extent (`9014fa07b`), which #1871 merged without, is restored in the same PR.

27 tests; 6 disable runs red.

**OPEN:**
- the probe owner switches to `dt64::dr`;
- `bind_dr64` waits on the `MethodSignature` vs `MethodRef` identity (EXO-50);
- the reservation as a global range (start + length).
