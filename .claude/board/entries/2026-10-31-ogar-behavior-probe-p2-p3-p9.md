# P2 / P3 / P9: what OGAR's existing structures express about behaviour

**Date 2026-10-31 · Status MEASURED (lab) · Crate `crates/ogar-behavior-probe` · Source map: 2026-10-31-coresearch-mandatory-codegraph.md**

Operator decision 2026-10-31: run P2, P3, P9. OGAR owns canonical behaviour; R2IL is one executable vocabulary; #1892 stays superseded. Correction carried into the probes: an `ogar-loco::Program` is not the whole ON × WHEN × DO. Behaviour is `ActionDef`/`KausalSpec` + Program body + vocabulary binding, so `body_rev` is not a semantic revision.

Results (real data: verbatim Odoo `account_payment_term.py` through ruff and `ogar-from-ruff`; the `ogar-vocab` action tables; OGAR's GCD example program). 19 tests; 8 disable runs red, 1 check found not load-bearing.

- **P2.** A canonical body revision is computable from existing data (renumber reachable bodies, hash shape, count, calls). Two gaps:
  - `ogar-loco::Vocabulary` has no name or version, so the caller must supply the interpretation tag.
  - A guard change on a real `ActionDef` leaves `body_rev` unchanged, and no existing revision detects it. `ActionDef` has no revision, `SigRev` is set by its registrant, and `FrameRevisions` do not cover the envelope.
  - Kill "body_rev is a semantic revision": fired.
- **P3.** None of the 21 harvested methods projects to a sound `MethodBinding`.
  - `on` is Direct for 10 of the 21.
  - `reads` is never complete.
  - Effect targets are Direct for 8, but effect algebra and arguments are NotRepresentable for all 21 (written values are not captured).
  - `id`, `sig_rev`, `population`, `truth` and `demand` are NotRepresentable.
  - A `StateGuard` cannot become a DT64 `Guard` without a value codebook.
  - The hand-authored DT64 fields remain the compatibility surface.
  - Kill "derivable today": fired.
- **P9.** 50 ActionDefs from four producers; 0 link a Program (category 1 empty). `ogar-loco::Program` has no frontend producer, the Elixir extractors are `todo!()`, and OCR actions are not ActionDefs. Kill "mandatory inspectable body exists": fired.

Owner gaps (no type proposed):
1. Vocabulary identity (OGAR).
2. An action-envelope revision, or plan validity checked against those fields (OGAR, then DT64).
3. Write-value capture (ruff / ogar-from-ruff).
4. A frontend lowering into Program, or a ruling that `body_source` plus facts is the body (OGAR).

P6 needs a lab body (the GCD program), since no harvested method has one.
