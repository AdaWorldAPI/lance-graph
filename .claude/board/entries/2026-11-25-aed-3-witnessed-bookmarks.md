# 2026-11-25 — D-AED-3: width and exponent witnessed with the frame

**Status:** TEST-PINNED (probe). Plan: `.claude/plans/adaptive-exact-decimal-binding-v1.md` D-AED-3.

- `FieldExps` (the owner-side declaration) gains `width_of`, `frame` and
  `check_frame`, defaulted for a stand-in with no store. `DecimalRecipe`
  records exponent, width and frame at compile and refuses a run with
  `ExpChanged`, `WidthChanged` or `FrameMoved(FrameRefusal)`.
- Reuse, no new descriptor system: the frame is the owner's DT64 seal, and
  the check is `FrameStore::check` (another store's seal is `ForeignFrame`
  even at equal revisions; a later data version is `SourceChanged`).
- Gate (`a_recipe_is_refused_under_a_moved_width_exponent_or_frame`): refused
  on another table equal in class, fields, rows and descriptor; refused after
  a write that moved no descriptor; refused, by width, after a write that
  widened the slice; runs on its unchanged owner and after recompiling.
  Disable runs: width check off and frame check off, each red.

OPEN: where the descriptor lives (plan §4.2: intake column shape, SoA
envelope or DR tile). The test owner pairs an intake table with a descriptor
it holds itself; intake has no numeric column type to carry it.
