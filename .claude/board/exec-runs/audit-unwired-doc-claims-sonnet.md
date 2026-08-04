## 2026-08-04 — audit_unwired_doc_claims sweep (report-only, Sonnet)

- Task: find doc-comments asserting a wiring/consumption relationship with no
  call site, per the 3 seed instances in kanban_actor.rs / batch_writer.rs.
- Mechanical sweep: 424 hits across 248 files (case-insensitive, doc-comment
  lines only, 12 trigger phrases per brief). 177 backtick-bearing (candidate
  concrete-symbol claims) vs 247 with none (VAGUE).
- Individually grep-verified 38 of the 177 candidates.
- UNWIRED-CLAIM confirmed (6 sites, 3 files):
  - kanban_actor.rs — the SAME false claim repeated 3x (lines ~27-31, ~74-78,
    ~294-298): "the #879 path consumes gate_decision_i4 directly via
    cycle_driver::shade_owner / run_cognitive_work_gated[_over]". Verified
    shade_owner/run_cognitive_work_gated[_over] have zero callers outside
    cycle_driver.rs itself; gate_decision_i4's only callers are cycle_driver's
    own wrapper, kanban_actor's own LEGACY drive_mul_advance, canonical_node's
    mul_phase_step (zero callers outside its own tests), and an unrelated
    sigma-tier-router consumer. Nothing in persist_sink::recover_and_apply
    (the actual #879 apply path) calls any of these.
  - batch_writer.rs:41-43 — re-verified the brief's given instance:
    recover_and_apply applies ls.slot.paired_move directly, never consults
    VersionScheduler::on_version. (This claim uses no mechanical trigger word
    — "fired inline" — so it did not surface in the 424-hit sweep; included
    per the brief's explicit naming.)
  - rbac.rs — NEW finding, not in the brief's seed set. Module header (lines
    1-9) + lines 141-142 assert relocating ClassRbac into the contract crate
    achieves "the keystone's impl ClassRbac for OgarClassView (Q5)" ("This is
    that placement"). grep confirms ZERO such impl exists anywhere. Worse:
    lance-graph-ogar/src/rbac_impl.rs (same repo) explicitly documents that
    exact form as an orphan-rule violation (E0117) and ships a different
    realization (OgarRbac<S> wrapper newtype + injected GrantSource) instead.
    rbac.rs's own stated reason for existing is falsified by its own
    downstream consumer.
- HONEST-DECLARED (good pattern) found repeated ~14x across the tree beyond
  the batch_writer.rs model: medcare_actor.rs, codegen_spine.rs, arcuate.rs,
  mailbox_scan.rs, cognition/op.rs, recipe_kernels.rs, traits.rs, columns.rs,
  transcode/mod.rs, transcode/parallelbetrieb.rs, inertia_data.rs, error.rs
  (archetype), sigker/codec.rs, canonical_node.rs (guid-v2-tail).
- Ruled out several plausible-looking false leads with grep before writing
  them up: odoo_blueprint "Drives X savant" claims (CurrencySelectionAdvisor
  etc.) are real data-table entries in lance_graph_contract::savants::SAVANTS,
  dispatched via savant_reasoners.rs — not missing code, just data-shaped
  rather than type-shaped, so my first `grep "struct CurrencySelectionAdvisor"`
  false-negatived. basin_placement_learning.rs vs mailbox_scan.rs "wired"
  claims are consistent (both refer to the already-wired PrefixDepth variant,
  not the separately-marked-unwired Hamming/value-decode means).
- Did not verify: remaining ~139 backtick-bearing lines (context-read only,
  not symbol-grepped); cross-repo consumer claims (q2, OGAR, tesseract-rs,
  woa-rs, ada-consciousness) — out of this repo's scope, not asserted either
  way. crates/jc hits seen but not ranked, per brief.
- Output: /tmp/audit_unwired_doc_claims.md. No .rs files touched. Did not
  touch crates/lance-graph-planner/examples/blw_*.rs (read-only, for context).
