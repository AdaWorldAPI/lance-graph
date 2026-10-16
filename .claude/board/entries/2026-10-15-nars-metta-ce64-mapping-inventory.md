# 2026-10-15 — Mapping inventory: Hyperon/MeTTa/PeTTa atoms and three NARS engines against CE64 and the polyfill crates

STATUS: measured | OUTCOME: `.claude/research/2026-10-15-nars-metta-ce64-mapping-inventory.md` — read from source
(hyperon-experimental `3f76dc4`, PeTTa `1d19a1d`, ONA `dc4efd0`, opennars `4d1bdb0`, Narsese.rs `16d4d18`) against
`causal-edge`, `lance-graph-contract`, `planner/nars`, `lance-graph-nal`, `-relalg`, `-metta`, OGAR `ogar-loco`. No meaning
of any atom changed; no byte proposed. | OPEN: relalg join + `μ` (blocks MeTTa conjunction and the NAL syllogism alike);
the per-row truth law (nal-polyfill §4 C) remains the one candidate byte; `lance-graph-ona` and `-cog` do not exist and
the inventory scopes what `-ona` honestly could be (oracle harness, argument-order adapter, lanes not bits).

Findings worth one line each: no foreign system carries a 59..63 or a 40..42 — the ladder and the Pearl mask are ours to
write; lance-graph's abduction/induction argument order matches OpenNARS Java, ONA and PeTTa use the swapped convention
(bind by rule shape, never by function name); ONA's `ATOMS_MAX = 255` fits the u8 S/P/O palette; `Narsese.rs` already
holds the parser `lance-graph-nal` needs.
