# ChatGPT-authored PRs #1601–#1617: plan and board links (2026-10-20)

**Status:** DONE (links only). **Basis:** VERIFIED-IN-CODE. Every PR body in #1591–#1617 was read through the GitHub API. #1591–#1600 carry a Claude Code footer; #1601–#1617 do not, so they are treated as ChatGPT-authored. Their files were read in full before linking. No content, D-id or plan was invented: each artifact is attached to the existing plan whose subject it is.

#1618 already gave all seventeen an arc entry and one shared recovery entry (`2026-10-19-post-merge-recovery-1601-1617.md`). What was missing: most artifacts named no plan or board file, and no plan named them. Measured before this change: of the 19 artifacts that now carry a back-link, 4 named a plan or board file (`2606.24948`, `2210.07278` and `2512.17373` a plan, the LEGO RFC a board entry), and their only incoming references were the arc and one lumped status row.

## Links added

Each artifact gains a `Plan and board (2026-10-20)` line under its title. Each owning plan gains a `Connected sources from #1601–#1617` section listing what it owns.

| PR | artifact | owning plan |
|---|---|---|
| #1601 | `.arxiv/2408.10592_…witnesses.md` | `d-holo-address-indexed-algebra-v1` |
| #1601 | `.arxiv/2606.24948_…two_hop_retrieval.md` | `d-holo-address-indexed-algebra-v1`, `d-holo-h14d-logic-mask-read-v1` |
| #1602 | `.claude/architecture/cartesian-epistemic-self-model.md` (already linked H14c) | `d-holo-h14c-thought-first-v1` (plan side added) |
| #1602 | `.claude/architecture/neurofunctional-rubicon-moore-plasticity.md` (already linked) | `ce64-hypothesis-layers-v1` (plan side added) |
| #1603 | `docs/research/linguistic-lego-*.md` (2) | `ce64-grammar-wiring-v1` (case and agreement, D-CGW-2/3) |
| #1604 | `docs/research/versioned-semantic-candidates-64-thoughts-rfc.md` (already linked) | `thought-git-api-v1` (plan side added); also cited by causal-pickaxe and souffle plans |
| #1605 | `causal-pickaxe-resident-git-lance13-temporal-v1.md` | is a plan; on the index; CPG-0, CPG-1..6 rows exist |
| #1606 | `tools/semantic_ring_oracle/` | status row VSC-0; README links `thought-git-api-v1` |
| #1607 | `data/qualia/MUSICAL_ANCHORS.md` | `ce64-grammar-wiring-v1` (D-CGW-14, #1571 qualia device) |
| #1608 | `data/qualia/MUSIC_ENRICHMENT.md` | `ce64-grammar-wiring-v1` (D-CGW-14) |
| #1609 | `docs/research/34-reasoning-recipes-lance-graph-plan.md` | `alpha-channel-rung-overlay-v1`, `thought-git-api-v1` |
| #1610 | three Hegel / Leibniz `.arxiv` notes | `dialectic-engine-v1` |
| #1611 | `2512.17373` dialectics | `dialectic-engine-v1`, `thought-git-api-v1` |
| #1611 | `2210.07278` meta-uncertainty | `jc-sbi-counterfactual-integration-v1` (which already cites it) |
| #1611 | `2310.18318` Hyperon | `metta-datalog-assimilation-v1` |
| #1611 | `2409.20212` GASM, `2412.01423` semantic maps | `ce64-grammar-wiring-v1` |
| #1611 | `2502.20233` SMASH | `d-holo-h14d-logic-mask-read-v1` (mask first, then read) |
| #1611 | `2510.24941` decorative CoT | `octopus-causal-cot-audit-v1` |
| #1611 | `2610.12692` distributed WCOJ | `d-holo-h14c-thought-first-v1`; the WCOJ plans carry the gather-violation banner, so the bookmark is attached to the authority instead |
| #1612 | Soufflé plan | is a plan; on the index; D-SRAM-0..7 rows exist |
| #1613 | JC R API plan | is a plan; on the index; D-JCR rows exist |
| #1614 | JC SBI plan | is a plan; on the index; D-JCSBI rows exist |
| #1615 | `crates/lance-graph-act-r/README.md` | `jc-sbi-counterfactual-integration-v1` (D-JCSBI-3, the ACT-R recovery fixture); status row "ACT-R crate" exists |
| #1616 | `crates/lance-graph-mask-risc/src/graph_hop.rs` | `ISS-GATHER-HOP-1616`; its doc comment now names `.claude/board/ISSUES.md` instead of a bare `ISSUES.md` |
| #1617 | three D-RMAP plans | are plans; on the index; D-RMAP rows exist |

## Open

- No plan owns the ACT-R crate itself; D-JCSBI-3 consumes it. Typed buffers, Mask-RISC matching and learning stay future work (README "Next").
- The `.arxiv` bookmarks marked READ LATER are unread sources; linking them claims nothing about their content.
