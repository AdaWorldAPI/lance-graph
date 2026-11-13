# CE64 logic-neutrino P0/P1: narrow executable lab

**Date:** 2026-11-13. **Status:** SOURCE WRITTEN; CARGO TESTS NOT YET EXECUTED. **Scope:** unadmitted research only.

**Research parent:** [Seven-branch logic-neutrino scratchpad, draft PR #2048](2026-11-13-ce64-logic-neutrino-grammar-spine.md). This lab is a separate PR, not a rewrite of that mindmap. It does not authorize changes to production admission, F###, CE64 bits, ClassView meaning or runtime.

## Question

Can a one-byte *candidate index* preserve two interpretations of the same visible grammatical construct, while existing DT64 WHEN/Quack/mask-risc machinery distinguishes them? Can a packed CausalEdge64 role be selected by the existing 64-bit ternary predicate without conflating its Epi5 topology?

## Changed surface

- New integration test in crates/grammar-constraints-probe/tests/logic_neutrino_p0.rs.
- One **test-only** causal-edge dev-dependency in crates/grammar-constraints-probe/Cargo.toml.
- This lab receipt. No production module touched.

This reuses the existing German-UD Haspelmath grammar constraints probe, the existing DT64 lab ClassView and the existing admitted mask-risc evaluator. It does not introduce a second grammar runtime.

## P0: rule selector is not the semantic authority

A test-local u8 enum indexes two possible rules: Recipient = 1, Experiencer = 2. The observed UD left corner (deprel = obl:arg, Case = Dat) proposes **both** for all the relevant arguments, including one with an unknown verb frame. This selector is not a registered OGAR recipe, F### word, MethodId or CE64 opcode.

The test registers the **existing** lab candidate methods using RuleDesc and DT64 BindStore, obtains real MethodRefs, calls when::compose, and executes the resulting Quack/mask-risc programs under the same sealed DR frame. Independent expected fixture labels:
- gid 105: Recipient, not Experiencer (geben).
- gid 302: Experiencer, not Recipient (gefallen).
- gid 405: Recipient, not Experiencer (geben).
- gid 505: neither rule admitted as a matching role (helfen has no frame entry; **abstain, not false**).
- gid 408: instrumental dative, not even a left-corner candidate.

**Controls that must discriminate:** substituting Experiencer RuleDesc for a Recipient MethodRef refuses with DescriptionMismatch; dropping its head-frame guard also refuses; a separate deliberately ablated rule set without the head-frame operand produces false positives; running a previously prepared method on a lexicon revised by Lexicon::swapped refuses LexiconRevision. The negative controls are only meaningful if they actually execute and discriminate.

**Important limit:** Physical::CoAddressed uses a producer-prepared head-frame lane. The producer does a cross-row lookup at ingestion. The test neither approves that producer for production nor relaxes the read-into-foreign-address restrictions.

## P1: CE64 role-mask equality

Using the existing CausalEdge64::set_orientation setter, construct a borrowed u64 edge lane. Existing Quack Cmp::MatchU64 selects the three orientation bits 43..45 with pattern = role_code << 43, care = 0b111 << 43; Agg::Count executes through the existing mask-risc evaluator.

**Independent oracle:** CausalEdge64::orientation() scalar accessor, over row counts 0,1,63,64,65,256,257. Flipping Epi5 Topology2 bit 59 must not change the role count; changing one Recipient to Core must decrease it by exactly one. These tests do not prove inferred Haspelmath case roles, general CE64 inference or legal graph traversal.

## How to run

~~~sh
cargo test --manifest-path crates/grammar-constraints-probe/Cargo.toml --test logic_neutrino_p0
~~~

Repository policy requires the sibling AdaWorldAPI/ndarray checkout at ../../../ndarray and forbids substituting crates.io. The current authoring environment has no Rust toolchain or GitHub network, so **tests are not claimed green**. Keep the PR draft until an actual runner records the test result and dependency revisions.

## Acceptance and kill conditions

Pass requires P0 typed distinct answers, nonvacuous ablation, wrong-description and revision refusals, and P1 CE64-oracle parity including both orthogonal bit mutations. A failed compile is not passing evidence. Stop the u8 hypothesis if the selector must itself encode all semantic obligations, if removal of WHEN does not create wrong candidates, or if role masks disagree with the canonical CE64 accessor.

**Explicitly deferred:** actual Ruff Rails and Kythe site parity; MeTTa/MORK multiset substitution; Datafrog fixpoint; board-game history/coal twins; NNUE delta refresh; XLOG chained joins and F100. The seven-branch mindmap remains open to all of them.