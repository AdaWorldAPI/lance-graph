# Wang, Li & Hammer 2018 — Self-models require internal experiences and mental operations

**Paper:** Pei Wang, Xiang Li, Patrick Hammer, *Self in NARS, an AGI System*. *Frontiers in Robotics and AI* 5:20 (2018).  
**Source:** https://www.frontiersin.org/journals/robotics-and-ai/articles/10.3389/frobt.2018.00020/full  
**DOI:** https://doi.org/10.3389/frobt.2018.00020  
**Read:** 2026-10-11.

## Punchline

> **NARS applies its ordinary experience-processing machinery to internal events, and supplements automatic attention with mentally executable acts of wondering, considering, doubting, and self-control. Trace recording is necessary for that design but is not equivalent to self-regulation.**

## Author claims and boundaries

The authors describe NARS under AIKR, with a *SELF* concept, mental operations and internal experience. They argue these provide early, incomplete self-awareness and self-control. Their broader philosophical interpretation involving consciousness is **their thesis**, not an established empirical test for phenomenal experience.

NARS does not assume complete access to its own internal states; it sees salient parts selected under resource limits.

### Mental operations (§3.3)

- generate a task of a different type (a judgment can trigger a goal or question);
- reduce confidence by *doubting* previously accepted evidence;
- deliberately activate/consider a concept without waiting for a new external task;
- infer and act over internal operations, not just external body actions.

NARS retains a fixed built-in meta-level (grammar, inference rules, mental operators); it does **not** claim it can rewrite arbitrary source code or invent unlimited new ISA instructions.

### Internal experience (§3.4)

- input experience is a stream of Narsese sentences with elapsed intervals;
- mental events can be represented similarly to external events;
- selected concepts and derivation relations become inspectable internal events;
- the system can answer questions about what it thought and why it derived a conclusion;
- sensory channels, internal sensations and global buffers integrate events;
- knowledge of its own processing is **incomplete, uncertain and revisable**.

### Feelings and appraisal (§3.5)

The paper defines event-level and system-level satisfaction measures used for resource allocation. It treats "emotional" concepts as internal symbolic descriptions of appraisal, without requiring anthropomorphic labels for our machine registers.

### Deliberate control (§4.3)

The paper draws a boundary between default hard-coded automatic mechanisms and actions resulting from case-by-case reflective deliberation. Not every operation performed by an agent on itself qualifies as self-control.

## Project transfer: thought ≠ edge ≠ witness

`
ALU/ISA contract    deterministic operations over bytes
Thought/Recipe      selects and authorizes operation
State edge/CE64     stores typed output/result under declared reading
Witness/receipt     records that a transition really happened
Self model          reconstructs past behavior from receipts + semantics
Mental act          chooses a *new* allowed question, focus or control action
`

**This mapping is a substrate proposal**, not the NARS implementation described in the paper.

## Falsifier chain

1. Produce a normal conclusion with a fully resolvable execution receipt.
2. Ask a self-query "what supporting steps led to this conclusion?" and require exact recorded steps, not prose reconstruction.
3. Alter only an evidence source, leaving compiled recipe fixed, and measure which explanation changes.
4. Introduce an expected-but-missing inference: detect actual missing step, not fake it.
5. Permit a deliberate `consider` operation to reorder bounded attention *without* changing truth bytes in unrelated states.
6. Request `doubt` on a belief with weak provenance; lower belief confidence under the explicit mental operation, **not** by changing the semantics of the u8 register.
7. Attempt unauthorized write to another owner's state; refuse.
8. Compare with an arm that merely records the same receipts but has no self-query or control: isolate whether self-modeling changes outcomes.

## Limits and negative controls

The article is theoretical and reports preliminary implementation, not proof of scalable AGI, universal consciousness or current parity with lance-graph. A cross-run self-report must never outrank immutable history. Avoid hard-coding "emotion" into an ALU bit without an explicit type/reading.

**Connections:** Wang 2026 NAL-9; Nolte et al. arXiv:2503.13467; D2 provenance; project first-divergence + ablation.
