# Wang's NAL-9 implementation — Internal operations are executed, buffered events

**Source:** Pei Wang, *NAL-9 Implementation*, 10-slide author-hosted presentation.  
**PDF:** https://cis.temple.edu/tagit/presentations/NAL%209.pdf  
**Read:** 2026-10-11. **Presentation**, not an independently verified arXiv paper. Date not asserted where PDF gives no explicit publication date.

## Punchline

> **NAL-9 closes a feedback loop by turning selected internal computations into inspectable experience and by enabling deliberated mental operations to influence future computations. A ninth recipe or bit is not NAL-9.**

## Actual slide content

- **Architecture:** memory/inference engine, task buffers (IEB and GEB), I/O channels with local buffers. Processing and resource allocation occur per working cycle.
- **Internal experience:** mental operation events in IEB, local item-level and global system-level feelings, actuators in working processes.
- **Uniform formalism:** external and internal experience represented/processed as Narsese; mental operators themselves are designed abstractions.
- **Local operations:** events of truth/desire/priority change, novelty, satisfaction, task invention (judgments, questions, goals), concept activation, reminders/hints.
- **Global operations:** status like satisfaction, health, novelty, load; appraisal can influence meta-goals but should not be confused with a directly executed objective.
- **Memory routine:** choose concept from bag; `^consider(c)` processes tasks; operation + premises + conclusions enter IEB.
- **Task buffers:** infer from novel tasks, move selected events across internal/global buffers and memory.
- **Channel routine:** `^observe`, `^request` / `^sense`, `^tell` / `^ask`, plus temporal/spatial inference in sensorimotor channels.
- **Issues still explicitly listed by Wang:** desire-value correlations, revised budget functions, task invention triggers, separating known (confidence) and believed (frequency), doubt/hesitation/trust, abstraction/generalization of problem strategies, revised OpenNARS design.

A slide listing an issue is **not evidence that it was implemented and validated**.

## Suggested minimal circuit

`
observed event -> operating recipe -> executed transition receipt
              -> self-reader -> internal experience
              -> bounded ^consider / ^doubt / new-question task
              -> authorized next recipe or focus change
              -> later outcome receipt
`

Test whether the control action actually changes future computation, and whether its diagnosis is supported by the prior receipt.

## Important distinctions

- NAL-9 is not nine serial steps or an AlphaTunnel rung.
- F/C are evidence-related values under a specific NARS reading; they are not automatically a stable hardware n count/CI.
- Internal self-observation and external Pearl intervention require different evidence rules.
- The presentation doesn't imply unrestricted self-modifying code.
- A register's bit-exactness enables replay, but a *self-model* needs explicit semantic and temporal linkage.

## Falsifiers

1. Inference trace enters internal readout exactly once per actual operation.
2. Inject a false self-report; immutable receipt rejects it.
3. Self-query influences attention only if a deliberate operator ran.
4. Repeat a root via two paths; internal attention must not inflate evidential confidence.
5. Deny write authority; no cross-owner effects from an observing focus.
6. Disable internal reader: measure exactly which correction behavior disappears.

**Further reading:** Wang et al. (2018), *Self in NARS*; Wang (2026), NARS introduction, NAL-1…9 ladder; Nolte et al. arXiv:2503.13467.
