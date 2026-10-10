# 2026-10-10 — CLASSID-LAYOUT-RULING: domain(8) | appid(8) | concept(16) — a missing shared-codebook id is not a blocker

**Status:** RULING (operator), recorded verbatim. Plan: `.claude/plans/cross-glove-business-parity-v1.md` §C.9.3.

**The ruling:** "Domain 8 bit / Appid 8 bit / Concept 16 bit / Together 32 bit classid" — "That's the whole purpose of" it.

**The misreading it corrects (D-XGP-2/3, this session):** "9 of `BillableWorkEntry`'s 12 edge targets are unminted, so nothing can be anchored until OGAR mints them." Wrong. They lack only a SHARED codebook id. Each app mints its own 16-bit concepts in its own appid slot; convergence runs through the shared domain byte.

**Do not repeat:** do not treat "absent from `ogar_vocab::class_ids` / `CODEBOOK`" as "unaddressable" or "blocked on a shared mint". Ask instead: which app owns the concept, and what is its `domain | appid | concept`?

**OPEN:** the code has two layouts (V3 mint, `domain:appid` high; render lens, concept high). Their reconciliation is not decided here.
