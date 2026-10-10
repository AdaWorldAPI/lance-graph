# 2026-10-10 — CLASSID-LAYOUT-RULING: domain(8)|appid(8)|concept(16); appid ≡ concept byte; domain immutable; 64k handed out

**Status:** RULING (operator), recorded verbatim. Plan: `.claude/plans/cross-glove-business-parity-v1.md` §C.9.3.

**The ruling:**
- "Domain 8 bit / Appid 8 bit / Concept 16 bit / Together 32 bit classid".
- Correction: "Appid/concept are synonymous". MedCare-rs uses "91..9E to have the concept id freed up"; before, "03:01..0E … was wasting the concept for defining the Ontology".
- "One is immutable, the other one is handed out in 64k size."

**Read it as:**
- the 16-bit codebook id `0xDDCC` = `domain:appid`;
- the domain is immutable;
- the low 16 bits are handed out in 64k blocks per `domain:appid`.

Today's `render_classid` high half already matches.

**The 16 bits are named `classview`** (operator spelling `domain : appid : classview`, OGAR `D-CLASSID-HI-U16-SPELLING`; `OGAR/crates/ogar-vocab/src/ports.rs:100` `PortSpec::classview()`). The classid comes in two widths:

- `domain:appid` (8:8) is the application-wide classid;
- `domain:appid:classview` (8:8:16) is the full classid, used e.g. for `ClassView` / `WideFieldMask`.

The same 16 bits also go by `APP_PREFIX` (older name, kept for existing callers) and the "custom half" (contract flip code). Prefer `classview`.

**Purpose of the 64k app-owned low half:** "so you can do cheap masking, spoG etc." — domain, domain:appid and app-defined sub-blocks are all selected by bit masks.

**Misreadings it corrects (this session):**
1. "9 of 12 edge targets are unminted, so blocked." Wrong: a missing shared-codebook id is not a blocker.
2. "appid and concept are different fields, so reverse `render_classid`." Wrong: they are the same byte, and reversal is not asked for.

**Do not repeat:** treating "absent from `ogar_vocab::class_ids`" as unaddressable; decoding classid halves by hand; reversing `render_classid`.

**OPEN:** how the per-app render prefix (today's low 16 bits) relates to the 64k concept space the ruling puts there.
