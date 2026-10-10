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

**RESOLVED (same day):** "Classview can be used for any compute masking, we explicitly expanded the ERB redmine fieldview pattern for risk mask of everything." Rendering is one use of `classview`; render prefixes are values in that 64k mask space; `render_classid` stays.

**Likely fossil (operator, 2026-10-10: "Per app render prefix might be a fossil. Before may-june the domain and appid was on the right side"). The ledger confirms the history:**

- OGAR `DISCOVERY-MAP.md` D-APPCLASS (2026-06-22): `classid = APP(hi u16) ‖ class(lo u16)`. The codebook id (`0xDDCC` = domain:appid) sat on the RIGHT, an app prefix on the left.
- D-CLASSID-CANON-HIGH-FLIP (2026-07-02) moved domain:appid to the left and says *"APP_PREFIX values are unchanged … only their position moves (hi → lo)"*.

So the per-app render prefixes (`0x0001` OpenProject … `0x000C` Hiro) are the June APP half, carried verbatim into the `classview` bits without being re-derived as classview values.

Today they still occupy `classview` values `0x0000`–`0x000C` and are read by `render_classid` / `PortSpec::classview()`. Whether to retire them, and what replaces them, is the operator's call and NOT decided here. Until then, do not mint new code that depends on `APP_PREFIX` being a classview value.

**odoo-rs dates from that era (operator, 2026-10-10; verified):** first commit 2026-06-17. Its classid code (`od-ontology/src/ogar.rs`, `tests/classid_pins.rs`) landed 2026-07-06/07, right after the flip. Its pinned ids carry the fossil prefix `0x0002` in the classview bits: `0x0202_0002`, `0x0103_0002`, `0x0204_0002`, plus `id & 0xFFFF == 0x0002`. They also appear in generated Python/C#/Rust (`CLASSID = 0x02020002`) and the od-server `/compile` JSON (render_classid audit, §C.9.4). If the prefix is retired, odoo-rs is the first consumer to re-pin.

**`classview = 0x1000` is the V3 migration marker (operator, 2026-10-10: "we used 1000 in classview as a V3 migration marker as opposed to V1/V2"; verified):**
- Examples: `lance-graph-contract` `canonical_node.rs` `CLASSID_OSINT_V3 = 0x0701_1000`, `FMA_V3 0x0A01_1000`, `CPIC_V3 0x0E01_1000`, `PROJECT_V3 0x0101_1000`, `ERP_V3 0x0202_1000`; `ogar-osm` `CLASSVIEW_V3_SUBSTRATE = 0x1000`; `ogar-ro` `0x0306_1000`; `ogar-dismech` `0x0333_1000`.
- The code calls it temporary by declaration; its retirement is the plan's P4 operator checkpoint.
- So the classview bits hold two historical uses today: render prefixes `0x0000`–`0x000C` (the June fossil) and the V3 marker `0x1000`. Only convention keeps them apart (`AppPrefix::from_prefix(0x1000)` is `None`).
- Neither is the general compute-mask use the ruling describes. Do not mint new meanings at `0x1000` or in `0x0000`–`0x000C`.
