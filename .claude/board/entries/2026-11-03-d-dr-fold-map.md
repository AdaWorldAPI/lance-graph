# 2026-11-03 — D-DR-FOLD-MAP: storage, addressing and algebra kept apart

**Status:** ARCHITECTURE MAP, read-only (`.claude/probes/2026-11-03-d-dr-fold/D-DR-FOLD-MAP.md`, tree `fb5e2d35b`). No code run; numbers quoted from earlier recorded measurements. No production change; mask-risc, EXO-140 and EXO-46 unchanged.

- **Storage (A):** no DR → physical-location translation, readahead or I/O/fold overlap exists in production. The address-mask → Lance prefilter chain is LAB only (H14c-4, H14d). A mask cuts bytes fetched only when presence is page-local. Every recorded `bytes_read` was served from the page cache, not read from disk. The physical read-planning contract is missing and has no owner.
- **Addressing (B):** `LocalMap` / `MapTable` / `FrameStore` (dt64) are the reusable coordinate mapping. No second binding store. `lance-graph-xlog::Cube` already keeps coordinate lifetime apart from presence lifetime (`DeltaLog` under one frame); its callers are lab crates. Reuse is amortization, not a different hot algorithm.
- **Algebra (C):** the frozen IR's only in-program cross-row reads are `Gather` / `EqU32Via`. `ScatterOrU32` is terminal-only. XLOG scatter-then-AND is LAB / UNCLASSIFIED. Tile compose (`relcomp`) is LAB and returns a transient C register. A vector×tile fold and cross-tile compose do not exist.
- For the D-BAG-FOLD shape and today's lowerings, Gather is required even for the hierarchical case. That claim covers that shape only. The 83 % Gather-avoidance figure is UNMEASURED.

**OPEN, owner decisions D1–D4:**
- D1: an in-program range-expand op;
- D2: the scatter-then-AND classification and outer-binding preservation;
- D3: binding lifetime vs presence writer;
- D4: owner of DR → Lance read planning.
