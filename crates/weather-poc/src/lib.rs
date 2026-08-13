//! Zarr → `NodeRow`: the ERA5 0.25° global grid on the canonical SoA row.
//!
//! **Plan:** `.claude/plans/weather-soa-bake-v1.md`. This crate exists because
//! the weather R&D arc ran entirely as Python over a Zarr file — one variable,
//! four hand-picked 16° boxes, three hand-typed timesteps — and the reason was
//! structural: the Zarr→`NodeRow` path did not exist and no plan specified one.
//!
//! # The shape, in four facts
//!
//! * **One cell is one node.** The ERA5 0.25° grid is 721 × 1440 =
//!   **1,038,240 cells**; the canonical row is 512 bytes; so a full global
//!   timestep is **0.495 GiB**. All 122 ERA5 fields per cell (17 surface + 7
//!   pressure-level × 13 levels + 14 static) fit at 1 byte/field inside the
//!   **292 free slab bytes** — see [`crate::key`] and the plan §2.1 for why the
//!   budget is 292 and not 384.
//! * **The key is the address.** `HEEL` = the 16° × 16° tile, `HIP` = position
//!   within it, `TWIG` dormant-reserved. A lat/lon grid is the literal-x/y case
//!   of the 3×4 cascade, so the arc's hand-picked 16° boxes become **HEEL-prefix
//!   range scans** rather than array slicing.
//! * **One timestep is one Lance version.** Per `E-MARKOV-TEMPORAL-STREAM-1`,
//!   episodic = Lance versions; a time series is a version-range read. This
//!   crate never writes a version writer of its own.
//! * **Nothing in the payload says what a byte means.** The (facet, pair, byte)
//!   → (variable, level, unit, floor) mapping is a ClassView-side manifest
//!   ([`crate::manifest`]), never a slot in the row — the le-contract §2
//!   slot-purity rule.
//!
//! # Status
//!
//! Wave W1 is under construction. `D-WXS-0` (the classid mint) is **blocked**
//! on an OGAR-side, operator-gated decision, and until it resolves the bake
//! must **refuse to write** rather than emit rows under `0x0000_0000` — that
//! value belongs to the zero-fallback ladder and a dataset carrying it is
//! indistinguishable from a bootstrap row.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod floor;
pub mod key;
pub mod manifest;
