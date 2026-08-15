//! Zarr -> `NodeRow`: the ERA5 0.25-degree global grid on the canonical SoA row.
//!
//! **Plan:** `.claude/plans/weather-soa-bake-v1.md`. This crate exists because
//! the weather R&D arc ran entirely as Python over a Zarr file and the
//! Zarr-to-row path did not exist.
//!
//! # The shape, in four facts
//!
//! * **One cell is one node.** The ERA5 0.25-degree grid is 721 x 1440 =
//!   **1,038,240 cells**; the canonical row is 512 bytes; so a full global
//!   timestep is **0.495 GiB**. The full 122-field capacity is a proven budget,
//!   but W1 deliberately commits only the 22 measured/needed fields across
//!   three L4 facets.
//! * **The key is the address.** `HEEL` = the 16-degree tile, `HIP` = position
//!   within it, `TWIG` dormant-reserved. The arc's hand-picked boxes become
//!   HEEL-prefix range scans rather than array slicing.
//! * **One timestep is one Lance version.** This standalone crate assembles
//!   the key + W1 facet image. With feature `canonical-row`, `canonical`
//!   places that image into the live `NodeRow` contract and can stream exact
//!   512-byte rows to the existing Lance publication path. No copied value-tail
//!   offset and no second dataset-version protocol are introduced.
//! * **Nothing in the payload says what a byte means.** The
//!   `(facet, pair, byte) -> (variable, level, unit, floor)` mapping is a
//!   ClassView-side manifest ([`crate::manifest`]), never a slot in the row.
//!
//! # Status
//!
//! W1 codec pieces and the streaming cell-image assembler are present.
//! Durable publication still requires a non-zero routable weather classid;
//! `0x0000_0000` is refused because it belongs to the canonical bootstrap
//! fallback ladder.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod bake;
/// Agreement bridge to the live 512-byte `NodeRow` contract.
#[cfg(feature = "canonical-row")]
pub mod canonical;
pub mod floor;
pub mod key;
pub mod lane;
pub mod manifest;
