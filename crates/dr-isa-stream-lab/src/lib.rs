//! DR G2L x ISA stream lab.
//!
//! The question: can one source-witnessed DT64 lane binding feed two very
//! different computations without a second value layout?
//!
//! * [`fixture`] is the only source of truth for values and WHERE masks.
//! * [`source`] owns the lanes: one bf16 slab with a uniform pitch, or
//!   separately allocated lanes. It implements DT64's [`LaneSource`].
//! * [`bind`] binds demanded fields through the DT64 lane socket.
//! * [`cube`] is ARM A: 16 WHERE masks over a 256 x 256 identity cube, folded
//!   positionally into `SUM(lane p) WHERE mask m`.
//! * [`gemm`] is ARM B: the same 16 x 16 answer as a BF16 tile GEMM,
//!   `C[p][m] = sum_i lane_p[i] * mask_m(i)`, with A read in place from the
//!   slab when its geometry allows and staged otherwise.
//! * [`oracle`] computes the answer from the fixture alone.
//! * [`shape`] is the D-DT64-GEMM-SHAPE-1/2 probe: a WHERE-selected
//!   population routed through the binding into a virtual 64 x 64 or 256 x 256 BF16
//!   object, consumed as 16 output tiles by four arms.
//!
//! [`LaneSource`]: lance_graph_dt64::LaneSource

pub mod bind;
pub mod cube;
pub mod fixture;
pub mod gemm;
pub mod int8;
pub mod oracle;
pub mod shape;
pub mod source;
