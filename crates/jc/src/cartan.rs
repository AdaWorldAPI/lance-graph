//! Cartan-Kuranishi: existence via prolongation to involutive form.
//!
//! DEFERRED — needs learned-attention-mask infrastructure (coupled revival
//! candidate 3). When activated: learn per-transition attention masks on
//! nibble positions from Animal Farm data; verify that learned mask widths
//! reproduce role_keys slice widths (2000/2000/2000/900/70/60/30). Match
//! would prove the layout is intrinsic geometry (Cartan characters), not
//! arbitrary design.
//!
//! Ref: Cartan 1945 / Kuranishi 1957.
//! See: EPIPHANIES.md [FORMAL-SCAFFOLD] coupled revival track.

use crate::PillarResult;

pub fn prove() -> PillarResult {
    PillarResult::deferred(
        "Cartan-Kuranishi",
        "needs learned-attention-mask module (coupled revival candidate 3). \
         When masks reproduce role_keys widths [2000,2000,2000,900,70,60,30], \
         that's experimental proof the layout is intrinsic Cartan-character spectrum.",
    )
}
