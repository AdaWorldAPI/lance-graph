//! callcenter-domain Layer-2 catalogue (per `I-VSA-IDENTITIES`).
//!
//! Sibling of [`crate::grammar::role_keys`] and the future
//! `persona::role_keys`: one identity fingerprint per concept, with
//! disjoint slice allocations. The 25 Odoo savants from
//! [`crate::savants::SAVANTS`] land here as the first set of
//! callcenter-domain identities.
//!
//! See `.claude/knowledge/vsa-switchboard-architecture.md` for the
//! three-layer Layer-2 catalogue doctrine and
//! `.claude/plans/odoo-savant-reasoners-v2.md` for the broader
//! composition-over-substrate reshape this module participates in.

pub mod role_keys;

pub use role_keys::{
    savant_role_key, savant_role_key_by_name, SAVANT_ROLE_KEYS, SAVANT_SLICE_END,
    SAVANT_SLICE_START, SAVANT_SLICE_WIDTH,
};
