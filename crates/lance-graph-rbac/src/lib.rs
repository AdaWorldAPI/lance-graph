//! Role-based access control for the Ada cognitive stack.
//!
//! Central RBAC crate consumed by lance-graph, smb-office-rs, OpenClaw,
//! and any future consumer. Ties permissions directly to the ontology:
//! roles gate property-depth access (PrefetchDepth), predicate writes,
//! and action triggers — not abstract ACLs.
//!
//! Depends only on `lance-graph-contract`.

pub mod access;
pub mod auth;
pub mod authorize;
pub mod permission;
pub mod policy;
pub mod role;

/// The §6 typed `granted` value-tenant surface, re-exported from the zero-dep
/// contract so kernel consumers reach it through `lance_graph_rbac` without an
/// extra import: [`ClassGrant`] `(target_classid, op_mask)` is the first-class
/// replacement for `project_role.permissions: text`, [`OpMask`] is its verb gate,
/// and [`grants_permit`] is the §5 stage-1 positive op-gate over a role's grant
/// slice. The richer [`permission::PermissionSpec`] (depth/predicate/action-name)
/// is the finer stage that layers above a passed verb gate.
pub use lance_graph_contract::rbac::{grants_permit, ClassGrant, OpMask};
