//! Role-based access control for the Ada cognitive stack.
//!
//! Central RBAC crate consumed by lance-graph, smb-office-rs, OpenClaw,
//! and any future consumer. Ties permissions directly to the ontology:
//! roles gate property-depth access (PrefetchDepth), predicate writes,
//! and action triggers — not abstract ACLs.
//!
//! Depends only on `lance-graph-contract`.

pub mod access;
pub mod permission;
pub mod policy;
pub mod role;
