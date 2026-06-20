//! `bridges::codebook` — shared canonical class_id constants for the
//! ports that emit through the OGAR codebook.
//!
//! Northstar plan §3 C4/C5 follow-up. Codex P1 on PR #559 flagged that
//! `OpenProjectBridge::entity("WorkPackage")` and
//! `RedmineBridge::entity("Issue")` minted **distinct** entity_type_ids
//! despite the PR body promising both routed to the same
//! `project_work_item` arm. Root cause: the registry's
//! [`crate::registry::RegistryState::append`] reuses `entity_type_id`
//! only when the exact `ogit_uri` matches, so per-port public names
//! never converge on a shared id by accident.
//!
//! This module is the single source of truth for the canonical class_ids;
//! both [`crate::bridges::OpenProjectBridge`] and
//! [`crate::bridges::RedmineBridge`] reference these constants in their
//! per-port public-name → class_id alias tables. Consumers that dispatch
//! on `EntityRef::schema_ptr.entity_type_id()` get the same id for
//! `WorkPackage`/`Issue`/etc. across both bridges — the cross-fork
//! convergence the codebook was calcified for.
//!
//! # Source of authority
//!
//! Mirrors `ogar-vocab::class_ids::*` (the calcified canonical codebook,
//! 32 promoted concepts). This crate intentionally **does not** depend
//! on `ogar-vocab` to avoid a cycle (OGAR depends on
//! lance-graph-contract, and lance-graph-ontology sits between them);
//! the constants below are the same values verbatim. Adding a
//! cross-crate `class_ids_match_ogar_vocab` smoke test in OGAR is a
//! roadmap item.
//!
//! # Coverage
//!
//! Project-management arm (0x01XX) — all 26 promoted concepts.
//! Commerce arm (0x02XX) and other arms — added as ports adopt them.

// ── Project-management arm (0x01XX) — 26 promoted concepts ────────────

/// `project` (0x0101) — the project itself. OpenProject `Project`,
/// Redmine `Project`.
pub const PROJECT: u16 = 0x0101;

/// `project_work_item` (0x0102) — the canonical work-item concept.
/// **OpenProject `WorkPackage`, Redmine `Issue` — the headline
/// convergence pin.**
pub const PROJECT_WORK_ITEM: u16 = 0x0102;

/// `billable_work_entry` (0x0103) — hours logged against a work item.
/// OpenProject `TimeEntry`, Redmine `TimeEntry`.
pub const BILLABLE_WORK_ENTRY: u16 = 0x0103;

/// `project_actor` (0x0104) — User / Principal in the port's terms.
pub const PROJECT_ACTOR: u16 = 0x0104;

/// `project_status` (0x0105) — OpenProject `Status`, Redmine
/// `IssueStatus`.
pub const PROJECT_STATUS: u16 = 0x0105;

/// `project_type` (0x0106) — OpenProject `Type`, Redmine `Tracker`.
pub const PROJECT_TYPE: u16 = 0x0106;

/// `priority` (0x0107).
pub const PRIORITY: u16 = 0x0107;

/// `project_membership` (0x0108) — OpenProject `Membership`, Redmine
/// `Member`.
pub const PROJECT_MEMBERSHIP: u16 = 0x0108;

/// `project_journal` (0x0109) — change history.
pub const PROJECT_JOURNAL: u16 = 0x0109;

/// `project_repository` (0x010A) — SCM repository attached to a project.
pub const PROJECT_REPOSITORY: u16 = 0x010A;

/// `project_version` (0x010B) — milestone / target version.
pub const PROJECT_VERSION: u16 = 0x010B;

/// `project_wiki_page` (0x010C).
pub const PROJECT_WIKI_PAGE: u16 = 0x010C;

/// `project_query` (0x010D) — saved filter / list-view spec.
pub const PROJECT_QUERY: u16 = 0x010D;

/// `project_attachment` (0x010E).
pub const PROJECT_ATTACHMENT: u16 = 0x010E;

/// `project_comment` (0x010F).
pub const PROJECT_COMMENT: u16 = 0x010F;

/// `project_custom_field` (0x0110).
pub const PROJECT_CUSTOM_FIELD: u16 = 0x0110;

/// `project_relation` (0x0111) — OpenProject `Relation`, Redmine
/// `IssueRelation`.
pub const PROJECT_RELATION: u16 = 0x0111;

/// `project_changeset` (0x0112).
pub const PROJECT_CHANGESET: u16 = 0x0112;

/// `project_watcher` (0x0113).
pub const PROJECT_WATCHER: u16 = 0x0113;

/// `project_news` (0x0114).
pub const PROJECT_NEWS: u16 = 0x0114;

/// `project_message` (0x0115) — forum message.
pub const PROJECT_MESSAGE: u16 = 0x0115;

/// `project_forum` (0x0116) — OpenProject `Forum`, Redmine `Board`.
pub const PROJECT_FORUM: u16 = 0x0116;

/// `project_role` (0x0117).
pub const PROJECT_ROLE: u16 = 0x0117;

/// `project_member_role` (0x0118) — join row between Membership and Role.
pub const PROJECT_MEMBER_ROLE: u16 = 0x0118;

/// `project_custom_value` (0x0119) — instance of a CustomField for one
/// work item.
pub const PROJECT_CUSTOM_VALUE: u16 = 0x0119;

/// `project_enabled_module` (0x011A) — per-project module toggle.
pub const PROJECT_ENABLED_MODULE: u16 = 0x011A;

#[cfg(test)]
mod tests {
    use super::*;

    /// Headline convergence pin: the canonical concept both
    /// OpenProject's `WorkPackage` and Redmine's `Issue` route through
    /// has the codebook id `0x0102`. The whole point of C4 + C5.
    #[test]
    fn project_work_item_is_0x0102() {
        assert_eq!(PROJECT_WORK_ITEM, 0x0102);
    }

    /// All 26 project-management arm ids are in the `0x01XX` block and
    /// strictly monotonic in the order they appear here — matches the
    /// codebook layout. Drift here means drift from the OGAR codebook.
    #[test]
    fn project_arm_ids_are_dense_monotone_0x0101_through_0x011a() {
        let ids = [
            PROJECT,
            PROJECT_WORK_ITEM,
            BILLABLE_WORK_ENTRY,
            PROJECT_ACTOR,
            PROJECT_STATUS,
            PROJECT_TYPE,
            PRIORITY,
            PROJECT_MEMBERSHIP,
            PROJECT_JOURNAL,
            PROJECT_REPOSITORY,
            PROJECT_VERSION,
            PROJECT_WIKI_PAGE,
            PROJECT_QUERY,
            PROJECT_ATTACHMENT,
            PROJECT_COMMENT,
            PROJECT_CUSTOM_FIELD,
            PROJECT_RELATION,
            PROJECT_CHANGESET,
            PROJECT_WATCHER,
            PROJECT_NEWS,
            PROJECT_MESSAGE,
            PROJECT_FORUM,
            PROJECT_ROLE,
            PROJECT_MEMBER_ROLE,
            PROJECT_CUSTOM_VALUE,
            PROJECT_ENABLED_MODULE,
        ];
        assert_eq!(ids.len(), 26);
        // All in the project-management arm.
        for id in ids {
            assert!(
                (0x0101..=0x011A).contains(&id),
                "id 0x{id:04X} outside project-arm block 0x0101..=0x011A"
            );
        }
        // Strictly monotone (codebook order pin).
        for w in ids.windows(2) {
            assert!(w[0] < w[1], "0x{:04X} should precede 0x{:04X}", w[0], w[1]);
        }
    }
}
