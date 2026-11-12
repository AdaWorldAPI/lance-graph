//! Ownership laws of [`MailboxSoA`](crate::mailbox_soa::MailboxSoA) that the
//! compiler enforces (D-SCHEMA-WARDROBE-P0). Compiled only under `cfg(doctest)`:
//! this module adds no item to the crate.
//!
//! Each law is a pair. The first block must compile; the second differs from
//! it by the one offending line and must not. Without the first block a
//! `compile_fail` would also pass on a typo.

/// One mailbox has one mutable borrower at a time.
///
/// ```
/// use cognitive_shader_driver::mailbox_soa::MailboxSoA;
/// use lance_graph_contract::kanban::KanbanColumn;
/// use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
/// let mut mb: MailboxSoA<8> = MailboxSoA::new(7, 0, 1.0);
/// let a = &mut mb;
/// a.advance_phase(KanbanColumn::CognitiveWork);
/// let b = &mut mb;
/// assert_eq!(b.phase(), KanbanColumn::CognitiveWork);
/// ```
///
/// ```compile_fail
/// use cognitive_shader_driver::mailbox_soa::MailboxSoA;
/// use lance_graph_contract::kanban::KanbanColumn;
/// use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
/// let mut mb: MailboxSoA<8> = MailboxSoA::new(7, 0, 1.0);
/// let a = &mut mb;
/// let b = &mut mb; // second adapter while the first still holds the mailbox
/// a.advance_phase(KanbanColumn::CognitiveWork);
/// assert_eq!(b.phase(), KanbanColumn::CognitiveWork);
/// ```
pub struct OneMutableBorrower;

/// A read view cannot advance the lifecycle.
///
/// ```
/// use cognitive_shader_driver::mailbox_soa::MailboxSoA;
/// use lance_graph_contract::kanban::KanbanColumn;
/// use lance_graph_contract::soa_view::MailboxSoaView;
/// let mb: MailboxSoA<8> = MailboxSoA::new(7, 0, 1.0);
/// let view: &dyn MailboxSoaView = &mb;
/// assert_eq!(view.phase(), KanbanColumn::Planning);
/// ```
///
/// ```compile_fail
/// use cognitive_shader_driver::mailbox_soa::MailboxSoA;
/// use lance_graph_contract::kanban::KanbanColumn;
/// use lance_graph_contract::soa_view::{MailboxSoaOwner, MailboxSoaView};
/// let mb: MailboxSoA<8> = MailboxSoA::new(7, 0, 1.0);
/// let view: &dyn MailboxSoaView = &mb;
/// view.advance_phase(KanbanColumn::CognitiveWork); // a view is not an owner
/// ```
pub struct AViewIsNotAnOwner;

/// A mailbox cannot be duplicated: there is no second resident population.
///
/// ```
/// use cognitive_shader_driver::mailbox_soa::MailboxSoA;
/// use lance_graph_contract::soa_view::MailboxSoaView;
/// let mb: MailboxSoA<8> = MailboxSoA::new(7, 0, 1.0);
/// let moved = mb;
/// assert_eq!(moved.mailbox_id(), 7);
/// ```
///
/// ```compile_fail
/// use cognitive_shader_driver::mailbox_soa::MailboxSoA;
/// use lance_graph_contract::soa_view::MailboxSoaView;
/// let mb: MailboxSoA<8> = MailboxSoA::new(7, 0, 1.0);
/// let copy = mb.clone(); // no Clone: an adapter cannot mint a twin
/// assert_eq!(copy.mailbox_id(), 7);
/// ```
///
/// ```compile_fail
/// use cognitive_shader_driver::mailbox_soa::MailboxSoA;
/// use lance_graph_contract::soa_view::MailboxSoaView;
/// let mb: MailboxSoA<8> = MailboxSoA::new(7, 0, 1.0);
/// let moved = mb;
/// assert_eq!(mb.mailbox_id(), 7); // the previous holder lost it
/// assert_eq!(moved.mailbox_id(), 7);
/// ```
pub struct NoSecondPopulation;
