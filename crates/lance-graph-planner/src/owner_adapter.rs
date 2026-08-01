//! D-MBX-A6 — the `Outcome → KanbanMove` emit adapter (write-on-behalf).
//!
//! Arm #1 of the mailbox-kanban executor (`.claude/v3/knowledge/mailbox-kanban-model.md`
//! §"The two executor arms"). A [`StyleStrategy`](crate::strategy::style_strategy)
//! **surfaces** a lifecycle intent on the D-MBX-A6 carrier
//! ([`StrategyOutcome::intended_move`]) but never emits it — the intent is a
//! **bootstrap sentinel** (`mailbox 0`, `witness_chain_position 0`, the
//! zero-fallback ladder). This module is the deferred half named on the
//! `StrategyOutcome::intended_move` doc and in `style_strategy.rs`'s slice-scope
//! header: it **rebinds** that sentinel to the live owner and **casts** it
//! write-on-behalf to the [`BatchWriter`].
//!
//! ## The causal placement (operator, 2026-08-01) — ahead of the write, on purpose
//!
//! A [`KanbanMove`] is the **destination written on the parcel before dispatch**:
//! the transition the completed thought *intends* the mailbox to become. It
//! travels WITH the write descriptor and is cast *ahead* of persistence, so the
//! thinker announces where it is going and resumes immediately (write latency is
//! masked). It is NOT the tugboat: this adapter does **not** advance any
//! lifecycle. The lifecycle **step** is the *delivery scan after Lance accepts
//! the write* — `try_advance_phase` applied post-persistence, on the successful
//! `LanceVersion`. **No successful write ⇒ no applied step.** That post-write
//! application is a SEPARATE seam (the version-completion path), which must apply
//! *this* paired move — never manufacture a generic `next_phases().first()`
//! transition merely because some version appeared. This module owns only the
//! pre-write cast half.
//!
//! ## What it does NOT do (the doctrine boundary)
//!
//! - It does **not** mutate a mailbox. `MailboxSoaOwner::advance_phase` /
//!   `try_advance_phase` is the SOLE mutator, applied by the post-write step.
//! - There is **no ack** and no confirmation bookkeeping
//!   (`E-ACK-ELIMINATED-1`). The cast records intent ahead of the write; the
//!   durable record is the row's own `LanceVersion`, read via `crate::temporal`.
//! - It does **not** write as *itself*. The cast is `on_behalf` of the live
//!   owner — the v3 write-on-behalf iron rule. A move that already names a live
//!   owner is **never re-owned** ([`rebind_bootstrap`] returns `None`), so this
//!   adapter can never steal ownership from whoever cast it.

use lance_graph_contract::collapse_gate::MailboxId;
use lance_graph_contract::kanban::KanbanMove;

use crate::batch_writer::{BatchWriter, CastId};
use crate::traits::StrategyOutcome;

/// The bootstrap-intent owner sentinel (`mailbox 0`) — the zero-fallback ladder's
/// "no live owner yet" address. Matches `StyleStrategy::intended_move`.
const BOOTSTRAP_OWNER: MailboxId = 0;

/// The bootstrap-intent cycle sentinel (`witness_chain_position 0`) — no live
/// `current_cycle` exists at plan time; the owner overwrites it on adoption.
const BOOTSTRAP_CYCLE: u32 = 0;

/// Rebind a **bootstrap** [`KanbanMove`] to the live owner mailbox.
///
/// Returns `Some` with `mailbox`/`witness_chain_position` bound to `owner` /
/// `owner_cycle` — and every other field (`from`, `to`, `exec`) preserved
/// bit-for-bit, because the Rubicon crossing itself (`Planning → CognitiveWork`,
/// and hence the *derived* −550 ms Σ-commit window and the `Elixir` backend) is a
/// structural fact the strategy already decided; only the *owner* was unknown at
/// plan time.
///
/// Returns `None` when `mv` is **not** the documented bootstrap sentinel — i.e.
/// it already names a live owner. Rebinding such a move would STEAL ownership
/// from whoever cast it, which the write-on-behalf iron rule forbids; the caller
/// emits nothing.
#[must_use]
pub fn rebind_bootstrap(mv: KanbanMove, owner: MailboxId, owner_cycle: u32) -> Option<KanbanMove> {
    if mv.mailbox != BOOTSTRAP_OWNER || mv.witness_chain_position != BOOTSTRAP_CYCLE {
        return None;
    }
    Some(KanbanMove {
        mailbox: owner,
        witness_chain_position: owner_cycle,
        ..mv
    })
}

/// Emit a strategy's bootstrap lifecycle intent **write-on-behalf** of the live
/// owner: rebind [`StrategyOutcome::intended_move`] to `owner` and `cast` it onto
/// `writer` (ahead of the write — the "parcel address before dispatch").
///
/// Returns the [`CastId`] of the staged intent, or `None` when there is nothing
/// to emit — either the outcome carries no intent (`intended_move == None`) or
/// the intent is not a rebindable bootstrap sentinel (already owned; see
/// [`rebind_bootstrap`]). Emitting nothing is the correct, silent outcome in
/// both cases — never an error and never a mailbox mutation.
///
/// `payload` is the write descriptor the sink drains (zero-copy: a descriptor,
/// not owned delta bytes — see [`BatchWriter`]); this adapter is payload-generic
/// and never inspects it.
pub fn emit_bootstrap_intent<P>(
    outcome: &StrategyOutcome,
    owner: MailboxId,
    owner_cycle: u32,
    writer: &mut BatchWriter<P>,
    payload: P,
) -> Option<CastId> {
    let rebound = rebind_bootstrap(outcome.intended_move?, owner, owner_cycle)?;
    Some(writer.cast(owner, vec![rebound], payload))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::kanban::{ExecTarget, KanbanColumn};

    /// The exact bootstrap intent a `StyleStrategy` surfaces (owner 0, cycle 0,
    /// the `Planning → CognitiveWork` crossing on the `Elixir` backend) — kept in
    /// lockstep with `StyleStrategy::intended_move`.
    fn bootstrap_move() -> KanbanMove {
        KanbanMove {
            mailbox: BOOTSTRAP_OWNER,
            from: KanbanColumn::Planning,
            to: KanbanColumn::CognitiveWork,
            witness_chain_position: BOOTSTRAP_CYCLE,
            exec: ExecTarget::Elixir,
        }
    }

    #[test]
    fn rebind_binds_the_sentinel_to_the_live_owner_and_preserves_the_crossing() {
        let rebound = rebind_bootstrap(bootstrap_move(), 42, 9).expect("bootstrap rebinds");
        // Anti-vacuity: the two sentinel fields ACTUALLY changed to the live values,
        // not merely "is Some" — 0 → 42 and 0 → 9.
        assert_ne!(
            rebound.mailbox, BOOTSTRAP_OWNER,
            "owner sentinel was rebound"
        );
        assert_eq!(rebound.mailbox, 42, "rebound to the live owner");
        assert_ne!(
            rebound.witness_chain_position, BOOTSTRAP_CYCLE,
            "cycle sentinel was rebound"
        );
        assert_eq!(
            rebound.witness_chain_position, 9,
            "to the owner's live cycle"
        );
        // The Rubicon crossing is preserved bit-for-bit (only the owner was
        // unknown at plan time) — so the DERIVED Σ-commit window is preserved too.
        assert_eq!(rebound.from, KanbanColumn::Planning);
        assert_eq!(rebound.to, KanbanColumn::CognitiveWork);
        assert_eq!(rebound.exec, ExecTarget::Elixir, "the backend survives");
    }

    #[test]
    fn rebind_refuses_to_re_own_a_move_that_already_names_a_live_owner() {
        // No-theft: a move already owned by mailbox 7 must NOT be rebound to 42.
        // The guard fires (returns None); the owner param does not leak in.
        let owned = KanbanMove {
            mailbox: 7,
            witness_chain_position: 3,
            ..bootstrap_move()
        };
        assert_eq!(
            rebind_bootstrap(owned, 42, 9),
            None,
            "an already-owned move is never re-owned (write-on-behalf iron rule)"
        );
        // A partial sentinel (owner 0 but a non-zero cycle) is also NOT a clean
        // bootstrap — it is refused rather than silently rebound.
        let partial = KanbanMove {
            mailbox: BOOTSTRAP_OWNER,
            witness_chain_position: 5,
            ..bootstrap_move()
        };
        assert_eq!(
            rebind_bootstrap(partial, 42, 9),
            None,
            "partial sentinel is refused"
        );
    }

    #[test]
    fn emit_casts_the_rebound_move_on_behalf_of_the_live_owner() {
        let mut w: BatchWriter<u8> = BatchWriter::new();
        let outcome = StrategyOutcome {
            reliability: 0.75,
            intended_move: Some(bootstrap_move()),
        };
        let cast = emit_bootstrap_intent(&outcome, 42, 9, &mut w, 0u8).expect("emits a cast");

        // The cast is recorded ON BEHALF OF the live owner (42), never as owner 0.
        assert_eq!(
            w.on_behalf_of(cast),
            Some(42),
            "write-on-behalf of the live owner"
        );
        let moves = w.intent_moves(cast).expect("intent recorded");
        assert_eq!(moves.len(), 1);
        // The staged move is the REBOUND one (live owner + cycle), not the sentinel.
        assert_eq!(moves[0].mailbox, 42);
        assert_eq!(moves[0].witness_chain_position, 9);
        assert_eq!(moves[0].to, KanbanColumn::CognitiveWork);
    }

    #[test]
    fn emit_is_a_silent_no_op_when_the_outcome_carries_no_intent() {
        let mut w: BatchWriter<u8> = BatchWriter::new();
        let outcome = StrategyOutcome {
            reliability: 0.5,
            intended_move: None,
        };
        assert_eq!(emit_bootstrap_intent(&outcome, 42, 9, &mut w, 0u8), None);
        // Non-vacuous silence: NOTHING was staged (the writer has no casts).
        assert!(w.casts().is_empty(), "no intent → no cast recorded");
    }

    #[test]
    fn emit_refuses_an_already_owned_intent_and_stages_nothing() {
        // The no-theft guarantee at the emit level: an outcome whose intent
        // already names a live owner produces no cast at all.
        let mut w: BatchWriter<u8> = BatchWriter::new();
        let outcome = StrategyOutcome {
            reliability: 0.9,
            intended_move: Some(KanbanMove {
                mailbox: 7,
                witness_chain_position: 3,
                ..bootstrap_move()
            }),
        };
        assert_eq!(emit_bootstrap_intent(&outcome, 42, 9, &mut w, 0u8), None);
        assert!(
            w.casts().is_empty(),
            "an owned intent is never re-cast on behalf of 42"
        );
    }
}
