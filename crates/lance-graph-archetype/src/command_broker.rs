//! `CommandBroker` — a channel-based queue for deferred world mutations.
//!
//! Per ADR-0001 Decision 1, `CommandBroker` is the archetype-side
//! equivalent of Bevy's `Commands` / the Python ECS `CommandBroker`:
//! a FIFO queue of world mutations that accumulates during a
//! Processor pass and is drained by the World at tick boundaries.
//!
//! Stub-only at this stage — `submit` accepts commands, `drain` returns
//! what was submitted in order. Actual application-to-World logic lands
//! in DU-2.7 together with Entity wiring.

/// A deferred world-mutation command. The three variants cover the
/// ECS-standard operations (spawn a new entity, despawn an existing
/// one, or update components on one). Payloads are opaque at the
/// scaffold stage; DU-2.7 will parameterise them over concrete
/// `Component` types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    /// Spawn a new entity. The `u64` is a placeholder for the
    /// eventually-to-be-typed component bundle identifier.
    Spawn(u64),

    /// Despawn an entity by its integer ID.
    Despawn(u64),

    /// Update the entity identified by the first `u64` with the
    /// component-bundle identifier in the second `u64`.
    Update(u64, u64),
}

/// FIFO queue of deferred commands.
///
/// Used by Processors to schedule world mutations without mutating the
/// World mid-pass (which would break the Arrow-batch transcode model).
/// The World's tick driver calls `drain` at tick boundaries and applies
/// the commands in order. The scaffold uses a `Vec<Command>`; DU-2.7
/// may upgrade to a `std::sync::mpsc::channel` for multi-processor
/// concurrency.
#[derive(Debug, Default, Clone)]
pub struct CommandBroker {
    queue: Vec<Command>,
}

impl CommandBroker {
    /// Construct an empty broker. No allocation is performed until the
    /// first `submit`.
    pub fn new() -> Self {
        Self { queue: Vec::new() }
    }

    /// Enqueue a command. O(1) amortised.
    pub fn submit(&mut self, cmd: Command) {
        self.queue.push(cmd);
    }

    /// Drain all queued commands in insertion order. Returns an owned
    /// `Vec<Command>`; the broker is empty afterwards. O(n).
    pub fn drain(&mut self) -> Vec<Command> {
        std::mem::take(&mut self.queue)
    }

    /// Read the current queue length without draining.
    pub fn len(&self) -> usize {
        self.queue.len()
    }

    /// `true` iff no commands are queued.
    pub fn is_empty(&self) -> bool {
        self.queue.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_broker_is_empty() {
        let b = CommandBroker::new();
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
    }

    #[test]
    fn submit_and_drain_preserves_order() {
        let mut b = CommandBroker::new();
        b.submit(Command::Spawn(1));
        b.submit(Command::Update(1, 7));
        b.submit(Command::Despawn(1));
        assert_eq!(b.len(), 3);

        let drained = b.drain();
        assert_eq!(
            drained,
            vec![
                Command::Spawn(1),
                Command::Update(1, 7),
                Command::Despawn(1)
            ]
        );
        assert!(b.is_empty());
    }
}
