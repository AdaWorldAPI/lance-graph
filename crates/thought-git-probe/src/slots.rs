//! D-TGB-1: the slot pool.
//!
//! Slot 0 is the givens (main); slots 1..=63 hold live thoughts. `branch`
//! takes a free slot without any decision; `free` (squash, discard or land)
//! returns it. When all 63 are taken, `branch` evicts one live thought
//! according to the pool's [`Eviction`] policy (operator, 2026-10-20: "The
//! easiest way in the beginning is Fifo vs pruning the least used lanes").
//!
//! A [`Slot`] handle carries the generation it was issued under. After its
//! slot is freed or evicted, the old handle is refused, so a recycled slot is
//! never read as the thought that used to live in it.

/// Live slots: 1..=63 (slot 0 is the givens).
pub const LIVE_SLOTS: usize = 63;

/// Which live thought gives way when the pool is full.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eviction {
    /// The thought branched longest ago.
    Fifo,
    /// The thought whose lane was used least; ties go to the oldest.
    LeastUsed,
}

/// A handle to a live slot, valid while its generation matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Slot {
    /// 1..=63.
    pub index: u8,
    pub generation: u32,
}

/// The thought that gave way to a new branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Evicted {
    pub thought: u32,
    pub slot: Slot,
    pub uses: u64,
}

/// A handle whose slot was freed or evicted since it was issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stale(pub Slot);

#[derive(Debug, Clone, Copy)]
struct Live {
    thought: u32,
    generation: u32,
    born: u64,
    uses: u64,
}

/// The 63 live slots and their eviction policy.
#[derive(Debug, Clone)]
pub struct SlotPool {
    policy: Eviction,
    live: [Option<Live>; LIVE_SLOTS],
    generations: [u32; LIVE_SLOTS],
    clock: u64,
}

impl SlotPool {
    #[must_use]
    pub fn new(policy: Eviction) -> Self {
        Self {
            policy,
            live: [None; LIVE_SLOTS],
            generations: [0; LIVE_SLOTS],
            clock: 0,
        }
    }

    /// Live thoughts right now.
    #[must_use]
    pub fn live(&self) -> usize {
        self.live.iter().filter(|s| s.is_some()).count()
    }

    /// Branch `thought` into a slot: a free one if there is one, otherwise
    /// the one the policy evicts, which is returned.
    pub fn branch(&mut self, thought: u32) -> (Slot, Option<Evicted>) {
        self.clock += 1;
        let (i, evicted) = match self.live.iter().position(Option::is_none) {
            Some(i) => (i, None),
            None => {
                let i = self.victim();
                let old = self.live[i].expect("pool is full");
                let e = Evicted {
                    thought: old.thought,
                    slot: self.handle(i, old.generation),
                    uses: old.uses,
                };
                self.release(i);
                (i, Some(e))
            }
        };
        let generation = self.generations[i];
        self.live[i] = Some(Live {
            thought,
            generation,
            born: self.clock,
            uses: 0,
        });
        (self.handle(i, generation), evicted)
    }

    /// Record one use of the slot's lane (a write or a read).
    ///
    /// # Errors
    /// [`Stale`] when the slot was freed or evicted since `slot` was issued.
    pub fn touch(&mut self, slot: Slot) -> Result<(), Stale> {
        let i = self.live_index(slot)?;
        if let Some(l) = self.live[i].as_mut() {
            l.uses += 1;
        }
        Ok(())
    }

    /// The thought living in `slot`.
    ///
    /// # Errors
    /// [`Stale`] as for [`touch`](Self::touch).
    pub fn thought(&self, slot: Slot) -> Result<u32, Stale> {
        let i = self.live_index(slot)?;
        Ok(self.live[i].expect("checked live").thought)
    }

    /// Free the slot: the thought was squashed, discarded or landed.
    ///
    /// # Errors
    /// [`Stale`] as for [`touch`](Self::touch).
    pub fn free(&mut self, slot: Slot) -> Result<(), Stale> {
        let i = self.live_index(slot)?;
        self.release(i);
        Ok(())
    }

    fn victim(&self) -> usize {
        let key = |l: &Live| match self.policy {
            Eviction::Fifo => (0, l.born),
            Eviction::LeastUsed => (l.uses, l.born),
        };
        self.live
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.as_ref().map(|l| (i, key(l))))
            .min_by_key(|&(_, k)| k)
            .map(|(i, _)| i)
            .expect("pool is full")
    }

    fn release(&mut self, i: usize) {
        self.live[i] = None;
        self.generations[i] = self.generations[i].wrapping_add(1);
    }

    fn live_index(&self, slot: Slot) -> Result<usize, Stale> {
        let i = usize::from(slot.index).wrapping_sub(1);
        match self.live.get(i).copied().flatten() {
            Some(l) if l.generation == slot.generation => Ok(i),
            _ => Err(Stale(slot)),
        }
    }

    fn handle(&self, i: usize, generation: u32) -> Slot {
        Slot {
            index: u8::try_from(i + 1).expect("63 slots fit in u8"),
            generation,
        }
    }
}
