//! D-TGB-1: the slot pool.
//!
//! Slot 0 is the givens (main); slots 1..=63 hold live thoughts. `branch`
//! takes a free slot without any decision; `free` (squash, discard or land)
//! returns it. A thought may branch from another live thought; the pool keeps
//! that parent link in its own bookkeeping (never in the CE64 W bits, which
//! name a lane and point nowhere).
//!
//! When all 63 are taken, `branch` makes room for the new thought:
//!
//! 1. The policy names the candidate: the oldest ([`Eviction::Fifo`]) or the
//!    least used ([`Eviction::LeastUsed`], ties to the oldest) (operator,
//!    2026-10-20: "The easiest way in the beginning is Fifo vs pruning the
//!    least used lanes").
//! 2. If the candidate is in a dependency chain, it is squashed instead of
//!    lost: into its live parent, or, if it has none, its live child is
//!    squashed into it (operator, same day: "Squashing if the least used are
//!    in a dependency chain"). Nothing is discarded; the caller merges the
//!    writes (`oracle::squash`).
//! 3. Only a candidate with no live parent and no live child is evicted.
//!
//! A [`Slot`] handle carries the generation it was issued under. After its
//! slot is freed, squashed away or evicted, the old handle is refused, so a
//! recycled slot is never read as the thought that used to live in it.

/// Live slots: 1..=63 (slot 0 is the givens).
pub const LIVE_SLOTS: usize = 63;

/// Which live thought is the candidate when the pool is full.
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

/// How a full pool made room for a new branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pressure {
    /// `from` was squashed into `into`; `from`'s handle is now stale. The
    /// caller merges `from`'s writes over `into`'s.
    Squashed {
        thought: u32,
        from: Slot,
        into: Slot,
    },
    /// The thought had no live parent or child and gave way.
    Evicted { thought: u32, slot: Slot, uses: u64 },
}

/// A handle whose slot was freed, squashed away or evicted since it was issued.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Stale(pub Slot);

#[derive(Debug, Clone, Copy)]
struct Live {
    thought: u32,
    generation: u32,
    born: u64,
    uses: u64,
    /// Index of the live parent this thought branched from.
    parent: Option<usize>,
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

    /// Branch `thought`, from main (`None`) or from a live thought.
    ///
    /// # Errors
    /// [`Stale`] when `from` is no longer live.
    pub fn branch(
        &mut self,
        thought: u32,
        from: Option<Slot>,
    ) -> Result<(Slot, Option<Pressure>), Stale> {
        let mut parent = from.map(|s| self.live_index(s)).transpose()?;
        self.clock += 1;
        let (i, pressure) = match self.live.iter().position(Option::is_none) {
            Some(i) => (i, None),
            None => self.make_room(parent),
        };
        // The new thought's parent may have just been squashed away; it
        // then hangs off whatever absorbed it.
        if let (Some(p), Some(Pressure::Squashed { from, into, .. })) = (parent, pressure)
            && usize::from(from.index) - 1 == p
        {
            parent = Some(usize::from(into.index) - 1);
        }
        let generation = self.generations[i];
        self.live[i] = Some(Live {
            thought,
            generation,
            born: self.clock,
            uses: 0,
            parent,
        });
        Ok((self.handle(i, generation), pressure))
    }

    /// Record one use of the slot's lane (a write or a read).
    ///
    /// # Errors
    /// [`Stale`] when the slot is no longer live under this handle.
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

    /// Free the slot: the thought was squashed, discarded or landed. Its
    /// children now hang off its parent.
    ///
    /// # Errors
    /// [`Stale`] as for [`touch`](Self::touch).
    pub fn free(&mut self, slot: Slot) -> Result<(), Stale> {
        let i = self.live_index(slot)?;
        let parent = self.live[i].expect("checked live").parent;
        self.reparent(i, parent);
        self.release(i);
        Ok(())
    }

    /// `keep`: the new thought's parent, never the candidate and never the
    /// slot a squash releases (branching from a thought must not be what
    /// removes it).
    fn make_room(&mut self, keep: Option<usize>) -> (usize, Option<Pressure>) {
        let c = self.candidate(keep);
        let cand = self.live[c].expect("pool is full");
        let (from, into) = if let Some(p) = cand.parent {
            (c, p)
        } else if let Some(child) = self.oldest_child(c) {
            // When that child is the new thought's parent, squash the root
            // into it instead, so the parent handle stays live.
            if Some(child) == keep {
                (c, child)
            } else {
                (child, c)
            }
        } else {
            let slot = self.handle(c, cand.generation);
            self.release(c);
            return (
                c,
                Some(Pressure::Evicted {
                    thought: cand.thought,
                    slot,
                    uses: cand.uses,
                }),
            );
        };
        let f = self.live[from].expect("live");
        let t = self.live[into].expect("live");
        let pressure = Pressure::Squashed {
            thought: f.thought,
            from: self.handle(from, f.generation),
            into: self.handle(into, t.generation),
        };
        if let Some(target) = self.live[into].as_mut() {
            target.uses += f.uses;
        }
        self.reparent(from, Some(into));
        // A child absorbing its parent takes the parent's place in the tree.
        if let Some(target) = self.live[into].as_mut()
            && target.parent == Some(into)
        {
            target.parent = f.parent;
        }
        self.release(from);
        (from, Some(pressure))
    }

    fn candidate(&self, keep: Option<usize>) -> usize {
        let key = |l: &Live| match self.policy {
            Eviction::Fifo => (0, l.born),
            Eviction::LeastUsed => (l.uses, l.born),
        };
        self.live
            .iter()
            .enumerate()
            .filter(|&(i, _)| Some(i) != keep)
            .filter_map(|(i, s)| s.as_ref().map(|l| (i, key(l))))
            .min_by_key(|&(_, k)| k)
            .map(|(i, _)| i)
            .expect("pool is full")
    }

    fn oldest_child(&self, parent: usize) -> Option<usize> {
        self.live
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.filter(|l| l.parent == Some(parent)).map(|l| (i, l.born)))
            .min_by_key(|&(_, born)| born)
            .map(|(i, _)| i)
    }

    /// Hand `i`'s children to `to`.
    fn reparent(&mut self, i: usize, to: Option<usize>) {
        for l in self.live.iter_mut().flatten() {
            if l.parent == Some(i) {
                l.parent = to;
            }
        }
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
