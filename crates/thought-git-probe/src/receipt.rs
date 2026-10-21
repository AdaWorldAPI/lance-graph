//! D-TGB-4: a receipt instead of history.
//!
//! A thought's receipt records what is needed to recompute its writes after
//! its live slot is gone: the sealed base it read, the program and its
//! reproducibility inputs, the seed, the execution configuration, the
//! sources it read and the provenance roots those sources carry. Replay runs
//! the program again from the receipt; it never reads the old slot.
//!
//! Comparison is by folds. A receipt carries two witnesses, one over the
//! values it read and one over the writes it produced. Replay recomputes
//! both and refuses on any difference. It does not keep the writes and does
//! not decode them to compare.
//!
//! A landed receipt lives in a [`MetadataTable`] keyed by
//! (dataset, data version, thought id), separate from the data version it
//! describes. Data and metadata are two writes, so the table distinguishes a
//! staged row from a sealed one. Replay accepts only a sealed row whose seal
//! still matches its contents, and only if the data version it names exists.
//!
//! Lab only. [`BaseRef`] stands in for D-CPG's `DurableRef` (dataset, branch,
//! version), which has no contract type yet; [`VersionStore`] stands in for
//! Lance. Nothing here touches `cycle_driver`, `LanceCycleWriter` or CE64.

use std::collections::BTreeMap;

use crate::fixtures::{Cell, Outcome, Thought};
use crate::oracle::{Givens, Version};
use crate::slots::{Slot, SlotPool, Stale};

/// A sealed version a thought reads from. Probe stand-in for D-CPG's
/// `DurableRef`; a version number alone is not enough once branches exist.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct BaseRef {
    pub dataset: u64,
    pub branch: u32,
    pub version: u64,
}

/// Sealed versions by reference. Probe stand-in for Lance.
#[derive(Debug, Clone, Default)]
pub struct VersionStore {
    pub givens: Givens,
    versions: BTreeMap<BaseRef, Version>,
}

impl VersionStore {
    #[must_use]
    pub fn new(givens: Givens) -> Self {
        Self {
            givens,
            versions: BTreeMap::new(),
        }
    }

    /// Seal `version` under `at`. A reference is sealed once; sealing it
    /// again with different contents is refused.
    ///
    /// # Errors
    /// [`AlreadySealed`] when `at` already holds a different version.
    pub fn seal(&mut self, at: BaseRef, version: Version) -> Result<(), AlreadySealed> {
        match self.versions.get(&at) {
            Some(v) if *v != version => Err(AlreadySealed(at)),
            Some(_) => Ok(()),
            None => {
                self.versions.insert(at, version);
                Ok(())
            }
        }
    }

    #[must_use]
    pub fn get(&self, at: BaseRef) -> Option<&Version> {
        self.versions.get(&at)
    }

    /// Test seam: replace the contents under `at` as if the sealed source had
    /// been changed behind the receipt's back. Replay must notice.
    pub fn tamper(&mut self, at: BaseRef, version: Version) {
        self.versions.insert(at, version);
    }
}

/// `at` already holds different contents.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AlreadySealed(pub BaseRef);

/// Which program produced the writes. The id names the template; `version`
/// is the template's revision, so an edited template is a different program.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ProgramId {
    pub id: u32,
    pub version: u32,
}

/// The one program this probe knows: for each target, read its source cell
/// from the base and write `source + jitter(seed, target)`, where an absent
/// source reads as 0 for the arithmetic only. Its inputs are the
/// reproducibility-relevant part of the receipt.
pub const DERIVE: ProgramId = ProgramId { id: 1, version: 1 };

/// The program's inputs as two lanes read at the same position `i`: the
/// source cell `sources[i]` feeds the target `targets[i]`. No pair is stored.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ProgramInputs {
    pub sources: Vec<Cell>,
    pub targets: Vec<Cell>,
}

impl ProgramInputs {
    /// # Panics
    /// When the lanes differ in length; a receipt with unequal lanes has no
    /// defined program.
    #[must_use]
    pub fn new(sources: Vec<Cell>, targets: Vec<Cell>) -> Self {
        assert_eq!(sources.len(), targets.len(), "lanes differ in length");
        Self { sources, targets }
    }
}

/// Execution settings that change the output. Recorded so a replay under a
/// different setting is refused rather than silently compared.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExecConfig {
    /// Revision of the jitter generator (SplitMix64, finalizer as published).
    pub rng_revision: u32,
    /// Jitter range: values are `0..modulus`.
    pub modulus: u64,
}

pub const EXEC_DEFAULT: ExecConfig = ExecConfig {
    rng_revision: 1,
    modulus: 1000,
};

/// A count plus an order-independent 64-bit fold. A witness, not a proof: two
/// different sets could fold equal, with probability about 2^-64 per pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Fold {
    pub count: u64,
    pub fold: u64,
}

impl Fold {
    fn add(&mut self, a: u64, b: u64) {
        self.count += 1;
        self.fold = self.fold.wrapping_add(mix(mix(a) ^ b.rotate_left(17)));
    }
}

/// SplitMix64's finalizer.
const fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

const fn cell_key(c: Cell) -> u64 {
    ((c.row as u64) << 8) | c.lane as u64
}

/// The fold of a set of writes.
#[must_use]
pub fn fold_writes(writes: &[(Cell, u64)]) -> Fold {
    let mut f = Fold::default();
    for &(c, v) in writes {
        f.add(cell_key(c), v);
    }
    f
}

/// The fold of the values the program read: each source cell with its value,
/// absence distinct from every value.
fn fold_sources(store: &VersionStore, base: &Version, inputs: &ProgramInputs) -> Fold {
    let mut f = Fold::default();
    for &src in &inputs.sources {
        let read = base.read(&store.givens, src);
        f.add(
            cell_key(src),
            read.map_or(u64::MAX, |v| mix(v) & !(1 << 63)),
        );
    }
    f
}

fn jitter(seed: u64, target: Cell, cfg: ExecConfig) -> u64 {
    mix(seed ^ mix(cell_key(target) ^ u64::from(cfg.rng_revision))) % cfg.modulus
}

/// Run the program. Pure: same arguments, same writes.
fn run(
    program: ProgramId,
    inputs: &ProgramInputs,
    seed: u64,
    cfg: ExecConfig,
    store: &VersionStore,
    base: &Version,
) -> Result<Vec<(Cell, u64)>, Refusal> {
    if program != DERIVE {
        return Err(Refusal::UnknownProgram(program));
    }
    Ok((0..inputs.targets.len())
        .map(|i| {
            let (src, dst) = (inputs.sources[i], inputs.targets[i]);
            let s = base.read(&store.givens, src).unwrap_or(0);
            (dst, s.wrapping_add(jitter(seed, dst, cfg)))
        })
        .collect())
}

/// Everything needed to recompute a thought's writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Receipt {
    pub thought: u32,
    pub base: BaseRef,
    pub program: ProgramId,
    pub inputs: ProgramInputs,
    /// `None` models a receipt written without its seed; replay refuses it.
    pub seed: Option<u64>,
    pub config: ExecConfig,
    /// The slot and generation the thought ran in. Provenance only: replay
    /// never consults it, and it is not part of [`Receipt::identity`].
    pub captured_in: Slot,
    /// Fold of the values read from `base`.
    pub sources: Fold,
    /// Provenance roots of those sources, as a bit set. Replayed material
    /// carries them as inherited, never as new.
    pub roots: u64,
    /// Fold of the writes the live run produced.
    pub output: Fold,
    pub outcome: Outcome,
}

/// The parts of a receipt that decide what replay produces. The live slot is
/// excluded: the same inputs in another slot are the same derivation.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ReceiptIdentity {
    pub thought: u32,
    pub base: BaseRef,
    pub program: ProgramId,
    pub inputs: ProgramInputs,
    pub seed: Option<u64>,
    pub config: ExecConfig,
    pub sources: Fold,
    pub roots: u64,
    pub output: Fold,
    /// Decides whether `next_version` lands the replayed writes.
    pub outcome: Outcome,
}

impl Receipt {
    #[must_use]
    pub fn identity(&self) -> ReceiptIdentity {
        ReceiptIdentity {
            thought: self.thought,
            base: self.base,
            program: self.program,
            inputs: self.inputs.clone(),
            seed: self.seed,
            config: self.config,
            sources: self.sources,
            roots: self.roots,
            output: self.output,
            outcome: self.outcome,
        }
    }
}

/// Why a capture or replay did not happen.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refusal {
    /// The slot handle is no longer live, or now holds another thought.
    Stale(Slot),
    /// The live slot holds `found`, not the receipt's thought.
    WrongThought {
        expected: u32,
        found: u32,
    },
    /// The base reference is not sealed in the store.
    UnknownBase(BaseRef),
    UnknownProgram(ProgramId),
    MissingSeed,
    /// The execution settings differ from the receipt's.
    ConfigChanged,
    /// The values read from the base differ from what the receipt recorded.
    SourceChanged {
        recorded: Fold,
        found: Fold,
    },
    /// The recomputed writes differ from the recorded output.
    OutputMismatch {
        recorded: Fold,
        found: Fold,
    },
    /// No metadata row under the key.
    MissingReceipt(MetaKey),
    /// The row was written but never sealed.
    Unsealed(MetaKey),
    /// The seal no longer matches the row's contents.
    Torn(MetaKey),
    /// The row names a data version that is not sealed.
    DataMissing(MetaKey),
    /// The landing names a different thought than the receipt.
    KeyMismatch(MetaKey),
}

impl From<Stale> for Refusal {
    fn from(s: Stale) -> Self {
        Self::Stale(s.0)
    }
}

/// What a thought declares before it runs.
#[derive(Debug, Clone)]
pub struct Plan {
    pub thought: u32,
    pub base: BaseRef,
    pub program: ProgramId,
    pub inputs: ProgramInputs,
    pub seed: u64,
    pub config: ExecConfig,
    pub roots: u64,
    pub outcome: Outcome,
}

/// Run a thought in its live slot and capture its receipt. Returns the
/// thought (its writes, for the caller to land or drop) and the receipt.
///
/// # Errors
/// [`Refusal::Stale`] / [`Refusal::WrongThought`] when `slot` does not hold
/// `plan.thought`; [`Refusal::UnknownBase`]; [`Refusal::UnknownProgram`].
pub fn capture(
    pool: &SlotPool,
    slot: Slot,
    store: &VersionStore,
    plan: Plan,
) -> Result<(Thought, Receipt), Refusal> {
    let found = pool.thought(slot)?;
    if found != plan.thought {
        return Err(Refusal::WrongThought {
            expected: plan.thought,
            found,
        });
    }
    let base = store
        .get(plan.base)
        .ok_or(Refusal::UnknownBase(plan.base))?;
    let writes = run(
        plan.program,
        &plan.inputs,
        plan.seed,
        plan.config,
        store,
        base,
    )?;
    let receipt = Receipt {
        thought: plan.thought,
        base: plan.base,
        program: plan.program,
        sources: fold_sources(store, base, &plan.inputs),
        inputs: plan.inputs,
        seed: Some(plan.seed),
        config: plan.config,
        captured_in: slot,
        roots: plan.roots,
        output: fold_writes(&writes),
        outcome: plan.outcome,
    };
    Ok((Thought::new(plan.thought, writes, plan.outcome), receipt))
}

/// A successful replay: the recomputed thought and the evidence it carries.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Replayed {
    pub thought: Thought,
    pub output: Fold,
    /// The receipt's roots, carried as inherited.
    pub inherited_roots: u64,
    /// Always 0: a replay has no root its receipt did not already name.
    pub independent_roots: u64,
}

/// Replay from the receipt alone, under `config`. Never reads a slot pool.
///
/// # Errors
/// Every [`Refusal`] from the seed, configuration, base, source and output
/// checks, in that order.
pub fn replay(
    receipt: &Receipt,
    store: &VersionStore,
    config: ExecConfig,
) -> Result<Replayed, Refusal> {
    let seed = receipt.seed.ok_or(Refusal::MissingSeed)?;
    if config != receipt.config {
        return Err(Refusal::ConfigChanged);
    }
    let base = store
        .get(receipt.base)
        .ok_or(Refusal::UnknownBase(receipt.base))?;
    let sources = fold_sources(store, base, &receipt.inputs);
    if sources != receipt.sources {
        return Err(Refusal::SourceChanged {
            recorded: receipt.sources,
            found: sources,
        });
    }
    let writes = run(receipt.program, &receipt.inputs, seed, config, store, base)?;
    let output = fold_writes(&writes);
    if output != receipt.output {
        return Err(Refusal::OutputMismatch {
            recorded: receipt.output,
            found: output,
        });
    }
    Ok(Replayed {
        thought: Thought::new(receipt.thought, writes, receipt.outcome),
        output,
        inherited_roots: receipt.roots,
        independent_roots: 0,
    })
}

/// Replay while the thought is still live: the slot must still hold it.
///
/// # Errors
/// [`Refusal::Stale`] / [`Refusal::WrongThought`] first, then as [`replay`].
pub fn replay_live(
    pool: &SlotPool,
    receipt: &Receipt,
    store: &VersionStore,
    config: ExecConfig,
) -> Result<Replayed, Refusal> {
    let found = pool.thought(receipt.captured_in)?;
    if found != receipt.thought {
        return Err(Refusal::WrongThought {
            expected: receipt.thought,
            found,
        });
    }
    replay(receipt, store, config)
}

/// Key of the metadata relation: the data version a thought landed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MetaKey {
    pub data: BaseRef,
    pub thought: u32,
}

/// One metadata row. `seal` is written last and covers every other field.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetaRow {
    pub receipt: Receipt,
    pub seal: Option<u64>,
}

const fn outcome_code(o: Outcome) -> u64 {
    match o {
        Outcome::Commit => 1,
        Outcome::Plan => 2,
        Outcome::Prune => 3,
        Outcome::Veto => 4,
    }
}

fn seal_of(key: MetaKey, r: &Receipt) -> u64 {
    let mut f = Fold::default();
    let ident = r.identity();
    f.add(key.data.dataset, key.data.version);
    f.add(u64::from(key.data.branch), u64::from(key.thought));
    f.add(ident.base.dataset, ident.base.version);
    f.add(u64::from(ident.base.branch), u64::from(ident.thought));
    f.add(
        u64::from(ident.program.id),
        u64::from(ident.program.version),
    );
    f.add(
        ident.seed.unwrap_or(u64::MAX),
        u64::from(ident.seed.is_some()),
    );
    f.add(u64::from(ident.config.rng_revision), ident.config.modulus);
    f.add(ident.sources.count, ident.sources.fold);
    f.add(ident.output.count, ident.output.fold);
    f.add(outcome_code(ident.outcome), 0);
    f.add(ident.roots, ident.inputs.targets.len() as u64);
    for i in 0..ident.inputs.targets.len() {
        let (s, d) = (ident.inputs.sources[i], ident.inputs.targets[i]);
        f.add(i as u64, (cell_key(s) << 32) | cell_key(d));
    }
    f.fold ^ f.count
}

/// The separate, version-linked relation holding landed receipts.
#[derive(Debug, Clone, Default)]
pub struct MetadataTable {
    rows: BTreeMap<MetaKey, MetaRow>,
}

impl MetadataTable {
    /// Step 1 of a landing, after the data version is sealed: write the row
    /// unsealed. The slot must still hold the thought.
    ///
    /// # Errors
    /// [`Refusal::Stale`] / [`Refusal::WrongThought`]; [`Refusal::KeyMismatch`]
    /// when `key.thought` is not the receipt's; [`Refusal::DataMissing`] when
    /// the data version is not sealed yet (data first, then metadata).
    pub fn stage(
        &mut self,
        pool: &SlotPool,
        store: &VersionStore,
        key: MetaKey,
        receipt: Receipt,
    ) -> Result<(), Refusal> {
        let found = pool.thought(receipt.captured_in)?;
        if found != receipt.thought {
            return Err(Refusal::WrongThought {
                expected: receipt.thought,
                found,
            });
        }
        if key.thought != receipt.thought {
            return Err(Refusal::KeyMismatch(key));
        }
        if store.get(key.data).is_none() {
            return Err(Refusal::DataMissing(key));
        }
        self.rows.insert(
            key,
            MetaRow {
                receipt,
                seal: None,
            },
        );
        Ok(())
    }

    /// Step 2: seal the staged row. Only a sealed row is a landing.
    ///
    /// # Errors
    /// [`Refusal::MissingReceipt`] when nothing was staged under `key`.
    pub fn seal(&mut self, key: MetaKey) -> Result<(), Refusal> {
        let row = self
            .rows
            .get_mut(&key)
            .ok_or(Refusal::MissingReceipt(key))?;
        row.seal = Some(seal_of(key, &row.receipt));
        Ok(())
    }

    #[must_use]
    pub fn row(&self, key: MetaKey) -> Option<&MetaRow> {
        self.rows.get(&key)
    }

    /// Test seam: a write that changed a row after it was sealed.
    pub fn row_mut(&mut self, key: MetaKey) -> Option<&mut MetaRow> {
        self.rows.get_mut(&key)
    }

    /// The sealed receipt under `key`, checked against its seal and against
    /// the data version it names.
    ///
    /// # Errors
    /// [`Refusal::MissingReceipt`], [`Refusal::Unsealed`], [`Refusal::Torn`],
    /// [`Refusal::DataMissing`].
    pub fn landed(&self, store: &VersionStore, key: MetaKey) -> Result<&Receipt, Refusal> {
        let row = self.rows.get(&key).ok_or(Refusal::MissingReceipt(key))?;
        let seal = row.seal.ok_or(Refusal::Unsealed(key))?;
        if seal != seal_of(key, &row.receipt) {
            return Err(Refusal::Torn(key));
        }
        if store.get(key.data).is_none() {
            return Err(Refusal::DataMissing(key));
        }
        Ok(&row.receipt)
    }

    /// Replay a landed thought from its sealed row. Needs no slot pool.
    ///
    /// # Errors
    /// As [`landed`](Self::landed), then as [`replay`].
    pub fn replay_landed(
        &self,
        store: &VersionStore,
        key: MetaKey,
        config: ExecConfig,
    ) -> Result<Replayed, Refusal> {
        replay(self.landed(store, key)?, store, config)
    }
}
