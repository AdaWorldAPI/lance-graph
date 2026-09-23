//! The boundary — the ONLY module of this crate that accepts text.
//!
//! ```text
//! external / raw input
//!       ↓ parse / normalize / canonicalize          (here)
//! KV  (ContentId → bytes)          CAM ((FieldId, ordinal) ↔ ContentId)
//!       ↓                                  ↓
//! fixed-width refs / ordinals / numeric lanes → AbiBatch → plan → fold
//!       ↓
//! aggregate coordinates → terminal → CAM → KV → human-readable text
//! ```
//!
//! **Human-readable identity belongs in CAM, not in the fold substrate.**
//!
//! * [`MemKv`] owns raw facts: it implements the contract's content-addressed
//!   [`ContentStore`] / [`ContentSink`] (the workspace's existing KV
//!   ownership surface — not a new trait). Raw variable-size values never
//!   enter a lane; a `U64` lane carries their [`ContentId`].
//! * [`CamLabels`] is a codebook per field: ordinal ↔ `ContentId` of the
//!   label text. It stores NO text and hashes no text at scan time — lookups
//!   key on the content address. Renaming an ordinal re-points it at new
//!   bytes; the ordinal, and so every coordinate, mask, fold and cached
//!   aggregate that uses it, is untouched (S5, S12).
//! * [`Catalog`] resolves a field NAME to a [`FieldId`] once, when an adapter
//!   (z8run config, a SQL front-end) builds a plan. A plan never holds the
//!   name (S4).
//!
//! Every lookup, insertion, persist and dereference is counted, so a test can
//! prove the fold did none of them and the terminal did only result-sized work.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use lance_graph_contract::content_store::{ContentId, ContentSink, ContentStore};

use crate::ids::FieldId;

/// Boundary counters (relaxed atomics: diagnostics, not synchronization).
#[derive(Debug, Default)]
pub struct BoundaryCounters {
    /// Raw values persisted to KV.
    pub kv_puts: AtomicU64,
    /// KV dereferences (all happen at a terminal adapter; the fold has no KV).
    pub kv_derefs: AtomicU64,
    /// CAM ordinal lookups (label → ordinal).
    pub cam_lookups: AtomicU64,
    /// CAM insertions (a new label minted an ordinal).
    pub cam_insertions: AtomicU64,
    /// CAM resolutions (ordinal → label), terminal only.
    pub cam_resolutions: AtomicU64,
}

fn bump(c: &AtomicU64) {
    c.fetch_add(1, Ordering::Relaxed);
}

impl BoundaryCounters {
    /// Read one counter.
    pub fn get(c: &AtomicU64) -> u64 {
        c.load(Ordering::Relaxed)
    }
}

/// An in-memory KV over the contract's content-addressed store traits.
#[derive(Debug, Default)]
pub struct MemKv {
    map: HashMap<ContentId, Box<[u8]>>,
    /// Counters.
    pub counters: BoundaryCounters,
}

impl ContentStore for MemKv {
    fn resolve(&self, id: ContentId) -> Option<&[u8]> {
        bump(&self.counters.kv_derefs);
        self.map.get(&id).map(|b| &b[..])
    }
}

impl ContentSink for MemKv {
    fn put(&mut self, bytes: &[u8]) -> ContentId {
        bump(&self.counters.kv_puts);
        let id = ContentId::of(bytes);
        self.map.entry(id).or_insert_with(|| bytes.into());
        id
    }
}

impl MemKv {
    /// Persist raw values; the lane that enters the batch is their addresses.
    pub fn persist<'a>(&mut self, raw: impl IntoIterator<Item = &'a [u8]>) -> Arc<[u64]> {
        raw.into_iter().map(|b| self.put(b).0).collect()
    }

    /// Dereference one reference as text (terminal use).
    pub fn text(&self, id: ContentId) -> Option<&str> {
        self.resolve(id).and_then(|b| std::str::from_utf8(b).ok())
    }
}

/// The CAM label codebook: per field, ordinal ↔ content address of its label.
#[derive(Debug, Default)]
pub struct CamLabels {
    fields: HashMap<FieldId, Codebook>,
    /// Counters.
    pub counters: BoundaryCounters,
}

#[derive(Debug, Default)]
struct Codebook {
    by_ordinal: Vec<ContentId>,
    by_content: HashMap<ContentId, u32>,
}

impl CamLabels {
    /// Canonicalize one textual value of `field` to its ordinal, minting one
    /// (and persisting the label text in `kv`) if it is new. Ingest only.
    pub fn intern(&mut self, field: FieldId, label: &str, kv: &mut MemKv) -> u32 {
        bump(&self.counters.cam_lookups);
        let id = ContentId::of_str(label);
        let book = self.fields.entry(field).or_default();
        if let Some(&o) = book.by_content.get(&id) {
            return o;
        }
        bump(&self.counters.cam_insertions);
        kv.put_str(label);
        let o = book.by_ordinal.len() as u32;
        book.by_ordinal.push(id);
        book.by_content.insert(id, o);
        o
    }

    /// Canonicalize a whole categorical column at ingest: text in, an
    /// ordinal lane out. The text does not survive past this call.
    pub fn canonicalize<'a>(
        &mut self,
        field: FieldId,
        raw: impl IntoIterator<Item = &'a str>,
        kv: &mut MemKv,
    ) -> Arc<[u32]> {
        raw.into_iter().map(|s| self.intern(field, s, kv)).collect()
    }

    /// Look up an existing label's ordinal (adapter use: resolving a filter
    /// literal before it enters a plan). Never mints.
    pub fn ordinal(&self, field: FieldId, label: &str) -> Option<u32> {
        bump(&self.counters.cam_lookups);
        self.fields
            .get(&field)?
            .by_content
            .get(&ContentId::of_str(label))
            .copied()
    }

    /// The domain size of a field's codebook (ordinals `0..domain`).
    pub fn domain(&self, field: FieldId) -> u32 {
        self.fields
            .get(&field)
            .map_or(0, |b| b.by_ordinal.len() as u32)
    }

    /// Resolve an ordinal to its label text — the terminal boundary only.
    pub fn resolve<'k>(&self, field: FieldId, ordinal: u32, kv: &'k MemKv) -> Option<&'k str> {
        bump(&self.counters.cam_resolutions);
        let id = *self.fields.get(&field)?.by_ordinal.get(ordinal as usize)?;
        kv.text(id)
    }

    /// Re-label an ordinal. The ordinal — the identity every coordinate,
    /// mask, fold and cache uses — does not change.
    pub fn rename(&mut self, field: FieldId, ordinal: u32, label: &str, kv: &mut MemKv) -> bool {
        let Some(book) = self.fields.get_mut(&field) else {
            return false;
        };
        let Some(slot) = book.by_ordinal.get_mut(ordinal as usize) else {
            return false;
        };
        let new = kv.put_str(label);
        book.by_content.remove(slot);
        *slot = new;
        book.by_content.insert(new, ordinal);
        true
    }
}

/// Field-name resolution for adapters: a name resolves to a [`FieldId`] once,
/// before a plan is built; the plan never carries it.
#[derive(Debug, Default, Clone)]
pub struct Catalog {
    names: Vec<(String, FieldId)>,
}

impl Catalog {
    /// Register a field name.
    pub fn with(mut self, name: impl Into<String>, id: FieldId) -> Self {
        self.names.push((name.into(), id));
        self
    }

    /// Resolve a name.
    pub fn field(&self, name: &str) -> Option<FieldId> {
        self.names
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, id)| *id)
    }

    /// The display name of a field (terminal headers, diagnostics).
    pub fn name(&self, id: FieldId) -> Option<&str> {
        self.names
            .iter()
            .find(|(_, i)| *i == id)
            .map(|(n, _)| n.as_str())
    }
}
