//! # lance-graph-report-ogar — analytical results as addressed projection sources
//!
//! **A report may become a paper. The reporting engine must not become a
//! paper engine.** This crate adds no rendering ontology, no report
//! document, no report template language. It makes a `ReportResult` one
//! more *object* OGAR's existing composition machinery can address:
//!
//! ```text
//! DocCompose ─ ObjectSlot("ogar://lance-graph/report/7@revision:3", view "r.by_c")
//!      │                      │
//!      │     ogar_doc_ir::resolve_slot ── lookup(mode) ── view_by_name ── class gate
//!      │                      │
//!      │           DocObjectSource::grid_of   ← THIS crate: a view over Arc<CellSpace>
//!      ▼                      ▼
//!  ResolvedDoc{ Slot{ SlotOutcome::Grid } }  ──►  ogar-render-typst::emit_grid
//! ```
//!
//! [`ReportSource`] wraps ANY existing [`DocObjectSource`] (the caller's
//! object world) and adds reports beside it, so one `DocCompose` can arrange
//! `projection(A) · projection(report) · projection(B)` and one `resolve_doc`
//! resolves all three. Composition stores addresses and projection choices;
//! nothing about the result is copied into it.
//!
//! **Pivot rotation is a projection change, not document regeneration.** Two
//! slots on the same report through two grid views resolve to the SAME
//! `Arc<CellSpace>` re-viewed ([`lance_graph_report::ReportResult::with_roles`]);
//! this crate never holds a batch, so it cannot scan, fold or recompute.
//!
//! **Resolution modes are OGAR's, unchanged.** `Live` is the newest published
//! result; `Revision(n)` is the result computed at source generation `n`
//! (kept by `Arc`, not by copy); `Snapshot(_)` has no content-addressed
//! store here and resolves to the slot's own fallback (ActionText's
//! missing-object path) — no bespoke report-freezing system.
//!
//! **Labels resolve at the boundary.** `grid_of` is called by the resolver on
//! the way to a renderer; it resolves CAM labels for the presented members
//! only, and keeps every axis member's ordinal tuple (`GridAxis::keys`) so a
//! rendered cell traces back to its aggregate coordinate.

use std::cell::Cell;
use std::collections::{BTreeMap, HashMap};

use lance_graph_contract::class_view::{ClassId, WideFieldMask};
use lance_graph_contract::ontology::DisplayTemplate;
use lance_graph_contract::selection::{NamedView, RailGraph, ViewId, ViewRegistry};
use lance_graph_report::render::Terminal;
use lance_graph_report::{CellValue, CoordSpec, ReportResult};
use ogar_doc_ir::compose::{ObjectRef, ResolutionMode};
use ogar_doc_ir::resolve::{
    DocObjectSource, GridAxis, ResolvedBlock, ResolvedDoc, ResolvedGrid, SlotOutcome,
};

/// The `ObjectRef::app` a report is addressed under.
pub const REPORT_APP: &str = "lance-graph";
/// The `ObjectRef::class` a report is addressed under.
pub const REPORT_OBJECT_CLASS: &str = "report";

/// A report address after resolution: which report, and which revision
/// (`None` = live).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ReportKey {
    /// Report id (the `ObjectRef::id`, parsed once at lookup).
    pub id: u64,
    /// Pinned source generation, or `None` for live.
    pub revision: Option<u64>,
}

/// A graph key: the wrapped source's own object, or a report.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Key<K> {
    /// An object of the wrapped source.
    Object(K),
    /// A report result.
    Report(ReportKey),
}

/// The wrapped graph: the inner graph plus report nodes (leaves, no rails).
pub struct Graph<'a, G> {
    inner: &'a G,
    report_class: ClassId,
}

impl<G: RailGraph> RailGraph for Graph<'_, G> {
    type Key = Key<G::Key>;

    fn class_of(&self, key: Self::Key) -> ClassId {
        match key {
            Key::Object(k) => self.inner.class_of(k),
            Key::Report(_) => self.report_class,
        }
    }

    fn present_mask(&self, key: Self::Key) -> WideFieldMask {
        match key {
            Key::Object(k) => self.inner.present_mask(k),
            Key::Report(_) => WideFieldMask::EMPTY,
        }
    }

    fn rail_target(&self, key: Self::Key, position: u8) -> Option<Self::Key> {
        match key {
            Key::Object(k) => self.inner.rail_target(k, position).map(Key::Object),
            Key::Report(_) => None,
        }
    }
}

/// A grid view's orientation: which canonical coordinates play rows and
/// columns. Pure presentation — two orientations of one report share cells.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Orientation {
    /// Row coordinates, outermost first.
    pub rows: Vec<CoordSpec>,
    /// Column coordinates, outermost first.
    pub columns: Vec<CoordSpec>,
}

/// Counters that make the zero-work claims checkable.
#[derive(Debug, Default)]
pub struct AdapterStats {
    /// Grids projected (each is a re-view + label resolution; never a fold).
    pub grids_projected: Cell<u64>,
}

/// Reports beside an existing object source, as one [`DocObjectSource`].
pub struct ReportSource<'a, S: DocObjectSource> {
    inner: &'a S,
    graph: Graph<'a, S::Graph>,
    /// Report id → results by source generation. Values are `ReportResult`s
    /// (an `Arc` to the shared aggregate space + a view), never cell copies.
    reports: HashMap<u64, BTreeMap<u64, ReportResult>>,
    views: Vec<(String, ViewId, Orientation)>,
    terminal: Terminal<'a>,
    /// Counters.
    pub stats: AdapterStats,
}

impl<'a, S: DocObjectSource> ReportSource<'a, S> {
    /// Wrap `inner`; reports resolve as class `report_class` (the caller's
    /// minted classid — this crate mints nothing), and their labels resolve
    /// through `terminal`'s CAM / KV / catalog.
    pub fn new(inner: &'a S, report_class: ClassId, terminal: Terminal<'a>) -> Self {
        Self {
            inner,
            graph: Graph {
                inner: inner.graph(),
                report_class,
            },
            reports: HashMap::new(),
            views: Vec::new(),
            terminal,
            stats: AdapterStats::default(),
        }
    }

    /// Publish a result under a report id. It becomes the live result, and
    /// stays addressable as `Revision(generation)` of its source afterwards.
    /// Stores an `Arc`-sharing handle; the cells are not copied.
    pub fn publish(&mut self, id: u64, result: ReportResult) {
        let generation = u64::from(result.space().key().source.generation);
        self.reports
            .entry(id)
            .or_default()
            .insert(generation, result);
    }

    /// Register a named grid view (a projection choice) in the caller's
    /// registry. The name is the membrane spelling a `DocCompose` carries;
    /// resolution turns it into the returned `ViewId` once.
    pub fn register_grid_view(
        &mut self,
        registry: &mut ViewRegistry,
        name: impl Into<String>,
        orientation: Orientation,
    ) -> ViewId {
        let id = registry.register(NamedView::new(
            self.graph.report_class,
            WideFieldMask::EMPTY,
            DisplayTemplate::Detail,
        ));
        self.views.push((name.into(), id, orientation));
        id
    }

    fn result(&self, k: ReportKey) -> Option<&ReportResult> {
        let revs = self.reports.get(&k.id)?;
        match k.revision {
            None => revs.values().next_back(),
            Some(r) => revs.get(&r),
        }
    }

    fn project(&self, r: &ReportResult) -> ResolvedGrid {
        let g = self.terminal.grid(r);
        let fmt = |v: &CellValue| match v {
            CellValue::Int(i) => i.to_string(),
            CellValue::Real(x) => format!("{x}"),
            CellValue::Null => String::new(),
        };
        let (row_keys, row_labels, cells) = g
            .pages
            .first()
            .map(|(_, rows)| {
                let keys = rows.iter().map(|row| row.key.clone()).collect();
                let labels = rows.iter().map(|row| row.labels.clone()).collect();
                let cells = rows
                    .iter()
                    .flat_map(|row| row.cells.iter().flatten().map(fmt))
                    .collect();
                (keys, labels, cells)
            })
            .unwrap_or_default();
        ResolvedGrid {
            rows: GridAxis {
                keys: row_keys,
                labels: row_labels,
            },
            columns: GridAxis {
                keys: g.column_keys,
                labels: g.columns,
            },
            measures: g.measures,
            cells,
        }
    }
}

impl<'a, S: DocObjectSource> DocObjectSource for ReportSource<'a, S> {
    type Graph = Graph<'a, S::Graph>;

    fn graph(&self) -> &Self::Graph {
        &self.graph
    }

    fn lookup(
        &self,
        target: &ObjectRef,
        mode: &ResolutionMode,
    ) -> Option<<Self::Graph as RailGraph>::Key> {
        if target.app != REPORT_APP || target.class != REPORT_OBJECT_CLASS {
            return self.inner.lookup(target, mode).map(Key::Object);
        }
        let id: u64 = target.id.parse().ok()?;
        let key = match mode {
            ResolutionMode::Live => ReportKey { id, revision: None },
            ResolutionMode::Revision(n) => ReportKey {
                id,
                revision: Some(*n),
            },
            // No content-addressed report store: the slot's fallback answers.
            ResolutionMode::Snapshot(_) => return None,
        };
        self.result(key).map(|_| Key::Report(key))
    }

    fn value_of(&self, key: <Self::Graph as RailGraph>::Key, position: u8) -> Option<String> {
        match key {
            Key::Object(k) => self.inner.value_of(k, position),
            Key::Report(_) => None,
        }
    }

    fn view_by_name(&self, name: &str) -> Option<ViewId> {
        self.views
            .iter()
            .find(|(n, _, _)| n == name)
            .map(|(_, id, _)| *id)
            .or_else(|| self.inner.view_by_name(name))
    }

    fn view_for_class(&self, class: ClassId) -> Option<ViewId> {
        self.inner.view_for_class(class)
    }

    fn grid_of(&self, key: <Self::Graph as RailGraph>::Key, view: ViewId) -> Option<ResolvedGrid> {
        match key {
            Key::Object(k) => self.inner.grid_of(k, view),
            Key::Report(rk) => {
                let (_, _, o) = self.views.iter().find(|(_, id, _)| *id == view)?;
                // A re-view of the SAME aggregate space: metadata only.
                let r = self.result(rk)?.with_roles(&o.rows, &o.columns, &[]).ok()?;
                self.stats
                    .grids_projected
                    .set(self.stats.grids_projected.get() + 1);
                Some(self.project(&r))
            }
        }
    }
}

/// Emit a resolved document as Typst source through OGAR's emitters — the
/// thin block walk every OGAR consumer writes today (OGAR's `DocRenderer`
/// trait is still spec-only; this walker adds no layout of its own).
pub fn emit_typst(doc: &ResolvedDoc) -> String {
    use ogar_render_askama::FieldView;
    use ogar_render_typst as t;
    let mut out = String::new();
    for b in &doc.blocks {
        match b {
            ResolvedBlock::Heading(h) => out.push_str(&t::emit_heading(h)),
            ResolvedBlock::Text(x) => out.push_str(&t::emit_text(x)),
            ResolvedBlock::Slot(rs) => match &rs.outcome {
                SlotOutcome::Resolved { fields, .. } => {
                    let rows: Vec<FieldView> = fields
                        .iter()
                        .map(|f| FieldView {
                            position: f.position,
                            label: f.label.clone(),
                            predicate: String::new(),
                            value: f.value.clone(),
                        })
                        .collect();
                    out.push_str(&t::emit_field_view(&rs.class_view, &rs.uri, &rows));
                }
                SlotOutcome::Grid { grid, .. } => {
                    let nm = grid.measures.len();
                    let mut header = vec![String::new()];
                    for c in &grid.columns.labels {
                        for m in &grid.measures {
                            header.push(if nm == 1 {
                                c.join("/")
                            } else {
                                format!("{} · {m}", c.join("/"))
                            });
                        }
                    }
                    let rows: Vec<Vec<String>> = grid
                        .rows
                        .labels
                        .iter()
                        .enumerate()
                        .map(|(r, l)| {
                            let mut row = vec![l.join("/")];
                            for c in 0..grid.columns.keys.len() {
                                for m in 0..nm {
                                    row.push(grid.cell(r, c, m).unwrap_or("").to_string());
                                }
                            }
                            row
                        })
                        .collect();
                    out.push_str(&t::emit_grid(&rs.class_view, &rs.uri, &header, &rows));
                }
                SlotOutcome::Fallback { content_sha256_hex } => {
                    out.push_str(&t::emit_fallback(&rs.class_view, content_sha256_hex));
                }
                SlotOutcome::Unresolvable => {
                    out.push_str(&t::emit_text(&format!("unresolvable: {}", rs.uri)));
                }
            },
        }
    }
    out
}
