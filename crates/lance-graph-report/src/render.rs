//! Terminal adapters — the materialization boundary.
//!
//! **A report may contain strings. The reporting engine does not.** This is
//! the one place aggregate coordinates meet text: each presented member is
//! resolved through CAM ([`CamLabels::resolve`]) exactly once per render, so
//! label work is proportional to the result's axis metadata, never to the
//! population. JSON and CSV walk the SAME [`Grid`]; the formats differ only
//! in how they print it (TEST 8).
//!
//! **Data export, not paper.** JSON and CSV are interchange terminals. Paged
//! and screen layout (HTML, Typst → PDF, A2UI) is NOT this crate's: a report
//! reaches those through OGAR's composition path — an `ObjectSlot` naming
//! the result, resolved by `ogar-doc-ir` (`SlotOutcome::Grid`) and emitted by
//! OGAR's renderers (see `lance-graph-report-ogar`). The reporting engine
//! does not become a paper engine; [`Grid`] is the renderer-neutral hand-off
//! both paths share.
//!
//! Detail exports ([`Terminal::detail_csv`]) obey **filter IDs first,
//! hydrate raw values last**: the selection runs as ids in the substrate,
//! the surviving rows are materialized by the one named materializer
//! (`lance_graph_mask_risc::materialize_rows`) HERE, and only those rows'
//! KV references are dereferenced.

use lance_graph_contract::content_store::ContentId;
use lance_graph_mask_risc::{
    execute_into, materialize_rows, words_for, Foreign, Out, Planes, Scratch,
};
use lance_graph_quack::{lower, Agg, Query};

use crate::batch::LaneData;
use crate::boundary::{CamLabels, Catalog, MemKv};
use crate::ids::FieldId;
use crate::plan::{AxisRole, CellValue, CoordSpec, ReportPlan};
use crate::result::ReportResult;
use crate::{AbiBatch, ReportError};

/// A rendered grid: the presented cells, labels resolved. Result-sized.
#[derive(Debug, Clone, PartialEq)]
pub struct Grid {
    /// Column keys: the ordinal tuple of each presented column (identity,
    /// kept for back-trace to the aggregate coordinate).
    pub column_keys: Vec<Vec<u32>>,
    /// Column header rows: one label tuple per column key.
    pub columns: Vec<Vec<String>>,
    /// Measure headers.
    pub measures: Vec<String>,
    /// Pages: (page labels, rows).
    pub pages: Vec<(Vec<String>, Vec<GridRow>)>,
    /// Grand total per measure.
    pub grand_total: Vec<CellValue>,
}

/// One presented row.
#[derive(Debug, Clone, PartialEq)]
pub struct GridRow {
    /// The row's ordinal tuple (identity, for back-trace).
    pub key: Vec<u32>,
    /// Row labels.
    pub labels: Vec<String>,
    /// `cells[col][measure]`.
    pub cells: Vec<Vec<CellValue>>,
    /// Row total per measure.
    pub total: Vec<CellValue>,
}

/// Render statistics.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RenderStats {
    /// Cells rendered.
    pub cells: u64,
    /// Output bytes.
    pub bytes: u64,
}

/// The terminal: the label store, the KV and the catalog, borrowed.
pub struct Terminal<'a> {
    /// CAM labels.
    pub cam: &'a CamLabels,
    /// KV.
    pub kv: &'a MemKv,
    /// Field names.
    pub catalog: &'a Catalog,
}

fn fmt_value(v: CellValue) -> String {
    match v {
        CellValue::Int(i) => i.to_string(),
        CellValue::Real(r) => format!("{r}"),
        CellValue::Null => String::new(),
    }
}

fn json_value(v: CellValue) -> String {
    match v {
        CellValue::Null => "null".to_string(),
        other => fmt_value(other),
    }
}

fn json_str(s: &str) -> String {
    let mut o = String::with_capacity(s.len() + 2);
    o.push('"');
    for c in s.chars() {
        match c {
            '"' => o.push_str("\\\""),
            '\\' => o.push_str("\\\\"),
            '\n' => o.push_str("\\n"),
            c if (c as u32) < 0x20 => o.push_str(&format!("\\u{:04x}", c as u32)),
            c => o.push(c),
        }
    }
    o.push('"');
    o
}

fn csv_field(s: &str) -> String {
    if s.contains([',', '"', '\n']) {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

impl Terminal<'_> {
    fn field_name(&self, f: FieldId) -> String {
        self.catalog
            .name(f)
            .map_or_else(|| f.to_string(), str::to_string)
    }

    fn member_label(&self, c: &CoordSpec, m: u32) -> String {
        match c {
            CoordSpec::Field(f) => self
                .cam
                .resolve(*f, m, self.kv)
                .map_or_else(|| m.to_string(), str::to_string),
            CoordSpec::Bucket { origin, width, .. } => {
                let lo = origin + width * i64::from(m);
                format!("[{lo},{})", lo + width)
            }
        }
    }

    fn labels(&self, r: &ReportResult, role: AxisRole, key: &[u32]) -> Vec<String> {
        let v = r.view();
        let dims = match role {
            AxisRole::Row => &v.rows,
            AxisRole::Column => &v.columns,
            AxisRole::Page => &v.pages,
        };
        dims.iter()
            .zip(key)
            .map(|(&d, &m)| self.member_label(&r.space().dims()[d].coord, m))
            .collect()
    }

    /// Resolve the presented grid. Labels are resolved once per presented
    /// member (memoized per dimension), values come from the shared cells.
    pub fn grid(&self, r: &ReportResult) -> Grid {
        let ms = r.measures();
        let cols = r.column_keys();
        let rows = r.row_keys();
        let pages = r.page_keys();
        let row_labels: Vec<Vec<String>> = rows
            .iter()
            .map(|k| self.labels(r, AxisRole::Row, k))
            .collect();
        Grid {
            column_keys: cols.clone(),
            columns: cols
                .iter()
                .map(|k| self.labels(r, AxisRole::Column, k))
                .collect(),
            measures: ms
                .iter()
                .map(|m| {
                    format!(
                        "{}({})",
                        m.op_name(),
                        m.field
                            .map_or_else(|| "*".to_string(), |f| self.field_name(f))
                    )
                })
                .collect(),
            pages: pages
                .iter()
                .map(|p| {
                    let body = rows
                        .iter()
                        .zip(&row_labels)
                        .map(|(rk, labels)| GridRow {
                            key: rk.clone(),
                            labels: labels.clone(),
                            cells: cols
                                .iter()
                                .map(|ck| ms.iter().map(|m| r.value(m, p, rk, ck)).collect())
                                .collect(),
                            total: ms.iter().map(|m| r.value(m, p, rk, &[])).collect(),
                        })
                        .collect();
                    (self.labels(r, AxisRole::Page, p), body)
                })
                .collect(),
            grand_total: ms.iter().map(|m| r.grand_total(m)).collect(),
        }
    }

    /// JSON: `{"columns":[[..]],"measures":[..],"pages":[{"page":[..],"rows":[{"row":[..],"cells":[[..]],"total":[..]}]}],"grand_total":[..]}`.
    pub fn json(&self, r: &ReportResult) -> (String, RenderStats) {
        let g = self.grid(r);
        let strs = |v: &[String]| {
            format!(
                "[{}]",
                v.iter().map(|s| json_str(s)).collect::<Vec<_>>().join(",")
            )
        };
        let vals = |v: &[CellValue]| {
            format!(
                "[{}]",
                v.iter()
                    .map(|x| json_value(*x))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        };
        let mut cells = 0u64;
        let pages = g
            .pages
            .iter()
            .map(|(p, rows)| {
                let rows = rows
                    .iter()
                    .map(|row| {
                        cells += (row.cells.len() * g.measures.len()) as u64;
                        format!(
                            "{{\"row\":{},\"cells\":[{}],\"total\":{}}}",
                            strs(&row.labels),
                            row.cells
                                .iter()
                                .map(|c| vals(c))
                                .collect::<Vec<_>>()
                                .join(","),
                            vals(&row.total)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{{\"page\":{},\"rows\":[{rows}]}}", strs(p))
            })
            .collect::<Vec<_>>()
            .join(",");
        let out = format!(
            "{{\"columns\":[{}],\"measures\":{},\"pages\":[{pages}],\"grand_total\":{}}}",
            g.columns
                .iter()
                .map(|c| strs(c))
                .collect::<Vec<_>>()
                .join(","),
            strs(&g.measures),
            vals(&g.grand_total)
        );
        let bytes = out.len() as u64;
        (out, RenderStats { cells, bytes })
    }

    /// CSV, long form per page: `page…,row…,column…,measure,value`.
    pub fn csv(&self, r: &ReportResult) -> (String, RenderStats) {
        let g = self.grid(r);
        let mut out = String::from("page,row,column,measure,value\n");
        let mut cells = 0u64;
        for (p, rows) in &g.pages {
            for row in rows {
                for (ci, col) in g.columns.iter().enumerate() {
                    for (mi, m) in g.measures.iter().enumerate() {
                        cells += 1;
                        out.push_str(&format!(
                            "{},{},{},{},{}\n",
                            csv_field(&p.join("/")),
                            csv_field(&row.labels.join("/")),
                            csv_field(&col.join("/")),
                            csv_field(m),
                            fmt_value(row.cells[ci][mi])
                        ));
                    }
                }
            }
        }
        let bytes = out.len() as u64;
        (out, RenderStats { cells, bytes })
    }

    /// Detail export: the rows the plan's selection keeps, with `fields`
    /// hydrated — `U64` references from KV, ordinals through CAM, integers
    /// as-is. The selection runs as ids; rows are materialized HERE by the
    /// one named materializer; hydration is proportional to the rows
    /// exported, never to the population.
    pub fn detail_csv(
        &self,
        plan: &ReportPlan,
        batch: &AbiBatch,
        fields: &[FieldId],
    ) -> Result<(String, RenderStats), ReportError> {
        let n = batch.n_rows();
        let filter = plan.selection.lower(batch)?;
        let prog = lower(&Query {
            filter,
            agg: Agg::Rows,
        })?;
        let (masks, lanes) = batch.views();
        let planes = Planes {
            n_rows: n,
            masks: &masks,
            lanes: &lanes,
        };
        let mut kept = vec![0u64; words_for(n)];
        let mut scratch = Scratch::for_program(&prog, n)?;
        execute_into(
            &prog,
            &planes,
            &Foreign::NONE,
            &mut scratch,
            Out::Mask(&mut kept),
        )?;
        let rows = materialize_rows(&kept, n);
        let mut out = fields
            .iter()
            .map(|f| csv_field(&self.field_name(*f)))
            .collect::<Vec<_>>()
            .join(",");
        out.push('\n');
        for &row in &rows {
            let line = fields
                .iter()
                .map(|f| {
                    let text = match batch.column(*f).map(|(_, c)| &c.lane) {
                        Some(LaneData::U64(v)) => self
                            .kv
                            .text(ContentId(v[row]))
                            .unwrap_or_default()
                            .to_string(),
                        Some(LaneData::U32(v)) => self
                            .cam
                            .resolve(*f, v[row], self.kv)
                            .map_or_else(|| v[row].to_string(), str::to_string),
                        Some(LaneData::I32(v)) => v[row].to_string(),
                        None => String::new(),
                    };
                    csv_field(&text)
                })
                .collect::<Vec<_>>()
                .join(",");
            out.push_str(&line);
            out.push('\n');
        }
        let bytes = out.len() as u64;
        Ok((
            out,
            RenderStats {
                cells: (rows.len() * fields.len()) as u64,
                bytes,
            },
        ))
    }
}
