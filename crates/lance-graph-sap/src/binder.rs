//! The CATS batch behind Quack's generic [`Binder`] (D-XGP-1).
//!
//! [`crate::query::CatsQuery`] addresses lanes through hard-coded `Col`
//! constants. This adapter resolves the same lanes BY NAME, so any frontend
//! that binds through `lance_graph_quack::bind` (a canonical-name front over
//! the ontology registry, a report catalog, a SQL surface) reaches CATS
//! without knowing its ordinals. It owns no state and copies no rows:
//!
//! - `field`: [`CatsSchema::resolve`](crate::schema::CatsSchema::resolve)
//!   (technical or C# name) plus the carrier → [`FieldKind`] mapping, and the
//!   derived `work_day` and `billable` lenses the bind already produced;
//! - `code`: the field's own domain: NUMC for PERNR, `X`/`true` for
//!   ABAP_BOOL, the batch dictionary otherwise. Nothing is minted;
//! - `live`: plane [`LIVE`], supplied by the executor as [`live_words`].
//!
//! Refused, deliberately, as unknown fields: optional fields (CATS stores NULL
//! as a sentinel, not a validity plane, so `<>` would keep NULL rows) and U64
//! instants (no `FieldKind`; `work_day` is their bindable lens).
use crate::bind::{numc, CatsBatch, BILLABLE, WORK_DAY};
use crate::schema::FIELDS;
use lance_graph_mask_risc::LaneKind;
use lance_graph_quack::bind::{Binder, BoundField, FieldKind, TableId};
use lance_graph_quack::{Col, Mask};

/// The one table this binder serves.
pub const TABLE: &str = "cats";
/// The live plane every bound query passes first.
pub const LIVE: Mask = Mask(0);
/// Name of the derived calendar-day lens (`YYYYMMDD`, `I32`).
pub const WORK_DAY_FIELD: &str = "work_day";
/// Name of the derived billable lens (`Code`: literals `true` / `false`).
pub const BILLABLE_FIELD: &str = "billable";

/// The live plane for `n` rows: every row live, tail bits clear. Its words are
/// `ceil(n / 64)`, the same population-mask size any selection already needs.
#[must_use]
pub fn live_words(n: usize) -> Vec<u64> {
    let mut words = vec![u64::MAX; n.div_ceil(64)];
    if let Some(last) = words.last_mut() {
        if !n.is_multiple_of(64) {
            *last = (1u64 << (n % 64)) - 1;
        }
    }
    words
}

/// Name and literal resolution over one bound batch.
pub struct CatsBinder<'a> {
    batch: &'a CatsBatch,
}

impl<'a> CatsBinder<'a> {
    #[must_use]
    pub fn new(batch: &'a CatsBatch) -> Self {
        Self { batch }
    }
}

impl Binder for CatsBinder<'_> {
    fn table(&self, name: &str) -> Option<TableId> {
        (name == TABLE).then_some(TableId(0))
    }
    fn live(&self, _: TableId) -> Mask {
        LIVE
    }
    fn field(&self, _: TableId, name: &str) -> Option<BoundField> {
        if name == WORK_DAY_FIELD {
            return Some(BoundField {
                col: Col(WORK_DAY as u16),
                kind: FieldKind::I32,
                validity: None,
            });
        }
        if name == BILLABLE_FIELD {
            return Some(BoundField {
                col: Col(BILLABLE as u16),
                kind: FieldKind::Code,
                validity: None,
            });
        }
        let col = self.batch.schema.resolve(name)?;
        let field = &FIELDS[usize::from(col.0)];
        if field.optional {
            return None;
        }
        let kind = match field.carrier {
            LaneKind::I32 => FieldKind::I32,
            LaneKind::U32 => FieldKind::Code,
            LaneKind::U64 | LaneKind::Strided => return None,
        };
        Some(BoundField {
            col,
            kind,
            validity: None,
        })
    }
    fn code(&self, _: TableId, col: Col, literal: &str) -> Option<u32> {
        let ordinal = usize::from(col.0);
        if ordinal == BILLABLE {
            return match literal {
                "true" => Some(1),
                "false" => Some(0),
                _ => None,
            };
        }
        match FIELDS.get(ordinal)?.native_type {
            "pernr_d" => numc(literal).ok(),
            "abap_bool" => match literal {
                "X" | "true" => Some(1),
                " " | "" | "false" => Some(0),
                _ => None,
            },
            _ => self.batch.dictionary_code(ordinal, literal),
        }
    }
}
