//! Terminal sinks only. Source hash profiles deliberately do not agree.
use crate::{
    bind::{format_decimal, BindError, CatsBatch},
    schema::{FIELDS, FIELD_COUNT},
};
use hmac::{Hmac, Mac};
use sha2::Sha512;

/// Pinned ABAP prepare_hash_data order, also used (without names) by SMB.
pub const HASH_ORDINALS: [usize; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 20, 21, 22];

/// The BAPI assignment order as a coordinate map over the canonical field
/// identity: position `k` of the BAPI parameter list reads canonical
/// ordinal `BAPI_ORDINALS[k]`. Same fields, same cardinality, different
/// coordinates — a permutation of a selection of [`FIELDS`], not a second
/// field vocabulary. [`bapi_sink`] reads through this map; the wire struct
/// it fills is the boundary's shape, never a copied normal form.
pub const BAPI_ORDINALS: [usize; 8] = [5, 9, 10, 11, 7, 8, 6, 20];

/// The BAPI parameter names, positionally aligned with [`BAPI_ORDINALS`].
pub const BAPI_PARAMETERS: [&str; 8] = [
    "EMPLOYEENUMBER",
    "WORKDATE",
    "HOURS",
    "ACTIVITYTYPE",
    "WBS_ELEMENT",
    "ORDERID",
    "CUST_SPEC_PR",
    "SHORTTEXT",
];

#[derive(Debug, Clone, Copy)]
pub enum HashProfile {
    /// Named, case-preserving fields. Decimal formatting depends on the SAP
    /// user's notation: the caller must provide the observed separator.
    SimafPortAbap { decimal_separator: char },
    /// Bare, trimmed, uppercase fields and an invariant two-place decimal.
    SmbMiddleware,
}

/// The source code only proves ASCII normalization and exact cents here.
/// Refuse higher-precision rounding and Unicode case conversion rather than
/// claim cross-runtime equivalence without running those runtimes.
pub fn ordered_hash_projection(
    batch: &CatsBatch,
    row: usize,
    profile: HashProfile,
) -> Result<String, BindError> {
    let mut parts = Vec::with_capacity(HASH_ORDINALS.len());
    for ordinal in HASH_ORDINALS {
        let mut text = batch.edge_value(ordinal, row)?.unwrap_or_default();
        if !text.is_ascii() {
            return Err(BindError("hash oracle supports ASCII only".into()));
        }
        if ordinal == 10 {
            text = exact_cents(&text)?;
        }
        match profile {
            HashProfile::SmbMiddleware => {
                if ordinal == 3 {
                    let tenant: i32 = text
                        .parse()
                        .map_err(|_| BindError("SMB TenantId requires Int32".into()))?;
                    if tenant.to_string() != text {
                        return Err(BindError(
                            "SMB TenantId would lose source formatting".into(),
                        ));
                    }
                }
                text = text.trim().to_ascii_uppercase();
            }
            HashProfile::SimafPortAbap { decimal_separator } => {
                if !['.', ','].contains(&decimal_separator) {
                    return Err(BindError("unsupported ABAP decimal notation".into()));
                }
                if text != text.trim() {
                    return Err(BindError(
                        "ABAP whitespace formatting requires a runtime oracle".into(),
                    ));
                }
                if ordinal == 10 {
                    text = text.replace('.', &decimal_separator.to_string());
                }
                text = format!("{}={text}", FIELDS[ordinal].csharp_name);
            }
        }
        parts.push(text);
    }
    Ok(parts.join("|"))
}

fn exact_cents(text: &str) -> Result<String, BindError> {
    let (whole, fraction) = text.split_once('.').unwrap_or((text, ""));
    let fraction = fraction.trim_end_matches('0');
    if fraction.len() > 2 {
        return Err(BindError(
            "hash rounding needs an explicit runtime oracle".into(),
        ));
    }
    Ok(format!("{whole}.{fraction:0<2}"))
}

/// `key` is actual key bytes; ABAP UTF-8 and SMB base64 constructor decoding
/// are separate caller-side key adapters, never query inputs.
pub fn hash_projection(projection: &str, key: &[u8]) -> String {
    let mut mac = Hmac::<Sha512>::new_from_slice(key).expect("HMAC accepts any key length");
    mac.update(projection.as_bytes());
    mac.finalize()
        .into_bytes()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

/// A requested DTO/JSON boundary can consume canonical values in C# field order.
/// This is the only whole-entry materialization; query execution never calls it.
pub fn csharp_fields(
    batch: &CatsBatch,
    row: usize,
) -> Result<[Option<String>; FIELD_COUNT], BindError> {
    let mut result = std::array::from_fn(|_| None);
    for (ordinal, value) in result.iter_mut().enumerate() {
        *value = batch.edge_value(ordinal, row)?;
    }
    Ok(result)
}

#[derive(Debug, PartialEq, Eq)]
pub struct ActivityTotal {
    pub activity_type: String,
    /// Exact decimal text for a C# decimal, JSON decimal or UI sink.
    pub hours: String,
}
pub fn activity_totals(batch: &CatsBatch, sums: &[i64]) -> Result<Vec<ActivityTotal>, BindError> {
    if sums.len() != batch.activity_groups() as usize {
        return Err(BindError("wrong group result length".into()));
    }
    let mut result = Vec::new();
    for (code, &sum) in sums.iter().enumerate() {
        if sum == 0 {
            continue;
        } // Hours are strictly positive: zero means no surviving entry.
        let label = batch
            .activity_label(code as u32)
            .ok_or_else(|| BindError("invalid activity code".into()))?;
        result.push(ActivityTotal {
            activity_type: label.into(),
            hours: format_decimal(sum, batch.scale()),
        });
    }
    Ok(result)
}

/// Exactly the eight assignments in the pinned ABAP processor's BAPI mapping.
/// No connectivity or SAP master-data validation is implied.
#[derive(Debug, PartialEq, Eq)]
pub struct BapiCatsInsert {
    pub employeenumber: String,
    pub workdate: String,
    pub hours: String,
    pub activitytype: String,
    pub wbs_element: String,
    pub orderid: String,
    pub cust_spec_pr: String,
    pub shorttext: String,
}
pub const BAPI_FUNCTION: &str = "BAPI_CATIMESHEETMGR_INSERT";

/// Explicit row sink. The substrate's sole materializer is used only here.
/// Aggregated hours MUST NOT be posted under an invented project/date.
pub fn bapi_sink(batch: &CatsBatch, kept: &[u64]) -> Result<Vec<BapiCatsInsert>, BindError> {
    if kept.len() != batch.alpha().len() || kept.iter().zip(batch.alpha()).any(|(a, b)| a & !b != 0)
    {
        return Err(BindError("invalid terminal selection mask".into()));
    }
    let mut result = Vec::new();
    for row in lance_graph_mask_risc::materialize_rows(kept, batch.len()) {
        let field =
            |i| -> Result<String, BindError> { Ok(batch.edge_value(i, row)?.unwrap_or_default()) };
        let [emp, date_o, hours, act, wbs, order, cust, notes_o] = BAPI_ORDINALS;
        let notes = field(notes_o)?;
        // ABAP source uses notes(50), not a documented safe truncation helper.
        // Restrict the fixture to at least 50 ASCII characters; do not silently
        // repair or claim parity for short strings / UTF-16 substring behavior.
        if !notes.is_ascii() || notes.len() < 50 {
            return Err(BindError(
                "BAPI notes(50) requires a 50-byte ASCII oracle fixture".into(),
            ));
        }
        let date = field(date_o)?;
        result.push(BapiCatsInsert {
            employeenumber: field(emp)?,
            workdate: date[..10].replace('-', ""),
            hours: field(hours)?,
            activitytype: field(act)?,
            wbs_element: field(wbs)?,
            orderid: field(order)?,
            cust_spec_pr: field(cust)?,
            shorttext: notes[..50].into(),
        });
    }
    Ok(result)
}
