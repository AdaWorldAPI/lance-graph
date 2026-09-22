//! Cold, fallible ingestion. The output owns numeric columns; execution borrows
//! them as the existing LaneRef. Dictionaries are lossless edge metadata.
use crate::schema::{CatsSchema, FIELDS, FIELD_COUNT};
use lance_graph_mask_risc::{words_for, LaneRef};

pub const EMPLOYEE: usize = 5;
pub const WORK_DATE: usize = 9;
pub const HOURS: usize = 10;
pub const ACTIVITY: usize = 11;
/// Derived date lens of the full UTC timestamp, bound once (YYYYMMDD).
pub const WORK_DAY: usize = FIELD_COUNT;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindError(pub String);
impl std::fmt::Display for BindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}
impl std::error::Error for BindError {}
fn error(s: &str) -> BindError {
    BindError(s.into())
}

/// Owning storage only, not another ABI type system. Borrowed values use LaneRef.
#[derive(Debug)]
enum Column {
    U32(Vec<u32>),
    I32(Vec<i32>),
    U64(Vec<u64>),
}
impl Column {
    fn borrow(&self) -> LaneRef<'_> {
        match self {
            Self::U32(v) => LaneRef::U32(v),
            Self::I32(v) => LaneRef::I32(v),
            Self::U64(v) => LaneRef::U64(v),
        }
    }
}

#[derive(Debug)]
pub struct CatsBatch {
    pub schema: CatsSchema,
    columns: Vec<Column>,
    dictionaries: [Vec<String>; FIELD_COUNT],
    alpha: Vec<u64>,
    len: usize,
    scale: u32,
}

impl CatsBatch {
    /// Column-oriented edge input in the descriptor's stable order. No DTOs are
    /// constructed. Null stays distinct from empty text and NUMC zero. ABAP
    /// callers spell bool as X/space, C# callers as true/false.
    pub fn bind(
        schema: CatsSchema,
        input: [&[Option<&str>]; FIELD_COUNT],
    ) -> Result<Self, BindError> {
        let len = input[0].len();
        if input.iter().any(|c| c.len() != len) {
            return Err(error("ragged input"));
        }
        let mut dictionaries: [Vec<String>; FIELD_COUNT] = std::array::from_fn(|_| Vec::new());
        let mut columns = Vec::with_capacity(FIELD_COUNT + 1);
        let mut scale = 0;
        for f in FIELDS {
            let index = usize::from(f.ordinal.0);
            let values = input[index];
            for value in values {
                if value.is_none() && !f.optional {
                    return Err(error(f.technical_name));
                }
                if !f.optional
                    && f.native_type != "abap_bool"
                    && value.is_some_and(|v| v.trim().is_empty())
                {
                    return Err(error("required field is blank"));
                }
                if let (Some(v), Some(width)) = (value, f.width) {
                    if v.encode_utf16().count() > width {
                        return Err(error("field width exceeded"));
                    }
                }
            }
            let column = if index == HOURS {
                let decimals: Vec<_> = values
                    .iter()
                    .map(|v| decimal(v.unwrap()))
                    .collect::<Result<_, _>>()?;
                scale = decimals.iter().map(|(_, s)| *s).max().unwrap_or(0);
                let mut lane = Vec::with_capacity(len);
                for (coefficient, s) in decimals {
                    let scaled = coefficient
                        .checked_mul(10i128.pow(scale - s))
                        .ok_or_else(|| error("decimal overflow"))?;
                    lane.push(
                        i32::try_from(scaled)
                            .map_err(|_| error("exact batch decimal scale exceeds I32 carrier"))?,
                    );
                }
                Column::I32(lane)
            } else if f.native_type == "pernr_d" {
                Column::U32(
                    values
                        .iter()
                        .map(|v| v.map(numc).unwrap_or(Ok(u32::MAX)))
                        .collect::<Result<_, _>>()?,
                )
            } else if f.native_type == "abap_bool" {
                Column::U32(
                    values
                        .iter()
                        .map(|v| match v.unwrap() {
                            "X" | "true" => Ok(1),
                            " " | "" | "false" => Ok(0),
                            _ => Err(error("invalid ABAP_BOOL")),
                        })
                        .collect::<Result<_, _>>()?,
                )
            } else if matches!(f.carrier, lance_graph_mask_risc::LaneKind::U64) {
                // Zero is a null sentinel; valid years start at 0001.
                Column::U64(
                    values
                        .iter()
                        .map(|v| v.map(utc).unwrap_or(Ok(0)))
                        .collect::<Result<_, _>>()?,
                )
            } else {
                let dict = &mut dictionaries[index];
                let mut codes = Vec::with_capacity(len);
                for value in values {
                    codes.push(if let Some(value) = value {
                        let position = if let Some(i) = dict.iter().position(|s| s == value) {
                            i
                        } else {
                            dict.push((*value).into());
                            dict.len() - 1
                        };
                        u32::try_from(position + 1).map_err(|_| error("dictionary too large"))?
                    } else {
                        0
                    });
                }
                Column::U32(codes)
            };
            columns.push(column);
        }
        let Column::U64(dates) = &columns[WORK_DATE] else {
            unreachable!()
        };
        columns.push(Column::I32(
            dates.iter().map(|v| (v / 1_000_000) as i32).collect(),
        ));
        let mut alpha = vec![u64::MAX; words_for(len)];
        if len % 64 != 0 {
            *alpha.last_mut().unwrap() = (1u64 << (len % 64)) - 1;
        }
        Ok(Self {
            schema,
            columns,
            dictionaries,
            alpha,
            len,
            scale,
        })
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn scale(&self) -> u32 {
        self.scale
    }
    pub fn alpha(&self) -> &[u64] {
        &self.alpha
    }
    pub fn lanes(&self) -> [LaneRef<'_>; FIELD_COUNT + 1] {
        std::array::from_fn(|i| self.columns[i].borrow())
    }
    /// Group cardinality includes reserved NULL code 0. Required ActivityType
    /// never uses it; an empty group remains an explicit zero sum.
    pub fn activity_groups(&self) -> u32 {
        (self.dictionaries[ACTIVITY].len() + 1) as u32
    }
    pub fn activity_label(&self, code: u32) -> Option<&str> {
        code.checked_sub(1)
            .and_then(|i| self.dictionaries[ACTIVITY].get(i as usize))
            .map(String::as_str)
    }
    /// Explicit terminal adapter only; never invoked by a fold.
    pub fn edge_value(&self, ordinal: usize, row: usize) -> Result<Option<String>, BindError> {
        let field = FIELDS
            .get(ordinal)
            .ok_or_else(|| error("invalid field ordinal"))?;
        if row >= self.len {
            return Err(error("invalid sink row"));
        }
        Ok(match &self.columns[ordinal] {
            Column::I32(v) => Some(format_decimal(i64::from(v[row]), self.scale)),
            Column::U64(v) => {
                if v[row] == 0 {
                    None
                } else {
                    Some(format_utc(v[row]))
                }
            }
            Column::U32(v) if field.native_type == "pernr_d" => {
                if v[row] == u32::MAX {
                    None
                } else {
                    Some(format!("{:08}", v[row]))
                }
            }
            Column::U32(v) if field.native_type == "abap_bool" => {
                Some(if v[row] == 1 { "true" } else { "false" }.into())
            }
            Column::U32(v) => v[row]
                .checked_sub(1)
                .map(|i| self.dictionaries[ordinal][i as usize].clone()),
        })
    }
}

pub fn numc(s: &str) -> Result<u32, BindError> {
    if s.len() != 8 || !s.bytes().all(|b| b.is_ascii_digit()) {
        return Err(error("PERNR requires eight ASCII digits"));
    }
    s.parse().map_err(|_| error("invalid PERNR"))
}

/// Exact positive C# decimal subset admitted by its [Range(0.01, 24.00)].
/// No native CATS scale is asserted. Up to 28 fractional digits are parsed;
/// binding rejects a batch whose exact common scale does not fit its ABI lane.
fn decimal(s: &str) -> Result<(i128, u32), BindError> {
    let (whole, fraction) = s.split_once('.').unwrap_or((s, ""));
    if whole.is_empty()
        || !whole.bytes().all(|b| b.is_ascii_digit())
        || !fraction.bytes().all(|b| b.is_ascii_digit())
        || fraction.len() > 28
    {
        return Err(error("invalid decimal"));
    }
    let fraction = fraction.trim_end_matches('0');
    let scale = fraction.len() as u32;
    let coefficient: i128 = format!("{whole}{fraction}")
        .parse()
        .map_err(|_| error("decimal overflow"))?;
    let power = 10i128.pow(scale);
    if coefficient > 24 * power || coefficient.checked_mul(100).is_none_or(|v| v < power) {
        return Err(error("hours outside C# range"));
    }
    Ok((coefficient, scale))
}

pub fn format_decimal(value: i64, scale: u32) -> String {
    if scale == 0 {
        return value.to_string();
    }
    let power = 10i64.pow(scale);
    format!(
        "{}{}.{:0width$}",
        if value < 0 { "-" } else { "" },
        value.unsigned_abs() / power as u64,
        value.unsigned_abs() % power as u64,
        width = scale as usize
    )
}

/// Lossless sortable YYYYMMDDhhmmss carrier for the corpus's strict UTC text.
/// Calendar validity is checked; time-of-day is never discarded for hashing.
pub fn utc(s: &str) -> Result<u64, BindError> {
    let b = s.as_bytes();
    if b.len() != 20
        || b[4] != b'-'
        || b[7] != b'-'
        || b[10] != b'T'
        || b[13] != b':'
        || b[16] != b':'
        || b[19] != b'Z'
    {
        return Err(error("strict UTC timestamp required"));
    }
    let mut digits = String::with_capacity(14);
    for (i, &c) in b.iter().enumerate() {
        if [4, 7, 10, 13, 16, 19].contains(&i) {
            continue;
        }
        if !c.is_ascii_digit() {
            return Err(error("invalid UTC digits"));
        }
        digits.push(c as char);
    }
    let n: u64 = digits.parse().map_err(|_| error("invalid UTC"))?;
    let year = n / 10_000_000_000;
    let month = (n / 100_000_000 % 100) as usize;
    let day = n / 1_000_000 % 100;
    let leap = year % 4 == 0 && (year % 100 != 0 || year % 400 == 0);
    let days = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    if year == 0
        || !(1..=12).contains(&month)
        || day == 0
        || day > days[month - 1]
        || n / 10_000 % 100 > 23
        || n / 100 % 100 > 59
        || n % 100 > 59
    {
        return Err(error("invalid calendar timestamp"));
    }
    Ok(n)
}
fn format_utc(n: u64) -> String {
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z",
        n / 10_000_000_000,
        n / 100_000_000 % 100,
        n / 1_000_000 % 100,
        n / 10_000 % 100,
        n / 100 % 100,
        n % 100
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lossless_adapters_reject_lossy_inputs() {
        assert_eq!(numc("00000042"), Ok(42));
        assert!(numc("42").is_err());
        assert!(numc("１２３４５６７８").is_err());
        assert_eq!(decimal("8.500"), Ok((85, 1)));
        assert_eq!(
            decimal("0.01234567890123456789"),
            Ok((1234567890123456789, 20))
        );
        for bad in ["0", "24.01", "NaN", "-1", "1.2.3"] {
            assert!(decimal(bad).is_err());
        }
        for good in [
            "2024-02-29T23:59:59Z",
            "0001-01-01T00:00:00Z",
            "9999-12-31T23:59:59Z",
        ] {
            assert_eq!(format_utc(utc(good).unwrap()), good);
        }
        for bad in [
            "2025-02-29T00:00:00Z",
            "2024-01-01T24:00:00Z",
            "2024-00-01T00:00:00Z",
            "2024-01-01T00:00:00+01:00",
        ] {
            assert!(utc(bad).is_err());
        }
    }
}
