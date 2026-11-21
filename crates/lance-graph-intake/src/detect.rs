//! Data as config: the shape of a column, read off its cells.
//!
//! A consumer hands in cells the way PostgreSQL prints them. This module
//! decides, from the non-NULL cells alone, the PostgreSQL type, the lane
//! encoding and the absent spelling of each column, and which column is the
//! key. The rules are checked in a fixed order and the first that holds for
//! every cell wins: bool, integer, timestamp, date, text.
//!
//! A type is only chosen when the cell text is exactly what PostgreSQL would
//! print for the parsed value (an integer is canonical, a timestamp has no
//! fraction and no zone), so the read side's spelling reproduces the input.
//! A lane holds 32 bits: an integer outside `u32`, or a date or timestamp
//! whose epoch does not fit `u32`, is text.
//!
//! [`parse_epoch_seconds`] and [`parse_epoch_days`] are the inverse of
//! `lance_graph_contract::pg_row::write_cell` for those encodings.

use std::collections::BTreeSet;

use lance_graph_contract::pg_row::{AbsentAs, LaneEncoding, PgColumn, PgType};

/// What the data says about one column.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Detected {
    pub col: PgColumn,
    /// Non-NULL cells that fit the chosen type.
    pub support: usize,
    /// Non-NULL cells.
    pub total: usize,
    /// The first row that broke a narrower type, when one did.
    pub witness: Option<usize>,
}

const TIMESTAMP_SHAPE: &[u8] = b"dddd-dd-dd dd:dd:dd";
const DATE_SHAPE: &[u8] = b"dddd-dd-dd";

const BOOL: PgColumn = PgColumn {
    ty: PgType::Bool,
    encoding: LaneEncoding::Bool,
    absent: AbsentAs::Null,
};
const INT4: PgColumn = PgColumn {
    ty: PgType::Int4,
    encoding: LaneEncoding::Signed,
    absent: AbsentAs::Null,
};
const INT8_UNSIGNED: PgColumn = PgColumn {
    ty: PgType::Int8,
    encoding: LaneEncoding::Unsigned,
    absent: AbsentAs::Null,
};
const TIMESTAMP: PgColumn = PgColumn {
    ty: PgType::Timestamp(None),
    encoding: LaneEncoding::EpochSeconds,
    absent: AbsentAs::Null,
};
const DATE: PgColumn = PgColumn {
    ty: PgType::Date,
    encoding: LaneEncoding::EpochDays,
    absent: AbsentAs::Null,
};

fn text_column(absent: AbsentAs) -> PgColumn {
    PgColumn {
        ty: PgType::Text,
        encoding: LaneEncoding::TextCode,
        absent,
    }
}

/// An integer, only when its text is what PostgreSQL prints for it.
pub(crate) fn parse_int(s: &str) -> Option<i64> {
    let v: i64 = s.parse().ok()?;
    (v.to_string() == s).then_some(v)
}

fn shape_is(s: &str, shape: &[u8]) -> bool {
    s.len() == shape.len()
        && s.bytes().zip(shape).all(|(b, &p)| {
            if p == b'd' {
                b.is_ascii_digit()
            } else {
                b == p
            }
        })
}

fn number(s: &str) -> Option<i64> {
    s.parse().ok()
}

fn is_leap(y: i64) -> bool {
    y % 4 == 0 && (y % 100 != 0 || y % 400 == 0)
}

fn days_in_month(y: i64, m: i64) -> i64 {
    match m {
        2 if is_leap(y) => 29,
        2 => 28,
        4 | 6 | 9 | 11 => 30,
        _ => 31,
    }
}

/// Days since 1970-01-01 for a proleptic Gregorian date, by Howard Hinnant's
/// `days_from_civil`.
fn days_from_civil(y: i64, m: i64, d: i64) -> i64 {
    let y = y - i64::from(m <= 2);
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let mp = if m > 2 { m - 3 } else { m + 9 };
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

/// `YYYY-MM-DD` as days since 1970-01-01, when it is a real date that fits
/// `u32`.
pub fn parse_epoch_days(s: &str) -> Option<u32> {
    if !shape_is(s, DATE_SHAPE) {
        return None;
    }
    let (y, m, d) = (number(&s[0..4])?, number(&s[5..7])?, number(&s[8..10])?);
    if !(1..=12).contains(&m) || d < 1 || d > days_in_month(y, m) {
        return None;
    }
    u32::try_from(days_from_civil(y, m, d)).ok()
}

/// `YYYY-MM-DD HH:MM:SS` as seconds since 1970-01-01 00:00:00, when it is a
/// real time that fits `u32`.
pub fn parse_epoch_seconds(s: &str) -> Option<u32> {
    if !shape_is(s, TIMESTAMP_SHAPE) {
        return None;
    }
    let days = i64::from(parse_epoch_days(&s[0..10])?);
    let (h, mi, se) = (
        number(&s[11..13])?,
        number(&s[14..16])?,
        number(&s[17..19])?,
    );
    if h > 23 || mi > 59 || se > 59 {
        return None;
    }
    u32::try_from(days * 86_400 + h * 3600 + mi * 60 + se).ok()
}

/// The absent spelling of a detected text column is always NULL: an observed
/// `''` is a present value, never evidence that `''` means absent. Reading
/// `''` as absent is a source policy, declared through `ColumnConfig::absent`
/// (`AbsentAs::EmptyText`), never inferred from the data.
fn text_absent() -> AbsentAs {
    AbsentAs::Null
}

/// The shape of one column from its cells, `None` being NULL.
pub fn detect_column(cells: &[Option<&str>]) -> Detected {
    let present: Vec<(usize, &str)> = cells
        .iter()
        .enumerate()
        .filter_map(|(row, cell)| cell.map(|s| (row, s)))
        .collect();
    let total = present.len();
    let done = |col: PgColumn, witness: Option<usize>| Detected {
        col,
        support: total,
        total,
        witness,
    };
    let text = |witness: Option<usize>| done(text_column(text_absent()), witness);

    if total == 0 {
        return done(text_column(AbsentAs::Null), None);
    }
    if present.iter().all(|&(_, s)| s == "t" || s == "f") {
        return done(BOOL, None);
    }
    let ints: Option<Vec<i64>> = present.iter().map(|&(_, s)| parse_int(s)).collect();
    if let Some(ints) = ints {
        let first_outside =
            |fits: fn(i64) -> bool| ints.iter().position(|&v| !fits(v)).map(|at| present[at].0);
        let fits_i32: fn(i64) -> bool = |v| i32::try_from(v).is_ok();
        let fits_u32: fn(i64) -> bool = |v| u32::try_from(v).is_ok();
        return match (first_outside(fits_i32), first_outside(fits_u32)) {
            (None, _) => done(INT4, None),
            (Some(broke_i32), None) => done(INT8_UNSIGNED, Some(broke_i32)),
            (Some(_), Some(broke_u32)) => text(Some(broke_u32)),
        };
    }
    if present.iter().all(|&(_, s)| shape_is(s, TIMESTAMP_SHAPE)) {
        return match present
            .iter()
            .find(|&&(_, s)| parse_epoch_seconds(s).is_none())
        {
            None => done(TIMESTAMP, None),
            Some(&(row, _)) => text(Some(row)),
        };
    }
    if present.iter().all(|&(_, s)| shape_is(s, DATE_SHAPE)) {
        return match present
            .iter()
            .find(|&&(_, s)| parse_epoch_days(s).is_none())
        {
            None => done(DATE, None),
            Some(&(row, _)) => text(Some(row)),
        };
    }
    text(None)
}

fn is_integer(col: &PgColumn) -> bool {
    matches!(col.ty, PgType::Int2 | PgType::Int4 | PgType::Int8)
        && matches!(col.encoding, LaneEncoding::Signed | LaneEncoding::Unsigned)
}

/// Whether column `k` has a non-NULL cell on every row and no two are equal.
fn is_distinct(cells: &[Vec<Option<&str>>], k: usize) -> bool {
    let mut seen = BTreeSet::new();
    cells.iter().all(|row| match row.get(k).copied().flatten() {
        Some(s) => seen.insert(s),
        None => false,
    })
}

/// The key column: the first integer column named `id` whose cells are all
/// non-NULL and distinct; else the first such integer column; else `None`.
/// `cells` is row-major.
pub fn detect_key(names: &[&str], cols: &[Detected], cells: &[Vec<Option<&str>>]) -> Option<usize> {
    let usable = |k: usize| is_integer(&cols[k].col) && is_distinct(cells, k);
    let width = names.len().min(cols.len());
    (0..width)
        .find(|&k| names[k] == "id" && usable(k))
        .or_else(|| (0..width).find(|&k| usable(k)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use lance_graph_contract::pg_row::{write_cell, Written};

    fn spelled(col: &PgColumn, value: u32) -> String {
        let mut out = String::new();
        let written = write_cell(col, true, value, None, &mut out).expect("written");
        assert_eq!(written, Written::Text);
        out
    }

    /// FAILS IF: a date's text does not parse back to the day it was written
    /// from, or the parse is off by a day anywhere in 1970..=2105.
    #[test]
    fn epoch_days_round_trip_with_write_cell() {
        let last = parse_epoch_days("2105-12-31").expect("last day");
        assert_eq!(parse_epoch_days("1970-01-01"), Some(0));
        for day in 0..=last {
            let text = spelled(&DATE, day);
            assert_eq!(parse_epoch_days(&text), Some(day), "{text}");
            assert_eq!(spelled(&DATE, parse_epoch_days(&text).expect("day")), text);
        }
    }

    /// FAILS IF: a timestamp does not parse back to the seconds it was
    /// written from.
    #[test]
    fn epoch_seconds_round_trip_with_write_cell() {
        for secs in [
            0,
            1,
            86_399,
            86_400,
            951_782_400,
            1_709_208_000 + 3_723,
            4_102_444_800,
            u32::MAX,
        ] {
            let text = spelled(&TIMESTAMP, secs);
            assert_eq!(parse_epoch_seconds(&text), Some(secs), "{text}");
            assert_eq!(
                spelled(&TIMESTAMP, parse_epoch_seconds(&text).expect("secs")),
                text
            );
        }
    }

    /// FAILS IF: an impossible date or time, a fraction, a zone or a value
    /// past `u32` is accepted.
    #[test]
    fn epoch_parsers_refuse_what_is_not_a_lane_value() {
        for bad in [
            "2023-02-30",
            "2023-13-01",
            "2023-00-10",
            "1969-12-31",
            "2023-1-01",
            "2023-01-01 ",
        ] {
            assert_eq!(parse_epoch_days(bad), None, "{bad}");
        }
        for bad in [
            "2023-01-01 24:00:00",
            "2023-01-01 00:60:00",
            "2023-01-01 00:00:60",
            "2023-01-01 00:00:00.5",
            "2023-01-01 00:00:00Z",
            "2106-02-07 06:28:16",
            "1969-12-31 23:59:59",
        ] {
            assert_eq!(parse_epoch_seconds(bad), None, "{bad}");
        }
        assert_eq!(parse_epoch_seconds("2106-02-07 06:28:15"), Some(u32::MAX));
    }

    /// FAILS IF: a non-canonical integer spelling is read as an integer, so
    /// the read side would print different text than was handed in.
    #[test]
    fn integers_must_be_canonical() {
        for bad in ["+5", "007", "-0", " 5", "5 ", ""] {
            assert_eq!(parse_int(bad), None, "{bad:?}");
        }
        assert_eq!(parse_int("-5"), Some(-5));
        assert_eq!(parse_int("0"), Some(0));
    }
}
