//! The PostgreSQL row polyfill contract: how a lane cell is spelled as
//! PostgreSQL's text output (D-QSP-VROW, D-REGC-10).
//!
//! A PostgreSQL row is never stored. It is a reading over source-owned lanes
//! at one position (`identity[index].key : identity[index].value`, EXO-332).
//! The only materialization is the text of the returned cells, written after
//! the fold. This module holds what every consumer must agree on for that
//! last step:
//!
//! - [`PgType`]: the admitted column types, as PostgreSQL's `\gdesc` names
//!   them. A type whose text form the lane cannot reproduce (for example a
//!   timestamp *with* time zone, which prints an offset the lane does not
//!   hold) is refused, never approximated.
//! - [`LaneEncoding`]: how a `u32` lane holds the value.
//! - [`AbsentAs`]: how an absent cell is spelled. Absent is one code, −0
//!   (presence 0 and value 0, [`crate::lane_codec`] § Absent, EXO-294). The
//!   column decides whether PostgreSQL shows it as NULL or as `''`
//!   (`.claude/knowledge/register-contract-reconciliation.md` C3).
//! - [`write_cell`]: the one spelling function.
//!
//! The absent rule here is the lane socket's (`lance-graph-dt64`
//! `PresenceSource`): a cell is absent iff its presence bit is 0 **and** its
//! value is 0. A presence bit of 0 with a nonzero value is a present value.
//!
//! Nothing here reads a lane, allocates beyond the caller's buffer, or knows
//! a source. The read side lives in `lance-graph-pg-row`.

use core::fmt::Write as _;

/// A column type the polyfill admits, as PostgreSQL's `\gdesc` names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PgType {
    /// `boolean`.
    Bool,
    /// `smallint`.
    Int2,
    /// `integer`.
    Int4,
    /// `bigint`.
    Int8,
    /// `text`.
    Text,
    /// `character varying(n)`, or unbounded when `None`.
    Varchar(Option<u32>),
    /// `timestamp(p) without time zone`, or the default precision when
    /// `None`. The lane holds whole seconds, so fractional output never
    /// arises.
    Timestamp(Option<u8>),
    /// `date`.
    Date,
}

/// Why [`PgType::parse`] or [`PgColumn::admit`] refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PgTypeRefusal {
    /// The type is not admitted (numeric, floating point, time zones, arrays,
    /// json, ...). Its text is kept so a report can name it.
    NotAdmitted(String),
    /// The lane encoding cannot hold this type's values.
    EncodingMismatch { ty: PgType, encoding: LaneEncoding },
    /// `''` for an absent cell is only spelled for a text column.
    EmptyTextOnNonText(PgType),
}

impl PgType {
    /// The type for a PostgreSQL type name, in `\gdesc` or SQL spelling.
    ///
    /// Accepts `boolean`/`bool`, `smallint`/`int2`, `integer`/`int`/`int4`,
    /// `bigint`/`int8`, `text`, `character varying[(n)]`/`varchar[(n)]`,
    /// `timestamp[(p)] [without time zone]`, `date`. Every other name is
    /// refused.
    pub fn parse(sql_type: &str) -> Result<PgType, PgTypeRefusal> {
        let t = sql_type.trim().to_ascii_lowercase();
        let refuse = || PgTypeRefusal::NotAdmitted(sql_type.trim().to_owned());
        let ty = match t.as_str() {
            "boolean" | "bool" => PgType::Bool,
            "smallint" | "int2" => PgType::Int2,
            "integer" | "int" | "int4" => PgType::Int4,
            "bigint" | "int8" => PgType::Int8,
            "text" => PgType::Text,
            "date" => PgType::Date,
            _ => {
                if let Some(rest) = strip_any(&t, &["character varying", "varchar"]) {
                    PgType::Varchar(paren_u32(rest).map_err(|_| refuse())?)
                } else if let Some(rest) = t.strip_prefix("timestamp") {
                    let (p, tail) = split_paren(rest).map_err(|_| refuse())?;
                    let tail = tail.trim();
                    if !(tail.is_empty() || tail == "without time zone") {
                        return Err(refuse());
                    }
                    let p = match p {
                        None => None,
                        Some(p) => Some(u8::try_from(p).map_err(|_| refuse())?),
                    };
                    PgType::Timestamp(p)
                } else {
                    return Err(refuse());
                }
            }
        };
        Ok(ty)
    }

    /// The name PostgreSQL's `\gdesc` prints for this type.
    pub fn gdesc(&self) -> String {
        match self {
            PgType::Bool => "boolean".into(),
            PgType::Int2 => "smallint".into(),
            PgType::Int4 => "integer".into(),
            PgType::Int8 => "bigint".into(),
            PgType::Text => "text".into(),
            PgType::Varchar(None) => "character varying".into(),
            PgType::Varchar(Some(n)) => format!("character varying({n})"),
            PgType::Timestamp(None) => "timestamp without time zone".into(),
            PgType::Timestamp(Some(p)) => format!("timestamp({p}) without time zone"),
            PgType::Date => "date".into(),
        }
    }

    fn is_text(&self) -> bool {
        matches!(self, PgType::Text | PgType::Varchar(_))
    }
}

fn strip_any<'a>(t: &'a str, prefixes: &[&str]) -> Option<&'a str> {
    prefixes.iter().find_map(|p| t.strip_prefix(p))
}

/// `"(n) rest"` → `(Some(n), " rest")`; `" rest"` → `(None, " rest")`.
fn split_paren(s: &str) -> Result<(Option<u32>, &str), ()> {
    let s = s.trim_start();
    let Some(inner) = s.strip_prefix('(') else {
        return Ok((None, s));
    };
    let close = inner.find(')').ok_or(())?;
    let n = inner[..close].trim().parse().map_err(|_| ())?;
    Ok((Some(n), &inner[close + 1..]))
}

fn paren_u32(s: &str) -> Result<Option<u32>, ()> {
    let (n, tail) = split_paren(s)?;
    if tail.trim().is_empty() {
        Ok(n)
    } else {
        Err(())
    }
}

/// How a `u32` lane holds a column's value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LaneEncoding {
    /// 0 is false, any other value is true.
    Bool,
    /// The value itself, unsigned.
    Unsigned,
    /// The value's bits as a two's-complement `i32`.
    Signed,
    /// Seconds since 1970-01-01 00:00:00, unsigned.
    EpochSeconds,
    /// Days since 1970-01-01, unsigned.
    EpochDays,
    /// A code into the column's code table ([`CodeText`]).
    TextCode,
}

/// How an absent cell (−0) is spelled for a column.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AbsentAs {
    /// SQL NULL.
    Null,
    /// The empty string, for a text column whose source spells "no value" as
    /// `''` (a `NOT NULL DEFAULT ''` column, D-REGC-10).
    EmptyText,
}

/// One output column: its type, the lane's encoding, the absent spelling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PgColumn {
    pub ty: PgType,
    pub encoding: LaneEncoding,
    pub absent: AbsentAs,
}

impl PgColumn {
    /// Whether this encoding can hold this type and this absent spelling
    /// applies. Checked once per column, before any cell is read.
    pub fn admit(&self) -> Result<(), PgTypeRefusal> {
        use LaneEncoding as E;
        use PgType as T;
        let ok = matches!(
            (self.ty, self.encoding),
            (T::Bool, E::Bool)
                | (T::Int2 | T::Int4 | T::Int8, E::Unsigned)
                | (T::Int2 | T::Int4 | T::Int8, E::Signed)
                | (T::Text | T::Varchar(_), E::TextCode)
                | (T::Timestamp(_), E::EpochSeconds)
                | (T::Date, E::EpochDays)
        );
        // An unsigned u32 does not fit a smallint or an integer.
        let fits = !matches!((self.ty, self.encoding), (T::Int2 | T::Int4, E::Unsigned));
        if !ok || !fits {
            return Err(PgTypeRefusal::EncodingMismatch {
                ty: self.ty,
                encoding: self.encoding,
            });
        }
        if self.absent == AbsentAs::EmptyText && !self.ty.is_text() {
            return Err(PgTypeRefusal::EmptyTextOnNonText(self.ty));
        }
        Ok(())
    }
}

/// A text column's code table, at the frame the lanes were read under.
pub trait CodeText {
    /// The string for `code`, or `None` when the table holds no such code.
    fn text(&self, code: u32) -> Option<&str>;
}

impl CodeText for Vec<String> {
    fn text(&self, code: u32) -> Option<&str> {
        self.get(code as usize).map(String::as_str)
    }
}

impl CodeText for Vec<&str> {
    fn text(&self, code: u32) -> Option<&str> {
        self.get(code as usize).copied()
    }
}

/// The absent rule of the lane socket: presence 0 and value 0.
#[inline]
pub fn is_absent(present: bool, value: u32) -> bool {
    !present && value == 0
}

/// What [`write_cell`] wrote.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Written {
    /// SQL NULL; nothing was appended.
    Null,
    /// Text was appended to the buffer.
    Text,
}

/// Why [`write_cell`] refused a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellRefusal {
    /// A text cell's code is not in its code table.
    UnknownCode(u32),
    /// A text column was written with no code table.
    NoCodeTable,
    /// An unsigned value does not fit the column type.
    OutOfRange(u32),
}

/// Append PostgreSQL's text output for one cell to `out`.
///
/// The column must have passed [`PgColumn::admit`]. `codes` is read only for
/// a [`LaneEncoding::TextCode`] column.
pub fn write_cell(
    col: &PgColumn,
    present: bool,
    value: u32,
    codes: Option<&dyn CodeText>,
    out: &mut String,
) -> Result<Written, CellRefusal> {
    if is_absent(present, value) {
        return Ok(match col.absent {
            AbsentAs::Null => Written::Null,
            AbsentAs::EmptyText => Written::Text,
        });
    }
    match col.encoding {
        LaneEncoding::Bool => out.push(if value != 0 { 't' } else { 'f' }),
        LaneEncoding::Unsigned => {
            if col.ty == PgType::Int8 || value <= i32::MAX as u32 {
                let _ = write!(out, "{value}");
            } else {
                return Err(CellRefusal::OutOfRange(value));
            }
        }
        LaneEncoding::Signed => {
            let v = value as i32;
            if col.ty == PgType::Int2 && i16::try_from(v).is_err() {
                return Err(CellRefusal::OutOfRange(value));
            }
            let _ = write!(out, "{v}");
        }
        LaneEncoding::EpochSeconds => {
            let (days, secs) = (value / 86_400, value % 86_400);
            write_date(days, out);
            let _ = write!(
                out,
                " {:02}:{:02}:{:02}",
                secs / 3600,
                secs / 60 % 60,
                secs % 60
            );
        }
        LaneEncoding::EpochDays => write_date(value, out),
        LaneEncoding::TextCode => {
            let codes = codes.ok_or(CellRefusal::NoCodeTable)?;
            out.push_str(codes.text(value).ok_or(CellRefusal::UnknownCode(value))?);
        }
    }
    Ok(Written::Text)
}

/// `YYYY-MM-DD` for `days` since 1970-01-01 (Gregorian, proleptic), by
/// Howard Hinnant's `civil_from_days`.
fn write_date(days: u32, out: &mut String) {
    let z = i64::from(days) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    let _ = write!(out, "{y:04}-{m:02}-{d:02}");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(
        col: PgColumn,
        present: bool,
        value: u32,
        codes: Option<&dyn CodeText>,
    ) -> Option<String> {
        let mut s = String::new();
        match write_cell(&col, present, value, codes, &mut s).expect("written") {
            Written::Null => None,
            Written::Text => Some(s),
        }
    }

    const INT: PgColumn = PgColumn {
        ty: PgType::Int4,
        encoding: LaneEncoding::Signed,
        absent: AbsentAs::Null,
    };

    /// FAILS IF: a zero with presence 1 reads as NULL, or −0 reads as a value,
    /// or presence 0 with a nonzero value reads as absent.
    #[test]
    fn absent_is_minus_zero_only() {
        assert_eq!(cell(INT, true, 0, None).as_deref(), Some("0"));
        assert_eq!(cell(INT, false, 0, None), None);
        assert_eq!(cell(INT, false, 7, None).as_deref(), Some("7"));
        assert_eq!(cell(INT, true, (-3i32) as u32, None).as_deref(), Some("-3"));
    }

    /// FAILS IF: an absent text cell under `EmptyText` is NULL, or a present
    /// `''` code differs from it.
    #[test]
    fn empty_text_is_absent_or_code() {
        let codes = vec!["", "open"];
        let col = PgColumn {
            ty: PgType::Text,
            encoding: LaneEncoding::TextCode,
            absent: AbsentAs::EmptyText,
        };
        assert_eq!(cell(col, false, 0, Some(&codes)).as_deref(), Some(""));
        assert_eq!(cell(col, true, 0, Some(&codes)).as_deref(), Some(""));
        assert_eq!(cell(col, true, 1, Some(&codes)).as_deref(), Some("open"));
        let null = PgColumn {
            absent: AbsentAs::Null,
            ..col
        };
        assert_eq!(cell(null, false, 0, Some(&codes)), None);
        let mut s = String::new();
        assert_eq!(
            write_cell(&null, true, 9, Some(&codes), &mut s),
            Err(CellRefusal::UnknownCode(9))
        );
    }

    /// FAILS IF: a date or timestamp is off by a day across a leap year or
    /// a century.
    #[test]
    fn dates_and_timestamps_match_postgresql() {
        let ts = PgColumn {
            ty: PgType::Timestamp(Some(3)),
            encoding: LaneEncoding::EpochSeconds,
            absent: AbsentAs::Null,
        };
        let date = PgColumn {
            ty: PgType::Date,
            encoding: LaneEncoding::EpochDays,
            absent: AbsentAs::Null,
        };
        // Values checked with PostgreSQL: to_timestamp(n) at time zone 'UTC'.
        for (secs, text) in [
            (1, "1970-01-01 00:00:01"),
            (951_782_400, "2000-02-29 00:00:00"),
            (1_709_208_000 + 3_723, "2024-02-29 13:02:03"),
            (4_102_444_800, "2100-01-01 00:00:00"),
        ] {
            assert_eq!(cell(ts, true, secs, None).as_deref(), Some(text));
        }
        assert_eq!(
            cell(date, true, 11_016, None).as_deref(),
            Some("2000-02-29")
        );
        assert_eq!(
            cell(date, true, 47_540, None).as_deref(),
            Some("2100-02-28")
        );
    }

    /// FAILS IF: a type with a time zone, numeric or float is admitted, or a
    /// spelling PostgreSQL prints is not parsed back to itself.
    #[test]
    fn types_parse_and_refuse() {
        for t in [
            "boolean",
            "smallint",
            "integer",
            "bigint",
            "text",
            "character varying",
            "character varying(255)",
            "timestamp without time zone",
            "timestamp(3) without time zone",
            "date",
        ] {
            assert_eq!(PgType::parse(t).expect(t).gdesc(), t);
        }
        for t in [
            "timestamp with time zone",
            "timestamp(3) with time zone",
            "numeric",
            "double precision",
            "jsonb",
            "integer[]",
        ] {
            assert!(PgType::parse(t).is_err(), "{t} admitted");
        }
    }

    /// FAILS IF: an encoding that cannot hold the type is admitted, or `''`
    /// is allowed for a non-text column.
    #[test]
    fn admit_checks_encoding_and_absent() {
        assert!(INT.admit().is_ok());
        let bad = PgColumn {
            encoding: LaneEncoding::Unsigned,
            ..INT
        };
        assert!(bad.admit().is_err(), "u32 into integer");
        let big = PgColumn {
            ty: PgType::Int8,
            ..bad
        };
        assert!(big.admit().is_ok());
        let empty_int = PgColumn {
            absent: AbsentAs::EmptyText,
            ..INT
        };
        assert_eq!(
            empty_int.admit(),
            Err(PgTypeRefusal::EmptyTextOnNonText(PgType::Int4))
        );
        let tz = PgColumn {
            ty: PgType::Timestamp(None),
            encoding: LaneEncoding::Unsigned,
            absent: AbsentAs::Null,
        };
        assert!(tz.admit().is_err());
    }
}
