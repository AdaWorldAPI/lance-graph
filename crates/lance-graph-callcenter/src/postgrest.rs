//! PostgREST-shape handler stub (DM-8).
//!
//! HTTP shape that mirrors PostgREST's REST conventions so SMB+MedCare
//! client SDKs can target a stable surface. NO live HTTP server here —
//! just request/response types + a pure-function dispatcher.
//!
//! META-AGENT (follow-up wiring, do NOT do here):
//!   - gate behind feature `postgrest`; add `pub mod postgrest;` to lib.rs;
//!   - add `postgrest = ["dep:serde", "dep:serde_json"]` to Cargo.toml [features].
//!   - epiphany E4 outlook (PR #278): add an OPTIONAL feature
//!     `datafusion-dispatch = ["dep:datafusion"]` that compiles the stub
//!     `parsed_query_to_plan` at the bottom of this file into a real
//!     PostgREST → DataFusion → RlsRewriter dispatcher. Keep it optional
//!     so the zero-dep posture of the crate is preserved by default.
//!
//! This module is dependency-free: it uses only `std`, and emits JSON via
//! a small hand-rolled writer so it compiles under the crate's default
//! (zero-dep) feature set. When the meta-agent wires `postgrest = […]`,
//! the body emitter can be swapped for `serde_json` without touching the
//! public surface.
//!
//! Scope (subset that SMB+MedCare actually need):
//!   - GET    /<table>?<filter>&select=…&order=…&limit=…&offset=…
//!   - POST   /<table>          (insert; not yet implemented → 501)
//!   - PATCH  /<table>?<filter> (update; not yet implemented → 501)
//!   - DELETE /<table>?<filter> (delete; not yet implemented → 501)
//!   - HEAD/OPTIONS             (CORS-ish preflight; → 501 in EchoHandler)
//!
//! Filter syntax: `col=op.value` — e.g. `id=eq.42`, `name=ilike.*foo*`.
//!
//! See Layer-2 callcenter membrane plan §DM-8 for follow-up PRs that
//! wire this to a Lance-backed query engine.

use std::collections::BTreeMap;

// ── HTTP method ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HttpMethod {
    Get,
    Post,
    Patch,
    Delete,
    Head,
    Options,
}

impl HttpMethod {
    /// Parse an ASCII method token, case-insensitive. Returns None on unknown.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_uppercase().as_str() {
            "GET" => Some(Self::Get),
            "POST" => Some(Self::Post),
            "PATCH" => Some(Self::Patch),
            "DELETE" => Some(Self::Delete),
            "HEAD" => Some(Self::Head),
            "OPTIONS" => Some(Self::Options),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
            Self::Patch => "PATCH",
            Self::Delete => "DELETE",
            Self::Head => "HEAD",
            Self::Options => "OPTIONS",
        }
    }
}

// ── Request / response ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PostgRestRequest {
    pub method: HttpMethod,
    /// Path after `/`, e.g. `"patients"`, `"patients?id=eq.42"`.
    pub path: String,
    /// Headers we care about: `Prefer`, `Accept`, `Accept-Profile`, etc.
    /// Keys are stored as-given (callers are expected to normalise case).
    pub headers: BTreeMap<String, String>,
    /// JSON body for POST/PATCH; empty for GET/DELETE.
    pub body: Vec<u8>,
}

#[derive(Debug, Clone)]
pub struct PostgRestResponse {
    pub status: u16,
    pub headers: BTreeMap<String, String>,
    pub body: Vec<u8>,
}

impl PostgRestResponse {
    /// Convenience constructor for an `application/json` reply.
    pub fn json(status: u16, body: Vec<u8>) -> Self {
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".to_string(), "application/json".to_string());
        Self {
            status,
            headers,
            body,
        }
    }

    /// Convenience constructor for a plain-text reply.
    pub fn text(status: u16, body: impl Into<String>) -> Self {
        let mut headers = BTreeMap::new();
        headers.insert("Content-Type".to_string(), "text/plain".to_string());
        Self {
            status,
            headers,
            body: body.into().into_bytes(),
        }
    }
}

// ── Parsed query ─────────────────────────────────────────────────────────────

/// Parsed PostgREST query: table, filters, embeds, ordering.
///
/// (Subset — full PostgREST spec is huge; we cover what SMB+MedCare need.)
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ParsedQuery {
    pub table: String,
    /// e.g. `id=eq.42` → `Filter { col:"id", op: Eq, val:"42" }`.
    pub filters: Vec<Filter>,
    /// e.g. `?select=id,name`.
    pub select: Option<String>,
    /// e.g. `?order=created.desc`.
    pub order: Option<String>,
    pub limit: Option<u64>,
    pub offset: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Filter {
    pub column: String,
    pub op: FilterOp,
    pub value: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FilterOp {
    Eq,
    Neq,
    Gt,
    Gte,
    Lt,
    Lte,
    Like,
    ILike,
    In,
    Is,
}

impl FilterOp {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "eq" => Some(Self::Eq),
            "neq" => Some(Self::Neq),
            "gt" => Some(Self::Gt),
            "gte" => Some(Self::Gte),
            "lt" => Some(Self::Lt),
            "lte" => Some(Self::Lte),
            "like" => Some(Self::Like),
            "ilike" => Some(Self::ILike),
            "in" => Some(Self::In),
            "is" => Some(Self::Is),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Eq => "eq",
            Self::Neq => "neq",
            Self::Gt => "gt",
            Self::Gte => "gte",
            Self::Lt => "lt",
            Self::Lte => "lte",
            Self::Like => "like",
            Self::ILike => "ilike",
            Self::In => "in",
            Self::Is => "is",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
}

impl ParseError {
    /// Construct a `ParseError` from any string-ish value.
    ///
    /// Public so downstream crates (e.g. an upcoming `PostgRestDispatcher`
    /// that wires PostgREST → DataFusion → RlsRewriter) can surface their
    /// own parse-time errors through this same type.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl From<String> for ParseError {
    fn from(message: String) -> Self {
        Self { message }
    }
}

impl From<&str> for ParseError {
    fn from(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "postgrest parse error: {}", self.message)
    }
}

impl std::error::Error for ParseError {}

// ── URL decoding (HIGH) ──────────────────────────────────────────────────────

/// Percent-decode a URL component. Handles `%XX` and `+` → space (form-encoding).
/// Returns None if malformed (incomplete %XX or invalid hex).
fn percent_decode(s: &str) -> Option<String> {
    let mut out = Vec::with_capacity(s.len());
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        match bytes[i] {
            b'+' => {
                out.push(b' ');
                i += 1;
            }
            b'%' if i + 2 < bytes.len() => {
                let hex = std::str::from_utf8(&bytes[i + 1..i + 3]).ok()?;
                let v = u8::from_str_radix(hex, 16).ok()?;
                out.push(v);
                i += 3;
            }
            b'%' => return None, // truncated escape
            c => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

// ── Table-name validation (MEDIUM) ───────────────────────────────────────────

/// Validate a PostgREST table name: ASCII alphanumeric + underscore, not
/// starting with `_` (rejects magic / reserved names), not empty.
///
/// This rejects path-traversal (`..`), dotted paths (`users.json`), and any
/// non-ASCII identifier. PostgREST's real surface is more permissive (quoted
/// identifiers can contain anything), but the SMB+MedCare subset only ever
/// uses simple snake_case names; tighter validation is the safer default.
fn is_valid_table_name(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') && !s.starts_with('_')
}

// ── Path parser ──────────────────────────────────────────────────────────────

/// Pure function — no I/O.  Parses path query string into [`ParsedQuery`].
///
/// Accepts `"<table>"` or `"<table>?<query>"`.  Each query param is either:
///   - a reserved keyword (`select=…`, `order=…`, `limit=…`, `offset=…`), or
///   - a filter of the shape `col=op.value` (e.g. `id=eq.42`).
pub fn parse_path(path: &str) -> Result<ParsedQuery, ParseError> {
    let trimmed = path.trim_start_matches('/');
    if trimmed.is_empty() {
        return Err(ParseError::new("empty path"));
    }

    let (table_part, query_part) = match trimmed.find('?') {
        Some(idx) => (&trimmed[..idx], Some(&trimmed[idx + 1..])),
        None => (trimmed, None),
    };

    if table_part.is_empty() {
        return Err(ParseError::new("missing table name"));
    }

    // Disallow path segments past the table — DM-8 stub is single-table.
    if table_part.contains('/') {
        return Err(ParseError::new(format!(
            "nested path not supported: {table_part}"
        )));
    }

    // Reject path-traversal (`..`), dotted paths (`users.json`), magic
    // names (`_internal`), and any non-ASCII-identifier table label.
    if !is_valid_table_name(table_part) {
        return Err(ParseError::new(format!("invalid table name: {table_part}")));
    }

    let mut parsed = ParsedQuery {
        table: table_part.to_string(),
        ..Default::default()
    };

    let Some(qs) = query_part else {
        return Ok(parsed);
    };
    if qs.is_empty() {
        return Ok(parsed);
    }

    for raw_param in qs.split('&') {
        if raw_param.is_empty() {
            continue;
        }
        let (key, value) = match raw_param.find('=') {
            Some(idx) => (&raw_param[..idx], &raw_param[idx + 1..]),
            None => return Err(ParseError::new(format!("param missing '=': {raw_param}"))),
        };
        if key.is_empty() {
            return Err(ParseError::new(format!("empty param key in '{raw_param}'")));
        }

        match key {
            "select" => {
                let decoded = percent_decode(value).ok_or_else(|| {
                    ParseError::new(format!("malformed urlencoding in select: {value}"))
                })?;
                parsed.select = Some(decoded);
            }
            "order" => {
                let decoded = percent_decode(value).ok_or_else(|| {
                    ParseError::new(format!("malformed urlencoding in order: {value}"))
                })?;
                parsed.order = Some(decoded);
            }
            "limit" => {
                parsed.limit = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_| ParseError::new(format!("bad limit: {value}")))?,
                );
            }
            "offset" => {
                parsed.offset = Some(
                    value
                        .parse::<u64>()
                        .map_err(|_| ParseError::new(format!("bad offset: {value}")))?,
                );
            }
            _ => {
                // Filter: value is "op.val".
                let (op_str, val) = match value.find('.') {
                    Some(idx) => (&value[..idx], &value[idx + 1..]),
                    None => {
                        return Err(ParseError::new(format!(
                            "filter value missing '.': {key}={value}"
                        )))
                    }
                };
                let op = FilterOp::parse(op_str)
                    .ok_or_else(|| ParseError::new(format!("unknown filter op: {op_str}")))?;
                // URL-decode AFTER `op.` split so e.g. `eq.foo%40bar.com`
                // yields `foo@bar.com` (and not a confused dot-split on a
                // decoded `.`). Op tokens themselves are ASCII and never
                // need decoding.
                let decoded = percent_decode(val).ok_or_else(|| {
                    ParseError::new(format!("malformed urlencoding in filter value: {val}"))
                })?;
                parsed.filters.push(Filter {
                    column: key.to_string(),
                    op,
                    value: decoded,
                });
            }
        }
    }

    Ok(parsed)
}

// ── Dispatcher trait ─────────────────────────────────────────────────────────

/// Concrete impls (in follow-up PRs) wire this to a Lance-backed query
/// engine via `lance-graph-callcenter::query`.
pub trait PostgRestHandler: Send + Sync {
    fn handle(&self, req: PostgRestRequest) -> PostgRestResponse;
}

// ── Echo handler (test fixture) ──────────────────────────────────────────────

/// Echo handler for tests — returns the parsed query as JSON.
///
/// - `GET /`            → `200` with `[]` (no tables advertised).
/// - `GET /<table>?…`   → `200` with parsed query as JSON.
/// - any other method   → `501 Not Implemented`.
#[derive(Debug, Default)]
pub struct EchoHandler;

impl PostgRestHandler for EchoHandler {
    fn handle(&self, req: PostgRestRequest) -> PostgRestResponse {
        if req.method != HttpMethod::Get {
            return PostgRestResponse::text(
                501,
                format!("method {} not implemented", req.method.as_str()),
            );
        }

        // Empty / "/" path → list of tables (currently empty).
        let trimmed = req.path.trim_start_matches('/');
        if trimmed.is_empty() {
            return PostgRestResponse::json(200, b"[]".to_vec());
        }

        match parse_path(&req.path) {
            Ok(q) => PostgRestResponse::json(200, encode_parsed_query(&q)),
            Err(e) => {
                let msg = format!("{{\"error\":{}}}", json_string(&e.message));
                PostgRestResponse::json(400, msg.into_bytes())
            }
        }
    }
}

// ── Hand-rolled JSON emitter ─────────────────────────────────────────────────
//
// Avoids pulling `serde_json` (which is gated behind `auth-jwt` in this
// crate's Cargo.toml — see meta-agent note at the top of the file).
// Output is deterministic and only used for the echo / error path; when
// `postgrest = ["dep:serde", "dep:serde_json"]` lands, swap this for
// `serde_json::to_vec(&q).unwrap()`.

fn encode_parsed_query(q: &ParsedQuery) -> Vec<u8> {
    let mut s = String::new();
    s.push('{');
    s.push_str("\"table\":");
    s.push_str(&json_string(&q.table));

    s.push_str(",\"filters\":[");
    for (i, f) in q.filters.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('{');
        s.push_str("\"column\":");
        s.push_str(&json_string(&f.column));
        s.push_str(",\"op\":");
        s.push_str(&json_string(f.op.as_str()));
        s.push_str(",\"value\":");
        s.push_str(&json_string(&f.value));
        s.push('}');
    }
    s.push(']');

    s.push_str(",\"select\":");
    push_optional_string(&mut s, q.select.as_deref());

    s.push_str(",\"order\":");
    push_optional_string(&mut s, q.order.as_deref());

    s.push_str(",\"limit\":");
    match q.limit {
        Some(n) => s.push_str(&n.to_string()),
        None => s.push_str("null"),
    }
    s.push_str(",\"offset\":");
    match q.offset {
        Some(n) => s.push_str(&n.to_string()),
        None => s.push_str("null"),
    }

    s.push('}');
    s.into_bytes()
}

fn push_optional_string(out: &mut String, v: Option<&str>) {
    match v {
        Some(s) => out.push_str(&json_string(s)),
        None => out.push_str("null"),
    }
}

/// Encode a Rust `&str` as a JSON string literal (with surrounding quotes).
///
/// Supplementary-plane characters (U+10000 and above) are emitted as a
/// UTF-16 surrogate pair `\uXXXX\uXXXX` per RFC 8259 §7. BMP characters
/// pass through as-is (UTF-8 in the JSON text is fine; we only escape
/// when we have to).
fn json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c if (c as u32) > 0xFFFF => {
                // Supplementary plane: encode as UTF-16 surrogate pair.
                //   high = 0xD800 + ((cp - 0x10000) >> 10)
                //   low  = 0xDC00 + ((cp - 0x10000) & 0x3FF)
                let cp = c as u32 - 0x10000;
                let high = 0xD800 + (cp >> 10);
                let low = 0xDC00 + (cp & 0x3FF);
                out.push_str(&format!("\\u{:04x}\\u{:04x}", high, low));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn req_get(path: &str) -> PostgRestRequest {
        PostgRestRequest {
            method: HttpMethod::Get,
            path: path.to_string(),
            headers: BTreeMap::new(),
            body: Vec::new(),
        }
    }

    #[test]
    fn parse_eq_filter() {
        let q = parse_path("patients?id=eq.42").expect("parse ok");
        assert_eq!(q.table, "patients");
        assert_eq!(q.filters.len(), 1);
        let f = &q.filters[0];
        assert_eq!(f.column, "id");
        assert_eq!(f.op, FilterOp::Eq);
        assert_eq!(f.value, "42");
        assert!(q.select.is_none());
        assert!(q.order.is_none());
        assert!(q.limit.is_none());
        assert!(q.offset.is_none());
    }

    #[test]
    fn parse_select_order_limit() {
        let q = parse_path("patients?select=id,name&order=name.asc&limit=10").expect("parse ok");
        assert_eq!(q.table, "patients");
        assert_eq!(q.select.as_deref(), Some("id,name"));
        assert_eq!(q.order.as_deref(), Some("name.asc"));
        assert_eq!(q.limit, Some(10));
        assert!(q.offset.is_none());
        assert!(q.filters.is_empty());
    }

    #[test]
    fn parse_offset_and_multiple_filters() {
        let q = parse_path("orders?status=eq.open&total=gte.100&offset=5").expect("parse ok");
        assert_eq!(q.table, "orders");
        assert_eq!(q.offset, Some(5));
        assert_eq!(q.filters.len(), 2);
        assert_eq!(q.filters[0].op, FilterOp::Eq);
        assert_eq!(q.filters[1].op, FilterOp::Gte);
        assert_eq!(q.filters[1].value, "100");
    }

    #[test]
    fn parse_table_only() {
        let q = parse_path("patients").expect("parse ok");
        assert_eq!(q.table, "patients");
        assert!(q.filters.is_empty());
    }

    #[test]
    fn parse_leading_slash_ok() {
        let q = parse_path("/patients?id=eq.7").expect("parse ok");
        assert_eq!(q.table, "patients");
        assert_eq!(q.filters[0].value, "7");
    }

    #[test]
    fn parse_empty_path_errors() {
        let err = parse_path("").expect_err("must error");
        assert!(err.message.contains("empty"));
    }

    #[test]
    fn parse_empty_path_slash_only_errors() {
        let err = parse_path("/").expect_err("must error");
        assert!(err.message.contains("empty"));
    }

    #[test]
    fn parse_unknown_op_errors() {
        let err = parse_path("patients?id=foo.42").expect_err("must error");
        assert!(err.message.contains("unknown filter op"));
    }

    #[test]
    fn parse_filter_missing_dot_errors() {
        let err = parse_path("patients?id=eq42").expect_err("must error");
        assert!(err.message.contains("missing '.'"));
    }

    #[test]
    fn parse_bad_limit_errors() {
        let err = parse_path("patients?limit=abc").expect_err("must error");
        assert!(err.message.contains("bad limit"));
    }

    #[test]
    fn parse_nested_path_errors() {
        let err = parse_path("patients/42").expect_err("must error");
        assert!(err.message.contains("nested path"));
    }

    #[test]
    fn echo_get_returns_parsed_query_as_json() {
        let h = EchoHandler;
        let resp = h.handle(req_get("patients?id=eq.42"));
        assert_eq!(resp.status, 200);
        assert_eq!(
            resp.headers.get("Content-Type").map(String::as_str),
            Some("application/json")
        );
        let body = std::str::from_utf8(&resp.body).expect("utf8");
        assert!(body.contains("\"table\":\"patients\""), "body={body}");
        assert!(body.contains("\"column\":\"id\""), "body={body}");
        assert!(body.contains("\"op\":\"eq\""), "body={body}");
        assert!(body.contains("\"value\":\"42\""), "body={body}");
    }

    #[test]
    fn echo_get_root_returns_empty_list() {
        let h = EchoHandler;
        let resp = h.handle(req_get("/"));
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, b"[]");
    }

    #[test]
    fn echo_post_returns_501() {
        let h = EchoHandler;
        let resp = h.handle(PostgRestRequest {
            method: HttpMethod::Post,
            path: "patients".to_string(),
            headers: BTreeMap::new(),
            body: b"{\"id\":1}".to_vec(),
        });
        assert_eq!(resp.status, 501);
        let body = std::str::from_utf8(&resp.body).expect("utf8");
        assert!(body.contains("POST"));
        assert!(body.contains("not implemented"));
    }

    #[test]
    fn echo_patch_delete_head_options_return_501() {
        let h = EchoHandler;
        for m in [
            HttpMethod::Patch,
            HttpMethod::Delete,
            HttpMethod::Head,
            HttpMethod::Options,
        ] {
            let resp = h.handle(PostgRestRequest {
                method: m.clone(),
                path: "patients".to_string(),
                headers: BTreeMap::new(),
                body: Vec::new(),
            });
            assert_eq!(resp.status, 501, "method {} should be 501", m.as_str());
        }
    }

    #[test]
    fn echo_get_bad_path_returns_400() {
        let h = EchoHandler;
        let resp = h.handle(req_get("patients?id=foo.42"));
        assert_eq!(resp.status, 400);
        let body = std::str::from_utf8(&resp.body).expect("utf8");
        assert!(body.contains("error"));
        assert!(body.contains("unknown filter op"));
    }

    #[test]
    fn http_method_parse_roundtrip() {
        for m in [
            HttpMethod::Get,
            HttpMethod::Post,
            HttpMethod::Patch,
            HttpMethod::Delete,
            HttpMethod::Head,
            HttpMethod::Options,
        ] {
            assert_eq!(HttpMethod::parse(m.as_str()), Some(m.clone()));
            // case-insensitive
            assert_eq!(HttpMethod::parse(&m.as_str().to_ascii_lowercase()), Some(m));
        }
        assert!(HttpMethod::parse("FROOB").is_none());
    }

    #[test]
    fn filter_op_parse_roundtrip() {
        for op in [
            FilterOp::Eq,
            FilterOp::Neq,
            FilterOp::Gt,
            FilterOp::Gte,
            FilterOp::Lt,
            FilterOp::Lte,
            FilterOp::Like,
            FilterOp::ILike,
            FilterOp::In,
            FilterOp::Is,
        ] {
            assert_eq!(FilterOp::parse(op.as_str()), Some(op));
        }
        assert!(FilterOp::parse("nope").is_none());
    }

    #[test]
    fn json_string_escapes() {
        assert_eq!(json_string("hi"), "\"hi\"");
        assert_eq!(json_string("a\"b"), "\"a\\\"b\"");
        assert_eq!(json_string("a\\b"), "\"a\\\\b\"");
        assert_eq!(json_string("a\nb"), "\"a\\nb\"");
        assert_eq!(json_string("\x01"), "\"\\u0001\"");
    }

    #[test]
    fn encode_parsed_query_includes_nulls() {
        let q = ParsedQuery {
            table: "t".to_string(),
            ..Default::default()
        };
        let bytes = encode_parsed_query(&q);
        let s = std::str::from_utf8(&bytes).unwrap();
        assert!(s.contains("\"select\":null"), "{s}");
        assert!(s.contains("\"order\":null"), "{s}");
        assert!(s.contains("\"limit\":null"), "{s}");
        assert!(s.contains("\"offset\":null"), "{s}");
        assert!(s.contains("\"filters\":[]"), "{s}");
    }

    // ── Filter-op coverage (LOW) — ilike / in / is / like ────────────────

    #[test]
    fn parse_ilike_filter() {
        let q = parse_path("users?email=ilike.*@example.com").expect("parse ok");
        assert_eq!(q.table, "users");
        assert_eq!(q.filters.len(), 1);
        let f = &q.filters[0];
        assert_eq!(f.column, "email");
        assert_eq!(f.op, FilterOp::ILike);
        assert_eq!(f.value, "*@example.com");
    }

    #[test]
    fn parse_in_filter() {
        let q = parse_path("users?id=in.(1,2,3)").expect("parse ok");
        assert_eq!(q.filters.len(), 1);
        let f = &q.filters[0];
        assert_eq!(f.column, "id");
        assert_eq!(f.op, FilterOp::In);
        assert_eq!(f.value, "(1,2,3)");
    }

    #[test]
    fn parse_is_null_filter() {
        let q = parse_path("users?status=is.null").expect("parse ok");
        let f = &q.filters[0];
        assert_eq!(f.column, "status");
        assert_eq!(f.op, FilterOp::Is);
        assert_eq!(f.value, "null");
    }

    #[test]
    fn parse_like_filter() {
        let q = parse_path("users?name=like.J*").expect("parse ok");
        let f = &q.filters[0];
        assert_eq!(f.column, "name");
        assert_eq!(f.op, FilterOp::Like);
        assert_eq!(f.value, "J*");
    }

    // ── URL-decode tests (HIGH) ──────────────────────────────────────────

    #[test]
    fn parse_url_decode_space() {
        let q = parse_path("users?name=eq.John%20Doe").expect("parse ok");
        assert_eq!(q.filters[0].value, "John Doe");
    }

    #[test]
    fn parse_url_decode_at_sign() {
        let q = parse_path("users?email=eq.foo%40bar.com").expect("parse ok");
        assert_eq!(q.filters[0].value, "foo@bar.com");
    }

    #[test]
    fn parse_url_decode_plus_sign() {
        // %2B → '+' (literal plus, NOT a space — that's only bare `+`).
        let q = parse_path("users?notes=eq.Hello%2BWorld").expect("parse ok");
        assert_eq!(q.filters[0].value, "Hello+World");
    }

    #[test]
    fn parse_url_decode_bare_plus_is_space() {
        // form-encoding convention: bare `+` decodes to space.
        let q = parse_path("users?name=eq.John+Doe").expect("parse ok");
        assert_eq!(q.filters[0].value, "John Doe");
    }

    #[test]
    fn parse_url_decode_select_and_order() {
        let q = parse_path("users?select=id%2Cname&order=created.desc").expect("parse ok");
        assert_eq!(q.select.as_deref(), Some("id,name"));
        assert_eq!(q.order.as_deref(), Some("created.desc"));
    }

    #[test]
    fn parse_url_decode_truncated_escape_errors() {
        let err = parse_path("users?name=eq.foo%2").expect_err("must error");
        assert!(
            err.message.contains("malformed urlencoding"),
            "{}",
            err.message
        );
    }

    #[test]
    fn parse_url_decode_bad_hex_errors() {
        let err = parse_path("users?name=eq.foo%ZZbar").expect_err("must error");
        assert!(
            err.message.contains("malformed urlencoding"),
            "{}",
            err.message
        );
    }

    // ── Table-validation tests (MEDIUM) ──────────────────────────────────

    #[test]
    fn parse_table_traversal_errors() {
        // `..` contains `.`, plus `/` → caught by either nested-path or
        // invalid-table-name guard. Either error message is acceptable.
        let err = parse_path("../../../etc/passwd").expect_err("must error");
        assert!(
            err.message.contains("nested path") || err.message.contains("invalid table name"),
            "{}",
            err.message
        );
    }

    #[test]
    fn parse_table_with_period_errors() {
        let err = parse_path("users.json?id=eq.1").expect_err("must error");
        assert!(
            err.message.contains("invalid table name"),
            "{}",
            err.message
        );
    }

    #[test]
    fn parse_table_underscore_prefix_errors() {
        let err = parse_path("_internal?id=eq.1").expect_err("must error");
        assert!(
            err.message.contains("invalid table name"),
            "{}",
            err.message
        );
    }

    #[test]
    fn parse_table_unicode_errors() {
        let err = parse_path("ünits?id=eq.1").expect_err("must error");
        assert!(
            err.message.contains("invalid table name"),
            "{}",
            err.message
        );
    }

    #[test]
    fn parse_table_leading_slash_still_works() {
        // Existing behaviour preserved.
        let q = parse_path("/users").expect("parse ok");
        assert_eq!(q.table, "users");
    }

    // ── ParseError public surface ────────────────────────────────────────

    #[test]
    fn parse_error_public_constructors() {
        let a = ParseError::new("a");
        let b: ParseError = "b".into();
        let c: ParseError = String::from("c").into();
        assert_eq!(a.message, "a");
        assert_eq!(b.message, "b");
        assert_eq!(c.message, "c");
        // Display impl is reachable.
        assert!(format!("{a}").contains("a"));
    }

    // ── Surrogate-pair JSON escaping (LOW) ───────────────────────────────

    #[test]
    fn json_string_supplementary_plane_emits_surrogate_pair() {
        // U+1F600 GRINNING FACE → 😀
        let out = json_string("\u{1F600}");
        assert_eq!(out, "\"\\ud83d\\ude00\"", "{out}");
    }

    #[test]
    fn json_string_bmp_passes_through() {
        // BMP chars are emitted as UTF-8, not escaped.
        let out = json_string("café");
        assert_eq!(out, "\"café\"");
    }
}

// ── EPIPHANY E4 SEED — PostgREST → DataFusion dispatch stub ──────────────────
//
// Doc-only stub for the upcoming `PostgRestDispatcher` (PR #278 outlook E4)
// that wires PostgREST → DataFusion → RlsRewriter. Compiled only under the
// optional `datafusion-dispatch` feature so the zero-dep default posture of
// the crate is preserved.
//
// META-AGENT (follow-up wiring, do NOT do here):
//   - add `datafusion-dispatch = ["dep:datafusion"]` to Cargo.toml [features];
//   - add `datafusion = { version = "...", optional = true }` to dependencies;
//   - flesh out `parsed_query_to_plan` to translate a `ParsedQuery` into a
//     `LogicalPlan` (TableScan → Filter → Projection → Sort → Limit), then
//     hand it to `crate::rls::RlsRewriter` for row-level-security overlay.

/// Convert a [`ParsedQuery`] to a DataFusion `LogicalPlan`.
///
/// **NOT IMPLEMENTED** in this PR. Stub interface for the upcoming
/// `PostgRestDispatcher` that wires PostgREST → DataFusion → RlsRewriter.
/// See PR #278 outlook epiphany E4.
#[cfg(feature = "datafusion-dispatch")]
pub fn parsed_query_to_plan(
    _query: &ParsedQuery,
    _ctx: &datafusion::execution::context::SessionContext,
) -> Result<datafusion::logical_expr::LogicalPlan, ParseError> {
    Err(ParseError::new(
        "parsed_query_to_plan: not yet implemented (PR #278 outlook E4)",
    ))
}
