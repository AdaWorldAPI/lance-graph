//! A read-only LDAP v3 server over a projected version, as a protocol
//! handler: request bytes in, response bytes out. No sockets — the host owns
//! the listener and calls [`Server::handle`] per PDU ([`frame`] splits a
//! stream into PDUs).
//!
//! Supported: simple bind (anonymous, or name + password checked by the
//! [`Authority`]), search (base, one-level, subtree; `and`, `or`, `not`,
//! equality, presence, substrings, `>=`, `<=`, approximate as equality;
//! attribute selection, `*`, `1.1`, types-only, size and time limits), the
//! rootDSE, unbind and abandon. Ordering (`>=`, `<=`) follows the
//! attribute: integers numerically, text case-insensitively, binary
//! Undefined. Filters nest at most [`MAX_FILTER_DEPTH`] deep. No control is
//! supported: a critical one answers `unavailableCriticalExtension`, a
//! non-critical one is ignored. Everything that writes — modify, add, delete,
//! modify DN — and compare and extended operations are answered
//! `unwillingToPerform`: the emulation never changes. A change to the
//! directory goes through simulate, validate and plan.
//!
//! **Access.** An anonymous session reads the rootDSE only, as Active
//! Directory does by default. A bound session sees an entry only when
//! [`Authority::visible`] says so, and an attribute only when
//! [`Authority::readable`] does; a refused entry is absent (no
//! "insufficient access" that would confirm it exists), and a refused
//! attribute is omitted. The check runs before an entry is encoded.

use crate::ad::{Entry, Projection, Value};
use ogar_dir_core::Dn;

// ── BER ──────────────────────────────────────────────────────────────────

/// Why a PDU could not be read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LdapError {
    /// Malformed BER.
    Malformed,
    /// A PDU longer than [`MAX_PDU`].
    TooLarge,
}

/// The largest request accepted.
pub const MAX_PDU: usize = 1 << 20;

#[derive(Clone, Copy)]
struct Tlv<'a> {
    tag: u8,
    value: &'a [u8],
}

fn read_tlv(b: &[u8]) -> Result<(Tlv<'_>, &[u8]), LdapError> {
    let (&tag, rest) = b.split_first().ok_or(LdapError::Malformed)?;
    if tag & 0x1f == 0x1f {
        return Err(LdapError::Malformed); // multi-byte tags are not LDAP
    }
    let (&l0, mut rest) = rest.split_first().ok_or(LdapError::Malformed)?;
    let len = if l0 < 0x80 {
        usize::from(l0)
    } else {
        let n = usize::from(l0 & 0x7f);
        if n == 0 || n > 4 || rest.len() < n {
            return Err(LdapError::Malformed);
        }
        let mut len = 0usize;
        for &x in &rest[..n] {
            len = (len << 8) | usize::from(x);
        }
        rest = &rest[n..];
        len
    };
    if len > MAX_PDU {
        return Err(LdapError::TooLarge);
    }
    if rest.len() < len {
        return Err(LdapError::Malformed);
    }
    Ok((
        Tlv {
            tag,
            value: &rest[..len],
        },
        &rest[len..],
    ))
}

fn children(b: &[u8]) -> Result<Vec<Tlv<'_>>, LdapError> {
    let mut out = Vec::new();
    let mut rest = b;
    while !rest.is_empty() {
        let (t, r) = read_tlv(rest)?;
        out.push(t);
        rest = r;
    }
    Ok(out)
}

fn int(t: &Tlv<'_>) -> Result<i64, LdapError> {
    if t.value.is_empty() || t.value.len() > 8 {
        return Err(LdapError::Malformed);
    }
    let mut v: i64 = if t.value[0] & 0x80 != 0 { -1 } else { 0 };
    for &x in t.value {
        v = (v << 8) | i64::from(x);
    }
    Ok(v)
}

fn text(t: &Tlv<'_>) -> Result<String, LdapError> {
    String::from_utf8(t.value.to_vec()).map_err(|_| LdapError::Malformed)
}

/// The byte length of the first complete PDU in `buf`, `None` while more
/// bytes are needed.
pub fn frame(buf: &[u8]) -> Result<Option<usize>, LdapError> {
    match buf.first() {
        None => return Ok(None),
        Some(&0x30) => {}
        Some(_) => return Err(LdapError::Malformed),
    }
    match header_len(buf) {
        None => Ok(None),
        Some(None) => Err(LdapError::Malformed),
        Some(Some((_, l))) if l > MAX_PDU => Err(LdapError::TooLarge),
        Some(Some((h, l))) => Ok((buf.len() >= h + l).then_some(h + l)),
    }
}

/// `None` = header incomplete; `Some(None)` = header invalid;
/// `Some(Some((header_len, body_len)))`.
fn header_len(b: &[u8]) -> Option<Option<(usize, usize)>> {
    let l0 = *b.get(1)?;
    if l0 < 0x80 {
        return Some(Some((2, usize::from(l0))));
    }
    let n = usize::from(l0 & 0x7f);
    if n == 0 || n > 4 {
        return Some(None);
    }
    let bytes = b.get(2..2 + n)?;
    Some(Some((
        2 + n,
        bytes.iter().fold(0usize, |a, &x| (a << 8) | usize::from(x)),
    )))
}

fn put(out: &mut Vec<u8>, tag: u8, value: &[u8]) {
    out.push(tag);
    let n = value.len();
    if n < 0x80 {
        out.push(n as u8);
    } else {
        let bytes = n.to_be_bytes();
        let skip = bytes.iter().take_while(|&&x| x == 0).count();
        out.push(0x80 | (bytes.len() - skip) as u8);
        out.extend_from_slice(&bytes[skip..]);
    }
    out.extend_from_slice(value);
}

fn tlv(tag: u8, value: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(value.len() + 6);
    put(&mut v, tag, value);
    v
}

fn put_int(tag: u8, v: i64) -> Vec<u8> {
    let bytes = v.to_be_bytes();
    let mut i = 0;
    while i < 7
        && ((bytes[i] == 0 && bytes[i + 1] & 0x80 == 0)
            || (bytes[i] == 0xff && bytes[i + 1] & 0x80 != 0))
    {
        i += 1;
    }
    tlv(tag, &bytes[i..])
}

// ── protocol ─────────────────────────────────────────────────────────────

/// LDAP result codes used here (RFC 4511 §4.1.9).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum ResultCode {
    /// success
    Success = 0,
    /// operationsError
    OperationsError = 1,
    /// protocolError
    ProtocolError = 2,
    /// sizeLimitExceeded
    SizeLimitExceeded = 4,
    /// timeLimitExceeded
    TimeLimitExceeded = 3,
    /// authMethodNotSupported
    AuthMethodNotSupported = 7,
    /// unavailableCriticalExtension
    UnavailableCriticalExtension = 12,
    /// noSuchObject
    NoSuchObject = 32,
    /// invalidCredentials
    InvalidCredentials = 49,
    /// unwillingToPerform
    UnwillingToPerform = 53,
}

/// Who may bind, and what each bound actor may read. Supplied by the host:
/// the IAM decision lives there, never in the protocol handler.
pub trait Authority {
    /// A bound principal.
    type Actor;
    /// Check a simple bind. `name` is as sent (a DN or a UPN); the password
    /// is not kept. `None` = invalid credentials.
    fn bind(&self, name: &str, password: &[u8]) -> Option<Self::Actor>;
    /// Whether `actor` may see `entry` at all.
    fn visible(&self, actor: &Self::Actor, entry: &Entry) -> bool;
    /// Whether `actor` may read attribute `name` of a visible `entry`.
    fn readable(&self, _actor: &Self::Actor, _entry: &Entry, _name: &str) -> bool {
        true
    }
}

/// One client's state.
pub struct Session<A> {
    actor: Option<A>,
    closed: bool,
}

impl<A> Default for Session<A> {
    fn default() -> Self {
        Self {
            actor: None,
            closed: false,
        }
    }
}

impl<A> Session<A> {
    /// The bound actor, `None` while anonymous.
    pub fn actor(&self) -> Option<&A> {
        self.actor.as_ref()
    }
    /// True after unbind: the host closes the connection.
    pub fn closed(&self) -> bool {
        self.closed
    }
}

/// A DN as comparable components: `(lowercase type, lowercase value)`,
/// leaf first.
type Key = Vec<(String, String)>;

fn key(dn: &str) -> Option<Key> {
    if dn.is_empty() {
        return Some(Vec::new());
    }
    Some(
        Dn::parse(dn)
            .ok()?
            .rdns
            .into_iter()
            .map(|r| (r.attr.to_lowercase(), r.value.to_lowercase()))
            .collect(),
    )
}

/// The served directory: one projected version under its naming context.
pub struct Directory {
    naming_context: String,
    nc_key: Key,
    nc_entry: Entry,
    entries: Vec<(Key, Entry)>,
}

impl Directory {
    /// Serve `p`, projected under `naming_context` (the same string given
    /// to [`crate::ad::project`]). `None` if it is not a DN.
    pub fn new(p: Projection, naming_context: &str) -> Option<Self> {
        let nc_key = key(naming_context).filter(|k| !k.is_empty())?;
        let dc: Vec<String> = Dn::parse(naming_context)
            .ok()?
            .rdns
            .into_iter()
            .map(|r| r.value)
            .collect();
        let nc_entry = Entry {
            dn: naming_context.to_string(),
            origin: crate::ad::Origin::Observed,
            node: None,
            attrs: vec![
                ("objectClass", Value::Text("top".into())),
                ("objectClass", Value::Text("domain".into())),
                ("objectClass", Value::Text("domainDNS".into())),
                ("dc", Value::Text(dc[0].clone())),
            ],
        };
        let entries = p
            .entries
            .into_iter()
            .map(|e| key(&e.dn).map(|k| (k, e)))
            .collect::<Option<Vec<_>>>()?;
        Some(Self {
            naming_context: naming_context.to_string(),
            nc_key,
            nc_entry,
            entries,
        })
    }

    fn root_dse(&self) -> Entry {
        Entry {
            dn: String::new(),
            origin: crate::ad::Origin::Observed,
            node: None,
            attrs: vec![
                ("objectClass", Value::Text("top".into())),
                ("namingContexts", Value::Text(self.naming_context.clone())),
                (
                    "defaultNamingContext",
                    Value::Text(self.naming_context.clone()),
                ),
                ("supportedLDAPVersion", Value::Text("3".into())),
                ("vendorName", Value::Text("dir-sim (emulation)".into())),
                ("isReadOnly", Value::Text("TRUE".into())),
            ],
        }
    }

    fn all(&self) -> impl Iterator<Item = (&Key, &Entry)> {
        std::iter::once((&self.nc_key, &self.nc_entry))
            .chain(self.entries.iter().map(|(k, e)| (k, e)))
    }
}

/// The handler: a directory and the authority deciding access.
pub struct Server<'a, P: Authority> {
    dir: &'a Directory,
    authority: &'a P,
}

#[derive(Debug)]
enum Filter {
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
    Eq(String, Vec<u8>),
    Ge(String, Vec<u8>),
    Le(String, Vec<u8>),
    Present(String),
    Sub {
        attr: String,
        initial: Option<Vec<u8>>,
        any: Vec<Vec<u8>>,
        last: Option<Vec<u8>>,
    },
    /// An extensible match: not supported, evaluates to Undefined.
    Unsupported,
}

/// The deepest filter accepted. A deeper one is malformed: parsing and
/// evaluation recurse, and an unauthenticated client must not be able to
/// exhaust the stack.
pub const MAX_FILTER_DEPTH: usize = 32;

fn parse_filter(t: &Tlv<'_>) -> Result<Filter, LdapError> {
    parse_filter_at(t, 0)
}

fn parse_filter_at(t: &Tlv<'_>, depth: usize) -> Result<Filter, LdapError> {
    if depth >= MAX_FILTER_DEPTH {
        return Err(LdapError::Malformed);
    }
    let sub = |t: &Tlv<'_>| parse_filter_at(t, depth + 1);
    let pair = |t: &Tlv<'_>| -> Result<(String, Vec<u8>), LdapError> {
        let c = children(t.value)?;
        if c.len() != 2 {
            return Err(LdapError::Malformed);
        }
        Ok((text(&c[0])?.to_lowercase(), c[1].value.to_vec()))
    };
    Ok(match t.tag {
        0xa0 => Filter::And(
            children(t.value)?
                .iter()
                .map(sub)
                .collect::<Result<_, _>>()?,
        ),
        0xa1 => Filter::Or(
            children(t.value)?
                .iter()
                .map(sub)
                .collect::<Result<_, _>>()?,
        ),
        0xa2 => {
            let c = children(t.value)?;
            if c.len() != 1 {
                return Err(LdapError::Malformed);
            }
            Filter::Not(Box::new(sub(&c[0])?))
        }
        0xa3 | 0xa8 => {
            let (a, v) = pair(t)?;
            Filter::Eq(a, v)
        }
        0xa5 => {
            let (a, v) = pair(t)?;
            Filter::Ge(a, v)
        }
        0xa6 => {
            let (a, v) = pair(t)?;
            Filter::Le(a, v)
        }
        0x87 => Filter::Present(text(t)?.to_lowercase()),
        0xa4 => {
            let c = children(t.value)?;
            if c.len() != 2 || c[1].tag != 0x30 {
                return Err(LdapError::Malformed);
            }
            let (mut initial, mut any, mut last) = (None, Vec::new(), None);
            for s in children(c[1].value)? {
                match s.tag {
                    0x80 if initial.is_none() && any.is_empty() && last.is_none() => {
                        initial = Some(s.value.to_vec())
                    }
                    0x81 if last.is_none() => any.push(s.value.to_vec()),
                    0x82 if last.is_none() => last = Some(s.value.to_vec()),
                    _ => return Err(LdapError::Malformed),
                }
            }
            Filter::Sub {
                attr: text(&c[0])?.to_lowercase(),
                initial,
                any,
                last,
            }
        }
        0xa9 => Filter::Unsupported,
        _ => return Err(LdapError::Malformed),
    })
}

fn is_integer(attr: &str) -> bool {
    matches!(attr, "useraccountcontrol" | "supportedldapversion")
}

/// `>=` / `<=` by the attribute's ordering rule: integers numerically, text
/// case-insensitively. A binary attribute has no ordering rule, and an
/// assertion value that is not an integer for an integer attribute is
/// invalid: both are Undefined.
fn order(
    attr: &str,
    v: &[u8],
    vals: Vec<Vec<u8>>,
    keep: impl Fn(std::cmp::Ordering) -> bool,
) -> Option<bool> {
    if is_binary(attr) {
        return None;
    }
    if is_integer(attr) {
        let num = |b: &[u8]| std::str::from_utf8(b).ok()?.parse::<i64>().ok();
        let v = num(v)?;
        return Some(vals.iter().filter_map(|x| num(x)).any(|x| keep(x.cmp(&v))));
    }
    let v = fold(attr, v);
    Some(vals.iter().any(|x| keep(x.as_slice().cmp(v.as_slice()))))
}

fn is_binary(attr: &str) -> bool {
    matches!(attr, "objectguid" | "msexchmailboxguid")
}

fn fold(attr: &str, v: &[u8]) -> Vec<u8> {
    if is_binary(attr) {
        v.to_vec()
    } else {
        String::from_utf8_lossy(v).to_lowercase().into_bytes()
    }
}

/// Values of `attr` on `e`, as `(raw, matching form)`; includes the
/// `distinguishedName` the entry has by definition.
fn values(e: &Entry, attr: &str) -> Vec<Vec<u8>> {
    if attr == "distinguishedname" {
        return vec![e.dn.as_bytes().to_vec()];
    }
    e.attrs
        .iter()
        .filter(|(n, _)| n.eq_ignore_ascii_case(attr))
        .map(|(_, v)| match v {
            Value::Text(t) => t.as_bytes().to_vec(),
            Value::Binary(b) => b.clone(),
        })
        .collect()
}

/// Three-valued filter evaluation (RFC 4511 §4.5.1.7): `None` = Undefined.
fn eval(f: &Filter, e: &Entry, readable: &dyn Fn(&str) -> bool) -> Option<bool> {
    let vals = |a: &str| -> Option<Vec<Vec<u8>>> {
        if !readable(a) {
            return Some(Vec::new()); // an unreadable attribute is absent
        }
        Some(values(e, a).into_iter().map(|v| fold(a, &v)).collect())
    };
    match f {
        Filter::And(fs) => {
            let mut out = Some(true);
            for f in fs {
                match eval(f, e, readable) {
                    Some(false) => return Some(false),
                    None => out = None,
                    Some(true) => {}
                }
            }
            out
        }
        Filter::Or(fs) => {
            let mut out = Some(false);
            for f in fs {
                match eval(f, e, readable) {
                    Some(true) => return Some(true),
                    None => out = None,
                    Some(false) => {}
                }
            }
            out
        }
        Filter::Not(f) => eval(f, e, readable).map(|b| !b),
        Filter::Present(a) => Some(!vals(a)?.is_empty()),
        Filter::Eq(a, v) => {
            let v = fold(a, v);
            Some(vals(a)?.contains(&v))
        }
        Filter::Ge(a, v) => order(a, v, vals(a)?, |o| o.is_ge()),
        Filter::Le(a, v) => order(a, v, vals(a)?, |o| o.is_le()),
        Filter::Sub {
            attr,
            initial,
            any,
            last,
        } => {
            let initial = initial.as_ref().map(|x| fold(attr, x));
            let any: Vec<Vec<u8>> = any.iter().map(|x| fold(attr, x)).collect();
            let last = last.as_ref().map(|x| fold(attr, x));
            Some(vals(attr)?.iter().any(|x| {
                let mut rest: &[u8] = x;
                if let Some(i) = &initial {
                    let Some(r) = rest.strip_prefix(i.as_slice()) else {
                        return false;
                    };
                    rest = r;
                }
                for a in any.iter().filter(|a| !a.is_empty()) {
                    match rest.windows(a.len()).position(|w| w == a.as_slice()) {
                        Some(p) => rest = &rest[p + a.len()..],
                        None => return false,
                    }
                }
                last.as_ref().is_none_or(|l| rest.ends_with(l))
            }))
        }
        Filter::Unsupported => None,
    }
}

/// Whether a `controls` element carries a control marked critical.
fn has_critical(ctrls: &Tlv<'_>) -> Result<bool, LdapError> {
    for ctrl in children(ctrls.value)? {
        let f = children(ctrl.value)?;
        if ctrl.tag != 0x30 || f.is_empty() || f[0].tag != 0x04 {
            return Err(LdapError::Malformed);
        }
        if f.get(1)
            .is_some_and(|b| b.tag == 0x01 && b.value.first().is_some_and(|&x| x != 0))
        {
            return Ok(true);
        }
    }
    Ok(false)
}

fn result(id: i64, op: u8, code: ResultCode, msg: &str) -> Vec<u8> {
    let mut body = put_int(0x0a, code as i64);
    body.extend(tlv(0x04, b""));
    body.extend(tlv(0x04, msg.as_bytes()));
    message(id, &tlv(op, &body))
}

fn message(id: i64, op: &[u8]) -> Vec<u8> {
    let mut body = put_int(0x02, id);
    body.extend_from_slice(op);
    tlv(0x30, &body)
}

const READ_ONLY: &str = "the directory emulation is read-only";

impl<'a, P: Authority> Server<'a, P> {
    /// A handler over `dir`, with access decided by `authority`.
    pub fn new(dir: &'a Directory, authority: &'a P) -> Self {
        Self { dir, authority }
    }

    /// Handle one request PDU; returns the response PDUs in order (none for
    /// unbind and abandon). A malformed PDU is answered with a notice of
    /// disconnection and closes the session.
    pub fn handle(&self, s: &mut Session<P::Actor>, pdu: &[u8]) -> Vec<Vec<u8>> {
        match self.dispatch(s, pdu) {
            Ok(out) => out,
            Err(_) => {
                s.closed = true;
                // Notice of Disconnection (RFC 4511 §4.4.1): an unsolicited
                // ExtendedResponse with messageID 0.
                let mut body = put_int(0x0a, ResultCode::ProtocolError as i64);
                body.extend(tlv(0x04, b""));
                body.extend(tlv(0x04, b"malformed request"));
                body.extend(tlv(0x8a, b"1.3.6.1.4.1.1466.20036"));
                vec![message(0, &tlv(0x78, &body))]
            }
        }
    }

    fn dispatch(&self, s: &mut Session<P::Actor>, pdu: &[u8]) -> Result<Vec<Vec<u8>>, LdapError> {
        let (m, rest) = read_tlv(pdu)?;
        if m.tag != 0x30 || !rest.is_empty() {
            return Err(LdapError::Malformed);
        }
        let c = children(m.value)?;
        if !(2..=3).contains(&c.len()) || c[0].tag != 0x02 {
            return Err(LdapError::Malformed);
        }
        let id = int(&c[0])?;
        if !(0..=i64::from(i32::MAX)).contains(&id) {
            return Err(LdapError::Malformed);
        }
        let op = c[1];
        // No control is supported: a critical one refuses the operation
        // (RFC 4511 §4.1.11); a non-critical one is ignored.
        if let Some(ctrls) = c.get(2) {
            if ctrls.tag != 0xa0 {
                return Err(LdapError::Malformed);
            }
            if has_critical(ctrls)? {
                let resp = match op.tag {
                    0x60 => Some(0x61),
                    0x63 => Some(0x65),
                    0x66 => Some(0x67),
                    0x68 => Some(0x69),
                    0x4a => Some(0x6b),
                    0x6c => Some(0x6d),
                    0x6e => Some(0x6f),
                    0x77 => Some(0x78),
                    0x42 | 0x50 => None,
                    _ => return Err(LdapError::Malformed),
                };
                if op.tag == 0x42 {
                    s.closed = true;
                }
                return Ok(resp
                    .map(|r| {
                        result(
                            id,
                            r,
                            ResultCode::UnavailableCriticalExtension,
                            "critical control not supported",
                        )
                    })
                    .into_iter()
                    .collect());
            }
        }
        Ok(match op.tag {
            0x60 => vec![self.bind(s, id, &op)?],
            0x42 => {
                s.closed = true;
                Vec::new()
            }
            0x50 => Vec::new(), // abandon: every search here completes at once
            0x63 => self.search(s, id, &op)?,
            0x66 => vec![result(id, 0x67, ResultCode::UnwillingToPerform, READ_ONLY)],
            0x68 => vec![result(id, 0x69, ResultCode::UnwillingToPerform, READ_ONLY)],
            0x4a => vec![result(id, 0x6b, ResultCode::UnwillingToPerform, READ_ONLY)],
            0x6c => vec![result(id, 0x6d, ResultCode::UnwillingToPerform, READ_ONLY)],
            0x6e => vec![result(
                id,
                0x6f,
                ResultCode::UnwillingToPerform,
                "compare is not supported",
            )],
            0x77 => vec![result(
                id,
                0x78,
                ResultCode::UnwillingToPerform,
                "no extended operations",
            )],
            _ => return Err(LdapError::Malformed),
        })
    }

    fn bind(&self, s: &mut Session<P::Actor>, id: i64, op: &Tlv<'_>) -> Result<Vec<u8>, LdapError> {
        let c = children(op.value)?;
        if c.len() != 3 || c[0].tag != 0x02 || c[1].tag != 0x04 {
            return Err(LdapError::Malformed);
        }
        // A bind always starts from anonymous (RFC 4513 §5.1).
        s.actor = None;
        if int(&c[0])? != 3 {
            return Ok(result(id, 0x61, ResultCode::ProtocolError, "LDAPv3 only"));
        }
        if c[2].tag != 0x80 {
            return Ok(result(
                id,
                0x61,
                ResultCode::AuthMethodNotSupported,
                "simple bind only",
            ));
        }
        let name = text(&c[1])?;
        let password = c[2].value;
        Ok(match (name.is_empty(), password.is_empty()) {
            (true, true) => result(id, 0x61, ResultCode::Success, ""),
            // Unauthenticated bind (RFC 4513 §5.1.2): refused.
            (false, true) => result(
                id,
                0x61,
                ResultCode::UnwillingToPerform,
                "unauthenticated bind",
            ),
            (true, false) => result(id, 0x61, ResultCode::InvalidCredentials, ""),
            (false, false) => match self.authority.bind(&name, password) {
                Some(a) => {
                    s.actor = Some(a);
                    result(id, 0x61, ResultCode::Success, "")
                }
                None => result(id, 0x61, ResultCode::InvalidCredentials, ""),
            },
        })
    }

    fn search(
        &self,
        s: &Session<P::Actor>,
        id: i64,
        op: &Tlv<'_>,
    ) -> Result<Vec<Vec<u8>>, LdapError> {
        let c = children(op.value)?;
        if c.len() != 8 || c[0].tag != 0x04 || c[1].tag != 0x0a || c[7].tag != 0x30 {
            return Err(LdapError::Malformed);
        }
        let base = text(&c[0])?;
        let scope = int(&c[1])?;
        let size_limit = int(&c[3])?;
        let time_limit = int(&c[4])?;
        let deadline = (time_limit > 0)
            .then(|| std::time::Instant::now() + std::time::Duration::from_secs(time_limit as u64));
        let types_only = c[5].value.first().is_some_and(|&b| b != 0);
        let filter = parse_filter(&c[6])?;
        let requested: Vec<String> = children(c[7].value)?
            .iter()
            .map(|t| text(t).map(|s| s.to_lowercase()))
            .collect::<Result<_, _>>()?;
        if !(0..=2).contains(&scope) {
            return Err(LdapError::Malformed);
        }
        let done = |code, msg: &str| result(id, 0x65, code, msg);

        // The rootDSE: readable by anyone.
        if base.is_empty() && scope == 0 {
            let dse = self.dir.root_dse();
            let mut out = Vec::new();
            if eval(&filter, &dse, &|_| true) == Some(true) {
                out.push(self.entry(id, &dse, &requested, types_only, &|_| true));
            }
            out.push(done(ResultCode::Success, ""));
            return Ok(out);
        }
        let Some(actor) = s.actor.as_ref() else {
            return Ok(vec![done(
                ResultCode::OperationsError,
                "a successful bind is required",
            )]);
        };
        let Some(base_key) = key(&base) else {
            return Ok(vec![done(ResultCode::NoSuchObject, "")]);
        };
        // The base must exist and be visible, or it is reported missing.
        let base_ok = self
            .dir
            .all()
            .any(|(k, e)| *k == base_key && self.authority.visible(actor, e));
        if !base_ok {
            return Ok(vec![done(ResultCode::NoSuchObject, "")]);
        }
        let in_scope = |k: &Key| match scope {
            0 => *k == base_key,
            1 => k.len() == base_key.len() + 1 && k[1..] == base_key[..],
            _ => k.len() >= base_key.len() && k[k.len() - base_key.len()..] == base_key[..],
        };
        let mut out = Vec::new();
        let mut sent = 0i64;
        for (k, e) in self.dir.all() {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                out.push(done(ResultCode::TimeLimitExceeded, ""));
                return Ok(out);
            }
            if !in_scope(k) || !self.authority.visible(actor, e) {
                continue;
            }
            let readable = |a: &str| self.authority.readable(actor, e, a);
            if eval(&filter, e, &readable) != Some(true) {
                continue;
            }
            if size_limit > 0 && sent == size_limit {
                out.push(done(ResultCode::SizeLimitExceeded, ""));
                return Ok(out);
            }
            out.push(self.entry(id, e, &requested, types_only, &readable));
            sent += 1;
        }
        out.push(done(ResultCode::Success, ""));
        Ok(out)
    }

    fn entry(
        &self,
        id: i64,
        e: &Entry,
        requested: &[String],
        types_only: bool,
        readable: &dyn Fn(&str) -> bool,
    ) -> Vec<u8> {
        let all = requested.is_empty() || requested.iter().any(|r| r == "*");
        let none = requested.len() == 1 && requested[0] == "1.1";
        let mut names: Vec<String> = Vec::new();
        for (n, _) in &e.attrs {
            let l = n.to_lowercase();
            if !names.contains(&l) {
                names.push(l);
            }
        }
        if requested.iter().any(|r| r == "distinguishedname") && !e.dn.is_empty() {
            names.push("distinguishedname".into());
        }
        let mut attrs = Vec::new();
        for l in names {
            let wanted = !none && (all || requested.contains(&l));
            if !wanted || !readable(&l) {
                continue;
            }
            let shown = e
                .attrs
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(&l))
                .map_or("distinguishedName", |(n, _)| *n);
            let mut vals = Vec::new();
            if !types_only {
                for v in values(e, &l) {
                    vals.extend(tlv(0x04, &v));
                }
            }
            let mut a = tlv(0x04, shown.as_bytes());
            a.extend(tlv(0x31, &vals));
            attrs.extend(tlv(0x30, &a));
        }
        let mut body = tlv(0x04, e.dn.as_bytes());
        body.extend(tlv(0x30, &attrs));
        message(id, &tlv(0x64, &body))
    }
}
