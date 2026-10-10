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
//! attribute is omitted. A value that names another entry (`member`) is
//! shown, and matched by a filter, only when that entry is visible too, so
//! neither a group's member list nor a `member=` filter reveals a hidden
//! entry. The check runs before an entry is encoded.

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

/// The TLVs in `b`, decoded one at a time.
fn each_child(b: &[u8]) -> impl Iterator<Item = Result<Tlv<'_>, LdapError>> {
    let mut rest = b;
    std::iter::from_fn(move || {
        if rest.is_empty() {
            return None;
        }
        Some(read_tlv(rest).map(|(t, r)| {
            rest = r;
            t
        }))
    })
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

/// An OCTET STRING (universal tag 4); any other tag is malformed.
fn octets<'a>(t: &Tlv<'a>) -> Result<&'a [u8], LdapError> {
    if t.tag != 0x04 {
        return Err(LdapError::Malformed);
    }
    Ok(t.value)
}

/// An LDAPString: an OCTET STRING holding UTF-8.
fn string(t: &Tlv<'_>) -> Result<String, LdapError> {
    String::from_utf8(octets(t)?.to_vec()).map_err(|_| LdapError::Malformed)
}

/// A BOOLEAN: universal tag 1 with exactly one content octet.
fn boolean(t: &Tlv<'_>) -> Result<bool, LdapError> {
    if t.tag != 0x01 || t.value.len() != 1 {
        return Err(LdapError::Malformed);
    }
    Ok(t.value[0] != 0)
}

/// The content of an implicitly tagged string, read as UTF-8; the caller
/// has already matched the tag.
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
    /// adminLimitExceeded: a server-side cap was reached
    AdminLimitExceeded = 11,
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
    /// Entry position by key, for resolving DN-valued attributes.
    index: std::collections::HashMap<Key, usize>,
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
        let index = entries
            .iter()
            .enumerate()
            .map(|(i, (k, _))| (k.clone(), i))
            .collect();
        Some(Self {
            naming_context: naming_context.to_string(),
            nc_key,
            nc_entry,
            entries,
            index,
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

    /// The entry named by `dn`, if it is served.
    fn resolve(&self, dn: &[u8]) -> Option<&Entry> {
        let k = key(std::str::from_utf8(dn).ok()?)?;
        if k == self.nc_key {
            return Some(&self.nc_entry);
        }
        self.index.get(&k).map(|&i| &self.entries[i].1)
    }

    fn all(&self) -> impl Iterator<Item = (&Key, &Entry)> {
        std::iter::once((&self.nc_key, &self.nc_entry))
            .chain(self.entries.iter().map(|(k, e)| (k, e)))
    }
}

/// A decoded SearchRequest (RFC 4511 §4.5.1).
struct SearchRequest {
    base: String,
    scope: i64,
    size_limit: i64,
    time_limit: i64,
    types_only: bool,
    filter: Filter,
    /// Lower-cased, sorted and deduplicated.
    requested: Vec<String>,
    /// More than [`MAX_ATTRIBUTE_SELECTORS`] distinct selectors were named;
    /// `requested` then holds only the first ones and must not be used.
    too_many_selectors: bool,
}

fn parse_search(op: &Tlv<'_>) -> Result<SearchRequest, LdapError> {
    let c = children(op.value)?;
    // RFC 4511 §4.5.1: baseObject OCTET STRING, scope ENUMERATED,
    // derefAliases ENUMERATED (0..=3), sizeLimit and timeLimit INTEGER,
    // typesOnly BOOLEAN, filter, attributes SEQUENCE.
    if c.len() != 8
        || c[0].tag != 0x04
        || c[1].tag != 0x0a
        || c[2].tag != 0x0a
        || c[3].tag != 0x02
        || c[4].tag != 0x02
        || c[7].tag != 0x30
        || !(0..=3).contains(&int(&c[2])?)
    {
        return Err(LdapError::Malformed);
    }
    let base = string(&c[0])?;
    let scope = int(&c[1])?;
    let size_limit = int(&c[3])?;
    let time_limit = int(&c[4])?;
    // sizeLimit and timeLimit are INTEGER (0 .. maxInt).
    let max_int = 0..=i64::from(i32::MAX);
    if !max_int.contains(&size_limit) || !max_int.contains(&time_limit) {
        return Err(LdapError::Malformed);
    }
    let types_only = boolean(&c[5])?;
    let filter = parse_filter(&c[6])?;
    // Decoded one selector at a time and deduplicated as they arrive, so a
    // repeated selector allocates nothing; past MAX_ATTRIBUTE_SELECTORS
    // distinct ones the rest are only shape-checked.
    let mut set = std::collections::BTreeSet::new();
    let mut too_many_selectors = false;
    let mut buf = String::new();
    for t in each_child(c[7].value) {
        let name = std::str::from_utf8(octets(&t?)?).map_err(|_| LdapError::Malformed)?;
        if too_many_selectors {
            continue;
        }
        buf.clear();
        buf.push_str(name);
        buf.make_ascii_lowercase();
        if !set.contains(buf.as_str()) {
            if set.len() == MAX_ATTRIBUTE_SELECTORS {
                too_many_selectors = true;
                continue;
            }
            set.insert(buf.clone());
        }
    }
    // Sorted and deduplicated: looked up by binary search per attribute.
    let requested: Vec<String> = set.into_iter().collect();
    if !(0..=2).contains(&scope) {
        return Err(LdapError::Malformed);
    }
    Ok(SearchRequest {
        base,
        scope,
        size_limit,
        time_limit,
        types_only,
        filter,
        requested,
        too_many_selectors,
    })
}

/// A PartialAttribute: `type` and a SET OF values, non-empty if `values`.
fn check_partial_attribute(t: &Tlv<'_>, values: bool) -> Result<(), LdapError> {
    let c = children(t.value)?;
    if t.tag != 0x30 || c.len() != 2 || c[1].tag != 0x31 {
        return Err(LdapError::Malformed);
    }
    string(&c[0])?;
    let vals = children(c[1].value)?;
    if values && vals.is_empty() {
        return Err(LdapError::Malformed);
    }
    vals.iter().try_for_each(|v| octets(v).map(drop))
}

/// Check a request's shape (RFC 4511 §4.2 - §4.12) before anything is
/// answered, so a malformed request disconnects even when the operation is
/// refused. A SearchRequest is checked by [`parse_search`].
fn check_request(op: &Tlv<'_>) -> Result<(), LdapError> {
    let bad = Err(LdapError::Malformed);
    match op.tag {
        // BindRequest: version, name, authentication.
        0x60 => {
            let c = children(op.value)?;
            if c.len() != 3 || c[0].tag != 0x02 {
                return bad;
            }
            int(&c[0])?;
            string(&c[1])?;
            // SaslCredentials ::= SEQUENCE { mechanism, credentials OPTIONAL }
            if c[2].tag == 0xa3 {
                let m = children(c[2].value)?;
                if !(1..=2).contains(&m.len()) {
                    return bad;
                }
                m.iter().try_for_each(|x| octets(x).map(drop))?;
            }
        }
        // UnbindRequest ::= [APPLICATION 2] NULL
        0x42 if !op.value.is_empty() => return bad,
        0x42 => {}
        // AbandonRequest ::= [APPLICATION 16] MessageID
        0x50 => {
            if !(0..=i64::from(i32::MAX)).contains(&int(op)?) {
                return bad;
            }
        }
        // DelRequest ::= [APPLICATION 10] LDAPDN
        0x4a => {
            text(op)?;
        }
        // ModifyRequest: object, changes SEQUENCE OF { operation, PartialAttribute }
        0x66 => {
            let c = children(op.value)?;
            if c.len() != 2 || c[1].tag != 0x30 {
                return bad;
            }
            string(&c[0])?;
            for ch in children(c[1].value)? {
                let m = children(ch.value)?;
                if ch.tag != 0x30 || m.len() != 2 || m[0].tag != 0x0a {
                    return bad;
                }
                if !(0..=3).contains(&int(&m[0])?) {
                    return bad;
                }
                check_partial_attribute(&m[1], false)?;
            }
        }
        // AddRequest: entry, attributes SEQUENCE OF Attribute (values 1..)
        0x68 => {
            let c = children(op.value)?;
            if c.len() != 2 || c[1].tag != 0x30 {
                return bad;
            }
            string(&c[0])?;
            for a in children(c[1].value)? {
                check_partial_attribute(&a, true)?;
            }
        }
        // ModifyDNRequest: entry, newrdn, deleteoldrdn, newSuperior [0] OPTIONAL
        0x6c => {
            let c = children(op.value)?;
            if !(3..=4).contains(&c.len()) {
                return bad;
            }
            string(&c[0])?;
            string(&c[1])?;
            boolean(&c[2])?;
            if let Some(sup) = c.get(3) {
                if sup.tag != 0x80 {
                    return bad;
                }
                text(sup)?;
            }
        }
        // CompareRequest: entry, ava { attributeDesc, assertionValue }
        0x6e => {
            let c = children(op.value)?;
            if c.len() != 2 || c[1].tag != 0x30 {
                return bad;
            }
            string(&c[0])?;
            let ava = children(c[1].value)?;
            if ava.len() != 2 {
                return bad;
            }
            string(&ava[0])?;
            octets(&ava[1])?;
        }
        // ExtendedRequest: requestName [0], requestValue [1] OPTIONAL
        0x77 => {
            let c = children(op.value)?;
            if c.is_empty() || c.len() > 2 || c[0].tag != 0x80 {
                return bad;
            }
            text(&c[0])?;
            if c.get(1).is_some_and(|v| v.tag != 0x81) {
                return bad;
            }
        }
        0x63 => {
            parse_search(op)?;
        }
        _ => return bad,
    }
    Ok(())
}

/// The handler: a directory and the authority deciding access.
pub struct Server<'a, P: Authority> {
    dir: &'a Directory,
    authority: &'a P,
    max_entries: usize,
    max_bytes: usize,
}

/// Distinct attribute selectors one search may name.
pub const MAX_ATTRIBUTE_SELECTORS: usize = 256;

/// Entries one search returns at most, whatever the client asks: Active
/// Directory's default MaxPageSize.
pub const DEFAULT_MAX_ENTRIES: usize = 1000;
/// Encoded entry bytes one search returns at most.
pub const DEFAULT_MAX_BYTES: usize = 16 << 20;

/// Attribute types the rootDSE and the naming-context entry carry, beyond
/// the projection schema in [`crate::ad::ATTRIBUTES`].
const SERVED_ATTRIBUTES: &[&str] = &[
    "distinguishedName",
    "dc",
    "namingContexts",
    "defaultNamingContext",
    "supportedLDAPVersion",
    "vendorName",
    "isReadOnly",
];

/// Whether the server recognises `attr` (lower-cased).
fn is_known(attr: &str) -> bool {
    crate::ad::ATTRIBUTES
        .iter()
        .chain(SERVED_ATTRIBUTES)
        .any(|a| a.eq_ignore_ascii_case(attr))
}

/// DN-valued attributes: matched as parsed distinguished names
/// (distinguishedNameMatch), with no ordering or substring rule.
fn is_dn(attr: &str) -> bool {
    matches!(
        attr,
        "distinguishedname" | "member" | "namingcontexts" | "defaultnamingcontext"
    )
}

/// The matching form of a DN: its parsed RDNs, case-folded. `None` if `v`
/// is not a DN.
fn dn_form(v: &[u8]) -> Option<Vec<u8>> {
    let k = key(std::str::from_utf8(v).ok()?)?;
    let mut out = Vec::new();
    for (a, v) in k {
        out.extend(a.as_bytes());
        out.push(0);
        out.extend(v.as_bytes());
        out.push(1);
    }
    Some(out)
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

/// The most filter nodes (every `and`, `or`, `not` and leaf counts) one
/// search may carry. Evaluation visits each node once per candidate entry,
/// so this bounds the work a single request can ask for.
pub const MAX_FILTER_NODES: usize = 256;

fn parse_filter(t: &Tlv<'_>) -> Result<Filter, LdapError> {
    let mut budget = MAX_FILTER_NODES;
    parse_filter_at(t, 0, &mut budget)
}

fn parse_filter_at(t: &Tlv<'_>, depth: usize, budget: &mut usize) -> Result<Filter, LdapError> {
    if depth >= MAX_FILTER_DEPTH || *budget == 0 {
        return Err(LdapError::Malformed);
    }
    *budget -= 1;
    let mut sub = |t: &Tlv<'_>| parse_filter_at(t, depth + 1, budget);
    let pair = |t: &Tlv<'_>| -> Result<(String, Vec<u8>), LdapError> {
        let c = children(t.value)?;
        if c.len() != 2 {
            return Err(LdapError::Malformed);
        }
        Ok((string(&c[0])?.to_lowercase(), octets(&c[1])?.to_vec()))
    };
    Ok(match t.tag {
        // and / or are SET SIZE (1..MAX) OF Filter (RFC 4511 §4.5.1).
        // Children are decoded one at a time, each against the budget, so a
        // wide set is refused before it is held in memory.
        0xa0 | 0xa1 => {
            let mut fs = Vec::new();
            for c in each_child(t.value) {
                fs.push(sub(&c?)?);
            }
            if fs.is_empty() {
                return Err(LdapError::Malformed);
            }
            if t.tag == 0xa0 {
                Filter::And(fs)
            } else {
                Filter::Or(fs)
            }
        }
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
            let attr = string(&c[0])?.to_lowercase();
            // substrings SEQUENCE SIZE (1..MAX), decoded one at a time.
            let (mut initial, mut any, mut last) = (None, Vec::new(), None);
            let mut parts = 0usize;
            for s in each_child(c[1].value) {
                let s = s?;
                parts += 1;
                // Each substring item counts against the filter budget.
                if *budget == 0 {
                    return Err(LdapError::Malformed);
                }
                *budget -= 1;
                match s.tag {
                    0x80 if initial.is_none() && any.is_empty() && last.is_none() => {
                        initial = Some(s.value.to_vec())
                    }
                    0x81 if last.is_none() => any.push(s.value.to_vec()),
                    0x82 if last.is_none() => last = Some(s.value.to_vec()),
                    _ => return Err(LdapError::Malformed),
                }
            }
            if parts == 0 {
                return Err(LdapError::Malformed);
            }
            Filter::Sub {
                attr,
                initial,
                any,
                last,
            }
        }
        // extensibleMatch: decoded for shape, never evaluated (Undefined).
        0xa9 => {
            // MatchingRuleAssertion ::= SEQUENCE { matchingRule [1]
            // OPTIONAL, type [2] OPTIONAL, matchValue [3], dnAttributes [4]
            // BOOLEAN DEFAULT FALSE }, fields in order, a rule or a type.
            let mut last = 0x80;
            let mut seen = [false; 5];
            for f in children(t.value)? {
                if !(0x81..=0x84).contains(&f.tag) || f.tag <= last {
                    return Err(LdapError::Malformed);
                }
                if f.tag == 0x84 && f.value.len() != 1 {
                    return Err(LdapError::Malformed);
                }
                // matchingRule (MatchingRuleId) and type (AttributeDescription)
                // are LDAPStrings: UTF-8, checked like every other one.
                if matches!(f.tag, 0x81 | 0x82) {
                    text(&f)?;
                }
                last = f.tag;
                seen[usize::from(f.tag - 0x80)] = true;
            }
            if !seen[3] || !(seen[1] || seen[2]) {
                return Err(LdapError::Malformed);
            }
            Filter::Unsupported
        }
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
    if is_binary(attr) || is_dn(attr) {
        return None;
    }
    if is_integer(attr) {
        let v = rfc_integer(v)?;
        return Some(
            vals.iter()
                .filter_map(|x| rfc_integer(x))
                .any(|x| keep(int_cmp(x, v))),
        );
    }
    let v = fold(attr, v)?;
    Some(vals.iter().any(|x| keep(x.as_slice().cmp(v.as_slice()))))
}

/// An RFC 4517 §3.3.16 Integer: `-`? digits, no leading zero, no `-0`;
/// unbounded in magnitude. Returns (negative, digits).
fn rfc_integer(b: &[u8]) -> Option<(bool, &[u8])> {
    let (neg, d) = match b {
        [b'-', rest @ ..] => (true, rest),
        _ => (false, b),
    };
    let ok = !d.is_empty()
        && d.iter().all(u8::is_ascii_digit)
        && (d.len() == 1 || d[0] != b'0')
        && !(neg && d == b"0");
    ok.then_some((neg, d))
}

/// integerOrderingMatch over [`rfc_integer`] values, at any magnitude.
fn int_cmp(a: (bool, &[u8]), b: (bool, &[u8])) -> std::cmp::Ordering {
    let mag = |x: &[u8], y: &[u8]| x.len().cmp(&y.len()).then_with(|| x.cmp(y));
    match (a.0, b.0) {
        (false, true) => std::cmp::Ordering::Greater,
        (true, false) => std::cmp::Ordering::Less,
        (false, false) => mag(a.1, b.1),
        (true, true) => mag(b.1, a.1),
    }
}

fn is_binary(attr: &str) -> bool {
    matches!(attr, "objectguid" | "msexchmailboxguid")
}

/// The matching form of a value or an equality / ordering assertion, or
/// `None` when it has none (a DN that does not parse, text that is not
/// UTF-8 or contains a prohibited character). A `None` assertion makes the
/// item Undefined; a `None` value matches nothing.
fn fold(attr: &str, v: &[u8]) -> Option<Vec<u8>> {
    if is_dn(attr) {
        return dn_form(v);
    }
    // Binary values compare as stored; integers are parsed by
    // [`rfc_integer`], which rejects anything but the canonical form.
    if is_binary(attr) || is_integer(attr) {
        return Some(v.to_vec());
    }
    prep(v, Piece::Whole)
}

/// Which part of an assertion a string is, for RFC 4518 §2.6.1
/// insignificant-space handling.
#[derive(Clone, Copy, PartialEq)]
enum Piece {
    /// An attribute value or an equality / ordering assertion.
    Whole,
    Initial,
    Any,
    Final,
}

/// RFC 4518 string preparation for caseIgnoreMatch and its ordering and
/// substrings rules: transcode (UTF-8 or nothing), map (map-to-nothing
/// characters dropped, every space separator to SPACE, case folded),
/// normalize (NFKC), prohibit (unassigned-free check reduced to the
/// prohibited ranges below), then insignificant spaces.
fn prep(v: &[u8], piece: Piece) -> Option<Vec<u8>> {
    use unicode_normalization::UnicodeNormalization;
    let s = std::str::from_utf8(v).ok()?;
    let mut mapped = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            // §2.2 mapped to nothing: soft hyphen, combining grapheme
            // joiner, Mongolian free variation selectors, variation
            // selectors, object replacement, zero-width space, and controls
            // other than the spaces mapped below.
            '\u{00AD}'
            | '\u{034F}'
            | '\u{1806}'
            | '\u{180B}'..='\u{180E}'
            | '\u{FE00}'..='\u{FE0F}'
            | '\u{FFFC}'
            | '\u{200B}' => {}
            '\u{0009}'..='\u{000D}' | '\u{0085}' => mapped.push(' '),
            c if c.is_control() => {}
            // §2.4 prohibited: private use, non-characters, the
            // replacement character.
            '\u{E000}'..='\u{F8FF}' | '\u{FDD0}'..='\u{FDEF}' | '\u{FFFD}' => return None,
            c if (c as u32 & 0xFFFE) == 0xFFFE => return None,
            c if c.is_whitespace() => mapped.push(' '),
            c => mapped.push(c),
        }
    }
    // Full case folding (RFC 3454 B.2) and NFKC, applied twice: B.2 is
    // the folding closed under NFKC, which one fold-then-normalize pass
    // does not reach for every character (e.g. a compatibility form whose
    // decomposition has upper case).
    let once = |x: &str| -> String { unicase::UniCase::new(x).to_folded_case().nfkc().collect() };
    let norm = once(&once(&mapped));
    let words: Vec<&str> = norm.split(' ').filter(|w| !w.is_empty()).collect();
    let mut out = String::with_capacity(norm.len() + 2);
    if piece == Piece::Whole {
        // §2.6.1: one SPACE each side, interior runs to two, empty to two.
        if words.is_empty() {
            return Some(b"  ".to_vec());
        }
        out.push(' ');
        out.push_str(&words.join("  "));
        out.push(' ');
        return Some(out.into_bytes());
    }
    if words.is_empty() {
        return Some(b" ".to_vec());
    }
    if piece == Piece::Initial || norm.starts_with(' ') {
        out.push(' ');
    }
    out.push_str(&words.join("  "));
    if piece == Piece::Final || norm.ends_with(' ') {
        out.push(' ');
    }
    Some(out.into_bytes())
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

/// What one reader may see of one entry.
struct Access<'x> {
    /// Whether an attribute (lower-cased) may be read.
    readable: &'x dyn Fn(&str) -> bool,
    /// Whether one value of an attribute may be shown: false for a
    /// reference to an entry the reader cannot see.
    shows: &'x dyn Fn(&str, &[u8]) -> bool,
}

/// Attributes whose values name other entries.
fn is_reference(attr: &str) -> bool {
    attr == "member"
}

/// Three-valued filter evaluation (RFC 4511 §4.5.1.7): `None` = Undefined.
fn eval(f: &Filter, e: &Entry, acc: &Access<'_>) -> Option<bool> {
    let vals = |a: &str| -> Option<Vec<Vec<u8>>> {
        // An unrecognised attribute type makes the item Undefined; a
        // recognised one that is unreadable or absent has no values.
        if !is_known(a) {
            return None;
        }
        if !(acc.readable)(a) {
            return Some(Vec::new());
        }
        Some(
            values(e, a)
                .into_iter()
                .filter(|v| (acc.shows)(a, v))
                .filter_map(|v| fold(a, &v))
                .collect(),
        )
    };
    match f {
        Filter::And(fs) => {
            let mut out = Some(true);
            for f in fs {
                match eval(f, e, acc) {
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
                match eval(f, e, acc) {
                    Some(true) => return Some(true),
                    None => out = None,
                    Some(false) => {}
                }
            }
            out
        }
        Filter::Not(f) => eval(f, e, acc).map(|b| !b),
        // present is FALSE, never Undefined, for an unrecognised type.
        Filter::Present(a) => Some(!vals(a).unwrap_or_default().is_empty()),
        // Integer equality is numeric, and Undefined for a non-integer
        // assertion (RFC 4517 integerMatch).
        Filter::Eq(a, v) if is_integer(a) => order(a, v, vals(a)?, |o| o.is_eq()),
        Filter::Eq(a, v) if is_dn(a) => {
            let vals = vals(a)?;
            Some(vals.contains(&dn_form(v)?))
        }
        Filter::Eq(a, v) => {
            let v = fold(a, v)?;
            Some(vals(a)?.contains(&v))
        }
        // distinguishedNameMatch, integerMatch and the binary GUIDs have no
        // substrings rule: the item is Undefined.
        Filter::Sub { attr, .. } if is_dn(attr) || is_integer(attr) || is_binary(attr) => {
            vals(attr)?;
            None
        }
        Filter::Ge(a, v) => order(a, v, vals(a)?, |o| o.is_ge()),
        Filter::Le(a, v) => order(a, v, vals(a)?, |o| o.is_le()),
        Filter::Sub {
            attr,
            initial,
            any,
            last,
        } => {
            // A piece with no matching form makes the item Undefined.
            let initial = match initial {
                Some(x) => Some(prep(x, Piece::Initial)?),
                None => None,
            };
            let any = any
                .iter()
                .map(|x| prep(x, Piece::Any))
                .collect::<Option<Vec<_>>>()?;
            let last = match last {
                Some(x) => Some(prep(x, Piece::Final)?),
                None => None,
            };
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
    // Every control is checked, not only those before the first critical one.
    let mut critical = false;
    for ctrl in children(ctrls.value)? {
        // Control ::= SEQUENCE { controlType LDAPOID, criticality BOOLEAN
        // DEFAULT FALSE, controlValue OCTET STRING OPTIONAL }
        if ctrl.tag != 0x30 {
            return Err(LdapError::Malformed);
        }
        let f = children(ctrl.value)?;
        let Some((oid, mut rest)) = f.split_first() else {
            return Err(LdapError::Malformed);
        };
        octets(oid)?;
        if let Some((b, r)) = rest.split_first().filter(|(b, _)| b.tag == 0x01) {
            critical |= boolean(b)?;
            rest = r;
        }
        match rest {
            [] => {}
            [v] => {
                octets(v)?;
            }
            _ => return Err(LdapError::Malformed),
        }
    }
    Ok(critical)
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
        Self {
            dir,
            authority,
            max_entries: DEFAULT_MAX_ENTRIES,
            max_bytes: DEFAULT_MAX_BYTES,
        }
    }

    /// Cap what one search returns, whatever its sizeLimit: at most
    /// `max_entries` entries and `max_bytes` encoded entry bytes. Reaching
    /// either ends the search with adminLimitExceeded.
    pub fn with_limits(mut self, max_entries: usize, max_bytes: usize) -> Self {
        self.max_entries = max_entries;
        self.max_bytes = max_bytes;
        self
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
        // MessageID 0 is reserved for unsolicited notifications (RFC 4511
        // §4.1.1.1): a request carries 1 .. maxInt.
        if !(1..=i64::from(i32::MAX)).contains(&id) {
            return Err(LdapError::Malformed);
        }
        let op = c[1];
        // Search parses itself when it runs; everything else, and a search
        // refused for a control, is shape-checked first.
        if op.tag != 0x63 {
            check_request(&op)?;
        }
        // No control is supported: a critical one refuses the operation
        // (RFC 4511 §4.1.11); a non-critical one is ignored.
        if let Some(ctrls) = c.get(2) {
            if ctrls.tag != 0xa0 {
                return Err(LdapError::Malformed);
            }
            if has_critical(ctrls)? {
                if op.tag == 0x63 {
                    parse_search(&op)?;
                }
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
                match op.tag {
                    0x42 => s.closed = true,
                    // Any bind, accepted or not, starts from anonymous.
                    0x60 => s.actor = None,
                    _ => {}
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
        let name = string(&c[1])?;
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
        let SearchRequest {
            base,
            scope,
            size_limit,
            time_limit,
            types_only,
            filter,
            requested,
            too_many_selectors,
        } = parse_search(op)?;
        // A deadline past what the clock can represent is no deadline.
        let deadline = (time_limit > 0)
            .then(|| {
                std::time::Instant::now()
                    .checked_add(std::time::Duration::from_secs(time_limit as u64))
            })
            .flatten();
        let done = |code, msg: &str| result(id, 0x65, code, msg);
        if too_many_selectors {
            return Ok(vec![done(
                ResultCode::AdminLimitExceeded,
                "too many attributes requested",
            )]);
        }

        // The rootDSE: readable by anyone.
        if base.is_empty() && scope == 0 {
            let dse = self.dir.root_dse();
            let acc = Access {
                readable: &|_| true,
                shows: &|_, _| true,
            };
            let mut out = Vec::new();
            if eval(&filter, &dse, &acc) == Some(true) {
                // The server caps hold for the rootDSE too.
                if self.max_entries == 0 {
                    out.push(done(ResultCode::AdminLimitExceeded, ""));
                    return Ok(out);
                }
                let pdu = self.entry(id, &dse, &requested, types_only, &acc);
                if pdu.len() > self.max_bytes {
                    out.push(done(ResultCode::AdminLimitExceeded, ""));
                    return Ok(out);
                }
                out.push(pdu);
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
        // Resolving the base counts against the time limit too.
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            return Ok(vec![done(ResultCode::TimeLimitExceeded, "")]);
        }
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
        let mut bytes = 0usize;
        for (k, e) in self.dir.all() {
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                out.push(done(ResultCode::TimeLimitExceeded, ""));
                return Ok(out);
            }
            if !in_scope(k) || !self.authority.visible(actor, e) {
                continue;
            }
            let readable = |a: &str| self.authority.readable(actor, e, a);
            let shows = |a: &str, v: &[u8]| {
                !is_reference(a)
                    || self
                        .dir
                        .resolve(v)
                        .is_some_and(|t| self.authority.visible(actor, t))
            };
            let acc = Access {
                readable: &readable,
                shows: &shows,
            };
            if eval(&filter, e, &acc) != Some(true) {
                continue;
            }
            if size_limit > 0 && sent == size_limit {
                out.push(done(ResultCode::SizeLimitExceeded, ""));
                return Ok(out);
            }
            // The entry cap is checked before the entry is built.
            if sent as usize >= self.max_entries {
                out.push(done(ResultCode::AdminLimitExceeded, ""));
                return Ok(out);
            }
            let pdu = self.entry(id, e, &requested, types_only, &acc);
            // Checking and projecting this candidate can outlast the limit:
            // nothing found after it is returned.
            if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
                out.push(done(ResultCode::TimeLimitExceeded, ""));
                return Ok(out);
            }
            if bytes + pdu.len() > self.max_bytes {
                out.push(done(ResultCode::AdminLimitExceeded, ""));
                return Ok(out);
            }
            bytes += pdu.len();
            out.push(pdu);
            sent += 1;
        }
        // The last candidate's checks can outlast the limit too.
        if deadline.is_some_and(|d| std::time::Instant::now() >= d) {
            out.push(done(ResultCode::TimeLimitExceeded, ""));
            return Ok(out);
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
        acc: &Access<'_>,
    ) -> Vec<u8> {
        let has = |r: &str| requested.binary_search_by(|x| x.as_str().cmp(r)).is_ok();
        let all = requested.is_empty() || has("*");
        let none = requested.len() == 1 && requested[0] == "1.1";
        let mut names: Vec<String> = Vec::new();
        for (n, _) in &e.attrs {
            let l = n.to_lowercase();
            if !names.contains(&l) {
                names.push(l);
            }
        }
        if has("distinguishedname") && !e.dn.is_empty() {
            names.push("distinguishedname".into());
        }
        let mut attrs = Vec::new();
        for l in names {
            let wanted = !none && (all || has(&l));
            if !wanted || !(acc.readable)(&l) {
                continue;
            }
            let shown = e
                .attrs
                .iter()
                .find(|(n, _)| n.eq_ignore_ascii_case(&l))
                .map_or("distinguishedName", |(n, _)| *n);
            let visible: Vec<Vec<u8>> = values(e, &l)
                .into_iter()
                .filter(|v| (acc.shows)(&l, v))
                .collect();
            // An attribute whose every value is hidden is omitted, even in
            // a types-only result: its presence alone would reveal them.
            if visible.is_empty() {
                continue;
            }
            let mut vals = Vec::new();
            if !types_only {
                for v in &visible {
                    vals.extend(tlv(0x04, v));
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

#[cfg(test)]
mod tests {
    use super::{prep, Piece};

    // RFC 3454 B.2 is full case folding: ß and ẞ fold to "ss", final sigma
    // to sigma. Lower-casing alone keeps them apart.
    #[test]
    fn preparation_uses_full_case_folding() {
        let p = |s: &str| prep(s.as_bytes(), Piece::Whole);
        assert_eq!(p("Straße"), p("STRASSE"));
        assert_eq!(p("\u{1E9E}"), p("ss"));
        assert_eq!(p("ΟΔΟΣ"), p("οδος"));
        assert_eq!(p("οδος"), p("οδοσ"));
        // Folding is not a wildcard.
        assert_ne!(p("Straße"), p("STRASE"));
        let f = |s: &str| prep(s.as_bytes(), Piece::Final);
        assert_eq!(f("STRASSE"), f("straße"));
    }
}
