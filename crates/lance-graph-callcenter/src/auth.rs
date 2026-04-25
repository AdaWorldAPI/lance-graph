//! DM-7 — JWT extraction middleware + `ActorContext` population.
//!
//! **Phase 1 (this file):** Extract and decode JWT payload (base64),
//! populate `ActorContext`. No signature verification — that requires
//! a JWK endpoint or static key, which is deployment-specific.
//!
//! **Phase 2 (future):** Plug in real verification via a `JwkSetProvider`
//! trait. The `JwtMiddleware::extract_actor` API won't change — only the
//! internal verification step gets wired.
//!
//! # JWT Payload Shape (expected claims)
//!
//! ```json
//! {
//!   "sub": "user@example.com",
//!   "tenant_id": 42,
//!   "roles": ["viewer", "editor"]
//! }
//! ```
//!
//! - `sub` (required) — maps to `ActorContext.actor_id`.
//! - `tenant_id` (required) — maps to `ActorContext.tenant_id` (`TenantId = u64`).
//! - `roles` (optional) — maps to `ActorContext.roles`. Defaults to `[]`.
//!
//! # Zero New Dependencies
//!
//! Uses `serde` + `serde_json` (already gated under `[auth]` feature).
//! Base64 URL-safe decoding is implemented inline (~40 lines) — no
//! `base64` crate, no `jsonwebtoken` crate.
//!
//! Plan: `.claude/plans/callcenter-membrane-v1.md` § DM-7

use lance_graph_contract::auth::{ActorContext, AuthError};
use serde::Deserialize;

/// JWT extraction middleware.
///
/// Phase 1: base64-decode the payload section of a JWT and extract
/// `sub`, `tenant_id`, and `roles` into an `ActorContext`.
///
/// No signature verification in Phase 1 — the token is trusted as-is.
/// Phase 2 will add a `JwkSetProvider` trait for real verification.
pub struct JwtMiddleware;

/// Deserialization target for the JWT payload claims we care about.
#[derive(Deserialize)]
struct JwtClaims {
    /// JWT `sub` claim — canonical actor identity.
    sub: Option<String>,
    /// Custom claim: tenant identifier.
    tenant_id: Option<u64>,
    /// Custom claim: actor roles. Optional; defaults to empty.
    #[serde(default)]
    roles: Vec<String>,
}

impl JwtMiddleware {
    /// Extract `ActorContext` from a raw JWT token string.
    ///
    /// The token should be in the standard `header.payload.signature`
    /// format. Only the payload section is decoded and parsed.
    ///
    /// # Phase 1 Limitations
    ///
    /// - **No signature verification.** The signature section is ignored.
    ///   Deploy behind a reverse proxy or API gateway that validates
    ///   signatures before traffic reaches this layer.
    /// - **No expiry checking.** `exp` / `nbf` / `iat` are ignored.
    ///   Phase 2 will enforce temporal validity.
    ///
    /// # Errors
    ///
    /// - `AuthError::MalformedToken` — token doesn't have 3 dot-separated parts.
    /// - `AuthError::InvalidBase64` — payload isn't valid base64url.
    /// - `AuthError::MissingSub` — payload JSON is missing the `sub` claim.
    /// - `AuthError::InvalidPayload` — payload JSON can't be parsed.
    pub fn extract_actor(token: &str) -> Result<ActorContext, AuthError> {
        // Split into header.payload.signature
        let parts: Vec<&str> = token.split('.').collect();
        if parts.len() != 3 {
            return Err(AuthError::MalformedToken);
        }

        // Decode payload (middle part)
        let payload_bytes = base64url_decode(parts[1])?;

        // Parse JSON
        let claims: JwtClaims = serde_json::from_slice(&payload_bytes)
            .map_err(|e| AuthError::InvalidPayload(e.to_string()))?;

        // Extract required fields
        let actor_id = claims.sub.ok_or(AuthError::MissingSub)?;
        if actor_id.is_empty() {
            return Err(AuthError::MissingSub);
        }

        let tenant_id = claims.tenant_id.unwrap_or(0);

        Ok(ActorContext::new(actor_id, tenant_id, claims.roles))
    }

    /// Extract `ActorContext` from an `Authorization: Bearer <token>` header value.
    ///
    /// Strips the `Bearer ` prefix if present, then delegates to `extract_actor`.
    pub fn extract_from_header(header_value: &str) -> Result<ActorContext, AuthError> {
        let token = header_value
            .strip_prefix("Bearer ")
            .or_else(|| header_value.strip_prefix("bearer "))
            .unwrap_or(header_value);
        Self::extract_actor(token)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MINIMAL BASE64URL DECODER
// ═══════════════════════════════════════════════════════════════════════════

/// Decode a base64url-encoded string (RFC 4648 §5) without padding.
///
/// JWT payloads use URL-safe base64 without padding characters (`=`).
/// This decoder handles both padded and unpadded inputs.
///
/// ~40 lines, no external crate. Handles the full base64url alphabet
/// (A-Z, a-z, 0-9, `-`, `_`).
fn base64url_decode(input: &str) -> Result<Vec<u8>, AuthError> {
    // Base64url alphabet → 6-bit value
    fn char_to_sextet(c: u8) -> Result<u8, AuthError> {
        match c {
            b'A'..=b'Z' => Ok(c - b'A'),
            b'a'..=b'z' => Ok(c - b'a' + 26),
            b'0'..=b'9' => Ok(c - b'0' + 52),
            b'-' => Ok(62),
            b'_' => Ok(63),
            b'=' => Ok(0), // padding — value ignored
            _ => Err(AuthError::InvalidBase64),
        }
    }

    // Strip padding for length calculation
    let stripped = input.trim_end_matches('=');
    let input_bytes = stripped.as_bytes();
    let len = input_bytes.len();

    if len == 0 {
        return Ok(Vec::new());
    }

    // Validate: base64 produces 3 output bytes per 4 input chars.
    // Without padding: len%4 can be 0, 2, or 3 (never 1).
    if len % 4 == 1 {
        return Err(AuthError::InvalidBase64);
    }

    let out_len = len * 3 / 4;
    let mut out = Vec::with_capacity(out_len);

    // Process full 4-char groups
    let full_groups = len / 4;
    for i in 0..full_groups {
        let base = i * 4;
        let a = char_to_sextet(input_bytes[base])?;
        let b = char_to_sextet(input_bytes[base + 1])?;
        let c = char_to_sextet(input_bytes[base + 2])?;
        let d = char_to_sextet(input_bytes[base + 3])?;

        out.push((a << 2) | (b >> 4));
        out.push((b << 4) | (c >> 2));
        out.push((c << 6) | d);
    }

    // Handle remaining 2 or 3 chars
    let remainder = len % 4;
    if remainder >= 2 {
        let base = full_groups * 4;
        let a = char_to_sextet(input_bytes[base])?;
        let b = char_to_sextet(input_bytes[base + 1])?;
        out.push((a << 2) | (b >> 4));

        if remainder == 3 {
            let c = char_to_sextet(input_bytes[base + 2])?;
            out.push((b << 4) | (c >> 2));
        }
    }

    Ok(out)
}

/// Encode bytes as base64url without padding (for test helpers).
#[cfg(test)]
fn base64url_encode(input: &[u8]) -> String {
    const ALPHABET: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";

    let mut out = String::with_capacity((input.len() + 2) / 3 * 4);

    for chunk in input.chunks(3) {
        let b0 = chunk[0] as usize;
        let b1 = chunk.get(1).copied().unwrap_or(0) as usize;
        let b2 = chunk.get(2).copied().unwrap_or(0) as usize;

        out.push(ALPHABET[(b0 >> 2)] as char);
        out.push(ALPHABET[((b0 & 0x03) << 4) | (b1 >> 4)] as char);

        if chunk.len() > 1 {
            out.push(ALPHABET[((b1 & 0x0F) << 2) | (b2 >> 6)] as char);
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(b2 & 0x3F)] as char);
        }
    }

    out
}

/// Build a minimal unsigned JWT from a JSON payload string (for tests).
#[cfg(test)]
fn make_test_jwt(payload_json: &str) -> String {
    let header = base64url_encode(b"{\"alg\":\"none\",\"typ\":\"JWT\"}");
    let payload = base64url_encode(payload_json.as_bytes());
    // No signature (Phase 1 doesn't verify)
    format!("{header}.{payload}.")
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Base64url decoder tests ──

    #[test]
    fn base64url_roundtrip() {
        let original = b"Hello, JWT world! \xF0\x9F\x94\x91";
        let encoded = base64url_encode(original);
        let decoded = base64url_decode(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    #[test]
    fn base64url_empty() {
        assert_eq!(base64url_decode("").unwrap(), Vec::<u8>::new());
    }

    #[test]
    fn base64url_padding_tolerance() {
        // "Hello" base64url = "SGVsbG8" (no padding) or "SGVsbG8=" (with padding)
        let expected = b"Hello";
        assert_eq!(base64url_decode("SGVsbG8").unwrap(), expected);
        assert_eq!(base64url_decode("SGVsbG8=").unwrap(), expected);
    }

    #[test]
    fn base64url_invalid_char() {
        assert_eq!(base64url_decode("!!!"), Err(AuthError::InvalidBase64));
    }

    #[test]
    fn base64url_invalid_length() {
        // len%4 == 1 is invalid
        assert_eq!(base64url_decode("A"), Err(AuthError::InvalidBase64));
    }

    // ── JWT extraction tests ──

    #[test]
    fn valid_jwt_full_claims() {
        let jwt = make_test_jwt(
            r#"{"sub":"user@example.com","tenant_id":42,"roles":["admin","viewer"]}"#,
        );
        let ctx = JwtMiddleware::extract_actor(&jwt).unwrap();
        assert_eq!(ctx.actor_id, "user@example.com");
        assert_eq!(ctx.tenant_id, 42);
        assert_eq!(ctx.roles, vec!["admin", "viewer"]);
        assert!(ctx.is_admin());
    }

    #[test]
    fn valid_jwt_minimal_claims() {
        let jwt = make_test_jwt(r#"{"sub":"bot-123","tenant_id":1}"#);
        let ctx = JwtMiddleware::extract_actor(&jwt).unwrap();
        assert_eq!(ctx.actor_id, "bot-123");
        assert_eq!(ctx.tenant_id, 1);
        assert!(ctx.roles.is_empty());
        assert!(!ctx.is_admin());
    }

    #[test]
    fn valid_jwt_empty_roles() {
        let jwt = make_test_jwt(r#"{"sub":"x","tenant_id":0,"roles":[]}"#);
        let ctx = JwtMiddleware::extract_actor(&jwt).unwrap();
        assert!(ctx.roles.is_empty());
    }

    #[test]
    fn valid_jwt_missing_tenant_defaults_to_zero() {
        let jwt = make_test_jwt(r#"{"sub":"x"}"#);
        let ctx = JwtMiddleware::extract_actor(&jwt).unwrap();
        assert_eq!(ctx.tenant_id, 0);
    }

    #[test]
    fn missing_sub_error() {
        let jwt = make_test_jwt(r#"{"tenant_id":1,"roles":["viewer"]}"#);
        assert_eq!(
            JwtMiddleware::extract_actor(&jwt),
            Err(AuthError::MissingSub)
        );
    }

    #[test]
    fn empty_sub_error() {
        let jwt = make_test_jwt(r#"{"sub":"","tenant_id":1}"#);
        assert_eq!(
            JwtMiddleware::extract_actor(&jwt),
            Err(AuthError::MissingSub)
        );
    }

    #[test]
    fn malformed_token_no_dots() {
        assert_eq!(
            JwtMiddleware::extract_actor("not-a-jwt"),
            Err(AuthError::MalformedToken)
        );
    }

    #[test]
    fn malformed_token_two_parts() {
        assert_eq!(
            JwtMiddleware::extract_actor("header.payload"),
            Err(AuthError::MalformedToken)
        );
    }

    #[test]
    fn malformed_token_four_parts() {
        assert_eq!(
            JwtMiddleware::extract_actor("a.b.c.d"),
            Err(AuthError::MalformedToken)
        );
    }

    #[test]
    fn invalid_base64_payload() {
        // Valid structure (3 parts) but middle part is bad base64
        assert!(matches!(
            JwtMiddleware::extract_actor("header.!!!invalid.sig"),
            Err(AuthError::InvalidBase64)
        ));
    }

    #[test]
    fn invalid_json_payload() {
        let header = base64url_encode(b"{}");
        let payload = base64url_encode(b"not json at all {{{");
        let token = format!("{header}.{payload}.");
        assert!(matches!(
            JwtMiddleware::extract_actor(&token),
            Err(AuthError::InvalidPayload(_))
        ));
    }

    #[test]
    fn extract_from_bearer_header() {
        let jwt = make_test_jwt(r#"{"sub":"user@test.com","tenant_id":7}"#);
        let header = format!("Bearer {jwt}");
        let ctx = JwtMiddleware::extract_from_header(&header).unwrap();
        assert_eq!(ctx.actor_id, "user@test.com");
        assert_eq!(ctx.tenant_id, 7);
    }

    #[test]
    fn extract_from_header_lowercase_bearer() {
        let jwt = make_test_jwt(r#"{"sub":"x","tenant_id":1}"#);
        let header = format!("bearer {jwt}");
        let ctx = JwtMiddleware::extract_from_header(&header).unwrap();
        assert_eq!(ctx.actor_id, "x");
    }

    #[test]
    fn extract_from_header_no_prefix() {
        let jwt = make_test_jwt(r#"{"sub":"x","tenant_id":1}"#);
        let ctx = JwtMiddleware::extract_from_header(&jwt).unwrap();
        assert_eq!(ctx.actor_id, "x");
    }
}
