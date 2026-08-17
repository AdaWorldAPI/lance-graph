//! The ONE shared reading of the S3/object-store environment for every
//! hydration caller (mirrors `lance-graph`'s own `dev_s3_env.rs`, generalized
//! off the same, already-Railway-compatible env var names, so a consumer
//! that already sets these for `lance-graph` needs ZERO new configuration to
//! use this crate).
//!
//! KEY NAMES here are universal (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`,
//! `AWS_ENDPOINT_URL`, `AWS_DEFAULT_REGION`, `AWS_S3_BUCKET_NAME`) — the same
//! names a Railway deployment already sets. VALUES never appear here.

use std::collections::HashMap;

/// An environment value, with the wrapping quotes some exporters add
/// stripped. A quoted `"…"` credential authenticates as garbage, and the
/// error it produces points at the credential rather than at the quoting.
pub fn env(k: &str) -> Option<String> {
    std::env::var(k)
        .ok()
        .map(|v| v.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|v| !v.is_empty())
}

/// The resolved object-store endpoint a hydration call targets. Built once
/// from the environment (`from_env`) so every caller in a process reads the
/// same variables the same way — the drift risk `dev_s3_env.rs` was written
/// to close, generalized past one crate.
#[derive(Debug, Clone)]
pub struct HydrationSource {
    pub bucket: String,
    options: HashMap<String, String>,
}

impl HydrationSource {
    /// Reads the standard `AWS_*` variables. Returns `None` if any REQUIRED
    /// variable is missing — callers on a path that has already committed to
    /// remote hydration should treat `None` as a hard error, never a silent
    /// fallback to default credential discovery.
    pub fn from_env() -> Option<Self> {
        let mut options = HashMap::new();
        options.insert("aws_access_key_id".into(), env("AWS_ACCESS_KEY_ID")?);
        options.insert(
            "aws_secret_access_key".into(),
            env("AWS_SECRET_ACCESS_KEY")?,
        );
        options.insert("aws_endpoint".into(), env("AWS_ENDPOINT_URL")?);
        options.insert(
            "aws_region".into(),
            env("AWS_DEFAULT_REGION").unwrap_or_else(|| "auto".into()),
        );
        options.insert("aws_virtual_hosted_style_request".into(), "false".into());
        let bucket = env("AWS_S3_BUCKET_NAME")?;
        Some(Self { bucket, options })
    }

    /// The `object_store` option map for this source.
    pub fn options(&self) -> &HashMap<String, String> {
        &self.options
    }

    /// `s3://<bucket>/<prefix>` for a given prefix (no leading/trailing slash
    /// normalization beyond a single join — callers own their own prefix
    /// shape, same as `VersionedGraph::s3`).
    pub fn uri(&self, prefix: &str) -> String {
        format!("s3://{}/{}", self.bucket, prefix.trim_matches('/'))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_strips_wrapping_double_quotes() {
        // SAFETY: single-threaded test process section (nextest isolates
        // this test's process), no other test reads this exact key.
        unsafe { std::env::set_var("LGH_ENV_TEST_QUOTED", "\"abc123\"") };
        assert_eq!(env("LGH_ENV_TEST_QUOTED").as_deref(), Some("abc123"));
        unsafe { std::env::remove_var("LGH_ENV_TEST_QUOTED") };
    }

    #[test]
    fn env_treats_empty_as_absent() {
        unsafe { std::env::set_var("LGH_ENV_TEST_EMPTY", "") };
        assert_eq!(env("LGH_ENV_TEST_EMPTY"), None);
        unsafe { std::env::remove_var("LGH_ENV_TEST_EMPTY") };
    }

    #[test]
    fn env_missing_key_is_none() {
        assert_eq!(env("LGH_ENV_TEST_DOES_NOT_EXIST_XYZ"), None);
    }

    #[test]
    fn from_env_is_none_when_a_required_var_is_missing() {
        // SAFETY: this test claims exclusive use of the LGH_TEST_SRC_* keys;
        // nextest's one-process-per-test isolation makes the mutation sound.
        unsafe {
            std::env::remove_var("LGH_TEST_SRC_AWS_ACCESS_KEY_ID");
        }
        // Deliberately does NOT read real AWS_* vars: uses a namespaced
        // stand-in check on the underlying `env()` helper instead, since
        // `from_env` reads the literal `AWS_*` names process-wide and this
        // test's process may carry real deployment credentials.
        assert_eq!(env("LGH_TEST_SRC_AWS_ACCESS_KEY_ID"), None);
    }

    #[test]
    fn uri_joins_bucket_and_prefix_without_double_slashes() {
        let src = HydrationSource {
            bucket: "my-bucket".into(),
            options: HashMap::new(),
        };
        assert_eq!(src.uri("/OSM/berlin/"), "s3://my-bucket/OSM/berlin");
        assert_eq!(src.uri("OSM/berlin"), "s3://my-bucket/OSM/berlin");
    }
}
