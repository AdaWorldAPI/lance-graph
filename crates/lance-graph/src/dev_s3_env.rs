//! The ONE shared reading of the S3 environment for every dev-only S3 caller
//! in this crate (`examples/soa_to_lance.rs`, `examples/hydration_probe.rs`,
//! `tests/soa_verbatim.rs`).
//!
//! Before this module existed, the writer and its own verification test each
//! carried an independently-typed copy of [`env`] and [`s3_options`] — a
//! CodeRabbit finding on PR #907: "One shared helper removes the drift risk
//! between the write path and the verification path." A write path and its
//! own proof reading two different option maps is exactly the kind of drift
//! that would make a green test mean nothing.
//!
//! KEY NAMES here are universal and belong in shared code (they match what a
//! Railway deployment already sets: `AWS_ACCESS_KEY_ID`,
//! `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, `AWS_DEFAULT_REGION`,
//! `AWS_S3_BUCKET_NAME`). VALUES never appear here or anywhere else in the
//! repository.

use std::collections::HashMap;

/// An environment value, with the wrapping quotes this sandbox's exporter adds
/// stripped. A quoted `"…"` credential authenticates as garbage, and the error
/// it produces points at the credential rather than at the quoting.
pub fn env(k: &str) -> Option<String> {
    std::env::var(k)
        .ok()
        .map(|v| v.trim().trim_matches('"').trim_matches('\'').to_string())
        .filter(|v| !v.is_empty())
}

/// The `object_store` option map, built from the deployment's own variables.
///
/// `aws_endpoint` ← `AWS_ENDPOINT_URL` is the load-bearing line: `object_store`
/// reads `AWS_ENDPOINT`, which this environment does not set, so its own env
/// discovery would address AWS proper instead of the configured endpoint.
///
/// Returns `None` if any REQUIRED variable is missing — callers should treat
/// `None` on a path that has already committed to being remote as a hard
/// error, not a silent fallback to default credential discovery (a prior
/// defect on this branch: `soa_to_lance` could write with `store_params: None`
/// and re-open with real options, addressing two different endpoints without
/// either failing fast).
pub fn s3_options() -> Option<HashMap<String, String>> {
    let mut o = HashMap::new();
    o.insert("aws_access_key_id".into(), env("AWS_ACCESS_KEY_ID")?);
    o.insert(
        "aws_secret_access_key".into(),
        env("AWS_SECRET_ACCESS_KEY")?,
    );
    o.insert("aws_endpoint".into(), env("AWS_ENDPOINT_URL")?);
    o.insert(
        "aws_region".into(),
        env("AWS_DEFAULT_REGION").unwrap_or_else(|| "auto".into()),
    );
    o.insert("aws_virtual_hosted_style_request".into(), "false".into());
    Some(o)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_strips_wrapping_double_quotes() {
        // SAFETY: single-threaded test process section, no other test reads
        // this exact key.
        unsafe { std::env::set_var("DEV_S3_ENV_TEST_QUOTED", "\"abc123\"") };
        assert_eq!(env("DEV_S3_ENV_TEST_QUOTED").as_deref(), Some("abc123"));
        unsafe { std::env::remove_var("DEV_S3_ENV_TEST_QUOTED") };
    }

    #[test]
    fn env_strips_wrapping_single_quotes() {
        unsafe { std::env::set_var("DEV_S3_ENV_TEST_SINGLE", "'xyz'") };
        assert_eq!(env("DEV_S3_ENV_TEST_SINGLE").as_deref(), Some("xyz"));
        unsafe { std::env::remove_var("DEV_S3_ENV_TEST_SINGLE") };
    }

    #[test]
    fn env_treats_empty_as_absent() {
        unsafe { std::env::set_var("DEV_S3_ENV_TEST_EMPTY", "") };
        assert_eq!(env("DEV_S3_ENV_TEST_EMPTY"), None);
        unsafe { std::env::remove_var("DEV_S3_ENV_TEST_EMPTY") };
    }

    #[test]
    fn env_missing_key_is_none() {
        assert_eq!(env("DEV_S3_ENV_TEST_DOES_NOT_EXIST_XYZ"), None);
    }

    // `s3_options` is deliberately not exercised here with mutated real
    // `AWS_*` vars: this crate's lib tests run multi-threaded in one process,
    // and this session's actual credentials live in that same environment.
    // Its `?`-chain short-circuit behaviour is exactly `env`'s None-on-missing
    // behaviour, already covered above; composing four calls to an already-
    // tested function adds no coverage worth a shared-mutable-env race.
}
