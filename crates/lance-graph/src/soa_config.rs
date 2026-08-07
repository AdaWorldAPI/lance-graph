//! Boot-time configuration for which SoA bakes a deployment knows about.
//!
//! A deployment reads exactly one YAML object from its object store at
//! `.config/<repo-name>/config.yaml` (see [`config_key`]) and that object
//! declares which bakes exist and which get hydrated to local disk. The same
//! binary then behaves identically in a dev container and on Railway,
//! because both read the same object at boot — there is no code path that
//! branches on "am I local or am I Railway", only a path that reads whatever
//! config object is sitting at that key.
//!
//! # Why config lives in the bucket, not in env vars or the binary
//!
//! Credentials belong in env vars (they are secrets, and secrets should
//! never live in an object a session's own tooling can list and read).
//! A manifest of *which bakes exist* is a different kind of thing entirely:
//! it changes far more often than code, and it must be readable by both
//! environments (dev container, Railway) without a redeploy. Baking it into
//! the binary means every new bake needs a rebuild+redeploy just to be
//! discoverable; putting it in an env var means it's invisible to `aws s3
//! ls` and has no place to grow structure. One YAML object in the bucket is
//! the one source of truth both environments already have credentialed
//! access to.
//!
//! # This is boot config, not a hot-path serialization violation
//!
//! This module is read **once, at startup**, exactly the same category as a
//! `build.rs` manifest parse or a `Cargo.toml` read. It is not on the
//! per-request or per-row hot path this workspace's Firewall doctrine
//! (ADR-022/023, "no serialization in the hot path; the IR is wire-truth")
//! protects. A future session should not "fix" this module by stripping
//! serde out of it — the ban is on serialization *between mailboxes during
//! cognition*, not on parsing a startup manifest.
//!
//! # Key names vs. values
//!
//! The struct field names, the YAML keys they map to, and the shapes below
//! are universal and belong in shared/public code — any deployment's config
//! object uses the same keys. The *values* that go in those keys (a bucket
//! name, a ledger prefix, a slab digest) are deployment configuration and
//! must never enter the repository, exactly like the `AWS_*` variables
//! [`crate::dev_s3_env`] reads: the shape is public, the content is not.

use std::fmt;

/// The schema version this build of `lance-graph` understands. A config
/// object declaring a different major version is refused outright (see
/// [`parse`]) rather than half-read — a future schema change is a loud
/// failure at boot, not a silent misinterpretation of fields that changed
/// meaning.
pub const CONFIG_SCHEMA_VERSION: u32 = 1;

/// The file name every deployment's config object uses.
pub const CONFIG_BASENAME: &str = "config.yaml";

/// The top-level prefix under which every repo's config object lives.
pub const CONFIG_ROOT: &str = ".config";

/// What to do when a bake's `hydrate`/refresh path finds a table that
/// already exists at the target name.
///
/// # The overwrite doctrine
///
/// - [`OnExisting::Refuse`] (the default): an existing table is never
///   silently overwritten. A refresh that lands on an occupied name is a
///   hard error, not a clobber.
/// - [`OnExisting::NewVersion`]: a refresh instead writes a **new**,
///   timestamped table (see [`versioned_table_name`]) and the config's
///   pointer (the `table` field naming which one is current) is updated to
///   point at it. The old table is left untouched on disk until an explicit,
///   separate purge.
///
/// **Why not rename-to-OLD instead of writing-new-then-repointing:** S3 has
/// no atomic rename. A Lance dataset is N objects under a prefix, and
/// "renaming" it is N copies plus N deletes — non-atomic by construction. A
/// crash partway through leaves the dataset split across two prefixes with
/// no single pointer that resolves to either half correctly. Flipping one
/// line of YAML (the `table` field) to name the new, already-fully-written
/// table is O(1) and atomic from a reader's point of view: the config *is*
/// the pointer, so a reader never observes a half-migrated state — it either
/// still sees the old table (config not yet updated) or the new one
/// (config updated), never a mix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OnExisting {
    #[default]
    Refuse,
    NewVersion,
}

/// One declared SoA bake: a name a caller looks it up by, the Lance table it
/// lives in, the classid that identifies its node layout, and whether this
/// deployment should pull it to local disk at boot.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
// A typo'd key (`slab_digset`, `hydarte`) would otherwise be SILENTLY ignored and
// its field defaulted — dropping a digest pin or flipping hydration off in a
// config whose whole purpose is to fail loudly at boot. Reject instead.
#[serde(deny_unknown_fields)]
pub struct BakeEntry {
    /// Human-facing key, unique within one config object. Not a filesystem
    /// path — just a lookup name (see [`SoaConfig::find`]).
    pub name: String,
    /// The Lance table name under `ledger_prefix`.
    pub table: String,
    /// Hex classid, e.g. `"0x0F010000"`, identifying this bake's node layout.
    ///
    /// **A FULL u32 classid** — the same width the canonical node key
    /// reserves at bytes `0..4`. Under the active `CanonHigh` order the
    /// LEFT (high) bytes carry the minted concept `0xDDCC` — domain in the
    /// most-significant byte, which is what makes classids sort and
    /// prefix-search hierarchically. **Read the left bytes to route.**
    ///
    /// The LOW half is not padding: it carries the app/render half —
    /// `ClassView` + `WideFieldMask` ergonomics and slot-schema switching.
    /// `0x0000` there means "no app skin", one legal value among many, and
    /// it is a slot a consumer FILLS — e.g. a session writing an ontology
    /// routing value into it. So a config carrying `…0000` is declaring the
    /// slot unset, not declaring it meaningless.
    ///
    /// # The addressing model, and the one place the CSS analogy breaks
    ///
    /// Low half + field mask compose into **screen-region addressing**, in
    /// the CSS sense: the low half selects the `ClassView` (the per-app
    /// template/skin) the way a selector picks an element, and the
    /// [`FieldMask`]/[`WideFieldMask`] selects which of that view's fields
    /// are in play the way declarations pick properties. That is the whole
    /// basis of a2ui-rs's "don't push pixels — address the screen": a
    /// `NodeDelta` carries a 16-byte key plus mask words, never a rendered
    /// region.
    ///
    /// **Where the analogy must not be followed:** a field mask is
    /// *presence, never semantics* (`class_view.rs` C2). `has(n)` answers
    /// "is field n populated here" — it must NEVER gate "field n means
    /// something different here." CSS's cascade does change which rule
    /// wins; a mask never changes what a field means. Read the analogy for
    /// addressing only.
    ///
    /// [`FieldMask`]: lance_graph_contract::class_view::FieldMask
    /// [`WideFieldMask`]: lance_graph_contract::class_view::WideFieldMask
    ///
    /// [`parse`] validates only that this is `0x`-prefixed hex fitting u32.
    /// It deliberately does **not** police the halves: a zero canon is a
    /// legal dormant state under the zero-fallback ladder, and pre-flip
    /// stored forms are legitimately read via `classid_canon_compat`. Use
    /// [`BakeEntry::classid_u32`] to read the value and the codebook's own
    /// accessors to split it.
    pub classid: String,
    /// Digest of the bake's slab, when the deployment pins one. Absent
    /// means "trust whatever is at `table` right now".
    #[serde(default)]
    pub slab_digest: Option<String>,
    /// Whether this deployment should pull the bake to local disk at boot
    /// (`true`) or read it remotely on demand (`false`).
    #[serde(default)]
    pub hydrate: bool,
}

/// The parsed, validated contents of one deployment's `config.yaml`.
#[derive(Debug, Clone, PartialEq, serde::Deserialize, serde::Serialize)]
#[serde(deny_unknown_fields)]
pub struct SoaConfig {
    /// Schema major version. [`parse`] refuses anything other than
    /// [`CONFIG_SCHEMA_VERSION`].
    pub version: u32,
    /// Per-repo ledger prefix, e.g. `"lance-graph/ledger"`. Combined with a
    /// bucket and a [`BakeEntry::table`] by [`SoaConfig::table_uri`].
    pub ledger_prefix: String,
    /// The declared bakes. Names and tables must each be unique — see
    /// [`parse`].
    pub bakes: Vec<BakeEntry>,
    /// What a refresh does when it finds an occupied table name. See
    /// [`OnExisting`] for the reasoning.
    #[serde(default)]
    pub on_existing: OnExisting,
}

/// Everything that can go wrong turning a YAML string into a validated
/// [`SoaConfig`]. Each variant names the offending value so a bad config
/// object fails loudly with something a human can grep the YAML for,
/// instead of a generic "invalid config" at boot.
#[derive(Debug, Clone, PartialEq)]
pub enum ConfigError {
    /// The YAML itself did not parse. Carries `serde_yaml`'s message.
    Yaml(String),
    /// `version` did not match [`CONFIG_SCHEMA_VERSION`].
    UnsupportedVersion { found: u32, supported: u32 },
    /// Two bakes declared the same `name`.
    DuplicateName(String),
    /// Two bakes declared the same `table`.
    DuplicateTable(String),
    /// A required string field was empty. Carries the field's name.
    EmptyField(&'static str),
    /// `classid` did not parse as a `0x`-prefixed hex value that fits u32.
    BadClassid(String),
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::Yaml(msg) => write!(f, "config.yaml did not parse: {msg}"),
            ConfigError::UnsupportedVersion { found, supported } => write!(
                f,
                "config.yaml declares version {found}, this build only supports version {supported}"
            ),
            ConfigError::DuplicateName(name) => {
                write!(f, "duplicate bake name in config.yaml: {name:?}")
            }
            ConfigError::DuplicateTable(table) => {
                write!(f, "duplicate bake table in config.yaml: {table:?}")
            }
            ConfigError::EmptyField(field) => {
                write!(f, "config.yaml field must not be empty: {field}")
            }
            ConfigError::BadClassid(classid) => {
                write!(
                    f,
                    "classid {classid:?} is not a 0x-prefixed hex value that fits u32"
                )
            }
        }
    }
}

impl std::error::Error for ConfigError {}

/// The object-store key a deployment's config object lives at:
/// `.config/<repo>/config.yaml`. Pure string composition, no I/O — the
/// caller is responsible for actually fetching the object at this key.
pub fn config_key(repo: &str) -> String {
    format!("{CONFIG_ROOT}/{repo}/{CONFIG_BASENAME}")
}

/// A `0x`-prefixed hex string as a `u32`, or `None` if it is not one (missing
/// prefix, empty, non-hex digit, or wider than u32).
fn parse_classid_hex(s: &str) -> Option<u32> {
    let digits = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
    if digits.is_empty() || !digits.chars().all(|c| c.is_ascii_hexdigit()) {
        return None;
    }
    u32::from_str_radix(digits, 16).ok()
}

/// Parse and validate a `config.yaml` body into a [`SoaConfig`].
///
/// Validation rejects, each with its own [`ConfigError`] variant:
/// - a `version` other than [`CONFIG_SCHEMA_VERSION`] (fail loud on a
///   schema a build doesn't understand, rather than half-reading it),
/// - two bakes sharing a `name`,
/// - two bakes sharing a `table` (two entries pointing at one table on
///   disk is always a mistake — one of them would be reading garbage or
///   racing the other's writes),
/// - an empty `ledger_prefix`, `name`, or `table`,
/// - a `classid` that is not `0x`-prefixed hex fitting u32,
/// - any unknown key, via `deny_unknown_fields` on both structs — a typo
///   must not silently default a field.
pub fn parse(yaml: &str) -> Result<SoaConfig, ConfigError> {
    let config: SoaConfig =
        serde_yaml::from_str(yaml).map_err(|e| ConfigError::Yaml(e.to_string()))?;

    if config.version != CONFIG_SCHEMA_VERSION {
        return Err(ConfigError::UnsupportedVersion {
            found: config.version,
            supported: CONFIG_SCHEMA_VERSION,
        });
    }

    if config.ledger_prefix.is_empty() {
        return Err(ConfigError::EmptyField("ledger_prefix"));
    }

    let mut seen_names: std::collections::HashSet<&str> = std::collections::HashSet::new();
    let mut seen_tables: std::collections::HashSet<&str> = std::collections::HashSet::new();

    for bake in &config.bakes {
        if bake.name.is_empty() {
            return Err(ConfigError::EmptyField("name"));
        }
        if bake.table.is_empty() {
            return Err(ConfigError::EmptyField("table"));
        }
        if !seen_names.insert(bake.name.as_str()) {
            return Err(ConfigError::DuplicateName(bake.name.clone()));
        }
        if !seen_tables.insert(bake.table.as_str()) {
            return Err(ConfigError::DuplicateTable(bake.table.clone()));
        }

        // Structural only: it must be 0x-prefixed hex that fits the u32 the
        // canonical node key reserves for it. NO semantic check on the halves
        // — a zero canon is a legal dormant/bootstrap state under the
        // zero-fallback ladder (CLAUDE.md: a zero tier is "not consulted",
        // never an error), and pre-flip stored forms are legitimately read by
        // `classid_canon_compat`. Rejecting either here would refuse valid
        // configs on an inference about intent this parser has no basis for.
        parse_classid_hex(&bake.classid)
            .ok_or_else(|| ConfigError::BadClassid(bake.classid.clone()))?;
    }

    Ok(config)
}

impl SoaConfig {
    /// The full `s3://` URI a bake's table lives at: bucket, this config's
    /// `ledger_prefix`, and the entry's own `table`, joined with `/`.
    pub fn table_uri(&self, bucket: &str, entry: &BakeEntry) -> String {
        format!("s3://{bucket}/{}/{}", self.ledger_prefix, entry.table)
    }

    /// The bakes this deployment should pull to local disk at boot.
    pub fn hydrate_set(&self) -> impl Iterator<Item = &BakeEntry> {
        self.bakes.iter().filter(|b| b.hydrate)
    }

    /// Look up a bake by its declared `name`.
    pub fn find(&self, name: &str) -> Option<&BakeEntry> {
        self.bakes.iter().find(|b| b.name == name)
    }
}

impl BakeEntry {
    /// The declared classid as the `u32` it is. Infallible on any entry that
    /// came through [`parse`] — validation already rejected anything that
    /// does not parse — so this is the accessor callers should use instead of
    /// re-parsing the string and re-deciding what a malformed one means.
    pub fn classid_u32(&self) -> Option<u32> {
        parse_classid_hex(&self.classid)
    }
}

/// Compute the versioned table name an [`OnExisting::NewVersion`] refresh
/// writes to. `berlin.lance` with nanos `N` becomes `berlin.<N>.lance`; a
/// table name with no `.lance` suffix gets `.<N>` appended.
///
/// Takes the timestamp as a **parameter**, deliberately never calling the
/// clock itself — that keeps this function pure and trivially testable, and
/// keeps "what time is it" a decision made once by the caller rather than
/// smeared across every call site that might need a versioned name.
pub fn versioned_table_name(table: &str, unix_nanos: u128) -> String {
    match table.strip_suffix(".lance") {
        Some(stem) => format!("{stem}.{unix_nanos}.lance"),
        None => format!("{table}.{unix_nanos}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn valid_yaml() -> &'static str {
        r#"
version: 1
ledger_prefix: "lance-graph/ledger"
on_existing: new_version
bakes:
  - name: berlin
    table: berlin.lance
    classid: "0x0F010000"
    slab_digest: "sha256:abc123"
    hydrate: true
  - name: munich
    table: munich.lance
    classid: "0x0F020000"
    hydrate: false
"#
    }

    #[test]
    fn valid_config_round_trips_with_expected_field_values() {
        let config = parse(valid_yaml()).expect("valid config must parse");
        assert_eq!(config.version, 1);
        assert_eq!(config.ledger_prefix, "lance-graph/ledger");
        assert_eq!(config.on_existing, OnExisting::NewVersion);
        assert_eq!(config.bakes.len(), 2);

        let berlin = config.find("berlin").expect("berlin must be found");
        assert_eq!(berlin.table, "berlin.lance");
        assert_eq!(berlin.classid, "0x0F010000");
        assert_eq!(berlin.slab_digest.as_deref(), Some("sha256:abc123"));
        assert!(berlin.hydrate);

        let munich = config.find("munich").expect("munich must be found");
        assert!(!munich.hydrate);
        assert!(munich.slab_digest.is_none());

        assert!(config.find("nonexistent").is_none());
    }

    #[test]
    fn on_existing_defaults_to_refuse_when_omitted() {
        let yaml = r#"
version: 1
ledger_prefix: "lance-graph/ledger"
bakes:
  - name: berlin
    table: berlin.lance
    classid: "0x0F010000"
"#;
        let config = parse(yaml).expect("must parse");
        assert_eq!(config.on_existing, OnExisting::Refuse);
    }

    #[test]
    fn rejects_unsupported_version() {
        let yaml = r#"
version: 2
ledger_prefix: "lance-graph/ledger"
bakes: []
"#;
        let err = parse(yaml).expect_err("wrong version must be rejected");
        assert_eq!(
            err,
            ConfigError::UnsupportedVersion {
                found: 2,
                supported: CONFIG_SCHEMA_VERSION
            }
        );

        // Paired case: same config with the version corrected passes,
        // proving the rejection above was specifically about the version
        // field and not some other defect in the fixture.
        let fixed = yaml.replace("version: 2", "version: 1");
        assert!(parse(&fixed).is_ok());
    }

    #[test]
    fn rejects_duplicate_bake_name() {
        let yaml = r#"
version: 1
ledger_prefix: "lance-graph/ledger"
bakes:
  - name: berlin
    table: berlin.lance
    classid: "0x0F010000"
  - name: berlin
    table: berlin2.lance
    classid: "0x0F020000"
"#;
        let err = parse(yaml).expect_err("duplicate name must be rejected");
        assert_eq!(err, ConfigError::DuplicateName("berlin".to_string()));

        let fixed = yaml.replacen(
            "name: berlin\n    table: berlin2.lance",
            "name: berlin2\n    table: berlin2.lance",
            1,
        );
        assert!(parse(&fixed).is_ok());
    }

    #[test]
    fn rejects_duplicate_bake_table() {
        let yaml = r#"
version: 1
ledger_prefix: "lance-graph/ledger"
bakes:
  - name: berlin
    table: shared.lance
    classid: "0x0F010000"
  - name: munich
    table: shared.lance
    classid: "0x0F020000"
"#;
        let err = parse(yaml).expect_err("duplicate table must be rejected");
        assert_eq!(err, ConfigError::DuplicateTable("shared.lance".to_string()));

        let fixed = yaml.replacen(
            "name: munich\n    table: shared.lance",
            "name: munich\n    table: munich.lance",
            1,
        );
        assert!(parse(&fixed).is_ok());
    }

    #[test]
    fn rejects_empty_ledger_prefix() {
        let yaml = r#"
version: 1
ledger_prefix: ""
bakes: []
"#;
        let err = parse(yaml).expect_err("empty ledger_prefix must be rejected");
        assert_eq!(err, ConfigError::EmptyField("ledger_prefix"));

        let fixed = yaml.replace(r#"ledger_prefix: """#, r#"ledger_prefix: "x""#);
        assert!(parse(&fixed).is_ok());
    }

    #[test]
    fn rejects_empty_bake_name() {
        let yaml = r#"
version: 1
ledger_prefix: "lance-graph/ledger"
bakes:
  - name: ""
    table: berlin.lance
    classid: "0x0F010000"
"#;
        let err = parse(yaml).expect_err("empty name must be rejected");
        assert_eq!(err, ConfigError::EmptyField("name"));

        let fixed = yaml.replace(r#"name: """#, r#"name: "berlin""#);
        assert!(parse(&fixed).is_ok());
    }

    #[test]
    fn rejects_empty_bake_table() {
        let yaml = r#"
version: 1
ledger_prefix: "lance-graph/ledger"
bakes:
  - name: berlin
    table: ""
    classid: "0x0F010000"
"#;
        let err = parse(yaml).expect_err("empty table must be rejected");
        assert_eq!(err, ConfigError::EmptyField("table"));

        let fixed = yaml.replace(r#"table: """#, r#"table: "berlin.lance""#);
        assert!(parse(&fixed).is_ok());
    }

    #[test]
    fn rejects_bad_classid() {
        let yaml = r#"
version: 1
ledger_prefix: "lance-graph/ledger"
bakes:
  - name: berlin
    table: berlin.lance
    classid: "F01"
"#;
        let err = parse(yaml).expect_err("classid without 0x prefix must be rejected");
        assert_eq!(err, ConfigError::BadClassid("F01".to_string()));

        let fixed = yaml.replace(r#"classid: "F01""#, r#"classid: "0xF010000""#);
        assert!(parse(&fixed).is_ok());
    }

    #[test]
    fn rejects_classid_with_non_hex_digits() {
        let yaml = r#"
version: 1
ledger_prefix: "lance-graph/ledger"
bakes:
  - name: berlin
    table: berlin.lance
    classid: "0xZZ"
"#;
        let err = parse(yaml).expect_err("non-hex digits must be rejected");
        assert_eq!(err, ConfigError::BadClassid("0xZZ".to_string()));
    }

    /// **A typo must not silently default a field.** `slab_digset` would
    /// otherwise be ignored and the digest pin dropped; `hydarte` would be
    /// ignored and hydration silently turned off. Two-sided: the typo is
    /// rejected, the correct spelling is accepted with its value intact.
    #[test]
    fn rejects_unknown_keys_instead_of_silently_defaulting_them() {
        let with = |key: &str, val: &str| {
            format!(
                "version: 1\nledger_prefix: \"lance-graph/ledger\"\nbakes:\n  \
                 - name: berlin\n    table: berlin.lance\n    \
                 classid: \"0x0F010000\"\n    {key}: {val}\n"
            )
        };

        for (typo, val) in [("slab_digset", "\"sha256:abc\""), ("hydarte", "true")] {
            assert!(
                parse(&with(typo, val)).is_err(),
                "typo {typo:?} must be REJECTED, not silently ignored and defaulted"
            );
        }

        // Correct spellings still work, and carry their values — proving the
        // rejection above is about the KEY being unknown, not about the
        // parser having become uniformly hostile.
        let ok = parse(&with("slab_digest", "\"sha256:abc\"")).expect("correct key must parse");
        assert_eq!(ok.bakes[0].slab_digest.as_deref(), Some("sha256:abc"));
        let ok = parse(&with("hydrate", "true")).expect("correct key must parse");
        assert!(ok.bakes[0].hydrate);

        // An unknown key at the TOP level is rejected too, not just in a bake.
        assert!(
            parse("version: 1\nledger_prefix: \"p\"\nbakes: []\nledgerprefix: \"typo\"\n").is_err(),
            "an unknown top-level key must be rejected"
        );
    }

    #[test]
    fn config_key_produces_the_exact_expected_string() {
        assert_eq!(config_key("lance-graph"), ".config/lance-graph/config.yaml");
        assert_eq!(config_key("q2"), ".config/q2/config.yaml");
    }

    #[test]
    fn versioned_table_name_on_dot_lance_suffix() {
        assert_eq!(
            versioned_table_name("berlin.lance", 1_700_000_000_000_000_000),
            "berlin.1700000000000000000.lance"
        );
    }

    #[test]
    fn versioned_table_name_on_bare_name() {
        assert_eq!(versioned_table_name("berlin", 42), "berlin.42");
    }

    #[test]
    fn hydrate_set_returns_only_hydrate_true_entries() {
        let config = parse(valid_yaml()).expect("must parse");
        let hydrated: Vec<&str> = config.hydrate_set().map(|b| b.name.as_str()).collect();
        // Fixture carries one true (berlin) and one false (munich) entry —
        // the count must be neither 0 (filter over-rejects) nor 2 (filter
        // is a no-op that returns everything).
        assert_eq!(hydrated.len(), 1);
        assert_eq!(hydrated, vec!["berlin"]);
    }

    #[test]
    fn table_uri_composes_correctly() {
        let config = parse(valid_yaml()).expect("must parse");
        let berlin = config.find("berlin").unwrap();
        assert_eq!(
            config.table_uri("my-bucket", berlin),
            "s3://my-bucket/lance-graph/ledger/berlin.lance"
        );
    }

    /// **The shipped example must load through THIS parser.**
    ///
    /// `examples/soa-config.example.yaml` is what the deployment docs point a
    /// human at, and it was authored by reading these structs rather than by
    /// running them — so without this test the example and the parser agree
    /// only by hand, and the first field rename desynchronises them silently.
    /// Documentation that no longer parses is worse than none: it is
    /// confidently wrong.
    ///
    /// `include_str!` resolves at COMPILE time relative to this source file,
    /// so a moved or deleted example is a build error rather than a test that
    /// quietly stops covering anything.
    #[test]
    fn the_shipped_example_config_parses_through_this_parser() {
        let example = include_str!("../examples/soa-config.example.yaml");
        let config = parse(example)
            .unwrap_or_else(|e| panic!("examples/soa-config.example.yaml does not parse: {e}"));

        // Not merely "it parsed" — the example exists to DEMONSTRATE, so
        // assert it still demonstrates. A single-entry or all-same-hydrate
        // example would parse fine while teaching nothing about the
        // distinction it is there to show.
        assert!(
            config.bakes.len() >= 3,
            "the example should show several bakes; found {}",
            config.bakes.len()
        );
        let hydrated = config.hydrate_set().count();
        assert!(
            hydrated > 0 && hydrated < config.bakes.len(),
            "the example must carry BOTH hydrate:true and hydrate:false entries \
             or it does not illustrate the choice; {hydrated} of {} are hydrated",
            config.bakes.len()
        );
    }
}
