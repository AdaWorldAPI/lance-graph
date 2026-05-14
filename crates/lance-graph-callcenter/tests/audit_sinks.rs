//! Integration tests for D-SDR-4b audit sinks: LanceAuditSink, JsonlAuditSink,
//! CompositeSink, and cross-format verify helpers.
//!
//! Run with:
//! ```
//! cargo test -p lance-graph-callcenter --features lance-sink,jsonl
//! ```

use lance_graph_callcenter::{
    audit_sink::{AuditError, AuditSink, CompositeSink, FanoutMode},
    super_domain::SuperDomain,
    unified_audit::{AuditChain, AuditMerkleRoot, AuthDecision, AuthOp, UnifiedAuditEvent},
    unified_bridge::{OgitFamily, OwlIdentity, TenantId},
};

// ── Helper: build a chain of N events ────────────────────────────────────────

fn make_events(n: usize, salt: u64) -> Vec<UnifiedAuditEvent> {
    let mut chain = AuditChain::new(SuperDomain::Healthcare, salt);
    (0..n)
        .map(|i| {
            let base = UnifiedAuditEvent {
                ts_unix_ms: 1_747_000_000_000 + i as u64,
                tenant: TenantId(42),
                super_domain: SuperDomain::Healthcare,
                owl: OwlIdentity::new(OgitFamily(7), 0x051c),
                op: AuthOp::Read,
                decision: AuthDecision::Allow,
                actor_role_hash: 0xCAFE_BABE_DEAD_BEEF,
                merkle_root: AuditMerkleRoot::GENESIS,
                prev_merkle: AuditMerkleRoot::GENESIS,
            };
            chain.advance(base)
        })
        .collect()
}

// ══════════════════════════════════════════════════════════════════════════════
// JsonlAuditSink tests (feature = "jsonl")
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "jsonl")]
mod jsonl_tests {
    use super::*;
    use lance_graph_callcenter::audit_sink::jsonl_sink::serialize_event;
    use lance_graph_callcenter::JsonlAuditSink;

    /// Round-trip 100 events through JsonlAuditSink, flush, then parse back
    /// and verify merkle integrity.
    #[test]
    fn jsonl_round_trip_100_events() {
        let dir = tempdir();
        let sink = JsonlAuditSink::new(&dir).expect("create JsonlAuditSink");

        let salt = 0xC0FFEE_u64;
        let events = make_events(100, salt);

        for ev in &events {
            sink.emit(*ev).expect("emit");
        }

        let root = sink.flush().expect("flush");
        assert_eq!(root, events.last().unwrap().merkle_root.raw());

        // Read back JSONL files and verify.
        let jsonl_files: Vec<_> = find_jsonl_files(&dir);
        assert!(
            !jsonl_files.is_empty(),
            "at least one JSONL file should exist"
        );

        let mut parsed: Vec<serde_json::Value> = Vec::new();
        for path in &jsonl_files {
            let content = std::fs::read_to_string(path).unwrap();
            for line in content.lines() {
                if !line.trim().is_empty() {
                    parsed.push(serde_json::from_str(line).unwrap());
                }
            }
        }
        assert_eq!(parsed.len(), 100, "all 100 events should be in JSONL");

        // Re-walk merkle chain from parsed JSONL.
        let genesis = AuditMerkleRoot::GENESIS;
        let mut prev = genesis;
        for (i, v) in parsed.iter().enumerate() {
            let ts_us: u64 = v["timestamp_us"].as_str().unwrap().parse().unwrap();
            let ts_ms = ts_us / 1000;
            let tenant_id = v["tenant_id"].as_u64().unwrap() as u32;
            let sd = v["super_domain"].as_u64().unwrap() as u8;
            let owl_hex = v["owl_identity"].as_str().unwrap();
            let owl = parse_owl_hex(owl_hex);
            let action = v["action"].as_u64().unwrap() as u8;
            let decision = v["decision"].as_u64().unwrap() as u8;
            let arh: u64 = v["actor_role_hash"].as_str().unwrap().parse().unwrap();
            let event_merkle: u64 = v["event_merkle"].as_str().unwrap().parse().unwrap();

            let mut cb = [0u8; 26];
            cb[0..8].copy_from_slice(&ts_ms.to_le_bytes());
            cb[8..12].copy_from_slice(&tenant_id.to_le_bytes());
            cb[12] = sd;
            cb[13..16].copy_from_slice(&owl);
            cb[16] = action;
            cb[17] = decision;
            cb[18..26].copy_from_slice(&arh.to_le_bytes());

            let expected = AuditMerkleRoot::chain(prev, salt, &cb);
            assert_eq!(expected.raw(), event_merkle, "chain break at JSONL row {i}");
            prev = expected;
        }
    }

    /// JSONL format test: parse output, verify hex owl_identity, decimal-string u64s.
    #[test]
    fn jsonl_format_owl_hex_and_decimal_strings() {
        let owl = OwlIdentity::new(OgitFamily(7), 0x051c);
        let owl_bytes = owl.to_canonical_bytes(); // [7, 0x1c, 0x05]
        let expected_hex = format!(
            "{:02x}{:02x}{:02x}",
            owl_bytes[0], owl_bytes[1], owl_bytes[2]
        );

        let ev = {
            let mut chain = AuditChain::new(SuperDomain::Healthcare, 0);
            chain.advance(UnifiedAuditEvent {
                ts_unix_ms: 1_747_180_800_000,
                tenant: TenantId(42),
                super_domain: SuperDomain::Healthcare,
                owl,
                op: AuthOp::Write,
                decision: AuthDecision::Deny,
                actor_role_hash: 0xDEAD_CAFE_BABE_0042,
                merkle_root: AuditMerkleRoot::GENESIS,
                prev_merkle: AuditMerkleRoot::GENESIS,
            })
        };

        let line = serialize_event(&ev).expect("serialize");
        let v: serde_json::Value = serde_json::from_str(&line).unwrap();

        // owl_identity must be 6-char lowercase hex.
        let owl_field = v["owl_identity"].as_str().unwrap();
        assert_eq!(owl_field.len(), 6, "owl_identity must be 6 chars");
        assert!(
            owl_field
                .chars()
                .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
            "owl_identity must be lowercase hex"
        );
        assert_eq!(owl_field, expected_hex);

        // timestamp_us must be a decimal string, not a JSON number.
        assert!(
            v["timestamp_us"].is_string(),
            "timestamp_us must be a string"
        );
        let ts: u64 = v["timestamp_us"].as_str().unwrap().parse().unwrap();
        assert_eq!(ts, ev.ts_unix_ms * 1000);

        // actor_role_hash, prev_merkle, event_merkle must be decimal strings.
        assert!(
            v["actor_role_hash"].is_string(),
            "actor_role_hash must be string"
        );
        assert!(v["prev_merkle"].is_string(), "prev_merkle must be string");
        assert!(v["event_merkle"].is_string(), "event_merkle must be string");

        // family_id must match owl_identity first byte.
        let family_id = v["family_id"].as_u64().unwrap() as u8;
        assert_eq!(
            family_id, owl_bytes[0],
            "family_id must match owl_identity[0]"
        );

        // payload must be null.
        assert!(v["payload"].is_null(), "payload must be null");
    }

    /// Checkpoint writes and reads back correctly.
    #[test]
    fn jsonl_checkpoint_atomic() {
        let dir = tempdir();
        let sink = JsonlAuditSink::new(&dir).unwrap();
        let events = make_events(3, 0x1234);
        for ev in &events {
            sink.emit(*ev).unwrap();
        }
        sink.flush().unwrap();
        sink.checkpoint().unwrap();

        let cp_path = dir.join("audit/_checkpoint.json");
        assert!(cp_path.exists(), "_checkpoint.json should exist");
        let content = std::fs::read_to_string(&cp_path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&content).unwrap();
        let root: u64 = v["last_merkle_root"].as_str().unwrap().parse().unwrap();
        assert_eq!(root, events.last().unwrap().merkle_root.raw());
    }

    fn parse_owl_hex(hex: &str) -> [u8; 3] {
        [
            u8::from_str_radix(&hex[0..2], 16).unwrap(),
            u8::from_str_radix(&hex[2..4], 16).unwrap(),
            u8::from_str_radix(&hex[4..6], 16).unwrap(),
        ]
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// LanceAuditSink tests (feature = "lance-sink")
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(feature = "lance-sink")]
mod lance_tests {
    use super::*;
    use arrow_schema::{DataType, Schema};
    use lance_graph_callcenter::audit_sink::lance_sink::audit_event_schema;
    use lance_graph_callcenter::LanceAuditSink;

    /// Verify the canonical Arrow schema has 12 columns with the right types.
    #[test]
    fn lance_schema_column_names_and_types() {
        let schema = audit_event_schema();
        let expected: &[(&str, DataType)] = &[
            ("timestamp_us", DataType::UInt64),
            ("tenant_id", DataType::UInt32),
            ("super_domain", DataType::UInt8),
            ("family_id", DataType::UInt8),
            ("owl_identity", DataType::FixedSizeBinary(3)),
            ("action", DataType::UInt8),
            ("decision", DataType::UInt8),
            ("actor_role_hash", DataType::UInt64),
            ("prev_merkle", DataType::UInt64),
            ("event_merkle", DataType::UInt64),
            ("payload", DataType::Binary),
            ("date_partition", DataType::Utf8),
        ];
        assert_eq!(schema.fields().len(), 12, "schema must have 12 columns");
        for (name, dtype) in expected {
            let field = schema
                .field_with_name(name)
                .unwrap_or_else(|_| panic!("column {name} not found in schema"));
            assert_eq!(field.data_type(), dtype, "column {name} has wrong type");
        }
        // payload must be nullable.
        let payload = schema.field_with_name("payload").unwrap();
        assert!(payload.is_nullable(), "payload must be nullable");
    }

    /// Write batch to Lance, read back via Lance reader, verify column presence.
    #[test]
    fn lance_write_and_read_back_schema() {
        use futures::TryStreamExt as _;

        let dir = tempdir();
        let sink = LanceAuditSink::new(&dir).expect("create LanceAuditSink");

        let events = make_events(10, 0xBEEF);
        for ev in &events {
            sink.emit(*ev).expect("emit");
        }
        sink.flush().expect("flush");

        // Find the lance partition directory.
        let audit_dir = dir.join("audit");
        let schema = audit_event_schema();

        // Walk partitions and read back using a fresh tokio runtime (not inside
        // a tokio::test runtime, so no nested runtime problem).
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();

        let total_rows = rt.block_on(async {
            let mut total_rows = 0usize;
            for sd_entry in std::fs::read_dir(&audit_dir).unwrap().flatten() {
                let sd_path = sd_entry.path();
                if !sd_path.is_dir() {
                    continue;
                }
                for date_entry in std::fs::read_dir(&sd_path).unwrap().flatten() {
                    let date_path = date_entry.path();
                    if !date_path.is_dir() {
                        continue;
                    }
                    let uri = format!("file://{}", date_path.display());
                    let dataset = match lance::dataset::Dataset::open(&uri).await {
                        Ok(ds) => ds,
                        Err(_) => continue,
                    };
                    let stream = dataset.scan().try_into_stream().await.unwrap();
                    let batches: Vec<arrow_array::RecordBatch> =
                        stream.try_collect().await.unwrap();

                    for batch in &batches {
                        // Verify all expected columns are present.
                        for field in schema.fields() {
                            assert!(
                                batch.column_by_name(field.name()).is_some(),
                                "column {} missing in Lance batch",
                                field.name()
                            );
                        }
                        total_rows += batch.num_rows();
                    }
                }
            }
            total_rows
        });

        assert_eq!(total_rows, 10, "all 10 events should be persisted in Lance");
    }

    /// Round-trip 100 events, verify merkle root from flush == last event's root.
    #[test]
    fn lance_round_trip_100_events() {
        let dir = tempdir();
        let sink = LanceAuditSink::new(&dir).unwrap();
        let events = make_events(100, 0xABCD);
        for ev in &events {
            sink.emit(*ev).unwrap();
        }
        let root = sink.flush().unwrap();
        assert_eq!(root, events.last().unwrap().merkle_root.raw());
    }

    /// Buffer full returns ChannelFull error.
    #[test]
    fn lance_buffer_full_returns_channel_full() {
        use lance_graph_callcenter::audit_sink::lance_sink::LANCE_BUFFER_CAPACITY;
        let dir = tempdir();
        let sink = LanceAuditSink::new(&dir).unwrap();
        let events = make_events(LANCE_BUFFER_CAPACITY + 1, 0);

        let mut full_err = false;
        for ev in &events {
            match sink.emit(*ev) {
                Ok(()) => {}
                Err(AuditError::ChannelFull(_)) => {
                    full_err = true;
                    break;
                }
                Err(e) => panic!("unexpected error: {e}"),
            }
        }
        assert!(full_err, "should get ChannelFull when buffer is full");
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// CompositeSink tests (always compiled — no feature requirement)
// ══════════════════════════════════════════════════════════════════════════════

mod composite_tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// Recording sink for testing.
    struct RecordingSink {
        events: Arc<Mutex<Vec<UnifiedAuditEvent>>>,
        fail: bool,
    }

    #[allow(dead_code)] // helpers retained for future test scenarios
    impl RecordingSink {
        fn new() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
                fail: false,
            }
        }
        fn new_failing() -> Self {
            Self {
                events: Arc::new(Mutex::new(Vec::new())),
                fail: true,
            }
        }
        fn count(&self) -> usize {
            self.events.lock().unwrap().len()
        }
    }

    impl AuditSink for RecordingSink {
        fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
            if self.fail {
                return Err(AuditError::ChannelFull("simulated failure".into()));
            }
            self.events.lock().unwrap().push(event);
            Ok(())
        }
        fn flush(&self) -> Result<u64, AuditError> {
            if self.fail {
                return Err(AuditError::ChannelFull("simulated flush failure".into()));
            }
            Ok(0)
        }
        fn checkpoint(&self) -> Result<(), AuditError> {
            if self.fail {
                return Err(AuditError::ChannelFull(
                    "simulated checkpoint failure".into(),
                ));
            }
            Ok(())
        }
    }

    /// BestEffort: when sink[0] fails, sink[1] still receives all events.
    #[test]
    fn composite_best_effort_one_fails_other_receives_all() {
        let good_events = Arc::new(Mutex::new(Vec::new()));
        let good_events_clone = Arc::clone(&good_events);

        struct GoodSink(Arc<Mutex<Vec<UnifiedAuditEvent>>>);
        impl AuditSink for GoodSink {
            fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
                self.0.lock().unwrap().push(event);
                Ok(())
            }
            fn flush(&self) -> Result<u64, AuditError> {
                Ok(0)
            }
            fn checkpoint(&self) -> Result<(), AuditError> {
                Ok(())
            }
        }

        let sink = CompositeSink::new(
            vec![
                Box::new(RecordingSink::new_failing()),
                Box::new(GoodSink(good_events_clone)),
            ],
            FanoutMode::BestEffort,
        );

        let events = make_events(5, 0x99);
        for ev in &events {
            // BestEffort: first error is returned but good sink still called.
            let _ = sink.emit(*ev);
        }

        let received = good_events.lock().unwrap().len();
        assert_eq!(
            received, 5,
            "good sink must receive all 5 events even when bad sink fails"
        );
    }

    /// FailFast: when sink[0] fails, sink[1] is NOT called.
    #[test]
    fn composite_fail_fast_aborts_on_first_error() {
        let good_events = Arc::new(Mutex::new(Vec::<UnifiedAuditEvent>::new()));
        let good_events_clone = Arc::clone(&good_events);

        struct GoodSink2(Arc<Mutex<Vec<UnifiedAuditEvent>>>);
        impl AuditSink for GoodSink2 {
            fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
                self.0.lock().unwrap().push(event);
                Ok(())
            }
            fn flush(&self) -> Result<u64, AuditError> {
                Ok(0)
            }
            fn checkpoint(&self) -> Result<(), AuditError> {
                Ok(())
            }
        }

        let sink = CompositeSink::new(
            vec![
                Box::new(RecordingSink::new_failing()),
                Box::new(GoodSink2(good_events_clone)),
            ],
            FanoutMode::FailFast,
        );

        let ev = make_events(1, 0)[0];
        let result = sink.emit(ev);
        assert!(result.is_err(), "FailFast should propagate the error");
        let received = good_events.lock().unwrap().len();
        assert_eq!(
            received, 0,
            "FailFast should not call subsequent sinks after error"
        );
    }

    /// BestEffort: two good sinks — both receive events in declaration order.
    #[test]
    fn composite_best_effort_preserves_ordering() {
        let events_a = Arc::new(Mutex::new(Vec::new()));
        let events_b = Arc::new(Mutex::new(Vec::new()));

        struct OrderSink(Arc<Mutex<Vec<u64>>>);
        impl AuditSink for OrderSink {
            fn emit(&self, event: UnifiedAuditEvent) -> Result<(), AuditError> {
                self.0.lock().unwrap().push(event.ts_unix_ms);
                Ok(())
            }
            fn flush(&self) -> Result<u64, AuditError> {
                Ok(0)
            }
            fn checkpoint(&self) -> Result<(), AuditError> {
                Ok(())
            }
        }

        let sink = CompositeSink::new(
            vec![
                Box::new(OrderSink(Arc::clone(&events_a))),
                Box::new(OrderSink(Arc::clone(&events_b))),
            ],
            FanoutMode::BestEffort,
        );

        let events = make_events(10, 0);
        for ev in &events {
            sink.emit(*ev).unwrap();
        }

        let a = events_a.lock().unwrap().clone();
        let b = events_b.lock().unwrap().clone();
        assert_eq!(a, b, "both sinks must receive events in same order");
        assert_eq!(a.len(), 10);
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Cross-verify: write same merkle chain to both sinks
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(all(feature = "lance-sink", feature = "jsonl"))]
mod cross_verify_tests {
    use super::*;
    use lance_graph_callcenter::audit_sink::jsonl_sink::serialize_event;
    use lance_graph_callcenter::{JsonlAuditSink, LanceAuditSink};

    /// Write the same 20-event chain to both sinks and verify that all
    /// event_merkle values agree between JSONL and Lance.
    #[test]
    fn cross_verify_same_chain_agrees() {
        use arrow_array::UInt64Array;
        use futures::TryStreamExt as _;

        let dir = tempdir();
        let jsonl_sink = JsonlAuditSink::new(&dir).unwrap();
        let lance_sink = LanceAuditSink::new(&dir).unwrap();

        let events = make_events(20, 0xDEAD_BEEF);

        for ev in &events {
            jsonl_sink.emit(*ev).unwrap();
            lance_sink.emit(*ev).unwrap();
        }
        jsonl_sink.flush().unwrap();
        lance_sink.flush().unwrap();

        // Collect JSONL event_merkle values.
        let jsonl_merkles: Vec<u64> = {
            let mut v = Vec::new();
            for path in find_jsonl_files(&dir) {
                let content = std::fs::read_to_string(&path).unwrap();
                for line in content.lines() {
                    if line.trim().is_empty() {
                        continue;
                    }
                    let rec: serde_json::Value = serde_json::from_str(line).unwrap();
                    let em: u64 = rec["event_merkle"].as_str().unwrap().parse().unwrap();
                    v.push(em);
                }
            }
            v
        };

        // Collect Lance event_merkle values using a fresh tokio runtime.
        let audit_dir = dir.join("audit");
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let lance_merkles: Vec<u64> = rt.block_on(async {
            let mut v = Vec::new();
            for sd_entry in std::fs::read_dir(&audit_dir).unwrap().flatten() {
                let sd_path = sd_entry.path();
                if !sd_path.is_dir() {
                    continue;
                }
                for date_entry in std::fs::read_dir(&sd_path).unwrap().flatten() {
                    let date_path = date_entry.path();
                    if !date_path.is_dir() {
                        continue;
                    }
                    let uri = format!("file://{}", date_path.display());
                    let dataset = match lance::dataset::Dataset::open(&uri).await {
                        Ok(ds) => ds,
                        Err(_) => continue,
                    };
                    let stream = dataset.scan().try_into_stream().await.unwrap();
                    let batches: Vec<arrow_array::RecordBatch> =
                        stream.try_collect().await.unwrap();
                    for batch in &batches {
                        let em_col = batch
                            .column_by_name("event_merkle")
                            .and_then(|c| c.as_any().downcast_ref::<UInt64Array>())
                            .expect("event_merkle column");
                        for i in 0..em_col.len() {
                            v.push(em_col.value(i));
                        }
                    }
                }
            }
            v
        });

        // Both should have 20 events.
        assert_eq!(jsonl_merkles.len(), 20, "JSONL must have 20 events");
        assert_eq!(lance_merkles.len(), 20, "Lance must have 20 events");

        // Sorted sets should be equal (partition ordering may differ).
        let mut js = jsonl_merkles.clone();
        js.sort_unstable();
        let mut ls = lance_merkles.clone();
        ls.sort_unstable();
        assert_eq!(js, ls, "JSONL and Lance merkle sets must agree");
    }
}

// ── Test utilities (reserved for future cross-verify tests) ──────────────────
#[allow(dead_code)]
fn tempdir() -> std::path::PathBuf {
    let tmp = std::env::temp_dir().join(format!(
        "audit_sink_test_{}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos(),
        std::process::id(),
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    tmp
}

#[allow(dead_code)]
fn find_jsonl_files(base: &std::path::Path) -> Vec<std::path::PathBuf> {
    let audit_dir = base.join("audit");
    let mut files = Vec::new();
    collect_jsonl_recursive(&audit_dir, &mut files);
    files.sort();
    files
}

#[allow(dead_code)]
fn collect_jsonl_recursive(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
    let Ok(rd) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in rd.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_jsonl_recursive(&path, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
            out.push(path);
        }
    }
}
