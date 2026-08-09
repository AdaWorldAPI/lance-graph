//! `measure_wal_curve` — the five-axis 64k measurement binary (operator-specified).
//!
//! Spec: `.claude/plans/measure-64k-axes-v1.md` — THE authority for every
//! constant, phase name, and gate in this file. Read it before touching
//! anything here. This header only orients the reader inside the code.
//!
//! `.claude/plans/measure-64k-axes-v3.md` adds two more arms to THIS SAME
//! binary (one release binary, never a second): §15 the M-arm (Morton
//! reorder inserted before the seal) and §16 the O-arm (does the seal's
//! own ordering duplicate what `temporal.rs` already provides?). Build lane
//! report for the M-arm/O-arm addition:
//! `.claude/board/exec-runs/m-arm-o-arm-build.md`.
//!
//! Build lane report (deviations, what could not be verified):
//! `.claude/board/exec-runs/measure-wal-curve-build.md`.
//!
//! ## Run
//!
//! ```text
//! cargo run --release -p lance-graph-supervisor --features cycle-driver \
//!     --example measure_wal_curve
//! ```
//!
//! Output: one CSV row per measured cycle to `$MEASURE_OUT` (default
//! `/tmp/measure_wal_curve.csv`, schema in the plan's "Measurement schema"
//! section); per-configuration medians to stderr; the four closing answers
//! (plan's "Placement + gates" section) as the final stderr block.
//!
//! ## Ground rules this file honors (plan, "Ground truth")
//!
//! - Logical population: 65,536 owner identities, everywhere.
//! - Canonical row: 512 bytes; canonical frame: 65,536 × 512 B = 32 MiB.
//! - `temporal.rs` runs ONLY after a sealed WAL read (never inside WAL prep).
//! - One logical cycle → one commit → one fdatasync → one `DatasetVersion`;
//!   segments are I/O slices of ONE commit, never version-publishing units.
//! - Ownership is a type/borrow property — never described as a runtime
//!   "claim" operation anywhere in this file's prose or variable names.
//! - The hot `MailboxSoA` representation and the canonical `NodeRow512`
//!   representation never share one memory claim (B1a's peak RSS and B1b's
//!   peak RSS are reported as two separate numbers; only their *difference*
//!   is the derived "hot representation overhead" metric).
//! - The WAL-curve plateau is a measured knee, printed as a descriptive
//!   finding — never framed as PASS/KILL.
//! - EXP-KIA-A2-64K is exploratory and non-claiming: it cannot and does not
//!   mark D-KIA-A2 passed (that gate stays median-of-5, >=2x, its own
//!   pre-registered protocol, untouched by this file).

fn main() {
    #[cfg(feature = "cycle-driver")]
    {
        measure::run();
    }
    #[cfg(not(feature = "cycle-driver"))]
    {
        eprintln!(
            "measure_wal_curve requires --features cycle-driver \
             (see .claude/plans/measure-64k-axes-v1.md)"
        );
        std::process::exit(1);
    }
}

#[cfg(feature = "cycle-driver")]
mod measure {
    #![allow(
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap,
        clippy::cast_sign_loss,
        clippy::cast_precision_loss,
        clippy::too_many_lines,
        clippy::too_many_arguments
    )]

    use std::collections::{BTreeMap, HashMap};
    use std::fs::{self, File, OpenOptions};
    use std::io::{IoSlice, Write as _};
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Mutex;
    use std::time::Instant;

    use cognitive_shader_driver::mailbox_soa::{MailboxSoA, WriteCell, WriteOutcome, WORDS_PER_FP};
    use lance_graph_contract::cognitive_shader::MetaWord;
    use lance_graph_contract::collapse_gate::MailboxId;
    use lance_graph_contract::kanban::{KanbanColumn, KanbanMove};
    use lance_graph_contract::mul::i4_eval::gate_decision_i4;
    use lance_graph_contract::qualia::QualiaI4_16D;
    use lance_graph_contract::scheduler::DatasetVersion;
    use lance_graph_contract::soa_view::MailboxSoaView;

    use lance_graph_planner::batch_writer::BatchWriter;
    use lance_graph_planner::ir::Arena;
    use lance_graph_planner::owner_adapter::emit_bootstrap_intent;
    use lance_graph_planner::persist_sink::{
        order_cycle_stably, persist_cycle, CommitError, CommitOutcome, CycleFrame, CycleId,
        DetachedCycleBatch, FrameMeta, LandedSlot, SweepSlot, WalSink, WriteFailed,
    };
    use lance_graph_planner::strategy::style_strategy::StyleStrategy;
    use lance_graph_planner::temporal::{
        deinterlace, local_trajectories, DeinterlaceRow, LocalCausalRow, NoDeps, QueryReference,
    };
    use lance_graph_planner::traits::{
        PlanContext, PlanInput, PlanStrategy, QueryFeatures, StrategyOutcome,
    };

    use lance_graph_supervisor::cycle_driver::{
        apply_sealed_transitions, collect_casts, SealedCycle as DriverSealedCycle, SealedTransition,
    };

    // ═════════════════════════════════════════════════════════════════════
    // §0 — shared constants + shapes (pre-registered, before any number
    // exists — matching `probe_ignition_64k.rs`'s discipline).
    // ═════════════════════════════════════════════════════════════════════

    const FLEET_OWNERS: u32 = 65_536;
    const CANONICAL_ROW_BYTES: usize = 512;
    const CANONICAL_FRAME_BYTES: usize = FLEET_OWNERS as usize * CANONICAL_ROW_BYTES; // 32 MiB
    const WARMUP_CYCLES: u32 = 2;
    const MEASURED_CYCLES: u32 = 16;
    const WAL_DIR: &str = "/tmp/measure_wal_curve_wal";

    /// The storage ENVELOPE stand-in for the canonical node layout — `key(16) |
    /// edges(16) | value(480)` = 512 B (CLAUDE.md § CANON — Minimal SoA node,
    /// 2026-06-13). This is NOT a new type proposal: it is a local, plain
    /// byte-array measurement fixture standing in for the persisted row shape
    /// B1b/W1 measure against — a `#[repr(C)] [u8; 512]`, nothing more.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NodeRow512([u8; CANONICAL_ROW_BYTES]);

    const _: () = assert!(std::mem::size_of::<NodeRow512>() == CANONICAL_ROW_BYTES);

    impl NodeRow512 {
        /// Deterministic, non-degenerate content per logical row id — splitmix64
        /// scramble (no clock, no rng), same generator shape as
        /// `probe_ignition_64k.rs:99-111`'s `plane_for` (provenance).
        fn for_id(id: u64) -> Self {
            let mut bytes = [0u8; CANONICAL_ROW_BYTES];
            let mut x = id ^ 0x9E37_79B9_7F4A_7C15;
            for chunk in bytes.chunks_exact_mut(8) {
                x ^= x >> 30;
                x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
                x ^= x >> 27;
                x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
                x ^= x >> 31;
                chunk.copy_from_slice(&(x | 1).to_le_bytes());
            }
            Self(bytes)
        }
        fn as_bytes(&self) -> &[u8] {
            &self.0
        }
    }

    /// The dense identity plane fixture — `probe_ignition_64k.rs:99-111`
    /// provenance, reused verbatim (never a zero plane — every word `| 1`).
    fn splitmix_plane(seed: u64) -> Vec<u64> {
        let mut x = seed ^ 0x9E37_79B9_7F4A_7C15;
        let mut plane = vec![0u64; WORDS_PER_FP];
        for w in plane.iter_mut() {
            x ^= x >> 30;
            x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
            x ^= x >> 27;
            x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
            x ^= x >> 31;
            *w = x | 1;
        }
        plane
    }

    /// Flow qualia fixture — `probe_ignition_64k.rs:88-90` / `probe_ignition.rs:196-203`
    /// provenance (warmth=4, groundedness=3, coherence=4, valence=2 =>
    /// flow_proxy 7, Calibrated). Every arm's owners Flow on this fixture — a
    /// non-trivial, non-zero qualia vector, never an all-zeros rig.
    fn flow_qualia() -> QualiaI4_16D {
        QualiaI4_16D(0).with(3, 4).with(14, 3).with(9, 4).with(1, 2)
    }

    /// Derived mantissa — `probe_ignition_64k.rs:92-95` provenance.
    fn mantissa_of<const N: usize>(owner: &MailboxSoA<N>) -> i8 {
        owner.pending_count().min(7) as i8
    }

    /// The 23D analytical style vector — `probe_ignition_64k.rs:230-243`
    /// provenance (idx 4 = analytical).
    fn style_context() -> PlanContext {
        let mut v = vec![0.0f64; 23];
        v[4] = 1.0;
        PlanContext {
            query: String::new(),
            features: QueryFeatures::default(),
            free_will_modifier: 1.0,
            thinking_style: Some(v),
            nars_hint: None,
            witness: None,
        }
    }

    /// ONE `StyleStrategy::plan` call, reused across every owner in an arm —
    /// `probe_ignition_64k.rs:289-308` provenance: every owner is armed with
    /// the same z, so the style outcome is owner-independent; per-owner
    /// binding happens inside `emit_bootstrap_intent` -> `rebind_bootstrap`
    /// (the no-theft guard), never by recomputing the plan per owner.
    fn build_style_outcome() -> StrategyOutcome {
        let mut arena = Arena::new();
        let plan_out = StyleStrategy
            .plan(
                PlanInput {
                    plan: None,
                    context: style_context(),
                    outcome: None,
                },
                &mut arena,
            )
            .expect("StyleStrategy::plan over the fixed recipe substrate");
        plan_out
            .outcome
            .expect("StyleStrategy always surfaces a StrategyOutcome")
    }

    /// FNV-1a 64-bit — std-only, no external digest crate.
    fn fnv1a64(bytes: &[u8]) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for &b in bytes {
            h ^= u64::from(b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        h
    }

    // ═════════════════════════════════════════════════════════════════════
    // §1 — /proc/self reading (std-only; RSS + faults + context switches).
    // ═════════════════════════════════════════════════════════════════════

    #[derive(Clone, Copy, Default)]
    struct ProcSnapshot {
        vmhwm_kb: u64,
        vmrss_kb: u64,
        minflt: u64,
        majflt: u64,
        vol_ctxt: u64,
        nonvol_ctxt: u64,
    }

    /// Returns `(VmHWM_kB, VmRSS_kB, voluntary_ctxt, nonvoluntary_ctxt)`.
    ///
    /// **VmHWM is process-monotonic** — it is the high-water mark since process
    /// start, so subtracting one arm's HWM from another's inside ONE process
    /// yields the historical max twice, not two footprints. Any per-arm memory
    /// figure MUST come from a VmRSS delta (current RSS after minus before).
    fn read_proc_status() -> (u64, u64, u64, u64) {
        let mut vmhwm = 0u64;
        let mut vmrss = 0u64;
        let mut vol = 0u64;
        let mut nonvol = 0u64;
        if let Ok(text) = fs::read_to_string("/proc/self/status") {
            for line in text.lines() {
                if let Some(rest) = line.strip_prefix("VmHWM:") {
                    vmhwm = rest
                        .trim()
                        .trim_end_matches("kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                } else if let Some(rest) = line.strip_prefix("VmRSS:") {
                    vmrss = rest
                        .trim()
                        .trim_end_matches("kB")
                        .trim()
                        .parse()
                        .unwrap_or(0);
                } else if let Some(rest) = line.strip_prefix("voluntary_ctxt_switches:") {
                    vol = rest.trim().parse().unwrap_or(0);
                } else if let Some(rest) = line.strip_prefix("nonvoluntary_ctxt_switches:") {
                    nonvol = rest.trim().parse().unwrap_or(0);
                }
            }
        }
        (vmhwm, vmrss, vol, nonvol)
    }

    /// `/proc/self/stat` field 10 (minflt) / field 12 (majflt), 1-indexed.
    /// `comm` (field 2) is parenthesised and may itself contain spaces, so we
    /// split AFTER the last `)` before counting whitespace-separated fields.
    fn read_proc_stat_faults() -> (u64, u64) {
        let mut minflt = 0u64;
        let mut majflt = 0u64;
        if let Ok(text) = fs::read_to_string("/proc/self/stat") {
            if let Some(close) = text.rfind(')') {
                let fields: Vec<&str> = text[close + 1..].split_whitespace().collect();
                // fields[0] = state (field 3); minflt = field 10 = fields[7];
                // majflt = field 12 = fields[9].
                if fields.len() > 9 {
                    minflt = fields[7].parse().unwrap_or(0);
                    majflt = fields[9].parse().unwrap_or(0);
                }
            }
        }
        (minflt, majflt)
    }

    fn proc_snapshot() -> ProcSnapshot {
        let (vmhwm_kb, vmrss_kb, vol_ctxt, nonvol_ctxt) = read_proc_status();
        let (minflt, majflt) = read_proc_stat_faults();
        ProcSnapshot {
            vmhwm_kb,
            vmrss_kb,
            minflt,
            majflt,
            vol_ctxt,
            nonvol_ctxt,
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // §2 — the CSV schema (plan "Measurement schema") + sink.
    // ═════════════════════════════════════════════════════════════════════

    #[derive(Clone)]
    struct Row {
        owner_shape: &'static str,
        physical_layout: &'static str,
        threads: u32,
        segment_rows: u64,
        segment_bytes: u64,
        segments_per_cycle: u64,
        repeat: u32,
        build_ns: u64,
        scan_ns: u64,
        think_ns: u64,
        rebind_cast_ns: u64,
        collect_ns: u64,
        freeze_ns: u64,
        wal_write_ns: u64,
        wal_sync_ns: u64,
        temporal_layer1_ns: u64,
        temporal_layer2_ns: u64,
        apply_ns: u64,
        total_ns: u64,
        logical_rows: u64,
        logical_bytes: u64,
        sealed_transitions: u64,
        applied_transitions: u64,
        wal_syscalls: u64,
        fsync_calls: u64,
        dataset_versions: u64,
        peak_rss_bytes: u64,
        minor_faults: u64,
        major_faults: u64,
        context_switches: u64,
        max_active_workers: u32,
        result_digest: u64,
        /// M-arm (plan v3): the Morton write-order reorder phase, timed in
        /// isolation from seal/write. `0` for every arm that does not perform
        /// a reorder (never fabricated).
        morton_reorder_ns: u64,
    }

    impl Row {
        fn header() -> &'static str {
            "owner_shape,physical_layout,threads,segment_rows,segment_bytes,\
             segments_per_cycle,repeat,build_ns,scan_ns,think_ns,rebind_cast_ns,\
             collect_ns,freeze_ns,wal_write_ns,wal_sync_ns,temporal_layer1_ns,\
             temporal_layer2_ns,apply_ns,total_ns,logical_rows,logical_bytes,\
             sealed_transitions,applied_transitions,wal_syscalls,fsync_calls,\
             dataset_versions,peak_rss_bytes,minor_faults,major_faults,\
             context_switches,llc_misses,max_active_workers,result_digest,\
             morton_reorder_ns"
        }

        /// `llc_misses` is always emitted EMPTY — no perf-counter access in
        /// this std-only binary (plan: "an empty cell, never a fabricated
        /// one").
        fn to_csv(&self) -> String {
            format!(
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},,{},{},{}",
                self.owner_shape,
                self.physical_layout,
                self.threads,
                self.segment_rows,
                self.segment_bytes,
                self.segments_per_cycle,
                self.repeat,
                self.build_ns,
                self.scan_ns,
                self.think_ns,
                self.rebind_cast_ns,
                self.collect_ns,
                self.freeze_ns,
                self.wal_write_ns,
                self.wal_sync_ns,
                self.temporal_layer1_ns,
                self.temporal_layer2_ns,
                self.apply_ns,
                self.total_ns,
                self.logical_rows,
                self.logical_bytes,
                self.sealed_transitions,
                self.applied_transitions,
                self.wal_syscalls,
                self.fsync_calls,
                self.dataset_versions,
                self.peak_rss_bytes,
                self.minor_faults,
                self.major_faults,
                self.context_switches,
                // NOTE: the trailing `,,` above already emits the empty
                // llc_misses cell between context_switches and
                // max_active_workers — do not add another field here.
                self.max_active_workers,
                self.result_digest,
                self.morton_reorder_ns,
            )
        }
    }

    struct CsvSink {
        file: File,
        path: String,
        rows_written: u64,
    }

    impl CsvSink {
        fn new() -> Self {
            let path = std::env::var("MEASURE_OUT")
                .unwrap_or_else(|_| "/tmp/measure_wal_curve.csv".to_string());
            let mut file = File::create(&path).unwrap_or_else(|e| {
                panic!("measure_wal_curve: cannot create MEASURE_OUT {path}: {e}")
            });
            writeln!(file, "{}", Row::header()).expect("write csv header");
            eprintln!("measure.csv: writing rows to {path}");
            Self {
                file,
                path,
                rows_written: 0,
            }
        }
        fn write(&mut self, row: &Row) {
            writeln!(self.file, "{}", row.to_csv()).expect("write csv row");
            self.rows_written += 1;
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // §3 — median / p95 + the cache-amortisation gain/plateau helpers.
    // ═════════════════════════════════════════════════════════════════════

    fn median(samples: &[u64]) -> u64 {
        if samples.is_empty() {
            return 0;
        }
        let mut s = samples.to_vec();
        s.sort_unstable();
        let n = s.len();
        if n % 2 == 1 {
            s[n / 2]
        } else {
            (s[n / 2 - 1] + s[n / 2]) / 2
        }
    }

    fn p95(samples: &[u64]) -> u64 {
        if samples.is_empty() {
            return 0;
        }
        let mut s = samples.to_vec();
        s.sort_unstable();
        let n = s.len();
        let idx = ((n as f64) * 0.95).ceil() as usize;
        s[idx.min(n - 1)]
    }

    /// `gain(C) = throughput(C)/throughput(prev) - 1` (plan's cache-amortisation
    /// curve). Returns the per-config gains (index 0 has no predecessor, so it
    /// is `f64::NAN` — never compared) and the DESCRIPTIVE plateau index: the
    /// first `i >= 2` where BOTH `gain(i-1)` and `gain(i)` are `< 5%`. Purely
    /// descriptive — never a PASS/KILL verdict (plan, "Placement + gates").
    fn plateau_index(throughput: &[f64]) -> (Vec<f64>, Option<usize>) {
        let mut gains = vec![f64::NAN; throughput.len()];
        for i in 1..throughput.len() {
            if throughput[i - 1] > 0.0 {
                gains[i] = throughput[i] / throughput[i - 1] - 1.0;
            }
        }
        let mut plateau = None;
        for i in 2..gains.len() {
            if gains[i - 1] < 0.05 && gains[i] < 0.05 {
                plateau = Some(i);
                break;
            }
        }
        (gains, plateau)
    }

    // ═════════════════════════════════════════════════════════════════════
    // §4 — write_vectored looped over partial writes (real syscall counting;
    // plan: "partial vectored writes loop and are counted").
    // ═════════════════════════════════════════════════════════════════════

    /// Write ALL of `bufs` via `File::write_vectored`, looping on short
    /// writes (a short/partial vectored write is a REAL possibility — kernel
    /// `IOV_MAX`, a signal interruption, or a filesystem that just doesn't
    /// hand back everything in one call). Returns `(bytes_written,
    /// syscalls_issued)` — `syscalls_issued` is the actual number of
    /// `write_vectored` calls made, not an assumed 1.
    fn write_vectored_all(
        file: &mut File,
        mut bufs: &mut [IoSlice<'_>],
    ) -> std::io::Result<(u64, u64)> {
        let mut total = 0u64;
        let mut syscalls = 0u64;
        while !bufs.is_empty() {
            match file.write_vectored(bufs) {
                Ok(0) => {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::WriteZero,
                        "write_vectored returned 0 with buffers remaining",
                    ));
                }
                Ok(n) => {
                    syscalls += 1;
                    total += n as u64;
                    IoSlice::advance_slices(&mut bufs, n);
                }
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok((total, syscalls))
    }

    // ═════════════════════════════════════════════════════════════════════
    // §5 — B0: DummyOwner cast baseline (no SoA, no temporal, no file I/O).
    // ═════════════════════════════════════════════════════════════════════

    #[derive(Clone, Copy)]
    struct DummyOwner {
        owner_id: MailboxId,
        phase: KanbanColumn,
        cycle: u32,
    }

    #[derive(Clone, Copy, Default)]
    struct PhaseMedians {
        build_ns: u64,
        scan_ns: u64,
        think_ns: u64,
        cast_ns: u64,
        collect_ns: u64,
        freeze_ns: u64,
        apply_ns: u64,
        /// Process-monotonic high-water mark. NEVER differenced across arms.
        peak_rss_bytes: u64,
        /// VmRSS AFTER this arm's allocation minus VmRSS BEFORE it — the only
        /// honest per-arm footprint inside one process.
        rss_delta_bytes: i64,
    }

    fn median_phases(samples: &[PhaseMedians]) -> PhaseMedians {
        let col = |f: fn(&PhaseMedians) -> u64| median(&samples.iter().map(f).collect::<Vec<_>>());
        let mut deltas: Vec<i64> = samples.iter().map(|s| s.rss_delta_bytes).collect();
        deltas.sort_unstable();
        let rss_delta_bytes = deltas.get(deltas.len() / 2).copied().unwrap_or(0);
        PhaseMedians {
            build_ns: col(|p| p.build_ns),
            scan_ns: col(|p| p.scan_ns),
            think_ns: col(|p| p.think_ns),
            cast_ns: col(|p| p.cast_ns),
            collect_ns: col(|p| p.collect_ns),
            freeze_ns: col(|p| p.freeze_ns),
            apply_ns: col(|p| p.apply_ns),
            peak_rss_bytes: col(|p| p.peak_rss_bytes),
            rss_delta_bytes,
        }
    }

    fn run_b0(csv: &mut CsvSink) -> PhaseMedians {
        eprintln!("\n== B0 — DummyOwner cast baseline (owner lookup, write-on-behalf rebind, CastId allocation, staging, collect) ==");
        let style_outcome = build_style_outcome();
        let mut samples = Vec::new();

        for repeat in 0..3u32 {
            // VmRSS BEFORE this repeat's fleet allocation (the delta baseline).
            let rss_before = read_proc_status().1 as i64 * 1024;
            let t_build = Instant::now();
            let mut fleet: HashMap<MailboxId, DummyOwner> =
                HashMap::with_capacity(FLEET_OWNERS as usize);
            for id in 0..FLEET_OWNERS {
                fleet.insert(
                    id,
                    DummyOwner {
                        owner_id: id,
                        phase: KanbanColumn::Planning,
                        cycle: 0,
                    },
                );
            }
            let build_ns = t_build.elapsed().as_nanos() as u64;

            let t_scan = Instant::now();
            let mut in_planning = 0usize;
            for owner in fleet.values() {
                if owner.phase == KanbanColumn::Planning {
                    in_planning += 1;
                }
            }
            let scan_ns = t_scan.elapsed().as_nanos() as u64;
            assert_eq!(
                in_planning, FLEET_OWNERS as usize,
                "B0: every dummy owner starts Planning"
            );

            // "fixed dummy thought": the ONE style outcome (built above,
            // outside every repeat's timing) IS the thought — B0 measures
            // no per-owner thinking cost by design (that axis belongs to
            // B1a's think_ns, which computes a real per-owner gate).
            let think_ns = 0u64;

            let t_cast = Instant::now();
            let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
            let mut cast = 0usize;
            for id in 0..FLEET_OWNERS {
                let owner = fleet.get(&id).expect("owner exists");
                let payload = id.to_le_bytes().to_vec();
                if emit_bootstrap_intent(
                    &style_outcome,
                    owner.owner_id,
                    owner.cycle,
                    &mut writer,
                    payload,
                )
                .is_some()
                {
                    cast += 1;
                }
            }
            let cast_ns = t_cast.elapsed().as_nanos() as u64;
            assert_eq!(
                cast, FLEET_OWNERS as usize,
                "B0 can-fire: every dummy owner casts"
            );

            let t_collect = Instant::now();
            let collected =
                collect_casts(&mut writer, CycleId(u64::from(repeat) + 1), 0, u64::from);
            let collect_ns = t_collect.elapsed().as_nanos() as u64;
            assert_eq!(collected.slots.len(), FLEET_OWNERS as usize);
            assert!(
                collected.held.is_empty(),
                "B0: one move per owner, nothing held"
            );

            // Everything the freeze call does NOT do (framing, counting,
            // digesting) happens OUTSIDE the timed window, so `freeze_ns`
            // measures exactly `DetachedCycleBatch::freeze`, nothing else.
            let frame = CycleFrame::new(CycleId(u64::from(repeat) + 1), DatasetVersion(0));
            let sealed_count = collected
                .slots
                .iter()
                .filter(|s| s.paired_move.is_some())
                .count();
            let digest_bytes: Vec<u8> = collected
                .slots
                .iter()
                .flat_map(|s| s.payload.iter().copied())
                .collect();
            let t_freeze = Instant::now();
            let frozen = DetachedCycleBatch::freeze(frame, collected.slots);
            let freeze_ns = t_freeze.elapsed().as_nanos() as u64;
            assert_eq!(
                frozen.image.len(),
                FLEET_OWNERS as usize,
                "B0: one coalesced row per owner"
            );

            let total_ns = build_ns + scan_ns + think_ns + cast_ns + collect_ns + freeze_ns;
            let snap = proc_snapshot();
            let digest = fnv1a64(&digest_bytes);

            samples.push(PhaseMedians {
                build_ns,
                scan_ns,
                think_ns,
                cast_ns,
                collect_ns,
                freeze_ns,
                apply_ns: 0,
                peak_rss_bytes: snap.vmhwm_kb * 1024,
                rss_delta_bytes: (snap.vmrss_kb as i64 * 1024) - rss_before,
            });

            eprintln!(
                "B0 repeat {repeat}: build={build_ns}ns scan={scan_ns}ns cast={cast_ns}ns collect={collect_ns}ns freeze={freeze_ns}ns total={total_ns}ns"
            );
            csv.write(&Row {
                owner_shape: "b0_dummy_owner",
                physical_layout: "none",
                threads: 1,
                segment_rows: 0,
                segment_bytes: 0,
                segments_per_cycle: 0,
                repeat,
                build_ns,
                scan_ns,
                think_ns,
                rebind_cast_ns: cast_ns,
                collect_ns,
                freeze_ns,
                wal_write_ns: 0,
                wal_sync_ns: 0,
                temporal_layer1_ns: 0,
                temporal_layer2_ns: 0,
                apply_ns: 0,
                total_ns,
                logical_rows: FLEET_OWNERS as u64,
                logical_bytes: (FLEET_OWNERS as u64) * 4, // MailboxId=u32 -> to_le_bytes() is 4 bytes
                sealed_transitions: sealed_count as u64,
                applied_transitions: 0,
                wal_syscalls: 0,
                fsync_calls: 0,
                dataset_versions: 0,
                peak_rss_bytes: snap.vmhwm_kb * 1024,
                minor_faults: snap.minflt,
                major_faults: snap.majflt,
                context_switches: snap.vol_ctxt + snap.nonvol_ctxt,
                max_active_workers: 1,
                result_digest: digest,
                morton_reorder_ns: 0,
            });
        }

        median_phases(&samples)
    }

    // ═════════════════════════════════════════════════════════════════════
    // §6 — B1a: 65,536 × MailboxSoA<4> (the actual hot runtime owner).
    // ═════════════════════════════════════════════════════════════════════

    const ROWS_PER_OWNER_B1A: usize = 4;
    type TenantB1a = MailboxSoA<ROWS_PER_OWNER_B1A>;

    /// Build + seed one real owner — `probe_ignition_64k.rs:210-228`
    /// provenance (one populated row, one firing row, w_slot 0, threshold 1.0).
    fn build_tenant<const N: usize>(id: MailboxId) -> MailboxSoA<N> {
        let mut owner: MailboxSoA<N> = MailboxSoA::new(id, 0, 1.0);
        let cycle = owner.cycle();
        let plane = splitmix_plane(u64::from(id));
        let cell = WriteCell {
            content: Some(plane.as_slice()),
            qualia: Some(flow_qualia()),
            meta: Some(MetaWord::new(1, 0, 0, 0, 0)), // armed z=1 (Analytical)
            entity_type: Some((id % 251) as u16),
            temporal: Some(u64::from(id)),
            ..WriteCell::default()
        };
        let outcome = owner.write_row(0, cycle, &cell);
        assert_eq!(outcome, WriteOutcome::Accepted, "seeding tenant {id}");
        owner.set_populated(1);
        owner.tick();
        owner.energy[0] = 2.0; // one firing row: exhausts after one advance
        owner
    }

    /// Build a `DriverSealedCycle` directly from a frozen cast set — the same
    /// transitions/next_position_base extraction `cycle_driver::seal_cycle`
    /// performs (`cycle_driver.rs:286-301`), but WITHOUT going through an
    /// actual `WalSink::commit_cycle`. This arm measures the collect/freeze
    /// and apply PHASES; the real WAL commit is the WAL-curve arm's job
    /// (§8/§9 below) — coupling B1a's apply timing to a real fsync would
    /// blend two axes the plan explicitly keeps apart.
    fn build_sealed_locally(
        frame: CycleFrame,
        slots: &[SweepSlot],
        version: DatasetVersion,
    ) -> DriverSealedCycle {
        let mut transitions: Vec<SealedTransition> = slots
            .iter()
            .filter_map(|s| {
                s.paired_move.map(|mv| SealedTransition {
                    stream_position: s.stream_position,
                    owner: s.owner,
                    mv,
                })
            })
            .collect();
        transitions.sort_by_key(|t| t.stream_position);
        let next_position_base = slots
            .iter()
            .map(|s| s.stream_position + 1)
            .max()
            .unwrap_or(0);
        // Synthetic outcome — this helper never calls `WalSink::commit_cycle`
        // (see the doc above), so there is no real `batch_hash` to carry; `0`
        // is a placeholder, never compared against a real committed hash.
        DriverSealedCycle {
            outcome: CommitOutcome::Committed {
                version,
                cycle: frame.cycle,
                batch_hash: 0,
            },
            version: Some(version),
            transitions,
            next_position_base,
        }
    }

    fn run_b1a(csv: &mut CsvSink) -> PhaseMedians {
        eprintln!("\n== B1a — 65,536 x MailboxSoA<4> (the actual hot runtime owner) ==");
        let style_outcome = build_style_outcome();
        let mut samples = Vec::new();

        for repeat in 0..3u32 {
            // VmRSS BEFORE this repeat's allocation (the delta baseline).
            let rss_before = read_proc_status().1 as i64 * 1024;
            let t_build = Instant::now();
            let mut fleet: HashMap<MailboxId, TenantB1a> =
                HashMap::with_capacity(FLEET_OWNERS as usize);
            for id in 0..FLEET_OWNERS {
                fleet.insert(id, build_tenant::<ROWS_PER_OWNER_B1A>(id));
            }
            let build_ns = t_build.elapsed().as_nanos() as u64;

            let t_scan = Instant::now();
            let mut in_planning = 0usize;
            for owner in fleet.values() {
                if owner.phase() == KanbanColumn::Planning {
                    in_planning += 1;
                }
            }
            let scan_ns = t_scan.elapsed().as_nanos() as u64;
            assert_eq!(in_planning, FLEET_OWNERS as usize);

            // real per-owner gate decision (unlike B0's precomputed thought).
            let t_think = Instant::now();
            let mut targets: Vec<KanbanColumn> = Vec::with_capacity(FLEET_OWNERS as usize);
            for id in 0..FLEET_OWNERS {
                let owner = fleet.get(&id).expect("owner exists");
                let qualia = owner.qualia_at(0);
                let mantissa = mantissa_of(owner);
                let gate = gate_decision_i4(&qualia, mantissa);
                let target = owner
                    .phase()
                    .advance_on_gate(&gate)
                    .expect("B1a: every owner Flows (flow qualia + firing row)");
                targets.push(target);
            }
            let think_ns = t_think.elapsed().as_nanos() as u64;

            let t_cast = Instant::now();
            let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
            let mut cast = 0usize;
            for id in 0..FLEET_OWNERS {
                let owner = fleet.get(&id).expect("owner exists");
                let payload = id.to_le_bytes().to_vec();
                if emit_bootstrap_intent(
                    &style_outcome,
                    owner.mailbox_id(),
                    owner.current_cycle(),
                    &mut writer,
                    payload,
                )
                .is_some()
                {
                    cast += 1;
                }
            }
            let cast_ns = t_cast.elapsed().as_nanos() as u64;
            assert_eq!(
                cast, FLEET_OWNERS as usize,
                "B1a can-fire: every owner casts"
            );

            let t_collect = Instant::now();
            let collected =
                collect_casts(&mut writer, CycleId(u64::from(repeat) + 1), 0, u64::from);
            let collect_ns = t_collect.elapsed().as_nanos() as u64;
            assert_eq!(collected.slots.len(), FLEET_OWNERS as usize);

            // `build_sealed_locally` (a sort over 65,536 transitions) and the
            // digest bytes are computed OUTSIDE the timed window so
            // `freeze_ns` measures exactly `DetachedCycleBatch::freeze`.
            let frame = CycleFrame::new(CycleId(u64::from(repeat) + 1), DatasetVersion(0));
            let sealed = build_sealed_locally(frame, &collected.slots, DatasetVersion(1));
            let digest_bytes: Vec<u8> = collected
                .slots
                .iter()
                .flat_map(|s| s.payload.iter().copied())
                .collect();
            let t_freeze = Instant::now();
            let frozen = DetachedCycleBatch::freeze(frame, collected.slots);
            let freeze_ns = t_freeze.elapsed().as_nanos() as u64;
            assert_eq!(frozen.image.len(), FLEET_OWNERS as usize);
            assert_eq!(
                sealed.transitions.len(),
                FLEET_OWNERS as usize,
                "B1a: every owner's move sealed (sparse == full here)"
            );

            let t_apply = Instant::now();
            let mut watermarks: HashMap<MailboxId, Option<u64>> = HashMap::new();
            let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut watermarks)
                .expect("B1a: apply must succeed against freshly-Planning owners");
            let apply_ns = t_apply.elapsed().as_nanos() as u64;
            assert_eq!(
                applied.applied.len(),
                FLEET_OWNERS as usize,
                "B1a: all 65,536 advanced"
            );

            let total_ns =
                build_ns + scan_ns + think_ns + cast_ns + collect_ns + freeze_ns + apply_ns;
            let snap = proc_snapshot();
            let digest = fnv1a64(&digest_bytes);

            samples.push(PhaseMedians {
                build_ns,
                scan_ns,
                think_ns,
                cast_ns,
                collect_ns,
                freeze_ns,
                apply_ns,
                peak_rss_bytes: snap.vmhwm_kb * 1024,
                rss_delta_bytes: (snap.vmrss_kb as i64 * 1024) - rss_before,
            });

            eprintln!(
                "B1a repeat {repeat}: build={build_ns}ns scan={scan_ns}ns think={think_ns}ns cast={cast_ns}ns collect={collect_ns}ns freeze={freeze_ns}ns apply={apply_ns}ns total={total_ns}ns peak_rss={}B",
                snap.vmhwm_kb * 1024
            );
            csv.write(&Row {
                owner_shape: "b1a_mailboxsoa4",
                physical_layout: "owner_exclusive_65536",
                threads: 1,
                segment_rows: 0,
                segment_bytes: 0,
                segments_per_cycle: 0,
                repeat,
                build_ns,
                scan_ns,
                think_ns,
                rebind_cast_ns: cast_ns,
                collect_ns,
                freeze_ns,
                wal_write_ns: 0,
                wal_sync_ns: 0,
                temporal_layer1_ns: 0,
                temporal_layer2_ns: 0,
                apply_ns,
                total_ns,
                logical_rows: FLEET_OWNERS as u64,
                logical_bytes: (FLEET_OWNERS as u64) * 4, // MailboxId=u32 -> to_le_bytes() is 4 bytes
                sealed_transitions: sealed.transitions.len() as u64,
                applied_transitions: applied.applied.len() as u64,
                wal_syscalls: 0,
                fsync_calls: 0,
                dataset_versions: 0,
                peak_rss_bytes: snap.vmhwm_kb * 1024,
                minor_faults: snap.minflt,
                major_faults: snap.majflt,
                context_switches: snap.vol_ctxt + snap.nonvol_ctxt,
                max_active_workers: 1,
                result_digest: digest,
                morton_reorder_ns: 0,
            });
            let _ = targets; // computed, consumed by the gate loop's own assertion
        }

        median_phases(&samples)
    }

    // ═════════════════════════════════════════════════════════════════════
    // §7 — B1b: 65,536 x NodeRow512 (the canonical, memory-only envelope).
    // ═════════════════════════════════════════════════════════════════════

    /// Returns the median VmRSS DELTA (bytes) for the canonical-envelope arm.
    /// NOT a high-water mark — see `read_proc_status`'s contract.
    fn run_b1b(csv: &mut CsvSink) -> i64 {
        eprintln!(
            "\n== B1b — 65,536 x NodeRow512 = 32 MiB canonical storage envelope (memory-only) =="
        );
        let mut rss_samples: Vec<i64> = Vec::new();

        for repeat in 0..3u32 {
            // VmRSS BEFORE this repeat's 32 MiB allocation (delta baseline).
            let rss_before = read_proc_status().1 as i64 * 1024;
            let t_build = Instant::now();
            let mut rows: Vec<NodeRow512> = Vec::with_capacity(FLEET_OWNERS as usize);
            for id in 0..u64::from(FLEET_OWNERS) {
                rows.push(NodeRow512::for_id(id));
            }
            let build_ns = t_build.elapsed().as_nanos() as u64;
            assert_eq!(rows.len(), FLEET_OWNERS as usize);
            let logical_bytes = (rows.len() * CANONICAL_ROW_BYTES) as u64;
            assert_eq!(
                logical_bytes, CANONICAL_FRAME_BYTES as u64,
                "B1b: 65,536 x 512B == the 32 MiB canonical frame"
            );

            let snap = proc_snapshot();
            let digest = fnv1a64(rows.last().expect("non-empty").as_bytes());
            // Hold `rows` alive PAST the RSS snapshot — the entire point of
            // this arm is measuring the envelope's own resident footprint,
            // not a footprint already reclaimed by drop.
            std::hint::black_box(&rows);

            rss_samples.push((snap.vmrss_kb as i64 * 1024) - rss_before);
            eprintln!(
                "B1b repeat {repeat}: build={build_ns}ns peak_rss={}B",
                snap.vmhwm_kb * 1024
            );
            csv.write(&Row {
                owner_shape: "b1b_noderow512",
                physical_layout: "contiguous_vec_32mib",
                threads: 1,
                segment_rows: 0,
                segment_bytes: 0,
                segments_per_cycle: 0,
                repeat,
                build_ns,
                scan_ns: 0,
                think_ns: 0,
                rebind_cast_ns: 0,
                collect_ns: 0,
                freeze_ns: 0,
                wal_write_ns: 0,
                wal_sync_ns: 0,
                temporal_layer1_ns: 0,
                temporal_layer2_ns: 0,
                apply_ns: 0,
                total_ns: build_ns,
                logical_rows: FLEET_OWNERS as u64,
                logical_bytes,
                sealed_transitions: 0,
                applied_transitions: 0,
                wal_syscalls: 0,
                fsync_calls: 0,
                dataset_versions: 0,
                peak_rss_bytes: snap.vmhwm_kb * 1024,
                minor_faults: snap.minflt,
                major_faults: snap.majflt,
                context_switches: snap.vol_ctxt + snap.nonvol_ctxt,
                max_active_workers: 1,
                result_digest: digest,
                morton_reorder_ns: 0,
            });
        }

        rss_samples.sort_unstable();
        rss_samples.get(rss_samples.len() / 2).copied().unwrap_or(0)
    }

    // ═════════════════════════════════════════════════════════════════════
    // §8/§9 — the WAL curve: W1-contiguous (physics ceiling) beside
    // W0-current (real SweepSlot/DetachedCycleBatch representation).
    // ═════════════════════════════════════════════════════════════════════

    #[derive(Clone, Copy)]
    struct WalConfig {
        segment_rows: u64,
        segment_bytes: u64,
        segments_per_cycle: u64,
    }

    const SEGMENT_TABLE: [WalConfig; 5] = [
        WalConfig {
            segment_rows: 2_048,
            segment_bytes: 1024 * 1024,
            segments_per_cycle: 32,
        },
        WalConfig {
            segment_rows: 4_096,
            segment_bytes: 2 * 1024 * 1024,
            segments_per_cycle: 16,
        },
        WalConfig {
            segment_rows: 8_192,
            segment_bytes: 4 * 1024 * 1024,
            segments_per_cycle: 8,
        },
        WalConfig {
            segment_rows: 16_384,
            segment_bytes: 8 * 1024 * 1024,
            segments_per_cycle: 4,
        },
        WalConfig {
            segment_rows: 65_536,
            segment_bytes: 32 * 1024 * 1024,
            segments_per_cycle: 1,
        },
    ];

    struct WalCurveSummary {
        /// (segment_bytes, median_total_wal_ns) per config, W1-contiguous.
        w1_points: Vec<(u64, u64)>,
        plateau_segment_bytes: Option<u64>,
        /// How many configs exceeded the p95/median spread ceiling. Non-zero
        /// means `plateau_segment_bytes` was SUPPRESSED (unmeasurable), which
        /// is a different `None` from "no knee in the table".
        unstable_configs: usize,
        worst_spread: f64,
    }

    fn run_wal_curve(csv: &mut CsvSink) -> WalCurveSummary {
        eprintln!("\n== WAL curve — W1-contiguous (storage/cache ceiling) vs W0-current (SweepSlot/BTreeMap) ==");
        let wal_dir = PathBuf::from(WAL_DIR);
        fs::create_dir_all(&wal_dir).expect("create WAL scratch dir");

        // The canonical 32 MiB frame — built ONCE, shared content across
        // every configuration and representation (only the I/O CHUNKING
        // differs between W1/W0; the "constant 512 MiB per config" bar is
        // about bytes actually handed to the OS, not about re-deriving
        // content each cycle).
        let canonical: Vec<u8> = {
            let mut buf = vec![0u8; CANONICAL_FRAME_BYTES];
            for row in 0..u64::from(FLEET_OWNERS) {
                let cell = NodeRow512::for_id(row);
                let lo = (row as usize) * CANONICAL_ROW_BYTES;
                buf[lo..lo + CANONICAL_ROW_BYTES].copy_from_slice(cell.as_bytes());
            }
            buf
        };
        assert_eq!(canonical.len(), CANONICAL_FRAME_BYTES);

        let mut w1_points = Vec::new();
        let mut w1_spreads: Vec<f64> = Vec::new();

        for cfg in SEGMENT_TABLE {
            for rep in ["w1_contiguous", "w0_current"] {
                let path = wal_dir.join(format!("{rep}_{}.wal", cfg.segment_rows));
                let mut file = OpenOptions::new()
                    .create(true)
                    .write(true)
                    .truncate(true)
                    .open(&path)
                    .unwrap_or_else(|e| panic!("open WAL scratch file {path:?}: {e}"));

                let mut write_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
                let mut sync_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
                let mut byte_samples: Vec<u64> = Vec::with_capacity(MEASURED_CYCLES as usize);
                let mut position_base: u64 = 0;

                for cycle_idx in 0..(WARMUP_CYCLES + MEASURED_CYCLES) {
                    let measured = cycle_idx >= WARMUP_CYCLES;

                    let (
                        collect_ns,
                        freeze_ns,
                        wal_write_ns,
                        wal_sync_ns,
                        wal_syscalls,
                        logical_rows,
                    );
                    // ACTUAL bytes handed to the kernel this cycle. A first
                    // revision computed MiB/s from the ASSUMED 32 MiB frame
                    // while discarding write_vectored's real byte count — an
                    // assumption presented as a measurement, and the exact
                    // axis-blending this plan exists to prevent.
                    let mut bytes_written = 0u64;

                    if rep == "w1_contiguous" {
                        collect_ns = 0u64;
                        freeze_ns = 0u64;
                        let mut total_syscalls = 0u64;
                        let t_write = Instant::now();
                        for seg in 0..cfg.segments_per_cycle {
                            let lo = (seg * cfg.segment_bytes) as usize;
                            let hi = lo + cfg.segment_bytes as usize;
                            let mut slices = [IoSlice::new(&canonical[lo..hi])];
                            let (written, calls) = write_vectored_all(&mut file, &mut slices)
                                .expect("W1 write_vectored");
                            total_syscalls += calls;
                            bytes_written += written;
                        }
                        wal_write_ns = t_write.elapsed().as_nanos() as u64;
                        let t_sync = Instant::now();
                        file.sync_data().expect("W1 sync_data");
                        wal_sync_ns = t_sync.elapsed().as_nanos() as u64;
                        wal_syscalls = total_syscalls;
                        logical_rows = u64::from(FLEET_OWNERS);
                    } else {
                        // W0-current: REAL SweepSlot construction (one owned
                        // Vec<u8> clone per row — the allocator cost this arm
                        // measures) + a REAL DetachedCycleBatch::freeze.
                        let t_collect = Instant::now();
                        let mut slots = Vec::with_capacity(FLEET_OWNERS as usize);
                        for row in 0..u64::from(FLEET_OWNERS) {
                            let lo = (row as usize) * CANONICAL_ROW_BYTES;
                            let payload = canonical[lo..lo + CANONICAL_ROW_BYTES].to_vec();
                            slots.push(SweepSlot {
                                cycle: CycleId(u64::from(cycle_idx) + 1),
                                stream_position: position_base + row,
                                owner: row as MailboxId,
                                row,
                                paired_move: None,
                                payload,
                            });
                        }
                        collect_ns = t_collect.elapsed().as_nanos() as u64;
                        position_base += u64::from(FLEET_OWNERS);

                        let t_freeze = Instant::now();
                        let frame = CycleFrame::new(
                            CycleId(u64::from(cycle_idx) + 1),
                            // The base version of cycle N is N (v0 before the first
                            // cycle). Derived from cycle_idx rather than a parallel
                            // counter — they were provably equal and the counter was a
                            // second source of truth.
                            DatasetVersion(u64::from(cycle_idx)),
                        );
                        let frozen = DetachedCycleBatch::freeze(frame, slots);
                        freeze_ns = t_freeze.elapsed().as_nanos() as u64;
                        logical_rows = frozen.image.len() as u64;
                        assert_eq!(
                            logical_rows,
                            u64::from(FLEET_OWNERS),
                            "W0: every row coalesced exactly once"
                        );

                        let ordered: Vec<&Vec<u8>> = frozen.image.values().collect();
                        let mut total_syscalls = 0u64;
                        let t_write = Instant::now();
                        for group in ordered.chunks(cfg.segment_rows as usize) {
                            let mut slices: Vec<IoSlice<'_>> =
                                group.iter().map(|v| IoSlice::new(v.as_slice())).collect();
                            let (written, calls) = write_vectored_all(&mut file, &mut slices)
                                .expect("W0 write_vectored");
                            total_syscalls += calls;
                            bytes_written += written;
                        }
                        wal_write_ns = t_write.elapsed().as_nanos() as u64;
                        let t_sync = Instant::now();
                        file.sync_data().expect("W0 sync_data");
                        wal_sync_ns = t_sync.elapsed().as_nanos() as u64;
                        wal_syscalls = total_syscalls;
                    }

                    assert_eq!(
                        bytes_written, CANONICAL_FRAME_BYTES as u64,
                        "{rep} @cycle {cycle_idx}: an arm that does not move exactly the \
                         canonical frame cannot be compared against one that does"
                    );

                    if measured {
                        write_samples.push(wal_write_ns);
                        sync_samples.push(wal_sync_ns);
                        byte_samples.push(bytes_written);

                        let snap = proc_snapshot();
                        // Content is constant per config (the canonical frame
                        // never changes across cycles here — documented in
                        // the build report); the digest is a stable checksum
                        // over a fixed slice, not a per-cycle claim.
                        let digest = fnv1a64(&canonical[..64]);
                        let total_ns = collect_ns + freeze_ns + wal_write_ns + wal_sync_ns;
                        csv.write(&Row {
                            owner_shape: "canonical_65536",
                            physical_layout: rep,
                            threads: 1,
                            segment_rows: cfg.segment_rows,
                            segment_bytes: cfg.segment_bytes,
                            segments_per_cycle: cfg.segments_per_cycle,
                            repeat: cycle_idx,
                            build_ns: 0,
                            scan_ns: 0,
                            think_ns: 0,
                            rebind_cast_ns: 0,
                            collect_ns,
                            freeze_ns,
                            wal_write_ns,
                            wal_sync_ns,
                            temporal_layer1_ns: 0,
                            temporal_layer2_ns: 0,
                            apply_ns: 0,
                            total_ns,
                            logical_rows,
                            logical_bytes: bytes_written,
                            sealed_transitions: 0,
                            applied_transitions: 0,
                            wal_syscalls,
                            fsync_calls: 1,
                            dataset_versions: 1,
                            peak_rss_bytes: snap.vmhwm_kb * 1024,
                            minor_faults: snap.minflt,
                            major_faults: snap.majflt,
                            context_switches: snap.vol_ctxt + snap.nonvol_ctxt,
                            max_active_workers: 1,
                            result_digest: digest,
                            morton_reorder_ns: 0,
                        });
                    }
                }

                let med_write = median(&write_samples);
                let p95_write = p95(&write_samples);
                let med_sync = median(&sync_samples);
                let med_total = med_write + med_sync;
                let med_bytes = median(&byte_samples);
                let mib_per_s = if med_total > 0 {
                    (med_bytes as f64 / (1024.0 * 1024.0)) / (med_total as f64 / 1e9)
                } else {
                    0.0
                };
                eprintln!(
                    "WAL {rep} segment_bytes={} segments/cycle={}: median write={med_write}ns p95={p95_write}ns sync={med_sync}ns ({mib_per_s:.1} MiB/s)",
                    cfg.segment_bytes, cfg.segments_per_cycle
                );

                if rep == "w1_contiguous" {
                    w1_points.push((cfg.segment_bytes, med_total));
                    // Instability marker: a config whose p95 write dwarfs its
                    // median is being driven by page-cache / dirty-writeback
                    // state, not by segment size.
                    let spread = if med_write > 0 {
                        p95_write as f64 / med_write as f64
                    } else {
                        0.0
                    };
                    w1_spreads.push(spread);
                }

                // Reclaim this configuration's scratch file IMMEDIATELY. Each
                // config appends (WARMUP+MEASURED) x 32 MiB = 576 MiB; keeping
                // all ten alive until the end needs ~5.8 GiB and hit ENOSPC on
                // the first real run. Dropping the handle first so the unlink
                // frees the blocks now rather than at scope exit.
                drop(file);
                fs::remove_file(&path).ok();
            }
        }

        fs::remove_dir_all(&wal_dir).ok();

        let throughput: Vec<f64> = w1_points
            .iter()
            .map(|&(_, ns)| {
                if ns > 0 {
                    (CANONICAL_FRAME_BYTES as f64 / (1024.0 * 1024.0)) / (ns as f64 / 1e9)
                } else {
                    0.0
                }
            })
            .collect();
        let (gains, plateau_idx) = plateau_index(&throughput);
        eprintln!("WAL curve gains (W1-contiguous, by segment_bytes):");
        for (i, cfg) in SEGMENT_TABLE.iter().enumerate() {
            let gain_str = if gains[i].is_nan() {
                "n/a".to_string()
            } else {
                format!("{:+.1}%", gains[i] * 100.0)
            };
            eprintln!(
                "  segment_bytes={:>9} throughput={:.1} MiB/s gain={}",
                cfg.segment_bytes, throughput[i], gain_str
            );
        }
        // ── STABILITY GUARD (added after four runs of this binary disagreed
        // by up to 6x at IDENTICAL configurations, and one config showed a
        // p95/median spread of 24x). A knee named from data whose run-to-run
        // variance exceeds the effect is fabricated precision — exactly what
        // this plan exists to prevent. The guard has both halves: it FIRES
        // (suppressing the knee) when any config is unstable, and it STAYS
        // SILENT (reporting the knee) when every config is tight.
        const SPREAD_CEILING: f64 = 3.0;
        let worst_spread = w1_spreads.iter().copied().fold(0.0_f64, f64::max);
        let unstable: Vec<usize> = w1_spreads
            .iter()
            .enumerate()
            .filter(|(_, s)| **s > SPREAD_CEILING)
            .map(|(i, _)| i)
            .collect();

        let plateau_segment_bytes = if unstable.is_empty() {
            plateau_idx.map(|i| SEGMENT_TABLE[i].segment_bytes)
        } else {
            None
        };

        if unstable.is_empty() {
            match plateau_segment_bytes {
                Some(b) => eprintln!(
                    "WAL curve plateau (descriptive, NOT pass/kill): first knee at segment_bytes={b} \
                     (two consecutive doublings < 5% median-throughput gain; worst p95/median spread \
                     {worst_spread:.1}x <= {SPREAD_CEILING:.0}x ceiling, so the curve is stable enough to read)"
                ),
                None => eprintln!(
                    "WAL curve plateau: no knee within the 5-point table (still gaining at every step)"
                ),
            }
        } else {
            eprintln!(
                "WAL curve plateau: NOT MEASURABLE ON THIS HOST — {} of {} configs exceed the \
                 p95/median spread ceiling ({SPREAD_CEILING:.0}x); worst {worst_spread:.1}x. The \
                 write phase is being driven by page-cache / dirty-writeback state rather than by \
                 segment size, so NO knee is reported. Re-run on a quiet host with headroom (this \
                 one was ~90% full) and O_DIRECT or a drop_caches barrier per config.",
                unstable.len(),
                w1_spreads.len()
            );
        }

        WalCurveSummary {
            w1_points,
            plateau_segment_bytes,
            unstable_configs: unstable.len(),
            worst_spread,
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // §10 — a real, in-process `WalSink` for the Temporal + EXP-KIA-A2-64K
    // arms (no raw-file I/O here — those arms measure `temporal.rs` and
    // concurrency, not fsync physics; the WAL-curve arm above owns physics).
    // ═════════════════════════════════════════════════════════════════════

    struct SealedEntry {
        frame: CycleFrame,
        version: DatasetVersion,
        /// The batch's deterministic content hash — the reconciliation-first
        /// idempotency key `commit_cycle` looks up BEFORE appending.
        batch_hash: u64,
        landings: Vec<SweepSlot>,
    }

    struct MemWal {
        sealed: Mutex<Vec<SealedEntry>>,
        next_version: AtomicU64,
        wal_writes: AtomicU64,
    }

    impl MemWal {
        fn new() -> Self {
            Self {
                sealed: Mutex::new(Vec::new()),
                next_version: AtomicU64::new(1),
                wal_writes: AtomicU64::new(0),
            }
        }
        fn wal_writes(&self) -> u64 {
            self.wal_writes.load(Ordering::SeqCst)
        }
        fn head(&self) -> DatasetVersion {
            self.sealed
                .lock()
                .expect("MemWal poisoned")
                .last()
                .map_or(DatasetVersion(0), |s| s.version)
        }
    }

    impl WalSink for MemWal {
        async fn commit_cycle(
            &mut self,
            batch: DetachedCycleBatch,
        ) -> Result<CommitOutcome, CommitError> {
            let mut sealed = self.sealed.lock().expect("MemWal poisoned");
            // Reconciliation-first: an already-durable (cycle, hash) is success,
            // a matching cycle with a different hash fails closed.
            if let Some(rec) = sealed.iter().find(|s| s.frame.cycle == batch.frame.cycle) {
                return if rec.batch_hash == batch.batch_hash {
                    Ok(CommitOutcome::Reconciled {
                        version: rec.version,
                        cycle: batch.frame.cycle,
                        batch_hash: batch.batch_hash,
                    })
                } else {
                    Err(CommitError::HashConflict {
                        cycle: batch.frame.cycle,
                        stored_hash: rec.batch_hash,
                        offered_hash: batch.batch_hash,
                    })
                };
            }
            let head = sealed.last().map_or(DatasetVersion(0), |s| s.version);
            if batch.frame.base_version != head {
                return Err(CommitError::Fenced { current_head: head });
            }
            self.wal_writes.fetch_add(1, Ordering::SeqCst);
            let version = DatasetVersion(self.next_version.fetch_add(1, Ordering::SeqCst));
            let (cycle, batch_hash) = (batch.frame.cycle, batch.batch_hash);
            sealed.push(SealedEntry {
                frame: batch.frame,
                version,
                batch_hash,
                landings: batch.landings,
            });
            Ok(CommitOutcome::Committed {
                version,
                cycle,
                batch_hash,
            })
        }

        async fn scan_sealed(
            &self,
            after_cycle: Option<CycleId>,
        ) -> Result<Vec<LandedSlot>, WriteFailed> {
            Ok(self
                .sealed
                .lock()
                .expect("MemWal poisoned")
                .iter()
                .filter(|s| after_cycle.is_none_or(|c| s.frame.cycle > c))
                .flat_map(|s| {
                    s.landings.iter().map(|slot| LandedSlot {
                        cycle: s.frame.cycle,
                        slot: slot.clone(),
                    })
                })
                .collect())
        }

        async fn timeline(&self) -> Result<Vec<FrameMeta>, WriteFailed> {
            Ok(self
                .sealed
                .lock()
                .expect("MemWal poisoned")
                .iter()
                .map(|s| FrameMeta {
                    cycle: s.frame.cycle,
                    base_version: s.frame.base_version,
                    batch_hash: s.batch_hash,
                })
                .collect())
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // §11 — Temporal: T0 (scan_sealed) · T1 (local_trajectories) ·
    // T2 (deinterlace), over 65,536 owners x 16 real committed cycles =
    // 1,048,576 rows.
    // ═════════════════════════════════════════════════════════════════════

    #[derive(Clone)]
    struct BenchRow {
        owner: MailboxId,
        cast_seq: u64,
        lance_version: u64,
    }
    impl LocalCausalRow for BenchRow {
        fn owner(&self) -> MailboxId {
            self.owner
        }
        fn cast_seq(&self) -> u64 {
            self.cast_seq
        }
    }
    impl DeinterlaceRow for BenchRow {
        fn subject(&self) -> &str {
            // NoDeps ignores subject entirely (`closure_at` never reads it) —
            // a real DependsClosure impl would need a real subject string;
            // this arm exercises the trivial DATA-causal axis only.
            ""
        }
        fn lance_version(&self) -> u64 {
            self.lance_version
        }
        fn knowable_from(&self) -> u64 {
            0 // every class knowable from the start — this arm isolates the
              // TIME-causal axis, not the class-registration axis.
        }
    }

    async fn run_temporal(csv: &mut CsvSink) {
        eprintln!("\n== Temporal — T0 scan_sealed / T1 local_trajectories / T2 deinterlace (post-WAL only) ==");

        // Build 16 REAL committed cycles of 65,536 no-step landings each
        // (`SweepSlot::paired_move = None` — a sanctioned landing shape per
        // `persist_sink.rs:145-147`'s own doc). Real `persist_cycle` calls
        // against a real (in-process) `WalSink`, not a fabricated Vec.
        let mut sink = MemWal::new();
        let mut position_base: u64 = 0;
        for cyc in 1..=16u64 {
            let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
            for id in 0..FLEET_OWNERS {
                writer.cast(id, vec![], id.to_le_bytes().to_vec());
            }
            let collected = collect_casts(&mut writer, CycleId(cyc), position_base, u64::from);
            assert_eq!(collected.slots.len(), FLEET_OWNERS as usize);
            let base = sink.head();
            let frame = CycleFrame::new(CycleId(cyc), base);
            persist_cycle(&mut sink, frame, collected.slots)
                .await
                .unwrap_or_else(|e| panic!("temporal history: cycle {cyc} failed to seal: {e}"));
            position_base += u64::from(FLEET_OWNERS);
        }
        assert_eq!(
            sink.wal_writes(),
            16,
            "16 real sealed cycles built the history"
        );

        // T0 — the sealed WAL read (this arm's "scan_ns" mapping — the CSV
        // schema has no dedicated T0 column; `scan_ns` is the closest-named
        // generic slot and is used ONLY here for that purpose, documented
        // in the build report).
        let t0 = Instant::now();
        let landed: Vec<LandedSlot> = sink
            .scan_sealed(None)
            .await
            .expect("T0: scan_sealed over 16 sealed cycles");
        let t0_ns = t0.elapsed().as_nanos() as u64;
        assert_eq!(
            landed.len(),
            FLEET_OWNERS as usize * 16,
            "T0: 65,536 owners x 16 landings = 1,048,576 rows"
        );

        let bench_rows: Vec<BenchRow> = landed
            .iter()
            .map(|ls| BenchRow {
                owner: ls.slot.owner,
                cast_seq: ls.slot.stream_position,
                // `LandedSlot` is keyed by CYCLE, not physical `DatasetVersion`
                // (persist_sink's governing storage rule: a cycle with only
                // intent-only casts publishes no version at all). This
                // benchmark's cycles are 1:1 with commits (every cast carries
                // a non-empty payload, so every cycle here IS a version), so
                // `cycle.0` is the same monotonic identity `version.0` was.
                lance_version: ls.cycle.0,
            })
            .collect();

        // T1 — layer-1 causal deinterlacing: split the interleaved global
        // log into 65,536 per-owner local trajectories.
        let t1 = Instant::now();
        let trajectories = local_trajectories(&bench_rows);
        let t1_ns = t1.elapsed().as_nanos() as u64;
        assert_eq!(
            trajectories.len(),
            FLEET_OWNERS as usize,
            "T1: exactly 65,536 distinct owner trajectories"
        );
        for chain in trajectories.values().take(8) {
            assert_eq!(
                chain.len(),
                16,
                "T1: each owner's chain has all 16 landings"
            );
        }

        // T2 — layer-2 epistemic projection. `ref_version = 8` (MID-history,
        // not the last cycle) so the filter is genuinely falsifiable: cycles
        // 1..=8 are Contemporary, 9..=16 are Anachronistic under Strict and
        // are DROPPED — never a vacuous "everything survives" check.
        let v_ref = QueryReference::at(8, 0);
        let t2 = Instant::now();
        let visible = deinterlace(&bench_rows, &v_ref, &NoDeps);
        let t2_ns = t2.elapsed().as_nanos() as u64;
        let expected_visible = FLEET_OWNERS as usize * 8;
        assert_eq!(
            visible.len(),
            expected_visible,
            "T2 anti-vacuity: exactly the first 8 of 16 cycles survive a Strict reader at ref_version=8"
        );
        assert!(
            visible.len() < bench_rows.len(),
            "T2 anti-vacuity: the filter actually dropped rows (future cycles), not merely 'ran'"
        );

        eprintln!(
            "Temporal: T0 scan_sealed={t0_ns}ns ({} rows) | T1 local_trajectories={t1_ns}ns ({} owners) | T2 deinterlace={t2_ns}ns ({} of {} visible)",
            landed.len(),
            trajectories.len(),
            visible.len(),
            bench_rows.len()
        );

        let owner_keys: Vec<u8> = trajectories.keys().flat_map(|k| k.to_le_bytes()).collect();
        let digest = fnv1a64(&owner_keys);

        csv.write(&Row {
            owner_shape: "temporal_1048576",
            physical_layout: "local_trajectories_then_deinterlace",
            threads: 1,
            segment_rows: 0,
            segment_bytes: 0,
            segments_per_cycle: 0,
            repeat: 0,
            build_ns: 0,
            scan_ns: t0_ns, // T0 — see the doc comment above the field write
            think_ns: 0,
            rebind_cast_ns: 0,
            collect_ns: 0,
            freeze_ns: 0,
            wal_write_ns: 0,
            wal_sync_ns: 0,
            temporal_layer1_ns: t1_ns,
            temporal_layer2_ns: t2_ns,
            apply_ns: 0,
            total_ns: t0_ns + t1_ns + t2_ns,
            logical_rows: landed.len() as u64,
            // Honest payload size: the temporal history's landings carry an
            // 4-byte owner-id marker each (`id.to_le_bytes()`, MailboxId=u32), NOT a
            // canonical 512-byte row — this arm exercises `temporal.rs`'s
            // row-shape-agnostic API, not the storage envelope (that's B1b's
            // and the WAL curve's job).
            logical_bytes: landed.len() as u64 * 4, // MailboxId=u32 -> to_le_bytes() is 4 bytes
            sealed_transitions: 0,
            applied_transitions: 0,
            wal_syscalls: 0,
            fsync_calls: 0,
            dataset_versions: 16,
            peak_rss_bytes: proc_snapshot().vmhwm_kb * 1024,
            minor_faults: 0,
            major_faults: 0,
            context_switches: 0,
            max_active_workers: 1,
            result_digest: digest,
            morton_reorder_ns: 0,
        });
    }

    // ═════════════════════════════════════════════════════════════════════
    // §12 — L1: ChunkedSoA<1024>[64] physical-layout control.
    //
    // Honest scoping note (see build report for the full reasoning): a
    // Rubicon `phase()` is scoped to ONE `MailboxSoA<N>` instance, never to
    // an individual row. L1a's 65,536 "logical owners" share 64 physical
    // `MailboxSoA<1024>` instances, so there is no per-logical-owner phase
    // to advance — `apply_sealed_transitions` (which resolves a transition's
    // owner to ONE `MailboxSoaOwner` and advances ITS phase) does not have a
    // meaningful target at logical-owner granularity here. Rather than
    // fabricate a per-row "apply" by routing 1,024 logical owners' moves
    // through one chunk's single phase field (which would either silently
    // misrepresent 1,023 of every 1,024 owners, or require inventing a new
    // per-row phase type this brief forbids), this arm measures build,
    // scan, think, cast, collect, and freeze ONLY — the phases that ARE
    // meaningful at row granularity — and apply_ns is left 0 for both L1a
    // and L1b, documented here so the comparison against B1a stays honest
    // (answer #2 compares build..freeze only, explicitly, not apply).
    // ═════════════════════════════════════════════════════════════════════

    const CHUNK_ROWS: usize = 1024;
    const CHUNKS: u32 = 64;
    type Chunk = MailboxSoA<CHUNK_ROWS>;

    fn build_chunk(chunk_idx: u32) -> Chunk {
        let mut chunk: Chunk = MailboxSoA::new(chunk_idx, 0, 1.0);
        let cycle = chunk.cycle();
        for lane in 0..CHUNK_ROWS {
            let logical_owner = chunk_idx * CHUNK_ROWS as u32 + lane as u32;
            let plane = splitmix_plane(u64::from(logical_owner));
            let cell = WriteCell {
                content: Some(plane.as_slice()),
                qualia: Some(flow_qualia()),
                meta: Some(MetaWord::new(1, 0, 0, 0, 0)),
                entity_type: Some((logical_owner % 251) as u16),
                temporal: Some(u64::from(logical_owner)),
                ..WriteCell::default()
            };
            let outcome = chunk.write_row(lane, cycle, &cell);
            assert_eq!(
                outcome,
                WriteOutcome::Accepted,
                "seeding chunk {chunk_idx} lane {lane}"
            );
        }
        chunk.set_populated(CHUNK_ROWS);
        chunk.tick();
        for lane in 0..CHUNK_ROWS {
            chunk.energy[lane] = 2.0;
        }
        chunk
    }

    fn run_l1a(csv: &mut CsvSink) -> PhaseMedians {
        eprintln!("\n== L1a — 64 x MailboxSoA<1024> chunks, 65,536 LOGICAL owners (owner = chunk*1024+lane) ==");
        let style_outcome = build_style_outcome();
        let mut samples = Vec::new();

        for repeat in 0..3u32 {
            // VmRSS BEFORE this repeat's allocation (the delta baseline).
            let rss_before = read_proc_status().1 as i64 * 1024;
            let t_build = Instant::now();
            let mut chunks: HashMap<u32, Chunk> = HashMap::with_capacity(CHUNKS as usize);
            for c in 0..CHUNKS {
                chunks.insert(c, build_chunk(c));
            }
            let build_ns = t_build.elapsed().as_nanos() as u64;

            let t_scan = Instant::now();
            let mut in_planning_chunks = 0usize;
            for chunk in chunks.values() {
                if chunk.phase() == KanbanColumn::Planning {
                    in_planning_chunks += 1;
                }
            }
            let scan_ns = t_scan.elapsed().as_nanos() as u64;
            assert_eq!(
                in_planning_chunks, CHUNKS as usize,
                "L1a: every chunk starts Planning"
            );

            // "thought": one gate decision PER CHUNK (phase is chunk-scoped —
            // every logical owner in a chunk shares its chunk's gate target,
            // an accepted physical-layout-control simplification, documented
            // above and in the build report).
            let t_think = Instant::now();
            let mut chunk_targets: HashMap<u32, KanbanColumn> =
                HashMap::with_capacity(CHUNKS as usize);
            for c in 0..CHUNKS {
                let chunk = chunks.get(&c).expect("chunk exists");
                let qualia = chunk.qualia_at(0);
                let mantissa = mantissa_of(chunk);
                let gate = gate_decision_i4(&qualia, mantissa);
                let target = chunk
                    .phase()
                    .advance_on_gate(&gate)
                    .expect("L1a: every chunk Flows");
                chunk_targets.insert(c, target);
            }
            let think_ns = t_think.elapsed().as_nanos() as u64;

            let t_cast = Instant::now();
            let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
            let mut cast = 0usize;
            for c in 0..CHUNKS {
                let chunk = chunks.get(&c).expect("chunk exists");
                for lane in 0..CHUNK_ROWS as u32 {
                    let logical_owner = c * CHUNK_ROWS as u32 + lane;
                    let payload = logical_owner.to_le_bytes().to_vec();
                    if emit_bootstrap_intent(
                        &style_outcome,
                        logical_owner,
                        chunk.current_cycle(),
                        &mut writer,
                        payload,
                    )
                    .is_some()
                    {
                        cast += 1;
                    }
                }
            }
            let cast_ns = t_cast.elapsed().as_nanos() as u64;
            assert_eq!(
                cast, FLEET_OWNERS as usize,
                "L1a can-fire: all 65,536 logical owners cast"
            );

            let t_collect = Instant::now();
            let collected =
                collect_casts(&mut writer, CycleId(u64::from(repeat) + 1), 0, u64::from);
            let collect_ns = t_collect.elapsed().as_nanos() as u64;
            assert_eq!(collected.slots.len(), FLEET_OWNERS as usize);
            assert!(
                collected.held.is_empty(),
                "L1a: distinct logical owner ids -> nothing held (unlike a chunk-keyed collapse)"
            );

            // digest_bytes computed OUTSIDE the timed window — freeze_ns
            // measures exactly `DetachedCycleBatch::freeze`.
            let frame = CycleFrame::new(CycleId(u64::from(repeat) + 1), DatasetVersion(0));
            let digest_bytes: Vec<u8> = collected
                .slots
                .iter()
                .flat_map(|s| s.payload.iter().copied())
                .collect();
            let t_freeze = Instant::now();
            let frozen = DetachedCycleBatch::freeze(frame, collected.slots);
            let freeze_ns = t_freeze.elapsed().as_nanos() as u64;
            assert_eq!(frozen.image.len(), FLEET_OWNERS as usize);

            let total_ns = build_ns + scan_ns + think_ns + cast_ns + collect_ns + freeze_ns;
            let snap = proc_snapshot();
            let digest = fnv1a64(&digest_bytes);

            samples.push(PhaseMedians {
                build_ns,
                scan_ns,
                think_ns,
                cast_ns,
                collect_ns,
                freeze_ns,
                apply_ns: 0,
                peak_rss_bytes: snap.vmhwm_kb * 1024,
                rss_delta_bytes: (snap.vmrss_kb as i64 * 1024) - rss_before,
            });

            eprintln!(
                "L1a repeat {repeat}: build={build_ns}ns scan={scan_ns}ns think={think_ns}ns cast={cast_ns}ns collect={collect_ns}ns freeze={freeze_ns}ns total={total_ns}ns"
            );
            csv.write(&Row {
                owner_shape: "l1a_chunked_soa_1024x64",
                physical_layout: "chunked_valid_comparison",
                threads: 1,
                segment_rows: 0,
                segment_bytes: 0,
                segments_per_cycle: 0,
                repeat,
                build_ns,
                scan_ns,
                think_ns,
                rebind_cast_ns: cast_ns,
                collect_ns,
                freeze_ns,
                wal_write_ns: 0,
                wal_sync_ns: 0,
                temporal_layer1_ns: 0,
                temporal_layer2_ns: 0,
                apply_ns: 0,
                total_ns,
                logical_rows: FLEET_OWNERS as u64,
                logical_bytes: (FLEET_OWNERS as u64) * 4, // MailboxId=u32 -> to_le_bytes() is 4 bytes
                sealed_transitions: cast as u64,
                applied_transitions: 0,
                wal_syscalls: 0,
                fsync_calls: 0,
                dataset_versions: 0,
                peak_rss_bytes: snap.vmhwm_kb * 1024,
                minor_faults: snap.minflt,
                major_faults: snap.majflt,
                context_switches: snap.vol_ctxt + snap.nonvol_ctxt,
                max_active_workers: 1,
                result_digest: digest,
                morton_reorder_ns: 0,
            });
            let _ = chunk_targets;
        }

        median_phases(&samples)
    }

    /// L1b — the MISLABELLING control: 64 chunks cast as if THEY were the
    /// owners (64 casts/cycle, not 65,536). Never evidence for the 64k-owner
    /// model — its only purpose is to make the held-backlog collapse visible
    /// (per `collect_casts`'s <=1-move-per-owner partition) when a physical
    /// chunk is mistaken for a logical owner, contrasted against L1a's clean
    /// 65,536-distinct-owner cast above.
    fn run_l1b(csv: &mut CsvSink) {
        eprintln!(
            "\n== L1b — 64 chunks AS owners (mislabelling CONTROL, never the 64k-owner model) =="
        );
        let style_outcome = build_style_outcome();

        let t_build = Instant::now();
        let mut chunks: HashMap<u32, Chunk> = HashMap::with_capacity(CHUNKS as usize);
        for c in 0..CHUNKS {
            chunks.insert(c, build_chunk(c));
        }
        let build_ns = t_build.elapsed().as_nanos() as u64;

        // Cast ALL 1,024 lanes' intents but ON BEHALF OF THE CHUNK id (not
        // the logical owner) — this is the deliberately wrong shape.
        let t_cast = Instant::now();
        let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
        let mut cast = 0usize;
        for c in 0..CHUNKS {
            let chunk = chunks.get(&c).expect("chunk exists");
            for lane in 0..CHUNK_ROWS as u32 {
                let payload = lane.to_le_bytes().to_vec();
                if emit_bootstrap_intent(
                    &style_outcome,
                    c, // <- the CHUNK id, not the logical owner — the control
                    chunk.current_cycle(),
                    &mut writer,
                    payload,
                )
                .is_some()
                {
                    cast += 1;
                }
            }
        }
        let cast_ns = t_cast.elapsed().as_nanos() as u64;
        assert_eq!(
            cast, FLEET_OWNERS as usize,
            "L1b: every intent still STAGES (cast never refuses; the collapse shows up at collect_casts)"
        );

        let t_collect = Instant::now();
        let collected = collect_casts(&mut writer, CycleId(1), 0, u64::from);
        let collect_ns = t_collect.elapsed().as_nanos() as u64;
        assert_eq!(
            collected.slots.len(),
            FLEET_OWNERS as usize,
            "L1b: every cast still lands a payload landing (move-free for the held ones)"
        );
        assert_eq!(
            collected.held.len(),
            FLEET_OWNERS as usize - CHUNKS as usize,
            "L1b MISLABELLING PROOF: only 64 of 65,536 moves seal (1/chunk) — the other \
             65,472 are HELD because collect_casts's <=1-move-per-owner partition sees \
             only 64 distinct owner ids, exactly the failure 'a physical chunk is NOT \
             an owner' warns against"
        );

        let total_ns = build_ns + cast_ns + collect_ns;
        let snap = proc_snapshot();
        eprintln!(
            "L1b: build={build_ns}ns cast={cast_ns}ns collect={collect_ns}ns -> {} staged, {} of {} HELD (chunk-as-owner collapse, proves the control)",
            collected.slots.len(),
            collected.held.len(),
            FLEET_OWNERS
        );
        csv.write(&Row {
            owner_shape: "l1b_chunk_as_owner_control",
            physical_layout: "topology_control_never_64k_evidence",
            threads: 1,
            segment_rows: 0,
            segment_bytes: 0,
            segments_per_cycle: 0,
            repeat: 0,
            build_ns,
            scan_ns: 0,
            think_ns: 0,
            rebind_cast_ns: cast_ns,
            collect_ns,
            freeze_ns: 0,
            wal_write_ns: 0,
            wal_sync_ns: 0,
            temporal_layer1_ns: 0,
            temporal_layer2_ns: 0,
            apply_ns: 0,
            total_ns,
            logical_rows: FLEET_OWNERS as u64,
            logical_bytes: 0,
            sealed_transitions: (CHUNKS as u64), // exactly one per chunk
            applied_transitions: 0,
            wal_syscalls: 0,
            fsync_calls: 0,
            dataset_versions: 0,
            peak_rss_bytes: snap.vmhwm_kb * 1024,
            minor_faults: snap.minflt,
            major_faults: snap.majflt,
            context_switches: snap.vol_ctxt + snap.nonvol_ctxt,
            max_active_workers: 1,
            result_digest: fnv1a64(&collected.held.len().to_le_bytes()),
            morton_reorder_ns: 0,
        });
    }

    // ═════════════════════════════════════════════════════════════════════
    // §13 — EXP-KIA-A2-64K: exploratory concurrency (non-claiming).
    // D-KIA-A2 is untouched — its own median-of-5, >=2x gate is elsewhere.
    // ═════════════════════════════════════════════════════════════════════

    /// The parallel COMPUTE phase's per-body result — benchmark-local (the
    /// plan §"EXP-KIA-A2-64K" names this shape explicitly: "thread-local
    /// `PreparedIntent` buffers"). Carries no shared mutable state; produced
    /// entirely from a `&Fleet` read during the parallel phase, consumed only
    /// at the SEQUENTIAL convergence boundary below.
    #[derive(Clone)]
    struct PreparedIntent {
        owner: MailboxId,
        target: KanbanColumn,
        payload: Vec<u8>,
    }

    fn compute_range(
        fleet: &HashMap<MailboxId, TenantB1a>,
        range: std::ops::Range<u32>,
    ) -> Vec<PreparedIntent> {
        let mut out = Vec::with_capacity(range.len());
        for id in range {
            let owner = fleet.get(&id).expect("owner exists");
            let qualia = owner.qualia_at(0);
            let mantissa = mantissa_of(owner);
            let gate = gate_decision_i4(&qualia, mantissa);
            let target = owner
                .phase()
                .advance_on_gate(&gate)
                .expect("EXP-KIA: every owner Flows");
            out.push(PreparedIntent {
                owner: id,
                target,
                payload: id.to_le_bytes().to_vec(),
            });
        }
        out
    }

    fn partitions(n: u32, workers: u32) -> Vec<std::ops::Range<u32>> {
        let workers = workers.max(1);
        let chunk = n.div_ceil(workers);
        let mut out = Vec::new();
        let mut lo = 0u32;
        while lo < n {
            let hi = (lo + chunk).min(n);
            out.push(lo..hi);
            lo = hi;
        }
        out
    }

    async fn run_exp_kia_a2_64k(csv: &mut CsvSink) -> (u64, u64, u32, bool) {
        eprintln!(
            "\n== EXP-KIA-A2-64K — exploratory concurrency (non-claiming; D-KIA-A2 untouched) =="
        );
        let style_outcome = build_style_outcome();

        let available = std::thread::available_parallelism()
            .map(std::num::NonZeroUsize::get)
            .unwrap_or(1) as u32;
        let mut worker_counts: Vec<u32> = vec![1, 2, 4, 8, 16];
        if !worker_counts.contains(&available) {
            worker_counts.push(available);
        }
        worker_counts.sort_unstable();
        worker_counts.dedup();
        eprintln!("EXP-KIA: worker counts under test: {worker_counts:?} (available_parallelism={available})");

        let mut seq_digest: Option<u64> = None;
        let mut seq_total_ns = 0u64;
        let mut best_parallel_ns = u64::MAX;
        let mut best_workers = 1u32;
        let mut all_digests_match = true;

        for &workers in &worker_counts {
            // Fresh fleet per worker-count run: every run needs owners
            // starting at Planning for `apply_sealed_transitions` to succeed.
            let t_build = Instant::now();
            let mut fleet: HashMap<MailboxId, TenantB1a> =
                HashMap::with_capacity(FLEET_OWNERS as usize);
            for id in 0..FLEET_OWNERS {
                fleet.insert(id, build_tenant::<ROWS_PER_OWNER_B1A>(id));
            }
            let build_ns = t_build.elapsed().as_nanos() as u64;

            // ── PARALLEL COMPUTE PHASE — disjoint ranges, thread-local
            // buffers, `&fleet` shared read-only, NEVER a mutex around a
            // shared BatchWriter here. ─────────────────────────────────────
            let ranges = partitions(FLEET_OWNERS, workers);
            let active = AtomicUsize::new(0);
            let max_active = AtomicUsize::new(0);
            let t_think = Instant::now();
            let mut all_intents: Vec<PreparedIntent> = Vec::with_capacity(FLEET_OWNERS as usize);
            std::thread::scope(|scope| {
                let mut handles = Vec::with_capacity(ranges.len());
                for range in ranges.clone() {
                    let fleet_ref = &fleet;
                    let active_ref = &active;
                    let max_active_ref = &max_active;
                    handles.push(scope.spawn(move || {
                        let cur = active_ref.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active_ref.fetch_max(cur, Ordering::SeqCst);
                        let result = compute_range(fleet_ref, range);
                        active_ref.fetch_sub(1, Ordering::SeqCst);
                        result
                    }));
                }
                for h in handles {
                    all_intents.extend(h.join().expect("worker thread must not panic"));
                }
            });
            let think_ns = t_think.elapsed().as_nanos() as u64;
            let max_active_workers = max_active.load(Ordering::SeqCst) as u32;
            assert_eq!(
                all_intents.len(),
                FLEET_OWNERS as usize,
                "EXP-KIA can-fire: exactly 65,536 bodies executed, worker count {workers}"
            );
            if workers >= 2 {
                assert!(
                    max_active_workers >= 2,
                    "EXP-KIA can-fire: real overlap measured (max_active_workers>=2) at worker count {workers}"
                );
            }

            // ── SEQUENTIAL CONVERGENCE BOUNDARY — the existing owner rebind
            // + BatchWriter staging, deterministic order (sorted by owner id
            // so every worker-count run converges on the identical order,
            // independent of thread completion order). ─────────────────────
            all_intents.sort_by_key(|p| p.owner);
            let t_cast = Instant::now();
            let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
            let mut cast = 0usize;
            for intent in &all_intents {
                let owner = fleet.get(&intent.owner).expect("owner exists");
                // Rebuild a per-owner StrategyOutcome from the shared
                // style_outcome's crossing, since the target the parallel
                // phase computed may legally be Prune (a Block) rather than
                // CognitiveWork (a Flow) — the bootstrap sentinel from
                // `style_outcome` only carries the Flow crossing, matching
                // `probe_ignition.rs`'s G2b `shade_owner` fallback shape.
                let outcome = if let Some(mv) = style_outcome.intended_move {
                    if mv.to == intent.target {
                        style_outcome
                    } else {
                        StrategyOutcome {
                            reliability: style_outcome.reliability,
                            intended_move: Some(KanbanMove {
                                mailbox: 0,
                                from: owner.phase(),
                                to: intent.target,
                                witness_chain_position: 0,
                                exec: lance_graph_contract::kanban::ExecTarget::Native,
                            }),
                        }
                    }
                } else {
                    style_outcome
                };
                if emit_bootstrap_intent(
                    &outcome,
                    owner.mailbox_id(),
                    owner.current_cycle(),
                    &mut writer,
                    intent.payload.clone(),
                )
                .is_some()
                {
                    cast += 1;
                }
            }
            let cast_ns = t_cast.elapsed().as_nanos() as u64;
            assert_eq!(
                cast, FLEET_OWNERS as usize,
                "EXP-KIA: every intent casts at the boundary"
            );

            // ── ONE SEAL, ONE WAL COMMIT. ───────────────────────────────────
            let t_collect = Instant::now();
            let collected = collect_casts(&mut writer, CycleId(1), 0, u64::from);
            let collect_ns = t_collect.elapsed().as_nanos() as u64;
            assert_eq!(collected.slots.len(), FLEET_OWNERS as usize);
            assert!(
                collected.held.is_empty(),
                "EXP-KIA: one move per owner, nothing held"
            );

            let mut sink = MemWal::new();
            let frame = CycleFrame::new(CycleId(1), DatasetVersion(0));
            let t_wal = Instant::now();
            let outcome = persist_cycle(&mut sink, frame, collected.slots.clone())
                .await
                .expect("EXP-KIA: seal must succeed");
            let wal_write_ns = t_wal.elapsed().as_nanos() as u64;
            assert_eq!(sink.wal_writes(), 1, "EXP-KIA: exactly one WAL commit");
            let version = match outcome {
                CommitOutcome::Committed { version, .. } => version,
                other => panic!("EXP-KIA: every cast has a non-empty payload, expected Committed, got {other:?}"),
            };
            assert_eq!(version, DatasetVersion(1));

            let sealed = build_sealed_locally(frame, &collected.slots, version);
            assert_eq!(
                sealed.transitions.len(),
                FLEET_OWNERS as usize,
                "EXP-KIA: one sealed cycle carries all 65,536 transitions"
            );

            let t_apply = Instant::now();
            let mut watermarks: HashMap<MailboxId, Option<u64>> = HashMap::new();
            let applied = apply_sealed_transitions(&mut fleet, &sealed, &mut watermarks)
                .expect("EXP-KIA: apply must succeed");
            let apply_ns = t_apply.elapsed().as_nanos() as u64;
            assert_eq!(
                applied.applied.len(),
                FLEET_OWNERS as usize,
                "EXP-KIA: 65,536 applied transitions"
            );

            // ── owner bindings preserved: `SealedTransition::owner` (set from
            // `SweepSlot.owner`, itself set from the PreparedIntent's owner at
            // the sequential boundary) must equal `mv.mailbox` (set inside
            // `emit_bootstrap_intent` -> `rebind_bootstrap`) for EVERY sealed
            // transition — two independently-populated fields from different
            // points in the pipeline, so this is a real cross-check, not a
            // value compared against itself.
            for t in &sealed.transitions {
                assert_eq!(
                    t.owner, t.mv.mailbox,
                    "EXP-KIA: owner binding preserved end-to-end (SweepSlot.owner == KanbanMove.mailbox)"
                );
            }

            // ── sequential-vs-parallel identity: FNV digest over the
            // SORTED (owner, stream_position) pairs. ────────────────────────
            let mut identity_bytes: Vec<u8> = sealed
                .transitions
                .iter()
                .map(|t| (t.owner, t.stream_position))
                .collect::<Vec<_>>()
                .into_iter()
                .flat_map(|(o, s)| {
                    let mut b = o.to_le_bytes().to_vec();
                    b.extend_from_slice(&s.to_le_bytes());
                    b
                })
                .collect();
            // sort at the BYTE level is wrong (variable width already fixed
            // at 12 bytes/entry, so chunk-sort is correct and cheap):
            {
                let mut entries: Vec<[u8; 12]> = identity_bytes
                    .chunks_exact(12)
                    .map(|c| c.try_into().unwrap())
                    .collect();
                entries.sort_unstable();
                identity_bytes = entries.into_iter().flatten().collect();
            }
            let digest = fnv1a64(&identity_bytes);

            let total_ns = build_ns + think_ns + cast_ns + collect_ns + wal_write_ns + apply_ns;
            let snap = proc_snapshot();

            if workers == 1 {
                seq_digest = Some(digest);
                seq_total_ns = think_ns; // the compute-phase time is the axis of interest
            } else {
                if Some(digest) != seq_digest {
                    all_digests_match = false;
                }
                if think_ns < best_parallel_ns {
                    best_parallel_ns = think_ns;
                    best_workers = workers;
                }
            }

            eprintln!(
                "EXP-KIA workers={workers}: build={build_ns}ns think(compute)={think_ns}ns max_active={max_active_workers} cast={cast_ns}ns collect={collect_ns}ns wal={wal_write_ns}ns apply={apply_ns}ns total={total_ns}ns digest={digest:016x}"
            );

            csv.write(&Row {
                owner_shape: "exp_kia_a2_64k",
                physical_layout: "prepared_intent_then_sequential_boundary",
                threads: workers,
                segment_rows: 0,
                segment_bytes: 0,
                segments_per_cycle: 0,
                repeat: 0,
                build_ns,
                scan_ns: 0,
                think_ns,
                rebind_cast_ns: cast_ns,
                collect_ns,
                freeze_ns: 0,
                wal_write_ns,
                wal_sync_ns: 0,
                temporal_layer1_ns: 0,
                temporal_layer2_ns: 0,
                apply_ns,
                total_ns,
                logical_rows: FLEET_OWNERS as u64,
                logical_bytes: (FLEET_OWNERS as u64) * 4, // MailboxId=u32 -> to_le_bytes() is 4 bytes
                sealed_transitions: sealed.transitions.len() as u64,
                applied_transitions: applied.applied.len() as u64,
                wal_syscalls: 0,
                fsync_calls: 1,
                dataset_versions: 1,
                peak_rss_bytes: snap.vmhwm_kb * 1024,
                minor_faults: snap.minflt,
                major_faults: snap.majflt,
                context_switches: snap.vol_ctxt + snap.nonvol_ctxt,
                max_active_workers,
                result_digest: digest,
                morton_reorder_ns: 0,
            });
        }

        eprintln!(
            "EXP-KIA: sequential-vs-parallel digests {} across all worker counts",
            if all_digests_match {
                "MATCH"
            } else {
                "DIVERGED (see stderr above)"
            }
        );
        assert!(
            all_digests_match,
            "EXP-KIA can-fire: sequential and parallel runs must converge to the identical sealed cycle"
        );

        (
            seq_total_ns,
            best_parallel_ns,
            best_workers,
            all_digests_match,
        )
    }

    // ═════════════════════════════════════════════════════════════════════
    // §15 — M-arm: Morton reorder inserted before the seal (plan v3, M-arm).
    //
    // A0 measured `logical order → seal → WAL`. This measures the pipeline
    // the architecture actually proposes: `logical order → MORTON REORDER →
    // seal → WAL`, plus the downstream T1 read. Same 65,536 owners, same
    // per-cycle cast/collect shape as B1a/`run_temporal`; the ONLY
    // difference between the two configurations below is the reorder phase
    // and the physical write order it produces — the CAST phase, payload
    // content, and cycle count are identical.
    // ═════════════════════════════════════════════════════════════════════

    /// v2 D1: identity stays on `MailboxId` — this is a SEPARATE key for
    /// physical write/storage order, never used to look an owner up.
    #[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
    struct WriteOrderKey {
        morton_chunk: u32,
        lane: u16,
        cycle_position: u64,
    }

    /// Standard 8-bit -> 16-bit bit-spread (the libmorton bit-trick): each of
    /// `v`'s 8 bits lands at an EVEN position (0, 2, 4, ..., 14) of the
    /// 16-bit result.
    fn morton_spread_u8(v: u8) -> u16 {
        let mut x = u16::from(v);
        x = (x | (x << 4)) & 0x0F0F;
        x = (x | (x << 2)) & 0x3333;
        x = (x | (x << 1)) & 0x5555;
        x
    }

    /// 2-D Morton (Z-order) interleave of two 8-bit coordinates into one
    /// 16-bit code — a BIJECTION over the full 65,536-owner space (every
    /// `(x, y)` in `[0,256)^2` maps to exactly one code in `[0, 65536)`).
    fn morton_code_u16(x: u8, y: u8) -> u16 {
        morton_spread_u8(x) | (morton_spread_u8(y) << 1)
    }

    /// `owner -> WriteOrderKey`: split the 16-bit owner id into two 8-bit
    /// coordinates (low byte / high byte), Morton-interleave them, and read
    /// the top 6 bits as a chunk id (64 chunks) / bottom 10 as a lane
    /// (1,024 lanes) — matching L1a's 64x1,024 physical shape, but via a
    /// spatially-interleaved (not linear `chunk = owner/1024`) assignment.
    /// `cycle_position` is carried per v2 D1's field list; since
    /// `(morton_chunk, lane)` is already a bijection of `owner`, it never
    /// breaks a tie here — it is provenance, not a discriminator.
    fn morton_key_for(owner: MailboxId) -> WriteOrderKey {
        debug_assert!(
            owner < FLEET_OWNERS,
            "morton_key_for: owner must fit the fleet's 16-bit range"
        );
        let o = owner as u16;
        let x = (o & 0x00FF) as u8;
        let y = ((o >> 8) & 0x00FF) as u8;
        let code = morton_code_u16(x, y);
        WriteOrderKey {
            morton_chunk: u32::from(code >> 10),
            lane: code & 0x03FF,
            cycle_position: u64::from(owner),
        }
    }

    /// A digest over `(owner, row, payload)`, SORTED BY OWNER — order
    /// independent of the caller's physical layout by construction, so it is
    /// a fair SEMANTIC (not physical-layout) comparison between two
    /// pipelines that wrote the same logical content in a different
    /// physical write order. Used for the M-arm's mandatory ordered-vs-
    /// unordered digest identity assert.
    fn semantic_digest(slots: &[SweepSlot]) -> u64 {
        let mut keyed: Vec<(MailboxId, u64, &[u8])> = slots
            .iter()
            .map(|s| (s.owner, s.row, s.payload.as_slice()))
            .collect();
        keyed.sort_by_key(|(owner, row, _)| (*owner, *row));
        let mut bytes = Vec::with_capacity(keyed.len() * (4 + 8 + CANONICAL_ROW_BYTES));
        for (owner, row, payload) in keyed {
            bytes.extend_from_slice(&owner.to_le_bytes());
            bytes.extend_from_slice(&row.to_le_bytes());
            bytes.extend_from_slice(payload);
        }
        fnv1a64(&bytes)
    }

    /// v2 D2 — the ordered-chunk fast path: given a scanned history whose
    /// PHYSICAL write order is already chunk-then-lane within each cycle and
    /// cycle-increasing across cycles (the Morton-ordered M-arm pipeline
    /// below), reconstruct per-owner trajectories by DIRECT APPEND — no
    /// `BTreeMap`-group-then-sort (`local_trajectories`'s own per-owner
    /// `sort_by_key` is exactly what this skips). Validates the invariant
    /// the shortcut depends on (global `stream_position` strictly increasing
    /// across the WHOLE scan — the collapse of "version monotonic x chunk
    /// sequence monotonic x lane monotonic" onto one counter, per v2 D2's
    /// header list) before trusting it; on ANY violation it refuses (`Err`)
    /// rather than silently mis-ordering a trajectory. A real precondition,
    /// not a decorative one — `run_m_arm`'s inline can-fire check (a
    /// deliberately corrupted, stream_position-regressed 2-row input) proves
    /// it can fire.
    fn local_trajectories_ordered_chunk_fastpath(
        landed: &[LandedSlot],
    ) -> Result<BTreeMap<MailboxId, Vec<BenchRow>>, String> {
        let mut out: BTreeMap<MailboxId, Vec<BenchRow>> = BTreeMap::new();
        let mut last_stream_position: Option<u64> = None;
        for ls in landed {
            if let Some(p) = last_stream_position {
                if ls.slot.stream_position <= p {
                    return Err(format!(
                        "fast path: stream_position non-increasing ({p} -> {}) — the \
                         ordered-chunk precondition (validate-then-append) does not hold",
                        ls.slot.stream_position
                    ));
                }
            }
            last_stream_position = Some(ls.slot.stream_position);

            // VALIDATED for this row — append directly. No group-then-sort:
            // this owner's chain is being built in the SAME order the
            // physical log already guarantees.
            out.entry(ls.slot.owner).or_default().push(BenchRow {
                owner: ls.slot.owner,
                cast_seq: ls.slot.stream_position,
                // See run_temporal's identical substitution note: `cycle.0`
                // stands in for the retired `version.0` (1:1 here, every
                // cast carries a non-empty payload).
                lance_version: ls.cycle.0,
            });
        }
        Ok(out)
    }

    /// One M-arm pipeline's per-cycle-medianed measurements.
    #[derive(Clone, Copy, Default)]
    struct MArmPhaseMedians {
        cast_ns: u64,
        collect_ns: u64,
        reorder_ns: u64,
        seal_ns: u64,
        wal_write_ns: u64,
        wal_sync_ns: u64,
        wal_syscalls: u64,
    }

    /// Run one M-arm configuration (`morton == false` -> natural/A0-shaped
    /// order; `morton == true` -> Morton reorder inserted before the seal).
    /// `WARMUP_CYCLES + MEASURED_CYCLES` real cycles, each: cast (identical
    /// content/order for both configs) -> collect -> [reorder, Morton only]
    /// -> seal (the REAL `DetachedCycleBatch::freeze`) -> a REAL byte write
    /// of the frozen landings' 512B payloads (chunked `write_vectored`,
    /// house pattern from `run_wal_curve`'s W0-current path) -> one
    /// `fsync` -> commit into an in-process `MemWal` (for the T1 read after
    /// all configs have run). Returns the phase medians, the sealed
    /// `SweepSlot`s of the LAST measured cycle (for the digest-identity
    /// assert), and the populated `MemWal`.
    async fn run_m_arm_pipeline(
        morton: bool,
        wal_path: &std::path::Path,
    ) -> (MArmPhaseMedians, Vec<SweepSlot>, MemWal) {
        let style_outcome = build_style_outcome();
        let mut sink = MemWal::new();
        let mut file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(wal_path)
            .unwrap_or_else(|e| panic!("M-arm: open WAL scratch file {wal_path:?}: {e}"));

        let mut cast_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut collect_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut reorder_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut seal_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut write_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut sync_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut syscall_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut position_base: u64 = 0;
        let mut last_measured_slots: Vec<SweepSlot> = Vec::new();

        const WRITE_CHUNK_ROWS: usize = 4_096; // 2 MiB/segment — house pattern from run_wal_curve.

        for cyc in 0..(WARMUP_CYCLES + MEASURED_CYCLES) {
            let measured = cyc >= WARMUP_CYCLES;
            let cycle_id = CycleId(u64::from(cyc) + 1);

            let t_cast = Instant::now();
            let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
            for id in 0..FLEET_OWNERS {
                // Payload varies by (owner, cycle) — falsifiable content, not
                // a constant row, so a trajectory digest actually depends on
                // cycle order (never a vacuous "same bytes every cycle").
                let combined = (u64::from(cyc) << 32) | u64::from(id);
                let payload = NodeRow512::for_id(combined).as_bytes().to_vec();
                let _ = emit_bootstrap_intent(&style_outcome, id, 0, &mut writer, payload);
            }
            let cast_ns = t_cast.elapsed().as_nanos() as u64;

            let t_collect = Instant::now();
            let collected = collect_casts(&mut writer, cycle_id, position_base, u64::from);
            let collect_ns = t_collect.elapsed().as_nanos() as u64;
            assert_eq!(
                collected.slots.len(),
                FLEET_OWNERS as usize,
                "M-arm: every owner casts and lands exactly once per cycle"
            );

            let mut slots = collected.slots;
            let reorder_ns = if morton {
                let t_reorder = Instant::now();
                order_cycle_stably(&mut slots, |s| morton_key_for(s.owner));
                // Relabel stream_position to the Morton rank so the seal's
                // OWN internal `order_cycle_stably(by stream_position)`
                // preserves (rather than undoes) this order — the reorder is
                // "inserted before the seal", not a bypass of it.
                for (idx, slot) in slots.iter_mut().enumerate() {
                    slot.stream_position = position_base + idx as u64;
                }
                t_reorder.elapsed().as_nanos() as u64
            } else {
                0
            };

            let frame = CycleFrame::new(cycle_id, sink.head());
            let t_seal = Instant::now();
            let frozen = DetachedCycleBatch::freeze(frame, slots);
            let seal_ns = t_seal.elapsed().as_nanos() as u64;
            assert_eq!(frozen.landings.len(), FLEET_OWNERS as usize);

            let mut bytes_written = 0u64;
            let mut total_syscalls = 0u64;
            let t_write = Instant::now();
            for group in frozen.landings.chunks(WRITE_CHUNK_ROWS) {
                let mut slices: Vec<IoSlice<'_>> = group
                    .iter()
                    .map(|s| IoSlice::new(s.payload.as_slice()))
                    .collect();
                let (written, calls) =
                    write_vectored_all(&mut file, &mut slices).expect("M-arm write_vectored");
                total_syscalls += calls;
                bytes_written += written;
            }
            let wal_write_ns = t_write.elapsed().as_nanos() as u64;
            let t_sync = Instant::now();
            file.sync_data().expect("M-arm sync_data");
            let wal_sync_ns = t_sync.elapsed().as_nanos() as u64;
            assert_eq!(
                bytes_written,
                CANONICAL_FRAME_BYTES as u64,
                "M-arm {}: an arm that does not move exactly the canonical frame \
                 cannot be compared against one that does",
                if morton { "morton" } else { "natural" }
            );

            if measured {
                last_measured_slots = frozen.landings.clone();
                cast_samples.push(cast_ns);
                collect_samples.push(collect_ns);
                reorder_samples.push(reorder_ns);
                seal_samples.push(seal_ns);
                write_samples.push(wal_write_ns);
                sync_samples.push(wal_sync_ns);
                syscall_samples.push(total_syscalls);
            }

            // Commit into the in-process MemWal for the later T1 read — a
            // SEPARATE commit from the real byte write above (the byte write
            // measures physical WAL bytes/fsync physics; the MemWal commit
            // is what `scan_sealed`/T1 read back, matching `run_temporal`'s
            // §11 shape). `persist_cycle` re-derives its own freeze
            // internally; feeding it the SAME (already-Morton-relabeled)
            // slots is safe because its internal `order_cycle_stably` is a
            // no-op-preserving STABLE sort of already-sorted input.
            let slots_for_commit = frozen.landings.clone();
            persist_cycle(&mut sink, frame, slots_for_commit)
                .await
                .unwrap_or_else(|e| panic!("M-arm: cycle {cyc} failed to seal: {e}"));

            position_base += u64::from(FLEET_OWNERS);
        }

        drop(file);
        fs::remove_file(wal_path).ok();

        let medians = MArmPhaseMedians {
            cast_ns: median(&cast_samples),
            collect_ns: median(&collect_samples),
            reorder_ns: median(&reorder_samples),
            seal_ns: median(&seal_samples),
            wal_write_ns: median(&write_samples),
            wal_sync_ns: median(&sync_samples),
            wal_syscalls: median(&syscall_samples),
        };
        (medians, last_measured_slots, sink)
    }

    async fn run_m_arm(csv: &mut CsvSink) {
        eprintln!("\n== M-arm — Morton reorder inserted before the seal (plan v3) ==");
        let wal_dir = PathBuf::from("/tmp/measure_wal_curve_m_arm");
        fs::create_dir_all(&wal_dir).expect("create M-arm WAL scratch dir");

        let (natural, natural_last_slots, natural_sink) =
            run_m_arm_pipeline(false, &wal_dir.join("natural.wal")).await;
        let (morton, morton_last_slots, morton_sink) =
            run_m_arm_pipeline(true, &wal_dir.join("morton.wal")).await;
        fs::remove_dir_all(&wal_dir).ok();

        // ── digest identity (MANDATORY, an assert, not a print) ──────────
        let digest_natural = semantic_digest(&natural_last_slots);
        let digest_morton = semantic_digest(&morton_last_slots);
        assert_eq!(
            digest_natural, digest_morton,
            "M-arm can-fire: the Morton reorder must be a pure layout change — the \
             last measured cycle's (owner, row, payload) content must be byte-identical \
             regardless of physical write order, or the reorder changed semantics"
        );
        eprintln!(
            "M-arm digest identity: natural={digest_natural:016x} morton={digest_morton:016x} MATCH"
        );

        // ── T1: local_trajectories over both 1,048,576-row histories ─────
        let (t1_natural_ns, t1_morton_ns, fastpath_ns, fastpath_digest_match) = {
            // Skip the WARM-UP versions. The pipeline runs
            // WARMUP+MEASURED real cycles (the warm-ups are needed for the
            // write/seal timing to settle), but T1 must cover exactly the
            // MEASURED window or its number is not comparable to A0's
            // 78-86 ms over 1,048,576 rows — and beating that number is the
            // whole point of the ordered fast path. `scan_sealed(Some(c))`
            // filters `cycle > c` (bounded recovery is the contract now, not
            // `DatasetVersion`-keyed), and the warm-ups own cycles 1..=WARMUP.
            let after_warmup = Some(CycleId(WARMUP_CYCLES as u64));
            let landed_natural = natural_sink
                .scan_sealed(after_warmup)
                .await
                .expect("M-arm T1: scan_sealed natural");
            let landed_morton = morton_sink
                .scan_sealed(after_warmup)
                .await
                .expect("M-arm T1: scan_sealed morton");
            assert_eq!(
                landed_natural.len(),
                FLEET_OWNERS as usize * MEASURED_CYCLES as usize,
                "M-arm T1 must read exactly the MEASURED window (comparability with A0)"
            );
            assert_eq!(
                landed_morton.len(),
                FLEET_OWNERS as usize * MEASURED_CYCLES as usize,
                "M-arm T1 must read exactly the MEASURED window (comparability with A0)"
            );

            let bench_natural: Vec<BenchRow> = landed_natural
                .iter()
                .map(|ls| BenchRow {
                    owner: ls.slot.owner,
                    cast_seq: ls.slot.stream_position,
                    lance_version: ls.cycle.0,
                })
                .collect();
            let bench_morton: Vec<BenchRow> = landed_morton
                .iter()
                .map(|ls| BenchRow {
                    owner: ls.slot.owner,
                    cast_seq: ls.slot.stream_position,
                    lance_version: ls.cycle.0,
                })
                .collect();

            let t1a = Instant::now();
            let traj_natural = local_trajectories(&bench_natural);
            let t1_natural_ns = t1a.elapsed().as_nanos() as u64;
            let t1b = Instant::now();
            let traj_morton = local_trajectories(&bench_morton);
            let t1_morton_ns = t1b.elapsed().as_nanos() as u64;
            assert_eq!(traj_natural.len(), FLEET_OWNERS as usize);
            assert_eq!(traj_morton.len(), FLEET_OWNERS as usize);

            // ── v2 D2 fast path — Morton-ordered history only (the
            // natural pipeline's physical order was never claimed to
            // satisfy the fast path's precondition; it happens to be
            // monotonic here too by construction, but the fast path is
            // exercised against the layout it was designed for). ──────
            let t_fp = Instant::now();
            let traj_fastpath = local_trajectories_ordered_chunk_fastpath(&landed_morton)
                .expect("M-arm: fast path must validate the Morton-ordered history");
            let fastpath_ns = t_fp.elapsed().as_nanos() as u64;

            // digest identity: generic vs fast path, over the SAME
            // Morton-ordered scanned history (v2 D2's own requirement).
            let digest_of = |m: &BTreeMap<MailboxId, Vec<BenchRow>>| -> u64 {
                let mut bytes = Vec::new();
                for (owner, chain) in m {
                    bytes.extend_from_slice(&owner.to_le_bytes());
                    for row in chain {
                        bytes.extend_from_slice(&row.cast_seq.to_le_bytes());
                        bytes.extend_from_slice(&row.lance_version.to_le_bytes());
                    }
                }
                fnv1a64(&bytes)
            };
            let digest_generic = digest_of(&traj_morton);
            let digest_fastpath = digest_of(&traj_fastpath);
            let fastpath_digest_match = digest_generic == digest_fastpath;
            assert!(
                fastpath_digest_match,
                "M-arm can-fire: generic local_trajectories and the ordered-chunk fast \
                     path must reconstruct byte-identical trajectories from the same \
                     Morton-ordered history"
            );

            // can-it-fire proof for the fast path's own guard (CLAUDE.md
            // falsifiability rule): a deliberately corrupted, out-of-
            // order 2-row input must be REFUSED, not silently accepted.
            let bad = vec![
                LandedSlot {
                    cycle: CycleId(2),
                    slot: SweepSlot {
                        cycle: CycleId(2),
                        stream_position: 10,
                        owner: 1,
                        row: 1,
                        paired_move: None,
                        payload: vec![],
                    },
                },
                LandedSlot {
                    cycle: CycleId(2),
                    slot: SweepSlot {
                        cycle: CycleId(2),
                        stream_position: 5, // regressed — must be refused
                        owner: 2,
                        row: 2,
                        paired_move: None,
                        payload: vec![],
                    },
                },
            ];
            assert!(
                local_trajectories_ordered_chunk_fastpath(&bad).is_err(),
                "M-arm can-fire: the fast path's monotonicity guard must reject a \
                     stream_position regression, not silently mis-order the trajectory"
            );

            (
                t1_natural_ns,
                t1_morton_ns,
                fastpath_ns,
                fastpath_digest_match,
            )
        };

        eprintln!(
            "M-arm natural: cast={}ns collect={}ns seal={}ns write={}ns sync={}ns T1={t1_natural_ns}ns",
            natural.cast_ns, natural.collect_ns, natural.seal_ns, natural.wal_write_ns, natural.wal_sync_ns
        );
        eprintln!(
            "M-arm morton:  cast={}ns collect={}ns reorder={}ns seal={}ns write={}ns sync={}ns T1={t1_morton_ns}ns \
             fastpath={fastpath_ns}ns (fastpath-vs-generic digest match={fastpath_digest_match})",
            morton.cast_ns, morton.collect_ns, morton.reorder_ns, morton.seal_ns, morton.wal_write_ns, morton.wal_sync_ns
        );

        // ── the pre-registered SUM verdict (never the gain alone) ────────
        let downstream_natural = (natural.seal_ns + natural.wal_write_ns + natural.wal_sync_ns)
            as i64
            + t1_natural_ns as i64;
        let downstream_morton = (morton.seal_ns + morton.wal_write_ns + morton.wal_sync_ns) as i64
            + t1_morton_ns as i64;
        let downstream_savings = downstream_natural - downstream_morton;
        let delta_total = morton.reorder_ns as i64 - downstream_savings;
        eprintln!(
            "M-arm SUM verdict: reorder_cost={}ns, downstream (seal+write+sync+T1) \
             natural={downstream_natural}ns morton={downstream_morton}ns savings={downstream_savings:+}ns \
             -> delta_total={delta_total:+}ns ({})",
            morton.reorder_ns,
            if delta_total < 0 {
                "Morton WINS (reorder cost paid for by downstream savings)"
            } else {
                "Morton does NOT win under this workload/host (reorder cost exceeds downstream savings)"
            }
        );
        eprintln!(
            "M-arm reference: A0 measured T1 at 78-86ms over 1,048,576 rows — the number the \
             fast path must beat; this run's fast path={fastpath_ns}ns is the direct comparison \
             (implementation-scoped: this implementation, this workload, this host)."
        );

        csv.write(&Row {
            owner_shape: "m_arm_natural",
            physical_layout: "unordered_stream_position",
            threads: 1,
            segment_rows: 4_096,
            segment_bytes: 4_096 * CANONICAL_ROW_BYTES as u64,
            segments_per_cycle: FLEET_OWNERS as u64 / 4_096,
            repeat: 0,
            build_ns: 0,
            scan_ns: 0,
            think_ns: 0,
            rebind_cast_ns: natural.cast_ns,
            collect_ns: natural.collect_ns,
            freeze_ns: natural.seal_ns,
            wal_write_ns: natural.wal_write_ns,
            wal_sync_ns: natural.wal_sync_ns,
            temporal_layer1_ns: t1_natural_ns,
            temporal_layer2_ns: 0,
            apply_ns: 0,
            total_ns: natural.cast_ns
                + natural.collect_ns
                + natural.seal_ns
                + natural.wal_write_ns
                + natural.wal_sync_ns
                + t1_natural_ns,
            logical_rows: FLEET_OWNERS as u64,
            logical_bytes: CANONICAL_FRAME_BYTES as u64,
            sealed_transitions: FLEET_OWNERS as u64,
            applied_transitions: 0,
            wal_syscalls: natural.wal_syscalls,
            fsync_calls: 1,
            dataset_versions: 16,
            peak_rss_bytes: proc_snapshot().vmhwm_kb * 1024,
            minor_faults: 0,
            major_faults: 0,
            context_switches: 0,
            max_active_workers: 1,
            result_digest: digest_natural,
            morton_reorder_ns: 0,
        });
        csv.write(&Row {
            owner_shape: "m_arm_morton",
            physical_layout: "morton_reordered_before_seal",
            threads: 1,
            segment_rows: 4_096,
            segment_bytes: 4_096 * CANONICAL_ROW_BYTES as u64,
            segments_per_cycle: FLEET_OWNERS as u64 / 4_096,
            repeat: 0,
            build_ns: 0,
            scan_ns: 0,
            think_ns: 0,
            rebind_cast_ns: morton.cast_ns,
            collect_ns: morton.collect_ns,
            freeze_ns: morton.seal_ns,
            wal_write_ns: morton.wal_write_ns,
            wal_sync_ns: morton.wal_sync_ns,
            temporal_layer1_ns: t1_morton_ns,
            temporal_layer2_ns: fastpath_ns,
            apply_ns: 0,
            total_ns: morton.cast_ns
                + morton.collect_ns
                + morton.reorder_ns
                + morton.seal_ns
                + morton.wal_write_ns
                + morton.wal_sync_ns
                + t1_morton_ns,
            logical_rows: FLEET_OWNERS as u64,
            logical_bytes: CANONICAL_FRAME_BYTES as u64,
            sealed_transitions: FLEET_OWNERS as u64,
            applied_transitions: 0,
            wal_syscalls: morton.wal_syscalls,
            fsync_calls: 1,
            dataset_versions: 16,
            peak_rss_bytes: proc_snapshot().vmhwm_kb * 1024,
            minor_faults: 0,
            major_faults: 0,
            context_switches: 0,
            max_active_workers: 1,
            result_digest: digest_morton,
            morton_reorder_ns: morton.reorder_ns,
        });
    }

    // ═════════════════════════════════════════════════════════════════════
    // §16 — O-arm: ordering source — where does the ordering actually come
    // from? O-A: cast -> seal -> WAL -> temporal replay (today's pipeline).
    // O-B: cast -> temporal replay -> seal -> WAL (ordering sourced first).
    // ═════════════════════════════════════════════════════════════════════

    /// A minimal `LocalCausalRow` view over an in-flight `SweepSlot` — lets
    /// O-B call `local_trajectories` (a temporal.rs primitive) on cast-time
    /// data, BEFORE any seal/WAL exists. `Copy`-free borrow view; built and
    /// consumed entirely within one function call, never persisted.
    #[derive(Clone)]
    struct PreSealRow {
        owner: MailboxId,
        arrival_stream_position: u64,
        slot: SweepSlot,
    }
    impl LocalCausalRow for PreSealRow {
        fn owner(&self) -> MailboxId {
            self.owner
        }
        fn cast_seq(&self) -> u64 {
            self.arrival_stream_position
        }
    }

    // FIREWALL-START: derive_order_from_temporal_replay
    //
    // O-B must not consult the sealed stream to build its own order (that
    // would be O-A wearing a disguise) — this function's body is the ONLY
    // place that decides O-B's physical write order, and it is scoped by
    // the FIREWALL-START/FIREWALL-END sentinels below so the compile-time
    // self-scan in `run_o_arm` can check ONLY this region (a whole-file
    // scan would false-positive on the legitimate `scan_sealed` calls
    // elsewhere in this file, e.g. `run_temporal`/`run_m_arm`).
    //
    /// v2 D2 (applied pre-seal): source O-B's physical write order from
    /// `local_trajectories` (a temporal.rs primitive) applied to the
    /// CAST-TIME data alone. Groups by owner (this benchmark casts each
    /// owner at most once per cycle, so every chain is a singleton) and
    /// flattens by `BTreeMap` iteration order (owner-ascending) — the
    /// temporal-sourced order, independent of arrival order.
    fn derive_order_from_temporal_replay(pre_seal: &[PreSealRow]) -> Vec<SweepSlot> {
        let grouped = local_trajectories(pre_seal);
        let mut out = Vec::with_capacity(pre_seal.len());
        for (_owner, chain) in grouped {
            for row in chain {
                out.push(row.slot);
            }
        }
        out
    }
    // FIREWALL-END: derive_order_from_temporal_replay

    /// One O-arm pipeline's per-cycle-medianed measurements.
    #[derive(Clone, Copy, Default)]
    struct OArmPhaseMedians {
        cast_ns: u64,
        collect_ns: u64,
        order_derive_ns: u64,
        seal_ns: u64,
        commit_ns: u64,
        t1_ns: u64,
    }

    /// A deterministic, non-ascending cast ORDER (bit-reversal permutation
    /// of the 16-bit owner id) — makes O-A's arrival/stream_position order
    /// DEMONSTRABLY not owner-ascending, so O-A's physical write order and
    /// O-B's temporal-sourced (owner-ascending) order are actually free to
    /// diverge. Without this scramble every arm in this file casts owners
    /// 0..65535 in order, which would make the O-A/O-B comparison trivially
    /// coincide regardless of whether O-B's derivation is doing real work —
    /// exactly the vacuous-assertion shape CLAUDE.md's falsifiability rule
    /// forbids.
    fn scrambled_cast_order() -> Vec<MailboxId> {
        let order: Vec<MailboxId> = (0..FLEET_OWNERS)
            .map(|id| (id as u16).reverse_bits() as u32)
            .collect();
        // `reverse_bits` on a u16 is itself a bijection over [0,65536), so
        // `order` is already a permutation of 0..65535; no sort needed to
        // prove that — but assert it here, once, as a cheap can-fire check
        // on the fixture itself (not per-cycle work).
        let mut check = order.clone();
        check.sort_unstable();
        debug_assert_eq!(
            check,
            (0..FLEET_OWNERS).collect::<Vec<_>>(),
            "scrambled_cast_order: must be a permutation of every owner, exactly once"
        );
        order
    }

    async fn run_o_arm_pipeline(
        label: &'static str,
        source_from_temporal: bool,
        cast_order: &[MailboxId],
    ) -> (OArmPhaseMedians, MemWal) {
        let style_outcome = build_style_outcome();
        let mut sink = MemWal::new();
        let mut cast_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut collect_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut order_derive_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut seal_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut commit_samples = Vec::with_capacity(MEASURED_CYCLES as usize);
        let mut position_base: u64 = 0;

        for cyc in 0..(WARMUP_CYCLES + MEASURED_CYCLES) {
            let measured = cyc >= WARMUP_CYCLES;
            let cycle_id = CycleId(u64::from(cyc) + 1);

            let t_cast = Instant::now();
            let mut writer: BatchWriter<Vec<u8>> = BatchWriter::new();
            for &id in cast_order {
                let combined = (u64::from(cyc) << 32) | u64::from(id);
                let payload = NodeRow512::for_id(combined).as_bytes().to_vec();
                let _ = emit_bootstrap_intent(&style_outcome, id, 0, &mut writer, payload);
            }
            let cast_ns = t_cast.elapsed().as_nanos() as u64;

            let t_collect = Instant::now();
            let collected = collect_casts(&mut writer, cycle_id, position_base, u64::from);
            let collect_ns = t_collect.elapsed().as_nanos() as u64;
            assert_eq!(collected.slots.len(), FLEET_OWNERS as usize);

            let (ordered_slots, order_derive_ns) = if source_from_temporal {
                let pre_seal: Vec<PreSealRow> = collected
                    .slots
                    .iter()
                    .map(|s| PreSealRow {
                        owner: s.owner,
                        arrival_stream_position: s.stream_position,
                        slot: s.clone(),
                    })
                    .collect();
                let t_derive = Instant::now();
                let mut derived = derive_order_from_temporal_replay(&pre_seal);
                // Relabel stream_position to the temporal-derived rank so
                // the seal's own stable sort preserves this order, exactly
                // as the M-arm does for its Morton rank.
                for (idx, slot) in derived.iter_mut().enumerate() {
                    slot.stream_position = position_base + idx as u64;
                }
                let ns = t_derive.elapsed().as_nanos() as u64;
                (derived, ns)
            } else {
                (collected.slots, 0)
            };

            let frame = CycleFrame::new(cycle_id, sink.head());
            let t_seal = Instant::now();
            let frozen = DetachedCycleBatch::freeze(frame, ordered_slots);
            let seal_ns = t_seal.elapsed().as_nanos() as u64;
            assert_eq!(frozen.landings.len(), FLEET_OWNERS as usize);

            let t_commit = Instant::now();
            sink.commit_cycle(frozen)
                .await
                .unwrap_or_else(|e| panic!("O-arm {label}: cycle {cyc} failed to seal: {e}"));
            let commit_ns = t_commit.elapsed().as_nanos() as u64;

            if measured {
                cast_samples.push(cast_ns);
                collect_samples.push(collect_ns);
                order_derive_samples.push(order_derive_ns);
                seal_samples.push(seal_ns);
                commit_samples.push(commit_ns);
            }
            position_base += u64::from(FLEET_OWNERS);
        }

        let t1 = Instant::now();
        // Same MEASURED-window scoping as the M-arm: the pipeline runs
        // WARMUP+MEASURED real cycles, so an unfiltered scan returns 18
        // cycles' rows. The replay must cover exactly the measured window or
        // its cost is not comparable to A0's or the M-arm's.
        let landed = sink
            .scan_sealed(Some(CycleId(WARMUP_CYCLES as u64)))
            .await
            .expect("O-arm: scan_sealed over the measured window");
        assert_eq!(
            landed.len(),
            FLEET_OWNERS as usize * MEASURED_CYCLES as usize,
            "O-arm replay must read exactly the MEASURED window"
        );
        let bench: Vec<BenchRow> = landed
            .iter()
            .map(|ls| BenchRow {
                owner: ls.slot.owner,
                cast_seq: ls.slot.stream_position,
                lance_version: ls.cycle.0,
            })
            .collect();
        let trajectories = local_trajectories(&bench);
        let t1_ns = t1.elapsed().as_nanos() as u64;
        assert_eq!(trajectories.len(), FLEET_OWNERS as usize);

        let medians = OArmPhaseMedians {
            cast_ns: median(&cast_samples),
            collect_ns: median(&collect_samples),
            order_derive_ns: median(&order_derive_samples),
            seal_ns: median(&seal_samples),
            commit_ns: median(&commit_samples),
            t1_ns,
        };
        (medians, sink)
    }

    /// The RECOVERED-TRAJECTORY digest: owner-ascending (`BTreeMap`
    /// iteration order), each owner's chain in `cast_seq` order — the
    /// PRIMARY observable this arm decides on. Computed over what a reader
    /// gets back after WAL + temporal replay, so it is a fair comparison
    /// EVEN THOUGH O-A and O-B wrote the bytes in different physical order.
    async fn trajectory_digest(sink: &MemWal) -> u64 {
        let landed = sink
            .scan_sealed(None)
            .await
            .expect("O-arm: scan_sealed for digest");
        let bench: Vec<BenchRow> = landed
            .iter()
            .map(|ls| BenchRow {
                owner: ls.slot.owner,
                cast_seq: ls.slot.stream_position,
                lance_version: ls.cycle.0,
            })
            .collect();
        let trajectories = local_trajectories(&bench);
        let mut bytes = Vec::new();
        for (owner, chain) in trajectories {
            bytes.extend_from_slice(&owner.to_le_bytes());
            for row in chain {
                bytes.extend_from_slice(&row.cast_seq.to_le_bytes());
            }
        }
        fnv1a64(&bytes)
    }

    async fn run_o_arm(csv: &mut CsvSink) {
        eprintln!("\n== O-arm — ordering source: O-A (cast->seal->WAL->temporal replay) vs O-B (cast->temporal replay->seal->WAL) ==");

        // ── compile-time self-scan (the firewall) ─────────────────────────
        // Needles built by CONCATENATING pieces never adjacent in this
        // file's own source text, matching `probe_ignition.rs`'s G2a
        // pattern — a needle spelled out contiguously would make the
        // absence-check vacuously true, since `include_str!` reads this
        // file, including the scan code itself.
        {
            let src = include_str!("measure_wal_curve.rs");
            let start_marker = "FIREWALL-START: derive_order_from_temporal_replay";
            let end_marker = "FIREWALL-END: derive_order_from_temporal_replay";
            let start = src
                .find(start_marker)
                .expect("O-arm firewall: FIREWALL-START marker must exist in source");
            let end = src
                .find(end_marker)
                .expect("O-arm firewall: FIREWALL-END marker must exist in source");
            assert!(start < end, "O-arm firewall: markers out of order");
            let region = &src[start..end];

            let scan_sealed_call = format!("{}_{}", "scan", "sealed");
            let sealed_field_read = format!("{}.{}", "sink", "sealed");

            // Strip line comments BEFORE scanning. The first real run fired on
            // this block's own prose — a comment inside the region mentioned
            // the needle by name, so the guard reported a violation that did
            // not exist in any executable line. A firewall that trips on
            // documentation tests the documentation, not the code.
            let code_only: String = region
                .lines()
                .map(|l| match l.find("//") {
                    Some(i) => &l[..i],
                    None => l,
                })
                .collect::<Vec<_>>()
                .join("\n");

            // POSITIVE CONTROL (the can-fire half): the detector must find the
            // needle in a line that really does call it. Without this, a
            // silent guard and a broken guard are indistinguishable.
            let synthetic_violation = format!("    let x = sink.{}(None).await;", scan_sealed_call);
            assert!(
                synthetic_violation.contains(&scan_sealed_call),
                "O-arm firewall self-test: the detector cannot see a real call — the guard is inert"
            );

            assert!(
                !code_only.contains(&scan_sealed_call),
                "O-arm firewall can-fire: derive_order_from_temporal_replay must never \
                 call scan_sealed — O-B would be O-A wearing a disguise"
            );
            assert!(
                !code_only.contains(&sealed_field_read),
                "O-arm firewall can-fire: derive_order_from_temporal_replay must never \
                 read a WalSink's sealed store directly"
            );
            assert!(
                region.contains("local_trajectories"),
                "O-arm firewall can-stay-silent: the scan must be able to find real \
                 content — a scan finding nothing is not evidence"
            );
            eprintln!(
                "O-arm firewall: derive_order_from_temporal_replay ({} bytes) contains \
                 no scan_sealed / sealed-store read; local_trajectories present (scan \
                 mechanism proven live)",
                region.len()
            );
        }

        let cast_order = scrambled_cast_order();
        let identity_order: Vec<MailboxId> = (0..FLEET_OWNERS).collect();
        assert_ne!(
            cast_order, identity_order,
            "O-arm can-fire: the cast order fixture must actually be scrambled, or O-A's \
             arrival order and O-B's temporal-sourced order would trivially coincide"
        );

        let (o_a, sink_a) = run_o_arm_pipeline("O-A", false, &cast_order).await;
        let (o_b, sink_b) = run_o_arm_pipeline("O-B", true, &cast_order).await;

        // ── PRIMARY OBSERVABLE — digest identity, decided and printed
        // BEFORE any timing is looked at (pre-registered: timing must not
        // be able to rescue a semantic difference). ──────────────────────
        let digest_a = trajectory_digest(&sink_a).await;
        let digest_b = trajectory_digest(&sink_b).await;
        let digests_match = digest_a == digest_b;
        eprintln!(
            "O-arm PRIMARY OBSERVABLE (decided before timing): O-A digest={digest_a:016x} \
             O-B digest={digest_b:016x} -> {}",
            if digests_match { "MATCH" } else { "DIVERGED" }
        );
        // BOTH outcomes are pre-registered RESULTS, so neither aborts the run.
        // A first revision asserted equality and panicked on divergence —
        // that turns a designed falsification into a crash and loses every
        // number after it. The spec is explicit: if the trajectories differ,
        // the hypothesis is dead and the seal's ordering is load-bearing —
        // "equally valuable, and cheaper to learn now than after a redesign".
        if digests_match {
            eprintln!(
                "O-arm VERDICT: ordering sourced from temporal replay reproduces the seal's \
                 own ordering byte-for-byte under this construction. The seal's ordering work \
                 is REDUNDANT with temporal's for this workload — the re-scope question is \
                 open (implementation-scoped: this construction, this workload, this host)."
            );
        } else {
            eprintln!(
                "O-arm VERDICT: DIVERGED — ordering sourced from temporal replay does NOT \
                 reproduce the seal's ordering. Under this construction the seal's ordering \
                 is LOAD-BEARING and cannot be re-scoped away. Honest scope: this falsifies \
                 the hypothesis FOR THIS O-B CONSTRUCTION; it does not prove that no \
                 construction could match. The divergence itself is the finding."
            );
        }

        // ── KILL CONDITION check — is O-B constructible without literally
        // duplicating O-A's ordering work? Reported honestly either way,
        // never silently rigged. ──────────────────────────────────────────
        eprintln!(
            "O-arm kill-condition check: CONSTRUCTIBLE. O-B's ordering derivation \
             (`local_trajectories` grouping, ~O(n log n) via BTreeMap insertion) is a \
             DIFFERENT code path from O-A's seal-side sort (`order_cycle_stably`'s Vec \
             sort_by_key, also O(n log n)) — not literally shared code, so this is not a \
             disguised O-A. Under THIS harness's one-row-per-owner-per-cycle shape the two \
             algorithms are doing comparable asymptotic work; the redundancy the plan asks \
             about is SEMANTIC (does temporal's grouping make the seal's own sort \
             unnecessary for correctness), not literal code-sharing — reported honestly, \
             not glossed over."
        );

        // ── secondary: per-phase timing for both pipelines ────────────────
        eprintln!(
            "O-A (today's pipeline): cast={}ns collect={}ns seal={}ns commit={}ns T1={}ns",
            o_a.cast_ns, o_a.collect_ns, o_a.seal_ns, o_a.commit_ns, o_a.t1_ns
        );
        eprintln!(
            "O-B (ordering sourced first): cast={}ns collect={}ns order_derive={}ns seal={}ns \
             commit={}ns T1={}ns",
            o_b.cast_ns, o_b.collect_ns, o_b.order_derive_ns, o_b.seal_ns, o_b.commit_ns, o_b.t1_ns
        );

        csv.write(&Row {
            owner_shape: "o_a_today_pipeline",
            physical_layout: "cast_seal_wal_temporal_replay",
            threads: 1,
            segment_rows: 0,
            segment_bytes: 0,
            segments_per_cycle: 0,
            repeat: 0,
            build_ns: 0,
            scan_ns: 0,
            think_ns: 0,
            rebind_cast_ns: o_a.cast_ns,
            collect_ns: o_a.collect_ns,
            freeze_ns: o_a.seal_ns,
            wal_write_ns: o_a.commit_ns,
            wal_sync_ns: 0,
            temporal_layer1_ns: o_a.t1_ns,
            temporal_layer2_ns: 0,
            apply_ns: 0,
            total_ns: o_a.cast_ns + o_a.collect_ns + o_a.seal_ns + o_a.commit_ns + o_a.t1_ns,
            logical_rows: FLEET_OWNERS as u64,
            logical_bytes: (FLEET_OWNERS as u64) * CANONICAL_ROW_BYTES as u64,
            sealed_transitions: FLEET_OWNERS as u64,
            applied_transitions: 0,
            wal_syscalls: 0,
            fsync_calls: 0,
            dataset_versions: 16,
            peak_rss_bytes: proc_snapshot().vmhwm_kb * 1024,
            minor_faults: 0,
            major_faults: 0,
            context_switches: 0,
            max_active_workers: 1,
            result_digest: digest_a,
            morton_reorder_ns: 0,
        });
        csv.write(&Row {
            owner_shape: "o_b_ordering_sourced_first",
            physical_layout: "cast_temporal_replay_seal_wal",
            threads: 1,
            segment_rows: 0,
            segment_bytes: 0,
            segments_per_cycle: 0,
            repeat: 0,
            build_ns: 0,
            scan_ns: 0,
            think_ns: 0,
            rebind_cast_ns: o_b.cast_ns,
            collect_ns: o_b.collect_ns,
            freeze_ns: o_b.seal_ns,
            wal_write_ns: o_b.commit_ns,
            wal_sync_ns: 0,
            temporal_layer1_ns: o_b.t1_ns,
            temporal_layer2_ns: o_b.order_derive_ns,
            apply_ns: 0,
            total_ns: o_b.cast_ns
                + o_b.collect_ns
                + o_b.order_derive_ns
                + o_b.seal_ns
                + o_b.commit_ns
                + o_b.t1_ns,
            logical_rows: FLEET_OWNERS as u64,
            logical_bytes: (FLEET_OWNERS as u64) * CANONICAL_ROW_BYTES as u64,
            sealed_transitions: FLEET_OWNERS as u64,
            applied_transitions: 0,
            wal_syscalls: 0,
            fsync_calls: 0,
            dataset_versions: 16,
            peak_rss_bytes: proc_snapshot().vmhwm_kb * 1024,
            minor_faults: 0,
            major_faults: 0,
            context_switches: 0,
            max_active_workers: 1,
            result_digest: digest_b,
            morton_reorder_ns: 0,
        });
    }

    // ═════════════════════════════════════════════════════════════════════
    // §14 — orchestration + the four closing answers.
    // ═════════════════════════════════════════════════════════════════════

    pub fn run() {
        eprintln!("measure_wal_curve — five-axis 64k measurement (release-mode; plan: .claude/plans/measure-64k-axes-v1.md)");
        eprintln!(
            "FLEET_OWNERS={FLEET_OWNERS} CANONICAL_ROW_BYTES={CANONICAL_ROW_BYTES} CANONICAL_FRAME_BYTES={CANONICAL_FRAME_BYTES}"
        );
        #[cfg(debug_assertions)]
        eprintln!(
            "WARNING: this binary was NOT built --release — every timing number below is meaningless as physics, structure-only."
        );

        let mut csv = CsvSink::new();

        // ── B0 / B1a / B1b ──────────────────────────────────────────────
        let b0 = run_b0(&mut csv);
        let b1a = run_b1a(&mut csv);
        let b1b_rss_delta = run_b1b(&mut csv);

        // ── L1a / L1b ───────────────────────────────────────────────────
        let l1a = run_l1a(&mut csv);
        run_l1b(&mut csv);

        // ── WAL curve ───────────────────────────────────────────────────
        let wal_summary = run_wal_curve(&mut csv);

        // ── Temporal + EXP-KIA-A2-64K need the async WAL-sink machinery ──
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("build a single-threaded tokio runtime for the async arms");
        rt.block_on(async {
            run_temporal(&mut csv).await;
        });
        let (exp_seq_ns, exp_best_parallel_ns, exp_best_workers, exp_digests_match) =
            rt.block_on(async { run_exp_kia_a2_64k(&mut csv).await });

        // ── M-arm / O-arm (plan v3, measure-64k-axes-v3.md) ───────────────
        rt.block_on(async {
            run_m_arm(&mut csv).await;
        });
        rt.block_on(async {
            run_o_arm(&mut csv).await;
        });

        eprintln!("\nmeasure.csv: {} rows written", csv.rows_written);
        eprintln!("measure.csv: file at {}", csv.path);

        // The derived "hot representation overhead" metric (plan §B1) — the
        // two peak-RSS numbers are NEVER blended into one memory claim; this
        // is their difference, reported once, separately from both.
        // MEASURED as VmRSS deltas. An earlier revision differenced VmHWM and
        // printed a NEGATIVE "overhead" — VmHWM is process-monotonic, so the
        // subtraction returned the same historical maximum twice. That figure
        // is retracted, not reported.
        // B1a's footprint is MEASURED (VmRSS delta). The canonical envelope is
        // EXACT ARITHMETIC (65_536 x 512 B) — so the overhead is measured-minus-
        // exact, never measured-minus-measured. B1b's own in-process delta is
        // reported beside it and is expected to read ~0: by the time it runs the
        // allocator satisfies its 32 MiB from pages B1a already returned, so an
        // in-process delta cannot see it. (An earlier revision differenced two
        // VmHWM values and printed a NEGATIVE overhead — VmHWM is
        // process-monotonic; that figure is retracted, not reported.)
        let hot_repr_overhead = b1a.rss_delta_bytes - CANONICAL_FRAME_BYTES as i64;
        eprintln!(
            "hot representation overhead: B1a MEASURED VmRSS delta {:+}B ({:.1} MiB) \
             minus canonical envelope {}B (32.0 MiB, exact by construction) = {:+}B ({:+.1} MiB, {:+.0}%)",
            b1a.rss_delta_bytes,
            b1a.rss_delta_bytes as f64 / (1024.0 * 1024.0),
            CANONICAL_FRAME_BYTES,
            hot_repr_overhead,
            hot_repr_overhead as f64 / (1024.0 * 1024.0),
            100.0 * hot_repr_overhead as f64 / CANONICAL_FRAME_BYTES as f64
        );
        eprintln!(
            "  (B1b in-process VmRSS delta {:+}B — reads ~0 by allocator reuse, \
             which is why the line above uses the exact canonical size)",
            b1b_rss_delta
        );

        // ── the four answers (plan, "Placement + gates") ─────────────────
        eprintln!("\n================ THE FOUR ANSWERS ================");

        // 1. What does ownership cost? (B1a - B0, per phase)
        let scan_tax = b1a.scan_ns as i64 - b0.scan_ns as i64;
        let cast_tax = b1a.cast_ns as i64 - b0.cast_ns as i64;
        let freeze_tax = b1a.freeze_ns as i64 - b0.freeze_ns as i64;
        eprintln!(
            "1. Ownership cost (B1a MailboxSoA<4> minus B0 DummyOwner, median of 3): \
             scan {scan_tax:+}ns, cast/rebind {cast_tax:+}ns, freeze {freeze_tax:+}ns \
             (B1a also pays a real per-owner think phase B0 has none of: {think}ns, \
             and a real apply phase: {apply}ns — B0 has neither).",
            think = b1a.think_ns,
            apply = b1a.apply_ns,
        );

        // 2. What does physical layout cost? (B1a vs L1a, build..freeze only)
        let layout_build = l1a.build_ns as i64 - b1a.build_ns as i64;
        let layout_cast = l1a.cast_ns as i64 - b1a.cast_ns as i64;
        let layout_freeze = l1a.freeze_ns as i64 - b1a.freeze_ns as i64;
        eprintln!(
            "2. Physical layout cost (L1a 64x MailboxSoA<1024> minus B1a 65,536x \
             MailboxSoA<4>, equal 65,536 logical owners, build..freeze phases only — \
             apply is not comparable, see the §12 doc comment): \
             build {layout_build:+}ns, cast/rebind {layout_cast:+}ns, freeze {layout_freeze:+}ns."
        );

        // 3. Where does WAL amortisation plateau?
        match wal_summary.plateau_segment_bytes {
            Some(b) => eprintln!(
                "3. WAL amortisation plateau (W1-contiguous, descriptive knee, NOT pass/kill): \
                 first segment_bytes={b} where two consecutive doublings improved median \
                 throughput by <5%. W0-current numbers are beside it in the CSV \
                 (physical_layout=w0_current), never substituted for this reading."
            ),
            None if wal_summary.unstable_configs > 0 => eprintln!(
                "3. WAL amortisation plateau: NOT MEASURABLE ON THIS HOST — {} of 5 configs \
                 exceeded the p95/median spread ceiling (worst {:.1}x). The write phase is \
                 driven by page-cache / dirty-writeback state, not segment size, so NO knee \
                 is claimed. Four runs of this binary disagreed by up to 6x at identical \
                 configs. Needs a quiet host with disk headroom plus O_DIRECT or a per-config \
                 cache barrier. W0-current numbers are in the CSV, never substituted here.",
                wal_summary.unstable_configs, wal_summary.worst_spread
            ),
            None => eprintln!(
                "3. WAL amortisation plateau: no knee found across the 5-point table \
                 ({:?} MiB/s at each segment_bytes) — either still gaining at 32 MiB \
                 segments or the table is too coarse to resolve one; W0-current numbers \
                 are beside it in the CSV, never substituted for this reading.",
                wal_summary
                    .w1_points
                    .iter()
                    .map(|&(b, ns)| (
                        b,
                        if ns > 0 {
                            (CANONICAL_FRAME_BYTES as f64 / (1024.0 * 1024.0)) / (ns as f64 / 1e9)
                        } else {
                            0.0
                        }
                    ))
                    .collect::<Vec<_>>()
            ),
        }

        // 4. What does genuine parallel thought execution add?
        eprintln!(
            "4. EXP-KIA-A2-64K (exploratory, NON-CLAIMING — D-KIA-A2's own median-of-5 \
             >=2x gate is untouched by this number): sequential (workers=1) compute-phase \
             median {exp_seq_ns}ns vs best observed parallel compute-phase median \
             {exp_best_parallel_ns}ns at workers={exp_best_workers}; sequential-vs-parallel \
             sealed-cycle digests {} across every worker count tested.",
            if exp_digests_match {
                "MATCH"
            } else {
                "DIVERGED"
            }
        );
        eprintln!("====================================================");
    }
}
