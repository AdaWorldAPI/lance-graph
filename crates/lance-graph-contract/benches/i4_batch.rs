//! Criterion benchmarks for i4_eval batch functions — D-CSV-13b
//!
//! SHIP gate: ≥4× AVX-512 vs scalar for dk/trust/flow/gate_disc at batch 1024.
//! LAND gate: ≥2× (records TD-D-CSV-13b-PERF-FLOOR-1 if ≥2 but <4).
//! mul_assess target: ≥2.5× (limited by scalar f64 finalize stage).

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use lance_graph_contract::mul::i4_eval::batch;
use lance_graph_contract::qualia::QualiaI4_16D;

fn make_inputs(n: usize) -> (Vec<QualiaI4_16D>, Vec<i8>) {
    let template: &[(i8, i8)] = &[
        (7, 5),
        (5, 4),
        (3, 3),
        (2, 2),
        (0, 2),
        (-1, 1),
        (-3, -2),
        (-5, -4),
        (6, 0),
        (1, -1),
        (4, 4),
        (-2, -3),
        (3, 1),
        (0, -1),
        (7, -5),
        (-4, 4),
    ];
    let mut qualia = Vec::with_capacity(n);
    let mut mantissas = Vec::with_capacity(n);
    for i in 0..n {
        let (coh, mant) = template[i % template.len()];
        let mut q = QualiaI4_16D::ZERO;
        q.set(9, coh); // dim 9 = coherence
        q.set(3, coh.saturating_add(1).clamp(-8, 7)); // warmth
        q.set(14, coh.saturating_sub(1).clamp(-8, 7)); // groundedness
        q.set(2, (mant.abs() % 6).clamp(0, 7)); // tension
        q.set(1, mant.clamp(-8, 7)); // valence
        qualia.push(q);
        mantissas.push(mant);
    }
    (qualia, mantissas)
}

fn bench_dk_position(c: &mut Criterion) {
    let mut group = c.benchmark_group("dk_position_batch");
    for &size in &[8usize, 64, 1024, 16384] {
        let (q, m) = make_inputs(size);
        let mut out = vec![lance_graph_contract::mul::DkPosition::MountStupid; size];

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |b, _| {
            b.iter(|| {
                batch::dk_position_batch(&q, &m, &mut out);
                criterion::black_box(&out);
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                batch::scalar_impl::dk_position_batch(&q, &m, &mut out);
                criterion::black_box(&out);
            });
        });
    }
    group.finish();
}

fn bench_trust_texture(c: &mut Criterion) {
    let mut group = c.benchmark_group("trust_texture_batch");
    for &size in &[8usize, 64, 1024, 16384] {
        let (q, _) = make_inputs(size);
        let mut out = vec![lance_graph_contract::mul::TrustTexture::Calibrated; size];

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |b, _| {
            b.iter(|| {
                batch::trust_texture_batch(&q, &mut out);
                criterion::black_box(&out);
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                batch::scalar_impl::trust_texture_batch(&q, &mut out);
                criterion::black_box(&out);
            });
        });
    }
    group.finish();
}

fn bench_flow_state(c: &mut Criterion) {
    let mut group = c.benchmark_group("flow_state_batch");
    for &size in &[8usize, 64, 1024, 16384] {
        let (q, m) = make_inputs(size);
        let mut out = vec![lance_graph_contract::mul::FlowState::Boredom; size];

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |b, _| {
            b.iter(|| {
                batch::flow_state_batch(&q, &m, &mut out);
                criterion::black_box(&out);
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                batch::scalar_impl::flow_state_batch(&q, &m, &mut out);
                criterion::black_box(&out);
            });
        });
    }
    group.finish();
}

fn bench_gate_decision_disc(c: &mut Criterion) {
    let mut group = c.benchmark_group("gate_decision_disc_batch");
    for &size in &[8usize, 64, 1024, 16384] {
        let (q, m) = make_inputs(size);
        let mut out = vec![0u8; size];

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |b, _| {
            b.iter(|| {
                batch::gate_decision_disc_batch(&q, &m, &mut out);
                criterion::black_box(&out);
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                batch::scalar_impl::gate_decision_disc_batch(&q, &m, &mut out);
                criterion::black_box(&out);
            });
        });
    }
    group.finish();
}

fn bench_mul_assess(c: &mut Criterion) {
    let mut group = c.benchmark_group("mul_assess_batch");
    for &size in &[8usize, 64, 1024, 16384] {
        let (q, m) = make_inputs(size);
        let dummy = || lance_graph_contract::mul::i4_eval::mul_assess_i4(&QualiaI4_16D::ZERO, 0);
        let mut out: Vec<_> = (0..size).map(|_| dummy()).collect();

        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("dispatch", size), &size, |b, _| {
            b.iter(|| {
                batch::mul_assess_batch(&q, &m, &mut out);
                criterion::black_box(&out);
            });
        });

        group.bench_with_input(BenchmarkId::new("scalar", size), &size, |b, _| {
            b.iter(|| {
                batch::scalar_impl::mul_assess_batch(&q, &m, &mut out);
                criterion::black_box(&out);
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_dk_position,
    bench_trust_texture,
    bench_flow_state,
    bench_gate_decision_disc,
    bench_mul_assess,
);
criterion_main!(benches);
