//! Cost of the ClassView × DR IAM behavior against the direct paths.
//!
//! ```text
//! cargo run --release --example classview_dr_cost            # 8,000 users
//! cargo run --release --example classview_dr_cost -- 65000
//! ```
//!
//! Four routes to "the roles user u holds", each checked equal to the direct
//! dir-sim answer before it is timed:
//!
//! * `direct-view`: dir-sim's own SID walk plus a grant map (the shape
//!   `ViewActors::for_principal` reads) — the mandatory negative control;
//! * `direct-csr`: the prepared CSR walk (`Prepared::roles_of`);
//! * `fold`: the compiled closure fold called directly (`Compiled::roles_of`);
//! * `classview-dr`: the same fold reached through the class's declaration
//!   (`compute_dag`) and the routing table, via `execute_compute_dag`.
//!
//! Preparation (`prepare` + `compile`) is reported separately, never folded
//! into a per-user figure.

use dir_rmap_probe::fixture::{Fixture, Spec};
use dir_rmap_probe::prep::{Chain, Compiled, Ordinals, words};
use dir_rmap_probe::{World, a_groups};
use lance_graph_contract::class_view::{
    ClassId, ClassView, ComputeEdge, compute_dag_topo_order, execute_compute_dag,
};
use lance_graph_contract::ontology::{DisplayTemplate, FieldRef};
use ogar_dir_core::Guid128;
use std::collections::HashMap;
use std::hint::black_box;
use std::time::{Duration, Instant};

const USER: ClassId = 1;
static DAG: [ComputeEdge; 1] = [ComputeEdge {
    target: 1,
    inputs: &[0],
}];
type Route = fn(&Compiled, u16) -> u64;
const DR: [Option<Route>; 2] = [None, Some(Compiled::roles_of)];

struct IamUser(Vec<FieldRef>);
impl ClassView for IamUser {
    fn fields(&self, _: ClassId) -> &[FieldRef] {
        &self.0
    }
    fn template(&self, _: ClassId) -> DisplayTemplate {
        DisplayTemplate::Detail
    }
    fn dolce_category_id(&self, _: ClassId) -> u8 {
        0
    }
    fn compute_dag(&self, _: ClassId) -> &[ComputeEdge] {
        &DAG
    }
}

fn via_classview(view: &dyn ClassView, c: &Compiled, u: u16) -> u64 {
    let mut out = (c, u, 0u64);
    execute_compute_dag(view.compute_dag(USER), &mut out, |s, t| {
        let route = DR[t as usize].ok_or(t)?;
        s.2 = route(s.0, s.1);
        Ok::<(), u8>(())
    })
    .expect("routed");
    out.2
}

/// The behavior compiled once per class: the declaration is resolved to the
/// routed executables in dependency order; per row only the routes run.
fn compile_class(view: &dyn ClassView) -> Vec<Route> {
    compute_dag_topo_order(view.compute_dag(USER))
        .expect("acyclic")
        .into_iter()
        .map(|t| DR[t as usize].expect("routed"))
        .collect()
}

fn median(mut v: Vec<Duration>) -> Duration {
    v.sort();
    v[v.len() / 2]
}

fn per_user(n: usize, reps: usize, mut f: impl FnMut()) -> f64 {
    let mut t = Vec::with_capacity(reps);
    for _ in 0..reps {
        let s = Instant::now();
        f();
        t.push(s.elapsed());
    }
    median(t).as_nanos() as f64 / n as f64
}

fn main() {
    let users: usize = std::env::args()
        .nth(1)
        .map_or(8_000, |s| s.parse().expect("user count"));
    let w = World::new(Fixture::generate(Spec::iam(users)));
    let v = w.view();
    let ord = Ordinals::of(&w.f, &v);
    let by_guid: HashMap<Guid128, u64> =
        w.f.grants.iter().fold(HashMap::new(), |mut m, &(g, r)| {
            *m.entry(w.f.groups[g as usize]).or_default() |= 1 << r;
            m
        });
    let direct = |u: &Guid128| {
        a_groups(&v, u, Chain::Security)
            .iter()
            .fold(0u64, |a, g| a | by_guid.get(g).copied().unwrap_or(0))
    };

    let s = Instant::now();
    let b = w.prepare();
    let t_prepare = s.elapsed();
    let s = Instant::now();
    let c = Compiled::compile(w.prepare());
    let t_compile = s.elapsed().saturating_sub(t_prepare);
    let view = IamUser(vec![
        FieldRef::new("urn:iam:user:memberOf", "member_of"),
        FieldRef::new("urn:iam:user:effectiveRoles", "effective_roles"),
    ]);

    let (mut scratch, mut queue) = (vec![0u64; words(v.groups_len())], Vec::new());
    for (i, ug) in w.f.users.iter().enumerate() {
        let want = direct(ug);
        let uo = ord.user[i];
        assert_eq!(b.roles_of(uo, &mut scratch, &mut queue), want, "csr {i}");
        assert_eq!(c.roles_of(uo), want, "fold {i}");
        assert_eq!(via_classview(&view, &c, uo), want, "classview-dr {i}");
    }

    let n = w.f.users.len();
    let reps = 15;
    let t_view = per_user(n, 5, || {
        for ug in &w.f.users {
            black_box(direct(ug));
        }
    });
    let t_csr = per_user(n, reps, || {
        for &uo in &ord.user {
            black_box(b.roles_of(uo, &mut scratch, &mut queue));
        }
    });
    let t_fold = per_user(n, reps, || {
        for &uo in &ord.user {
            black_box(c.roles_of(black_box(uo)));
        }
    });
    let routes = compile_class(&view);
    for &uo in &ord.user {
        assert_eq!(routes[0](&c, uo), c.roles_of(uo), "compiled");
    }
    let t_compiled = per_user(n, reps, || {
        for &uo in &ord.user {
            for r in &routes {
                black_box(r(&c, black_box(uo)));
            }
        }
    });
    let t_dr = per_user(n, reps, || {
        for &uo in &ord.user {
            black_box(via_classview(&view, &c, black_box(uo)));
        }
    });

    println!(
        "users {n}, groups {}, all {n} answers checked equal to direct-view",
        v.groups_len()
    );
    println!("per user (median of runs):");
    println!("  direct-view   {t_view:>10.1} ns   (existing dir-sim path)");
    println!("  direct-csr    {t_csr:>10.1} ns");
    println!("  fold          {t_fold:>10.1} ns");
    println!(
        "  classview-dr  {t_dr:>10.1} ns   (dispatch over fold: {:+.1} ns)",
        t_dr - t_fold
    );
    println!(
        "  classview-dr compiled once per class {t_compiled:>6.1} ns   (over fold: {:+.1} ns)",
        t_compiled - t_fold
    );
    // P0: a millisecond result is a failed gate, never a measurement.
    let prep = t_prepare + t_compile;
    if prep >= Duration::from_millis(1) {
        println!("preparation, once per version: FAILED GATE (millisecond scale)");
    } else {
        println!(
            "preparation, once per version: {:.1} us",
            prep.as_secs_f64() * 1e6
        );
    }
}
