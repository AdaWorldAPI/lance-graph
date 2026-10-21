//! D-RU-1 (U1) lab probe: a batch of JC statistical requests answered from
//! shared power-sum passes. Plan: `.claude/plans/d-relalg-unified-v1.md` §3.
//!
//! STATUS: PROBE (results only). Nothing here is wired into JC, Quack or a
//! production path.
//!
//! A request names a row selection (a [`relalg::Rel`]) and a statistic over
//! `I32` lanes. Every statistic here is a projection of grouped sufficient
//! statistics, the sums mask-risc folds exactly:
//!
//! - mean, variance and one-way ANOVA read `(n, Σx, Σx²)`;
//! - covariance, Pearson, the simple regression line and its R² read
//!   `(n, Σx, Σy, Σx², Σy², Σxy)`.
//!
//! Two ways to answer a batch:
//!
//! - [`answer_separately`]: every request on its own. It evaluates its
//!   selection into a mask, then runs one power-sum pass over it.
//! - [`answer_batched`]: requests are reduced to the distinct selections and,
//!   per selection, to the lane pairs they need. Each selection is evaluated
//!   once, and each needed pair is folded once, keyed by the batch's group
//!   lane. One cross pass over `(x, y)` keyed by group answers every request
//!   that needs `x`, `y` or the pair, grouped or not: an ungrouped statistic
//!   merges the groups exactly ([`CrossPowerSums::checked_merge`]), and a
//!   univariate one reads the `x` side of the cross sums. Lanes no pair
//!   covers get one univariate pass.
//!
//! **Exactness.** The sums are integers (`u64`, `i64`, `u128`, `i128`) and
//! the executor refuses a plane past `2^32` rows instead of wrapping. Merging
//! groups is checked addition. Both arms therefore hand JC bit-identical
//! inputs, and every projection is the same `f64` arithmetic on the same
//! integers: the batched answers equal the separate ones exactly, not within
//! a tolerance.
//!
//! **Versions.** A batch reads one snapshot. Every request carries the version
//! it was built for, and a batch over a table at another version, or mixing
//! versions, is refused ([`BatchError::VersionMismatch`]).
//!
//! **Cost accounting.** Every mask-risc program the probe executes spans all
//! `n_rows` rows, so [`Cost`] counts executed programs, the values each one
//! reads (from the lanes the program names), and the mask bytes written. The
//! counts come from the code that runs the programs, not from a model.

use jc::stats::{
    anova_from_power_sums, pearson_from_cross_power_sums, r_squared_from_cross_power_sums,
    sample_covariance_from_cross_power_sums, simple_regression_from_cross_power_sums, Anova,
    SimpleRegression,
};
use lance_graph_mask_risc::{
    execute_into, scratch_words_for, words_for, ExecError, Foreign, GroupKey, LaneRef, Operand,
    Out, Planes, Program, Scratch, Terminal, Value,
};
use lance_graph_relalg::{normalize, run, Atom, Measure, Query, Rel, RelError};
use ndarray::simd::{CrossPowerSums, PowerSums};

/// One statistic over `I32` lanes of the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stat {
    /// `Σx / n`.
    Mean(u16),
    /// The unbiased sample variance.
    Variance(u16),
    /// One-way ANOVA of the lane across the table's groups.
    Anova(u16),
    /// The unbiased sample covariance.
    Covariance(u16, u16),
    /// Pearson's r.
    Pearson(u16, u16),
    /// The simple least-squares line `y = α + β·x` (first lane is `x`).
    Regression(u16, u16),
    /// `R²` of that line.
    RSquared(u16, u16),
}

impl Stat {
    /// The lanes the statistic reads, `(x, Some(y))` for a pair.
    fn lanes(self) -> (u16, Option<u16>) {
        match self {
            Stat::Mean(x) | Stat::Variance(x) | Stat::Anova(x) => (x, None),
            Stat::Covariance(x, y)
            | Stat::Pearson(x, y)
            | Stat::Regression(x, y)
            | Stat::RSquared(x, y) => (x, Some(y)),
        }
    }

    /// Whether the statistic is per group (needs the group lane).
    fn grouped(self) -> bool {
        matches!(self, Stat::Anova(_))
    }
}

/// One request: a statistic over a row selection, built for one version.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Request {
    /// The table version the request was built against.
    pub version: u64,
    /// The rows the statistic is taken over.
    pub select: Rel,
    /// The statistic.
    pub stat: Stat,
}

/// An answer. `None` inside means the statistic is undefined on the
/// selection (JC's degeneracy rules: too few rows, a constant lane, …).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Answer {
    /// A scalar statistic.
    Scalar(Option<f64>),
    /// A regression line.
    Line(Option<SimpleRegression>),
    /// An ANOVA result.
    Anova(Option<Anova>),
}

/// The table a batch runs over: resident lanes owned by the caller.
#[derive(Debug, Clone, Copy)]
pub struct Table<'a> {
    /// The version the lanes are at.
    pub version: u64,
    /// Row count every lane spans.
    pub n_rows: usize,
    /// The lanes. Statistic lanes are `I32`; `group` and `zero` are `U32`.
    pub lanes: &'a [LaneRef<'a>],
    /// The `U32` group lane grouped statistics are taken over.
    pub group: u16,
    /// Group universe: every value of `group` is below it.
    pub groups: usize,
    /// A `U32` lane of zeros, the key of an ungrouped pass when a request is
    /// answered on its own. Caller-owned, as the other lanes are.
    pub zero: u16,
}

/// Why a batch was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BatchError {
    /// A request was built for another version than the table holds.
    VersionMismatch {
        request: usize,
        request_version: u64,
        table_version: u64,
    },
    /// The selection was decided from its description alone (all rows or
    /// none). The probe runs only selections that need a pass.
    SymbolicSelection { request: usize },
    /// relalg refused the selection.
    Relalg(RelError),
    /// mask-risc refused a pass.
    Exec(ExecError),
    /// Merging group sums overflowed (cannot happen below `2^32` rows).
    MergeOverflow,
}

impl From<RelError> for BatchError {
    fn from(e: RelError) -> Self {
        BatchError::Relalg(e)
    }
}

impl From<ExecError> for BatchError {
    fn from(e: ExecError) -> Self {
        BatchError::Exec(e)
    }
}

/// What answering a batch cost, counted where the programs run.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Cost {
    /// mask-risc programs executed (selection passes plus power-sum passes).
    pub programs: usize,
    /// Of those, selection passes (relalg evaluating a selection to a mask).
    pub selection_passes: usize,
    /// Of those, power-sum passes.
    pub sum_passes: usize,
    /// Rows traversed: every program spans the whole table.
    pub rows_traversed: u64,
    /// Lane values read: per program, `n_rows` per lane it names.
    pub values_read: u64,
    /// Mask bytes materialized (one selection mask per selection pass).
    pub mask_bytes: u64,
}

impl Cost {
    fn selection(&mut self, n: usize, lanes: usize) {
        self.programs += 1;
        self.selection_passes += 1;
        self.rows_traversed += n as u64;
        self.values_read += (n * lanes) as u64;
        self.mask_bytes += (words_for(n) * 8) as u64;
    }

    fn sums(&mut self, n: usize, lanes: usize) {
        self.programs += 1;
        self.sum_passes += 1;
        self.rows_traversed += n as u64;
        self.values_read += (n * lanes) as u64;
    }
}

/// The distinct lanes a normalized selection reads.
fn selection_lanes(rel: &Rel, out: &mut Vec<u16>) {
    match rel {
        Rel::All | Rel::Empty | Rel::Plane(_) => {}
        Rel::Atom(Atom::I32 { lane, .. } | Atom::U32 { lane, .. }) => {
            if !out.contains(lane) {
                out.push(*lane);
            }
        }
        Rel::And(xs) | Rel::Or(xs) => xs.iter().for_each(|x| selection_lanes(x, out)),
        Rel::Not(x) => selection_lanes(x, out),
    }
}

fn check_versions(t: &Table<'_>, reqs: &[Request]) -> Result<(), BatchError> {
    for (i, r) in reqs.iter().enumerate() {
        if r.version != t.version {
            return Err(BatchError::VersionMismatch {
                request: i,
                request_version: r.version,
                table_version: t.version,
            });
        }
    }
    Ok(())
}

/// Evaluates `select` into a mask (one relalg program).
fn select_mask(
    t: &Table<'_>,
    select: &Rel,
    request: usize,
    cost: &mut Cost,
) -> Result<Vec<u64>, BatchError> {
    let planes = Planes {
        n_rows: t.n_rows,
        masks: &[],
        lanes: t.lanes,
    };
    let mut keep = vec![0u64; words_for(t.n_rows)];
    let r = run(
        &Query {
            rel: select.clone(),
            measure: Measure::Keep,
        },
        &planes,
        Some(&mut keep),
    )?;
    if r.programs == 0 {
        return Err(BatchError::SymbolicSelection { request });
    }
    let mut lanes = Vec::new();
    selection_lanes(&normalize(select), &mut lanes);
    cost.selection(t.n_rows, lanes.len());
    Ok(keep)
}

fn exec(t: &Table<'_>, mask: &[u64], terminal: Terminal, out: Out<'_>) -> Result<(), BatchError> {
    let masks = [mask];
    let planes = Planes {
        n_rows: t.n_rows,
        masks: &masks,
        lanes: t.lanes,
    };
    let program = Program::new(Vec::new(), terminal);
    let words = words_for(t.n_rows).max(1);
    let slots = program.scratch_slots as usize;
    let mut buf = vec![0u64; scratch_words_for(words, slots).expect("scratch size")];
    let mut scratch = Scratch::over(&mut buf, words, slots).expect("scratch carves");
    let v = execute_into(&program, &planes, &Foreign::NONE, &mut scratch, out)?;
    debug_assert!(matches!(
        v,
        Value::GroupPowerSums | Value::GroupCrossPowerSums
    ));
    Ok(())
}

/// Grouped `(n, Σx, Σx²)` of lane `x` under `mask`, keyed by `key`.
fn power_pass(
    t: &Table<'_>,
    mask: &[u64],
    key: u16,
    groups: usize,
    x: u16,
    cost: &mut Cost,
) -> Result<Vec<PowerSums>, BatchError> {
    let mut out = vec![PowerSums::default(); groups];
    exec(
        t,
        mask,
        Terminal::GroupPowerSumsI32 {
            mask: Operand::Plane(0),
            key: GroupKey::Lane(key),
            val: x,
        },
        Out::PowerSums(&mut out),
    )?;
    cost.sums(t.n_rows, 2);
    Ok(out)
}

/// Grouped cross sums of `(x, y)` under `mask`, keyed by `key`.
fn cross_pass(
    t: &Table<'_>,
    mask: &[u64],
    key: u16,
    groups: usize,
    x: u16,
    y: u16,
    cost: &mut Cost,
) -> Result<Vec<CrossPowerSums>, BatchError> {
    let mut out = vec![CrossPowerSums::default(); groups];
    exec(
        t,
        mask,
        Terminal::GroupCrossPowerSumsI32 {
            mask: Operand::Plane(0),
            key: GroupKey::Lane(key),
            x,
            y,
        },
        Out::CrossPowerSums(&mut out),
    )?;
    cost.sums(t.n_rows, 3);
    Ok(out)
}

/// The `x` side of cross sums, as univariate sums.
fn x_side(c: &CrossPowerSums) -> PowerSums {
    PowerSums {
        n: c.n,
        sum: c.sum_x,
        sum_sq: c.sum_x_sq,
    }
}

/// The `y` side of cross sums, as univariate sums.
fn y_side(c: &CrossPowerSums) -> PowerSums {
    PowerSums {
        n: c.n,
        sum: c.sum_y,
        sum_sq: c.sum_y_sq,
    }
}

/// The cross sums with `x` and `y` swapped.
fn swapped(c: &CrossPowerSums) -> CrossPowerSums {
    CrossPowerSums {
        n: c.n,
        sum_x: c.sum_y,
        sum_y: c.sum_x,
        sum_x_sq: c.sum_y_sq,
        sum_y_sq: c.sum_x_sq,
        sum_xy: c.sum_xy,
    }
}

fn merge_power(g: &[PowerSums]) -> Result<PowerSums, BatchError> {
    g.iter()
        .try_fold(PowerSums::default(), |a, b| a.checked_merge(*b))
        .ok_or(BatchError::MergeOverflow)
}

fn merge_cross(g: &[CrossPowerSums]) -> Result<CrossPowerSums, BatchError> {
    g.iter()
        .try_fold(CrossPowerSums::default(), |a, b| a.checked_merge(*b))
        .ok_or(BatchError::MergeOverflow)
}

/// `Σx / n`, `None` for an empty selection.
pub fn mean(m: &PowerSums) -> Option<f64> {
    (m.n > 0).then(|| m.sum as f64 / m.n as f64)
}

/// The unbiased sample variance, formed exactly: `(nΣx² − (Σx)²)` in `i128`
/// before the one division. `None` for `n < 2`.
///
/// Candidate for `jc::stats` (it has no univariate projection yet); kept here
/// so the probe does not change JC.
pub fn variance(m: &PowerSums) -> Option<f64> {
    if m.n < 2 {
        return None;
    }
    let n = i128::from(m.n);
    let sum = i128::from(m.sum);
    let sum_sq = i128::try_from(m.sum_sq).ok()?;
    let centered = n.checked_mul(sum_sq)?.checked_sub(sum.checked_mul(sum)?)?;
    Some(centered as f64 / (n * (n - 1)) as f64)
}

fn univariate_answer(stat: Stat, total: &PowerSums, groups: &[PowerSums]) -> Answer {
    match stat {
        Stat::Mean(_) => Answer::Scalar(mean(total)),
        Stat::Variance(_) => Answer::Scalar(variance(total)),
        Stat::Anova(_) => Answer::Anova(anova_from_power_sums(groups)),
        _ => unreachable!("univariate"),
    }
}

fn pair_answer(stat: Stat, m: &CrossPowerSums) -> Answer {
    match stat {
        Stat::Covariance(..) => Answer::Scalar(sample_covariance_from_cross_power_sums(m)),
        Stat::Pearson(..) => Answer::Scalar(pearson_from_cross_power_sums(m)),
        Stat::Regression(..) => Answer::Line(simple_regression_from_cross_power_sums(m)),
        Stat::RSquared(..) => Answer::Scalar(r_squared_from_cross_power_sums(m)),
        _ => unreachable!("pair"),
    }
}

/// Every request on its own: its selection, then one pass for its
/// statistic (keyed by the group lane only when the statistic is grouped).
pub fn answer_separately(
    t: &Table<'_>,
    reqs: &[Request],
) -> Result<(Vec<Answer>, Cost), BatchError> {
    check_versions(t, reqs)?;
    let mut cost = Cost::default();
    let mut out = Vec::with_capacity(reqs.len());
    for (i, r) in reqs.iter().enumerate() {
        let mask = select_mask(t, &r.select, i, &mut cost)?;
        let (key, groups) = if r.stat.grouped() {
            (t.group, t.groups)
        } else {
            (t.zero, 1)
        };
        out.push(match r.stat.lanes() {
            (x, None) => {
                let g = power_pass(t, &mask, key, groups, x, &mut cost)?;
                univariate_answer(r.stat, &merge_power(&g)?, &g)
            }
            (x, Some(y)) => {
                let g = cross_pass(t, &mask, key, groups, x, y, &mut cost)?;
                pair_answer(r.stat, &merge_cross(&g)?)
            }
        });
    }
    Ok((out, cost))
}

/// What a batch was reduced to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reduction {
    /// Distinct selections.
    pub selections: usize,
    /// Cross passes (one per distinct unordered pair per selection).
    pub cross_passes: usize,
    /// Univariate passes (lanes no pair covers).
    pub power_passes: usize,
}

/// The batch reduced to distinct selections and, per selection, to one pass
/// per distinct unordered lane pair plus one per lane no pair covers. Every
/// pass is keyed by the group lane, so grouped and ungrouped requests share
/// it.
pub fn answer_batched(
    t: &Table<'_>,
    reqs: &[Request],
) -> Result<(Vec<Answer>, Cost, Reduction), BatchError> {
    check_versions(t, reqs)?;
    let mut cost = Cost::default();
    let mut red = Reduction::default();

    // Distinct selections, compared after normalization.
    let mut sels: Vec<(Rel, usize)> = Vec::new();
    let sel_of: Vec<usize> = reqs
        .iter()
        .enumerate()
        .map(|(i, r)| {
            let n = normalize(&r.select);
            match sels.iter().position(|(s, _)| *s == n) {
                Some(p) => p,
                None => {
                    sels.push((n, i));
                    sels.len() - 1
                }
            }
        })
        .collect();
    red.selections = sels.len();

    let mut out: Vec<Option<Answer>> = vec![None; reqs.len()];
    for (s, (_, first)) in sels.iter().enumerate() {
        let members: Vec<usize> = (0..reqs.len()).filter(|&i| sel_of[i] == s).collect();
        let mask = select_mask(t, &reqs[*first].select, *first, &mut cost)?;

        // Distinct unordered pairs, then the lanes no pair covers.
        let mut pairs: Vec<(u16, u16)> = Vec::new();
        for &i in &members {
            if let (x, Some(y)) = reqs[i].stat.lanes() {
                let p = (x.min(y), x.max(y));
                if !pairs.contains(&p) {
                    pairs.push(p);
                }
            }
        }
        let mut singles: Vec<u16> = Vec::new();
        for &i in &members {
            if let (x, None) = reqs[i].stat.lanes() {
                if !pairs.iter().any(|&(a, b)| a == x || b == x) && !singles.contains(&x) {
                    singles.push(x);
                }
            }
        }

        let mut cross: Vec<((u16, u16), Vec<CrossPowerSums>)> = Vec::with_capacity(pairs.len());
        for &(a, b) in &pairs {
            cross.push((
                (a, b),
                cross_pass(t, &mask, t.group, t.groups, a, b, &mut cost)?,
            ));
        }
        red.cross_passes += pairs.len();
        let mut power: Vec<(u16, Vec<PowerSums>)> = Vec::with_capacity(singles.len());
        for &x in &singles {
            power.push((x, power_pass(t, &mask, t.group, t.groups, x, &mut cost)?));
        }
        red.power_passes += singles.len();

        // Per-group univariate sums of `x`, from a pair pass or its own.
        let groups_of = |x: u16| -> Vec<PowerSums> {
            for ((a, b), g) in &cross {
                if *a == x {
                    return g.iter().map(x_side).collect();
                }
                if *b == x {
                    return g.iter().map(y_side).collect();
                }
            }
            power
                .iter()
                .find(|(l, _)| *l == x)
                .map(|(_, g)| g.clone())
                .expect("every lane is covered")
        };

        for &i in &members {
            let stat = reqs[i].stat;
            out[i] = Some(match stat.lanes() {
                (x, None) => {
                    let g = groups_of(x);
                    univariate_answer(stat, &merge_power(&g)?, &g)
                }
                (x, Some(y)) => {
                    let (_, g) = cross
                        .iter()
                        .find(|(p, _)| *p == (x.min(y), x.max(y)))
                        .expect("every pair is covered");
                    let m = merge_cross(g)?;
                    pair_answer(stat, &if x <= y { m } else { swapped(&m) })
                }
            });
        }
    }
    Ok((
        out.into_iter()
            .map(|a| a.expect("every request answered"))
            .collect(),
        cost,
        red,
    ))
}
