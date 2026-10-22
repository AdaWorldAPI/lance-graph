//! Independent, scalar reference for grid-supply versus telecom-path failures.
//!
//! Intended for offline Bangladesh-outage fixtures and later compiler differential
//! tests, not an estimate of real Bangladesh grid/tower failure probabilities.
//! A `CascadeResult` is a grid stress/line-trip witness, **not** an observation
//! that a tower lost grid supply or service.

/// Three-valued physical observation: `Unknown` means *not established*, not down.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Observed {
    Up,
    Down,
    Unknown,
}

impl Observed {
    /// Strong-Kleene conjunction (conservatively evaluate missing observations).
    pub const fn and(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Down, _) | (_, Self::Down) => Self::Down,
            (Self::Up, Self::Up) => Self::Up,
            _ => Self::Unknown,
        }
    }

    /// Strong-Kleene disjunction.
    pub const fn or(self, rhs: Self) -> Self {
        match (self, rhs) {
            (Self::Up, _) | (_, Self::Up) => Self::Up,
            (Self::Down, Self::Down) => Self::Down,
            _ => Self::Unknown,
        }
    }

    pub const fn not(self) -> Self {
        match self {
            Self::Up => Self::Down,
            Self::Down => Self::Up,
            Self::Unknown => Self::Unknown,
        }
    }
}

/// All input signals refer to the **same tower**, time window and pinned source
/// generation. `telecom_path` covers radio equipment and upstream backhaul;
/// `backup_supply` covers battery/generator/solar availability. Do not substitute
/// the Bangladesh A10H synthetic `grid_status` for independent `grid_supply`:
/// that script derives its value from the tower outage label itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiteFrame {
    pub grid_supply: Observed,
    pub backup_supply: Observed,
    pub telecom_path: Observed,
    /// Independently observed end-user tower service, when available.
    pub observed_tower_service: Observed,
}

/// Descriptive mechanism attribution based ONLY on fully resolved *inputs*.
/// Not a statistically inferred cause and not a certification of grid physics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mechanism {
    Neither,
    GridPower,
    TelecomPath,
    Both,
    Unresolved,
}

/// Independent end-to-end service observation versus the model prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleCheck {
    Agrees,
    Contradiction,
    InsufficientEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiteAssessment {
    /// Grid-source failure, irrespective of backup success or telecom status.
    pub grid_supply_outage: Observed,
    /// Failure of usable site power (grid OR backup).
    pub power_unavailable: Observed,
    /// Independent non-power telecom-path failure.
    pub telecom_path_failure: Observed,
    pub predicted_tower_service: Observed,
    pub mechanism: Mechanism,
    pub check: OracleCheck,
}

/// Mathematical reference: `service = (grid_supply OR backup_supply)
/// AND telecom_path`. Keep this scalar implementation independent of any
/// egglog rewrite selection or R2IL/Quack/mask-risc lowering.
///
/// The model is intentionally minimal and only applies when the observed
/// signals have the defined meaning. An upstream shared backhaul power outage
/// must be expressed as a telecom-path failure or modeled separately, never
/// silently inferred from a grid line-trip or a tower's offline label.
pub fn assess_site(frame: SiteFrame) -> SiteAssessment {
    let power = frame.grid_supply.or(frame.backup_supply);
    let predicted = power.and(frame.telecom_path);
    let power_failure = power.not();
    let telecom_failure = frame.telecom_path.not();
    let mechanism = match (power_failure, telecom_failure) {
        (Observed::Down, Observed::Down) => Mechanism::Neither,
        (Observed::Up, Observed::Down) => Mechanism::GridPower,
        (Observed::Down, Observed::Up) => Mechanism::TelecomPath,
        (Observed::Up, Observed::Up) => Mechanism::Both,
        _ => Mechanism::Unresolved,
    };
    let check = match (predicted, frame.observed_tower_service) {
        (Observed::Unknown, _) | (_, Observed::Unknown) => OracleCheck::InsufficientEvidence,
        (lhs, rhs) if lhs == rhs => OracleCheck::Agrees,
        _ => OracleCheck::Contradiction,
    };
    SiteAssessment {
        grid_supply_outage: frame.grid_supply.not(),
        power_unavailable: power_failure,
        telecom_path_failure: telecom_failure,
        predicted_tower_service: predicted,
        mechanism,
        check,
    }
}

/// A stable, already-resolved tower identity and a common time window. It is
/// NOT a local ordinal from two unrelated tables; resolving foreign identities
/// belongs to an authorized, version-pinned binding stage before compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SiteFocus {
    pub tower_guid: u128,
    pub window_start: i64,
    pub window_end: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundObservation {
    pub focus: SiteFocus,
    pub source_generation: u64,
    pub value: Observed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BoundSiteFrame {
    pub grid_supply: BoundObservation,
    pub backup_supply: BoundObservation,
    pub telecom_path: BoundObservation,
    pub observed_tower_service: BoundObservation,
}

/// Independent generation pins: different source streams need not have the
/// same numeric generation, but each must match its own attested snapshot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SourcePins {
    pub grid: u64,
    pub backup: u64,
    pub telecom: u64,
    pub service: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    InvalidWindow,
    CrossFocus,
    StaleSource,
}

/// A cold compiler may only optimize after this check; arithmetic on local
/// positions cannot ever substitute for matching tower identities and time.
/// The returned oracle assessment is *not* a commit acknowledgement.
pub fn assess_bound(frame: BoundSiteFrame, pins: SourcePins) -> Result<SiteAssessment, Refusal> {
    let focus = frame.grid_supply.focus;
    if focus.window_start >= focus.window_end {
        return Err(Refusal::InvalidWindow);
    }
    if frame.backup_supply.focus != focus
        || frame.telecom_path.focus != focus
        || frame.observed_tower_service.focus != focus
    {
        return Err(Refusal::CrossFocus);
    }
    if frame.grid_supply.source_generation != pins.grid
        || frame.backup_supply.source_generation != pins.backup
        || frame.telecom_path.source_generation != pins.telecom
        || frame.observed_tower_service.source_generation != pins.service
    {
        return Err(Refusal::StaleSource);
    }
    Ok(assess_site(SiteFrame {
        grid_supply: frame.grid_supply.value,
        backup_supply: frame.backup_supply.value,
        telecom_path: frame.telecom_path.value,
        observed_tower_service: frame.observed_tower_service.value,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    const fn f(grid: Observed, backup: Observed, telecom: Observed, service: Observed) -> SiteFrame {
        SiteFrame {
            grid_supply: grid,
            backup_supply: backup,
            telecom_path: telecom,
            observed_tower_service: service,
        }
    }

    #[test]
    fn grid_trips_but_backup_keeps_telecom_up() {
        let a = assess_site(f(Observed::Down, Observed::Up, Observed::Up, Observed::Up));
        assert_eq!(a.grid_supply_outage, Observed::Up);
        assert_eq!(a.power_unavailable, Observed::Down);
        assert_eq!(a.predicted_tower_service, Observed::Up);
        assert_eq!(a.mechanism, Mechanism::Neither);
        assert_eq!(a.check, OracleCheck::Agrees);
    }

    #[test]
    fn grid_failure_causes_tower_service_loss_without_backup() {
        let a = assess_site(f(Observed::Down, Observed::Down, Observed::Up, Observed::Down));
        assert_eq!(a.mechanism, Mechanism::GridPower);
        assert_eq!(a.check, OracleCheck::Agrees);
    }

    #[test]
    fn healthy_grid_with_telecom_path_failure_is_not_grid_failure() {
        let a = assess_site(f(Observed::Up, Observed::Unknown, Observed::Down, Observed::Down));
        assert_eq!(a.grid_supply_outage, Observed::Down);
        assert_eq!(a.mechanism, Mechanism::TelecomPath);
        assert_eq!(a.check, OracleCheck::Agrees);
    }

    #[test]
    fn concurrent_grid_and_telecom_faults_keep_both_causes() {
        let a = assess_site(f(Observed::Down, Observed::Down, Observed::Down, Observed::Down));
        assert_eq!(a.mechanism, Mechanism::Both);
    }

    #[test]
    fn unknown_backup_is_not_false_and_cannot_prove_grid_caused_service_loss() {
        let a = assess_site(f(Observed::Down, Observed::Unknown, Observed::Up, Observed::Down));
        assert_eq!(a.predicted_tower_service, Observed::Unknown);
        assert_eq!(a.mechanism, Mechanism::Unresolved);
        assert_eq!(a.check, OracleCheck::InsufficientEvidence);
    }

    #[test]
    fn observed_outage_contradicting_healthy_inputs_is_refused() {
        let a = assess_site(f(Observed::Up, Observed::Up, Observed::Up, Observed::Down));
        assert_eq!(a.check, OracleCheck::Contradiction);
    }

    #[test]
    fn missing_telecom_status_is_not_a_proven_hardware_fault() {
        let a = assess_site(f(Observed::Up, Observed::Down, Observed::Unknown, Observed::Down));
        assert_eq!(a.mechanism, Mechanism::Unresolved);
        assert_eq!(a.check, OracleCheck::InsufficientEvidence);
    }

    #[test]
    fn total_truth_table_kleene_and_commutes_with_pure_operand_order() {
        let vals = [Observed::Up, Observed::Down, Observed::Unknown];
        for &a in &vals {
            assert_eq!(a.and(a), a);
            assert_eq!(a.or(a), a);
            assert_eq!(a.not().not(), a);
            for &b in &vals {
                assert_eq!(a.and(b), b.and(a));
                assert_eq!(a.or(b), b.or(a));
                assert_eq!(a.and(b), a.not().or(b.not()).not());
                for &c in &vals {
                    assert_eq!(a.and(b).and(c), a.and(b.and(c)));
                    assert_eq!(a.or(b).or(c), a.or(b.or(c)));
                    assert_eq!(a.and(b.or(c)), a.and(b).or(a.and(c)));
                }
            }
        }
    }

    #[test]
    fn stale_source_refuses_even_when_local_site_and_values_agree() {
        let focus = SiteFocus { tower_guid: 7001, window_start: 0, window_end: 60 };
        let a = BoundObservation { focus, source_generation: 17, value: Observed::Up };
        let b = BoundObservation { source_generation: 18, ..a };
        let f = BoundSiteFrame {
            grid_supply: a,
            backup_supply: b,
            telecom_path: a,
            observed_tower_service: a,
        };
        let pins = SourcePins { grid: 17, backup: 18, telecom: 17, service: 17 };
        assert_eq!(assess_bound(f, pins).unwrap().mechanism, Mechanism::Neither);
        assert_eq!(assess_bound(f, SourcePins { backup: 19, ..pins }), Err(Refusal::StaleSource));
    }

    #[test]
    fn same_row_ordinal_different_tower_guid_refuses_join() {
        let focus = SiteFocus { tower_guid: 7001, window_start: 0, window_end: 60 };
        let site_2 = SiteFocus { tower_guid: 7002, ..focus };
        let a = BoundObservation { focus, source_generation: 1, value: Observed::Up };
        let f = BoundSiteFrame {
            grid_supply: a,
            backup_supply: BoundObservation { focus: site_2, ..a },
            telecom_path: a,
            observed_tower_service: a,
        };
        let pins = SourcePins { grid: 1, backup: 1, telecom: 1, service: 1 };
        assert_eq!(assess_bound(f, pins), Err(Refusal::CrossFocus));
        assert_eq!(assess_bound(BoundSiteFrame {
            backup_supply: a,
            grid_supply: BoundObservation { focus: SiteFocus { window_end: 0, ..focus }, ..a },
            ..f
        }, pins), Err(Refusal::InvalidWindow));
    }

    #[test]
    fn known_grid_status_never_equates_to_tower_availability() {
        // The two are different observations. Generator is entirely independent
        // of a tower's on/off status (the historical Bangladesh bug).
        let generator_protected = assess_site(f(Observed::Down, Observed::Up, Observed::Up, Observed::Up));
        let independently_failed = assess_site(f(Observed::Up, Observed::Up, Observed::Down, Observed::Down));
        assert_ne!(generator_protected.grid_supply_outage, independently_failed.grid_supply_outage);
        assert_ne!(generator_protected.predicted_tower_service, independently_failed.predicted_tower_service);
    }
}