//! Lab spike: Crepe's rule analysis as a cold optimizer in front of relalg.
//!
//! STATUS: PROBE (results only). Not wired into Quack, R2IL or any production
//! path. Plan: `.claude/plans/crepe-rule-optimizer-spike-v1.md`.
//!
//! The pipeline, one stage per module:
//!
//! 1. `crepe_plan!` (AdaWorldAPI/crepe fork) checks the rules exactly as
//!    `crepe!` does (names, arity, strata, negation cycles) and reports crepe's
//!    own left-to-right bound/free analysis as data ([`plan`]). No HashMap
//!    index, HashSet store or fixpoint loop is generated.
//! 2. [`logic`]: labels to ids; the admitted subset (flat, positive,
//!    non-recursive, over `@input` relations) or a named refusal.
//! 3. [`optimize`]: binding propagation, then duplicate-atom and duplicate-rule
//!    elimination where the answer is duplicate-insensitive.
//! 4. [`mork_api`]: each rule as a MORK `query_multi` pattern (D-MPS-9 shape),
//!    admitted or refused by name.
//! 5. [`native`]: the star read out of the pattern, laid out by the D-MPS-8
//!    relationship table, the joint mask computed by relalg (`And` of presence
//!    planes) as one mask-risc program; terminals read the mask.
//!    - [`lance_store`]: the same star over Lance 13 tables (written through
//!      lancedb 0.40), each side read under a relalg-built address mask with
//!      `with_row_addr_prefilter` (D-HOLO-H14d stage 3).
//!    - [`addressed`]: arm B, the addressed semijoin. Each relation is a lane
//!      in key order, attested with `SealedFacetLane`; a side reads only the
//!      ranges of its demanded keys.
//!    - [`cse`]: every rule's star over one shared table, the parts several
//!      rules share evaluated once (D-RU-3 / U3).
//! 6. [`reference`]: an independent nested-loop evaluator over the resolved,
//!    unoptimized rules. The unmodified `crepe!` evaluator, datafrog and MORK
//!    are the other oracles (tests only).

pub mod addressed;
pub mod coaddressed;
pub mod cse;
pub mod lance_store;
pub mod logic;
pub mod mork_api;
pub mod native;
pub mod optimize;
pub mod pipeline;
pub mod plan;
pub mod programs;
pub mod reference;
