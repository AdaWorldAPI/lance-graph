//! Generic consumer hot-plug — the plug-and-play pattern EVERY consumer
//! migrates to (operator, 2026-07-07).
//!
//! Three roles, three homes:
//!
//! 1. **This module (the SOCKET, zero-dep):** the shapes a consumer uses to
//!    declare WHICH classids it hot-plugs, and the [`CapabilityAuthority`]
//!    trait the authority implements. No OGAR dep — the contract is a
//!    workspace member and MUST stay dependency-free (a path dep here breaks
//!    every CI cargo invocation at workspace-load time; learned 2026-07-07).
//! 2. **OGAR (the AUTHORITY):** resolves the hot-plugged classids to BOTH the
//!    vocab rows and the action definitions, and verifies the registration
//!    (expected consumer, coverage both directions, ids minted exactly once).
//! 3. **The consumer:** declares one [`HotPlug`] const naming its classids +
//!    covered capabilities, calls `activate` in its own binary/tests — drift
//!    bangs once, no pins, no serialization, no per-consumer plug crate.
//!
//! The classid is the join key on BOTH sides: the consumer says "0x0805,
//! 0x0808, 0x0809 are hot", the authority hands back the concepts and every
//! action whose subject is one of those ids.

/// A consumer's hot-plug declaration: which classids it activates and which
/// capability names its executor covers. One `const` per consumer — the
/// whole registration surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HotPlug {
    /// Consumer name (crate name by convention) — the authority checks it
    /// against the expected-executor list of the tables it resolves.
    pub consumer: &'static str,
    /// The canon-high concept ids the consumer hot-plugs.
    pub classids: &'static [u16],
    /// Capability names the consumer's executor covers.
    pub covered: &'static [&'static str],
}

/// What the authority returns for a green activation: the vocab rows and
/// capability names resolved for the hot-plugged classids. Plain owned
/// `std` types — zero-dep, no serialization.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Activation {
    /// `(concept, classid)` vocab rows for every hot-plugged id.
    pub concepts: Vec<(String, u16)>,
    /// Capability names whose subject is one of the hot-plugged ids.
    pub capabilities: Vec<String>,
}

/// Why an activation failed — each arm is one named bang.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ActivationDrift {
    /// A hot-plugged classid is not minted in the authority's codebook.
    UnknownClassid(u16),
    /// The consumer is not an expected executor for a resolved table.
    UnexpectedConsumer(String),
    /// The authority declares a capability on a hot-plugged id that the
    /// consumer does not cover.
    Uncovered(String),
    /// The consumer claims a capability the authority does not declare on
    /// its hot-plugged ids.
    Undeclared(String),
    /// A hot-plugged classid resolves to no declared capability at all —
    /// plugging it is either premature or the table was forgotten.
    NoCapabilitiesFor(u16),
    /// The authority resolved a concept that this crate's zero-dep wire
    /// mirror ([`crate::ogar_codebook`]) does not carry at the same id.
    ///
    /// This is the drift the retired compile-time `COUNT_FUSE` existed to
    /// catch — now reported **per plug**, for the ids a consumer actually
    /// uses, instead of as a global equality assert that failed every build.
    /// The authority is the only place both sides are in scope: a consumer
    /// holding OGAR can see the mirror, and a mirror-only consumer cannot
    /// call [`CapabilityAuthority::activate`] at all.
    MirrorDrift {
        /// Concept name as the authority resolved it.
        concept: String,
        /// The id the authority is authoritative for.
        authority_id: u16,
        /// What the mirror said — `None` when the concept is missing entirely.
        mirror_id: Option<u16>,
    },
}

/// Cross-check an authority's resolved concepts against this crate's zero-dep
/// wire mirror, so a stale mirror is caught at the plug rather than silently
/// mis-resolving in a consumer that reads the mirror instead of the authority.
///
/// Split from the mirror lookup so the checker itself is testable against a
/// deliberately-wrong table — with the real mirror there is (by construction)
/// no concept that disagrees, so a test using it could only ever assert the
/// happy path.
#[must_use]
pub fn mirror_disagreement<F>(
    concepts: &[(String, u16)],
    mirror_lookup: F,
) -> Option<ActivationDrift>
where
    F: Fn(&str) -> Option<u16>,
{
    concepts.iter().find_map(|(concept, authority_id)| {
        let mirror_id = mirror_lookup(concept);
        (mirror_id != Some(*authority_id)).then(|| ActivationDrift::MirrorDrift {
            concept: concept.clone(),
            authority_id: *authority_id,
            mirror_id,
        })
    })
}

/// [`mirror_disagreement`] against the real [`crate::ogar_codebook`] mirror.
#[must_use]
pub fn verify_against_mirror(concepts: &[(String, u16)]) -> Option<ActivationDrift> {
    mirror_disagreement(concepts, crate::ogar_codebook::canonical_concept_id)
}

impl core::fmt::Display for ActivationDrift {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnknownClassid(id) => write!(f, "hot-plugged classid 0x{id:04X} is not minted"),
            Self::UnexpectedConsumer(c) => write!(f, "consumer `{c}` is not an expected executor"),
            Self::Uncovered(cap) => write!(f, "declared capability `{cap}` has no consumer arm"),
            Self::Undeclared(cap) => write!(f, "consumer covers `{cap}` which is not declared"),
            Self::NoCapabilitiesFor(id) => {
                write!(f, "classid 0x{id:04X} resolves to no declared capability")
            }
            Self::MirrorDrift {
                concept,
                authority_id,
                mirror_id,
            } => match mirror_id {
                Some(m) => write!(
                    f,
                    "wire mirror has `{concept}`=0x{m:04X} but the authority says 0x{authority_id:04X}"
                ),
                None => write!(
                    f,
                    "wire mirror is missing `{concept}` (authority: 0x{authority_id:04X})"
                ),
            },
        }
    }
}

impl std::error::Error for ActivationDrift {}

/// Implemented by the authority (OGAR side, same binary): resolve a
/// [`HotPlug`] to its [`Activation`] or the first [`ActivationDrift`].
pub trait CapabilityAuthority {
    /// Verify the plug and hand back the vocab + capability surface for
    /// exactly the hot-plugged classids.
    fn activate(&self, plug: &HotPlug) -> Result<Activation, ActivationDrift>;
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TinyAuthority;
    impl CapabilityAuthority for TinyAuthority {
        fn activate(&self, plug: &HotPlug) -> Result<Activation, ActivationDrift> {
            if plug.classids.contains(&0xDEAD) {
                return Err(ActivationDrift::UnknownClassid(0xDEAD));
            }
            Ok(Activation {
                concepts: plug
                    .classids
                    .iter()
                    .map(|&id| (format!("c{id:04x}"), id))
                    .collect(),
                capabilities: plug.covered.iter().map(|s| (*s).to_string()).collect(),
            })
        }
    }

    #[test]
    fn socket_shape_round_trips_through_a_trait_object() {
        let plug = HotPlug {
            consumer: "demo",
            classids: &[0x0805],
            covered: &["recognize_line"],
        };
        let auth: &dyn CapabilityAuthority = &TinyAuthority;
        let act = auth.activate(&plug).unwrap();
        assert_eq!(act.concepts, vec![("c0805".to_string(), 0x0805)]);
        assert_eq!(act.capabilities, vec!["recognize_line".to_string()]);
        assert!(matches!(
            auth.activate(&HotPlug {
                classids: &[0xDEAD],
                ..plug
            }),
            Err(ActivationDrift::UnknownClassid(0xDEAD))
        ));
    }

    /// The drift class the retired `COUNT_FUSE` guarded: a concept the
    /// AUTHORITY knows that the MIRROR does not carry at the same id.
    ///
    /// Codex caught that hot-plug alone could not see this — `activate`
    /// consults only OGAR, so a stale mirror activated green while
    /// mirror-reading consumers resolved `None`. The checker closes that,
    /// and it is tested against a deliberately-wrong table because the real
    /// mirror is (by construction) never wrong — a test using it could only
    /// assert the happy path and would pass with the checker deleted.
    #[test]
    fn a_concept_missing_from_the_mirror_is_named_drift_not_silence() {
        let resolved = vec![("textline".to_string(), 0x0805u16)];

        // Missing entirely — the add-a-concept-without-mirroring-it case,
        // i.e. exactly what happened to osm_street_node on 2026-08-14.
        assert_eq!(
            mirror_disagreement(&resolved, |_| None),
            Some(ActivationDrift::MirrorDrift {
                concept: "textline".to_string(),
                authority_id: 0x0805,
                mirror_id: None,
            })
        );

        // Present at the WRONG id — the case a length-equality fuse could
        // never have caught at all, since the counts still match.
        assert_eq!(
            mirror_disagreement(&resolved, |_| Some(0x0806)),
            Some(ActivationDrift::MirrorDrift {
                concept: "textline".to_string(),
                authority_id: 0x0805,
                mirror_id: Some(0x0806),
            })
        );

        // Agreement is silent — without this the checker could "pass" by
        // objecting to everything, which carries no information.
        assert_eq!(mirror_disagreement(&resolved, |_| Some(0x0805)), None);
    }
}
