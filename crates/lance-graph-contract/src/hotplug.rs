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
//! 2. **OGAR (the AUTHORITY):** resolves the hot-plugged classids to the vocab
//!    rows, the action definitions, AND the storage reading
//!    ([`Activation::read_modes`]), and verifies the registration (expected
//!    consumer, coverage both directions, ids minted exactly once).
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

/// What the authority returns for a green activation: the vocab rows, the
/// capability names, and the STORAGE READING resolved for the hot-plugged
/// classids. Plain owned `std` types — zero-dep, no serialization.
///
/// **No `Default`, deliberately.** `Activation::default()` would be a
/// green-looking activation carrying no reading at all — precisely the shape
/// a consumer then turns into [`ReadMode::DEFAULT`] (V1). An activation is
/// something an authority RESOLVED; there is no meaningful empty one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Activation {
    /// `(concept, classid)` vocab rows for every hot-plugged id.
    pub concepts: Vec<(String, u16)>,
    /// Capability names whose subject is one of the hot-plugged ids.
    pub capabilities: Vec<String>,
    /// `(concept id, ReadMode)` — how a row addressed under each hot-plugged
    /// concept is READ: which tail the key carries, which value tenants
    /// materialise, how the edge block is carved.
    ///
    /// **Why this rides the activation rather than a registry entry**
    /// (D-BLOCKS-HOTPLUG-1, operator, 2026-09-07). The reading is not
    /// recorded in the key — [`crate::canonical_node::NodeGuid`] has no
    /// tail-variant accessor, and `decode()` (V1 `family:u24 ++ identity:u24`)
    /// versus `decode_v2()` (leaf + `u16`/`u16`) is chosen by classid alone.
    /// So SOMETHING must map a classid to its reading, and
    /// [`BUILTIN_READ_MODES`](crate::canonical_node::classid_read_mode)
    /// *"holds only the canon builtins"* — every entry there is a canon
    /// DOMAIN. Registering each consumer's seat there would make adding a
    /// frontend an edit to this crate plus a substrate recompile: the central
    /// lockstep the retired `COUNT_FUSE` belonged to, rebuilt one layer up.
    ///
    /// The authority already resolves the other two infos for exactly the
    /// hot-plugged ids; the reading is the third, from the same call, keyed
    /// the same way. A consumer composes its own full `u32`
    /// (`concept << 16 | its app prefix`) and reads rows with the mode the
    /// authority handed it — never by asking the canon registry about a class
    /// the canon does not own.
    ///
    /// **Owned, and per-plug — corrected 2026-09-07.** This was `&'static`,
    /// argued as "plug-and-play at COMPILE time: a reading is looked up, not
    /// computed". That argument had a consequence I did not weigh: a
    /// `&'static` table cannot be built per plug, so the authority could only
    /// hand back a table it already held ALL of, which forced an
    /// all-or-nothing guard and, behind it, one hard-coded row per consumer
    /// seat. That is the central lockstep `D-BLOCKS-HOTPLUG-1` retired,
    /// rebuilt one level up — and its failure mode is the worst kind: a
    /// consumer missing from the table keeps compiling and silently loses its
    /// V3 tail, discoverable only weeks later and nowhere near the edit.
    ///
    /// Owned lets the authority answer for exactly the ids a plug declared,
    /// derived from the plug rather than looked up in a shared list. Being
    /// plugged in IS the declaration
    /// ([`ReadMode::PLUG_AND_PLAY_V3`](crate::canonical_node::ReadMode::PLUG_AND_PLAY_V3)),
    /// and an authority may still override any single concept.
    ///
    /// **Fails closed by MECHANISM, not by prose** (operator, 2026-09-07:
    /// *"don't silently enforce V1 fallback in hotplug, that's
    /// unacceptable"*). This field is PRIVATE and the only lookup is
    /// [`Activation::read_mode_for`], which returns a `Result` — there is no
    /// `Option` to `.unwrap_or(ReadMode::DEFAULT)` and no public slice to
    /// scan. An earlier cut left it public with a doc comment saying an empty
    /// slice "is not assume-the-default"; a rule a caller can violate with a
    /// one-liner is not a rule.
    ///
    /// An empty table is still legitimate — a capability-only consumer (an
    /// executor that mints no keys) has no reading to declare. What the type
    /// now guarantees is that *asking* for an absent one bangs instead of
    /// quietly yielding a V1 tail.
    read_modes: Vec<(u16, crate::canonical_node::ReadMode)>,
}

impl Activation {
    /// Build an activation. Authority-side constructor — the fields are not
    /// public, so this is how [`CapabilityAuthority`] implementations return
    /// one.
    #[must_use]
    pub fn new(
        concepts: Vec<(String, u16)>,
        capabilities: Vec<String>,
        read_modes: Vec<(u16, crate::canonical_node::ReadMode)>,
    ) -> Self {
        Self {
            concepts,
            capabilities,
            read_modes,
        }
    }

    /// The reading for one hot-plugged concept — the ONLY lookup, and it
    /// fails closed.
    ///
    /// Returns [`ActivationDrift::NoReadingFor`] when the authority declared
    /// no reading for `concept`. It deliberately does NOT return an `Option`:
    /// an `Option` invites `.unwrap_or(ReadMode::DEFAULT)`, and
    /// [`ReadMode::DEFAULT`] is a **V1** tail. A consumer that mints keys
    /// under a hot-plugged seat must get its reading from here or bang; there
    /// is no third outcome.
    ///
    /// [`ReadMode::DEFAULT`]: crate::canonical_node::ReadMode::DEFAULT
    ///
    /// # Errors
    ///
    /// [`ActivationDrift::NoReadingFor`] if `concept` has no declared reading.
    pub fn read_mode_for(
        &self,
        concept: u16,
    ) -> Result<crate::canonical_node::ReadMode, ActivationDrift> {
        self.read_modes
            .iter()
            .find_map(|(c, m)| (*c == concept).then_some(*m))
            .ok_or(ActivationDrift::NoReadingFor(concept))
    }

    /// Every declared `(concept, reading)` pair — the AUDIT surface.
    ///
    /// For tests and authority conformance checks that assert the whole
    /// table. Consumers resolving a single seat use
    /// [`read_mode_for`](Activation::read_mode_for): this returns a plain
    /// slice, so scanning it and defaulting is once again expressible, and
    /// that is exactly what the lookup exists to avoid.
    #[must_use]
    pub fn declared_readings(&self) -> &[(u16, crate::canonical_node::ReadMode)] {
        &self.read_modes
    }
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
    /// [`Activation::read_mode_for`] was asked for a concept the authority
    /// declared no reading for.
    ///
    /// The named bang that replaces a silent [`ReadMode::DEFAULT`] (V1)
    /// fallback. A consumer minting keys under a hot-plugged seat reaches
    /// this instead of quietly producing legacy-tailed rows.
    ///
    /// [`ReadMode::DEFAULT`]: crate::canonical_node::ReadMode::DEFAULT
    NoReadingFor(u16),
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
            Self::NoReadingFor(id) => write!(
                f,
                "no storage reading declared for concept 0x{id:04X} \
                 (a V1 default is never substituted)"
            ),
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
            Ok(Activation::new(
                plug.classids
                    .iter()
                    .map(|&id| (format!("c{id:04x}"), id))
                    .collect(),
                plug.covered.iter().map(|s| (*s).to_string()).collect(),
                // This toy authority declares no reading — an authority that
                // has none says so, and asking it for one bangs.
                Vec::new(),
            ))
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

    /// Asking for an absent reading BANGS — it never yields V1.
    ///
    /// Operator, 2026-09-07: *"don't silently enforce V1 fallback in hotplug,
    /// that's unacceptable."* The shape being removed is
    /// `act.read_modes.iter().find(…).map(…).unwrap_or(ReadMode::DEFAULT)` —
    /// a one-liner that compiles, reads as careful, and mints legacy-tailed
    /// keys forever. It is no longer expressible: the field is private and
    /// the lookup returns a `Result`.
    ///
    /// Two-sided on the same activation, which is what makes it a test rather
    /// than a restatement of the code: the DECLARED concept resolves, and an
    /// undeclared one bangs. A lookup that answered for everything, or for
    /// nothing, would fail one half.
    #[test]
    fn an_undeclared_reading_bangs_instead_of_defaulting_to_v1() {
        use crate::canonical_node::{EdgeCodecFlavor, ReadMode, TailVariant, ValueSchema};

        const V3_SEAT: ReadMode = ReadMode {
            tail_variant: TailVariant::V3,
            value_schema: ValueSchema::Bootstrap,
            edge_codec: EdgeCodecFlavor::CoarseOnly,
        };
        let act = Activation::new(Vec::new(), Vec::new(), vec![(0x1717, V3_SEAT)]);

        // Can-fire on the happy half: a declared seat resolves to its own
        // reading, not to some other row of the table.
        assert_eq!(act.read_mode_for(0x1717), Ok(V3_SEAT));
        assert_eq!(
            act.read_mode_for(0x1717).unwrap().tail_variant,
            TailVariant::V3
        );

        // …and an undeclared one is a NAMED bang. The anti-vacuity assertion
        // is the second one: the error must not be some value that happens to
        // compare unequal — there must be no ReadMode on this path at all.
        assert_eq!(
            act.read_mode_for(0x1718),
            Err(ActivationDrift::NoReadingFor(0x1718))
        );
        assert!(act.read_mode_for(0x1718).is_err());

        // The V1 fallback this replaces would have returned DEFAULT here.
        // Pinned so the difference is measured, not asserted in prose.
        assert_eq!(ReadMode::DEFAULT.tail_variant, TailVariant::V1);
        assert_ne!(Ok(ReadMode::DEFAULT), act.read_mode_for(0x1718));
    }

    /// An authority with NO readings still activates — and still bangs.
    ///
    /// The silence twin at the other end: a capability-only consumer (an
    /// executor that mints no keys) legitimately declares an empty table, so
    /// an empty table must not be an activation failure. What must never
    /// happen is that consumer's lookup quietly succeeding with V1.
    #[test]
    fn an_empty_table_activates_but_still_refuses_to_invent_a_reading() {
        let plug = HotPlug {
            consumer: "demo",
            classids: &[0x0805],
            covered: &["recognize_line"],
        };
        let act = TinyAuthority.activate(&plug).expect("activation is green");

        assert!(act.declared_readings().is_empty(), "no reading declared");
        assert_eq!(
            act.read_mode_for(0x0805),
            Err(ActivationDrift::NoReadingFor(0x0805)),
            "an empty table is not 'assume the default'"
        );
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
