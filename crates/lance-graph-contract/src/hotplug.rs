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
//!
//! # SPOG × slab metadata → tenant reading (the ONE resolution path)
//!
//! A row is read under three separate concerns, and this module is where they
//! meet — never a second registry beside it:
//!
//! | concern | carrier | answers |
//! |---|---|---|
//! | SPOG context | the row key, via [`crate::spog_tenants::graph_of`] | which concept/graph this row belongs to |
//! | OGAR registry | [`Activation`] (from [`CapabilityAuthority::activate`]) | is that concept plugged, and how its class reads |
//! | slab metadata | [`SlabDeclaration`] (beside the slab, never inside it) | how THIS physical slab was written |
//!
//! [`Activation::resolve_tenant_reading`] combines them into one
//! [`ReadMode`](crate::canonical_node::ReadMode) or a named
//! [`ActivationDrift`]. Tenant bytes stay content-blind: nothing on this path
//! decodes a payload to decide how to read it.

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

/// What a physical slab DECLARES about how it was written — the slab half of
/// the SPOG × slab resolution.
///
/// It is metadata that lives BESIDE the slab (a group header, the way a
/// palette group carries its gamma once rather than per cell), never a field
/// decoded from tenant bytes: the payload stays content-blind. Where the
/// declaration is physically stored is not this type's concern; the type
/// fixes only what a declaration may say and how it is checked.
///
/// A declaration is a CLAIM, not an authority. It is checked against the
/// reading the OGAR authority resolved for the same concept, and it may only
/// NARROW that reading — see [`Activation::resolve_tenant_reading`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlabDeclaration {
    /// The concept (canon-high half of the classid) the slab was written
    /// under. Must equal the SPOG graph of the row being read.
    pub concept: u16,
    /// How the slab was written.
    pub read_mode: crate::canonical_node::ReadMode,
    /// [`crate::soa_envelope::ENVELOPE_LAYOUT_VERSION`] at write time. A slab
    /// written under another layout is refused, never reinterpreted.
    pub layout_version: u8,
}

impl Activation {
    /// Resolve the reading for one row: SPOG context × slab declaration,
    /// validated by this activation. The single resolution path.
    ///
    /// 1. **SPOG** — the row's concept is [`crate::spog_tenants::graph_of`]
    ///    of its key. Read from the key, never from the payload.
    /// 2. **Registry** — that concept must have a reading in this activation
    ///    ([`read_mode_for`](Activation::read_mode_for)); otherwise
    ///    [`ActivationDrift::NoReadingFor`]. No slab declaration can stand in
    ///    for a missing authority reading.
    /// 3. **Slab** — with no declaration, the authority's reading is the
    ///    answer. With one:
    ///    - it must name the same concept
    ///      ([`ActivationDrift::SlabConceptMismatch`]);
    ///    - it must carry the current envelope layout
    ///      ([`ActivationDrift::SlabLayoutVersion`]);
    ///    - its tail and edge codec must EQUAL the authority's — they read the
    ///      key and edge block, which are class semantics, not slab choices;
    ///    - its value schema may be NARROWER (a subset of tenants: the slab
    ///      wrote fewer than the class allows), never wider.
    ///
    ///    Any other difference is [`ActivationDrift::SlabReadingConflict`].
    ///    The declared reading is returned when it passes, because it is the
    ///    one that matches the bytes actually written.
    ///
    /// # Errors
    ///
    /// The named [`ActivationDrift`] arms above. There is no fallback reading
    /// on any path.
    pub fn resolve_tenant_reading(
        &self,
        key: crate::canonical_node::NodeGuid,
        slab: Option<&SlabDeclaration>,
    ) -> Result<crate::canonical_node::ReadMode, ActivationDrift> {
        let concept = crate::spog_tenants::graph_of(key);
        let authority = self.read_mode_for(concept)?;
        let Some(slab) = slab else {
            return Ok(authority);
        };
        if slab.concept != concept {
            return Err(ActivationDrift::SlabConceptMismatch {
                spog: concept,
                slab: slab.concept,
            });
        }
        if slab.layout_version != crate::soa_envelope::ENVELOPE_LAYOUT_VERSION {
            return Err(ActivationDrift::SlabLayoutVersion {
                slab: slab.layout_version,
                expected: crate::soa_envelope::ENVELOPE_LAYOUT_VERSION,
            });
        }
        let declared = slab.read_mode;
        let narrows = declared
            .value_schema
            .field_mask()
            .is_subset_of(authority.value_schema.field_mask());
        if declared.tail_variant != authority.tail_variant
            || declared.edge_codec != authority.edge_codec
            || !narrows
        {
            return Err(ActivationDrift::SlabReadingConflict {
                concept,
                authority,
                declared,
            });
        }
        Ok(declared)
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
    /// A slab declaration names a different concept than the row's SPOG
    /// graph ([`crate::spog_tenants::graph_of`] of its key).
    SlabConceptMismatch {
        /// The concept read from the row key.
        spog: u16,
        /// The concept the slab declared.
        slab: u16,
    },
    /// A slab was written under another envelope layout version.
    SlabLayoutVersion {
        /// The version the slab declared.
        slab: u8,
        /// [`crate::soa_envelope::ENVELOPE_LAYOUT_VERSION`].
        expected: u8,
    },
    /// A slab declaration contradicts or widens the authority's reading.
    SlabReadingConflict {
        /// The concept both readings are for.
        concept: u16,
        /// What the authority resolved.
        authority: crate::canonical_node::ReadMode,
        /// What the slab declared.
        declared: crate::canonical_node::ReadMode,
    },
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
            Self::SlabConceptMismatch { spog, slab } => write!(
                f,
                "slab declares concept 0x{slab:04X} but the row's SPOG graph is 0x{spog:04X}"
            ),
            Self::SlabLayoutVersion { slab, expected } => write!(
                f,
                "slab written under envelope layout v{slab}, this build reads v{expected}"
            ),
            Self::SlabReadingConflict {
                concept,
                authority,
                declared,
            } => write!(
                f,
                "slab reading {declared:?} for concept 0x{concept:04X} contradicts or \
                 widens the authority's {authority:?}"
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

    /// SPOG × slab resolution fixture: one plugged concept (0x0901) whose
    /// authority reading is V3 / Full / CoarseOnly.
    mod slab_resolution {
        use super::super::*;
        use crate::canonical_node::{
            EdgeCodecFlavor, NodeGuid, ReadMode, TailVariant, ValueSchema,
        };
        use crate::soa_envelope::ENVELOPE_LAYOUT_VERSION;

        const AUTH: ReadMode = ReadMode::PLUG_AND_PLAY_V3;

        fn act() -> Activation {
            Activation::new(Vec::new(), Vec::new(), vec![(0x0901, AUTH)])
        }

        /// A row whose SPOG graph (canon-high half of the classid) is `concept`.
        fn row(concept: u16) -> NodeGuid {
            NodeGuid::new(u32::from(concept) << 16, 1, 2, 3, 0x66, 7)
        }

        fn slab(concept: u16, read_mode: ReadMode) -> SlabDeclaration {
            SlabDeclaration {
                concept,
                read_mode,
                layout_version: ENVELOPE_LAYOUT_VERSION,
            }
        }

        /// No declaration: the authority's reading. An unplugged concept bangs
        /// even WITH a declaration — a slab cannot stand in for the registry.
        #[test]
        fn the_authority_answers_and_a_slab_cannot_replace_it() {
            assert_eq!(act().resolve_tenant_reading(row(0x0901), None), Ok(AUTH));
            assert_eq!(
                act().resolve_tenant_reading(row(0x0902), Some(&slab(0x0902, AUTH))),
                Err(ActivationDrift::NoReadingFor(0x0902))
            );
        }

        /// The slab may NARROW the value schema, and the narrower reading is
        /// what comes back (it matches the bytes written). Two-sided: the same
        /// difference in the other direction (widening) is refused.
        #[test]
        fn a_slab_narrows_the_value_schema_but_never_widens_it() {
            let narrow = ReadMode {
                value_schema: ValueSchema::Bootstrap,
                ..AUTH
            };
            let got = act().resolve_tenant_reading(row(0x0901), Some(&slab(0x0901, narrow)));
            assert_eq!(got, Ok(narrow));
            assert_ne!(got, Ok(AUTH), "anti-vacuity: the declaration was used");

            let bootstrap_auth = Activation::new(Vec::new(), Vec::new(), vec![(0x0901, narrow)]);
            assert_eq!(
                bootstrap_auth.resolve_tenant_reading(row(0x0901), Some(&slab(0x0901, AUTH))),
                Err(ActivationDrift::SlabReadingConflict {
                    concept: 0x0901,
                    authority: narrow,
                    declared: AUTH,
                })
            );
        }

        /// Tail and edge codec read the key and edge block: class semantics, so
        /// a slab that disagrees on either is refused, not obeyed.
        #[test]
        fn tail_and_edge_codec_must_match_the_authority() {
            for declared in [
                ReadMode {
                    tail_variant: TailVariant::V1,
                    ..AUTH
                },
                ReadMode {
                    edge_codec: EdgeCodecFlavor::Pq32x4,
                    ..AUTH
                },
            ] {
                assert!(matches!(
                    act().resolve_tenant_reading(row(0x0901), Some(&slab(0x0901, declared))),
                    Err(ActivationDrift::SlabReadingConflict { .. })
                ));
            }
        }

        /// The declaration must be about THIS row's SPOG graph and THIS layout.
        #[test]
        fn concept_and_layout_version_are_checked() {
            assert_eq!(
                act().resolve_tenant_reading(row(0x0901), Some(&slab(0x0902, AUTH))),
                Err(ActivationDrift::SlabConceptMismatch {
                    spog: 0x0901,
                    slab: 0x0902,
                })
            );
            let stale = SlabDeclaration {
                layout_version: ENVELOPE_LAYOUT_VERSION.wrapping_sub(1),
                ..slab(0x0901, AUTH)
            };
            assert_eq!(
                act().resolve_tenant_reading(row(0x0901), Some(&stale)),
                Err(ActivationDrift::SlabLayoutVersion {
                    slab: ENVELOPE_LAYOUT_VERSION.wrapping_sub(1),
                    expected: ENVELOPE_LAYOUT_VERSION,
                })
            );
            // Silence twin: a matching declaration passes unchanged.
            assert_eq!(
                act().resolve_tenant_reading(row(0x0901), Some(&slab(0x0901, AUTH))),
                Ok(AUTH)
            );
        }
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
