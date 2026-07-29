//! `UnicharCompress` (the recoder) **keystone** — `classid → ClassView →
//! content store → adapter`, the sibling wiring of
//! [`crate::unicharset_adapter`] over [`crate::unicharcompress::UnicharCompress`].
//!
//! [`crate::unicharcompress`] proved the load-side leaf: byte-parity vs
//! libtesseract on the real `eng.lstm-recoder` (112 pass-through entries; see
//! that module's own doc comments and `examples/recoder_dump.rs`). This
//! module composes that leaf through the same three movable parts
//! [`crate::unicharset_adapter`] proved for `UNICHARSET`, and in doing so
//! proves the dispatch pattern is **class-agnostic**, not fitted to one call
//! shape: `EncodeUnichar` returns a *sequence* of codes (not a scalar id or a
//! borrowed string), and the keystone routes it cleanly without bending the
//! DO-out vocabulary to fit.
//!
//! | Core movable part | Here |
//! |---|---|
//! | identity = `classid` | the `u32` classid parameter (bound OGAR-side — the OGAR codebook's `("recoder", 0x0802)` concept slot in [`crate::ogar_codebook`], never minted here) |
//! | composition = `classid → ClassView` | [`methods_for`] over the harvested `has_function` manifest gates which methods a class composes |
//! | state = classid-keyed content tier | [`RecoderStore`] resolves `classid → &UnicharCompress` (consumer-provided; the adapter holds NO state — `I-VSA-IDENTITIES` content-store tier) |
//! | invocation | [`invoke_recoder`] — the thin DO-in ([`RecoderCall`]) / DO-out ([`RecoderOut`]) shape a [`crate::orchestration::UnifiedStep`] would call |
//!
//! # Why a second keystone, not a copy-paste
//!
//! [`crate::unicharset_adapter`] proved the pattern for a class whose methods
//! return a scalar (`Option<u32>`) or a borrowed string (`Option<&str>`). If
//! this module just repeated that shape, it would prove nothing new — a
//! dispatch that only ever sees scalar/string outputs could still secretly be
//! fitted to that one shape. `UnicharCompress::encode` returns
//! `Option<&RecodedCharId>` — a **sequence** (`length` + up to 9 codes), not a
//! scalar — so this module is the first place the keystone has to decide how
//! a compound DO-out reads. The decision made here (documented on
//! [`RecoderOut::Codes`]): the DO-out stays a **borrowed slice**
//! (`&'a [i32]`), never the internal [`RecodedCharId`] wrapper — the dispatch
//! vocabulary stays thin and class-agnostic on both sides of the keystone,
//! exactly as [`crate::unicharset_adapter::UniCharOut::Unichar`] borrows
//! `&str` rather than some internal string-table row type.
//!
//! # Naming note
//!
//! The Rust type is [`crate::unicharcompress::UnicharCompress`] (mirroring
//! the C++ `UnicharCompress` class name), but this module — like
//! `examples/recoder_dump.rs` before it — is named after the **OGAR codebook
//! concept** it composes: `("recoder", 0x0802)` in
//! [`crate::ogar_codebook::CODEBOOK`]. That is the stable, cross-repo name
//! every consumer (OGAR, tesseract-rs) uses in prose for this class.
//! `unicharset_adapter.rs` happens to share its base name with
//! `unicharset.rs` because the codebook concept (`"unicharset"`, `0x0801`)
//! and the Rust type share a name too; here they diverge, and the codebook
//! concept name wins — consistent with this crate's own precedent.
//!
//! # Errors
//!
//! See [`invoke_recoder`]'s `# Errors` section; this module reuses
//! [`crate::unicharset_adapter::DispatchError`] unchanged — both its existing
//! variants (`MethodNotComposed`, `NoContentStore`) already cover every
//! recoder dispatch failure mode, so no new variant was needed.

use crate::codegen_manifest::{methods_for, ClassMethods};
use crate::unicharcompress::{RecodedCharId, UnicharCompress};
use crate::unicharset_adapter::DispatchError;

/// The classid-keyed **content-store tier**: resolve a `classid` to its
/// loaded [`UnicharCompress`] (recoder). Implemented by the consumer (e.g. a
/// `HashMap<u32, UnicharCompress>` loaded from `.lstm-recoder` components);
/// the contract owns only this vocabulary — the same dependency inversion as
/// [`crate::unicharset_adapter::UniCharSetStore`].
///
/// The content NEVER lives on the node row or the adapter — a variable-length
/// id↔codes table is not fixed-width SoA state; it rides this tier, keyed by
/// the identity (`I-VSA-IDENTITIES`; the core-first doctrine's "content-store
/// tier").
pub trait RecoderStore {
    /// The [`UnicharCompress`] bound to `classid`, or `None` if no store is bound.
    fn recoder(&self, classid: u32) -> Option<&UnicharCompress>;
}

/// A typed call into the recoder adapter set — the **DO-in**. Each variant's
/// [`method_name`](RecoderCall::method_name) is the C++ method the harvested
/// `has_function` manifest must list for the class to compose it. This is
/// exactly the "recognizer runtime surface" [`crate::unicharcompress`]'s
/// module docs scope the load-side leaf to: `EncodeUnichar`, `DecodeUnichar`,
/// and `code_range`. (`ComputeEncoding`, the training-side table builder, and
/// the beam-search trie accessors `IsValidFirstCode` / `GetFinalCodes` /
/// `GetNextCodes` stay out of this dispatch surface — those maps are Core
/// content the recognizer's beam search consumes directly, not a step this
/// keystone routes.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoderCall<'a> {
    /// `RecodedCharID EncodeUnichar(UNICHAR_ID unichar_id) const` — id → code
    /// sequence (`unicharcompress.cpp:295-301`).
    EncodeUnichar(u32),
    /// `UNICHAR_ID DecodeUnichar(const RecodedCharID &code) const` — code
    /// sequence → id (`unicharcompress.cpp:305-315`). The raw codes stay a
    /// primitive `&'a [i32]` on the DO-in side (never the internal
    /// [`RecodedCharId`] wrapper); the adapter builds the lookup key via
    /// [`RecodedCharId::from_codes`] immediately before dispatching to
    /// [`UnicharCompress::decode`].
    DecodeUnichar(&'a [i32]),
    /// `int code_range() const` — the lattice width (`unicharcompress.h:171`).
    /// Arity 0: no id or codes needed, just the loaded table's own width.
    CodeRange,
}

impl RecoderCall<'_> {
    /// The C++ method name this call dispatches to — the key the ClassView
    /// method manifest must contain for the call to be composed. These match
    /// the `UnicharCompress` member names the `ruff_cpp_spo` harvest emits.
    #[must_use]
    pub const fn method_name(&self) -> &'static str {
        match self {
            Self::EncodeUnichar(_) => "EncodeUnichar",
            Self::DecodeUnichar(_) => "DecodeUnichar",
            Self::CodeRange => "code_range",
        }
    }
}

/// The adapter's typed result — the **DO-out**.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecoderOut<'a> {
    /// The code sequence for an id (`None` = out of range), from
    /// [`RecoderCall::EncodeUnichar`]. Borrows the codes slice straight out
    /// of the content-store's [`RecodedCharId`] (zero-copy; the store
    /// outlives the call) — a **borrowed slice**, not the internal
    /// [`RecodedCharId`] wrapper, keeping the DO-out vocabulary thin exactly
    /// like [`crate::unicharset_adapter::UniCharOut::Unichar`] borrows `&str`
    /// rather than a string-table row. This is the shape decision the module
    /// docs call out: `EncodeUnichar` is the first method in the keystone
    /// whose C++ return is a sequence, not a scalar.
    Codes(Option<&'a [i32]>),
    /// The id for a code sequence (`None` = the C++ `INVALID_UNICHAR_ID`
    /// sentinel — an unknown or ill-formed code), from
    /// [`RecoderCall::DecodeUnichar`].
    Id(Option<u32>),
    /// The lattice width, from [`RecoderCall::CodeRange`]. Infallible once
    /// the content store resolves — unlike the other two calls, there is no
    /// per-argument "out of range" case.
    CodeRange(i32),
}

/// **The keystone, applied to a second class.** Invoke a `UnicharCompress`
/// (recoder) adapter through the `classid → ClassView` method manifest and
/// the classid-keyed content store — the same steps
/// [`crate::unicharset_adapter::invoke_unicharset`] proved, now over a class
/// whose methods return sequences rather than scalars/strings.
///
/// 1. **ClassView composition gate**: [`methods_for(registry, classid)`](methods_for)
///    must contain a method whose name is the call's
///    [`method_name`](RecoderCall::method_name) — i.e. the harvested
///    `has_function` manifest says this class composes this adapter.
///    Otherwise [`DispatchError::MethodNotComposed`] (an unconfigured classid
///    composes nothing — the zero-fallback ladder).
/// 2. **Content-store tier**: [`store.recoder(classid)`](RecoderStore::recoder)
///    supplies the loaded recoder; the adapter holds no state of its own.
/// 3. **Adapter leaf**: route to [`UnicharCompress::encode`] /
///    [`UnicharCompress::decode`] / [`UnicharCompress::code_range`]. The
///    `DecodeUnichar` call's raw `&[i32]` codes are wrapped into a
///    [`RecodedCharId`] via [`RecodedCharId::from_codes`] immediately before
///    the leaf call — a cheap, infallible, stack-only conversion; the
///    wrapper never crosses the DO-in/DO-out boundary.
///
/// Byte-parity is inherited from [`UnicharCompress`] (byte-identical vs
/// libtesseract on the real `eng.lstm-recoder`); this proves the dispatch
/// path is faithful and the composition gate holds for a second,
/// structurally different class.
///
/// # Errors
///
/// [`DispatchError::MethodNotComposed`] if the class's manifest does not
/// list the called method; [`DispatchError::NoContentStore`] if no
/// `UnicharCompress` is bound to the classid. No new error variant was
/// needed — both of [`DispatchError`]'s existing variants already cover
/// every recoder dispatch failure mode (see the module docs).
pub fn invoke_recoder<'a, S: RecoderStore + ?Sized>(
    registry: &[ClassMethods],
    store: &'a S,
    classid: u32,
    call: &RecoderCall<'_>,
) -> Result<RecoderOut<'a>, DispatchError> {
    let method = call.method_name();
    // 1. ClassView composition gate (the harvested has_function manifest).
    if !methods_for(registry, classid)
        .iter()
        .any(|m| m.name == method)
    {
        return Err(DispatchError::MethodNotComposed { classid, method });
    }
    // 2. Content-store tier — state lives here, never on the adapter.
    let recoder = store
        .recoder(classid)
        .ok_or(DispatchError::NoContentStore { classid })?;
    // 3. The proven adapter leaf.
    Ok(match *call {
        RecoderCall::EncodeUnichar(id) => {
            RecoderOut::Codes(recoder.encode(id).map(|rc| rc.codes()))
        }
        RecoderCall::DecodeUnichar(codes) if codes.len() > RecodedCharId::MAX_CODE_LEN => {
            // An overlong slice must be REJECTED before the key is built.
            // `RecodedCharId::from_codes` silently truncates to the first
            // `MAX_CODE_LEN` codes, so a 10-code input whose 9-code prefix
            // happens to be a real code would otherwise decode SUCCESSFULLY
            // to that prefix's id — aliasing an ill-formed input onto a valid
            // answer, and breaking this arm's contract that an ill-formed
            // sequence yields `None`. Truncation is lossy in a way that is
            // invisible after the fact, so the guard has to live here, at the
            // DO boundary, not inside the constructor.
            RecoderOut::Id(None)
        }
        RecoderCall::DecodeUnichar(codes) => {
            let key = RecodedCharId::from_codes(codes);
            let id = recoder.decode(&key);
            // `decode` returns the C++ INVALID_UNICHAR_ID sentinel (-1) for
            // an unknown/ill-formed code; every valid id is a non-negative
            // Vec index (bounded well under i32::MAX by the
            // 50_000_000-element sanity cap in
            // `UnicharCompress::from_le_bytes`), so `< 0` is exactly the
            // sentinel check — converted to a clean `None` here, mirroring
            // `UniCharOut::Id`.
            RecoderOut::Id((id >= 0).then_some(id as u32))
        }
        RecoderCall::CodeRange => RecoderOut::CodeRange(recoder.code_range()),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codegen_manifest::MethodSig;
    use crate::ogar_codebook::{canonical_concept_id, render_classid};
    use std::collections::HashMap;

    /// The real OGAR-minted classid for the Tesseract `UnicharCompress`
    /// (recoder) class under the Core render lens: `ogar_codebook::CODEBOOK`'s
    /// `("recoder", 0x0802)` concept slot (the `0x08XX` OCR domain), composed
    /// via `render_classid(prefix=Core=0x0000, concept=0x0802)` — the bare
    /// concept in the CANON (high) half, no app render lens. Unlike
    /// `unicharset_adapter`'s test classid (an arbitrary `0x0001_0001`
    /// placeholder), this one is derived from the live codebook, so a
    /// renumbering there is caught by
    /// `classid_matches_the_live_ogar_codebook_entry` below rather than
    /// silently drifting.
    const TESS_RECODER: u32 = render_classid(0x0000, 0x0802);

    #[test]
    fn classid_matches_the_live_ogar_codebook_entry() {
        assert_eq!(canonical_concept_id("recoder"), Some(0x0802));
        assert_eq!(TESS_RECODER, render_classid(0x0000, 0x0802));
    }

    /// The ClassView method manifest the `ruff_cpp_spo` harvest emits for
    /// `UnicharCompress` — the three byte-parity-proven leaves this dispatch
    /// surface covers.
    const RECODER_METHODS: &[MethodSig] = &[
        MethodSig {
            name: "EncodeUnichar",
            params: &["UNICHAR_ID"],
            ret: Some("RecodedCharID"),
            is_const: true,
            is_static: false,
            overrides: None,
        },
        MethodSig {
            name: "DecodeUnichar",
            params: &["const RecodedCharID &"],
            ret: Some("UNICHAR_ID"),
            is_const: true,
            is_static: false,
            overrides: None,
        },
        MethodSig {
            name: "code_range",
            params: &[],
            ret: Some("int"),
            is_const: true,
            is_static: false,
            overrides: None,
        },
    ];

    const REGISTRY: &[ClassMethods] = &[ClassMethods {
        classid: TESS_RECODER,
        methods: RECODER_METHODS,
    }];

    /// An in-memory content store — what a consumer builds from
    /// `.lstm-recoder` components. The contract owns the trait; the
    /// consumer owns the answers.
    struct MemStore(HashMap<u32, UnicharCompress>);

    impl RecoderStore for MemStore {
        fn recoder(&self, classid: u32) -> Option<&UnicharCompress> {
            self.0.get(&classid)
        }
    }

    /// Build a minimal `.lstm-recoder` byte buffer in the exact
    /// little-endian wire form [`crate::unicharcompress`]'s module docs
    /// specify (mirrors that module's own private test helper of the same
    /// shape, duplicated here since it is not reachable across module
    /// boundaries). No real `eng.lstm-recoder` fixture is reachable from this
    /// crate — that binary component lives only in the tesseract-rs corpus,
    /// a sibling repo — so this in-memory recoder is constructed directly: a
    /// pass-through shape (every entry length-1), the same shape family
    /// `eng.lstm-recoder`'s 112 real entries take.
    fn build_recoder_bytes(entries: &[(i8, &[i32])]) -> Vec<u8> {
        let mut b = Vec::new();
        b.extend_from_slice(&u32::try_from(entries.len()).unwrap().to_le_bytes());
        for (self_norm, codes) in entries {
            b.push(*self_norm as u8);
            b.extend_from_slice(&i32::try_from(codes.len()).unwrap().to_le_bytes());
            for &c in *codes {
                b.extend_from_slice(&c.to_le_bytes());
            }
        }
        b
    }

    /// ids 0/1/2 -> codes `[0]`/`[1]`/`[2]` (code_range = 3) — a pass-through
    /// recoder in miniature, the same shape family as `eng.lstm-recoder`.
    fn store_with_passthrough_sample() -> MemStore {
        let bytes = build_recoder_bytes(&[(1, &[0]), (1, &[1]), (1, &[2])]);
        let rec = UnicharCompress::from_le_bytes(&bytes).expect("valid");
        let mut m = HashMap::new();
        m.insert(TESS_RECODER, rec);
        MemStore(m)
    }

    /// The full keystone path for `EncodeUnichar`: classid → ClassView →
    /// content store → adapter leaf, id 1 -> codes `[1]`.
    #[test]
    fn keystone_encode_dispatches_with_inherited_byte_parity() {
        let store = store_with_passthrough_sample();
        assert_eq!(
            invoke_recoder(
                REGISTRY,
                &store,
                TESS_RECODER,
                &RecoderCall::EncodeUnichar(1)
            ),
            Ok(RecoderOut::Codes(Some(&[1][..]))),
            "the encode path survives classid -> ClassView -> store -> adapter"
        );
        assert_eq!(
            invoke_recoder(
                REGISTRY,
                &store,
                TESS_RECODER,
                &RecoderCall::EncodeUnichar(99)
            ),
            Ok(RecoderOut::Codes(None)),
            "out-of-range id is a clean None, not a panic"
        );
    }

    /// The full keystone path for `DecodeUnichar`: codes `[1]` -> id 1, and
    /// an unknown code -> a clean `None` (never the C++ raw -1 sentinel
    /// leaking through the dispatch boundary).
    #[test]
    fn keystone_decode_dispatches_with_inherited_byte_parity() {
        let store = store_with_passthrough_sample();
        assert_eq!(
            invoke_recoder(
                REGISTRY,
                &store,
                TESS_RECODER,
                &RecoderCall::DecodeUnichar(&[1])
            ),
            Ok(RecoderOut::Id(Some(1)))
        );
        assert_eq!(
            invoke_recoder(
                REGISTRY,
                &store,
                TESS_RECODER,
                &RecoderCall::DecodeUnichar(&[42])
            ),
            Ok(RecoderOut::Id(None)),
            "unknown code decodes to a clean None, not the raw -1 sentinel"
        );
    }

    /// An empty codes slice is the C++ "ill-formed" (`length == 0`) decode
    /// input — also a clean `None`, not a panic or a spurious id 0.
    #[test]
    fn keystone_decode_empty_codes_is_clean_none() {
        let store = store_with_passthrough_sample();
        assert_eq!(
            invoke_recoder(
                REGISTRY,
                &store,
                TESS_RECODER,
                &RecoderCall::DecodeUnichar(&[])
            ),
            Ok(RecoderOut::Id(None))
        );
    }

    /// **Regression (codex P2 on PR #864).** An OVERLONG codes slice must
    /// decode to `None`, never alias onto its own truncated prefix.
    ///
    /// `RecodedCharId::from_codes` keeps only the first
    /// [`RecodedCharId::MAX_CODE_LEN`] codes and silently drops the rest, so
    /// before the guard a 10-code input whose 9-code prefix was a REAL code
    /// decoded successfully to that prefix's id — an ill-formed input
    /// producing a confident, valid-looking answer. Truncation is invisible
    /// after construction, so the length check has to happen at the DO
    /// boundary.
    ///
    /// The fixture is built to make the aliasing reachable: a single entry
    /// whose code IS a full-length 9-sequence, so the 10-code probe's prefix
    /// is exactly that valid code. A pass-through (length-1) recoder could
    /// not falsify this at all — the prefix would never resolve.
    #[test]
    fn keystone_decode_overlong_codes_is_none_not_a_truncated_alias() {
        let full: [i32; 9] = [1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(
            full.len(),
            RecodedCharId::MAX_CODE_LEN,
            "the fixture must sit exactly at the truncation boundary"
        );
        let bytes = build_recoder_bytes(&[(1, &full[..])]);
        let rec = UnicharCompress::from_le_bytes(&bytes).expect("valid");
        let mut m = HashMap::new();
        m.insert(TESS_RECODER, rec);
        let store = MemStore(m);

        // The exact-length sequence is a real code — this is what makes the
        // overlong case dangerous rather than merely wrong.
        assert_eq!(
            invoke_recoder(
                REGISTRY,
                &store,
                TESS_RECODER,
                &RecoderCall::DecodeUnichar(&full[..])
            ),
            Ok(RecoderOut::Id(Some(0))),
            "the 9-code sequence itself must still decode"
        );

        // One code too many: the prefix is valid, so truncation WOULD have
        // returned Some(0). The guard must return None instead.
        let overlong: [i32; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        assert_eq!(
            invoke_recoder(
                REGISTRY,
                &store,
                TESS_RECODER,
                &RecoderCall::DecodeUnichar(&overlong[..])
            ),
            Ok(RecoderOut::Id(None)),
            "an overlong sequence must NOT alias onto its truncated prefix"
        );
    }

    /// **Regression (codex P2 on PR #864).** `DispatchError` is SHARED by
    /// every keystone adapter, so its `Display` must not name one specific
    /// content type — it used to say "no UniCharSet content store", which
    /// misidentifies the component when a *recoder* store is the one missing.
    #[test]
    fn missing_store_error_message_names_no_specific_content_type() {
        let store = MemStore(HashMap::new());
        let err = invoke_recoder(
            REGISTRY,
            &store,
            TESS_RECODER,
            &RecoderCall::DecodeUnichar(&[1]),
        )
        .expect_err("an empty store cannot answer");
        let msg = err.to_string();
        assert!(
            !msg.contains("UniCharSet"),
            "shared error must not name UniCharSet when a recoder is missing: {msg}"
        );
        assert!(
            msg.contains("content store"),
            "still says what is missing: {msg}"
        );
    }

    /// `code_range` dispatches with no id/codes argument at all.
    #[test]
    fn keystone_code_range_dispatches() {
        let store = store_with_passthrough_sample();
        assert_eq!(
            invoke_recoder(REGISTRY, &store, TESS_RECODER, &RecoderCall::CodeRange),
            Ok(RecoderOut::CodeRange(3))
        );
    }

    /// The composition gate: a class whose manifest composes only
    /// `EncodeUnichar` refuses `DecodeUnichar`, even though the content
    /// store could answer it.
    #[test]
    fn keystone_classview_gate_rejects_uncomposed_method() {
        const ENCODE_ONLY: &[ClassMethods] = &[ClassMethods {
            classid: TESS_RECODER,
            methods: &[MethodSig {
                name: "EncodeUnichar",
                params: &["UNICHAR_ID"],
                ret: Some("RecodedCharID"),
                is_const: true,
                is_static: false,
                overrides: None,
            }],
        }];
        let store = store_with_passthrough_sample();
        assert_eq!(
            invoke_recoder(
                ENCODE_ONLY,
                &store,
                TESS_RECODER,
                &RecoderCall::DecodeUnichar(&[1])
            ),
            Err(DispatchError::MethodNotComposed {
                classid: TESS_RECODER,
                method: "DecodeUnichar",
            })
        );
    }

    /// Zero-fallback: an unregistered classid composes nothing, so the gate
    /// refuses before the store is even consulted.
    #[test]
    fn keystone_unregistered_classid_composes_nothing() {
        let store = store_with_passthrough_sample();
        assert_eq!(
            invoke_recoder(
                REGISTRY,
                &store,
                0xDEAD_BEEF,
                &RecoderCall::EncodeUnichar(0)
            ),
            Err(DispatchError::MethodNotComposed {
                classid: 0xDEAD_BEEF,
                method: "EncodeUnichar",
            })
        );
    }

    /// A composed method with no bound content store is a typed error, not
    /// a panic.
    #[test]
    fn keystone_missing_content_store_is_typed_error() {
        let empty = MemStore(HashMap::new());
        assert_eq!(
            invoke_recoder(
                REGISTRY,
                &empty,
                TESS_RECODER,
                &RecoderCall::EncodeUnichar(0)
            ),
            Err(DispatchError::NoContentStore {
                classid: TESS_RECODER,
            })
        );
    }

    #[test]
    fn call_method_names_match_the_harvest() {
        assert_eq!(RecoderCall::EncodeUnichar(0).method_name(), "EncodeUnichar");
        assert_eq!(
            RecoderCall::DecodeUnichar(&[0]).method_name(),
            "DecodeUnichar"
        );
        assert_eq!(RecoderCall::CodeRange.method_name(), "code_range");
    }
}
