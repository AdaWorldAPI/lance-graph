//! Hand-annotated sentences in UD German-GSD convention, and an independent
//! gold role per argument.
//!
//! The annotation convention was checked against the GSD treebank (UD
//! master, CC BY-SA 4.0; fetched for measurement, not committed):
//!
//! - a dative recipient of *geben* is `obl:arg` + `Case=Dat`
//!   (`train-s1712`: "gab Shell Mitschuld", Shell `obl:arg` Dat);
//! - a dative experiencer of *gefallen* is ALSO `obl:arg` + `Case=Dat`
//!   (`test-s258`: "das gefällt uns", uns `obl:arg` Dat);
//! - `zur` is a multiword token `zu` (`case`) + `der` (`det`, Dat), and its
//!   noun is `obl` + `Case=Dat` under *bringen* ("zur Reparatur gebracht");
//! - `iobj` is not used for these (1 occurrence in GSD train, 0 in dev/test).
//!
//! Gold roles are a separate, single-annotator linguistic analysis (not
//! adjudicated), kept apart from the rules: they are what the rules are
//! scored against, never an input to them.

use crate::Tok;

macro_rules! t {
    ($gid:expr, $form:expr, $lemma:expr, $deprel:expr, $case:expr, $head:expr) => {
        Tok {
            gid: $gid,
            form: $form,
            lemma: $lemma,
            deprel: $deprel,
            case: $case,
            head: $head,
        }
    };
}

/// A. Die Mutter gibt dem Kind das Buch.
pub const A: [Tok; 8] = [
    t!(101, "Die", "der", "det", "Nom", 102),
    t!(102, "Mutter", "Mutter", "nsubj", "Nom", 103),
    t!(103, "gibt", "geben", "root", "", 0),
    t!(104, "dem", "der", "det", "Dat", 105),
    t!(105, "Kind", "Kind", "obl:arg", "Dat", 103),
    t!(106, "das", "der", "det", "Acc", 107),
    t!(107, "Buch", "Buch", "obj", "Acc", 103),
    t!(108, ".", ".", "punct", "", 103),
];

/// B. Die Mutter bringt das Buch zur Schule. (`zur` = `zu` + `der`)
pub const B: [Tok; 9] = [
    t!(201, "Die", "der", "det", "Nom", 202),
    t!(202, "Mutter", "Mutter", "nsubj", "Nom", 203),
    t!(203, "bringt", "bringen", "root", "", 0),
    t!(204, "das", "der", "det", "Acc", 205),
    t!(205, "Buch", "Buch", "obj", "Acc", 203),
    t!(206, "zu", "zu", "case", "", 208),
    t!(207, "der", "der", "det", "Dat", 208),
    t!(208, "Schule", "Schule", "obl", "Dat", 203),
    t!(209, ".", ".", "punct", "", 203),
];

/// C. Dem Kind gefällt das Buch.
pub const C: [Tok; 6] = [
    t!(301, "Dem", "der", "det", "Dat", 302),
    t!(302, "Kind", "Kind", "obl:arg", "Dat", 303),
    t!(303, "gefällt", "gefallen", "root", "", 0),
    t!(304, "das", "der", "det", "Nom", 305),
    t!(305, "Buch", "Buch", "nsubj", "Nom", 303),
    t!(306, ".", ".", "punct", "", 303),
];

/// D (control). Die Mutter gibt dem Kind mit dem Löffel den Brei.
/// `mit dem Löffel` is a dative `obl` under GIVE that is NOT a recipient.
pub const D: [Tok; 11] = [
    t!(401, "Die", "der", "det", "Nom", 402),
    t!(402, "Mutter", "Mutter", "nsubj", "Nom", 403),
    t!(403, "gibt", "geben", "root", "", 0),
    t!(404, "dem", "der", "det", "Dat", 405),
    t!(405, "Kind", "Kind", "obl:arg", "Dat", 403),
    t!(406, "mit", "mit", "case", "", 408),
    t!(407, "dem", "der", "det", "Dat", 408),
    t!(408, "Löffel", "Löffel", "obl", "Dat", 403),
    t!(409, "den", "der", "det", "Acc", 410),
    t!(410, "Brei", "Brei", "obj", "Acc", 403),
    t!(411, ".", ".", "punct", "", 403),
];

/// E (control). Die Mutter hilft dem Kind. *helfen* has no frame entry.
pub const E: [Tok; 6] = [
    t!(501, "Die", "der", "det", "Nom", 502),
    t!(502, "Mutter", "Mutter", "nsubj", "Nom", 503),
    t!(503, "hilft", "helfen", "root", "", 0),
    t!(504, "dem", "der", "det", "Dat", 505),
    t!(505, "Kind", "Kind", "obl:arg", "Dat", 503),
    t!(506, ".", ".", "punct", "", 503),
];

/// Gold function of an argument token (Haspelmath dative-map names).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gold {
    Recipient,
    Experiencer,
    Direction,
    /// A dative-map function the frame lexicon does not cover.
    Beneficiary,
    /// Not a dative-map function (e.g. an instrument).
    NotOnMap,
}

/// The independent gold annotation (single annotator).
pub const GOLD: [(u64, Gold); 6] = [
    (105, Gold::Recipient),
    (208, Gold::Direction),
    (302, Gold::Experiencer),
    (405, Gold::Recipient),
    (408, Gold::NotOnMap),
    (505, Gold::Beneficiary),
];

/// Every sentence, in corpus order.
pub fn corpus() -> Vec<Tok> {
    let mut v = Vec::new();
    for s in [&A[..], &B[..], &C[..], &D[..], &E[..]] {
        v.extend_from_slice(s);
    }
    v
}

pub fn gids(toks: &[Tok]) -> Vec<u64> {
    toks.iter().map(|t| t.gid).collect()
}
