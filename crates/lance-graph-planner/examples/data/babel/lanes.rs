// Curated verbatim multilingual lexeme fixture — PROBE-BABEL-STANCES slice 1.
// Sources (all public domain): Luther Bible 1545 (original orthography, e.g.
// "gewar", "erkandte", "nacket"); Bible Kralická 1613; Clementine Vulgate;
// Septuagint (Rahlfs); Targum Onkelos (Aramaic, Hebrew script, unpointed).
// Each row's comment carries the relevant clause quote (curator's best
// reconstruction from the named public-domain edition) and a confidence
// marker: VERIFIED (curator is confident this matches the edition's actual
// wording) or CHECK (reconstruction from general knowledge of the language/
// edition's patterns, not a character-verified quote — orchestrator must
// verify against a facsimile/critical text before this row is trusted for
// anything beyond the demo). The en-kjv lane (lane 0) is provided by the
// orchestrator from the local Gutenberg #10 corpus, machine-verified, and is
// NOT duplicated here.
//
// Anchor-consistency note: rows marked "(anchor)" reproduce a KNOW-pair
// surface/root the orchestrator had already verified before this file was
// curated (Luther 3:7/4:1, LXX 3:7/4:1, Vulgate 3:7/4:1, BKR 3:7/4:1, Onkelos
// 3:7/4:1 root ידע). No discrepancy was found against any anchor fact.
//
// DEVIATION FLAGGED: cs-bkr's DIE pair does NOT share one root across its two
// addresses (unlike every other lane) — see the BKR DIE rows below and the
// worker report for detail. This is recorded faithfully per instruction, not
// forced into the same-root pattern the other four lanes show.

/// One curated lexeme: which lane, which grid pair, which verse address,
/// the surface form in that edition, and its lexical root (the stem that
/// decides root-sharing across a pair's two addresses). `pos`/`morph`/`prag`
/// carry the LEVEL-SPLIT fields — Morphologie (`morph`, the prefix|stem
/// decomposition of `root`), Syntax (`pos`), and Pragmatik (`prag`) —
/// alongside the pre-existing Semantik level, which stays put in
/// `root`/`surface` (never duplicated into the new fields). `prag` marks
/// the Hebrew ידע ("yada") euphemism calqued through every lane's KNOW
/// pair at Gen 4:1.
pub struct LaneLex {
    pub lane: &'static str,
    pub pair: &'static str, // "KNOW" | "NAKED" | "DIE"
    pub addr: (u8, u8),     // (chapter, verse)
    pub surface: &'static str,
    pub root: &'static str,
    pub pos: &'static str,   // syntax axis: "v" | "adj"
    pub morph: &'static str, // "prefix|stem" decomposition of the root; "|root" when no prefix
    pub prag: &'static str,  // "plain" | "euphemism-calque"
}

pub const CURATED: &[LaneLex] = &[
    // ══════════════════════════════════════════════════════════════════
    // de-luther1545 — Luther Bible 1545, original orthography
    // ══════════════════════════════════════════════════════════════════

    // Gen 3:7 "Da wurden jr beyder Augen auffgethan, vnd wurden gewar, das
    //          sie nacket waren, vnd flochten Feigenbletter zusamen, vnd
    //          machten jnen Schürtze." — [VERIFIED, anchor]
    LaneLex { lane: "de-luther1545", pair: "KNOW", addr: (3, 7), surface: "gewar", root: "gewahr", pos: "adj", morph: "|gewahr", prag: "plain" },
    // Gen 4:1 "VND Adam erkandte sein Weib Heua, vnd sie ward schwanger,
    //          vnd gebar den Cain, vnd sprach, Ich habe den Man des HERRN."
    //          — [VERIFIED, anchor]
    LaneLex { lane: "de-luther1545", pair: "KNOW", addr: (4, 1), surface: "erkandte", root: "kennen", pos: "v", morph: "er|kennen", prag: "euphemism-calque" },
    // Gen 2:25 "Vnd sie waren beide nacket, der Mensch vnd sein Weib, vnd
    //           schemeten sich nicht." — [VERIFIED]
    LaneLex { lane: "de-luther1545", pair: "NAKED", addr: (2, 25), surface: "nacket", root: "nacket", pos: "adj", morph: "|nacket", prag: "plain" },
    // Gen 3:7 (same clause as the KNOW row above) "...vnd wurden gewar, das
    //          sie nacket waren..." — [VERIFIED]
    LaneLex { lane: "de-luther1545", pair: "NAKED", addr: (3, 7), surface: "nacket", root: "nacket", pos: "adj", morph: "|nacket", prag: "plain" },
    // Gen 2:17 "aber von dem Baum des erkentnis Guts vnd Böses soltu nicht
    //           essen, Denn welches tages du dauon issest, wirstu des
    //           Todes sterben." — [CHECK: curator confident of the
    //           "des Todes sterben" doubling-construction and the root, but
    //           not fully confident of exact clause word order / spelling
    //           of "Todes" (1545 print may show "tods") without a facsimile]
    LaneLex { lane: "de-luther1545", pair: "DIE", addr: (2, 17), surface: "sterben", root: "sterben", pos: "v", morph: "|sterben", prag: "plain" },
    // Gen 3:4 "Da sprach die Schlange zum Weibe, Jr werdet mit nichten des
    //          Todes sterben." — [CHECK: curator recalls the famous
    //          "mit nichten" emphatic-negation choice with reasonable
    //          confidence, but has not verified the exact clause order
    //          against a facsimile]
    LaneLex { lane: "de-luther1545", pair: "DIE", addr: (3, 4), surface: "sterben", root: "sterben", pos: "v", morph: "|sterben", prag: "plain" },

    // ══════════════════════════════════════════════════════════════════
    // cs-bkr — Bible Kralická (1613)
    // ══════════════════════════════════════════════════════════════════

    // Gen 3:7 "Tedy otevříny jsou oči obou dvou, a poznali, že jsou nazí..."
    // — [CHECK: curator is not fully confident of exact 1613 orthography /
    //   word order; reconstruction from general knowledge of BKR style —
    //   anchor confirms root "poznati" at both KNOW addresses]
    // morph note (operator steer, prefix-meaning-switch): Czech is the
    // aspectual case of the same phenomenon Latin/German show lexically —
    // po-/u-/ze- are perfectivizing prefixes on a bare stem, so the KNOW pair
    // shares its prefix (agreement) while the DIE pair switches it
    // (u|mříti ↔ ze|mříti) — divergence attributable to morphology ALONE.
    LaneLex { lane: "cs-bkr", pair: "KNOW", addr: (3, 7), surface: "poznali", root: "poznati", pos: "v", morph: "po|znati", prag: "plain" },
    // Gen 4:1 "Adam pak poznal Evu ženu svou, kterážto počavši, porodila
    //          Kaina..." — [CHECK: same caveat as above; anchor confirms
    //          root]
    LaneLex { lane: "cs-bkr", pair: "KNOW", addr: (4, 1), surface: "poznal", root: "poznati", pos: "v", morph: "po|znati", prag: "euphemism-calque" },
    // Gen 2:25 "Byli pak oba dva nazí, Adam i žena jeho, a nestyděli se."
    // — [CHECK: reconstruction, not a verified quote]
    LaneLex { lane: "cs-bkr", pair: "NAKED", addr: (2, 25), surface: "nazí", root: "nahý", pos: "adj", morph: "|nahý", prag: "plain" },
    // Gen 3:7 (same clause as KNOW row above) "...a poznali, že jsou
    //          nazí..." — [CHECK: reconstruction]
    LaneLex { lane: "cs-bkr", pair: "NAKED", addr: (3, 7), surface: "nazí", root: "nahý", pos: "adj", morph: "|nahý", prag: "plain" },
    // Gen 2:17 "...nebo v který bys koli den z něho jedl, smrtí umřeš."
    // — [CHECK: reconstruction of the Hebrew-doubling-mirroring
    //   "smrtí umřeš" (instrumental noun "smrtí" + perfective verb
    //   "umřeš", prefix u-) — curator has moderate confidence in the
    //   pattern, low confidence in exact wording]
    LaneLex { lane: "cs-bkr", pair: "DIE", addr: (2, 17), surface: "umřeš", root: "umříti", pos: "v", morph: "u|mříti", prag: "plain" },
    // Gen 3:4 "I řekl had ženě: Nikoli, nezemřete smrtí." — [CHECK:
    //   reconstruction; curator flags this as the requested "check what
    //   BKR actually uses" case]
    //
    // ⚠ DEVIATION (flagged per brief instruction, NOT overwritten to fit
    // the expected pattern): the DIE-pair verb at 3:4 is "nezemřete", built
    // on the perfective prefix "ze-" (zemříti), whereas 2:17 uses "umřeš"
    // on the perfective prefix "u-" (umříti). Both share the bare Slavic
    // root morpheme "mří-" (infinitive "mříti"/"mřít"), but as DISTINCT
    // prefixed lemmas they read as two different roots under this fixture's
    // root-per-row bookkeeping — unlike every other lane's DIE pair (Luther,
    // Vulgate, LXX all keep one root at both addresses). This may reflect a
    // genuine BKR translator choice (varying the perfectivizing prefix
    // between the flat statement at 2:17 and the serpent's negated denial
    // at 3:4) or may be a curator memory error; orchestrator should verify
    // against a BKR facsimile/critical edition before trusting this
    // row pair as data (as opposed to as a flagged CHECK).
    LaneLex { lane: "cs-bkr", pair: "DIE", addr: (3, 4), surface: "nezemřete", root: "zemříti", pos: "v", morph: "ze|mříti", prag: "plain" },

    // ══════════════════════════════════════════════════════════════════
    // la-vulgate — Clementine Vulgate
    // ══════════════════════════════════════════════════════════════════

    // Gen 3:7 "Et aperti sunt oculi amborum; cumque cognovissent se esse
    //          nudos, consuerunt folia ficus, et fecerunt sibi
    //          perizomata." — [VERIFIED, anchor]
    // morph note (operator steer, prefix-meaning-switch): cognōscō is itself
    // prefixed — con- + (g)nōscō — so the decomposition is carried explicitly.
    // Same prefix at both KNOW addresses → the morph channel reads agreement.
    LaneLex { lane: "la-vulgate", pair: "KNOW", addr: (3, 7), surface: "cognovissent", root: "cognosco", pos: "v", morph: "co|gnosco", prag: "plain" },
    // Gen 4:1 "Adam vero cognovit uxorem suam Hevam, quae concepit et
    //          peperit Cain, dicens: Possedi hominem per Deum." —
    //          [VERIFIED, anchor]
    LaneLex { lane: "la-vulgate", pair: "KNOW", addr: (4, 1), surface: "cognovit", root: "cognosco", pos: "v", morph: "co|gnosco", prag: "euphemism-calque" },
    // Gen 2:25 "Erat autem uterque nudus, Adam scilicet et uxor ejus: et
    //           non erubescebant." — [VERIFIED]
    LaneLex { lane: "la-vulgate", pair: "NAKED", addr: (2, 25), surface: "nudus", root: "nudus", pos: "adj", morph: "|nudus", prag: "plain" },
    // Gen 3:7 (same clause as KNOW row above) "...cumque cognovissent se
    //          esse nudos..." — [VERIFIED]
    LaneLex { lane: "la-vulgate", pair: "NAKED", addr: (3, 7), surface: "nudos", root: "nudus", pos: "adj", morph: "|nudus", prag: "plain" },
    // Gen 2:17 "De ligno autem scientiae boni et mali ne comedas: in
    //           quocumque enim die comederis ex eo, morte morieris." —
    //           [VERIFIED: high-confidence reconstruction of a widely
    //           quoted verse, reflecting the Hebrew infinitive-absolute
    //           doubling with "morte morieris"]
    LaneLex { lane: "la-vulgate", pair: "DIE", addr: (2, 17), surface: "morieris", root: "morior", pos: "v", morph: "|morior", prag: "plain" },
    // Gen 3:4 "Dixit autem serpens ad mulierem: Nequaquam morte
    //          moriemini." — [VERIFIED: "moriemini" is 2nd person plural,
    //          matching the plural Hebrew תְּמֻתוּן despite the serpent
    //          addressing the woman alone — a well-attested crux; curator
    //          is confident in this reading]
    LaneLex { lane: "la-vulgate", pair: "DIE", addr: (3, 4), surface: "moriemini", root: "morior", pos: "v", morph: "|morior", prag: "plain" },

    // ══════════════════════════════════════════════════════════════════
    // el-lxx — Septuagint (Rahlfs)
    // ══════════════════════════════════════════════════════════════════

    // Gen 3:7 "καὶ διηνοίχθησαν οἱ ὀφθαλμοὶ τῶν δύο, καὶ ἔγνωσαν ὅτι
    //          γυμνοὶ ἦσαν..." — [VERIFIED, anchor]
    LaneLex { lane: "el-lxx", pair: "KNOW", addr: (3, 7), surface: "ἔγνωσαν", root: "γινώσκω", pos: "v", morph: "|γινώσκω", prag: "plain" },
    // Gen 4:1 "Αδαμ δὲ ἔγνω Ευαν τὴν γυναῖκα αὐτοῦ, καὶ συλλαβοῦσα ἔτεκεν
    //          τὸν Καιν..." — [VERIFIED, anchor]
    LaneLex { lane: "el-lxx", pair: "KNOW", addr: (4, 1), surface: "ἔγνω", root: "γινώσκω", pos: "v", morph: "|γινώσκω", prag: "euphemism-calque" },
    // Gen 2:25 "καὶ ἦσαν οἱ δύο γυμνοί, ὅ τε Αδαμ καὶ ἡ γυνὴ αὐτοῦ, καὶ
    //           οὐκ ᾐσχύνοντο." — [VERIFIED]
    LaneLex { lane: "el-lxx", pair: "NAKED", addr: (2, 25), surface: "γυμνοί", root: "γυμνός", pos: "adj", morph: "|γυμνός", prag: "plain" },
    // Gen 3:7 (same clause as KNOW row above) "...καὶ ἔγνωσαν ὅτι γυμνοὶ
    //          ἦσαν..." — [VERIFIED]
    LaneLex { lane: "el-lxx", pair: "NAKED", addr: (3, 7), surface: "γυμνοὶ", root: "γυμνός", pos: "adj", morph: "|γυμνός", prag: "plain" },
    // Gen 2:17 "ἀπὸ δὲ τοῦ ξύλου τοῦ γινώσκειν καλὸν καὶ πονηρόν, οὐ
    //           φάγεσθε ἀπ᾿ αὐτοῦ· ᾗ δ᾿ ἂν ἡμέρᾳ φάγητε ἀπ᾿ αὐτοῦ, θανάτῳ
    //           ἀποθανεῖσθε." — [VERIFIED: the dative-of-instrument +
    //           future-passive "θανάτῳ ἀποθανεῖσθε" is the well-known LXX
    //           rendering of the Hebrew infinitive-absolute doubling]
    LaneLex { lane: "el-lxx", pair: "DIE", addr: (2, 17), surface: "ἀποθανεῖσθε", root: "ἀποθνῄσκω", pos: "v", morph: "|ἀποθνῄσκω", prag: "plain" },
    // Gen 3:4 "καὶ εἶπεν ὁ ὄφις τῇ γυναικί Οὐ θανάτῳ ἀποθανεῖσθε." —
    //          [VERIFIED: same construction repeated for the serpent's
    //          negated denial]
    LaneLex { lane: "el-lxx", pair: "DIE", addr: (3, 4), surface: "ἀποθανεῖσθε", root: "ἀποθνῄσκω", pos: "v", morph: "|ἀποθνῄσκω", prag: "plain" },

    // ══════════════════════════════════════════════════════════════════
    // arc-onkelos — Targum Onkelos (Aramaic, Hebrew script, unpointed)
    // ══════════════════════════════════════════════════════════════════
    //
    // Every row in this lane is [CHECK]: the curator has reasonable
    // confidence in the consonantal ROOT (ידע "know", ערטל "naked", מות
    // "die" — all cognate to, and largely coextensive with, the Hebrew
    // Vorlage's own roots, which Onkelos usually mirrors closely) but is
    // NOT quoting from a critical edition (e.g. Sperber) and cannot
    // guarantee the exact unpointed SURFACE spelling (verb binyan, plural
    // marker placement, accusative "ית" positioning) letter-for-letter.
    // Orchestrator should verify against Sperber's Targum Onkelos text
    // before trusting these surfaces as data.

    // Gen 3:7 "וידעו ארי ערטילאין אנון" (roughly: "and they knew that they
    //          [were] naked") — [CHECK; root ידע matches anchor]
    LaneLex { lane: "arc-onkelos", pair: "KNOW", addr: (3, 7), surface: "וידעו", root: "ידע", pos: "v", morph: "|ידע", prag: "plain" },
    // Gen 4:1 "ואדם ידע ית חוה אתתיה" (roughly: "and Adam knew Eve his
    //          wife") — [CHECK; root ידע matches anchor]
    LaneLex { lane: "arc-onkelos", pair: "KNOW", addr: (4, 1), surface: "ידע", root: "ידע", pos: "v", morph: "|ידע", prag: "euphemism-calque" },
    // Gen 2:25 "והוו תרויהון ערטילאין אדם ואתתיה ולא מתביישין" (roughly:
    //           "and the two of them were naked, Adam and his wife, and
    //           were not ashamed") — [CHECK]
    LaneLex { lane: "arc-onkelos", pair: "NAKED", addr: (2, 25), surface: "ערטילאין", root: "ערטל", pos: "adj", morph: "|ערטל", prag: "plain" },
    // Gen 3:7 (same clause as KNOW row above) "...ארי ערטילאין אנון..." —
    //          [CHECK]
    LaneLex { lane: "arc-onkelos", pair: "NAKED", addr: (3, 7), surface: "ערטילאין", root: "ערטל", pos: "adj", morph: "|ערטל", prag: "plain" },
    // Gen 2:17 "ארי ביומא דתיכול מיניה מיתא תמות" (roughly: "for on the
    //           day you eat of it, dying you shall die" — Onkelos mirrors
    //           the Hebrew infinitive-absolute doubling with an infinitive
    //           + finite pair on the same root) — [CHECK]
    LaneLex { lane: "arc-onkelos", pair: "DIE", addr: (2, 17), surface: "תמות", root: "מות", pos: "v", morph: "|מות", prag: "plain" },
    // Gen 3:4 "לא מיתא תמותון" (roughly: "not dying shall you [pl.] die" —
    //          same doubling construction, plural verb matching the plural
    //          Hebrew תְּמֻתוּן) — [CHECK]
    LaneLex { lane: "arc-onkelos", pair: "DIE", addr: (3, 4), surface: "תמותון", root: "מות", pos: "v", morph: "|מות", prag: "plain" },
];
