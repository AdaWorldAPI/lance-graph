//! Makor citations (`rule -> Shulchan Arukh Yoreh De'ah siman:seif`) addressed
//! the way the KJV corpus is: a [`NiblePath`] from the outline, via
//! `deepnsm_v2::hydrate`. No classid, no `NodeRow`, no key minting.
//!
//! Mapping: YD is one book, siman = chapter, seif = verse. A citation is the
//! triple `(rule, CITES, seif)` laid at its seif's verse node. A siman or
//! seif beyond the two-nibble level capacity (255) has no node and is counted
//! in `Hydration::unaddressed`, never folded onto another node.
//!
//! The book is always book 0: `CorpusToc::from_markers` opens a new book only
//! at marker `(1, 1)` (or implicitly for the first marker), and YD citations
//! never contain 1:1.
//!
//! Names exist only in [`parse_makor`] (the membrane); [`MakorHydration`]
//! holds ordinals and paths.

use std::collections::HashMap;

use deepnsm_v2::hydrate::{Hydration, hydrate};
use deepnsm_v2::spo::Spo;
use deepnsm_v2::toc::TocLevel;
use deepnsm_v2::vocab::WordId;
use lance_graph_contract::hhtl::NiblePath;

use crate::Refusal;

/// Predicate: "rule cites seif".
pub const CITES: WordId = 1;

/// Basin nibble of the makor tree. A lab choice; any nibble works.
pub const MAKOR_BASIN: u8 = 0x0D;

/// Citations laid over the outline. Holds no names.
#[derive(Debug, Clone)]
pub struct MakorHydration {
    pub hydration: Hydration,
    /// Distinct rules (ordinals `0..rules`).
    pub rules: usize,
}

/// Parse `rule \t siman \t seif` lines; `#` lines and blank lines are skipped.
/// More than `u16::MAX` distinct rules are refused.
pub fn parse_makor(text: &str) -> Result<Vec<(String, u16, u16)>, Refusal> {
    let mut out = Vec::new();
    let mut names: std::collections::HashSet<&str> = std::collections::HashSet::new();
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let bad = || Refusal::Format(line.to_owned());
        let mut c = line.split('\t');
        let (Some(rule), Some(s), Some(f), None) = (c.next(), c.next(), c.next(), c.next()) else {
            return Err(bad());
        };
        if rule.is_empty() {
            return Err(bad());
        }
        let s: u16 = s.parse().map_err(|_| bad())?;
        let f: u16 = f.parse().map_err(|_| bad())?;
        names.insert(rule);
        if names.len() > usize::from(u16::MAX) + 1 {
            return Err(bad());
        }
        out.push((rule.to_owned(), s, f));
    }
    Ok(out)
}

/// Lay the citations over the outline of their seifim (distinct, in input
/// order). Rule ordinals are first-appearance order. More distinct rules or
/// seifim than a `WordId` holds are refused, never saturated.
pub fn hydrate_makor(rows: &[(String, u16, u16)]) -> Result<MakorHydration, Refusal> {
    let mut seif_ix: HashMap<(u16, u16), usize> = HashMap::new();
    let mut markers: Vec<(u16, u16)> = Vec::new();
    let mut rule_ix: HashMap<&str, u16> = HashMap::new();
    let mut per_verse: Vec<Vec<Spo>> = Vec::new();
    for (rule, s, f) in rows {
        let i = *seif_ix.entry((*s, *f)).or_insert_with(|| {
            markers.push((*s, *f));
            per_verse.push(Vec::new());
            markers.len() - 1
        });
        let r = match rule_ix.get(rule.as_str()) {
            Some(&r) => r,
            None => {
                let r = u16::try_from(rule_ix.len()).map_err(|_| Refusal::TooMany("rule"))?;
                rule_ix.insert(rule.as_str(), r);
                r
            }
        };
        let seif_ordinal = u16::try_from(i + 1).map_err(|_| Refusal::TooMany("seif"))?;
        per_verse[i].push(Spo::new(r, CITES, seif_ordinal));
    }
    Ok(MakorHydration {
        hydration: hydrate(&markers, &per_verse, MAKOR_BASIN),
        rules: rule_ix.len(),
    })
}

/// The verse node of `siman:seif`, if the outline spawned one.
pub fn path_of(h: &MakorHydration, siman: u16, seif: u16) -> Option<NiblePath> {
    h.hydration
        .entries
        .iter()
        .find(|e| {
            e.level == TocLevel::Verse && e.book == 0 && e.chapter == siman && e.verse == seif
        })
        .map(|e| e.path)
}
