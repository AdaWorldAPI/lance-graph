//! Source arm: one name in two namespaces, judged by rust-analyzer.
//!
//! The rules read lanes a scanner fills from the fixture's text: the byte
//! after the identifier, and whether the site lies in a function signature.
//! They resolve nothing. The authority is the committed pin, which is
//! rust-analyzer's own record of which symbol each `Foo` names.

use hidden_state_twins_probe::{digest, Family, Rule, TwinRefusal};

const SRC: &str = include_str!("../fixtures/ns_twins/src/lib.rs");
const PIN: &str = include_str!("../fixtures/ns_twins_foo.pin.tsv");

const CLASS_SITE: u32 = 0x7F32_0001;

// Lanes of a site row.
const NEXT: u16 = 0;
const IN_SIGNATURE: u16 = 1;

// NEXT codes.
const BRACE: u32 = 1;
const PAREN: u32 = 2;
const COMMA: u32 = 3;

// Rules, one bit each. Two name the struct, two the function.
const STRUCT_BRACE: u8 = 1 << 0;
const STRUCT_IN_SIGNATURE: u8 = 1 << 1;
const FN_PAREN: u8 = 1 << 2;
const FN_VALUE: u8 = 1 << 3;
const ANY: u8 = STRUCT_BRACE | STRUCT_IN_SIGNATURE | FN_PAREN | FN_VALUE;
const STRUCT: u8 = STRUCT_BRACE | STRUCT_IN_SIGNATURE;
const FUNCTION: u8 = FN_PAREN | FN_VALUE;

/// Byte offsets of the identifier `Foo` outside comment lines.
fn sites() -> Vec<usize> {
    let bytes = SRC.as_bytes();
    let ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
    let mut out = Vec::new();
    let mut line_start = 0;
    for line in SRC.split_inclusive('\n') {
        if !line.trim_start().starts_with("//") {
            for (i, _) in line.match_indices("Foo") {
                let at = line_start + i;
                let before = at > 0 && ident(bytes[at - 1]);
                let after = bytes.get(at + 3).is_some_and(|&b| ident(b));
                if !before && !after {
                    out.push(at);
                }
            }
        }
        line_start += line.len();
    }
    out
}

/// The producer. `NEXT` is one byte of the row's own text. `IN_SIGNATURE`
/// is not: it says whether the site lies between a `fn` and the `{` that
/// opens its body, which is read from the rest of the line.
fn site_lanes(sites: &[usize]) -> Vec<Vec<u32>> {
    let bytes = SRC.as_bytes();
    let mut lanes = vec![Vec::new(); 2];
    for &at in sites {
        let next = bytes[at + 3..]
            .iter()
            .copied()
            .find(|b| *b != b' ')
            .unwrap_or(0);
        lanes[usize::from(NEXT)].push(match next {
            b'{' => BRACE,
            b'(' => PAREN,
            b',' => COMMA,
            _ => 0,
        });
        let line_start = SRC[..at].rfind('\n').map_or(0, |i| i + 1);
        let head = &SRC[line_start..at];
        let in_signature = head.contains("fn ") && !head.contains('{');
        lanes[usize::from(IN_SIGNATURE)].push(u32::from(in_signature));
    }
    lanes
}

fn site_family(with_signature: bool) -> Family {
    let comma = |sig: u32| {
        let mut when = vec![(NEXT, vec![COMMA])];
        if with_signature {
            when.push((IN_SIGNATURE, vec![sig]));
        }
        when
    };
    Family {
        class: CLASS_SITE,
        lanes: 2,
        rules: vec![
            Rule {
                name: "StructBrace",
                when: vec![(NEXT, vec![BRACE])],
            },
            Rule {
                name: "StructInSignature",
                when: comma(1),
            },
            Rule {
                name: "FnParen",
                when: vec![(NEXT, vec![PAREN])],
            },
            Rule {
                name: "FnValue",
                when: comma(0),
            },
        ],
        producer: digest(&["ns_twins", "text-scan-v1"]),
    }
}

/// rust-analyzer's record: byte offset -> which symbol.
fn authority() -> Vec<(usize, u8)> {
    let mut out: Vec<(usize, u8)> = PIN
        .lines()
        .filter(|l| !l.starts_with('#') && !l.is_empty())
        .map(|l| {
            let c: Vec<&str> = l.split('\t').collect();
            let symbol = match c[0] {
                "ns_twins::Foo#" => STRUCT,
                "ns_twins::Foo()." => FUNCTION,
                other => panic!("unexpected moniker {other}"),
            };
            (c[3].parse().expect("byte offset"), symbol)
        })
        .collect();
    out.sort_unstable();
    out
}

#[test]
fn namespace_twins_need_the_position_as_a_lane() {
    let sites = sites();
    let pin = authority();
    // The scanner and rust-analyzer name the same six sites.
    assert_eq!(sites, pin.iter().map(|p| p.0).collect::<Vec<_>>());
    assert_eq!(pin.iter().filter(|p| p.1 == STRUCT).count(), 3);
    assert_eq!(pin.iter().filter(|p| p.1 == FUNCTION).count(), 3);

    let lanes = site_lanes(&sites);
    // Every site has the same left corner, so the selector proposes all four.
    let candidates = vec![ANY; sites.len()];

    // The twins: both read `(Foo,`. One is the struct, the other the function.
    let twins: Vec<usize> = (0..sites.len())
        .filter(|&r| lanes[usize::from(NEXT)][r] == COMMA)
        .collect();
    assert_eq!(twins.len(), 2);
    for &r in &twins {
        assert_eq!(&SRC[sites[r] - 1..sites[r] + 4], "(Foo,");
    }
    assert_ne!(pin[twins[0]].1, pin[twins[1]].1);

    let family = site_family(true);
    let got = family
        .bind()
        .unwrap()
        .decide(&family, &lanes, &candidates)
        .unwrap();
    assert_eq!(got.composed, 4);
    for (r, &(at, symbol)) in pin.iter().enumerate() {
        let holds = got.holds[r];
        assert_eq!(holds.count_ones(), 1, "byte {at}: exactly one rule holds");
        assert_eq!(
            holds & symbol,
            holds,
            "byte {at}: the rule names rust-analyzer's symbol"
        );
    }

    // RED control: drop the position from the guards. The twins collapse and
    // each is given both symbols.
    let ablated = site_family(false);
    let wrong = ablated
        .bind()
        .unwrap()
        .decide(&ablated, &lanes, &candidates)
        .unwrap();
    for &r in &twins {
        assert_eq!(wrong.holds[r], STRUCT_IN_SIGNATURE | FN_VALUE);
        assert_ne!(
            wrong.holds[r] & !pin[r].1,
            0,
            "the ablation must name the wrong symbol"
        );
    }
    assert_eq!(
        family.bind().unwrap().decide(&ablated, &lanes, &candidates),
        Err(TwinRefusal::DescriptionMismatch)
    );

    // A narrower selector loses answers and composes fewer rules.
    let narrow = vec![STRUCT; sites.len()];
    let fewer = family
        .bind()
        .unwrap()
        .decide(&family, &lanes, &narrow)
        .unwrap();
    assert_eq!(fewer.composed, 2);
    assert_eq!(fewer.holds.iter().filter(|h| **h != 0).count(), 3);
}
