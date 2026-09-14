//! The generated 256-arm ternlog dispatch — a runtime `imm: u8` routed to a
//! const-generic `ndarray::simd::mask_ternlog::<IMM>` word.
//!
//! [`ir::MaskOp::Ternlog`](crate::ir::MaskOp::Ternlog) carries its immediate as
//! a plain `u8`: it comes from a fused truth table computed at plan-build time
//! (`fuse::fuse`) or from a caller who already knows the VPTERNLOG index they
//! want. `ndarray::simd::mask_ternlog`/`mask_ternlog_assign`, though, take the
//! immediate as a **const generic** (`mask_ternlog<const IMM: i32>`) — the
//! same shape every other realization in that facade uses to let a backend
//! pick its VPTERNLOGQ encoding at compile time rather than branch on it at
//! run time. A const generic can only ever be instantiated with a literal, so
//! there is exactly one way to route a byte that is not known until run time
//! into it: enumerate all 256 literals in a `match` and let the byte select
//! the arm. That is all [`ternlog_dispatch`] and [`ternlog_dispatch_assign`]
//! are — this crate's ONE fan-out from "immediate, at run time" to "immediate,
//! at compile time", named once so the executor never needs a second copy of
//! either 256-arm table.
//!
//! ## Why the body is generated, not hand-typed
//!
//! 256 near-identical arms, twice, is exactly the shape a human miscounts —
//! drops an arm, duplicates one, or gets the order wrong in a way a reviewer
//! skims past. [`tools/gen_ternlog_dispatch.py`](../../tools/gen_ternlog_dispatch.py)
//! (relative to this crate's root) emits the region between the
//! `GEN-TERNLOG-DISPATCH-BEGIN`/`END` markers below deterministically from
//! `range(256)`, so the only way an arm goes missing is the generator itself
//! being wrong — one place to look, not 256.
//!
//! ## Regenerating
//!
//! Run from the repository root (or any directory — paths resolve relative to
//! the script's own location):
//!
//! ```text
//! python3 crates/lance-graph-mask-risc/tools/gen_ternlog_dispatch.py
//! ```
//!
//! The script only ever rewrites the region between the two markers; the
//! module doc, the `use` line, and the `#[cfg(test)]` module below are
//! hand-written and are never touched by a regenerate.
//!
//! To check the committed file is current without writing to it (the
//! invocation CI's regenerate-and-diff gate runs):
//!
//! ```text
//! python3 crates/lance-graph-mask-risc/tools/gen_ternlog_dispatch.py --check
//! ```
//!
//! `--check` exits non-zero the moment the committed region would differ from
//! a fresh run (a hand edit inside the markers, or the generator itself
//! changing what it emits without the file being regenerated) and exits 0
//! when the committed file already matches. CI runs this on every PR that
//! touches either the generator or this file, so a stale table fails the Rust Tests workflow.

use ndarray::simd::{mask_ternlog, mask_ternlog_assign};
// GEN-TERNLOG-DISPATCH-BEGIN
/// `dst = table[imm](a, b, c)` — the runtime immediate routed to the const-generic facade word.
pub fn ternlog_dispatch(imm: u8, a: &[u64], b: &[u64], c: &[u64], dst: &mut [u64]) {
    match imm {
        0 => mask_ternlog::<0>(a, b, c, dst),
        1 => mask_ternlog::<1>(a, b, c, dst),
        2 => mask_ternlog::<2>(a, b, c, dst),
        3 => mask_ternlog::<3>(a, b, c, dst),
        4 => mask_ternlog::<4>(a, b, c, dst),
        5 => mask_ternlog::<5>(a, b, c, dst),
        6 => mask_ternlog::<6>(a, b, c, dst),
        7 => mask_ternlog::<7>(a, b, c, dst),
        8 => mask_ternlog::<8>(a, b, c, dst),
        9 => mask_ternlog::<9>(a, b, c, dst),
        10 => mask_ternlog::<10>(a, b, c, dst),
        11 => mask_ternlog::<11>(a, b, c, dst),
        12 => mask_ternlog::<12>(a, b, c, dst),
        13 => mask_ternlog::<13>(a, b, c, dst),
        14 => mask_ternlog::<14>(a, b, c, dst),
        15 => mask_ternlog::<15>(a, b, c, dst),
        16 => mask_ternlog::<16>(a, b, c, dst),
        17 => mask_ternlog::<17>(a, b, c, dst),
        18 => mask_ternlog::<18>(a, b, c, dst),
        19 => mask_ternlog::<19>(a, b, c, dst),
        20 => mask_ternlog::<20>(a, b, c, dst),
        21 => mask_ternlog::<21>(a, b, c, dst),
        22 => mask_ternlog::<22>(a, b, c, dst),
        23 => mask_ternlog::<23>(a, b, c, dst),
        24 => mask_ternlog::<24>(a, b, c, dst),
        25 => mask_ternlog::<25>(a, b, c, dst),
        26 => mask_ternlog::<26>(a, b, c, dst),
        27 => mask_ternlog::<27>(a, b, c, dst),
        28 => mask_ternlog::<28>(a, b, c, dst),
        29 => mask_ternlog::<29>(a, b, c, dst),
        30 => mask_ternlog::<30>(a, b, c, dst),
        31 => mask_ternlog::<31>(a, b, c, dst),
        32 => mask_ternlog::<32>(a, b, c, dst),
        33 => mask_ternlog::<33>(a, b, c, dst),
        34 => mask_ternlog::<34>(a, b, c, dst),
        35 => mask_ternlog::<35>(a, b, c, dst),
        36 => mask_ternlog::<36>(a, b, c, dst),
        37 => mask_ternlog::<37>(a, b, c, dst),
        38 => mask_ternlog::<38>(a, b, c, dst),
        39 => mask_ternlog::<39>(a, b, c, dst),
        40 => mask_ternlog::<40>(a, b, c, dst),
        41 => mask_ternlog::<41>(a, b, c, dst),
        42 => mask_ternlog::<42>(a, b, c, dst),
        43 => mask_ternlog::<43>(a, b, c, dst),
        44 => mask_ternlog::<44>(a, b, c, dst),
        45 => mask_ternlog::<45>(a, b, c, dst),
        46 => mask_ternlog::<46>(a, b, c, dst),
        47 => mask_ternlog::<47>(a, b, c, dst),
        48 => mask_ternlog::<48>(a, b, c, dst),
        49 => mask_ternlog::<49>(a, b, c, dst),
        50 => mask_ternlog::<50>(a, b, c, dst),
        51 => mask_ternlog::<51>(a, b, c, dst),
        52 => mask_ternlog::<52>(a, b, c, dst),
        53 => mask_ternlog::<53>(a, b, c, dst),
        54 => mask_ternlog::<54>(a, b, c, dst),
        55 => mask_ternlog::<55>(a, b, c, dst),
        56 => mask_ternlog::<56>(a, b, c, dst),
        57 => mask_ternlog::<57>(a, b, c, dst),
        58 => mask_ternlog::<58>(a, b, c, dst),
        59 => mask_ternlog::<59>(a, b, c, dst),
        60 => mask_ternlog::<60>(a, b, c, dst),
        61 => mask_ternlog::<61>(a, b, c, dst),
        62 => mask_ternlog::<62>(a, b, c, dst),
        63 => mask_ternlog::<63>(a, b, c, dst),
        64 => mask_ternlog::<64>(a, b, c, dst),
        65 => mask_ternlog::<65>(a, b, c, dst),
        66 => mask_ternlog::<66>(a, b, c, dst),
        67 => mask_ternlog::<67>(a, b, c, dst),
        68 => mask_ternlog::<68>(a, b, c, dst),
        69 => mask_ternlog::<69>(a, b, c, dst),
        70 => mask_ternlog::<70>(a, b, c, dst),
        71 => mask_ternlog::<71>(a, b, c, dst),
        72 => mask_ternlog::<72>(a, b, c, dst),
        73 => mask_ternlog::<73>(a, b, c, dst),
        74 => mask_ternlog::<74>(a, b, c, dst),
        75 => mask_ternlog::<75>(a, b, c, dst),
        76 => mask_ternlog::<76>(a, b, c, dst),
        77 => mask_ternlog::<77>(a, b, c, dst),
        78 => mask_ternlog::<78>(a, b, c, dst),
        79 => mask_ternlog::<79>(a, b, c, dst),
        80 => mask_ternlog::<80>(a, b, c, dst),
        81 => mask_ternlog::<81>(a, b, c, dst),
        82 => mask_ternlog::<82>(a, b, c, dst),
        83 => mask_ternlog::<83>(a, b, c, dst),
        84 => mask_ternlog::<84>(a, b, c, dst),
        85 => mask_ternlog::<85>(a, b, c, dst),
        86 => mask_ternlog::<86>(a, b, c, dst),
        87 => mask_ternlog::<87>(a, b, c, dst),
        88 => mask_ternlog::<88>(a, b, c, dst),
        89 => mask_ternlog::<89>(a, b, c, dst),
        90 => mask_ternlog::<90>(a, b, c, dst),
        91 => mask_ternlog::<91>(a, b, c, dst),
        92 => mask_ternlog::<92>(a, b, c, dst),
        93 => mask_ternlog::<93>(a, b, c, dst),
        94 => mask_ternlog::<94>(a, b, c, dst),
        95 => mask_ternlog::<95>(a, b, c, dst),
        96 => mask_ternlog::<96>(a, b, c, dst),
        97 => mask_ternlog::<97>(a, b, c, dst),
        98 => mask_ternlog::<98>(a, b, c, dst),
        99 => mask_ternlog::<99>(a, b, c, dst),
        100 => mask_ternlog::<100>(a, b, c, dst),
        101 => mask_ternlog::<101>(a, b, c, dst),
        102 => mask_ternlog::<102>(a, b, c, dst),
        103 => mask_ternlog::<103>(a, b, c, dst),
        104 => mask_ternlog::<104>(a, b, c, dst),
        105 => mask_ternlog::<105>(a, b, c, dst),
        106 => mask_ternlog::<106>(a, b, c, dst),
        107 => mask_ternlog::<107>(a, b, c, dst),
        108 => mask_ternlog::<108>(a, b, c, dst),
        109 => mask_ternlog::<109>(a, b, c, dst),
        110 => mask_ternlog::<110>(a, b, c, dst),
        111 => mask_ternlog::<111>(a, b, c, dst),
        112 => mask_ternlog::<112>(a, b, c, dst),
        113 => mask_ternlog::<113>(a, b, c, dst),
        114 => mask_ternlog::<114>(a, b, c, dst),
        115 => mask_ternlog::<115>(a, b, c, dst),
        116 => mask_ternlog::<116>(a, b, c, dst),
        117 => mask_ternlog::<117>(a, b, c, dst),
        118 => mask_ternlog::<118>(a, b, c, dst),
        119 => mask_ternlog::<119>(a, b, c, dst),
        120 => mask_ternlog::<120>(a, b, c, dst),
        121 => mask_ternlog::<121>(a, b, c, dst),
        122 => mask_ternlog::<122>(a, b, c, dst),
        123 => mask_ternlog::<123>(a, b, c, dst),
        124 => mask_ternlog::<124>(a, b, c, dst),
        125 => mask_ternlog::<125>(a, b, c, dst),
        126 => mask_ternlog::<126>(a, b, c, dst),
        127 => mask_ternlog::<127>(a, b, c, dst),
        128 => mask_ternlog::<128>(a, b, c, dst),
        129 => mask_ternlog::<129>(a, b, c, dst),
        130 => mask_ternlog::<130>(a, b, c, dst),
        131 => mask_ternlog::<131>(a, b, c, dst),
        132 => mask_ternlog::<132>(a, b, c, dst),
        133 => mask_ternlog::<133>(a, b, c, dst),
        134 => mask_ternlog::<134>(a, b, c, dst),
        135 => mask_ternlog::<135>(a, b, c, dst),
        136 => mask_ternlog::<136>(a, b, c, dst),
        137 => mask_ternlog::<137>(a, b, c, dst),
        138 => mask_ternlog::<138>(a, b, c, dst),
        139 => mask_ternlog::<139>(a, b, c, dst),
        140 => mask_ternlog::<140>(a, b, c, dst),
        141 => mask_ternlog::<141>(a, b, c, dst),
        142 => mask_ternlog::<142>(a, b, c, dst),
        143 => mask_ternlog::<143>(a, b, c, dst),
        144 => mask_ternlog::<144>(a, b, c, dst),
        145 => mask_ternlog::<145>(a, b, c, dst),
        146 => mask_ternlog::<146>(a, b, c, dst),
        147 => mask_ternlog::<147>(a, b, c, dst),
        148 => mask_ternlog::<148>(a, b, c, dst),
        149 => mask_ternlog::<149>(a, b, c, dst),
        150 => mask_ternlog::<150>(a, b, c, dst),
        151 => mask_ternlog::<151>(a, b, c, dst),
        152 => mask_ternlog::<152>(a, b, c, dst),
        153 => mask_ternlog::<153>(a, b, c, dst),
        154 => mask_ternlog::<154>(a, b, c, dst),
        155 => mask_ternlog::<155>(a, b, c, dst),
        156 => mask_ternlog::<156>(a, b, c, dst),
        157 => mask_ternlog::<157>(a, b, c, dst),
        158 => mask_ternlog::<158>(a, b, c, dst),
        159 => mask_ternlog::<159>(a, b, c, dst),
        160 => mask_ternlog::<160>(a, b, c, dst),
        161 => mask_ternlog::<161>(a, b, c, dst),
        162 => mask_ternlog::<162>(a, b, c, dst),
        163 => mask_ternlog::<163>(a, b, c, dst),
        164 => mask_ternlog::<164>(a, b, c, dst),
        165 => mask_ternlog::<165>(a, b, c, dst),
        166 => mask_ternlog::<166>(a, b, c, dst),
        167 => mask_ternlog::<167>(a, b, c, dst),
        168 => mask_ternlog::<168>(a, b, c, dst),
        169 => mask_ternlog::<169>(a, b, c, dst),
        170 => mask_ternlog::<170>(a, b, c, dst),
        171 => mask_ternlog::<171>(a, b, c, dst),
        172 => mask_ternlog::<172>(a, b, c, dst),
        173 => mask_ternlog::<173>(a, b, c, dst),
        174 => mask_ternlog::<174>(a, b, c, dst),
        175 => mask_ternlog::<175>(a, b, c, dst),
        176 => mask_ternlog::<176>(a, b, c, dst),
        177 => mask_ternlog::<177>(a, b, c, dst),
        178 => mask_ternlog::<178>(a, b, c, dst),
        179 => mask_ternlog::<179>(a, b, c, dst),
        180 => mask_ternlog::<180>(a, b, c, dst),
        181 => mask_ternlog::<181>(a, b, c, dst),
        182 => mask_ternlog::<182>(a, b, c, dst),
        183 => mask_ternlog::<183>(a, b, c, dst),
        184 => mask_ternlog::<184>(a, b, c, dst),
        185 => mask_ternlog::<185>(a, b, c, dst),
        186 => mask_ternlog::<186>(a, b, c, dst),
        187 => mask_ternlog::<187>(a, b, c, dst),
        188 => mask_ternlog::<188>(a, b, c, dst),
        189 => mask_ternlog::<189>(a, b, c, dst),
        190 => mask_ternlog::<190>(a, b, c, dst),
        191 => mask_ternlog::<191>(a, b, c, dst),
        192 => mask_ternlog::<192>(a, b, c, dst),
        193 => mask_ternlog::<193>(a, b, c, dst),
        194 => mask_ternlog::<194>(a, b, c, dst),
        195 => mask_ternlog::<195>(a, b, c, dst),
        196 => mask_ternlog::<196>(a, b, c, dst),
        197 => mask_ternlog::<197>(a, b, c, dst),
        198 => mask_ternlog::<198>(a, b, c, dst),
        199 => mask_ternlog::<199>(a, b, c, dst),
        200 => mask_ternlog::<200>(a, b, c, dst),
        201 => mask_ternlog::<201>(a, b, c, dst),
        202 => mask_ternlog::<202>(a, b, c, dst),
        203 => mask_ternlog::<203>(a, b, c, dst),
        204 => mask_ternlog::<204>(a, b, c, dst),
        205 => mask_ternlog::<205>(a, b, c, dst),
        206 => mask_ternlog::<206>(a, b, c, dst),
        207 => mask_ternlog::<207>(a, b, c, dst),
        208 => mask_ternlog::<208>(a, b, c, dst),
        209 => mask_ternlog::<209>(a, b, c, dst),
        210 => mask_ternlog::<210>(a, b, c, dst),
        211 => mask_ternlog::<211>(a, b, c, dst),
        212 => mask_ternlog::<212>(a, b, c, dst),
        213 => mask_ternlog::<213>(a, b, c, dst),
        214 => mask_ternlog::<214>(a, b, c, dst),
        215 => mask_ternlog::<215>(a, b, c, dst),
        216 => mask_ternlog::<216>(a, b, c, dst),
        217 => mask_ternlog::<217>(a, b, c, dst),
        218 => mask_ternlog::<218>(a, b, c, dst),
        219 => mask_ternlog::<219>(a, b, c, dst),
        220 => mask_ternlog::<220>(a, b, c, dst),
        221 => mask_ternlog::<221>(a, b, c, dst),
        222 => mask_ternlog::<222>(a, b, c, dst),
        223 => mask_ternlog::<223>(a, b, c, dst),
        224 => mask_ternlog::<224>(a, b, c, dst),
        225 => mask_ternlog::<225>(a, b, c, dst),
        226 => mask_ternlog::<226>(a, b, c, dst),
        227 => mask_ternlog::<227>(a, b, c, dst),
        228 => mask_ternlog::<228>(a, b, c, dst),
        229 => mask_ternlog::<229>(a, b, c, dst),
        230 => mask_ternlog::<230>(a, b, c, dst),
        231 => mask_ternlog::<231>(a, b, c, dst),
        232 => mask_ternlog::<232>(a, b, c, dst),
        233 => mask_ternlog::<233>(a, b, c, dst),
        234 => mask_ternlog::<234>(a, b, c, dst),
        235 => mask_ternlog::<235>(a, b, c, dst),
        236 => mask_ternlog::<236>(a, b, c, dst),
        237 => mask_ternlog::<237>(a, b, c, dst),
        238 => mask_ternlog::<238>(a, b, c, dst),
        239 => mask_ternlog::<239>(a, b, c, dst),
        240 => mask_ternlog::<240>(a, b, c, dst),
        241 => mask_ternlog::<241>(a, b, c, dst),
        242 => mask_ternlog::<242>(a, b, c, dst),
        243 => mask_ternlog::<243>(a, b, c, dst),
        244 => mask_ternlog::<244>(a, b, c, dst),
        245 => mask_ternlog::<245>(a, b, c, dst),
        246 => mask_ternlog::<246>(a, b, c, dst),
        247 => mask_ternlog::<247>(a, b, c, dst),
        248 => mask_ternlog::<248>(a, b, c, dst),
        249 => mask_ternlog::<249>(a, b, c, dst),
        250 => mask_ternlog::<250>(a, b, c, dst),
        251 => mask_ternlog::<251>(a, b, c, dst),
        252 => mask_ternlog::<252>(a, b, c, dst),
        253 => mask_ternlog::<253>(a, b, c, dst),
        254 => mask_ternlog::<254>(a, b, c, dst),
        255 => mask_ternlog::<255>(a, b, c, dst),
    }
}

/// `a = table[imm](a, b, c)` in place — the aliasing form the executor uses when `dst` is `a`.
pub fn ternlog_dispatch_assign(imm: u8, a: &mut [u64], b: &[u64], c: &[u64]) {
    match imm {
        0 => mask_ternlog_assign::<0>(a, b, c),
        1 => mask_ternlog_assign::<1>(a, b, c),
        2 => mask_ternlog_assign::<2>(a, b, c),
        3 => mask_ternlog_assign::<3>(a, b, c),
        4 => mask_ternlog_assign::<4>(a, b, c),
        5 => mask_ternlog_assign::<5>(a, b, c),
        6 => mask_ternlog_assign::<6>(a, b, c),
        7 => mask_ternlog_assign::<7>(a, b, c),
        8 => mask_ternlog_assign::<8>(a, b, c),
        9 => mask_ternlog_assign::<9>(a, b, c),
        10 => mask_ternlog_assign::<10>(a, b, c),
        11 => mask_ternlog_assign::<11>(a, b, c),
        12 => mask_ternlog_assign::<12>(a, b, c),
        13 => mask_ternlog_assign::<13>(a, b, c),
        14 => mask_ternlog_assign::<14>(a, b, c),
        15 => mask_ternlog_assign::<15>(a, b, c),
        16 => mask_ternlog_assign::<16>(a, b, c),
        17 => mask_ternlog_assign::<17>(a, b, c),
        18 => mask_ternlog_assign::<18>(a, b, c),
        19 => mask_ternlog_assign::<19>(a, b, c),
        20 => mask_ternlog_assign::<20>(a, b, c),
        21 => mask_ternlog_assign::<21>(a, b, c),
        22 => mask_ternlog_assign::<22>(a, b, c),
        23 => mask_ternlog_assign::<23>(a, b, c),
        24 => mask_ternlog_assign::<24>(a, b, c),
        25 => mask_ternlog_assign::<25>(a, b, c),
        26 => mask_ternlog_assign::<26>(a, b, c),
        27 => mask_ternlog_assign::<27>(a, b, c),
        28 => mask_ternlog_assign::<28>(a, b, c),
        29 => mask_ternlog_assign::<29>(a, b, c),
        30 => mask_ternlog_assign::<30>(a, b, c),
        31 => mask_ternlog_assign::<31>(a, b, c),
        32 => mask_ternlog_assign::<32>(a, b, c),
        33 => mask_ternlog_assign::<33>(a, b, c),
        34 => mask_ternlog_assign::<34>(a, b, c),
        35 => mask_ternlog_assign::<35>(a, b, c),
        36 => mask_ternlog_assign::<36>(a, b, c),
        37 => mask_ternlog_assign::<37>(a, b, c),
        38 => mask_ternlog_assign::<38>(a, b, c),
        39 => mask_ternlog_assign::<39>(a, b, c),
        40 => mask_ternlog_assign::<40>(a, b, c),
        41 => mask_ternlog_assign::<41>(a, b, c),
        42 => mask_ternlog_assign::<42>(a, b, c),
        43 => mask_ternlog_assign::<43>(a, b, c),
        44 => mask_ternlog_assign::<44>(a, b, c),
        45 => mask_ternlog_assign::<45>(a, b, c),
        46 => mask_ternlog_assign::<46>(a, b, c),
        47 => mask_ternlog_assign::<47>(a, b, c),
        48 => mask_ternlog_assign::<48>(a, b, c),
        49 => mask_ternlog_assign::<49>(a, b, c),
        50 => mask_ternlog_assign::<50>(a, b, c),
        51 => mask_ternlog_assign::<51>(a, b, c),
        52 => mask_ternlog_assign::<52>(a, b, c),
        53 => mask_ternlog_assign::<53>(a, b, c),
        54 => mask_ternlog_assign::<54>(a, b, c),
        55 => mask_ternlog_assign::<55>(a, b, c),
        56 => mask_ternlog_assign::<56>(a, b, c),
        57 => mask_ternlog_assign::<57>(a, b, c),
        58 => mask_ternlog_assign::<58>(a, b, c),
        59 => mask_ternlog_assign::<59>(a, b, c),
        60 => mask_ternlog_assign::<60>(a, b, c),
        61 => mask_ternlog_assign::<61>(a, b, c),
        62 => mask_ternlog_assign::<62>(a, b, c),
        63 => mask_ternlog_assign::<63>(a, b, c),
        64 => mask_ternlog_assign::<64>(a, b, c),
        65 => mask_ternlog_assign::<65>(a, b, c),
        66 => mask_ternlog_assign::<66>(a, b, c),
        67 => mask_ternlog_assign::<67>(a, b, c),
        68 => mask_ternlog_assign::<68>(a, b, c),
        69 => mask_ternlog_assign::<69>(a, b, c),
        70 => mask_ternlog_assign::<70>(a, b, c),
        71 => mask_ternlog_assign::<71>(a, b, c),
        72 => mask_ternlog_assign::<72>(a, b, c),
        73 => mask_ternlog_assign::<73>(a, b, c),
        74 => mask_ternlog_assign::<74>(a, b, c),
        75 => mask_ternlog_assign::<75>(a, b, c),
        76 => mask_ternlog_assign::<76>(a, b, c),
        77 => mask_ternlog_assign::<77>(a, b, c),
        78 => mask_ternlog_assign::<78>(a, b, c),
        79 => mask_ternlog_assign::<79>(a, b, c),
        80 => mask_ternlog_assign::<80>(a, b, c),
        81 => mask_ternlog_assign::<81>(a, b, c),
        82 => mask_ternlog_assign::<82>(a, b, c),
        83 => mask_ternlog_assign::<83>(a, b, c),
        84 => mask_ternlog_assign::<84>(a, b, c),
        85 => mask_ternlog_assign::<85>(a, b, c),
        86 => mask_ternlog_assign::<86>(a, b, c),
        87 => mask_ternlog_assign::<87>(a, b, c),
        88 => mask_ternlog_assign::<88>(a, b, c),
        89 => mask_ternlog_assign::<89>(a, b, c),
        90 => mask_ternlog_assign::<90>(a, b, c),
        91 => mask_ternlog_assign::<91>(a, b, c),
        92 => mask_ternlog_assign::<92>(a, b, c),
        93 => mask_ternlog_assign::<93>(a, b, c),
        94 => mask_ternlog_assign::<94>(a, b, c),
        95 => mask_ternlog_assign::<95>(a, b, c),
        96 => mask_ternlog_assign::<96>(a, b, c),
        97 => mask_ternlog_assign::<97>(a, b, c),
        98 => mask_ternlog_assign::<98>(a, b, c),
        99 => mask_ternlog_assign::<99>(a, b, c),
        100 => mask_ternlog_assign::<100>(a, b, c),
        101 => mask_ternlog_assign::<101>(a, b, c),
        102 => mask_ternlog_assign::<102>(a, b, c),
        103 => mask_ternlog_assign::<103>(a, b, c),
        104 => mask_ternlog_assign::<104>(a, b, c),
        105 => mask_ternlog_assign::<105>(a, b, c),
        106 => mask_ternlog_assign::<106>(a, b, c),
        107 => mask_ternlog_assign::<107>(a, b, c),
        108 => mask_ternlog_assign::<108>(a, b, c),
        109 => mask_ternlog_assign::<109>(a, b, c),
        110 => mask_ternlog_assign::<110>(a, b, c),
        111 => mask_ternlog_assign::<111>(a, b, c),
        112 => mask_ternlog_assign::<112>(a, b, c),
        113 => mask_ternlog_assign::<113>(a, b, c),
        114 => mask_ternlog_assign::<114>(a, b, c),
        115 => mask_ternlog_assign::<115>(a, b, c),
        116 => mask_ternlog_assign::<116>(a, b, c),
        117 => mask_ternlog_assign::<117>(a, b, c),
        118 => mask_ternlog_assign::<118>(a, b, c),
        119 => mask_ternlog_assign::<119>(a, b, c),
        120 => mask_ternlog_assign::<120>(a, b, c),
        121 => mask_ternlog_assign::<121>(a, b, c),
        122 => mask_ternlog_assign::<122>(a, b, c),
        123 => mask_ternlog_assign::<123>(a, b, c),
        124 => mask_ternlog_assign::<124>(a, b, c),
        125 => mask_ternlog_assign::<125>(a, b, c),
        126 => mask_ternlog_assign::<126>(a, b, c),
        127 => mask_ternlog_assign::<127>(a, b, c),
        128 => mask_ternlog_assign::<128>(a, b, c),
        129 => mask_ternlog_assign::<129>(a, b, c),
        130 => mask_ternlog_assign::<130>(a, b, c),
        131 => mask_ternlog_assign::<131>(a, b, c),
        132 => mask_ternlog_assign::<132>(a, b, c),
        133 => mask_ternlog_assign::<133>(a, b, c),
        134 => mask_ternlog_assign::<134>(a, b, c),
        135 => mask_ternlog_assign::<135>(a, b, c),
        136 => mask_ternlog_assign::<136>(a, b, c),
        137 => mask_ternlog_assign::<137>(a, b, c),
        138 => mask_ternlog_assign::<138>(a, b, c),
        139 => mask_ternlog_assign::<139>(a, b, c),
        140 => mask_ternlog_assign::<140>(a, b, c),
        141 => mask_ternlog_assign::<141>(a, b, c),
        142 => mask_ternlog_assign::<142>(a, b, c),
        143 => mask_ternlog_assign::<143>(a, b, c),
        144 => mask_ternlog_assign::<144>(a, b, c),
        145 => mask_ternlog_assign::<145>(a, b, c),
        146 => mask_ternlog_assign::<146>(a, b, c),
        147 => mask_ternlog_assign::<147>(a, b, c),
        148 => mask_ternlog_assign::<148>(a, b, c),
        149 => mask_ternlog_assign::<149>(a, b, c),
        150 => mask_ternlog_assign::<150>(a, b, c),
        151 => mask_ternlog_assign::<151>(a, b, c),
        152 => mask_ternlog_assign::<152>(a, b, c),
        153 => mask_ternlog_assign::<153>(a, b, c),
        154 => mask_ternlog_assign::<154>(a, b, c),
        155 => mask_ternlog_assign::<155>(a, b, c),
        156 => mask_ternlog_assign::<156>(a, b, c),
        157 => mask_ternlog_assign::<157>(a, b, c),
        158 => mask_ternlog_assign::<158>(a, b, c),
        159 => mask_ternlog_assign::<159>(a, b, c),
        160 => mask_ternlog_assign::<160>(a, b, c),
        161 => mask_ternlog_assign::<161>(a, b, c),
        162 => mask_ternlog_assign::<162>(a, b, c),
        163 => mask_ternlog_assign::<163>(a, b, c),
        164 => mask_ternlog_assign::<164>(a, b, c),
        165 => mask_ternlog_assign::<165>(a, b, c),
        166 => mask_ternlog_assign::<166>(a, b, c),
        167 => mask_ternlog_assign::<167>(a, b, c),
        168 => mask_ternlog_assign::<168>(a, b, c),
        169 => mask_ternlog_assign::<169>(a, b, c),
        170 => mask_ternlog_assign::<170>(a, b, c),
        171 => mask_ternlog_assign::<171>(a, b, c),
        172 => mask_ternlog_assign::<172>(a, b, c),
        173 => mask_ternlog_assign::<173>(a, b, c),
        174 => mask_ternlog_assign::<174>(a, b, c),
        175 => mask_ternlog_assign::<175>(a, b, c),
        176 => mask_ternlog_assign::<176>(a, b, c),
        177 => mask_ternlog_assign::<177>(a, b, c),
        178 => mask_ternlog_assign::<178>(a, b, c),
        179 => mask_ternlog_assign::<179>(a, b, c),
        180 => mask_ternlog_assign::<180>(a, b, c),
        181 => mask_ternlog_assign::<181>(a, b, c),
        182 => mask_ternlog_assign::<182>(a, b, c),
        183 => mask_ternlog_assign::<183>(a, b, c),
        184 => mask_ternlog_assign::<184>(a, b, c),
        185 => mask_ternlog_assign::<185>(a, b, c),
        186 => mask_ternlog_assign::<186>(a, b, c),
        187 => mask_ternlog_assign::<187>(a, b, c),
        188 => mask_ternlog_assign::<188>(a, b, c),
        189 => mask_ternlog_assign::<189>(a, b, c),
        190 => mask_ternlog_assign::<190>(a, b, c),
        191 => mask_ternlog_assign::<191>(a, b, c),
        192 => mask_ternlog_assign::<192>(a, b, c),
        193 => mask_ternlog_assign::<193>(a, b, c),
        194 => mask_ternlog_assign::<194>(a, b, c),
        195 => mask_ternlog_assign::<195>(a, b, c),
        196 => mask_ternlog_assign::<196>(a, b, c),
        197 => mask_ternlog_assign::<197>(a, b, c),
        198 => mask_ternlog_assign::<198>(a, b, c),
        199 => mask_ternlog_assign::<199>(a, b, c),
        200 => mask_ternlog_assign::<200>(a, b, c),
        201 => mask_ternlog_assign::<201>(a, b, c),
        202 => mask_ternlog_assign::<202>(a, b, c),
        203 => mask_ternlog_assign::<203>(a, b, c),
        204 => mask_ternlog_assign::<204>(a, b, c),
        205 => mask_ternlog_assign::<205>(a, b, c),
        206 => mask_ternlog_assign::<206>(a, b, c),
        207 => mask_ternlog_assign::<207>(a, b, c),
        208 => mask_ternlog_assign::<208>(a, b, c),
        209 => mask_ternlog_assign::<209>(a, b, c),
        210 => mask_ternlog_assign::<210>(a, b, c),
        211 => mask_ternlog_assign::<211>(a, b, c),
        212 => mask_ternlog_assign::<212>(a, b, c),
        213 => mask_ternlog_assign::<213>(a, b, c),
        214 => mask_ternlog_assign::<214>(a, b, c),
        215 => mask_ternlog_assign::<215>(a, b, c),
        216 => mask_ternlog_assign::<216>(a, b, c),
        217 => mask_ternlog_assign::<217>(a, b, c),
        218 => mask_ternlog_assign::<218>(a, b, c),
        219 => mask_ternlog_assign::<219>(a, b, c),
        220 => mask_ternlog_assign::<220>(a, b, c),
        221 => mask_ternlog_assign::<221>(a, b, c),
        222 => mask_ternlog_assign::<222>(a, b, c),
        223 => mask_ternlog_assign::<223>(a, b, c),
        224 => mask_ternlog_assign::<224>(a, b, c),
        225 => mask_ternlog_assign::<225>(a, b, c),
        226 => mask_ternlog_assign::<226>(a, b, c),
        227 => mask_ternlog_assign::<227>(a, b, c),
        228 => mask_ternlog_assign::<228>(a, b, c),
        229 => mask_ternlog_assign::<229>(a, b, c),
        230 => mask_ternlog_assign::<230>(a, b, c),
        231 => mask_ternlog_assign::<231>(a, b, c),
        232 => mask_ternlog_assign::<232>(a, b, c),
        233 => mask_ternlog_assign::<233>(a, b, c),
        234 => mask_ternlog_assign::<234>(a, b, c),
        235 => mask_ternlog_assign::<235>(a, b, c),
        236 => mask_ternlog_assign::<236>(a, b, c),
        237 => mask_ternlog_assign::<237>(a, b, c),
        238 => mask_ternlog_assign::<238>(a, b, c),
        239 => mask_ternlog_assign::<239>(a, b, c),
        240 => mask_ternlog_assign::<240>(a, b, c),
        241 => mask_ternlog_assign::<241>(a, b, c),
        242 => mask_ternlog_assign::<242>(a, b, c),
        243 => mask_ternlog_assign::<243>(a, b, c),
        244 => mask_ternlog_assign::<244>(a, b, c),
        245 => mask_ternlog_assign::<245>(a, b, c),
        246 => mask_ternlog_assign::<246>(a, b, c),
        247 => mask_ternlog_assign::<247>(a, b, c),
        248 => mask_ternlog_assign::<248>(a, b, c),
        249 => mask_ternlog_assign::<249>(a, b, c),
        250 => mask_ternlog_assign::<250>(a, b, c),
        251 => mask_ternlog_assign::<251>(a, b, c),
        252 => mask_ternlog_assign::<252>(a, b, c),
        253 => mask_ternlog_assign::<253>(a, b, c),
        254 => mask_ternlog_assign::<254>(a, b, c),
        255 => mask_ternlog_assign::<255>(a, b, c),
    }
}
// GEN-TERNLOG-DISPATCH-END

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    /// Deterministic pseudo-random `u64` stream (splitmix64) — no external
    /// crate, reproducible byte-for-byte across every run and every machine.
    fn splitmix64(state: &mut u64) -> u64 {
        *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = *state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// One seeded triple of `n`-word `u64` planes, all independent draws from
    /// the same stream.
    fn seeded_triple(seed: u64, n: usize) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
        let mut s = seed;
        let a: Vec<u64> = (0..n).map(|_| splitmix64(&mut s)).collect();
        let b: Vec<u64> = (0..n).map(|_| splitmix64(&mut s)).collect();
        let c: Vec<u64> = (0..n).map(|_| splitmix64(&mut s)).collect();
        (a, b, c)
    }

    /// The bit-serial truth-table oracle — one bit at a time, independent of
    /// every SIMD backend and of this file's own generated dispatch.
    fn bit_serial_ternlog(a: u64, b: u64, c: u64, imm: u8) -> u64 {
        let mut r = 0u64;
        for bit in 0..64u32 {
            let idx = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1);
            if (imm >> idx) & 1 == 1 {
                r |= 1u64 << bit;
            }
        }
        r
    }

    /// FAILS IF: `ternlog_dispatch` disagrees with the bit-serial reference for
    /// ANY of the 256 immediates on one fixed random triple, or that triple
    /// itself is degenerate. A degenerate triple (e.g. all-zero words) makes
    /// `idx` constant across every bit, so the reference could only ever
    /// return one of two possible vectors (`imm & 1` replicated) regardless
    /// of which table is under test — a dispatch bug that swapped, say, arm
    /// 0x60 for arm 0x68 would then still pass by coincidence. The seeded
    /// triple is checked to produce far more than two distinct outputs across
    /// the 256 tables before the real assertion is trusted.
    #[test]
    fn ternlog_dispatch_matches_the_bit_serial_reference_for_every_immediate() {
        let (a, b, c) = seeded_triple(0xD1CE_5EED_C0FF_EE01, 5);
        let mut distinct: HashSet<Vec<u64>> = HashSet::new();

        for imm in 0u8..=255u8 {
            let expect: Vec<u64> = (0..a.len())
                .map(|i| bit_serial_ternlog(a[i], b[i], c[i], imm))
                .collect();

            let mut dst = vec![0xDEAD_BEEFu64; a.len()];
            ternlog_dispatch(imm, &a, &b, &c, &mut dst);
            assert_eq!(dst, expect, "ternlog_dispatch mismatch at imm={imm:#04x}");

            distinct.insert(expect);
        }

        assert!(
            distinct.len() >= 100,
            "the seeded triple produced only {} distinct outputs across all 256 \
             immediates — too degenerate to discriminate tables (want a triple \
             whose bits vary, not a constant one)",
            distinct.len()
        );
    }

    /// FAILS IF: `ternlog_dispatch_assign` (in-place, `a` doubles as `dst`)
    /// disagrees with `ternlog_dispatch` (out-of-place) for ANY immediate on
    /// the same triple — the executor picks between the two entry points by
    /// whether `dst` aliases an input, and both must realize the identical
    /// table.
    #[test]
    fn ternlog_dispatch_assign_matches_dispatch_out_of_place_for_every_immediate() {
        let (a, b, c) = seeded_triple(0x5EED_BA5E_0000_00FF, 5);

        for imm in 0u8..=255u8 {
            let mut expect = vec![0u64; a.len()];
            ternlog_dispatch(imm, &a, &b, &c, &mut expect);

            let mut got = a.clone();
            ternlog_dispatch_assign(imm, &mut got, &b, &c);

            assert_eq!(
                got, expect,
                "ternlog_dispatch_assign mismatch at imm={imm:#04x}"
            );
        }
    }

    const BEGIN_MARKER: &str = "// GEN-TERNLOG-DISPATCH-BEGIN";
    const END_MARKER: &str = "// GEN-TERNLOG-DISPATCH-END";

    /// Parses `N` out of a line shaped `"... {prefix}N>(...)"`, or `None` if
    /// `prefix` doesn't occur on this line at all.
    fn extract_imm(line: &str, prefix: &str) -> Option<u32> {
        let start = line.find(prefix)? + prefix.len();
        let rest = &line[start..];
        let end = rest.find('>')?;
        rest[..end].parse::<u32>().ok()
    }

    /// FAILS IF: a hand edit inside the markers drops an arm, duplicates one,
    /// reorders them, or adds a wildcard/fallback arm — this reads the
    /// crate's OWN committed source (`include_str!`), so it catches a manual
    /// edit even if nobody re-ran the generator afterward. Marker detection is
    /// LINE-exact (a line whose trimmed content equals the marker text)
    /// rather than a raw substring search over the whole file: a raw search
    /// would also match this test's own `BEGIN_MARKER`/`END_MARKER` string
    /// literals two lines below, double-counting the markers before the real
    /// check ever ran (caught while writing this test — the naive version
    /// failed on itself). The two arm prefixes are chosen so
    /// `mask_ternlog::<N>(` never matches a `mask_ternlog_assign::<N>(` line
    /// (the latter has `_assign` immediately after `mask_ternlog`, never
    /// `::<`), so the two counts cannot bleed into each other either.
    #[test]
    fn the_generated_region_carries_exactly_256_ordered_arms_for_each_entry_point() {
        let src = include_str!("ternlog_dispatch.rs");
        let src_lines: Vec<&str> = src.lines().collect();

        let begins: Vec<usize> = src_lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.trim() == BEGIN_MARKER)
            .map(|(i, _)| i)
            .collect();
        let ends: Vec<usize> = src_lines
            .iter()
            .enumerate()
            .filter(|(_, line)| line.trim() == END_MARKER)
            .map(|(i, _)| i)
            .collect();
        assert_eq!(
            begins.len(),
            1,
            "expected exactly one bare BEGIN marker line, found {}",
            begins.len()
        );
        assert_eq!(
            ends.len(),
            1,
            "expected exactly one bare END marker line, found {}",
            ends.len()
        );
        assert!(
            begins[0] < ends[0],
            "BEGIN marker line must precede END marker line"
        );

        let region_lines = &src_lines[begins[0]..=ends[0]];

        let dispatch_ns: Vec<u32> = region_lines
            .iter()
            .filter_map(|line| extract_imm(line, "=> mask_ternlog::<"))
            .collect();
        let assign_ns: Vec<u32> = region_lines
            .iter()
            .filter_map(|line| extract_imm(line, "=> mask_ternlog_assign::<"))
            .collect();

        let expected: Vec<u32> = (0..=255).collect();
        assert_eq!(
            dispatch_ns, expected,
            "ternlog_dispatch arms missing, duplicated, or reordered"
        );
        assert_eq!(
            assign_ns, expected,
            "ternlog_dispatch_assign arms missing, duplicated, or reordered"
        );
    }
}
