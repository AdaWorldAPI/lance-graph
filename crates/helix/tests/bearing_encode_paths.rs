//! MEASUREMENT: the two candidate `bearing -> Signed360` encodes, for the WIND
//! reuse of this codec. Records which one fits and by how much.
//!
//! `encode_signed` derives ALL THREE direction-bearing fields from `n` alone
//! (`residue.rs:182-204`) — there is no bearing entry point. The knowledge doc
//! `helix-cartesian-vs-fisher2z.md` prescribes `nearest spherical-Fibonacci
//! (n, sign)` for encoding a 3D normal. **That prescription does not fit wind.**
//!
//! The golden spiral couples latitude and azimuth through ONE index: to reach
//! `y ~ 0` (horizontal) you need `n ~ N-1`, and those few `n` have their azimuth
//! already fixed at `n*phi`. You cannot independently choose a bearing at the
//! horizon — and the lattice's latitude density is `~ sin(2*lat)`, sparsest
//! exactly there. Normals (q2 `/helix`, FMA body mesh) spread over the whole
//! sphere and never hit this; wind bearings cluster at the horizon and always do.
//!
//! Path B writes `(polar, azimuth)` directly and leaves `rim` carrying
//! `(place, n)` — licensed by the doctrine's own split: *"the rim is the METRIC
//! carrier, not a render input"* and *"direction is place-INDEPENDENT; only the
//! metric is place-coupled"*. It does mean `(polar, azimuth)` are no longer
//! functions of the same `n` as `rim`; per that split, that is the intent.
//!
//! Measured (N=65536): horizontal bearings — Path A 1.9-2.7 deg, Path B ~0.000 deg.
//! Over 24 (bearing x elevation) cases — Path A 0.972 deg, Path B 0.097 deg, 10x.
use helix::placement::{HemispherePoint, Sign};
use helix::residue::{ResidueEncoder, Signed360};

fn dir(bearing_deg: f64, elev_deg: f64) -> (f64,f64,f64) {
    let (b,e) = (bearing_deg.to_radians(), elev_deg.to_radians());
    (e.cos()*b.sin(), e.cos()*b.cos(), e.sin())   // (x=east, z=north, y=up)
}
fn ang(a:(f64,f64,f64), b:(f64,f64,f64)) -> f64 {
    (a.0*b.0+a.1*b.1+a.2*b.2).clamp(-1.0,1.0).acos().to_degrees()
}
fn decode(s: &Signed360) -> (f64,f64,f64) {
    let y = if s.polar>=128 {(s.polar as f64-128.0)/127.0} else {-((127.0-s.polar as f64)/127.0)};
    let r = (1.0-y*y).max(0.0).sqrt();
    let a = s.azimuth as f64/65536.0*std::f64::consts::TAU;
    (r*a.sin(), r*a.cos(), y)
}

#[test]
fn measure() {
    const N: usize = 65_536;
    let enc = ResidueEncoder::new(N);

    // Path A (doctrine): nearest spherical-Fibonacci (n, sign), then encode_signed.
    let nearest = |target:(f64,f64,f64)| -> (usize, Sign) {
        let mut best = (f64::MAX, 0usize, Sign::Pos);
        for n in 0..N {
            for sg in [Sign::Pos, Sign::Neg] {
                let p = HemispherePoint::signed_lift(n, N, sg);
                let v = (p.r*p.azimuth.sin(), p.r*p.azimuth.cos(), p.y);
                let e = ang(target, v);
                if e < best.0 { best = (e, n, sg); }
            }
        }
        (best.1, best.2)
    };
    // Path B: write (polar, azimuth) directly from the bearing; rim still from (place, n).
    let direct = |target:(f64,f64,f64), rim_n: usize| -> Signed360 {
        let y = target.2;
        let mag = (y.abs()*127.0).round().clamp(0.0,127.0) as u8;
        let polar = if y >= 0.0 { 128+mag } else { 127-mag };
        let az = target.0.atan2(target.1).rem_euclid(std::f64::consts::TAU);
        let azimuth = ((az/std::f64::consts::TAU)*65536.0).round() as u64 as u16;
        let mut s = enc.encode_signed(0x1234, rim_n, if y>=0.0 {Sign::Pos} else {Sign::Neg});
        s.polar = polar; s.azimuth = azimuth; s
    };

    println!("\nwind bearings (elevation 0 = horizontal), N={N}");
    println!("  bearing   elev    PathA nearest-n    PathB direct-field");
    let (mut sa, mut sb) = (0.0f64, 0.0f64);
    let mut c = 0;
    for bear in [0.0, 22.5, 45.0, 90.0, 137.5, 180.0, 270.0, 315.0] {
        for elev in [0.0, 5.0, 30.0] {
            let t = dir(bear, elev);
            let (n, sg) = nearest(t);
            let ea = ang(t, decode(&enc.encode_signed(0x1234, n, sg)));
            let eb = ang(t, decode(&direct(t, n)));
            if elev == 0.0 { println!("  {bear:>6.1}°  {elev:>4.0}°      {ea:>8.4}°          {eb:>8.4}°"); }
            sa += ea; sb += eb; c += 1;
        }
    }
    println!("\n  MEAN over {c} (bearing x elevation) cases:  PathA {:.4}°   PathB {:.4}°   ratio {:.2}x",
             sa/c as f64, sb/c as f64, (sa/c as f64)/(sb/c as f64));
}
