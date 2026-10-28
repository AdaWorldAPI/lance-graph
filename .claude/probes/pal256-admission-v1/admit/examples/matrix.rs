//! Amendment 6.5 P0: code-space compatibility matrix. Each producer encodes its own full code range
//! (every valid code, plus its NaN sentinel if it has one); the artifact is offered to every consumer.
//! A cell is "bind" or the refusal reason. Usage: matrix OUT.json
use admit::*;
use ndarray::hpc::codec8::{Asym, Calibration, CodeSpace, NaNAtMin, NanPolicy, NoNaN, Stores, Sym, TiesAway, TiesEven, I8, U8};

struct One;
impl Calibration for One {
    const SCALE: f32 = 1.0;
    const ZERO_POINT: i32 = 0;
}

fn payload<T: Stores<S>, S: CodeSpace, N: NanPolicy<S>>() -> Vec<T::Elem> {
    let mut v: Vec<T::Elem> = (S::MIN..=S::MAX).map(T::store).collect();
    if let Some(n) = N::NAN_CODE {
        v.push(T::store(n));
    }
    v
}

macro_rules! codecs {
    ($m:ident) => {
        $m!(
            ("I8/Asym/TiesAway/NoNaN (TfliteAct)", I8, Asym, TiesAway, NoNaN),
            ("I8/Asym/TiesEven/NoNaN (OnnxQ)", I8, Asym, TiesEven, NoNaN),
            ("I8/Sym/TiesAway/NoNaN (TfliteWeight)", I8, Sym, TiesAway, NoNaN),
            ("I8/Sym/TiesAway/NaNAtMin (SymNaN)", I8, Sym, TiesAway, NaNAtMin),
            ("U8/Asym/TiesAway/NoNaN (biased)", U8, Asym, TiesAway, NoNaN),
            ("U8/Sym/TiesAway/NaNAtMin (biased SymNaN)", U8, Sym, TiesAway, NaNAtMin)
        )
    };
}

macro_rules! run {
    ($(($pn:expr, $pt:ty, $ps:ty, $pr:ty, $pnn:ty)),*) => {{
        let mut arts: Vec<(&str, Artifact)> = Vec::new();
        $( arts.push(($pn, produce::<$pt, $ps, $pr, $pnn, One>(&payload::<$pt, $ps, $pnn>(), "lab.op", 1, None))); )*
        arts
    }};
}

macro_rules! consume {
    ($arts:expr; $(($cn:expr, $ct:ty, $cs:ty, $cr:ty, $cnn:ty)),*) => {{
        let mut rows: Vec<String> = Vec::new();
        for (pn, a) in $arts.iter() {
            let mut cells: Vec<String> = Vec::new();
            $(
                let cell = match bind(a, &Expect::<$ct, $cs, $cr, $cnn, One>::new("lab.op", 1, None)) {
                    Ok(_) => "bind".to_string(),
                    Err(r) => format!("{:?}", r).split([' ', '{']).next().unwrap().to_string(),
                };
                cells.push(format!("\"{}\": \"{}\"", $cn, cell));
            )*
            rows.push(format!("  \"{}\": {{{}}}", pn, cells.join(", ")));
        }
        rows
    }};
}

fn main() {
    let out = std::env::args().nth(1).expect("OUT.json");
    let arts = codecs!(run);
    macro_rules! go { ($($t:tt)*) => { consume!(arts; $($t)*) }; }
    let rows = codecs!(go);
    let s = format!("{{\n \"rows_are_producers_columns_are_consumers\": true,\n \"matrix\": {{\n{}\n }}\n}}\n", rows.join(",\n"));
    std::fs::write(&out, &s).unwrap();
    print!("{s}");
}
