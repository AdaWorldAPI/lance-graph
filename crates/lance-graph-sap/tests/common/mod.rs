use lance_graph_sap::{
    bind::CatsBatch,
    schema::{CatsSchema, FIELD_COUNT},
};

pub fn fixture(n: usize) -> [Vec<Option<&'static str>>; FIELD_COUNT] {
    let values: Vec<_> = include_str!("../../fixtures/cats.txt")
        .lines()
        .map(|v| if v == "\\N" { None } else { Some(v) })
        .collect();
    let base: [Option<&str>; FIELD_COUNT] = values.try_into().unwrap();
    std::array::from_fn(|i| vec![base[i]; n])
}

pub fn bind(input: &[Vec<Option<&str>>; FIELD_COUNT]) -> CatsBatch {
    CatsBatch::bind(
        CatsSchema::new(42, 0),
        std::array::from_fn(|i| input[i].as_slice()),
    )
    .unwrap()
}
