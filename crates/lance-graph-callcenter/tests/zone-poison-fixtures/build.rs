// Zone-poison fixture build script.
//
// Mirrors the core logic of lance-graph-callcenter/build.rs but scans only
// the local src/external_intent.rs (the deliberately-poisoned Zone 2 file).
// Always runs in "strict" mode: any Serialize derive on a public type causes
// cargo::error= + std::process::exit(1).
//
// The subprocess compile-fail probe in zone_serialize_check_compile_fail.rs
// runs `cargo build` on this fixture and asserts the process exits non-zero
// with stderr containing "D-CASCADE-V1-1 zone_serialize_check:".

fn derive_has_serialize(attr: &syn::Attribute) -> Option<String> {
    if !attr.path().is_ident("derive") {
        return None;
    }
    let mut hit: Option<String> = None;
    let _ = attr.parse_nested_meta(|meta| {
        if let Some(last) = meta.path.segments.last() {
            if last.ident == "Serialize" {
                let full = meta
                    .path
                    .segments
                    .iter()
                    .map(|s| s.ident.to_string())
                    .collect::<Vec<_>>()
                    .join("::");
                hit = Some(full);
            }
        }
        Ok(())
    });
    hit
}

fn scan_file(file: &syn::File) -> Vec<(String, String)> {
    let mut hits = Vec::new();
    for item in &file.items {
        let (ident, attrs, vis) = match item {
            syn::Item::Struct(s) => (s.ident.to_string(), &s.attrs, &s.vis),
            syn::Item::Enum(e) => (e.ident.to_string(), &e.attrs, &e.vis),
            _ => continue,
        };
        if !matches!(vis, syn::Visibility::Public(_)) {
            continue;
        }
        for attr in attrs {
            if let Some(derive_name) = derive_has_serialize(attr) {
                hits.push((ident.clone(), derive_name));
            }
        }
    }
    hits
}

fn main() {
    let manifest = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let path = std::path::Path::new(&manifest).join("src/external_intent.rs");
    println!("cargo:rerun-if-changed={}", path.display());

    let src = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("zone-poison-fixture: cannot read {}: {}", path.display(), e));
    let file = syn::parse_file(&src)
        .unwrap_or_else(|e| panic!("zone-poison-fixture: cannot parse {}: {}", path.display(), e));

    let hits = scan_file(&file);
    if hits.is_empty() {
        // Fixture is broken — it MUST contain a Serialize derive.
        println!(
            "cargo:warning=zone-poison-fixture: no Serialize derive found in {}; fixture is invalid",
            path.display()
        );
        // Treat missing poison as a fixture integrity failure (still non-zero).
        println!(
            "cargo::error=D-CASCADE-V1-1 zone_serialize_check: fixture integrity error — \
             src/external_intent.rs must contain a pub struct/enum with #[derive(Serialize)]"
        );
        std::process::exit(1);
    }

    for (ident, derive_name) in &hits {
        println!(
            "cargo:warning=ZONE-SERIALIZE-VIOLATION [Zone 2] {} :: pub struct/enum `{}` carries \
             `#[derive({})]` — Zone 1/2 types may NOT serialize",
            path.display(),
            ident,
            derive_name
        );
    }

    let first = &hits[0];
    println!(
        "cargo::error=D-CASCADE-V1-1 zone_serialize_check: `{}` in {} (Zone 2) carries \
         `#[derive({})]` — Zone 1/2 types may NOT serialize. Move to Zone 3 \
         (transcode/phoenix/postgrest/drain/supabase) or remove the derive.",
        first.0,
        path.display(),
        first.1
    );
    std::process::exit(1);
}
