# Rotation vs Error Correction

## Core Finding (2026-03-26)

Google TurboQuant (ICLR 2026) validates bgz17/CAM-PQ design choices:

- **Same principle, different application**: TurboQuant compresses KV-cache (flat key→value). bgz17 encodes knowledge graph topology (SPO triples with nodes, edges, truth values).

- **Codebook advantage**: Google criticizes PQ/FAISS for dataset-dependent training and large codebooks. bgz17's Fibonacci codebook is dataset-independent by construction — same 4097 entries for any data.

- **Three resolution modes**: 3×12-bit CAM (compact SPO address), 3×16kbit Hamming (broadband scan), 3×FP32 (full precision after hydration). TurboQuant has one mode: 3-bit compressed.

- **HHTL cascade**: HEEL→HIP→TWIG→LEAF with 90% rejection per stage. TurboQuant scans all candidates at one resolution. bgz17 resolves progressively.

- **Graph metadata**: A bgz17 fingerprint returns "Palantir DEVELOPS Gotham (confidence: 0.94)". TurboQuant returns "distance: 0.23". The path through the graph is the explanation.

## Potential Improvements from TurboQuant

1. **Polar decomposition before Fibonacci**: Radius → natural HEEL, angles → HIP/TWIG/LEAF. ~300 LOC in cam_pq.rs. Not yet implemented.

2. **Squeeze-and-Excitation for HHTL**: Query-adaptive cascade routing. Skip uninformative stages. ~50 LOC. Not yet implemented.

## Full Spec

See `docs/ROTATION_VS_ERROR_CORRECTION.md`
