# Word Frequency Data (COCA)

Word frequency lists from the Corpus of Contemporary American English (COCA, 1 billion words).
Source: [wordfrequency.info](https://www.wordfrequency.info)

## Files

### For Rust transcoding (start here)

| File | Rows | Description |
|------|------|-------------|
| `word_rank_lookup.csv` | 5,050 | **Primary**: rank → word → PoS → freq. Direct lookup table for tokenizer. |
| `lemmas_compact.csv` | 5,050 | Rank, lemma, PoS, freq, perMil, dispersion. Core metrics only. |
| `forms_compact.csv` | 5,050 | Rank, word, freq, #texts, %caps. Surface forms. |

### Full data

| File | Rows × Cols | Description |
|------|-------------|-------------|
| `lemmas_5k.csv` | 5,050 × 25 | Full lemma list with per-genre frequencies (blog, web, TV/movie, spoken, fiction, magazine, news, academic) |
| `word_forms.csv` | 11,460 × 6 | Lemma → surface form mappings (e.g., "be" → is, was, 's, are, were, been) |
| `forms_5k.csv` | 5,050 × 21 | Top 5K individual word forms with per-genre breakdown |
| `subgenres_5k.csv` | 5,050 × 195 | Fine-grained subgenre frequencies (96 subgenres × raw + per-million) |

## Column Reference

### PoS Tags
- `a` = article/determiner
- `v` = verb
- `c` = conjunction
- `i` = preposition
- `p` = pronoun
- `n` = noun
- `j` = adjective
- `r` = adverb
- `t` = particle/infinitive marker
- `d` = modal/auxiliary
- `e` = existential (there)
- `x` = not/negation

### Genre Codes
- `blog` = blog posts
- `web` = general web pages
- `TVM` = TV and movie subtitles
- `spok` = spoken (interviews, conversations)
- `fic` = fiction
- `mag` = magazine
- `news` = newspaper
- `acad` = academic

### Key Metrics
- `freq` = raw frequency (total count across 1B words)
- `perMil` = frequency per million words
- `disp` = dispersion (0-1, how evenly distributed across texts; 1.0 = perfectly even)
- `range` = number of texts containing the word
- `%caps` = percentage of occurrences that are capitalized

## Rust Usage

```rust
// Load the compact lookup for DeepNSM tokenization
use std::collections::HashMap;

struct WordEntry {
    rank: u16,
    pos: String,
    freq: u64,
}

fn load_word_ranks(csv_path: &str) -> HashMap<String, WordEntry> {
    // Parse word_rank_lookup.csv
    // rank,word,pos,freq
    // 1,the,a,50033612
    // 2,be,v,32394756
    // ...
}
```

## CAM-PQ Codebook (for Rust)

Product Quantization over 96D distributional vectors from COCA subgenre frequencies.

### Architecture

```
96D subgenre frequency vector (log-normalized, z-scored)
  → 6 subspaces × 16D
    → k-means (256 centroids per subspace)
      → 6-byte CAM fingerprint per word
```

### Files

| File | Size | Description |
|------|------|-------------|
| `codebook_pq.bin` | 96 KB | Flat binary: `[6][256][16] × f32`. Load directly in Rust. |
| `codebook_pq.json` | 504 KB | Same data as JSON + normalization params (mean, std). |
| `cam_codes.bin` | 30 KB | 5,050 words × 6 bytes. Word index `i` → `cam_codes[i*6..(i+1)*6]`. |
| `nsm_primes.json` | 11 KB | 63 NSM semantic primes → rank, CAM code, word index. |
| `word_cam_index.json` | 374 KB | word string → rank + CAM code + PoS. |

### Rust Usage

```rust
use std::fs;

// Load codebook: [6][256][16] f32
let codebook_bytes = fs::read("codebook_pq.bin")?;
let codebook: &[f32] = bytemuck::cast_slice(&codebook_bytes);
// codebook[subspace * 256 * 16 + centroid * 16 + dim]

// Load CAM codes: 5050 × 6 bytes
let cam_codes = fs::read("cam_codes.bin")?;
// Word at index i: &cam_codes[i*6..(i+1)*6]

// Distance between two words via codebook lookup
fn cam_distance(a: &[u8; 6], b: &[u8; 6], codebook: &[f32]) -> f32 {
    let mut dist = 0.0f32;
    for s in 0..6 {
        let offset_a = s * 256 * 16 + a[s] as usize * 16;
        let offset_b = s * 256 * 16 + b[s] as usize * 16;
        for d in 0..16 {
            let diff = codebook[offset_a + d] - codebook[offset_b + d];
            dist += diff * diff;
        }
    }
    dist.sqrt()
}
```

### Semantic Distance Verification

Distributional CAM-PQ distances capture NSM semantic relationships:

| Pair | Distance | Notes |
|------|----------|-------|
| i / you | 3.32 | Closest — same genre distribution (conversational) |
| think / know | 3.85 | Close — both mental predicates |
| before / after | 4.46 | Close — both temporal markers |
| live / die | 5.35 | Moderate — same domain, different valence |
| good / bad | 6.60 | Moderate — evaluators, inverse polarity |
| think / big | 8.16 | Far — different semantic fields |
| the / this | 11.47 | Far — very different distributional profiles |
| say / word | 15.22 | Farthest — different genre concentrations |

### NSM Prime Coverage

63/65 Wierzbicka semantic primes found in top 5K COCA words (96.9% coverage).
Each prime has a 6-byte CAM fingerprint that serves as a semantic anchor in the codebook.
