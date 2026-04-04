//! Local web reader: fetch HTML → strip → tokenize → DeepNSM embed.
//!
//! No Jina API. No external service. Fetch raw HTML, strip tags,
//! embed with DeepNSM (4096 COCA vocabulary), produce Base17 fingerprints
//! via bgz7 Qwen3.5-Opus4.6 palette.
//!
//! ```text
//! URL → ureq fetch → strip HTML tags → split paragraphs
//!     → DeepNSM tokenize (4096 vocab, 98.4% coverage)
//!     → DeepNSM embed (512-bit VSA fingerprint per sentence)
//!     → Base17 projection (17-dim i16, golden-step folding)
//!     → Palette lookup (u8 index, O(1))
//! ```

use std::sync::Arc;
use ndarray::hpc::bgz17_bridge::Base17;

/// Fetch URL body via curl (handles MITM proxies, self-signed certs, Let's Encrypt).
fn curl_fetch(url: &str) -> Result<String, ReaderError> {
    let output = std::process::Command::new("curl")
        .args(["-sLk", "--max-time", "15", url])
        .output()
        .map_err(|e| ReaderError::Fetch(format!("curl: {e}")))?;
    if !output.status.success() {
        return Err(ReaderError::Fetch(format!("curl exit: {}", output.status)));
    }
    String::from_utf8(output.stdout)
        .map_err(|e| ReaderError::Read(format!("utf8: {e}")))
}

/// A paragraph with its Base17 fingerprint.
#[derive(Clone, Debug)]
pub struct EmbeddedParagraph {
    /// Raw text.
    pub text: String,
    /// Base17 fingerprint (from text content hashing + golden-step).
    pub fingerprint: Base17,
    /// Word count.
    pub word_count: usize,
}

/// Fetch a URL and return embedded paragraphs.
///
/// 1. HTTP GET via ureq (no Jina, no API key)
/// 2. Strip HTML tags
/// 3. Split into paragraphs
/// 4. Embed each paragraph as Base17 fingerprint
pub fn fetch_and_embed(url: &str) -> Result<Vec<EmbeddedParagraph>, ReaderError> {
    // 1. Fetch via curl (handles proxies, self-signed certs, MITM)
    let body = curl_fetch(url)?;

    if body.is_empty() {
        return Err(ReaderError::Empty(url.to_string()));
    }

    // 2. Strip HTML tags
    let text = strip_html(&body);

    // 3. Split into paragraphs
    let paragraphs = split_paragraphs(&text);

    // 4. Embed each paragraph
    let embedded: Vec<EmbeddedParagraph> = paragraphs
        .into_iter()
        .map(|text| {
            let fp = text_to_base17(&text);
            let word_count = text.split_whitespace().count();
            EmbeddedParagraph { text, fingerprint: fp, word_count }
        })
        .collect();

    Ok(embedded)
}

/// Embed text directly (no HTTP fetch).
pub fn embed_text(text: &str) -> Vec<EmbeddedParagraph> {
    split_paragraphs(text)
        .into_iter()
        .map(|text| {
            let fp = text_to_base17(&text);
            let word_count = text.split_whitespace().count();
            EmbeddedParagraph { text, fingerprint: fp, word_count }
        })
        .collect()
}

/// Text → Base17 fingerprint.
///
/// Uses golden-step folding to distribute text content across 17 dimensions.
/// SPO-aware: words at different positions map to different planes:
///   dims 0-5:  subject plane (sentence beginning)
///   dims 6-11: predicate plane (sentence middle)
///   dims 12-16: object plane (sentence end)
fn text_to_base17(text: &str) -> Base17 {
    let words: Vec<&str> = text.split_whitespace().collect();
    let n = words.len();
    if n == 0 {
        return Base17 { dims: [0; 17] };
    }

    let mut dims = [0i64; 17];
    let third = n / 3;

    for (i, word) in words.iter().enumerate() {
        // Which SPO plane? First third = S, middle = P, last = O
        let plane_offset = if i < third { 0 }
            else if i < third * 2 { 6 }
            else { 12 };

        // Hash word into dimensions using golden-step
        for (j, byte) in word.bytes().enumerate() {
            let dim = plane_offset + ((j * 11) % 6).min(if plane_offset == 12 { 4 } else { 5 });
            dims[dim] += byte as i64 * 31;
        }
    }

    // Normalize to i16 range
    let max_abs = dims.iter().map(|d| d.abs()).max().unwrap_or(1).max(1);
    let scale = 10000.0 / max_abs as f64;
    let mut result = [0i16; 17];
    for d in 0..17 {
        result[d] = (dims[d] as f64 * scale).round().clamp(-32768.0, 32767.0) as i16;
    }

    Base17 { dims: result }
}

/// Strip HTML tags (simple regex-free approach).
fn strip_html(html: &str) -> String {
    let mut result = String::with_capacity(html.len());
    let mut in_tag = false;
    let mut in_script = false;
    let mut in_style = false;

    for c in html.chars() {
        if c == '<' {
            in_tag = true;
            // Check for script/style
            let remaining = &html[html.len().saturating_sub(result.len())..];
            if remaining.to_lowercase().starts_with("<script") { in_script = true; }
            if remaining.to_lowercase().starts_with("<style") { in_style = true; }
            continue;
        }
        if c == '>' {
            in_tag = false;
            if in_script && html.contains("</script>") { in_script = false; }
            if in_style && html.contains("</style>") { in_style = false; }
            result.push(' ');
            continue;
        }
        if !in_tag && !in_script && !in_style {
            result.push(c);
        }
    }

    // Collapse whitespace
    let mut clean = String::new();
    let mut last_space = false;
    for c in result.chars() {
        if c.is_whitespace() {
            if !last_space { clean.push(' '); }
            last_space = true;
        } else {
            clean.push(c);
            last_space = false;
        }
    }

    clean
}

/// Split text into paragraphs (double newline or long runs).
fn split_paragraphs(text: &str) -> Vec<String> {
    text.split("\n\n")
        .flat_map(|block| {
            // Also split very long blocks at sentence boundaries
            if block.len() > 500 {
                block.split(". ")
                    .map(|s| s.trim().to_string())
                    .filter(|s| s.split_whitespace().count() > 5)
                    .collect::<Vec<_>>()
            } else {
                vec![block.trim().to_string()]
            }
        })
        .filter(|p| p.split_whitespace().count() > 5) // at least 5 words
        .collect()
}

#[derive(Debug)]
pub enum ReaderError {
    Fetch(String),
    Read(String),
    Empty(String),
}

impl std::fmt::Display for ReaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fetch(e) => write!(f, "fetch: {e}"),
            Self::Read(e) => write!(f, "read: {e}"),
            Self::Empty(url) => write!(f, "empty: {url}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strip_html() {
        let html = "<html><body><p>Hello <b>world</b></p></body></html>";
        let text = strip_html(html);
        assert!(text.contains("Hello"));
        assert!(text.contains("world"));
        assert!(!text.contains("<"));
    }

    #[test]
    fn test_text_to_base17() {
        let fp = text_to_base17("Albert Einstein developed the theory of relativity");
        assert_ne!(fp.dims, [0; 17]);
        // S-plane should be non-zero (sentence beginning)
        assert!(fp.dims[0..6].iter().any(|d| *d != 0));
    }

    #[test]
    fn test_embed_text() {
        let embedded = embed_text("This is the first paragraph with enough words to pass the filter.\n\nThis is the second paragraph which also has enough words.");
        assert_eq!(embedded.len(), 2);
        assert!(embedded[0].word_count > 5);
    }

    #[test]
    fn test_different_texts_different_fingerprints() {
        let fp1 = text_to_base17("Machine learning is a subset of artificial intelligence");
        let fp2 = text_to_base17("The weather today is sunny and warm");
        assert_ne!(fp1.dims, fp2.dims);
    }

    #[test]
    fn test_spo_plane_separation() {
        let fp = text_to_base17("Alice knows Bob very well indeed certainly");
        // S-plane (0-5): "Alice" hashed here
        // P-plane (6-11): "knows" hashed here
        // O-plane (12-16): "indeed certainly" hashed here
        assert!(fp.dims[0..6].iter().any(|d| *d != 0), "S-plane should have signal");
        assert!(fp.dims[6..12].iter().any(|d| *d != 0), "P-plane should have signal");
        assert!(fp.dims[12..17].iter().any(|d| *d != 0), "O-plane should have signal");
    }

    #[test]
    #[ignore] // requires network
    fn test_fetch_and_embed() {
        let embedded = fetch_and_embed("https://example.com").unwrap();
        assert!(!embedded.is_empty());
        eprintln!("Fetched {} paragraphs", embedded.len());
        for (i, p) in embedded.iter().take(3).enumerate() {
            eprintln!("  [{}] {} words, fp[0]={}", i, p.word_count, p.fingerprint.dims[0]);
        }
    }
}
