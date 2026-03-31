//! HTML structure classifier using Reader LM palette + crate::simd from ndarray.

use ndarray::hpc::bgz17_bridge::Base17;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HtmlStructure {
    Heading, Paragraph, ListItem, TableCell,
    CodeBlock, Link, Image, BlockQuote, Unknown,
}

pub fn classify_structure(text: &str) -> HtmlStructure {
    let t = text.trim();
    if t.starts_with('#') { HtmlStructure::Heading }
    else if t.starts_with("- ") || t.starts_with("* ") || t.starts_with("1.") { HtmlStructure::ListItem }
    else if t.starts_with('|') && t.ends_with('|') { HtmlStructure::TableCell }
    else if t.starts_with("```") || t.starts_with("    ") { HtmlStructure::CodeBlock }
    else if t.starts_with('[') && t.contains("](") { HtmlStructure::Link }
    else if t.starts_with("![") { HtmlStructure::Image }
    else if t.starts_with('>') { HtmlStructure::BlockQuote }
    else if t.len() > 20 { HtmlStructure::Paragraph }
    else { HtmlStructure::Unknown }
}

pub fn text_fingerprint(text: &str) -> Base17 {
    let mut dims = [0i64; 17];
    for (i, byte) in text.bytes().enumerate() {
        dims[(i * 11) % 17] += byte as i64;
    }
    let max_abs = dims.iter().map(|d| d.abs()).max().unwrap_or(1).max(1);
    let scale = 10000.0 / max_abs as f64;
    let mut result = [0i16; 17];
    for d in 0..17 { result[d] = (dims[d] as f64 * scale).round().clamp(-32768.0, 32767.0) as i16; }
    Base17 { dims: result }
}

pub fn segment_markdown(markdown: &str) -> Vec<(HtmlStructure, String)> {
    markdown.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| (classify_structure(l), l.to_string()))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn test_heading() { assert_eq!(classify_structure("# Title"), HtmlStructure::Heading); }
    #[test] fn test_list() { assert_eq!(classify_structure("- item"), HtmlStructure::ListItem); }
    #[test] fn test_code() { assert_eq!(classify_structure("```rust"), HtmlStructure::CodeBlock); }
    #[test] fn test_table() { assert_eq!(classify_structure("| a | b |"), HtmlStructure::TableCell); }
    #[test] fn test_link() { assert_eq!(classify_structure("[x](http://y)"), HtmlStructure::Link); }
    #[test] fn test_quote() { assert_eq!(classify_structure("> text"), HtmlStructure::BlockQuote); }
    #[test] fn test_paragraph() { assert_eq!(classify_structure("Long enough paragraph text here."), HtmlStructure::Paragraph); }
    #[test] fn test_fingerprint() { assert_ne!(text_fingerprint("hello").dims, [0; 17]); }
}
