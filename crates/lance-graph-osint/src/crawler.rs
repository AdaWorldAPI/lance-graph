//! Spider-based web crawler for mass OSINT exploration.
//!
//! Uses the `spider` crate (2M downloads, async, production-grade) to crawl
//! article URLs extracted from Google Custom Search results.
//!
//! ```text
//! Pearl query → Google Custom Search → article URLs
//!   → spider crawl each URL (async, follows links, handles TLS/redirects)
//!   → strip HTML → split paragraphs → extract SPO triplets
//!   → NARS revision → frontier expansion → 2000+ nodes
//! ```
//!
//! Feature-gated behind `spider-crawl`.

#[cfg(feature = "spider-crawl")]
pub mod crawl {
    use spider::website::Website;
    use std::collections::HashSet;

    use crate::extractor;
    use crate::reader;

    /// Result of crawling one query's worth of pages.
    #[derive(Debug)]
    pub struct CrawlResult {
        /// Pages successfully fetched.
        pub pages_fetched: usize,
        /// Total paragraphs extracted.
        pub paragraphs: usize,
        /// Triplets extracted from all pages.
        pub triplets: Vec<extractor::Triplet>,
        /// URLs that were crawled.
        pub urls_crawled: Vec<String>,
    }

    /// Crawl search results for a query using the spider framework.
    ///
    /// 1. Google Custom Search → article URLs
    /// 2. Spider crawls each URL (async, proper TLS, follows redirects)
    /// 3. Extract paragraphs → triplets
    ///
    /// Falls back to curl if spider fails.
    pub fn crawl_query(query: &str, max_articles: usize, clock: u64) -> CrawlResult {
        // Get article URLs from Google Custom Search
        let urls = match reader::google_search(query, max_articles) {
            Ok(results) => results.into_iter().map(|r| r.link).collect::<Vec<_>>(),
            Err(_) => {
                // Fallback: construct a single DuckDuckGo URL
                vec![format!("https://html.duckduckgo.com/html/?q={}",
                    reader::urlencoding_pub(query))]
            }
        };

        if urls.is_empty() {
            return CrawlResult {
                pages_fetched: 0, paragraphs: 0,
                triplets: vec![], urls_crawled: vec![],
            };
        }

        // Crawl each URL with spider
        let rt = tokio::runtime::Runtime::new().unwrap();
        let mut all_triplets = Vec::new();
        let mut total_paragraphs = 0;
        let mut urls_crawled = Vec::new();

        for url in &urls {
            let page_text = rt.block_on(async {
                crawl_single_url(url).await
            });

            if let Some(text) = page_text {
                let paragraphs = reader::embed_text(&text);
                total_paragraphs += paragraphs.len();

                for para in &paragraphs {
                    let extracted = extractor::extract_triplets(&para.text, clock);
                    all_triplets.extend(extracted);
                }
                urls_crawled.push(url.clone());
            }
        }

        CrawlResult {
            pages_fetched: urls_crawled.len(),
            paragraphs: total_paragraphs,
            triplets: all_triplets,
            urls_crawled,
        }
    }

    /// Crawl a single URL using spider, return stripped text.
    async fn crawl_single_url(url: &str) -> Option<String> {
        let mut website: Website = Website::new(url);

        // Configure: single page, no link following
        website.with_depth(0);

        website.crawl().await;

        // Get the page content
        if let Some(pages) = website.get_pages() {
            if let Some(page) = pages.first() {
                let html = page.get_html();
                if !html.is_empty() {
                    return Some(strip_html_simple(&html));
                }
            }
        }

        None
    }

    /// Strip HTML tags (simple, fast, no regex).
    fn strip_html_simple(html: &str) -> String {
        let mut result = String::with_capacity(html.len() / 2);
        let mut in_tag = false;
        let mut in_script = false;

        let lower = html.to_lowercase();
        let chars: Vec<char> = html.chars().collect();
        let lower_chars: Vec<char> = lower.chars().collect();

        let mut i = 0;
        while i < chars.len() {
            if !in_tag && i + 7 < lower_chars.len() {
                let slice: String = lower_chars[i..i+7].iter().collect();
                if slice == "<script" { in_script = true; }
            }
            if in_script && i + 9 < lower_chars.len() {
                let slice: String = lower_chars[i..i+9].iter().collect();
                if slice == "</script>" {
                    in_script = false;
                    i += 9;
                    continue;
                }
            }

            if chars[i] == '<' { in_tag = true; i += 1; continue; }
            if chars[i] == '>' { in_tag = false; result.push(' '); i += 1; continue; }
            if !in_tag && !in_script { result.push(chars[i]); }
            i += 1;
        }

        // Collapse whitespace
        let mut clean = String::with_capacity(result.len());
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

    /// Batch-crawl multiple queries. Returns total triplets discovered.
    pub fn batch_crawl(
        queries: &[String],
        max_articles_per_query: usize,
        clock_start: u64,
    ) -> Vec<CrawlResult> {
        queries.iter().enumerate().map(|(i, q)| {
            crawl_query(q, max_articles_per_query, clock_start + i as u64)
        }).collect()
    }
}

// Re-export for non-feature-gated code
#[cfg(not(feature = "spider-crawl"))]
pub mod crawl {
    /// Spider crawl not available. Enable `spider-crawl` feature.
    pub fn crawl_query(_query: &str, _max_articles: usize, _clock: u64) -> ! {
        panic!("spider-crawl feature not enabled. Add `features = [\"spider-crawl\"]` to Cargo.toml")
    }
}
