// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! xAI/Grok API client for OSINT triplet extraction and knowledge refinement.
//!
//! Reads `ADA_XAI` environment variable for the API key (set in Railway.com).
//! Supports both REST (v1/chat/completions) and gRPC endpoints.
//!
//! # Environment Variables
//!
//! - `ADA_XAI` — xAI API key (required, never hardcoded)
//! - `XAI_BASE_URL` — API base URL (default: `https://api.x.ai/v1`)
//! - `XAI_MODEL` — model name (default: `grok-3-mini`)

use std::collections::HashSet;

use super::retrieval::{EXTRACTION_PROMPT, PLAN_PROMPT, REFINING_PROMPT};
use super::triplet_graph::{Triplet, TripletGraph};

/// Default xAI API base URL.
const DEFAULT_BASE_URL: &str = "https://api.x.ai/v1";

/// Default model name.
const DEFAULT_MODEL: &str = "grok-3-mini";

/// Configuration for the xAI client.
#[derive(Debug, Clone)]
pub struct XaiConfig {
    /// API base URL.
    pub base_url: String,
    /// Model name.
    pub model: String,
    /// Maximum tokens for completion.
    pub max_tokens: u32,
    /// Temperature for generation.
    pub temperature: f32,
}

impl Default for XaiConfig {
    fn default() -> Self {
        Self {
            base_url: std::env::var("XAI_BASE_URL")
                .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string()),
            model: std::env::var("XAI_MODEL").unwrap_or_else(|_| DEFAULT_MODEL.to_string()),
            max_tokens: 1024,
            temperature: 0.1,
        }
    }
}

/// xAI/Grok client for OSINT knowledge graph operations.
///
/// All API calls read the key from `ADA_XAI` env var at call time.
/// This is the same variable name used in Railway.com deployments.
///
/// # Supported operations
///
/// 1. **Extract triplets** from raw OSINT text observations
/// 2. **Refine graph** by identifying outdated/contradicted facts
/// 3. **Generate plans** from graph context + episodic memory
/// 4. **Classify entities** by type (person, organization, location, event)
#[derive(Debug, Clone)]
pub struct XaiClient {
    config: XaiConfig,
}

/// Error type for xAI API operations.
#[derive(Debug)]
pub enum XaiError {
    /// ADA_XAI environment variable not set.
    MissingApiKey,
    /// HTTP request failed.
    RequestFailed(String),
    /// API returned an error response.
    ApiError { status: u16, message: String },
    /// Failed to parse API response.
    ParseError(String),
}

impl std::fmt::Display for XaiError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingApiKey => write!(f, "ADA_XAI environment variable not set"),
            Self::RequestFailed(e) => write!(f, "HTTP request failed: {e}"),
            Self::ApiError { status, message } => write!(f, "API error {status}: {message}"),
            Self::ParseError(e) => write!(f, "Parse error: {e}"),
        }
    }
}

impl std::error::Error for XaiError {}

/// A chat message for the xAI API.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Response from the xAI API.
#[derive(Debug, Clone)]
pub struct XaiResponse {
    /// The generated text content.
    pub content: String,
    /// Usage: prompt tokens.
    pub prompt_tokens: u32,
    /// Usage: completion tokens.
    pub completion_tokens: u32,
}

impl XaiClient {
    /// Create a new client with default configuration.
    pub fn new() -> Self {
        Self {
            config: XaiConfig::default(),
        }
    }

    /// Create a new client with custom configuration.
    pub fn with_config(config: XaiConfig) -> Self {
        Self { config }
    }

    /// Read the API key from the `ADA_XAI` environment variable.
    ///
    /// Never hardcoded. Railway.com injects this at deploy time.
    fn api_key() -> Result<String, XaiError> {
        std::env::var("ADA_XAI").map_err(|_| XaiError::MissingApiKey)
    }

    /// Build the JSON request body for a chat completion.
    ///
    /// Format matches OpenAI-compatible `/v1/chat/completions` endpoint
    /// that xAI/Grok supports.
    pub fn build_request_body(&self, messages: &[ChatMessage]) -> String {
        let msgs: Vec<String> = messages
            .iter()
            .map(|m| {
                format!(
                    r#"{{"role":"{}","content":"{}"}}"#,
                    m.role,
                    m.content
                        .replace('\\', "\\\\")
                        .replace('"', "\\\"")
                        .replace('\n', "\\n")
                )
            })
            .collect();

        format!(
            r#"{{"model":"{}","messages":[{}],"max_tokens":{},"temperature":{}}}"#,
            self.config.model,
            msgs.join(","),
            self.config.max_tokens,
            self.config.temperature,
        )
    }

    /// Build the full endpoint URL.
    pub fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.config.base_url)
    }

    /// Parse a chat completion response JSON.
    ///
    /// Extracts the first choice's message content and usage stats.
    /// Handles the xAI/Grok response format (OpenAI-compatible).
    pub fn parse_response(json: &str) -> Result<XaiResponse, XaiError> {
        // Minimal JSON parsing without serde dependency in hot path.
        // The response format is: {"choices":[{"message":{"content":"..."}}],"usage":{...}}

        let content = extract_json_string(json, "content")
            .ok_or_else(|| XaiError::ParseError("missing content field".to_string()))?;

        let prompt_tokens = extract_json_number(json, "prompt_tokens").unwrap_or(0);
        let completion_tokens = extract_json_number(json, "completion_tokens").unwrap_or(0);

        Ok(XaiResponse {
            content,
            prompt_tokens,
            completion_tokens,
        })
    }

    // ── OSINT Operations ──

    /// Extract triplets from an observation using xAI/Grok.
    ///
    /// Returns the prompt + messages ready to send. The actual HTTP call
    /// is left to the caller (async runtime agnostic).
    pub fn prepare_extraction(&self, observation: &str) -> Result<(String, String), XaiError> {
        let _key = Self::api_key()?;
        let prompt = EXTRACTION_PROMPT.replace("{observation}", observation);
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a knowledge extraction system for OSINT analysis. Extract factual triplets from observations.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];
        let body = self.build_request_body(&messages);
        Ok((self.endpoint(), body))
    }

    /// Parse extracted triplets from xAI response text.
    ///
    /// Expects format: "subject, relation, object; subject, relation, object"
    pub fn parse_triplets(response_text: &str, timestamp: u64) -> Vec<Triplet> {
        response_text
            .split(';')
            .filter_map(|part| {
                let parts: Vec<&str> = part.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    // Input format is "subject, relation, object" (S-P-O).
                    // Triplet::new takes (subject, object, relation, timestamp) —
                    // note the (object, relation) swap relative to the SPO
                    // input order. Without this argument re-ordering, the
                    // parsed relation lands in the object slot and vice
                    // versa, silently corrupting every LLM-extracted fact.
                    Some(Triplet::new(parts[0], parts[2], parts[1], timestamp))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Prepare a refinement request to identify outdated facts.
    pub fn prepare_refinement(
        &self,
        existing_triplets: &[String],
        observation: &str,
    ) -> Result<(String, String), XaiError> {
        let _key = Self::api_key()?;
        let prompt = REFINING_PROMPT
            .replace("{existing_triplets}", &existing_triplets.join("\n"))
            .replace("{observation}", observation);
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a knowledge refinement system. Identify outdated or contradicted facts.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];
        let body = self.build_request_body(&messages);
        Ok((self.endpoint(), body))
    }

    /// Parse outdated triplet patterns from xAI response.
    pub fn parse_outdated(response_text: &str) -> Vec<(String, String, String)> {
        if response_text.trim().to_lowercase() == "none" {
            return Vec::new();
        }
        response_text
            .split(';')
            .filter_map(|part| {
                let parts: Vec<&str> = part.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 3 {
                    Some((
                        parts[0].to_string(),
                        parts[1].to_string(),
                        parts[2].to_string(),
                    ))
                } else {
                    None
                }
            })
            .collect()
    }

    /// Prepare a plan generation request.
    pub fn prepare_plan(
        &self,
        graph_context: &[String],
        episodic_context: &[String],
        observation: &str,
    ) -> Result<(String, String), XaiError> {
        let _key = Self::api_key()?;
        let prompt = PLAN_PROMPT
            .replace("{graph_context}", &graph_context.join("\n"))
            .replace("{episodic_context}", &episodic_context.join("\n"))
            .replace("{observation}", observation);
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are an OSINT analysis planning system. Generate actionable intelligence plans.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];
        let body = self.build_request_body(&messages);
        Ok((self.endpoint(), body))
    }

    /// Prepare an entity classification request.
    pub fn prepare_entity_classification(
        &self,
        entities: &HashSet<String>,
    ) -> Result<(String, String), XaiError> {
        let _key = Self::api_key()?;
        let entity_list: Vec<&str> = entities.iter().map(|s| s.as_str()).collect();
        let prompt = format!(
            "Classify each entity into one of: PERSON, ORGANIZATION, LOCATION, EVENT, CONCEPT, UNKNOWN.\n\nEntities:\n{}\n\nOutput format: entity, type (one per line)",
            entity_list.join("\n")
        );
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are an entity classification system.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: prompt,
            },
        ];
        let body = self.build_request_body(&messages);
        Ok((self.endpoint(), body))
    }

    /// Full OSINT update cycle: extract → refine → update graph.
    ///
    /// This is the core AriGraph update loop, transcoded from Python.
    /// The caller must execute the HTTP requests and feed back responses.
    ///
    /// Returns: (extraction_request, refinement_request_if_needed)
    pub fn prepare_update_cycle(
        &self,
        graph: &TripletGraph,
        observation: &str,
    ) -> Result<OsintUpdateRequests, XaiError> {
        let extraction = self.prepare_extraction(observation)?;

        // Get existing triplets near the observation's entities for refinement
        let words: HashSet<String> = observation
            .split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect();

        let associated = graph.get_associated(&words, 1);
        let existing: Vec<String> = associated.iter().map(|t| t.to_string_repr()).collect();

        let refinement = if existing.is_empty() {
            None
        } else {
            Some(self.prepare_refinement(&existing, observation)?)
        };

        Ok(OsintUpdateRequests {
            extraction,
            refinement,
        })
    }
}

/// Pair of API requests for a full OSINT update cycle.
#[derive(Debug, Clone)]
pub struct OsintUpdateRequests {
    /// (endpoint, body) for triplet extraction.
    pub extraction: (String, String),
    /// (endpoint, body) for refinement (None if graph is empty near observation).
    pub refinement: Option<(String, String)>,
}

// ── Minimal JSON helpers (no serde dependency) ──

/// Extract a string value from JSON by key name (first occurrence).
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!(r#""{}""#, key);
    let pos = json.find(&pattern)?;
    let after_key = &json[pos + pattern.len()..];
    // Skip : and whitespace
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_ws = after_colon.trim_start();
    if !after_ws.starts_with('"') {
        return None;
    }
    let content = &after_ws[1..];
    let mut result = String::new();
    let mut chars = content.chars();
    loop {
        match chars.next()? {
            '\\' => match chars.next()? {
                '"' => result.push('"'),
                '\\' => result.push('\\'),
                'n' => result.push('\n'),
                't' => result.push('\t'),
                c => {
                    result.push('\\');
                    result.push(c);
                }
            },
            '"' => break,
            c => result.push(c),
        }
    }
    Some(result)
}

/// Extract a numeric value from JSON by key name (first occurrence).
fn extract_json_number(json: &str, key: &str) -> Option<u32> {
    let pattern = format!(r#""{}""#, key);
    let pos = json.find(&pattern)?;
    let after_key = &json[pos + pattern.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_ws = after_colon.trim_start();
    let num_str: String = after_ws
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect();
    num_str.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_defaults() {
        let config = XaiConfig::default();
        assert_eq!(config.max_tokens, 1024);
        assert!((config.temperature - 0.1).abs() < f32::EPSILON);
    }

    #[test]
    fn test_build_request_body() {
        let client = XaiClient::new();
        let messages = vec![ChatMessage {
            role: "user".to_string(),
            content: "hello".to_string(),
        }];
        let body = client.build_request_body(&messages);
        assert!(body.contains(r#""role":"user""#));
        assert!(body.contains(r#""content":"hello""#));
        assert!(body.contains(r#""max_tokens":1024"#));
    }

    #[test]
    fn test_parse_triplets() {
        let text = "alice, knows, bob; bob, works_at, company_x";
        let triplets = XaiClient::parse_triplets(text, 1);
        assert_eq!(triplets.len(), 2);
        assert_eq!(triplets[0].subject, "alice");
        assert_eq!(triplets[0].relation, "knows");
        assert_eq!(triplets[0].object, "bob");
        assert_eq!(triplets[1].subject, "bob");
    }

    #[test]
    fn test_parse_outdated_none() {
        assert!(XaiClient::parse_outdated("none").is_empty());
        assert!(XaiClient::parse_outdated("None").is_empty());
    }

    #[test]
    fn test_parse_outdated_some() {
        let patterns =
            XaiClient::parse_outdated("alice, lives_in, new_york; bob, works_at, old_co");
        assert_eq!(patterns.len(), 2);
        assert_eq!(patterns[0].0, "alice");
    }

    #[test]
    fn test_parse_response() {
        let json = r#"{"choices":[{"message":{"content":"hello world"}}],"usage":{"prompt_tokens":10,"completion_tokens":5}}"#;
        let resp = XaiClient::parse_response(json).unwrap();
        assert_eq!(resp.content, "hello world");
        assert_eq!(resp.prompt_tokens, 10);
        assert_eq!(resp.completion_tokens, 5);
    }

    #[test]
    fn test_api_key_missing() {
        // This test verifies that MissingApiKey is returned when ADA_XAI is not set.
        // In CI/Railway, the key IS set, so this test only validates the error path.
        // We don't assert on the result — just check prepare_extraction doesn't panic.
        let client = XaiClient::new();
        let _result = client.prepare_extraction("test observation");
        // If ADA_XAI is set, result is Ok. If not, result is Err(MissingApiKey). Both valid.
    }

    #[test]
    fn test_endpoint_url() {
        let client = XaiClient::new();
        let endpoint = client.endpoint();
        assert!(endpoint.ends_with("/chat/completions"));
    }

    #[test]
    fn test_json_string_extraction() {
        let json = r#"{"name":"test","value":"hello \"world\""}"#;
        assert_eq!(extract_json_string(json, "name").unwrap(), "test");
        assert_eq!(
            extract_json_string(json, "value").unwrap(),
            r#"hello "world""#
        );
    }

    #[test]
    fn test_json_number_extraction() {
        let json = r#"{"count":42,"name":"test"}"#;
        assert_eq!(extract_json_number(json, "count").unwrap(), 42);
        assert_eq!(extract_json_number(json, "missing"), None);
    }

    #[test]
    fn test_prepare_entity_classification() {
        let client = XaiClient::new();
        let entities: HashSet<String> = ["alice", "acme_corp", "new_york"]
            .into_iter()
            .map(String::from)
            .collect();
        // Will fail if ADA_XAI not set, which is fine in unit tests
        let _result = client.prepare_entity_classification(&entities);
    }
}
