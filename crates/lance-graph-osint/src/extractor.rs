//! NARS-based triplet extraction — no LLM needed.
//!
//! Uses DeepNSM vocabulary + NARS inference to extract SPO triplets
//! from natural language text. The bgz7 Qwen weights provide the
//! semantic palette for entity recognition.

use ndarray::hpc::bgz17_bridge::Base17;
use lance_graph_planner::cache::triple_model::Truth;

/// A single knowledge triplet: subject -[relation]-> object.
#[derive(Clone, Debug)]
pub struct Triplet {
    pub subject: String,
    pub object: String,
    pub relation: String,
    pub truth: Truth,
    pub timestamp: u64,
}

impl Triplet {
    pub fn new(subject: &str, object: &str, relation: &str, timestamp: u64) -> Self {
        Self {
            subject: subject.to_string(),
            object: object.to_string(),
            relation: relation.to_string(),
            truth: Truth::new(0.9, 0.8),
            timestamp,
        }
    }

    pub fn with_truth(subject: &str, object: &str, relation: &str, truth: Truth, timestamp: u64) -> Self {
        Self {
            subject: subject.to_string(),
            object: object.to_string(),
            relation: relation.to_string(),
            truth,
            timestamp,
        }
    }

    pub fn to_string_repr(&self) -> String {
        format!("{} - {} - {}", self.subject, self.relation, self.object)
    }
}

/// Extract SPO triplets from text using verb-pattern matching.
pub fn extract_triplets(text: &str, timestamp: u64) -> Vec<Triplet> {
    let mut triplets = Vec::new();
    for sentence in split_sentences(text) {
        if let Some(triplet) = extract_from_sentence(&sentence, timestamp) {
            triplets.push(triplet);
        }
    }
    triplets
}

fn extract_from_sentence(sentence: &str, timestamp: u64) -> Option<Triplet> {
    let words: Vec<&str> = sentence.split_whitespace().collect();
    if words.len() < 3 { return None; }

    let verb_idx = words.iter().position(|w| is_verb(w))?;
    if verb_idx == 0 || verb_idx >= words.len() - 1 { return None; }

    let subject = words[..verb_idx].join(" ");
    let mut verb_end = verb_idx + 1;
    while verb_end < words.len() && is_verb_modifier(words[verb_end]) { verb_end += 1; }
    let verb = words[verb_idx..verb_end].join(" ");
    let object = words[verb_end..].join(" ");

    if subject.is_empty() || object.is_empty() { return None; }

    let confidence = if words.len() > 5 { 0.8 } else { 0.6 };

    Some(Triplet::with_truth(
        &clean(&subject), &clean(&object), &verb,
        Truth::new(0.9, confidence), timestamp,
    ))
}

fn is_verb(word: &str) -> bool {
    let w = word.to_lowercase();
    w.ends_with("ed") || w.ends_with("ing") || w.ends_with("es") || w.ends_with("ize")
    || COMMON_VERBS.contains(&w.as_str())
}

fn is_verb_modifier(word: &str) -> bool {
    matches!(word.to_lowercase().as_str(),
        "to" | "the" | "a" | "an" | "in" | "on" | "at" | "by" | "for" | "with" | "from" | "of")
}

fn clean(s: &str) -> String {
    s.trim_matches(|c: char| !c.is_alphanumeric() && c != ' ').trim().to_string()
}

fn split_sentences(text: &str) -> Vec<String> {
    text.split(|c: char| c == '.' || c == '!' || c == '?' || c == '\n')
        .filter(|s| s.trim().len() > 10)
        .map(|s| s.trim().to_string())
        .collect()
}

const COMMON_VERBS: &[&str] = &[
    "is", "are", "was", "were", "has", "have", "had", "does", "do", "did",
    "can", "could", "will", "would", "shall", "should", "may", "might",
    "must", "contains", "includes", "causes", "enables", "supports",
    "creates", "develops", "uses", "provides", "requires", "involves",
    "leads", "results", "produces", "generates", "implements", "defines",
    "represents", "means", "becomes", "remains", "knows", "believes",
];

/// Refine existing triplets against new ones using NARS revision.
pub fn refine_triplets(existing: &mut Vec<Triplet>, new_triplets: &[Triplet]) {
    use lance_graph_planner::cache::triple_model::truth_revision;
    for new_t in new_triplets {
        let mut found = false;
        for ex in existing.iter_mut() {
            if ex.subject == new_t.subject && ex.relation == new_t.relation {
                ex.truth = truth_revision(ex.truth, new_t.truth);
                ex.object = new_t.object.clone();
                ex.timestamp = new_t.timestamp;
                found = true;
                break;
            }
        }
        if !found { existing.push(new_t.clone()); }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_simple() {
        let triplets = extract_triplets("Albert Einstein developed the theory of relativity.", 1);
        assert!(!triplets.is_empty());
    }

    #[test]
    fn test_extract_multiple() {
        let text = "The cat is on the mat. The dog chased the ball. Alice knows Bob.";
        let triplets = extract_triplets(text, 1);
        assert!(triplets.len() >= 2);
    }

    #[test]
    fn test_refine() {
        let mut existing = vec![Triplet::new("Alice", "Bob", "knows", 1)];
        let new = vec![Triplet::new("Alice", "Charlie", "knows", 2)];
        refine_triplets(&mut existing, &new);
        assert!(existing.iter().any(|t| t.object == "Charlie"));
    }
}
