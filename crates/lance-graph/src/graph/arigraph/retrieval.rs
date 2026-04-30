// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Combined retrieval over triplet graph and episodic memory.
//!
//! Merges BFS graph expansion with fingerprint-based episode recall
//! to produce unified context for LLM agent prompts.

use std::collections::{HashMap, HashSet};

use super::episodic::EpisodicMemory;
use super::triplet_graph::TripletGraph;

// ============================================================================
// Production prompts — transcoded verbatim from Python AriGraph prompts.py
// ============================================================================

/// Production prompt for extracting triplets from observations.
///
/// Transcoded from Python `prompt_extraction_current` (prompts.py:35-60).
/// 590 words of detailed instructions with examples and edge cases.
pub const EXTRACTION_PROMPT: &str = r#"Objective: The main goal is to meticulously gather information from input text and organize this data into a clear, structured knowledge graph.

Guidelines for Building the Knowledge Graph:

Creating Nodes and Triplets: Nodes should depict entities or concepts, similar to Wikipedia nodes. Use a structured triplet format to capture data, as follows: "subject, relation, object". For example, from "Albert Einstein, born in Germany, is known for developing the theory of relativity," extract "Albert Einstein, country of birth, Germany; Albert Einstein, developed, Theory of Relativity."
Remember that you should break complex triplets like "John, position, engineer in Google" into simple triplets like "John, position, engineer", "John, work at, Google".
Length of your triplet should not be more than 7 words. You should extract only concrete knowledges, any assumptions must be described as hypothesis.
For example, from phrase "John have scored many points and potentially will be winner" you should extract "John, scored many, points; John, could be, winner" and should not extract "John, will be, winner".
Remember that object and subject must be an atomary units while relation can be more complex and long.
If observation states that you take item, the triplet should be: 'item, is in, inventory' and nothing else.

Do not miss important information. If observation is 'book involves story about knight, who needs to kill a dragon', triplets should be 'book, involves, knight', 'knight, needs to kill, dragon'. If observation involves some type of notes, do not forget to include triplets about entities this note includes.
There could be connections between distinct parts of observations. For example if there is information in the beginning of the observation that you are in location, and in the end it states that there is an exit to the east, you should extract triplet: 'location, has exit, east'.
Several triplets can be extracted, that contain information about the same node. For example 'kitchen, contains, apple', 'kitchen, contains, table', 'apple, is on, table'. Do not miss this type of connections.
Other examples of triplets: 'room z, contains, black locker'; 'room x, has exit, east', 'apple, is on, table', 'key, is in, locker', 'apple, to be, grilled', 'potato, to be, sliced', 'stove, used for, frying', 'recipe, requires, green apple', 'recipe, requires, potato'.
Do not include triplets that state the current location of an agent like 'you, are in, location'.
Do not use 'none' as one of the entities.
If there is information that you read something, do not forget to include triplets that state that entity that you read contains information that you extract.

Example of triplets you have extracted before: {example}

Observation: {observation}

Remember that triplets must be extracted in format: "subject_1, relation_1, object_1; subject_2, relation_2, object_2; ..."

Extracted triplets: "#;

/// Production prompt for identifying outdated/contradicted triplets.
///
/// Transcoded from Python `prompt_refining_items` (prompts.py:1-33).
/// 400 words with 4 worked examples showing when to replace and when NOT to.
pub const REFINING_PROMPT: &str = r#"You will be provided with list of existing triplets and list of new triplets. Triplets are in the following format: "subject, relation, object".
The triplets denote facts about the environment. Some triplets from the list of existing triplets can be replaced with one of the new triplets. For example, the item was taken from the locker and the existing triplet "item, is in, locker" should be replaced with the new triplet "item, is in, inventory".

Sometimes there are no triplets to replace:
Example of existing triplets: "Golden locker, state, open"; "Room K, is west of, Room I"; "Room K, has exit, east".
Example of new triplets: "Room T, is north of, Room N"; "Room T, has exit, south".
Example of replacing: []. Nothing to replace here.

Sometimes several triplets can be replaced with one:
Example of existing triplets: "kitchen, contains, broom"; "broom, is on, floor".
Example of new triplets: "broom, is in, inventory".
Example of replacing: [["kitchen, contains, broom" -> "broom, is in, inventory"], ["broom, is on, floor" -> "broom, is in, inventory"]]. Because broom changed location.

Ensure that triplets are only replaced if they contain redundant or conflicting information about the same aspect of an entity. Triplets should not be replaced if they provide distinct or complementary information. If there is uncertainty, prioritize retaining the existing triplet.
Example of existing triplets: "apple, to be, cooked", "knife, used for, cutting", "apple, has been, sliced"
Example of new triplets: "apple, is on, table", "kitchen, contains, knife", "apple, has been, grilled".
Example of replacing: []. Nothing to replace here. These describe different properties.

Another example:
Example of existing triplets: "brush, used for, painting".
Example of new triplets: "brush, is in, art class".
Example of replacing: []. Nothing to replace. Different properties.

Do not replace triplets if they carry different type of information about entities! It is better to leave a triplet than to replace one that has important information.
If you find a triplet in Existing triplets which semantically duplicates some triplet in New triplets, replace it. However do not replace if they refer to different things.
####

Generate only replacing, no descriptions are needed.
Existing triplets: {ex_triplets}.
New triplets: {new_triplets}.
####
Warning! Replacing must be generated strictly in following format: [[outdated_triplet_1 -> actual_triplet_1], [outdated_triplet_2 -> actual_triplet_2], ...], you MUST NOT include any descriptions in answer.
Replacing: "#;

/// Prompt for generating plans from retrieved context.
pub const PLAN_PROMPT: &str = r#"You are a planning system. Given the current context from a knowledge graph and episodic memory, generate a plan.

Graph context (known facts):
{graph_context}

Episodic context (past experiences):
{episodic_context}

Current observation: {observation}

Plan:"#;

/// System prompt for the plan agent.
///
/// Transcoded from Python `system_plan_agent` (system_prompts.py:9-47).
pub const SYSTEM_PLAN_AGENT: &str = r#"You are a planner within an agent system. Your role is to create a concise plan to achieve your main goal or modify your current plan based on new information received.
Make sure your sub-goals will benefit the achievement of your main goal. If your main goal is an ongoing complex process, also put sub-goals that can immediately benefit achieving something from your main goal.
If you need to find something, put it into sub-goal.
If you wish to alter or delete a sub-goal within the current plan, confirm that this sub-goal has been achieved according to the current observation or is no longer relevant. Until then do not change wording in "sub_goal" elements.
Pay attention to your inventory, what items you are carrying, when setting the sub-goals.
Pay attention to information from your memory module, it is important.
There should always be at least one sub-goal.

Write your answer exactly in this json format:
{
  "main_goal": "...",
  "plan_steps": [
    {"sub_goal_1": "...", "reason": "..."},
    {"sub_goal_2": "...", "reason": "..."}
  ],
  "your_emotion": {"your_current_emotion": "emotion", "reason_behind_emotion": "..."}
}

Do not write anything else."#;

/// System prompt for the action selection agent.
///
/// Transcoded from Python `system_action_agent_sub_expl` (system_prompts.py:50-65).
pub const SYSTEM_ACTION_AGENT: &str = r#"You are an action selector within an agent system. Your role involves receiving information about an agent and the state of the environment alongside a list of possible actions.
Your primary objective is to choose an action from the list of possible actions that aligns with the goals outlined in the plan, giving precedence to main goal or sub-goals in the order they appear.
Prioritize sub-goals that can be solved by performing single action in current situation, like 'take something', over long term sub-goals.
In tasks centered around exploration or locating something, prioritize actions that guide the agent to previously unexplored areas.
Performing same action typically will not provide different results, so if you are stuck, try to perform other actions.
You may choose actions only from the list of possible actions. You must choose strictly one action.

Write your answer exactly in this json format:
{
  "reason_for_action": "reason",
  "action_to_take": "selected action"
}

Do not write anything else."#;

/// Prompt for determining if a plan requires exploration.
///
/// Transcoded from Python `if_exp_prompt` (system_prompts.py:6-7).
pub const EXPLORATION_CHECK_PROMPT: &str = r#"You will be provided with sub-goals and reasons for it from plan of an agent. Your task is to state if this sub goals require exploration of the environment, finding or locating something.
Answer with just True or False."#;

/// Reflex prompt for learning from mistakes.
///
/// Transcoded from Python `reflex_prompt` (prompts.py:127-132).
pub const REFLEX_PROMPT: &str = r#"You are a learner in a system of AI agents. Your task is to find useful patterns in observations and explain it for future usage. Namely, you should find the inefficiency in previous behaviour and the patterns that can help to avoid this inefficiency.
Your answer must be brief and accurate and contain only three sentences.
####
{for_reflex}
####
Your answer: "#;

/// Configuration for the retrieval pipeline.
#[derive(Debug, Clone)]
pub struct RetrievalConfig {
    /// Maximum BFS depth for graph association retrieval.
    pub max_depth: usize,
    /// Maximum number of graph triplets to return.
    pub top_k: usize,
    /// Minimum truth expectation for including a triplet.
    pub threshold: f32,
    /// Number of episodic memory entries to retrieve.
    pub episodic_k: usize,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            max_depth: 2,
            top_k: 20,
            threshold: 0.4,
            episodic_k: 5,
        }
    }
}

/// Result of a combined retrieval query.
#[derive(Debug, Clone)]
pub struct RetrievalResult {
    /// Triplet strings from the knowledge graph.
    pub graph_context: Vec<String>,
    /// Observation strings from episodic memory.
    pub episodic_context: Vec<String>,
    /// All entities found during retrieval.
    pub entities: HashSet<String>,
}

/// Combines triplet graph BFS with episodic memory retrieval.
///
/// Given seed entities (from a query or observation), expands through the
/// knowledge graph via BFS and retrieves similar episodes by fingerprint
/// distance.
pub struct OsintRetriever<'a> {
    /// Reference to the knowledge graph.
    graph: &'a TripletGraph,
    /// Reference to the episodic memory.
    memory: &'a EpisodicMemory,
    /// Retrieval configuration.
    config: RetrievalConfig,
}

impl<'a> OsintRetriever<'a> {
    /// Create a new retriever with default configuration.
    pub fn new(graph: &'a TripletGraph, memory: &'a EpisodicMemory) -> Self {
        Self {
            graph,
            memory,
            config: RetrievalConfig::default(),
        }
    }

    /// Create a new retriever with custom configuration.
    pub fn with_config(
        graph: &'a TripletGraph,
        memory: &'a EpisodicMemory,
        config: RetrievalConfig,
    ) -> Self {
        Self {
            graph,
            memory,
            config,
        }
    }

    /// Retrieve combined context from graph and episodic memory.
    ///
    /// 1. Expands seed entities through the graph via BFS up to `max_depth`.
    /// 2. Filters triplets by truth expectation threshold.
    /// 3. Retrieves top-k similar episodes from memory.
    /// 4. Returns the merged result.
    pub fn retrieve(&self, seed_entities: &HashSet<String>, query: &str) -> RetrievalResult {
        // Graph BFS retrieval
        let associated = self
            .graph
            .get_associated(seed_entities, self.config.max_depth);

        let mut graph_context: Vec<String> = Vec::new();
        let mut entities: HashSet<String> = seed_entities.clone();

        for triplet in &associated {
            if triplet.truth.expectation() < self.config.threshold {
                continue;
            }
            graph_context.push(triplet.to_string_repr());
            entities.insert(triplet.subject.clone());
            entities.insert(triplet.object.clone());

            if graph_context.len() >= self.config.top_k {
                break;
            }
        }

        // Episodic memory retrieval
        let episodes = self.memory.top_k(query, self.config.episodic_k);
        let episodic_context: Vec<String> = episodes
            .into_iter()
            .map(|ep| ep.observation.clone())
            .collect();

        RetrievalResult {
            graph_context,
            episodic_context,
            entities,
        }
    }

    /// Retrieve with per-entity depth tracking.
    ///
    /// Returns the standard retrieval result plus a map from each entity
    /// to the BFS depth at which it was first encountered.
    pub fn retrieve_with_depths(
        &self,
        seed_entities: &HashSet<String>,
        query: &str,
    ) -> (RetrievalResult, HashMap<String, usize>) {
        let mut depth_map: HashMap<String, usize> = HashMap::new();

        // Track depths during manual BFS
        let mut current: HashSet<String> = seed_entities.iter().map(|e| e.to_lowercase()).collect();
        for entity in &current {
            depth_map.insert(entity.clone(), 0);
        }

        let mut all_graph_context: Vec<String> = Vec::new();
        let mut all_entities: HashSet<String> = seed_entities.clone();
        let mut seen_indices: HashSet<usize> = HashSet::new();

        for depth in 0..self.config.max_depth {
            let mut next: HashSet<String> = HashSet::new();

            for entity in &current {
                if let Some(indices) = self.graph.entity_index.get(entity) {
                    for &idx in indices {
                        if seen_indices.contains(&idx) {
                            continue;
                        }
                        let triplet = &self.graph.triplets[idx];
                        if triplet.is_deleted() {
                            continue;
                        }
                        if triplet.truth.expectation() < self.config.threshold {
                            continue;
                        }
                        seen_indices.insert(idx);

                        all_graph_context.push(triplet.to_string_repr());
                        all_entities.insert(triplet.subject.clone());
                        all_entities.insert(triplet.object.clone());

                        let subj_lower = triplet.subject.to_lowercase();
                        let obj_lower = triplet.object.to_lowercase();
                        let other = if subj_lower == *entity {
                            &obj_lower
                        } else {
                            &subj_lower
                        };

                        if other != "itself" && !depth_map.contains_key(other) {
                            depth_map.insert(other.clone(), depth + 1);
                            next.insert(other.clone());
                        }
                    }
                }
            }

            current = next;

            if all_graph_context.len() >= self.config.top_k {
                all_graph_context.truncate(self.config.top_k);
                break;
            }
        }

        // Episodic memory retrieval
        let episodes = self.memory.top_k(query, self.config.episodic_k);
        let episodic_context: Vec<String> = episodes
            .into_iter()
            .map(|ep| ep.observation.clone())
            .collect();

        let result = RetrievalResult {
            graph_context: all_graph_context,
            episodic_context,
            entities: all_entities,
        };

        (result, depth_map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::arigraph::triplet_graph::Triplet;

    fn setup() -> (TripletGraph, EpisodicMemory) {
        let mut graph = TripletGraph::new();
        graph.add_triplets(&[
            Triplet::new("alice", "bob", "knows", 1),
            Triplet::new("bob", "carol", "knows", 2),
            Triplet::new("carol", "dave", "knows", 3),
        ]);

        let mut memory = EpisodicMemory::new(10);
        memory.add(
            "Alice met Bob at the park",
            &["alice - knows - bob".to_string()],
            1,
        );
        memory.add(
            "Carol introduced Dave to the group",
            &["carol - knows - dave".to_string()],
            2,
        );

        (graph, memory)
    }

    #[test]
    fn test_combined_retrieval() {
        let (graph, memory) = setup();
        let retriever = OsintRetriever::new(&graph, &memory);

        let seed: HashSet<String> = ["alice".to_string()].into_iter().collect();
        let result = retriever.retrieve(&seed, "alice and friends");

        assert!(!result.graph_context.is_empty());
        assert!(!result.episodic_context.is_empty());
        assert!(result.entities.contains("alice"));
        assert!(result.entities.contains("bob"));
    }

    #[test]
    fn test_empty_graph_retrieval() {
        let graph = TripletGraph::new();
        let memory = EpisodicMemory::new(10);
        let retriever = OsintRetriever::new(&graph, &memory);

        let seed: HashSet<String> = ["nobody".to_string()].into_iter().collect();
        let result = retriever.retrieve(&seed, "test");

        assert!(result.graph_context.is_empty());
        assert!(result.episodic_context.is_empty());
    }

    #[test]
    fn test_config_defaults() {
        let config = RetrievalConfig::default();
        assert_eq!(config.max_depth, 2);
        assert_eq!(config.top_k, 20);
        assert!((config.threshold - 0.4).abs() < f32::EPSILON);
        assert_eq!(config.episodic_k, 5);
    }

    #[test]
    fn test_retrieve_with_depths() {
        let (graph, memory) = setup();
        let retriever = OsintRetriever::new(&graph, &memory);

        let seed: HashSet<String> = ["alice".to_string()].into_iter().collect();
        let (result, depths) = retriever.retrieve_with_depths(&seed, "alice network");

        assert!(!result.graph_context.is_empty());
        assert_eq!(*depths.get("alice").unwrap(), 0);
        // bob is 1 hop from alice
        assert_eq!(*depths.get("bob").unwrap(), 1);
    }

    #[test]
    fn test_custom_config() {
        let (graph, memory) = setup();
        let config = RetrievalConfig {
            max_depth: 1,
            top_k: 1,
            threshold: 0.0,
            episodic_k: 1,
        };
        let retriever = OsintRetriever::with_config(&graph, &memory, config);

        let seed: HashSet<String> = ["alice".to_string()].into_iter().collect();
        let result = retriever.retrieve(&seed, "alice");

        // top_k=1 limits graph context
        assert!(result.graph_context.len() <= 1);
        // episodic_k=1 limits episodic context
        assert!(result.episodic_context.len() <= 1);
    }
}
