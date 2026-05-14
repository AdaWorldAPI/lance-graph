//! Query Layer - GQL Alchemy Syntax
//!
//! Supports both RedisGraph Cypher syntax and ISO GQL alchemy patterns
//! for vector-enhanced graph queries.
//!
//! # Query Styles
//!
//! ## RedisGraph Cypher (Compatible)
//! ```cypher
//! MATCH (n:Person)-[r:KNOWS]->(m:Person)
//! WHERE n.embedding ~> query_vector < 0.3
//! RETURN n, m
//! ```
//!
//! ## GQL Alchemy (Extended)
//! ```gql
//! FROM graph
//! MATCH (a)-[BIND verb]->(b)
//! WHERE RESONANCE(a, query) > 0.8
//! UNBIND a FROM edge USING verb
//! RETURN CLEANUP(result)
//! ```
//!
//! ## Vector Operations in Queries
//!
//! - `~>` : Hamming similarity operator
//! - `BIND(a, b)` : XOR binding
//! - `UNBIND(bound, key)` : XOR unbinding
//! - `RESONANCE(vec, query)` : Find best match in cleanup memory
//! - `CLEANUP(vec)` : Map noisy vector to clean concept

mod executor;
mod parser;
mod transpiler;

pub use executor::{QueryExecutor, QueryResult};
pub use parser::{NodePattern, QueryAst, QueryParser, RelationPattern, VectorOp};
pub use transpiler::{CypherTranspiler, GqlTranspiler};
