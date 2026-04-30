//! Arena-allocated expressions.

use super::Node;

/// Expression handle — points into the expression arena.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExprNode(pub Node);

/// Expression variants (arena-allocated, from Polars AExpr pattern).
#[derive(Debug, Clone)]
pub enum AExpr {
    /// Column reference: variable.property
    Column { variable: String, property: String },

    /// Literal value.
    Literal(Literal),

    /// Binary operation.
    BinaryOp {
        left: ExprNode,
        op: BinaryOp,
        right: ExprNode,
    },

    /// Unary operation.
    UnaryOp { op: UnaryOp, input: ExprNode },

    /// Function call.
    Function { name: String, args: Vec<ExprNode> },

    /// RESONATE(fingerprint, query, threshold) — first-class resonance query.
    Resonate {
        fingerprint: ExprNode,
        query: ExprNode,
        threshold: f64,
    },

    /// Hamming distance between two fingerprints.
    HammingDistance { left: ExprNode, right: ExprNode },

    /// NARS truth value expression.
    TruthValue { frequency: f64, confidence: f64 },

    /// Cast expression.
    Cast { input: ExprNode, to_type: DataType },

    /// CASE WHEN ... THEN ... ELSE ... END
    Case {
        conditions: Vec<(ExprNode, ExprNode)>,
        else_result: Option<ExprNode>,
    },

    /// Exists subquery.
    Exists { subquery: Node },

    /// Wildcard (*) — expanded during binding.
    Wildcard,

    /// Parameter placeholder ($name).
    Parameter { name: String },
}

/// Literal values.
#[derive(Debug, Clone)]
pub enum Literal {
    Null,
    Bool(bool),
    Int64(i64),
    Float64(f64),
    String(String),
    /// Binary fingerprint (Container).
    Fingerprint(Vec<u64>),
    List(Vec<Literal>),
    Map(Vec<(String, Literal)>),
}

/// Binary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    // Comparison
    Eq,
    Neq,
    Lt,
    Lte,
    Gt,
    Gte,
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    // Logical
    And,
    Or,
    // String
    Contains,
    StartsWith,
    EndsWith,
    // Bitwise (for fingerprints)
    Xor,
    BitwiseAnd,
    BitwiseOr,
}

/// Unary operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Negate,
    IsNull,
    IsNotNull,
    PopCount,
}

/// Data types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    Bool,
    Int64,
    Float64,
    String,
    /// Fixed-width binary fingerprint (N × u64).
    Fingerprint(usize),
    List(Box<DataType>),
    Node,
    Relationship,
    Path,
    TruthValue,
}
