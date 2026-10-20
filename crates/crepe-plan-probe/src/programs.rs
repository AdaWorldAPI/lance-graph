//! The fixture programs. Each is written once as `crepe_plan!` (the plan this
//! probe optimizes) and, where it is used as an oracle, once more as the
//! unmodified `crepe!` evaluator with the same rules.

/// `Answer(x, z)`: grandchildren of `x` through an allowed middle generation,
/// plus a sponsor branch that only ever answers for `x = 5`.
pub mod family {
    use crate::plan::*;
    use crepe::crepe_plan;
    crepe_plan! {
        @input
        struct Parent(u32, u32);
        @input
        struct Allowed(u32);
        @input
        struct Sponsor(u32);
        @output
        struct Answer(u32, u32);
        Answer(x, z) <- Parent(x, y), Parent(y, z), Allowed(y);
        Answer(5, z) <- Sponsor(z);
    }

    /// The same rules through crepe's own evaluator (test oracle only).
    pub mod oracle {
        use crepe::crepe;
        crepe! {
            @input
            pub struct Parent(pub u32, pub u32);
            @input
            pub struct Allowed(pub u32);
            @input
            pub struct Sponsor(pub u32);
            @output
            pub struct Answer(pub u32, pub u32);
            Answer(x, z) <- Parent(x, y), Parent(y, z), Allowed(y);
            Answer(5, z) <- Sponsor(z);
        }
    }
}

/// The `family` program with renamed variables and a duplicated atom and rule.
pub mod family_noisy {
    use crate::plan::*;
    use crepe::crepe_plan;
    crepe_plan! {
        @input
        struct Parent(u32, u32);
        @input
        struct Allowed(u32);
        @input
        struct Sponsor(u32);
        @output
        struct Answer(u32, u32);
        Answer(a, c) <- Parent(a, b), Allowed(b), Parent(b, c), Allowed(b);
        Answer(p, q) <- Parent(p, m), Allowed(m), Parent(m, q);
        Answer(5, w) <- Sponsor(w);
    }
}

/// The two-relation join MORK's oracle answers: `T(x, z) <- R(x, y), S(y, z)`.
pub mod two_way {
    use crate::plan::*;
    use crepe::crepe_plan;
    crepe_plan! {
        @input
        struct R(u32, u32);
        @input
        struct S(u32, u32);
        @output
        struct T(u32, u32, u32);
        T(x, y, z) <- R(x, y), S(y, z);
    }
}

/// A unary star `R(y), S(y), Q(y)`.
pub mod star3 {
    use crate::plan::*;
    use crepe::crepe_plan;
    crepe_plan! {
        @input
        struct R(u32);
        @input
        struct S(u32);
        @input
        struct Q(u32);
        @output
        struct Hit(u32);
        Hit(y) <- R(y), S(y), Q(y);
    }
}

/// Shapes the spike refuses.
pub mod refused {
    use crate::plan::*;
    use crepe::crepe_plan;
    crepe_plan! {
        @input
        struct E(u32, u32);
        @input
        struct B(u32);
        @output
        struct Chain(u32, u32);
        @output
        struct Tri(u32);
        @output
        struct Loop(u32);
        @output
        struct Neg(u32);
        @output
        struct Reach(u32, u32);
        Chain(a, d) <- E(a, b), E(b, c), E(c, d);
        Tri(a) <- E(a, b), E(b, c), E(c, a);
        Loop(a) <- E(a, a);
        Neg(a) <- E(a, _), !B(a);
        Reach(a, b) <- E(a, b);
        Reach(a, c) <- E(a, b), Reach(b, c);
    }
}
