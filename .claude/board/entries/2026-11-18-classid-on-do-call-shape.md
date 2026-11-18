# CLASSID-ON-DO-1: the classid is the call; knowledge document for every session (2026-11-18)

**Status:** CURRENT-CONTRACT for the addressing reading; the call-shape goal is
a WORKING-MODEL with nothing measured.

`.claude/knowledge/classid-on-do-call-shape.md` connects D-DT64-7 (behaviour
addressing and registration) to consumers, from the owner's statements of
2026-11-18, quoted there verbatim:

- A classid's left side is ON, its right `0xF###` is DO; `0xFFFF` is the global
  algebra primitive. WHEN is what `register` supplies.
- The `BindStore` is the registry: DT is the class view, DR is the field mask.
- No foreign key: a relation is a class view, and the intake declares the
  reading before any query.
- Goal: behaviour called like a typed SIMD primitive, `[ON]::[DO]()`, over a
  bag the cube lends. The lending half exists (`LaneBinding::borrow_all`, and
  a method executed over it in `foundry-parity-lab/src/method.rs`); the cube
  implements `RelationSource` only, so its bag is not lendable through it yet.

It also lists eight mistakes one session made the same day, each with the check that would have caught it. Listed in `CLAUDE.md`
§ Knowledge Base as mandatory before proposing a registry, binding a consumer,
or writing a join.

Open: whether `MethodBinding.on` is redundant; the effect class on the binding
and the call-parameter channel (EXO-291); `EqU32Via` / `Semijoin` admission
(EXO-140); cube sides filtering on a property (EXO-327).
