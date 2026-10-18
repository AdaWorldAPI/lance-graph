# C-01/C-02 semantic ring reference oracle

**Scope: test-only independent oracle, not the Rust semantic candidate runtime.**

The pre-registered C-01 and C-02 fixtures in
`docs/research/versioned-semantic-candidates-fixtures.json` make a strong
claim: rotating a four-sentence ring does not change its justified canonical
relations, but the ring must **not close** unless the last indefinite key is
identified with the first by an explicit, independent witness.

`docs/research/semantic-ring-bound-gold-v1.json` adds manually annotated
subject/predicate/object bindings. The first and last key remain **different
identities**. The fixture's synthetic attestation can be activated only when
its exact root and endpoints match the registered gold record. Grammar
parsing, NodeGuid/codebook minting and witness verification against a real
source are deliberately **out of scope**. This avoids treating the English
word `key` as a foreign-key join or inventing coreference.

```sh
python3 tools/semantic_ring_oracle/ring_reference.py
python3 -m unittest discover -s tools/semantic_ring_oracle -p 'test_*.py' -v
```

The oracle does not read `case.expected` during evaluation. It checks four
rotations, negative closure, spoofed/misdirected witnesses, missing claims,
duplicate sentence bindings, and a changed predicate that preserves
unlabeled graph topology while changing the semantic claim set.

**Next integration gate:** have a real DeepNSM/UD binder emit version-pinned
canonical bound relations (or an explicit `Unbound`) with source-root receipts;
compare its output and downstream R2IL/Quack selection against this reference.
A test passing here certifies only the reference fixtures, **not** parsing,
coreference accuracy, semantic alignment, actual independent evidence, a
foreign-address join, or production C-01/C-02 behavior. Do not import the
Python oracle into a production execution path or lower semantic equality to
an unverified mask AND.
