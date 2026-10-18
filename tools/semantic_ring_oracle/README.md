# C-01/C-02 semantic ring reference oracle

**Test-only independent oracle. Not the Rust semantic candidate runtime.**

This experiment is a versioned identity assertion, not a newly minted
'witness object'. The four bound SPO relations live at fixed symbolic
identities. The identity assertion itself has a **fixed claim address**
`corefers(last_sentence.key,first_sentence.key)`. Its **sealed version** is
the witness of its admission:

- C-02 reads `(symbolic-ring, main, v16)`: the claim does not yet exist.
  Four sentences look cyclic, but their distinct key identities make the
  cycle open.
- C-01 reads `(symbolic-ring, main, v17)`: the exact same semantic identities
  are read; version 17 has witnessed the added identity assertion, and the
  cycle closes in all four rotations.
- Re-reading C-02 at v17 closes the cycle **without reminting any identity**.
  An older horizon stays unresolved even if its narrative says a witness
  exists. Version visibility is not inferred from a source-root label.

This directly follows the D-TGA decision in
`.claude/plans/thought-git-api-v1.md`: a sealed version is the PR entry;
`blame` returns a landing version plus original source bits. The active
thought slot `0..63` is recyclable. The history witness is the claim's
**thought address at the sealed version**, not a reusable slot number, an
external pointer, a new witness NodeGuid or a bare numeric Lance version.
The tuple in this **symbolic** fixture includes dataset and branch because
real durable versions need those to disambiguate identity.

**Evidence root != witness coordinate.** Independent source roots are
metadata in the version receipt for NAL/Gadamer revision. Three thoughts
copying one observation do not yield three independent sources, and neither
a version number nor an address bit alone proves a claim *true*. The fixture
contains a synthetic source root, not external proof of real-world coreference.

The preregistered C-01/C-02 source JSON in
`docs/research/versioned-semantic-candidates-fixtures.json` remains unchanged.
Its older `external_identity_witness` field is retained for traceability
but is **not** the executable witness index. The new v2 gold is held in
`docs/research/semantic-ring-bound-gold-v1.json` (original path retained),
with explicitly pinned read horizons and a sealed claim-address receipt.

```sh
python3 tools/semantic_ring_oracle/ring_reference.py
python3 -m unittest discover -s tools/semantic_ring_oracle -p 'test_*.py' -v
```

The reference never consults `case.expected` during evaluation. Controls
include four rotations, pre/post version, wrong dataset/branch, spurious
root labels, wrong claim address, missing version, predicate mutation,
truncated facts and source-root de-duplication.

**Next integration gate:** have the actual DeepNSM/UD binder produce
canonical bound claim addresses and their stamped, versioned thought writes
(or `Unbound`), then differential-check real mask/R2IL execution against this
reference. No parser, real-world evidence authentication, NodeGuid allocation,
Lance13 integration, generalized relational join, causal inference or
production C-01/C-02 semantic closure is claimed here.
