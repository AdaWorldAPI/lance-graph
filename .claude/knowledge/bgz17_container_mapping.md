# bgz17 × Container 0: Word-by-Word Mapping

## The Container

Container 0 is 256 words × 64 bits = 2KB. It IS the BitVec `[u64; 256]`.
It IS the `FixedSizeBinary(2048)` in cognitive_nodes.lance.
It IS the `Fingerprint` type in spo/builder.rs.

When the Cascade does `words_hamming_sampled(query, candidate, 16)`,
it's sampling THIS container — hitting W0, W16, W32, W48... at stride 16.
When HHTL extracts the scent byte, it's reading the Boolean lattice of
THIS container's Hamming profile against another container.

bgz17 doesn't replace the container. It provides a COMPRESSED VIEW of it.

## Word Map: What bgz17 Can and Cannot Encode

```
WORD RANGE    FIELD                    bgz17 VIEW          ACTION
───────────────────────────────────────────────────────────────────

W0            DN address               content_hash         Base17 CANNOT encode this.
              (identity)               (u64 from dims)      But hash(Base17) approximates
                                                            identity for dedup. Not same
                                                            as DN path — DN is structural,
                                                            Base17 is perceptual.
                                                            KEEP W0 as-is.

W1            node_kind|flags|         NOT ENCODED          Discrete metadata. Not a 
              schema_version|                               signal — a label. bgz17
              provenance_hash                               encodes continuous signals.
                                                            KEEP W1 as-is.

W2            created_ms|modified_ms   temporal_quantile    Base17 averages across octaves,
                                       (u8, 1 byte)        destroying temporal info.
                                                            The u8 quantile in bgz17
                                                            captures RELATIVE time only.
                                                            KEEP W2 for absolute time.
                                                            ADD temporal_quantile for
                                                            relative ordering in scope.

W3            label_hash|tree_depth    NOT ENCODED          Structural position in DN tree.
                                                            Not a signal. KEEP W3 as-is.

───────────────────────────────────────────────────────────────────
W4-7          NARS truth values        wisdom_freq/conf     Base17 CANNOT encode truth
              freq, conf, evidence,    (f32 × 2)            values — they're independent
              horizon                                       evidence, not geometry.
                                                            truth.rs TruthValue IS the
                                                            encoding. TruthGate IS the
                                                            filter. NARS revision rule
                                                            IS the update. All exist.
                                                            
                                                            bgz17 can DERIVE approximate
                                                            freq from palette_distance
                                                            and conf from sign_agreement,
                                                            but these are INFERIOR to the
                                                            stored NARS values which
                                                            accumulate across encounters.
                                                            
                                                            KEEP W4-7 as-is.
                                                            DO NOT derive from bgz17.

───────────────────────────────────────────────────────────────────
W8-11         DN rung | gate_state     NOT ENCODED          Cognitive depth markers.
                                                            Discrete state machine, not
                                                            continuous signal.
                                                            KEEP W8-11 as-is.

W12-15        7-layer markers          CAN PREFETCH         Per-layer activation could
                                                            correlate with palette distance
                                                            per layer. But the 7 layers
                                                            are a cognitive architecture
                                                            choice, not a signal property.
                                                            KEEP W12-15 as-is.
                                                            bgz17 can use these as
                                                            ROUTING hints (which layer
                                                            is active → which precision).

───────────────────────────────────────────────────────────────────
W16-31        INLINE EDGES             ★ THIS IS THE        64 edges × (verb:8 + target:8)
              64 × (verb + target)     bgz17 SWEET SPOT     packed into 16 words.
                                                            
                                       Each edge is 16 bits: 
                                         verb(8) = relationship type
                                         target(8) = target node index
                                       
                                       bgz17 palette index is ALSO
                                       8 bits per plane. The palette
                                       IS the verb vocabulary:
                                         palette_p = predicate archetype
                                                   ≈ verb class
                                         palette_o = object archetype
                                                   ≈ target class
                                       
                                       The 256×256 distance matrix
                                       between palette entries IS the
                                       adjacency strength between
                                       verb classes × target classes.
                                       
                                       ACTION: bgz17 provides the
                                       DISTANCE METRIC for the inline
                                       edges. W16-31 stores the edges.
                                       bgz17 tells you how SIMILAR
                                       any two edges are without
                                       loading full planes.
                                       
                                       The compose_table[verb_a][verb_b]
                                       tells you what 2-hop traversal
                                       through these inline edges means.

───────────────────────────────────────────────────────────────────
W32-39        RL / Q-values / rewards  NOT ENCODED          Learning state. Changes with
                                                            policy updates, not with
                                                            content. bgz17 encodes content.
                                                            KEEP W32-39 as-is.

───────────────────────────────────────────────────────────────────
W40-47        Bloom filter (512 bits)  PARTIALLY REDUNDANT  The Bloom filter tests 
                                       WITH PALETTE          "have I seen this entity?"
                                                            
                                       The palette index tests
                                       "is this entity close to
                                       archetype k?" — different
                                       question but overlapping use.
                                       
                                       Bloom: exact membership, O(1),
                                       false positives possible.
                                       Palette: approximate membership
                                       by archetype, O(1), no false
                                       positives but loses identity.
                                       
                                       KEEP W40-47.
                                       Bloom answers "seen before?"
                                       Palette answers "similar to?"
                                       Both needed.

W48-55        Graph metrics            CAN DERIVE SOME      Degree = edge_count in scope.
              (degree, centrality)                           Centrality computable from
                                                            palette_csr archetype graph.
                                                            But stored metrics are EXACT.
                                                            bgz17 metrics are approximate.
                                                            KEEP W48-55 for exact values.
                                                            bgz17 provides fast ESTIMATES
                                                            for routing decisions.

───────────────────────────────────────────────────────────────────
W56-63        Qualia                   ★ THIS IS bgz17's   18 qualia channels × f16.
              18 channels × f16        GROUNDING LAYER       
              + 8 spare slots                               Base17 has 17 dimensions.
                                                            18 qualia channels ≈ 17+1.
                                                            
                                       The 17 base dimensions
                                       ARE the qualia channels
                                       (minus one for the DC/mean).
                                       
                                       Base17.dims[0..16] maps to
                                       qualia channels 0..16.
                                       The 18th channel is the
                                       overall magnitude (= octave
                                       envelope mean).
                                       
                                       i16 fixed-point (×256) in
                                       Base17 has MORE precision
                                       than f16 in the container
                                       for values in [-128, +127].
                                       
                                       ACTION: Base17 IS the
                                       compressed qualia state.
                                       W56-63 stores the full
                                       f16 qualia for cold path.
                                       Base17 serves the hot path.
                                       
                                       The palette_s/p/o indices
                                       are QUALIA ARCHETYPES:
                                       "this node's subject aspect
                                       is qualia pattern #42."

───────────────────────────────────────────────────────────────────
W64-79        Rung history +           NOT ENCODED          Temporal reasoning trace.
              collapse gate                                 Discrete state machine.
                                                            KEEP as-is.

W80-95        Representation           NOT ENCODED          Language/encoding metadata.
              descriptor                                    Discrete. KEEP as-is.

───────────────────────────────────────────────────────────────────
W96-111       DN-Sparse adjacency      ★ RESERVED FOR       This is EXACTLY where
              (RESERVED)               PALETTE CSR           the 256×256 palette
                                                            distance matrix or the
                                                            compose table belongs.
                                                            
                                       16 words = 128 bytes = 
                                       enough for a 11×11 u8
                                       distance matrix or a
                                       sparse subset of the
                                       full 256×256.
                                       
                                       OR: 16 words = 64 palette
                                       edge entries (verb:8 + 
                                       palette_target:8 per edge).
                                       Same format as W16-31 but
                                       with palette indices instead
                                       of raw target IDs.
                                       
                                       ACTION: When implemented,
                                       W96-111 stores the LOCAL
                                       palette CSR — this node's
                                       view of its archetype
                                       neighborhood.

───────────────────────────────────────────────────────────────────
W112-125      Reserved                 bgz17 ANNEX          14 words = 112 bytes.
                                                            
                                       Base17 S+P+O = 102 bytes.
                                       Fits in 13 words (104 bytes).
                                       Leaves 1 word spare.
                                       
                                       W112-124: base17_s (34B) +
                                                 base17_p (34B) +
                                                 base17_o (34B) = 102B
                                       W125:     palette_s(8) +
                                                 palette_p(8) +
                                                 palette_o(8) +
                                                 temporal_q(8) +
                                                 spare(32) = 8B
                                       
                                       ACTION: Store Base17 + palette
                                       indices in the RESERVED words.
                                       No schema change needed — they
                                       live inside the existing 2KB
                                       container.

W126-127      Checksum + version       KEEP AS-IS           Integrity verification.
                                                            The checksum covers ALL
                                                            256 words including the
                                                            bgz17 annex in W112-125.

───────────────────────────────────────────────────────────────────

WIDE META W128-255 (CogRecord8K only):

W128-143      SPO Crystal              ★ bgz17 DECODE       8 packed SPO triples.
              (16 words)               TARGET                The crystallized output
                                                            of the accumulator.
                                                            
                                       Base17 IS the compressed
                                       crystal. Decode to fill
                                       W128-143 on demand.
                                       
                                       OR: store Base17 in
                                       W112-124 and derive
                                       W128-143 when needed.

W224-239      Extended edge overflow   PALETTE EDGES         64 more edges (128 total).
              (16 words)                                     Same format as W16-31.
                                                            bgz17 palette distances
                                                            apply to these edges too.

W144-159      Hybrid Crystal           DEFERRED             
W160-175      Extended NARS evidence   DEFERRED             
W176-191      Scent index              ★ bgz17 REPLACES     16 words = 128 bytes =
                                                            128 scent values OR
                                                            palette lookup indices
                                                            for 128 neighbors.
                                                            
                                       ACTION: Use for palette
                                       indices of top-128
                                       neighbors. Distance via
                                       matrix lookup.

W192-207      Causal graph             DEFERRED             
W208-223      10-Layer cognitive       DEFERRED             
W240-251      DN tree spine cache      DEFERRED             
W254          Wide checksum            KEEP                 
W255          Embedding format tag     KEEP                 
```

## Summary: What Goes Where

```
INSIDE THE EXISTING 2KB CONTAINER (no schema change):
  W112-124:  Base17 S+P+O (102 bytes, hot-path compressed encoding)
  W125:      palette_s + palette_p + palette_o + temporal_q (4 bytes)
  W96-111:   local palette CSR (when implemented)
  W176-191:  palette neighbor indices (replaces deferred scent index)

OUTSIDE THE CONTAINER (scope-level, amortized):
  scopes.lance: codebook_s/p/o (26KB) + dist_matrix_s/p/o (384KB)
  These are PER-SCOPE, not per-node. Loaded once, cached in L1.

STAYS EXACTLY AS DESIGNED:
  W0-3:     Identity + temporal + structural (not a signal)
  W4-7:     NARS truth values (independent evidence, not geometry)
  W8-15:    Cognitive state (discrete state machine, not continuous)
  W16-31:   Inline edges (bgz17 adds DISTANCE METRIC, not replacement)
  W32-39:   RL state (learning, not content)
  W40-47:   Bloom filter (exact membership, different from palette similarity)
  W48-55:   Graph metrics (exact, bgz17 provides fast estimates)
  W56-63:   Qualia channels (full f16, bgz17 Base17 = compressed version)
  W64-95:   Temporal trace + representation (discrete, not continuous)
  W126-127: Checksum + version (covers all words including bgz17 annex)
```

## The Cascade Sampling Insight

When `Cascade::query()` does `words_hamming_sampled(query, candidate, 16)`,
it samples every 16th word: W0, W16, W32, W48, W64, W80, W96, W112, ...

At stride 16, it hits:
```
W0   = DN address        (structural, high entropy → good discriminator)
W16  = first inline edge (topological, good discriminator)
W32  = first RL value    (learning state, medium discriminator)
W48  = first graph metric(degree/centrality, good discriminator)
W64  = first rung history(temporal, medium discriminator)
W80  = first repr desc   (encoding metadata, low discriminator)
W96  = DN-sparse / palette CSR (RESERVED → when filled, great discriminator)
W112 = Base17 word 0     (bgz17 compressed encoding → excellent discriminator)
...
```

The Cascade's 1/16 sample ALREADY hits W112 — the first word of the Base17 annex.
This means the Cascade automatically benefits from bgz17 without ANY code change.
Just filling W112-125 with Base17 data makes the Cascade's stage 1 more discriminating
because it now samples a word that captures the node's compressed perceptual identity.

At stride 4 (stage 2), it additionally hits W4, W8, W12, W20, W24, W28, ...
getting NARS truth, cognitive depth, more inline edges, and more Base17 words.

The Cascade doesn't need to know about bgz17. It just needs the words filled.

## The Scent Index (W176-191) Replacement

Currently deferred. The design reserved 16 words = 128 bytes for a "scent index."
With bgz17, this becomes 128 palette neighbor indices:

```
W176-191: [u8; 128] = palette indices for this node's 128 nearest neighbors
         + implicit distance via matrix[my_palette][neighbor_palette]
```

This is a LOCAL view of the scope's distance structure, stored per-node.
Combined with W16-31 (inline edges with target IDs), you can:
1. W16-31: find WHO my neighbors are (64 edge targets)
2. W176-191: find HOW SIMILAR they are (128 palette distances)
3. Matrix lookup: O(1) per neighbor, no plane loading needed

## What bgz17 Prefetches vs Recalculates

```
PREFETCH (read from container, no computation):
  W112-124: Base17 dims → immediate L1 distance computation
  W125[0:2]: palette indices → immediate matrix lookup
  W4-7: NARS truth → TruthGate filter (no derivation needed)
  W16-31: inline edges → neighbor identity

RECALCULATE (derived on the fly, never stored):
  scent byte: from palette_distance(my_palette, query_palette)
  ZeckF64: from zeckf64_from_base(my_base17, query_base17)
  Cascade sigma/mu: from scope-level palette distance distribution
  BNN activation: from sign bits of Base17 dims
  CLAM tree position: from palette CSR archetype membership

PROMOTE (hot → cold, async):
  wisdom updates to W4-7 (NARS revision when patterns recur)
  new edges to W16-31 or W224-239 (discovered relationships)
  graph metrics to W48-55 (recomputed after topology change)
```

## Cypher/NARS Cold Path: How Promoted Wisdom Joins

```
CYPHER:
  MATCH (a)-[r]->(b)
  WHERE b.truth_conf > 0.8 AND r.verb_class = 'KNOWS'
  RETURN a.name, b.name

EXECUTION:
  DataFusion reads cognitive_nodes.lance:
    → scans W4-7 of each container for truth_conf (columnar access via Arrow)
    → filter: conf > 0.8
  DataFusion reads W16-31 for inline edges:
    → filter: verb(8) matches KNOWS label_hash
    → join: target(8) → resolved via scope node_ids

  No hot path. No bgz17 computation. Pure column scan on stored metadata.
  The wisdom was promoted earlier by the async background task.
  DataFusion never sees Base17 or palette indices — those are hot-path only.
```

## What This Means for the Session Prompts

Sessions A-D stay as designed. One addition to Session B:

```
DELIVERABLE 6 (Session B): Container Annex Writer

fn write_bgz17_annex(container: &mut [u64; 256], base17: &SpoBase17, palette: &PaletteEdge) {
    // W112-124: pack Base17 S+P+O into 13 words (102 bytes → 104 bytes with padding)
    let bytes = base17.to_bytes(); // 102 bytes
    for i in 0..13 {
        let offset = i * 8;
        let end = (offset + 8).min(bytes.len());
        let mut word = [0u8; 8];
        word[..end-offset].copy_from_slice(&bytes[offset..end]);
        container[112 + i] = u64::from_le_bytes(word);
    }
    
    // W125: palette indices + temporal quantile packed into one word
    container[125] = (palette.s_idx as u64)
        | ((palette.p_idx as u64) << 8)
        | ((palette.o_idx as u64) << 16)
        | ((palette.temporal_quantile as u64) << 24);
}

fn read_bgz17_annex(container: &[u64; 256]) -> (SpoBase17, PaletteEdge) {
    let mut bytes = [0u8; 104];
    for i in 0..13 {
        bytes[i*8..(i+1)*8].copy_from_slice(&container[112 + i].to_le_bytes());
    }
    let base17 = SpoBase17::from_bytes(&bytes[..102]);
    
    let w125 = container[125];
    let palette = PaletteEdge {
        s_idx: w125 as u8,
        p_idx: (w125 >> 8) as u8,
        o_idx: (w125 >> 16) as u8,
    };
    let temporal_q = (w125 >> 24) as u8;
    
    (base17, palette)
}
```

This writes bgz17 data INTO the existing container at the RESERVED words.
No new Lance columns. No schema change. The 2KB container already has room.
The checksum at W126-127 automatically covers the bgz17 annex.
The Cascade's stride-16 sampling automatically hits W112 (first Base17 word).
