# lance-graph-cognitive

crate: lance-graph-cognitive
path: crates/lance-graph-cognitive
workspace: excluded
digest: 93f956ac0d69ca66
evidence legend: ../index.md#evidence-legend

Generated; the code is the truth. See ../index.md.

## Depends on

- dep holograph normal — crates/lance-graph-cognitive/Cargo.toml:18 CONFIRMED
- dep lance-graph-contract normal — crates/lance-graph-cognitive/Cargo.toml:19 CONFIRMED
- dep ndarray normal — crates/lance-graph-cognitive/Cargo.toml:17 CONFIRMED
- dep serde normal optional — crates/lance-graph-cognitive/Cargo.toml:22 CONFIRMED

## Depended on by

- rdep deepnsm normal optional — crates/deepnsm/Cargo.toml:40 CONFIRMED
- rdep lance-graph normal optional — crates/lance-graph/Cargo.toml:98 CONFIRMED

## Public items

- struct PackedDn [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:34 CONFIRMED
- const PackedDn::MAX_DEPTH [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:38 CONFIRMED
- fn PackedDn::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:41 CONFIRMED
- const PackedDn::ROOT [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:51 CONFIRMED
- fn PackedDn::depth [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:54 CONFIRMED
- fn PackedDn::component [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:64 CONFIRMED
- fn PackedDn::parent [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:77 CONFIRMED
- fn PackedDn::child [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:88 CONFIRMED
- fn PackedDn::ancestors [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:98 CONFIRMED
- fn PackedDn::subtree_range [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:112 CONFIRMED
- fn PackedDn::is_ancestor_of [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:126 CONFIRMED
- fn PackedDn::common_depth [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:142 CONFIRMED
- fn PackedDn::hex [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:155 CONFIRMED
- fn PackedDn::from_hex [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:160 CONFIRMED
- fn PackedDn::raw [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:166 CONFIRMED
- fn PackedDn::from_path [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:177 CONFIRMED
- struct InlineEdge [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:233 CONFIRMED
- const InlineEdge::EMPTY [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:242 CONFIRMED
- fn InlineEdge::pack [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:249 CONFIRMED
- fn InlineEdge::unpack [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:255 CONFIRMED
- fn InlineEdge::is_empty [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:264 CONFIRMED
- struct EdgeDescriptor [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:283 CONFIRMED
- fn EdgeDescriptor::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:287 CONFIRMED
- fn EdgeDescriptor::verb_id [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:295 CONFIRMED
- fn EdgeDescriptor::weight [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:301 CONFIRMED
- fn EdgeDescriptor::weight_raw [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:307 CONFIRMED
- fn EdgeDescriptor::target_dn_low [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:313 CONFIRMED
- fn EdgeDescriptor::raw [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:319 CONFIRMED
- const EdgeDescriptor::EMPTY [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:324 CONFIRMED
- fn EdgeDescriptor::is_empty [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:326 CONFIRMED
- struct InlineEdgeView [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:336 CONFIRMED
- fn InlineEdgeView<'a>::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:342 CONFIRMED
- fn InlineEdgeView<'a>::get [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:348 CONFIRMED
- fn InlineEdgeView<'a>::count [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:357 CONFIRMED
- fn InlineEdgeView<'a>::iter [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:368 CONFIRMED
- fn InlineEdgeView<'a>::first_empty [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:376 CONFIRMED
- fn InlineEdgeView<'a>::find [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:386 CONFIRMED
- struct InlineEdgeIter [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:398 CONFIRMED
- struct InlineEdgeViewMut [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:424 CONFIRMED
- fn InlineEdgeViewMut<'a>::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:430 CONFIRMED
- fn InlineEdgeViewMut<'a>::get [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:436 CONFIRMED
- fn InlineEdgeViewMut<'a>::set [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:446 CONFIRMED
- fn InlineEdgeViewMut<'a>::add [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:455 CONFIRMED
- fn InlineEdgeViewMut<'a>::remove [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:466 CONFIRMED
- fn InlineEdgeViewMut<'a>::clear [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:478 CONFIRMED
- fn InlineEdgeViewMut<'a>::count [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:485 CONFIRMED
- const MAX_CSR_EDGES [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:513 CONFIRMED
- struct CsrOverflowView [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:516 CONFIRMED
- fn CsrOverflowView<'a>::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:521 CONFIRMED
- fn CsrOverflowView<'a>::edge_count [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:527 CONFIRMED
- fn CsrOverflowView<'a>::row_count [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:533 CONFIRMED
- fn CsrOverflowView<'a>::edge [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:539 CONFIRMED
- fn CsrOverflowView<'a>::edges [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:545 CONFIRMED
- struct CsrOverflowViewMut [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:555 CONFIRMED
- fn CsrOverflowViewMut<'a>::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:560 CONFIRMED
- fn CsrOverflowViewMut<'a>::set_edge_count [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:565 CONFIRMED
- fn CsrOverflowViewMut<'a>::set_row_count [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:570 CONFIRMED
- fn CsrOverflowViewMut<'a>::set_edge [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:576 CONFIRMED
- fn CsrOverflowViewMut<'a>::push_edge [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:582 CONFIRMED
- fn CsrOverflowViewMut<'a>::clear [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:593 CONFIRMED
- struct AdjacencyView [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:606 CONFIRMED
- fn AdjacencyView<'a>::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:611 CONFIRMED
- fn AdjacencyView<'a>::total_edges [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:616 CONFIRMED
- fn AdjacencyView<'a>::inline [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:623 CONFIRMED
- fn AdjacencyView<'a>::overflow [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/adjacency.rs:628 CONFIRMED
- enum CacheError [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:11 CONFIRMED
- struct ContainerCache [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:37 CONFIRMED
- fn ContainerCache::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:53 CONFIRMED
- fn ContainerCache::len [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:65 CONFIRMED
- fn ContainerCache::is_empty [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:71 CONFIRMED
- fn ContainerCache::write [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:76 CONFIRMED
- fn ContainerCache::read [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:98 CONFIRMED
- fn ContainerCache::generation [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:104 CONFIRMED
- fn ContainerCache::slot_popcount [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:110 CONFIRMED
- fn ContainerCache::recompute_spine [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:115 CONFIRMED
- fn ContainerCache::dirty_indices [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:147 CONFIRMED
- fn ContainerCache::validate [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:168 CONFIRMED
- fn ContainerCache::mark_dirty [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:181 CONFIRMED
- fn ContainerCache::is_dirty [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:198 CONFIRMED
- fn ContainerCache::clear_all_dirty [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:209 CONFIRMED
- fn ContainerCache::push [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:217 CONFIRMED
- fn ContainerCache::push_spine_slot [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/cache.rs:241 CONFIRMED
- fn xor_parity [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/delta.rs:10 CONFIRMED
- fn recover [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/delta.rs:22 CONFIRMED
- fn delta_encode [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/delta.rs:32 CONFIRMED
- fn delta_decode [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/delta.rs:39 CONFIRMED
- fn chain_encode [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/delta.rs:46 CONFIRMED
- fn chain_decode [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/delta.rs:61 CONFIRMED
- const DN_PREFIX [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:26 CONFIRMED
- const SPINE_PREFIX [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:29 CONFIRMED
- fn dn_key [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:37 CONFIRMED
- fn spine_key [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:43 CONFIRMED
- fn walk_to_root_keys [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:48 CONFIRMED
- fn children_pattern [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:54 CONFIRMED
- fn subtree_pattern [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:65 CONFIRMED
- struct RedisPipeline [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:79 CONFIRMED
- enum RedisCommand [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:86 CONFIRMED
- fn RedisPipeline::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:100 CONFIRMED
- fn RedisPipeline::get_dn [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:107 CONFIRMED
- fn RedisPipeline::set_dn [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:113 CONFIRMED
- fn RedisPipeline::mget_dns [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:120 CONFIRMED
- fn RedisPipeline::walk_to_root [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:127 CONFIRMED
- fn RedisPipeline::propagate_dirty [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:137 CONFIRMED
- fn RedisPipeline::len [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:150 CONFIRMED
- fn RedisPipeline::is_empty [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:155 CONFIRMED
- fn cog_record_to_bytes [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:165 CONFIRMED
- fn cog_record_from_bytes [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:183 CONFIRMED
- trait CogRecordSerde [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:198 CONFIRMED
- fn build_record [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/dn_redis.rs:236 CONFIRMED
- struct ContainerGraph [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:35 CONFIRMED
- fn ContainerGraph::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:45 CONFIRMED
- fn ContainerGraph::node_count [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:54 CONFIRMED
- fn ContainerGraph::insert [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:59 CONFIRMED
- fn ContainerGraph::get [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:76 CONFIRMED
- fn ContainerGraph::get_mut [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:82 CONFIRMED
- fn ContainerGraph::contains [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:88 CONFIRMED
- fn ContainerGraph::dns [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:93 CONFIRMED
- fn ContainerGraph::iter [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:98 CONFIRMED
- fn ContainerGraph::children_of [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:103 CONFIRMED
- fn ContainerGraph::walk_to_root [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:108 CONFIRMED
- fn ContainerGraph::adjacency [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:125 CONFIRMED
- fn ContainerGraph::inline_edges [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:132 CONFIRMED
- fn ContainerGraph::overflow_edges [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:139 CONFIRMED
- fn ContainerGraph::add_edge [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:147 CONFIRMED
- fn ContainerGraph::remove_edge [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:161 CONFIRMED
- fn ContainerGraph::add_overflow_edge [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:179 CONFIRMED
- fn ContainerGraph::fingerprint [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:190 CONFIRMED
- fn ContainerGraph::hamming [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:195 CONFIRMED
- fn ContainerGraph::quick_distance [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:202 CONFIRMED
- fn ContainerGraph::outgoing [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:214 CONFIRMED
- fn ContainerGraph::degree [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:222 CONFIRMED
- fn ContainerGraph::nearest_k [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:231 CONFIRMED
- fn ContainerGraph::mget [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:244 CONFIRMED
- fn ContainerGraph::subtree [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/graph.rs:249 CONFIRMED
- const SPLIT_THRESHOLD [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/insert.rs:30 CONFIRMED
- const RESONANCE_THRESHOLD [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/insert.rs:37 CONFIRMED
- enum InsertResult [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/insert.rs:41 CONFIRMED
- fn InsertResult::leaf_idx [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/insert.rs:75 CONFIRMED
- fn InsertResult::containers_written [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/insert.rs:84 CONFIRMED
- fn InsertResult::spines_dirtied [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/insert.rs:93 CONFIRMED
- fn insert_leaf [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/insert.rs:118 CONFIRMED
- const W_DN_ADDR [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:36 CONFIRMED
- const W_TYPE [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:41 CONFIRMED
- const W_TIME [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:44 CONFIRMED
- const W_LABEL [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:47 CONFIRMED
- const W_NARS_BASE [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:50 CONFIRMED
- const W_DN_RUNG [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:53 CONFIRMED
- const W_LAYER_BASE [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:56 CONFIRMED
- const W_EDGE_BASE [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:59 CONFIRMED
- const W_EDGE_END [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:60 CONFIRMED
- const MAX_INLINE_EDGES [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:63 CONFIRMED
- const W_RL_BASE [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:66 CONFIRMED
- const W_BLOOM_BASE [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:69 CONFIRMED
- const W_GRAPH_BASE [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:72 CONFIRMED
- const W_QUALIA_BASE [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:75 CONFIRMED
- const W_RUNG_HIST [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:78 CONFIRMED
- const W_REPR_BASE [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:81 CONFIRMED
- const W_ADJ_BASE [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:84 CONFIRMED
- const W_RESERVED [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:87 CONFIRMED
- const W_CHECKSUM [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:90 CONFIRMED
- const SCHEMA_VERSION [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:93 CONFIRMED
- struct MetaView [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:100 CONFIRMED
- fn MetaView<'a>::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:106 CONFIRMED
- fn MetaView<'a>::dn_addr [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:114 CONFIRMED
- fn MetaView<'a>::node_kind [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:122 CONFIRMED
- fn MetaView<'a>::container_count [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:128 CONFIRMED
- fn MetaView<'a>::geometry [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:134 CONFIRMED
- fn MetaView<'a>::flags [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:141 CONFIRMED
- fn MetaView<'a>::schema_version [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:147 CONFIRMED
- fn MetaView<'a>::provenance_hash [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:153 CONFIRMED
- fn MetaView<'a>::created_ms [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:161 CONFIRMED
- fn MetaView<'a>::modified_ms [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:167 CONFIRMED
- fn MetaView<'a>::label_hash [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:175 CONFIRMED
- fn MetaView<'a>::tree_depth [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:181 CONFIRMED
- fn MetaView<'a>::branch [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:187 CONFIRMED
- fn MetaView<'a>::nars_frequency [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:195 CONFIRMED
- fn MetaView<'a>::nars_confidence [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:201 CONFIRMED
- fn MetaView<'a>::nars_positive_evidence [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:207 CONFIRMED
- fn MetaView<'a>::nars_negative_evidence [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:213 CONFIRMED
- fn MetaView<'a>::rung_level [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:221 CONFIRMED
- fn MetaView<'a>::gate_state [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:227 CONFIRMED
- fn MetaView<'a>::layer_bitmap [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:233 CONFIRMED
- fn MetaView<'a>::layer_marker [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:241 CONFIRMED
- fn MetaView<'a>::inline_edge [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:271 CONFIRMED
- fn MetaView<'a>::inline_edge_count [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:280 CONFIRMED
- fn MetaView<'a>::q_value [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:295 CONFIRMED
- fn MetaView<'a>::bloom_contains [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:305 CONFIRMED
- fn MetaView<'a>::in_degree [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:322 CONFIRMED
- fn MetaView<'a>::out_degree [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:328 CONFIRMED
- fn MetaView<'a>::pagerank [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:334 CONFIRMED
- fn MetaView<'a>::clustering [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:340 CONFIRMED
- fn MetaView<'a>::stored_checksum [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:348 CONFIRMED
- fn MetaView<'a>::verify_checksum [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:353 CONFIRMED
- struct MetaViewMut [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:367 CONFIRMED
- fn MetaViewMut<'a>::new [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:373 CONFIRMED
- fn MetaViewMut<'a>::as_view [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:378 CONFIRMED
- fn MetaViewMut<'a>::set_dn_addr [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:384 CONFIRMED
- fn MetaViewMut<'a>::set_node_kind [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:390 CONFIRMED
- fn MetaViewMut<'a>::set_container_count [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:394 CONFIRMED
- fn MetaViewMut<'a>::set_geometry [pub] cfg=feature="wip" — crates/lance-graph-cognitive/src/container_bs/meta.rs:398 CONFIRMED
- truncated (1194 total), see symbols.tsv

## Imports from in-scope crates

- none

## Calls

CALL_GRAPH=ARTIFACT_MISSING
- calls ARTIFACT_MISSING — no ruff_charon artifact for this crate UNKNOWN

## Architecture

- none

Generated from Cargo manifests — extracted relationships only.

```mermaid
flowchart LR
  n10["deepnsm"]
  n25["holograph"]
  n28["lance-graph"]
  n37["lance-graph-cognitive"]
  n39["lance-graph-contract"]
  n10 --> n37
  n28 --> n37
  n37 --> n25
  n37 --> n39
```
