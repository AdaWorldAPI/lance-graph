# Reading and prompts — deforestation and the immaterial handover

Companion to `IMMATERIAL_HANDOVER.md` and `BOUND_COMPUTATION.md`. Papers are real. Prompts are for a session that must not turn the metaphor into a framework.

Repo: https://github.com/AdaWorldAPI/lance-graph

## Zoom out — the law, before databases

- Philip Wadler, "Deforestation: transforming programs to eliminate trees", Theoretical Computer Science, 1990. The root. Deletes an intermediate tree when the consumer can fold the producer. Restricted to treeless form.
- Andrew Gill, John Launchbury, Simon Peyton Jones, "A Short Cut to Deforestation", FPCA 1993. One equation: `foldr k z (build g) = g k z`. Does not fuse `zip`.
- Duncan Coutts, Roman Leshchinskiy, Don Stewart, "Stream Fusion: From Lists to Streams to Nothing at All", ICFP 2007. Step functions so `zip` and `filter` can fuse. Residual is the function passed into `map`.
- Oleg Kiselyov, Aggelos Biboudis, Nick Palladinos, Yannis Smaragdakis, "Stream Fusion, to Completeness", arXiv:1612.06668. Names the residual the shortcut still leaves.
- Yijia Chen, Lionel Parreaux, "The Long Way to Deforestation", ICFP 2024, arXiv:2410.02232, revised October 2025. General deforestation. Average 8.2 percent on 38 programs. Shortcut fusion remains the industrial system.

Prompt, zoom out: "Which of these delete the consumer's input, and which only delete the producer's output? Quote the residual each paper admits it cannot remove."

## The database cut — positions, not tuples

- Daniel Abadi, Daniel Myers, David DeWitt, Samuel Madden, "Materialization Strategies in a Column-Oriented DBMS", ICDE 2007. Late materialization emits positions as a range, a list, or a bitmap, then stitches. Also measures the reversal: the position list can cost more than the tuple.
- Evgeniy Klyuchikov, Elena Mikhailova, George Chernishev, "Hybrid Materialization in a Disk-Based Column-Store", arXiv:2304.08532. Ultra-late still builds the position set. Hybrid exists because that set is sometimes the new tree.
- Liwen Sun, Michael Franklin, Sanjay Krishnan, Reynold Xin, "Fine-grained partitioning for aggressive data skipping", SIGMOD 2014. Zone-map ancestor at block grain. 2–5× over range blocking, on a system that reorganizes tuples to make skips possible.
- Peter Boncz, Martin Kersten, "MIL Primitives for Querying a Fragmented World", VLDB Journal 1999. The column-store pipeline as primitives over fragments, not rows.

Prompt, this altitude: "Abadi's bitmap is Gill's `build`. Where does Abadi say the bitmap loses, and is that the same failure as `zip`?"

## Zoom in — the join the shortcut cannot eat

- Scott Kovach et al., "Fast Collection Operations from Indexed Stream Fusion", arXiv:2507.06456, July 2025. Joins over collections with no intermediate allocation, by folding an indexed stream. Mechanized in Lean, ported to Rust. Does not use the word deforestation.
- Destination-passing style, the compiler technique in which a caller passes the address the callee writes into, so the callee does not allocate the result. This is the nearest named relative of `handover(i) = Destination { space, ordinal }`. Do not cite a specific arXiv id for it without opening the paper.

Prompt, zoom in: "Indexed stream fusion handles `zip` by making the index the structure. Is a `u16` ordinal that kind of index, or is it Abadi's position list under another name? What would falsify the distinction?"

## The handover — no paper is this note

The polyline, the superconductor, and the unopened box are not literature. The literature pieces they are made of:

- The polyline is provenance without re-walking. Todd Green, Grigoris Karvounarakis, Val Tannen, "Provenance Semirings", PODS 2007. The bound path stored beside the ref is a provenance polynomial of one term, not a semiring implementation.
- The unopened box is late materialization plus a typed observe. Abadi 2007, and the report boundary that already counts lookups.
- The address is destination-passing style plus the CAM ordinal that survives a rename (`crates/lance-graph-report/src/boundary.rs`).

Prompt, on the note: "Read `.grok/board/IMMATERIAL_HANDOVER.md`. Which sentence is Abadi, which is Gill, and which is only a metaphor? Do not promote a metaphor to a type."

## Prompts that change zoom

Widen. "Report, Quack, and lane-fold each bind once. Name the intermediate each still builds. Say which of those intermediates is Gill's `build` and which is the residual Gill could not delete."

Zoom out. "If the handover is destination-passing style, what does the caller own, and what is the callee forbidden to allocate? Answer from the technique, then check it against `SourceRef { id, generation }`."

Zoom in. "Set `ordered` and `invertible` on Count, Sum, Min, Max, Exists without using the words route, cat, or superconductor. `ordered` means one thing. If it means two, stop."

Adjacent, do not absorb. "Provenance semirings explain a bound path. They do not justify a `RouteRef`. Say what a semiring would add that a stored path does not, and whether this tree needs it."

Reject path. "Argue that the immaterial handover is early materialization with the copy deleted from the diagram. Use Abadi's reversal result. If the argument holds, the note is false."
