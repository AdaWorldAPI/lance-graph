# NARS reference oracles — observed outputs (2026-10-11)

Every truth value below was copied from a raw output file in `ext/ona_out/` or `ext/onr_out/`.
A fixture's own `expected:` comment is recorded separately and never treated as an observation.

## Pinned revisions (verified with `git rev-parse HEAD`)

| engine | repo | HEAD |
|---|---|---|
| ONA (OpenNARS for Applications) | github.com/opennars/OpenNARS-for-Applications | `dc4efd0abd520cdb79bf53bfa3c285ebb24f2e8a` |
| OpenNARS for Research | github.com/opennars/opennars | `4d1bdb0f599ccf6ed44edfc398f02f930713d30f` |

Cloned shallow into `ext/ona` and `ext/onr`.

## ONA build

- `build.sh` is two `gcc` invocations, nothing downloaded or installed:
  stage 1 builds NAR, `./NAR NAL_GenerateRuleTable > src/RuleTable.c`, stage 2 rebuilds with the table.
- Command: `cd ext/ona && sh ./build.sh` -> exit 0, ~2.6 MB `NAR` binary. Log: `ext/ona_build.log`.
- Fixtures run as the README documents: `./NAR shell < examples/nal/<name>.nal`. Each ran in under 7 s.
- `reasoningbench` run twice: outputs byte-identical (deterministic).

## ONA evaluation.py (upstream harness)

`python3 evaluation.py` -> exit 0 in 8 min 6 s. Full output: `ext/ona_out/evaluation.txt`. Observed:
- C system tests: "System tests successful!"
- "Narsese integration tests successful!"; English examples skipped (no nltk).
- Q&A global: avg answer time 30.28, avg answer confidence 0.5547681, 138 questions, 113 answered, answer ratio 0.8188.
- Pong ratio 0.8206, Pong2 0.9454, Alien 0.9946, Cartpole 0.8330, Robot move_success_ratio 0.2692; sheep counted 0 of 20.

**Harness defects found (read from evaluation.py):**
1. It matches only lines starting `Comment: expected: `. `reasoningbench.nal` writes `//--expected:`, which the shell echoes as
   `Comment: --expected: ...`. So none of its 14 expectations are parsed, and evaluation.py reports
   "Passed ./examples/nal/reasoningbench.nal" vacuously (0 questions counted).
2. On a failed expectation it prints `Failure ...` and calls `exit(0)`, so its exit status cannot signal failure. No `Failure` line appeared in this run.
3. An Answer passes when its confidence is >= the expected confidence; frequency is never compared.

## ONA reasoningbench.nal — question / fixture's expected / observed

Setup: 9 rules (flower, grandfather/grandmother via father/mother chains, grandchild inversion, `succ` numbers), 1000 cycles,
facts, 10 cycles, questions; later `<1 --> num>.` + 5 cycles + 4 questions. Raw: `ext/ona_out/reasoningbench.txt`.
"exp" = the `//--expected` comment in the fixture. "obs" = the observed `Answer:` line (f, c) and its stamp.

| # | question | exp (f, c) | obs answer | obs (f, c) | stamp | match |
|---|---|---|---|---|---|---|
| 1 | `<(sam_father_father * sam) --> ?1>?` | grandfather (1.0, 0.729) | grandfather | (1.000000, 0.729000) | [14,3,13] | yes |
| 2 | `<(sam * sam_father_father) --> ?1>?` | grandchild (1.0, 0.6561) | grandchild | (1.000000, 0.656100) | [14,7,3,13] | yes |
| 3 | `<(sam_mother_father * sam) --> ?1>?` | grandfather (1.0, 0.729) | grandfather | (1.000000, 0.729000) | [18,16,4] | yes |
| 4 | `<(sam * sam_mother_father) --> ?1>?` | grandchild (1.0, 0.6561) | grandchild | (1.000000, 0.656100) | [18,7,16,4] | yes |
| 5 | `<(sam_father_mother * sam) --> ?1>?` | grandmother (1.0, 0.729) | grandmother | (1.000000, 0.729000) | [15,13,5] | yes |
| 6 | `<(sam * sam_father_mother) --> ?1>?` | grandchild (1.0, 0.6561) | grandchild | (1.000000, 0.656100) | [15,8,13,5] | yes |
| 7 | `<(sam_mother_mother * sam) --> ?1>?` | grandmother (1.0, 0.729) | grandmother | (1.000000, 0.729000) | [17,6,16] | yes |
| 8 | `<(sam * sam_mother_mother) --> ?1>?` | grandchild (1.0, 0.6561) | grandchild | (1.000000, 0.656100) | [17,8,6,16] | yes |
| 9 | `<?1 --> flower>?` | `{rose}` (1.0, 0.729) | `{rose}` | (1.000000, **0.815686**) | [10,2,11,2,1] | **MISMATCH on c** |
| 10 | `<(succ /1 1) --> num>?` | (1.0, 0.81) | same | (1.000000, 0.810000) | [19,9] | yes |
| 11 | `<(succ /1 (succ /1 1)) --> num>?` | (1.0, 0.729) | same | (1.000000, 0.729000) | [19,9,9] | yes |
| 12 | 3x succ | (1.0, 0.6561) | same | (1.000000, 0.656100) | [19,9,9,9] | yes |
| 13 | 4x succ | (1.0, 0.59049) | same | (1.000000, 0.590490) | [19,9,9,9,9] | yes |
| 14 | 5x succ (commented out in fixture, not asked) | (1.0, 0.531441) | — | not run | — | n/a |

Notes on what was observed:
- Answers 1-8, 10-13 equal the chained deduction confidences 0.9^k (k = 2..5): 0.81, 0.729, 0.6561, 0.59049.
- #9: observed c = 0.815686 exceeds the fixture's 0.729. Its stamp `[10,2,11,2,1]` contains both rules (1 green, 2 red)
  and both facts (10 green, 11 red), and evidence 2 appears twice. Arithmetic check (mine, not ONA's): revising two
  independent c=0.729 conclusions would give c=0.8433 (w=2.690 each), not 0.815686, so the observed value is NOT simple
  revision of the two 0.729 deductions; its derivation path is unexplained here (w(0.815686)=4.4255). Treat 0.815686 as an
  observed fact to reproduce, not as a value derivable from the formula sheet without tracing ONA.
  Under evaluation.py's rule (obtained >= expected) it would pass anyway; under an exact-match oracle it does not.
- The `{grass}` candidate (green only) was not the answer; `?1` returned the single best answer.
- Statement inputs before the first `1000` are echoed twice (`Input:` line printed twice per rule); questions/facts later once.
- Interesting stamp detail: grandmother answers (#5, #7) do not carry stamp 7/8 (the grandchild rules); #2 carries
  rule 7 (grandfather->grandchild) and #6/#8 carry rule 8 — consistent with the rules used.

## ONA other fixtures — expected comment vs observed

| fixture | fixture's `//expected:` | observed (line in raw output) | match |
|---|---|---|---|
| toothbrush.nal | `^lighter`, `^reshape`, `^wait`, `^unscrew` each with args `({SELF} * {toothbrush1})` | all four executed with exactly those args (lines 22, 35, 48, 61) | yes |
| activequestion.nal | `^go` and `^say` with args `({SELF} * bedroom)` | `^go executed with args ({SELF} * bedroom)` (l.30), `^say ... ({SELF} * bedroom)` (l.39) | yes |
| plandeep.nal | `^left executed with args` | `^left executed with args ` (l.96) | yes |
| self.nal | `^say ... ({SELF} * (want * beer))` | same (l.11), decision expectation=0.550672 | yes |
| self2.nal | `^say ... ({SELF} * (want * beer))` | same (l.9), decision expectation=0.578198 | yes |
| self3.nal | `^say ... ({SELF} * {PIZZA})` | same (l.37), decision expectation=0.518912 | yes |

evaluation.py also printed "Passed" for each of these six (it checks execution lines by exact string).

## ONA `{SELF}` — what it is and what it is not

**What `{SELF}` does in self.nal / self2.nal / self3.nal: goal-directed operation selection, not introspection.**
- `{SELF}` appears only as the first argument of an operation: `<({SELF} * X) --> ^op>`. In ONA's source
  (`Narsese.c:716-726`, `Narsese_isOperation`) an operation term is `<(*,{SELF},x) --> ^op>`, and `{SELF}` (or a variable)
  in that slot is what marks it executable. `Decision.c:296-302` builds `({SELF} * arg)` when filling a decision's
  arguments; `Decision.c:399-400` rejects an operation whose first argument did not resolve to `SELF`.
- self.nal: given the rule `<(#1 &/ <({SELF} * $1) --> ^say>) =/> <$1 --> [hear]>>` plus a demonstration that
  `<(want * pizza) --> [hear]>` co-occurred with sam getting pizza, the goal `<({SELF} * beer) --> get>!` made ONA print
  `decision expectation=0.550672 ...` and `^say executed with args ({SELF} * (want * beer))`. It generalised the
  observed sam/pizza episode to itself/beer and chose to *say* "want beer".
- self2.nal: same goal, but the demonstration is sam's own `^say` (imitation). Observed: a derived temporal implication
  `<(<{SELF} --> [hungry]> &/ <({SELF} * (want * beer)) --> ^say>) =/> <({SELF} * beer) --> get>>` with c=0.241351, decision
  expectation 0.578198, then `^say ... ({SELF} * (want * beer))`.
- self3.nal: babbling -> hearing association learned from `<({SELF} * {babababa}) --> ^say>` followed by hearing it;
  later, a goal to *see* vPizza (previously preceded by hearing PIZZA) produced derived subgoals `<{PIZZA} --> [hear]>!`
  (f=0.964915, c=0.251427) and decision `^say ... ({SELF} * {PIZZA})` (expectation 0.518912). Self-generated sound as
  a means to an end.
- In all three, `{SELF}` names the agent as the actor of its own motor operations. Nothing reads beliefs or inference steps.

**Does ONA expose an operation that reads its own inference steps/beliefs as objects? No (at this SHA).**
- Registered Narsese operators (`Shell.c:37-46`, `NAR_AddOperation`): `^left ^right ^up ^down ^say ^pick ^drop ^go
  ^activate ^deactivate` — all bound to `Shell_op_nop` (they return an empty Feedback). `OPERATIONS_MAX` is 10
  (`Config.h:132`). `*setopname` can rename op slots (toothbrush.nal uses it), still nop actions.
- Repo-wide grep for `^believe`, `^want`, `^doubt`, `^wonder`, `^anticipate`, `^remind`, `^evaluate`: **zero hits**
  (src, examples, Python, misc). OpenNARS-style mental operators do not exist in ONA.
- The only self-inspection is out-of-band shell commands that print to stdout, not Narsese terms the reasoner can
  reason about: `*stats`, `*concepts`, `*cycling_belief_events`, `*cycling_goal_events`, `*inverted_atom_index`,
  `*occurrence_time_index`, `*opconfig`, `*query <threshold> <narsese>` (Shell.c:123-402). Operation feedback
  (`Decision.c:195-252`) only re-enters the executed operation term as a belief event (`NAR_AddInputBelief(feedbackTerm)`),
  i.e. "I did ^say(x)" — the action, not the inference that chose it.
- So ONA is an oracle for NAL-1..6 Q&A and NAL-7/8 temporal/procedural decisions, not for introspective (NAL-9) behaviour.

## OpenNARS for Research — test corpus at the pinned SHA

`src/main/resources/nal/` holds `single_step/`, `multi_step/`, `application/`, `stability/`.

| dir | files (incl. README.txt) | .nal files |
|---|---|---|
| single_step | 216 | **215** |
| multi_step | 24 | 24 |
| application | 5 | 5 (detective, detective2, toothbrush, toothbrush2, vision) |
| stability | 1 | 1 (long_term_stability.nal) |

Naming scheme in single_step: `nal<LEVEL>.<n>.nal` (numeric n), plus named variants `nal<LEVEL>.<topic>.nal`.
Per level (count of `nal<L>.*.nal`):

| NAL | count | files |
|---|---|---|
| 1 | 9 | nal1.0 .. nal1.8 |
| 2 | 20 | nal2.0 .. nal2.19 |
| 3 | 16 | nal3.0 .. nal3.15 |
| 4 | 9 | nal4.0 .. nal4.8 |
| 5 | 30 | nal5.0 .. nal5.29 |
| 6 | 35 | nal6.0 .. nal6.27, nal6.birdClaimedByBob, can_of_worms, nlp1, nlp2, redundant, symmetry, uncle |
| 7 | 30 | nal7.0-3, 6-8, 15, 18, 19, 35, 36, concurrentEqual.res, concurrentImpl.{abd,ded,goal,ind}, conj.decompose, decomposeCompound, eventInduction1/2, par, predictiveImpl.goal, retroImplSeq, retrospectiveImpl.goal, revrev, vardetach1-4 |
| 8 | 50 (+1 `nal8_list.nal`) | nal8.1.0-1.27 (gaps), 8.2.1-4, 8.3.0-4, 8.4.0-7 (gaps), 8.5.0-7, nal8.add, nal8.count, nal8.reflect |
| 9 | 13 | nal9.4, anticipate1, believe1, believe2, doubt, evaluate1, evaluate2, hesitate, want1, want12, want2, wonder1, wonder2 |
| other | 2 | desire.1.nal, notcontain.nal |

9+20+16+9+30+35+30+50+13 = 212, + nal8_list + desire.1 + notcontain = 215.

### How upstream runs and checks a .nal file

- Runner: `src/test/java/org/opennars/core/NALTest.java` (JUnit4 Parameterized over `/nal/single_step/`, `/nal/multi_step/`,
  `/nal/application/`; `NALTestSome` = subset). `sample(path)`: `new Nar()` (config `config/defaultConfig.xml` from resources),
  `OutputCondition.getConditions(n, fileText, 5)`, `n.addInputFile(path)`, then `n.cycles(1550)` (minCycles), and
  `assertTrue(success)` where success = at least one condition and all conditions succeeded.
- File syntax (`single_step/README.txt`): a bare number = run that many cycles; `'text` = comment echoed as `ECHO:`;
  `''outputMustContain('<narsese> %f;c%')` = the output buffer must contain a line with that substring
  (`OutputContainsCondition`). Truth is printed and matched at **2 decimals** (`%0.87;0.91%`), not full precision.
- Note: because the runner always runs 1550 cycles after input, an expectation passes if the line appears at any point —
  it is not "after exactly N cycles".
- `Memory.randomNumber = new Random(randomSeed)` (Memory.java:87): seeded. Two reruns of nal4.0 and nal8.1.0 were identical
  after masking the per-run stamp-base id (`(-5576437971530929954,0)`-style), so outputs are reproducible.

## OpenNARS for Research — build

- **Upstream build BLOCKED.** `mvn -B -q -DskipTests -Dmaven.javadoc.skip=true package` (JDK 21, fresh local repo) ->
  `[FATAL] Non-resolvable parent POM for org.opennars:opennars:3.0.4-SNAPSHOT: ... org.opennars:opennars-parent:pom:1-SNAPSHOT
  (absent): Could not find artifact ... in ossrh.snapshots (https://oss.sonatype.org/content/repositories/snapshots)`.
  The parent POM only existed in the retired Sonatype OSSRH snapshot repo. Log: `ext/onr_build.log`.
- **Workaround used (not upstream's build, recorded so it can be judged):**
  1. Fetched the three declared deps from Maven Central with `maven-dependency-plugin:3.6.1:copy`: commons-lang3 3.7,
     guava 25.1-jre (versions from pom.xml), and junit **4.12** + hamcrest-core 1.3 (junit's version is managed by the missing
     parent, so 4.12 is my choice; the tests only use the JUnit4 API).
  2. `javac --release 8 -encoding UTF-8` over all 154 `src/main/java` + `src/test/java` files -> exit 0 (`ext/onr_classes`).
  3. Driver `scripts/src/RunNal.java` (authored here) calls upstream `new NALTest(path).testNAL(path)` unchanged, with
     `NALTest.showOutput = showSuccess = true`, and catches the AssertionError to print PASS/FAIL. Classpath adds
     `onr/src/main/resources` (config + nal files).
  Command: `java -cp scripts/classes:ext/onr_classes:ext/onr/src/main/resources:ext/onr_libs/* RunNal <file.nal>`.

## OpenNARS for Research — observed runs (9 single_step files, all via upstream NALTest)

Raw outputs: `ext/onr_out/<file>.txt`. Truth values are as printed (2 decimals). "upstream verdict" is NALTest's own.
"in OUT" = count of `OUT:` lines containing the expected substring.

| file | level / rule | input | fixture's expected line | observed? (in OUT) | upstream verdict |
|---|---|---|---|---|---|
| nal1.0 | NAL-1 revision | `<bird --> swimmer>.` + `%0.10;0.60%`, 1 cycle | `<bird --> swimmer>. %0.87;0.91%` | yes (1) | PASS |
| nal1.1 | NAL-1 deduction | `<bird --> animal>.`, `<robin --> bird>.`, 3 cycles | `<robin --> animal>. %1.00;0.81%` | yes (2) | PASS |
| nal2.0 | NAL-2 similarity revision | `<robin <-> swan>.` + `%0.10;0.60%` | `<robin <-> swan>. %0.87;0.91%` | yes (1) | PASS |
| nal4.0 | NAL-4 product -> image | `<(*,acid,base) --> reaction>.`, 2 cycles | `<acid --> (/,reaction,_,base)>. %1.00;0.90%` and `<base --> (/,reaction,acid,_)>. %1.00;0.90%` | yes (139), yes (40) | PASS |
| nal5.1 | NAL-5 implication deduction | `<<robin-->bird> ==> <robin-->animal>>`, `<<robin-->[flying]> ==> <robin-->bird>>`, 14 cycles | `<<robin --> [flying]> ==> <robin --> animal>>. %1.00;0.81%` | yes (5) | PASS |
| nal6.1 | NAL-6 variable unification | `<<$x-->bird> ==> <$x-->animal>>`, `<<$y-->robin> ==> <$y-->bird>>`, 3 cycles | `<<$1 --> robin> ==> <$1 --> animal>>. %1.00;0.81%` and `<<$1 --> animal> ==> <$1 --> robin>>. %1.00;0.45%` | yes (4), yes (4) | PASS |
| nal8.1.0 | NAL-8 goal + procedural impl | goal `<{t001} --> [opened]>!`, `<(&/,hold,at,^open) =/> opened>.`, 20 cycles | `(&/,<(*,SELF,{t002}) --> hold>,<(*,SELF,{t001}) --> at>,(^open,{t001}))! %1.00;0.81%` | yes (26) | PASS |
| nal9.believe1 | NAL-9 internal experience | `<a --> b>.`, `<a --> b>?`, 2 cycles | `(^believe,{SELF},<a --> b>,TRUE). :!0: %1.00;0.90%` | yes (6) | PASS |
| nal9.wonder1 | NAL-9 internal experience | `<a --> b>?`, 10 cycles | `(^wonder,{SELF},<a --> b>). :!0: %1.00;0.90%` | yes (35) | PASS |

Also observed (beyond the expectation lines), useful as extra oracle points:
- nal1.1 also produced `<animal --> robin>. %1.00;0.45%` (the reverse, abductive/inductive-strength conclusion).
- nal5.1 also produced `<<robin --> animal> ==> <robin --> [flying]>>. %1.00;0.45%`.
- nal2.0 / nal1.0 revisions both print exactly `%0.87;0.91%` for (1.0,0.9) + (0.1,0.6).
- nal4.0: besides the two images (c=0.90), many second-order structural outputs at c=0.81, e.g.
  `<acid --> (/,reaction,_,base)>. %1.00;0.81%` (128 lines) and nested images like `<base --> (/,reaction,(/,(*,acid,base),_,base),_)>. %1.00;0.81%`.
- nal8.1.0 derived subgoals: `<(*,SELF,{t002}) --> hold>! %1.00;0.73%`, an eternal->event variant `<(*,SELF,{t002}) --> hold>! :!-5: %1.00;0.66%`,
  image-rewritten `(&/,<{t002} --> (/,hold,SELF,_)>,...)! %1.00;0.81%`, and internal-experience terms
  `<{SELF} --> (/,^want,_,<{t001} --> [opened]>,TRUE)>. :!0: %1.00;0.90%` (43 lines). No `EXE:` line occurred.
- Note OpenNARS writes the agent as bare `SELF` inside products in NAL-8 fixtures and as `{SELF}` in mental-operator terms.

## ONA vs OpenNARS on the same rule (observed)

| rule | ONA observed | OpenNARS observed |
|---|---|---|
| 2-step deduction, (1,0.9)x(1,0.9) | c=0.810000 (`<(succ /1 1) --> num>`) | `%1.00;0.81%` |
| 3-step chain | c=0.729000 (grandfather) | — (not run) |
| revision (1,0.9)+(0.1,0.6) | — (not run) | `%0.87;0.91%` |

## NAL-9 / `{SELF}` comparison

- OpenNARS for Research HAS mental operators: `src/main/java/org/opennars/operator/mental/` contains Anticipate, Believe, Consider,
  Doubt, Evaluate, Feel, FeelBusy, FeelSatisfied, Hesitate, Name, Register, Remind, Want, Wonder. Observed: inputting a belief
  and a question produces `(^believe,{SELF},<a --> b>,TRUE). :!0:` (time 0, i.e. at input) and `(^wonder,{SELF},<a --> b>). :!0:`;
  a goal produces `<{SELF} --> (/,^want,_,<goal>,TRUE)>`. These are Narsese events about the system's own beliefs/questions/goals
  that re-enter inference (internal experience).
- ONA has none of these (see the ONA `{SELF}` section). For introspective (NAL-9) oracle behaviour, OpenNARS is the only one of the two.

## BLOCKED / not done, and why

1. **OpenNARS upstream Maven build** — parent POM `org.opennars:opennars-parent:1-SNAPSHOT` is not resolvable (only ever hosted on
   the retired oss.sonatype.org snapshot repo). Exact error quoted in the build section. Worked around by direct javac; the junit
   version (4.12) is my choice, not upstream's.
2. **ONA English integration tests** — skipped by evaluation.py itself ("install python3 and nltk"); not attempted.
3. **reasoningbench via evaluation.py** — not blocked, but its 14 expectations are never checked by the upstream harness
   (`//--expected` vs `Comment: expected: `); the table above is the only check of them.
4. reasoningbench #14 (5x succ) is commented out in the fixture, so it was never asked.
5. OpenNARS: only 9 of 215 single_step files run (as asked); multi_step/application not run.
6. ONA #9 (`{rose} --> flower`, c=0.815686) — derivation not traced; the value does not equal simple revision of two 0.729 deductions.


## Addendum (orchestrator, 2026-10-11)

- **Counts re-checked:** `single_step/` at `4d1bdb0` holds 216 entries, 215 `.nal`
  files plus `README.txt`. (A separate inventory worker reported 230; that figure
  was wrong and is not used anywhere.)
- **Audit fixture `revision_points.nal`** (in this folder) was run through the same
  OpenNARS driver. Observed revised OUT lines: `<a1 --> b1>. %0.87;0.91%`,
  `<a2 --> b2>. %0.45;0.77%`, `<a3 --> b3>. %0.50;0.95%`. Pairs a4 and a5 printed
  no revised line (their revised truth rounds to an input's printed value), so they
  are not used. These three points are the expected values in
  `crates/lance-graph-planner/tests/nal_revision_oracle.rs`.
- Raw outputs (`ona_out/`, `onr_out/`) and the compiled OpenNARS classes are NOT in
  the repo. Reproduce with the pinned commits and the commands above.
