# 2026-10-04 — DeReKo profile and TEKAMOLO order: edited German books vs Bible translations (D-LXC-28)

**Status:** MEASURED (debug-0, release).
- Instruments: `crates/deepnsm-v2/examples/{dereko_profile,tekamolo_order}.rs`.
- Texts, none committed:
  - Project Gutenberg public-domain novels;
  - the PD Bible bundle (Luther 1545 in modernised spelling, Elberfelder 1905);
  - getbible Elberfelder 1871 and Schlachter 1951;
  - UD GSD and HDT.
- Schlachter 1951 was used only as a local measurement input: its copyright status is unclear.

Operator: test DeReWo against other literature, and compare TEKAMOLO by position between edited books (*lektorierte Bücher*) and the grammar-distorted Bible.

## 1. DeReKo-2014 profile (top-100k word / lemma / STTS / frequency list)

| text | tokens | covered | Spearman | POS-ambiguous | ADJD |
|---|---|---|---|---|---|
| UD GSD (news/web) | 222,938 | 90.2 % | 0.670 | 13.4 % | 2.42 % |
| UD HDT (IT news) | 630,538 | 91.7 % | 0.509 | 15.2 % | 2.39 % |
| Kafka, *Verwandlung* 1915 | 19,230 | 94.9 % | 0.595 | 18.1 % | 3.69 % |
| Mann, *Buddenbrooks* 1901 | 230,254 | 91.2 % | 0.570 | 16.2 % | 3.44 % |
| Fontane, *Effi Briest* 1895 | 95,624 | 93.6 % | 0.620 | 18.7 % | 3.36 % |
| Storm, *Schimmelreiter* 1888 | 38,473 | 91.9 % | 0.523 | 17.3 % | 2.28 % |
| Goethe, *Faust I* (verse) | 30,959 | 88.6 % | 0.547 | 18.9 % | 3.47 % |
| Schlachter 1951 | 708,524 | 91.9 % | 0.475 | 18.7 % | 1.12 % |
| Elberfelder 1905 | 722,778 | 91.2 % | 0.450 | 18.0 % | 0.91 % |
| Elberfelder 1871 | 718,761 | 91.3 % | 0.452 | 18.0 % | 0.92 % |
| Luther 1545 (modernised spelling) | 696,534 | 89.0 % | 0.468 | 18.9 % | 1.20 % |

- **Token coverage barely separates genres** (89–95 %).
- **Rank agreement with contemporary usage is lower for every Bible** (0.45–0.48) than for the novels (0.52–0.62).
- **ADJD** (adjective used predicatively or adverbially, *früh*, *schnell*) is about 3.4 % of tokens in the novels and about 1 % in every Bible, including the 20th-century Schlachter.
- Type-OOV rates are not comparable across texts: they grow with text length and proper names.

## 2. TEKAMOLO order by position

**Method.**
- A clause is the span between punctuation marks.
- Lanes are read from **unambiguous** cue words only: no prepositions, no modal particles, no *da* / *so* / *noch*.
- A pair of two different lanes in one clause counts as in order when Te < Ka < Mo < Lo holds.

**Round 1 (pre-registered): KILL.**
- Prediction: every Bible lies below the pooled edited texts, with non-overlapping 95 % intervals.
- The pool included the two news treebanks, which sit at chance (GSD 50.0 %, HDT 46.5 %), as do the Bibles (45.5–50.0 %).
- The novels alone are higher: Storm 74.3 %, Fontane 67.5 %, Mann 60.3 %. Pooled, 101/157 = 64.3 % vs the Bibles 73/155 = 47.1 %, non-overlapping. That split was made after seeing the data.

**Round 2 (pre-registered before download; novels fixed by Gutenberg id).**
- Novels: *Dr. Mabuse*, *Taugenichts*, *Liebe ist ewig*, *Zauberberg I*, *Beate Hoyermann*, *Elixiere des Teufels*.
- *Winnetou I* was excluded before any run: no plain-text file on Gutenberg.

| | in order | 95 % Wilson | pairs |
|---|---|---|---|
| fresh novels, pooled | **58.9 %** | [51.1, 66.2] | 158 |
| four Bibles, pooled | **47.1 %** | [39.4, 54.9] | 155 |

**Verdict: KILL on the pre-registered non-overlap criterion.** The direction replicates (+11.8 points; two-proportion z ≈ 2.1, p ≈ 0.04), but the intervals overlap by 3.8 points. The bottleneck is that only about 150 clauses per group contain two unambiguous cues.

**The robust difference is density, not order (both rounds).**

| | Te per 10k tokens | Mo per 10k tokens | Ka per 10k tokens |
|---|---|---|---|
| novels | 47–93 | 16–55 | 7–11 |
| news | 51–53 | 13–17 | — |
| Bibles | 14–18 | 4–5 | 16–20 (Luther's *darum*, *deshalb*) |

Biblical German rarely stacks two adverbials in one clause; it chains short clauses with *und* (parataxis). This agrees with the ADJD gap in §1.

**OPEN.**
- Order needs more two-lane clauses: a lane reader with ambiguous cues resolved (D-LXC-25 lanes for PPs, D-LXC-20 adj/adv for bare adverbs), not a larger cue list.
- No public-domain contemporary German Bible exists. Luther 2017, Einheitsübersetzung 2016, Elberfelder 2006, Schlachter 2000 and the BasisBibel are copyrighted. The *Offene Bibel* is an openly licensed, partial contemporary translation (licence to verify).
