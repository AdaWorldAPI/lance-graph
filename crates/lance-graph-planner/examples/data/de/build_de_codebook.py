#!/usr/bin/env python3
"""Build the German COCA-shaped codebook from UD treebanks (GSD + HDT).

Emits the loader-shaped TSVs the lance-graph examples read:

  lexicon.tsv        word<TAB>lemma<TAB>pos<TAB>rank        (COCA shape, PoS letters n/v/j/r/i/d/p)
  article_case.tsv   form<TAB>case<TAB>count                (der/den/dem — case-decidable NP fronting)
  satzklammer.tsv    pattern<TAB>count                      (bracket geometry: V2 / verb-final / aux…V)
  valency.tsv        verb<TAB>relation<TAB>case<TAB>count   (mined government frames = ArgumentStatus)
  tekamolo.tsv       lane<TAB>lemma<TAB>relation<TAB>count  (adverbial → Temporal/Kausal/Modal/Lokal)
"""
import sys, glob, collections


def verb_cluster(head, toks):
    """One clause's verb cluster: its lexical head + the auxiliaries UD hangs
    off it (`gesehen` + `habe`, `kommen` + `muss`). German splits the finite and
    lexical verb across the Satzklammer, so any question about the bracket must
    be asked of the cluster, never of a single token."""
    if head is None:
        return []
    return [head] + [x for x in toks
                     if x["head"] == head["id"] and x["upos"] in ("VERB", "AUX")
                     and x["rel"] in ("aux", "aux:pass", "cop")]


def finite_of(cluster):
    """The cluster's FINITE member — the left bracket, and the token whose
    position decides verb-finality."""
    return next((v for v in cluster if v["feats"].get("VerbForm") == "Fin"), None)

# UD UPOS → the single-letter PoS the COCA loader expects.
POS = {
    "NOUN": "n", "PROPN": "n", "VERB": "v", "AUX": "v", "ADJ": "j",
    "ADV": "r", "ADP": "i", "DET": "d", "PRON": "p", "NUM": "m",
    "CCONJ": "c", "SCONJ": "c", "PART": "t", "INTJ": "x",
}

# German TEKAMOLO cue lexicons (function words + high-frequency adverbials).
# The lane assignment is the grammatical circumstance-frame, not semantics.
TEMPORAL = {
    "heute", "gestern", "morgen", "jetzt", "dann", "damals", "bald", "spät", "früh",
    "immer", "nie", "oft", "manchmal", "wieder", "schon", "noch", "seit", "während",
    "bevor", "nachdem", "sobald", "bis", "danach", "zuvor", "jährlich", "täglich",
    "monatlich", "wöchentlich", "anschließend", "zunächst", "schließlich",
}
KAUSAL = {
    "weil", "denn", "da", "deshalb", "deswegen", "daher", "darum", "also", "folglich",
    "somit", "wegen", "aufgrund", "dadurch", "damit", "sodass", "obwohl", "trotz",
    "dennoch", "falls", "wenn", "sofern", "andernfalls", "infolge", "mithin",
}
MODAL = {
    "so", "sehr", "gut", "schnell", "langsam", "gern", "kaum", "fast", "genau",
    "wirklich", "vielleicht", "wohl", "eigentlich", "sicher", "leider", "natürlich",
    "möglicherweise", "angeblich", "offenbar", "durchaus", "keineswegs", "unbedingt",
    "gemeinsam", "plötzlich", "allmählich", "sorgfältig", "deutlich",
}
LOKAL = {
    "hier", "dort", "da", "oben", "unten", "vorn", "hinten", "links", "rechts",
    "überall", "nirgends", "draußen", "drinnen", "in", "an", "auf", "bei", "über",
    "unter", "vor", "hinter", "neben", "zwischen", "nach", "zu", "aus", "von",
}
# `da` is both Kausal (causal conjunction) and Lokal (deictic); `wenn` temporal-or-
# conditional. Ambiguity is recorded, never silently resolved: a lemma may appear in
# several lanes and the consumer treats multi-lane lemmas as undecided-by-lexicon.

WECHSEL_PREPS = {"an", "auf", "hinter", "in", "neben", "über", "unter", "vor", "zwischen"}

MODALS = {"können", "müssen", "sollen", "wollen", "dürfen", "mögen", "werden", "haben", "sein"}


def sentences(paths):
    """Yield one sentence at a time as a list of token dicts."""
    for p in paths:
        toks = []
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                if not line:
                    if toks:
                        yield toks
                    toks = []
                    continue
                if line.startswith("#"):
                    continue
                c = line.split("\t")
                if len(c) < 8 or "-" in c[0] or "." in c[0]:
                    continue  # multiword ranges / empty nodes
                feats = {}
                if c[5] != "_":
                    for f in c[5].split("|"):
                        k, _, v = f.partition("=")
                        feats[k] = v
                toks.append({
                    "id": int(c[0]), "form": c[1], "lemma": c[2], "upos": c[3],
                    "feats": feats, "head": int(c[6]) if c[6].isdigit() else 0,
                    "rel": c[7].split(":")[0], "fullrel": c[7],
                })
        if toks:
            yield toks


def main(paths, outdir):
    freq = collections.Counter()          # (word, lemma, pos) → count
    art = collections.Counter()           # (form, case) → count
    klammer = collections.Counter()       # bracket pattern → count
    valency = collections.Counter()       # (verb, rel, case) → count
    tekamolo = collections.Counter()      # (lane, lemma, rel) → count
    wechsel = collections.Counter()       # (prep, case, reading, rel) → count
    refl = collections.Counter()          # (verb, case, rel) → count
    relpro = collections.Counter()        # (form, case, rel, clause-shape) → count
    nsent = 0

    for toks in sentences(paths):
        nsent += 1
        by_id = {t["id"]: t for t in toks}
        for t in toks:
            pos = POS.get(t["upos"])
            if pos and t["form"]:
                freq[(t["form"].lower(), t["lemma"].lower(), pos)] += 1
            # (a) case-bearing determiners + pronouns — what makes German NP
            #     fronting decidable where English needs a witness.
            if t["upos"] in ("DET", "PRON") and "Case" in t["feats"]:
                art[(t["form"].lower(), t["feats"]["Case"])] += 1
            # (b) mined government frames: verb → (relation, case of dependent)
            if t["rel"] in ("obj", "obl", "iobj", "nsubj") and t["head"] in by_id:
                h = by_id[t["head"]]
                if h["upos"] in ("VERB", "AUX") and "Case" in t["feats"]:
                    valency[(h["lemma"].lower(), t["rel"], t["feats"]["Case"])] += 1
            # (c2) WECHSELPRÄPOSITIONEN — the shipped `grammar::wechsel` ambiguity
            #      COLLAPSED BY CASE, free: an/auf/hinter/in/neben/über/unter/vor/
            #      zwischen take Acc = directional (wohin, goal/change-of-state) vs
            #      Dat = static (wo, location). German marks morphologically what
            #      `WechselAmbiguity` would otherwise ticket to an LLM.
            if t["upos"] == "ADP" and t["lemma"].lower() in WECHSEL_PREPS and t["head"] in by_id:
                np_head = by_id[t["head"]]  # the ADP's head is its NP in UD
                case = np_head["feats"].get("Case")
                if case in ("Acc", "Dat"):
                    # Record the CASE and its GOVERNOR; do NOT assert a spatial
                    # reading here. `an dich denken` is Acc with no direction —
                    # a lexically governed frame, not a Wechsel alternation. The
                    # alternation itself is the evidence, resolved at emit time.
                    gov = by_id.get(np_head["head"])
                    gv = (gov["lemma"].lower()
                          if gov and gov["upos"] in ("VERB", "AUX") else "-")
                    wechsel[(t["lemma"].lower(), gv, case, np_head["rel"])] += 1
            # (c3) REFLEXIVES — `sich` Acc vs Dat is a VERB-FRAME discriminator
            #      (sich[Acc] waschen vs sich[Dat] etwas vorstellen).
            if t["lemma"].lower() == "sich" and t["head"] in by_id:
                h = by_id[t["head"]]
                if h["upos"] in ("VERB", "AUX"):
                    refl[(h["lemma"].lower(), t["feats"].get("Case", "-"), t["rel"])] += 1
            # (c4) RELATIVE PRONOUNS — the relativizer's CASE gives the antecedent's
            #      role INSIDE the relative clause (der Mann, DEN ich sah → antecedent
            #      is the object), and the clause is verb-final: coreference rails +
            #      right-corner commitment in one signal (the FSM Pos::Rel feeder).
            if "Rel" in t["feats"].get("PronType", "").split(",") and t["head"] in by_id:
                h = by_id[t["head"]]
                # Verb-finality is a property of the FINITE verb, not of the
                # relativizer's head. In periphrastic clauses (`das ich gesehen
                # habe`) the relativizer attaches to the PARTICIPLE while finite
                # `habe` is a later aux child — comparing against the participle
                # reverses the classification. Resolve the clause's verb cluster
                # and anchor on its finite member. (Codex #850 P1.)
                cluster = verb_cluster(h, toks)
                anchor = finite_of(cluster) or h
                cl_ids = {v["id"] for v in cluster}
                others = [x for x in toks if x["head"] == h["id"]
                          and x["rel"] != "punct" and x["id"] not in cl_ids]
                vfinal = bool(others) and anchor["id"] > max(x["id"] for x in others)
                relpro[(t["form"].lower(), t["feats"].get("Case", "-"), t["rel"],
                        "verb-final" if vfinal else "verb-nonfinal")] += 1
            # (c) TEKAMOLO lanes from adverbials/obliques
            if t["rel"] in ("advmod", "obl", "advcl", "mark", "cc"):
                lem = t["lemma"].lower()
                for lane, cues in (("Temporal", TEMPORAL), ("Kausal", KAUSAL),
                                   ("Modal", MODAL), ("Lokal", LOKAL)):
                    if lem in cues:
                        tekamolo[(lane, lem, t["rel"])] += 1

        # (d) Satzklammer geometry — the bracket this substrate parses natively.
        #     The Vorfeld is measured in CONSTITUENTS, not tokens: V2 means exactly
        #     ONE top-level constituent precedes the finite verb (`Der große Hund
        #     bellt` is V2 even though the verb is token 4).
        #     CLAUSE-LOCAL: the leftmost finite verb in the SENTENCE is not the
        #     matrix predicate — in `Wenn es regnet, bleibe ich zuhause` it is
        #     `regnet`, yielding a spurious V3+ instead of the matrix V2 headed by
        #     `bleibe`. Anchor on the ROOT's verb cluster, and take the bracket's
        #     nonfinite members from THAT cluster (a sentence-wide `nonfin` sweep
        #     pairs the predicate with a verb from another clause). (Codex #850 P1.)
        root = next((t for t in toks if t["rel"] == "root"
                     and t["upos"] in ("VERB", "AUX")), None)
        cluster = verb_cluster(root, toks) if root else []
        f0 = finite_of(cluster)
        if f0:
            nonfin = [v for v in cluster
                      if v["feats"].get("VerbForm") in ("Inf", "Part")]
            # Top-level constituents = direct dependents of the clause's lexical
            # head (the ROOT), minus the verb cluster itself; the Vorfeld is those
            # lying before the FINITE verb (`Der große Hund bellt` is V2 even
            # though the verb is token 4 — constituents, not tokens).
            cl_ids = {v["id"] for v in cluster}
            deps = [t for t in toks if t["head"] == root["id"]
                    and t["rel"] != "punct" and t["id"] not in cl_ids]
            vorfeld = [d for d in deps if d["id"] < f0["id"]]
            pos_class = ("V1" if not vorfeld else "V2" if len(vorfeld) == 1
                         else f"V{min(len(vorfeld) + 1, 4)}+")
            # WHAT occupies the Vorfeld — the fronting inventory (the reason German
            # donates regression tests: any constituent may front, case decides role).
            if len(vorfeld) == 1:
                v = vorfeld[0]
                klammer[("vorfeld", f"{v['rel']}:{v['feats'].get('Case', '-')}")] += 1
            if nonfin:
                span = max(x["id"] for x in nonfin) - f0["id"]
                klammer[(f"{pos_class}+bracket", "span>=3" if span >= 3 else "span<3")] += 1
            else:
                klammer[(pos_class, "simple")] += 1
            # Subordinate clauses: FINITE verb at the right corner. AUX is
            # included because German modals (können/müssen/sollen/wollen/dürfen/
            # mögen) are UPOS=AUX in UD — `weil er kommen muss` is headed by the
            # modal and was previously skipped entirely. Same finite-anchor rule
            # as the relativizer block. (CodeRabbit + Codex #850.)
            for t in toks:
                if t["rel"] in ("advcl", "ccomp", "csubj", "acl", "acl:relcl") \
                        and t["upos"] in ("VERB", "AUX"):
                    sub = verb_cluster(t, toks)
                    anchor = finite_of(sub) or t
                    sub_ids = {v["id"] for v in sub}
                    others = [x for x in toks if x["head"] == t["id"]
                              and x["rel"] != "punct" and x["id"] not in sub_ids]
                    if others:
                        vf = anchor["id"] > max(x["id"] for x in others)
                        klammer[("subordinate",
                                 "verb-final" if vf else "verb-nonfinal")] += 1

    import os
    os.makedirs(outdir, exist_ok=True)
    w = lambda name: open(os.path.join(outdir, name), "w", encoding="utf-8")

    # lexicon.tsv — COCA shape: word, lemma, pos, rank (1 = most frequent)
    agg = collections.Counter()
    for (word, lemma, pos), c in freq.items():
        agg[(word, lemma, pos)] += c
    with w("lexicon.tsv") as fh:
        fh.write("# German lexicon from UD German-GSD + German-HDT (CC BY-SA / CC BY-NC-SA)\n")
        fh.write("# word\tlemma\tpos\trank   (pos: n v j r i d p m c t x)\n")
        seen = set()
        for rank, ((word, lemma, pos), c) in enumerate(agg.most_common(), 1):
            if word in seen:
                continue  # keep the most frequent reading per surface form
            seen.add(word)
            fh.write(f"{word}\t{lemma}\t{pos}\t{rank}\n")

    with w("article_case.tsv") as fh:
        fh.write("# form\tcase\tcount — case-bearing determiners/pronouns (raw)\n")
        for (form, case), c in art.most_common():
            fh.write(f"{form}\t{case}\t{c}\n")

    # article_decidability.tsv — the HONEST boundary, German's analog of
    # PronounCase::Ambiguous. `den` is Acc (masc sg) AND Dat (plural), so it does
    # NOT decide alone; `dem` is Dat-only and DOES. Purity = share of the dominant
    # case; a consumer commits on case only above its threshold.
    forms = collections.defaultdict(collections.Counter)
    for (form, case), c in art.items():
        forms[form][case] += c
    with w("article_decidability.tsv") as fh:
        fh.write("# form\tdominant_case\tpurity_ppm\ttotal\tverdict   "
                 "(Decisive >= 950000ppm, Dominant >= 800000, Ambiguous below)\n")
        rows = []
        for form, cases in forms.items():
            tot = sum(cases.values())
            case, n = cases.most_common(1)[0]
            purity = (n * 1_000_000) // tot
            verdict = ("Decisive" if purity >= 950_000
                       else "Dominant" if purity >= 800_000 else "Ambiguous")
            rows.append((tot, form, case, purity, verdict))
        for tot, form, case, purity, verdict in sorted(rows, reverse=True):
            if tot >= 20:
                fh.write(f"{form}\t{case}\t{purity}\t{tot}\t{verdict}\n")

    with w("satzklammer.tsv") as fh:
        fh.write("# pattern\tdetail\tcount — finite-verb position + bracket span\n")
        for (pat, detail), c in klammer.most_common():
            fh.write(f"{pat}\t{detail}\t{c}\n")

    with w("valency.tsv") as fh:
        fh.write("# verb\trelation\tcase\tcount — mined government frames (ArgumentStatus source)\n")
        for (v, rel, case), c in valency.most_common():
            if c >= 2:
                fh.write(f"{v}\t{rel}\t{case}\t{c}\n")

    with w("tekamolo.tsv") as fh:
        fh.write("# lane\tlemma\trelation\tcount — adverbial cue → TEKAMOLO lane\n")
        for (lane, lem, rel), c in tekamolo.most_common():
            fh.write(f"{lane}\t{lem}\t{rel}\t{c}\n")

    # wechsel.tsv — case does NOT by itself mean directional/static. `an dich
    # denken` is Acc without direction; temporal and other governed senses are
    # likewise not spatial. The ALTERNATION is the evidence: a (prep, governor)
    # pair attested with BOTH cases is a live Wechsel contrast, so the spatial
    # reading is licensed; a pair locked to one case is lexically GOVERNED and
    # gets no semantic reading at all. Same discipline as the English extractor's
    # recipient-only PP fronting. (Codex #850 P2.)
    pair_cases = collections.defaultdict(set)
    for (prep, gov, case, _rel) in wechsel:
        pair_cases[(prep, gov)].add(case)
    with w("wechsel.tsv") as fh:
        fh.write("# prep\tgovernor\tcase\tframe\treading\trelation\tcount\n")
        fh.write("#   frame=alternating → the (prep,governor) pair is attested with BOTH cases,\n")
        fh.write("#     so the Wechsel contrast is live: Acc = directional (wohin) · Dat = static (wo).\n")
        fh.write("#   frame=governed → pair locked to ONE case (lexical/idiomatic frame, e.g.\n")
        fh.write("#     `an+Acc denken`): reading is '-' — case here encodes government, NOT space.\n")
        for (prep, gov, case, rel), c in wechsel.most_common():
            alternating = len(pair_cases[(prep, gov)]) > 1
            frame = "alternating" if alternating else "governed"
            reading = ("directional" if case == "Acc" else "static") if alternating else "-"
            fh.write(f"{prep}\t{gov}\t{case}\t{frame}\t{reading}\t{rel}\t{c}\n")

    with w("reflexive.tsv") as fh:
        fh.write("# verb\tcase\trelation\tcount — sich[Acc] vs sich[Dat] verb-frame discriminator\n")
        for (v, case, rel), c in refl.most_common():
            if c >= 2:
                fh.write(f"{v}\t{case}\t{rel}\t{c}\n")

    with w("relative_pronoun.tsv") as fh:
        fh.write("# form\tcase\trelation\tclause_shape\tcount — relativizer case = antecedent role IN the clause\n")
        for (form, case, rel, shape), c in relpro.most_common():
            fh.write(f"{form}\t{case}\t{rel}\t{shape}\t{c}\n")

    print(f"sentences: {nsent}")
    print(f"lexicon:   {len(seen)} surface forms")
    print(f"articles:  {len(art)} (form,case) pairs")
    print(f"valency:   {sum(1 for c in valency.values() if c >= 2)} frames (support>=2)")
    print(f"tekamolo:  {len(tekamolo)} cue rows")
    print(f"klammer:   {len(klammer)} patterns")
    print(f"wechsel:   {len(wechsel)} prep-case readings ({sum(wechsel.values())} tokens)")
    print(f"reflexive: {sum(1 for c in refl.values() if c>=2)} verb frames")
    print(f"relpro:    {len(relpro)} relativizer rows ({sum(relpro.values())} tokens)")


if __name__ == "__main__":
    main(sys.argv[1:-1], sys.argv[-1])
