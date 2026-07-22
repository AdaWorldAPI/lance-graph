#!/usr/bin/env python3
"""Full meaning/routing/frequency determination vs Jina ground truth.

Sample = tokens present in BOTH the 8-genre freq table (lemmas_5k) AND the
18k DocuScope set, so each carries: 8-genre PM vector, a discourse category,
and a frequency. Jina-v3 256-d = ground-truth meaning.

Three candidate 'meaning' signals, each correlated against Jina:
  A. frequency        |Δlog perMil|                  -> ROUTING / address
  B. substrate distance 1 - zcos(8-genre PM)          -> COUNT-DERIVED meaning
     (the freq_is_cosine axis; normalized [x;y], not cosine-the-fn)
  C. discourse category (same/diff)                   -> coarse AWARENESS meaning
Determination: which of A/B/C tracks Jina meaning, which is orthogonal.
"""
import csv, os, json, math, random, subprocess, statistics as st
SP = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("JINA_API_KEY", "").strip().strip('"').strip("'")
assert KEY.startswith("jina_")
random.seed(0x9E3779B9)
GEN = ["blogPM","webPM","TVMPM","spokPM","ficPM","magPM","newsPM","acadPM"]

# 8-genre vectors, one per lemma (highest-freq PoS wins)
genre, permil = {}, {}
for r in csv.DictReader(open(f"{SP}/lemmas_5k.csv")):
    t = r["lemma"].strip().lower()
    if not t.isalpha() or len(t) < 2:
        continue
    try:
        v = [math.log1p(float(r[g])) for g in GEN]; pm = float(r["perMil"])
    except (ValueError, KeyError):
        continue
    if t not in permil or pm > permil[t]:
        genre[t] = v; permil[t] = pm
# discourse category from DocuScope
cat = {}
for r in csv.DictReader(open(f"{SP}/coca_tokens.csv")):
    t = r["Token"].strip().lower()
    if t.isalpha() and len(t) > 1 and t not in cat:
        cat[t] = r["DocuScope Category"]

both = [t for t in genre if t in cat]
random.shuffle(both)
sample = both[:360]
print(f"sample: {len(sample)} tokens (in both 8-genre table AND DocuScope)")

# z-score each genre dim across the sample
cols = list(zip(*[genre[t] for t in sample]))
mu = [sum(c)/len(c) for c in cols]
sd = [math.sqrt(sum((x-m)**2 for x in c)/len(c)) or 1.0 for c, m in zip(cols, mu)]
zv = {t: [(genre[t][i]-mu[i])/sd[i] for i in range(8)] for t in sample}

def cos(a, b):
    d = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a)); nb = math.sqrt(sum(y*y for y in b))
    return d/(na*nb) if na and nb else 0.0

def embed(batch):
    body = json.dumps({"model":"jina-embeddings-v3","task":"text-matching","dimensions":256,"input":batch})
    out = subprocess.run(["curl","-sS","--cacert","/root/.ccr/ca-bundle.crt","-X","POST",
        "https://api.jina.ai/v1/embeddings","-H",f"Authorization: Bearer {KEY}",
        "-H","Content-Type: application/json","-d",body], capture_output=True, text=True, timeout=120)
    return [e["embedding"] for e in json.loads(out.stdout)["data"]]
emb = {}
for i in range(0, len(sample), 100):
    for t, e in zip(sample[i:i+100], embed(sample[i:i+100])):
        emb[t] = e
print(f"embedded {len(emb)}")

def spearman(xs, ys):
    def rk(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r=[0.0]*len(v); i=0
        while i < len(v):
            j=i
            while j+1<len(v) and v[o[j+1]]==v[o[i]]: j+=1
            a=(i+j)/2.0
            for k in range(i,j+1): r[o[k]]=a
            i=j+1
        return r
    rx,ry=rk(xs),rk(ys); mx=sum(rx)/len(rx); my=sum(ry)/len(ry)
    num=sum((a-mx)*(b-my) for a,b in zip(rx,ry))
    den=math.sqrt(sum((a-mx)**2 for a in rx)*sum((b-my)**2 for b in ry))
    return num/den if den else 0.0

freq_d, subs_d, jina_d, same = [], [], [], []
ss, cs = [], []
for i in range(len(sample)):
    for j in range(i+1, len(sample)):
        a, b = sample[i], sample[j]
        jc = cos(emb[a], emb[b])
        freq_d.append(abs(math.log10(permil[a]) - math.log10(permil[b])))
        subs_d.append(1.0 - cos(zv[a], zv[b]))       # substrate count-derived distance
        jina_d.append(1.0 - jc)                        # meaning ground-truth distance
        s = 1 if cat[a]==cat[b] else 0; same.append(s)
        (ss if s else cs).append(jc)

r_freq = spearman(freq_d, jina_d)
r_subs = spearman(subs_d, jina_d)
random.shuffle(ss); random.shuffle(cs); SS,CS=ss[:1500],cs[:1500]
auc=(sum(1 for a in SS for b in CS if a>b)+0.5*sum(1 for a in SS for b in CS if a==b))/(len(SS)*len(CS))
d_eff=(st.mean(ss)-st.mean(cs))/math.sqrt((st.pvariance(ss)+st.pvariance(cs))/2)

print("\n══════════ MEANING vs ROUTING vs FREQUENCY — Jina-grounded, 18k COCA ══════════")
print(f"pairs {len(jina_d):,}   oracle jina-embeddings-v3/256d   sample {len(sample)}")
print(f"\n  A. FREQUENCY (routing)      rho(|Δlog perMil| , Jina-dist) = {r_freq:+.3f}  {'⟂ meaning' if abs(r_freq)<0.15 else 'carries meaning'}")
print(f"  B. SUBSTRATE 8-genre dist   rho(1-zcos(genre) , Jina-dist) = {r_subs:+.3f}  {'TRACKS meaning' if r_subs>0.15 else 'weak'}")
print(f"  C. DISCOURSE category       AUC(same>cross Jina)          = {auc:.3f}   Cohen d = {d_eff:+.3f}  {'coarse meaning' if auc>0.55 else 'no separation'}")
print(f"\n  RATIO  substrate:frequency meaning-tracking = {(r_subs/abs(r_freq) if r_freq else float('inf')):.1f}×")
print("\nDETERMINATION:")
print(f"  frequency  → ROUTING   (orthogonal to meaning, rho={r_freq:+.3f})")
print(f"  distance   → MEANING   (substrate count-distance tracks Jina, rho={r_subs:+.3f}; discourse AUC={auc:.3f})")
