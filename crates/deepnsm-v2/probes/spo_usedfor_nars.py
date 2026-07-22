#!/usr/bin/env python3
"""Can the 0.828 substrate carry SPO 2^3 NARS reasoning over a 'used for' group?

'used for' is FUNCTIONAL, not similarity — so reasoning rides on
analogical relation-generalization: if S1 used_for O and the substrate
clusters S2≈S1, NARS transfers S2 used_for O. Test whether the substrate
(48-bit POINT vs 96-bit DISTRIBUTION vs Jina-full) recovers the used_for
GROUP structure that makes that transfer sound.

Metric 1 — function clustering: k-NN purity of same-used_for subjects.
Metric 2 — analogical transfer (the NARS 2^3 SP->O leg): hold out each
  subject's object; predict it by its nearest same-substrate neighbor's
  object; accuracy = does similarity transfer the used_for relation.
Metric 3 — relation vs similarity: does adding the 'used for' predicate
  token move S toward its functional O (SP->O) vs raw S->O similarity.
"""
import os, json, subprocess, numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("JINA_API_KEY","").strip().strip('"').strip("'"); assert KEY.startswith("jina_")
rng = np.random.default_rng(7)

# used_for groups: object(function) -> similar subjects that share it
GROUPS = {
 "pounding":   ["hammer","mallet","sledgehammer"],
 "cutting":    ["knife","scalpel","cleaver","blade"],
 "writing":    ["pen","pencil","marker"],
 "sweeping":   ["broom","besom"],
 "digging":    ["shovel","spade","trowel"],
 "cooking":    ["stove","oven","skillet"],
 "measuring":  ["ruler","tape","caliper"],
 "fastening":  ["screw","bolt","rivet","nail"],
 "illuminating":["lamp","lantern","torch","flashlight"],
 "drinking":   ["cup","mug","glass","goblet"],
 "sitting":    ["chair","stool","bench"],
 "sailing":    ["boat","ship","yacht","canoe"],
 "flying":     ["plane","jet","helicopter"],
 "cleaning":   ["mop","sponge","cloth"],
 "typing":     ["keyboard","typewriter"],
}
subs, func = [], []
for o, ss in GROUPS.items():
    for s in ss:
        subs.append(s); func.append(o)
objs = list(GROUPS.keys())
print(f"{len(subs)} subjects across {len(objs)} used_for groups")

def embed(batch, dim=96):
    body = json.dumps({"model":"jina-embeddings-v3","task":"text-matching","dimensions":dim,"input":batch})
    out = subprocess.run(["curl","-sS","--cacert","/root/.ccr/ca-bundle.crt","-X","POST",
        "https://api.jina.ai/v1/embeddings","-H",f"Authorization: Bearer {KEY}",
        "-H","Content-Type: application/json","-d",body], capture_output=True, text=True, timeout=120)
    return np.asarray([e["embedding"] for e in json.loads(out.stdout)["data"]], dtype=np.float64)

S = embed(subs)                    # subjects, Jina 96-d
O = embed(objs)                    # objects (the functions)
P = embed(["used for"])            # the predicate token

def kmeans(X,k,it,seed):
    r=np.random.default_rng(seed); C=X[r.choice(len(X),min(k,len(X)),replace=False)].copy()
    for _ in range(it):
        d=(X*X).sum(1)[:,None]+(C*C).sum(1)[None,:]-2*X@C.T; a=d.argmin(1)
        for j in range(len(C)):
            m=a==j
            if m.any(): C[j]=X[m].mean(0)
    return C,a
def pq(X,nsub):
    d=X.shape[1]//nsub; R=np.empty_like(X)
    # train codebook on a big generic vocab so it's not fit to these 45 words
    for s in range(nsub):
        C,_=kmeans(BIG[:,s*d:(s+1)*d],256,20,seed=100+s)
        sub=X[:,s*d:(s+1)*d]
        dd=(sub*sub).sum(1)[:,None]+(C*C).sum(1)[None,:]-2*sub@C.T
        R[:,s*d:(s+1)*d]=C[dd.argmin(1)]
    return R
# generic codebook-training vocab (independent of the 45 test words)
import csv
gen=[r["Token"].strip().lower() for r in csv.DictReader(open(f"{SP}/coca_tokens.csv"))
     if r["Token"].strip().isalpha() and len(r["Token"].strip())>1]
rng.shuffle(gen); gen=gen[:1500]
BIG=embed_all=[]
for i in range(0,len(gen),100): embed_all.extend(embed(gen[i:i+100]))
BIG=np.asarray(embed_all,dtype=np.float64)
print(f"codebook train vocab {BIG.shape}")

S48, S96 = pq(S,6), pq(S,12)
def norm(a): return a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-12)
reps = {"Jina-full":norm(S), "96-bit DIST":norm(S96), "48-bit POINT":norm(S48)}

def knn_purity(X, k=1):
    sim = X@X.T; np.fill_diagonal(sim,-9)
    nn = sim.argsort(1)[:,-k:]
    return np.mean([func[i]==func[j] for i in range(len(X)) for j in nn[i]])
def transfer_acc(X):
    # NARS analogical SP->O: predict each subject's object via nearest neighbor's object
    sim=X@X.T; np.fill_diagonal(sim,-9); nn=sim.argmax(1)
    return np.mean([func[nn[i]]==func[i] for i in range(len(X))])

print("\n═════ SPO 2^3 'used for' reasoning capacity vs meaning fidelity ═════")
print(f"{'representation':<14} {'kNN@1 purity':>12} {'kNN@2 purity':>12} {'analogical SP->O':>16}")
for name,X in reps.items():
    print(f"{name:<14} {knn_purity(X,1):>12.3f} {knn_purity(X,2):>12.3f} {transfer_acc(X):>16.3f}")

# Metric 3: does the 'used for' predicate token move a subject toward its true function-object?
On = norm(O); Pn = norm(P)[0]
def sp_o_rank(X):
    # for each subject: rank objects by distance from (S + P) vs from S alone; is true O higher with P?
    with_p, without_p = [], []
    Xs = X  # subjects in same space
    # map objects into same normalized space via Jina-full for a fair O target
    for i,s in enumerate(subs):
        true_o = objs.index(func[i])
        base = norm((S[i])[None,:])[0]
        withp = norm((S[i]+P[0])[None,:])[0]
        r_base = (On@base); r_with = (On@withp)
        without_p.append((-r_base).argsort().tolist().index(true_o))
        with_p.append((-r_with).argsort().tolist().index(true_o))
    return np.mean(without_p), np.mean(with_p)
rb, rw = sp_o_rank(reps["Jina-full"])
print(f"\nMetric 3 (predicate token): mean rank of true used_for-object (0=top)")
print(f"  S alone      : {rb:.2f}")
print(f"  S + 'used for': {rw:.2f}   → predicate token {'HELPS (relation≠similarity)' if rw<rb else 'no help'}")

pur = knn_purity(reps["96-bit DIST"],1)
print(f"\nVERDICT: 96-bit substrate kNN@1 function-purity = {pur:.3f} "
      f"({'supports' if pur>0.5 else 'too weak for'} analogical used_for NARS)")
