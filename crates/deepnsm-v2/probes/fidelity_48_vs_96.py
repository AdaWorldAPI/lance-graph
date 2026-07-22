#!/usr/bin/env python3
"""deepnsm (CAM-PQ 48, POINT) vs deepnsm_v2 (CAM-PQ 96, DISTRIBUTION):
how much of Jina's MEANING does each substrate preserve?

Real Jina-v3 96-d embeddings (ground truth) quantized two ways:
  v1  48-bit POINT        : 6 subspaces × 16-d, 256 centroids each  → 6 bytes
  v2  96-bit DISTRIBUTION : 12 sub-subspaces × 8-d, 256 centroids   → 12 bytes  (each 16-d subspace split into a 256:256 pair)
Then Spearman rho( substrate cosine-distance , Jina cosine-distance )
on random pairs. v2 (finer quantization) should preserve MORE meaning.
Routing (frequency) already shown orthogonal — this is the MEANING axis.
"""
import csv, os, json, math, subprocess, numpy as np
SP = os.path.dirname(os.path.abspath(__file__))
KEY = os.environ.get("JINA_API_KEY", "").strip().strip('"').strip("'")
assert KEY.startswith("jina_")
rng = np.random.default_rng(0x9E3779B9)

# vocabulary: single-word alpha tokens from DocuScope (broad coverage)
toks = []
seen = set()
for r in csv.DictReader(open(f"{SP}/coca_tokens.csv")):
    t = r["Token"].strip().lower()
    if t.isalpha() and len(t) > 1 and t not in seen:
        seen.add(t); toks.append(t)
rng.shuffle(toks)
toks = toks[:3000]
print(f"vocab: {len(toks)} single-word tokens")

def embed(batch):
    body = json.dumps({"model":"jina-embeddings-v3","task":"text-matching","dimensions":96,"input":batch})
    out = subprocess.run(["curl","-sS","--cacert","/root/.ccr/ca-bundle.crt","-X","POST",
        "https://api.jina.ai/v1/embeddings","-H",f"Authorization: Bearer {KEY}",
        "-H","Content-Type: application/json","-d",body], capture_output=True, text=True, timeout=180)
    return [e["embedding"] for e in json.loads(out.stdout)["data"]]
E = []
for i in range(0, len(toks), 100):
    E.extend(embed(toks[i:i+100]))
E = np.asarray(E, dtype=np.float64)   # N×96 Jina ground truth
print(f"embedded {E.shape}")

def kmeans(X, k=256, iters=20, seed=0):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = (X*X).sum(1)[:,None] + (C*C).sum(1)[None,:] - 2*X@C.T
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = X[m].mean(0)
    return C, a

def pq_reconstruct(X, n_sub):
    """Split 96-d into n_sub equal subspaces, PQ-quantize each, return reconstruction."""
    d = X.shape[1] // n_sub
    R = np.empty_like(X)
    mse = 0.0
    for s in range(n_sub):
        sub = X[:, s*d:(s+1)*d]
        C, a = kmeans(sub, 256, 20, seed=1000+s)
        rec = C[a]
        R[:, s*d:(s+1)*d] = rec
        mse += ((sub - rec)**2).sum()
    return R, mse / X.size

R48, mse48 = pq_reconstruct(E, 6)    # POINT:        6 × 16-d  → 6 bytes  (48-bit)
R96, mse96 = pq_reconstruct(E, 12)   # DISTRIBUTION: 12 × 8-d  → 12 bytes (96-bit)

def norm(a): return a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
En, R48n, R96n = norm(E), norm(R48), norm(R96)

# random eval pairs
N = len(toks)
I = rng.integers(0, N, 300_000); J = rng.integers(0, N, 300_000)
ok = I != J; I, J = I[ok], J[ok]
def cdist(A): return 1.0 - (A[I]*A[J]).sum(1)   # cosine distance over pairs
dJ, d48, d96 = cdist(En), cdist(R48n), cdist(R96n)

def spearman(x, y):
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    rx = rx - rx.mean(); ry = ry - ry.mean()
    return float((rx*ry).sum() / math.sqrt((rx*rx).sum()*(ry*ry).sum()))
r48 = spearman(d48, dJ); r96 = spearman(d96, dJ)

print("\n═══════ MEANING FIDELITY vs JINA — CAM-PQ 48 (POINT) vs 96 (DISTRIBUTION) ═══════")
print(f"vocab {N}   eval pairs {len(I):,}   ground truth jina-v3/96d")
print(f"\n  deepnsm    48-bit POINT        6×16-d   recon MSE = {mse48:.5f}   rho(dist, Jina) = {r48:.4f}")
print(f"  deepnsm_v2 96-bit DISTRIBUTION 12×8-d   recon MSE = {mse96:.5f}   rho(dist, Jina) = {r96:.4f}")
print(f"\n  meaning preserved: 48→{r48:.3f}  96→{r96:.3f}   Δρ = {r96-r48:+.4f}  ({100*(r96-r48)/r48:+.1f}%)")
print(f"  reconstruction err: 96-bit is {100*(mse48-mse96)/mse48:.1f}% lower MSE than 48-bit")
print("\nVERDICT: " + ("96-bit DISTRIBUTION preserves MORE Jina meaning than 48-bit POINT"
      if r96 > r48 else "no fidelity gain from 96-bit — investigate"))
