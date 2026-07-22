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

CA = os.environ.get("JINA_CA_BUNDLE", "")  # optional; empty = system trust store
def embed(batch):
    body = json.dumps({"model":"jina-embeddings-v3","task":"text-matching","dimensions":96,"input":batch})
    args = ["curl","-sS","-f","--config","-","-X","POST","https://api.jina.ai/v1/embeddings",
            "-H","Content-Type: application/json","-d",body]
    if CA:
        args[2:2] = ["--cacert", CA]
    # Authorization rides stdin config, never argv (invisible to process listings).
    out = subprocess.run(args, input=f'header = "Authorization: Bearer {KEY}"\n',
                         capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise RuntimeError(f"jina embed failed (curl exit {out.returncode}): {out.stderr[:200]}")
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

def pq_reconstruct(X_train, X_eval, n_sub):
    """HELD-OUT PQ: fit each subspace codebook on X_train ONLY, then encode and
    reconstruct X_eval (disjoint) — out-of-sample fidelity, not memorization
    (the in-sample version overstated absolute fidelity; PR #801 review)."""
    d = X_eval.shape[1] // n_sub
    R = np.empty_like(X_eval)
    mse = 0.0
    for s in range(n_sub):
        C, _ = kmeans(X_train[:, s*d:(s+1)*d], 256, 20, seed=1000+s)
        sub = X_eval[:, s*d:(s+1)*d]
        dist = (sub*sub).sum(1)[:,None] + (C*C).sum(1)[None,:] - 2*sub@C.T
        rec = C[dist.argmin(1)]
        R[:, s*d:(s+1)*d] = rec
        mse += ((sub - rec)**2).sum()
    return R, mse / X_eval.size

# Disjoint split: codebooks trained on 2000, scored on the held-out 1000.
N_TRAIN = 2000
Etr, Eev = E[:N_TRAIN], E[N_TRAIN:]
R48, mse48 = pq_reconstruct(Etr, Eev, 6)    # POINT:        6 x 16-d  -> 6 bytes  (48-bit)
R96, mse96 = pq_reconstruct(Etr, Eev, 12)   # DISTRIBUTION: 12 x 8-d  -> 12 bytes (96-bit)
E = Eev  # all downstream scoring runs on the held-out set only
toks = toks[N_TRAIN:]

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
print(f"vocab {N}   eval pairs {len(I):,}   ground truth jina-v3/96d   HELD-OUT eval (codebooks fit on disjoint 2000)")
print(f"\n  deepnsm    48-bit POINT        6×16-d   recon MSE = {mse48:.5f}   rho(dist, Jina) = {r48:.4f}")
print(f"  deepnsm_v2 96-bit DISTRIBUTION 12×8-d   recon MSE = {mse96:.5f}   rho(dist, Jina) = {r96:.4f}")
print(f"\n  meaning preserved: 48→{r48:.3f}  96→{r96:.3f}   Δρ = {r96-r48:+.4f}  ({100*(r96-r48)/r48:+.1f}%)")
print(f"  reconstruction err: 96-bit is {100*(mse48-mse96)/mse48:.1f}% lower MSE than 48-bit")
print("\nVERDICT: " + ("96-bit DISTRIBUTION preserves MORE Jina meaning than 48-bit POINT"
      if r96 > r48 else "no fidelity gain from 96-bit — investigate"))
