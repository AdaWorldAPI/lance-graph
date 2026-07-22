import numpy as np, math, struct
rng = np.random.default_rng(0x9E3779B9)
E = np.load("bible_vocab_emb96.npy").astype(np.float64)
vocab = [w for w in open("bible_vocab.txt").read().splitlines() if w.strip()]
N = len(E); assert N == len(vocab)

def kmeans(X, k=256, iters=25, seed=0):
    r = np.random.default_rng(seed)
    C = X[r.choice(len(X), k, replace=False)].copy()
    for _ in range(iters):
        d = (X*X).sum(1)[:,None] + (C*C).sum(1)[None,:] - 2*X@C.T
        a = d.argmin(1)
        for j in range(k):
            m = a == j
            if m.any(): C[j] = X[m].mean(0)
    return C

def assign(X, C):
    d = (X*X).sum(1)[:,None] + (C*C).sum(1)[None,:] - 2*X@C.T
    return d.argmin(1)

def pq_fit(Xtr, n_sub, dim_total=96):
    d = dim_total // n_sub
    return [kmeans(Xtr[:, s*d:(s+1)*d], 256, 25, seed=1000+s) for s in range(n_sub)]
def pq_encode(X, books):
    n_sub = len(books); d = X.shape[1] // n_sub
    return np.stack([assign(X[:, s*d:(s+1)*d], books[s]) for s in range(n_sub)], axis=1)
def pq_recon(codes, books, d):
    return np.concatenate([books[s][codes[:, s]] for s in range(len(books))], axis=1)

def rq_fit(Xtr):  # 96-bit POINT control: 6 subspaces × (coarse-256 + residual-256)
    d = 16; books = []
    for s in range(6):
        sub = Xtr[:, s*d:(s+1)*d]
        C1 = kmeans(sub, 256, 25, seed=2000+s)
        res = sub - C1[assign(sub, C1)]
        C2 = kmeans(res, 256, 25, seed=3000+s)
        books.append((C1, C2))
    return books
def rq_recon(X, books):
    d = 16; R = np.empty_like(X)
    for s in range(6):
        sub = X[:, s*d:(s+1)*d]; C1, C2 = books[s]
        r1 = C1[assign(sub, C1)]
        R[:, s*d:(s+1)*d] = r1 + C2[assign(sub - r1, C2)]
    return R

def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64); ry = np.argsort(np.argsort(y)).astype(np.float64)
    rx -= rx.mean(); ry -= ry.mean()
    return float((rx*ry).sum()/math.sqrt((rx*rx).sum()*(ry*ry).sum()))
def norm(a): return a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-12)

# ── held-out measurement: train 10,000 / eval 2,543 ──
perm = rng.permutation(N); tr, ev = perm[:10000], perm[10000:]
Xtr, Xev = E[tr], E[ev]
b96 = pq_fit(Xtr, 12); b48 = pq_fit(Xtr, 6); brq = rq_fit(Xtr)
R96 = pq_recon(pq_encode(Xev, b96), b96, 8)
R48 = pq_recon(pq_encode(Xev, b48), b48, 16)
RRQ = rq_recon(Xev, brq)
En, n96, n48, nrq = norm(Xev), norm(R96), norm(R48), norm(RRQ)
I = rng.integers(0, len(ev), 300_000); J = rng.integers(0, len(ev), 300_000)
ok = I != J; I, J = I[ok], J[ok]
dJ = 1 - (En[I]*En[J]).sum(1)
r96 = spearman(1-(n96[I]*n96[J]).sum(1), dJ)
r48 = spearman(1-(n48[I]*n48[J]).sum(1), dJ)
rrq = spearman(1-(nrq[I]*nrq[J]).sum(1), dJ)
mse = lambda R: float(((Xev-R)**2).mean())
print(f"HELD-OUT (train 10000 / eval {len(ev)}, pairs {ok.sum():,}) — Bible vocab, Jina-v3 96d")
print(f"  48-bit POINT   6×16d PQ          rho {r48:.4f}  mse {mse(R48):.5f}")
print(f"  96-bit POINT   6×(coarse+resid)  rho {rrq:.4f}  mse {mse(RRQ):.5f}   <- equal-budget point control")
print(f"  96-bit DIST    12×8d PQ          rho {r96:.4f}  mse {mse(R96):.5f}")
gate = r96 >= 0.6 and r96 > r48
print(f"  GATE rho96>=0.6 and rho96>rho48: {'PASS' if gate else 'KILL'}")

# ── artifact: train on ALL data, encode all words ──
bart = pq_fit(E, 12)
codes = pq_encode(E, bart).astype(np.uint8)
# d_max: 99.9th percentile of pairwise code distances (for [0,1] similarity clamp)
recon_all = pq_recon(codes.astype(np.int64), bart, 8)
K = rng.integers(0, N, 200_000); L = rng.integers(0, N, 200_000)
okk = K != L
dd = ((recon_all[K[okk]]-recon_all[L[okk]])**2).sum(1)
d_max = float(np.quantile(dd, 0.999))
print(f"artifact: 12 axes × 256 × 8d; d_max(p99.9) = {d_max:.4f}")

with open("cam96_codebook.bin","wb") as f:
    f.write(b"CAM96CB1")
    f.write(struct.pack("<IIIf", 12, 8, 256, d_max))
    for C in bart:
        f.write(C.astype("<f4").tobytes())
with open("cam96_codes.bin","wb") as f:
    f.write(b"CAM96WD1")
    f.write(struct.pack("<I", N))
    f.write(codes.tobytes())
import os
print("codebook", os.path.getsize("cam96_codebook.bin"), "B; codes", os.path.getsize("cam96_codes.bin"), "B")
