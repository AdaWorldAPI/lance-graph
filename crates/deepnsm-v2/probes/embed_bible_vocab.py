import json, os, subprocess, time, numpy as np
KEY = os.environ.get("JINA_API_KEY","").strip().strip('"').strip("'"); assert KEY.startswith("jina_")
CA = os.environ.get("JINA_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
vocab = [w for w in open("bible_vocab.txt").read().splitlines() if w.strip()]
def embed(batch):
    body = json.dumps({"model":"jina-embeddings-v3","task":"text-matching","dimensions":96,"input":batch})
    args = ["curl","-sS","-f","--config","-","-X","POST","https://api.jina.ai/v1/embeddings",
            "-H","Content-Type: application/json","-d",body]
    if CA and os.path.exists(CA):
        args[2:2] = ["--cacert", CA]
    out = subprocess.run(args, input=f'header = "Authorization: Bearer {KEY}"\n',
                         capture_output=True, text=True, timeout=180)
    if out.returncode != 0:
        raise RuntimeError(f"curl exit {out.returncode}: {out.stderr[:200]}")
    return [e["embedding"] for e in json.loads(out.stdout)["data"]]
# resume from partial if present
E = list(np.load("partial_emb.npy")) if os.path.exists("partial_emb.npy") else []
start = len(E)
print(f"resuming at {start}/{len(vocab)}", flush=True)
for i in range(start, len(vocab), 100):
    for attempt in range(3):
        try:
            E.extend(embed(vocab[i:i+100])); break
        except Exception as ex:
            if attempt == 2: raise
            time.sleep(2**attempt)
    if (i//100) % 20 == 0:
        np.save("partial_emb.npy", np.asarray(E, dtype=np.float32))
        print(f"batch {i//100}/{(len(vocab)+99)//100}", flush=True)
E = np.asarray(E, dtype=np.float32)
np.save("bible_vocab_emb96.npy", E)
print("saved", E.shape)
