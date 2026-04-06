#!/usr/bin/env python3
"""
OSINT Pipeline Prototype: search → crawl → clean → embed → think → learn

Demonstrates the full loop:
  1. Query → Google search (via requests)
  2. Top-N URLs → fetch HTML (spider-like)
  3. HTML → clean text (Reader-LM style, simplified)
  4. Text → sentences → codebook centroids
  5. Centroids → thinking engine (f32, softmax T=0.01)
  6. Peaks → contrastive table update
  7. NARS confidence check → if low, generate new queries
"""

import requests
import json
import struct
import numpy as np
import re
import time
from urllib.parse import quote_plus
from bs4 import BeautifulSoup

# ═══ CONFIG ═══
CODEBOOK_PATH = "/tmp/codebooks/jina-v5-256"
N_CENT = 256

def load_table():
    """Load f32 cosine table and codebook index."""
    table = np.fromfile(f"{CODEBOOK_PATH}/cosine_matrix_{N_CENT}x{N_CENT}.f32",
                        dtype=np.float32).reshape(N_CENT, N_CENT)
    idx = np.fromfile(f"{CODEBOOK_PATH}/codebook_index.u16", dtype=np.uint16)
    return table, idx

def softmax_think(table, centroid_ids, cycles=10, T=0.01):
    """Signed softmax thinking on the codebook table."""
    n = table.shape[0]
    energy = np.zeros(n, dtype=np.float32)
    for c in centroid_ids:
        if c < n:
            energy[c] += 1.0
    total = energy.sum()
    if total > 0:
        energy /= total

    inv_t = 1.0 / T
    for _ in range(cycles):
        nxt = table.T @ energy
        mx = nxt.max()
        exps = np.exp((nxt - mx) * inv_t)
        energy = exps / exps.sum()

    return energy

def energy_to_peaks(energy, k=5):
    """Extract top-k peaks from energy distribution."""
    idx = np.argsort(-energy)[:k]
    return [(int(i), float(energy[i])) for i in idx]

# ═══ STEP 1: SEARCH ═══
def search_google(query, n=5):
    """Fetch search results via Google (simplified)."""
    # Use DuckDuckGo HTML (no API key needed)
    url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; OSINT-Pipeline/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        results = []
        for a in soup.select("a.result__a")[:n]:
            href = a.get("href", "")
            title = a.get_text(strip=True)
            if href and title:
                results.append({"url": href, "title": title})
        return results
    except Exception as e:
        print(f"  Search error: {e}")
        return []

# ═══ STEP 2: CRAWL ═══
def fetch_page(url, timeout=10):
    """Fetch a page and extract text (Reader-LM style, simplified)."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; OSINT-Pipeline/1.0)"}
    try:
        resp = requests.get(url, headers=headers, timeout=timeout)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")

        # Remove script, style, nav, footer
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Extract text from paragraphs
        paragraphs = []
        for p in soup.find_all(["p", "h1", "h2", "h3", "li"]):
            text = p.get_text(strip=True)
            if len(text) > 30:  # skip short fragments
                paragraphs.append(text)

        return "\n\n".join(paragraphs[:20])  # cap at 20 paragraphs
    except Exception as e:
        return f"[Error: {e}]"

# ═══ STEP 3: TEXT → CENTROIDS ═══
def text_to_centroids(text, codebook_idx):
    """Map text bytes to codebook centroids (simplified tokenization)."""
    # Real pipeline: Qwen3 BPE tokenizer → token IDs → codebook_idx
    # Simplified: byte-level → codebook_idx (lossy but functional)
    token_ids = [b for b in text.encode("utf-8") if b < len(codebook_idx)]
    centroids = [int(codebook_idx[t]) for t in token_ids]
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in centroids:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique

# ═══ STEP 4: CONTRASTIVE UPDATE ═══
def contrastive_update(table, centroids_a, centroids_b, similarity, alpha=0.01):
    """Update table from observed similarity between two texts."""
    updates = 0
    total_error = 0.0
    for ca in centroids_a[:10]:  # cap at 10 centroids per text
        for cb in centroids_b[:10]:
            if ca != cb and ca < table.shape[0] and cb < table.shape[0]:
                old = table[ca, cb]
                error = similarity - old
                table[ca, cb] += alpha * error
                table[cb, ca] += alpha * error  # symmetric
                total_error += abs(error)
                updates += 1
    return updates, total_error / max(updates, 1)

# ═══ STEP 5: NARS CONFIDENCE ═══
class NarsTruth:
    def __init__(self, frequency=0.5, confidence=0.0):
        self.f = frequency
        self.c = confidence

    def revise(self, observation, weight=1.0):
        """NARS revision: update truth from new observation."""
        k = 1.0  # evidential horizon
        w1 = self.c / (1 - self.c + 1e-10) if self.c < 1 else 100
        w2 = weight
        w = w1 + w2
        new_f = (w1 * self.f + w2 * observation) / w
        new_c = w / (w + k)
        self.f = new_f
        self.c = new_c

    def __repr__(self):
        return f"<f={self.f:.3f}, c={self.c:.3f}>"

# ═══ MAIN LOOP ═══
def osint_loop(initial_query, max_rounds=3, results_per_round=3):
    """The full OSINT loop: search → crawl → think → learn → repeat."""

    table, codebook_idx = load_table()
    table = table.copy()  # mutable copy for learning

    nars = {}  # per-centroid-pair NARS truth
    all_texts = []
    total_updates = 0

    query = initial_query

    for round_num in range(max_rounds):
        print(f"\n{'='*70}")
        print(f"  ROUND {round_num + 1}: query = \"{query}\"")
        print(f"{'='*70}")

        # Search
        print(f"\n  [1] Searching...")
        results = search_google(query, n=results_per_round)
        if not results:
            print("  No results found. Trying alternate query...")
            query = f"{query} explained"
            results = search_google(query, n=results_per_round)

        for r in results:
            print(f"    {r['title'][:60]}")

        # Crawl + Clean
        print(f"\n  [2] Crawling {len(results)} pages...")
        texts = []
        for r in results:
            url = r.get("url", "")
            if not url.startswith("http"):
                continue
            text = fetch_page(url)
            if text and not text.startswith("[Error"):
                texts.append({"url": url, "title": r["title"], "text": text[:2000]})
                print(f"    ✓ {r['title'][:40]}: {len(text)} chars")
            else:
                print(f"    ✗ {r['title'][:40]}: failed")

        if not texts:
            print("  No text fetched. Skipping round.")
            continue

        # Map to centroids
        print(f"\n  [3] Mapping to codebook centroids...")
        text_centroids = []
        for t in texts:
            cents = text_to_centroids(t["text"], codebook_idx)
            text_centroids.append(cents)
            print(f"    {t['title'][:40]}: {len(cents)} unique centroids")

        # Think
        print(f"\n  [4] Thinking (softmax T=0.01, 10 cycles)...")
        energies = []
        for i, cents in enumerate(text_centroids):
            energy = softmax_think(table, cents)
            peaks = energy_to_peaks(energy, k=3)
            energies.append(energy)
            print(f"    Text {i+1}: peaks = {[(p[0], f'{p[1]:.4f}') for p in peaks]}")

        # Pairwise similarity from thinking
        print(f"\n  [5] Pairwise thinking similarity:")
        for i in range(len(energies)):
            for j in range(i+1, len(energies)):
                dot = np.dot(energies[i], energies[j])
                ni = np.linalg.norm(energies[i])
                nj = np.linalg.norm(energies[j])
                cos = dot / (ni * nj) if ni > 0 and nj > 0 else 0
                print(f"    {texts[i]['title'][:25]} ↔ {texts[j]['title'][:25]}: cos={cos:.4f}")

                # Contrastive update
                updates, mae = contrastive_update(
                    table, text_centroids[i], text_centroids[j], cos)
                total_updates += updates

                # NARS update
                for ca in text_centroids[i][:5]:
                    for cb in text_centroids[j][:5]:
                        key = (min(ca, cb), max(ca, cb))
                        if key not in nars:
                            nars[key] = NarsTruth()
                        nars[key].revise(cos)

        # NARS confidence check
        print(f"\n  [6] NARS confidence ({len(nars)} pairs tracked):")
        low_conf = [(k, v) for k, v in nars.items() if v.c < 0.5]
        high_conf = [(k, v) for k, v in nars.items() if v.c >= 0.5]
        print(f"    High confidence (c≥0.5): {len(high_conf)} pairs")
        print(f"    Low confidence (c<0.5):  {len(low_conf)} pairs")
        print(f"    Total table updates:     {total_updates}")

        # Generate next query from low-confidence pairs
        if low_conf and round_num < max_rounds - 1:
            # Find the most uncertain pair
            worst = min(low_conf, key=lambda x: x[1].c)
            query = f"{initial_query} {worst[0][0]} {worst[0][1]} relationship"
            print(f"\n  [7] Low confidence → new query: \"{query}\"")

        all_texts.extend(texts)

    # Final summary
    print(f"\n{'='*70}")
    print(f"  OSINT LOOP COMPLETE")
    print(f"{'='*70}")
    print(f"  Rounds:        {max_rounds}")
    print(f"  Texts crawled: {len(all_texts)}")
    print(f"  Table updates: {total_updates}")
    print(f"  NARS pairs:    {len(nars)}")
    high = sum(1 for v in nars.values() if v.c >= 0.5)
    print(f"  High conf:     {high}/{len(nars)} ({high/max(len(nars),1)*100:.0f}%)")

    return table, nars, all_texts

if __name__ == "__main__":
    table, nars, texts = osint_loop(
        "CRISPR gene editing therapeutic applications",
        max_rounds=3,
        results_per_round=3
    )
