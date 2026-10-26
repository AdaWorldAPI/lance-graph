#!/usr/bin/env python3
"""D-DQM P1: does a chunk-touched predictor explain the Lance 13 masked-read bytes
already measured in D-HOLO H14c and H14d?

Pre-registration: `.claude/plans/duckdb-dr-quack-machine-room-reuse-v1.md` § A.6 (P1).
Offline only. Reads the committed board entries, regenerates each cell's mask from the
probes' own seeded generators, and compares. No Lance run, no runtime code.

Inputs (read, never written):
  H14c  .claude/board/entries/2026-10-15-d-holo-h14c-thought-first.md   (Lance CSV block)
        generator: crates/holo-lance-masked-io-probe/src/main.rs  (`mix`, `present`, main loop)
  H14d  .claude/board/entries/2026-10-16-d-holo-h14d-logic-mask-read.md (rows `2-4`)
        generator: crates/holo-h14d-probe/src/main.rs  (`mix`, `draw`, `clustered_threshold`,
        `present`, `in_aperture`, `expr(Form::And, k)`)

Decisions fixed before any number was computed (none may be changed to rescue a result):
  * cell unit: H14c one per (pattern, density, k) for strategy `combined` vs `unmasked`,
    after asserting `bytes_read` is identical across the three materialization styles
    (the entry reports 0 of 120 cells differ; asserted here, not assumed). `independent`
    is excluded: it reads `coord` once per lane, a different projection from the
    baseline, so its ratio is not masked/unmasked of one read.
    H14d one per (pattern, density, width, k), form `and` (the only form read from Lance).
  * all cells count, including zero-survivor cells; a sensitivity line without them is
    printed beside the primary result, never instead of it.
  * baseline predictor: nonzero 64-word groups / total groups (4,096 rows per group,
    16 groups in a 65,536-row tile).
  * kill (A.6): |predicted - measured| > 0.10 in more than 10 % of cells, or Spearman
    rank correlation < 0.9.
  * one follow-up only if the baseline fails: the pre-registered 64 KiB range-merge
    hypothesis, per column, chunk bytes = 4,096 x the column's plain value width,
    touched groups separated by at most floor(65536 / chunk_bytes) untouched groups are
    read too (lance-io `is_close_together`, scheduler.rs:1172), columns weighted by plain
    width. Reported separately; never relabelled as the baseline.

Coordinate witness (checked from the probe source, not inferred from the CSV): both probes
write coordinates 0..65535 in order as ONE batch into ONE fragment (H14c
`max_rows_per_file = 1 << 20`, H14d `max_rows_per_file = N`), without stable row ids
(H14c `enable_stable_row_ids: false`; H14d default). So `_rowid` = row address = coordinate,
and slot c of a regenerated mask is row offset c. Each regenerated mask is also checked
against the probe's own recorded row count (H14c `rows_emitted`, H14d `survivors`).

Run: python3 -I .claude/probes/p1-chunk-touched-v1/p1.py [repo_root]
"""
import csv
import math
import os
import sys

M64 = (1 << 64) - 1
N = 65_536
W = N // 64
GROUP = 4096
GROUPS = N // GROUP
LANES = 8


def mix(z):
    z = (z + 0x9E3779B97F4A7C15) & M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


# ---------------------------------------------------------------- H14c generator

def h14c_lane(pattern, d, l):
    """Mask of coordinates where lane l is present, as W little-endian u64 words."""
    if d >= 1.0:
        return [M64] * W
    t = min(int(d * float(M64)), M64)  # Rust `(d * u64::MAX as f64) as u64` (saturating)
    words = [0] * W
    if pattern == "uniform":
        base = 0x1A9E ^ (l << 40)
        for c in range(N):
            if mix(base ^ c) < t:
                words[c >> 6] |= 1 << (c & 63)
    else:
        base = 0xC1 ^ (l << 40)
        for w in range(W):
            if mix(base ^ w) < t:
                words[w] = M64
    return words


# ---------------------------------------------------------------- H14d generator

def h14d_draw(l, key):
    return mix(((l << 40) ^ key ^ 0x5151) & M64)


def rust_round(x):
    return math.floor(x + 0.5) if x >= 0 else math.ceil(x - 0.5)


def h14d_lane(pattern, d, l):
    words = [0] * W
    if pattern == "uniform":
        two53 = float(1 << 53)
        for i in range(N):
            if (h14d_draw(l, i) >> 11) / two53 < d:
                words[i >> 6] |= 1 << (i & 63)
        return words
    m = int(rust_round(d * W))
    if m == 0:
        return words
    h = sorted(h14d_draw(l, w) for w in range(W))
    t = h[m - 1]
    for w in range(W):
        if h14d_draw(l, w) <= t:
            words[w] = M64
    return words


# ---------------------------------------------------------------- predictors

def touched_groups(words):
    """One flag per 64-word group (4,096 rows): does the group hold any selected row."""
    per = GROUP // 64
    return [any(words[g * per:(g + 1) * per]) for g in range(GROUPS)]


def merged(touched, max_gap):
    """Touched groups plus every untouched gap of <= max_gap groups between two touched ones."""
    out = list(touched)
    idx = [g for g, t in enumerate(touched) if t]
    for a, b in zip(idx, idx[1:]):
        if b - a - 1 <= max_gap:
            for g in range(a + 1, b):
                out[g] = True
    return out


def merge_prediction(touched, widths):
    total = sum(widths)
    s = 0.0
    for wbytes in widths:
        chunk = GROUP * wbytes
        gap = 65536 // chunk
        s += wbytes * sum(merged(touched, gap)) / GROUPS
    return s / total


def popcount(words):
    return sum(bin(w).count("1") for w in words)


def shifted(words, k):
    """Mask moved k rows towards higher addresses; k is a multiple of 64 here."""
    assert k % 64 == 0
    s = k // 64
    return [0] * s + words[: W - s]


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    vy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return cov / (vx * vy)


# ---------------------------------------------------------------- inputs

def rows(path):
    with open(path) as f:
        return [line.rstrip("\n").split(",") for line in f if line.startswith("csv,")]


def h14c_cells(root):
    path = os.path.join(root, ".claude/board/entries/2026-10-15-d-holo-h14c-thought-first.md")
    by = {}
    for r in rows(path):
        if len(r) != 13 or r[1] not in ("unmasked", "combined", "independent"):
            continue
        _, strat, pat, dens, k, style, emitted, nbytes = r[:8]
        by.setdefault((pat, dens, int(k)), {}).setdefault(strat, []).append(
            (style, int(emitted), int(nbytes)))
    cells = []
    for (pat, dens, k), s in sorted(by.items()):
        for strat in ("unmasked", "combined"):
            assert len(s[strat]) == 3, (pat, dens, k, strat)
            assert len({b for _, _, b in s[strat]}) == 1, ("style changed bytes", pat, dens, k, strat)
        cells.append(dict(fixture="H14c", pattern=pat, density=float(dens), width=N, k=k,
                          rows=s["combined"][0][1], masked=s["combined"][0][2],
                          unmasked=s["unmasked"][0][2]))
    return cells


def h14d_cells(root):
    path = os.path.join(root, ".claude/board/entries/2026-10-16-d-holo-h14d-logic-mask-read.md")
    by = {}
    for r in rows(path):
        if len(r) != 10 or r[1] != "2-4" or r[5] != "and":
            continue
        _, _, pat, dens, width, _, k, _, metric, val = r
        by.setdefault((pat, dens, int(width), int(k)), {})[metric] = val
    cells = []
    for (pat, dens, width, k), m in sorted(by.items()):
        cells.append(dict(fixture="H14d", pattern=pat, density=float(dens), width=width, k=k,
                          rows=int(m["survivors"]), masked=int(m["lance_bytes_masked"]),
                          unmasked=int(m["lance_bytes_unmasked"]),
                          returned=int(m["lance_rows_returned"])))
    return cells


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    cells = h14c_cells(root) + h14d_cells(root)

    lane_cache = {}

    def lane(fix, pat, d, l):
        key = (fix, pat, d, l)
        if key not in lane_cache:
            lane_cache[key] = (h14c_lane if fix == "H14c" else h14d_lane)(pat, d, l)
        return lane_cache[key]

    for c in cells:
        m = [M64] * W
        for l in range(c["k"]):
            m = [a & b for a, b in zip(m, lane(c["fixture"], c["pattern"], c["density"], l))]
        if c["fixture"] == "H14d":
            # aperture: i < width (widths 1024, 8192, 65536 are word multiples)
            m = [x if w * 64 < c["width"] else 0 for w, x in enumerate(m)]
        sel = popcount(m)
        c["mask_ok"] = sel == c["rows"] and c.get("returned", sel) == sel
        c["selected"] = sel
        t = touched_groups(m)
        c["touched"] = sum(t)
        c["pred"] = sum(t) / GROUPS
        c["meas"] = c["masked"] / c["unmasked"]
        c["err"] = abs(c["pred"] - c["meas"])
        widths = [4] + ([16] * c["k"] if c["fixture"] == "H14c" else [4] * c["k"])
        c["pred_merge"] = merge_prediction(t, widths)
        c["err_merge"] = abs(c["pred_merge"] - c["meas"])
        c["pred_shift2048"] = sum(touched_groups(shifted(m, 2048))) / GROUPS

    bad = [c for c in cells if not c["mask_ok"]]
    with open(os.path.join(out_dir, "results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["fixture", "pattern", "density", "width", "k", "selected", "rows_recorded",
                    "mask_reconstructed", "touched_groups", "pred_ratio", "measured_ratio",
                    "abs_err", "masked_bytes", "unmasked_bytes", "pred_shift2048",
                    "pred_merge64k", "abs_err_merge64k"])
        for c in cells:
            w.writerow([c["fixture"], c["pattern"], c["density"], c["width"], c["k"], c["selected"],
                        c["rows"], c["mask_ok"], c["touched"], f'{c["pred"]:.4f}',
                        f'{c["meas"]:.4f}', f'{c["err"]:.4f}', c["masked"], c["unmasked"],
                        f'{c["pred_shift2048"]:.4f}', f'{c["pred_merge"]:.4f}',
                        f'{c["err_merge"]:.4f}'])

    def verdict(label, cs, key_pred, key_err):
        n = len(cs)
        over = sum(1 for c in cs if c[key_err] > 0.10)
        rho = spearman([c[key_pred] for c in cs], [c["meas"] for c in cs])
        fired = []
        if over / n > 0.10:
            fired.append(f"error>0.10 in {over}/{n} = {100*over/n:.1f}% of cells (> 10%)")
        if rho < 0.9:
            fired.append(f"Spearman {rho:.3f} < 0.9")
        print(f"{label}: cells={n} err>0.10={over} ({100*over/n:.1f}%) spearman={rho:.3f} "
              f"-> {'FAIL: ' + '; '.join(fired) if fired else 'PASS'}")
        return not fired

    print(f"cells: {len(cells)} (H14c {sum(c['fixture']=='H14c' for c in cells)}, "
          f"H14d {sum(c['fixture']=='H14d' for c in cells)}); mask reconstruction mismatches: {len(bad)}")
    if bad:
        print("INCONCLUSIVE: regenerated masks disagree with recorded row counts:")
        for c in bad:
            print("  ", c["fixture"], c["pattern"], c["density"], c["width"], c["k"], c["selected"], c["rows"])
        return
    thresholds_ok = verdict("baseline thresholds (all cells, the registered population)", cells, "pred", "err")
    nz = [c for c in cells if c["selected"] > 0]
    verdict("baseline thresholds (sensitivity: cells with a selection)", nz, "pred", "err")
    for fx in ("H14c", "H14d"):
        verdict(f"baseline thresholds ({fx} only)", [c for c in cells if c["fixture"] == fx], "pred", "err")
    over = sum(1 for c in nz if c["pred"] > c["meas"])
    print(f"  direction on cells with a selection: over-predicted {over} of {len(nz)}")

    # Can-fire controls (A.6): uniform 1 % ~ 1.00 and clustered 1 % ~ 0.2, k = 1, full
    # tile. Each must sit within the kill tolerance (0.10) of its measured ratio.
    print("controls (1 %, k = 1, full tile):")
    controls_ok = True
    for c in cells:
        if c["density"] == 0.01 and c["k"] == 1 and c["width"] == N:
            hit = c["err"] <= 0.10
            controls_ok &= hit
            print(f"  {c['fixture']} {c['pattern']}: selected={c['selected']} touched={c['touched']}/16 "
                  f"pred={c['pred']:.3f} measured={c['meas']:.3f} -> {'ok' if hit else 'MISS'}")
    # Can-stay-silent control (A.6): +2048 rows must raise the clustered prediction.
    # Its premise (every run crosses a chunk boundary) holds only for runs as long as a
    # chunk; these runs are 64 rows at 64-row alignment, so it is reported, not scored.
    clu = [c for c in cells if c["pattern"] == "clustered" and c["selected"] > 0]
    raised = sum(1 for c in clu if c["pred_shift2048"] > c["pred"])
    same = sum(1 for c in clu if c["pred_shift2048"] == c["pred"])
    print(f"  shift +2048 (not scored, premise false for 64-row runs): raised {raised}, "
          f"equal {same}, lowered {len(clu) - raised - same} of {len(clu)} clustered cells")

    baseline_ok = thresholds_ok and controls_ok
    print(f"BASELINE: {'PASS' if baseline_ok else 'FAIL'} "
          f"(thresholds {'pass' if thresholds_ok else 'fire'}, controls {'pass' if controls_ok else 'miss'})")
    if not baseline_ok:
        print("one registered follow-up, reported separately:")
        verdict("  64 KiB merge hypothesis (all cells)", cells, "pred_merge", "err_merge")
        verdict("  64 KiB merge hypothesis (cells with a selection)", nz, "pred_merge", "err_merge")
        for c in cells:
            if c["density"] == 0.01 and c["k"] == 1 and c["width"] == N:
                print(f"    control {c['fixture']} {c['pattern']}: pred_merge={c['pred_merge']:.3f} "
                      f"measured={c['meas']:.3f}")

if __name__ == "__main__":
    main()
