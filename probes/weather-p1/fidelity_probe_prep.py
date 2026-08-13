"""D-WXS-7/D-WXS-8 prep stage: quantize the fetched grid-scale fields and
write raw (truth_distance, code_distance) pair arrays for the Rust compute
stage (`crates/weather-poc/examples/fidelity_probe.rs`) to score with
`jc::reliability::spearman` -- per bar B6 (`.claude/plans/weather-soa-bake-v1.md`
W3), the metric must be COMPUTED WITH jc, not with scipy; this stage only
prepares the arrays.

Quantization formula matches `crates/weather-poc/src/floor.rs::CalibratedFloor`
exactly (itself a documented re-expression of `helix::quantize::RollingFloor`):
N uniform buckets over a percentile-trimmed [lo, hi] window, decode to bucket
centre. Re-implemented here in Python for the SAME reason `floor.rs` is
zero-dep by construction (`.claude/plans/weather-soa-bake-v1.md` sec 2.3) --
this stage fetches real data over HTTP, which the Rust crate deliberately
cannot do, so the two languages meet at a data file, not a dependency edge
(the D-WXS-12 shape: parity is measured, not imported).

Standardization (z = (anom - mean) / std) matches probes/weather-p1/p2_probe.py
exactly, extended from 1 timestep/3 variables to 3 REAL seasons (see
fidelity_probe_fetch.py's module doc for how those seasons and their variable
sets were found by a live availability sweep, not assumed)."""
import json
import struct
import numpy as np

RNG_SEED = 20260813  # pre-registered, distinct from D-CZ's 20260812
LO_PCT, HI_PCT = 0.4, 99.6  # matches floor.rs LO_PERCENTILE/HI_PERCENTILE
N_PAIRS = 200_000  # matches p2_probe.py's N
LADDER = [16, 64, 256]  # bar B6(c)'s resolution ladder

SEASONS = {
    'winter': ['2m_temperature', '2m_dewpoint_temperature', '10m_u_component_of_wind',
               'mean_sea_level_pressure', 'total_cloud_cover'],
    'spring': ['2m_temperature', '2m_dewpoint_temperature', '10m_u_component_of_wind',
               'total_column_water_vapour'],
    'summer': ['2m_temperature', '10m_u_component_of_wind', '10m_v_component_of_wind',
               'total_cloud_cover'],
}
UNITS = {
    '2m_temperature': 'K', '2m_dewpoint_temperature': 'K', 'sea_surface_temperature': 'K',
    '10m_u_component_of_wind': 'm/s', '10m_v_component_of_wind': 'm/s',
    'mean_sea_level_pressure': 'Pa', 'total_column_water_vapour': 'kg/m2',
    'total_cloud_cover': '1',
}


def load_z(season, var):
    """Load a fetched field, return its z-standardized flat array (anomaly /
    std, matching p2_probe.py's transform -- identity-after-scaling, §12.1
    measured arctanh as actively harmful on this bell-shaped anomaly shape)."""
    a = np.load(f'fixture/season_{season}/{var}.npy').astype(np.float64).ravel()
    assert np.isfinite(a).all(), f"{season}/{var}: caller must pre-filter to usable fields"
    z = (a - a.mean()) / a.std()
    return z


def quantize(z, lo, hi, n_levels, decode_perm=None):
    """CalibratedFloor::quantize/bucket_center, re-expressed. `decode_perm`
    is a permutation applied to the CENTRES table (not the index assignment)
    -- the "shuffled decode table" bar B6(b) names: same bucket a value lands
    in, but a scrambled mapping from bucket -> represented value. Returns
    (index_array_u16, decoded_value_array_f64)."""
    lo, hi = float(lo), float(hi)
    if hi <= lo:
        hi = lo + 1e-9
    width = (hi - lo) / n_levels
    idx = np.clip(((z - lo) / width).astype(np.int64), 0, n_levels - 1)
    centres = lo + (np.arange(n_levels) + 0.5) * width
    if decode_perm is not None:
        centres = centres[decode_perm]
    return idx, centres[idx]


def percentile_window(z):
    lo, hi = np.percentile(z, [LO_PCT, HI_PCT])
    return float(lo), float(hi)


def write_f64(path, arr):
    """Flat little-endian f64, no header -- the Rust reader is a std-only
    `f64::from_le_bytes` loop, matching the crate's zero-dep philosophy."""
    arr.astype('<f8').tofile(path)


def main():
    rng = np.random.default_rng(RNG_SEED)
    out_dir = 'fixture/fidelity_pairs'
    import os
    os.makedirs(out_dir, exist_ok=True)
    manifest = {'seed': RNG_SEED, 'n_pairs': N_PAIRS, 'ladder': LADDER,
                'lo_percentile': LO_PCT, 'hi_percentile': HI_PCT, 'seasons': {}}

    for season, varlist in SEASONS.items():
        z = {v: load_z(season, v) for v in varlist}
        units_here = sorted(set(UNITS[v] for v in varlist))
        print(f"\n== {season}: {len(varlist)} vars, units={units_here} ==")
        assert len(varlist) >= 4 and len(units_here) >= 2, \
            f"{season} does not meet bar B7's floor -- fetch stage should have caught this"

        n_cells = len(next(iter(z.values())))
        pool = np.concatenate([z[v] for v in varlist])
        shared_lo, shared_hi = percentile_window(pool)
        per_var_window = {v: percentile_window(z[v]) for v in varlist}

        season_manifest = {'n_cells': n_cells, 'units': {v: UNITS[v] for v in varlist},
                            'shared_window': [shared_lo, shared_hi], 'pairs': {}}

        # ---- within-variable sample indices (shared across resolutions) ----
        ia_w = rng.integers(0, n_cells, N_PAIRS)
        ib_w = rng.integers(0, n_cells, N_PAIRS)

        # ---- the K x K pair (bar B6): the SAME two variables, both unit K,
        # that the plan's own committed measurement (0.999556, BELOW the
        # 0.9996 bar) came from -- re-measured here at real grid scale,
        # across 3 real seasons, rather than assumed to still hold. Only
        # winter and spring carry 2 distinct K variables; summer's K set is
        # {2m_temperature} alone, so summer's "K pair" is the WITHIN-variable
        # self-pair instead (a legitimate degenerate case, reported as such).
        k_vars = [v for v in varlist if UNITS[v] == 'K']
        if len(k_vars) >= 2:
            va, vb = k_vars[0], k_vars[1]
            pair_name = f"{va} x {vb} (K x K)"
        else:
            va = vb = k_vars[0] if k_vars else varlist[0]
            pair_name = f"{va} (within, only-K-available)"

        ia = rng.integers(0, n_cells, N_PAIRS)
        ib = rng.integers(0, n_cells, N_PAIRS)
        truth = np.abs(z[va][ia] - z[vb][ib])
        write_f64(f"{out_dir}/{season}_kxk_truth.f64", truth)
        for n_levels in LADDER:
            idx_a, _ = quantize(z[va], shared_lo, shared_hi, n_levels)
            idx_b, _ = quantize(z[vb], shared_lo, shared_hi, n_levels)
            code = np.abs(idx_a[ia].astype(np.float64) - idx_b[ib].astype(np.float64))
            write_f64(f"{out_dir}/{season}_kxk_code_L{n_levels}.f64", code)
        # shuffle control at the top resolution (256): permute the DECODE
        # table (not the index assignment), decode both sides through it,
        # take |decoded_a - decoded_b| as the "code distance" a reader with
        # a scrambled codebook would compute.
        perm = rng.permutation(256)
        _, dec_a = quantize(z[va], shared_lo, shared_hi, 256, decode_perm=perm)
        _, dec_b = quantize(z[vb], shared_lo, shared_hi, 256, decode_perm=perm)
        shuffled_code = np.abs(dec_a[ia] - dec_b[ib])
        write_f64(f"{out_dir}/{season}_kxk_code_shuffled.f64", shuffled_code)
        season_manifest['pairs']['kxk'] = {'name': pair_name, 'va': va, 'vb': vb}

        # ---- bar B7: every cross-variable pair, shared vs per-variable floor,
        # at resolution 256 (the shipped resolution) ----
        cross_pairs = {}
        for i, va2 in enumerate(varlist):
            for vb2 in varlist[i + 1:]:
                key = f"{va2}_x_{vb2}"
                ia2 = rng.integers(0, n_cells, N_PAIRS)
                ib2 = rng.integers(0, n_cells, N_PAIRS)
                truth2 = np.abs(z[va2][ia2] - z[vb2][ib2])
                write_f64(f"{out_dir}/{season}_{key}_truth.f64", truth2)

                idx_a_sh, _ = quantize(z[va2], shared_lo, shared_hi, 256)
                idx_b_sh, _ = quantize(z[vb2], shared_lo, shared_hi, 256)
                code_sh = np.abs(idx_a_sh[ia2].astype(np.float64) - idx_b_sh[ib2].astype(np.float64))
                write_f64(f"{out_dir}/{season}_{key}_code_shared.f64", code_sh)

                lo_a, hi_a = per_var_window[va2]
                lo_b, hi_b = per_var_window[vb2]
                idx_a_pv, _ = quantize(z[va2], lo_a, hi_a, 256)
                idx_b_pv, _ = quantize(z[vb2], lo_b, hi_b, 256)
                code_pv = np.abs(idx_a_pv[ia2].astype(np.float64) - idx_b_pv[ib2].astype(np.float64))
                write_f64(f"{out_dir}/{season}_{key}_code_pervar.f64", code_pv)

                cross_pairs[key] = {'va': va2, 'vb': vb2, 'unit_va': UNITS[va2], 'unit_vb': UNITS[vb2],
                                     'same_unit': UNITS[va2] == UNITS[vb2]}
                print(f"  cross pair {va2:28s} x {vb2:28s}  units=({UNITS[va2]},{UNITS[vb2]})")
        season_manifest['pairs']['cross'] = cross_pairs

        # ---- stay-silent twin: within-variable, shared floor must not cost
        # resolution vs per-variable floor. One representative variable
        # (the first in varlist) at resolution 256. ----
        v0 = varlist[0]
        truth_w = np.abs(z[v0][ia_w] - z[v0][ib_w])
        write_f64(f"{out_dir}/{season}_within_truth.f64", truth_w)
        idx_sh, _ = quantize(z[v0], shared_lo, shared_hi, 256)
        code_w_sh = np.abs(idx_sh[ia_w].astype(np.float64) - idx_sh[ib_w].astype(np.float64))
        write_f64(f"{out_dir}/{season}_within_code_shared.f64", code_w_sh)
        lo0, hi0 = per_var_window[v0]
        idx_pv0, _ = quantize(z[v0], lo0, hi0, 256)
        code_w_pv = np.abs(idx_pv0[ia_w].astype(np.float64) - idx_pv0[ib_w].astype(np.float64))
        write_f64(f"{out_dir}/{season}_within_code_pervar.f64", code_w_pv)
        # empty-bucket count under the shared floor, for this variable
        occ = np.bincount(idx_sh, minlength=256)
        season_manifest['within_variable_control'] = {
            'var': v0, 'shared_empty_buckets': int((occ == 0).sum()),
        }

        manifest['seasons'][season] = season_manifest

        # Plain key:value sidecar for the Rust reader -- deliberately NOT
        # JSON, so the Rust example needs no parsing crate (matches the
        # crate's zero-dep-by-construction philosophy; only `jc` is added,
        # as a dev-dependency, for the one thing Rust must do: compute
        # Spearman WITH jc, per bar B6's own wording).
        with open(f'{out_dir}/{season}_meta.txt', 'w') as f:
            f.write(f"within_var:{v0}\n")
            f.write(f"within_empty_buckets:{season_manifest['within_variable_control']['shared_empty_buckets']}\n")
            f.write(f"kxk_name:{pair_name}\n")
            f.write(f"kxk_va:{va}\n")
            f.write(f"kxk_vb:{vb}\n")
            for key, info in cross_pairs.items():
                f.write(f"cross:{key}:{info['unit_va']}:{info['unit_vb']}\n")

    json.dump(manifest, open(f'{out_dir}/manifest.json', 'w'), indent=2)
    print(f"\nwrote {out_dir}/manifest.json + raw f64 pair arrays + per-season meta.txt")


if __name__ == '__main__':
    main()
