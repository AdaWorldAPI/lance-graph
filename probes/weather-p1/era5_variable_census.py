"""D-WXS-1a — the ERA5 WeatherBench2 Zarr variable census, as a committed,
re-runnable probe.

`weather-soa-bake-v1.md` §0.6 asserts a variable census in prose (17 surface
+ 7 pressure-level x 13 levels = 91 + 14 static = 122 fields/cell, 92,044
timesteps) with no artifact behind it. That is exactly the chat-only-figure
defect this arc has already caught three times (see the plan's §0 C1/⊘C3
note: "a title match is not an existence check; list the bucket"). This
probe reads the store's own `.zmetadata` and emits the census as JSON, so
the numbers terminate at bytes fetched from the store rather than at a
sentence someone typed.

Classification is by array shape rank, excluding the four coordinate
arrays (`time`, `level`, `latitude`, `longitude`):

  rank 3  [T, 721, 1440]        -> time-varying SURFACE field
  rank 4  [T, n_levels, 721, 1440] -> time-varying PRESSURE-LEVEL field
                                       (contributes n_levels fields, one
                                       per level)
  rank 2  [721, 1440]           -> STATIC field (no time axis)

Anything that does not match one of those three shapes is reported under
`unclassified` rather than silently dropped or silently forced into a
bucket -- an unanticipated array shape is itself a finding.

Run with `--selftest` to compare the fetched census against the constants
this plan already committed. A mismatch there means one of two things:
the store changed shape since the plan was written, or the plan's prose
was wrong. Either way the probe's own fetched answer wins -- that is the
point of terminating at the artifact instead of the prose.
"""

import json
import pathlib
import sys
import urllib.error
import urllib.request

STORE = ("https://storage.googleapis.com/weatherbench2/datasets/era5/"
         "1959-2022-6h-1440x721.zarr")
METADATA_URL = STORE + "/.zmetadata"

# The four coordinate arrays. Every other array in the store is a data
# variable (or an unanticipated shape, reported separately).
COORD_ARRAYS = {"time", "level", "latitude", "longitude"}

HERE = pathlib.Path(__file__).parent
OUT = HERE / "era5_variable_census.json"

# ---------------------------------------------------------------- fetch

def _fetch_bytes(url, timeout=120):
    """Fetch `url`, trying the default (proxied) opener first and falling
    back to a proxy-bypassed opener on failure.

    This environment's outbound HTTPS goes through a pre-configured
    agent proxy; a 403 or TLS error against a public GCS bucket is
    usually the proxy, not the resource (see CLAUDE.md's own "GitHub
    access matrix" finding, which generalizes past GitHub: verify with
    the proxy bypassed before concluding a store is unreachable).
    """
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read()
    except Exception:  # noqa: BLE001 -- broad on purpose, see docstring
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(url, timeout=timeout) as resp:
            return resp.read()


def fetch_metadata():
    """Fetch and parse the store's consolidated `.zmetadata`.

    Returns the top-level `"metadata"` dict: keys are `"<array>/.zarray"`
    (plus `.zgroup` / `.zattrs` entries, which this probe ignores), values
    carry `"shape"` and `"dtype"` among other zarr array-metadata fields.
    """
    raw = _fetch_bytes(METADATA_URL)
    doc = json.loads(raw)
    return doc["metadata"]


# ------------------------------------------------------------ classify

def array_entries(metadata):
    """Yield (array_name, zarray_dict) for every `.zarray` entry in the
    consolidated metadata, in the order the JSON gave them (not sorted --
    sorting happens once, at emission, on the classified name lists)."""
    for key, val in metadata.items():
        if key.endswith("/.zarray"):
            name = key[: -len("/.zarray")]
            yield name, val


def classify(metadata):
    """Classify every non-coordinate array by shape rank.

    Returns a dict with `surface`, `pressure_level` (name -> shape),
    `static`, and `unclassified` (name -> shape, for anything that
    doesn't match one of the three canonical shapes) plus the resolved
    grid dims, timestep count, and level count pulled from the
    coordinate arrays themselves (never hardcoded).
    """
    coord_shapes = {}
    for name, za in array_entries(metadata):
        if name in COORD_ARRAYS:
            coord_shapes[name] = za["shape"]

    missing = COORD_ARRAYS - set(coord_shapes)
    if missing:
        raise RuntimeError(
            f"store is missing expected coordinate array(s): {sorted(missing)} "
            "-- cannot classify without a time/level/lat/lon reference")

    n_time = coord_shapes["time"][0]
    n_levels = coord_shapes["level"][0]
    n_lat = coord_shapes["latitude"][0]
    n_lon = coord_shapes["longitude"][0]
    grid_shape = [n_lat, n_lon]

    surface, pressure_level, static, unclassified = {}, {}, {}, {}

    for name, za in array_entries(metadata):
        if name in COORD_ARRAYS:
            continue
        shape = za["shape"]
        rank = len(shape)
        if rank == 2 and shape == grid_shape:
            static[name] = shape
        elif rank == 3 and shape[0] == n_time and shape[1:] == grid_shape:
            surface[name] = shape
        elif (rank == 4 and shape[0] == n_time and shape[1] == n_levels
              and shape[2:] == grid_shape):
            pressure_level[name] = shape
        else:
            unclassified[name] = shape

    return {
        "surface": surface,
        "pressure_level": pressure_level,
        "static": static,
        "unclassified": unclassified,
        "n_time": n_time,
        "n_levels": n_levels,
        "n_lat": n_lat,
        "n_lon": n_lon,
    }


# --------------------------------------------------------------- report

def build_report(metadata):
    """Assemble the full committed-JSON census from a classified store."""
    c = classify(metadata)

    surface_names = sorted(c["surface"])
    pressure_names = sorted(c["pressure_level"])
    static_names = sorted(c["static"])
    unclassified_names = sorted(c["unclassified"])
    coord_names = sorted(COORD_ARRAYS)

    n_surface = len(surface_names)
    n_pressure_vars = len(pressure_names)
    n_levels = c["n_levels"]
    n_pressure_fields = n_pressure_vars * n_levels
    n_static = len(static_names)
    total_fields_per_cell = n_surface + n_pressure_fields + n_static

    n_lat, n_lon = c["n_lat"], c["n_lon"]
    n_cells = n_lat * n_lon
    n_time = c["n_time"]

    values_per_timestep = n_cells * total_fields_per_cell
    bytes_per_timestep_512b_rows = n_cells * 512

    return {
        "probe": "era5_variable_census",
        "store_url": STORE,
        "metadata_url": METADATA_URL,
        "coordinate_arrays_excluded": coord_names,
        "surface_variables": surface_names,
        "pressure_level_variables": pressure_names,
        "static_variables": static_names,
        "unclassified_arrays": {n: c["unclassified"][n] for n in unclassified_names},
        "counts": {
            "surface": n_surface,
            "pressure_level_variables": n_pressure_vars,
            "levels": n_levels,
            "pressure_level_fields": n_pressure_fields,
            "static": n_static,
            "unclassified": len(unclassified_names),
        },
        "total_fields_per_cell": total_fields_per_cell,
        "grid": {"lat": n_lat, "lon": n_lon, "n_cells": n_cells},
        "n_timesteps": n_time,
        "derived": {
            "values_per_timestep": values_per_timestep,
            "bytes_per_timestep_512B_rows": bytes_per_timestep_512b_rows,
        },
    }


# --------------------------------------------------------------- selftest

# The constants `weather-soa-bake-v1.md` §0.6 already committed in prose.
# A mismatch means either the store changed since the plan was written, or
# the plan's prose figure was wrong -- either way this probe's fetched
# answer is the one that wins, because it terminates at the artifact.
EXPECTED = {
    "surface": 17,
    "pressure_level_variables": 7,
    "levels": 13,
    "pressure_level_fields": 91,
    "static": 14,
    "total_fields_per_cell": 122,
    "n_cells": 1_038_240,
    "n_timesteps": 92_044,
    "values_per_timestep": 126_665_280,
    "bytes_per_timestep_512B_rows": 531_578_880,
}


def selftest(report):
    """Compare the fetched report against EXPECTED. Fails loudly (exit 1)
    and prints every mismatch on disagreement -- never adjusts EXPECTED
    to make a run pass."""
    got = {
        "surface": report["counts"]["surface"],
        "pressure_level_variables": report["counts"]["pressure_level_variables"],
        "levels": report["counts"]["levels"],
        "pressure_level_fields": report["counts"]["pressure_level_fields"],
        "static": report["counts"]["static"],
        "total_fields_per_cell": report["total_fields_per_cell"],
        "n_cells": report["grid"]["n_cells"],
        "n_timesteps": report["n_timesteps"],
        "values_per_timestep": report["derived"]["values_per_timestep"],
        "bytes_per_timestep_512B_rows":
            report["derived"]["bytes_per_timestep_512B_rows"],
    }
    mismatches = {k: (EXPECTED[k], got[k]) for k in EXPECTED if EXPECTED[k] != got[k]}
    return mismatches, got


# -------------------------------------------------------------------- main

def main():
    run_selftest = "--selftest" in sys.argv

    metadata = fetch_metadata()
    report = build_report(metadata)

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"store: {STORE}")
    print(f"metadata: {METADATA_URL}")
    print(f"surface variables:        {report['counts']['surface']}")
    print(f"pressure-level variables: {report['counts']['pressure_level_variables']}"
          f"  x {report['counts']['levels']} levels"
          f" = {report['counts']['pressure_level_fields']} fields")
    print(f"static variables:         {report['counts']['static']}")
    print(f"unclassified arrays:      {report['counts']['unclassified']}"
          + (f"  {sorted(report['unclassified_arrays'])}"
             if report['unclassified_arrays'] else ""))
    print(f"total fields/cell:        {report['total_fields_per_cell']}")
    print(f"grid: {report['grid']['lat']} x {report['grid']['lon']}"
          f" = {report['grid']['n_cells']} cells")
    print(f"timesteps: {report['n_timesteps']}")
    print(f"values/timestep: {report['derived']['values_per_timestep']}")
    print(f"bytes/timestep @512B rows: "
          f"{report['derived']['bytes_per_timestep_512B_rows']}")
    print(f"\nwrote {OUT}")

    if run_selftest:
        mismatches, got = selftest(report)
        if mismatches:
            print("\nSELFTEST: FAIL -- fetched census disagrees with the plan's "
                  "committed constants (weather-soa-bake-v1.md §0.6):")
            for k, (exp, act) in sorted(mismatches.items()):
                print(f"  {k:32s} expected={exp!r:>14}  got={act!r:>14}")
            print("\nThis means either the store changed shape since the plan "
                  "was written, or the plan's prose figure was wrong. The "
                  "fetched answer above is authoritative; do not edit EXPECTED "
                  "to silence this without updating the plan too.")
            sys.exit(1)
        else:
            print(f"\nSELFTEST: PASS -- all {len(EXPECTED)} committed constants "
                  "match the fetched census.")


if __name__ == "__main__":
    main()
