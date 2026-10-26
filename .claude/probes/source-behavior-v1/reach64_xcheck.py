# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "mpmath"]
# ///
"""Cross-check of reach64's residue rule against a plain unbounded-coin DP on [0, 2^21], per calibration.
Usage: reach64_xcheck.py REACH64 TRACE63 OUT.txt   -> "XC cal LIM representable mismatches" lines."""
import subprocess, sys
REACH, TRACE, OUT = sys.argv[1:4]
src = open("softmax63.py").read(); g = {"__name__": "x"}; sys.argv = ["x", "x", "x", "/dev/null"]
exec(src[: src.index("# --- prepared per-calibration weights")], g)
lines = []
for ci, (s, z, b) in enumerate(g["CAL"]):
    t = subprocess.run([TRACE], input=f"softexp {g['f32bits'](s)} {g['f32bits'](b)}\n", capture_output=True, text=True, check=True).stdout.splitlines()[1:]
    lines.append(subprocess.run([REACH, str(1 << 21)], input=f"C {ci}\n" + "\n".join(t) + "\nQ 0\n", capture_output=True, text=True, check=True).stdout.strip())
open(OUT, "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
