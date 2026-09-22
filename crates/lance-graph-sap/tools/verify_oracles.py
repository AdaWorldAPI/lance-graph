#!/usr/bin/env python3
"""Execute pinned C# implementations and compare all fields/hash with Rust.

python3 tools/verify_oracles.py EVIDENCE_ROOT [dotnet] [cargo]
An ABAP runtime is NOT emulated or claimed by this harness.
"""
from pathlib import Path
import subprocess
import sys
import hashlib
import hmac

here = Path(__file__).resolve().parents[1]
root = Path(sys.argv[1]).resolve()
dotnet = sys.argv[2] if len(sys.argv) > 2 else 'dotnet'
cargo = sys.argv[3] if len(sys.argv) > 3 else 'cargo'
subprocess.run([sys.executable, str(here/'tools/verify_sources.py'), str(root)], check=True)
subprocess.run([dotnet, 'build', str(here/'tools/csharp/Oracle.csproj'), '-o', str(here/'target/csharp-oracle'), '-p:EvidenceRoot='+str(root), '--nologo'], check=True)
cs = subprocess.check_output([dotnet, str(here/'target/csharp-oracle/Oracle.dll'), str(here/'fixtures/cats.txt'), str(here/'schema.tsv')], text=True)
rs = subprocess.check_output([cargo, '+stable', 'run', '--quiet', '--manifest-path', str(here/'Cargo.toml'), '--example', 'oracle'], text=True)
assert cs.splitlines() == rs.splitlines(), (cs, rs)
lines = rs.splitlines()
assert len(lines) == 25
assert hmac.new(b'fixture-key', lines[-2].encode(), hashlib.sha512).hexdigest() == lines[-1]
print('PASS: 23 fields + ordered projection + HMAC: original SIMAFPort/SMB C# == Rust; Python HMAC agrees.')
