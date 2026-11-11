#!/usr/bin/env bash
# Install protoc and the protobuf includes that lance's build needs.
#
# No apt: the hosted runners cannot reach their apt mirror ("Temporary failure
# resolving 'azure.archive.ubuntu.com'", every job, 2026-11-11). protoc comes
# from the protobuf GitHub release instead, checked against a pinned digest.
# 21.12 is the release apt shipped as 3.21.12.
set -euo pipefail

version=21.12
sha256=3a4c1e5f2516c639d3079b1586e703fc7bcfa2136d58bda24d1d54f949c315e8

if command -v protoc >/dev/null 2>&1; then
  echo "protoc already present: $(protoc --version)"
  exit 0
fi
[ "$(uname -m)" = x86_64 ] || { echo "install-protoc: no build pinned for $(uname -m)" >&2; exit 1; }

zip=$(mktemp --suffix=.zip)
curl -fsSL --retry 5 --retry-all-errors -o "$zip" \
  "https://github.com/protocolbuffers/protobuf/releases/download/v${version}/protoc-${version}-linux-x86_64.zip"
echo "${sha256}  ${zip}" | sha256sum -c -
sudo unzip -oq "$zip" 'bin/protoc' 'include/*' -d /usr/local
sudo chmod 755 /usr/local/bin/protoc
sudo chmod -R a+rX /usr/local/include/google
echo "protoc: $(protoc --version)"
