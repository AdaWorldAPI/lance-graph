#!/usr/bin/env bash
# Install protoc and the protobuf includes that lance's build needs.
#
# apt first: it is the runner's cached mirror and the usual path. When apt
# cannot reach its mirror (observed 2026-11-11: every job failed for over an
# hour on "Temporary failure resolving 'azure.archive.ubuntu.com'"), take the
# same protoc release from GitHub instead, checked against a pinned digest.
set -euo pipefail

done_if_present() {
  if command -v protoc >/dev/null 2>&1; then
    echo "protoc: $(protoc --version) from $1"
    exit 0
  fi
}
done_if_present "the image"

apt_opts=(-o Acquire::Retries=2 -o Acquire::http::Timeout=20 -o Acquire::https::Timeout=20)
if sudo apt-get update "${apt_opts[@]}" &&
  sudo apt-get install -y "${apt_opts[@]}" protobuf-compiler libprotobuf-dev; then
  done_if_present "apt"
fi
echo "::warning::apt could not install protobuf-compiler; falling back to the GitHub release"

version=21.12 # the release apt ships as 3.21.12
sha256=3a4c1e5f2516c639d3079b1586e703fc7bcfa2136d58bda24d1d54f949c315e8
[ "$(uname -m)" = x86_64 ] || { echo "install-protoc: no fallback for $(uname -m)" >&2; exit 1; }
zip=$(mktemp --suffix=.zip)
curl -fsSL --retry 5 --retry-all-errors -o "$zip" \
  "https://github.com/protocolbuffers/protobuf/releases/download/v${version}/protoc-${version}-linux-x86_64.zip"
echo "${sha256}  ${zip}" | sha256sum -c -
sudo unzip -oq "$zip" 'bin/protoc' 'include/*' -d /usr/local
sudo chmod 755 /usr/local/bin/protoc
done_if_present "the GitHub release"
echo "install-protoc: protoc still missing" >&2
exit 1
