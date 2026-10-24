#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
g++ -O3 -std=c++17 -march=native -o fractal_bench main.cpp
./fractal_bench "${1:-15}" "${2:-5}" "${3:-9}" > measurements.csv
python3 visualize.py
