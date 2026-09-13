#!/bin/bash
# ClickBench: install WaveDB on a fresh Ubuntu machine (the official harness runs this first).
# Needs: python3 (>= 3.10) with venv, git. Uses the package manager only if python3 is missing.
set -e
if ! command -v python3 >/dev/null; then
    apt-get update -qq && apt-get install -y -qq python3 python3-venv python3-pip git
fi
git clone --depth 1 https://github.com/Winkman4000/WaveDB.git wavedb 2>/dev/null || true
cd wavedb
python3 -m venv venv && . venv/bin/activate
pip install -q --upgrade pip
pip install -q numpy pyarrow zstandard numba sqlglot pandas duckdb
echo "WaveDB installed in $(pwd)"
