#!/bin/bash
# THE FLOOR BOARD (Jackson, 2026-09-29: "make this the version that we run when we launch the board
# from here on out"). The configuration a ClickBench submission may run: no sidecars (indices,
# censuses, pre-aggregates), no answers from the load statistics, nothing built by a query.
#   bench/board_floor.sh DB_DIR OUT_DIR [REFEREE_DIR]
# 1. refuses a database holding anything beyond the data, its load statistics, its text lengths
#    and the catalog; 2. verifies all 43 against the parquet; 3. the true-cold board (fresh process
#    per query, files and numba caches evicted); 4. the score against the referees, when given.
set -e
cd "$(dirname "$0")/.."
DB=${1:?usage: bench/board_floor.sh DB_DIR OUT_DIR [REFEREE_DIR]}
OUT=${2:?usage: bench/board_floor.sh DB_DIR OUT_DIR [REFEREE_DIR]}
REF=${3:-}
P=${PYTHON:-python3}
PARQ=${HITS_PARQUET:-/workspace/data/hits.parquet}
export PYTHONPATH=src WDB_SIDECARS=0 WDB_LOAD_ANSWERS=0
mkdir -p "$OUT"
EXTRA=$(ls "$DB" | grep -v -E '^hits_[0-9]+\.wdb$|\.stats\.npz$|\.clen\.|\.rlen\.|^catalog\.json$|^routing_ledger\.jsonl$|^shelves\.json$' || true)
if [ -n "$EXTRA" ]; then
    echo "THE FLOOR REFUSES: $DB holds derived files:"; echo "$EXTRA" | head -20
    echo "(load with benchmark/clickbench/load.sh, or 'bin/wdb sidecars $DB drop')"; exit 2
fi
echo "floor: $DB  load $(cut -d' ' -f1-3 /proc/loadavg)  $(date)" | tee "$OUT/board.out"
$P bench/verify_correctness.py src "$DB" "$PARQ" bench/clickbench_queries.sql 300 > "$OUT/verify.log" 2>&1 || true
tail -2 "$OUT/verify.log" | tee -a "$OUT/board.out"
$P bench/true_cold.py "$DB" bench/clickbench_queries.sql > "$OUT/board.jsonl" 2> "$OUT/board.err"
NEW=$(ls "$DB" | grep -v -E '^hits_[0-9]+\.wdb$|\.stats\.npz$|\.clen\.|\.rlen\.|^catalog\.json$|^routing_ledger\.jsonl$|^shelves\.json$' || true)
[ -n "$NEW" ] && { echo "THE FLOOR BROKE: files born during the board:"; echo "$NEW"; } | tee -a "$OUT/board.out"
if [ -n "$REF" ]; then
    $P bench/board_vs.py "$OUT/board.jsonl" "$REF" | tee -a "$OUT/board.out"
fi
echo "done $(date)  load $(cut -d' ' -f1-3 /proc/loadavg)" | tee -a "$OUT/board.out"
