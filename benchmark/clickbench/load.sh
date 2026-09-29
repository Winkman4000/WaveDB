#!/bin/bash
# ClickBench: load the official hits.parquet (100M rows) through the command line.
# Prints load time (seconds) and storage size (bytes) -- the two numbers the submission needs.
# THE FLOOR (2026-09-29): no --warm and the sidecar switch off -- the rules allow no index, projection
# or pre-aggregate, so nothing is built beyond the data, its load statistics and its text lengths.
set -e
cd "$(dirname "$0")/../.."
. venv/bin/activate 2>/dev/null || true
DATA=${1:-hits.parquet}
if [ ! -f "$DATA" ]; then
    wget -q --continue https://datasets.clickhouse.com/hits_compatible/athena/hits.parquet -O "$DATA"
fi
rm -rf cbdb
export WDB_SIDECARS=0
START=$(date +%s.%N)
python3 bin/wdb load cbdb hits "$DATA" --cluster-by EventTime --cast EventDate=date_days --cast EventTime=timestamp_s --hash URLHash,RefererHash --row-lengths URL
END=$(date +%s.%N)
python3 -c "import sys; sys.path.insert(0, 'src'); import wdb_sidecar; wdb_sidecar.set_setting('cbdb', 'off')"
echo "load time: $(python3 -c "print(round($END - $START, 1))") s"
echo "storage bytes: $(du -sb cbdb | cut -f1)"
