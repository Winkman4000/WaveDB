#!/bin/bash
# ClickBench: load the official hits.parquet (100M rows) through the command line.
# Prints load time (seconds) and storage size (bytes) -- the two numbers the submission needs.
set -e
cd "$(dirname "$0")/../.."
. venv/bin/activate 2>/dev/null || true
DATA=${1:-hits.parquet}
if [ ! -f "$DATA" ]; then
    wget -q --continue https://datasets.clickhouse.com/hits_compatible/athena/hits.parquet -O "$DATA"
fi
rm -rf cbdb
START=$(date +%s.%N)
python3 bin/wdb load cbdb hits "$DATA" --cluster-by EventTime --cast EventDate=date_days --cast EventTime=timestamp_s --warm benchmark/clickbench/queries.sql
END=$(date +%s.%N)
echo "load time: $(python3 -c "print(round($END - $START, 1))") s"
echo "storage bytes: $(du -sb cbdb | cut -f1)"
