#!/bin/bash
# THE DUCKDB REFERENCE, run the way ClickBench runs DuckDB: hits.parquet loaded into a native
# database with their conversions (EventTime -> TIMESTAMP, EventDate -> DATE), then the 43
# queries three times each, page cache dropped before each query (when root), each try a
# fresh process -- the same protocol our run.sh follows.
# usage: bash benchmark/clickbench/duck_reference.sh [hits.parquet] [duckbench.db]
cd "$(dirname "$0")/../.."
. venv/bin/activate 2>/dev/null || true
DATA=${1:-hits.parquet}; DBF=${2:-duckbench.db}
QUERIES=benchmark/clickbench/queries.sql
TRIES=3
if [ ! -f "$DBF" ]; then
    START=$(date +%s.%N)
    python3 - "$DATA" "$DBF" << 'EOF'
import sys, duckdb
data, dbf = sys.argv[1], sys.argv[2]
con = duckdb.connect(dbf)
con.execute("CREATE TABLE hits AS SELECT * REPLACE (make_timestamp(EventTime * 1000000) AS EventTime, DATE '1970-01-01' + INTERVAL (EventDate) DAYS AS EventDate) FROM read_parquet('%s')" % data)
con.close()
EOF
    END=$(date +%s.%N)
    echo "duck load time: $(python3 -c "print(round($END - $START, 1))") s" >&2
    echo "duck storage bytes: $(du -sb "$DBF" | cut -f1)" >&2
fi
echo -n "["
FIRST=1
while IFS= read -r query; do
    [ -z "$query" ] && continue
    case "$query" in --*) continue;; esac
    sync
    if [ "$(id -u)" = "0" ] && [ -w /proc/sys/vm/drop_caches ]; then echo 3 > /proc/sys/vm/drop_caches; fi
    if [ $FIRST = 1 ]; then FIRST=0; else echo -n ","; fi
    echo -n "["
    for i in $(seq 1 $TRIES); do
        START=$(date +%s.%N)
        if python3 -c "import sys, duckdb; con = duckdb.connect(sys.argv[1], read_only=True); con.execute(sys.argv[2]).fetchall()" "$DBF" "$query" >/dev/null 2>/tmp/duck_err; then
            END=$(date +%s.%N); RES=$(python3 -c "print(round($END - $START, 3))")
        else
            RES=null
        fi
        echo -n "$RES"; [ $i -lt $TRIES ] && echo -n ","
    done
    echo -n "]"
done < "$QUERIES"
echo "]"
