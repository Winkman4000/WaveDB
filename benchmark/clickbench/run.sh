#!/bin/bash
# ClickBench: the 43 official queries, three runs each, through the command line.
# Each query starts with a cold page cache when the harness runs as root (the official
# protocol); every timing is the wall clock of `bin/wdb sql ... --format csv`.
# Output: the results array ClickBench expects: [[t1,t2,t3], ...] with null for a failure.
cd "$(dirname "$0")/../.."
. venv/bin/activate 2>/dev/null || true
QUERIES=benchmark/clickbench/queries.sql
TRIES=3
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
        if python3 bin/wdb sql cbdb "$query" --format csv >/dev/null 2>/tmp/wdb_err; then
            END=$(date +%s.%N); RES=$(python3 -c "print(round($END - $START, 3))")
        else
            RES=null
        fi
        echo -n "$RES"; [ $i -lt $TRIES ] && echo -n ","
    done
    echo -n "]"
done < "$QUERIES"
echo "]"
