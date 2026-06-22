#!/usr/bin/env python3
"""bench/bench_scanpair_kernel.py -- verify the scanpair AVX-512 kernel vs DuckDB on 100M.

Run on the pod (has the 100M segment + parquet oracle). Confirms the kernel path fires, is
bit-exact vs DuckDB, and measures warm end-to-end-through-db.run ratio on the high-match cases that
the numpy path lost. This is the gate before merging scanpair-kernel -> main.

Usage (on pod):
  /workspace/venv/bin/python3 bench/bench_scanpair_kernel.py
Edit DB_DIR / PARQUET below if paths differ.
"""
import sys, time, hashlib
sys.path.insert(0, '/workspace/WaveDB/src')
import numpy as np
from wdb_db import Database
from wdb_engine import Segment
import wdb_scanpair, wdb_kernel, duckdb

DB_DIR  = '/workspace/data/cb25db'
SEGMENT = '/workspace/data/cb25db/hits_0.wdb'
PARQUET = '/workspace/data/hits.parquet'
P = lambda *a: print(*a, flush=True)

HITS = ("WITH hits AS (SELECT * REPLACE ((DATE '1970-01-01' + EventDate) AS EventDate,"
        "(TIMESTAMP '1970-01-01' + to_seconds(EventTime)) AS EventTime) "
        "FROM read_parquet('%s'))" % PARQUET)

def hsh(rows):
    m = hashlib.md5()
    for r in sorted([tuple(round(v, 3) if isinstance(v, float) else v for v in row)
                     for row in rows], key=repr):
        m.update(repr(r).encode())
    return m.hexdigest()

def main():
    P("kernel available:", wdb_kernel._load() is not None)
    db = Database.open(DB_DIR); seg = Segment(SEGMENT)
    con = duckdb.connect()

    # dominant values chosen self-referentially (no peeking at data semantics)
    def dom(col):
        return int(np.asarray(seg._typed_dict(col))[np.bincount(seg.codes(col)).argmax()])
    cases = [
        ('CounterID = 62',                 "CounterID = 62"),
        ('RegionID = dominant (~18%)',      f"RegionID = {dom('RegionID')}"),
        ('ResolutionWidth = dom (~24%)',    f"ResolutionWidth = {dom('ResolutionWidth')}"),
        ('RegionID = 2 (~7%)',              "RegionID = 2"),
    ]
    GB = "GROUP BY SearchEngineID, SearchPhrase ORDER BY c DESC LIMIT 10"
    SEL = "SELECT SearchEngineID, SearchPhrase, COUNT(*) AS c FROM hits WHERE"

    P(f"{'case':<34} {'fired':<6} {'exact':<6} {'wave_ms':>8} {'duck_ms':>8} {'ratio':>7}")
    for label, pred in cases:
        q = f"{SEL} {pred} {GB}"
        h0 = wdb_scanpair._HITS
        try:
            rows, _ = db.run(q)
        except Exception as e:
            P(f"{label:<34} ERR {type(e).__name__}: {str(e)[:50]}"); continue
        fired = wdb_scanpair._HITS - h0 >= 1
        ts = []
        for _ in range(5):
            t = time.perf_counter(); rows, _ = db.run(q); ts.append((time.perf_counter() - t) * 1000)
        wave = float(np.median(ts))
        t = time.time(); d = con.execute(HITS + ' ' + q).fetchall(); duck = (time.time() - t) * 1000
        exact = hsh(rows) == hsh(d)
        P(f"{label:<34} {str(fired):<6} {str(exact):<6} {wave:>8.0f} {duck:>8.0f} {duck/wave:>6.2f}x")
        if not exact:
            P(f"    wdb {rows[:2]}"); P(f"    duck {d[:2]}")
    P("DONE")

if __name__ == '__main__':
    main()
