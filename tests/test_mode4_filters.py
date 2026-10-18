"""Filters and keys on mode-4 columns (2026-10-18): a mode-4 column's codes are row positions, not dictionary
codes, so a read that looks a literal up in a value dictionary finds a meaningless code. Built with the suite's
mode-4 toys allowed (WDB_SEQ_NARROW_OK / WDB_SEQ_REPEATS_OK, as tests/run.py sets them); every answer is checked
against DuckDB over the same parquet -- whatever read takes it."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
from wdb_db import Database
import wdb_encode, wdb_join, controller

_D = None


def _fixture():
    global _D
    if _D is not None:
        return _D
    rng = np.random.default_rng(23); N = 300_000
    rare = np.zeros(N, np.int64); ix = rng.choice(N, 900, replace=False); rare[ix] = rng.integers(1, 40, 900)
    df = pd.DataFrame({'tiny': rng.integers(0, 7, N).astype('int64'), 'rare': rare,
                       'one': np.full(N, 7, np.int64),
                       'wide': rng.integers(0, 5000, N).astype('int64'),
                       'seq': np.repeat(np.arange(N // 100), 100).astype('int64')})
    d = os.path.join(tempfile.gettempdir(), 'm4f_%s' % uuid.uuid4().hex[:8])
    db = Database.create(d)
    db.run("CREATE TABLE t (tiny BIGINT, rare BIGINT, one BIGINT, wide BIGINT, seq BIGINT)")
    pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
    saved = {k: os.environ.get(k) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    os.environ['WDB_SEQ_NARROW_OK'] = '1'; os.environ['WDB_SEQ_REPEATS_OK'] = '1'
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    db.cat.add_segment('t', 't_0.wdb')
    seg, _ = wdb_join._solo_segment(db, 't')
    _D = (db, pq, seg)
    return _D


Q = [
    "SELECT rare, tiny, COUNT(*) AS c FROM t WHERE rare = 3 GROUP BY rare, tiny ORDER BY c DESC, tiny LIMIT 10",
    "SELECT rare, tiny, COUNT(*) AS c FROM t WHERE rare <> 0 GROUP BY rare, tiny ORDER BY c DESC, rare, tiny LIMIT 10",
    "SELECT one, tiny, COUNT(*) AS c FROM t WHERE one <> 7 GROUP BY one, tiny ORDER BY c DESC LIMIT 10",
    "SELECT one, tiny, COUNT(*) AS c FROM t WHERE one = 7 GROUP BY one, tiny ORDER BY c DESC, tiny LIMIT 10",
    "SELECT tiny, COUNT(*) AS c FROM t WHERE rare = 3 GROUP BY tiny ORDER BY c DESC, tiny LIMIT 10",
    "SELECT wide, COUNT(*) AS c FROM t WHERE rare = 5 GROUP BY wide ORDER BY c DESC, wide LIMIT 10",
    "SELECT COUNT(*) FROM t WHERE rare = 3",
    "SELECT COUNT(*) FROM t WHERE one <> 7",
    "SELECT seq, COUNT(*) AS c FROM t WHERE seq = 1234 GROUP BY seq",
    "SELECT tiny, COUNT(*) AS c FROM t WHERE seq = 1234 GROUP BY tiny ORDER BY c DESC, tiny LIMIT 10",
    "SELECT rare, COUNT(*) AS c FROM t WHERE tiny = 2 GROUP BY rare ORDER BY c DESC, rare LIMIT 10",
]


def _rows(res):
    rows = res[0] if isinstance(res, tuple) else res
    return [tuple(int(x) for x in r) for r in rows]


def test_mode4_filters_against_duckdb():
    db, pq, seg = _fixture()
    con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM '%s'" % pq)
    bad = []
    for q in Q:
        try:
            got = _rows(db.run(q)); served = controller._SERVED[0]
        except Exception as e:
            bad.append((q, 'raised %r' % e)); continue
        want = [tuple(int(x) for x in r) for r in con.execute(q).fetchall()]
        if got != want:
            bad.append((q, served, got[:3], want[:3]))
    # the board's form: ORDER BY the count alone -- ties may come in any order, so check legality: the counts are
    # DuckDB's top counts and every row is a real group with its true count
    for q, g in (("SELECT rare, tiny, COUNT(*) AS c FROM t WHERE rare = 3 GROUP BY rare, tiny ORDER BY c DESC LIMIT 10",
                  "rare = 3"),
                 ("SELECT rare, wide, COUNT(*) AS c FROM t WHERE rare <> 0 GROUP BY rare, wide ORDER BY c DESC LIMIT 10",
                  "rare <> 0")):
        try:
            got = _rows(db.run(q)); served = controller._SERVED[0]
        except Exception as e:
            bad.append((q, 'raised %r' % e)); continue
        a, b = q.split('SELECT ')[1].split(',')[:2]; b = b.strip()
        truth = {(int(x[0]), int(x[1])): int(x[2]) for x in con.execute(
            "SELECT %s, %s, COUNT(*) FROM t WHERE %s GROUP BY %s, %s" % (a, b, g, a, b)).fetchall()}
        want = sorted(truth.values(), reverse=True)[:10]
        if [r[2] for r in got] != want or any(truth.get((r[0], r[1])) != r[2] for r in got):
            bad.append((q, served, got[:3], want[:3]))
    assert not bad, bad
