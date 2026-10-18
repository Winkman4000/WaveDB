"""wdb_pairtop's general filter (2026-10-17, the Q14 board): one '=' / '<>' between ANY column and a literal,
applied by code at the survivor rows and the filler pool. A near-unique key's pairs are mostly count 1, so a
LIMIT has many legal answers: checked for LEGALITY against DuckDB (the counts are DuckDB's top-k counts, every row
is a real group of the filtered rows with its true count)."""
import sys, os, uuid, tempfile, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
from wdb_db import Database
import wdb_encode, wdb_pairtop

_D = None


def _fixture():
    global _D
    if _D is not None:
        return _D
    rng = np.random.default_rng(31); N = 200_000
    u = rng.permutation(N).astype('int64')
    dup = rng.choice(N, 300, replace=False)                 # a few repeated ids, so real count-2+ pairs exist
    u[dup[:150]] = u[dup[150:]]
    df = pd.DataFrame({'u': u, 'a': rng.integers(0, 6, N).astype('int64'),
                       's': rng.integers(0, 40, N).astype('int64'),
                       'w': rng.choice(['', 'x', 'y'], N, p=[.7, .2, .1])})
    d = os.path.join(tempfile.gettempdir(), 'ptf_%s' % uuid.uuid4().hex[:8])
    db = Database.create(d)
    db.run("CREATE TABLE t (u BIGINT, a BIGINT, s BIGINT, w VARCHAR)")
    pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
    wdb_encode.encode(pq, os.path.join(d, 't_0.wdb')); db.cat.add_segment('t', 't_0.wdb')
    _D = (db, df, pq, d)
    return _D


def _legal(rows, pq, where, k1, k2, lim):
    con = duckdb.connect()
    truth = {(x[0], x[1]): x[2] for x in con.execute(
        f"SELECT {k1}, {k2}, COUNT(*) FROM '{pq}' WHERE {where} GROUP BY {k1}, {k2}").fetchall()}
    want = sorted(truth.values(), reverse=True)[:lim]
    got = [int(r[2]) for r in rows]
    assert got == want, (where, got, want)
    for r in rows:
        assert truth.get((r[0], r[1])) == int(r[2]), (where, r)
    assert len({(r[0], r[1]) for r in rows}) == len(rows)


def test_general_filter_on_each_column():
    db, df, pq, d = _fixture()
    for where in ('a <> 2', 'a = 4', 's <> 7', 'u <> %d' % int(df.u.iloc[0]), "w <> 'x'", "w = 'y'"):
        for k1, k2 in (('u', 'a'), ('s', 'u')):
            h0 = wdb_pairtop._HITS
            q = 'SELECT %s, %s, COUNT(*) AS c FROM t WHERE %s GROUP BY %s, %s ORDER BY c DESC LIMIT 10' % (
                k1, k2, where, k1, k2)
            rows = [tuple(x.decode() if isinstance(x, (bytes, bytearray)) else x for x in r) for r in db.run(q)[0]]
            # an '=', or a filter on a third column, may be an earlier read's (wherescan); the Q14 shape is
            # '<>' on one of the keys, and that is pairtop's
            if '<>' in where and where.split()[0] in (k1, k2):
                assert wdb_pairtop._HITS == h0 + 1, ('pairtop did not serve', q)
            _legal(rows, pq, where, k1, k2, 10)


def test_absent_literal_hands_on():
    db, df, pq, d = _fixture()
    h0 = wdb_pairtop._HITS
    db.run('SELECT u, a, COUNT(*) AS c FROM t WHERE a <> 99 GROUP BY u, a ORDER BY c DESC LIMIT 10')
    assert wdb_pairtop._HITS == h0, 'an absent literal was guessed'
