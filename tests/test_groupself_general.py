"""groupself across dictionary modes (2026-10-18, the single-key board): GROUP BY K with the WHERE on K itself --
int-dictionary keys (mode 2) as well as modes 0/1, '=' / '<>' / IN / NOT IN, a near-unique key whose counts all
tie at the LIMIT (the canonical top: count desc, code asc), a one-value key filtered away. Checked for legality
against DuckDB: the counts are DuckDB's top counts and every row is a real group with its true count. A second
ORDER BY term over a tie must be followed exactly, whichever read answers."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
from wdb_db import Database
import wdb_encode, wdb_join, wdb_groupself, controller

_D = None


def _fixture():
    global _D
    if _D is not None:
        return _D
    rng = np.random.default_rng(29); N = 400_002
    df = pd.DataFrame({
        'big': (rng.integers(0, 50_000, N) * 1_000_003 + 17).astype('int64'),   # wide ints: an int dictionary
        'uniq': rng.permutation(N).astype('int64') * 7 + 3,                      # every value once
        'one': np.full(N, 9, np.int64),
        'small': rng.integers(0, 12, N).astype('int64'),
        'even': rng.permutation(np.arange(N) % 6).astype('int64'),             # six values, exactly tied counts
    })
    d = os.path.join(tempfile.gettempdir(), 'gsg_%s' % uuid.uuid4().hex[:8])
    db = Database.create(d)
    db.run("CREATE TABLE t (big BIGINT, uniq BIGINT, one BIGINT, small BIGINT, even BIGINT)")
    pq = os.path.join(d, 's.parquet'); df.to_parquet(pq, index=False)
    saved = {k: os.environ.pop(k, None) for k in ('WDB_SEQ_NARROW_OK', 'WDB_SEQ_REPEATS_OK')}
    try:
        wdb_encode.encode(pq, os.path.join(d, 't_0.wdb'))
    finally:
        for k, v in saved.items():
            if v is not None: os.environ[k] = v
    db.cat.add_segment('t', 't_0.wdb')
    seg, _ = wdb_join._solo_segment(db, 't')
    _D = (db, df, pq, seg)
    return _D


def _legal(con, q, rows, key, where, lim):
    w = (' WHERE ' + where) if where else ''
    truth = {int(r[0]): int(r[1]) for r in con.execute('SELECT %s, COUNT(*) FROM t%s GROUP BY %s' % (key, w, key)).fetchall()}
    want = sorted(truth.values(), reverse=True)[:lim]
    got = [int(r[1]) for r in rows]
    assert got == want, (q, got[:5], want[:5])
    for r in rows:
        assert truth.get(int(r[0])) == int(r[1]), (q, r)
    assert len({int(r[0]) for r in rows}) == len(rows)


def test_groupself_modes_filters_ties():
    db, df, pq, seg = _fixture()
    con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM '%s'" % pq)
    modes = {c: seg.cols[c]['mode'] for c in ('big', 'uniq', 'one', 'small')}
    topbig = int(df['big'].value_counts().index[0]); u0 = int(df['uniq'].iloc[0])
    cases = [('big', 'big <> %d' % topbig), ('big', 'big = %d' % topbig), ('big', 'big IN (%d, %d)' % (topbig, int(df.big.iloc[5]))),
             ('big', 'big NOT IN (%d)' % topbig), ('uniq', 'uniq <> %d' % u0), ('uniq', None), ('one', 'one <> 9'),
             ('small', 'small <> 3'), ('small', 'small = 3'), ('even', 'even <> 2'), ('even', None)]
    served = {}
    for key, where in cases:
        q = 'SELECT %s, COUNT(*) AS c FROM t%s GROUP BY %s ORDER BY c DESC LIMIT 3' % (
            key, (' WHERE ' + where) if where else '', key)
        h0 = wdb_groupself._HITS
        res = db.run(q); rows = res[0] if isinstance(res, tuple) else res
        served[q] = (controller._SERVED[0], wdb_groupself._HITS > h0)
        _legal(con, q, rows, key, where, 3)
    # a second ORDER BY term over an exact tie: the answer must follow it, whichever read takes it
    for q in ('SELECT even, COUNT(*) AS c FROM t GROUP BY even ORDER BY c DESC, even DESC LIMIT 3',
              'SELECT even, COUNT(*) AS c FROM t WHERE even <> 1 GROUP BY even ORDER BY c DESC, even DESC LIMIT 3',
              'SELECT even, COUNT(*) AS c FROM t WHERE even <> 1 GROUP BY even ORDER BY c DESC, even LIMIT 3'):
        res = db.run(q); rows = res[0] if isinstance(res, tuple) else res
        want = con.execute(q).fetchall()
        assert [(int(a), int(b)) for a, b in rows] == [(int(a), int(b)) for a, b in want], (q, rows, want)
    # the int-dictionary keys with a WHERE on themselves are groupself's now
    for q, (rd, hit) in served.items():
        k9 = q.split()[1].rstrip(',')
        if ' WHERE ' in q and k9 in ('big', 'uniq') and modes[k9] == 2:
            assert hit, (q, rd, modes)
