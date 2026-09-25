"""THE BACK-REFERENCE (tag 20, Jackson 2026-09-24): an operator-declared hash column is stored as, per
row, its code or the gap back to the previous copy inside its block. Every read must give back exactly
the column: the full decode, reads at rows (sorted, unsorted, across blocks), windows, and SQL answers
equal to DuckDB's."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode
from wdb_db import Database

TMP = tempfile.gettempdir()
_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(20)
    n = 200000
    pool = rng.integers(-(1 << 62), 1 << 62, 6000, dtype=np.int64)       # hash-like values
    pick = rng.zipf(1.3, n) % pool.size                                    # a few hot, a long tail
    near = rng.random(n) < 0.3                                             # local repeats: copy a row
    back = np.maximum(0, np.arange(n) - rng.integers(1, 40, n))           # a few rows back
    h = pool[pick]
    h[near] = h[back[near]]
    df = pd.DataFrame({'h': h, 'g': rng.integers(0, 7, n).astype(np.int64),
                       'k': rng.integers(0, 1000, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/he_{t}.parquet'
    df.to_parquet(pq, index=False)
    d = f'{TMP}/hedb_{t}'
    db = Database.create(d)
    db.cat.add_table('tbl', [['h', 'int'], ['g', 'int'], ['k', 'int']])
    keep = {k: os.environ.get(k) for k in ('WDB_HASH_COLS', 'WDB_E20_BR', 'WDB_SEQ_NARROW_OK')}
    os.environ.update(WDB_HASH_COLS='h', WDB_E20_BR='512', WDB_SEQ_NARROW_OK='0')
    try:
        wdb_encode.encode(pq, os.path.join(d, 'tbl_0.wdb'))
    finally:
        for k, v in keep.items():
            if v is None: os.environ.pop(k, None)
            else: os.environ[k] = v
    db.cat.add_segment('tbl', 'tbl_0.wdb')
    _FIX = (db, pq, os.path.join(d, 'tbl_0.wdb'), df)
    return _FIX


def _seg():
    from wdb_engine import Segment
    return Segment(_fixture()[2])


def test_hash_column_is_tag20():
    s = _seg()
    assert s.cols['h'].get('code_enc') == 20, s.cols['h'].get('code_enc')
    assert s.cols['g'].get('code_enc') != 20 and s.cols['k'].get('code_enc') != 20


def test_full_decode_gives_the_column_back():
    s = _seg(); df = _fixture()[3]
    vals = np.asarray(s.values('h')).astype(np.int64)
    assert np.array_equal(vals, df['h'].to_numpy()), 'values differ'


def test_reads_at_rows_and_windows():
    s = _seg()
    full = np.asarray(s._raw_codes('h')).astype(np.int64)
    s2 = _seg()                                              # a fresh segment: no full decode cached
    rng = np.random.default_rng(3)
    for rows in (np.sort(rng.choice(s2.N, 777, replace=False)), rng.choice(s2.N, 999, replace=False),
                 np.array([0, 511, 512, 513, s2.N - 1], dtype=np.int64), np.array([5], dtype=np.int64)):
        got = np.asarray(s2.codes_at('h', rows)).astype(np.int64)
        assert np.array_equal(got, full[rows]), ('rows', rows[:5])
    for lo, hi in ((0, 10), (500, 530), (1023, 1030), (s2.N - 7, s2.N)):
        got = np.asarray(s2._raw_codes_range('h', lo, hi)).astype(np.int64)
        assert np.array_equal(got, full[lo:hi]), ('window', lo, hi)


def _duck(pq, sql):
    return duckdb.connect().execute(sql.replace('FROM tbl', "FROM read_parquet('%s')" % pq)).fetchall()


def _eq(sql):
    db, pq, _, _ = _fixture()
    a = [tuple(int(x) for x in r) for r in db.run(sql)[0]]
    b = [tuple(int(x) for x in r) for r in _duck(pq, sql)]
    assert a == b, (sql, a[:5], b[:5])


def test_sql_answers_equal_duckdb():
    df = _fixture()[3]
    hot = int(df['h'].value_counts().index[0]); rare = int(df['h'].value_counts().index[-1])
    _eq('SELECT COUNT(*) FROM tbl WHERE h = %d' % hot)
    _eq('SELECT COUNT(*) FROM tbl WHERE h = %d' % rare)
    _eq('SELECT COUNT(*) FROM tbl WHERE h = %d AND g = 3' % hot)
    _eq('SELECT h, COUNT(*) AS c FROM tbl GROUP BY h ORDER BY c DESC, h LIMIT 10')
    _eq('SELECT g, COUNT(DISTINCT h) FROM tbl GROUP BY g ORDER BY g')
    _eq('SELECT h, g, COUNT(*) AS c FROM tbl WHERE k < 100 GROUP BY h, g ORDER BY c DESC, h, g LIMIT 10')
    _eq('SELECT MIN(h), MAX(h) FROM tbl')
