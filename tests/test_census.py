"""THE CENSUS OF THE LOAD (Jackson, 2026-09-24): the load writes rows-per-code for every small
dictionary column; COUNT(*) WHERE k <op> v and GROUP BY k COUNT(*) are answered from those counts
without a row decoded. Every answer must equal DuckDB's. Also: GROUP BY k WHERE k <> v with no
ORDER BY must not bring the excluded key back as a count-1 group (the singleton-law hole)."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_blockstats, wdb_gbcount
from wdb_db import Database

TMP = tempfile.gettempdir()
_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(31)
    n = 200000
    a = np.zeros(n, np.int64)
    hit = rng.random(n) < 0.012                                   # ~1% off the big exception
    a[hit] = rng.choice([2, 3, 7, 13, 16, 21, 44, 62], hit.sum())
    s = np.array(['', 'alpha', 'beta', 'gamma', 'delta'], dtype=object)[rng.integers(0, 5, n)]
    big = rng.permutation(n).astype(np.int64)                     # past VCNT_MAX? no -- n distinct:
    df = pd.DataFrame({'a': a, 's': s, 'g': rng.integers(0, 40, n).astype(np.int64), 'big': big})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/cs_{t}.parquet'
    df.to_parquet(pq, index=False)
    d = f'{TMP}/csdb_{t}'
    db = Database.create(d)
    db.cat.add_table('tbl', [['a', 'int'], ['s', 'string'], ['g', 'int'], ['big', 'int']])
    prev = os.environ.get('WDB_SEQ_NARROW_OK')              # the suite lets narrow columns become
    os.environ['WDB_SEQ_NARROW_OK'] = '0'                    # sequences (mode 4); here 'a' must be a
    try:                                                     # dictionary column, as AdvEngineID is
        wdb_encode.encode(pq, os.path.join(d, 'tbl_0.wdb'))
    finally:
        if prev is None: os.environ.pop('WDB_SEQ_NARROW_OK', None)
        else: os.environ['WDB_SEQ_NARROW_OK'] = prev
    from wdb_engine import Segment as _S
    _c = _S(os.path.join(d, 'tbl_0.wdb')).cols
    assert _c['a']['mode'] in (0, 1, 2) and _c['s']['mode'] in (0, 1, 2), ('fixture encodings', _c['a']['mode'], _c['s']['mode'])
    db.cat.add_segment('tbl', 'tbl_0.wdb')
    _FIX = (db, pq, os.path.join(d, 'tbl_0.wdb'))
    return _FIX


def _duck(pq, sql):
    return duckdb.connect().execute(sql.replace('FROM tbl', "FROM read_parquet('%s')" % pq)).fetchall()


def _norm(rows):
    out = []
    for r in rows:
        out.append(tuple(x.decode() if isinstance(x, (bytes, bytearray)) else (int(x) if isinstance(x, (np.integer,)) else x) for x in r))
    return out


def test_load_writes_the_census():
    db, pq, w = _fixture()
    from wdb_engine import Segment
    seg = Segment(w)
    va = wdb_blockstats.vcnt_from_load(seg, 'a')
    assert va is not None and int(va.sum()) == seg.N and va.size == seg.cols['a']['V']
    assert wdb_blockstats.vcnt_from_load(seg, 's') is not None
    assert wdb_blockstats.vcnt_from_load(seg, 'big') is None          # 200,000 codes: past VCNT_MAX


def test_counts_equal_duckdb_and_come_from_the_census():
    db, pq, w = _fixture()
    qs = ["SELECT COUNT(*) FROM tbl WHERE a <> 0", "SELECT COUNT(*) FROM tbl WHERE a = 0",
          "SELECT COUNT(*) FROM tbl WHERE a = 7", "SELECT COUNT(*) FROM tbl WHERE a = 5",
          "SELECT COUNT(*) FROM tbl WHERE a IN (2, 3, 99)", "SELECT COUNT(*) FROM tbl WHERE a NOT IN (0, 2)",
          "SELECT COUNT(*) FROM tbl WHERE a > 5", "SELECT COUNT(*) FROM tbl WHERE a BETWEEN 3 AND 21",
          "SELECT COUNT(*) FROM tbl WHERE s <> ''", "SELECT COUNT(*) FROM tbl WHERE s = 'beta'"]
    for q in qs:
        h0 = wdb_gbcount._HITS
        got = db.run(q)[0]
        assert _norm(got) == _norm(_duck(pq, q)), (q, got)
        assert wdb_gbcount._HITS == h0 + 1, ('not served by the census', q)


def test_group_counts_equal_duckdb():
    db, pq, w = _fixture()
    for q in ["SELECT a, COUNT(*) FROM tbl WHERE a <> 0 GROUP BY a ORDER BY COUNT(*) DESC, a",
              "SELECT s, COUNT(*) FROM tbl GROUP BY s ORDER BY COUNT(*) DESC, s",
              "SELECT a, COUNT(*) AS c FROM tbl GROUP BY a ORDER BY c DESC, a"]:
        got = db.run(q)[0]
        assert _norm(got) == _norm(_duck(pq, q)), (q, got)


def test_excluded_key_stays_out_unordered():
    """GROUP BY k WHERE k <> v, no ORDER BY: served from the census AND from the decoded heavy list
    (the census hidden) -- the excluded key must never return as a count-1 group."""
    db, pq, w = _fixture()
    q = "SELECT a, COUNT(*) FROM tbl WHERE a <> 0 GROUP BY a"
    exp = sorted(_norm(_duck(pq, q)))
    assert sorted(_norm(db.run(q)[0])) == exp
    prev = wdb_blockstats.vcnt_from_load
    wdb_blockstats.vcnt_from_load = lambda seg, col: None
    try:
        wdb_gbcount._CACHE.clear()
        assert sorted(_norm(db.run(q)[0])) == exp
    finally:
        wdb_blockstats.vcnt_from_load = prev
        wdb_gbcount._CACHE.clear()
