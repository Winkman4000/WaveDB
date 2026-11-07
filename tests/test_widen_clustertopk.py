"""cluster_topk, widened: SELECT cols FROM t WHERE col <> <literal> ORDER BY clusterkey [DESC][, sec...] LIMIT k for any
literal of the column's kind (text or int), on columns with NULLs (NULL <> x is not TRUE: the NULL code is excluded),
a sparse column whose stored default is the literal, one whose default is not, a literal the column never holds, a
front-coded (mode 1) text column and an int-dictionary (mode 2) column. A clustered fixture (encode cluster_by='tm');
every answer DuckDB's. The ORDER BY ends in a unique key, or projects only its own keys, so there is no tie at the cut."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

_FX = None
_DIR = [None]
N = 300000


def _fixture():
    """N rows, clustered by tm (one tick per ~4 rows, shuffled): id unique; s sparse (default 'x' 80%, '' 5%,
    'v0'..'v999'); e sparse with default '' (80%, else 'e0'..'e99'); n no '' ('n0'..'n39'); nn (NULL 20%, '' 10%,
    'm0'..'m6'); g (NULL 40%, 'g0' 30%, 'g1' 20%, 'g2' 10%); url (>50k distinct: front-coded, '' 3%); c (300 ints);
    ki (ints 0..9 with NULL 25%); u (100k distinct ints: an int dictionary, mode 2)"""
    global _FX
    if _FX is not None:
        return _FX
    import uuid, tempfile, duckdb, wdb_encode
    import numpy as np, pyarrow as pa, pyarrow.parquet as pq
    from wdb_db import Database
    rng = np.random.default_rng(11)
    i = np.arange(N, dtype=np.int64)
    tm = (np.datetime64('2013-07-01T00:00:00') + rng.integers(0, N // 4, N).astype('timedelta64[s]')).astype('datetime64[s]')
    h = rng.random(N); hv = rng.integers(0, 1000, N)
    s = np.where(h < 0.80, 'x', np.where(h < 0.85, '', np.char.add('v', hv.astype(str))))
    he = rng.random(N)
    e = np.where(he < 0.80, '', np.char.add('e', rng.integers(0, 100, N).astype(str)))
    n = np.char.add('n', rng.integers(0, 40, N).astype(str))
    h3 = rng.integers(0, 10, N); m7 = rng.integers(0, 7, N)
    nn = [None if x < 2 else ('' if x == 2 else 'm%d' % y) for x, y in zip(h3.tolist(), m7.tolist())]
    hg = rng.random(N)
    g = [None if x < 0.4 else ('g0' if x < 0.7 else ('g1' if x < 0.9 else 'g2')) for x in hg.tolist()]
    hu = rng.random(N)
    url = np.where(hu < 0.03, '', np.char.add('http://site/', rng.integers(0, 200000, N).astype(str)))
    c = rng.integers(0, 300, N).astype(np.int32)
    kv = rng.integers(0, 10, N); kn = rng.random(N) < 0.25
    ki = pa.array(kv, pa.int64(), mask=kn)
    u = rng.integers(0, 100000, N).astype(np.int64)
    tbl = pa.table({'id': i, 'tm': pa.array(tm, pa.timestamp('s')), 's': pa.array(s.tolist(), pa.string()),
                    'e': pa.array(e.tolist(), pa.string()), 'n': pa.array(n.tolist(), pa.string()),
                    'nn': pa.array(nn, pa.string()), 'g': pa.array(g, pa.string()),
                    'url': pa.array(url.tolist(), pa.string()), 'c': c, 'ki': ki, 'u': u})
    dd = os.path.join(tempfile.gettempdir(), 'wctk_%s' % uuid.uuid4().hex[:8]); os.makedirs(dd, exist_ok=True)
    try:
        import helpers as _H; _H.register_dir(dd)
    except Exception:
        pass
    _DIR[0] = dd
    p = os.path.join(dd, 't.parquet'); pq.write_table(tbl, p)
    con = duckdb.connect(); con.execute("CREATE VIEW t AS SELECT * FROM read_parquet('%s')" % p)
    db = Database.create(dd)
    db.cat.add_table('t', [['id', 'int'], ['tm', 'datetime'], ['s', 'str'], ['e', 'str'], ['n', 'str'],
                           ['nn', 'str'], ['g', 'str'], ['url', 'str'], ['c', 'int'], ['ki', 'int'], ['u', 'int']])
    wdb_encode.encode(p, os.path.join(dd, 't_0.wdb'), cluster_by='tm')
    db.cat.add_segment('t', 't_0.wdb')
    _FX = (db, con)
    return _FX


def _rows(r):
    out = []
    for x in (r[0] if isinstance(r, tuple) else r):
        t = []
        for v in x:
            if hasattr(v, 'isoformat'):
                v = str(v)[:19]
            elif isinstance(v, (bytes, bytearray)):
                v = v.decode()
            t.append(v)
        out.append(tuple(t))
    return out


def _check(q, served='cluster_topk'):
    import controller
    db, con = _fixture()
    got = _rows(db.run(q))
    assert controller._SERVED[0] == served, (q, controller._SERVED[0])
    want = _rows(con.execute(q).fetchall())
    assert got == want, (q, got[:5], want[:5])


def _both(where, proj='id, tm', lim=20, sec=', id'):
    for d in ('', ' DESC'):
        _check("SELECT %s FROM t WHERE %s ORDER BY tm%s%s LIMIT %d" % (proj, where, d, sec, lim))


def test_layout():
    """the fixture holds the shapes the tests name: clustered by tm, s/e sparse, url front-coded, u an int dict"""
    from wdb_engine import Segment
    _fixture()
    seg = Segment(os.path.join(_DIR[0], 't_0.wdb'))
    assert seg.cluster_meta()['key'] == 'tm'
    import wdb_policies as P
    assert seg.cols['s'].get('code_enc') in (8, 9) and P.sparse_default_is(seg, 's', 'x')
    assert seg.cols['e'].get('code_enc') in (8, 9) and P.sparse_default_is(seg, 'e', '')
    assert seg.cols['url']['mode'] == 1 and seg.cols['u']['mode'] == 2
    assert seg.cols['nn']['has_null'] and seg.cols['g']['has_null'] and seg.cols['ki']['has_null']


def test_old_literal_empty():
    _both("s <> ''")
    _both("e <> ''")                               # sparse, default '': the old fast case
    _both("s <> ''", proj='s', sec=', s')          # the original Q24-like shape


def test_sparse_default_literal():
    _both("s <> 'x'")
    _both("e <> 'e5'")


def test_existing_non_default_value():
    _both("s <> 'v7'")
    _both("g <> 'g2'")
    _both("n <> 'n3'", proj='n, id', sec=', n, id')


def test_absent_literal():
    _both("s <> 'zz'")
    _both("n <> 'zz'")
    _both("g <> 'zz'")                             # absent, but NULLs still never pass


def test_null_columns():
    _both("nn <> ''", proj='nn, id')
    _both("nn <> 'm3'", proj='nn, id')
    _both("g <> 'g0'", proj='g, id')


def test_no_empty_in_column():
    _both("n <> ''")


def test_front_coded():
    _both("url <> ''", proj='url, id')
    _both("url <> 'http://site/17'")             # absent from this fixture
    _, con = _fixture()
    for d in ('', ' DESC'):                         # present: the url of the first (last) key value
        v = con.execute("SELECT url FROM t WHERE url <> '' ORDER BY tm%s, id LIMIT 1" % d).fetchall()[0][0]
        _check("SELECT url, id FROM t WHERE url <> '%s' ORDER BY tm%s, id LIMIT 20" % (v, d))


def test_int_columns():
    _both("c <> 7", proj='c, id')
    _both("c <> -3")                               # absent
    _both("c <> 2.5")                              # no int equals it
    _both("ki <> 4", proj='ki, id')                # NULLs never pass
    _both("u <> %d" % 12345, proj='u, id')         # int dictionary (mode 2)


def test_reversed_and_limit_spans():
    _both("'' <> s")
    _both("s <> 'x'", lim=3000)                   # past many key values and batches


def test_declines_kind_mismatch():
    """a literal of the other kind: not a code test (DuckDB would cast) -- the read declines, the answer stays exact"""
    import controller
    db, con = _fixture()
    for q in ("SELECT id FROM t WHERE c <> '7' ORDER BY tm, id LIMIT 10",
              "SELECT id FROM t WHERE tm <> '2013-07-01 00:00:05' ORDER BY tm, id LIMIT 10"):
        got = _rows(db.run(q))
        assert controller._SERVED[0] != 'cluster_topk', q
        assert got == _rows(con.execute(q).fetchall()), q


def _time(q, reps=20):
    db, _ = _fixture()
    db.run(q)
    t0 = time.perf_counter()
    for _ in range(reps):
        db.run(q)
    return (time.perf_counter() - t0) / reps * 1000


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'time':
        for q in ("SELECT e FROM t WHERE e <> '' ORDER BY tm, e LIMIT 10",
                  "SELECT e FROM t WHERE e <> '' ORDER BY tm DESC, e LIMIT 10",
                  "SELECT s FROM t WHERE s <> '' ORDER BY tm, s LIMIT 10"):
            print('%.2f ms  %s' % (_time(q), q))
        sys.exit(0)
    fails = 0; n = 0
    for nm, fn in list(globals().items()):
        if nm.startswith('test_') and callable(fn):
            n += 1
            try:
                fn(); print('PASS', nm)
            except Exception as ex:
                fails += 1; print('FAIL', nm, repr(ex)[:400])
    print('%d/%d passed' % (n - fails, n))
    sys.exit(1 if fails else 0)
