"""THE THREE READS (Jackson, 2026-09-23): differentiation (codes), identification (a decision per
distinct string from the dictionary AS STORED -- chunk by chunk, prefix carry, no whole-dictionary
blob), return (answer strings only). Identification must equal the plain definition on every
dictionary string, across chunk seams, multi-byte text, newlines and empty strings."""
import sys, os, uuid, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import numpy as np, pandas as pd, duckdb
import wdb_encode, wdb_sql, wdb_strings
from wdb_engine import Segment

TMP = tempfile.gettempdir()
_FIX = None


def _fixture():
    global _FIX
    if _FIX is not None:
        return _FIX
    rng = np.random.default_rng(3)
    vals = set()
    while len(vals) < 60000:
        k = int(rng.integers(0, 4000)); x = int(rng.integers(0, 10 ** 6)); r = int(rng.integers(0, 12))
        if r == 0: v = 'http://www.site%d.com/p/%d' % (k, x)
        elif r == 1: v = 'https://site%d.com/q?x=%d' % (k, x)
        elif r == 2: v = 'https://www.google.com/search?q=%d' % x
        elif r == 3: v = 'https://пример%d.рф/страница/%d' % (k, x)
        elif r == 4: v = 'http://mail.google.%d.ru/%d' % (k, x)
        elif r == 5: v = 'http://site%d.com/a\n%d' % (k, x)
        elif r == 6: v = 'ftp://site%d.com/%d' % (k, x)
        elif r == 7: v = 'Google Поиск %d' % x
        elif r == 8: v = 'http://www./%d' % x
        elif r == 9: v = 'http://site%d.com' % k
        elif r == 10: v = 'https://www.site%d.com/%dgoogle' % (k, x)
        else: v = ''
        vals.add(v)
    vals = sorted(vals)
    n = 200000
    ref = np.array(vals, dtype=object)[rng.integers(0, len(vals), n)]
    df = pd.DataFrame({'ref': ref, 'g': rng.integers(0, 50, n).astype(np.int64)})
    t = uuid.uuid4().hex[:8]; pq = f'{TMP}/tr_{t}.parquet'; w = f'{TMP}/tr_{t}.wdb'
    df.to_parquet(pq, index=False); wdb_encode.encode(pq, w)
    _FIX = (Segment(w), w, pq, df)
    return _FIX


def _dict(seg, col):
    V0 = int(seg.cols[col].get('n_dict') or seg.cols[col]['V'])
    return [bytes(seg.fetch(col, i)) if not isinstance(seg.fetch(col, i), str) else seg.fetch(col, i).encode()
            for i in range(V0)]


def test_chunked_and_differentiation_is_code_order():
    seg, w, pq, df = _fixture()
    c = seg.cols['ref']
    assert c['mode'] == 1 and c.get('chunked') and c['nch'] >= 3, (c['mode'], c.get('chunked'))
    d = _dict(seg, 'ref')
    assert d == sorted(d)                                  # code order IS string order (MIN/ORDER BY)
    codes = np.asarray(wdb_strings.differentiate(seg, 'ref'))
    assert np.array_equal(np.array(d, dtype=object)[codes], np.array([v.encode() for v in df['ref']], dtype=object))


def test_identify_contains_equals_definition():
    seg, w, pq, df = _fixture()
    d = _dict(seg, 'ref')
    for n1, n2 in ((b'google', b''), (b'.google.', b''), (b'Google', b''), (b'www.', b''), (b'/p/1', b''),
                   ('Поиск'.encode(), b''), ('рф/стр'.encode(), b''), (b'\n', b''), (b'http', b'google'),
                   (b'site1', b'/q'), (b'zzzz', b'')):
        keep = wdb_strings.identify_contains(seg, 'ref', n1, n2)
        exp = [(n1 in s) and (not n2 or n2 in s[s.find(n1) + len(n1):]) for s in d]
        assert np.array_equal(keep[:len(d)], np.array(exp)), (n1, n2)


def test_charlens_equal_definition():
    seg, w, pq, df = _fixture()
    d = _dict(seg, 'ref')
    got = np.asarray(seg.dict_charlens('ref'))[:len(d)]
    assert np.array_equal(got, np.array([len(s.decode('utf-8')) for s in d])), 'char lengths'


def test_host_road_one_read_equals_old_road():
    import wdb_regexgroup as RG
    seg, w, pq, df = _fixture()
    spec = {'col': 'ref', 'pat': r'^https?://(?:www\.)?([^/]+)/.*$', 'rep': '\\1', 'lenfn': None}
    pcl = RG._prefix_class(spec['pat'], spec['rep'])
    new = RG._derive_runs_one_read(seg, 'ref', spec, pcl)
    old = RG._derive_runs(Segment(w), 'ref', spec, pcl)
    assert new is not None and old is not None
    assert np.array_equal(new[0], old[0]) and np.array_equal(np.asarray(new[1]), np.asarray(old[1]))
    pairs = np.unique(np.stack([np.asarray(old[2], np.int64), np.asarray(new[2], np.int64)], 1), axis=0)
    assert len(pairs) == len(old[3]) == len(new[3])       # the same partition
    for og, ng in pairs.tolist():
        ol = old[3][og]; ol = ol if isinstance(ol, bytes) else str(ol).encode()
        assert ol == new[3][ng], (ol, new[3][ng])


def test_queries_match_duck():
    seg, w, pq, df = _fixture()
    con = duckdb.connect()
    for sql in ["SELECT COUNT(*) FROM TBL WHERE ref LIKE '%google%'",
                "SELECT g, COUNT(*) FROM TBL WHERE ref LIKE '%google%' AND ref NOT LIKE '%.google.%' GROUP BY g ORDER BY g",
                "SELECT COUNT(*) FROM TBL WHERE ref NOT LIKE '%Поиск%'",
                "SELECT g, AVG(length(ref)), COUNT(*) FROM TBL WHERE ref <> '' GROUP BY g ORDER BY g",
                "SELECT REGEXP_REPLACE(ref, '^https?://(?:www\\.)?([^/]+)/.*$', '\\1') AS k, AVG(length(ref)) AS l, "
                "COUNT(*) AS c, MIN(ref) FROM TBL WHERE ref <> '' GROUP BY k HAVING COUNT(*) > 40 ORDER BY l DESC, k LIMIT 25"]:
        duck = con.execute(sql.replace('TBL', f"'{pq}'")).fetchall()
        rows, _ = wdb_sql.execute(Segment(w), sql.replace('TBL', 'tbl'))
        norm = lambda rs: sorted(tuple(round(float(x), 6) if isinstance(x, float) else
                                       (x.decode() if isinstance(x, bytes) else x) for x in r) for r in rs)
        assert norm(rows) == norm(duck), sql
